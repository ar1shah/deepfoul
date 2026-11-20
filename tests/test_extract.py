"""Full-fight extract pass without video or a GPU: frame grid, parts + resume, cache round trips,
identity from cache, the weak-corner guard, --all selection and the golden baseline verdicts."""

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import deepfoul.io.video as video
from deepfoul.config import load_config
from deepfoul.perception import extract as ex
from deepfoul.perception.identity import smooth_labels, smooth_track, split_segments
from deepfoul.viz.id_report import baseline_verdict

CFG = load_config()
H, W = 720, 1280


# ---------------------------------------------------------------- grid and parts


def test_stride_for():
    assert ex.stride_for(60.0, 30) == 2
    assert ex.stride_for(50.0, 30) == 2
    assert ex.stride_for(30.0, 30) == 1
    assert ex.stride_for(29.97, 30) == 1
    assert ex.stride_for(23.976, 30) == 1
    assert ex.stride_for(60.0, None) == 1


def shots_of(bounds):
    return [{"shot_id": i, "start_frame": a, "end_frame": b, "start_s": a / 30, "end_s": b / 30}
            for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_parts_start_at_shots_on_the_grid_and_cover_the_window(stride):
    shots = shots_of([101, 140, 171, 260, 263, 400])
    parts = ex.plan_parts(shots, 30.0, stride, chunk_s=2.0)
    starts = {s["start_frame"] for s in shots}
    grid = set(range(101, 400, stride))
    for p in parts:
        assert p["start_frame"] in grid
        assert ex.grid_frame(p["start_frame"] - stride + 1, 101, stride) == p["start_frame"]
        assert any(s <= p["start_frame"] < s + stride for s in starts)
    covered = [f for p in parts for f in range(p["start_frame"], p["end_frame"], stride)]
    assert covered == sorted(grid)
    assert sum(p["frames"] for p in parts) == len(grid)


# ---------------------------------------------------------------- synthetic pass


@dataclass
class F:
    idx: int
    t: float
    image: np.ndarray


FPS = 30.0
IMG = np.zeros((H, W, 3), np.uint8)
IMG[:, : W // 2] = (20, 20, 200)   # red half, blue half: histograms aren't empty
IMG[:, W // 2:] = (200, 60, 20)


def fake_frames(row, raw_dir, start_s=None, end_s=None, stride=1):
    """Stands in for iter_fight_frames: frames 0..599 at 30 fps, same seek rule."""
    kept = 0
    for i in range(600):
        t = i / FPS
        if start_s is not None and t < start_s - 1e-3:
            continue
        if end_s is not None and t >= end_s - 1e-3:
            break
        if kept % stride == 0:
            yield F(i, t, IMG)
        kept += 1


def make_detect():
    """Two people standing apart."""

    def detect(image):
        return (np.array([[200, 100, 400, 650], [800, 100, 1000, 650]], np.float32),
                np.array([0.9, 0.9], np.float32))
    return detect


def fake_pose(image, boxes):
    """Keypoints laid out inside each box, all confident."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    k = np.zeros((len(boxes), 133, 2), np.float32)
    rng = np.linspace(0.05, 0.95, 133)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        k[i, :, 0] = x1 + (x2 - x1) * rng
        k[i, :, 1] = y1 + (y2 - y1) * rng[::-1]
        # shoulders, hips, knees, ankles: a standing person
        cx, w, h = (x1 + x2) / 2, x2 - x1, y2 - y1
        for j, (dx, fy) in {5: (-.2, .2), 6: (.2, .2), 11: (-.15, .5), 12: (.15, .5),
                            13: (-.15, .7), 14: (.15, .7), 15: (-.15, .9),
                            16: (.15, .9)}.items():
            k[i, j] = (cx + dx * w, y1 + fy * h)
    return k, np.ones((len(boxes), 133), np.float32)


def job_for(tmp_path, stride_target):
    shots = shots_of([10, 70, 75, 200, 330, 331, 500])
    doc = {"fps": FPS, "window": [10 / FPS, 500 / FPS], "detector": {}, "shots": shots}
    cfg = {**CFG, "paths": {**CFG["paths"], "keypoints": tmp_path / "kp",
                            "cache": tmp_path / "cache", "raw": tmp_path},
           "extract": {"target_fps": stride_target, "chunk_s": 2.0}}
    row = {"fight_id": "f1", "red_shorts": "red", "blue_shorts": "blue",
           "action_start_s": "", "action_end_s": ""}
    return ex.plan_job(row, cfg, doc, "performance"), cfg


class Stop(Exception):
    pass


@pytest.mark.parametrize("target", [30, 15])   # stride 1 and stride 2
def test_resume_gives_the_same_rows_as_one_run(tmp_path, monkeypatch, target):
    monkeypatch.setattr(video, "iter_fight_frames", fake_frames)
    one, cfg1 = job_for(tmp_path / "a", target)
    assert one.stride == (1 if target == 30 else 2)
    ex.run_perception(one, cfg1, make_detect(), fake_pose)
    full_b, full_f = ex.assemble(one)

    two, cfg2 = job_for(tmp_path / "b", target)
    assert len(two.parts) >= 3

    def stop_after_two(job, p, timing):
        if p["part"] == 1:
            raise Stop

    with pytest.raises(Stop):
        ex.run_perception(two, cfg2, make_detect(), fake_pose, stop_after_two)
    again, _ = job_for(tmp_path / "b", target)
    assert again.status == "resume" and 0 < again.frames_done < again.frames_total
    ex.run_perception(again, cfg2, make_detect(), fake_pose)
    res_b, res_f = ex.assemble(again)

    assert full_f["frame_idx"].tolist() == res_f["frame_idx"].tolist()
    assert (np.diff(full_f["frame_idx"]) % one.stride == 0).all()
    assert full_f["frame_idx"].iloc[0] == 10
    cols = ["frame_idx", "det_idx", "track_id", "x1", "shot_id", "guard"]
    pd.testing.assert_frame_equal(full_b[cols], res_b[cols])
    assert np.array_equal(np.stack(full_b["kpts"]), np.stack(res_b["kpts"]))
    assert not one.extra.get("grid_warnings") and not again.extra.get("grid_warnings")


def test_parquet_round_trip_keeps_unknown_hists(tmp_path):
    df = pd.DataFrame({"frame_idx": [1, 2], "kpts": [np.arange(266, dtype=np.float32)] * 2,
                       "hist_shorts": [np.full(80, 1 / 80, np.float32), None],
                       "clothed": [0.2, np.nan]})
    p = tmp_path / "x.parquet"
    ex.write_parquet(df, p, {"a": 1})
    back = ex.read_boxes(p)
    assert back["hist_shorts"][1] is None
    assert back["kpts"][0].shape == (266,) and back["kpts"][0].dtype == np.float32
    assert ex.read_meta(p) == {"a": 1}


def test_identity_from_cache_equals_in_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "iter_fight_frames", fake_frames)
    job, cfg = job_for(tmp_path, 30)
    ex.run_perception(job, cfg, make_detect(), fake_pose)
    boxes, frames = ex.assemble(job)
    prof = ex.make_profile("f1", job.row, boxes, 10 / FPS, cfg)
    a = ex.identify(boxes, prof, cfg, FPS)
    ex.write_parquet(boxes, tmp_path / "b.parquet")
    b = ex.identify(ex.read_boxes(tmp_path / "b.parquet"), prof, cfg, FPS)
    assert a["label"].tolist() == b["label"].tolist()
    assert set(a["label"]) <= {"red", "blue", "ref", "merged", "none"}
    s = ex.summarize(boxes.assign(corner=a["label"].values), frames)
    assert s["frames"] == len(frames)


# ---------------------------------------------------------------- weak-corner guard


def guard_frame(rows):
    base = {"track_id": "", "shot_id": 1, "fallback": False, "margin": 0.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_guard_clears_an_official_and_lets_the_pair_box_merge():
    rows = [  # frame 0: two fighters side by side, overlapping
        {"frame_idx": 0, "raw_label": "red", "cost": .1, "cost_red": .1, "cost_blue": .6,
         "cost_ref": .9, "x1": 0, "y1": 0, "x2": 100, "y2": 300},
        {"frame_idx": 0, "raw_label": "blue", "cost": .1, "cost_red": .6, "cost_blue": .1,
         "cost_ref": .9, "x1": 80, "y1": 0, "x2": 180, "y2": 300},
        # frame 1: one wide box for the pair, plus a seated official that looks like a ref
        {"frame_idx": 1, "raw_label": "blue", "cost": .2, "cost_red": .5, "cost_blue": .2,
         "cost_ref": .9, "x1": 0, "y1": 0, "x2": 200, "y2": 300},
        {"frame_idx": 1, "raw_label": "red", "cost": .5, "cost_red": .5, "cost_blue": .8,
         "cost_ref": .2, "x1": 500, "y1": 100, "x2": 600, "y2": 300},
    ]
    cfg = {**CFG["identity"], "assign": {**CFG["identity"]["assign"], "weak_corner_cost": 0.4}}
    out = smooth_labels(guard_frame(rows), 30.0, cfg)
    assert out.sort_index()["label"].tolist() == ["red", "blue", "merged", "none"]
    off = {**cfg, "assign": {**cfg["assign"], "weak_corner_cost": None}}
    out = smooth_labels(guard_frame(rows), 30.0, off)
    assert out.sort_index()["label"].tolist()[2:] == ["blue", "red"]   # 1.3 behaviour


def test_guard_keeps_a_fighter_with_a_middling_cost():
    rows = [{"frame_idx": 0, "raw_label": "red", "cost": .45, "cost_red": .45, "cost_blue": .7,
             "cost_ref": .9, "x1": 0, "y1": 0, "x2": 100, "y2": 300}]
    cfg = {**CFG["identity"], "assign": {**CFG["identity"]["assign"], "weak_corner_cost": 0.4}}
    assert smooth_labels(guard_frame(rows), 30.0, cfg)["label"].tolist() == ["red"]


# ---------------------------------------------------------------- smoothing


def smooth_track_old(frames, labels, fps, cfg):
    """The 1.3 O(n^2) version, kept here as the reference."""
    sm = cfg["smooth"]
    w = sm["smooth_s"] * fps
    segs = split_segments(labels, sm["split_frames"])
    out = []
    for f, s, own in zip(frames, segs, labels):
        votes = {}
        for j in range(len(frames)):
            if segs[j] == s and abs(frames[j] - f) <= w:
                votes[labels[j]] = votes.get(labels[j], 0) + 1
        top = max(votes.values())
        winners = [k for k, v in votes.items() if v == top]
        out.append(own if own in winners else winners[0])
    return out


def test_sliding_smoothing_matches_the_old_version():
    rnd = random.Random(0)
    cfg = CFG["identity"]
    for _ in range(300):
        n = rnd.randint(1, 120)
        frames = sorted(rnd.sample(range(400), n))
        # runs of labels with noise, so segments split sometimes
        labels, lab = [], rnd.choice("rbn")
        for _ in range(n):
            if rnd.random() < 0.08:
                lab = rnd.choice("rbnm")
            labels.append(lab if rnd.random() > 0.2 else rnd.choice("rbnm"))
        for fps in (30.0, 15.0):
            assert smooth_track(frames, labels, fps, cfg) == smooth_track_old(frames, labels,
                                                                               fps, cfg)


# ---------------------------------------------------------------- CLI selection, baseline


def test_extract_all_skips_test_and_runs_train_before_val():
    from deepfoul.cli import _extract_rows

    rows = _extract_rows([], None)
    splits = [r["split"].strip() for r in rows]
    assert "test" not in splits
    assert splits == sorted(splits, key=lambda s: {"train": 0, "val": 1}[s])
    assert splits.count("train") >= 1 and splits.count("val") >= 1
    # explicit ids still reach test fights (6.0)
    test_ids = [r["fight_id"] for r in load_csv() if r["split"].strip() == "test"]
    assert _extract_rows(test_ids[:1], None)[0]["split"].strip() == "test"


def load_csv():
    from deepfoul.io.fetch import read_fights

    return read_fights(CFG["paths"]["labels"] / "fights.csv")


def test_baseline_verdicts():
    base = {"both_or_merged_pct": "90", "corner_jumps": "3", "status": "pass",
            "fix_section": ""}
    s = {"both_or_merged_pct": 89.0, "flips": 5}
    assert baseline_verdict(s, base) == "ok"
    assert baseline_verdict({**s, "both_or_merged_pct": 86.0}, base) == "REGRESSION"
    assert baseline_verdict({**s, "flips": 6}, base) == "REGRESSION"
    known = {**base, "corner_jumps": "10", "status": "known_fail", "fix_section": "4.1"}
    assert baseline_verdict({**s, "flips": 9}, known) == "known fail (4.1)"
    assert baseline_verdict({**s, "flips": 1}, known) == "FIXED? check by eye"
    assert baseline_verdict(s, None) == "no baseline"


def test_keep_awake_is_safe():
    from deepfoul.io.power import keep_awake

    with keep_awake() as ok:
        assert ok in (True, False)


def test_guard_spares_a_downed_fighter_that_was_confidently_that_corner_before():
    """golden_05: Petrosyan crumpled on the mat looks clothed and ref-like for a moment."""
    stand = {"raw_label": "blue", "cost": .1, "cost_red": .7, "cost_blue": .1, "cost_ref": .9,
             "x1": 500, "y1": 0, "x2": 600, "y2": 300}
    down = {"raw_label": "blue", "cost": .6, "cost_red": .8, "cost_blue": .6, "cost_ref": .4,
            "x1": 400, "y1": 200, "x2": 700, "y2": 330}
    other = {"raw_label": "red", "cost": .05, "cost_red": .05, "cost_blue": .6, "cost_ref": .7,
             "x1": 0, "y1": 0, "x2": 100, "y2": 330}
    rows = [{**other, "frame_idx": f, "track_id": "1-0"} for f in range(30)]
    rows += [{**stand, "frame_idx": f, "track_id": "1-1"} for f in range(10)]
    rows += [{**down, "frame_idx": f, "track_id": "1-1"} for f in range(10, 30)]
    base = {"shot_id": 1, "fallback": False, "margin": 0.0}
    df = pd.DataFrame([{**base, **r} for r in rows])
    cfg = {**CFG["identity"], "assign": {**CFG["identity"]["assign"], "weak_corner_cost": 0.4}}
    out = smooth_labels(df, 30.0, cfg)
    assert (out.loc[out["track_id"] == "1-1", "label"] == "blue").all()
    # the same box with no confident past (an untracked fragment) is cleared
    lone = pd.DataFrame([{**base, **other, "frame_idx": 0, "track_id": "1-0"},
                         {**base, **down, "frame_idx": 0, "track_id": ""}])
    assert smooth_labels(lone, 30.0, cfg).sort_index()["label"].tolist() == ["red", "none"]
