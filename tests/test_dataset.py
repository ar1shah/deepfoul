"""Dataset builder (2.3): splits, windowing, note categories, audit, keypoint loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deepfoul.config import load_config
from deepfoul.dataset.build import audit, build, categorize, grid_inside, subtract, union
from deepfoul.dataset.windows import coverage_arrays, window_keypoints
from deepfoul.labeling.queues import read_csv

CFG = load_config()
D = CFG["dataset"]
CATS = D["note_categories"]
W = D["window_s"]

FIGHTS = {
    "ufc1_a_b": {"fight_id": "ufc1_a_b", "split": "train", "action_start_s": "0", "action_end_s": "1000"},
    "ufc2_c_d": {"fight_id": "ufc2_c_d", "split": "val", "action_start_s": "0", "action_end_s": "1000"},
    "ufc3_e_f": {"fight_id": "ufc3_e_f", "split": "test", "action_start_s": "0", "action_end_s": "1000"},
}


def ev(eid, fid, s, e, foul="clean", off="none", contact="no", notes="", source="mined"):
    return {"event_id": eid, "fight_id": fid, "start_s": str(s), "end_s": str(e),
            "foul_type": foul, "offender": off, "contact": contact, "ref_stopped": "no",
            "speed": "live", "source": source, "notes": notes}


def item(fid, t, i):
    return {"item_id": f"{fid}_{i}", "fight_id": fid, "t_s": float(t), "score": "", "reason": ""}


def run(events, items):
    return build(events, FIGHTS, items, CFG).windows


# ---------- note categories ----------
@pytest.mark.parametrize("note,cat", [
    ("", "clean"),
    ("round break", "round break"),
    ("Round Break", "round break"),
    ("round break, replay being shown", "replay"),
    ("ref warning, recovery period for foul", "foul timeout"),
    ("round break, recovery after foul", "foul timeout"),
    ("round break after foul commited", "foul timeout"),
    ("round break, fight waved off after foul", "foul timeout"),
    ("foul timeout", "foul timeout"),
    ("ref warning", "stoppage"),
    ("slip", "stoppage"),
    ("cut", "stoppage"),
    ("glove touch", "stoppage"),
    ("other", "other stoppage"),
    ("big uppercut lands", "clean"),        # whole words: "uppercut" is not "cut"
    ("slipped a jab", "clean"),
])
def test_categorize(note, cat):
    assert categorize(note, CATS)[0] == cat


def test_every_real_clean_note_has_a_sensible_category():
    rows = read_csv(CFG["paths"]["labels"] / "events.csv")
    for r in rows:
        if r["foul_type"] != "clean" or not r["notes"].strip():
            continue
        name, use = categorize(r["notes"], CATS)
        if "replay" in r["notes"] or "foul" in r["notes"] or "recovery" in r["notes"]:
            assert use == "exclude", r
        assert name != "clean", r          # every noted clean row today is a stoppage of some kind


# ---------- splits ----------
def test_test_fights_never_get_windows():
    events = [ev("e1", "ufc3_e_f", 100, 100.8, "eye_poke", "red", "yes"),
              ev("e2", "ufc1_a_b", 100, 100.8, "low_blow", "blue", "yes")]
    items = [item("ufc3_e_f", 100, 1), item("ufc1_a_b", 100, 1)]
    b = build(events, FIGHTS, items, CFG)
    assert b.dropped_test == ["ufc3_e_f"]
    assert set(b.windows["fight_id"]) == {"ufc1_a_b"}
    assert not (b.windows["split"] == "test").any()
    assert not any(c.fight_id == "ufc3_e_f" for c in b.clips)


def test_unknown_fight_is_an_error():
    with pytest.raises(ValueError, match="not in fights.csv"):
        build([ev("e1", "ufc9_x_y", 1, 2)], FIGHTS, [], CFG)


def test_each_fight_one_split_from_fights_csv():
    events = [ev("e1", "ufc1_a_b", 100, 100.8, "eye_poke", "red", "yes"),
              ev("e2", "ufc2_c_d", 200, 200.8, "low_blow", "blue", "yes")]
    df = run(events, [])
    assert df.groupby("fight_id")["split"].unique().map(list).to_dict() == \
           {"ufc1_a_b": ["train"], "ufc2_c_d": ["val"]}


# ---------- windows ----------
def test_positive_windows_hold_half_the_event():
    events = [ev("e1", "ufc1_a_b", 100.1, 100.9, "low_blow", "blue", "yes")]
    df = run(events, [])
    pos = df[df["label"] == "low_blow"]
    assert len(pos) in (8, 9) and (pos["offender"] == "blue").all()   # 9 when 50% lands on both ends
    for t0, t1 in zip(pos["t0"], pos["t1"]):
        assert t1 - t0 == pytest.approx(W)
        assert min(t1, 100.9) - max(t0, 100.1) >= 0.4 - 1e-6
    assert np.allclose(np.diff(sorted(pos["t0"])), D["pos_stride_s"])


def test_clean_only_inside_reviewed_clips_and_outside_buffers():
    events = [ev("c1", "ufc1_a_b", 199.5, 200.5),                       # clean clip at 200
              ev("f1", "ufc1_a_b", 300.0, 300.6, "eye_poke", "red", "yes")]  # foul clip at 301
    items = [item("ufc1_a_b", 200, 1), item("ufc1_a_b", 301, 2), item("ufc1_a_b", 500, 3)]
    df = run(events, items)
    clean = df[df["label"] == "clean"]
    in_clip = lambda t0, t1, c: c - 4 <= t0 and t1 <= c + 4
    assert all(in_clip(a, b, 200) or in_clip(a, b, 301) for a, b in zip(clean["t0"], clean["t1"]))
    assert ((clean["t0"] >= 196) & (clean["t1"] <= 204)).sum() == 13   # the whole 8 s clip
    # unlabeled time in the foul clip is clean, except the 2 s buffer around the foul
    buf = D["buffer_s"]["eye_poke"]
    foul_clip = clean[(clean["t0"] >= 297)]
    assert len(foul_clip) > 0
    assert ((foul_clip["t1"] <= 300.0 - buf) | (foul_clip["t0"] >= 300.6 + buf)).all()
    # item 3 has no label: never reviewed, no windows
    assert not ((df["t0"] > 450)).any()


def test_excluded_and_stoppage_clips():
    events = [ev("c1", "ufc1_a_b", 199.5, 200.5, notes="round break, replay being shown"),
              ev("c2", "ufc1_a_b", 399.5, 400.5, notes="ref warning, recovery period after foul"),
              ev("c3", "ufc1_a_b", 599.5, 600.5, notes="slip")]
    items = [item("ufc1_a_b", t, i) for i, t in enumerate((200, 400, 600))]
    b = build(events, FIGHTS, items, CFG)
    df = b.windows
    assert not ((df["t0"] < 450)).any()                    # replay + timeout clips: nothing
    hn = df[df["label"] == "hard_negative"]
    assert len(hn) == 13 and (hn["source_kind"] == "stoppage_clip").all()
    assert (hn["weight"] == D["hard_neg_weight"]).all() and (hn["event_id"] == "c3").all()
    assert (df["label"] == "clean").sum() == 0
    assert sorted(c.use for c in b.clips) == ["exclude", "exclude", "hard_negative"]


def test_hard_negative_events_and_priority():
    # an open hand right before a low blow: windows that fit both are low blow; windows that fit
    # the open hand but only graze the low blow are dropped
    events = [ev("o1", "ufc1_a_b", 98.0, 99.0, "open_hand", "red"),
              ev("l1", "ufc1_a_b", 99.2, 100.0, "low_blow", "blue", "yes"),
              ev("h1", "ufc1_a_b", 150.0, 150.5, "hard_negative", "none", "yes")]
    df = run(events, [])
    lb, oh = df[df["label"] == "low_blow"], df[df["label"] == "open_hand"]
    for t0, t1 in zip(oh["t0"], oh["t1"]):
        assert min(t1, 100.0) - max(t0, 99.2) <= 0       # never touches the low blow
    assert len(lb) in (8, 9) and len(oh) > 0
    hn = df[df["label"] == "hard_negative"]
    assert len(hn) in (8, 9) and (hn["weight"] == D["hard_neg_weight"]).all()


def test_full_fight_labels_review_the_whole_window():
    fights = {"ufc1_a_b": {**FIGHTS["ufc1_a_b"], "action_start_s": "100", "action_end_s": "140"}}
    events = [ev("f1", "ufc1_a_b", 110, 110.5, "eye_poke", "red", "yes", source="full_fight"),
              ev("c1", "ufc1_a_b", 125, 130, notes="round break", source="full_fight")]
    df = build(events, fights, [], CFG).windows
    clean = df[df["label"] == "clean"]
    assert clean["t0"].min() >= 100 and clean["t1"].max() <= 140
    assert not ((clean["t0"] < 130) & (clean["t1"] > 125)).any()      # the round break span
    assert ((clean["t0"] >= 112.5) & (clean["t1"] <= 125)).any()


def test_window_ids_unique_and_deterministic():
    events = [ev("c1", "ufc1_a_b", 199.5, 200.5),
              ev("f1", "ufc1_a_b", 201, 201.6, "eye_poke", "red", "yes")]
    items = [item("ufc1_a_b", 200, 1)]
    a, b = run(events, items), run(events, items)
    assert a["window_id"].is_unique
    pd.testing.assert_frame_equal(a, b)


def test_interval_helpers():
    assert union([(5, 6), (0, 2), (1, 3)]) == [(0, 3), (5, 6)]
    assert subtract([(0, 10)], [(2, 3), (8, 12)]) == [(0, 2), (3, 8)]
    assert list(grid_inside(0.1, 3.0, 0.5, 2.0)) == [0.5, 1.0]


# ---------- audit ----------
def test_audit_flags():
    events = [ev("e1", "ufc1_a_b", 100, 100.1, "eye_poke", "red", "yes"),          # too short
              ev("e2", "ufc1_a_b", 200, 200.5), ev("e3", "ufc1_a_b", 200, 200.5),   # duplicate
              ev("e4", "ufc1_a_b", 300, 300.5, off="red"),                          # clean + offender
              ev("e5", "ufc1_a_b", 900, 900.5)]                                     # outside clips
    items = [item("ufc1_a_b", 100, 1), item("ufc1_a_b", 200, 2), item("ufc1_a_b", 300, 3)]
    b = build(events, FIGHTS, items, CFG)
    known = [{"fight_id": "ufc1_a_b", "live_s": "500", "foul_type": "low_blow"}]
    w = "\n".join(audit(events, known, b.clips, D))
    assert "odd length 0.10 s: e1" in w
    assert "duplicate: e3 = e2" in w
    assert "clean with offender/contact: e4" in w
    assert "clean row outside every reviewed clip: e5" in w
    assert "known foul without a label: ufc1_a_b 08:20.00 low_blow" in w
    assert "foul label not in known_fouls.csv: e1" in w


# ---------- keypoint loader ----------
def test_window_keypoints_and_coverage(tmp_path):
    fid = "ufc1_a_b"
    t = np.round(np.arange(10) / 10, 3)
    pd.DataFrame({"t": t}).to_parquet(tmp_path / f"{fid}_frames.parquet")
    kp = lambda v: np.full(266, v, np.float32).tolist()
    rows = [  # frame 0: red + blue; frame 1: merged (raw blue) + red; frame 2: nothing; 3: ref
        {"t": t[0], "corner": "red", "raw_label": "red", "kpts": kp(1), "kpt_scores": [0.9] * 133, "score": 0.9},
        {"t": t[0], "corner": "blue", "raw_label": "blue", "kpts": kp(2), "kpt_scores": [0.8] * 133, "score": 0.9},
        {"t": t[1], "corner": "merged", "raw_label": "blue", "kpts": kp(3), "kpt_scores": [0.7] * 133, "score": 0.9},
        {"t": t[1], "corner": "red", "raw_label": "red", "kpts": kp(4), "kpt_scores": [0.6] * 133, "score": 0.9},
        {"t": t[3], "corner": "ref", "raw_label": "ref", "kpts": kp(5), "kpt_scores": [0.5] * 133, "score": 0.9},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / f"{fid}.parquet")
    out = window_keypoints(tmp_path, fid, 0.0, 0.35)
    assert len(out["t"]) == 4
    assert out["red"]["kpts"][0, 0, 0] == 1 and out["blue"]["kpts"][0, 0, 0] == 2
    assert out["blue"]["kpts"][1, 0, 0] == 3 and out["merged"][1]     # merged -> raw_label side
    assert out["red"]["kpts"][1, 0, 0] == 4
    assert np.isnan(out["red"]["kpts"][2]).all() and out["red"]["scores"][3].sum() == 0
    ft, fig = coverage_arrays(tmp_path, fid)
    assert len(ft) == 10 and list(fig) == [t[0], t[1]]
    assert coverage_arrays(tmp_path, "ufc9_x_y") is None
