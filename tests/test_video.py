"""Frame reader and shot detection on a synthetic video with known cuts."""

import shutil

import numpy as np
import pytest

from deepfoul.io.previews import clip_name, fmt_ts, frame_label, golden_selection, pick_cuts
from deepfoul.io.shots import (detect_shots, first_frame_at, load_shots, save_shots,
                               shot_index_at, shot_index_at_frame, shots_from_cuts, stale_reason)
from deepfoul.io.video import ffmpeg_frame, iter_fight_frames, iter_frames, video_info

FPS, N, W, H = 25, 150, 192, 128
CUTS = [50, 100]   # first frame of each new shot
SHOTS_CFG = {"detector": "adaptive", "adaptive_threshold": 3.0, "min_content_val": 15.0,
             "min_scene_len_s": 0.5}


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """3 shots (different textures/colors) with a slowly moving block, cuts at CUTS."""
    import cv2

    path = tmp_path_factory.mktemp("vid") / "synthetic.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(0)
    bases = [np.clip(rng.normal(m, 25, (H, W, 3)), 0, 255).astype(np.uint8)
             for m in ((40, 40, 200), (200, 60, 40), (60, 200, 60))]
    for i in range(N):
        img = bases[sum(i >= c for c in CUTS)].copy()
        x = (i * 2) % (W - 20)
        img[50:70, x:x + 20] = 255
        wr.write(img)
    wr.release()
    return path


def test_reader_order_timestamps_and_window(video):
    frames = list(iter_frames(video))
    assert [f.idx for f in frames] == list(range(N))
    assert all(abs(f.t - f.idx / FPS) < 1e-3 for f in frames)
    part = list(iter_frames(video, 2.0, 4.0))          # half-open [2, 4)
    assert part[0].idx == 50 and part[-1].idx == 99


def test_reader_stride_and_offgrid_start(video):
    part = list(iter_frames(video, 2.01, 3.0, stride=3))
    assert part[0].idx == 51                            # first frame at or after 2.01 s
    assert [f.idx for f in part] == list(range(51, 75, 3))


def test_seek_matches_linear_decode(video):
    linear = {f.idx: f.image for f in iter_frames(video)}
    seeked = next(iter_frames(video, 4.6))
    assert seeked.idx == 115 and np.array_equal(seeked.image, linear[115])


def test_fight_frames_clamped_to_window(video):
    row = {"fight_id": video.stem, "action_start_s": "1", "action_end_s": "3"}
    frames = list(iter_fight_frames(row, video.parent, start_s=0.0, end_s=10.0))
    assert frames[0].idx == 25 and frames[-1].idx == 74


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_reader_matches_ffmpeg(video):
    info = video_info(video)
    assert info["flags"] == [] and info["width"] == W
    f = next(iter_frames(video, 3.33))
    ref = ffmpeg_frame(video, 3.33, W, H)
    assert np.abs(f.image.astype(int) - ref.astype(int)).mean() < 1.0


def test_detect_shots_finds_cuts_inside_window(video):
    doc = detect_shots(video, 0.0, None, SHOTS_CFG)
    starts = [s["start_frame"] for s in doc["shots"]]
    assert len(starts) == 3 and starts[0] == 0
    assert all(abs(a - b) <= 1 for a, b in zip(starts[1:], CUTS))
    assert doc["shots"][-1]["end_frame"] == N
    assert doc["detector"]["min_scene_len_frames"] == round(0.5 * FPS)
    inner = detect_shots(video, 2.4, 5.0, SHOTS_CFG)    # the cut at frame 50 is outside
    assert [s["start_frame"] for s in inner["shots"]] == [60, inner["shots"][1]["start_frame"]]
    assert abs(inner["shots"][1]["start_frame"] - 100) <= 1


def test_shots_half_open_lookup():
    shots = shots_from_cuts([5, 50, 100, 200], 10, 150, 25.0)
    assert [(s["start_frame"], s["end_frame"]) for s in shots] == [(10, 50), (50, 100), (100, 150)]
    assert shot_index_at_frame(shots, 49) == 0
    assert shot_index_at_frame(shots, 50) == 1        # frame exactly at a cut -> new shot
    assert shot_index_at_frame(shots, 9) is None and shot_index_at_frame(shots, 150) is None
    assert shot_index_at(shots, 2.0) == 1             # 50 / 25 s, exactly at the cut
    assert shot_index_at(shots, 1.999) == 0
    assert shot_index_at(shots, 6.0) is None


def test_first_frame_at():
    assert first_frame_at(10.0, 30000 / 1001) == 300  # 9.977 s < 10 <= 10.01 s
    assert first_frame_at(2.0, 25) == 50 and first_frame_at(2.01, 25) == 51


def test_cache_roundtrip_and_staleness(tmp_path):
    doc = {"fight_id": "x", **{"fps": 25.0, "window": [0.0, None], "scenedetect_version": None,
                               "detector": None, "shots": []}}
    from deepfoul.io import shots as shots_mod
    doc["scenedetect_version"] = shots_mod.scenedetect.__version__
    doc["detector"] = shots_mod.detector_params(SHOTS_CFG, 25.0)
    save_shots(doc, tmp_path / "x_shots.json")
    loaded = load_shots(tmp_path / "x_shots.json")
    assert loaded == doc
    assert stale_reason(loaded, (0.0, None), SHOTS_CFG) is None
    assert "detector" in stale_reason(loaded, (0.0, None), {**SHOTS_CFG, "adaptive_threshold": 2.5})
    assert "window" in stale_reason(loaded, (5.0, None), SHOTS_CFG)


def test_per_fight_override(video):
    from deepfoul.io.shots import detector_params, shots_cfg_for

    cfg = {**SHOTS_CFG, "content_threshold": 27.0, "per_fight": {"f1": {"detector": "content"}}}
    assert shots_cfg_for(cfg, "f2")["detector"] == "adaptive"
    c = shots_cfg_for(cfg, "f1")
    assert detector_params(c, 25.0)["name"] == "content" and "per_fight" not in c
    doc = detect_shots(video, 0.0, None, c)
    assert all(abs(a - b) <= 1 for a, b in zip([s["start_frame"] for s in doc["shots"][1:]], CUTS))
    with pytest.raises(ValueError):
        detector_params({**SHOTS_CFG, "detector": "nope"}, 25.0)


def test_labels_and_names():
    assert fmt_ts(727.48) == "12:07.48" and fmt_ts(0.0) == "00:00.00"
    assert fmt_ts(3725.5) == "62:05.50"
    assert frame_label(727.48, 36274) == "12:07.48  f36274"
    assert clip_name("ufc276_munhoz_omalley", 727.0, "eye_poke") == \
        "ufc276_munhoz_omalley_12-07_eye_poke.mp4"
    assert pick_cuts(list(range(100)), 3) == [0, 50, 99] and pick_cuts([1, 2], 5) == [1, 2]


def test_golden_selection_filters():
    fights = [{"fight_id": "a", "split": "train"}, {"fight_id": "b", "split": "val"},
              {"fight_id": "c", "split": "train"}]
    fouls = [{"fight_id": "a", "visibility": "clear_live"},
             {"fight_id": "a", "visibility": "hard_live"},
             {"fight_id": "b", "visibility": "clear_live"},
             {"fight_id": "c", "visibility": "clear_live"}]
    got = golden_selection(fights, fouls, exclude={"c"})
    assert got == [fouls[0]]
