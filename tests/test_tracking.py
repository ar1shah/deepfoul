"""Detection parsing, box filters and the shot-aware tracking loop, without video or a GPU."""

from dataclasses import dataclass

import numpy as np
import pytest

from deepfoul.perception.detect import parse_dets
from deepfoul.perception.filters import drop_reasons
from deepfoul.perception.track import guard_frames, make_tracker, track_stream

H, W = 1000, 2000
FCFG = {"min_height_frac": 0.10, "rel_min_height_frac": 0.4, "ref_score_min": 0.5,
        "min_bottom_frac": 0.35, "border_margin_frac": 0.01, "edge_max_height_frac": 0.25,
        "max_people": 3}
BT = {"lost_track_buffer": 30, "track_activation_threshold": 0.5, "minimum_consecutive_frames": 2,
      "minimum_iou_threshold": 0.1, "high_conf_det_threshold": 0.5}


def test_parse_dets_scales_clips_and_thresholds():
    dets = np.array([[[10, 20, 110, 220, 0.9], [0, 0, 50, 50, 0.05], [600, 300, 800, 400, 0.2]]])
    boxes, scores = parse_dets(dets, ratio=0.5, score_min=0.1, frame_hw=(700, 1500))
    assert scores.tolist() == pytest.approx([0.9, 0.2])
    assert boxes[0].tolist() == [20, 40, 220, 440]
    assert boxes[1].tolist() == [1200, 600, 1500, 700]  # clipped to the frame


def test_parse_dets_rejects_output_without_nms():
    with pytest.raises(ValueError):
        parse_dets(np.zeros((1, 10, 85)), 1.0, 0.1, (H, W))


def test_drop_reasons_each_rule():
    boxes = np.array([
        [900, 100, 1100, 900],    # fighter: kept
        [500, 300, 540, 380],     # small (8% tall) and rel_small
        [1300, 100, 1400, 300],   # high: bottom at 30%; rel_small (20% vs 80%)
        [0, 600, 100, 800],       # at left edge, 20% tall -> edge (and rel_small)
        [1500, 150, 1700, 850],   # 70% tall: kept
    ], dtype=float)
    scores = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    r = drop_reasons(boxes, scores, (H, W), FCFG)
    assert r[0] == "" and r[4] == ""
    assert r[1] == "small+rel_small"
    assert r[2] == "rel_small+high"
    assert r[3] == "rel_small+edge"


def test_rel_small_ignores_low_score_giants():
    boxes = np.array([[0, 0, 2000, 1000], [900, 300, 1000, 700]], dtype=float)  # junk + 40% box
    r = drop_reasons(boxes, np.array([0.2, 0.9]), (H, W), FCFG)
    assert r[1] == ""


def test_max_people_keeps_tallest():
    boxes = np.array([[i * 300, 1000 - h, i * 300 + 100, 1000] for i, h in
                      enumerate([500, 900, 700, 600, 800])], dtype=float)
    r = drop_reasons(boxes, np.full(5, 0.9), (H, W), FCFG)
    assert r == ["max_people", "", "", "max_people", ""]


def test_guard_frames():
    shots = [{"shot_id": 0, "start_frame": 0, "end_frame": 10},
             {"shot_id": 1, "start_frame": 10, "end_frame": 20}]
    assert guard_frames(shots, 1) == {9, 10}
    assert guard_frames(shots, 2) == {8, 9, 10, 11}


@dataclass
class F:
    idx: int
    t: float
    image: np.ndarray


def test_track_stream_resets_and_guards_at_cuts():
    shots = [{"shot_id": 0, "start_frame": 0, "end_frame": 10},
             {"shot_id": 1, "start_frame": 10, "end_frame": 20}]
    img = np.zeros((H, W, 3), np.uint8)

    def detect(_):  # one person standing still, plus a crowd head
        return np.array([[900, 100, 1100, 900], [100, 50, 140, 90]], float), np.array([0.9, 0.9])

    tracker = make_tracker(BT, 30.0)
    out = list(track_stream((F(i, i / 30, img) for i in range(20)), shots, detect, tracker, FCFG))
    assert [r.guard for r in out] == [i in (9, 10) for i in range(20)]
    fighter = [r.rows[0]["track_id"] for r in out]
    crowd = [r.rows[1] for r in out]
    assert all(not c["kept"] and c["track_id"] == "" for c in crowd)
    assert fighter[9] == fighter[10] == ""                       # guard frames: not tracked
    assert set(fighter[1:9]) == {"0-0"} and fighter[0] == ""     # confirmed on the 2nd frame
    assert set(fighter[12:]) == {"1-0"} and fighter[11] == ""    # reset: new shot, ids restart
    assert len(out[0].rows) == 2 and out[0].rows[0]["shot_id"] == 0
