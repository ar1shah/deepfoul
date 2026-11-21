"""1.5 overlay renderer, cache range reads, contact-sheet frame picks, report flags, bench math."""

import numpy as np
import pandas as pd
import pytest

from deepfoul.bench import summarize_timing
from deepfoul.io.fights import parse_time
from deepfoul.viz.extract_report import check_flags
from deepfoul.viz.overlay import (BLUE, EYES, FINGERTIPS, LOWER, RED, REF, draw_frame,
                                  read_range, sheet_frames)

NAMES = {"red": "Battle", "blue": "Loosa"}


def blank():
    return np.zeros((720, 1280, 3), np.uint8)


def row(label, box, kpts=None, scores=None):
    x1, y1, x2, y2 = box
    return {"frame_idx": 0, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "corner": label,
            "fallback": False, "kpts": kpts, "kpt_scores": scores}


def has_color(img, color, region=None):
    sub = img if region is None else img[region[1]:region[3], region[0]:region[2]]
    return bool((sub == np.array(color, np.uint8)).all(axis=-1).any())


def test_keypoint_indices_match_the_coco_wholebody_layout():
    assert FINGERTIPS == [95, 99, 103, 107, 111, 116, 120, 124, 128, 132]
    assert EYES[:2] == [1, 2] and EYES[2:] == list(range(59, 71))
    assert LOWER == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def test_red_and_blue_boxes_are_drawn_with_their_colors():
    img = draw_frame(blank(), pd.DataFrame([row("red", (100, 100, 300, 600)),
                                            row("blue", (700, 100, 900, 600))]), NAMES)
    assert has_color(img, RED, (90, 90, 310, 610))
    assert has_color(img, BLUE, (690, 90, 910, 610))


def test_ref_is_hidden_unless_asked_and_none_is_grey():
    rows = pd.DataFrame([row("ref", (100, 100, 300, 600)), row("none", (700, 100, 900, 600))])
    ref_area, none_area = (slice(80, 620), slice(80, 320)), (slice(80, 620), slice(680, 920))
    img = draw_frame(blank(), rows, NAMES)
    assert not img[ref_area].any()          # ref hidden
    assert img[none_area].any()             # thin anti-aliased grey box on the unlabelled one
    img = draw_frame(blank(), rows, NAMES, show_ref=True, show_none=False)
    assert img[ref_area].any() and has_color(img, REF)   # filled tag in the ref color
    assert not img[none_area].any()


def test_low_confidence_keypoints_are_never_drawn():
    k = np.zeros((133, 2), np.float32)
    s = np.zeros(133, np.float32)
    tip = FINGERTIPS[1]
    k[tip] = (640, 360)
    s[tip] = 0.9
    box = (400, 50, 900, 700)
    img = draw_frame(blank(), pd.DataFrame([row("red", box, k.ravel(), s)]), NAMES, skeleton=True,
                     kpt_thr=0.3)
    assert img[360, 640].any()                         # confident fingertip: a dot
    s[tip] = 0.1                                       # same point, masked
    img = draw_frame(blank(), pd.DataFrame([row("red", box, k.ravel(), s)]), NAMES, skeleton=True,
                     kpt_thr=0.3)
    assert not img[340:380, 620:660].any()


def test_skeleton_only_on_fighters():
    k = np.full((133, 2), 640, np.float32)
    s = np.ones(133, np.float32)
    img = draw_frame(blank(), pd.DataFrame([row("ref", (400, 50, 900, 700), k.ravel(), s)]), NAMES,
                     skeleton=True, show_ref=False)
    assert not img.any()


def test_read_range_filters_on_frame_idx(tmp_path):
    df = pd.DataFrame({"frame_idx": np.arange(100), "x": np.arange(100.0)})
    p = tmp_path / "f.parquet"
    df.to_parquet(p, row_group_size=10)
    got = read_range(p, 25, 40)
    assert got["frame_idx"].tolist() == list(range(25, 40))


def test_sheet_frames_merged_and_single():
    frames = pd.DataFrame({"frame_idx": [0, 1, 2, 3, 4]})
    boxes = pd.DataFrame({"frame_idx": [0, 0, 1, 2, 2, 3],
                          "corner": ["red", "blue", "merged", "red", "ref", "none"]})
    assert sheet_frames(boxes, frames, "merged").tolist() == [1]
    assert sheet_frames(boxes, frames, "single").tolist() == [2]
    with pytest.raises(ValueError):
        sheet_frames(boxes, frames, "other")


def test_parse_time():
    assert parse_time("14:03.8") == pytest.approx(843.8)
    assert parse_time("843.8") == pytest.approx(843.8)
    assert parse_time("1:02:03") == 3723.0
    with pytest.raises(ValueError):
        parse_time("14:3x")


def test_check_flags():
    check = {"bom_of_visible_min": 82, "visible_min": 88}
    assert check_flags({"bom_of_visible_pct": 90, "any_fighter_pct": 97}, check) == ""
    assert check_flags({"bom_of_visible_pct": 78, "any_fighter_pct": 92}, check) == "**check**"
    assert check_flags({"bom_of_visible_pct": 81, "any_fighter_pct": 83},
                       check) == "**check**, low visible"
    assert check_flags({}, check) == ""


def test_bench_summary_math():
    t = {"frames": 100, "boxes": 250, "det_s": 5.0, "pose_s": 4.0, "desc_s": 6.0,
         "wall_s": 10.0, "describe_workers": 3}
    s = summarize_timing(t)
    assert s["det_fps"] == pytest.approx(20.0) and s["det_ms"] == pytest.approx(50.0)
    assert s["pose_fps"] == pytest.approx(25.0)
    assert s["pose_ms_per_box"] == pytest.approx(16.0)
    assert s["boxes_per_frame"] == pytest.approx(2.5)
    assert s["pass_fps"] == pytest.approx(10.0) and s["desc_ms"] == pytest.approx(60.0)
