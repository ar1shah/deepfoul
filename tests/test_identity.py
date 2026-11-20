"""Identity logic on synthetic data: histograms, clothed shins, assignment, merge, smoothing."""

import cv2
import numpy as np
import pandas as pd
import pytest

from deepfoul.config import load_config
from deepfoul.perception.appearance import (Appearance, clothed_score, describe, histogram,
                                            seed_fraction, seed_histogram, similarity)
from deepfoul.perception.identity import (BoxObs, Profile, assign_frame, build_profile,
                                          check_lookahead, smooth_labels, smooth_track,
                                          split_segments)

CFG = load_config()["identity"]
SAT, VMIN = CFG["sat_min"], CFG["v_min"]


def patch(bgr, n=400):
    """n pixels of one BGR color -> HSV pixel array."""
    img = np.full((1, n, 3), bgr, np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3)


def hist_of(bgr):
    return histogram(patch(bgr), SAT, VMIN)


BLACK, NAVY, RED, GRAY, WHITE, SKIN = (15, 15, 15), (90, 30, 20), (30, 30, 200), (130, 130, 130), (240, 240, 240), (120, 150, 200)


def test_histograms_separate_dark_and_colored():
    assert similarity(hist_of(BLACK), hist_of(BLACK)) == pytest.approx(1.0)
    assert similarity(hist_of(BLACK), hist_of(NAVY)) < 0.5      # navy is chromatic, black is not
    assert similarity(hist_of(GRAY), hist_of(WHITE)) < 0.5      # achromatic, different value
    assert similarity(hist_of(RED), hist_of(BLACK)) == pytest.approx(0.0)


def test_seed_histograms_cover_their_colors():
    for name, bgr in (("black", BLACK), ("red", RED), ("gray", GRAY), ("white", WHITE)):
        seed = seed_histogram(name, CFG["colors"], SAT, VMIN)
        assert seed_fraction(hist_of(bgr), seed) > 0.9, name
    assert seed_fraction(hist_of(RED), seed_histogram("black", CFG["colors"], SAT, VMIN)) < 0.1


def test_clothed_score():
    # fighter: black shorts, bare shins that look like the face skin
    assert clothed_score(1.0, hist_of(BLACK), hist_of(SKIN)) == pytest.approx(0.0)
    # ref: black pants from hip to ankle
    assert clothed_score(0.0, hist_of(BLACK), hist_of(BLACK)) == pytest.approx(1.0)
    # no face visible: the fabric cue alone still separates them
    assert clothed_score(None, hist_of(BLACK), hist_of(SKIN)) < 0.2 < 0.8 < clothed_score(
        None, hist_of(BLACK), hist_of(BLACK))
    assert clothed_score(None, None, None) is None


def person_image(shorts_bgr, legs_bgr, size=(400, 200)):
    """A crude standing person: skin everywhere, shorts band, legs colored (skin or pants)."""
    h, w = size
    img = np.full((h, w, 3), SKIN, np.uint8)
    img[200:240] = shorts_bgr
    img[240:390] = legs_bgr
    kp = np.zeros((133, 2), np.float32)
    sc = np.zeros(133, np.float32)
    pts = {5: (70, 80), 6: (130, 80), 11: (80, 200), 12: (120, 200), 13: (80, 290),
           14: (120, 290), 15: (80, 380), 16: (120, 380),
           24: (85, 30), 26: (88, 45), 52: (100, 35), 38: (115, 30), 36: (112, 45)}
    for i, p in pts.items():
        kp[i], sc[i] = p, 0.9
    return img, kp, sc


RCFG = {**CFG["regions"], "sat_min": SAT, "v_min": VMIN}


def test_describe_fighter_vs_ref():
    img, kp, sc = person_image(RED, SKIN)
    fighter = describe(img, kp, sc, np.array([0, 0, 200, 400], np.float32), RCFG, 0.3)
    img, kp, sc = person_image(BLACK, BLACK)
    ref = describe(img, kp, sc, np.array([0, 0, 200, 400], np.float32), RCFG, 0.3)
    assert fighter.shorts is not None and seed_fraction(
        fighter.shorts, seed_histogram("red", CFG["colors"], SAT, VMIN)) > 0.5
    assert fighter.clothed < 0.4 < 0.8 < ref.clothed
    # missing knees -> no shorts depth, no shins: unknown, not guessed
    sc2 = sc.copy()
    sc2[[13, 14, 15, 16]] = 0.0
    part = describe(img, kp, sc2, np.array([0, 0, 200, 400], np.float32), RCFG, 0.3)
    assert part.shins is None and part.clothed is None


def app(shorts=None, clothed=0.1, torso=None, thigh=None):
    return Appearance(shorts, torso, thigh, None, None, None, clothed, 100.0)


def profile_rb():
    p = Profile("f", seed_histogram("red", CFG["colors"], SAT, VMIN),
                seed_histogram("black", CFG["colors"], SAT, VMIN))
    p.red, p.blue = hist_of(RED), hist_of(BLACK)
    p.ref_torso = hist_of(NAVY)
    return p


def box(x):
    return np.array([x, 100, x + 100, 400], np.float32)


def test_assignment_with_none_and_ref():
    p = profile_rb()
    obs = [BoxObs(box(0), app(hist_of(BLACK))), BoxObs(box(300), app(hist_of(RED))),
           BoxObs(box(600), app(hist_of(WHITE), clothed=0.95, torso=hist_of(NAVY))),
           BoxObs(box(900), app(hist_of(WHITE), clothed=None))]          # crowd
    labels = [r["label"] for r in assign_frame(obs, p, CFG)]
    assert labels == ["blue", "red", "ref", "none"]


def test_black_shorts_fighter_is_not_the_ref():
    """Holland (black shorts, bare shins) vs a ref in black pants."""
    p = profile_rb()
    p.blue = hist_of(BLACK)
    obs = [BoxObs(box(0), app(hist_of(BLACK), clothed=0.1)),
           BoxObs(box(300), app(hist_of(BLACK), clothed=0.95)),
           BoxObs(box(600), app(hist_of(RED), clothed=0.1))]
    labels = [r["label"] for r in assign_frame(obs, p, CFG)]
    assert labels == ["blue", "ref", "red"]


def test_third_person_fallback():
    p = profile_rb()
    p.ref_torso = None
    obs = [BoxObs(box(0), app(hist_of(BLACK))), BoxObs(box(300), app(hist_of(RED))),
           BoxObs(box(600), app(hist_of(WHITE), clothed=0.45))]
    out = assign_frame(obs, p, CFG)
    assert out[2]["label"] == "ref" and out[2]["fallback"]


def test_build_profile_learns_and_warns():
    frames = [[(box(0), app(hist_of(RED))), (box(300), app(hist_of(BLACK)))] for _ in range(12)]
    p = build_profile("f", "red", "black", frames, CFG)
    assert p.n["red"] == 12 and p.red is not None and p.warnings[-1].startswith("no ref")
    assert similarity(p.red, hist_of(RED)) > 0.99
    few = build_profile("f", "red", "black", frames[:3], CFG)
    assert few.red is None and "seeds only" in few.warnings[0]
    # names that match nothing seen: no teaching frames, seeds only, and a warning
    p = build_profile("f", "pink", "yellow", frames, CFG)
    assert p.red is None and "seeds only" in p.warnings[0]


def test_drift_warnings():
    from deepfoul.perception.identity import drift_warnings

    p = profile_rb()
    p.n["ref"] = 3
    assert drift_warnings(p, CFG["profile"]) == []
    p.red, p.blue = hist_of(BLACK), hist_of(RED)          # learned the wrong way round
    assert any("swapped" in w for w in drift_warnings(p, CFG["profile"]))
    p.red, p.blue = hist_of(WHITE), hist_of(BLACK)        # red named red, looks white
    assert any("drifted" in w for w in drift_warnings(p, CFG["profile"]))


def test_split_segments_on_stable_flip_only():
    blip = ["red"] * 10 + ["blue"] * 2 + ["red"] * 10
    assert len(set(split_segments(blip, 8))) == 1
    swap = ["red"] * 10 + ["blue"] * 10
    assert split_segments(swap, 8) == [0] * 10 + [1] * 10


def test_smooth_track_majority_and_swap():
    frames = list(range(22))
    labs = ["red"] * 10 + ["blue"] + ["red"] * 11
    assert smooth_track(frames, labs, 30.0, CFG) == ["red"] * 22
    labs = ["red"] * 11 + ["blue"] * 11
    assert smooth_track(frames, labs, 30.0, CFG) == labs     # a real swap is kept


def test_smoothing_respects_lookahead():
    check_lookahead(CFG, 30.0, 2.0)
    bad = {**CFG, "smooth": {"smooth_s": 3.0, "split_frames": 8}}
    with pytest.raises(ValueError):
        check_lookahead(bad, 30.0, 2.0)
    # the vote at frame t never looks past t + smooth_s
    frames = list(range(40))
    labs = ["red"] * 20 + ["blue"] * 20
    out = smooth_track(frames, labs, 30.0, CFG)
    w = int(CFG["smooth"]["smooth_s"] * 30)
    assert out[:20 - w] == ["red"] * (20 - w)


def test_smooth_labels_dedupes_per_frame():
    df = pd.DataFrame({"frame_idx": [0, 0, 0], "track_id": ["", "", ""], "shot_id": [1, 1, 1],
                       "raw_label": ["red", "red", "blue"], "cost": [0.2, 0.5, 0.1],
                       "x1": [0, 300, 600], "y1": [0, 0, 0], "x2": [100, 400, 700],
                       "y2": [300, 300, 300]})
    out = smooth_labels(df, 30.0, CFG)
    assert out["label"].tolist() == ["red", "none", "blue"]


def test_dressed_ref_in_close_up_is_not_a_fighter():
    """Legs out of frame (clothed unknown): the ref's black shirt + pants match black shorts,
    but not the fighter's bare-skin torso profile."""
    p = profile_rb()
    p.blue = hist_of(BLACK)
    p.red_torso, p.blue_torso = hist_of(SKIN), hist_of(SKIN)
    p.ref_torso = hist_of(BLACK)
    obs = [BoxObs(box(0), app(hist_of(BLACK), clothed=None, torso=hist_of(BLACK))),
           BoxObs(box(300), app(hist_of(RED), clothed=0.1, torso=hist_of(SKIN)))]
    labels = [r["label"] for r in assign_frame(obs, p, CFG)]
    assert labels[1] == "red" and labels[0] != "blue"


def merge_df(rows):
    """rows: (frame, label, box) -> the columns apply_merges reads."""
    return pd.DataFrame([{"frame_idx": f, "shot_id": 1, "label": lab, "x1": b[0], "y1": b[1],
                          "x2": b[2], "y2": b[3]} for f, lab, b in rows])


def test_merge_when_the_pair_closes_into_one_box():
    from deepfoul.perception.identity import apply_merges

    left, right, wide = (0, 100, 150, 400), (120, 100, 280, 400), (0, 100, 280, 400)
    df = merge_df([(0, "red", left), (0, "blue", right), (1, "red", wide), (2, "red", wide)])
    out = apply_merges(df, 30.0, CFG["assign"])
    assert out["label"].tolist() == ["red", "blue", "merged", "merged"]


def test_no_merge_when_the_other_fighter_walked_away():
    from deepfoul.perception.identity import apply_merges

    left, far, wide = (0, 100, 150, 400), (900, 100, 1050, 400), (0, 100, 200, 400)
    df = merge_df([(0, "red", left), (0, "blue", far), (1, "red", wide)])
    assert apply_merges(df, 30.0, CFG["assign"])["label"].tolist() == ["red", "blue", "red"]
    # other fighter last seen too long ago
    df = merge_df([(0, "red", left), (0, "blue", (100, 100, 250, 400)), (60, "red", wide)])
    assert apply_merges(df, 30.0, CFG["assign"])["label"].iloc[-1] == "red"


def test_merge_with_a_fragment_inside_the_wide_box():
    from deepfoul.perception.identity import apply_merges

    left, right, wide, frag = ((0, 100, 150, 400), (120, 100, 280, 400), (0, 100, 280, 400),
                               (150, 250, 270, 400))
    df = merge_df([(0, "red", left), (0, "blue", right), (1, "red", wide), (1, "blue", frag)])
    out = apply_merges(df, 30.0, CFG["assign"])
    assert out["label"].tolist() == ["red", "blue", "merged", "none"]
