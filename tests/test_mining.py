"""Stoppage finder (2.2): signals on synthetic keypoint sequences, peaks, queues, the check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deepfoul.config import load_config
from deepfoul.labeling.queues import load_queue, write_queue
from deepfoul.mining.miner import (build_mined, candidates, extras, fight_queue, item_id,
                                   known_ranks, pick_peaks, reason)
from deepfoul.mining.signals import compute_signals

FPS = 30.0
TORSO = 100.0


@pytest.fixture(scope="module")
def mcfg():
    return load_config()["mining"]


def pose(hands: str = "guard", bend: bool = False, phase: float = 0.0, amp: float = 0.0):
    """133 x 2 keypoints around a hip midpoint at (0, 0), torso 100 px; scores 1 on the body set
    (0-16), 0 elsewhere. hands: guard | down | eye | groin. amp moves elbows, knees and ankles."""
    k = np.zeros((133, 2), np.float32)
    s = np.zeros(133, np.float32)
    sho_y, head_y, sx = -100.0, -140.0, 0.0
    if bend:
        sho_y, head_y, sx = 0.0, 0.0, 100.0
    pts = {0: (sx, head_y), 1: (sx - 8, head_y - 5), 2: (sx + 8, head_y - 5),
           3: (sx - 15, head_y), 4: (sx + 15, head_y), 5: (sx - 40, sho_y), 6: (sx + 40, sho_y),
           11: (-20, 0), 12: (20, 0), 13: (-25, 80), 14: (25, 80), 15: (-25, 160), 16: (25, 160),
           7: (sx - 50, sho_y + 40), 8: (sx + 50, sho_y + 40)}
    hand = {"guard": ((sx - 15, head_y + 20), (sx + 15, head_y + 20)),
            "down": ((-45, 15), (45, 15)),
            "eye": ((sx - 3, head_y - 3), (sx + 15, head_y + 20)),
            "groin": ((-3, 4), (45, 15))}[hands]
    pts[9], pts[10] = hand
    wob = amp * np.sin(phase)
    for i, (x, y) in pts.items():
        k[i] = (x + (wob if i in (7, 8, 13, 14, 15, 16) else 0.0), y)
        s[i] = 1.0
    return k, s


def rows(spec, t0=0.0, fps=FPS, x0=-200.0, track="0-0", corner="red", shot=0, dup=False):
    """spec: list of (seconds, pose kwargs). One fighter box per frame."""
    out, t = [], t0
    for secs, kw in spec:
        n = int(round(secs * fps))
        for i in range(n):
            j = i - (i % 2) if dup else i            # dup: every other frame repeats
            k, s = pose(**{**kw, "phase": kw.get("speed", 0.0) * j / fps}
                        if "speed" in kw else kw)
            k = k + np.array([x0, 500.0], np.float32)
            out.append({"t": t + i / fps, "shot_id": shot, "track_id": track, "corner": corner,
                        "x1": x0 - 60, "y1": 300.0, "x2": x0 + 60, "y2": 700.0,
                        "kpts": k.ravel(), "kpt_scores": s, "torso_len": TORSO})
        t += n / fps
    return out


def moving(hands="guard", amp=80.0):
    return {"hands": hands, "amp": amp, "speed": 2 * np.pi * 1.1}


def still(hands="guard", bend=False):
    return {"hands": hands, "bend": bend}


def frame(rs):
    df = pd.DataFrame(rs)
    return df.sort_values("t", kind="stable").reset_index(drop=True)


def two(spec_red, spec_blue=None, **kw):
    spec_blue = spec_blue or spec_red
    return frame(rows(spec_red, x0=-200, track="0-0", corner="red", **kw)
                 + rows(spec_blue, x0=200, track="0-1", corner="blue", **kw))


def sig(df, mcfg, end=None):
    end = end or float(df["t"].max()) + 1 / FPS
    return compute_signals(df, 0.0, end, mcfg, 0.3, 2.0)


def at(s, t):
    return s.iloc[int(np.argmin(np.abs(s["t"].to_numpy() - t)))]


# patch pose() so rows() can pass "speed" through as an oscillation rate
_pose = pose


def pose(hands="guard", bend=False, phase=0.0, amp=0.0, speed=None):  # noqa: F811
    return _pose(hands, bend, phase, amp)


# ---------------------------------------------------------------- signals


def test_pause_fires_when_motion_drops(mcfg):
    s = sig(two([(8, moving()), (6, still())]), mcfg)
    assert at(s, 8.0)["pause"] > 0.8
    assert s[s["t"] < 6]["pause"].max() < 0.2        # steady action: no pause


def test_pause_across_a_cut_compares_the_new_shot(mcfg):
    """High motion, a cut (new shot, new track ids), then low motion = a high pause score."""
    a = rows([(8, moving())], track="0-0", corner="red", shot=0) + \
        rows([(8, moving())], x0=200, track="0-1", corner="blue", shot=0)
    b = rows([(6, still())], t0=8.0, track="1-0", corner="red", shot=1) + \
        rows([(6, still())], t0=8.0, x0=200, track="1-1", corner="blue", shot=1)
    s = sig(frame(a + b), mcfg)
    assert s.loc[(s["t"] >= 7.5) & (s["t"] <= 8.5), "pause"].max() > 0.8


def test_duplicate_frames_are_not_stopped(mcfg):
    """A 30 fps feed where every other frame repeats (Garbrandt-Long) still reads as moving."""
    s = sig(two([(10, moving())], dup=True), mcfg)
    assert np.nanmedian(s["motion"]) > 1.0
    assert s[s["t"] < 8.0]["pause"].max() < 0.2       # bins with a full 2 s lookahead


def test_guard_drop_after_a_guard(mcfg):
    s = sig(two([(12, still("guard")), (5, still("down"))]), mcfg)
    assert at(s, 12.0)["guard"] > 0.9
    assert s[s["t"] < 10]["guard"].max() == 0        # hands still up 2 s ahead


def test_no_guard_signal_without_a_guard_before(mcfg):
    s = sig(two([(17, still("down"))]), mcfg)
    assert s["guard"].max() == 0


def test_eye_hold_vs_normal_guard(mcfg):
    s = sig(two([(4, still("guard")), (2, still("eye")), (4, still("guard"))],
                [(10, still("guard"))]), mcfg)
    assert at(s, 4.0)["eye"] == 1.0
    assert s[s["t"] < 1.5]["eye"].max() == 0          # a guard is not a hand on the eye


def test_groin_hold(mcfg):
    s = sig(two([(4, still("down")), (2, still("groin")), (4, still("down"))],
                [(10, still("down"))]), mcfg)
    assert at(s, 4.0)["groin"] == 1.0
    assert s[s["t"] < 1.5]["groin"].max() == 0


def test_bent_over_and_still(mcfg):
    s = sig(two([(4, still()), (2, still(bend=True)), (4, still())]), mcfg)
    assert at(s, 4.0)["bend"] == 1.0
    assert s[s["t"] < 1.5]["bend"].max() == 0


def test_ref_rush(mcfg):
    fighters = rows([(6, still())], x0=-200) + rows([(6, still())], x0=200, track="0-1",
                                                     corner="blue")
    ref = []
    for i in range(int(6 * FPS)):
        t = i / FPS
        x = 900.0 if t < 3 else 900.0 - 1500.0 * min(t - 3, 0.5)   # 1.5 heights/0.5 s... fast
        k, s = _pose("down")
        ref.append({"t": t, "shot_id": 0, "track_id": "0-2", "corner": "ref", "x1": x - 60,
                    "y1": 300.0, "x2": x + 60, "y2": 700.0, "kpts": k.ravel(),
                    "kpt_scores": s, "torso_len": TORSO})
    s = sig(frame(fighters + ref), mcfg)
    assert at(s, 2.5)["ref"] > 0.8
    assert at(s, 0.0)["ref"] < 0.05


def test_signals_use_at_most_the_lookahead(mcfg):
    """Changing everything after t + lookahead leaves the signal at t unchanged."""
    a = sig(two([(10, moving()), (10, still("down"))]), mcfg, end=20.0)
    b = sig(two([(10, moving()), (10, still("guard"))]), mcfg, end=20.0)
    early = a["t"] < 10.0 - 2.0 - 1e-9
    cols = ["guard", "pause", "eye", "groin", "bend", "closeup", "ref"]
    pd.testing.assert_frame_equal(a.loc[early, cols], b.loc[early, cols])


# ---------------------------------------------------------------- peaks, reasons, queues


def test_pick_peaks_gap_and_order():
    t = np.arange(0, 60, 0.25)
    s = np.zeros_like(t)
    s[[40, 44, 200]] = [0.9, 1.0, 0.5]               # 10 s, 11 s (too close), 50 s
    assert pick_peaks(t, s, 5, 10.0) == [44, 200]


def test_candidates_place_time_before_the_peak(mcfg):
    t = np.arange(100, 160, 0.25)
    d = {k: np.zeros_like(t) for k in ("guard", "pause", "ref", "eye", "groin", "bend",
                                       "closeup")}
    d["guard"][np.argmin(np.abs(t - 130))] = 1.0
    d["groin"][np.argmin(np.abs(t - 130))] = 0.8
    c = candidates(pd.DataFrame({"t": t, **d}), {"guard": 1.0}, mcfg, 10.0, (100.0, 160.0))
    assert c[0]["t_s"] == pytest.approx(130 - mcfg["onset_back_s"])
    assert c[0]["reason"] == "guard · groin"         # names only, no numbers
    assert not any(ch.isdigit() for ch in c[0]["reason"])


def test_reason_weak():
    row = pd.Series({k: 0.0 for k in ("guard", "pause", "ref", "eye", "groin", "bend",
                                      "closeup")})
    assert reason(row, 0.3) == "weak"


def test_known_ranks_hit_window():
    cands = [{"t_s": 100.0, "rank": 1}, {"t_s": 200.0, "rank": 2}, {"t_s": 196.0, "rank": 3}]
    fouls = [{"fight_id": "f", "live_s": "203.5"}, {"fight_id": "f", "live_s": "150"},
             {"fight_id": "f", "live_s": "199"}]
    got = [f["rank"] for f in known_ranks(cands, fouls, 4.0, 4.0)]
    assert got == [2, None, 2]                        # best rank whose clip window covers it


def test_extras_for_misses_only():
    ranked = [{"fight_id": "f", "live_s": "150", "rank": None},
              {"fight_id": "f", "live_s": "203", "rank": 2}]
    ex = extras("f", ranked)
    assert ex == [{"item_id": item_id("f", 150.0), "fight_id": "f", "t_s": 150.0, "score": "",
                   "reason": "extra"}]


def test_build_mined_train_only_with_extras(tmp_path):
    q = tmp_path
    write_queue(fight_queue("tr1", [{"t_s": 300.0, "score": 0.9, "reason": "guard"},
                                    {"t_s": 100.0, "score": 0.5, "reason": "guard"}]),
                q / "mined_tr1.csv")
    write_queue(fight_queue("va1", [{"t_s": 50.0, "score": 0.9, "reason": "guard"}]),
                q / "mined_va1.csv")
    fouls = [{"fight_id": "tr1", "live_s": "302"}, {"fight_id": "tr1", "live_s": "500"},
             {"fight_id": "va1", "live_s": "900"}]
    items, extra = build_mined(q, ["tr1"], fouls, 4.0, 4.0)
    assert [it["t_s"] for it in items] == [100.0, 300.0, 500.0]   # by time, val excluded
    assert [e["t_s"] for e in extra] == [500.0]
    assert items[-1]["reason"] == "extra"
    write_queue(items, q / "mined.csv")
    again = load_queue(q / "mined.csv")
    assert [it["item_id"] for it in again] == [it["item_id"] for it in items]


def test_item_ids_are_stable():
    assert item_id("ufc1_a_b", 812.25) == "ufc1_a_b_0081225"
    assert item_id("ufc1_a_b", 812.25) == item_id("ufc1_a_b", 812.2500001)
