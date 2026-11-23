"""Hard-negative and random queues (2.3) on synthetic keypoints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deepfoul.config import load_config
from deepfoul.mining.hardneg import candidates, random_items, strike_frames

CFG = load_config()
H = CFG["hardneg"]
FPS = 30


def body(head, hip, torso=200.0):
    """133 x 2 keypoints: head points at `head`, hips at `hip`, everything else far away."""
    k = np.full((133, 2), -5000.0, np.float32)
    k[[0, 1, 2, 3, 4]] = head
    k[[11, 12]] = hip
    k[[5, 6]] = (hip[0], hip[1] - torso)
    return k


def scene(kind: str, n: int = 30, extended: bool = False) -> pd.DataFrame:
    """Blue stands still. Red's right hand (kind='face') or right foot (kind='groin') travels
    fast to blue's head / hips over frames 10-15, then goes back."""
    rows = []
    blue = body((800.0, 200.0), (800.0, 450.0))
    for i in range(n):
        t = i / FPS
        red = body((300.0, 200.0), (300.0, 450.0))
        target = (780.0, 200.0) if kind == "face" else (790.0, 450.0)
        start = (380.0, 200.0) if kind == "face" else (330.0, 600.0)
        a = np.clip((i - 10) / 5, 0, 1) if i <= 15 else max(0.0, 1 - (i - 15) / 5)
        p = np.array(start) + a * (np.array(target) - np.array(start))
        if kind == "face":
            red[10] = p
            red[112:133] = p                      # the right hand: a fist (tips at the wrist)
            if extended:
                for tip in (4, 8, 12, 16, 20):
                    red[112 + tip] = p + (100.0, 0.0)
        else:
            red[16] = p                           # right ankle
        for corner, k, tid in (("red", red, 1), ("blue", blue, 2)):
            rows.append({"t": round(t, 4), "track_id": tid, "corner": corner,
                         "kpts": k.reshape(-1).tolist(), "kpt_scores": [0.9] * 133,
                         "torso_len": 200.0, "score": 0.9, "shot_id": 0})
    return pd.DataFrame(rows)


def best(df):
    f = strike_frames(df, 0.3, H)
    return f.loc[f["score"].idxmax()]


def test_fist_at_the_face_scores():
    r = best(scene("face"))
    assert r["score"] > 0.5 and r["zone"] == "face" and r["striker"] == "red"


def test_extended_hand_at_the_face_is_not_a_hard_negative():
    assert strike_frames(scene("face", extended=True), 0.3, H)["score"].max() == 0


def test_kick_at_the_groin_scores():
    r = best(scene("groin"))
    assert r["score"] > 0.5 and r["zone"] == "groin" and r["striker"] == "red"


def test_merged_frames_have_no_opponent():
    df = scene("face")
    df.loc[df["corner"] == "blue", "corner"] = "merged"
    assert len(strike_frames(df, 0.3, H)) == 0


def _frames(t_hit):
    t = np.round(np.arange(0, 60, 0.25), 2)
    return pd.DataFrame({"t": t, "score": np.where(t == t_hit, 0.9, 0.0),
                         "zone": "groin", "striker": "blue"})


def test_candidates_need_no_stoppage_and_distance():
    quiet = pd.DataFrame({"t": np.arange(0, 60, 0.25), "guard": 0.0})
    c = candidates(_frames(30.0), quiet, (0, 60), [], H, 0.25)
    assert [x["t_s"] for x in c] == [30.0] and c[0]["reason"] == "strike at groin · blue"
    stop = quiet.assign(guard=np.where((quiet["t"] > 32) & (quiet["t"] < 33), 0.8, 0.0))
    assert candidates(_frames(30.0), stop, (0, 60), [], H, 0.25) == []      # a stoppage follows
    assert candidates(_frames(30.0), quiet, (0, 60), [35.0], H, 0.25) == []  # near a label
    assert candidates(_frames(2.0), quiet, (0, 60), [], H, 0.25) == []      # clip not in window


def test_random_items_spaced_deterministic():
    a = random_items("ufc1_a_b", (100, 400), [200.0], 10, 10.0, seed=0)
    assert a == random_items("ufc1_a_b", (100, 400), [200.0], 10, 10.0, seed=0)
    ts = [x["t_s"] for x in a]
    assert len(ts) == 10 and all(104 <= t <= 396 for t in ts)
    assert all(abs(t - 200) >= 10 for t in ts)
    assert min(np.diff(sorted(ts))) >= 10
