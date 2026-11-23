"""Hard-negative candidates (2.3): a strike lands near a foul zone and the fight goes on.

From the keypoint cache, on frames with exactly one red and one blue box (a merged box has one
skeleton, so there's no opponent to measure against):
- face: one fighter's hand point (wrists, hand tips) comes within `head_r` torso lengths of the
  other's head centre (nose, eyes, ears), fast, with the fingers not extended (a fist; an
  extended hand at the face is an open hand, which is a positive, not a hard negative).
- groin: one fighter's knee, ankle or foot point comes within `groin_r` of the other's hip
  midpoint, fast.
Score per frame = proximity (1 at the zone centre, 0 at the radius) x speed (capped at
`speed_ref` torso lengths per second); a 0.25 s bin keeps its best frame. A candidate is kept
only if no stoppage follows: the 2.2 guard signal stays under `stop_max` for `stop_after_s`.
This peeks past the lookahead, which is fine for picking clips to label (not for detection).

Also `random_items`: plain random moments for unbiased clean labels.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from deepfoul.mining.miner import item_id, pick_peaks
from deepfoul.mining.signals import HANDS, HEAD, _quiet

LEGS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]          # knees, ankles, feet
HAND_BASE = {"left": 91, "right": 112}                    # hand wrist point; tips at +4, 8, ...
TIPS = [4, 8, 12, 16, 20]
WRIST_HAND = {9: "left", 10: "right"}


def hand_of(point: int) -> str:
    if point in WRIST_HAND:
        return WRIST_HAND[point]
    return "left" if point < 112 else "right"


def _corner_frames(df: pd.DataFrame, corner: str, conf: float) -> pd.DataFrame:
    """One row per frame for this corner (best box), with masked keypoints and per-point speed
    (torso lengths / s, from the previous frame of the same track)."""
    d = df[df["corner"] == corner].sort_values(["t", "score"], ascending=[True, False])
    d = d.drop_duplicates("t").sort_values(["track_id", "t"]).reset_index(drop=True)
    n = len(d)
    if n == 0:
        return d.assign(k=[], v=[])
    k = np.stack(d["kpts"].to_numpy()).reshape(n, 133, 2).astype(np.float32)
    s = np.stack(d["kpt_scores"].to_numpy()).reshape(n, 133)
    k = np.where((s >= conf)[..., None], k, np.nan)
    torso = d["torso_len"].to_numpy(np.float32)
    torso = np.where(torso > 1, torso, np.nan)
    v = np.full((n, 133), np.nan, np.float32)
    same = (d["track_id"].to_numpy()[1:] == d["track_id"].to_numpy()[:-1])
    dt = np.diff(d["t"].to_numpy())
    ok = same & (dt > 0) & (dt < 0.2)
    with np.errstate(all="ignore"):
        step = np.linalg.norm(k[1:] - k[:-1], axis=2) / dt[:, None] / torso[1:, None]
    v[1:] = np.where(ok[:, None], step, np.nan)
    d["k"], d["v"], d["torso"] = list(k), list(v), torso
    return d[["t", "k", "v", "torso"]]


def strike_frames(df: pd.DataFrame, conf: float, h: dict) -> pd.DataFrame:
    """Per frame with one red and one blue box: best strike score, zone and striker."""
    red, blue = _corner_frames(df, "red", conf), _corner_frames(df, "blue", conf)
    j = red.merge(blue, on="t", suffixes=("_r", "_b")).sort_values("t").reset_index(drop=True)
    if not len(j):
        return pd.DataFrame(columns=["t", "score", "zone", "striker"])
    K = {"red": np.stack(j["k_r"]), "blue": np.stack(j["k_b"])}
    V = {"red": np.stack(j["v_r"]), "blue": np.stack(j["v_b"])}
    T = {"red": j["torso_r"].to_numpy(), "blue": j["torso_b"].to_numpy()}
    n = len(j)
    best = np.zeros(n)
    zone = np.array([""] * n, dtype=object)
    striker = np.array([""] * n, dtype=object)
    rows = np.arange(n)
    with np.errstate(all="ignore"), _quiet():
        for a, b in (("red", "blue"), ("blue", "red")):
            ka, kb, va = K[a], K[b], V[a]
            scale = np.nanmean(np.stack([T[a], T[b]]), axis=0)
            targets = {"face": (np.nanmean(kb[:, HEAD], axis=1), HANDS, h["head_r"]),
                       "groin": (np.nanmean(kb[:, [11, 12]], axis=1), LEGS, h["groin_r"])}
            for z, (centre, pts, r) in targets.items():
                d = np.linalg.norm(ka[:, pts] - centre[:, None], axis=2) / scale[:, None]
                prox = np.clip(1 - d / r, 0, 1)
                spd = np.clip(va[:, pts] / h["speed_ref"], 0, 1)
                s = np.nan_to_num(prox * spd)
                i = np.argmax(s, axis=1)
                sc = s[rows, i]
                if z == "face":   # an extended hand at the face is an open hand: not ours
                    ext = np.full(n, np.nan)
                    for side, base in HAND_BASE.items():
                        tips = np.linalg.norm(ka[:, [base + t for t in TIPS]] - ka[:, [base]],
                                              axis=2)
                        e = np.nanmedian(tips, axis=1) / T[a]
                        use = np.array([hand_of(pts[x]) == side for x in i])
                        ext = np.where(use, e, ext)
                    sc = np.where(ext > h["open_min"], 0.0, sc)
                better = sc > best
                best = np.where(better, sc, best)
                zone = np.where(better, z, zone)
                striker = np.where(better, a, striker)
    return pd.DataFrame({"t": j["t"].to_numpy(), "score": best, "zone": zone,
                         "striker": striker})


def candidates(frames: pd.DataFrame, guard: pd.DataFrame, window: tuple[float, float],
               avoid: list[float], h: dict, step: float) -> list[dict]:
    """Ranked strike moments with no stoppage after, away from every `avoid` time."""
    if not len(frames):
        return []
    w0, w1 = window
    f = frames.assign(bin=np.floor((frames["t"] - w0) / step).astype(int))
    best = f.loc[f.groupby("bin")["score"].idxmax()].reset_index(drop=True)
    t = best["t"].to_numpy()
    s = best["score"].to_numpy().copy()
    # no stoppage after: guard signal max over [t, t + stop_after_s]
    gt, gv = guard["t"].to_numpy(), guard["guard"].to_numpy()
    for i, ti in enumerate(t):
        sel = (gt >= ti) & (gt <= ti + h["stop_after_s"])
        if sel.any() and gv[sel].max() >= h["stop_max"]:
            s[i] = 0
    av = np.asarray(sorted(avoid), float)
    if len(av):
        near = np.abs(t[:, None] - av[None, :]).min(axis=1) < h["gap_s"]
        s[near] = 0
    s[(t < w0 + 4) | (t > w1 - 4)] = 0
    # each zone gets its own quota (face strikes are ~2x as common and would crowd out the
    # groin ones); groin first, then face away from the groin picks
    picked: list[int] = []
    zones = best["zone"].to_numpy()
    for z in ("groin", "face"):
        sz = np.where(zones == z, s, 0.0)
        for j in picked:
            sz[np.abs(t - t[j]) < h["gap_s"]] = 0
        picked += pick_peaks(t, sz, h["per_zone"][z], h["gap_s"])
    out = []
    for rank, i in enumerate(sorted(picked, key=lambda i: -s[i]), start=1):
        r = best.iloc[i]
        what = "strike at face" if r["zone"] == "face" else "strike at groin"
        out.append({"t_s": round(float(t[i]), 2), "score": round(float(s[i]), 3), "rank": rank,
                    "reason": f"{what} · {r['striker']}"})
    return out


def queue_items(fid: str, cands: list[dict], prefix: str) -> list[dict]:
    return [{"item_id": f"{prefix}_{item_id(fid, c['t_s'])}", "fight_id": fid, "t_s": c["t_s"],
             "score": c.get("score", ""), "reason": c["reason"]} for c in cands]


def random_items(fid: str, window: tuple[float, float], avoid: list[float], n: int, gap: float,
                 seed: int, pad: float = 4.0) -> list[dict]:
    """n random moments in the window (clip fully inside), each >= gap from every avoid time
    and from each other. Deterministic per (seed, fight)."""
    rng = random.Random(f"{seed}-{fid}")
    w0, w1 = window[0] + pad, window[1] - pad
    picked: list[float] = []
    for _ in range(5000):
        if len(picked) >= n or w1 <= w0:
            break
        t = round(rng.uniform(w0, w1), 2)
        if all(abs(t - a) >= gap for a in [*avoid, *picked]):
            picked.append(t)
    return [{"t_s": t, "reason": "random", "score": ""} for t in sorted(picked)]
