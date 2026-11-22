"""Stoppage signals from the keypoint cache (2.2), on a fixed time grid.

Every signal at grid bin b looks at most `lookahead_s` into the future (bins b .. b + ahead - 1);
the past is free. Signals are 0-1. Geometry is in torso lengths, so zoom doesn't matter.

Per fighter box (red / blue / merged; own-body geometry only, so a merged box's one skeleton is
still usable) and per tracklet (ByteTrack ids are per shot, so nothing is compared across a cut):
- motion: body keypoints relative to the box's own hip midpoint, displacement between grid bins
  of the same track / torso / dt. Hip-relative = camera pans don't count; displacement over a
  whole bin = duplicate frames (Garbrandt-Long) don't read as "stopped".
- hand_head: nearest own hand point (wrists, hand tips) to own head centre.
- hand_hip: nearest own hand point to own hip midpoint.
- bend: shoulder-mid -> hip-mid angle from vertical, degrees.

Signals:
- guard: every fighter's hands come down from the face (min hand_head over fighter boxes above
  guard_down) after a stretch with a guard up (below guard_up) over the last guard_before_s.
  This is how the known stoppages look: fighters drop their hands when the ref steps in.
- pause: median fighter motion drops from [b - before_s, b) to the lookahead. Bins without a
  fighter are skipped, so across a cut the "after" side is the new shot's motion.
- ref: the ref moves fast relative to the fighters' midpoint (sustained over 0.5 s).
- eye / groin: one fighter holds a hand on their own head / groin for hold_s.
- bend: one fighter bent over (bend > bend_deg) and still for hold_s.
- closeup: share of the lookahead without both corners in frame.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

BODY = list(range(17))                      # COCO body set: nose, eyes, ears, shoulders ... ankles
HEAD = [0, 1, 2, 3, 4]
HANDS = [9, 10, 91, 95, 99, 103, 107, 111, 112, 116, 120, 124, 128, 132]  # wrists + hand tips
FIGHTER = ("red", "blue", "merged")
SIGNALS = ("guard", "pause", "ref", "eye", "groin", "bend", "closeup")
COLUMNS = ["t", "shot_id", "track_id", "corner", "x1", "y1", "x2", "y2", "kpts", "kpt_scores",
           "torso_len"]


class _quiet(warnings.catch_warnings):
    """Silence numpy's 'mean of empty slice' / 'all-NaN slice' warnings (NaN is the answer)."""

    def __enter__(self):
        super().__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)


def box_features(df: pd.DataFrame, conf: float) -> pd.DataFrame:
    """Per-row own-body geometry: rel (17 x 2 hip-relative body coords in torsos, NaN = masked),
    hand_head, hand_hip, bend."""
    n = len(df)
    k = np.stack(df["kpts"].to_numpy()).reshape(n, 133, 2).astype(np.float32)
    s = np.stack(df["kpt_scores"].to_numpy()).reshape(n, 133)
    k = np.where((s >= conf)[..., None], k, np.nan)
    torso = df["torso_len"].to_numpy(np.float32)
    torso = np.where(torso > 1, torso, np.nan)
    with np.errstate(all="ignore"), _quiet():
        hip = np.nanmean(k[:, [11, 12]], axis=1)          # NaN unless a hip is confident
        sho = np.nanmean(k[:, [5, 6]], axis=1)
        head = np.nanmean(k[:, HEAD], axis=1)
        rel = (k[:, BODY] - hip[:, None]) / torso[:, None, None]
        hands = k[:, HANDS]
        hand_head = np.nanmin(np.linalg.norm(hands - head[:, None], axis=2), axis=1) / torso
        hand_hip = np.nanmin(np.linalg.norm(hands - hip[:, None], axis=2), axis=1) / torso
        v = sho - hip
        bend = np.degrees(np.arccos(np.clip(-v[:, 1] / np.linalg.norm(v, axis=1), -1, 1)))
    out = pd.DataFrame({"hand_head": hand_head, "hand_hip": hand_hip, "bend": bend},
                       index=df.index)
    out["rel"] = list(rel)
    return out


def _bins(t, t0: float, step: float) -> np.ndarray:
    return np.floor((np.asarray(t, float) - t0) / step + 1e-9).astype(np.int64)


def per_track_bins(df: pd.DataFrame, feats: pd.DataFrame, t0: float, step: float) -> pd.DataFrame:
    """One row per (track, bin): the track's last box in the bin, with its motion since the
    previous bin of the same track (NaN for a track's first bin or after a gap)."""
    d = pd.concat([df[["t", "track_id", "corner"]], feats], axis=1)
    d = d[d["track_id"].astype(str) != ""]
    d["bin"] = _bins(d["t"].to_numpy(), t0, step)
    last = d.sort_values("t").groupby(["track_id", "bin"], sort=False).tail(1)
    last = last.sort_values(["track_id", "bin"]).reset_index(drop=True)
    rel = np.stack(last["rel"].to_numpy()) if len(last) else np.zeros((0, 17, 2))
    same = last["track_id"].to_numpy()[1:] == last["track_id"].to_numpy()[:-1]
    gap = np.diff(last["bin"].to_numpy()) == 1
    with np.errstate(all="ignore"), _quiet():
        disp = np.nanmean(np.linalg.norm(rel[1:] - rel[:-1], axis=2), axis=1) / step
    motion = np.full(len(last), np.nan)
    motion[1:] = np.where(same & gap, disp, np.nan)
    last["motion"] = motion
    return last.drop(columns=["rel"])


def _ahead(x, ahead: int, fn: str) -> np.ndarray:
    """fn of x[b : b + ahead], ignoring NaN (NaN if all NaN)."""
    r = pd.Series(np.asarray(x, float)[::-1]).rolling(ahead, min_periods=1)
    return getattr(r, fn)().to_numpy()[::-1]


def _behind(x, n: int, fn: str) -> np.ndarray:
    """fn of x[b - n : b] (strictly before b), ignoring NaN."""
    r = pd.Series(np.asarray(x, float)).shift(1).rolling(n, min_periods=1)
    return getattr(r, fn)().to_numpy()


def held(tb: pd.DataFrame, flag: np.ndarray, nb: int, ahead: int) -> np.ndarray:
    """Per bin b: the most bins any single track spends flagged in [b, b + ahead)."""
    out = np.zeros(nb)
    d = pd.DataFrame({"track_id": tb["track_id"].to_numpy(), "bin": tb["bin"].to_numpy(),
                      "flag": np.asarray(flag, bool)})
    for _, g in d[d["flag"]].groupby("track_id", sort=False):
        fb = np.sort(g["bin"].to_numpy())
        b = np.arange(max(0, fb[0] - ahead + 1), fb[-1] + 1)
        cnt = np.searchsorted(fb, b + ahead) - np.searchsorted(fb, b)
        np.maximum.at(out, b[b < nb], cnt[b < nb])
    return out


def compute_signals(df: pd.DataFrame, start_s: float, end_s: float, mcfg: dict,
                    conf: float, lookahead_s: float) -> pd.DataFrame:
    """Signals (and the raw per-bin measures behind them) on the grid [start_s, end_s).
    df = cache rows (COLUMNS) inside the window."""
    step = mcfg["step_s"]
    ahead = max(1, int(round(min(mcfg["after_s"], lookahead_s) / step)))
    before_n = max(1, int(round(mcfg["before_s"] / step)))
    guard_n = max(1, int(round(mcfg["guard_before_s"] / step)))
    hold_n = max(1, int(round(mcfg["hold_s"] / step)))
    nb = int(np.ceil((end_s - start_s) / step))
    out = pd.DataFrame({"t": start_s + step * np.arange(nb)})

    fig = df[df["corner"].isin(FIGHTER)]
    cols = ["track_id", "bin", "motion", "hand_head", "hand_hip", "bend", "corner"]
    tb = pd.DataFrame({c: pd.Series(dtype=object if c in ("track_id", "corner") else float)
                       for c in cols})
    if len(fig):
        tb = per_track_bins(fig, box_features(fig, conf), start_s, step)
        tb = tb[(tb["bin"] >= 0) & (tb["bin"] < nb)].reset_index(drop=True)

    def per_bin(values, how: str) -> np.ndarray:
        a = np.full(nb, np.nan)
        if not len(tb):
            return a
        g = pd.Series(np.asarray(values, float)).groupby(tb["bin"].to_numpy()).agg(how).dropna()
        a[g.index.to_numpy(np.int64)] = g.to_numpy()
        return a

    with np.errstate(all="ignore"), _quiet():
        motion = per_bin(tb["motion"], "mean")
        out["motion"] = motion
        hh = per_bin(tb["hand_head"], "min")
        out["hand_head"] = hh

        # guard: hands come down from the face after a stretch with a guard up
        down = np.where(np.isnan(hh), np.nan, (hh > mcfg["guard_down"]).astype(float))
        up = np.where(np.isnan(hh), np.nan, (hh < mcfg["guard_up"]).astype(float))
        out["guard"] = np.nan_to_num(_ahead(down, ahead, "mean") * _behind(up, guard_n, "mean"))

        # pause: fighter motion drops (NaN bins skipped, so a cut compares shot to shot)
        before = _behind(motion, before_n, "median")
        after = _ahead(motion, ahead, "median")
        drop = np.clip((before - after) / before, 0, 1)
        active = np.clip(before / mcfg["active_motion"], 0, 1)
        out["pause"] = np.nan_to_num(drop * active)

    # held own-body poses, per track
    still = (tb["motion"].astype(float).fillna(0) < mcfg["still_motion"]).to_numpy()
    flags = {"eye": (tb["hand_head"].astype(float) < mcfg["eye_r"]).to_numpy(),
             "groin": (tb["hand_hip"].astype(float) < mcfg["groin_r"]).to_numpy(),
             "bend": (tb["bend"].astype(float) > mcfg["bend_deg"]).to_numpy() & still}
    for name, flag in flags.items():
        out[name] = np.clip(held(tb, flag, nb, ahead) / hold_n, 0, 1)

    out["ref"], out["ref_speed"] = _ref_signal(df, start_s, step, nb, ahead, mcfg)

    # close-up: both corners not in frame, over the lookahead
    one = np.zeros(nb)
    if len(fig):
        both = (pd.DataFrame({"bin": _bins(fig["t"].to_numpy(), start_s, step),
                              "red": (fig["corner"] == "red").to_numpy(),
                              "blue": (fig["corner"] == "blue").to_numpy()})
                .groupby("bin")[["red", "blue"]].any())
        both = both[(both.index >= 0) & (both.index < nb)]
        one[both.index.to_numpy(np.int64)] = ~(both["red"] & both["blue"]).to_numpy()
    out["closeup"] = _ahead(one, ahead, "mean")
    return out


def _ref_signal(df: pd.DataFrame, t0: float, step: float, nb: int, ahead: int,
                mcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(signal, raw sustained speed) of the ref relative to the fighters' midpoint, in ref
    heights per second."""
    zero = (np.zeros(nb), np.zeros(nb))
    if not len(df):
        return zero
    d = pd.DataFrame({"bin": _bins(df["t"].to_numpy(), t0, step),
                      "cx": ((df["x1"] + df["x2"]) / 2).to_numpy(),
                      "cy": ((df["y1"] + df["y2"]) / 2).to_numpy(),
                      "h": (df["y2"] - df["y1"]).clip(lower=1).to_numpy(),
                      "corner": df["corner"].to_numpy(),
                      "track_id": df["track_id"].astype(str).to_numpy()})
    d = d[(d["bin"] >= 0) & (d["bin"] < nb)]
    fm = d[d["corner"].isin(FIGHTER)].groupby("bin")[["cx", "cy"]].mean()
    ref = d[(d["corner"] == "ref") & (d["track_id"] != "")]
    ref = ref.groupby(["track_id", "bin"])[["cx", "cy", "h"]].last().reset_index()
    ref = ref.join(fm.rename(columns={"cx": "cx_f", "cy": "cy_f"}), on="bin").dropna()
    if not len(ref):
        return zero
    ref = ref.sort_values(["track_id", "bin"])
    rx = (ref["cx"] - ref["cx_f"]) / ref["h"]
    ry = (ref["cy"] - ref["cy_f"]) / ref["h"]
    same = ref["track_id"].eq(ref["track_id"].shift()) & ref["bin"].diff().eq(1)
    ref["speed"] = (np.hypot(rx.diff(), ry.diff()) / step).where(same)
    speed = np.zeros(nb)
    g = ref.groupby("bin")["speed"].max().dropna()
    speed[g.index.to_numpy(np.int64)] = g.to_numpy()
    sm = pd.Series(speed).rolling(max(1, int(round(0.5 / step))), min_periods=1).mean().to_numpy()
    return np.clip(_ahead(sm, ahead, "max") / mcfg["ref_speed"], 0, 1), sm
