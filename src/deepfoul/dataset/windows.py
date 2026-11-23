"""Reading the keypoint cache for dataset windows (2.3). Read-only; no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIGHTER = ("red", "blue", "merged")


def coverage_arrays(kp_dir: Path, fid: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(sorted times of every processed frame, sorted times of frames with a red / blue /
    merged box), or None when the fight isn't cached."""
    frames, rows = kp_dir / f"{fid}_frames.parquet", kp_dir / f"{fid}.parquet"
    if not frames.is_file() or not rows.is_file():
        return None
    ft = np.sort(pd.read_parquet(frames, columns=["t"])["t"].to_numpy())
    r = pd.read_parquet(rows, columns=["t", "corner"])
    fig = np.unique(r.loc[r["corner"].isin(FIGHTER), "t"].to_numpy())
    return ft, fig


def window_keypoints(kp_dir: Path, fid: str, t0: float, t1: float) -> dict:
    """Per processed frame in [t0, t1): red and blue keypoints (T, 133, 2) + scores (T, 133),
    NaN / 0 where that fighter has no box. A `merged` box goes to its raw_label corner (the
    corner before merging, see PROGRESS 1.5), so its one skeleton isn't lost.

    Returns {"t": (T,), "red": {...}, "blue": {...}, "merged": (T,) bool}."""
    frames = pd.read_parquet(kp_dir / f"{fid}_frames.parquet", columns=["t"],
                             filters=[("t", ">=", float(t0)), ("t", "<", float(t1))])
    t = np.sort(frames["t"].to_numpy())
    rows = pd.read_parquet(kp_dir / f"{fid}.parquet",
                           columns=["t", "corner", "raw_label", "kpts", "kpt_scores", "score"],
                           filters=[("t", ">=", float(t0)), ("t", "<", float(t1))])
    out = {"t": t, "merged": np.zeros(len(t), bool)}
    idx = {v: i for i, v in enumerate(t)}
    for c in ("red", "blue"):
        out[c] = {"kpts": np.full((len(t), 133, 2), np.nan, np.float32),
                  "scores": np.zeros((len(t), 133), np.float32)}
    rows = rows.assign(side=np.where(rows["corner"] == "merged", rows["raw_label"], rows["corner"]))
    # one box per corner per frame: the direct label wins over a merged stand-in, then box score
    rows = rows[rows["side"].isin(("red", "blue"))].assign(m=rows["corner"] == "merged")
    rows = rows.sort_values(["t", "side", "m", "score"], ascending=[True, True, True, False])
    for r in rows.drop_duplicates(["t", "side"]).itertuples():
        i = idx.get(r.t)
        if i is None:
            continue
        out[r.side]["kpts"][i] = np.asarray(r.kpts, np.float32).reshape(133, 2)
        out[r.side]["scores"][i] = np.asarray(r.kpt_scores, np.float32)
        out["merged"][i] |= bool(r.m)
    return out
