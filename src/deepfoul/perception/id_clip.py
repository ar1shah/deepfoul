"""Identity on a time span of a fight: the passes behind `deepfoul id-golden`.

1. GPU pass (cached): `extract.perceive_span`, the same in-order pass `deepfoul extract` runs on
   a whole fight (detect -> filter -> ByteTrack -> RTMW -> color descriptors from the frame in
   memory). Writes <cache>/<clip>_boxes.parquet + <clip>_frames.parquet. `--from-cache` skips it.
2. Identity (no pixels): `extract.identify`, per-frame assignment against the fight profile, then
   smoothing within tracklets.
3. Render pass (CPU): decode and draw red / blue / merged boxes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deepfoul.perception.appearance import describe
from deepfoul.perception.extract import make_profile, read_boxes, region_cfg
from deepfoul.perception.identity import Profile, build_profile



# ---------------------------------------------------------------- profiles


def profile_samples(row: dict, cfg: dict, detect, pose, raw_dir: Path, fps: float):
    """(box, Appearance) per kept box on frames every sample_every_s over the first seed_s of
    the fight window. Only the start of the fight is used, so this is live-compatible."""
    from deepfoul.io.fights import window
    from deepfoul.io.video import iter_fight_frames
    from deepfoul.perception.filters import drop_reasons

    icfg = cfg["identity"]
    pc, rc = icfg["profile"], region_cfg(icfg)
    start = window(row)[0] or 0.0
    stride = max(1, round(pc["sample_every_s"] * fps))
    samples = []
    for f in iter_fight_frames(row, raw_dir, start, start + pc["seed_s"], stride):
        boxes, scores = detect(f.image)
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        reasons = drop_reasons(boxes, scores, f.image.shape[:2], cfg["tracking"]["filters"])
        kept = boxes[[r == "" for r in reasons]]
        k, s = pose(f.image, kept)
        samples.append([(b, describe(f.image, kk, ss, b, rc, cfg["kpt_conf_min"]))
                        for b, kk, ss in zip(kept, k, s)])
    return samples


def get_profile(row: dict, cfg: dict, detect, pose, raw_dir: Path, cache_dir: Path, fps: float,
                rebuild: bool = False) -> Profile:
    fid = row["fight_id"].strip()
    path = cache_dir / f"{fid}_profile.npz"
    if path.exists() and not rebuild:
        return Profile.load(path)
    from deepfoul.io.fights import window

    kp = cfg["paths"]["keypoints"] / f"{fid}.parquet"
    if kp.exists():   # same rows `deepfoul extract` learned it from
        cols = ["frame_idx", "t", "x1", "y1", "x2", "y2", "hist_shorts", "hist_torso",
                "hist_thigh", "hist_shins", "shin_skin", "torso_skin", "clothed", "torso_len"]
        prof = make_profile(fid, row, read_boxes(kp, cols), window(row)[0] or 0.0, cfg)
    elif detect is None:
        raise RuntimeError(f"{fid}: no cached profile at {path}; run without --from-cache once")
    else:
        samples = profile_samples(row, cfg, detect, pose, raw_dir, fps)
        prof = build_profile(fid, row["red_shorts"], row["blue_shorts"], samples, cfg["identity"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    prof.save(path)
    return prof


# ---------------------------------------------------------------- render


def render_identity(row: dict, start_s: float, end_s: float, tracks: pd.DataFrame,
                    ident: pd.DataFrame, raw_dir: Path, out: Path, show_ref: bool = False,
                    show_none: bool = True) -> None:
    """id-golden's render: the 1.5 overlay renderer on a clip cache (labels in `label`)."""
    from deepfoul.viz.overlay import render_clip

    render_clip(row, start_s, end_s, tracks, ident, raw_dir, out, label_col="label",
                show_ref=show_ref, show_none=show_none)
