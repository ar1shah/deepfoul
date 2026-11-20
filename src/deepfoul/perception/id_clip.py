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

RED, BLUE, PURPLE, GREY, REF = (40, 40, 230), (230, 120, 30), (200, 60, 200), (150, 150, 150), (0, 215, 255)


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
    import cv2

    from deepfoul.io.previews import clip_writer, frame_label, put_label
    from deepfoul.io.video import iter_fight_frames

    names = {"red": row["red_fighter"].split()[-1], "blue": row["blue_fighter"].split()[-1]}
    meta = tracks.groupby("frame_idx").agg(shot_id=("shot_id", "first"), guard=("guard", "first"))
    by_frame = {f: g for f, g in ident.groupby("frame_idx")} if not ident.empty else {}
    video = raw_dir / f"{row['fight_id'].strip()}.mp4"
    write = None
    with_stack = []
    try:
        for f in iter_fight_frames(row, raw_dir, start_s, end_s):
            img = f.image
            h, w = img.shape[:2]
            scale = h / 720
            if write is None:
                cm = clip_writer(video, f.t, end_s, (w, h), out)
                write = cm.__enter__()
                with_stack.append(cm)
            for _, r in (by_frame.get(f.idx, pd.DataFrame()).iterrows()):
                lab = r["label"]
                p1, p2 = (round(r.x1), round(r.y1)), (round(r.x2), round(r.y2))
                if lab in ("red", "blue", "merged"):
                    color = {"red": RED, "blue": BLUE, "merged": PURPLE}[lab]
                    text = (f"{lab.upper()} {names[lab]}" if lab != "merged" else "RED+BLUE")
                    cv2.rectangle(img, p1, p2, color, max(2, round(3 * scale)), cv2.LINE_AA)
                    _tag(img, text, p1, color, scale)
                elif lab == "ref" and show_ref:
                    cv2.rectangle(img, p1, p2, REF, 1, cv2.LINE_AA)
                    _tag(img, "ref" + (" (3rd)" if r.get("fallback") else ""), p1, REF, scale * 0.8)
                elif lab == "none" and show_none:
                    cv2.rectangle(img, p1, p2, GREY, 1, cv2.LINE_AA)
            shot = meta["shot_id"].get(f.idx, None)
            guard = bool(meta["guard"].get(f.idx, False))
            label = f"{frame_label(f.t, f.idx)}  shot {'-' if shot is None else shot}"
            write(put_label(img, label + ("  CUT" if guard else ""), scale))
    finally:
        for cm in with_stack:
            cm.__exit__(None, None, None)


def _tag(img, text, p1, color, scale):
    import cv2

    fs = 0.55 * scale
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    x, y = p1[0], max(int(th + 6), p1[1])
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                max(1, round(scale)), cv2.LINE_AA)
