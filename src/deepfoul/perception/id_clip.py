"""Identity on a time span of a fight: the passes behind `deepfoul id-golden`.

1. GPU pass (cached): detect -> filter -> ByteTrack -> RTMW on kept boxes, in frame order.
   Writes <cache>/<clip>_tracks.parquet and <clip>_pose.npz. `--from-cache` skips it.
2. Descriptor pass (CPU): re-decode the frames and read each kept box's colors from its
   pose-guided regions.
3. Identity (no pixels): per-frame assignment against the fight profile, then smoothing within
   tracklets.
4. Render pass (CPU): re-decode and draw red / blue / merged boxes.

The passes are split for tuning speed (identity re-runs without the GPU). Each one only reads
frames up to t + lookahead, so the same logic runs streaming with a short frame buffer.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from deepfoul.perception.appearance import describe
from deepfoul.perception.identity import BoxObs, Profile, assign_frame, build_profile, smooth_labels

RED, BLUE, PURPLE, GREY, REF = (40, 40, 230), (230, 120, 30), (200, 60, 200), (150, 150, 150), (0, 215, 255)


def region_cfg(icfg: dict) -> dict:
    return {**icfg["regions"], "sat_min": icfg["sat_min"], "v_min": icfg["v_min"]}


# ---------------------------------------------------------------- pose cache


def save_pose(path: Path, frame_idx, det_idx, kpts, scores) -> None:
    np.savez_compressed(path, frame_idx=np.asarray(frame_idx, np.int64),
                        det_idx=np.asarray(det_idx, np.int64),
                        kpts=np.asarray(kpts, np.float32).reshape(-1, 133, 2),
                        scores=np.asarray(scores, np.float32).reshape(-1, 133))


def load_pose(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    z = np.load(path)
    return {(int(f), int(d)): (k, s) for f, d, k, s in
            zip(z["frame_idx"], z["det_idx"], z["kpts"], z["scores"])}


# ---------------------------------------------------------------- 1. GPU pass


def track_pose_clip(row: dict, start_s: float, end_s: float, shots: list[dict], fps: float,
                    detect, pose, cfg: dict, raw_dir: Path):
    """Returns (tracks DataFrame, pose arrays, timing dict)."""
    from deepfoul.io.video import iter_fight_frames
    from deepfoul.perception.track import COLUMNS, make_tracker, track_stream

    tcfg = cfg["tracking"]
    tracker = make_tracker(tcfg["bytetrack"], fps)
    rows, meta = [], []
    pf, pd_, pk, ps = [], [], [], []
    det_s = pose_s = 0.0
    t0 = time.perf_counter()
    for res in track_stream(iter_fight_frames(row, raw_dir, start_s, end_s), shots, detect,
                            tracker, tcfg["filters"], tcfg["cut_guard_frames"], pose):
        rows.extend(res.rows)
        meta.append((res.idx, res.t, res.shot_id, res.guard))
        det_s += res.det_s
        pose_s += res.pose_s
        di, k, s = res.pose
        pf += [res.idx] * len(di)
        pd_ += list(di)
        pk += list(k)
        ps += list(s)
    wall = time.perf_counter() - t0
    timing = {"frames": len(meta), "det_s": det_s, "pose_s": pose_s, "wall_s": wall,
              "pose_boxes": len(pf)}
    return pd.DataFrame(rows, columns=COLUMNS), meta, (pf, pd_, pk, ps), timing


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
    if detect is None:
        raise RuntimeError(f"{fid}: no cached profile at {path}; run without --from-cache once")
    samples = profile_samples(row, cfg, detect, pose, raw_dir, fps)
    icfg = cfg["identity"]
    prof = build_profile(fid, row["red_shorts"], row["blue_shorts"], samples,
                         {**icfg, "colors": icfg["colors"]})
    cache_dir.mkdir(parents=True, exist_ok=True)
    prof.save(path)
    return prof


# ---------------------------------------------------------------- 2-3. descriptors + identity


def identify_clip(row: dict, start_s: float, end_s: float, tracks: pd.DataFrame,
                  pose: dict, prof: Profile, cfg: dict, raw_dir: Path, fps: float) -> pd.DataFrame:
    """One row per kept box: tracks columns + appearance summary + raw_label / label / cost."""
    from deepfoul.io.video import iter_fight_frames

    icfg = cfg["identity"]
    rc = region_cfg(icfg)
    kept = tracks[tracks["kept"]].copy()
    by_frame = {f: g for f, g in kept.groupby("frame_idx")}
    out = []
    for f in iter_fight_frames(row, raw_dir, start_s, end_s):
        g = by_frame.get(f.idx)
        if g is None:
            continue
        obs, recs = [], []
        for _, r in g.iterrows():
            box = np.array([r.x1, r.y1, r.x2, r.y2], np.float32)
            kp = pose.get((int(r.frame_idx), int(r.det_idx)))
            if kp is None:
                continue
            app = describe(f.image, kp[0], kp[1], box, rc, cfg["kpt_conf_min"])
            o = BoxObs(box, app)
            obs.append(o)
            recs.append(r)
        if not obs:
            continue
        for r, o, a in zip(recs, obs, assign_frame(obs, prof, icfg)):
            out.append({**r.to_dict(), "clothed": o.app.clothed, "shin_skin": o.app.shin_skin,
                        "torso_skin": o.app.torso_skin,
                        "has_shorts": o.app.shorts is not None, "raw_label": a["label"],
                        "cost": a["cost"], "margin": a["margin"], "fallback": a["fallback"],
                        "cost_red": a["cost_red"], "cost_blue": a["cost_blue"],
                        "cost_ref": a["cost_ref"]})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return smooth_labels(df, fps, icfg)


# ---------------------------------------------------------------- 4. render


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
