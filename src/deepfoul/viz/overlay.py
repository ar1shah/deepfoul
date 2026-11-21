"""Overlay renderer v1 (1.5): red / blue / merged boxes, corner tags, optional skeleton, timestamp.

Everything is drawn from cached rows (data/keypoints/<fight_id>.parquet or an id-golden clip
cache), so rendering needs no GPU. Only frames that are in the cache are written, so with a frame
rate cap (stride 2) the output plays at the processed rate and every overlay is from its own frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RED, BLUE, PURPLE, GREY, REF = (40, 40, 230), (230, 120, 30), (200, 60, 200), (150, 150, 150), (0, 215, 255)
CORNER_COLOR = {"red": RED, "blue": BLUE, "merged": PURPLE}

# COCO-WholeBody 133 points worth checking by eye (checked against rtmlib's coco133 skeleton)
FINGERTIPS = [91 + o for o in (4, 8, 12, 16, 20)] + [112 + o for o in (4, 8, 12, 16, 20)]
EYES = [1, 2] + list(range(59, 71))          # body-set eyes + the 12 face eye-contour points
LOWER = [11, 12, 13, 14, 15, 16] + list(range(17, 23))   # hips, knees, ankles, feet
HIGHLIGHTS = {"poke": FINGERTIPS + EYES, "low_blow": LOWER, "both": FINGERTIPS + EYES + LOWER,
              "none": []}


def read_range(path: Path, start_frame: int, end_frame: int,
               columns: list[str] | None = None) -> pd.DataFrame:
    """Rows of a keypoints / frames parquet with start_frame <= frame_idx < end_frame, read with
    a pyarrow filter so a whole-fight file isn't loaded."""
    df = pd.read_parquet(path, columns=columns, filters=[("frame_idx", ">=", int(start_frame)),
                                                         ("frame_idx", "<", int(end_frame))])
    for c in [c for c in df.columns if c in ("kpts", "kpt_scores")]:
        df[c] = [None if v is None else np.asarray(v, np.float32) for v in df[c]]
    return df


def tag(img, text, p1, color, scale):
    import cv2

    fs = 0.55 * scale
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    x, y = p1[0], max(int(th + 6), p1[1])
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                max(1, round(scale)), cv2.LINE_AA)


def draw_skeleton(img, kpts: np.ndarray, scores: np.ndarray, color, kpt_thr: float,
                  highlight: list[int], scale: float):
    """rtmlib's coco133 skeleton (points under kpt_thr are skipped, i.e. masked points are never
    drawn), then the highlight points as larger dots with a white rim in the corner color."""
    import cv2
    from rtmlib import draw_skeleton as rtm_draw

    k = np.asarray(kpts, np.float32).reshape(-1, 2)
    s = np.asarray(scores, np.float32).reshape(-1)
    img = rtm_draw(img, k[None], s[None], kpt_thr=kpt_thr,   # rtmlib wants (instances, 133, 2) radius=max(1, round(1.5 * scale)),
                   line_width=max(1, round(1.5 * scale)))
    r = max(2, round(3 * scale))
    for i in highlight:
        if s[i] >= kpt_thr:
            c = (int(k[i, 0]), int(k[i, 1]))
            cv2.circle(img, c, r + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
    return img


def draw_frame(img, rows: pd.DataFrame, names: dict, label_col: str = "corner",
               skeleton: bool = False, show_ref: bool = False, show_none: bool = True,
               kpt_thr: float = 0.3, highlight: list[int] | None = None):
    """Draw one frame's rows. Skeletons only on red / blue / merged boxes, never on the ref,
    crowd or officials."""
    import cv2

    scale = img.shape[0] / 720
    highlight = HIGHLIGHTS["both"] if highlight is None else highlight
    for r in rows.itertuples(index=False):
        lab = getattr(r, label_col)
        p1, p2 = (round(r.x1), round(r.y1)), (round(r.x2), round(r.y2))
        if lab in CORNER_COLOR:
            color = CORNER_COLOR[lab]
            if skeleton and getattr(r, "kpts", None) is not None:
                img = draw_skeleton(img, r.kpts, r.kpt_scores, color, kpt_thr, highlight, scale)
            cv2.rectangle(img, p1, p2, color, max(2, round(3 * scale)), cv2.LINE_AA)
            tag(img, f"{lab.upper()} {names[lab]}" if lab != "merged" else "RED+BLUE", p1,
                color, scale)
        elif lab == "ref" and show_ref:
            cv2.rectangle(img, p1, p2, REF, 1, cv2.LINE_AA)
            tag(img, "ref" + (" (3rd)" if getattr(r, "fallback", False) else ""), p1, REF,
                scale * 0.8)
        elif lab == "none" and show_none:
            cv2.rectangle(img, p1, p2, GREY, 1, cv2.LINE_AA)
    return img


def fighter_names(row: dict) -> dict:
    return {"red": row["red_fighter"].split()[-1], "blue": row["blue_fighter"].split()[-1]}


def render_clip(row: dict, start_s: float, end_s: float, frames: pd.DataFrame,
                boxes: pd.DataFrame, raw_dir: Path, out: Path, label_col: str = "corner",
                skeleton: bool = False, show_ref: bool = False, show_none: bool = True,
                kpt_thr: float = 0.3, highlight: list[int] | None = None,
                speed: float = 1.0) -> int:
    """Decode [start_s, end_s) and write every frame that is in `frames` (the processed grid) with
    its overlay. Output rate = processed rate x speed; audio only at speed 1. Returns frames
    written."""
    from deepfoul.io.previews import clip_writer, frame_label, put_label
    from deepfoul.io.video import iter_fight_frames, video_info

    video = raw_dir / f"{row['fight_id'].strip()}.mp4"
    names = fighter_names(row)
    meta = frames.set_index("frame_idx")[["shot_id", "guard"]]
    by_frame = {f: g for f, g in boxes.groupby("frame_idx")} if not boxes.empty else {}
    stride = _stride(frames)
    from fractions import Fraction

    info = video_info(video)
    src = Fraction(info["r_frame_rate"]) if info.get("r_frame_rate") else Fraction(info["fps"])
    rate = src / stride * Fraction(speed).limit_denominator(1000)
    fps_out = f"{rate.numerator}/{rate.denominator}"
    empty = boxes.iloc[:0]
    n = 0
    cm = write = None
    try:
        for f in iter_fight_frames(row, raw_dir, start_s, end_s):
            if f.idx not in meta.index:
                continue
            img = f.image
            if write is None:
                cm = clip_writer(video, f.t, end_s, (img.shape[1], img.shape[0]), out,
                                 fps=fps_out, audio=speed == 1.0)
                write = cm.__enter__()
            img = draw_frame(img, by_frame.get(f.idx, empty), names, label_col, skeleton,
                             show_ref, show_none, kpt_thr, highlight)
            shot, guard = meta.at[f.idx, "shot_id"], bool(meta.at[f.idx, "guard"])
            text = f"{frame_label(f.t, f.idx)}  shot {'-' if pd.isna(shot) else shot}"
            text += ("  CUT" if guard else "") + (f"  x{speed:g}" if speed != 1.0 else "")
            write(put_label(img, text, img.shape[0] / 720))
            n += 1
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
    return n


def _stride(frames: pd.DataFrame) -> int:
    d = np.diff(np.sort(frames["frame_idx"].to_numpy()))
    return int(np.median(d)) if len(d) else 1


# ---------------------------------------------------------------- contact sheets


def sheet_frames(boxes: pd.DataFrame, frames: pd.DataFrame, kind: str) -> np.ndarray:
    """frame_idx values for a sheet: 'merged' = a merged box; 'single' = exactly one fighter
    (red or blue, no merged box)."""
    has = {c: set(boxes.loc[boxes["corner"] == c, "frame_idx"]) for c in ("red", "blue",
                                                                          "merged")}
    fi = frames["frame_idx"]
    red, blue, mer = (fi.isin(has[c]) for c in ("red", "blue", "merged"))
    if kind == "merged":
        sel = mer
    elif kind == "single":
        sel = (red ^ blue) & ~mer
    else:
        raise ValueError(f"unknown sheet kind {kind!r}")
    return fi[sel].to_numpy()


def write_sheet(row: dict, picks: list[int], boxes: pd.DataFrame, frames: pd.DataFrame,
                raw_dir: Path, out: Path, fps: float, kpt_thr: float = 0.3,
                captions: dict | None = None, cols: int = 4, tile_w: int = 480) -> None:
    """Contact sheet of single frames (ffmpeg accurate seek), each drawn with the overlay and
    labelled with its time (and an optional caption, e.g. a tile number)."""
    import cv2

    from deepfoul.io.previews import fmt_ts, put_label
    from deepfoul.io.video import ffmpeg_frame, video_info

    video = raw_dir / f"{row['fight_id'].strip()}.mp4"
    info = video_info(video)
    names = fighter_names(row)
    tmap = frames.set_index("frame_idx")["t"]
    tiles = []
    for i, fidx in enumerate(picks):
        t = float(tmap[fidx])
        img = ffmpeg_frame(video, fidx / fps, info["width"], info["height"]).copy()
        img = draw_frame(img, boxes[boxes["frame_idx"] == fidx], names, skeleton=False,
                         show_ref=True, show_none=True, kpt_thr=kpt_thr)
        img = cv2.resize(img, (tile_w, round(img.shape[0] * tile_w / img.shape[1])))
        cap = f"#{i + 1} {fmt_ts(t)}" + (f" {captions[fidx]}" if captions and fidx in captions
                                          else "")
        tiles.append(put_label(img, cap, 0.5))
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
