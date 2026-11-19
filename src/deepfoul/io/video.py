"""Frame reader: frames in order with original-file timestamps, plus video stream info."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepfoul.io.fights import window
from deepfoul.io.probe import VFR_TOLERANCE, _rate, ffprobe

SEEK_MARGIN_S = 2.0     # seek this far before start_s, then decode forward to the exact frame
TIME_EPS_S = 1e-3       # pts are ms-rounded; well under one frame at 60 fps


@dataclass
class Frame:
    idx: int            # frame index in the original file (round(t * fps))
    t: float            # seconds in the original file, as VLC shows it
    image: np.ndarray   # BGR


def video_info(path: Path) -> dict:
    """ffprobe stats of the first video stream, plus flags for things that break frame timing."""
    probe = ffprobe(path)
    v = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = probe.get("format", {})
    r_fps, avg_fps = _rate(v.get("r_frame_rate")), _rate(v.get("avg_frame_rate"))
    start = float(v.get("start_time") or fmt.get("start_time") or 0.0)
    dur = fmt.get("duration") or v.get("duration")
    info = {
        "width": v.get("width"), "height": v.get("height"),
        "r_frame_rate": v.get("r_frame_rate"), "avg_frame_rate": v.get("avg_frame_rate"),
        "r_fps": r_fps, "avg_fps": avg_fps, "fps": avg_fps or r_fps,
        "frame_count": int(v["nb_frames"]) if v.get("nb_frames") else None,
        "duration_s": float(dur) if dur else None,
        "start_time": start,
        "flags": [],
    }
    if abs(start) > TIME_EPS_S:
        info["flags"].append(f"nonzero start_time ({start:.3f}s)")
    if r_fps and avg_fps and abs(r_fps - avg_fps) / r_fps > VFR_TOLERANCE:
        info["flags"].append(f"variable frame rate (r={r_fps:.3f}, avg={avg_fps:.3f})")
    return info


def iter_frames(path: Path, start_s: float | None = None, end_s: float | None = None,
                stride: int = 1) -> Iterator[Frame]:
    """Yield frames with start_s <= t < end_s, in order, keeping every `stride`-th frame.

    Seeks to SEEK_MARGIN_S before start_s and decodes forward, so the first frame is exactly the
    first one at or after start_s (same frame `ffmpeg -ss start_s` gives).
    """
    import cv2

    if stride < 1:
        raise ValueError("stride must be >= 1")
    start = max(start_s or 0.0, 0.0)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if start > SEEK_MARGIN_S:
            cap.set(cv2.CAP_PROP_POS_MSEC, (start - SEEK_MARGIN_S) * 1000)
        first, kept = True, 0
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if first:
                first = False
                if t > start + TIME_EPS_S and start > SEEK_MARGIN_S:
                    # Seek overshot: start over and decode from the beginning (slow but exact).
                    cap.release()
                    cap = cv2.VideoCapture(str(path))
                    continue
            if t < start - TIME_EPS_S:
                continue
            if end_s is not None and t >= end_s - TIME_EPS_S:
                break
            if kept % stride == 0:
                ok, img = cap.retrieve()
                if not ok:
                    break
                yield Frame(idx=round(t * fps), t=t, image=img)
            kept += 1
    finally:
        cap.release()


def iter_fight_frames(row: dict[str, str], raw_dir: Path, start_s: float | None = None,
                      end_s: float | None = None, stride: int = 1) -> Iterator[Frame]:
    """iter_frames on <raw_dir>/<fight_id>.mp4, defaulting to (and clamped to) the fight window."""
    w_start, w_end = window(row)
    start = max(x for x in (start_s, w_start, 0.0) if x is not None)
    ends = [x for x in (end_s, w_end) if x is not None]
    end = min(ends) if ends else None
    yield from iter_frames(raw_dir / f"{row['fight_id'].strip()}.mp4", start, end, stride)


def ffmpeg_frame(path: Path, t: float, width: int, height: int) -> np.ndarray:
    """One BGR frame decoded by ffmpeg with an accurate input seek. Independent of OpenCV."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.6f}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, np.uint8)[: width * height * 3].reshape(height, width, 3)


def frame_check(path: Path, times: list[float]) -> list[dict]:
    """Check the reader's timeline against ffmpeg and against decoding from further back.

    For each t: the reader's first frame at t vs ffmpeg's frame at t (mean abs pixel diff, and
    which of the reader's neighbouring frames is closest), and the frame reached by seeking to t
    vs the same index reached by decoding from 10 s earlier.
    """
    info = video_info(path)
    w, h = info["width"], info["height"]
    rows = []
    for t in times:
        near = list(iter_frames(path, t - 1.5 / info["fps"], t + 2.5 / info["fps"]))
        seek = next(iter_frames(path, t))
        ref = ffmpeg_frame(path, t, w, h).astype(np.int16)
        diffs = {f.idx - seek.idx: float(np.abs(f.image.astype(np.int16) - ref).mean())
                 for f in near}
        earlier = next(f for f in iter_frames(path, max(t - 10.0, 0.0), t + 1.0)
                       if f.idx == seek.idx)
        rows.append({
            "t": t, "idx": seek.idx, "frame_t": seek.t,
            "diff_ffmpeg": diffs.get(0),
            "best_offset": min(diffs, key=diffs.get),
            "diffs": diffs,
            "seek_vs_linear": float(np.abs(seek.image.astype(np.int16)
                                           - earlier.image.astype(np.int16)).mean()),
            "seek_t_vs_linear_t": abs(seek.t - earlier.t),
        })
    return rows
