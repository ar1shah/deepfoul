"""ByteTrack within each shot.

Frames are processed strictly in order. On a shot change the tracker is reset, and frames within
`cut_guard_frames` of a cut (with 1: the frame before the cut and the cut frame) are "guard"
frames: detected and drawn, but not fed to the tracker, so a frame blended across a cut can't
seed or corrupt a track. Knowing that the *next* frame is a cut is 1 frame of lookahead, well
inside the 2.0 s budget; shots come from the cached shot file.

Track ids restart at every reset, so they are made unique per fight as "<shot_id>-<local_id>".
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from deepfoul.io.shots import shot_index_at_frame
from deepfoul.perception.filters import drop_reasons

COLUMNS = ["frame_idx", "t", "shot_id", "guard", "det_idx", "x1", "y1", "x2", "y2", "score",
           "kept", "drop_reason", "track_id"]


@dataclass
class FrameResult:
    idx: int
    t: float
    shot_id: int | None
    guard: bool
    rows: list[dict] = field(default_factory=list)
    det_s: float = 0.0
    image: np.ndarray | None = field(default=None, repr=False)  # the frame, for drawing


def guard_frames(shots: list[dict], n: int) -> set[int]:
    """Frame indices within n frames of a cut: cut - n <= idx < cut + n."""
    cuts = [s["start_frame"] for s in shots[1:]]
    return {c + k for c in cuts for k in range(-n, n)}


def make_tracker(bt_cfg: dict, fps: float):
    from trackers import ByteTrackTracker

    return ByteTrackTracker(frame_rate=fps, **bt_cfg)


def track_stream(frames: Iterable, shots: list[dict], detect: Callable, tracker,
                 filter_cfg: dict, guard_n: int = 1) -> Iterator[FrameResult]:
    """Detect, filter and track each frame (objects with .idx, .t, .image) in order."""
    import supervision as sv

    guards = guard_frames(shots, guard_n)
    prev_shot = object()
    for f in frames:
        shot = shot_index_at_frame(shots, f.idx)
        if shot != prev_shot:
            tracker.reset()
            prev_shot = shot
        t0 = time.perf_counter()
        boxes, scores = detect(f.image)
        det_s = time.perf_counter() - t0
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        reasons = drop_reasons(boxes, scores, f.image.shape[:2], filter_cfg)
        kept = np.array([r == "" for r in reasons], dtype=bool)
        guard = f.idx in guards or shot is None
        tids = [""] * len(boxes)
        if not guard:
            keep_idx = np.flatnonzero(kept)
            dets = sv.Detections(xyxy=boxes[keep_idx], confidence=scores[keep_idx],
                                 data={"det_idx": keep_idx})
            out = tracker.update(dets)
            pairs = zip(out.data["det_idx"], out.tracker_id) if len(out) else []
            for di, tid in pairs:
                if tid >= 0:
                    tids[int(di)] = f"{shot}-{int(tid)}"
        rows = [{"frame_idx": f.idx, "t": f.t, "shot_id": shot, "guard": guard, "det_idx": i,
                 "x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                 "score": float(s), "kept": bool(k), "drop_reason": r, "track_id": tid}
                for i, (b, s, k, r, tid) in enumerate(zip(boxes, scores, kept, reasons, tids))]
        yield FrameResult(f.idx, f.t, shot, guard, rows, det_s, f.image)



def track_clip(row: dict[str, str], start_s: float, end_s: float, shots: list[dict], fps: float,
               detect: Callable, cfg: dict, raw_dir, render_to=None):
    """Track [start_s, end_s) of a fight, optionally rendering the debug overlay to `render_to`.

    Returns (DataFrame with every detection, list of (frame_idx, t, shot_id, guard) per frame,
    total detection seconds).
    """
    import pandas as pd

    from deepfoul.io.previews import clip_writer
    from deepfoul.io.video import iter_fight_frames
    from deepfoul.viz.debug_render import draw_tracking, make_annotators

    tcfg = cfg["tracking"]
    tracker = make_tracker(tcfg["bytetrack"], fps)
    frames = iter_fight_frames(row, raw_dir, start_s, end_s)
    video = raw_dir / f"{row['fight_id'].strip()}.mp4"
    rows, meta, det_s = [], [], 0.0
    write = anns = None
    with ExitStack() as stack:
        for res in track_stream(frames, shots, detect, tracker, tcfg["filters"],
                                tcfg["cut_guard_frames"]):
            rows.extend(res.rows)
            meta.append((res.idx, res.t, res.shot_id, res.guard))
            det_s += res.det_s
            if render_to is not None:
                if write is None:
                    h, w = res.image.shape[:2]
                    anns = make_annotators(h)
                    write = stack.enter_context(clip_writer(video, res.t, end_s, (w, h), render_to))
                write(draw_tracking(res.image, res, anns))
    return pd.DataFrame(rows, columns=COLUMNS), meta, det_s
