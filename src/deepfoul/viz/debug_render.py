"""Debug overlay for 1.2: tracked boxes with ids, unconfirmed and dropped boxes, time and shot."""

from __future__ import annotations

import numpy as np

from deepfoul.io.previews import frame_label, put_label

GREY = (150, 150, 150)
WHITE = (255, 255, 255)


def draw_tracking(img: np.ndarray, res, annotators=None) -> np.ndarray:
    """res: a track.FrameResult. Tracked boxes in a color per track id (label "<id> <score>"),
    kept-but-unconfirmed boxes thin white, filtered-out boxes thin grey with the rule name."""
    import cv2
    import supervision as sv

    box_ann, label_ann = annotators or make_annotators(img.shape[0])
    scale = img.shape[0] / 720
    for r in res.rows:
        if r["track_id"]:
            continue
        p1, p2 = (round(r["x1"]), round(r["y1"])), (round(r["x2"]), round(r["y2"]))
        color = WHITE if r["kept"] else GREY
        cv2.rectangle(img, p1, p2, color, 1, cv2.LINE_AA)
        text = f"? {r['score']:.2f}" if r["kept"] else f"{r['drop_reason']} {r['score']:.2f}"
        cv2.putText(img, text, (p1[0] + 2, p2[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale,
                    color, 1, cv2.LINE_AA)
    tracked = [r for r in res.rows if r["track_id"]]
    if tracked:
        ids = [r["track_id"] for r in tracked]
        det = sv.Detections(
            xyxy=np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in tracked], np.float32),
            confidence=np.array([r["score"] for r in tracked], np.float32),
            tracker_id=np.array([int(i.split("-")[1]) for i in ids]))
        img = box_ann.annotate(img, det)
        img = label_ann.annotate(img, det, labels=[f"{i} {r['score']:.2f}"
                                                   for i, r in zip(ids, tracked)])
    shot = "-" if res.shot_id is None else res.shot_id
    label = f"{frame_label(res.t, res.idx)}  shot {shot}" + ("  CUT" if res.guard else "")
    return put_label(img, label, scale)


def make_annotators(frame_h: int):
    import supervision as sv

    scale = frame_h / 720
    return (sv.BoxAnnotator(thickness=max(2, round(2 * scale)),
                            color_lookup=sv.ColorLookup.TRACK),
            sv.LabelAnnotator(text_scale=0.5 * scale, text_thickness=max(1, round(scale)),
                              color_lookup=sv.ColorLookup.TRACK))
