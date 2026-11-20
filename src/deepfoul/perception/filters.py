"""Geometric box filters that drop the crowd and camera crew. Thresholds from `tracking.filters`.

Deliberately loose: picking the fighters and the ref out of what survives is 1.3's job.
"""

from __future__ import annotations

import numpy as np

RULES = ("small", "rel_small", "high", "edge", "max_people")
GEOM_RULES = RULES[:-1]


def drop_reasons(boxes: np.ndarray, scores: np.ndarray, frame_hw: tuple[int, int],
                 cfg: dict) -> list[str]:
    """One string per box: "" if kept, else every failed rule joined with "+".

    small:      box height < min_height_frac of the frame
    rel_small:  box height < rel_min_height_frac of the tallest box scoring >= ref_score_min
                (background crowd in tight shots, where everyone in the cage is huge)
    high:       box bottom above min_bottom_frac of the frame (feet in the crowd behind the cage)
    edge:       touches the left/right border (within border_margin_frac) and is shorter than
                edge_max_height_frac (partial camera crew / officials at the frame edge)
    max_people: survived the rules above but is beyond the max_people tallest boxes
    """
    h, w = frame_hw
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    bh = (boxes[:, 3] - boxes[:, 1]) / h
    margin = cfg["border_margin_frac"] * w
    at_edge = (boxes[:, 0] <= margin) | (boxes[:, 2] >= w - margin)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    confident = bh[scores >= cfg["ref_score_min"]]
    ref_h = confident.max() if len(confident) else 0.0
    flags = {
        "small": bh < cfg["min_height_frac"],
        "rel_small": bh < cfg["rel_min_height_frac"] * ref_h,
        "high": boxes[:, 3] / h < cfg["min_bottom_frac"],
        "edge": at_edge & (bh < cfg["edge_max_height_frac"]),
    }
    reasons = [[r for r in GEOM_RULES if flags[r][i]] for i in range(len(boxes))]
    survivors = [i for i, r in enumerate(reasons) if not r]
    for i in sorted(survivors, key=lambda i: -bh[i])[cfg["max_people"]:]:
        reasons[i].append("max_people")
    return ["+".join(r) for r in reasons]
