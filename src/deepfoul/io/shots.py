"""Camera-cut (shot) detection with PySceneDetect, cached per fight.

Shots are half-open: shot i covers start_frame <= idx < end_frame (start_s <= t < end_s).
A frame exactly at a cut belongs to the new shot. 1.2 resets tracks whenever the shot id
of consecutive frames changes; prefer `shot_index_at_frame` (exact integers) over seconds.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from pathlib import Path

import scenedetect
from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video


def first_frame_at(t: float, fps: float) -> int:
    """Index of the first frame with time >= t (same rule as the frame reader)."""
    return max(0, math.ceil(t * fps - 1e-3 * fps))


def shots_cfg_for(shots_cfg: dict, fight_id: str) -> dict:
    """The shots config with this fight's `per_fight` overrides applied."""
    base = {k: v for k, v in shots_cfg.items() if k != "per_fight"}
    return {**base, **(shots_cfg.get("per_fight") or {}).get(fight_id, {})}


def detector_params(shots_cfg: dict, fps: float) -> dict:
    """Detector name + params as used. min_scene_len_s is converted with this video's fps."""
    name = shots_cfg["detector"]
    params = {"name": name}
    if name == "adaptive":
        params["adaptive_threshold"] = float(shots_cfg["adaptive_threshold"])
        params["min_content_val"] = float(shots_cfg["min_content_val"])
    elif name == "content":
        params["content_threshold"] = float(shots_cfg["content_threshold"])
    else:
        raise ValueError(f"unknown shot detector {name!r}")
    params["min_scene_len_s"] = float(shots_cfg["min_scene_len_s"])
    params["min_scene_len_frames"] = max(1, round(shots_cfg["min_scene_len_s"] * fps))
    return params


def make_detector(params: dict):
    if params["name"] == "adaptive":
        return AdaptiveDetector(adaptive_threshold=params["adaptive_threshold"],
                                min_scene_len=params["min_scene_len_frames"],
                                min_content_val=params["min_content_val"])
    return ContentDetector(threshold=params["content_threshold"],
                           min_scene_len=params["min_scene_len_frames"])


def shots_from_cuts(cut_frames: list[int], start_frame: int, end_frame: int,
                    fps: float) -> list[dict]:
    """Half-open shots tiling [start_frame, end_frame), split at every cut strictly inside it."""
    inner = sorted({c for c in cut_frames if start_frame < c < end_frame})
    bounds = [start_frame, *inner, end_frame]
    return [{"shot_id": i, "start_frame": a, "end_frame": b,
             "start_s": round(a / fps, 4), "end_s": round(b / fps, 4)}
            for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]


def detect_shots(video: Path, start_s: float, end_s: float | None, shots_cfg: dict) -> dict:
    """Run the detector over [start_s, end_s) and return the cache document (minus fight_id).

    Decodes from frame 0 (no seek) so PySceneDetect's frame numbers match the file exactly;
    cuts before start_s are dropped.
    """
    stream = open_video(str(video))
    fps = float(stream.frame_rate)
    params = detector_params(shots_cfg, fps)
    sm = SceneManager()
    sm.add_detector(make_detector(params))
    sm.detect_scenes(stream, end_time=end_s)
    total = stream.duration.frame_num
    start_frame = first_frame_at(start_s, fps)
    end_frame = min(first_frame_at(end_s, fps), total) if end_s is not None else total
    cuts = [c.frame_num for c in sm.get_cut_list(show_warning=False)]
    return {
        "fps": fps,
        "window": [start_s, end_s],
        "detector": params,
        "scenedetect_version": scenedetect.__version__,
        "intervals": "half-open [start, end)",
        "shots": shots_from_cuts(cuts, start_frame, end_frame, fps),
    }


def shots_path(cache_dir: Path, fight_id: str) -> Path:
    return cache_dir / f"{fight_id}_shots.json"


def save_shots(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load_shots(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stale_reason(doc: dict, window: tuple[float, float | None], shots_cfg: dict) -> str | None:
    """Why a cached shot file no longer matches the config / fight window, or None if it's fresh."""
    want = detector_params(shots_cfg, doc["fps"])
    if doc.get("detector") != want:
        return f"detector params changed ({doc.get('detector')} -> {want})"
    if doc.get("scenedetect_version") != scenedetect.__version__:
        return f"scenedetect {doc.get('scenedetect_version')} -> {scenedetect.__version__}"
    if list(doc.get("window", [])) != list(window):
        return f"fight window changed ({doc.get('window')} -> {list(window)})"
    return None


def shot_index_at_frame(shots: list[dict], idx: int) -> int | None:
    """Shot id containing frame idx, or None outside the window."""
    i = bisect_right([s["start_frame"] for s in shots], idx) - 1
    return shots[i]["shot_id"] if i >= 0 and idx < shots[i]["end_frame"] else None


def shot_index_at(shots: list[dict], t: float) -> int | None:
    """Shot id containing time t (seconds), or None outside the window."""
    i = bisect_right([s["start_s"] for s in shots], t) - 1
    return shots[i]["shot_id"] if i >= 0 and t < shots[i]["end_s"] else None


def cut_frames(doc: dict) -> list[int]:
    return [s["start_frame"] for s in doc["shots"][1:]]
