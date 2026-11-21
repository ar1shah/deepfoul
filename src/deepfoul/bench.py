"""`deepfoul bench`: processing speed of every pipeline stage on this machine's GPU.

Runs on a fixed span of a fight (configs `bench:`), after an untimed warm-up:
decode only, then the full extract pass (`perceive_span`: detect + track + RTMW + color
descriptors, with per-stage timers), identity from those rows, and the overlay renderer with
skeletons from the keypoint cache. Peak GPU memory is sampled with nvidia-smi during the pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from datetime import date
from pathlib import Path


def summarize_timing(t: dict) -> dict:
    """Per-stage numbers from a perceive_span timing dict.

    Detection and pose are synchronous ONNX calls timed inside the pass, so their fps is the
    rate that stage alone could run at. Descriptors run on CPU threads that overlap the GPU, so
    their cost per frame is CPU time, not wall time."""
    n, boxes = t["frames"], t["boxes"]

    def rate(s):
        return n / s if s else None

    return {"frames": n, "boxes_per_frame": boxes / n if n else 0.0,
            "det_ms": 1000 * t["det_s"] / n, "det_fps": rate(t["det_s"]),
            "pose_ms": 1000 * t["pose_s"] / n, "pose_fps": rate(t["pose_s"]),
            "pose_ms_per_box": 1000 * t["pose_s"] / boxes if boxes else None,
            "desc_ms": 1000 * t["desc_s"] / n, "desc_workers": t.get("describe_workers", 0),
            "pass_fps": rate(t["wall_s"]), "pass_ms": 1000 * t["wall_s"] / n}


class VramSampler:
    """Peak `memory.used` (MiB, whole GPU) from nvidia-smi every `every_s` while active."""

    def __init__(self, every_s: float = 0.5):
        self.every_s, self.peak, self._stop = every_s, None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def read() -> int | None:
        if not shutil.which("nvidia-smi"):
            return None
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True, timeout=10, check=True).stdout
            return int(out.strip().splitlines()[0])
        except (subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _run(self):
        while not self._stop.is_set():
            v = self.read()
            if v is not None:
                self.peak = max(self.peak or 0, v)
            self._stop.wait(self.every_s)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def fight_frames(kp_dir: Path, split_ok=("train", "val"), fights: dict | None = None) -> int:
    """Processed frames over every cached non-test fight (for the batch-time estimate)."""
    import pyarrow.parquet as pq

    n = 0
    for p in kp_dir.glob("*_frames.parquet"):
        fid = p.name[: -len("_frames.parquet")]
        if fights and fights.get(fid, {}).get("split", "").strip() not in split_ok:
            continue
        n += pq.read_metadata(p).num_rows
    return n


def run_bench(cfg: dict, row: dict, start_s: float, seconds: float, detect, pose,
              out_dir: Path, machine: dict, fights: dict) -> dict:
    from deepfoul.io.fights import window
    from deepfoul.io.shots import first_frame_at, load_shots, shots_path
    from deepfoul.io.video import iter_fight_frames
    from deepfoul.perception.extract import (grid_frame, identify, keypoints_paths,
                                             perceive_span, stride_for)
    from deepfoul.perception.identity import Profile
    from deepfoul.viz.overlay import read_range, render_clip

    fid = row["fight_id"].strip()
    doc = load_shots(shots_path(cfg["paths"]["cache"], fid))
    fps = doc["fps"]
    stride = stride_for(fps, cfg["extract"]["target_fps"])
    w0 = first_frame_at(window(row)[0] or 0.0, fps)
    warm = cfg["bench"]["warmup_s"]
    f_warm = grid_frame(first_frame_at(start_s - warm, fps), w0, stride)
    f0 = grid_frame(first_frame_at(start_s, fps), w0, stride)
    f1 = first_frame_at(start_s + seconds, fps)
    raw = cfg["paths"]["raw"]
    res: dict = {"fight_id": fid, "start_s": start_s, "seconds": seconds, "fps": fps,
                 "stride": stride, **machine}

    # decode only (same reader and stride as the pass)
    t0 = time.perf_counter()
    nd = sum(1 for _ in iter_fight_frames(row, raw, f0 / fps, f1 / fps, stride))
    res["decode_fps"] = nd / (time.perf_counter() - t0)

    # warm-up (CUDA init, cuDNN autotune), untimed
    perceive_span(row, f_warm, f0, doc["shots"], fps, stride, detect, pose, cfg, raw)

    base_vram = VramSampler.read()
    with VramSampler() as vs:
        boxes, frames, timing = perceive_span(row, f0, f1, doc["shots"], fps, stride, detect,
                                              pose, cfg, raw)
    res.update(summarize_timing(timing))
    res["vram_peak_mib"], res["vram_idle_mib"] = vs.peak, base_vram

    # identity on those rows (no pixels)
    prof_path = cfg["paths"]["cache"] / "identity" / f"{fid}_profile.npz"
    if prof_path.exists():
        t0 = time.perf_counter()
        identify(boxes, Profile.load(prof_path), cfg, fps / stride)
        res["identity_fps"] = len(frames) / (time.perf_counter() - t0)

    # overlay renderer with skeletons, from the keypoint cache
    kp = keypoints_paths(cfg["paths"]["keypoints"], fid)
    if kp["boxes"].exists():
        fr = read_range(kp["frames"], f0, f1)
        bx = read_range(kp["boxes"], f0, f1)
        tmp = out_dir / "_bench_render.mp4"
        t0 = time.perf_counter()
        n = render_clip(row, f0 / fps, f1 / fps, fr, bx, raw, tmp, skeleton=True,
                        kpt_thr=cfg["kpt_conf_min"])
        res["render_fps"] = n / (time.perf_counter() - t0)
        tmp.unlink(missing_ok=True)

    total = fight_frames(cfg["paths"]["keypoints"], fights=fights)
    res["batch_frames"] = total
    res["batch_hours"] = total / res["pass_fps"] / 3600 if res.get("pass_fps") else None
    return res


def _f(v, fmt="{:.1f}"):
    return "-" if v is None else fmt.format(v)


def write_bench(res: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = (res.get("gpu") or "cpu").replace("NVIDIA ", "").replace("GeForce ", "")
    slug = "".join(c if c.isalnum() else "-" for c in slug).strip("-").lower()
    stem = out_dir / f"{date.today().isoformat()}_{slug}"
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    rows = [
        ("Decode only (OpenCV, CPU)", f"{_f(res['decode_fps'])} fps", ""),
        ("Person detection (YOLOX)", f"{_f(res['det_fps'])} fps", f"{_f(res['det_ms'])} ms/frame"),
        ("Pose (RTMW, batched)", f"{_f(res['pose_fps'])} fps",
         f"{_f(res['pose_ms'])} ms/frame, {_f(res['pose_ms_per_box'])} ms/box, "
         f"{_f(res['boxes_per_frame'], '{:.2f}')} boxes/frame"),
        ("Color descriptors (CPU)", f"{_f(1000 / res['desc_ms'] if res['desc_ms'] else None)}"
         " fps per thread", f"{_f(res['desc_ms'])} ms/frame CPU, {res['desc_workers']} threads "
         "overlapping the GPU"),
        ("**Full extract pass**", f"**{_f(res['pass_fps'])} fps**",
         f"{_f(res['pass_ms'])} ms/frame wall (decode + detect + track + pose + descriptors)"),
        ("Identity (from rows, CPU)", f"{_f(res.get('identity_fps'))} fps", ""),
        ("Overlay render + skeletons", f"{_f(res.get('render_fps'))} fps",
         "decode + draw + H.264 encode"),
    ]
    lines = [f"# Benchmark {date.today().isoformat()}", "",
             f"Machine: {res.get('gpu') or 'no GPU'} ({res.get('vram_gb')} GB), profile "
             f"`{res.get('profile')}`, pose mode `{res.get('pose_mode')}`, providers "
             f"{res.get('providers')}. rtmlib {res.get('rtmlib')}, onnxruntime "
             f"{res.get('onnxruntime')}.", "",
             f"Span: `{res['fight_id']}` {res['start_s']:.0f}-{res['start_s'] + res['seconds']:.0f}"
             f" s, {res['frames']} frames at {res['fps']:.2f} fps (stride {res['stride']}), after "
             "an untimed warm-up.", "",
             "| stage | speed | detail |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines += ["", f"GPU memory (whole card): {_f(res.get('vram_idle_mib'), '{:.0f}')} MiB idle "
              f"with models loaded, peak {_f(res.get('vram_peak_mib'), '{:.0f}')} MiB during the "
              "pass.", "",
              f"Batch estimate: {res['batch_frames']:,} processed frames over the cached non-test "
              f"fights -> **{_f(res.get('batch_hours'))} h** of extract at this speed "
              "(plus ~2 min of identity per fight)."]
    md = stem.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md
