"""Full-fight perception pass (1.4): data/keypoints/<fight_id>.parquet.

One in-order pass over the fight window: detect -> filter -> ByteTrack -> RTMW on the kept boxes
-> appearance descriptors from the frame already in memory. Every kept box becomes one row with
its box, track, 133 keypoints + scores and color histograms. Identity (red / blue / ref / merged /
none) runs afterwards from those rows alone, so `--from-cache` re-runs it without GPU or decode.

Resume: the window is cut into parts of whole shots (about `extract.chunk_s` long). The tracker
resets at every cut anyway, so a part started on its own gives exactly the rows an uninterrupted
run gives. Parts land in data/keypoints/<fight_id>.parts/ and are merged at the end.

Frame rate cap: with `extract.target_fps`, every `stride`-th frame is processed, on a grid counted
from the first frame of the window (resumed parts stay on the same grid). frame_idx and t are
always the original file's.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from deepfoul.perception.appearance import Appearance, describe
from deepfoul.perception.identity import BoxObs, assign_frame, build_profile, smooth_labels

PERCEPTION_VERSION = 1   # bump when the pass's output changes for the same config
IDENTITY_VERSION = 2     # 2: weak-corner guard, sliding-window smoothing
HISTS = ("shorts", "torso", "thigh", "shins")
SCALARS = ("shin_skin", "torso_skin", "clothed", "torso_len")
BOX_COLS = ["frame_idx", "t", "shot_id", "guard", "det_idx", "x1", "y1", "x2", "y2", "score",
            "track_id"]
ID_COLS = ["raw_label", "corner", "cost", "margin", "fallback", "cost_red", "cost_blue",
           "cost_ref"]
FRAME_COLS = ["frame_idx", "t", "shot_id", "guard", "n_dets", "n_kept"]


def region_cfg(icfg: dict) -> dict:
    return {**icfg["regions"], "sat_min": icfg["sat_min"], "v_min": icfg["v_min"]}


# ---------------------------------------------------------------- frame grid and parts


def stride_for(fps: float, target_fps: float | None) -> int:
    """Every stride-th frame: 60 and 50 fps -> 2 at target 30; 30, 29.97 and 23.976 -> 1."""
    if not target_fps:
        return 1
    return max(1, round(fps / target_fps))


def grid_frame(frame: int, f0: int, stride: int) -> int:
    """First frame at or after `frame` on the grid f0 + k * stride."""
    return f0 + max(0, math.ceil((frame - f0) / stride)) * stride


def grid_count(start: int, end: int, stride: int) -> int:
    """Grid frames in [start, end) when start is on the grid."""
    return max(0, math.ceil((end - start) / stride))


def plan_parts(shots: list[dict], fps: float, stride: int, chunk_s: float) -> list[dict]:
    """Whole shots grouped into parts of at least chunk_s (the last one may be shorter).

    Each part starts on the stride grid (counted from the window's first frame) at or after its
    first shot's start, and ends where the next part starts.
    """
    if not shots:
        return []
    f0 = shots[0]["start_frame"]
    starts, acc = [shots[0]["start_frame"]], 0
    for s in shots:
        if acc >= chunk_s * fps:
            starts.append(s["start_frame"])
            acc = 0
        acc += s["end_frame"] - s["start_frame"]
    starts = [grid_frame(s, f0, stride) for s in starts]
    ends = starts[1:] + [shots[-1]["end_frame"]]
    parts = [{"part": i, "start_frame": a, "end_frame": b, "frames": grid_count(a, b, stride)}
             for i, (a, b) in enumerate(zip(starts, ends))]
    return [p for p in parts if p["frames"] > 0]


# ---------------------------------------------------------------- the pass


def box_records(res, rc: dict, conf: float) -> list[dict]:
    """One dict per kept box of a track_stream FrameResult (pose must have run)."""
    keep_idx, kpts, scores = res.pose
    by_det = {r["det_idx"]: r for r in res.rows}
    out = []
    for di, k, s in zip(keep_idx, kpts, scores):
        r = by_det[int(di)]
        box = np.array([r["x1"], r["y1"], r["x2"], r["y2"]], np.float32)
        app = describe(res.image, k, s, box, rc, conf)
        rec = {c: r[c] for c in BOX_COLS}
        rec["kpts"] = np.asarray(k, np.float32).reshape(-1)
        rec["kpt_scores"] = np.asarray(s, np.float32).reshape(-1)
        for h in HISTS:
            v = getattr(app, h)
            rec[f"hist_{h}"] = None if v is None else np.asarray(v, np.float32)
        for c in SCALARS:
            v = getattr(app, c)
            rec[c] = np.nan if v is None else float(v)
        out.append(rec)
    return out


def perceive_span(row: dict, start_frame: int, end_frame: int, shots: list[dict], fps: float,
                  stride: int, detect, pose, cfg: dict, raw_dir: Path):
    """Run the pass on grid frames [start_frame, end_frame). start_frame must be on the grid.

    Returns (boxes DataFrame, frames DataFrame, timing dict).
    """
    from deepfoul.io.video import iter_fight_frames
    from deepfoul.perception.track import make_tracker, track_stream

    tcfg = cfg["tracking"]
    rc, conf = region_cfg(cfg["identity"]), cfg["kpt_conf_min"]
    tracker = make_tracker(tcfg["bytetrack"], fps / stride)
    frames = iter_fight_frames(row, raw_dir, start_frame / fps, end_frame / fps, stride)
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    recs, metas = [], []
    det_s = pose_s = desc_s = 0.0
    workers = cfg.get("extract", {}).get("describe_workers", 0)

    def described(res):
        t1 = time.perf_counter()
        return res, box_records(res, rc, conf), time.perf_counter() - t1

    def collect(item):
        nonlocal desc_s
        res, rs, dt = item
        desc_s += dt
        recs.extend(rs)
        metas.append({"frame_idx": res.idx, "t": res.t, "shot_id": res.shot_id,
                      "guard": res.guard, "n_dets": len(res.rows), "n_kept": len(rs)})

    t0 = time.perf_counter()
    stream = track_stream(frames, shots, detect, tracker, tcfg["filters"],
                          tcfg["cut_guard_frames"], pose)
    if workers:
        # descriptors (CPU) overlap the next frames' detection + pose (GPU); results are taken
        # in submission order, so rows stay in frame order
        pending: deque = deque()
        with ThreadPoolExecutor(workers) as pool:
            for res in stream:
                det_s += res.det_s
                pose_s += res.pose_s
                pending.append(pool.submit(described, res))
                while len(pending) > 2 * workers:
                    collect(pending.popleft().result())
            while pending:
                collect(pending.popleft().result())
    else:
        for res in stream:
            det_s += res.det_s
            pose_s += res.pose_s
            collect(described(res))
    timing = {"frames": len(metas), "boxes": len(recs), "det_s": det_s, "pose_s": pose_s,
              "desc_s": desc_s, "wall_s": time.perf_counter() - t0, "describe_workers": workers}
    return boxes_frame(recs), pd.DataFrame(metas, columns=FRAME_COLS), timing


def boxes_frame(recs: list[dict]) -> pd.DataFrame:
    cols = (BOX_COLS + ["kpts", "kpt_scores"] + [f"hist_{h}" for h in HISTS] + list(SCALARS))
    df = pd.DataFrame(recs, columns=cols)
    if df.empty:
        return df
    df["shot_id"] = df["shot_id"].astype("Int64")
    return df


# ---------------------------------------------------------------- identity from rows


def appearance_of(r) -> Appearance:
    def h(name):
        v = r[f"hist_{name}"]
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else np.asarray(v,
                                                                                           np.float64)

    def f(name):
        v = r[name]
        return None if v is None or pd.isna(v) else float(v)

    return Appearance(h("shorts"), h("torso"), h("thigh"), h("shins"), f("shin_skin"),
                      f("torso_skin"), f("clothed"), float(r["torso_len"]))


def profile_samples_from_boxes(boxes: pd.DataFrame, start_s: float, cfg: dict):
    """Per sampled frame in the first seed_s of the window (one frame per sample_every_s
    bucket): (box, Appearance) per kept box. Same boxes and pose as the separate 1.3 sampling."""
    pc = cfg["identity"]["profile"]
    d = boxes[(boxes["t"] >= start_s) & (boxes["t"] < start_s + pc["seed_s"])]
    if d.empty:
        return []
    bucket = np.floor((d["t"] - start_s) / pc["sample_every_s"]).astype(int)
    first = d.groupby(bucket)["frame_idx"].transform("min")
    d = d[d["frame_idx"] == first]
    return [[(np.array([r.x1, r.y1, r.x2, r.y2], np.float32), appearance_of(r._asdict()))
             for r in g.itertuples(index=False)] for _, g in d.groupby("frame_idx", sort=True)]


def make_profile(fid: str, row: dict, boxes: pd.DataFrame, start_s: float, cfg: dict):
    icfg = cfg["identity"]
    return build_profile(fid, row["red_shorts"], row["blue_shorts"],
                         profile_samples_from_boxes(boxes, start_s, cfg), icfg)


def identify(boxes: pd.DataFrame, prof, cfg: dict, fps_eff: float) -> pd.DataFrame:
    """Identity columns for every row of `boxes` (same index, same order)."""
    icfg = cfg["identity"]
    if boxes.empty:
        return pd.DataFrame(index=boxes.index, columns=["raw_label", "cost", "margin", "fallback",
                                                        "cost_red", "cost_blue", "cost_ref",
                                                        "label"])
    order = boxes.sort_values(["frame_idx", "det_idx"], kind="stable")
    recs = {}
    cols = list(order.columns)
    for _, g in order.groupby("frame_idx", sort=True):
        rows = [dict(zip(cols, v)) for v in g.itertuples(index=False, name=None)]
        obs = [BoxObs(np.array([r["x1"], r["y1"], r["x2"], r["y2"]], np.float32),
                      appearance_of(r)) for r in rows]
        for i, a in zip(g.index, assign_frame(obs, prof, icfg)):
            recs[i] = {"raw_label": a["label"], "cost": a["cost"], "margin": a["margin"],
                       "fallback": a["fallback"], "cost_red": a["cost_red"],
                       "cost_blue": a["cost_blue"], "cost_ref": a["cost_ref"]}
    ident = pd.DataFrame.from_dict(recs, orient="index").reindex(boxes.index)
    for c in ("cost", "margin", "cost_red", "cost_blue", "cost_ref"):
        ident[c] = ident[c].astype(float)
    ident["fallback"] = ident["fallback"].astype(bool)
    base = boxes[["frame_idx", "t", "shot_id", "track_id", "x1", "y1", "x2", "y2"]]
    smoothed = smooth_labels(pd.concat([base, ident], axis=1), fps_eff, icfg)
    ident["label"] = smoothed["label"].reindex(boxes.index)
    return ident


# ---------------------------------------------------------------- hashes and files


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def perception_hash(cfg: dict, pose_mode: str, shots_doc: dict, stride: int) -> str:
    icfg = cfg["identity"]
    return _hash({"v": PERCEPTION_VERSION, "det": cfg["det"], "tracking": {
        k: v for k, v in cfg["tracking"].items() if k != "checks"}, "pose": cfg["pose"],
        "regions": icfg["regions"], "sat_min": icfg["sat_min"], "v_min": icfg["v_min"],
        "kpt_conf_min": cfg["kpt_conf_min"], "pose_mode": pose_mode, "stride": stride,
        "shots": [shots_doc["window"], shots_doc["detector"], shots_doc["shots"]]})


def identity_hash(cfg: dict) -> str:
    return _hash({"v": IDENTITY_VERSION, "identity": cfg["identity"],
                  "lookahead_s": cfg["lookahead_s"]})


def keypoints_paths(kp_dir: Path, fid: str) -> dict[str, Path]:
    return {"boxes": kp_dir / f"{fid}.parquet", "frames": kp_dir / f"{fid}_frames.parquet",
            "parts": kp_dir / f"{fid}.parts"}


def write_parquet(df: pd.DataFrame, path: Path, meta: dict | None = None) -> None:
    """Atomic write (tmp + rename), zstd, optional JSON metadata under b'deepfoul'."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    if meta is not None:
        md = dict(table.schema.metadata or {})
        md[b"deepfoul"] = json.dumps(meta, default=str).encode()
        table = table.replace_schema_metadata(md)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def read_meta(path: Path) -> dict | None:
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.exists():
        return None
    md = pq.read_schema(path).metadata or {}
    raw = md.get(b"deepfoul")
    return json.loads(raw) if raw else None


def read_boxes(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns)
    for c in [c for c in df.columns if c.startswith("hist_") or c in ("kpts", "kpt_scores")]:
        df[c] = [None if v is None else np.asarray(v, np.float32) for v in df[c]]
    return df


def model_info(detector, pose) -> dict:
    """Model file names + SHA-256, and the rtmlib / onnxruntime versions."""
    from importlib.metadata import version

    out = {"rtmlib": version("rtmlib"), "onnxruntime": version("onnxruntime-gpu")
           if _has_dist("onnxruntime-gpu") else version("onnxruntime")}
    for name, m in (("detector", detector.model), ("pose", pose.model)):
        p = Path(m.onnx_model)
        out[name] = {"file": p.name, "sha256": _sha256(p)}
    return out


def _has_dist(name: str) -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        version(name)
        return True
    except PackageNotFoundError:
        return False


_SHA: dict[tuple, str] = {}


def _sha256(p: Path) -> str:
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime)
    if key not in _SHA:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


# ---------------------------------------------------------------- one fight


@dataclass
class FightJob:
    row: dict
    fid: str
    fps: float
    stride: int
    shots_doc: dict
    parts: list[dict]
    paths: dict[str, Path]
    phash: str
    ihash: str
    frames_total: int = 0
    frames_done: int = 0      # already in finished parts before this run
    status: str = ""          # todo | resume | identity | up_to_date
    reason: str = ""
    extra: dict = field(default_factory=dict)


def plan_job(row: dict, cfg: dict, shots_doc: dict, pose_mode: str, force: bool = False) -> FightJob:
    fid = row["fight_id"].strip()
    fps = float(shots_doc["fps"])
    stride = stride_for(fps, cfg["extract"]["target_fps"])
    parts = plan_parts(shots_doc["shots"], fps, stride, cfg["extract"]["chunk_s"])
    paths = keypoints_paths(cfg["paths"]["keypoints"], fid)
    job = FightJob(row, fid, fps, stride, shots_doc, parts, paths,
                   perception_hash(cfg, pose_mode, shots_doc, stride), identity_hash(cfg))
    job.frames_total = sum(p["frames"] for p in parts)
    meta = read_meta(paths["boxes"])
    if force:
        job.status, job.reason = "todo", "--force"
    elif meta and meta.get("perception_hash") == job.phash:
        if meta.get("identity_hash") == job.ihash:
            job.status = "up_to_date"
        else:
            job.status, job.reason = "identity", "identity config changed"
        job.frames_done = job.frames_total
    else:
        if meta:
            job.reason = "perception config or shots changed"
        pmeta = _parts_meta(paths["parts"])
        if pmeta and pmeta.get("perception_hash") == job.phash:
            done = done_parts(paths["parts"])
            job.frames_done = sum(p["frames"] for p in parts if p["part"] in done)
            job.status = "resume" if done else "todo"
        else:
            job.status = "todo"
    return job


def _parts_meta(parts_dir: Path) -> dict | None:
    p = parts_dir / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def part_path(parts_dir: Path, i: int, kind: str = "boxes") -> Path:
    return parts_dir / (f"part_{i:05d}.parquet" if kind == "boxes" else f"part_{i:05d}_frames.parquet")


def done_parts(parts_dir: Path) -> set[int]:
    """A part is done when its boxes file exists (written last, atomically)."""
    if not parts_dir.exists():
        return set()
    return {int(p.stem.split("_")[1]) for p in parts_dir.glob("part_?????.parquet")}


def run_perception(job: FightJob, cfg: dict, detect, pose, on_part: Callable | None = None,
                   force: bool = False) -> dict:
    """Run every part that isn't done. Returns summed timing for this run."""
    pdir = job.paths["parts"]
    pmeta = _parts_meta(pdir)
    if force or (pmeta and pmeta.get("perception_hash") != job.phash):
        shutil.rmtree(pdir, ignore_errors=True)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "meta.json").write_text(json.dumps({"perception_hash": job.phash,
                                                "stride": job.stride}))
    done = done_parts(pdir)
    total = {"frames": 0, "boxes": 0, "det_s": 0.0, "pose_s": 0.0, "desc_s": 0.0, "wall_s": 0.0}
    for p in job.parts:
        if p["part"] in done:
            continue
        boxes, frames, timing = perceive_span(job.row, p["start_frame"], p["end_frame"],
                                              job.shots_doc["shots"], job.fps, job.stride,
                                              detect, pose, cfg, cfg["paths"]["raw"])
        if len(frames) and frames["frame_idx"].iloc[0] != p["start_frame"]:
            job.extra.setdefault("grid_warnings", []).append(
                f"part {p['part']}: first frame {frames['frame_idx'].iloc[0]} != "
                f"planned {p['start_frame']}")
        write_parquet(frames, part_path(pdir, p["part"], "frames"))
        write_parquet(boxes, part_path(pdir, p["part"]))
        for k in total:
            total[k] += timing[k]
        job.frames_done += p["frames"]
        if on_part:
            on_part(job, p, timing)
    return total


def assemble(job: FightJob) -> tuple[pd.DataFrame, pd.DataFrame]:
    pdir = job.paths["parts"]
    boxes = [read_boxes(part_path(pdir, p["part"])) for p in job.parts]
    frames = [pd.read_parquet(part_path(pdir, p["part"], "frames")) for p in job.parts]
    b = pd.concat([x for x in boxes if not x.empty], ignore_index=True) if any(
        not x.empty for x in boxes) else boxes_frame([])
    f = pd.concat(frames, ignore_index=True)
    return b, f


def finalize(job: FightJob, cfg: dict, boxes: pd.DataFrame, frames: pd.DataFrame,
             base_meta: dict, rebuild_profile: bool = True) -> dict:
    """Profile + identity on the assembled rows; write the final parquet files."""
    from deepfoul.io.fights import window
    from deepfoul.perception.identity import Profile

    start = window(job.row)[0] or 0.0
    prof_path = cfg["paths"]["cache"] / "identity" / f"{job.fid}_profile.npz"
    if prof_path.exists() and not rebuild_profile:
        prof = Profile.load(prof_path)
    else:
        prof = make_profile(job.fid, job.row, boxes, start, cfg)
        prof_path.parent.mkdir(parents=True, exist_ok=True)
        prof.save(prof_path)
    t0 = time.perf_counter()
    ident = identify(boxes, prof, cfg, job.fps / job.stride)
    id_s = time.perf_counter() - t0
    out = boxes.drop(columns=[c for c in ID_COLS if c in boxes.columns])
    for c in ID_COLS:
        out[c] = ident["label" if c == "corner" else c].values
    meta = {**base_meta, "fight_id": job.fid, "fps": job.fps, "stride": job.stride,
            "fps_eff": job.fps / job.stride, "window": job.shots_doc["window"],
            "perception_hash": job.phash, "identity_hash": job.ihash,
            "perception_version": PERCEPTION_VERSION, "identity_version": IDENTITY_VERSION,
            "profile": {"n": prof.n, "warnings": prof.warnings}, "identity_s": id_s,
            "columns": {"kpts": "133 x (x, y) frame pixels, flattened",
                        "kpt_scores": "133", "hist_*": "80-bin HSV histograms, null = unknown",
                        "corner": "red | blue | ref | merged | none"}}
    write_parquet(frames, job.paths["frames"], {"fight_id": job.fid})
    write_parquet(out, job.paths["boxes"], meta)
    return {"meta": meta, "profile": prof, "boxes": out, "frames": frames}


def summarize(boxes: pd.DataFrame, frames: pd.DataFrame) -> dict:
    """Frame-level corner coverage for the report."""
    n = len(frames)
    if n == 0:
        return {"frames": 0}
    has = {lab: boxes.loc[boxes["corner"] == lab, "frame_idx"].unique()
           for lab in ("red", "blue", "ref", "merged")}
    fi = frames["frame_idx"]
    red, blue, ref, mer = (fi.isin(has[k]) for k in ("red", "blue", "ref", "merged"))
    both = red & blue
    bom = both | mer
    anyf = red | blue | mer
    return {"frames": n, "both_pct": both.mean() * 100, "both_or_merged_pct": bom.mean() * 100,
            "any_fighter_pct": anyf.mean() * 100,
            "bom_of_visible_pct": (bom[anyf].mean() * 100) if anyf.any() else 0.0,
            "ref_pct": ref.mean() * 100, "merged_pct": mer.mean() * 100,
            "boxes_per_frame": len(boxes) / n}
