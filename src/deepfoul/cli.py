"""`deepfoul <command>` entry point."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import typer

from deepfoul.config import load_config, pick_profile

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _gpu_info() -> tuple[str | None, float | None]:
    """(GPU name, VRAM in GB) from nvidia-smi, or (None, None)."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
        name, mib = [s.strip() for s in out.split(",")]
        return name, round(int(mib) / 1024, 1)
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None, None


def _tool_version(cmd: list[str]) -> str:
    if not shutil.which(cmd[0]):
        return "MISSING"
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (subprocess.SubprocessError, IndexError):
        return "error"


def _pb(field: int, value: bytes | int) -> bytes:
    """Encode one protobuf field (varint or length-delimited)."""
    def varint(n: int) -> bytes:
        out = bytearray()
        while True:
            b, n = n & 0x7F, n >> 7
            out.append(b | (0x80 if n else 0))
            if not n:
                return bytes(out)
    if isinstance(value, int):
        return varint(field << 3) + varint(value)
    return varint(field << 3 | 2) + varint(len(value)) + value


def _tiny_add_model() -> bytes:
    """ONNX model y = x + x on a float[4], built by hand (no `onnx` package needed)."""
    def value_info(name: str) -> bytes:
        dim = _pb(1, _pb(1, 4))                              # TensorShapeProto.dim{dim_value: 4}
        tensor = _pb(1, 1) + _pb(2, dim)                     # elem_type FLOAT, shape
        return _pb(1, name.encode()) + _pb(2, _pb(1, tensor))
    node = _pb(1, b"x") + _pb(1, b"x") + _pb(2, b"y") + _pb(4, b"Add")
    graph = _pb(1, node) + _pb(2, b"doctor") + _pb(11, value_info("x")) + _pb(12, value_info("y"))
    return _pb(1, 9) + _pb(7, graph) + _pb(8, _pb(2, 17))    # ir_version 9, opset 17


def _cuda_session_check() -> tuple[bool, str]:
    """Run a tiny ONNX graph on CUDA and report which provider actually executed it."""
    import numpy as np
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(_tiny_add_model(),
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        sess.run(None, {"x": np.ones(4, dtype=np.float32)})
    except Exception as e:  # noqa: BLE001 - report any ORT failure
        return False, f"error: {e}"
    used = sess.get_providers()
    return used[0] == "CUDAExecutionProvider", ", ".join(used)


def _init_ort():
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls") and sys.platform != "darwin":
        ort.preload_dlls()  # CUDA/cuDNN DLLs from the nvidia-* pip wheels
    return ort


@app.command()
def doctor() -> None:
    """Report machine profile, GPU, onnxruntime providers and external tools."""
    cfg = load_config()
    ort = _init_ort()
    gpu, vram = _gpu_info()
    profile = pick_profile(cfg, vram)

    rows = [
        ("OS", f"{platform.system()} {platform.version()}"),  # release() says "10" on Windows 11
        ("Python", platform.python_version()),
        ("GPU", f"{gpu} ({vram} GB)" if gpu else "none"),
        ("onnxruntime", ort.__version__),
        ("available providers", ", ".join(ort.get_available_providers())),
    ]
    cuda_ok, used = _cuda_session_check() if gpu else (False, "skipped (no NVIDIA GPU)")
    rows += [
        ("CUDA session test", f"{'OK' if cuda_ok else 'FAIL'} -> {used}"),
        ("profile", f"{profile} (pose_mode={cfg['profiles'][profile]['pose_mode']}, "
                    f"batch_size={cfg['profiles'][profile]['batch_size']})"),
        ("ffmpeg", _tool_version(["ffmpeg", "-version"])),
        ("yt-dlp", _tool_version(["yt-dlp", "--version"])),
        ("repo root", str(cfg["root"])),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        typer.echo(f"{k:<{width}}  {v}")

    if gpu and not cuda_ok:
        typer.secho("\nCUDAExecutionProvider is NOT active. Check onnxruntime-gpu / driver versions.",
                    fg="red", err=True)
        raise typer.Exit(1)
    if "MISSING" in (rows[-3][1], rows[-2][1]):
        raise typer.Exit(1)


@app.command()
def fetch(
    csv_path: Path = typer.Option(None, "--csv", help="Fights CSV (default: labels/fights.csv)."),
    only: str = typer.Option(None, "--only", help="Download a single fight_id."),
) -> None:
    """Download every video in fights.csv into data/raw/<fight_id>.mp4 (skips existing)."""
    from deepfoul.io.fetch import fetch_all, read_fights

    cfg = load_config()
    rows = read_fights(csv_path or cfg["paths"]["labels"] / "fights.csv")
    res = fetch_all(rows, cfg["paths"]["raw"], cfg["fetch"]["max_height"], only=only)
    for k, ids in res.items():
        typer.echo(f"{k}: {len(ids)} {ids if ids else ''}")
    if res["failed"]:
        raise typer.Exit(1)


@app.command("fetch-check")
def fetch_check(
    csv_path: Path = typer.Option(None, "--csv", help="Fights CSV (default: labels/fights.csv)."),
) -> None:
    """ffprobe every downloaded video, flag problems, write contact sheets + outputs/fetch_check/report.md."""
    from deepfoul.io.fetch import read_fights
    from deepfoul.io.probe import check_all

    cfg = load_config()
    rows = read_fights(csv_path or cfg["paths"]["labels"] / "fights.csv")
    out_dir = cfg["paths"]["outputs"] / "fetch_check"
    for r in check_all(rows, cfg["paths"]["raw"], out_dir):
        typer.echo(f"{r['fight_id']:<36} {r['status']:<10} {'; '.join(r['flags']) or 'ok'}")
    typer.echo(f"report: {out_dir / 'report.md'}")


def _fight_rows(fight_ids: list[str], split: str | None = None) -> list[dict[str, str]]:
    """fights.csv rows for the given ids (all rows of `split` if no ids given)."""
    from deepfoul.io.fetch import read_fights

    cfg = load_config()
    rows = {r["fight_id"].strip(): r for r in read_fights(cfg["paths"]["labels"] / "fights.csv")}
    if not fight_ids:
        return [r for r in rows.values() if split is None or r["split"].strip() == split]
    missing = [f for f in fight_ids if f not in rows]
    if missing:
        raise typer.BadParameter(f"not in fights.csv: {', '.join(missing)}")
    return [rows[f] for f in fight_ids]


def _ensure_shots(row: dict, cfg: dict, force: bool = False) -> dict:
    """The fight's shot file, (re)built if missing, stale or forced."""
    import time

    from deepfoul.io.fights import window
    from deepfoul.io.shots import (detect_shots, load_shots, save_shots, shots_cfg_for,
                                   shots_path, stale_reason)

    fid = row["fight_id"].strip()
    video = cfg["paths"]["raw"] / f"{fid}.mp4"
    start, end = window(row)
    win = (start or 0.0, end)
    out = shots_path(cfg["paths"]["cache"], fid)
    shots_cfg = shots_cfg_for(cfg["shots"], fid)
    if out.exists() and not force:
        doc = load_shots(out)
        reason = stale_reason(doc, win, shots_cfg)
        if reason is None:
            return doc
        typer.echo(f"{fid}: shots re-running, {reason}")
    t0 = time.perf_counter()
    doc = {"fight_id": fid, **detect_shots(video, win[0], win[1], shots_cfg)}
    save_shots(doc, out)
    lens = [s["end_s"] - s["start_s"] for s in doc["shots"]]
    typer.echo(f"{fid}: {len(lens)} shots, median {sorted(lens)[len(lens) // 2]:.1f}s, "
               f"{time.perf_counter() - t0:.0f}s -> {out}")
    return doc


def _extract_rows(fight_ids: list[str], split: str | None) -> list[dict]:
    """Explicit ids, a split, or (neither) --all: every non-test fight, train first, then val."""
    if fight_ids or split:
        return _fight_rows(fight_ids or [], split)
    order = {"train": 0, "val": 1}
    picked = [r for r in _fight_rows([], None) if r["split"].strip() in order]
    return sorted(picked, key=lambda r: order[r["split"].strip()])  # stable: CSV order kept


@app.command()
def extract(
    fight_ids: list[str] = typer.Argument(None, help="fight_ids from fights.csv."),
    all_: bool = typer.Option(False, "--all", help="Every downloaded non-test fight (train, then "
                              "val) that isn't extracted yet."),
    split: str = typer.Option(None, "--split", help="Every fight in this split."),
    shots: bool = typer.Option(False, "--shots", help="Only detect camera cuts."),
    from_cache: bool = typer.Option(False, "--from-cache", help="Re-run identity from the cached "
                                    "keypoints (no GPU, no decode)."),
    force: bool = typer.Option(False, "--force", help="Re-run even if up to date."),
    rebuild_profile: bool = typer.Option(False, "--rebuild-profile",
                                         help="With --from-cache: re-learn the fight profile."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List what would run, then stop."),
) -> None:
    """Shots + detect/track/pose + identity in one in-order pass per fight.

    Writes data/keypoints/<fight_id>.parquet (one row per kept box: corner, box, 133 keypoints,
    scores, color histograms) and <fight_id>_frames.parquet, plus outputs/extract_report.md.
    An interrupted run resumes from the last finished part.
    """
    import shutil
    import time
    import traceback

    import pandas as pd

    from deepfoul.io.fights import window
    from deepfoul.io.power import keep_awake
    from deepfoul.io.shots import load_shots, shots_path
    from deepfoul.io.video import video_info
    from deepfoul.perception.extract import (assemble, finalize, plan_job, read_boxes,
                                             read_meta, run_perception, stride_for, summarize)
    from deepfoul.perception.identity import check_lookahead
    from deepfoul.viz.extract_report import fmt_dur, write_extract_report

    if not (fight_ids or split or all_):
        raise typer.BadParameter("give fight_ids, --split or --all")
    cfg = load_config()
    rows = _extract_rows(fight_ids or [], split)
    if shots:
        for row in rows:
            fid = row["fight_id"].strip()
            if not (cfg["paths"]["raw"] / f"{fid}.mp4").exists():
                typer.secho(f"{fid}: no video, run `deepfoul fetch`", fg="red", err=True)
                continue
            _ensure_shots(row, cfg, force)
        return

    gpu, vram = _gpu_info()
    mode = cfg["profiles"][pick_profile(cfg, vram)]["pose_mode"]
    gpu_s = f"{gpu} ({vram} GB)" if gpu else "none"
    report = cfg["paths"]["outputs"] / "extract_report.md"
    log_dir = cfg["paths"]["outputs"] / "extract_logs"

    # ---- plan: status and frame estimate per fight
    results = []
    for row in rows:
        fid = row["fight_id"].strip()
        r = {"fight_id": fid, "split": row["split"].strip(), "row": row, "frames_est": 0}
        video = cfg["paths"]["raw"] / f"{fid}.mp4"
        if not video.exists():
            r["status"] = "no video"
            results.append(r)
            continue
        r["status"] = "pending"
        sp = shots_path(cfg["paths"]["cache"], fid)
        if sp.exists():
            job = plan_job(row, cfg, load_shots(sp), mode, force)
            r.update(job=job, stride=job.stride, frames_est=job.frames_total - job.frames_done)
            if job.status == "up_to_date" and not from_cache:
                r.update(status="up to date", frames_est=0)
        else:   # no shot file yet: estimate from the window and the frame rate
            info = video_info(video)
            start, end = window(row)
            end = end if end is not None else info["duration_s"]
            st = stride_for(info["fps"], cfg["extract"]["target_fps"])
            r.update(stride=st, frames_est=int((end - (start or 0.0)) * info["fps"] / st))
        results.append(r)

    todo = [r for r in results if r["status"] == "pending"]
    typer.echo(f"{len(results)} fights, {len(todo)} to run:")
    for r in results:
        job = r.get("job")
        extra = (f"{job.status} {job.reason}".strip() if job and r["status"] == "pending" else
                 "no shots yet" if r["status"] == "pending" else "")
        typer.echo(f"  {r['fight_id']:<36} {r['split']:<5} {r['status']:<10} "
                   + (f"~{r['frames_est']} frames (stride {r['stride']}) {extra}"
                      if r["status"] == "pending" else ""))
    if dry_run:
        return

    det = pose = None
    models: dict = {}

    def get_models():
        nonlocal det, pose
        if det is None:
            from deepfoul.perception.detect import PersonDetector
            from deepfoul.perception.extract import model_info
            from deepfoul.perception.pose import PoseEstimator

            _init_ort()
            device = cfg["device"] if cfg["device"] != "auto" else ("cuda" if gpu else "cpu")
            det = PersonDetector(mode, device, cfg["det"]["score_min"])
            pose = PoseEstimator(mode, device, cfg["pose"]["batch"])
            typer.echo(f"detector + pose ({mode}) on {det.providers[0]} / {pose.providers[0]}, "
                       f"pose batched: {pose.batched}")
            models.update(model_info(det, pose))
        return det, pose

    run_t0 = time.perf_counter()
    run_frames = 0

    def write_report():
        write_extract_report([{k: v for k, v in r.items() if k not in ("row", "job")}
                              for r in results], report, gpu_s,
                             time.perf_counter() - run_t0, cfg["extract"]["check"])

    def fill_from_file(r, job):
        meta = read_meta(job.paths["boxes"])
        b = read_boxes(job.paths["boxes"], ["frame_idx", "corner"])
        f = pd.read_parquet(job.paths["frames"])
        r.update(summary=summarize(b, f), profile_n=meta["profile"]["n"],
                 warnings=meta["profile"]["warnings"] + meta.get("grid_warnings", []),
                 size_mb=job.paths["boxes"].stat().st_size / 1e6, stride=meta["stride"])

    def on_part(job, p, timing):
        nonlocal run_frames
        run_frames += timing["frames"]
        el = time.perf_counter() - run_t0
        fps = run_frames / el if el else 0.0
        left_fight = job.frames_total - job.frames_done
        left_run = left_fight + sum(x["frames_est"] for x in results
                                    if x["status"] == "pending")
        eta = (lambda n: fmt_dur(n / fps)) if fps else (lambda n: "-")
        typer.echo(f"{job.fid} part {p['part'] + 1}/{len(job.parts)} "
                   f"{100 * job.frames_done / max(job.frames_total, 1):.0f}% | "
                   f"{timing['frames'] / max(timing['wall_s'], 1e-9):.1f} fps (run {fps:.1f}) | "
                   f"fight ETA {eta(left_fight)} | run ETA {eta(left_run)}")

    for r in results:
        if r["status"] == "up to date":
            try:
                fill_from_file(r, r["job"])
            except Exception:  # noqa: BLE001 - a broken old file shouldn't stop the run
                pass
    write_report()
    with keep_awake() as awake:
        if todo and not from_cache:
            typer.echo("keeping the PC awake while this runs" if awake else
                       "could not keep the PC awake (not Windows?)")
        for r in todo:
            fid, row = r["fight_id"], r["row"]
            t0 = time.perf_counter()
            r["status"] = "running"
            try:
                doc = _ensure_shots(row, cfg)
                check_lookahead(cfg["identity"], doc["fps"], cfg["lookahead_s"])
                job = plan_job(row, cfg, doc, mode, force and not from_cache)
                r.update(job=job, stride=job.stride)
                if from_cache:
                    if job.status not in ("up_to_date", "identity"):
                        raise RuntimeError("no up-to-date keypoints to re-run identity on "
                                           f"({job.reason or 'not extracted'}); run without "
                                           "--from-cache")
                    meta = read_meta(job.paths["boxes"])
                    keep = {k: meta[k] for k in ("models", "perception_timing", "grid_warnings")
                            if k in meta}
                    finalize(job, cfg, read_boxes(job.paths["boxes"]),
                             pd.read_parquet(job.paths["frames"]), keep, rebuild_profile)
                    r["status"] = "identity re-run"
                else:
                    resumed = job.status == "resume"
                    timing = {}
                    if job.status != "identity":
                        typer.echo(f"{fid}: {job.status} {job.reason}, {len(job.parts)} parts, "
                                   f"{job.frames_total} frames (stride {job.stride})")
                        d, p = get_models()
                        timing = run_perception(job, cfg, d, p, on_part, force)
                        r["run_frames"], r["run_s"] = timing["frames"], timing["wall_s"]
                        boxes, frames = assemble(job)
                        base = {"models": dict(models), "perception_timing": timing,
                                "grid_warnings": job.extra.get("grid_warnings", [])}
                    else:
                        meta = read_meta(job.paths["boxes"])
                        boxes = read_boxes(job.paths["boxes"])
                        frames = pd.read_parquet(job.paths["frames"])
                        base = {k: meta[k] for k in ("models", "perception_timing",
                                                     "grid_warnings") if k in meta}
                    finalize(job, cfg, boxes, frames, base, rebuild_profile=True)
                    shutil.rmtree(job.paths["parts"], ignore_errors=True)
                    r["status"] = ("identity re-run" if job.status == "identity" else
                                   "resumed" if resumed else "done")
                r["time_s"] = time.perf_counter() - t0
                fill_from_file(r, job)
                s = r["summary"]
                typer.echo(f"{fid}: {r['status']} in {fmt_dur(r['time_s'])}, both "
                           f"{s['both_pct']:.0f}%, both or merged {s['both_or_merged_pct']:.0f}% "
                           f"({s['bom_of_visible_pct']:.0f}% of frames with a fighter)")
            except KeyboardInterrupt:
                r["status"] = "interrupted"
                write_report()
                typer.secho(f"{fid}: interrupted, run the same command again to resume",
                            fg="yellow")
                raise typer.Exit(130)
            except Exception as e:  # noqa: BLE001 - one bad fight must not stop the batch
                log_dir.mkdir(parents=True, exist_ok=True)
                log = log_dir / f"{fid}.log"
                log.write_text(traceback.format_exc(), encoding="utf-8")
                r.update(status="failed", error=f"{type(e).__name__}: {e}",
                         log=log.relative_to(cfg["paths"]["outputs"]).as_posix(),
                         time_s=time.perf_counter() - t0)
                typer.secho(f"{fid}: FAILED {r['error']} (log {log})", fg="red", err=True)
            r["frames_est"] = 0
            write_report()
    typer.echo(f"report -> {report}")


@app.command()
def render(
    fight_id: str = typer.Argument(None, help="fight_id from fights.csv (not needed with --golden)."),
    start: str = typer.Option(None, "--start", help="mm:ss(.cc) or seconds; default window start."),
    end: str = typer.Option(None, "--end", help="mm:ss(.cc) or seconds; default window end."),
    skeleton: bool = typer.Option(False, "--skeleton", help="Draw the 133-point skeletons."),
    highlight: str = typer.Option("both", "--highlight",
                                  help="Big dots with --skeleton: poke (fingertips + eyes), "
                                  "low_blow (hips, knees, ankles, feet), both, none."),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed, e.g. 0.25 (no audio)."),
    show_ref: bool = typer.Option(False, "--show-ref", help="Thin yellow box on the ref."),
    show_none: bool = typer.Option(True, "--show-none/--hide-none",
                                   help="Thin grey boxes on unlabelled people."),
    out: Path = typer.Option(None, "--out", help="Output .mp4 (default outputs/renders/)."),
    sheet: str = typer.Option(None, "--sheet", help="Contact sheet instead of a video: merged | "
                              "single (exactly one fighter), to outputs/extract_check/."),
    n: int = typer.Option(24, "--n", help="Frames on a --sheet."),
    seed: int = typer.Option(0, "--seed", help="Random seed for --sheet."),
    golden: bool = typer.Option(False, "--golden", help="Render every golden clip with "
                                "skeletons (plus 0.25x for the foul clips)."),
    only: list[str] = typer.Option(None, "--only", help="With --golden: only these event_ids."),
    tag: str = typer.Option("1.5", "--tag", help="With --golden: outputs/renders/<tag>/."),
) -> None:
    """Draw red / blue / merged boxes (and skeletons) from the keypoint cache. No GPU."""
    from deepfoul.io.fetch import read_fights
    from deepfoul.io.fights import parse_time, window
    from deepfoul.io.previews import fmt_ts
    from deepfoul.io.shots import first_frame_at
    from deepfoul.perception.extract import keypoints_paths, read_meta
    from deepfoul.viz.overlay import HIGHLIGHTS, read_range, render_clip

    cfg = load_config()
    if highlight not in HIGHLIGHTS:
        raise typer.BadParameter(f"--highlight must be one of {', '.join(HIGHLIGHTS)}")
    if speed <= 0:
        raise typer.BadParameter("--speed must be > 0")
    fights = {r["fight_id"].strip(): r for r in read_fights(cfg["paths"]["labels"] / "fights.csv")}
    if golden:
        _render_golden(cfg, fights, only, tag)
        return
    if not fight_id or fight_id not in fights:
        raise typer.BadParameter(f"unknown fight_id {fight_id!r}")
    row = fights[fight_id]
    paths = keypoints_paths(cfg["paths"]["keypoints"], fight_id)
    meta = read_meta(paths["boxes"])
    if meta is None:
        raise typer.BadParameter(f"{fight_id}: no keypoint cache, run `deepfoul extract` first")
    fps = meta["fps"]
    w0, w1 = window(row)
    s0 = parse_time(start) if start else (w0 or 0.0)
    s1 = parse_time(end) if end else (w1 if w1 is not None else meta["window"][1])
    f0, f1 = first_frame_at(s0, fps), first_frame_at(s1, fps)
    frames = read_range(paths["frames"], f0, f1)
    if sheet:
        _render_sheet(cfg, row, paths, frames, f0, f1, fps, sheet, n, seed)
        return
    boxes = read_range(paths["boxes"], f0, f1, None if skeleton else
                       ["frame_idx", "x1", "y1", "x2", "y2", "corner", "fallback"])
    name = (f"{fight_id}_{fmt_ts(s0).replace(':', '')}-{fmt_ts(s1).replace(':', '')}"
            + ("_skel" if skeleton else "") + (f"_x{speed:g}" if speed != 1 else "") + ".mp4")
    out = out or (cfg["paths"]["outputs"] / "renders" / name)
    wrote = render_clip(row, s0, s1, frames, boxes, cfg["paths"]["raw"], out, skeleton=skeleton,
                        show_ref=show_ref, show_none=show_none, kpt_thr=cfg["kpt_conf_min"],
                        highlight=HIGHLIGHTS[highlight], speed=speed)
    typer.echo(f"{wrote} frames -> {out}")


def _render_sheet(cfg, row, paths, frames, f0, f1, fps, kind, n, seed):
    import numpy as np

    from deepfoul.viz.overlay import read_range, sheet_frames, write_sheet

    fid = row["fight_id"].strip()
    boxes = read_range(paths["boxes"], f0, f1, ["frame_idx", "x1", "y1", "x2", "y2", "corner",
                                                "fallback"])
    pool = sheet_frames(boxes, frames, kind)
    if not len(pool):
        typer.echo(f"{fid}: no {kind} frames in range")
        return
    picks = sorted(np.random.default_rng(seed).choice(pool, min(n, len(pool)), replace=False))
    out = cfg["paths"]["outputs"] / "extract_check" / f"{fid}_{kind}.jpg"
    write_sheet(row, [int(p) for p in picks], boxes, frames, cfg["paths"]["raw"], out, fps,
                cfg["kpt_conf_min"])
    typer.echo(f"{len(picks)} of {len(pool)} {kind} frames -> {out}")


def _render_golden(cfg, fights, only, tag):
    import csv

    from deepfoul.io.previews import fmt_ts
    from deepfoul.io.shots import first_frame_at
    from deepfoul.perception.extract import keypoints_paths, read_meta, summarize
    from deepfoul.viz.id_report import read_baseline
    from deepfoul.viz.overlay import HIGHLIGHTS, read_range, render_clip

    out_dir = cfg["paths"]["outputs"] / "renders" / tag
    with (cfg["paths"]["labels"] / "golden.csv").open(encoding="utf-8-sig", newline="") as f:
        golden = [g for g in csv.DictReader(f) if not only or g["event_id"] in only]
    base = read_baseline(cfg["paths"]["labels"] / "golden_baseline.csv")
    hl = {"eye_poke": "poke", "low_blow": "low_blow"}
    lines = [f"# Golden renders ({tag})", "",
             "Drawn from the full-fight keypoint caches (`data/keypoints/`), skeletons on. Points "
             f"under {cfg['kpt_conf_min']} confidence are not drawn. Big dots: fingertips + eyes "
             "on the poke clip, hips / knees / ankles / feet on the low blow clip, all of them "
             "elsewhere. Foul clips also get a 0.25x version (no audio). Numbers are from the "
             "full-fight cache, so tracklets that start before the clip can differ slightly from "
             "the clip caches the baseline was measured on.", "",
             "| clip | fight | span | type | both | both or merged | baseline b-or-m | renders |",
             "|---|---|---|---|---|---|---|---|"]
    for g in golden:
        fid, cid = g["fight_id"], g["event_id"]
        row = fights[fid]
        paths = keypoints_paths(cfg["paths"]["keypoints"], fid)
        fps = read_meta(paths["boxes"])["fps"]
        s0, s1 = float(g["start_s"]), float(g["end_s"])
        f0, f1 = first_frame_at(s0, fps), first_frame_at(s1, fps)
        frames = read_range(paths["frames"], f0, f1)
        boxes = read_range(paths["boxes"], f0, f1)
        kind = g["foul_type"]
        links = []
        for sp in ([1.0, 0.25] if kind in hl else [1.0]):
            sfx = "" if sp == 1 else f"_x{sp:g}"
            out = out_dir / f"{cid}_{fid}_{kind}{sfx}.mp4"
            render_clip(row, s0, s1, frames, boxes, cfg["paths"]["raw"], out, skeleton=True,
                        kpt_thr=cfg["kpt_conf_min"], highlight=HIGHLIGHTS[hl.get(kind, "both")],
                        speed=sp)
            links.append(f"[{sp:g}x]({out.name})")
            typer.echo(f"{cid} {sp:g}x -> {out}")
        s = summarize(boxes, frames)
        b = base.get(cid, {})
        lines.append(f"| {cid} | {fid} | {fmt_ts(s0)}-{fmt_ts(s1)} | {kind} | "
                     f"{s['both_pct']:.0f}% | {s['both_or_merged_pct']:.0f}% | "
                     f"{b.get('both_or_merged_pct', '-')}% | {' '.join(links)} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"index -> {out_dir / 'renders.md'}")


@app.command()
def bench(
    fight: str = typer.Option(None, "--fight", help="fight_id (default: configs bench.fight_id)."),
    start: str = typer.Option(None, "--start", help="mm:ss(.cc) or seconds (default from config)."),
    seconds: float = typer.Option(None, "--seconds", help="Timed span length (default 60)."),
) -> None:
    """Speed of every stage on this machine: decode, detection, pose, descriptors, the full
    extract pass, identity and the renderer. Writes outputs/bench/<date>_<gpu>.md + .json."""
    from importlib.metadata import version

    from deepfoul.bench import run_bench, write_bench
    from deepfoul.io.fetch import read_fights
    from deepfoul.io.fights import parse_time
    from deepfoul.perception.detect import PersonDetector
    from deepfoul.perception.pose import PoseEstimator

    cfg = load_config()
    bc = cfg["bench"]
    fid = fight or bc["fight_id"]
    fights = {r["fight_id"].strip(): r for r in read_fights(cfg["paths"]["labels"] / "fights.csv")}
    if fid not in fights:
        raise typer.BadParameter(f"unknown fight_id {fid!r}")
    s0 = parse_time(start) if start else float(bc["start_s"])
    secs = seconds or float(bc["seconds"])
    gpu, vram = _gpu_info()
    prof = pick_profile(cfg, vram)
    mode = cfg["profiles"][prof]["pose_mode"]
    _init_ort()
    device = cfg["device"] if cfg["device"] != "auto" else ("cuda" if gpu else "cpu")
    det = PersonDetector(mode, device, cfg["det"]["score_min"])
    pose = PoseEstimator(mode, device, cfg["pose"]["batch"])
    typer.echo(f"detector + pose ({mode}) on {det.providers[0]} / {pose.providers[0]}, "
               f"pose batched: {pose.batched}; timing {fid} {s0:.0f}-{s0 + secs:.0f} s ...")
    ort_name = "onnxruntime-gpu" if gpu else "onnxruntime"
    machine = {"gpu": gpu, "vram_gb": vram, "profile": prof, "pose_mode": mode,
               "providers": f"{det.providers[0]} / {pose.providers[0]}",
               "rtmlib": version("rtmlib"), "onnxruntime": version(ort_name)}
    out_dir = cfg["paths"]["outputs"] / "bench"
    res = run_bench(cfg, fights[fid], s0, secs, det, pose, out_dir, machine, fights)
    md = write_bench(res, out_dir)
    typer.echo(md.read_text(encoding="utf-8"))
    typer.echo(f"report -> {md}")


@app.command("shots-check")
def shots_check(
    fight_ids: list[str] = typer.Argument(None, help="Fights to draw cut sheets for."),
    n: int = typer.Option(20, "--n", help="Total cuts on the sheets, split evenly over the fights."),
) -> None:
    """Cut contact sheets + outputs/shots_check/report.md (all cached fights in the report)."""
    from deepfoul.io.previews import cut_contact_sheet, shot_summary, write_shots_report
    from deepfoul.io.shots import load_shots

    cfg = load_config()
    out_dir = cfg["paths"]["outputs"] / "shots_check"
    sheet_ids = fight_ids or []
    for fid in sheet_ids:
        if not (cfg["paths"]["cache"] / f"{fid}_shots.json").exists():
            raise typer.BadParameter(f"{fid}: no shot file, run `deepfoul extract {fid} --shots`")
    per = -(-n // len(sheet_ids)) if sheet_ids else 0
    entries = []
    for path in sorted(cfg["paths"]["cache"].glob("*_shots.json")):
        doc = load_shots(path)
        fid = doc["fight_id"]
        e = {"fight_id": fid, "summary": shot_summary(doc)}
        if fid in sheet_ids:
            sheet = out_dir / f"{fid}_cuts.jpg"
            e["cuts"] = cut_contact_sheet(cfg["paths"]["raw"] / f"{fid}.mp4", doc, per, sheet)
            e["sheet"] = sheet.name
        entries.append(e)
        s = e["summary"]
        typer.echo(f"{fid:<36} {s['shots']:>4} shots  >60s: {len(s['long'])}  <1s: {len(s['short'])}")
    write_shots_report(entries, out_dir / "report.md")
    typer.echo(f"report: {out_dir / 'report.md'}")


GOLDEN_EXCLUDE = {"ufc326_garbrandt_long", "ufcfn20210313_edwards_muhammad"}  # video quality


@app.command("golden-candidates")
def golden_candidates() -> None:
    """Preview clips of every clear_live known foul in train fights -> outputs/golden_candidates/."""
    from deepfoul.io.fetch import read_fights
    from deepfoul.io.previews import (clip_name, fmt_ts, golden_selection, read_known_fouls,
                                      write_preview_clip)

    cfg = load_config()
    labels = cfg["paths"]["labels"]
    fouls = golden_selection(read_fights(labels / "fights.csv"),
                             read_known_fouls(labels / "notes" / "known_fouls.csv"), GOLDEN_EXCLUDE)
    pad = float(cfg["previews"]["golden_pad_s"])
    out_dir = cfg["paths"]["outputs"] / "golden_candidates"
    lines = ["# Golden candidates", "",
             f"Clear-live known fouls in train fights, live time +- {pad:.0f} s. "
             "Burned-in text is the original video time (mm:ss.cc) and frame index.", "",
             "| clip | fight_id | live | type | offender | notes |", "|---|---|---|---|---|---|"]
    for f in fouls:
        live = float(f["live_s"])
        name = clip_name(f["fight_id"], live, f["foul_type"])
        n = write_preview_clip(cfg["paths"]["raw"] / f"{f['fight_id']}.mp4",
                               max(live - pad, 0.0), live + pad, out_dir / name)
        typer.echo(f"{name}: {n} frames")
        lines.append(f"| [{name}]({name}) | {f['fight_id']} | {fmt_ts(live)} | {f['foul_type']} | "
                     f"{f['offender']} | {f['notes']} |")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"index: {out_dir / 'index.md'}")


@app.command("frame-check")
def frame_check_cmd(
    fight_id: str = typer.Argument(...),
    times: list[float] = typer.Option(None, "--t", help="Times to check (default: 5 in the window)."),
) -> None:
    """Compare the frame reader to ffmpeg (pixel diff) and to decoding from 10 s earlier."""
    from deepfoul.io.fights import window
    from deepfoul.io.video import frame_check, video_info

    cfg = load_config()
    row = _fight_rows([fight_id])[0]
    video = cfg["paths"]["raw"] / f"{fight_id}.mp4"
    info = video_info(video)
    typer.echo(f"{fight_id}: {info['width']}x{info['height']} r={info['r_frame_rate']} "
               f"avg={info['avg_frame_rate']} start_time={info['start_time']} "
               f"flags={info['flags'] or 'none'}")
    if not times:
        start, end = window(row)
        start, end = start or 0.0, end or info["duration_s"]
        # Off-grid times (not on a frame boundary) so rounding mistakes show up.
        times = [round(start + p * (end - start), 3) + 0.013 for p in (0.05, 0.27, 0.5, 0.73, 0.95)]
    bad = 0
    for r in frame_check(video, times):
        ok = r["best_offset"] == 0 and r["diff_ffmpeg"] < 1.0 and r["seek_vs_linear"] == 0.0
        bad += not ok
        typer.echo(f"t={r['t']:.3f} -> f{r['idx']} ({r['frame_t']:.3f}s)  "
                   f"ffmpeg diff {r['diff_ffmpeg']:.2f}  best offset {r['best_offset']:+d}  "
                   f"neighbours {', '.join(f'{k:+d}:{v:.1f}' for k, v in sorted(r['diffs'].items()))}  "
                   f"seek vs linear {r['seek_vs_linear']:.2f}  {'OK' if ok else 'MISMATCH'}")
    if bad:
        raise typer.Exit(1)


@app.command("pose-check")
def pose_check(
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(None, help="Output image (default: outputs/pose_check.jpg)."),
) -> None:
    """Run rtmlib whole-body pose on one image and save it with skeletons drawn."""
    import cv2

    cfg = load_config()
    _init_ort()
    from rtmlib import Wholebody, draw_skeleton

    gpu, vram = _gpu_info()
    profile = pick_profile(cfg, vram)
    mode = cfg["profiles"][profile]["pose_mode"]
    device = cfg["device"] if cfg["device"] != "auto" else ("cuda" if gpu else "cpu")

    img = cv2.imread(str(image))
    if img is None:
        raise typer.BadParameter(f"could not read {image}")
    wb = Wholebody(mode=mode, backend="onnxruntime", device=device)
    keypoints, scores = wb(img)
    vis = draw_skeleton(img.copy(), keypoints, scores, openpose_skeleton=False, kpt_thr=0.4)

    out = out or cfg["paths"]["outputs"] / "pose_check.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    typer.echo(f"profile={profile} mode={mode} device={device}")
    typer.echo(f"det providers:  {wb.det_model.session.get_providers()}")
    typer.echo(f"pose providers: {wb.pose_model.session.get_providers()}")
    typer.echo(f"people={len(keypoints)} keypoints shape={getattr(keypoints, 'shape', None)}")
    typer.echo(f"saved {out}")


@app.command("track-golden")
def track_golden(
    only: list[str] = typer.Option(None, "--only", help="Only these golden event_ids / check_ids."),
    no_render: bool = typer.Option(False, "--no-render", help="Skip the debug videos."),
    no_checks: bool = typer.Option(False, "--no-checks", help="Skip the extra check spans."),
) -> None:
    """Detect + track every golden clip (and the config's extra check spans).

    Writes debug renders to outputs/renders/1.2/, every detection to
    data/cache/golden/<id>_tracks.parquet, and outputs/renders/1.2/report.md.
    """
    import csv
    import time

    from deepfoul.io.fetch import read_fights
    from deepfoul.io.shots import load_shots, shots_path
    from deepfoul.perception.detect import PersonDetector
    from deepfoul.perception.track import track_clip
    from deepfoul.viz.track_report import clip_summary, write_report

    cfg = load_config()
    _init_ort()
    gpu, vram = _gpu_info()
    mode = cfg["profiles"][pick_profile(cfg, vram)]["pose_mode"]
    device = cfg["device"] if cfg["device"] != "auto" else ("cuda" if gpu else "cpu")
    detector = PersonDetector(mode, device, cfg["det"]["score_min"])
    typer.echo(f"detector {mode} on {detector.providers[0]}")

    fights = {r["fight_id"].strip(): r for r in read_fights(cfg["paths"]["labels"] / "fights.csv")}
    with (cfg["paths"]["labels"] / "golden.csv").open(encoding="utf-8-sig", newline="") as f:
        jobs = [(g["event_id"], g["fight_id"], float(g["start_s"]), float(g["end_s"]),
                 f"{g['foul_type']}: {g['notes']}", True) for g in csv.DictReader(f)]
    if not no_checks:
        jobs += [(c["check_id"], c["fight_id"], float(c["start_s"]), float(c["end_s"]),
                  c.get("note", "extra check"), False) for c in cfg["tracking"]["checks"]]
    if only:
        jobs = [j for j in jobs if j[0] in only]

    out_dir = cfg["paths"]["outputs"] / "renders" / "1.2"
    cache_dir = cfg["paths"]["cache"] / "golden"
    cache_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip_id, fid, start, end, note, golden in jobs:
        shots_file = shots_path(cfg["paths"]["cache"], fid)
        if not shots_file.exists():
            typer.secho(f"{clip_id}: no shot file, run `deepfoul extract {fid} --shots`", fg="red")
            continue
        doc = load_shots(shots_file)
        kind = note.split(":")[0].replace(" ", "_")
        render = out_dir / f"{clip_id}_{fid}_{kind}.mp4" if golden and not no_render else None
        t0 = time.perf_counter()
        df, meta, det_s = track_clip(fights[fid], start, end, doc["shots"], doc["fps"], detector,
                                     cfg, cfg["paths"]["raw"], render)
        df.insert(0, "clip_id", clip_id)
        df.to_parquet(cache_dir / f"{clip_id}_tracks.parquet", index=False)
        s = clip_summary(clip_id, fid, start, end, note, df, meta, doc["fps"], det_s,
                         render.name if render else None)
        summaries.append(s)
        typer.echo(f"{clip_id}: {len(meta)} frames, {sum(x['tracks'] for x in s['shots'])} tracks "
                   f"in {len(s['shots'])} shots, det {s['det_fps']:.0f} fps, "
                   f"{time.perf_counter() - t0:.0f}s")
    report = out_dir / "report.md"
    write_report(summaries, report, f"{gpu} ({vram} GB)" if gpu else "none", detector.providers)
    typer.echo(f"report -> {report}")


@app.command("shorts-check")
def shorts_check(
    fight_ids: list[str] = typer.Argument(None, help="fight_ids (default: every fight)."),
    before: float = typer.Option(180, "--before", help="Seconds before action_start_s."),
    after: float = typer.Option(60, "--after", help="Seconds after action_start_s."),
    step: float = typer.Option(5, "--step", help="Seconds between sheet frames."),
    save: list[str] = typer.Option(None, "--save", help="fight_id=mm:ss.cc: save that full frame "
                                   "to outputs/shorts_check/<fight_id>.jpg instead of a sheet."),
) -> None:
    """Contact sheets around round 1 for reading shorts colors (outputs/shorts_check/sheets/)."""
    import cv2
    import numpy as np

    from deepfoul.io.fights import window
    from deepfoul.io.previews import time_sheet
    from deepfoul.io.video import ffmpeg_frame, video_info

    cfg = load_config()
    out_dir = cfg["paths"]["outputs"] / "shorts_check"
    if save:
        for item in save:
            fid, _, ts = item.partition("=")
            m, _, s = ts.partition(":")
            t = int(m) * 60 + float(s)
            video = cfg["paths"]["raw"] / f"{fid}.mp4"
            info = video_info(video)
            img = ffmpeg_frame(video, t, info["width"], info["height"])
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{fid}.jpg"), np.ascontiguousarray(img))
            typer.echo(f"{fid} {ts} -> {out_dir / f'{fid}.jpg'}")
        _shorts_report(cfg, out_dir)
        return
    for row in _fight_rows(fight_ids or []):
        fid = row["fight_id"].strip()
        video = cfg["paths"]["raw"] / f"{fid}.mp4"
        if not video.exists():
            typer.secho(f"{fid}: no video", fg="red", err=True)
            continue
        start = window(row)[0] or 0.0
        times = list(np.arange(max(0.0, start - before), start + after, step))
        time_sheet(video, times, out_dir / "sheets" / f"{fid}.jpg")
        typer.echo(f"{fid}: {len(times)} frames")
    _shorts_report(cfg, out_dir)


def _shorts_report(cfg: dict, out_dir: Path) -> None:
    from deepfoul.io.fetch import read_fights
    from deepfoul.io.previews import write_shorts_report

    rows = read_fights(cfg["paths"]["labels"] / "fights.csv")
    images = {p.stem for p in out_dir.glob("*.jpg")}
    write_shorts_report(rows, cfg["identity"]["similar_pairs"], images, out_dir / "report.md")
    typer.echo(f"index -> {out_dir / 'renders.md'}")


@app.command("id-golden")
def id_golden(
    only: list[str] = typer.Option(None, "--only", help="Only these golden event_ids / check_ids."),
    from_cache: bool = typer.Option(False, "--from-cache",
                                    help="Reuse cached boxes + pose + colors (no GPU, no decode "
                                    "except for renders)."),
    no_render: bool = typer.Option(False, "--no-render", help="Skip the videos."),
    show_ref: bool = typer.Option(False, "--show-ref", help="Draw a thin labelled box on the ref."),
    rebuild_profiles: bool = typer.Option(False, "--rebuild-profiles",
                                          help="Re-learn the fight profiles."),
    no_checks: bool = typer.Option(False, "--no-checks", help="Skip the extra check spans."),
    tag: str = typer.Option("1.4", "--tag", help="Output folder under outputs/renders/."),
) -> None:
    """Detect + track + pose + red/blue/ref identity on every golden clip (and the checks).

    Same pass as `deepfoul extract`, on the golden spans. Renders to outputs/renders/<tag>/ (with
    --show-ref: .../show_ref/), caches to data/cache/identity/, writes
    outputs/renders/<tag>/report.md with each clip compared to labels/golden_baseline.csv.
    """
    import csv
    import json

    from deepfoul.io.fetch import read_fights
    from deepfoul.io.shots import first_frame_at, load_shots, shots_path
    from deepfoul.perception.extract import (identify, perceive_span, read_boxes, stride_for,
                                             write_parquet)
    from deepfoul.perception.id_clip import get_profile, render_identity
    from deepfoul.perception.identity import check_lookahead
    from deepfoul.viz.id_report import clip_identity_summary, read_baseline, write_id_report

    cfg = load_config()
    detector = pose = None
    gpu, vram = _gpu_info()
    if not from_cache or rebuild_profiles:
        from deepfoul.perception.detect import PersonDetector
        from deepfoul.perception.pose import PoseEstimator

        _init_ort()
        mode = cfg["profiles"][pick_profile(cfg, vram)]["pose_mode"]
        device = cfg["device"] if cfg["device"] != "auto" else ("cuda" if gpu else "cpu")
        detector = PersonDetector(mode, device, cfg["det"]["score_min"])
        pose = PoseEstimator(mode, device, cfg["pose"]["batch"])
        typer.echo(f"detector + pose ({mode}) on {detector.providers[0]} / {pose.providers[0]}, "
                   f"pose batched: {pose.batched}")

    fights = {r["fight_id"].strip(): r for r in read_fights(cfg["paths"]["labels"] / "fights.csv")}
    with (cfg["paths"]["labels"] / "golden.csv").open(encoding="utf-8-sig", newline="") as f:
        jobs = [(g["event_id"], g["fight_id"], float(g["start_s"]), float(g["end_s"]),
                 f"{g['foul_type']}: {g['notes']}", True) for g in csv.DictReader(f)]
    if not no_checks:
        jobs += [(c["check_id"], c["fight_id"], float(c["start_s"]), float(c["end_s"]),
                  c.get("note", "extra check"), False) for c in cfg["tracking"]["checks"]]
    if only:
        jobs = [j for j in jobs if j[0] in only]

    out_dir = cfg["paths"]["outputs"] / "renders" / tag
    cache_dir = cfg["paths"]["cache"] / "identity"
    cache_dir.mkdir(parents=True, exist_ok=True)
    profiles, summaries = {}, []
    for clip_id, fid, start, end, note, golden in jobs:
        row = fights[fid]
        doc = load_shots(shots_path(cfg["paths"]["cache"], fid))
        fps = doc["fps"]
        stride = stride_for(fps, cfg["extract"]["target_fps"])
        check_lookahead(cfg["identity"], fps, cfg["lookahead_s"])
        if fid not in profiles:
            profiles[fid] = get_profile(row, cfg, detector, pose, cfg["paths"]["raw"], cache_dir,
                                        fps, rebuild_profiles)
            p = profiles[fid]
            typer.echo(f"{fid}: profile from {p.n} teaching frames"
                       + "".join(f"\n  WARNING {w}" for w in p.warnings))
        boxes_path = cache_dir / f"{clip_id}_boxes.parquet"
        frames_path = cache_dir / f"{clip_id}_frames.parquet"
        timing_path = cache_dir / f"{clip_id}_timing.json"
        timing = None
        if from_cache:
            if not boxes_path.exists():
                typer.secho(f"{clip_id}: no cache (1.3 caches have no colors), run without "
                            "--from-cache", fg="red")
                continue
            boxes, frames = read_boxes(boxes_path), pd_read(frames_path)
            if timing_path.exists():
                timing = json.loads(timing_path.read_text())
        else:
            boxes, frames, timing = perceive_span(
                row, first_frame_at(start, fps), first_frame_at(end, fps), doc["shots"], fps,
                stride, detector, pose, cfg, cfg["paths"]["raw"])
            write_parquet(boxes, boxes_path)
            write_parquet(frames, frames_path)
            timing_path.write_text(json.dumps(timing))
        ident = identify(boxes, profiles[fid], cfg, fps / stride)
        ident = pd_concat([boxes.drop(columns=["kpts", "kpt_scores"]), ident])
        ident.drop(columns=[c for c in ident.columns if c.startswith("hist_")]).to_parquet(
            cache_dir / f"{clip_id}_identity.parquet", index=False)
        kind = note.split(":")[0].replace(" ", "_")
        render = None
        if golden and not no_render:
            render = (out_dir / "show_ref" if show_ref else out_dir) / f"{clip_id}_{fid}_{kind}.mp4"
            render_identity(row, start, end, frames, ident, cfg["paths"]["raw"], render, show_ref)
        s = clip_identity_summary(clip_id, fid, start, end, note, frames, ident, fps, timing,
                                  render.relative_to(out_dir).as_posix() if render else None,
                                  step=stride)
        summaries.append(s)
        typer.echo(f"{clip_id}: both fighters {s['both_pct']:.0f}% of frames, "
                   f"both or merged {s['both_or_merged_pct']:.0f}%, "
                   f"flips {s['flips']}, ref {s['ref_frames']}, merged {s['merged_frames']}"
                   + (f", pass {timing['frames'] / timing['wall_s']:.1f} fps" if timing else ""))
    report = out_dir / "report.md"
    write_id_report(summaries, profiles, fights, report, f"{gpu} ({vram} GB)" if gpu else "none",
                    cfg["identity_issue_checks"],
                    read_baseline(cfg["paths"]["labels"] / "golden_baseline.csv"),
                    f"Identity check ({tag})")
    typer.echo(f"report -> {report}")


def pd_concat(frames):
    import pandas as pd

    return pd.concat(frames, axis=1)


def pd_read(path):
    import pandas as pd

    return pd.read_parquet(path)


@app.command()
def mine(
    fight_ids: list[str] = typer.Argument(None, help="fight_ids (train or val) to mine."),
    all_: bool = typer.Option(False, "--all", help="Every train fight with a keypoint cache."),
    check: bool = typer.Option(False, "--check", help="Also score the val fights for the "
                               "known-foul recall check (in memory, never queued)."),
    sheets: bool = typer.Option(True, "--sheets/--no-sheets", help="Top-candidate contact "
                                "sheets (ffmpeg, a few seconds per fight)."),
) -> None:
    """Stoppage finder: rank candidate foul moments from the keypoint cache into label queues."""
    import csv

    from deepfoul.io.fights import window
    from deepfoul.labeling.queues import write_queue
    from deepfoul.mining.miner import (build_mined, fight_queue, known_ranks, load_rows,
                                       mine_fight, variant_cfg, write_fight_queue)
    from deepfoul.mining.report import plot_signals, write_report, write_top_sheet
    from deepfoul.perception.extract import keypoints_paths

    cfg = load_config()
    mcfg, lab = cfg["mining"], cfg["labeling"]
    labels = cfg["paths"]["labels"]
    every = _fight_rows([], None)
    has_cache = {r["fight_id"].strip() for r in every
                 if keypoints_paths(cfg["paths"]["keypoints"], r["fight_id"].strip())["boxes"].exists()}
    if not fight_ids and not all_ and not check:
        raise typer.BadParameter("give fight_ids, --all or --check")
    rows = _fight_rows(fight_ids) if fight_ids else []
    if all_:
        rows += [r for r in every if r["split"].strip() == "train"]
    if check:
        rows += [r for r in every if r["split"].strip() == "val"]
    seen, picked = set(), []
    for r in rows:
        fid = r["fight_id"].strip()
        if fid in seen:
            continue
        seen.add(fid)
        if r["split"].strip() == "test":
            raise typer.BadParameter(f"{fid} is a test fight: never mined")
        if fid not in has_cache:
            typer.echo(f"{fid}: no keypoint cache, skipped (run `deepfoul extract`)")
            continue
        picked.append(r)
    with (labels / "notes" / "known_fouls.csv").open(newline="", encoding="utf-8") as f:
        fouls = list(csv.DictReader(f))
    qdir, odir = cfg["paths"]["queues"], cfg["paths"]["outputs"] / "mining"
    pb, pa = lab["pad_before_s"], lab["pad_after_s"]
    results = []
    for r in picked:
        fid, split = r["fight_id"].strip(), r["split"].strip()
        w0, w1 = window(r)
        kp = keypoints_paths(cfg["paths"]["keypoints"], fid)["boxes"]
        if w1 is None:
            w1 = float(pd_read_meta(kp)["window"][1])
        w = (w0 or 0.0, w1)
        df = load_rows(kp, *w)
        mine_f = [f for f in fouls if f["fight_id"] == fid]
        out = {}
        for variant in ("untuned", "tuned"):
            out[variant] = mine_fight(df, w, variant_cfg(mcfg, variant), cfg["kpt_conf_min"],
                                      cfg["lookahead_s"], lab["min_gap_s"])
        sig, cands = out["tuned"]
        queued = split == "train" or (split == "val" and fid in (fight_ids or []))
        if queued:
            write_fight_queue(qdir, fid, fight_queue(fid, cands))
        plot_signals(fid, sig, cands, mine_f, mcfg["onset_back_s"], odir / f"{fid}_signals.png")
        if sheets and split == "train":
            write_top_sheet(cfg["paths"]["raw"] / f"{fid}.mp4", cands, mcfg["onset_back_s"],
                            odir / f"{fid}_top.jpg")
        ranks = {v: known_ranks(out[v][1], mine_f, pb, pa) for v in out}
        results.append({"fight_id": fid, "split": split, "cands": cands, "ranks": ranks,
                        "queued": queued})
        typer.echo(f"{fid} ({split}): {len(cands)} candidates, known fouls "
                   f"{[_r['rank'] for _r in ranks['tuned']]}" + ("" if queued else ", not queued"))
    train_ids = [r["fight_id"].strip() for r in every if r["split"].strip() == "train"]
    items, extra = build_mined(qdir, train_ids, fouls, pb, pa)
    write_queue(items, qdir / "mined.csv")
    write_report(odir / "report.md", results, len(items), extra, mcfg["per_fight"])
    typer.echo(f"mined.csv: {len(items)} items ({len(extra)} extra) -> {qdir / 'mined.csv'}")
    typer.echo(f"report: {odir / 'report.md'}")


def pd_read_meta(path):
    from deepfoul.perception.extract import read_meta
    return read_meta(path)


@app.command()
def label(
    queue: str = typer.Option(None, "--queue", help="Queue name in data/queues/ (default: practice)."),
    fight: str = typer.Option(None, "--fight", help="Full-fight mode for this fight_id."),
    events: Path = typer.Option(None, "--events", help="Events CSV to write (default: "
                                "labels/events.csv; the practice queue uses its own file)."),
    port: int = typer.Option(None, "--port", help="Port (default from config, 8000)."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open a browser tab."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild the practice queue."),
) -> None:
    """Local labeling app: a queue of moments, or a whole fight with --fight."""
    import webbrowser

    import uvicorn

    from deepfoul.io.fetch import read_fights
    from deepfoul.labeling.app import create_app
    from deepfoul.labeling.queues import (build_practice_queue, load_queue, queue_path,
                                          read_csv, write_queue)

    cfg = load_config()
    lab = cfg["labeling"]
    labels = cfg["paths"]["labels"]
    if queue and fight:
        raise typer.BadParameter("give --queue or --fight, not both")
    fights = {f["fight_id"].strip(): f for f in read_fights(labels / "fights.csv")}
    items, skips_path, widen_path = None, None, None
    if fight:
        if fight not in fights:
            raise typer.BadParameter(f"unknown fight_id {fight!r}")
        if not (cfg["paths"]["raw"] / f"{fight}.mp4").is_file():
            raise typer.BadParameter(f"{fight}: no video, run `deepfoul fetch`")
        mode, default_events = "fight", labels / "events.csv"
    else:
        queue = queue or "practice"
        qpath = queue_path(cfg, queue)
        if queue == "practice":
            default_events = cfg["root"] / lab["practice_events"]
            if rebuild or not qpath.exists():
                available = {fid for fid in fights if (cfg["paths"]["raw"] / f"{fid}.mp4").is_file()}
                built = build_practice_queue(
                    list(fights.values()), read_csv(labels / "golden.csv"),
                    read_csv(labels / "notes" / "known_fouls.csv"), available,
                    n=lab["practice_n"], seed=lab["practice_seed"], gap=lab["min_gap_s"],
                    pad_before=lab["pad_before_s"], pad_after=lab["pad_after_s"])
                write_queue(built, qpath)
                typer.echo(f"practice queue: {len(built)} items -> {qpath}")
        else:
            default_events = labels / "events.csv"
            if not qpath.exists():
                raise typer.BadParameter(f"no queue file {qpath}")
        items = load_queue(qpath)
        skips_path = qpath.with_suffix(".skips.json")
        widen_path = qpath.with_suffix(".widen.json")
        mode = "queue"
    events_path = (events if events else default_events).resolve()
    if events_path == (labels / "events.csv").resolve():
        typer.secho("writing labels/events.csv: git pull before labeling, commit and push after "
                    "(event ids are max + 1 and collide across machines otherwise)", fg="yellow")
    port = port or lab["port"]
    url = f"http://localhost:{port}"
    typer.echo(f"{mode}: {queue or fight} -> {events_path}")
    typer.echo(f"open {url} in Chrome or Edge (Ctrl+C stops)")
    app_ = create_app(cfg, events_path, mode, queue=items, queue_name=queue, fight_id=fight,
                      skips_path=skips_path, widen_path=widen_path)
    if not no_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app_, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    app()
