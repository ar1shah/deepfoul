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


@app.command()
def extract(
    fight_ids: list[str] = typer.Argument(None, help="fight_ids from fights.csv."),
    shots: bool = typer.Option(False, "--shots", help="Detect camera cuts (the only stage so far)."),
    split: str = typer.Option(None, "--split", help="Run every fight in this split instead."),
    force: bool = typer.Option(False, "--force", help="Re-run even if the cache is up to date."),
) -> None:
    """Run perception stages on fights. Writes data/cache/<fight_id>_shots.json."""
    import time

    from deepfoul.io.fights import window
    from deepfoul.io.shots import (detect_shots, load_shots, save_shots, shots_cfg_for,
                                   shots_path, stale_reason)

    if not shots:
        raise typer.BadParameter("nothing to do: pass --shots (pose extraction lands in 1.4)")
    if not fight_ids and not split:
        raise typer.BadParameter("give fight_ids or --split")
    cfg = load_config()
    for row in _fight_rows(fight_ids or [], split):
        fid = row["fight_id"].strip()
        video = cfg["paths"]["raw"] / f"{fid}.mp4"
        if not video.exists():
            typer.secho(f"{fid}: no video at {video}, run `deepfoul fetch`", fg="red", err=True)
            continue
        start, end = window(row)
        win = (start or 0.0, end)
        out = shots_path(cfg["paths"]["cache"], fid)
        shots_cfg = shots_cfg_for(cfg["shots"], fid)
        if out.exists() and not force:
            reason = stale_reason(load_shots(out), win, shots_cfg)
            if reason is None:
                typer.echo(f"{fid}: up to date ({out.name})")
                continue
            typer.echo(f"{fid}: re-running, {reason}")
        t0 = time.perf_counter()
        doc = {"fight_id": fid, **detect_shots(video, win[0], win[1], shots_cfg)}
        save_shots(doc, out)
        lens = [s["end_s"] - s["start_s"] for s in doc["shots"]]
        typer.echo(f"{fid}: {len(lens)} shots, median {sorted(lens)[len(lens) // 2]:.1f}s, "
                   f"{time.perf_counter() - t0:.0f}s -> {out}")


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


if __name__ == "__main__":
    app()
