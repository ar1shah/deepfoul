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
