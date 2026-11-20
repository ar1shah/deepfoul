"""Review outputs for Ari: cut contact sheets and golden candidate preview clips."""

from __future__ import annotations

import csv
import itertools
import statistics
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from deepfoul.io.video import iter_frames, video_info

LONG_SHOT_S = 60.0   # longer than this: possible missed cut
SHORT_SHOT_S = 1.0   # shorter than this: likely a flash or replay-wipe fake cut


def fmt_ts(t: float) -> str:
    """Seconds -> 'mm:ss.cc' (hundredths, like VLC). Minutes are not wrapped into hours."""
    cs = round(t * 100)
    return f"{cs // 6000:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def frame_label(t: float, idx: int) -> str:
    return f"{fmt_ts(t)}  f{idx}"


def put_label(img: np.ndarray, text: str, scale: float = 1.0) -> np.ndarray:
    import cv2

    org = (12, round(40 * scale))
    for color, thick in (((0, 0, 0), round(6 * scale)), ((255, 255, 255), round(2 * scale))):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, color, max(thick, 1),
                    cv2.LINE_AA)
    return img


def pick_cuts(cuts: list[int], n: int) -> list[int]:
    """n cuts spread evenly over the list (all of them if there are fewer)."""
    if len(cuts) <= n:
        return list(cuts)
    return [cuts[round(i)] for i in np.linspace(0, len(cuts) - 1, n)]


def cut_pair(video: Path, cut_frame: int, fps: float):
    """(last frame before the cut, first frame of the new shot) from the frame reader."""
    t = cut_frame / fps
    frames = list(iter_frames(video, t - 2.5 / fps, t + 1.5 / fps))
    before = [f for f in frames if f.idx < cut_frame]
    after = [f for f in frames if f.idx >= cut_frame]
    return (before[-1] if before else None), (after[0] if after else None)


def cut_contact_sheet(video: Path, doc: dict, n: int, out: Path, tile_w: int = 480) -> list[dict]:
    """One row per sampled cut: before | after, labeled. Returns per-cut stats for the report."""
    import cv2

    fps = doc["fps"]
    cuts = [s["start_frame"] for s in doc["shots"][1:]]
    rows, stats = [], []
    for k, c in enumerate(pick_cuts(cuts, n), 1):
        before, after = cut_pair(video, c, fps)
        tiles = []
        for f, side in ((before, "before"), (after, "after")):
            img = f.image if f is not None else np.zeros((270, tile_w, 3), np.uint8)
            img = cv2.resize(img, (tile_w, round(img.shape[0] * tile_w / img.shape[1])))
            if f is not None:
                put_label(img, f"#{k} {side} {frame_label(f.t, f.idx)}", 0.6)
            tiles.append(img)
        h = min(t.shape[0] for t in tiles)
        rows.append(np.hstack([t[:h] for t in tiles]))
        diff = (float(np.abs(before.image.astype(np.int16) - after.image.astype(np.int16)).mean())
                if before is not None and after is not None else None)
        stats.append({"n": k, "cut_frame": c, "t": c / fps, "diff": diff})
    w = max(r.shape[1] for r in rows) if rows else tile_w * 2
    sep = np.full((4, w, 3), 255, np.uint8)
    sheet = np.vstack([x for r in rows for x in (r, sep)]) if rows else np.zeros((10, w, 3), np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return stats


def shot_summary(doc: dict) -> dict:
    lens = [s["end_s"] - s["start_s"] for s in doc["shots"]]
    return {
        "shots": len(lens),
        "median_s": statistics.median(lens) if lens else 0.0,
        "max_s": max(lens) if lens else 0.0,
        "long": [s for s in doc["shots"] if s["end_s"] - s["start_s"] > LONG_SHOT_S],
        "short": [s for s in doc["shots"] if s["end_s"] - s["start_s"] < SHORT_SHOT_S],
    }


def write_shots_report(entries: list[dict], path: Path) -> None:
    """entries: {fight_id, summary, sheet, cuts} per fight."""
    span = lambda s: f"{fmt_ts(s['start_s'])}-{fmt_ts(s['end_s'])} ({s['end_s'] - s['start_s']:.2f}s)"  # noqa: E731
    lines = ["# Shot check", "",
             "Shots are half-open `[start, end)`. Times are in the original video (mm:ss.cc).", "",
             "| fight_id | shots | median | max | > 60 s | < 1 s | sheet |",
             "|---|---|---|---|---|---|---|"]
    for e in entries:
        s = e["summary"]
        sheet = f"[{e['sheet']}]({e['sheet']})" if e.get("sheet") else "-"
        lines.append(f"| {e['fight_id']} | {s['shots']} | {s['median_s']:.1f}s | {s['max_s']:.1f}s"
                     f" | {len(s['long'])} | {len(s['short'])} | {sheet} |")
    for e in entries:
        s = e["summary"]
        lines += ["", f"## {e['fight_id']}", ""]
        if e.get("cuts"):
            lines += ["Sampled cuts on the sheet (mean abs pixel diff before -> after):", ""]
            for c in e["cuts"]:
                diff = "-" if c["diff"] is None else f"{c['diff']:.1f}"
                lines.append(f"- #{c['n']} {fmt_ts(c['t'])} f{c['cut_frame']}  diff {diff}")
            lines.append("")
        lines.append(f"Possible missed cuts (shots > {LONG_SHOT_S:.0f} s): "
                     + (", ".join(span(x) for x in s["long"]) or "none"))
        lines.append("")
        lines.append(f"Possible fake cuts (shots < {SHORT_SHOT_S:.0f} s): "
                     + (", ".join(span(x) for x in s["short"]) or "none"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_known_fouls(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def golden_selection(fights: list[dict[str, str]], fouls: list[dict[str, str]],
                     exclude: set[str]) -> list[dict[str, str]]:
    """clear_live fouls from train fights, minus excluded fights."""
    train = {r["fight_id"].strip() for r in fights if r["split"].strip() == "train"}
    return [f for f in fouls if f["fight_id"] in train and f["fight_id"] not in exclude
            and f["visibility"].strip() == "clear_live"]


def clip_name(fight_id: str, live_s: float, foul_type: str) -> str:
    s = int(live_s)
    return f"{fight_id}_{s // 60:02d}-{s % 60:02d}_{foul_type}.mp4"


@contextmanager
def clip_writer(video: Path, first_t: float, end_s: float, size: tuple[int, int], out: Path):
    """Pipe BGR frames into an H.264 mp4 at the source frame rate, muxing the raw video's audio
    from first_t to end_s. Yields a write(img) function. The raw file is only read."""
    info = video_info(video)
    w, h = size
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", info["r_frame_rate"] or str(info["fps"]), "-i", "-",
           "-ss", f"{first_t:.6f}", "-t", f"{end_s - first_t:.6f}", "-i", str(video),
           "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        yield lambda img: proc.stdin.write(np.ascontiguousarray(img).tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {out}")


def write_preview_clip(video: Path, start_s: float, end_s: float, out: Path) -> int:
    """Re-encode [start_s, end_s) of the raw video to `out` (H.264 + AAC) with the original
    timestamp and frame index burned in. The raw file is only read. Returns frames written."""
    frames = iter_frames(video, start_s, end_s)
    first = next(frames)
    h, w = first.image.shape[:2]
    n = 0
    with clip_writer(video, first.t, end_s, (w, h), out) as write:
        for f in itertools.chain([first], frames):
            write(put_label(f.image, frame_label(f.t, f.idx), h / 720))
            n += 1
    return n


def time_sheet(video: Path, times: list[float], out: Path, cols: int = 6,
               tile_w: int = 400) -> None:
    """Contact sheet of single frames at `times` (ffmpeg seeks), each labeled with its time."""
    import cv2

    from deepfoul.io.video import ffmpeg_frame

    info = video_info(video)
    tiles = []
    for t in times:
        img = ffmpeg_frame(video, t, info["width"], info["height"]).copy()
        img = cv2.resize(img, (tile_w, round(img.shape[0] * tile_w / img.shape[1])))
        tiles.append(put_label(img, fmt_ts(t), 0.6))
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def shorts_similarity(red: str, blue: str, similar_pairs: list[list[str]]) -> str:
    """'' if the main shorts colors are distinct, else why they look alike."""
    from deepfoul.io.fights import shorts_names

    r, b = (shorts_names(red) or [""])[0], (shorts_names(blue) or [""])[0]
    if not r or not b:
        return "missing"
    if r == b:
        return f"same main color ({r})"
    if {r, b} in [set(p) for p in similar_pairs]:
        return f"similar ({r} / {b})"
    return ""


def write_shorts_report(rows: list[dict[str, str]], similar_pairs: list[list[str]],
                        images: set[str], path: Path) -> None:
    lines = ["# Shorts colors (1.3)", "",
             "Read from the broadcast trunk-color graphic where there is one. Main color first, "
             "accents after `+`. Corners from UFC.com (`labels/notes/starter_fights.md`).", "",
             "| fight_id | split | red corner | red shorts | blue corner | blue shorts | flag | image |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fid = r["fight_id"].strip()
        flag = shorts_similarity(r["red_shorts"], r["blue_shorts"], similar_pairs)
        cross = [c for c, own in (("red", "red_shorts"), ("blue", "blue_shorts"))
                 if c != (r[own].split("+")[0].strip() or c) and
                 (r["blue_shorts" if c == "red" else "red_shorts"].split("+")[0].strip() == c)]
        notes = [flag] + (["shorts color = opposite corner"] if cross else [])
        img = f"[{fid}.jpg]({fid}.jpg)" if fid in images else "-"
        lines.append(f"| {fid} | {r['split']} | {r['red_fighter']} | {r['red_shorts']} | "
                     f"{r['blue_fighter']} | {r['blue_shorts']} | {'; '.join(n for n in notes if n) or ''} | {img} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
