"""Check downloaded videos: ffprobe stats, sanity flags, contact sheets, a markdown report."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from deepfoul.io.fights import window

MIN_HEIGHT = 720
NO_WINDOW_MIN_S, NO_WINDOW_MAX_S = 5 * 60, 45 * 60
VFR_TOLERANCE = 0.01


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def _rate(s: str | None) -> float | None:
    try:
        r = Fraction(s)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return float(r) if r else None


def summarize(probe: dict) -> dict:
    """Duration, size, frame rates and codec of the first video stream."""
    v = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
    dur = probe.get("format", {}).get("duration") or v.get("duration")
    return {
        "duration_s": float(dur) if dur else None,
        "width": v.get("width"),
        "height": v.get("height"),
        "r_fps": _rate(v.get("r_frame_rate")),
        "avg_fps": _rate(v.get("avg_frame_rate")),
        "codec": v.get("codec_name"),
    }


def flags_for(row: dict[str, str], info: dict) -> list[str]:
    """Things that look wrong about a downloaded video, given its fights.csv row."""
    flags = []
    dur = info.get("duration_s")
    _, end = window(row)
    if dur is None:
        flags.append("no duration")
    elif end is not None:
        if dur < end:
            flags.append(f"shorter than window end ({dur:.0f}s < {end:.0f}s)")
    elif not NO_WINDOW_MIN_S <= dur <= NO_WINDOW_MAX_S:
        flags.append(f"odd length for a full fight ({dur / 60:.1f} min)")
    h = info.get("height")
    if h is None or h < MIN_HEIGHT:
        flags.append(f"under {MIN_HEIGHT}p ({h}p)")
    r, avg = info.get("r_fps"), info.get("avg_fps")
    if r and avg and abs(r - avg) / r > VFR_TOLERANCE:
        flags.append(f"variable frame rate (r={r:.2f}, avg={avg:.2f})")
    if "unofficial upload" in (row.get("notes") or ""):
        flags.append("unofficial upload")
    return flags


def bucket(row: dict[str, str]) -> str:
    for part in (row.get("notes") or "").split(";"):
        part = part.strip()
        if part.startswith("bucket="):
            return part[len("bucket="):]
    return "-"


def sheet_times(row: dict[str, str], duration_s: float) -> list[float]:
    """10%, 50% and 90% of the fight window (whole video if there's no window)."""
    start, end = window(row)
    start = start or 0.0
    end = min(end, duration_s) if end else duration_s
    return [start + p * (end - start) for p in (0.1, 0.5, 0.9)]


def contact_sheet(video: Path, times: list[float], out: Path, width: int = 640) -> bool:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((360, width, 3), np.uint8)
        frame = cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])))
        label = f"{int(t // 60):02d}:{int(t % 60):02d}"
        for color, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(frame, label, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, thick)
        tiles.append(frame)
    cap.release()
    h = min(t.shape[0] for t in tiles)
    out.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out), np.hstack([t[:h] for t in tiles]))


def _mmss(s: float | None) -> str:
    return "-" if s is None else f"{int(s // 60)}:{int(s % 60):02d}"


def check_all(rows: list[dict[str, str]], raw_dir: Path, out_dir: Path) -> list[dict]:
    """Probe every row, write contact sheets and report.md into out_dir. Returns the result rows."""
    results = []
    for row in rows:
        fid = row["fight_id"].strip()
        video = raw_dir / f"{fid}.mp4"
        res = {"fight_id": fid, "split": row.get("split", ""), "bucket": bucket(row),
               "status": "", "info": {}, "flags": [], "sheet": ""}
        if video.exists():
            res["status"] = "downloaded"
            try:
                res["info"] = summarize(ffprobe(video))
                res["flags"] = flags_for(row, res["info"])
                sheet = out_dir / f"{fid}.jpg"
                if res["info"]["duration_s"] and contact_sheet(
                        video, sheet_times(row, res["info"]["duration_s"]), sheet):
                    res["sheet"] = sheet.name
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                res["flags"] = [f"ffprobe failed: {e}"]
        else:
            res["status"] = "failed" if (row.get("source_url") or "").strip() else "no URL"
        results.append(res)
    write_report(results, out_dir / "report.md")
    return results


def write_report(results: list[dict], path: Path) -> None:
    lines = ["# Fetch check", "",
             "| fight_id | split | bucket | status | duration | resolution | fps | flags | sheet |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        i = r["info"]
        res = f"{i['width']}x{i['height']}" if i.get("height") else "-"
        fps = f"{i['avg_fps']:.2f}" if i.get("avg_fps") else "-"
        sheet = f"[{r['sheet']}]({r['sheet']})" if r["sheet"] else "-"
        lines.append(f"| {r['fight_id']} | {r['split']} | {r['bucket']} | {r['status']} | "
                     f"{_mmss(i.get('duration_s'))} | {res} | {fps} | "
                     f"{'; '.join(r['flags']) or 'ok'} | {sheet} |")
    lines += ["", "## Summary per bucket", "",
              "| bucket | downloaded | failed | no URL | flagged |", "|---|---|---|---|---|"]
    for b in sorted({r["bucket"] for r in results}):
        rs = [r for r in results if r["bucket"] == b]
        n = {s: sum(r["status"] == s for r in rs) for s in ("downloaded", "failed", "no URL")}
        flagged = sum(bool(r["flags"]) for r in rs)
        lines.append(f"| {b} | {n['downloaded']} | {n['failed']} | {n['no URL']} | {flagged} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
