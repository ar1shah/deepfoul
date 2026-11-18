"""Download every video in fights.csv with yt-dlp into data/raw/<fight_id>.mp4."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


def read_fights(csv_path: Path) -> list[dict[str, str]]:
    # utf-8-sig: tolerate the BOM that Excel / PowerShell add when editing CSVs.
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if row.get("fight_id", "").strip()]


def ytdlp_command(url: str, out_path: Path, max_height: int) -> list[str]:
    return [
        "yt-dlp",
        "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", str(out_path),
        url,
    ]


def fetch_all(rows: list[dict[str, str]], raw_dir: Path, max_height: int,
              only: str | None = None, run=subprocess.run) -> dict[str, list[str]]:
    """Download missing videos. Returns {"downloaded": [...], "skipped": [...], "failed": [...]}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[str]] = {"downloaded": [], "skipped": [], "failed": []}
    for row in rows:
        fid = row["fight_id"].strip()
        if only and fid != only:
            continue
        out = raw_dir / f"{fid}.mp4"
        url = row.get("source_url", "").strip()
        if out.exists():
            result["skipped"].append(fid)
            continue
        if not url:
            result["failed"].append(fid)
            continue
        proc = run(ytdlp_command(url, out, max_height))
        ok = proc.returncode == 0 and out.exists()
        result["downloaded" if ok else "failed"].append(fid)
    return result
