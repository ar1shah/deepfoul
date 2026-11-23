"""`deepfoul ingest` (2.3): new fights from a Cowork draft -> fetch -> extract -> mine, + a report.

Each stage reuses the existing command (fetch_all in-process; extract and mine as subprocesses,
so a crash in one fight's GPU pass can't take the report down). Every stage skips what is
already up to date, so ingest can be re-run after a failure or an overnight interruption.
Test fights are only fetched (they're extracted in 6.0 and never mined).
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from deepfoul.io.fights import FIGHT_COLS, shorts_names, validate_fights, window


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def merge_draft(fights_path: Path, draft_path: Path, colors: set[str] | None,
                archive_dir: Path, today: str | None = None) -> tuple[list[str], list[str]]:
    """Append the draft's rows to fights.csv. Returns (new fight_ids, problems). On any problem
    nothing is written. The draft is moved to archive_dir/<date>_fights_draft.csv."""
    fights = read_rows(fights_path)
    draft = read_rows(draft_path)
    problems = []
    if draft:
        missing = [c for c in FIGHT_COLS if c not in draft[0]]
        if missing:
            return [], [f"draft is missing columns: {', '.join(missing)}"]
    have = {r["fight_id"].strip() for r in fights}
    dup = [r["fight_id"].strip() for r in draft if r["fight_id"].strip() in have]
    if dup:
        problems.append(f"already in fights.csv: {', '.join(dup)}")
    # validate the merged table, but only report lines from the draft
    merged = fights + [{c: (r.get(c) or "").strip() for c in FIGHT_COLS} for r in draft]
    n0 = len(fights) + 2
    for p in validate_fights(merged, colors):
        line = int(p.split()[1]) if p.startswith("line ") else 0
        if line >= n0:
            problems.append(f"draft {p.replace(f'line {line}', f'row {line - n0 + 1}', 1)}")
    if problems:
        return [], problems
    text = fights_path.read_text(encoding="utf-8")
    with fights_path.open("a", newline="", encoding="utf-8") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        w = csv.DictWriter(f, FIGHT_COLS, lineterminator="\n")
        for r in draft:
            w.writerow({c: (r.get(c) or "").strip() for c in FIGHT_COLS})
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / f"{today or date.today().isoformat()}_fights_draft.csv"
    k = 2
    while dest.exists():
        dest = archive_dir / f"{today or date.today().isoformat()}_fights_draft_{k}.csv"
        k += 1
    shutil.move(str(draft_path), dest)
    return [r["fight_id"].strip() for r in draft], []


def blockers(row: dict[str, str], has_video: bool) -> list[str]:
    """Why a fight can't be extracted yet (empty = ready)."""
    out = []
    if not has_video:
        out.append("no video")
    if not shorts_names(row.get("red_shorts", "")) or not shorts_names(row.get("blue_shorts", "")):
        out.append("shorts colors missing (run `deepfoul shorts-check`)")
    if window(row)[0] is None:
        out.append("no fight window (action_start_s)")
    return out


def run_cli(args: list[str], log: Path) -> int:
    """Run `python -m deepfoul <args>`, streaming to the console and to log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n$ deepfoul {' '.join(args)}\n")
        f.flush()
        proc = subprocess.Popen([sys.executable, "-m", "deepfoul", *args], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def cache_stats(kp_dir: Path, fid: str) -> dict | None:
    from deepfoul.perception.extract import keypoints_paths, summarize
    p = keypoints_paths(kp_dir, fid)
    if not p["boxes"].exists() or not p["frames"].exists():
        return None
    return summarize(pd.read_parquet(p["boxes"], columns=["frame_idx", "corner"]),
                     pd.read_parquet(p["frames"], columns=["frame_idx"]))


def queue_info(qdir: Path, fid: str, fouls: list[dict], pad: float) -> tuple[int, list[str]]:
    """(items in data/queues/mined_<fid>.csv, rank of each known foul or 'miss')."""
    p = qdir / f"mined_{fid}.csv"
    if not p.exists():
        return 0, []
    items = read_rows(p)
    ranks = []
    for kf in fouls:
        if kf["fight_id"] != fid:
            continue
        t = float(kf["live_s"])
        hit = [i for i, it in enumerate(items, 1) if abs(float(it["t_s"]) - t) <= pad]
        ranks.append(f"{kf['foul_type']} {int(t // 60):02d}:{t % 60:02.0f} "
                     + (f"#{hit[0]}" if hit else "miss"))
    return len(items), ranks


def write_report(out: Path, rows: list[dict], merged: list[str], notes: list[str]) -> None:
    L = [f"# Ingest report ({date.today().isoformat()})", ""]
    if merged:
        L += [f"Merged from the draft: {', '.join(merged)}.", ""]
    L += ["| fight | split | fetch | extract | b/m of visible | queue | known fouls (rank) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        bom = f"{r['stats']['bom_of_visible_pct']:.0f}%" if r.get("stats") else "-"
        L.append(f"| {r['fight_id']} | {r['split']} | {r['fetch']} | {r['extract']} | {bom} | "
                 f"{r.get('queue', '-')} | {'; '.join(r.get('ranks') or []) or '-'} |")
    L += [""] + [f"- {n}" for n in notes]
    L += ["", "Extract details: outputs/extract_report.md. Mining details and contact sheets: "
          "outputs/mining/report.md."]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
