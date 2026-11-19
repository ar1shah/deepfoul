"""labels/fights.csv schema, time parsing and validation."""

from __future__ import annotations

import re
from datetime import date

FIGHT_COLS = ["fight_id", "event", "date", "red_fighter", "blue_fighter", "red_shorts",
              "blue_shorts", "source_url", "split", "action_start_s", "action_end_s", "notes"]
SPLITS = {"train", "val", "test"}

_FIGHT = r"ufc(?:fn\d{8}|\d+)_[a-z0-9]+_[a-z0-9]+"
ID_RE = re.compile(rf"^(?:{_FIGHT}|comp_[a-z0-9]+_\d{{2}}|clip_{_FIGHT})$")
URL_RE = re.compile(r"^https://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}")


def parse_mmss(s: str) -> int:
    """'04:13' -> 253, '0:43' -> 43, '1:02:03' -> 3723."""
    parts = s.strip().split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"bad time {s!r}, expected mm:ss or h:mm:ss")
    nums = [int(p) for p in parts]
    if any(n >= 60 for n in nums[1:]):
        raise ValueError(f"bad time {s!r}, minutes/seconds must be < 60")
    total = 0
    for n in nums:
        total = total * 60 + n
    return total


def window(row: dict[str, str]) -> tuple[float | None, float | None]:
    """(action_start_s, action_end_s) as floats, None where blank."""
    def f(k: str) -> float | None:
        v = (row.get(k) or "").strip()
        return float(v) if v else None
    return f("action_start_s"), f("action_end_s")


def validate_fights(rows: list[dict[str, str]]) -> list[str]:
    """Return a list of human-readable problems (empty if all rows are fine)."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(rows, start=2):  # line 1 is the header
        fid = (row.get("fight_id") or "").strip()
        where = f"line {i} ({fid or 'no id'})"
        if not ID_RE.match(fid):
            problems.append(f"{where}: bad fight_id format")
        if fid in seen:
            problems.append(f"{where}: duplicate fight_id")
        seen.add(fid)
        if (row.get("split") or "").strip() not in SPLITS:
            problems.append(f"{where}: split must be one of {sorted(SPLITS)}")
        d = (row.get("date") or "").strip()
        if d:
            try:
                date.fromisoformat(d)
            except ValueError:
                problems.append(f"{where}: bad date {d!r}, expected YYYY-MM-DD")
        url = (row.get("source_url") or "").strip()
        if url and not URL_RE.match(url):
            problems.append(f"{where}: source_url is not a YouTube link")
        try:
            start, end = window(row)
        except ValueError:
            problems.append(f"{where}: action_start_s / action_end_s must be numbers")
            continue
        if (start is not None and start < 0) or (start is not None and end is not None and start >= end):
            problems.append(f"{where}: bad window {start}-{end}")
    return problems
