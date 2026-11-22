"""labels/events.csv schema, validation and crash-safe reads/writes."""

from __future__ import annotations

import csv
import io
import os
import re
import threading
from pathlib import Path

EVENT_COLS = ["event_id", "fight_id", "start_s", "end_s", "foul_type", "offender", "contact",
              "ref_stopped", "speed", "source", "notes"]
FOUL_TYPES = {"eye_poke", "low_blow", "open_hand", "hard_negative", "clean"}
FOULS_NEEDING_OFFENDER = {"eye_poke", "low_blow", "open_hand"}
OFFENDERS = {"red", "blue", "none"}
CONTACT = {"yes", "no", "unsure"}
YES_NO = {"yes", "no"}
SPEEDS = {"live", "slowmo"}
SOURCES = {"full_fight", "compilation", "mined"}
_ID_RE = re.compile(r"^ev_(\d+)$")


def default_source(fight_id: str, mode: str) -> str:
    """compilation for comp_* videos, else full_fight (--fight) or mined (a queue)."""
    if fight_id.startswith("comp_"):
        return "compilation"
    return "full_fight" if mode == "fight" else "mined"


def validate_event(row: dict, windows: dict[str, tuple[float | None, float | None]] | None = None
                   ) -> tuple[list[str], list[str]]:
    """(errors, warnings) for one event row. `windows` maps fight_id -> fight window."""
    errors: list[str] = []
    warnings: list[str] = []
    fid = str(row.get("fight_id") or "").strip()
    if not fid:
        errors.append("fight_id is missing")
    elif windows is not None and fid not in windows:
        errors.append(f"unknown fight_id {fid!r}")
    foul = row.get("foul_type") or ""
    if foul not in FOUL_TYPES:
        errors.append("pick a foul type (1-5)")
    offender = row.get("offender") or ""
    if offender not in OFFENDERS:
        errors.append("offender must be red, blue or none")
    elif foul in FOULS_NEEDING_OFFENDER and offender == "none":
        errors.append(f"{foul} needs an offender: press R or B")
    for key, allowed, hint in [("contact", CONTACT, "press C"),
                               ("ref_stopped", YES_NO, "press T"),
                               ("speed", SPEEDS, "press M"),
                               ("source", SOURCES, "")]:
        if (row.get(key) or "") not in allowed:
            errors.append(f"{key} is not set" + (f" ({hint})" if hint else ""))
    try:
        start, end = float(row.get("start_s")), float(row.get("end_s"))
    except (TypeError, ValueError):
        errors.append("start_s / end_s must be numbers")
    else:
        if start < 0 or end < start:
            errors.append("need 0 <= start <= end (use [ and ])")
        elif windows is not None and fid in windows:
            w0, w1 = windows[fid]
            if (w0 is not None and end < w0) or (w1 is not None and start > w1):
                warnings.append("event is outside the fight window")
    return errors, warnings


def _clean(row: dict) -> dict[str, str]:
    out = {c: "" if row.get(c) is None else str(row.get(c)).strip() for c in EVENT_COLS}
    for c in ("start_s", "end_s"):
        if out[c]:
            out[c] = f"{round(float(out[c]), 2):g}"
    out["notes"] = " ".join(out["notes"].split())  # one line, so the CSV stays hand-readable
    return out


class EventStore:
    """events.csv with one-line appends (flushed + fsynced) and atomic rewrites for edits."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write_all([])
        self._check_header()

    def _check_header(self) -> None:
        with self.path.open(newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
        if header != EVENT_COLS:
            raise ValueError(f"{self.path}: header {header} does not match {EVENT_COLS}")

    def list(self, fight_id: str | None = None) -> list[dict[str, str]]:
        with self.path.open(newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if (r.get("event_id") or "").strip()]
        return [r for r in rows if fight_id is None or r["fight_id"] == fight_id]

    def get(self, event_id: str) -> dict[str, str] | None:
        return next((r for r in self.list() if r["event_id"] == event_id), None)

    def _next_id(self, rows: list[dict[str, str]]) -> str:
        nums = [int(m.group(1)) for r in rows if (m := _ID_RE.match(r["event_id"]))]
        return f"ev_{(max(nums) + 1 if nums else 1):05d}"

    def add(self, row: dict) -> dict[str, str]:
        with self._lock:
            out = _clean(row)
            out["event_id"] = self._next_id(self.list())
            buf = io.StringIO()
            csv.DictWriter(buf, EVENT_COLS, lineterminator="\n").writerow(out)
            with self.path.open("rb") as f:  # a file saved without a final newline
                f.seek(-1, os.SEEK_END)
                needs_nl = f.read(1) not in (b"\n", b"\r")
            with self.path.open("a", newline="", encoding="utf-8") as f:
                f.write(("\n" if needs_nl else "") + buf.getvalue())
                f.flush()
                os.fsync(f.fileno())
            return out

    def update(self, event_id: str, row: dict) -> dict[str, str]:
        with self._lock:
            rows = self.list()
            for i, r in enumerate(rows):
                if r["event_id"] == event_id:
                    out = _clean(row)
                    out["event_id"] = event_id
                    rows[i] = out
                    self._write_all(rows)
                    return out
            raise KeyError(event_id)

    def delete(self, event_id: str) -> dict[str, str]:
        with self._lock:
            rows = self.list()
            keep = [r for r in rows if r["event_id"] != event_id]
            if len(keep) == len(rows):
                raise KeyError(event_id)
            self._write_all(keep)
            return next(r for r in rows if r["event_id"] == event_id)

    def _write_all(self, rows: list[dict[str, str]]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, EVENT_COLS, lineterminator="\n")
            w.writeheader()
            w.writerows({c: r.get(c, "") for c in EVENT_COLS} for r in rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
