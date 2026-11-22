"""Label queues: data/queues/<name>.csv (item_id, fight_id, t_s, score, reason).

2.2 (mined) and 2.3 (hardneg) write the same format. An item is done when any event of the same
fight overlaps its clip window, so status needs no local state and carries across machines.
Skips are local (data/queues/<name>.skips.json).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from deepfoul.io.fights import window

QUEUE_COLS = ["item_id", "fight_id", "t_s", "score", "reason"]


def queue_path(cfg: dict, name: str) -> Path:
    return cfg["paths"]["queues"] / f"{name}.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_queue(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, QUEUE_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows({c: it.get(c, "") for c in QUEUE_COLS} for it in items)


def load_queue(path: Path) -> list[dict]:
    items = read_csv(path)
    for it in items:
        it["t_s"] = float(it["t_s"])
    return items


def golden_t(clip: dict[str, str], fouls: list[dict[str, str]]) -> float:
    """Foul clips: the known foul's live time inside the clip. Other clips: the midpoint."""
    a, b = float(clip["start_s"]), float(clip["end_s"])
    if clip["foul_type"] in ("eye_poke", "low_blow"):
        for f in fouls:
            if (f["fight_id"] == clip["fight_id"] and f["foul_type"] == clip["foul_type"]
                    and a <= float(f["live_s"]) <= b):
                return float(f["live_s"])
    return (a + b) / 2


def _far(t: float, spans: list[tuple[float, float]], gap: float) -> bool:
    """t is at least `gap` seconds from every span [a, b]."""
    return all(t < a - gap or t > b + gap for a, b in spans)


def build_practice_queue(fights: list[dict[str, str]], golden: list[dict[str, str]],
                         fouls: list[dict[str, str]], available: set[str], n: int = 30,
                         seed: int = 0, gap: float = 10.0, pad_before: float = 4.0,
                         pad_after: float = 4.0) -> list[dict]:
    """Golden clips + the other train known fouls + random train moments, shuffled.

    Only train fights with a video. Every item is >= gap from every other item of its fight, and
    random moments are >= gap from every known foul (live and replay) and golden clip range.
    """
    by_id = {f["fight_id"]: f for f in fights}
    train = {fid for fid, f in by_id.items() if f["split"].strip() == "train" and fid in available}
    items: list[dict] = []
    spans: dict[str, list[tuple[float, float]]] = {fid: [] for fid in train}

    def ok(fid: str, t: float) -> bool:
        return all(abs(t - it["t_s"]) >= gap for it in items if it["fight_id"] == fid)

    for clip in golden:
        fid = clip["fight_id"]
        if fid not in train:
            continue
        spans[fid].append((float(clip["start_s"]), float(clip["end_s"])))
        t = golden_t(clip, fouls)
        if ok(fid, t):
            items.append({"fight_id": fid, "t_s": t, "reason": f"{clip['event_id']} {clip['foul_type']}"})
    for f in fouls:
        fid = f["fight_id"]
        if fid not in train:
            continue
        for k in ("live_s", "replay_s"):
            if (f.get(k) or "").strip():
                spans[fid].append((float(f[k]), float(f[k])))
        t = float(f["live_s"])
        if ok(fid, t):
            items.append({"fight_id": fid, "t_s": t, "reason": f"known foul {f['foul_type']}"})

    rng = random.Random(seed)
    order = sorted(train)
    tries = 0
    while len(items) < n and tries < 10000:
        tries += 1
        fid = rng.choice(order)
        w0, w1 = window(by_id[fid])
        lo, hi = (w0 or 0.0) + pad_before, (w1 or 0.0) - pad_after
        if w1 is None or hi <= lo:
            continue
        t = round(rng.uniform(lo, hi), 1)
        if ok(fid, t) and _far(t, spans[fid], gap):
            items.append({"fight_id": fid, "t_s": t, "reason": "random"})
    rng.shuffle(items)
    for i, it in enumerate(items[:n], start=1):
        it["item_id"] = f"practice_{i:02d}"
        it["score"] = ""
    return items[:n]


def item_done(item: dict, events: list[dict[str, str]], pad_before: float, pad_after: float) -> bool:
    """Any event of the same fight overlaps the item's clip window."""
    a, b = item["t_s"] - pad_before, item["t_s"] + pad_after
    return any(e["fight_id"] == item["fight_id"] and float(e["start_s"]) <= b
               and float(e["end_s"]) >= a for e in events)


class Skips:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.ids = set(json.loads(path.read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError):
            self.ids = set()

    def set(self, item_id: str, skipped: bool = True) -> None:
        (self.ids.add if skipped else self.ids.discard)(item_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.ids)), encoding="utf-8")
        tmp.replace(self.path)
