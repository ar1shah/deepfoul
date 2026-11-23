"""Dataset builder (2.3): labels/events.csv + reviewed regions -> labeled training windows.

The output is an index (one row per window: fight, split, t0, t1, label, weight ...). Features
come in 3.2 and read the keypoints for a window with `dataset.windows.window_keypoints`.

Where windows come from:
- Events (eye_poke, low_blow, open_hand, hard_negative): every window on a `pos_stride_s` grid
  that holds at least `pos_min_overlap` of the event's span gets the event's label. A window
  that fits two events takes the label higher in `priority`; one that only partly touches a
  higher-priority event is dropped.
- Clean: only **reviewed** time. That's the clip [t - 4, t + 4] of every labeled queue item
  (Ari reviewed the whole clip, even where the saved clean span is the default 1 s), or a whole
  fight window once it has full_fight labels. Unlabeled time in a reviewed clip is clean, except
  within `buffer_s` of an event (windup and reaction).
- Clean rows are sorted by their note (`note_categories`, whole-word keywords): replays, foul
  timeouts, round breaks and "other" stoppages exclude their clip; cut / ref warning / slip /
  glove touch make their clip a hard negative; anything else is clean.
- Splits come from fights.csv. Test fights never get a window.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from deepfoul.labeling.queues import clip_window, done_items

LABELS = ("eye_poke", "low_blow", "open_hand", "hard_negative", "clean")
FOULS = ("eye_poke", "low_blow")
EVENT_LABELS = ("eye_poke", "low_blow", "open_hand", "hard_negative")
COLUMNS = ["window_id", "fight_id", "split", "t0", "t1", "label", "offender", "weight",
           "event_id", "source_kind", "coverage"]
EPS = 1e-6


def fmt(t: float) -> str:
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:05.2f}"


def categorize(note: str, categories: list[dict]) -> tuple[str, str]:
    """(category name, use) of a clean row's note. Whole-word, case-insensitive keyword match,
    first category wins, so "uppercut" is not "cut". No match = ("clean", "clean")."""
    for c in categories:
        for kw in c["keywords"]:
            if re.search(rf"\b{re.escape(kw)}\b", note or "", re.IGNORECASE):
                return c["name"], c["use"]
    return "clean", "clean"


def assign_splits(fight_ids: set[str], fights: dict[str, dict]) -> dict[str, str]:
    """fight_id -> split from fights.csv. An unknown fight is an error, never a guess."""
    unknown = sorted(f for f in fight_ids if f not in fights)
    if unknown:
        raise ValueError(f"events for fights not in fights.csv: {', '.join(unknown)}")
    return {f: fights[f]["split"].strip() for f in fight_ids}


# ---------- intervals ----------
def union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def subtract(spans: list[tuple[float, float]], cut: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = union(spans)
    for c0, c1 in union(cut):
        nxt = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if a < c0:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        out = nxt
    return out


def grid_inside(a: float, b: float, stride: float, w: float) -> np.ndarray:
    """Window starts on the absolute grid k * stride with [t0, t0 + w] inside [a, b]."""
    k0, k1 = int(np.ceil((a - EPS) / stride)), int(np.floor((b - w + EPS) / stride))
    return np.arange(k0, k1 + 1) * stride if k1 >= k0 else np.zeros(0)


def overlap(t0: float, t1: float, s: float, e: float) -> float:
    return max(0.0, min(t1, e) - max(t0, s))


def fits(t0: float, t1: float, s: float, e: float, min_share: float) -> bool:
    if e - s < EPS:                                   # a point event: it has to be inside
        return t0 <= s <= t1
    return overlap(t0, t1, s, e) >= min_share * (e - s) - EPS


@dataclass
class Clip:
    fight_id: str
    a: float
    b: float
    item_id: str
    use: str = "clean"               # clean | hard_negative | exclude
    why: list[str] = field(default_factory=list)


# ---------- reviewed regions ----------
def reviewed_clips(items: list[dict], events: list[dict], fights: dict[str, dict],
                   pad_before: float, pad_after: float) -> list[Clip]:
    """The clips Ari labeled: every queue item closed by an event, plus the whole window of
    fights with full_fight labels."""
    done = done_items(items, events, pad_before, pad_after)
    clips, seen = [], set()
    for it in items:
        if it["item_id"] in done and it["item_id"] not in seen:
            seen.add(it["item_id"])
            a, b = clip_window(it, pad_before, pad_after)
            clips.append(Clip(it["fight_id"], a, b, it["item_id"]))
    for fid in sorted({e["fight_id"] for e in events if e["source"] == "full_fight"}):
        w0, w1 = _window(fights[fid])
        clips.append(Clip(fid, w0, w1, "full_fight"))
    return clips


def _window(f: dict) -> tuple[float, float]:
    a, b = (f.get("action_start_s") or "").strip(), (f.get("action_end_s") or "").strip()
    return (float(a) if a else 0.0), (float(b) if b else float("inf"))


def apply_clean_rows(clips: list[Clip], events: list[dict], categories: list[dict]) -> list[dict]:
    """Sort every clean row into a category and mark the clips it sits in. Returns one record
    per clean row (for the summary)."""
    rows = []
    for e in events:
        if e["foul_type"] != "clean":
            continue
        name, use = categorize(e["notes"], categories)
        s, t = float(e["start_s"]), float(e["end_s"])
        hit = [c for c in clips if c.fight_id == e["fight_id"] and s <= c.b and t >= c.a]
        for c in hit:
            if c.item_id == "full_fight":
                continue            # a full-fight clean row marks only its own span (see build)
            rank = {"clean": 0, "hard_negative": 1, "exclude": 2}
            if rank[use] > rank[c.use]:
                c.use = use
            if use != "clean":
                c.why.append(f"{e['event_id']} {name}")
        rows.append({**e, "category": name, "use": use, "in_clip": bool(hit)})
    return rows


# ---------- windows ----------
def fight_windows(fid: str, split: str, events: list[dict], clips: list[Clip],
                  clean_rows: list[dict], cfg: dict, window: tuple[float, float]) -> list[dict]:
    """All windows of one fight (coverage is filled in later)."""
    d = cfg
    W, prio = d["window_s"], d["priority"]
    buf = d["buffer_s"]
    w0, w1 = window
    evs = [e for e in events if e["foul_type"] in EVENT_LABELS]
    spans = [(float(e["start_s"]), float(e["end_s"])) for e in evs]
    zones = [(s - buf[e["foul_type"]], t + buf[e["foul_type"]]) for e, (s, t) in zip(evs, spans)]
    out: list[dict] = []

    def touches_zone(t0: float, t1: float) -> bool:
        return any(t0 < z1 and t1 > z0 for z0, z1 in zones)

    # event windows: label each candidate start once, from every event it fits
    starts: set[float] = set()
    for s, t in spans:
        starts.update(np.round(grid_inside(s - W, t + W, d["pos_stride_s"], W), 6).tolist())
    for t0 in sorted(starts):
        t1 = t0 + W
        if t0 < w0 - EPS or t1 > w1 + EPS:
            continue
        fit = [(prio.index(e["foul_type"]), -overlap(t0, t1, s, t), e)
               for e, (s, t) in zip(evs, spans) if fits(t0, t1, s, t, d["pos_min_overlap"])]
        if not fit:
            continue
        rank, _, best = min(fit, key=lambda x: (x[0], x[1]))
        # partly touching a higher-priority event = ambiguous, drop
        if any(prio.index(e["foul_type"]) < rank and overlap(t0, t1, s, t) > 0
               and not fits(t0, t1, s, t, d["pos_min_overlap"])
               for e, (s, t) in zip(evs, spans)):
            continue
        out.append(_row(fid, split, t0, t1, best["foul_type"], best["offender"],
                        best["event_id"], "event", d))

    # clip windows: clean and stoppage hard negatives, away from every event's buffer zone
    fclips = [c for c in clips if c.fight_id == fid]
    queue_clips = [c for c in fclips if c.item_id != "full_fight"]
    full = [c for c in fclips if c.item_id == "full_fight"]
    frows = [r for r in clean_rows if r["fight_id"] == fid]

    def span(r) -> tuple[float, float]:
        return float(r["start_s"]), float(r["end_s"])

    # hard negatives: (region, event_id) = a stoppage clip, or a stoppage row's own span in a
    # full-fight-labeled fight
    hn = [((c.a, c.b), c.why[0].split()[0]) for c in queue_clips if c.use == "hard_negative"]
    if full:
        hn += [(span(r), r["event_id"]) for r in frows if r["use"] == "hard_negative"]
    # clean: clean queue clips + full windows, minus every clip / row that isn't clean
    region = union([(c.a, c.b) for c in queue_clips if c.use == "clean"]
                   + [(c.a, c.b) for c in full])
    region = subtract(region, [(c.a, c.b) for c in queue_clips if c.use != "clean"]
                      + ([span(r) for r in frows if r["use"] != "clean"] if full else []))
    jobs = [((a, b), "clean", "", "clean_clip") for a, b in region]
    jobs += [(sp, "hard_negative", ev, "stoppage_clip") for sp, ev in hn]
    taken: set[tuple[float, str]] = set()
    for (a, b), label, ev, kind in jobs:
        a, b = max(a, w0), min(b, w1)
        for t0 in grid_inside(a, b, d["neg_stride_s"], W):
            t1 = t0 + W
            key = (round(float(t0), 6), label)
            if key in taken or touches_zone(t0, t1):
                continue
            taken.add(key)
            out.append(_row(fid, split, t0, t1, label, "none", ev, kind, d))
    return out


def _row(fid, split, t0, t1, label, offender, event_id, kind, d) -> dict:
    return {"fight_id": fid, "split": split, "t0": round(float(t0), 3), "t1": round(float(t1), 3),
            "label": label, "offender": offender or "none",
            "weight": d["hard_neg_weight"] if label == "hard_negative" else 1.0,
            "event_id": event_id, "source_kind": kind}


@dataclass
class Built:
    windows: pd.DataFrame
    clips: list[Clip]
    clean_rows: list[dict]
    dropped_test: list[str]
    no_cache: list[str]
    no_frames: Counter


def build(events: list[dict], fights: dict[str, dict], items: list[dict], cfg: dict,
          coverage_fn=None) -> Built:
    """events.csv rows + fights.csv + queue items -> windows. coverage_fn(fid) returns
    (frame times, fighter frame times) sorted arrays, or None when the fight has no cache."""
    d, lab = cfg["dataset"], cfg["labeling"]
    split_of = assign_splits({e["fight_id"] for e in events}, fights)
    dropped_test = sorted({f for f, s in split_of.items() if s == "test"})
    events = [e for e in events if split_of[e["fight_id"]] != "test"]
    items = [it for it in items if it["fight_id"] in fights
             and fights[it["fight_id"]]["split"].strip() != "test"]
    clips = reviewed_clips(items, events, fights, lab["pad_before_s"], lab["pad_after_s"])
    clean_rows = apply_clean_rows(clips, events, d["note_categories"])

    rows, no_cache, no_frames = [], [], Counter()
    for fid in sorted({e["fight_id"] for e in events}):
        fe = [e for e in events if e["fight_id"] == fid]
        ws = fight_windows(fid, split_of[fid], fe, clips, clean_rows, d, _window(fights[fid]))
        cov = coverage_fn(fid) if coverage_fn else None
        if coverage_fn and cov is None:
            no_cache.append(fid)
            continue
        for w in ws:
            if cov is None:
                w["coverage"] = float("nan")
            else:
                ft, fig = cov
                n = np.searchsorted(ft, w["t1"]) - np.searchsorted(ft, w["t0"])
                if n == 0:
                    no_frames[fid] += 1
                    continue
                k = np.searchsorted(fig, w["t1"]) - np.searchsorted(fig, w["t0"])
                w["coverage"] = round(float(k / n), 3)
            rows.append(w)
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "window_id"])
    df = df.sort_values(["fight_id", "t0", "label"]).reset_index(drop=True)
    df.insert(0, "window_id", [f"{f}_{int(round(t * 100)):07d}_{lab_}" for f, t, lab_
                               in zip(df["fight_id"], df["t0"], df["label"])])
    assert not (df["split"] == "test").any(), "a test fight got into the dataset"
    assert df.groupby("fight_id")["split"].nunique().le(1).all(), "a fight in two splits"
    return Built(df, clips, clean_rows, dropped_test, no_cache, no_frames)


# ---------- audit ----------
def audit(events: list[dict], known: list[dict], clips: list[Clip], d: dict) -> list[str]:
    """Label checks re-run on every build (the 2.3 audit, automated)."""
    warn = []
    seen: dict[tuple, str] = {}
    lo, hi = d["odd_length_s"]
    for e in events:
        s, t = float(e["start_s"]), float(e["end_s"])
        key = (e["fight_id"], e["foul_type"], s, t, e["offender"])
        if key in seen:
            warn.append(f"duplicate: {e['event_id']} = {seen[key]} ({e['fight_id']} {fmt(s)})")
        seen.setdefault(key, e["event_id"])
        if e["foul_type"] in ("eye_poke", "low_blow", "open_hand") and not lo <= t - s <= hi:
            warn.append(f"odd length {t - s:.2f} s: {e['event_id']} {e['foul_type']} "
                        f"({e['fight_id']} {fmt(s)})")
        if e["foul_type"] == "clean" and (e["offender"] != "none" or e["contact"] != "no"):
            warn.append(f"clean with offender/contact: {e['event_id']} ({e['fight_id']} {fmt(s)})")
        # a clean row is only meaningful inside the clip it was reviewed in (a foul found by
        # widening sits outside its clip on purpose)
        if e["foul_type"] == "clean" and e["source"] == "mined" and not any(
                c.fight_id == e["fight_id"] and s <= c.b and t >= c.a for c in clips):
            warn.append(f"clean row outside every reviewed clip: {e['event_id']} ({e['fight_id']} {fmt(s)})")
    labeled = {e["fight_id"] for e in events}
    fouls = [e for e in events if e["foul_type"] in FOULS]
    tol = d["known_match_s"]

    def near(kf, e):
        return (e["fight_id"] == kf["fight_id"] and e["foul_type"] == kf["foul_type"]
                and float(e["start_s"]) - tol <= float(kf["live_s"]) <= float(e["end_s"]) + tol)

    for kf in known:
        if kf["fight_id"] in labeled and not any(near(kf, e) for e in fouls):
            warn.append(f"known foul without a label: {kf['fight_id']} {fmt(float(kf['live_s']))} "
                        f"{kf['foul_type']}")
    for e in fouls:
        if not any(near(kf, e) for kf in known):
            warn.append(f"foul label not in known_fouls.csv: {e['event_id']} {e['foul_type']} "
                        f"({e['fight_id']} {fmt(float(e['start_s']))})")
    return warn
