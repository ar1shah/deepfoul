"""Stoppage finder (2.2): signals -> score -> ranked candidates -> label queues.

Queues: data/queues/mined_<fight_id>.csv (ranked, one per fight) and data/queues/mined.csv (all
train fights, by fight then time, plus "extra" items for train known fouls the top N missed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deepfoul.labeling.queues import QUEUE_COLS, load_queue, write_queue
from deepfoul.mining.signals import COLUMNS, SIGNALS, compute_signals


def variant_cfg(mcfg: dict, variant: str) -> dict:
    """'tuned' = the mining config as is; 'untuned' = with mining.untuned overrides."""
    if variant == "tuned":
        return mcfg
    return {**mcfg, **mcfg["untuned"]}


def mine_fight(df: pd.DataFrame, window: tuple[float, float], mcfg: dict, conf: float,
               lookahead_s: float, gap: float) -> tuple[pd.DataFrame, list[dict]]:
    """(signals, ranked candidates) for one fight's cache rows inside its window."""
    sig = compute_signals(df, window[0], window[1], mcfg, conf, lookahead_s)
    return sig, candidates(sig, mcfg["weights"], mcfg, gap, window)


def load_rows(kp_path: Path, start_s: float, end_s: float) -> pd.DataFrame:
    return pd.read_parquet(kp_path, columns=COLUMNS,
                           filters=[("t", ">=", float(start_s)), ("t", "<", float(end_s))])


def score(sig: pd.DataFrame, weights: dict) -> np.ndarray:
    return sum(weights.get(k, 0.0) * sig[k].to_numpy() for k in SIGNALS)


def reason(row: pd.Series, min_signal: float, n: int = 3) -> str:
    """Signal names only (no numbers): the strongest signals at the peak, at most n."""
    parts = sorted(((row[k], k) for k in SIGNALS if row[k] >= min_signal), reverse=True)
    return " · ".join(k for _, k in parts[:n]) or "weak"


def pick_peaks(t: np.ndarray, s: np.ndarray, n: int, gap: float) -> list[int]:
    """Indices of the n best bins, each at least `gap` seconds from every bin picked before."""
    picked: list[int] = []
    for i in np.argsort(-s, kind="stable"):
        if s[i] <= 0 or len(picked) >= n:
            break
        if all(abs(t[i] - t[j]) >= gap for j in picked):
            picked.append(int(i))
    return picked


def item_id(fid: str, t: float) -> str:
    return f"{fid}_{int(round(t * 100)):07d}"        # centiseconds: stable across re-mines


def candidates(sig: pd.DataFrame, weights: dict, mcfg: dict, gap: float,
               window: tuple[float, float]) -> list[dict]:
    """Ranked candidates (rank 1 = best). t_s = peak bin - onset_back_s, kept in the window."""
    s = score(sig, weights)
    t = sig["t"].to_numpy()
    out = []
    for rank, i in enumerate(pick_peaks(t, s, mcfg["per_fight"], gap), start=1):
        ts = float(np.clip(t[i] - mcfg["onset_back_s"], window[0], window[1]))
        out.append({"t_s": round(ts, 2), "score": round(float(s[i]), 3), "rank": rank,
                    "reason": reason(sig.iloc[i], mcfg["min_signal"])})
    return out


def known_ranks(cands: list[dict], fouls: list[dict], pad_before: float,
                pad_after: float) -> list[dict]:
    """For each known foul: the best rank whose clip window [t - pad_before, t + pad_after]
    contains live_s, or None (a miss)."""
    out = []
    for f in fouls:
        live = float(f["live_s"])
        hits = [c["rank"] for c in cands if c["t_s"] - pad_before <= live <= c["t_s"] + pad_after]
        out.append({**f, "rank": min(hits) if hits else None})
    return out


def fight_queue(fid: str, cands: list[dict]) -> list[dict]:
    return [{"item_id": item_id(fid, c["t_s"]), "fight_id": fid, "t_s": c["t_s"],
             "score": c["score"], "reason": c["reason"]} for c in cands]


def extras(fid: str, ranked: list[dict]) -> list[dict]:
    """Queue items for missed known fouls: t = live_s, neutral reason, no score."""
    return [{"item_id": item_id(fid, float(f["live_s"])), "fight_id": fid,
             "t_s": float(f["live_s"]), "score": "", "reason": "extra"}
            for f in ranked if f["rank"] is None]


def build_mined(queue_dir: Path, train_ids: list[str], fouls: list[dict], pad_before: float,
                pad_after: float) -> tuple[list[dict], list[dict]]:
    """(mined.csv items, the extras among them).

    mined.csv = every train fight's ranked mined_<fid>.csv, plus an "extra" item at live_s for
    each of that fight's known fouls no item's clip window covers, sorted by fight (fights.csv
    order) then time. Built from the per-fight files, so re-mining one fight keeps the rest."""
    items, extra = [], []
    for fid in train_ids:
        p = queue_dir / f"mined_{fid}.csv"
        if not p.exists():
            continue
        q = load_queue(p)
        ranked = [{**it, "rank": r} for r, it in enumerate(q, start=1)]
        mine = [f for f in fouls if f["fight_id"] == fid]
        extra += extras(fid, known_ranks(ranked, mine, pad_before, pad_after))
        items += q
    items += extra
    order = {fid: i for i, fid in enumerate(train_ids)}
    items.sort(key=lambda it: (order.get(it["fight_id"], 1 << 30), it["t_s"]))
    return items, extra


def write_fight_queue(queue_dir: Path, fid: str, items: list[dict]) -> Path:
    p = queue_dir / f"mined_{fid}.csv"
    write_queue(items, p)
    return p


__all__ = ["QUEUE_COLS", "build_mined", "candidates", "compute_signals", "extras",
           "fight_queue", "known_ranks", "load_rows", "pick_peaks", "reason", "score",
           "write_fight_queue"]
