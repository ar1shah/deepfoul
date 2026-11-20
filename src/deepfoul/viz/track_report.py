"""Per-clip / per-shot tracking stats and outputs/renders/1.2/report.md."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from deepfoul.io.previews import fmt_ts
from deepfoul.perception.filters import RULES

LONG_FRAC = 0.8        # a track covering >= this fraction of its shot's tracked frames is "full"
SHORT_S = 1.0          # tracks shorter than this are "short"
LATE_START_S = 0.5     # a track first seen this long after its shot's first tracked frame is "late"


def shot_stats(df: pd.DataFrame, meta: list[tuple], fps: float) -> list[dict]:
    """One dict per shot that has tracked (non-guard) frames in the clip."""
    frames = pd.DataFrame(meta, columns=["frame_idx", "t", "shot_id", "guard"])
    live = frames[~frames["guard"] & frames["shot_id"].notna()]
    out = []
    for shot, fr in live.groupby("shot_id", sort=True):
        n = len(fr)
        d = df[(df["shot_id"] == shot) & df["frame_idx"].isin(fr["frame_idx"])]
        tracked = d[d["track_id"] != ""]
        per_track = tracked.groupby("track_id")["frame_idx"].agg(["count", "min"])
        first = fr["frame_idx"].min()
        out.append({
            "shot_id": int(shot), "start_t": fr["t"].min(), "end_t": fr["t"].max(),
            "frames": n, "dur_s": n / fps,
            "kept_per_frame": d["kept"].sum() / n,
            "tracks": len(per_track),
            "full": int((per_track["count"] >= LONG_FRAC * n).sum()),
            "short": int((per_track["count"] < SHORT_S * fps).sum()),
            "late": int((per_track["min"] > first + LATE_START_S * fps).sum()),
            "track_frames": per_track["count"].sort_values(ascending=False).tolist(),
        })
    return out


def drop_counts(df: pd.DataFrame) -> dict:
    """Dropped box-frames per rule (a box can fail several), plus boxes dropped ONLY by 'high'."""
    dropped = df[~df["kept"]]["drop_reason"]
    c = Counter(r for reasons in dropped for r in reasons.split("+"))
    return {**{r: c.get(r, 0) for r in RULES}, "only_high": int((dropped == "high").sum()),
            "total": len(dropped)}


def clip_summary(clip_id: str, fight_id: str, start_s: float, end_s: float, note: str,
                 df: pd.DataFrame, meta: list[tuple], fps: float, det_s: float,
                 render: str | None) -> dict:
    return {"clip_id": clip_id, "fight_id": fight_id, "start_s": start_s, "end_s": end_s,
            "note": note, "frames": len(meta), "det_fps": len(meta) / det_s if det_s else 0.0,
            "shots": shot_stats(df, meta, fps), "drops": drop_counts(df), "render": render,
            "kept_per_frame": df["kept"].sum() / max(len(meta), 1)}


def write_report(summaries: list[dict], path: Path, gpu: str, providers: list[str]) -> None:
    lines = [
        "# Tracking check (1.2)", "",
        f"GPU: {gpu}. Detector providers: {', '.join(providers)}.", "",
        "Per shot, only frames fed to the tracker count (guard frames at cuts are skipped).",
        f"**full** = tracks covering >= {LONG_FRAC:.0%} of the shot, **short** = tracks under "
        f"{SHORT_S:.0f} s, **late** = tracks first seen > {LATE_START_S} s into the shot "
        "(a proxy for ID switches or people entering). Ideal stand-up shot: 3 tracks, 3 full, "
        "0 short, 0 late.", "",
        "| clip | fight | span | frames | det fps | kept boxes/frame | shots | tracks | full | short | late | dropped (small/rel/high/edge/max) | only high |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        sh, d = s["shots"], s["drops"]
        tot = lambda k: sum(x[k] for x in sh)  # noqa: E731
        name = f"[{s['clip_id']}]({s['render']})" if s["render"] else s["clip_id"]
        lines.append(
            f"| {name} | {s['fight_id']} | {fmt_ts(s['start_s'])}-{fmt_ts(s['end_s'])} | "
            f"{s['frames']} | {s['det_fps']:.0f} | {s['kept_per_frame']:.1f} | {len(sh)} | "
            f"{tot('tracks')} | {tot('full')} | {tot('short')} | {tot('late')} | "
            f"{d['small']}/{d['rel_small']}/{d['high']}/{d['edge']}/{d['max_people']} | {d['only_high']} |")
    for s in summaries:
        lines += ["", f"## {s['clip_id']} · {s['fight_id']} · {s['note']}", "",
                  "| shot | span | dur | kept/frame | tracks | full | short | late | frames per track |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for x in s["shots"]:
            lines.append(
                f"| {x['shot_id']} | {fmt_ts(x['start_t'])}-{fmt_ts(x['end_t'])} | "
                f"{x['dur_s']:.1f}s | {x['kept_per_frame']:.1f} | {x['tracks']} | {x['full']} | "
                f"{x['short']} | {x['late']} | {', '.join(map(str, x['track_frames']))} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
