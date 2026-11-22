"""Mining report (2.2): outputs/mining/report.md, signal plots and top-candidate sheets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deepfoul.io.previews import fmt_ts

PLOT_SIGNALS = ("guard", "pause", "ref", "eye", "groin", "bend", "closeup")


def plot_signals(fid: str, sig: pd.DataFrame, cands: list[dict], fouls: list[dict],
                 onset_back: float, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = sig["t"].to_numpy() / 60
    fig, axes = plt.subplots(len(PLOT_SIGNALS) + 1, 1, figsize=(22, 13), sharex=True)
    for ax, k in zip(axes, PLOT_SIGNALS):
        ax.plot(t, sig[k], lw=0.6)
        ax.set_ylabel(k, rotation=0, ha="right")
        ax.set_ylim(0, 1.05)
    axes[-1].plot(t, sig["hand_head"].clip(upper=2), lw=0.5, color="gray")
    axes[-1].set_ylabel("hand_head", rotation=0, ha="right")
    for ax in axes:
        for f in fouls:
            ax.axvline(float(f["live_s"]) / 60, color="red", lw=1.2, alpha=0.7)
        for c in cands:
            ax.axvline((c["t_s"] + onset_back) / 60, color="green", lw=0.6, alpha=0.5, ls="--")
    axes[0].set_title(f"{fid}: signals (red = known foul, green dashed = top candidates' peaks)")
    axes[-1].set_xlabel("minutes into the video")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=70)
    plt.close(fig)


def write_top_sheet(video: Path, cands: list[dict], at_s: float, out: Path, cols: int = 5,
                    tile_w: int = 384) -> None:
    """One frame per candidate at t_s + at_s (where the stoppage shows), captioned rank/time."""
    import cv2

    from deepfoul.io.previews import put_label
    from deepfoul.io.video import ffmpeg_frame, video_info

    info = video_info(video)
    tiles = []
    for c in sorted(cands, key=lambda c: c["rank"]):
        img = ffmpeg_frame(video, c["t_s"] + at_s, info["width"], info["height"]).copy()
        img = cv2.resize(img, (tile_w, round(img.shape[0] * tile_w / img.shape[1])))
        tiles.append(put_label(img, f"#{c['rank']} {fmt_ts(c['t_s'])}", 0.5))
    if not tiles:
        return
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])


def _rank(r) -> str:
    return "miss" if r is None else str(r)


def write_report(out: Path, results: list[dict], mined_n: int, extra: list[dict],
                 per_fight: int, sec_per_item: float = 20.0) -> None:
    """results: one dict per scored fight: fight_id, split, cands, ranks {variant: [...]}."""
    L = ["# Stoppage finder report (2.2)", ""]
    for split in ("train", "val"):
        rows = [r for r in results if r["split"] == split]
        fouls = [(r, i) for r in rows for i in range(len(r["ranks"]["tuned"]))]
        if not fouls:
            continue
        hit = {v: sum(r["ranks"][v][i]["rank"] is not None for r, i in fouls)
               for v in ("untuned", "tuned")}
        note = " (held out: reported, never tuned on, never queued)" if split == "val" else ""
        L += [f"## Known fouls, {split}{note}", "",
              f"In a top-{per_fight} clip window: **untuned {hit['untuned']}/{len(fouls)}, "
              f"tuned {hit['tuned']}/{len(fouls)}**.", "",
              "| fight | foul | live | untuned rank | tuned rank |", "|---|---|---|---|---|"]
        for r, i in fouls:
            f = r["ranks"]["tuned"][i]
            L.append(f"| {r['fight_id']} | {f['foul_type']} | {fmt_ts(float(f['live_s']))} | "
                     f"{_rank(r['ranks']['untuned'][i]['rank'])} | {_rank(f['rank'])} |")
        L.append("")
    L += ["## Queue", "",
          f"`data/queues/mined.csv`: {mined_n} items ({len(extra)} extra for missed train "
          f"fouls), about {mined_n * sec_per_item / 60:.0f} min at {sec_per_item:.0f} s each.",
          "", "| fight | split | items | queued |", "|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['fight_id']} | {r['split']} | {len(r['cands'])} | "
                 f"{'yes' if r['queued'] else 'no'} |")
    L += ["", "Contact sheets `<fight_id>_top.jpg` show each train candidate at its peak "
          "(queue time + onset_back_s): round breaks (corner split screens) and foul timeouts "
          "(red +m:ss clock) are easy to spot there.", ""]
    for r in results:
        L += [f"## {r['fight_id']} ({r['split']})", "",
              f"![signals]({r['fight_id']}_signals.png)", "",
              "| rank | time | score | reason |", "|---|---|---|---|"]
        for c in r["cands"]:
            L.append(f"| {c['rank']} | {fmt_ts(c['t_s'])} | {c['score']:.2f} | {c['reason']} |")
        L.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
