"""outputs/dataset/summary.md for build-dataset (2.3)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from deepfoul.dataset.build import LABELS, Built, fmt


def write_summary(out: Path, b: Built, events: list[dict], warnings: list[str], d: dict) -> str:
    df = b.windows
    L: list[str] = ["# Dataset summary (build-dataset)", ""]
    L += [f"Windows: **{len(df)}** ({d['window_s']} s long; event stride {d['pos_stride_s']} s, "
          f"clean stride {d['neg_stride_s']} s; hard-negative weight {d['hard_neg_weight']}).", ""]
    L += ["**Clean windows come only from reviewed clips:** the 8 s clips of labeled queue items "
          "(today: the mined queue). Unreviewed fight time is never used. The mined clips were "
          "picked by guard drops, so clean examples lean toward clinches, stoppages and hands "
          "coming down, not ordinary exchanges.", ""]

    L += ["## Windows per class per split", "", "| split | " + " | ".join(LABELS) + " | total |",
          "|---|" + "---|" * (len(LABELS) + 1)]
    for split in ("train", "val"):
        s = df[df["split"] == split]
        L.append(f"| {split} | " + " | ".join(str(int((s['label'] == l).sum())) for l in LABELS)
                 + f" | {len(s)} |")
    if b.dropped_test:
        L += ["", f"Test fights with labels, left out on purpose: {', '.join(b.dropped_test)}."]

    L += ["", "## Per fight (windows; events in brackets)", "",
          "| fight | split | " + " | ".join(LABELS) + " | mean coverage |",
          "|---|---|" + "---|" * (len(LABELS) + 1)]
    ev_count = Counter((e["fight_id"], e["foul_type"]) for e in events)
    for fid, s in df.groupby("fight_id"):
        cells = []
        for l in LABELS:
            n = int((s["label"] == l).sum())
            cells.append(f"{n} ({ev_count[(fid, l)]})" if l != "clean" else str(n))
        L.append(f"| {fid} | {s['split'].iloc[0]} | " + " | ".join(cells)
                 + f" | {s['coverage'].mean():.2f} |")
    if b.no_cache:
        L += ["", f"No keypoint cache (no windows): {', '.join(b.no_cache)}."]
    if b.no_frames:
        L += ["", "Windows dropped for having no cached frames: "
              + ", ".join(f"{f} {n}" for f, n in sorted(b.no_frames.items())) + "."]

    used = Counter(r["category"] for r in b.clean_rows)
    uses = {r["category"]: r["use"] for r in b.clean_rows}
    clips = Counter(c.use for c in b.clips)
    L += ["", "## Clean rows by note category", "",
          "| category | rows | what it does |", "|---|---|---|"]
    what = {"clean": "clip gives clean windows", "hard_negative": "clip gives hard-negative "
            "windows", "exclude": "clip gives no windows (not live action / may hold a replay)"}
    for cat, n in used.most_common():
        L.append(f"| {cat} | {n} | {what[uses[cat]]} |")
    L += ["", f"Reviewed clips: {len(b.clips)} ({clips['clean']} clean, {clips['hard_negative']} "
          f"stoppage hard negative, {clips['exclude']} excluded).", ""]

    L += ["### Every clean row with a note, and its category", "",
          "| event | fight | time | note | category |", "|---|---|---|---|---|"]
    for r in b.clean_rows:
        if (r["notes"] or "").strip():
            L.append(f"| {r['event_id']} | {r['fight_id']} | {fmt(float(r['start_s']))} | "
                     f"{r['notes']} | {r['category']} ({r['use']}) |")
    n_plain = sum(1 for r in b.clean_rows if not (r["notes"] or "").strip())
    L += ["", f"Plus {n_plain} clean rows with no note (category clean).", ""]

    L += ["## Labeled events", "", "| event | fight | time | type | offender | windows |",
          "|---|---|---|---|---|---|"]
    per_ev = df[df["source_kind"] == "event"].groupby("event_id").size()
    for e in events:
        if e["foul_type"] != "clean":
            L.append(f"| {e['event_id']} | {e['fight_id']} | {fmt(float(e['start_s']))}-"
                     f"{fmt(float(e['end_s']))} | {e['foul_type']} | {e['offender']} | "
                     f"{int(per_ev.get(e['event_id'], 0))} |")

    L += ["", "## Label audit (re-run on every build)", ""]
    L += [f"- {w}" for w in warnings] if warnings else ["- no warnings"]
    text = "\n".join(L) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def write_window_sheet(raw_dir: Path, windows, out: Path, cols: int = 5,
                       tile_w: int = 384) -> int:
    """One tile per window at its middle frame, captioned label / fight / time. For checking
    by eye that windows sit where the labels say."""
    import cv2
    import numpy as np

    from deepfoul.io.previews import put_label
    from deepfoul.io.video import ffmpeg_frame, video_info

    tile_h = round(tile_w * 9 / 16)
    tiles, info = [], {}
    for w in windows.itertuples():
        video = raw_dir / f"{w.fight_id}.mp4"
        if w.fight_id not in info:
            info[w.fight_id] = video_info(video)
        vi = info[w.fight_id]
        mid = (w.t0 + w.t1) / 2
        img = ffmpeg_frame(video, mid, vi["width"], vi["height"]).copy()
        h = round(img.shape[0] * tile_w / img.shape[1])
        img = cv2.resize(img, (tile_w, min(h, tile_h)))
        tile = np.zeros((tile_h, tile_w, 3), np.uint8)
        tile[:img.shape[0]] = img
        short = "-".join(w.fight_id.split("_")[1:])
        tiles.append(put_label(tile, f"{w.label} {w.offender} {short} {fmt(mid)}", 0.42))
    if not tiles:
        return 0
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return len(windows)
