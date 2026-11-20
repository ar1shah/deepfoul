"""outputs/renders/1.3/report.md: per-clip identity stats, Ari's 1.2 issues, profile warnings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from deepfoul.io.previews import fmt_ts
from deepfoul.perception.identity import iou

JUMP_IOU = 0.3   # a corner label moving to a box with IoU below this vs last frame = a jump


def label_jumps(ident: pd.DataFrame, labels=("red", "blue")) -> pd.DataFrame:
    """Frames where a corner label moves to a different-looking box within a shot."""
    rows = []
    for lab in labels:
        d = ident[ident["label"] == lab].sort_values("frame_idx")
        prev = None
        for r in d.itertuples():
            box = (r.x1, r.y1, r.x2, r.y2)
            if (prev is not None and prev[0] == r.frame_idx - 1 and prev[1] == r.shot_id
                    and iou(prev[2], box) < JUMP_IOU):
                rows.append({"frame_idx": r.frame_idx, "t": r.t, "label": lab})
            prev = (r.frame_idx, r.shot_id, box)
    return pd.DataFrame(rows, columns=["frame_idx", "t", "label"])


def frame_table(tracks: pd.DataFrame, ident: pd.DataFrame) -> pd.DataFrame:
    """One row per frame: time, shot, guard, and how many boxes carry each label."""
    fr = tracks.groupby("frame_idx").agg(t=("t", "first"), shot_id=("shot_id", "first"),
                                         guard=("guard", "first"))
    for lab in ("red", "blue", "ref", "merged", "none"):
        if ident.empty:
            fr[lab] = 0
            continue
        has = ident[ident["label"] == lab].groupby("frame_idx").size()
        fr[lab] = has.reindex(fr.index, fill_value=0)
    # possible missed ref: both fighters found, no ref, and an unlabelled box with clothed legs
    if ident.empty:
        fr["ref_missed"] = 0
    else:
        miss = ident[(ident["label"] == "none") & (ident["clothed"].fillna(0) >= 0.5)]
        cand = miss.groupby("frame_idx").size().reindex(fr.index, fill_value=0)
        fr["ref_missed"] = ((cand > 0) & (fr["ref"] == 0) & (fr["red"] > 0)
                            & (fr["blue"] > 0)).astype(int)
    return fr


def clip_identity_summary(clip_id, fight_id, start_s, end_s, note, tracks, ident, fps, timing,
                          render) -> dict:
    fr = frame_table(tracks, ident)
    jumps = (label_jumps(ident) if not ident.empty
             else pd.DataFrame(columns=["frame_idx", "t", "label"]))
    n = len(fr)
    both = (fr["red"] > 0) & (fr["blue"] > 0)
    shots = []
    for shot, g in fr.groupby("shot_id", sort=True):
        js = jumps[jumps["frame_idx"].isin(g.index)]
        shots.append({"shot_id": int(shot), "start_t": g["t"].min(), "end_t": g["t"].max(),
                      "frames": len(g), "both": ((g["red"] > 0) & (g["blue"] > 0)).mean() * 100,
                      "merged": int((g["merged"] > 0).sum()), "ref": int((g["ref"] > 0).sum()),
                      "ref_missed": int(g["ref_missed"].sum()), "jumps": len(js)})
    return {"clip_id": clip_id, "fight_id": fight_id, "start_s": start_s, "end_s": end_s,
            "note": note, "render": render, "frames": n, "fps": fps,
            "both_pct": both.mean() * 100 if n else 0.0,
            "both_or_merged_pct": (both | (fr["merged"] > 0)).mean() * 100 if n else 0.0,
            "flips": len(jumps), "ref_frames": int((fr["ref"] > 0).sum()),
            "ref_missed": int(fr["ref_missed"].sum()),
            "merged_frames": int((fr["merged"] > 0).sum()),
            "none_boxes": int(fr["none"].sum()), "shots": shots, "timing": timing,
            "frames_df": fr, "jumps_df": jumps}


def issue_rows(summaries: list[dict], checks: list[dict]) -> list[str]:
    by = {s["clip_id"]: s for s in summaries}
    out = []
    for c in checks:
        s = by.get(c["clip"])
        if s is None:
            continue
        fr, js = s["frames_df"], s["jumps_df"]
        w = fr[(fr["t"] >= c["t"] - 1.5) & (fr["t"] < c["t"] + 1.5)]
        jw = js[(js["t"] >= c["t"] - 1.5) & (js["t"] < c["t"] + 1.5)]
        k = max(len(w), 1)

        def pct(col):
            return (w[col] > 0).sum() * 100 / k

        out.append(f"| #{c['n']} | {c['clip']} | {fmt_ts(c['t'])} | {c['note']} | "
                   f"{pct('red'):.0f}% / {pct('blue'):.0f}% | {pct('merged'):.0f}% | "
                   f"{pct('ref'):.0f}% | {len(jw)} |")
    return out


def write_id_report(summaries: list[dict], profiles: dict, fights: dict, path: Path, gpu: str,
                    checks: list[dict] | None = None) -> None:
    lines = ["# Identity check (1.3)", "", f"GPU: {gpu}.", "",
             "Labels come from matching each kept box against the fight's profiles (shorts colors "
             "for red/blue, clothed shins + ref clothing for the ref), smoothed within ByteTrack "
             "tracklets. **both** = frames with a red and a blue box. **jumps** = a corner label "
             "moving to a different box between consecutive frames of one shot (an ID flip). "
             "**ref missed** = both fighters found and an unlabelled box with clothed legs, but no "
             "ref. Renders draw nothing on the ref (use `--show-ref`).", "",
             "| clip | fight | span | frames | both | both or merged | jumps | ref frames | "
             "ref missed | merged | none boxes | det+pose fps |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        name = f"[{s['clip_id']}]({s['render']})" if s["render"] else s["clip_id"]
        t = s["timing"]
        fps = f"{t['frames'] / t['wall_s']:.1f}" if t else "cached"
        lines.append(f"| {name} | {s['fight_id']} | {fmt_ts(s['start_s'])}-{fmt_ts(s['end_s'])} | "
                     f"{s['frames']} | {s['both_pct']:.0f}% | {s['both_or_merged_pct']:.0f}% | "
                     f"{s['flips']} | {s['ref_frames']} | {s['ref_missed']} | "
                     f"{s['merged_frames']} | {s['none_boxes']} | {fps} |")
    if checks:
        lines += ["", "## Ari's 1.2 issues, now", "",
                  "Within +-1.5 s of each time: share of frames with a red / blue box, a merged "
                  "box and the ref, plus corner-label jumps.", "",
                  "| # | clip | time | issue | red / blue | merged | ref | jumps |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += issue_rows(summaries, checks)
    lines += ["", "## Profiles", "",
              "| fight | red shorts | blue shorts | teaching frames (red / blue / ref) | warnings |",
              "|---|---|---|---|---|"]
    for fid, p in profiles.items():
        r = fights[fid]
        lines.append(f"| {fid} | {r['red_shorts']} | {r['blue_shorts']} | "
                     f"{p.n['red']} / {p.n['blue']} / {p.n['ref']} | "
                     f"{'; '.join(p.warnings) or 'none'} |")
    timed = [s["timing"] for s in summaries if s["timing"]]
    if timed:
        fr = sum(t["frames"] for t in timed)
        det, pose, wall = (sum(t[k] for t in timed) for k in ("det_s", "pose_s", "wall_s"))
        boxes = sum(t["pose_boxes"] for t in timed)
        lines += ["", "## Speed", "",
                  f"{fr} frames: detection {fr / det:.1f} fps, pose {fr / pose:.1f} fps "
                  f"({boxes / fr:.1f} kept boxes per frame), whole GPU pass incl. decode and "
                  f"tracking {fr / wall:.1f} fps."]
    for s in summaries:
        lines += ["", f"## {s['clip_id']} · {s['fight_id']} · {s['note']}", "",
                  "| shot | span | frames | both | jumps | ref frames | ref missed | merged |",
                  "|---|---|---|---|---|---|---|---|"]
        for sh in s["shots"]:
            lines.append(f"| {sh['shot_id']} | {fmt_ts(sh['start_t'])}-{fmt_ts(sh['end_t'])} | "
                         f"{sh['frames']} | {sh['both']:.0f}% | {sh['jumps']} | {sh['ref']} | "
                         f"{sh['ref_missed']} | {sh['merged']} |")
        js = s["jumps_df"]
        if len(js):
            lines += ["", "Jumps at: " + ", ".join(f"{fmt_ts(t)} {lab}" for t, lab in
                                                  zip(js["t"], js["label"]))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
