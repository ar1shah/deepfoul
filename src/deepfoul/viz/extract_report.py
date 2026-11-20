"""outputs/extract_report.md: one row per fight of an `extract` run."""

from __future__ import annotations

from pathlib import Path


def fmt_dur(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _pct(v) -> str:
    return "-" if v is None else f"{v:.0f}%"


def write_extract_report(results: list[dict], path: Path, gpu: str, total_s: float,
                         check_pct: float) -> None:
    lines = ["# Extract report (1.4)", "", f"GPU: {gpu}. Run time so far: {fmt_dur(total_s)}.",
             "",
             "Columns: **both** = frames with a red and a blue box. **both or merged** also "
             "counts frames where the two fighters share one `merged` box (clinch). **visible** = "
             "frames with at least one fighter (red, blue or merged); **b/m of visible** = both "
             "or merged among those, so between-round breaks, crowd and close-up shots with no "
             f"fighter don't count against it. **check** = both or merged under {check_pct:.0f}%."
             " Time and fps are for this run's GPU pass (a resumed fight only counts the parts "
             "done this run).", "",
             "| fight | split | status | time | frames (stride) | fps | both | both or merged | "
             "visible | b/m of visible | ref | profile red/blue/ref | warnings | size | check |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r.get("summary") or {}
        prof = r.get("profile_n")
        warn = "; ".join(r.get("warnings") or []) or ("-" if r["status"] != "failed" else
                                                     r.get("error", "see log"))
        fps = (f"{r['run_frames'] / r['run_s']:.1f}" if r.get("run_s") and r.get("run_frames")
               else "-")
        bom = s.get("both_or_merged_pct")
        check = "**check**" if bom is not None and bom < check_pct else ""
        frames = (f"{s['frames']} ({r['stride']})" if s.get("frames") else
                  (f"~{r['frames_est']} est ({r.get('stride', '?')})" if r.get("frames_est") else "-"))
        prof_s = "/".join(str(prof[k]) for k in ("red", "blue", "ref")) if prof else "-"
        size = f"{r['size_mb']:.0f} MB" if r.get("size_mb") is not None else "-"
        cells = [r["fight_id"], r["split"], r["status"], fmt_dur(r.get("time_s")), frames, fps,
                 _pct(s.get("both_pct")), _pct(bom), _pct(s.get("any_fighter_pct")),
                 _pct(s.get("bom_of_visible_pct")), _pct(s.get("ref_pct")), prof_s, warn, size,
                 check]
        lines.append("| " + " | ".join(cells) + " |")
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        lines += ["", "## Failed", ""] + [f"- {r['fight_id']}: {r.get('error', '')} "
                                          f"(log: `{r.get('log', '')}`)" for r in failed]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
