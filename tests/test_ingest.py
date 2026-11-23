"""`deepfoul ingest` (2.3): draft merge and readiness checks (no video, no GPU)."""

from __future__ import annotations

import csv

from deepfoul.config import load_config
from deepfoul.ingest import blockers, merge_draft, read_rows
from deepfoul.io.fights import FIGHT_COLS

COLORS = set(load_config()["identity"]["colors"])


def fight(fid, split="train", **kw):
    row = {c: "" for c in FIGHT_COLS}
    row.update(fight_id=fid, event="UFC 1", date="2025-01-01", red_fighter="A B",
               blue_fighter="C D", red_shorts="red", blue_shorts="black",
               source_url="https://www.youtube.com/watch?v=abcdefghijk", split=split,
               action_start_s="10", action_end_s="900", notes="")
    row.update(kw)
    return row


def write(path, rows, cols=FIGHT_COLS):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, lineterminator="\n")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)


def setup(tmp_path, draft_rows, cols=FIGHT_COLS):
    fights, draft = tmp_path / "fights.csv", tmp_path / "fights_draft.csv"
    write(fights, [fight("ufc100_aa_bb")])
    write(draft, draft_rows, cols)
    return fights, draft, tmp_path / "drafts"


def test_merge_appends_and_archives(tmp_path):
    fights, draft, arch = setup(tmp_path, [fight("ufc101_cc_dd"), fight("ufc102_ee_ff", "val")])
    new, problems = merge_draft(fights, draft, COLORS, arch, today="2026-11-23")
    assert problems == [] and new == ["ufc101_cc_dd", "ufc102_ee_ff"]
    assert [r["fight_id"] for r in read_rows(fights)] == ["ufc100_aa_bb", "ufc101_cc_dd", "ufc102_ee_ff"]
    assert not draft.exists() and (arch / "2026-11-23_fights_draft.csv").exists()


def test_merge_refuses_duplicates_and_bad_rows_writing_nothing(tmp_path):
    for bad, what in ((fight("ufc100_aa_bb"), "already in fights.csv"),
                      (fight("ufc103_gg_hh", "holdout"), "split must be one of"),
                      (fight("ufc104_ii_jj", red_shorts="mauve"), "unknown color"),
                      (fight("ufc105_kk_ll", action_start_s="900", action_end_s="10"), "bad window")):
        fights, draft, arch = setup(tmp_path, [fight("ufc101_cc_dd"), bad])
        before = fights.read_text(encoding="utf-8")
        new, problems = merge_draft(fights, draft, COLORS, arch)
        assert new == [] and any(what in p for p in problems), (what, problems)
        assert fights.read_text(encoding="utf-8") == before and draft.exists()


def test_merge_refuses_a_draft_missing_columns(tmp_path):
    cols = [c for c in FIGHT_COLS if c != "split"]
    fights, draft, arch = setup(tmp_path, [fight("ufc101_cc_dd")], cols)
    new, problems = merge_draft(fights, draft, COLORS, arch)
    assert new == [] and "missing columns: split" in problems[0]


def test_blockers():
    assert blockers(fight("ufc1_a_b"), True) == []
    why = blockers(fight("ufc1_a_b", blue_shorts="", action_start_s=""), False)
    assert len(why) == 3 and why[0] == "no video"
