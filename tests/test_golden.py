"""The committed golden set is well-formed and every clip sits inside a train fight's window."""

import csv

from deepfoul.config import repo_root
from deepfoul.io.fetch import read_fights
from deepfoul.io.fights import window

EVENT_COLS = ["event_id", "fight_id", "start_s", "end_s", "foul_type", "offender", "contact",
              "ref_stopped", "speed", "source", "notes"]


def _golden():
    path = repo_root() / "labels" / "golden.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def test_golden_schema_matches_events():
    events = (repo_root() / "labels" / "events.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    cols, rows = _golden()
    assert cols == EVENT_COLS == events.split(",")
    assert rows, "golden set is empty"
    assert len({r["event_id"] for r in rows}) == len(rows)


def test_golden_rows_valid():
    fights = {r["fight_id"].strip(): r for r in read_fights(repo_root() / "labels" / "fights.csv")}
    for r in _golden()[1]:
        fight = fights[r["fight_id"]]
        assert fight["split"].strip() == "train", r["event_id"]
        start, end = float(r["start_s"]), float(r["end_s"])
        assert 10 <= end - start <= 30, r["event_id"]
        w0, w1 = window(fight)
        assert (w0 is None or start >= w0) and (w1 is None or end <= w1), r["event_id"]
        assert r["foul_type"] in {"eye_poke", "low_blow", "open_hand", "hard_negative", "clean"}
        assert r["offender"] in {"red", "blue", "none"}
        assert r["contact"] in {"yes", "no", "unsure"}
        assert r["ref_stopped"] in {"yes", "no"}
        assert r["speed"] in {"live", "slowmo"}
        assert r["source"] in {"full_fight", "compilation", "mined"}
        assert (r["offender"] == "none") == (r["foul_type"] in {"clean", "hard_negative"})
