import csv

import pytest

from deepfoul.config import repo_root
from deepfoul.io.fetch import read_fights
from deepfoul.io.fights import parse_mmss, validate_fights, window

GOOD = {"fight_id": "ufc276_munhoz_omalley", "date": "2022-07-02", "split": "train",
        "source_url": "https://www.youtube.com/watch?v=e6DTwSONguQ",
        "action_start_s": "139", "action_end_s": "848"}


@pytest.mark.parametrize("s,expected", [("04:13", 253), ("0:43", 43), ("2:00", 120),
                                        ("26:11", 1571), ("1:02:03", 3723)])
def test_parse_mmss(s, expected):
    assert parse_mmss(s) == expected


@pytest.mark.parametrize("s", ["", "43", "4:61", "a:10", "1:2:3:4"])
def test_parse_mmss_rejects(s):
    with pytest.raises(ValueError):
        parse_mmss(s)


@pytest.mark.parametrize("fid", ["ufc276_munhoz_omalley", "ufcfn20251101_cortesacosta_delija",
                                 "comp_eyepokes_01", "clip_ufc300_gaethje_holloway",
                                 "clip_ufcfn20240316_battle_loosa"])
def test_valid_ids(fid):
    assert validate_fights([{**GOOD, "fight_id": fid}]) == []


@pytest.mark.parametrize("change,needle", [
    ({"fight_id": "UFC276_Munhoz"}, "fight_id format"),
    ({"fight_id": "comp_eyepokes_1"}, "fight_id format"),
    ({"split": "holdout"}, "split"),
    ({"date": "2022-13-01"}, "date"),
    ({"source_url": "https://vimeo.com/123"}, "YouTube"),
    ({"action_start_s": "900"}, "window"),
    ({"action_end_s": "abc"}, "numbers"),
])
def test_validation_errors(change, needle):
    probs = validate_fights([{**GOOD, **change}])
    assert len(probs) == 1 and needle in probs[0]


def test_duplicates_and_blanks():
    blank = {**GOOD, "source_url": "", "action_start_s": "", "action_end_s": "", "date": ""}
    assert validate_fights([blank]) == []
    assert "duplicate" in validate_fights([GOOD, GOOD])[0]


def test_committed_fights_csv_is_valid():
    rows = read_fights(repo_root() / "labels" / "fights.csv")
    assert rows and validate_fights(rows) == []


def test_known_fouls_inside_fight_windows():
    labels = repo_root() / "labels"
    fights = {r["fight_id"]: r for r in read_fights(labels / "fights.csv")}
    with (labels / "notes" / "known_fouls.csv").open(newline="", encoding="utf-8-sig") as f:
        fouls = list(csv.DictReader(f))
    assert fouls
    for foul in fouls:
        start, end = window(fights[foul["fight_id"]])
        live = float(foul["live_s"])
        assert start <= live <= end, foul
        if foul["replay_s"]:
            assert live < float(foul["replay_s"]) <= end, foul
        assert foul["foul_type"] in {"eye_poke", "low_blow"}
        assert {foul["offender"], foul["victim"]} == {"red", "blue"}
        assert foul["ref_paused"] in {"yes", "no"}
        assert foul["visibility"] in {"clear_live", "hard_live"}
