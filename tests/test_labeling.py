"""Labeling app (2.1): event store, validation, practice queue, done derivation, API."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deepfoul.config import load_config
from deepfoul.io.fetch import read_fights
from deepfoul.io.fights import window
from deepfoul.labeling.app import create_app
from deepfoul.labeling.events_store import (EVENT_COLS, EventStore, default_source,
                                            validate_event)
from deepfoul.labeling.queues import (Skips, build_practice_queue, done_items, golden_t,
                                      is_edit_queue, item_done, load_queue, read_csv, write_queue)

CFG = load_config()
LABELS = CFG["paths"]["labels"]
FIGHTS = read_fights(LABELS / "fights.csv")
GOLDEN = read_csv(LABELS / "golden.csv")
FOULS = read_csv(LABELS / "notes" / "known_fouls.csv")
ALL_IDS = {f["fight_id"] for f in FIGHTS}
WINDOWS = {f["fight_id"]: window(f) for f in FIGHTS}
FID = "ufc258_barber_grasso"


def row(**kw) -> dict:
    base = {"fight_id": FID, "start_s": 600.0, "end_s": 601.0, "foul_type": "eye_poke",
            "offender": "red", "contact": "yes", "ref_stopped": "no", "speed": "live",
            "source": "mined", "notes": ""}
    return {**base, **kw}


# ---------- event store ----------
def test_store_creates_header_and_round_trips(tmp_path):
    p = tmp_path / "events.csv"
    s = EventStore(p)
    assert p.read_text(encoding="utf-8").splitlines()[0] == ",".join(EVENT_COLS)
    a = s.add(row(start_s=600.123, end_s=601.456, notes="two\nlines"))
    assert a["event_id"] == "ev_00001" and a["start_s"] == "600.12" and a["notes"] == "two lines"
    got = EventStore(p).list()  # a fresh store object, as after a crash/restart
    assert got == [a]


def test_store_ids_update_delete(tmp_path):
    s = EventStore(tmp_path / "e.csv")
    ids = [s.add(row())["event_id"] for _ in range(3)]
    assert ids == ["ev_00001", "ev_00002", "ev_00003"]
    s.update("ev_00002", row(foul_type="clean", offender="none", contact="no"))
    assert s.get("ev_00002")["foul_type"] == "clean"
    s.delete("ev_00001")
    assert [r["event_id"] for r in s.list()] == ["ev_00002", "ev_00003"]
    assert s.add(row())["event_id"] == "ev_00004"  # max + 1, never reuses a deleted id
    with pytest.raises(KeyError):
        s.delete("ev_09999")
    assert len(s.list(FID)) == 3 and s.list("ufc299_omalley_vera") == []


def test_store_refuses_bad_header(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("event_id,fight_id\n", encoding="utf-8")
    with pytest.raises(ValueError):
        EventStore(p)


def test_store_appends_after_missing_final_newline(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text(",".join(EVENT_COLS), encoding="utf-8")  # header, no newline
    s = EventStore(p)
    s.add(row())
    s.add(row())
    assert len(s.list()) == 2


def test_committed_events_csv_header_matches():
    EventStore(LABELS / "events.csv")  # only reads: the file exists, so nothing is written


# ---------- validation ----------
def test_validate_ok_and_enums():
    assert validate_event(row(), WINDOWS) == ([], [])
    errs, _ = validate_event(row(foul_type="", contact="maybe", fight_id="nope"), WINDOWS)
    assert any("foul type" in e for e in errs)
    assert any("contact" in e for e in errs)
    assert any("unknown fight_id" in e for e in errs)


@pytest.mark.parametrize("foul", ["eye_poke", "low_blow", "open_hand"])
def test_fouls_need_an_offender(foul):
    errs, _ = validate_event(row(foul_type=foul, offender="none"), WINDOWS)
    assert any("needs an offender" in e for e in errs)
    assert validate_event(row(foul_type=foul, offender="blue"), WINDOWS)[0] == []


def test_clean_and_hard_negative_allow_no_offender():
    assert validate_event(row(foul_type="clean", offender="none", contact="no"), WINDOWS)[0] == []
    assert validate_event(row(foul_type="hard_negative", offender="none", contact="yes"),
                          WINDOWS)[0] == []
    # hard negative starts with contact unset: refused until C is pressed
    errs, _ = validate_event(row(foul_type="hard_negative", offender="none", contact=""), WINDOWS)
    assert any("contact" in e for e in errs)


def test_validate_times_and_window_warning():
    assert any("start <= end" in e for e in validate_event(row(start_s=5, end_s=4), WINDOWS)[0])
    errs, warns = validate_event(row(start_s=1.0, end_s=2.0), WINDOWS)
    w0, _ = WINDOWS[FID]
    assert w0 and w0 > 2 and errs == [] and warns == ["event is outside the fight window"]


def test_default_source():
    assert default_source("comp_eyepokes_01", "queue") == "compilation"
    assert default_source(FID, "queue") == "mined"
    assert default_source(FID, "fight") == "full_fight"


# ---------- practice queue ----------
def _practice(seed=0):
    return build_practice_queue(FIGHTS, GOLDEN, FOULS, ALL_IDS, n=30, seed=seed, gap=10.0)


def test_practice_queue_contents():
    items = _practice()
    assert len(items) == 30
    assert len({it["item_id"] for it in items}) == 30
    splits = {f["fight_id"]: f["split"] for f in FIGHTS}
    assert all(splits[it["fight_id"]] == "train" for it in items)
    for it in items:
        w0, w1 = WINDOWS[it["fight_id"]]
        assert w0 + 4 <= it["t_s"] <= w1 - 4
    reasons = [it["reason"] for it in items]
    for g in GOLDEN:
        assert any(r.startswith(g["event_id"] + " ") for r in reasons)
    # every train known foul not already a golden clip is in (6 at 0.2, 11 after 2.3's finds);
    # random moments fill the rest
    n_golden = sum(not r.startswith("known foul") and r != "random" for r in reasons)
    n_known = sum(r.startswith("known foul") for r in reasons)
    assert n_known >= 6 and n_golden + n_known + sum(r == "random" for r in reasons) == 30
    assert _practice() == items and _practice(seed=1) != items


def test_practice_queue_spacing():
    items = _practice()
    by_fight: dict[str, list[float]] = {}
    for it in items:
        by_fight.setdefault(it["fight_id"], []).append(it["t_s"])
    for ts in by_fight.values():
        ts.sort()
        assert all(b - a >= 10.0 for a, b in zip(ts, ts[1:]))
    for it in (x for x in items if x["reason"] == "random"):
        for f in FOULS:
            if f["fight_id"] == it["fight_id"]:
                for k in ("live_s", "replay_s"):
                    if f[k]:
                        assert abs(it["t_s"] - float(f[k])) >= 10.0
        for g in GOLDEN:
            if g["fight_id"] == it["fight_id"]:
                assert (it["t_s"] <= float(g["start_s"]) - 10 or it["t_s"] >= float(g["end_s"]) + 10)


def test_practice_one_label_closes_one_item():
    """An event covering one item's whole clip window never marks another item done."""
    items = _practice()
    for it in items:
        ev = {"fight_id": it["fight_id"], "start_s": it["t_s"] - 4, "end_s": it["t_s"] + 4}
        assert sum(item_done(x, [ev], 4.0, 4.0) for x in items) == 1


def test_golden_t():
    g = {x["event_id"]: x for x in GOLDEN}
    assert golden_t(g["golden_01"], FOULS) == 489.0   # poke: the known foul's live time
    assert golden_t(g["golden_02"], FOULS) == 302.0   # low blow
    assert golden_t(g["golden_03"], FOULS) == 544.0   # clean: midpoint of 529-559


def test_item_done_uses_clip_window():
    it = {"fight_id": FID, "t_s": 100.0}
    ev = lambda a, b, fid=FID: {"fight_id": fid, "start_s": a, "end_s": b}
    assert item_done(it, [ev(103.5, 103.9)], 4, 4)       # inside [96, 104], not within t+-1
    assert item_done(it, [ev(90, 96.5)], 4, 4)            # overlaps the left edge
    assert not item_done(it, [ev(104.5, 106)], 4, 4)
    assert not item_done(it, [ev(99, 101, fid="ufc299_omalley_vera")], 4, 4)


def test_queue_file_and_skips(tmp_path):
    items = _practice()
    p = tmp_path / "practice.csv"
    write_queue(items, p)
    back = load_queue(p)
    assert [(x["item_id"], x["t_s"]) for x in back] == [(x["item_id"], x["t_s"]) for x in items]
    s = Skips(tmp_path / "practice.skips.json")
    s.set("practice_03")
    assert Skips(tmp_path / "practice.skips.json").ids == {"practice_03"}
    s.set("practice_03", False)
    assert Skips(tmp_path / "practice.skips.json").ids == set()


# ---------- API ----------
@pytest.fixture
def client(tmp_path):
    cfg = {**CFG, "paths": {**CFG["paths"], "raw": tmp_path / "raw"}}
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / f"{FID}.mp4").write_bytes(bytes(range(256)) * 4)
    queue = [{"item_id": "q_01", "fight_id": FID, "t_s": 600.5, "score": "", "reason": "test"}]
    app = create_app(cfg, tmp_path / "events.csv", "queue", queue=queue, queue_name="t",
                     skips_path=tmp_path / "t.skips.json", warm_probe=False)
    return TestClient(app)


def test_api_event_crud_and_status(client):
    s = client.get("/api/session").json()
    assert s["items"][0]["status"] == "todo" and FID in s["fights"]
    r = client.post("/api/events", json={k: v for k, v in row().items() if k != "source"})
    assert r.status_code == 200
    ev = r.json()["event"]
    assert ev["event_id"] == "ev_00001" and ev["source"] == "mined"
    assert client.get("/api/session").json()["items"][0]["status"] == "done"
    r = client.put(f"/api/events/{ev['event_id']}", json=row(foul_type="low_blow"))
    assert r.json()["event"]["foul_type"] == "low_blow"
    assert client.get(f"/api/events?fight_id={FID}").json()[0]["foul_type"] == "low_blow"
    assert client.delete("/api/events/ev_00001").status_code == 200
    assert client.get("/api/events").json() == []
    assert client.delete("/api/events/ev_00001").status_code == 404


def test_api_rejects_invalid_rows(client):
    r = client.post("/api/events", json=row(offender="none"))
    assert r.status_code == 422
    assert any("needs an offender" in e for e in r.json()["detail"]["errors"])
    assert client.get("/api/events").json() == []


def test_api_skip(client):
    assert client.post("/api/skip/q_01", json={"skipped": True}).json()["skipped"] is True
    assert client.get("/api/session").json()["items"][0]["status"] == "skipped"
    assert client.post("/api/skip/nope", json={}).status_code == 404


def test_api_video_range_and_unknown_fight(client):
    r = client.get(f"/video/{FID}", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert r.headers["content-range"] == "bytes 0-99/1024"
    assert client.get("/video/not_a_fight").status_code == 404
    assert client.get("/video/..%2F..%2Fpyproject").status_code == 404
    assert client.get("/").status_code == 200


# ---------- 2.2: nearest-item done rule, widen, skip reasons ----------
def _q(*ts, fid=FID):
    return [{"item_id": f"i{k}", "fight_id": fid, "t_s": t} for k, t in enumerate(ts)]


def _ev(a, b, fid=FID):
    return {"fight_id": fid, "start_s": a, "end_s": b}


def test_done_items_one_label_closes_the_nearest_item():
    items = _q(100.0, 108.0)                              # windows [96,104] and [104,112] touch
    assert done_items(items, [_ev(103.0, 105.0)], 4, 4) == {"i0"}   # 103 is nearer to 100
    assert done_items(items, [_ev(105.0, 106.0)], 4, 4) == {"i1"}
    assert done_items(items, [_ev(103.0, 103.5), _ev(109.0, 110.0)], 4, 4) == {"i0", "i1"}


def test_done_items_widened_label_closes_its_item():
    items = _q(100.0, 120.0)
    ev = [_ev(93.0, 93.5)]                                # outside [96, 104]
    assert done_items(items, ev, 4, 4) == set()
    assert done_items(items, ev, 4, 4, widen_s=4.0, widen={"i0": 1}) == {"i0"}
    # widened far enough to overlap the neighbour: still only the nearest item closes
    ev = [_ev(115.5, 116.5)]                              # in i0 widened x3 and in i1
    assert done_items(items, ev, 4, 4, widen_s=4.0, widen={"i0": 3}) == {"i1"}
    assert done_items(items, [_ev(99, 101, fid="ufc299_omalley_vera")], 4, 4) == set()


def test_skips_keep_a_reason_and_read_the_old_list(tmp_path):
    p = tmp_path / "q.skips.json"
    p.write_text('["a", "b"]', encoding="utf-8")          # 2.1 format
    s = Skips(p)
    assert s.ids == {"a", "b"}
    s.set("c", reason="replay")
    back = Skips(p)
    assert back.data == {"a": "", "b": "", "c": "replay"}
    back.set("a", False)
    assert Skips(p).ids == {"b", "c"}


@pytest.fixture
def qclient(tmp_path):
    cfg = {**CFG, "paths": {**CFG["paths"], "raw": tmp_path / "raw"}}
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / f"{FID}.mp4").write_bytes(b"0" * 64)
    queue = [{"item_id": "q_01", "fight_id": FID, "t_s": 600.0, "score": "", "reason": "guard"},
             {"item_id": "q_02", "fight_id": FID, "t_s": 615.0, "score": "", "reason": "guard"}]
    app = create_app(cfg, tmp_path / "events.csv", "queue", queue=queue, queue_name="t",
                     skips_path=tmp_path / "t.skips.json", widen_path=tmp_path / "t.widen.json",
                     warm_probe=False)
    return TestClient(app)


def test_api_widen_and_done(qclient):
    s = qclient.get("/api/session").json()
    assert s["config"]["scrub_s"] == 0.5 and s["config"]["widen_s"] == 4.0
    assert s["config"]["clean_reasons"][:2] == ["cut", "ref warning"]
    assert [it["widen"] for it in s["items"]] == [0, 0]
    assert qclient.post("/api/widen/q_01", json={"n": 1}).json()["widen"] == 1
    assert qclient.post("/api/widen/q_01", json={"n": -1}).status_code == 422
    assert qclient.post("/api/widen/nope", json={"n": 1}).status_code == 404
    r = qclient.post("/api/events", json=row(start_s=593.0, end_s=593.5, foul_type="clean",
                                             offender="none", contact="no"))
    assert r.status_code == 200
    st = {it["item_id"]: it for it in qclient.get("/api/session").json()["items"]}
    assert st["q_01"]["widen"] == 1
    assert (st["q_01"]["status"], st["q_02"]["status"]) == ("done", "todo")


def test_api_skip_reason(qclient):
    r = qclient.post("/api/skip/q_02", json={"skipped": True, "reason": "replay"}).json()
    assert r == {"item_id": "q_02", "skipped": True, "reason": "replay"}
    it = qclient.get("/api/session").json()["items"][1]
    assert (it["status"], it["skip_reason"]) == ("skipped", "replay")
    assert qclient.get("/api/events").json() == []        # nothing written to events.csv


# ---------- edit queues (2.3 relabel) ----------
def test_edit_queue_file_round_trip(tmp_path):
    p = tmp_path / "relabel.csv"
    write_queue([{"item_id": "r_01", "fight_id": FID, "t_s": 600.0, "score": "",
                  "reason": "dup -> delete one", "event_ids": ["ev_00001", "ev_00002"]}], p)
    back = load_queue(p)
    assert back[0]["event_ids"] == ["ev_00001", "ev_00002"] and is_edit_queue(back)
    assert not is_edit_queue(_q(600.0))


@pytest.fixture
def eclient(tmp_path):
    cfg = {**CFG, "paths": {**CFG["paths"], "raw": tmp_path / "raw"}}
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / f"{FID}.mp4").write_bytes(b"0" * 64)
    store = EventStore(tmp_path / "events.csv")
    a = store.add(row(start_s=600.0, end_s=601.0, foul_type="clean", offender="none", contact="no"))
    b = store.add(row(start_s=600.0, end_s=601.0, foul_type="clean", offender="none", contact="no"))
    c = store.add(row(start_s=640.0, end_s=640.0, foul_type="clean", offender="none", contact="no"))
    queue = [{"item_id": "r_01", "fight_id": FID, "t_s": 600.5, "score": "", "reason": "dup",
              "event_ids": [a["event_id"], b["event_id"]]},
             {"item_id": "r_02", "fight_id": FID, "t_s": 635.0, "score": "", "reason": "outside",
              "event_ids": [c["event_id"]]}]
    app = create_app(cfg, tmp_path / "events.csv", "queue", queue=queue, queue_name="relabel",
                     skips_path=tmp_path / "r.skips.json", widen_path=tmp_path / "r.widen.json",
                     done_path=tmp_path / "r.done.json", warm_probe=False)
    return TestClient(app), (a, b, c), tmp_path


def test_edit_queue_todo_despite_labels_then_done(eclient):
    client, (a, b, c), tmp = eclient
    s = client.get("/api/session").json()
    assert s["edit_queue"] is True
    st = {it["item_id"]: it for it in s["items"]}
    # both items overlap labels, but an edit queue isn't closed by overlap
    assert (st["r_01"]["status"], st["r_02"]["status"]) == ("todo", "todo")
    assert st["r_01"]["event_ids"] == [a["event_id"], b["event_id"]]
    assert st["r_02"]["events_span"] == [640.0, 640.0]      # the page widens its clip to this
    # delete the duplicate, then mark the item done (what the page does after a delete)
    assert client.delete(f"/api/events/{b['event_id']}").status_code == 200
    assert client.post("/api/done/r_01", json={"done": True}).json()["done"] is True
    # a PUT on the other item, then done
    r = client.put(f"/api/events/{c['event_id']}",
                   json=row(start_s=634.5, end_s=635.5, foul_type="clean", offender="none",
                            contact="no"))
    assert r.status_code == 200
    client.post("/api/done/r_02", json={})
    st = {it["item_id"]: it for it in client.get("/api/session").json()["items"]}
    assert (st["r_01"]["status"], st["r_02"]["status"]) == ("done", "done")
    assert st["r_01"]["events_span"] == [600.0, 601.0]       # only the remaining event
    assert (tmp / "r.done.json").is_file()
    assert client.post("/api/done/nope", json={}).status_code == 404


def test_done_endpoint_only_for_edit_queues(qclient):
    assert qclient.get("/api/session").json()["edit_queue"] is False
    assert qclient.post("/api/done/q_01", json={}).status_code == 404


def test_delete_then_undo_restores_the_row(eclient):
    # undo of a delete = POST the same values back (the page does this); new id, same fields
    client, (a, b, c), _ = eclient
    gone = client.delete(f"/api/events/{a['event_id']}").json()["event"]
    back = client.post("/api/events", json={k: v for k, v in gone.items() if k != "event_id"}).json()
    ev = back["event"]
    assert ev["event_id"] != a["event_id"]
    assert {k: ev[k] for k in EVENT_COLS if k != "event_id"} == \
           {k: a[k] for k in EVENT_COLS if k != "event_id"}


def test_quick_reasons_include_foul_timeout():
    assert CFG["labeling"]["clean_reasons"][6] == "foul timeout"
