"""Local labeling app (2.1): FastAPI + one plain HTML/JS page. Bind to 127.0.0.1 only.

The browser streams the raw video straight from data/raw/ with HTTP Range requests, so every time
it reports is original-file time and nothing is cut or re-encoded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from deepfoul.io.fetch import read_fights
from deepfoul.io.fights import window
from deepfoul.labeling.events_store import (EventStore, default_source, validate_event)
from deepfoul.labeling.queues import EDIT_COL, LocalMap, Skips, done_items, is_edit_queue

STATIC = Path(__file__).parent / "static"


def create_app(cfg: dict, events_path: Path, mode: str, queue: list[dict] | None = None,
               queue_name: str | None = None, fight_id: str | None = None,
               skips_path: Path | None = None, widen_path: Path | None = None,
               done_path: Path | None = None, warm_probe: bool = True) -> FastAPI:
    """mode 'queue' (queue + queue_name) or 'fight' (fight_id). An edit queue (items with
    event_ids) keeps its done items in done_path instead of deriving them from overlap."""
    lab = cfg["labeling"]
    fights = {f["fight_id"].strip(): f for f in read_fights(cfg["paths"]["labels"] / "fights.csv")}
    windows = {fid: window(f) for fid, f in fights.items()}
    raw = cfg["paths"]["raw"]
    store = EventStore(events_path)
    skips = Skips(skips_path) if skips_path else None
    widen = LocalMap(widen_path) if widen_path else None
    queue = queue or []
    edit = is_edit_queue(queue)
    if edit and done_path is None:
        raise ValueError("an edit queue needs done_path")
    done_map = LocalMap(done_path) if edit else None
    app = FastAPI(title="DeepFoul labeling", docs_url=None, redoc_url=None)

    def video_path(fid: str) -> Path:
        if fid not in fights:
            raise HTTPException(404, f"unknown fight_id {fid!r}")
        p = raw / f"{fid}.mp4"
        if not p.is_file():
            raise HTTPException(404, f"{fid}: no video in data/raw, run `deepfoul fetch`")
        return p

    @lru_cache(maxsize=None)
    def probe(fid: str) -> dict:
        from deepfoul.io.probe import ffprobe
        from deepfoul.io.video import video_info
        p = video_path(fid)
        streams = ffprobe(p).get("streams", [])
        codec = {s["codec_type"]: s.get("codec_name") for s in streams if s.get("codec_type")}
        info = video_info(p)
        return {"fps": info["fps"], "duration_s": info["duration_s"],
                "width": info["width"], "height": info["height"],
                "video_codec": codec.get("video"), "audio_codec": codec.get("audio")}

    def fight_info(fid: str) -> dict:
        f = fights[fid]
        w0, w1 = windows[fid]
        return {"fight_id": fid, "event": f.get("event", ""), "split": f["split"],
                "red_fighter": f["red_fighter"], "blue_fighter": f["blue_fighter"],
                "red_shorts": f["red_shorts"], "blue_shorts": f["blue_shorts"],
                "window": [w0, w1], "has_video": (raw / f"{fid}.mp4").is_file()}

    def items_with_status() -> list[dict]:
        wid = widen.data if widen else {}
        events = store.list()
        if edit:
            done = set(done_map.data)
            by_id = {e["event_id"]: e for e in events}
        else:
            done = done_items(queue, events, lab["pad_before_s"], lab["pad_after_s"],
                              lab["widen_s"], wid)
        out = []
        for it in queue:
            iid = it["item_id"]
            status = ("done" if iid in done
                      else "skipped" if skips and iid in skips.ids else "todo")
            row = {**it, "status": status, "widen": int(wid.get(iid, 0)),
                   "skip_reason": skips.data.get(iid, "") if skips else ""}
            if edit:  # the clip also covers the listed events that still exist
                spans = [(float(by_id[e]["start_s"]), float(by_id[e]["end_s"]))
                         for e in it[EDIT_COL] if e in by_id]
                row["events_span"] = ([min(s for s, _ in spans), max(t for _, t in spans)]
                                      if spans else None)
            out.append(row)
        return out

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/session")
    def session() -> dict:
        ids = [fight_id] if mode == "fight" else sorted({it["fight_id"] for it in queue})
        return {"mode": mode, "queue_name": queue_name, "fight_id": fight_id, "edit_queue": edit,
                "events_path": str(events_path), "items": items_with_status(),
                "fights": {fid: fight_info(fid) for fid in ids if fid in fights},
                "config": {k: lab[k] for k in ("pad_before_s", "pad_after_s", "default_event_pad_s",
                                               "scrub_s", "scrub_big_s", "widen_s",
                                               "clean_reasons", "replay_skip_reason")}}

    @app.get("/api/fight/{fid}")
    def fight(fid: str) -> dict:
        return {**fight_info(video_path(fid).stem), **probe(fid)}

    @app.get("/video/{fid}")
    def video(fid: str) -> FileResponse:
        return FileResponse(video_path(fid), media_type="video/mp4")

    @app.get("/api/events")
    def events(fight_id: str | None = None) -> list[dict]:
        return store.list(fight_id)

    def checked(row: dict) -> tuple[dict, list[str]]:
        row = dict(row)
        if not row.get("source") and row.get("fight_id"):
            row["source"] = default_source(row["fight_id"], mode)
        errors, warnings = validate_event(row, windows)
        if errors:
            raise HTTPException(422, {"errors": errors, "warnings": warnings})
        return row, warnings

    @app.post("/api/events")
    def add(row: dict = Body(...)) -> dict:
        row, warnings = checked(row)
        return {"event": store.add(row), "warnings": warnings}

    @app.put("/api/events/{event_id}")
    def update(event_id: str, row: dict = Body(...)) -> dict:
        row, warnings = checked(row)
        try:
            return {"event": store.update(event_id, row), "warnings": warnings}
        except KeyError:
            raise HTTPException(404, f"no event {event_id}")

    @app.delete("/api/events/{event_id}")
    def delete(event_id: str) -> dict:
        try:
            return {"event": store.delete(event_id)}
        except KeyError:
            raise HTTPException(404, f"no event {event_id}")

    @app.post("/api/skip/{item_id}")
    def skip(item_id: str, body: dict = Body(default={})) -> dict:
        if skips is None or item_id not in {it["item_id"] for it in queue}:
            raise HTTPException(404, f"no queue item {item_id}")
        skips.set(item_id, bool(body.get("skipped", True)), str(body.get("reason", "")))
        return {"item_id": item_id, "skipped": item_id in skips.ids,
                "reason": skips.data.get(item_id, "")}

    @app.post("/api/done/{item_id}")
    def set_done(item_id: str, body: dict = Body(default={})) -> dict:
        """Edit queues only: mark an item fixed (the page calls this after a save or delete)."""
        if done_map is None or item_id not in {it["item_id"] for it in queue}:
            raise HTTPException(404, f"no edit-queue item {item_id}")
        if body.get("done", True):
            done_map.put(item_id, True)
        else:
            done_map.drop(item_id)
        return {"item_id": item_id, "done": item_id in done_map.data}

    @app.post("/api/widen/{item_id}")
    def set_widen(item_id: str, body: dict = Body(default={})) -> dict:
        if widen is None or item_id not in {it["item_id"] for it in queue}:
            raise HTTPException(404, f"no queue item {item_id}")
        n = int(body.get("n", 0))
        if n < 0:
            raise HTTPException(422, "n must be >= 0")
        if n:
            widen.put(item_id, n)
        else:
            widen.drop(item_id)
        return {"item_id": item_id, "widen": n}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def warm() -> None:  # ffprobe takes ~2 s per file; do it before the page asks
        ids = [fight_id] if mode == "fight" else list(dict.fromkeys(it["fight_id"] for it in queue))
        for fid in ids:
            if fid in fights and (raw / f"{fid}.mp4").is_file():
                try:
                    probe(fid)
                except Exception:  # the page shows the error when it asks for this fight
                    pass

    if warm_probe:
        import threading
        threading.Thread(target=warm, daemon=True).start()
    return app
