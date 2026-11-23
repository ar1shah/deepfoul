// DeepFoul labeling page. Plain JS, no build step. Times are seconds of the original video file.
"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video");
const preloadEl = $("preload");
const RATES = [0.25, 0.5, 1, 1.5, 2];
const FOUL_KEYS = { "1": "eye_poke", "2": "low_blow", "3": "open_hand", "4": "hard_negative", "5": "clean" };
const NAMES = { eye_poke: "eye poke", low_blow: "low blow", open_hand: "open hand",
                hard_negative: "hard negative", clean: "clean" };

const S = {
  session: null, cfg: null, items: [], idx: -1,
  fightId: null, fight: null, events: [],
  t: 0,                 // presented frame time (requestVideoFrameCallback mediaTime)
  rateIdx: 2, clipStart: null, clipEnd: null, stoppedAtEnd: false,
  draft: null, undo: [], saved: 0, startedAt: Date.now(),
};

// ---------- helpers ----------
function fmt(t) {
  if (t == null || isNaN(t)) return "--:--.--";
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}
function msg(text, kind = "ok") { const el = $("msg"); el.textContent = text; el.className = `msg ${kind}`; }
async function api(method, url, body) {
  const r = await fetch(url, { method, headers: { "Content-Type": "application/json" },
                               body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = data.detail;
    const text = d && d.errors ? d.errors.join("; ") : (typeof d === "string" ? d : `HTTP ${r.status}`);
    throw new Error(text);
  }
  return data;
}
const fps = () => (S.fight && S.fight.fps) || 30;
const isChromium = () => {
  const brands = (navigator.userAgentData && navigator.userAgentData.brands) || [];
  return brands.some((b) => /Chromium|Google Chrome|Microsoft Edge/.test(b.brand)) ||
         /Chrome\/|Edg\//.test(navigator.userAgent);
};

// ---------- frame clock ----------
function onFrame(_now, meta) {
  S.t = meta.mediaTime;
  if (S.clipEnd != null && !video.paused && S.t >= S.clipEnd && !S.stoppedAtEnd) {
    video.pause();
    S.stoppedAtEnd = true;
  }
  renderClock();
  video.requestVideoFrameCallback(onFrame);
}
video.requestVideoFrameCallback(onFrame);
video.addEventListener("play", renderClock);
video.addEventListener("pause", renderClock);

// ---------- loading ----------
async function loadFight(fid) {
  if (S.fightId === fid) return;
  S.fightId = fid;
  $("video-msg").classList.add("hidden");
  S.fight = await api("GET", `/api/fight/${fid}`).catch((e) => {
    showVideoError(`${fid}: ${e.message}`); return null;
  });
  if (!S.fight) return;
  $("fight-name").textContent = `${fid} · RED ${S.fight.red_fighter} (${S.fight.red_shorts}) vs ` +
                                `BLUE ${S.fight.blue_fighter} (${S.fight.blue_shorts})`;
  $("red-name").textContent = `red · ${S.fight.red_fighter.split(" ").slice(-1)[0]}`;
  $("blue-name").textContent = `blue · ${S.fight.blue_fighter.split(" ").slice(-1)[0]}`;
  await new Promise((resolve) => {
    video.addEventListener("loadedmetadata", resolve, { once: true });
    video.addEventListener("error", resolve, { once: true });
    video.src = `/video/${fid}`;
  });
  video.playbackRate = RATES[S.rateIdx];   // a new src resets the rate; keep the labeler's
  await refreshEvents();
}

function showVideoError(text) {
  const el = $("video-msg"); el.textContent = text; el.classList.remove("hidden");
}
video.addEventListener("error", () => {
  const f = S.fight || {};
  const code = video.error ? video.error.code : "?";
  showVideoError(`Can't play ${S.fightId} (video ${f.video_codec || "?"}, audio ${f.audio_codec || "?"}), ` +
                 `media error ${code}. Use Chrome or Edge.`);
});

async function refreshEvents() {
  S.events = S.fightId ? await api("GET", `/api/events?fight_id=${S.fightId}`) : [];
  renderEvents();
  renderTimeline();
}

function bounds() {
  const dur = video.duration || Infinity;
  if (S.clipStart != null) return [Math.max(0, S.clipStart), Math.min(dur, S.clipEnd)];
  const w = (S.fight && S.fight.window) || [];
  return [Math.max(0, w[0] != null ? w[0] : 0), Math.min(dur, w[1] != null ? w[1] : Infinity)];
}

function seek(t) {
  const [lo, hi] = bounds();
  video.currentTime = Math.min(Math.max(t, lo), hi);
  S.t = video.currentTime;       // until the next presented frame reports mediaTime
  S.stoppedAtEnd = S.clipEnd != null && S.t >= S.clipEnd;
  renderClock();
}

function blankDraft() {
  return { event_id: null, fight_id: S.fightId, start_s: null, end_s: null, foul_type: "",
           offender: "none", contact: "", ref_stopped: "no", speed: "live", notes: "" };
}

async function gotoItem(i, { play = true } = {}) {
  if (i < 0 || i >= S.items.length) {
    msg(i >= S.items.length ? "End of the queue. Nothing left to label." : "Start of the queue.", "warn");
    return;
  }
  S.loading = true;
  S.idx = i;
  const it = S.items[i];
  $("item-reason").textContent = `· loading item ${i + 1}…`;
  try { await loadFight(it.fight_id); } finally { S.loading = false; }
  setClip(it);
  S.draft = blankDraft();
  const listed = editEvents(it);
  if (listed.length) S.draft = draftFrom(listed[0]);
  // the mined reason (signal names) says what to look for; the practice queue's would give
  // its answers away, so it stays hidden there
  const why = S.session.queue_name !== "practice" && it.reason ? ` · ${it.reason}` : "";
  $("item-reason").textContent = `· item ${i + 1}/${S.items.length} at ${fmt(it.t_s)}${why}`;
  seek(S.clipStart);
  if (play) video.play().catch(() => {});
  preloadNext();
  renderAll();
  if (it.event_ids) {
    const gone = it.event_ids.filter((id) => !S.events.some((e) => e.event_id === id));
    msg(listed.length ? `Editing ${listed[0].event_id}` +
        (listed.length > 1 ? ` (also listed: ${listed.slice(1).map((e) => e.event_id).join(", ")})` : "") +
        `. Fix: ${it.reason}` + (gone.length ? ` · already deleted: ${gone.join(", ")}` : "")
        : `Listed events are all deleted. Fix: ${it.reason}`, "warn");
  }
}

// Edit queues (relabel): the item's listed events that still exist, in the listed order.
function editEvents(it) {
  if (!it || !it.event_ids) return [];
  return it.event_ids.map((id) => S.events.find((e) => e.event_id === id)).filter(Boolean);
}
function draftFrom(ev) { return { ...ev, start_s: parseFloat(ev.start_s), end_s: parseFloat(ev.end_s) }; }
async function markDone() {
  if (!S.session.edit_queue || S.idx < 0) return;
  const it = S.items[S.idx];
  await api("POST", `/api/done/${it.item_id}`, { done: true });
  it.status = "done";
}

function setClip(it) {
  const extra = (it.widen || 0) * S.cfg.widen_s;
  S.clipStart = it.t_s - S.cfg.pad_before_s - extra;
  S.clipEnd = it.t_s + S.cfg.pad_after_s + extra;
  if (it.events_span) {  // edit queue: the listed events are always inside the clip
    S.clipStart = Math.min(S.clipStart, it.events_span[0] - S.cfg.default_event_pad_s);
    S.clipEnd = Math.max(S.clipEnd, it.events_span[1] + S.cfg.default_event_pad_s);
  }
}

async function widen() {
  if (S.session.mode !== "queue" || S.idx < 0) return;
  const it = S.items[S.idx];
  const n = (it.widen || 0) + 1;
  try { await api("POST", `/api/widen/${it.item_id}`, { n }); }
  catch (e) { msg(`Not widened: ${e.message}`, "err"); return; }
  it.widen = n;
  setClip(it);
  S.stoppedAtEnd = false;
  renderTimeline();
  msg(`Clip widened by ${n * S.cfg.widen_s} s on each side (W again for more).`, "ok");
}

function quickReason(i) {
  const r = (S.cfg.clean_reasons || [])[i];
  if (!r || !S.draft) return;
  setFoul("clean");
  S.draft.notes = r;
  $("notes").value = r;
  renderDraft();
  msg(`clean · ${r}. Enter saves.`, "ok");
}

function preloadNext() {
  const next = S.items.slice(S.idx + 1).find((it) => it.status === "todo");
  if (!next || next.fight_id === S.fightId) return;
  const src = `/video/${next.fight_id}`;
  if (!preloadEl.src.endsWith(src)) {
    preloadEl.src = src;
    preloadEl.addEventListener("loadedmetadata",
      () => { preloadEl.currentTime = Math.max(0, next.t_s - S.cfg.pad_before_s); }, { once: true });
  }
}

function nextTodo(from) {
  for (let i = from + 1; i < S.items.length; i++) if (S.items[i].status === "todo") return i;
  return S.items.length;
}

// ---------- draft ----------
function effective(d) {
  const pad = S.cfg.default_event_pad_s;
  const center = S.session.mode === "queue" && S.idx >= 0 ? S.items[S.idx].t_s : S.t;
  return { start: d.start_s != null ? d.start_s : center - pad,
           end: d.end_s != null ? d.end_s : center + pad };
}

function setFoul(v) {
  const d = S.draft;
  d.foul_type = v;
  if (v === "clean") { d.offender = "none"; d.contact = "no"; }
  if (v === "hard_negative") { d.offender = "none"; d.contact = ""; }
  renderDraft();
}
function setOffender(v) { S.draft.offender = S.draft.offender === v ? "none" : v; renderDraft(); }
function cycleContact() {
  const order = ["yes", "no", "unsure"];
  S.draft.contact = order[(order.indexOf(S.draft.contact) + 1) % order.length];
  renderDraft();
}
function setStart() {
  S.draft.start_s = +S.t.toFixed(3);
  if (S.draft.end_s != null && S.draft.end_s < S.draft.start_s) S.draft.end_s = S.draft.start_s;
  renderDraft();
}
function setEnd() {
  S.draft.end_s = +S.t.toFixed(3);
  if (S.draft.start_s != null && S.draft.start_s > S.draft.end_s) S.draft.start_s = S.draft.end_s;
  renderDraft();
}

function loadEvent(ev) {
  S.draft = draftFrom(ev);
  seek(S.draft.start_s);
  renderAll();
  msg(`Editing ${ev.event_id}. Enter saves the change.`, "warn");
}

async function save(next) {
  const d = S.draft;
  if (!d || !S.fightId) return;
  d.notes = $("notes").value;
  const { start, end } = effective(d);
  const row = { ...d, fight_id: S.fightId, start_s: +start.toFixed(2), end_s: +end.toFixed(2) };
  delete row.event_id;
  S.saving = true;
  try {
    const res = d.event_id ? await api("PUT", `/api/events/${d.event_id}`, row)
                           : await api("POST", "/api/events", row);
    const ev = res.event;
    S.saved += 1;
    const prev = d.event_id ? S.events.find((x) => x.event_id === d.event_id) : null;
    S.undo.push({ event: ev, idx: S.idx, prev });
    const warn = res.warnings && res.warnings.length ? ` (${res.warnings.join("; ")})` : "";
    msg(`${d.event_id ? "Updated" : "Saved"} ${ev.event_id}: ${NAMES[ev.foul_type]} ` +
        `${ev.offender} ${fmt(+ev.start_s)}-${fmt(+ev.end_s)}${warn}`, warn ? "warn" : "ok");
    if (S.session.mode === "queue" && S.idx >= 0) S.items[S.idx].status = "done";
    await markDone();
    await refreshEvents();
    if (next && S.session.mode === "queue") {
      await gotoItem(nextTodo(S.idx));
      return;
    }
    S.draft = blankDraft();
    renderAll();
  } catch (e) {
    msg(`Not saved: ${e.message}`, "err");
  } finally {
    S.saving = false;
  }
}

async function deleteEvent() {
  const d = S.draft;
  if (!d || !d.event_id) { msg("Nothing to delete: load an event first (click it, or open a relabel item).", "warn"); return; }
  const ev = S.events.find((x) => x.event_id === d.event_id);
  if (!ev) return;
  S.saving = true;
  try {
    await api("DELETE", `/api/events/${ev.event_id}`);
    S.undo.push({ deleted: ev, idx: S.idx });
    await markDone();
    await refreshEvents();
    const rest = editEvents(S.session.mode === "queue" && S.idx >= 0 ? S.items[S.idx] : null);
    S.draft = rest.length ? draftFrom(rest[0]) : blankDraft();
    renderAll();
    msg(`deleted ${ev.event_id}, U to undo` + (rest.length ? ` · now editing ${rest[0].event_id}` : ""), "warn");
  } catch (e) {
    msg(`Not deleted: ${e.message}`, "err");
  } finally {
    S.saving = false;
  }
}

async function undo() {
  const last = S.undo.pop();
  if (!last) { msg("Nothing to undo.", "warn"); return; }
  if (last.deleted) {  // undo a delete: add the row back (it gets a new id)
    let ev;
    try {
      const { event_id, ...row } = last.deleted;
      ev = (await api("POST", "/api/events", row)).event;
    } catch (e) { msg(`Undo failed: ${e.message}`, "err"); return; }
    await refreshEvents();
    S.draft = draftFrom(ev);
    renderAll();
    msg(`Restored ${last.deleted.event_id} as ${ev.event_id}.`, "warn");
    return;
  }
  if (last.prev) {  // undo an edit: put the old values back
    try {
      const { event_id, ...row } = last.prev;
      await api("PUT", `/api/events/${event_id}`, row);
    } catch (e) { msg(`Undo failed: ${e.message}`, "err"); return; }
    await refreshEvents();
    msg(`Restored ${last.prev.event_id} to its values before the edit.`, "warn");
    return;
  }
  try {
    await api("DELETE", `/api/events/${last.event.event_id}`);
  } catch (e) { msg(`Undo failed: ${e.message}`, "err"); return; }
  S.saved -= 1;
  if (S.session.mode === "queue" && last.idx >= 0) {
    const s = await api("GET", "/api/session");
    S.items = s.items;
    await gotoItem(last.idx, { play: false });
  } else {
    await refreshEvents();
  }
  const ev = last.event;
  S.draft = { ...ev, event_id: null, start_s: +ev.start_s, end_s: +ev.end_s };
  seek(S.draft.start_s);
  renderAll();
  msg(`Deleted ${ev.event_id}. Its values are back in the form; fix and press Enter.`, "warn");
}

async function skip(reason = "") {
  if (S.session.mode !== "queue" || S.idx < 0) return;
  const it = S.items[S.idx];
  try { await api("POST", `/api/skip/${it.item_id}`, { skipped: true, reason }); }
  catch (e) { msg(`Not skipped: ${e.message}`, "err"); return; }
  if (it.status === "todo") it.status = "skipped";
  it.skip_reason = reason;
  msg(`Skipped ${it.item_id}${reason ? ` (${reason})` : ""}.`, "warn");
  await gotoItem(nextTodo(S.idx));
}

function togglePlay() {
  if (video.paused) {
    if (S.clipEnd != null && S.t >= S.clipEnd - 1 / fps()) seek(S.clipStart);
    S.stoppedAtEnd = false;
    video.play().catch(() => {});
  } else {
    video.pause();
  }
}
function step(frames) {
  video.pause();
  const f = Math.round(S.t * fps()) + frames;
  seek(Math.max(0, f / fps() + 0.001));
}
function setRate(delta) {
  S.rateIdx = Math.min(RATES.length - 1, Math.max(0, S.rateIdx + delta));
  video.playbackRate = video.defaultPlaybackRate = RATES[S.rateIdx];
  renderClock();
}

// ---------- rendering ----------
function tlRange() {
  if (S.session.mode === "queue" && S.clipStart != null) return [S.clipStart, S.clipEnd];
  const w = (S.fight && S.fight.window) || [];
  return [w[0] != null ? w[0] : 0, w[1] != null ? w[1] : (video.duration || 1)];
}
function pct(t) { const [a, b] = tlRange(); return `${Math.min(100, Math.max(0, (t - a) / (b - a) * 100))}%`; }

function renderClock() {
  $("time").textContent = fmt(S.t);
  $("frame").textContent = `f${Math.round(S.t * fps())}`;
  $("rate").textContent = `${RATES[S.rateIdx]}x`;
  $("playstate").textContent = video.paused ? (S.stoppedAtEnd ? "clip end · Space replays" : "paused") : "";
  $("tl-head").style.left = pct(S.t);
  if (S.draft && (S.draft.start_s == null || S.draft.end_s == null)) renderDraftSpan();
}

function renderDraftSpan() {
  if (!S.draft) return;
  const { start, end } = effective(S.draft);
  const el = $("tl-draft");
  el.style.left = pct(start);
  el.style.width = `calc(${pct(end)} - ${pct(start)})`;
  el.classList.toggle("auto", S.draft.start_s == null || S.draft.end_s == null);
  $("start-s").textContent = fmt(start) + (S.draft.start_s == null ? " (auto)" : "");
  $("end-s").textContent = fmt(end) + (S.draft.end_s == null ? " (auto)" : "");
}

function renderDraft() {
  const d = S.draft;
  if (!d) return;
  $("draft-title").textContent = d.event_id ? `Editing ${d.event_id}` : "New event";
  document.querySelectorAll(".choices").forEach((box) => {
    const f = box.dataset.field;
    box.querySelectorAll("button").forEach((b) => b.classList.toggle("on", d[f] === b.dataset.v));
  });
  if (document.activeElement !== $("notes")) $("notes").value = d.notes || "";
  $("quick").querySelectorAll("button").forEach((b) => b.classList.toggle("on",
    d.foul_type === "clean" && d.notes === (S.cfg.clean_reasons || [])[+b.dataset.i]));
  renderDraftSpan();
}

function renderTimeline() {
  const marks = $("tl-marks");
  marks.innerHTML = "";
  const [a, b] = tlRange();
  for (const ev of S.events) {
    const s = +ev.start_s, e = +ev.end_s;
    if (e < a || s > b) continue;
    const m = document.createElement("div");
    m.className = `tl-mark t-${ev.foul_type}` + (S.draft && S.draft.event_id === ev.event_id ? " editing" : "");
    m.style.left = pct(s);
    m.style.width = `calc(${pct(e)} - ${pct(s)})`;
    m.title = `${ev.event_id} ${NAMES[ev.foul_type]} ${ev.offender} ${fmt(s)}-${fmt(e)}`;
    m.addEventListener("click", (evt) => { evt.stopPropagation(); loadEvent(ev); });
    marks.appendChild(m);
  }
  const cand = $("tl-cand"), base = $("tl-base"), tag = $("tl-widen");
  const it = S.session.mode === "queue" && S.idx >= 0 ? S.items[S.idx] : null;
  if (it) {
    cand.classList.remove("hidden"); cand.style.left = pct(it.t_s);
  } else cand.classList.add("hidden");
  if (it && it.widen) {
    const s0 = it.t_s - S.cfg.pad_before_s, s1 = it.t_s + S.cfg.pad_after_s;
    base.classList.remove("hidden");
    base.style.left = pct(s0);
    base.style.width = `calc(${pct(s1)} - ${pct(s0)})`;
    tag.textContent = `widened +${it.widen * S.cfg.widen_s} s each side`;
  } else {
    base.classList.add("hidden");
    tag.textContent = "";
  }
  $("tl-left").textContent = fmt(a);
  $("tl-right").textContent = fmt(b);
  renderClock();
}

function renderEvents() {
  const ul = $("event-list");
  ul.innerHTML = "";
  if (!S.events.length) { ul.innerHTML = '<li class="muted">none yet</li>'; return; }
  for (const ev of [...S.events].sort((x, y) => x.start_s - y.start_s)) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="dot t-${ev.foul_type}"></span>${fmt(+ev.start_s)} ` +
                   `${NAMES[ev.foul_type]} · ${ev.offender} <span class="muted">${ev.event_id}</span>`;
    if (S.draft && S.draft.event_id === ev.event_id) li.classList.add("current");
    li.addEventListener("click", () => loadEvent(ev));
    ul.appendChild(li);
  }
}

function renderQueue() {
  const ul = $("queue-list");
  if (S.session.mode !== "queue") { $("queue-title").classList.add("hidden"); ul.classList.add("hidden"); return; }
  ul.innerHTML = "";
  S.items.forEach((it, i) => {
    const li = document.createElement("li");
    const why = it.status === "skipped" && it.skip_reason ? ` (${it.skip_reason})` : "";
    li.innerHTML = `<span class="st-${it.status}">${it.status === "done" ? "✓" : it.status === "skipped" ? "–" : "·"}` +
                   ` ${it.item_id}</span> <span class="muted">${it.fight_id.split("_").slice(1).join("-")} ${fmt(it.t_s)}${why}</span>`;
    if (i === S.idx) li.classList.add("current");
    li.addEventListener("click", () => gotoItem(i));
    ul.appendChild(li);
  });
  const done = S.items.filter((x) => x.status === "done").length;
  const skipped = S.items.filter((x) => x.status === "skipped").length;
  $("progress").textContent = `${done}/${S.items.length} done · ${skipped} skipped`;
  const cur = ul.querySelector(".current");
  if (cur) cur.scrollIntoView({ block: "nearest" });
}

function renderPace() {
  const sec = Math.round((Date.now() - S.startedAt) / 1000);
  $("pace").textContent = `this session: ${S.saved} saved in ${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
}
setInterval(renderPace, 1000);

function renderAll() { renderDraft(); renderEvents(); renderTimeline(); renderQueue(); renderPace(); }

// ---------- input ----------
$("timeline").addEventListener("click", (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const [a, b] = tlRange();
  seek(a + (e.clientX - r.left) / r.width * (b - a));
});
document.querySelectorAll(".choices").forEach((box) => {
  box.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || !S.draft) return;
    const f = box.dataset.field;
    if (f === "foul_type") setFoul(b.dataset.v); else { S.draft[f] = b.dataset.v; renderDraft(); }
  });
});
$("save-next").addEventListener("click", () => save(true));
$("save-stay").addEventListener("click", () => save(false));
$("skip").addEventListener("click", () => skip());
$("skip-replay").addEventListener("click", () => skip(S.cfg.replay_skip_reason));
$("quick").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) quickReason(+b.dataset.i);
});
$("delete-ev").addEventListener("click", () => deleteEvent());
$("new-draft").addEventListener("click", () => { S.draft = blankDraft(); renderAll(); msg(""); });
$("help-btn").addEventListener("click", () => $("help").classList.toggle("hidden"));
$("help").addEventListener("click", () => $("help").classList.add("hidden"));
$("notes").addEventListener("input", () => { if (S.draft) S.draft.notes = $("notes").value; });

document.addEventListener("keydown", (e) => {
  const inNotes = document.activeElement === $("notes");
  if (inNotes) {
    if (e.key === "Escape") { $("notes").blur(); e.preventDefault(); }
    else if (e.key === "Enter") { $("notes").blur(); e.preventDefault(); save(!e.shiftKey); }
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  // Punctuation keys by physical position too, so [ ] , . - = work on any keyboard layout.
  const CODES = { BracketLeft: "[", BracketRight: "]", Comma: ",", Period: ".", Minus: "-", Equal: "=" };
  const k = (!e.shiftKey && CODES[e.code]) || e.key;
  if (S.loading || S.saving) {  // a key now would land on the previous item's draft
    if (k !== "?" && k !== "Escape") { e.preventDefault(); msg("Loading the next clip…", "warn"); }
    return;
  }
  let handled = true;
  // quick reasons by physical key: 6 7 8 9, Shift+6 7 8 (Shift+digit types ^ & * on US)
  const QUICK = e.shiftKey ? { Digit6: 4, Digit7: 5, Digit8: 6 } : { Digit6: 0, Digit7: 1, Digit8: 2, Digit9: 3 };
  if (e.code in QUICK) { quickReason(QUICK[e.code]); e.preventDefault(); return; }
  if (e.code === "Digit0" && !e.shiftKey) { skip(S.cfg.replay_skip_reason); e.preventDefault(); return; }
  if (FOUL_KEYS[k]) setFoul(FOUL_KEYS[k]);
  else if (k === "r" || k === "R") setOffender("red");
  else if (k === "b" || k === "B") setOffender("blue");
  else if (k === "c" || k === "C") cycleContact();
  else if (k === "t" || k === "T") { S.draft.ref_stopped = S.draft.ref_stopped === "yes" ? "no" : "yes"; renderDraft(); }
  else if (k === "m" || k === "M") { S.draft.speed = S.draft.speed === "slowmo" ? "live" : "slowmo"; renderDraft(); }
  else if (k === "[") setStart();
  else if (k === "]") setEnd();
  else if (k === " ") togglePlay();
  else if (k === "ArrowLeft") seek(S.t - (e.shiftKey ? S.cfg.scrub_big_s : S.cfg.scrub_s));
  else if (k === "ArrowRight") seek(S.t + (e.shiftKey ? S.cfg.scrub_big_s : S.cfg.scrub_s));
  else if (k === "w" || k === "W") widen();
  else if (k === ",") step(-1);
  else if (k === ".") step(1);
  else if (k === "-") setRate(-1);
  else if (k === "=" || k === "+") setRate(1);
  else if (k === "Enter") save(!e.shiftKey);
  else if (k === "s" || k === "S") skip();
  else if (k === "u" || k === "U") undo();
  else if (k === "Delete" || k === "Backspace") deleteEvent();   // Mac's delete key = Backspace
  else if (k === "n" || k === "N") $("notes").focus();
  else if (k === "?") $("help").classList.toggle("hidden");
  else if (k === "Escape") $("help").classList.add("hidden");
  else handled = false;
  if (handled) e.preventDefault();
});

// ---------- start ----------
async function start() {
  if (!isChromium()) {
    const b = $("banner");
    b.textContent = "This page needs Chrome or Edge: the fight videos are VP9 / AV1 with Opus audio, " +
                    "which other browsers may not play. Open http://localhost:" + location.port + " in Chrome.";
    b.classList.remove("hidden");
  }
  S.session = await api("GET", "/api/session");
  S.cfg = S.session.config;
  S.items = S.session.items;
  $("events-path").textContent = S.session.events_path;
  const QK = ["6", "7", "8", "9", "⇧6", "⇧7", "⇧8"];
  $("quick").innerHTML = (S.cfg.clean_reasons || []).slice(0, QK.length)
    .map((r, i) => `<button data-i="${i}"><kbd>${QK[i]}</kbd> ${r}</button>`).join("");
  $("help-quick").textContent = (S.cfg.clean_reasons || []).slice(0, QK.length)
    .map((r, i) => `${QK[i]} ${r}`).join(", ");
  $("help-scrub").textContent = `${S.cfg.scrub_s} s`;
  $("help-scrub-big").textContent = `${S.cfg.scrub_big_s} s`;
  $("help-widen").textContent = `${S.cfg.widen_s} s`;
  if (S.session.mode !== "queue") {
    $("skip-replay").classList.add("hidden");
    $("skip").classList.add("hidden");
  }
  $("mode").textContent = S.session.mode === "queue" ? `queue: ${S.session.queue_name}` :
                          `full fight: ${S.session.fight_id}`;
  if (S.session.mode === "queue") {
    if (!S.items.length) { msg("The queue is empty.", "warn"); return; }
    const first = S.items.findIndex((it) => it.status === "todo");
    if (first < 0) msg("Every item in this queue is done or skipped. Click one to revisit.", "warn");
    await gotoItem(first < 0 ? 0 : first, { play: first >= 0 });
  } else {
    await loadFight(S.session.fight_id);
    S.clipStart = S.clipEnd = null;
    const w = S.fight && S.fight.window;
    if (w && w[0] != null) seek(w[0]);
    S.draft = blankDraft();
    $("item-reason").textContent = "· full fight";
    renderAll();
  }
}

// Exposed for automated checks: the presented frame's time, and a canvas grab of it.
window.__deepfoul = {
  state: S,
  grab() {
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    return { t: S.t, width: c.width, height: c.height, png: c.toDataURL("image/png") };
  },
  seekAndWait(t) {
    return new Promise((resolve) => {
      video.pause();
      video.addEventListener("seeked", () => {
        video.requestVideoFrameCallback((_n, meta) => resolve(meta.mediaTime));
      }, { once: true });
      seek(t);
    });
  },
};

start().catch((e) => msg(`Failed to start: ${e.message}`, "err"));
