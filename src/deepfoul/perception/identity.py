"""Red / blue / ref identity from fight-level appearance profiles (1.3).

IDs are not carried across frames. Every kept box is matched against the fight's profiles
(red shorts, blue shorts, ref clothing), so labels survive cuts and re-entries by construction.
ByteTrack tracklets are only used to smooth labels over a short window.

Labels: "red", "blue", "ref", "merged" (one box holding both fighters, e.g. a clinch), "none"
(crowd, crew, cageside officials, or not enough evidence).

Lookahead: smoothing uses frames up to t + smooth_s and a split needs split_frames future frames,
both checked against `lookahead_s` so a live version stays possible.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from deepfoul.perception.appearance import (Appearance, seed_fraction, seed_histogram,
                                            similarity)

SLOTS = ("red", "blue", "ref")


# ---------------------------------------------------------------- profiles


@dataclass
class Profile:
    fight_id: str
    seed_red: np.ndarray
    seed_blue: np.ndarray
    red: np.ndarray | None = None        # refined shorts histograms
    blue: np.ndarray | None = None
    red_torso: np.ndarray | None = None  # refined fighter torso (bare skin, or a top)
    blue_torso: np.ndarray | None = None
    ref_torso: np.ndarray | None = None  # refined ref clothing
    ref_thigh: np.ndarray | None = None
    n: dict = field(default_factory=lambda: {"red": 0, "blue": 0, "ref": 0})
    warnings: list[str] = field(default_factory=list)

    def fighter_hist(self, c: str) -> np.ndarray:
        h = self.red if c == "red" else self.blue
        return h if h is not None else (self.seed_red if c == "red" else self.seed_blue)

    def save(self, path) -> None:
        arrays = {k: getattr(self, k) for k in ("seed_red", "seed_blue", "red", "blue",
                                                 "red_torso", "blue_torso", "ref_torso",
                                                 "ref_thigh")
                  if getattr(self, k) is not None}
        np.savez(path, fight_id=self.fight_id, n=np.array([self.n[s] for s in SLOTS]),
                 warnings=np.array(self.warnings, dtype=object), **arrays)

    @classmethod
    def load(cls, path) -> "Profile":
        z = np.load(path, allow_pickle=True)
        p = cls(str(z["fight_id"]), z["seed_red"], z["seed_blue"])
        for k in ("red", "blue", "red_torso", "blue_torso", "ref_torso", "ref_thigh"):
            if k in z:
                setattr(p, k, z[k])
        p.n = dict(zip(SLOTS, (int(x) for x in z["n"])))
        p.warnings = [str(w) for w in z["warnings"]]
        return p


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def contained(inner, outer) -> float:
    """Share of `inner`'s area that lies inside `outer`."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / area if area > 0 else 0.0


def _mean(hists: list[np.ndarray]) -> np.ndarray | None:
    if not hists:
        return None
    m = np.mean(hists, 0)
    return m / m.sum()


def build_profile(fight_id: str, red_name: str, blue_name: str,
                  samples: list[list[tuple[np.ndarray, Appearance]]], cfg: dict) -> Profile:
    """samples: per sampled frame from the start of the fight, (box, appearance) per kept box.

    Seeds come from the shorts color names. A frame teaches the profiles when exactly one box
    clearly matches red, another clearly matches blue, the two don't overlap and neither has
    clothed shins. A second round re-picks frames with the refined profiles. Boxes with clothed
    shins in those frames teach the ref profile.
    """
    seeds = {c: seed_histogram(n, cfg["colors"], cfg["sat_min"], cfg["v_min"])
             for c, n in (("red", red_name), ("blue", blue_name))}
    prof = Profile(fight_id, seeds["red"], seeds["blue"])
    pc = cfg["profile"]

    def score(app, c, refined):
        if refined:
            return similarity(app.shorts, prof.fighter_hist(c))
        return seed_fraction(app.shorts, seeds[c])

    for refined in (False, True):
        hists = {"red": [], "blue": [], "torso": [], "thigh": [], "red_torso": [],
                 "blue_torso": []}
        margin = pc["refined_margin"] if refined else pc["seed_margin"]
        for frame in samples:
            if not frame:
                continue
            # fighters are the big people in the cage: smaller boxes (crowd, corner, cageside)
            # never teach the profiles
            tallest = max(b[3] - b[1] for b, _ in frame)
            fighters = [(b, a) for b, a in frame if a.shorts is not None
                        and b[3] - b[1] >= pc["min_rel_height"] * tallest
                        and (a.clothed is None or a.clothed < pc["fighter_clothed_max"])]
            picks = {}
            for c, o in (("red", "blue"), ("blue", "red")):
                scored = sorted(((score(a, c, refined) - score(a, o, refined), score(a, c, refined),
                                  k) for k, (b, a) in enumerate(fighters)), reverse=True)
                # the best box must clearly match this color, and clearly beat any runner-up
                if (scored and scored[0][0] >= margin and scored[0][1] >= pc["min_score"]
                        and (len(scored) < 2 or scored[0][0] - scored[1][0] >= margin)):
                    picks[c] = fighters[scored[0][2]]
            if len(picks) != 2 or picks["red"][0] is picks["blue"][0]:
                continue
            (rb, ra), (bb, ba) = picks["red"], picks["blue"]
            if iou(rb, bb) > pc["max_iou"]:
                continue
            hists["red"].append(ra.shorts)
            hists["blue"].append(ba.shorts)
            for c, a in (("red", ra), ("blue", ba)):
                if a.torso is not None:
                    hists[f"{c}_torso"].append(a.torso)
            for b, a in frame:
                if (a.clothed is not None and a.clothed >= pc["ref_clothed_min"]
                        and iou(b, rb) < pc["max_iou"] and iou(b, bb) < pc["max_iou"]):
                    if a.torso is not None:
                        hists["torso"].append(a.torso)
                    if a.thigh is not None:
                        hists["thigh"].append(a.thigh)
        if len(hists["red"]) >= pc["min_samples"]:
            prof.red, prof.blue = _mean(hists["red"]), _mean(hists["blue"])
            prof.red_torso, prof.blue_torso = _mean(hists["red_torso"]), _mean(hists["blue_torso"])
        prof.ref_torso, prof.ref_thigh = _mean(hists["torso"]), _mean(hists["thigh"])
        prof.n = {"red": len(hists["red"]), "blue": len(hists["blue"]),
                  "ref": max(len(hists["torso"]), len(hists["thigh"]))}
    prof.warnings = drift_warnings(prof, pc)
    return prof


def drift_warnings(prof: Profile, pc: dict) -> list[str]:
    """Flag refined profiles that wandered off their color name, or look swapped."""
    out = []
    if prof.red is None:
        return [f"red/blue not refined ({prof.n['red']} good frames < {pc['min_samples']}); "
                "using the color-name seeds only"]
    for c, own, other in (("red", prof.seed_red, prof.seed_blue),
                          ("blue", prof.seed_blue, prof.seed_red)):
        h = prof.fighter_hist(c)
        f_own, f_other = seed_fraction(h, own), seed_fraction(h, other)
        if f_other > f_own:
            out.append(f"{c} profile looks more like the {('blue' if c == 'red' else 'red')} "
                       f"color name ({f_other:.2f} vs {f_own:.2f}): possibly swapped")
        elif f_own < pc["drift_min"]:
            out.append(f"{c} profile drifted from its color name (only {f_own:.2f} of its mass "
                       "in the named color's bins)")
    if prof.n["ref"] == 0:
        out.append("no ref samples at the start of the fight; ref relies on clothed shins only")
    return out


# ---------------------------------------------------------------- per-frame assignment


@dataclass
class BoxObs:
    """What identity needs about one kept box (no pixels)."""
    box: np.ndarray
    app: Appearance


def cue_weights(prof: Profile, ac: dict) -> tuple[float, float]:
    """(shorts weight, torso weight): a floor plus how distinct the two fighters are on it."""
    def distinct(p, q):
        sim = similarity(p, q)
        return 0.0 if sim is None else 1.0 - sim

    lo, gain = ac["cue_weight_floor"], ac["cue_weight_gain"]
    w_s = lo + gain * distinct(prof.fighter_hist("red"), prof.fighter_hist("blue"))
    w_t = (lo + gain * distinct(prof.red_torso, prof.blue_torso)
           if prof.red_torso is not None and prof.blue_torso is not None else 0.0)
    return w_s, w_t


def slot_costs(o: BoxObs, prof: Profile, cfg: dict) -> dict[str, float]:
    """Cost (lower is better, roughly 0..1+) of each slot for one box."""
    a, ac = o.app, cfg["assign"]
    costs = {}
    # each cue counts more the better it separates the two fighters: shorts always matter;
    # tops matter a lot in women's fights (black top vs white top), bare torsos barely separate
    # men but still keep a dressed ref off a fighter slot
    w_s, w_t = cue_weights(prof, ac)
    for c in ("red", "blue"):
        s = similarity(a.shorts, prof.fighter_hist(c))
        t = similarity(a.torso, prof.red_torso if c == "red" else prof.blue_torso)
        if s is None and t is None:
            cost = 1.0
        elif t is None:
            cost = 1.0 - s
        elif s is None:
            cost = 1.0 - t if w_t >= w_s else 1.0
        else:
            cost = 1.0 - (w_s * s + w_t * t) / (w_s + w_t)
        if a.clothed is not None:
            cost += ac["fighter_clothed_penalty"] * max(0.0, a.clothed - ac["clothed_free"])
        # a torso that looks like the ref's shirt rather than this fighter's torso (skin or
        # top) is the ref, e.g. in a close-up with the legs out of frame
        t_ref = similarity(a.torso, prof.ref_torso)
        if t_ref is not None and t is not None:
            cost += ac["ref_torso_penalty"] * max(0.0, t_ref - t)
        costs[c] = cost
    parts, weights = [], []
    if a.clothed is not None:
        parts.append(a.clothed)
        weights.append(ac["ref_clothed_weight"])
    sims = [s for s in (similarity(a.torso, prof.ref_torso), similarity(a.thigh, prof.ref_thigh))
            if s is not None]
    if sims:
        parts.append(max(sims))
        weights.append(ac["ref_profile_weight"])
    costs["ref"] = 1.0 - float(np.average(parts, weights=weights)) if parts else 1.0
    return costs


def assign_frame(obs: list[BoxObs], prof: Profile, cfg: dict) -> list[dict]:
    """Best one-to-one assignment of boxes to {red, blue, ref}, each box possibly 'none'.

    Exhaustive over at most max_people boxes x 3 slots (tiny). A box only takes a slot when its
    cost is under that slot's max cost. Then the third-person fallback. Merged boxes are found
    later, over time (`apply_merges`).
    """
    ac = cfg["assign"]
    max_cost = {"red": ac["fighter_max_cost"], "blue": ac["fighter_max_cost"],
                "ref": ac["ref_max_cost"]}
    costs = [slot_costs(o, prof, cfg) for o in obs]
    n = len(obs)
    best, best_total = [None] * n, float("inf")
    # each slot takes one box (under its max cost) or stays empty: at most (n+1)^3 combos
    options = [[None] + [i for i in range(n) if costs[i][s] <= max_cost[s]] for s in SLOTS]
    for pick in itertools.product(*options):
        chosen = [i for i in pick if i is not None]
        if len(chosen) != len(set(chosen)):
            continue
        # every unassigned box pays a flat cost: prefer filling slots with good matches
        total = (sum(costs[i][s] for s, i in zip(SLOTS, pick) if i is not None)
                 + ac["none_cost"] * (n - len(chosen)))
        if total < best_total - 1e-9:
            best_total = total
            best = [None] * n
            for s, i in zip(SLOTS, pick):
                if i is not None:
                    best[i] = s
    out = []
    for i, c in enumerate(best):
        others = sorted(v for k, v in costs[i].items() if k != c)
        own = costs[i][c] if c else None
        out.append({"label": c or "none", "cost": own,
                    "margin": (others[0] - own) if own is not None and others else None,
                    "fallback": False, **{f"cost_{k}": v for k, v in costs[i].items()}})
    labels = [r["label"] for r in out]
    # third-person fallback: both fighters found, one other box left, nobody is the ref
    if "red" in labels and "blue" in labels and "ref" not in labels:
        rest = [i for i, lab in enumerate(labels) if lab == "none"]
        if len(rest) == 1 and costs[rest[0]]["ref"] <= ac["fallback_ref_max_cost"]:
            out[rest[0]].update(label="ref", cost=costs[rest[0]]["ref"], fallback=True)
    return out


# ---------------------------------------------------------------- smoothing


def check_lookahead(cfg: dict, fps: float, lookahead_s: float) -> None:
    sm = cfg["smooth"]
    need = max(sm["smooth_s"], sm["split_frames"] / fps)
    if need > lookahead_s:
        raise ValueError(f"identity smoothing needs {need:.2f} s of future frames, more than "
                         f"lookahead_s={lookahead_s}")


def split_segments(labels: list[str], split_frames: int) -> list[int]:
    """Segment id per position. A new segment starts where a run of one label lasting at least
    split_frames begins and differs from the label that dominated the current segment."""
    seg, out, start = 0, [], 0
    counts: dict[str, int] = {}
    runs = [(k, len(list(g))) for k, g in itertools.groupby(labels)]
    for lab, n in runs:
        dominant = max(counts, key=counts.get) if counts else None
        if dominant is not None and lab != dominant and n >= split_frames:
            seg += 1
            counts = {}
        counts[lab] = counts.get(lab, 0) + n
        out += [seg] * n
        start += n
    return out


def smooth_track(frames: list[int], labels: list[str], fps: float, cfg: dict) -> list[str]:
    """Majority label within the same segment over [t - smooth_s, t + smooth_s].

    Ties go to the box's own label if it is among the winners, else to the winner that occurs
    first in the window. Sliding window, O(n) per tracklet (frames must be ascending).
    """
    from bisect import bisect_left

    sm = cfg["smooth"]
    w = sm["smooth_s"] * fps
    segs = split_segments(labels, sm["split_frames"])
    n = len(frames)
    pos: dict[str, list[int]] = {}
    for j, lab in enumerate(labels):
        pos.setdefault(lab, []).append(j)
    counts: dict[str, int] = {}
    lo = hi = 0
    out = []
    for i in range(n):
        while hi < n and segs[hi] == segs[i] and frames[hi] - frames[i] <= w:
            counts[labels[hi]] = counts.get(labels[hi], 0) + 1
            hi += 1
        while segs[lo] != segs[i] or frames[i] - frames[lo] > w:
            counts[labels[lo]] -= 1
            lo += 1
        top = max(counts.values())
        winners = [k for k, v in counts.items() if v == top]
        if labels[i] in winners:
            out.append(labels[i])
        else:
            out.append(min(winners, key=lambda k: pos[k][bisect_left(pos[k], lo)]))
    return out


def weak_corner_guard(df, ac: dict):
    """Leave a corner empty rather than give it to a box that matches it weakly.

    A red/blue box becomes 'none' when all hold:
      - its own corner cost is above `weak_corner_cost` and above its ref cost (it looks more
        like a clothed person: a seated official, crowd behind the fence),
      - it is shorter than `weak_corner_max_rel_height` x the tallest box in the frame,
      - its tracklet never matched this corner with a cost under `weak_corner_trusted_cost`
        earlier (past only). A fighter crumpled on the mat after a knockdown looks just as
        clothed and ref-like for a moment, but was confidently that fighter before.
    Runs before `apply_merges`, so a freed corner can still become 'merged'.
    """
    thr = ac.get("weak_corner_cost")
    if thr is None or df.empty or "cost_ref" not in df:
        return df
    own = df["cost_red"].where(df["label"] == "red", df["cost_blue"])
    h = df["y2"] - df["y1"]
    rel = h / h.groupby(df["frame_idx"]).transform("max")
    weak = (df["label"].isin(["red", "blue"]) & (own > thr) & (own > df["cost_ref"])
            & (rel < ac.get("weak_corner_max_rel_height", 1.0)))
    trusted = ac.get("weak_corner_trusted_cost")
    if trusted is not None and "track_id" in df:
        for c in ("red", "blue"):
            good = (df["track_id"] != "") & (df[f"cost_{c}"] < trusted)
            first = df.loc[good].groupby("track_id")["frame_idx"].min()
            since = df["track_id"].map(first)
            weak &= ~((df["label"] == c) & since.notna() & (since <= df["frame_idx"]))
    df.loc[weak, "label"] = "none"
    return df


def smooth_labels(df, fps: float, cfg: dict):
    """df rows: one per kept box with frame_idx, track_id, raw_label, cost. Adds `label`:
    smoothed within each tracklet, then de-duplicated per frame (each of red / blue / ref /
    merged at most once; the lowest raw cost keeps it, the others fall back to 'none')."""
    df = df.copy()
    df["label"] = df["raw_label"]
    tracked = df[df["track_id"] != ""]
    for _, g in tracked.groupby("track_id"):
        g = g.sort_values("frame_idx")
        df.loc[g.index, "label"] = smooth_track(g["frame_idx"].tolist(), g["raw_label"].tolist(),
                                                fps, cfg)
    cost = df["cost"].fillna(9.0)
    for lab in ("red", "blue", "ref", "merged"):
        m = df["label"] == lab
        keep = cost[m].groupby(df.loc[m, "frame_idx"]).idxmin()
        df.loc[m & ~df.index.isin(keep.values), "label"] = "none"
    df = weak_corner_guard(df, cfg["assign"])
    df = apply_merges(df, fps, cfg["assign"])
    # fighters are among the big people in frame; a small red/blue box is crowd or crew
    h = df["y2"] - df["y1"]
    rel = h / h.groupby(df["frame_idx"]).transform("max")
    small = df["label"].isin(["red", "blue"]) & (rel < cfg["assign"]["fighter_min_rel_height"])
    df.loc[small, "label"] = "none"
    return df


def apply_merges(df, fps: float, ac: dict):
    """Mark clinch boxes as 'merged', using the past only.

    A fighter's box becomes 'merged' when the other fighter has no box of their own (missing,
    or only a fragment lying inside it) and the other fighter's last box, seen at most
    merge_memory_s ago in the same shot, overlapped this one. A fighter who just walked out of
    frame never overlapped the remaining one, so a lone fighter stays red or blue. While the
    pair stays merged, the merged box stands in as both fighters' last box.
    """
    mem = ac["merge_memory_s"] * fps
    last: dict[str, tuple] = {}
    df = df.sort_values("frame_idx")
    for f, g in df.groupby("frame_idx", sort=True):
        shot = g["shot_id"].iloc[0] if "shot_id" in g else None
        labs = g["label"]
        boxes = {i: (r["x1"], r["y1"], r["x2"], r["y2"]) for i, r in g.iterrows()}
        merged_box = None
        for have, miss in (("red", "blue"), ("blue", "red")):
            if (labs == have).sum() != 1 or merged_box is not None:
                continue
            i = labs.index[labs == have][0]
            xb = boxes[i]
            js = labs.index[labs == miss]
            j = js[0] if len(js) else None
            if j is not None and contained(boxes[j], xb) < ac["merge_contain_min"]:
                continue
            prev = last.get(miss)
            if prev is None or f - prev[0] > mem or prev[2] != shot:
                continue
            w, h = xb[2] - xb[0], xb[3] - xb[1]
            if w / max(h, 1e-6) < ac["merge_aspect_min"]:
                continue
            if iou(prev[1], xb) >= ac["merge_iou_min"] or contained(prev[1], xb) >= 0.5:
                df.loc[i, "label"] = "merged"
                if j is not None:
                    df.loc[j, "label"] = "none"
                merged_box = xb
        if merged_box is not None:
            # both fighters are inside the merged box: no other box can be red or blue
            others = g.index[(df.loc[g.index, "label"].isin(["red", "blue"]))]
            df.loc[others, "label"] = "none"
            last["red"] = last["blue"] = (f, merged_box, shot)
        else:
            for c in ("red", "blue"):
                idx = labs.index[labs == c]
                if len(idx):
                    last[c] = (f, boxes[idx[0]], shot)
    return df
