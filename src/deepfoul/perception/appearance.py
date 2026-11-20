"""Per-box appearance from pose-guided body regions (1.3).

Regions are drawn from keypoints (only points with score >= kpt_conf_min count):
  shorts: hips, plus each hip -> `shorts_depth` of the way to its knee, as thick strokes
  thigh:  the lower part of the thigh (hip->knee from thigh_from to 0.9), kept as part of the
          ref clothing profile
  shins:  knee -> ankle strokes
  torso:  the shoulder/hip quad
  skin:   two upper-cheek triangles from the face landmarks (the person's own skin color)
Stroke widths scale with torso length, so regions work at any zoom.

Colors are summarized as HSV histograms with HUE_BINS x SAT_BINS chromatic bins plus V_BINS
achromatic bins (pixels with low saturation or low value), so black / white / gray shorts
separate as well as colored ones. A region with fewer than `min_pixels` pixels is None
(unknown), never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HUE_BINS, SAT_BINS, V_BINS = 18, 4, 8
N_BINS = HUE_BINS * SAT_BINS + V_BINS

L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16
# Upper-cheek triangles (COCO-WholeBody face points: jaw 23-39, nose bridge 50-53).
CHEEKS = ((24, 26, 52), (38, 36, 52))


@dataclass
class Appearance:
    shorts: np.ndarray | None   # N_BINS histogram (sums to 1) or None
    torso: np.ndarray | None
    thigh: np.ndarray | None
    shins: np.ndarray | None
    shin_skin: float | None     # fraction of shin pixels that look like the face skin
    torso_skin: float | None
    clothed: float | None       # 0 bare shins .. 1 long pants (see clothed_score)
    torso_len: float            # pixels


def hsv_bins(hsv: np.ndarray, sat_min: int, v_min: int) -> np.ndarray:
    """HSV pixels (N x 3, OpenCV ranges) -> bin index per pixel."""
    h, s, v = hsv[:, 0].astype(np.int32), hsv[:, 1].astype(np.int32), hsv[:, 2].astype(np.int32)
    chroma = (s >= sat_min) & (v >= v_min)
    hb = np.minimum(h * HUE_BINS // 180, HUE_BINS - 1)
    sb = np.minimum((s - sat_min).clip(0) * SAT_BINS // max(1, 256 - sat_min), SAT_BINS - 1)
    vb = np.minimum(v * V_BINS // 256, V_BINS - 1)
    return np.where(chroma, hb * SAT_BINS + sb, HUE_BINS * SAT_BINS + vb)


def histogram(hsv: np.ndarray, sat_min: int, v_min: int) -> np.ndarray:
    hist = np.bincount(hsv_bins(hsv, sat_min, v_min), minlength=N_BINS).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def similarity(p: np.ndarray | None, q: np.ndarray | None) -> float | None:
    """Bhattacharyya coefficient: 1 identical, 0 disjoint."""
    if p is None or q is None:
        return None
    return float(np.sqrt(p * q).sum())


def seed_histogram(name_spec: str, colors: dict, sat_min: int, v_min: int,
                   accent_weight: float = 0.3) -> np.ndarray:
    """A prior histogram from a shorts name like 'black+gold': the bins whose centers fall in
    the named HSV ranges (main color weight 1, each accent `accent_weight`)."""
    from deepfoul.io.fights import shorts_names

    # one representative pixel per (h, s, v) grid cell
    h, s, v = np.meshgrid(np.arange(0, 180, 2), np.arange(0, 256, 8), np.arange(0, 256, 8),
                          indexing="ij")
    grid = np.stack([h.ravel(), s.ravel(), v.ravel()], 1)
    bins = hsv_bins(grid, sat_min, v_min)
    hist = np.zeros(N_BINS)
    for k, name in enumerate(shorts_names(name_spec)):
        inside = np.zeros(len(grid), bool)
        for h0, s0, v0, h1, s1, v1 in colors[name]:
            inside |= ((grid[:, 0] >= h0) & (grid[:, 0] <= h1) & (grid[:, 1] >= s0)
                       & (grid[:, 1] <= s1) & (grid[:, 2] >= v0) & (grid[:, 2] <= v1))
        h_name = np.bincount(bins[inside], minlength=N_BINS).astype(np.float64)
        if h_name.sum():
            hist += (1.0 if k == 0 else accent_weight) * h_name / h_name.sum()
    return hist / max(hist.sum(), 1e-9)


def seed_fraction(hist: np.ndarray | None, seed: np.ndarray) -> float | None:
    """Share of a histogram's mass that falls in bins the seed color covers."""
    if hist is None:
        return None
    return float(hist[seed > 0].sum())


def _ok(kpts, scores, idx, conf) -> bool:
    return all(scores[i] >= conf for i in idx)


def _lerp(a, b, t):
    return a + (b - a) * t


def region_masks(kpts: np.ndarray, scores: np.ndarray, shape: tuple[int, int], cfg: dict,
                 conf: float) -> tuple[dict[str, np.ndarray | None], float]:
    """Boolean masks (frame-sized crops are avoided by drawing into a box-local canvas by the
    caller) for each region; None if its keypoints are missing."""
    import cv2

    h, w = shape
    sho, hip = (L_SHO, R_SHO), (L_HIP, R_HIP)
    torso_len = None
    if _ok(kpts, scores, sho + hip, conf):
        torso_len = float(np.linalg.norm(kpts[list(sho)].mean(0) - kpts[list(hip)].mean(0)))
    elif _ok(kpts, scores, (L_SHO, L_HIP), conf):
        torso_len = float(np.linalg.norm(kpts[L_SHO] - kpts[L_HIP]))
    elif _ok(kpts, scores, (R_SHO, R_HIP), conf):
        torso_len = float(np.linalg.norm(kpts[R_SHO] - kpts[R_HIP]))
    if not torso_len or torso_len < 4:
        return {k: None for k in ("shorts", "thigh", "shins", "torso", "skin")}, 0.0
    width = lambda f: max(2, int(round(f * torso_len)))  # noqa: E731
    pt = lambda p: tuple(int(round(c)) for c in p)  # noqa: E731

    def canvas():
        return np.zeros((h, w), np.uint8)

    out: dict[str, np.ndarray | None] = {}
    # shorts: each hip -> shorts_depth toward its knee, plus the hip line
    m, any_ = canvas(), False
    for hp, kn in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
        if _ok(kpts, scores, (hp, kn), conf):
            cv2.line(m, pt(kpts[hp]), pt(_lerp(kpts[hp], kpts[kn], cfg["shorts_depth"])), 1,
                     width(cfg["shorts_width"]))
            any_ = True
    if any_ and _ok(kpts, scores, hip, conf):
        cv2.line(m, pt(kpts[L_HIP]), pt(kpts[R_HIP]), 1, width(cfg["shorts_width"]))
    out["shorts"] = m.astype(bool) if any_ else None
    # thigh: lower thigh (fabric on a ref, skin on a fighter)
    m, any_ = canvas(), False
    for hp, kn in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
        if _ok(kpts, scores, (hp, kn), conf):
            cv2.line(m, pt(_lerp(kpts[hp], kpts[kn], cfg["thigh_from"])),
                     pt(_lerp(kpts[hp], kpts[kn], 0.9)), 1, width(cfg["limb_width"]))
            any_ = True
    out["thigh"] = m.astype(bool) if any_ else None
    # shins
    m, any_ = canvas(), False
    for kn, an in ((L_KNEE, L_ANK), (R_KNEE, R_ANK)):
        if _ok(kpts, scores, (kn, an), conf):
            cv2.line(m, pt(_lerp(kpts[kn], kpts[an], 0.15)), pt(_lerp(kpts[kn], kpts[an], 0.85)),
                     1, width(cfg["limb_width"]))
            any_ = True
    out["shins"] = m.astype(bool) if any_ else None
    # torso quad
    if _ok(kpts, scores, sho + hip, conf):
        m = canvas()
        quad = np.array([kpts[L_SHO], kpts[R_SHO], kpts[R_HIP], kpts[L_HIP]])
        center = quad.mean(0)
        quad = center + (quad - center) * 0.8   # stay off the arms and the waistband
        cv2.fillPoly(m, [quad.round().astype(np.int32)], 1)
        out["torso"] = m.astype(bool)
    else:
        out["torso"] = None
    # skin: upper cheeks
    m, any_ = canvas(), False
    for tri in CHEEKS:
        if _ok(kpts, scores, tri, conf):
            cv2.fillPoly(m, [kpts[list(tri)].round().astype(np.int32)], 1)
            any_ = True
    out["skin"] = m.astype(bool) if any_ else None
    return out, torso_len


def skin_fraction(lab_pixels: np.ndarray, skin_lab: np.ndarray, cfg: dict) -> float:
    """Share of pixels whose chroma (a, b) is close to the face skin chroma and whose lightness
    is within a wide band of it (shins are lit differently from the face)."""
    mu = np.median(skin_lab, 0)
    d_ab = np.linalg.norm(lab_pixels[:, 1:].astype(np.float64) - mu[1:], axis=1)
    d_l = np.abs(lab_pixels[:, 0].astype(np.float64) - mu[0])
    return float(((d_ab <= cfg["skin_ab_tol"]) & (d_l <= cfg["skin_l_tol"])).mean())


def clothed_score(shin_skin: float | None, shorts: np.ndarray | None,
                  shins: np.ndarray | None) -> float | None:
    """0 = bare shins (fighter), 1 = long pants (ref). Mean of 'shins are not skin' (vs the
    person's own face) and 'shins look like the fabric at the hips' (one fabric hip to ankle =
    pants; a fighter's shorts never look like bare shins), using whichever is known."""
    parts = []
    if shin_skin is not None:
        parts.append(1.0 - shin_skin)
    sim = similarity(shorts, shins)
    if sim is not None:
        parts.append(sim)
    return float(np.mean(parts)) if parts else None


def describe(image: np.ndarray, kpts: np.ndarray, scores: np.ndarray, box: np.ndarray,
             cfg: dict, conf: float) -> Appearance:
    """Appearance of one person. Work happens inside the box (padded by 10%) for speed."""
    import cv2

    H, W = image.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    cx2, cy2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return Appearance(None, None, None, None, None, None, None, 0.0)
    local = kpts - np.array([cx1, cy1], np.float32)
    masks, torso_len = region_masks(local, scores, crop.shape[:2], cfg, conf)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    sat_min, v_min, min_px = cfg["sat_min"], cfg["v_min"], cfg["min_pixels"]

    def hist(name):
        m = masks[name]
        if m is None or m.sum() < min_px:
            return None
        return histogram(hsv[m], sat_min, v_min)

    hists = {k: hist(k) for k in ("shorts", "torso", "thigh", "shins")}
    skin = masks["skin"]
    skin_px = lab[skin] if skin is not None and skin.sum() >= cfg["min_skin_pixels"] else None

    def skin_frac(name):
        m = masks[name]
        if skin_px is None or m is None or m.sum() < min_px:
            return None
        return skin_fraction(lab[m], skin_px, cfg)

    shin_skin = skin_frac("shins")
    return Appearance(hists["shorts"], hists["torso"], hists["thigh"], hists["shins"],
                      shin_skin, skin_frac("torso"),
                      clothed_score(shin_skin, hists["shorts"], hists["shins"]), torso_len)

