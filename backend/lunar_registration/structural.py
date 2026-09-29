"""
Dense structural-NCC correspondence generator (Stage 4 last-resort arm).
========================================================================
WHY THIS EXISTS: on this pair class (cross-sensor, cross-sun-angle lunar
imagery) the keypoint matcher arms collapse - the Lowe ratio test rejects
almost everything on repetitive crater terrain, leaving 1-10 matches across
all rotation hypotheses - so Stage 4 never receives enough points to estimate
a homography and the run fails closed with no transform at all. This
generator bypasses detection/description entirely: it derives dense
correspondences from the structural (Sobel gradient-magnitude) maps that the
verification gate already trusts (`verify.compute_structural_ncc`), so the
pipeline can still estimate a homography when every matcher arm produced
nothing.

Pipeline contract: by the time this runs, Stage 3 has ALREADY applied the
coarse scale and rotation to `src_img`, so translation is the only remaining
unknown. The generator:
  1. estimates that translation by Hann-windowed phase correlation on a
     <= 512px working copy of the pair,
  2. warps the source into the reference frame with H0 = translation(dx, dy),
  3. takes the best TM_CCOEFF_NORMED peak per reference-grid cell on the
     gradient maps (template = the cell's own patch; search window = the same
     cell grown by `cfg.structural_search_radius_px`),
  4. maps each peak back to source coordinates through inv(H0),
then returns a normal `MatchResult` so Stage 4 estimates a homography from it
exactly as it would from a matcher arm. Fewer than
`cfg.structural_min_correspondences` peaks - or degenerate input (empty,
tiny, featureless) - yields an empty `MatchResult` the caller discards; this
module never raises for bad input.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .matching import MatchResult
from .verify import gradient_magnitude


# Long-side cap for the phase-correlation working frame: keeps the FFT cheap
# while retaining enough structure for a reliable translation prior.
_PHASE_WORK_MAX_SIDE = 512
# Smallest grid cell (px) worth template-matching. Below this the patch
# degenerates and the per-cell NCC is meaningless, so we bail out instead.
_MIN_CELL_PX = 7
# A patch must carry at least this much gradient energy or the NCC is
# numerically meaningless. Two floors, because gradient-magnitude scale
# depends on input units (uint8 images -> hundreds, float images -> ~0.05):
# an absolute floor for exact-zero (constant) patches, and a fraction of the
# owning map's global std for merely near-constant ones. This also defuses a
# cv2.matchTemplate(TM_CCOEFF_NORMED) trap: a zero-variance patch returns
# exactly 1.0 (its 0/0 is swallowed by an epsilon), which would otherwise
# read as a perfect structural match on flat terrain.
_MIN_PATCH_STD = 1e-6
_MIN_PATCH_STD_REL = 0.01
# Reported in MatchResult.method / .provenance so callers can trace the arm.
_METHOD = "structural_ncc"


def _empty_result() -> MatchResult:
    """Zero-length result the caller discards (degenerate input or too few peaks)."""
    return MatchResult(
        method=_METHOD,
        pts_src=np.zeros((0, 2), np.float32),
        pts_dst=np.zeros((0, 2), np.float32),
        scores=np.zeros((0,), np.float32),
        provenance=_METHOD,
    )


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    """Contiguous float32 grayscale copy used by the phase-correlation prior."""
    arr = np.asarray(img)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _pad_top_left(img: np.ndarray, canvas_h: int, canvas_w: int) -> np.ndarray:
    """Zero-pad `img` to the canvas anchored at the top-left corner.

    Phase correlation needs equal shapes. Padding only at the bottom/right
    keeps both images' coordinate origins at (0, 0), so the relative content
    displacement between them - the quantity we measure - is preserved.
    """
    h, w = img.shape[:2]
    if h == canvas_h and w == canvas_w:
        return img
    out = np.zeros((canvas_h, canvas_w), np.float32)
    out[:h, :w] = img
    return out


def _phase_translation_prior(src_img: np.ndarray, ref_img: np.ndarray) -> Tuple[float, float]:
    """Translation (dx, dy) in reference pixels that best aligns source onto reference.

    Both images are scaled by ONE similarity factor (long side <= 512), so a
    pair with unequal sizes keeps a per-axis-consistent pixel spacing; each
    working copy is then top-left-aligned and zero-padded to a common canvas
    and phase-correlated with a Hann window. The measured shift is scaled
    back to full-resolution reference pixels per axis. Structureless input
    (zero response, NaNs) falls back to a zero prior rather than raising.
    """
    src = _to_gray_f32(src_img)
    ref = _to_gray_f32(ref_img)
    h0, w0 = src.shape[:2]
    h1, w1 = ref.shape[:2]

    long_side = max(h0, w0, h1, w1)
    scale = min(1.0, _PHASE_WORK_MAX_SIDE / float(long_side))
    ws0, hs0 = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
    ws1, hs1 = max(1, int(round(w1 * scale))), max(1, int(round(h1 * scale)))

    if scale < 1.0:
        src_w = cv2.resize(src, (ws0, hs0), interpolation=cv2.INTER_AREA)
        ref_w = cv2.resize(ref, (ws1, hs1), interpolation=cv2.INTER_AREA)
    else:
        src_w, ref_w = src, ref

    canvas_h, canvas_w = max(hs0, hs1), max(ws0, ws1)
    src_c = _pad_top_left(src_w, canvas_h, canvas_w)
    ref_c = _pad_top_left(ref_w, canvas_h, canvas_w)

    hann = cv2.createHanningWindow((canvas_w, canvas_h), cv2.CV_32F)
    (dx_w, dy_w), response = cv2.phaseCorrelate(src_c, ref_c, window=hann)

    if not (np.isfinite(dx_w) and np.isfinite(dy_w) and np.isfinite(response)):
        return 0.0, 0.0
    if response <= 0.0:
        # Flat / structureless pair: no usable peak, fall back to no prior.
        return 0.0, 0.0

    # Working pixels -> full-resolution reference pixels (per axis).
    return float(dx_w) * (w1 / float(ws1)), float(dy_w) * (h1 / float(hs1))


def _local_std(img: np.ndarray, side: int) -> np.ndarray:
    """Std-dev of every `side` x `side` patch of `img`, indexed by patch top-left.

    `cv2.boxFilter` responds at the kernel CENTRE, so the response belonging to
    the patch with top-left (i, j) sits at (i + side // 2, j + side // 2); the
    slice re-indexes it to match `cv2.matchTemplate`'s top-left result layout.
    Used to blank out (near-)constant match positions whose NCC would be
    numerically meaningless (see _MIN_PATCH_STD).
    """
    half = side // 2
    mean = cv2.boxFilter(img, cv2.CV_32F, (side, side), normalize=True)
    sq_mean = cv2.boxFilter(img * img, cv2.CV_32F, (side, side), normalize=True)
    var = sq_mean - mean * mean
    np.maximum(var, 0.0, out=var)
    std = np.sqrt(var)
    h, w = img.shape[:2]
    return std[half:half + h - side + 1, half:half + w - side + 1]


def structural_ncc_correspondences(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    cfg: Optional[PipelineConfig] = None,
) -> MatchResult:
    """Generate dense correspondences from structural (gradient) maps.

    WHY: keypoint matchers collapse on this pair class (ratio-test rejection
    on repetitive crater terrain), starving Stage 4 of points. Instead of
    keypoints, this takes the best NCC peak per grid cell of the Sobel
    gradient maps - after a phase-correlation translation prior warps the
    (already rotation/scale-aligned) source into the reference frame - and
    maps each peak back to source coordinates through inv(H0). Returns a
    normal `MatchResult` (`method="structural_ncc"`) for Stage 4 to estimate
    a homography from, or an empty one when fewer than
    `cfg.structural_min_correspondences` peaks survive (the caller discards
    it). Never raises for degenerate/flat input.
    """
    cfg = cfg or PipelineConfig()
    if not cfg.structural_fallback_enabled:
        return _empty_result()

    if src_img is None or ref_img is None:
        return _empty_result()
    src = np.asarray(src_img)
    ref = np.asarray(ref_img)
    if src.ndim < 2 or ref.ndim < 2 or src.size == 0 or ref.size == 0:
        return _empty_result()

    h_ref, w_ref = int(ref.shape[0]), int(ref.shape[1])
    grid = int(cfg.structural_grid)
    if grid <= 0:
        return _empty_result()
    if min(w_ref // grid, h_ref // grid) < _MIN_CELL_PX:
        # Image only spans a few (or sub-pixel) cells: nothing meaningful to match.
        return _empty_result()

    t0 = time.perf_counter()

    # 1-2. Translation prior by phase correlation, then warp src into the
    # reference frame (Stage 3 already removed scale/rotation upstream).
    dx, dy = _phase_translation_prior(src, ref)
    h0 = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
    h0_inv = np.linalg.inv(h0)
    src_warp = src if src.dtype in (np.uint8, np.float32) else src.astype(np.float32)
    aligned = cv2.warpPerspective(src_warp, h0, (w_ref, h_ref))

    # 3. Structural maps - the exact representation verify.compute_structural_ncc uses.
    g_aligned = gradient_magnitude(aligned)
    g_ref = gradient_magnitude(ref)
    g_ref_std = float(g_ref.std())
    g_aligned_std = float(g_aligned.std())
    if g_ref_std <= _MIN_PATCH_STD or g_aligned_std <= _MIN_PATCH_STD:
        # Featureless image on either side: no structure to correspond.
        return _empty_result()

    radius = int(cfg.structural_search_radius_px)
    min_ncc = float(cfg.structural_min_ncc)
    # Per-map patch floors: absolute floor for constant patches, relative
    # floor (fraction of global structure) for near-constant ones.
    ref_std_floor = max(_MIN_PATCH_STD, _MIN_PATCH_STD_REL * g_ref_std)
    aligned_std_floor = max(_MIN_PATCH_STD, _MIN_PATCH_STD_REL * g_aligned_std)
    q_pts: List[List[float]] = []          # matched content, reference-frame coords
    pts_dst: List[List[float]] = []        # template centre in reference coords
    scores: List[float] = []

    # 4. Per-cell best peak on the gradient maps.
    for row in range(grid):
        y0 = int(round(row * h_ref / grid))
        y1 = int(round((row + 1) * h_ref / grid))
        for col in range(grid):
            x0 = int(round(col * w_ref / grid))
            x1 = int(round((col + 1) * w_ref / grid))

            # Patch side = cell side, made odd so it has an integer centre.
            side = min(x1 - x0, y1 - y0)
            if side < _MIN_CELL_PX:
                continue
            if side % 2 == 0:
                side -= 1
            half = side // 2

            # Template: gradient patch centred in the cell (clamped at borders).
            px = min(max(int(round((x0 + x1) / 2.0)) - half, 0), w_ref - side)
            py = min(max(int(round((y0 + y1) / 2.0)) - half, 0), h_ref - side)
            template = g_ref[py:py + side, px:px + side]
            if float(template.std()) < ref_std_floor:
                continue  # featureless cell: NCC would be NaN/meaningless

            # Search window: the same cell grown by the search radius.
            wx0, wy0 = max(0, x0 - radius), max(0, y0 - radius)
            wx1, wy1 = min(w_ref, x1 + radius), min(h_ref, y1 + radius)
            window = g_aligned[wy0:wy1, wx0:wx1]
            if window.shape[0] < side or window.shape[1] < side:
                continue  # defensive: matchTemplate requires window >= template

            res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            res = np.asarray(res)
            res[~np.isfinite(res)] = -1.0  # flat window -> NaN
            # Blank out positions whose own patch is (near-)constant: their
            # NCC is numerical noise (matchTemplate even reports 1.0 there).
            res[_local_std(window, side) < aligned_std_floor] = -1.0
            _, best, _, loc = cv2.minMaxLoc(res)
            if best < min_ncc:
                continue

            # 5. Map back to source coordinates: the content found at reference-frame
            # position q of `aligned` came from source position inv(H0) @ q.
            q_pts.append([wx0 + loc[0] + half, wy0 + loc[1] + half])
            pts_dst.append([px + half, py + half])  # dst = cell centre in ref coords
            scores.append(float(best))

    # 7. Too few peaks -> empty MatchResult for the caller to discard.
    if len(q_pts) < int(cfg.structural_min_correspondences):
        return _empty_result()

    q_arr = np.asarray(q_pts, dtype=np.float64)
    hom = np.hstack([q_arr, np.ones((len(q_arr), 1), dtype=np.float64)])
    mapped = (h0_inv @ hom.T).T
    pts_src = (mapped[:, :2] / mapped[:, 2:3]).astype(np.float32)

    return MatchResult(
        method=_METHOD,
        pts_src=pts_src,
        pts_dst=np.asarray(pts_dst, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        provenance=_METHOD,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )
