"""
Cause-Branched Subpixel Refinement and Overlap Seam Reconciliation (§§5, 6).
=============================================================================
- Routes tiles based on environmental cause: illum_delta vs texture_energy.
- Never uses raw ECC under severe illumination delta (> 30 deg).
- Computes subpixel shift and 2x2 covariance.
- Seam checker verifies median prediction disagreement in ground units across tile borders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Task 8: "subpixel" literally means < 1 px — corrections beyond this bound
# are gross source-vs-reference offsets, not registration residuals (the
# report observed 123.5054/271.0260 px reported as subpixel refinement).
SUBPIXEL_MAX_PX: float = 1.0

# Minimum cv2.phaseCorrelate response to consider the peak trustworthy;
# below this the peak is dominated by noise/repetitive texture.
WEAK_PEAK_MIN_RESPONSE: float = 0.2


def compute_texture_energy(img: np.ndarray) -> float:
    """Compute normalized gradient energy as a measure of structural texture."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    g_float = gray.astype(np.float32)
    if g_float.max() > 1.0:
        g_float = g_float / 255.0

    gx = cv2.Sobel(g_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_float, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.mean(mag))


def choose_refiner(
    illum_delta_deg: float,
    texture_energy: float,
    shadow_frac: float = 0.0,
) -> str:
    """Select appropriate subpixel refiner based on environmental cause (§5).

    High illum_delta -> phase-congruency NCC or bi-channel only, NEVER raw ECC.
    Low illum_delta + low texture -> ECC/phase OK.
    Both bad -> coarse_inflated / skip.
    """
    if illum_delta_deg > 30.0 or shadow_frac > 0.2:
        # High illumination variation: phase-congruency NCC / bi-channel only
        if texture_energy < 0.01:
            return "skip"
        return "pc_ncc"
    else:
        # Low illumination difference
        if texture_energy < 0.01:
            return "coarse_inflated"
        elif texture_energy < 0.03:
            return "phase"
        else:
            return "ecc"


def _non_subpixel_reason(dx: float, dy: float, method_name: str) -> Optional[str]:
    """Task 8: a correction beyond +/-SUBPIXEL_MAX_PX cannot be a subpixel
    refinement — name it instead of silently reporting it as one."""
    if max(abs(dx), abs(dy)) > SUBPIXEL_MAX_PX:
        return (
            f"non-subpixel correction from {method_name}: dx={dx:.4f}, "
            f"dy={dy:.4f} px exceeds +-{SUBPIXEL_MAX_PX}px — measures the "
            "gross source/reference offset, not a registration residual"
        )
    return None


def refine_tile(
    tile_src: np.ndarray,
    tile_ref: np.ndarray,
    illum_delta_deg: float = 0.0,
    texture_energy: Optional[float] = None,
    shadow_frac: float = 0.0,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute cause-branched subpixel refinement on a tile pair.

    Returns {dx, dy, cov2x2, method, low_precision, reason} in reference
    pixels. `reason` is None when no precision objection applies, otherwise
    it names WHY the estimate is low precision (weak phase-correlation peak,
    correction beyond +/-SUBPIXEL_MAX_PX, refiner skipped) — a bare boolean
    could not say (Task 8). The pipeline adds the structural
    matcher-independence objection on top of these value-level ones.
    """
    if texture_energy is None:
        texture_energy = compute_texture_energy(tile_src)

    if method is None:
        method = choose_refiner(illum_delta_deg, texture_energy, shadow_frac)

    def _to_gray(im: np.ndarray) -> np.ndarray:
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_f = im.astype(np.float32)
        if im_f.max() > 1.0:
            im_f = im_f / 255.0
        return im_f

    g0 = _to_gray(tile_src)
    g1 = _to_gray(tile_ref)

    # Ensure matching spatial dimensions for ECC and phase correlation
    h1, w1 = g1.shape[:2]
    if g0.shape[:2] != (h1, w1):
        g0 = cv2.resize(g0, (w1, h1), interpolation=cv2.INTER_LINEAR)

    if method in ("skip", "coarse_inflated"):
        return {
            "dx": 0.0,
            "dy": 0.0,
            "cov2x2": np.diag([25.0, 25.0]).astype(np.float64),
            "method": method,
            "low_precision": True,
            "reason": (
                f"refiner '{method}' skipped: illumination/texture conditions "
                "too poor for a subpixel estimate"
            ),
        }

    if method == "ecc":
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        try:
            _, M = cv2.findTransformECC(
                (g0 * 255).astype(np.uint8),
                (g1 * 255).astype(np.uint8),
                warp_matrix,
                cv2.MOTION_TRANSLATION,
                criteria,
            )
            dx = float(M[0, 2])
            dy = float(M[1, 2])
            cov = np.diag([0.05, 0.05]).astype(np.float64)
            ecc_reason = _non_subpixel_reason(dx, dy, "ECC")
            return {"dx": dx, "dy": dy, "cov2x2": cov, "method": "ecc",
                    "low_precision": ecc_reason is not None, "reason": ecc_reason}
        except cv2.error:
            # Fall back to phase correlation on ECC divergence
            method = "phase"

    # Phase correlation / pc_ncc
    h, w = g0.shape[:2]
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(g0, g1, window=hann)
    sigma = float(max(0.1, 1.0 / max(response, 1e-3)))
    cov = np.diag([sigma, sigma]).astype(np.float64)

    # Task 8: name every precision objection instead of a bare boolean.
    reasons = []
    if response < WEAK_PEAK_MIN_RESPONSE:
        reasons.append(
            f"weak phase-correlation peak: response={response:.3f} < "
            f"{WEAK_PEAK_MIN_RESPONSE}"
        )
    non_sub = _non_subpixel_reason(float(dx), float(dy), "phase correlation")
    if non_sub:
        reasons.append(non_sub)
    reason = "; ".join(reasons) if reasons else None

    return {
        "dx": float(dx),
        "dy": float(dy),
        "cov2x2": cov,
        "method": method,
        "low_precision": reason is not None,
        "reason": reason,
    }


def check_seam(
    warpA: np.ndarray,
    warpB: np.ndarray,
    overlap_bbox: Tuple[int, int, int, int],
    tau_seam_meters: float = 2.0,
    gsd_ref: float = 0.5,
) -> Dict[str, Any]:
    """Reconcile seam overlap between adjacent tiles in ground units (§6).

    Checks whether median prediction disagreement < tau_seam_meters.
    Tags escalation as 'clean', 'edge_artifact', or 'matcher_failure'.
    """
    x0, y0, x1, y1 = overlap_bbox
    xs = np.linspace(x0, x1, 5)
    ys = np.linspace(y0, y1, 5)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

    ptsA = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), warpA.astype(np.float32)).reshape(-1, 2)
    ptsB = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), warpB.astype(np.float32)).reshape(-1, 2)

    disagree_px = np.linalg.norm(ptsA - ptsB, axis=1)
    median_disagree_px = float(np.median(disagree_px))
    median_disagree_ground = median_disagree_px * max(float(gsd_ref), 1e-4)

    agree = bool(median_disagree_ground <= tau_seam_meters)

    tag = "clean" if agree else ("edge_artifact" if median_disagree_ground < tau_seam_meters * 2.0 else "matcher_failure")

    return {
        "agree": agree,
        "median_disagree_ground": median_disagree_ground,
        "median_disagree_px": median_disagree_px,
        "tag": tag,
    }
