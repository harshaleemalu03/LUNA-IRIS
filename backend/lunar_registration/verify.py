"""
Orthogonal Cheap-Gate and Rigid-Only Hypothesis Competition (§§1, 4).
======================================================================
- Rejects confidently-wrong periodic crater shifts via rigid similarity projection
  and margin competition.
- Enforces orthogonal escalation: residual 256px phase-correlation agreement after
  warping the source by the candidate homography, structural-NCC on gradient maps,
  scale prior consistency, and coverage.
- Time-bounded verification budget (fails open to robust path if exceeded).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of `img` - the structural representation.

    Shared by `compute_structural_ncc` (the verification gate) and the
    structural-NCC correspondence generator (`structural.py`) so both reason
    about exactly the same structural maps. Gradient magnitude is robust to
    inverted shadow orientations under changing sun angles on the lunar surface.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def compute_structural_ncc(
    img0: np.ndarray,
    img1: np.ndarray,
    H: np.ndarray,
    max_patches: int = 200,
) -> float:
    """Compute structural normalized cross-correlation on gradient maps under homography H.

    Sobel gradient magnitudes are robust to inverted shadow orientations under changing
    sun angles on the lunar surface.
    """
    if H is None:
        return 0.0

    h, w = img1.shape[:2]

    g0 = gradient_magnitude(img0)
    g1 = gradient_magnitude(img1)

    warped_g0 = cv2.warpPerspective(g0, H.astype(np.float64), (w, h))
    mask = cv2.warpPerspective(
        np.ones_like(g0, dtype=np.uint8) * 255, H.astype(np.float64), (w, h)
    ) > 0

    if float(mask.mean()) < 0.02:
        return 0.0

    x = warped_g0[mask].ravel().astype(np.float64)
    y = g1[mask].ravel().astype(np.float64)

    # Subsample if large to keep verification fast
    max_pts = max_patches * 64
    if len(x) > max_pts:
        step = len(x) // max_pts
        x = x[::step]
        y = y[::step]

    xz = x - x.mean()
    yz = y - y.mean()
    denom = math.sqrt(float((xz**2).sum() * (yz**2).sum()))
    if denom <= 1e-8:
        return 0.0

    return float(np.clip((xz * yz).sum() / denom, -1.0, 1.0))


def _residual_phase_agree_256(
    img0: np.ndarray,
    img1: np.ndarray,
    H: np.ndarray,
) -> Dict[str, float]:
    """Residual 256px phase correlation after warping img0 into img1's frame with H.

    WHY a residual instead of a direct source-vs-reference comparison: comparing H's
    raw translation against the raw phase shift between the two images structurally
    cannot agree once the pair is rotated or rescaled (measured 475.6px disagreement
    at ~11deg of rotation vs tau_agree=24), so correct transforms were rejected.
    Warping first leaves a ~0 residual for a correct H regardless of rotation/scale.

    Dimensionally correct for unequal image sizes: each image is mapped into the
    shared 256x256 frame by S = diag(256/w, 256/h), so the warp matrix expressed in
    that frame is H' = S1 @ H @ inv(S0). The residual is scaled back to reference
    pixels. `peak_sharpness` comes from the SAME correlation, i.e. it is now measured
    on the ALIGNED pair (a correct H sharpens the peak; gate_cheap's sigma_logscale
    uses it).
    """
    def _to_gray_256(im: np.ndarray) -> np.ndarray:
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(im.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
        if resized.max() > 1.0:
            resized = resized / 255.0
        return resized

    g0 = _to_gray_256(img0)
    g1 = _to_gray_256(img1)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    s0 = np.diag([256.0 / w0, 256.0 / h0, 1.0])
    s1 = np.diag([256.0 / w1, 256.0 / h1, 1.0])
    h_prime = s1 @ np.asarray(H, dtype=np.float64) @ np.linalg.inv(s0)

    warped_g0 = cv2.warpPerspective(g0, h_prime, (256, 256))

    hann = cv2.createHanningWindow((256, 256), cv2.CV_32F)
    (dx_256, dy_256), response = cv2.phaseCorrelate(warped_g0, g1, window=hann)

    # Scale the residual back to img1 (reference) pixels
    dx = float(dx_256 * (w1 / 256.0))
    dy = float(dy_256 * (h1 / 256.0))

    return {"dx": dx, "dy": dy, "peak_sharpness": float(response)}


def gate_cheap(
    warp_candidate: Optional[np.ndarray],
    img0: np.ndarray,
    img1: np.ndarray,
    gsd_src: float = 1.0,
    gsd_ref: float = 1.0,
    t_struct: float = 0.25,
    tau_agree: float = 24.0,
    k_sigma: float = 3.0,
    budget_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Orthogonal escalation gate (§1).

    Checks:
    1. Residual 256px phase agreement: warp img0 by the candidate homography into
       img1's frame, then require the leftover phase shift within tau_agree. The
       residual is ~0 for a correct H at any rotation/scale, so this accepts rotated
       pairs the old translation-equality comparison could never pass.
    2. Implied scale within k_sigma * sigma_logscale of GSD prior (sigma derived from
       the phase peak measured on the ALIGNED pair).
    3. Structural-NCC on gradient maps >= t_struct (default matches
       PipelineConfig.orthogonal_gate_t_struct).
    Budget capped at budget_ms; over-budget fails open to robust path.

    `phase_agree_px` (check #1 residual magnitude in reference pixels) is reported on
    every path where check #1 ran; it is None when no warp was available to check.
    """
    t0 = time.perf_counter()

    if warp_candidate is None:
        cost = (time.perf_counter() - t0) * 1000.0
        return {
            "pass": False,
            "reason": "no_warp_emitted",
            "cost_ms": cost,
            "phase_agree_px": None,
        }

    # 1. Residual phase correlation agreement (rotation/scale aware)
    residual = _residual_phase_agree_256(img0, img1, warp_candidate)
    err_agree = math.hypot(residual["dx"], residual["dy"])

    if err_agree > tau_agree:
        cost = (time.perf_counter() - t0) * 1000.0
        return {
            "pass": False,
            "reason": f"phase_correlation_disagreement: {err_agree:.1f}px > {tau_agree:.1f}px",
            "cost_ms": cost,
            "phase_agree_px": err_agree,
        }

    # 2. Implied scale vs GSD prior
    s_prior = max(float(gsd_ref) / max(float(gsd_src), 1e-4), 1e-4)
    det = abs(warp_candidate[0, 0] * warp_candidate[1, 1] - warp_candidate[0, 1] * warp_candidate[1, 0])
    implied_scale = math.sqrt(max(det, 1e-6))
    log_err = abs(math.log(max(implied_scale, 1e-4) / s_prior))
    sigma_logscale = max(0.1, 0.5 * (1.0 - min(residual["peak_sharpness"], 1.0)))

    if log_err > k_sigma * sigma_logscale:
        cost = (time.perf_counter() - t0) * 1000.0
        return {
            "pass": False,
            "reason": f"implied_scale_violation: log_err {log_err:.2f} > {k_sigma * sigma_logscale:.2f}",
            "cost_ms": cost,
            "phase_agree_px": err_agree,
        }

    # 3. Structural-NCC
    struct_ncc = compute_structural_ncc(img0, img1, warp_candidate)
    if struct_ncc < t_struct:
        cost = (time.perf_counter() - t0) * 1000.0
        return {
            "pass": False,
            "reason": f"structural_ncc_low: {struct_ncc:.3f} < {t_struct:.3f}",
            "cost_ms": cost,
            "phase_agree_px": err_agree,
        }

    cost = (time.perf_counter() - t0) * 1000.0
    if budget_ms is not None and cost > budget_ms:
        return {
            "pass": False,
            "reason": f"verification_budget_exceeded ({cost:.1f}ms > {budget_ms:.1f}ms) -> fail open",
            "cost_ms": cost,
            "phase_agree_px": err_agree,
        }

    return {
        "pass": True,
        "reason": "all_checks_passed",
        "cost_ms": cost,
        "struct_ncc": struct_ncc,
        "phase_agree_px": err_agree,
    }


def compete_rigid(
    candidates: List[Dict[str, Any]],
    terrain_spacing_px: float = 24.0,
    lambda_dof: float = 0.1,
    margin_min: float = 0.05,
) -> Dict[str, Any]:
    """Rigid-only hypothesis competition (§4).

    Resolves repeating crater false-matching:
    1. Scores: score = fit + coverage - lambda_dof * dof_resid
    2. Identifies competing modes separated by > terrain_spacing_px.
    3. If margin Δ = best - second < margin_min, flags ambiguous=True.
    """
    if not candidates:
        return {
            "winner": None,
            "margin": 0.0,
            "ambiguous": True,
            "reason": "no_candidates",
        }

    # Prioritize candidates with verified inliers over arms that produced 0 inliers
    candidates_with_inliers = [
        c for c in candidates
        if c.get("metrics") is not None and getattr(c["metrics"], "n_inliers", 0) > 0
    ]
    candidates_to_score = candidates_with_inliers if candidates_with_inliers else candidates

    scored: List[Dict[str, Any]] = []
    for c in candidates_to_score:
        warp = np.asarray(c.get("warp", np.eye(3)), dtype=np.float64)
        fit = float(c.get("fit", 0.0))
        cov = float(c.get("coverage", 0.0))
        dof_resid = float(c.get("dof_resid", 0.0))
        score = fit + cov - (lambda_dof * dof_resid)

        tx = float(warp[0, 2])
        ty = float(warp[1, 2])

        entry = dict(c)
        entry["score"] = score
        entry["tx"] = tx
        entry["ty"] = ty
        scored.append(entry)

    # Sort descending by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    winner = scored[0]

    if len(scored) == 1:
        return {
            "winner": winner,
            "margin": float("inf"),
            "ambiguous": False,
            "all_scored": scored,
        }

    # Find competing hypotheses separated by > terrain_spacing_px
    best_tx, best_ty = winner["tx"], winner["ty"]
    competing_runners = [
        c for c in scored[1:]
        if math.hypot(c["tx"] - best_tx, c["ty"] - best_ty) >= terrain_spacing_px
    ]

    if not competing_runners:
        margin = float(winner["score"] - scored[1]["score"])
        return {
            "winner": winner,
            "margin": margin,
            "ambiguous": False,
            "all_scored": scored,
        }

    second_best = competing_runners[0]
    margin = float(winner["score"] - second_best["score"])
    ambiguous = bool(margin < margin_min)

    return {
        "winner": winner,
        "runner_up": second_best,
        "margin": margin,
        "ambiguous": ambiguous,
        "all_scored": scored,
    }
