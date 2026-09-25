"""Tasks 2 + 3: rotation/scale-aware gate check #1 and the t_struct recalibration.

WHY this file exists: gate check #1 used to compare the candidate homography's raw
translation against the raw source-vs-reference phase shift. On a pair rotated ~11
degrees that quantity structurally cannot agree with tau_agree=24 (measured 475.6px
on the canonical pair), so CORRECT transforms were rejected. Check #1 now measures a
RESIDUAL: warp the source by the candidate H, then phase-correlate the warped source
against the reference — ~0 for a correct H at any rotation/scale.

Check #3's t_struct was likewise recalibrated from 0.40 (above what this pair class
can achieve: exhaustive-search best 0.317) to 0.25, so an aligned control in the
measured (0.25, 0.40) band must now pass while identity/garbage H still fails.
"""
import inspect
import math

import cv2
import numpy as np

from lunar_registration.config import PipelineConfig
from lunar_registration.verify import compute_structural_ncc, gate_cheap

# One config surface: the gate thresholds under test are read from PipelineConfig so
# a future recalibration cannot silently desynchronise test and config.
_CFG = PipelineConfig()
TAU_AGREE_PX = _CFG.orthogonal_gate_tau_agree
T_STRUCT = _CFG.orthogonal_gate_t_struct
OLD_T_STRUCT = 0.40                 # pre-recalibration threshold (measured unreachable)
STRUCT_BAND = (0.25, 0.40)          # measured band the new threshold must admit

CONTROL_ROTATION_DEG = 11.0         # rotation at which the old check structurally failed
CONTROL_TRANSLATION_PX = (4.0, -3.0)  # small: the identity-H residual stays inside tau
IMAGE_SIZE_PX = (256, 256)          # (w, h) of the control pair
TRANSLATION_ERROR_PX = 100.0        # deliberate wrong-translation offset (>> tau_agree)
STRUCT_NOISE_SIGMA = 55.0           # photometric noise landing struct_ncc mid-band
NOISE_SEED = 5
UNEQUAL_SRC_PX = (300, 400)         # (w, h) of the unequal-size source
UNEQUAL_REF_PX = (512, 512)         # (w, h) of the unequal-size reference
PAIR_SEED = 11


def _synthetic_lunar(size=IMAGE_SIZE_PX, seed=42) -> np.ndarray:
    """Crater-like circles on a flat background — enough local structure for both the
    phase-correlation peak and the gradient NCC (copied from the e2e generator so
    this file stays self-contained).
    """
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.zeros((h, w), dtype=np.uint8) + 120
    for _ in range(15):
        cx = int(rng.integers(20, w - 20))
        cy = int(rng.integers(20, h - 20))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r, 200, 2)          # rim
        cv2.circle(img, (cx - 2, cy - 2), r - 2, 40, -1)   # inner shadow
        cv2.circle(img, (cx + 2, cy + 2), r - 4, 150, -1)  # lit floor
    return img


def _control_pair():
    """(src, ref, H_gt): ref is src rotated by CONTROL_ROTATION_DEG plus a small
    translation. warpAffine's default convention makes H_gt the forward map from
    source pixel coordinates to reference pixel coordinates — the ground truth.
    """
    src = _synthetic_lunar(seed=PAIR_SEED)
    m = cv2.getRotationMatrix2D(
        (IMAGE_SIZE_PX[0] / 2.0, IMAGE_SIZE_PX[1] / 2.0), CONTROL_ROTATION_DEG, 1.0
    )
    m[0, 2] += CONTROL_TRANSLATION_PX[0]
    m[1, 2] += CONTROL_TRANSLATION_PX[1]
    H_gt = np.vstack([m, [0.0, 0.0, 1.0]])
    ref = cv2.warpAffine(src, m, IMAGE_SIZE_PX)
    return src, ref, H_gt


def _unequal_pair():
    """(src 300x400, ref 512x512, H_gt): unequal sizes exercise the
    S1 @ H @ inv(S0) frame mapping inside check #1.
    """
    src = _synthetic_lunar(size=UNEQUAL_SRC_PX, seed=PAIR_SEED + 1)
    sw, sh = UNEQUAL_SRC_PX
    rw, rh = UNEQUAL_REF_PX
    m = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), CONTROL_ROTATION_DEG, 1.0)
    # rotate about the source centre, then land the source centre on the reference
    # centre (+ a small translation) so the content stays inside the reference frame
    m[0, 2] += (rw - sw) / 2.0 + CONTROL_TRANSLATION_PX[0]
    m[1, 2] += (rh - sh) / 2.0 + CONTROL_TRANSLATION_PX[1]
    H_gt = np.vstack([m, [0.0, 0.0, 1.0]])
    ref = cv2.warpPerspective(src, H_gt, UNEQUAL_REF_PX)
    return src, ref, H_gt


def _add_photometric_noise(img: np.ndarray, sigma: float, seed: int = NOISE_SEED) -> np.ndarray:
    """Degrade radiometry but not geometry: the pair stays aligned for check #1 while
    the gradient maps decorrelate enough to land structural NCC mid-band.
    """
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _old_translation_disagreement(img0: np.ndarray, img1: np.ndarray, H: np.ndarray) -> float:
    """The quantity the OLD check compared: H's translation vs the raw source/reference
    phase shift. Kept here (not in verify.py) as the regression witness — at 11deg it
    exceeds tau_agree, which is exactly why correct transforms used to be rejected.
    """
    def _to_gray_256(im: np.ndarray) -> np.ndarray:
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        out = cv2.resize(im.astype(np.float32), IMAGE_SIZE_PX, interpolation=cv2.INTER_AREA)
        return out / 255.0 if out.max() > 1.0 else out

    hann = cv2.createHanningWindow(IMAGE_SIZE_PX, cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(_to_gray_256(img0), _to_gray_256(img1), window=hann)
    # both images are IMAGE_SIZE_PX here, so the 256-frame shift is already in px
    return math.hypot(float(H[0, 2]) - dx, float(H[1, 2]) - dy)


def test_ground_truth_h_passes_phase_check_despite_rotation():
    """AC Task 2: known 11deg rotation + translation, H = ground truth ->
    phase_agree_px <= tau_agree, and the gate as a whole passes."""
    src, ref, H_gt = _control_pair()

    res = gate_cheap(H_gt, src, ref)

    assert res["phase_agree_px"] is not None
    assert res["phase_agree_px"] <= TAU_AGREE_PX, res
    assert res["pass"] is True, res
    # regression witness: the old translation-only comparison rejected THIS pair
    assert _old_translation_disagreement(src, ref, H_gt) > TAU_AGREE_PX


def test_wrong_translation_exceeds_tau_agree():
    """AC Task 2: correct rotation but translation off by 100px ->
    phase_agree_px > tau_agree, gate fails check #1."""
    src, ref, H_gt = _control_pair()
    H_bad = H_gt.copy()
    H_bad[0, 2] += TRANSLATION_ERROR_PX

    res = gate_cheap(H_bad, src, ref)

    assert res["pass"] is False
    assert res["phase_agree_px"] > TAU_AGREE_PX, res
    assert res["reason"].startswith("phase_correlation_disagreement")
    # the wrong rotation is still correct, so the failure is about translation alone
    assert np.isclose(np.linalg.det(H_bad[:2, :2]), 1.0)


def test_unequal_image_sizes_handled():
    """AC Task 2: 300x400 source vs 512x512 reference — no shape errors, and check #1
    still separates the ground-truth H from a translation-offset one."""
    src, ref, H_gt = _unequal_pair()

    res_gt = gate_cheap(H_gt, src, ref)
    assert res_gt["phase_agree_px"] is not None
    assert res_gt["phase_agree_px"] <= TAU_AGREE_PX, res_gt

    H_bad = H_gt.copy()
    H_bad[0, 2] += TRANSLATION_ERROR_PX
    res_bad = gate_cheap(H_bad, src, ref)
    assert res_bad["pass"] is False
    assert res_bad["phase_agree_px"] > TAU_AGREE_PX, res_bad
    assert res_bad["reason"].startswith("phase_correlation_disagreement")


def test_struct_ncc_band_passes_new_threshold_and_failed_old_one():
    """AC Task 3: an aligned pair with struct_ncc in the measured (0.25, 0.40) band
    passes the gate now, while the pre-recalibration 0.40 threshold rejected it."""
    src, ref, H_gt = _control_pair()
    degraded_ref = _add_photometric_noise(ref, STRUCT_NOISE_SIGMA)

    res = gate_cheap(H_gt, src, degraded_ref)
    # numeric band asserted explicitly, not just "above the threshold"
    assert STRUCT_BAND[0] < res["struct_ncc"] < STRUCT_BAND[1], res
    assert res["pass"] is True, res

    old = gate_cheap(H_gt, src, degraded_ref, t_struct=OLD_T_STRUCT)
    assert old["pass"] is False
    assert old["reason"].startswith("structural_ncc_low")
    assert f"{OLD_T_STRUCT:.3f}" in old["reason"]


def test_default_t_struct_matches_config():
    """AC Task 3: gate_cheap's default agrees with the recalibrated config (0.25)."""
    default = inspect.signature(gate_cheap).parameters["t_struct"].default
    assert default == T_STRUCT == 0.25
    assert T_STRUCT == _CFG.orthogonal_gate_t_struct


def test_identity_transform_still_rejected_by_structural_ncc():
    """AC Task 3: identity/garbage H (struct_ncc well below 0.25) still fails check #3
    with the structural_ncc_low reason — the recalibration did not open that door."""
    src, ref, _ = _control_pair()
    H_identity = np.eye(3, dtype=np.float64)

    ncc = compute_structural_ncc(src, ref, H_identity)
    assert ncc < STRUCT_BAND[0]

    res = gate_cheap(H_identity, src, ref)
    assert res["pass"] is False
    assert res["reason"].startswith("structural_ncc_low")
    # check #1 passed first, so the rejection really came from check #3
    assert res["phase_agree_px"] <= TAU_AGREE_PX, res


def test_no_warp_emitted_reports_phase_agree_none():
    """API contract: early returns carry phase_agree_px = None (nothing was checked)."""
    src, ref, _ = _control_pair()

    res = gate_cheap(None, src, ref)

    assert res["pass"] is False
    assert res["reason"] == "no_warp_emitted"
    assert res["phase_agree_px"] is None


def test_budget_ms_semantics_unchanged():
    """An exhausted budget still fails open, and reports the residual that was measured."""
    src, ref, H_gt = _control_pair()

    res = gate_cheap(H_gt, src, ref, budget_ms=0.0)

    assert res["pass"] is False
    assert res["reason"].startswith("verification_budget_exceeded")
    assert res["phase_agree_px"] <= TAU_AGREE_PX, res
