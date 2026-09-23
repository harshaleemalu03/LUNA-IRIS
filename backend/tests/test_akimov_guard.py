"""Task 7: Akimov guard — prevent silent exp-B keypoint annihilation.

exp-B failure modes this guards (both produced SILENT zero-keypoint runs):

1. **Enabled without both angles**: the `pwift_akimov` branch is entered
   (OHRC/LROC sensor) but only one — or neither — of the per-pixel angle
   maps is available. Code silently computed unweighted maps (W=1), so
   callers believed Akimov weighting was active while it was inert.
2. **w_soft collapse**: uniform near-terminator angles (e.g. inc=89/em=20
   sidecar scalars expanded to maps) give akimov_weight a zero-variance raw
   w0 below `w_soft_lo` everywhere (the hi-lo<1e-4 branch skips
   normalization) -> `w_soft` is all-zero -> `pc_pw_o` (pwift.py Eq 7) and
   `Sc(x)` (Eq 9) are identically zero -> **0 keypoints, no warning**.

Guards: (1) warn + document W=1; (2) hard warning + fall back to the
unweighted path so structural energy and keypoints survive.
"""
import warnings

import cv2
import numpy as np
import pytest

from lunar_registration.pwift import (
    akimov_weight,
    detect_keypoints_pw,
    photometric_weighted_structural_maps,
)


def _structured_img(seed=3, size=64):
    """Small image with real structure so phase congruency has energy to
    form keypoints on (flat/noise images would fail for unrelated reasons)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint8) + 120
    for _ in range(8):
        cx = int(rng.integers(10, size - 10))
        cy = int(rng.integers(10, size - 10))
        r = int(rng.integers(4, 12))
        cv2.circle(img, (cx, cy), r, 200, 2)
        cv2.circle(img, (cx - 1, cy - 1), r - 2, 40, -1)
    return img.astype(np.float32) / 255.0


def _collapse_angles(shape, inc_deg=89.0, emi_deg=20.0):
    """Uniform near-terminator geometry: raw w0 is zero-variance AND below
    w_soft_lo=0.10 everywhere -> the exact exp-B collapse configuration."""
    inc = np.full(shape, inc_deg, dtype=np.float32)
    emi = np.full(shape, emi_deg, dtype=np.float32)
    return inc, emi


def test_akimov_branch_warns_without_both_angles():
    """Guard 1: partial or missing angle maps -> warn that Akimov weighting
    is INERT (W=1), instead of computing silently."""
    img = _structured_img()
    inc_only = np.full(img.shape, 60.0, dtype=np.float32)

    # partial: incidence provided, emission missing
    with pytest.warns(UserWarning, match="INERT"):
        maps = photometric_weighted_structural_maps(
            img, incidence_deg=inc_only, n_scales=2, n_orient=4)
    assert np.all(maps["w"] == 1.0)  # documented W=1 behaviour

    # absent: neither map (reference-style call, or source angles never loaded)
    with pytest.warns(UserWarning, match="INERT"):
        maps = photometric_weighted_structural_maps(
            img, n_scales=2, n_orient=4)
    assert np.all(maps["w"] == 1.0)


def test_w_soft_collapse_falls_back_to_unweighted():
    """Guard 2: exp-B collapse geometry -> hard warning + unweighted
    fallback (w_soft and M_PW alive), instead of all-zero maps."""
    img = _structured_img(seed=5)
    inc, emi = _collapse_angles(img.shape)

    # pre-condition: this IS the collapse geometry
    w_raw = akimov_weight(inc, emi)
    assert np.all(w_raw < 0.10), "expected raw weights below w_soft_lo"

    with pytest.warns(UserWarning, match="collapse"):
        maps = photometric_weighted_structural_maps(
            img, incidence_deg=inc, emission_deg=emi,
            n_scales=2, n_orient=4)

    assert np.any(maps["w_soft"] > 0)      # fell back — not annihilated
    assert np.all(maps["w"] == 1.0)        # unweighted path
    assert np.any(maps["M_PW"] > 0)        # structural energy survived


def test_keypoints_survive_forced_inc_em_collapse():
    """AC verify: a forced inc/em run no longer yields 0 source keypoints
    silently — the collapse is warned about and keypoints are found."""
    img = _structured_img(seed=7)
    inc, emi = _collapse_angles(img.shape)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        maps = photometric_weighted_structural_maps(
            img, incidence_deg=inc, emission_deg=emi,
            n_scales=2, n_orient=4)

    keypoints = detect_keypoints_pw(
        maps["M_PW"], maps["m_PW"], maps["w_soft"], maps["mask"])

    assert len(keypoints) > 0, (
        "forced inc/em collapse produced zero keypoints")
    assert [w for w in caught if "collapse" in str(w.message)], (
        "collapse must not be silent")
