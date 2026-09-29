"""
Tests for lunar_registration.photometric - matcher-input normalization.

Covers the Task 4 acceptance criteria: the dtype/shape/finiteness contract
for every mode, no mutation of the input, the "none" pass-through,
illumination-shift invariance (the point of the feature), colour and uint8
inputs, the unknown-mode error, degenerate (flat) input safety, and the
config dispatcher used by the pipeline later.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lunar_registration.config import PipelineConfig
from lunar_registration.photometric import (
    VALID_MODES,
    normalize_for_matching,
    normalize_for_matching_cfg,
)

MODES = ("gradient", "weber", "clahe", "none")
ACTIVE_MODES = ("gradient", "weber", "clahe")

# Illumination change applied to the base image: a nonlinear sensor/exposure
# response, an exposure-gain difference, and a left-to-right shading ramp -
# the ways the two acquisitions on the canonical pair differ photometrically.
EXPOSURE_GAMMA = 2.2
EXPOSURE_GAIN = 0.85
EXPOSURE_OFFSET = 0.10
SHADE_RAMP_LO = 0.60
SHADE_RAMP_HI = 1.40

# Required mutual-NCC improvement of the normalized pair over the raw pair.
NCC_MARGIN = 0.05

# Flat/constant image used for the degenerate-input guards.
FLAT_VALUE_U8 = 137


def _synthetic_lunar_image(size=(256, 256), seed: int = 10) -> np.ndarray:
    """Unit-range float copy of the crater generator from test_pipeline_e2e.

    Copied locally on purpose: tests must not import from other test modules.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros(size, dtype=np.uint8) + 120
    for _ in range(15):
        cx = int(rng.integers(20, size[1] - 20))
        cy = int(rng.integers(20, size[0] - 20))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r, 200, 2)              # outer rim
        cv2.circle(img, (cx - 2, cy - 2), r - 2, 40, -1)   # inner shadow
        cv2.circle(img, (cx + 2, cy + 2), r - 4, 150, -1)  # sunlight floor
    return img.astype(np.float32) / 255.0


def _illumination_shifted(img: np.ndarray) -> np.ndarray:
    """Re-derive `img` under a different exposure/sun-angle: gamma response,
    exposure gain and a shading ramp across the frame - the photometric
    difference the normalization is supposed to remove."""
    height, width = img.shape[:2]
    ramp = np.linspace(SHADE_RAMP_LO, SHADE_RAMP_HI, width, dtype=np.float32)
    shifted = np.power(img, EXPOSURE_GAMMA) * EXPOSURE_GAIN * ramp[np.newaxis, :]
    shifted = shifted + EXPOSURE_OFFSET
    return np.clip(shifted, 0.0, 1.0).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation of two equally sized images."""
    av = a.astype(np.float64).ravel()
    bv = b.astype(np.float64).ravel()
    av -= av.mean()
    bv -= bv.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0.0:
        return 0.0
    return float(av @ bv / denom)


def _mutual_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual (both directions) NCC: the worse of the two directional scores."""
    return min(_ncc(a, b), _ncc(b, a))


@pytest.mark.parametrize("mode", MODES)
def test_modes_are_float32_same_shape_finite_and_do_not_mutate(mode):
    img = _synthetic_lunar_image()
    before = img.copy()

    out = normalize_for_matching(img, mode)

    assert out.dtype == np.float32
    assert out.shape == img.shape
    assert np.isfinite(out).all()
    if mode in ACTIVE_MODES:
        assert out.min() >= 0.0
        assert out.max() <= 1.0
    # The caller's array must be untouched (pipeline reuses it later).
    np.testing.assert_array_equal(img, before)


def test_none_returns_unchanged_copy():
    img = _synthetic_lunar_image()
    out = normalize_for_matching(img, "none")

    np.testing.assert_array_equal(out, img)  # values identical to the input
    assert out.dtype == np.float32
    assert out is not img
    assert not np.shares_memory(out, img)  # a copy, not a view

    # uint8 input: identical values, still cast to float32, input untouched.
    u8 = np.rint(img * 255.0).astype(np.uint8)
    u8_before = u8.copy()
    out_u8 = normalize_for_matching(u8, "none")
    np.testing.assert_array_equal(out_u8, u8)
    assert out_u8.dtype == np.float32
    np.testing.assert_array_equal(u8, u8_before)


@pytest.mark.parametrize("mode", ["gradient", "weber"])
def test_normalization_beats_raw_ncc_under_illumination_change(mode):
    base = _synthetic_lunar_image()
    shifted = _illumination_shifted(base)

    raw_ncc = _mutual_ncc(base, shifted)
    norm_ncc = _mutual_ncc(
        normalize_for_matching(base, mode),
        normalize_for_matching(shifted, mode),
    )

    assert norm_ncc > raw_ncc + NCC_MARGIN, (
        f"{mode}: normalized mutual NCC {norm_ncc:.4f} should beat raw "
        f"{raw_ncc:.4f} by at least {NCC_MARGIN}"
    )


@pytest.mark.parametrize("mode", ACTIVE_MODES)
def test_uint8_and_colour_inputs_are_accepted_and_normalized(mode):
    base = _synthetic_lunar_image()
    base_u8 = np.rint(base * 255.0).astype(np.uint8)
    colour_f32 = np.stack([base, np.roll(base, 9, axis=1), np.flipud(base)], axis=2)
    colour_u8 = np.stack([base_u8, np.roll(base_u8, 9, axis=1), np.flipud(base_u8)], axis=2)

    for img in (base_u8, colour_f32, colour_u8):
        before = img.copy()
        out = normalize_for_matching(img, mode)

        assert out.dtype == np.float32
        assert out.shape == img.shape
        assert np.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        np.testing.assert_array_equal(img, before)


def test_unknown_mode_raises_value_error_naming_valid_modes():
    img = _synthetic_lunar_image()

    with pytest.raises(ValueError) as exc:
        normalize_for_matching(img, "equalize")

    message = str(exc.value)
    assert "valid modes" in message
    for mode in VALID_MODES:
        assert mode in message


@pytest.mark.parametrize("mode", MODES)
def test_flat_image_returns_finite_output(mode):
    flat_u8 = np.full((64, 64), FLAT_VALUE_U8, dtype=np.uint8)
    flat_zero = np.zeros((64, 64), dtype=np.float32)

    for img in (flat_u8, flat_zero):
        out = normalize_for_matching(img, mode)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert not np.isnan(out).any()

    # A constant image has no edges: gradient mode must fall back to zeros
    # instead of dividing by its (zero) percentile.
    grad = normalize_for_matching(flat_u8, "gradient")
    assert np.count_nonzero(grad) == 0


def test_cfg_dispatcher_reads_neural_input_normalization():
    img = _synthetic_lunar_image()

    cfg = PipelineConfig()
    assert cfg.neural_input_normalization == "clahe"  # plan default
    np.testing.assert_array_equal(
        normalize_for_matching_cfg(img, cfg),
        normalize_for_matching(img, "clahe"),
    )

    cfg.neural_input_normalization = "gradient"
    np.testing.assert_array_equal(
        normalize_for_matching_cfg(img, cfg),
        normalize_for_matching(img, "gradient"),
    )
