"""
Photometric normalization of matcher inputs (pre-neural-matching)
================================================================
Why this exists: a deep matcher (RoMa2 / EfficientLoFTR) scores
correspondences from local appearance, but on the canonical lunar pair the
source and the reference were acquired under different exposure and
sun-angle conditions. The same terrain therefore arrives with different
absolute brightness, different contrast, and a smooth brightness ramp
across the frame - and a network trained on photometrically consistent
pairs degrades badly (measured on the canonical pair: EfficientLoFTR
collapsed to 5 inliers).

This module strips the illumination component from the image BEFORE the
matcher sees it, so the network only has to explain geometry. It is
deliberately NOT Stage 2 (`illumination.py`): Stage 2 builds the PWIFT
structural representation for the classical arm, while this module only
reshapes the plain image arrays the neural arms consume (pipeline Stage 3),
selected by `PipelineConfig.neural_input_normalization`.

Contract: the input may be 2-D grayscale or 3-channel, uint8 or float; the
output is float32 with the same shape as the input and all values finite,
and the caller's array is never modified. For 3-channel input the active
modes normalize the image's luma and replicate it across the channels so
the shape is preserved (matching.py's `_prepare_gray_tensor` reduces
EfficientLoFTR input to luma; RoMa2's `_prepare_tensor` accepts either).
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig


# ---- valid modes (public: callers and tests validate against this) ----
VALID_MODES: tuple[str, ...] = ("gradient", "weber", "clahe", "none")

# ---- luma reduction for 3-channel input (BT.601, same weights matching.py
# applies when it reduces matcher input to grayscale) ----
LUMA_R: float = 0.299
LUMA_G: float = 0.587
LUMA_B: float = 0.114

# ---- value-range handling ----
U8_MAX: float = 255.0         # 8-bit rendering used by the "clahe" mode
FLOAT_UNIT_MAX: float = 1.0   # float values at/below this are treated as [0, 1]

# ---- "gradient" mode ----
GRADIENT_SOBEL_KSIZE: int = 3     # 3x3 Sobel: standard, less noise-amplifying
GRADIENT_PERCENTILE: float = 99.5  # robust scale - hot/dead pixels can't own it
GRADIENT_MIN_SCALE: float = 1e-6   # below this the image has no edges at all

# ---- "weber" mode ----
# 31x31 kernel (cv2 derives sigma ~= 5 px from it): wide enough that the
# local mean estimates illumination rather than texture, narrow enough to
# stay local across a crater rim.
WEBER_KERNEL_SIZE: int = 31
WEBER_EPS: float = 1e-3       # keeps the denominator > 0 in pitch-black shadow
WEBER_SATURATION: float = 1.0  # Weber ratio beyond +/-1 saturates to 0 / 1

# ---- "clahe" mode ----
CLAHE_CLIP_LIMIT: float = 2.0  # same defaults illumination.correct_iirs uses
CLAHE_TILE_GRID: int = 8       # 8x8 tiles: local enough for uneven terrain


def normalize_for_matching(img: np.ndarray, mode: str = "gradient") -> np.ndarray:
    """Normalize `img` for a deep matcher according to `mode`.

    WHY each mode exists - the illumination problem it addresses:

    - "gradient": Sobel gradient magnitude of the grayscale image, divided
      by its own 99.5th percentile and clipped to [0, 1]. A global
      brightness OFFSET does not change gradients at all, and a global
      gain cancels out in the percentile division, so brightness and
      contrast differences between source and reference are removed by
      construction and only edge structure reaches the matcher. This is
      the measured best mode on the canonical pair (EfficientLoFTR
      inliers 5 -> 25-29).
    - "weber": local contrast ``(x - local_mean) / (local_mean + eps)``
      mapped into [0, 1]. Weber contrast is a RATIO, so a multiplicative
      illumination change (sun angle / exposure gain, including a smooth
      shading ramp across the frame) cancels out - the dominant
      photometric difference between the two acquisitions.
    - "clahe": contrast-limited adaptive histogram equalization on the
      8-bit rendering, back to [0, 1]. For pairs whose global histograms
      barely overlap, each local tile is stretched onto a comparable
      contrast scale; the clip limit stops that stretch from amplifying
      sensor noise in flat terrain.
    - "none": the raw input as an unchanged float32 copy - the opt-out
      for callers that must keep raw imagery (the PWIFT arm always does).

    Parameters
    ----------
    img:
        2-D grayscale or 3-channel image, uint8 or float. Floats are
        expected in [0, 1] (repo convention); [0, 255] floats are
        tolerated and rescaled instead of being clipped to a constant.
    mode:
        One of `VALID_MODES`; defaults to "gradient" (measured best).

    Returns
    -------
    float32 array with the same shape as `img` and finite values. The
    active modes return values in [0, 1] and never modify `img`.

    Raises
    ------
    ValueError
        Unknown `mode` (message names the valid modes), or an image that
        is neither 2-D nor 3-channel.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown normalization mode {mode!r}; valid modes: {', '.join(VALID_MODES)}"
        )

    src = np.asarray(img)
    # Operate on a float32 COPY: the pipeline keeps feeding the same arrays
    # to later stages (PWIFT maps are built from the raw imagery), so the
    # caller's array must never be modified here.
    work = np.array(src, dtype=np.float32, copy=True)
    # Keep the all-finite contract even for degenerate callers: NaN/inf
    # pixels collapse to the unit-range extremes instead of propagating.
    np.nan_to_num(work, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    if mode == "none":
        # "none" = raw input: pass through untouched, values unchanged.
        return work

    # ---- shared preparation: reduce to a unit-range grayscale copy ----
    if work.ndim == 2:
        gray = work
        n_channels = 0
    elif work.ndim == 3 and work.shape[2] in (1, 3):
        n_channels = int(work.shape[2])
        if n_channels == 3:
            gray = LUMA_R * work[..., 0] + LUMA_G * work[..., 1] + LUMA_B * work[..., 2]
        else:
            gray = work[..., 0]
    else:
        raise ValueError(
            f"Expected a 2-D grayscale or 3-channel image, got shape {work.shape}"
        )

    # Scale into [0, 1]: integer dtypes are unambiguous (uint8 -> 255,
    # uint16 -> 65535); floats follow the repo's unit-range convention but
    # tolerate the common cv2 [0, 255] float convention as well.
    if np.issubdtype(src.dtype, np.integer):
        value_scale = float(np.iinfo(src.dtype).max)
    else:
        value_scale = 1.0 if float(gray.max()) <= FLOAT_UNIT_MAX else U8_MAX
    gray = np.clip(gray / value_scale, 0.0, 1.0)

    # ---- mode dispatch ----
    if mode == "gradient":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=GRADIENT_SOBEL_KSIZE)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=GRADIENT_SOBEL_KSIZE)
        # np.hypot rather than cv2.magnitude: cv2's IPP-backed magnitude is
        # not bit-stable run-to-run here (1-ulp jitter), and callers/tests
        # compare normalized outputs for equality.
        magnitude = np.hypot(gx, gy)
        # p99.5 rather than max: one specular pixel must not set the scale.
        normalizer = float(np.percentile(magnitude, GRADIENT_PERCENTILE))
        if normalizer < GRADIENT_MIN_SCALE:
            # Flat/constant image: no edges at all -> zeros, never 0/0 -> NaN.
            out = np.zeros_like(gray)
        else:
            out = np.clip(magnitude / normalizer, 0.0, 1.0)
    elif mode == "weber":
        local_mean = cv2.GaussianBlur(gray, (WEBER_KERNEL_SIZE, WEBER_KERNEL_SIZE), sigmaX=0)
        # gray >= 0 and WEBER_EPS > 0, so the denominator is never 0.
        ratio = (gray - local_mean) / (local_mean + WEBER_EPS)
        # Monotone map of the signed Weber ratio onto [0, 1]:
        # <= -WEBER_SATURATION -> 0, 0 -> 0.5, >= +WEBER_SATURATION -> 1.
        out = np.clip(0.5 + 0.5 * (ratio / WEBER_SATURATION), 0.0, 1.0)
    else:
        # mode == "clahe" - validated against VALID_MODES above.
        gray_u8 = (gray * U8_MAX).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=(CLAHE_TILE_GRID, CLAHE_TILE_GRID),
        )
        out = clahe.apply(gray_u8).astype(np.float32) / U8_MAX

    out = out.astype(np.float32, copy=False)
    if n_channels:
        # Preserve the input shape: the normalized luma is what the matcher
        # consumes, so carry it in every channel instead of dropping them.
        out = np.repeat(out[..., np.newaxis], n_channels, axis=2)
    return out


def normalize_for_matching_cfg(img: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Normalize `img` with the mode selected by `cfg`.

    Stage-3 callers hold a `PipelineConfig`, not a mode string, so this
    makes the eventual pipeline wiring a single line that reads
    `cfg.neural_input_normalization` (default "gradient", the measured
    best mode).
    """
    return normalize_for_matching(img, cfg.neural_input_normalization)
