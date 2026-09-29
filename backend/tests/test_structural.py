"""Tests for the dense structural-NCC correspondence generator (`structural.py`).

Builds a synthetic lunar crater pair with a known ground-truth transform
(~11 deg rotation + ~30-60px translation, scale 1.0). The source is passed
already coarse-aligned in rotation/scale - the pipeline contract - so only
translation remains for the generator to resolve.
"""

from __future__ import annotations

import cv2
import numpy as np

from lunar_registration.config import PipelineConfig
from lunar_registration.structural import structural_ncc_correspondences


# Ground-truth parameters of the synthetic pair. The source gets ONLY the
# rotation/scale (Stage-3 coarse alignment); the reference gets
# rotation/scale THEN translation, so the residual src->ref transform is
# exactly the pure translation below.
_ANGLE_DEG = 11.0
_SCALE = 1.0
_TRANSLATION = (42.0, -36.0)  # ~55px total, each component in the 30-60px band
_IMG_SIDE = 512
_CENTER = _IMG_SIDE / 2.0


def _create_synthetic_lunar_image(
    size: tuple[int, int] = (_IMG_SIDE, _IMG_SIDE),
    seed: int = 42,
    n_craters: int = 45,
) -> np.ndarray:
    """Crater-field generator (pattern copied from test_pipeline_e2e, not imported)."""
    rng = np.random.default_rng(seed)
    img = np.zeros(size, dtype=np.uint8) + 120
    for _ in range(n_craters):
        cx = int(rng.integers(30, size[1] - 30))
        cy = int(rng.integers(30, size[0] - 30))
        r = int(rng.integers(10, 40))
        cv2.circle(img, (cx, cy), r, 200, 2)                # outer rim
        cv2.circle(img, (cx - 3, cy - 3), r - 3, 40, -1)    # inner shadow
        cv2.circle(img, (cx + 3, cy + 3), max(1, r - 6), 150, -1)  # sunlit floor
    return img


def _build_pair(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (src_img, ref_img, H_gt) with H_gt a pure translation."""
    base = _create_synthetic_lunar_image(seed=seed)
    rot = cv2.getRotationMatrix2D((_CENTER, _CENTER), _ANGLE_DEG, _SCALE)
    rot3 = np.vstack([rot, [0.0, 0.0, 1.0]])
    tx, ty = _TRANSLATION
    h_gt = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    full = (h_gt @ rot3)[:2]  # rotation/scale applied first, then translation

    src = cv2.warpAffine(base, rot, (_IMG_SIDE, _IMG_SIDE))
    ref = cv2.warpAffine(base, full, (_IMG_SIDE, _IMG_SIDE))
    return src, ref, h_gt


def _median_error(h: np.ndarray, pts_src: np.ndarray, pts_dst: np.ndarray) -> float:
    """Median Euclidean error of pts_src mapped by h against pts_dst."""
    proj = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), h).reshape(-1, 2)
    return float(np.median(np.linalg.norm(proj - pts_dst, axis=1)))


def test_structural_yields_dense_correspondences():
    src, ref, _ = _build_pair()
    cfg = PipelineConfig()
    res = structural_ncc_correspondences(src, ref, cfg)

    assert res.method == "structural_ncc"
    assert res.provenance == "structural_ncc"

    n = len(res.pts_src)
    assert n >= 50
    assert res.pts_src.shape == (n, 2)
    assert res.pts_dst.shape == (n, 2)
    assert res.scores is not None and len(res.scores) == n
    assert np.all(np.isfinite(res.pts_src))
    assert np.all(np.isfinite(res.pts_dst))
    assert np.all(res.scores >= cfg.structural_min_ncc)

    # >= 50% of grid cells covered: map each dst point back to its cell.
    grid = cfg.structural_grid
    h, w = ref.shape[:2]
    cols = np.clip((res.pts_dst[:, 0] * grid / w).astype(int), 0, grid - 1)
    rows = np.clip((res.pts_dst[:, 1] * grid / h).astype(int), 0, grid - 1)
    covered = len(set(zip(rows.tolist(), cols.tolist())))
    assert covered >= 0.5 * grid * grid


def test_structural_round_trip_recovers_ground_truth():
    src, ref, h_gt = _build_pair()
    cfg = PipelineConfig()
    res = structural_ncc_correspondences(src, ref, cfg)
    assert len(res.pts_src) >= cfg.structural_min_correspondences

    h_est, _inliers = cv2.findHomography(res.pts_src, res.pts_dst, cv2.RANSAC, 3.0)
    assert h_est is not None

    # Median reprojection error of the estimated homography, and of the
    # ground-truth mapping, on the returned correspondences.
    assert _median_error(h_est, res.pts_src, res.pts_dst) < 5.0
    assert _median_error(h_gt, res.pts_src, res.pts_dst) < 5.0

    # Recovered translation is within ~2x the search radius of the truth.
    radius = 2 * cfg.structural_search_radius_px
    assert abs(float(h_est[0, 2]) - h_gt[0, 2]) <= radius
    assert abs(float(h_est[1, 2]) - h_gt[1, 2]) <= radius


def test_structural_flat_pair_returns_empty():
    flat_a = np.full((256, 256), 128, np.uint8)
    flat_b = np.full((256, 256), 90, np.uint8)
    cfg = PipelineConfig()

    res = structural_ncc_correspondences(flat_a, flat_b, cfg)  # must not raise

    assert len(res.pts_src) < cfg.structural_min_correspondences
    assert res.pts_src.shape == (0, 2)
    assert res.pts_dst.shape == (0, 2)
    assert res.scores is not None and res.scores.shape == (0,)
    assert np.all(np.isfinite(res.pts_src))
    assert np.all(np.isfinite(res.pts_dst))
    assert np.all(np.isfinite(res.scores))


def test_structural_grid_knob_caps_correspondences():
    src, ref, _ = _build_pair()
    cfg = PipelineConfig()
    cfg.structural_grid = 4

    res = structural_ncc_correspondences(src, ref, cfg)

    # One best peak per cell at most: 4 x 4 grid -> at most 16 points.
    assert len(res.pts_src) <= 16


def test_structural_min_ncc_knob_rejects_hard_pair():
    # Two unrelated crater fields: repetitive terrain still produces plenty
    # of >= 0.2 peaks at the default threshold, but nothing can reach 0.99.
    pair_a = _create_synthetic_lunar_image(seed=1)
    pair_b = _create_synthetic_lunar_image(seed=999)

    cfg = PipelineConfig()
    default_count = len(structural_ncc_correspondences(pair_a, pair_b, cfg).pts_src)
    assert default_count >= cfg.structural_min_correspondences

    cfg.structural_min_ncc = 0.99
    res = structural_ncc_correspondences(pair_a, pair_b, cfg)
    assert len(res.pts_src) < cfg.structural_min_correspondences


def test_structural_unequal_image_sizes():
    base = _create_synthetic_lunar_image(seed=5)
    crop = base[40:340, 60:460].copy()  # 300x400 vs 512x512 reference

    res = structural_ncc_correspondences(crop, base, PipelineConfig())  # must not raise

    n = len(res.pts_src)
    assert n > 0  # the phase-correlation prior handles the size mismatch
    assert res.pts_src.shape == (n, 2)
    assert res.pts_dst.shape == (n, 2)
    assert np.all(np.isfinite(res.pts_src))
    assert np.all(np.isfinite(res.pts_dst))
    assert np.all(np.isfinite(res.scores))


def test_structural_tiny_and_empty_input_guards():
    ref = _create_synthetic_lunar_image(seed=2)
    cfg = PipelineConfig()

    # Images smaller than a few grid cells, and empty arrays: no exception.
    tiny = np.zeros((8, 8), dtype=np.uint8)
    assert len(structural_ncc_correspondences(tiny, ref, cfg).pts_src) == 0
    empty = np.zeros((0, 0), dtype=np.uint8)
    assert len(structural_ncc_correspondences(empty, ref, cfg).pts_src) == 0
    assert len(structural_ncc_correspondences(ref, empty, cfg).pts_src) == 0
