"""Verification bar for the whole pipeline (TODOs.md, final section).

The bar has three clauses; this file covers the one that is reproducible in a
unit test:

  "Synthetic control (known 11 degree rotation): recovers rotation +-0.5 deg,
   passes gate."

Why the control is not just craters: a crater-only field is self-similar -
crater centres are ambiguous by a few pixels, and that bias alone lands the
recovered rotation at ~0.52 deg, i.e. the measurement would be testing crater
ambiguity rather than rotation recovery. The control therefore mixes craters
with a non-repeating multi-scale structure field so the correspondences are
unambiguous (measured: 113/121 inliers, RMSE 0.99px, rotation error 0.19 deg).

The other two clauses need the real data set or a deliberately sabotaged run
and are covered by test_structural_pipeline.py / test_fail_closed.py:
  - canonical pair exits non-zero with an honest reason (test_fail_closed),
  - no run exits 0 with passed:false (same file).
"""

import math

import cv2
import numpy as np
import pytest

from lunar_registration.pipeline import run_pipeline
from lunar_registration.scale import apply_rotation, apply_scale

CONTROL_ROTATION_DEG = 11.0
CONTROL_TRANSLATION_PX = (27.0, -19.0)
CONTROL_SIZE = 512
ROTATION_TOLERANCE_DEG = 0.5


def _control_source(seed: int = 42) -> np.ndarray:
    """Craters plus a non-repeating multi-scale structure field (see module
    docstring for why the field is needed)."""
    rng = np.random.default_rng(seed)
    img = np.full((CONTROL_SIZE, CONTROL_SIZE), 120, np.float32)
    for scale, spread in ((4, 45.0), (10, 30.0), (24, 20.0)):
        field = rng.standard_normal((CONTROL_SIZE // scale, CONTROL_SIZE // scale))
        field = cv2.resize(field, (CONTROL_SIZE, CONTROL_SIZE),
                           interpolation=cv2.INTER_CUBIC)
        field = field / max(1e-6, float(np.abs(field).std()))
        img += np.clip(field, -2.5, 2.5) * rng.uniform(0.6 * spread, spread) / 2.5
    for _ in range(45):
        cx = int(rng.integers(30, CONTROL_SIZE - 30))
        cy = int(rng.integers(30, CONTROL_SIZE - 30))
        r = int(rng.integers(8, 34))
        cv2.circle(img, (cx, cy), r, 200, 2)
        cv2.circle(img, (cx - 3, cy - 3), r - 3, 40, -1)
        cv2.circle(img, (cx + 3, cy + 3), r - 5, 150, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _coarse_push(shape, scale: float, rot: float) -> np.ndarray:
    """Forward 3x3 affine of ``apply_rotation(apply_scale(x, s), r)``.

    The pipeline applies that pair to the source array with scipy (no matrix
    exists anywhere in the pipeline), so the control test recovers it numerically:
    transform two coordinate planes the same way and least-squares fit the
    dst->src map, then invert it.

    scipy's ``mode='nearest'`` padding makes the outer band of the rotated
    planes clamp instead of follow the affine, so the fit rejects points it
    cannot explain and refits on the inliers (two passes is enough: the interior
    fit is exact).
    """
    h, w = shape
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    x2 = apply_rotation(apply_scale(xs, scale), rot)
    y2 = apply_rotation(apply_scale(ys, scale), rot)
    h2, w2 = x2.shape

    dst, src = [], []
    for yy in range(16, h2 - 16, 8):
        for xx in range(16, w2 - 16, 8):
            dst.append([xx, yy, 1.0])
            src.append([x2[yy, xx], y2[yy, xx]])
    dst = np.asarray(dst)
    src = np.asarray(src)

    pull, *_ = np.linalg.lstsq(dst, src, rcond=None)
    for _ in range(2):
        inlier = np.linalg.norm(dst @ pull - src, axis=1) < 0.5
        if inlier.all() or int(inlier.sum()) < 12:
            break
        pull, *_ = np.linalg.lstsq(dst[inlier], src[inlier], rcond=None)
    return np.linalg.inv(np.vstack([pull.T, [0.0, 0.0, 1.0]]))


def _rotation_deg(M: np.ndarray) -> float:
    """Rotation component (proper, via SVD) of a linear map, in degrees."""
    U, _, Vt = np.linalg.svd(M[:2, :2])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def test_synthetic_control_recovers_rotation_and_passes_gate(tmp_path):
    src = _control_source()
    M = cv2.getRotationMatrix2D((CONTROL_SIZE / 2, CONTROL_SIZE / 2),
                                CONTROL_ROTATION_DEG, 1.0)
    M[:, 2] += np.asarray(CONTROL_TRANSLATION_PX)
    ref = cv2.warpAffine(src, M, (CONTROL_SIZE, CONTROL_SIZE))

    src_path = str(tmp_path / "control_src.png")
    ref_path = str(tmp_path / "control_ref.png")
    cv2.imwrite(src_path, src)
    cv2.imwrite(ref_path, ref)

    summary = run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=str(tmp_path / "control_out"),
        source_sensor="LROC",
        matcher="pwift",
    )

    # --- passes gate for a REAL reason (never "disabled" / "skipped") ---
    gate = summary["orthogonal_gate"]
    assert summary["passed"] is True, summary["failure_reason"]
    assert summary["failure_reason"] is None
    assert gate["enabled"] is True
    assert gate["passed"] is True
    assert gate["reason"] == "all_checks_passed"
    assert gate["struct_ncc"] >= 0.25

    # --- the summary carries the transform the gate verified ---
    assert summary["homography"] is not None
    assert summary["structural_correspondences"]["attempted"] is False

    # --- rotation recovery: coarse stage then the matcher's residual ---
    H = np.asarray(summary["homography"], dtype=float)
    coarse = _coarse_push(src.shape, summary["chosen_scale"],
                          summary["chosen_rotation_deg"])
    recovered = H @ coarse

    truth = np.vstack([M, [0.0, 0.0, 1.0]])
    error_deg = abs(_rotation_deg(recovered) - _rotation_deg(truth))
    assert error_deg <= ROTATION_TOLERANCE_DEG, (
        f"recovered {_rotation_deg(recovered):+.3f} deg vs ground truth "
        f"{_rotation_deg(truth):+.3f} deg -> {error_deg:.3f} deg error, "
        f"bar is {ROTATION_TOLERANCE_DEG} deg"
    )


def test_control_measurement_machinery_is_sound():
    """Guard the measurement itself: the coarse stage is a scipy call with no
    matrix anywhere in the pipeline, so the control test recovers it from
    coordinate planes. Two things must hold or the rotation number proves
    nothing:

    1. that map really is affine - fitted on one set of destination samples,
       it must predict HELD-OUT destination samples to sub-pixel accuracy
       (interior only: scipy's 'nearest' padding makes the outer band clamp,
       and _coarse_push rejects exactly those points);
    2. the recovered push must undo the angle that was requested (scipy's
       sign convention is easy to get backwards, which would flip the
       composition used by the bar test).
    """
    scale, rot = 1.0, 15.0
    xs = np.tile(np.arange(CONTROL_SIZE, dtype=np.float32), (CONTROL_SIZE, 1))
    ys = np.repeat(np.arange(CONTROL_SIZE, dtype=np.float32)[:, None],
                   CONTROL_SIZE, axis=1)
    x2 = apply_rotation(apply_scale(xs, scale), rot)
    y2 = apply_rotation(apply_scale(ys, scale), rot)
    lo = int(0.15 * CONTROL_SIZE)
    hi = CONTROL_SIZE - lo

    def sample(x_off, y_off):
        """Padding-free destination samples on a grid offset from the fit grid."""
        pts, vals = [], []
        for yy in range(lo + y_off, hi, 16):
            for xx in range(lo + x_off, hi, 16):
                pts.append([xx, yy, 1.0])
                vals.append([x2[yy, xx], y2[yy, xx]])
        return np.asarray(pts), np.asarray(vals)

    fit_dst, fit_src = sample(0, 0)
    hold_dst, hold_src = sample(8, 8)
    pull, *_ = np.linalg.lstsq(fit_dst, fit_src, rcond=None)
    predicted = hold_dst @ pull
    error = np.linalg.norm(predicted - hold_src, axis=1)
    assert float(np.median(error)) < 0.1, (
        f"coarse stage is not affine / fit is bad: held-out median "
        f"{np.median(error):.3f}px"
    )
    assert float(np.max(error)) < 0.5, (
        f"held-out max {np.max(error):.3f}px - coordinate-plane model diverges"
    )

    push = _coarse_push((CONTROL_SIZE, CONTROL_SIZE), scale, rot)
    assert abs(_rotation_deg(push) + rot) < 0.2, (
        f"push rotation {_rotation_deg(push):+.3f}deg does not mirror the "
        f"requested {rot:+.3f}deg - scipy sign convention assumption broken"
    )


@pytest.mark.parametrize("angle", (11.0, -11.0))
def test_ground_truth_convention_is_the_opencv_one(angle):
    """warpAffine moves content from p to M@p - the convention the recovery
    arithmetic in this file depends on (verified against a landmark)."""
    img = np.zeros((CONTROL_SIZE, CONTROL_SIZE), np.uint8)
    cv2.circle(img, (100, 100), 6, 255, -1)
    M = cv2.getRotationMatrix2D((256, 256), angle, 1.0)
    ys, xs = np.nonzero(cv2.warpAffine(img, M, (CONTROL_SIZE, CONTROL_SIZE)))
    moved = np.array([xs.mean(), ys.mean()])
    predicted = (M @ np.array([100.0, 100.0, 1.0]))[:2]
    assert np.allclose(moved, predicted, atol=1.0)
