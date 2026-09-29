"""Task 6: Stage 3 must hand photometrically NORMALIZED copies of
`src_scaled` / `ref.data` to the NEURAL matcher arms while the PWIFT arm
keeps receiving the raw imagery (PWIFT consumes the Stage-2 illumination
maps built from raw input).

Fake matchers stand in for roma2/pwift (no model loading), and a Stage-2
spy captures the raw arrays at the boundary where they enter illumination
correction — an independent witness of what "raw" means, without
re-running the scale/rotation search.
"""
import cv2
import numpy as np

import lunar_registration.pipeline as pl
from lunar_registration.config import PipelineConfig
from lunar_registration.matching import MatchResult


# Synthetic frame the pair and the fake correspondences are built on.
_PAIR_SIZE = (256, 256)
# Grid the fake matchers' correspondences over: comfortably inside the
# frame so photometric patch extraction (fusion) stays in-bounds.
_GRID_LO, _GRID_HI, _GRID_N = 30.0, 220.0, 8
# Distinct grid offsets keep the two arms' proposals from collapsing into
# each other at fusion's ~1 px deduplication radius.
_NEURAL_OFFSET = (6.0, 5.0)


def _synthetic_lunar_image(size=_PAIR_SIZE, seed=42):
    """Synthetic crater field — same pattern as test_pipeline_e2e.py
    (copied locally; tests never import from each other)."""
    rng = np.random.default_rng(seed)
    img = np.zeros(size, dtype=np.uint8) + 120
    for _ in range(15):
        cx = int(rng.integers(20, size[1] - 20))
        cy = int(rng.integers(20, size[0] - 20))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r, 200, 2)
        cv2.circle(img, (cx - 2, cy - 2), r - 2, 40, -1)
        cv2.circle(img, (cx + 2, cy + 2), r - 4, 150, -1)
    return img


def _related_pair(tmp_path, tag):
    """src/ref pair related by a known 2-degree / 4 px warp; returns the
    paths plus the 2x3 matrix so the fakes can emit consistent matches."""
    src = _synthetic_lunar_image(seed=11)
    M = cv2.getRotationMatrix2D((128, 128), 2.0, 1.0)
    M[0, 2] += 4.0
    M[1, 2] -= 3.0
    ref = cv2.warpAffine(src, M, _PAIR_SIZE)
    src_path = str(tmp_path / f"{tag}_src.tif")
    ref_path = str(tmp_path / f"{tag}_ref.tif")
    cv2.imwrite(src_path, src)
    cv2.imwrite(ref_path, ref)
    return src_path, ref_path, M


def _correspondences(M, offset=(0.0, 0.0)):
    """Grid points with destinations under the pair's warp: a healthy,
    RANSAC-consistent MatchResult (>= _GRID_N**2 points) so every arm
    carries real inlier support instead of random garbage."""
    axis = np.linspace(_GRID_LO, _GRID_HI, _GRID_N)
    gx, gy = np.meshgrid(axis, axis)
    pts = np.stack([gx.ravel() + offset[0], gy.ravel() + offset[1]], axis=1)
    dst = (M @ np.hstack([pts, np.ones((pts.shape[0], 1))]).T).T
    return pts.astype(np.float32), dst.astype(np.float32)


class _RecordingMatcher:
    """Fake arm: records exactly the arrays it was handed (as references,
    so identity against the raw inputs is checkable) and answers with a
    valid MatchResult built from precomputed correspondences."""

    def __init__(self, method, correspondences):
        self.method = method
        self.pts_src, self.pts_dst = correspondences
        self.received = None  # (src, ref) as handed over

    def match(self, src, ref, **kwargs):
        self.received = (src, ref)
        return MatchResult(
            method=self.method,
            pts_src=self.pts_src.copy(),
            pts_dst=self.pts_dst.copy(),
        )


def _fake_get_matcher(neural, pwift):
    """Stand-in for pipeline.get_matcher; fails loudly on anything but the
    two arms under test so no real model can be loaded unnoticed."""
    def get_matcher(name, cfg=None):
        if name == "pwift":
            return pwift
        if name in pl.NEURAL_MATCHERS:
            return neural
        raise AssertionError(f"unexpected matcher requested: {name!r}")
    return get_matcher


def _record_raw_inputs(monkeypatch):
    """Capture the RAW `src_scaled` / `ref.data` objects where they enter
    Stage 2 — which consumes them un-normalized by construction. That is
    the independent "raw" every arm's inputs are asserted against."""
    raw = {}
    real_iic = pl.apply_illumination_correction

    def spy(img, sensor, *args, reference=None, **kwargs):
        # Stage 2's first call is the source, carrying reference=ref.data
        # (the second re-passes ref.data with reference=None).
        if reference is not None:
            raw["src"], raw["ref"] = img, reference
        return real_iic(img, sensor, *args, reference=reference, **kwargs)

    monkeypatch.setattr(pl, "apply_illumination_correction", spy)
    return raw


def _run(tmp_path, monkeypatch, tag, **run_kwargs):
    """Full pipeline run on the synthetic pair with both fakes installed;
    returns (summary, neural fake, pwift fake, raw Stage-3 inputs)."""
    src_path, ref_path, M = _related_pair(tmp_path, tag)
    neural = _RecordingMatcher("roma2", _correspondences(M, offset=_NEURAL_OFFSET))
    pwift = _RecordingMatcher("pwift", _correspondences(M))
    raw = _record_raw_inputs(monkeypatch)
    monkeypatch.setattr(pl, "get_matcher", _fake_get_matcher(neural, pwift))
    summary = pl.run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=str(tmp_path / f"out_{tag}"),
        source_sensor="LROC",
        matcher="roma2",
        **run_kwargs,
    )
    return summary, neural, pwift, raw


def test_neural_arm_gets_normalized_inputs_pwift_keeps_raw(tmp_path, monkeypatch):
    """AC: default mode 'clahe' — the neural arm receives arrays that
    DIFFER from the raw inputs; the PWIFT arm receives the raw inputs."""
    assert PipelineConfig().neural_input_normalization == "clahe"  # plan default
    _, neural, pwift, raw = _run(tmp_path, monkeypatch, "grad")

    n_src, n_ref = neural.received
    p_src, p_ref = pwift.received

    # PWIFT's inputs are the raw Stage-3 arrays — same objects, not just
    # equal values (its Stage-2 illumination maps are built from them).
    assert p_src is raw["src"]
    assert p_ref is raw["ref"]

    # The neural arm sees normalized copies: shape preserved, values changed.
    assert n_src.shape == raw["src"].shape
    assert n_ref.shape == raw["ref"].shape
    assert not np.array_equal(n_src, raw["src"])
    assert not np.array_equal(n_ref, raw["ref"])


def test_none_mode_passes_the_original_arrays_through(tmp_path, monkeypatch):
    """AC: mode 'none' hands the ORIGINAL arrays to the neural arm —
    identity included, not merely value-equal copies."""
    _, neural, pwift, raw = _run(
        tmp_path, monkeypatch, "none",
        cfg=PipelineConfig(neural_input_normalization="none"),
    )

    n_src, n_ref = neural.received
    assert n_src is raw["src"]
    assert n_ref is raw["ref"]
    assert np.array_equal(n_src, raw["src"])
    assert np.array_equal(n_ref, raw["ref"])
    # PWIFT stays raw whatever the mode says.
    assert pwift.received[0] is raw["src"]
    assert pwift.received[1] is raw["ref"]


def test_run_completes_with_both_arms_recorded(tmp_path, monkeypatch):
    """AC: the wired-up run completes without raising and reports both
    evaluated arms in summary['metrics']."""
    summary, neural, pwift, _ = _run(tmp_path, monkeypatch, "done")

    assert neural.received is not None
    assert pwift.received is not None
    assert set(summary["metrics"]) >= {"roma2", "pwift"}
