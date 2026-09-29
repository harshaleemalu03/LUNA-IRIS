"""Task 8: structural-NCC last-resort arm, pipeline level (Stage 4).

Covers the trigger contract (attempted ONLY when no matcher arm produced a
transform), the shared estimation path (the arm's correspondences travel
the same RANSAC -> reprojection cleanup -> metrics -> compete_rigid -> gate
route and are judged honestly), the report block present on every run, and
the two miss paths (config switch off / generator starved) that must leave
the run behaving exactly as it does without the arm.

Crater-generator pattern copied locally from test_pipeline_e2e.py — tests
never import from each other.
"""
import json
import os

import cv2
import numpy as np

import lunar_registration.pipeline as pl
from lunar_registration.config import PipelineConfig
from lunar_registration.matching import MatchResult


_PAIR_SIZE = (256, 256)
_CENTER = (128, 128)
# Known ground-truth warp of the synthetic pair: 2 deg rotation about the
# centre plus a (4, -3) px translation.
_ROT_DEG = 2.0
_TRANSLATION = (4.0, -3.0)
# Garbage correspondences: >= 4 points so they ENTER estimation (the
# results pool is never empty -> no RuntimeError) while remaining
# RANSAC-unsupportable.
_GARBAGE_N = 24
_GARBAGE_SEED = 7
# Summary block contract (shape asserted on every run below).
_REPORT_KEYS = {"attempted", "n_correspondences", "used", "coverage"}


def _create_synthetic_lunar_image(size=_PAIR_SIZE, seed=42):
    """Synthetic crater field (pattern copied from test_pipeline_e2e.py)."""
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
    """(src_path, ref_path) of a pair related by the known warp above."""
    src = _create_synthetic_lunar_image(seed=11)
    M = cv2.getRotationMatrix2D(_CENTER, _ROT_DEG, 1.0)
    M[0, 2] += _TRANSLATION[0]
    M[1, 2] += _TRANSLATION[1]
    ref = cv2.warpAffine(src, M, _PAIR_SIZE)
    src_path = str(tmp_path / f"{tag}_src.tif")
    ref_path = str(tmp_path / f"{tag}_ref.tif")
    cv2.imwrite(src_path, src)
    cv2.imwrite(ref_path, ref)
    return src_path, ref_path


class _ZeroMatcher:
    """Sabotaged arm: every requested matcher answers with ZERO points, so
    no arm can ever produce a transform (the AC's '0 usable points')."""

    def __init__(self, method):
        self.method = method

    def match(self, src, ref, **kwargs):
        return MatchResult(
            method=self.method,
            pts_src=np.zeros((0, 2), np.float32),
            pts_dst=np.zeros((0, 2), np.float32),
        )


class _GarbageMatcher:
    """Arm returning uncorrelated random points: they enter estimation but
    no homography supports them."""

    def __init__(self, method):
        self.method = method

    def match(self, src, ref, **kwargs):
        rng = np.random.default_rng(_GARBAGE_SEED)
        h, w = src.shape[:2]
        pts_src = rng.uniform(0, 1, size=(_GARBAGE_N, 2)) * np.array([w, h])
        pts_dst = rng.uniform(0, 1, size=(_GARBAGE_N, 2)) * np.array([w, h])
        return MatchResult(
            method=self.method,
            pts_src=pts_src.astype(np.float32),
            pts_dst=pts_dst.astype(np.float32),
        )


def _sabotage_matchers(monkeypatch, matcher_cls):
    """Install `matcher_cls` as the factory for EVERY requested arm."""
    monkeypatch.setattr(
        pl, "get_matcher", lambda name, cfg=None: matcher_cls(name))


def _stub_no_transform(monkeypatch):
    """Estimator stub: no hypothesis ever carries a homography.

    WHY: makes 'no arm produced a transform' deterministic instead of
    leaving it to RANSAC's luck on random points (the same stub pattern
    test_fail_closed.py uses for its no-transform scenario).
    """
    from lunar_registration.viewpoint import HomographyResult

    monkeypatch.setattr(
        pl, "estimate_viewpoint_transform",
        lambda pts_src, *a, **k: HomographyResult(
            H=None, inlier_mask=np.zeros(len(pts_src), dtype=bool)),
    )


def _run(tmp_path, tag, matcher="pwift", **run_kwargs):
    src_path, ref_path = _related_pair(tmp_path, tag)
    out_dir = str(tmp_path / f"out_{tag}")
    summary = pl.run_pipeline(
        source_path=src_path, reference_path=ref_path, out_dir=out_dir,
        source_sensor="LROC", matcher=matcher, **run_kwargs,
    )
    return summary, out_dir


# ---------------------------------------------------------------------------
# 1. Fires when the matcher arms produce nothing at all
# ---------------------------------------------------------------------------

def test_structural_arm_rescues_a_run_with_zero_matcher_points(tmp_path, monkeypatch):
    """AC: zero-point matchers -> the arm is attempted AND used, a transform
    is estimated from structural correspondences, and the run reaches the
    gate (i.e. gets past the old 'no transform' dead end)."""
    _sabotage_matchers(monkeypatch, _ZeroMatcher)
    summary, out_dir = _run(tmp_path, "zeros")

    rep = summary["structural_correspondences"]
    cfg = PipelineConfig()
    print("\nstructural report:", rep)
    print("gate:", summary["orthogonal_gate"])
    print("passed:", summary["passed"], "| failure:", summary["failure_reason"])
    print("metrics arms:", summary["metrics"])

    assert set(rep) == _REPORT_KEYS
    assert rep["attempted"] is True
    assert rep["used"] is True
    assert rep["n_correspondences"] >= cfg.structural_min_correspondences
    assert rep["coverage"] > 0.0

    # The run got past "no transform": a transform WAS estimated and judged
    # by the gate (never the skip reason the old dead end reported). On this
    # pair the arm's estimate is good enough to pass honestly: gate pass,
    # passed: True, products written (values printed while developing).
    assert summary["orthogonal_gate"]["reason"] != "skipped: no transform"
    assert summary["orthogonal_gate"]["passed"] is True
    assert summary["passed"] is True, summary["failure_reason"]
    assert summary["outputs"].get("registered_png")
    # The structural arm's correspondences entered Stage 4 through the
    # shared path: its entry is in the competition metrics.
    assert "structural_ncc" in summary["metrics"]
    # The report block also lands in the on-disk summary.json.
    with open(os.path.join(out_dir, "summary.json")) as f:
        on_disk = json.load(f)
    assert on_disk["structural_correspondences"] == rep


# ---------------------------------------------------------------------------
# 2. Never fires on a run that already has a transform
# ---------------------------------------------------------------------------

def test_structural_arm_stays_silent_on_a_healthy_run(tmp_path):
    """AC: a normal PWIFT run yields a real transform -> attempted False
    (the arm must not run on a run that already has one)."""
    summary, _ = _run(tmp_path, "healthy")

    rep = summary["structural_correspondences"]
    print("\nstructural report:", rep)
    print("passed:", summary["passed"], "| failure:", summary["failure_reason"])

    assert set(rep) == _REPORT_KEYS
    assert rep["attempted"] is False
    assert rep["used"] is False
    assert rep["n_correspondences"] == 0
    assert rep["coverage"] == 0.0
    # Sanity: this run DID get a transform from PWIFT (otherwise the
    # attempted=False above could be trivially true for the wrong reason).
    assert summary["passed"] is True, summary["failure_reason"]


# ---------------------------------------------------------------------------
# 3. Honours the config switch
# ---------------------------------------------------------------------------

def test_config_switch_off_keeps_the_run_failing_closed(tmp_path, monkeypatch):
    """AC: structural_fallback_enabled=False -> the arm never fires even
    though no arm produced a transform, and the run fails closed exactly as
    the pre-arm code did (the sabotage still left usable matches, so the
    verdict is no_transform, not the no-matches RuntimeError)."""
    _sabotage_matchers(monkeypatch, _GarbageMatcher)
    _stub_no_transform(monkeypatch)
    cfg = PipelineConfig(structural_fallback_enabled=False)
    summary, out_dir = _run(tmp_path, "switchoff", cfg=cfg)

    rep = summary["structural_correspondences"]
    print("\nstructural report:", rep)
    print("failure:", summary["failure_reason"])

    assert set(rep) == _REPORT_KEYS
    assert rep["attempted"] is False
    assert rep["used"] is False
    assert summary["passed"] is False
    assert summary["failure_reason"].startswith("no_transform")
    assert not summary["outputs"].get("registered_png")
    assert os.path.exists(os.path.join(out_dir, "summary.json"))


# ---------------------------------------------------------------------------
# 4. Generator returning nothing: attempted but not used
# ---------------------------------------------------------------------------

def test_starved_generator_leaves_the_run_unchanged(tmp_path, monkeypatch):
    """AC: the generator yields nothing -> attempted True, used False, and
    the run behaves EXACTLY as it does without the arm (same fail-closed
    verdict as an identical run with the switch off — compared below).

    Starvation threshold: 0.99 (the suggested value) does NOT starve this
    RELATED pair — after Stage-3 alignment its identical crater patches
    legitimately score >= 0.99 (observed while developing) — so the
    threshold sits above NCC's theoretical maximum of 1.0 instead: no cell
    can ever be accepted, which provably starves the generator."""
    _sabotage_matchers(monkeypatch, _GarbageMatcher)
    _stub_no_transform(monkeypatch)

    cfg = PipelineConfig(structural_min_ncc=1.01)
    summary, out_dir = _run(tmp_path, "starved", cfg=cfg)
    print("\nstructural report:", summary["structural_correspondences"])

    rep = summary["structural_correspondences"]
    assert set(rep) == _REPORT_KEYS
    assert rep["attempted"] is True
    assert rep["used"] is False
    assert rep["n_correspondences"] < cfg.structural_min_correspondences
    assert rep["coverage"] == 0.0
    assert summary["passed"] is False
    assert summary["failure_reason"].startswith("no_transform")
    assert not summary["outputs"].get("registered_png")
    assert os.path.exists(os.path.join(out_dir, "summary.json"))

    # Control: the identical sabotage with the arm switched OFF must reach
    # the same verdict — the starved arm changed nothing.
    cfg_off = PipelineConfig(
        structural_fallback_enabled=False, structural_min_ncc=1.01)
    control, _ = _run(tmp_path, "starved_ctl", cfg=cfg_off)
    assert control["structural_correspondences"]["attempted"] is False
    assert control["passed"] == summary["passed"]
    assert control["failure_reason"] == summary["failure_reason"]
    assert control["outputs"] == summary["outputs"]
