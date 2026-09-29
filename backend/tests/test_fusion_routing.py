"""Task 7: a BARE neural matcher name must take the same path as its
`hybrid_pwift_<name>` counterpart — PWIFT runs alongside, and fusion +
rigid competition arbitrate between the arms (plan-todos.md: "Neural arms
always compete against PWIFT"). `matcher="pwift"` stays single-arm. A
PWIFT failure degrades to the neural arm alone with the contingency
recorded, and `POST /api/register` surfaces `contingency_fallback` in
BOTH responses.

Fake matchers only — no real model loading.
"""
from pathlib import Path

import cv2
import numpy as np
from starlette.testclient import TestClient

import lunar_registration.pipeline as pl
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


class _FakeMatcher:
    """Fake arm answering with a valid MatchResult built from precomputed
    correspondences (no model loading)."""

    def __init__(self, method, correspondences):
        self.method = method
        self.pts_src, self.pts_dst = correspondences

    def match(self, src, ref, **kwargs):
        return MatchResult(
            method=self.method,
            pts_src=self.pts_src.copy(),
            pts_dst=self.pts_dst.copy(),
        )


class _ExplodingPwift:
    """PWIFT arm that dies the way a broken runtime would."""

    def match(self, *args, **kwargs):
        raise RuntimeError("pwift exploded")


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


def _run(tmp_path, monkeypatch, tag, matcher, pwift=None):
    """Full run on the synthetic pair with fake arms wired into
    get_matcher (no model loading); `pwift` overrides the healthy default."""
    src_path, ref_path, M = _related_pair(tmp_path, tag)
    neural = _FakeMatcher("roma2", _correspondences(M, offset=_NEURAL_OFFSET))
    if pwift is None:
        pwift = _FakeMatcher("pwift", _correspondences(M))
    monkeypatch.setattr(pl, "get_matcher", _fake_get_matcher(neural, pwift))
    return pl.run_pipeline(
        source_path=src_path,
        reference_path=ref_path,
        out_dir=str(tmp_path / f"out_{tag}"),
        source_sensor="LROC",
        matcher=matcher,
    )


def test_bare_neural_name_routes_through_fusion_and_competition(tmp_path, monkeypatch):
    """AC: matcher="roma2" behaves like hybrid_pwift_roma2 — the fused arm
    is evaluated alongside the originals, so compete_rigid has >= 2 arms."""
    summary = _run(tmp_path, monkeypatch, "route", "roma2")

    assert "hybrid_pwift_roma2" in summary["metrics"]
    assert len(summary["metrics"]) >= 2


def test_pwift_failure_degrades_to_the_neural_arm(tmp_path, monkeypatch):
    """AC: a PWIFT crash must not raise — the neural arm alone completes
    the run and the contingency records the degradation."""
    summary = _run(tmp_path, monkeypatch, "pwn", "roma2", pwift=_ExplodingPwift())

    cont = summary["contingency_fallback"]
    assert cont["triggered"] is True
    assert cont["original_matcher"] == "roma2"
    assert cont["fallback_matcher"] == "roma2"
    assert "pwift exploded" in cont["reason"]
    # Without PWIFT there is nothing to fuse — the neural arm stands alone.
    assert set(summary["metrics"]) == {"roma2"}


def test_pwift_stays_single_arm(tmp_path, monkeypatch):
    """AC: matcher="pwift" is unchanged — one arm, no fusion, and no
    contingency (its matches carry real inlier support)."""
    summary = _run(tmp_path, monkeypatch, "single", "pwift")

    assert set(summary["metrics"]) == {"pwift"}
    assert not any(m.startswith("hybrid_pwift_") for m in summary["metrics"])
    assert summary["contingency_fallback"]["triggered"] is False


# --------------------------------------------------
# API visibility: contingency_fallback in BOTH responses
# --------------------------------------------------

def _api_client(monkeypatch, run_summary):
    from api import app as api_app

    def fake_run_pipeline(**kwargs):
        out = Path(kwargs["out_dir"])
        stem = Path(kwargs["source_path"]).stem
        (out / f"{stem}_matches.png").write_bytes(b"png")
        return run_summary

    monkeypatch.setattr(api_app, "run_pipeline", fake_run_pipeline)
    return TestClient(api_app.app)


def _post(client):
    return client.post(
        "/api/register",
        files={
            "source": ("s.tif", b"fake", "image/tiff"),
            "reference": ("r.tif", b"fake", "image/tiff"),
        },
        data={"sensor": "OHRC"},
    )


def test_api_carries_contingency_fallback_in_both_responses(monkeypatch):
    """AC: POST /api/register surfaces contingency_fallback as a top-level
    truth field in the 422 detail AND in the 200 success body."""
    contingency = {
        "triggered": True,
        "original_matcher": "roma2",
        "fallback_matcher": "roma2",
        "reason": "pwift exploded",
    }

    fail_summary = {
        "passed": False,
        "failure_reason": "verification_gate_failed: stub_disagreement:999.0px",
        "subpixel_refine": {"low_precision": True},
        "orthogonal_gate": {"enabled": True, "passed": False, "reason": "stub"},
        "contingency_fallback": contingency,
    }
    resp = _post(_api_client(monkeypatch, fail_summary))
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["contingency_fallback"] == contingency

    ok_summary = {
        "passed": True,
        "failure_reason": None,
        "subpixel_refine": {"low_precision": False},
        "orthogonal_gate": {"enabled": True, "passed": True, "reason": "ok"},
        "contingency_fallback": contingency,
    }
    resp = _post(_api_client(monkeypatch, ok_summary))
    assert resp.status_code == 200, resp.text
    assert resp.json()["contingency_fallback"] == contingency
