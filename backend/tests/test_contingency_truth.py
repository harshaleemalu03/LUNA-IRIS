"""Task 11: contingency_fallback must tell the truth about the winning arm.

The verification report: `triggered` reported False while the arm that WON
the rigid competition carried 0 geometrically valid inliers — the matcher
delivered nothing verifiable, yet the summary claimed "no contingency
occurred". Fail-honestly means the status flag matches reality even when no
alternate arm remained to run (fallback_matcher stays None: contingency
detected, nothing left to try).

Design: `note_winning_arm_support` (pipeline.py) is a pure update over the
contingency dict — unit-testable without a pipeline run; the integration
test runs the full pipeline with a garbage matcher so the competition's
winner provably has 0 inliers, and asserts the summary reflects it.
"""
import cv2
import numpy as np

from lunar_registration.config import PipelineConfig


def _related_pair(tmp_path, seed=11):
    """Same synthetic family as test_subpixel_refine_honesty (proven to
    yield a transform through the full pipeline — so only the matcher is
    under test here, not the scale search / illumination stages)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((256, 256), dtype=np.uint8) + 120
    for _ in range(15):
        cx = int(rng.integers(20, 236))
        cy = int(rng.integers(20, 236))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r, 200, 2)
        cv2.circle(img, (cx - 2, cy - 2), r - 2, 40, -1)
        cv2.circle(img, (cx + 2, cy + 2), r - 4, 150, -1)
    M = cv2.getRotationMatrix2D((128, 128), 2.0, 1.0)
    M[0, 2] += 4.0
    M[1, 2] -= 3.0
    ref = cv2.warpAffine(img, M, (256, 256))
    src_path = str(tmp_path / "ct_src.tif")
    ref_path = str(tmp_path / "ct_ref.tif")
    cv2.imwrite(src_path, img)
    cv2.imwrite(ref_path, ref)
    return src_path, ref_path


def test_zero_inlier_winner_marks_contingency():
    """A winner with 0 inliers is a contingency even with no fallback arm."""
    from lunar_registration.pipeline import note_winning_arm_support

    contingency = {
        "triggered": False,
        "original_matcher": "hybrid_pwift_roma2",
        "fallback_matcher": None,
        "reason": None,
    }
    out = note_winning_arm_support(contingency, "hybrid_pwift_roma2", 0)
    assert out["triggered"] is True
    assert out["fallback_matcher"] is None  # nothing left to fall back to
    assert out["reason"] is not None
    assert "hybrid_pwift_roma2" in out["reason"]
    assert "0 geometrically valid" in out["reason"]


def test_supported_winner_leaves_contingency_untouched():
    """A winner with real inlier support is NOT a contingency — the flag
    must stay exactly as the matcher stage left it (no false positives)."""
    from lunar_registration.pipeline import note_winning_arm_support

    contingency = {
        "triggered": False,
        "original_matcher": "pwift",
        "fallback_matcher": None,
        "reason": None,
    }
    out = note_winning_arm_support(contingency, "pwift", 12)
    assert out["triggered"] is False
    assert out["reason"] is None


def test_already_triggered_reason_keeps_both_facts():
    """If the matcher stage already recorded a contingency AND the winner
    then shows 0 inliers, the summary must carry BOTH facts — the earlier
    reason must not be silently dropped."""
    from lunar_registration.pipeline import note_winning_arm_support

    first_fact = "Neural matcher 'roma2' yielded < 4 matches; degraded to PWIFT only."
    contingency = {
        "triggered": True,
        "original_matcher": "hybrid_pwift_roma2",
        "fallback_matcher": "pwift",
        "reason": first_fact,
    }
    out = note_winning_arm_support(contingency, "pwift", 0)
    assert out["triggered"] is True
    assert out["fallback_matcher"] == "pwift"  # cascade record preserved
    assert first_fact in out["reason"]
    assert "0 geometrically valid" in out["reason"]


def test_pipeline_summary_truthful_when_winner_has_zero_inliers(tmp_path, monkeypatch):
    """AC: the winning arm has 0 inliers -> summary reports a contingency.

    The matcher is replaced by one returning uncorrelated random points:
    >= 4 points so the result ENTERS the competition, but no homography can
    support them, so the winner's metrics end up with 0 inliers — the exact
    state the old code reported as `triggered: False`."""
    import lunar_registration.pipeline as pl
    from lunar_registration.matching import MatchResult

    src_path, ref_path = _related_pair(tmp_path)

    class _GarbageMatcher:
        def match(self, src, ref, **kwargs):
            rng = np.random.default_rng(7)
            n = 24
            h, w = src.shape[:2]
            pts_src = rng.uniform(0, 1, size=(n, 2)) * np.array([w, h])
            pts_dst = rng.uniform(0, 1, size=(n, 2)) * np.array([w, h])
            return MatchResult(
                method="pwift",
                pts_src=pts_src.astype(np.float32),
                pts_dst=pts_dst.astype(np.float32),
            )

    monkeypatch.setattr(pl, "get_matcher", lambda name, cfg: _GarbageMatcher())
    # Stub the gate: this task tests contingency truth, not the gate. (With
    # H=None the gate is skipped anyway; stub keeps the run deterministic if
    # RANSAC were to hallucinate a transform.)
    monkeypatch.setattr(pl, "gate_cheap", lambda *a, **k: {
        "pass": True, "reason": "stubbed: task-11 contingency-truth test",
        "cost_ms": 0.0, "struct_ncc": 1.0,
    })

    summary = pl.run_pipeline(
        source_path=src_path, reference_path=ref_path,
        out_dir=str(tmp_path / "out_cont"),
        source_sensor="LROC", matcher="pwift",
        # Task 8 edit: the structural-NCC last-resort arm would now rescue
        # this transform-less run (winner with REAL inliers), destroying the
        # scenario under test. Disabling it pins the pre-Task-8 world this
        # Task-11 test is about: no arm produces a transform at all.
        cfg=PipelineConfig(structural_fallback_enabled=False),
    )

    # Sanity: competition ran, and the winning arm PROVABLY had 0 inliers
    # (the exact scenario this task is about — not just "some run happened").
    best_method = summary["best_method"]
    assert summary["metrics"][best_method]["n_inliers"] == 0

    cont = summary["contingency_fallback"]
    assert cont["triggered"] is True, (
        "summary claims no contingency although the winning arm had 0 inliers"
    )
    assert cont["reason"] is not None
    assert "0 geometrically valid" in cont["reason"]
    assert cont["fallback_matcher"] is None
