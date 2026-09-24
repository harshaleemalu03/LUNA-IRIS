"""Task 6: peak-quality acceptance for the scale/rotation search.

The coarse-to-fine search used to silently return its argmax even when the
argmax sat on the EDGE of the candidate domain (0.575-band / -180 / 3.0
observed in the verification report). An edge argmax means one of two things,
both untrustworthy:

1. the similarity surface was still rising toward the domain edge -> the true
   optimum is OUTSIDE the searched range (or outside the GSD-prior band);
2. the surface was flat/uninformative -> Python's stable-sort tie order
   decided the winner, which lands on insertion-order edges.

The contract: boundary picks and flat surfaces return "no confident
alignment" (None + a reason) instead of a confidently-wrong argmax, and the
pipeline fails closed on it instead of resampling the source with a garbage
scale and matching anyway.
"""
import os

import cv2
import numpy as np
import pytest

from lunar_registration.pwift import no_confident_alignment_reason


# Candidate sets matching PipelineConfig defaults (config.py:124-126)
_SCALE_CANDS = (0.5, 0.707, 1.0, 1.414, 2.0, 3.0)
_ROT_COARSE = (-180.0, -120.0, -60.0, 0.0, 60.0, 120.0)
_NONFLAT_KEYS = [(5, 0.5, -1.0, 7), (3, 0.4, -2.0, 9)]


def test_scale_upper_boundary_pick_is_not_confident():
    """3.0 observed: argmax at the top of the searched scale candidates
    (full OHRC range or prior-band upper edge) -> optimum wants to go
    higher than anything searched -> no confident alignment."""
    reason = no_confident_alignment_reason(
        best_scale=3.0, best_rot=0.0,
        scale_candidates=_SCALE_CANDS,
        coarse_rotation_candidates_deg=_ROT_COARSE,
        best_key=_NONFLAT_KEYS[0], all_keys=_NONFLAT_KEYS,
    )
    assert reason is not None
    assert "scale" in reason
    assert "boundary" in reason


def test_scale_lower_boundary_pick_is_not_confident():
    """0.575-class pick: argmax at the FIRST candidate (band_lo/range lo)
    must fail just like the upper edge."""
    reason = no_confident_alignment_reason(
        best_scale=0.5, best_rot=0.0,
        scale_candidates=_SCALE_CANDS,
        coarse_rotation_candidates_deg=_ROT_COARSE,
        best_key=_NONFLAT_KEYS[0], all_keys=_NONFLAT_KEYS,
    )
    assert reason is not None
    assert "scale" in reason
    assert "boundary" in reason


def test_rotation_min_boundary_pick_is_not_confident():
    """-180 observed: argmax at the edge of the coarse rotation domain."""
    reason = no_confident_alignment_reason(
        best_scale=1.0, best_rot=-180.0,
        scale_candidates=_SCALE_CANDS,
        coarse_rotation_candidates_deg=_ROT_COARSE,
        best_key=_NONFLAT_KEYS[0], all_keys=_NONFLAT_KEYS,
    )
    assert reason is not None
    assert "rotation" in reason


def test_rotation_refined_past_coarse_domain_is_not_confident():
    """Refinement can push the pick beyond the coarse grid (e.g. 120+30=150);
    leaving the searched domain is the same untrustworthy symptom."""
    reason = no_confident_alignment_reason(
        best_scale=1.0, best_rot=150.0,
        scale_candidates=_SCALE_CANDS,
        coarse_rotation_candidates_deg=_ROT_COARSE,
        best_key=_NONFLAT_KEYS[0], all_keys=_NONFLAT_KEYS,
    )
    assert reason is not None
    assert "rotation" in reason


def test_flat_similarity_surface_is_not_confident():
    """Every hypothesis scored identically -> the argmax (wherever it sits,
    even interior) is a tie-order artifact, not a peak."""
    flat = [(0, 0.0, -1e9, 42)] * 8
    reason = no_confident_alignment_reason(
        best_scale=1.0, best_rot=15.0,  # interior values: only flatness is wrong
        scale_candidates=_SCALE_CANDS,
        coarse_rotation_candidates_deg=_ROT_COARSE,
        best_key=flat[0], all_keys=flat,
    )
    assert reason is not None
    assert "flat" in reason


def test_interior_peak_returns_no_reason():
    """The happy path: non-flat surface, scale and rotation both strictly
    inside the searched domains -> confident (None = no objection)."""
    reason = no_confident_alignment_reason(
        best_scale=1.0, best_rot=0.0,
        scale_candidates=_SCALE_CANDS,
        coarse_rotation_candidates_deg=_ROT_COARSE,
        best_key=_NONFLAT_KEYS[0], all_keys=_NONFLAT_KEYS,
    )
    assert reason is None


def test_single_fixed_scale_is_not_a_false_boundary():
    """lo==hi sensor configs (LROC-vs-LROC, scale_range=(1.0,1.0)) pass a
    single fixed scale - that is configuration, not a search, so it must not
    be flagged as a boundary pick."""
    reason = no_confident_alignment_reason(
        best_scale=1.0, best_rot=0.0,
        scale_candidates=(1.0,),
        coarse_rotation_candidates_deg=_ROT_COARSE,
        best_key=_NONFLAT_KEYS[0], all_keys=_NONFLAT_KEYS,
    )
    assert reason is None


def test_search_returns_none_on_flat_surface(monkeypatch):
    """End-to-end through coarse_to_fine_rotation_scale: a surface where all
    hypotheses tie must return None (and say so), not an argmax."""
    import lunar_registration.pwift as pw
    from lunar_registration.config import PipelineConfig

    monkeypatch.setattr(
        pw, "run_pwift_stage", lambda *a, **k: (None, None, []))
    monkeypatch.setattr(
        pw, "_evaluate_rs_candidate", lambda *a, **k: (0, 0.0, -1e9, 42))

    rng = np.random.default_rng(3)
    src = rng.random((256, 256)).astype(np.float32)
    ref = rng.random((256, 256)).astype(np.float32)

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pw.coarse_to_fine_rotation_scale(src, ref, PipelineConfig())

    assert result is None
    assert [w for w in caught if "no confident alignment" in str(w.message)]


def test_pipeline_fails_closed_when_scale_search_unconfident(tmp_path, monkeypatch):
    """select_best_scale returning None (no confident alignment) must make
    the run fail closed: passed False, no_confident_alignment reason, no
    registered products, zero GCPs, summary.json still written."""
    import lunar_registration.pipeline as pl

    # Same synthetic family proven to produce matches in test_fail_closed
    rng = np.random.default_rng(11)
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
    ref_img = cv2.warpAffine(img, M, (256, 256))
    src_path = str(tmp_path / "sc_src.tif")
    ref_path = str(tmp_path / "sc_ref.tif")
    cv2.imwrite(src_path, img)
    cv2.imwrite(ref_path, ref_img)

    out_dir = str(tmp_path / "out_scaleconf")
    monkeypatch.setattr(pl, "select_best_scale", lambda *a, **k: None)

    summary = pl.run_pipeline(
        source_path=src_path, reference_path=ref_path, out_dir=out_dir,
        source_sensor="LROC", matcher="pwift",
    )

    assert summary["passed"] is False
    assert summary["failure_reason"].startswith("no_confident_alignment")
    assert summary["miho_gcps"]["count"] == 0
    assert not summary["outputs"].get("registered_png")
    assert os.path.exists(os.path.join(out_dir, "summary.json"))
    assert summary["subpixel_refine"]["low_precision"] is True
