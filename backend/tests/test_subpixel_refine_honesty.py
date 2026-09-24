"""Task 8: subpixel-refine honesty — degenerate dx/dy detected, not implied.

The verification report observed refine dx/dy that were IDENTICAL across
different matchers (123.5054 / 271.0260 with pwift and roma2 alike). The
structural cause: `pipeline.py` calls `refine_tile(src_scaled, ref.data, ...)`
— inputs that never include the matcher's transform — so the refine measures
the GROSS source-vs-reference offset, not a registration residual. It also
reported those numbers as if they were subpixel corrections (a strong
phase-correlation peak on a 123px shift yielded `low_precision: false`).

Guards added:
1. `refine_tile` names WHY its estimate is low precision: weak
   phase-correlation peak, correction beyond +/-SUBPIXEL_MAX_PX, or a skipped
   refiner — a `reason` field instead of a bare boolean.
2. The pipeline adds the structural objection (matcher-independent inputs)
   and forces `low_precision: true` with that reason.
3. summary.json and the API (success + 422 detail) surface the reason.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from lunar_registration.refine import SUBPIXEL_MAX_PX, refine_tile


def _noise(seed, size=256):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(size, size)).astype(np.float32)


def test_refine_weak_peak_is_flagged_with_reason():
    """Unrelated tiles -> phase-correlation peak is weak -> low_precision
    with a reason that names the weak peak (was: boolean only)."""
    out = refine_tile(_noise(1), _noise(2), method="phase")
    assert out["low_precision"] is True
    assert out["reason"] is not None
    assert "weak" in out["reason"]


def test_refine_non_subpixel_correction_is_flagged_with_reason():
    """A gross 91px shift with a STRONG peak is not a subpixel correction ->
    low_precision with a reason (was: low_precision false, no reason, i.e.
    the report's 123.5/271.0 px 'subpixel' claim)."""
    tile = _noise(3)
    shifted = np.roll(tile, 91, axis=1)
    out = refine_tile(tile, shifted, method="phase")
    assert max(abs(out["dx"]), abs(out["dy"])) > SUBPIXEL_MAX_PX
    assert out["low_precision"] is True
    assert out["reason"] is not None
    assert "non-subpixel" in out["reason"]


def test_refine_strong_subpixel_peak_reports_no_objection():
    """Perfect peak at zero shift -> no refine-level objection: the reason
    field is the ABSENCE of one (None), not a missing key."""
    tile = _noise(4)
    out = refine_tile(tile, tile.copy(), method="phase")
    assert out["low_precision"] is False
    assert out["reason"] is None


def _related_pair(tmp_path, seed=11):
    """Same synthetic family as test_fail_closed (proven to yield a
    transform through the full pipeline)."""
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
    src_path = str(tmp_path / "sp_src.tif")
    ref_path = str(tmp_path / "sp_ref.tif")
    cv2.imwrite(src_path, img)
    cv2.imwrite(ref_path, ref)
    return src_path, ref_path


def test_refine_degenerate_flagged_and_identical_across_matchers(tmp_path, monkeypatch):
    """AC verify: two different matchers on the same pair -> the refine
    values come out IDENTICAL (matcher-independent by construction) and BOTH
    runs are flagged low_precision with the structural reason."""
    import lunar_registration.pipeline as pl

    src_path, ref_path = _related_pair(tmp_path)
    # Stub the gate: this task tests REFINE honesty, not the gate.
    monkeypatch.setattr(pl, "gate_cheap", lambda *a, **k: {
        "pass": True, "reason": "stubbed: task-8 subpixel-refine test",
        "cost_ms": 0.0, "struct_ncc": 1.0,
    })

    summaries = {}
    for matcher in ("pwift", "roma2"):
        summaries[matcher] = pl.run_pipeline(
            source_path=src_path, reference_path=ref_path,
            out_dir=str(tmp_path / f"out_sp_{matcher}"),
            source_sensor="LROC", matcher=matcher,
        )

    sum_pwift, sum_roma = summaries["pwift"], summaries["roma2"]
    assert sum_pwift["passed"] is True, sum_pwift.get("failure_reason")
    assert sum_roma["passed"] is True, sum_roma.get("failure_reason")

    sp, sr = sum_pwift["subpixel_refine"], sum_roma["subpixel_refine"]
    # The degeneracy itself: refine never sees the matcher's transform.
    assert (sp["dx"], sp["dy"]) == (sr["dx"], sr["dy"])
    # The honesty guard: both flagged, both with the structural reason.
    for out in (sp, sr):
        assert out["low_precision"] is True
        assert out["reason"] is not None
        assert "matcher-independent" in out["reason"]

    # summary.json on disk carries it too
    import json
    on_disk = json.loads(
        (Path(summaries["pwift"]["outputs"]["summary_path"])
         if "summary_path" in summaries["pwift"].get("outputs", {})
         else Path(tmp_path / "out_sp_pwift" / "summary.json")).read_text()
    )
    assert "matcher-independent" in on_disk["subpixel_refine"]["reason"]


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


def test_api_success_surfaces_subpixel_refine_reason(monkeypatch):
    client = _api_client(monkeypatch, {
        "passed": True,
        "failure_reason": None,
        "subpixel_refine": {
            "low_precision": True, "dx": 123.5054, "dy": 271.0260,
            "reason": "matcher-independent inputs: gross offset",
        },
        "orthogonal_gate": {"enabled": True, "passed": True, "reason": "ok"},
    })
    resp = _post(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["low_precision"] is True
    assert "matcher-independent" in body["subpixel_refine_reason"]


def test_api_failure_detail_surfaces_subpixel_refine_reason(monkeypatch):
    client = _api_client(monkeypatch, {
        "passed": False,
        "failure_reason": "verification_gate_failed: stub: 999.0px > 24.0px",
        "subpixel_refine": {
            "low_precision": True, "dx": None, "dy": None,
            "reason": "skipped: verification_gate_failed: stub",
        },
        "orthogonal_gate": {"enabled": True, "passed": False, "reason": "stub"},
    })
    resp = _post(client)
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "skipped" in detail["subpixel_refine_reason"]
