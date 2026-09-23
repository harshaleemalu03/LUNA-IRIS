"""Task 4: verification must fail closed — no transform or gate FAIL means
`passed: false`, no registered products, and (via the API) an error status.

Covers the two fail-open modes proven in the verification report:
1. gate ran and FAILED -> pipeline warned "Proceeding with flagged confidence"
   and still wrote outputs (exit 0)
2. primary_H was None -> gate never ran -> orthogonal_gate.passed defaulted
   to True ("disabled" reason while enabled: true) and nothing failed
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient


def _synthetic(size=(256, 256), seed=42):
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


def _related_pair(tmp_path, seed=11):
    src = _synthetic(seed=seed)
    M = cv2.getRotationMatrix2D((128, 128), 2.0, 1.0)
    M[0, 2] += 4.0
    M[1, 2] -= 3.0
    ref = cv2.warpAffine(src, M, (256, 256))
    src_path = tmp_path / "fc_src.tif"
    ref_path = tmp_path / "fc_ref.tif"
    cv2.imwrite(str(src_path), src)
    cv2.imwrite(str(ref_path), ref)
    return str(src_path), str(ref_path)


def test_gate_fail_is_fail_closed(tmp_path, monkeypatch):
    """Gate FAIL -> passed False, reason recorded, NO registered products,
    and the 'Proceeding with flagged confidence' escape hatch is gone."""
    import warnings

    import lunar_registration.pipeline as pl

    src, ref = _related_pair(tmp_path)
    out_dir = str(tmp_path / "out_gatefail")

    monkeypatch.setattr(
        pl, "gate_cheap",
        lambda *a, **k: {"pass": False, "reason": "stub_disagreement:999.0px",
                         "cost_ms": 0.0, "struct_ncc": 0.0},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = pl.run_pipeline(
            source_path=src, reference_path=ref, out_dir=out_dir,
            source_sensor="LROC", matcher="pwift",
        )

    assert summary["passed"] is False
    assert summary["failure_reason"].startswith("verification_gate_failed")
    assert "stub_disagreement" in summary["failure_reason"]
    assert not summary["outputs"].get("registered_png")
    assert os.path.exists(os.path.join(out_dir, "summary.json"))  # honest report still written
    assert not [w for w in caught if "Proceeding with flagged confidence" in str(w.message)]


def test_no_transform_is_fail_closed(tmp_path, monkeypatch):
    """primary_H None -> passed False + gate reported as not-passed, instead
    of the old orthogonal_gate.passed=True/'disabled' default."""
    import lunar_registration.pipeline as pl
    from lunar_registration.viewpoint import HomographyResult

    src, ref = _related_pair(tmp_path, seed=13)
    out_dir = str(tmp_path / "out_notransform")

    monkeypatch.setattr(
        pl, "estimate_viewpoint_transform",
        lambda pts_src, *a, **k: HomographyResult(
            H=None, inlier_mask=np.zeros(len(pts_src), dtype=bool)),
    )

    summary = pl.run_pipeline(
        source_path=src, reference_path=ref, out_dir=out_dir,
        source_sensor="LROC", matcher="pwift",
    )

    assert summary["passed"] is False
    assert summary["failure_reason"].startswith("no_transform")
    assert summary["orthogonal_gate"]["passed"] is False
    assert not summary["outputs"].get("registered_png")
    assert os.path.exists(os.path.join(out_dir, "summary.json"))


def _client(monkeypatch, run_summary):
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


def test_api_failed_registration_returns_422(monkeypatch):
    """A pipeline run that fails verification must be an HTTP error with a
    structured reason — never a 200 'success'."""
    client = _client(monkeypatch, {
        "passed": False,
        "failure_reason": "verification_gate_failed: stub_disagreement:999.0px",
        "subpixel_refine": {"low_precision": True, "dx": None, "dy": None},
        "orthogonal_gate": {"enabled": True, "passed": False,
                            "reason": "stub_disagreement:999.0px"},
        "condition_routing": {"regime": "polar_grazing", "incidence_deg": 84.9,
                              "resolved_matcher": "hybrid_pwift_roma2"},
        "gsd_scale_prior": 1.006,
    })

    resp = _post(client)

    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["passed"] is False
    assert detail["failure_reason"].startswith("verification_gate_failed")
    assert detail["low_precision"] is True
    assert detail["summary_path"]
    assert detail["condition_routing"]["regime"] == "polar_grazing"
    assert detail["gsd_scale_prior"] == pytest.approx(1.006)


def test_api_success_surfaces_passed_and_low_precision(monkeypatch):
    """Success response must carry the honesty flags too (low_precision was
    computed in every run but never surfaced)."""
    client = _client(monkeypatch, {
        "passed": True,
        "failure_reason": None,
        "subpixel_refine": {"low_precision": False, "dx": 1.5, "dy": -2.5},
        "orthogonal_gate": {"enabled": True, "passed": True, "reason": "ok"},
    })

    resp = _post(client)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["passed"] is True
    assert body["low_precision"] is False
    assert body["orthogonal_gate_passed"] is True


def test_cli_main_exits_nonzero_when_registration_failed(monkeypatch, tmp_path):
    """CLI contract: failed verification -> SystemExit nonzero (the demo keys
    off exit codes; pre-Task-4 main() always exited 0 after a print)."""
    import lunar_registration.pipeline as pl

    monkeypatch.setattr(
        pl, "run_pipeline",
        lambda **kw: {"passed": False, "failure_reason": "no_transform: x"},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--source", "s.tif", "--reference", "r.tif",
         "--out-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exc:
        pl.main()

    assert exc.value.code == 1


def test_cli_main_exits_zero_when_registration_passed(monkeypatch, tmp_path):
    """A passing run must NOT exit — main() returns and the process exits 0."""
    import lunar_registration.pipeline as pl

    monkeypatch.setattr(
        pl, "run_pipeline",
        lambda **kw: {"passed": True, "failure_reason": None},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--source", "s.tif", "--reference", "r.tif",
         "--out-dir", str(tmp_path)],
    )

    pl.main()  # no SystemExit raised
