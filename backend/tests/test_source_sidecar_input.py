"""Explicit source sidecar XML input — API upload field + CLI flag.

WHY: uploaded TIFFs land in a fresh per-run directory, so filesystem sibling
auto-discovery (`resolve_sidecar_xml`) never fires on the API path — routing
silently fell back to the 30 deg placeholder unless the client hand-typed
`incidence_deg`. These tests pin the explicit-input contract:

  * an explicit path overrides sibling auto-discovery,
  * omitting the input keeps the honest logged fallback (input is optional),
  * an explicit-but-broken file fails loudly (422 / exception) — registration
    must never reroute on a placeholder guess after the client sent a file,
  * the explicit value actually reaches condition routing.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest


def _xml(incidence: float) -> bytes:
    """Minimal product XML carrying the routing-relevant angle tag."""
    return (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<Product>"
        f"<Solar_incidence_angle_in_degree>{incidence}"
        "</Solar_incidence_angle_in_degree>"
        "</Product>"
    ).encode()


def _tiny_image(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _tiny_pair(tmp_path):
    """Small rotated pair so pipeline-level tests exercise the real run."""
    src_img = np.zeros((256, 256), dtype=np.uint8) + 120
    cv2.circle(src_img, (128, 128), 30, 200, 2)
    cv2.circle(src_img, (100, 100), 15, 40, -1)
    M = cv2.getRotationMatrix2D((128, 128), 2.0, 1.0)
    M[0, 2] += 4.0
    M[1, 2] -= 3.0
    ref_img = cv2.warpAffine(src_img, M, (256, 256))
    src = tmp_path / "src.png"
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(src), src_img)
    cv2.imwrite(str(ref), ref_img)
    return str(src), str(ref)


def _api_client_and_capture(monkeypatch):
    """API harness: capture run_pipeline kwargs instead of running it."""
    from starlette.testclient import TestClient

    from api import app as api_app

    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["out_dir"])
        stem = Path(kwargs["source_path"]).stem
        (out / f"{stem}_matches.png").write_bytes(b"png")
        return {
            "passed": True,
            "failure_reason": None,
            "subpixel_refine": {"low_precision": False},
            "orthogonal_gate": {"passed": True},
        }

    monkeypatch.setattr(api_app, "run_pipeline", fake_run_pipeline)
    return TestClient(api_app.app), captured


# ---------------------------------------------------------------- attach level


def test_explicit_sidecar_path_overrides_sibling_discovery(tmp_path):
    """The explicit path wins even when a different XML sits beside the image."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    src = img_dir / "tile.png"
    _tiny_image(src)
    # Sibling auto-discovery would find this one first...
    (img_dir / "tile.xml").write_bytes(_xml(84.896724))
    # ...but the explicit input lives elsewhere and must win.
    explicit = tmp_path / "explicit" / "PROD.xml"
    explicit.parent.mkdir()
    explicit.write_bytes(_xml(71.2))

    from lunar_registration.preprocessing import load_image

    loaded = load_image(
        str(src), sensor_hint="LROC", sidecar_xml_path=str(explicit)
    )

    assert loaded.sidecar_incidence_deg == pytest.approx(71.2)
    assert loaded.sidecar_path == str(explicit)


def test_missing_explicit_sidecar_fails_loudly(tmp_path):
    """A caller-supplied path that does not exist is an error, not a fallback."""
    src = tmp_path / "tile.png"
    _tiny_image(src)

    from lunar_registration.preprocessing import SidecarXmlError, load_image

    missing = tmp_path / "PROD.xml"
    with pytest.raises(SidecarXmlError, match="PROD.xml"):
        load_image(str(src), sensor_hint="LROC", sidecar_xml_path=str(missing))


def test_unparseable_explicit_sidecar_fails_loudly(tmp_path):
    """Bytes that are not XML must abort, not degrade to the placeholder."""
    src = tmp_path / "tile.png"
    _tiny_image(src)
    junk = tmp_path / "PROD.xml"
    junk.write_bytes(b"\x00 this is not xml")

    from lunar_registration.preprocessing import SidecarXmlError, load_image

    with pytest.raises(SidecarXmlError, match="PROD.xml"):
        load_image(str(src), sensor_hint="LROC", sidecar_xml_path=str(junk))


def test_untagged_explicit_sidecar_fails_loudly(tmp_path):
    """Well-formed XML without ANY recognized angle tag cannot steer routing."""
    src = tmp_path / "tile.png"
    _tiny_image(src)
    empty = tmp_path / "PROD.xml"
    empty.write_bytes(b"<Product><Nothing>1</Nothing></Product>")

    from lunar_registration.preprocessing import SidecarXmlError, load_image

    with pytest.raises(SidecarXmlError, match="PROD.xml"):
        load_image(str(src), sensor_hint="LROC", sidecar_xml_path=str(empty))


def test_omitted_sidecar_keeps_logged_fallback(tmp_path):
    """Input is OPTIONAL: no explicit path and no sibling must not raise."""
    src = tmp_path / "tile.png"
    _tiny_image(src)

    from lunar_registration.preprocessing import load_image

    loaded = load_image(str(src), sensor_hint="LROC")

    assert loaded.sidecar_incidence_deg is None
    assert loaded.sidecar_path is None


# ------------------------------------------------------------ pipeline level


def test_run_pipeline_hard_errors_on_broken_explicit_sidecar(tmp_path):
    """The error must propagate out of run_pipeline (drives CLI exit 1)."""
    import lunar_registration.pipeline as pl
    from lunar_registration.preprocessing import SidecarXmlError

    src, ref = _tiny_pair(tmp_path)
    junk = tmp_path / "PROD.xml"
    junk.write_bytes(b"not xml")

    with pytest.raises(SidecarXmlError):
        pl.run_pipeline(
            source_path=src,
            reference_path=ref,
            out_dir=str(tmp_path / "out"),
            source_sensor="LROC",
            matcher="pwift",
            source_sidecar_xml=str(junk),
        )


def test_run_pipeline_feeds_explicit_sidecar_to_routing(tmp_path, monkeypatch):
    """The explicit XML's incidence must reach condition routing."""
    import lunar_registration.pipeline as pl

    src, ref = _tiny_pair(tmp_path)
    xml = tmp_path / "PROD.xml"
    xml.write_bytes(_xml(71.2))

    captured = {}
    real = pl.determine_adaptive_matcher

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(pl, "determine_adaptive_matcher", spy)

    summary = pl.run_pipeline(
        source_path=src,
        reference_path=ref,
        out_dir=str(tmp_path / "out"),
        source_sensor="LROC",
        matcher="pwift",
        source_sidecar_xml=str(xml),
    )

    assert captured["incidence_deg"] == pytest.approx(71.2)
    assert summary["condition_routing"]["incidence_deg"] == pytest.approx(71.2)


# ----------------------------------------------------------------- API level


def test_register_forwards_source_xml_upload(monkeypatch):
    """POST /api/register must save the XML and forward its explicit path."""
    client, captured = _api_client_and_capture(monkeypatch)

    resp = client.post(
        "/api/register",
        files={
            "source": ("s.tif", b"fake", "image/tiff"),
            "reference": ("r.tif", b"fake", "image/tiff"),
            "source_xml": ("PROD.xml", _xml(84.896724), "application/xml"),
        },
        data={"sensor": "OHRC"},
    )

    assert resp.status_code == 200, resp.text
    forwarded = Path(captured["source_sidecar_xml"])
    assert forwarded.name == "PROD.xml"
    assert forwarded.exists()
    assert forwarded.read_bytes() == _xml(84.896724)


def test_register_without_xml_forwards_none(monkeypatch):
    """Omitting the XML stays legal: forwarded as None (logged-placeholder path)."""
    client, captured = _api_client_and_capture(monkeypatch)

    resp = client.post(
        "/api/register",
        files={
            "source": ("s.tif", b"fake", "image/tiff"),
            "reference": ("r.tif", b"fake", "image/tiff"),
        },
        data={"sensor": "OHRC"},
    )

    assert resp.status_code == 200, resp.text
    assert captured["source_sidecar_xml"] is None


def test_register_sanitizes_xml_filename(monkeypatch):
    """A hostile filename must not escape the per-run upload directory."""
    client, captured = _api_client_and_capture(monkeypatch)

    resp = client.post(
        "/api/register",
        files={
            "source": ("s.tif", b"fake", "image/tiff"),
            "reference": ("r.tif", b"fake", "image/tiff"),
            "source_xml": ("../../evil.xml", _xml(50.0), "application/xml"),
        },
        data={"sensor": "OHRC"},
    )

    assert resp.status_code == 200, resp.text
    forwarded = Path(captured["source_sidecar_xml"])
    assert forwarded.name == "evil.xml"


def test_register_maps_broken_explicit_xml_to_422(monkeypatch):
    """An unusable client XML is a client error (422), never a 500 or a reroute."""
    from starlette.testclient import TestClient

    from api import app as api_app
    from lunar_registration.preprocessing import SidecarXmlError

    def fake_broken(**kwargs):
        raise SidecarXmlError("explicit sidecar XML not found: /run/PROD.xml")

    monkeypatch.setattr(api_app, "run_pipeline", fake_broken)
    client = TestClient(api_app.app, raise_server_exceptions=False)

    resp = client.post(
        "/api/register",
        files={
            "source": ("s.tif", b"fake", "image/tiff"),
            "reference": ("r.tif", b"fake", "image/tiff"),
            "source_xml": ("PROD.xml", _xml(84.9), "application/xml"),
        },
        data={"sensor": "OHRC"},
    )

    assert resp.status_code == 422, resp.text
    assert "PROD.xml" in resp.json()["detail"]


# ----------------------------------------------------------------- CLI level


def test_cli_flag_forwards_source_sidecar_xml(monkeypatch):
    """--source-sidecar-xml must land on run_pipeline's source_sidecar_xml."""
    import lunar_registration.pipeline as pl

    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(pl, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "luna",
            "--source", "s.tif",
            "--reference", "r.tif",
            "--out-dir", "out",
            "--source-sidecar-xml", "/data/PROD.xml",
        ],
    )

    pl.main()

    assert captured["source_sidecar_xml"] == "/data/PROD.xml"
