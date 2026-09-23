"""Task 3: metadata must drive the scale prior and routing — no CLI flags.

Wiring points:
- sidecar XML incidence -> determine_adaptive_matcher (routing) when no
  manual/map incidence exists
- reference image sensor: auto-detect from filename instead of hardcoded LROC
- computed GSD ratio -> select_best_scale(prior_scale=...)
"""
import shutil

import cv2
import numpy as np
import pytest

XML = """<?xml version="1.0"?>
<Product>
  <Sun_azimuth_in_degree>19.910131</Sun_azimuth_in_degree>
  <Sun_elevation_in_degree>5.103276</Sun_elevation_in_degree>
  <Solar_incidence_angle_in_degree>{inc}</Solar_incidence_angle_in_degree>
</Product>
"""

ROUTING_STUB_INFO_KEYS = (
    "mode", "regime", "incidence_deg", "real_time_requested",
    "resolved_matcher", "reason",
)


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


def _related_pair(tmp_path, stem="PROD123"):
    src = _synthetic(seed=7)
    M = cv2.getRotationMatrix2D((128, 128), 2.0, 1.0)
    M[0, 2] += 4.0
    M[1, 2] -= 3.0
    ref = cv2.warpAffine(src, M, (256, 256))
    src_path = tmp_path / f"{stem}_source_at_5m.tif"
    ref_path = tmp_path / f"{stem}_reference_at_5m.tif"
    cv2.imwrite(str(src_path), src)
    cv2.imwrite(str(ref_path), ref)
    return src_path, ref_path


def test_routing_receives_sidecar_incidence_without_flags(tmp_path, monkeypatch):
    """With no --incidence-deg and no angle maps, routing must receive the
    sidecar XML incidence (canonical pair: 84.9 -> polar branch)."""
    import lunar_registration.pipeline as pl

    src_path, ref_path = _related_pair(tmp_path)
    (tmp_path / "PROD123.xml").write_text(XML.format(inc="84.9"))

    captured = {}

    def fake_determine(**kwargs):
        captured.update(kwargs)
        inc = kwargs.get("incidence_deg")
        return "pwift", {
            "mode": "adaptive",
            "regime": "polar_grazing" if inc and inc >= 70 else "subpixel_cartography",
            "incidence_deg": inc if inc is not None else 30.0,
            "real_time_requested": False,
            "resolved_matcher": "pwift",
            "reason": "stub",
        }

    monkeypatch.setattr(pl, "determine_adaptive_matcher", fake_determine)
    # Keep the scale search out of the way: this test is about routing inputs.
    monkeypatch.setattr(pl, "select_best_scale", lambda *a, **k: (1.0, 0.0))

    summary = pl.run_pipeline(
        source_path=str(src_path),
        reference_path=str(ref_path),
        out_dir=str(tmp_path / "out"),
        source_sensor="OHRC",
        matcher="auto",
    )

    assert captured.get("incidence_deg") == pytest.approx(84.9)
    assert summary["condition_routing"]["regime"] == "polar_grazing"


def test_scale_prior_computed_from_gsd_metadata(tmp_path, monkeypatch):
    """The GSD ratio from georeferencing tags must reach select_best_scale
    as prior_scale (canonical pair: 1.006; synthetic here: 10m/5m -> 2.0)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    import lunar_registration.pipeline as pl

    src_path, ref_path = _related_pair(tmp_path, stem="GEO")
    moon_crs = CRS.from_string(
        'PROJCS["PolarStereographic Moon",GEOGCS["GCS_Moon",'
        'DATUM["D_Moon",SPHEROID["Moon",1737400,0.0,0.0]],'
        'PRIMEM["Reference_Meridian",0.0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90.0],'
        'PARAMETER["central_meridian",0.0],UNIT["metre",1.0]]'
    )
    for path, scale in ((src_path, 10.0), (ref_path, 5.0)):
        with rasterio.open(str(path), "r+") as ds:
            ds.crs = moon_crs
            ds.transform = Affine(scale, 0, 0, 0, -scale, 0)

    captured = {}

    def fake_select(a, b, c, cfg, prior_scale=None, **kw):
        captured["prior_scale"] = prior_scale
        return 1.0, 0.0

    monkeypatch.setattr(pl, "select_best_scale", fake_select)

    summary = pl.run_pipeline(
        source_path=str(src_path),
        reference_path=str(ref_path),
        out_dir=str(tmp_path / "out"),
        source_sensor="OHRC",
        matcher="pwift",
    )

    assert captured["prior_scale"] == pytest.approx(2.0)
    assert summary["gsd_scale_prior"] == pytest.approx(2.0)


def test_reference_sensor_autodetects_ohrc(tmp_path):
    """A canonical OHRC reference filename must detect as OHRC — not be
    force-labeled LROC (pipeline hardcoded sensor_hint='LROC')."""
    import warnings

    from lunar_registration.preprocessing import load_image

    src_path, ref_path = _related_pair(
        tmp_path, stem="OHRXXD18CHO2436502NNNN25039175231280_V2_1_01")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_image(str(ref_path))  # no sensor hint

    assert loaded.sensor.name == "OHRC"
    assert not [w for w in caught if "Could not auto-detect sensor" in str(w.message)]


def test_unknown_filename_falls_back_to_lroc(tmp_path):
    """Unrecognizable names keep a neutral LROC fallback — changed from OHRC
    so auto-detection can be the default without silently flipping prior
    defaults for generic test/upload files."""
    import warnings

    from lunar_registration.preprocessing import load_image

    _, ref_path = _related_pair(tmp_path, stem="unknown")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_image(str(ref_path))

    assert loaded.sensor.name == "LROC"
    assert [w for w in caught if "Could not auto-detect sensor" in str(w.message)]


def test_reference_sensor_autodetects_iirs(tmp_path):
    """IIRS product prefix must detect too (same gap as OHRXXD): the easy-pair
    reference would otherwise fall back to LROC — wrong illumination branch
    and a nonsense approx-GSD prior (80/0.5)."""
    import warnings

    import cv2

    from lunar_registration.preprocessing import load_image

    p = tmp_path / "IIRXXD18CHO2686502NNNN25244140531312_V2_1_reference.tif"
    cv2.imwrite(str(p), _synthetic(seed=5))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_image(str(p))

    assert loaded.sensor.name == "IIRS"
    assert not [w for w in caught if "Could not auto-detect sensor" in str(w.message)]
