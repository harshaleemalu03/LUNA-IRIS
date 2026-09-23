"""Task 2: metadata reader — GeoTIFF tags + sidecar XML angles into LoadedImage."""
import math
import os

import pytest
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

from lunar_registration.preprocessing import LoadedImage, load_image

R_MOON_M = 1_737_400.0

MOON_POLAR = CRS.from_string(
    'PROJCS["PolarStereographic Moon",GEOGCS["GCS_Moon",'
    'DATUM["D_Moon",SPHEROID["Moon",1737400,0.0,0.0]],'
    'PRIMEM["Reference_Meridian",0.0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90.0],'
    'PARAMETER["central_meridian",0.0],UNIT["metre",1.0]]'
)

XML_TEMPLATE = """<?xml version="1.0"?>
<Product>
  <Sun_azimuth_in_degree>{az}</Sun_azimuth_in_degree>
  <Sun_elevation_in_degree>{el}</Sun_elevation_in_degree>
  <Solar_incidence_angle_in_degree>{inc}</Solar_incidence_angle_in_degree>
</Product>
"""


def _write_tif(path, *, crs=None, transform=None, gcps=None, shape=(64, 64)):
    data = (np_random(shape) * 255).astype("uint8")
    with rasterio.open(
        str(path), "w", driver="GTiff", height=shape[0], width=shape[1],
        count=1, dtype="uint8", crs=crs, transform=transform,
    ) as ds:
        ds.write(data, 1)
        if gcps:
            # rasterio's GCP setter requires a CRS; degree tiepoints are
            # geographic (the real OHRC files store them without a dataset CRS).
            ds.gcps = (
                [rasterio.control.GroundControlPoint(*g) for g in gcps],
                CRS.from_epsg(4326),
            )


def np_random(shape):
    import numpy as np
    return np.random.default_rng(0).random(shape)


def test_geotiff_pixel_scale_read(tmp_path):
    """A projected GeoTIFF's pixel scale must become pixel_scale_m/gsd_m."""
    p = tmp_path / "proj.tif"
    _write_tif(p, crs=MOON_POLAR, transform=Affine(5.0, 0, 0, 0, -5.0, 0))

    loaded = load_image(str(p), sensor_hint="LROC")

    assert loaded.pixel_scale_m == (5.0, 5.0)
    assert loaded.gsd_m == pytest.approx(5.0)


def test_corner_tiepoints_gsd_read(tmp_path):
    """Corner tiepoints in degrees (no CRS) must yield tiepoints + meter gsd."""
    p = tmp_path / "tie.tif"
    # 100x100 px spanning exactly 1 degree of longitude and latitude at the
    # equator -> gsd = (pi*R/180 meters per degree) / 100 px meters/px.
    gcps = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 100.0, 1.0, 0.0),
        (100.0, 0.0, 0.0, 1.0),
        (100.0, 100.0, 1.0, 1.0),
    ]
    _write_tif(p, gcps=gcps, shape=(100, 100))

    loaded = load_image(str(p), sensor_hint="LROC")

    assert loaded.corner_tiepoints is not None
    assert len(loaded.corner_tiepoints) == 4
    assert loaded.gsd_m == pytest.approx(math.pi * R_MOON_M / 180.0 / 100.0, rel=0.02)


def test_sidecar_xml_angles_read(tmp_path):
    """Sidecar XML (product stem, shared by source+reference) must be found
    and its illumination angles attached as scalars."""
    p = tmp_path / "PROD123_source_at_5m.tif"
    _write_tif(p, crs=MOON_POLAR, transform=Affine(5.0, 0, 0, 0, -5.0, 0))
    (tmp_path / "PROD123.xml").write_text(
        XML_TEMPLATE.format(az="19.910131", el="5.103276", inc="84.896724")
    )

    loaded = load_image(str(p), sensor_hint="OHRC")

    assert loaded.sidecar_path is not None
    assert loaded.sidecar_path.endswith("PROD123.xml")
    assert loaded.sidecar_incidence_deg == pytest.approx(84.896724)
    assert loaded.sidecar_sun_elevation_deg == pytest.approx(5.103276)
    assert loaded.sidecar_sun_azimuth_deg == pytest.approx(19.910131)


def test_missing_metadata_logs_field_names(tmp_path, caplog):
    """Absent metadata degrades to None placeholders, logging WHICH field
    was missing so fallback defaults are never silent."""
    import logging

    p = tmp_path / "bare.tif"
    _write_tif(p)  # identity transform, no crs, no gcps, no sidecar

    with caplog.at_level(logging.WARNING, logger="lunar_registration.preprocessing"):
        loaded = load_image(str(p), sensor_hint="LROC")

    assert loaded.gsd_m is None
    assert loaded.sidecar_incidence_deg is None
    text = caplog.text
    assert "Solar_incidence_angle_in_degree" in text
    assert "georefer" in text.lower()


def test_rotated_footprint_tiepoints_gsd(tmp_path):
    """Regression: the real OHRC source has a footprint ROTATED in lon/lat
    space (lon changes along rows too). GSD must come from per-axis edge
    lengths (geodesic), not axis-aligned lon/lat bounding spans.

    Geometry mirrors OHRXXD18CHO2436502..._source_at_5m.tif (648x5059 px,
    ~5 m/px ground truth from the verification report)."""
    p = tmp_path / "rot_source_at_5m.tif"
    gcps = [
        (0.0, 0.0, 26.723277, -84.574934),
        (0.0, 648.0, 27.813661, -84.540314),
        (5059.0, 0.0, 29.93406, -85.353908),
        (5059.0, 648.0, 31.180635, -85.313377),
    ]
    _write_tif(p, gcps=gcps, shape=(5059, 648))

    loaded = load_image(str(p), sensor_hint="OHRC")

    # ~5 m/px both axes (report: 4.98 row / 5.08 col); the old bounding-span
    # formula returned ~11.5 (2.3x too large).
    assert loaded.gsd_m == pytest.approx(5.0, rel=0.1)
