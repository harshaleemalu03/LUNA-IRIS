"""Task 12: tiepoint-derived coarse transform seeding the estimator.

WHY: matching must produce >= 4 correspondences before ANY transform
hypothesis is evaluated, and RANSAC starts from nothing — on the canonical
pairs the fitted homography comes out unsupported ("no_transform") even
when both rasters carry corner tiepoints in a shared map frame. Metadata
gives a FREE coarse hypothesis (source pixel -> map -> reference pixel)
with zero matching cost.

HONESTY CONTRACT: the seed is never trusted — it must survive the same
reprojection-inlier support test as fitted hypotheses
(`_evaluate_seed_hypothesis`), so wrong/stale metadata LOSES instead of
being believed. When tiepoints are absent or degenerate, derivation
returns H=None with a reason naming the cause (placeholders must stay
traceable). For windowed/tiled loads the tiepoints are SHIFTED by the
known crop origin — exact arithmetic, not a guess.

Seeding is metadata wiring only: `estimate_viewpoint_transform` gains an
optional seed_H (default None = byte-for-byte old behavior); no matcher
family is touched.
"""
import cv2
import numpy as np

from lunar_registration.preprocessing import (
    LoadedImage,
    derive_tiepoint_coarse_transform,
)
from lunar_registration.config import PipelineConfig, get_sensor_config
from lunar_registration.viewpoint import (
    HomographyResult,
    estimate_viewpoint_transform,
    insert_seed_hypothesis,
)


def _loaded(tiepoints, h=100, w=200, path="synthetic.tif"):
    return LoadedImage(
        data=np.zeros((h, w), dtype=np.float32),
        path=path,
        sensor=get_sensor_config("LROC"),
        corner_tiepoints=tiepoints,
    )


# Four corner tiepoints, row/col -> map x/y (matches attach_file_metadata's
# (row, col, x, y) tuple order).
_SRC_ID = [(0, 0, 0.0, 0.0), (0, 199, 199.0, 0.0),
           (99, 0, 0.0, 99.0), (99, 199, 199.0, 99.0)]


def test_derive_identical_placement_is_identity():
    """Same tiepoints on both rasters -> the coarse transform is identity;
    no reason (derivation succeeded)."""
    out = derive_tiepoint_coarse_transform(_loaded(_SRC_ID), _loaded(_SRC_ID))
    assert out["reason"] is None
    assert out["H"] is not None
    assert np.allclose(out["H"], np.eye(3), atol=1e-9)
    assert abs(out["scale"] - 1.0) < 1e-9
    assert abs(out["rotation_deg"]) < 1e-6


def test_derive_known_scale_ratio():
    """Reference covers 2 map units per pixel, source 1 -> source pixel
    maps to half the reference pixel coordinate: scale 0.5, rotation 0."""
    ref = _loaded([(r, c, 2.0 * c, 2.0 * r) for r, c, _, _ in _SRC_ID])
    out = derive_tiepoint_coarse_transform(_loaded(_SRC_ID), ref)
    assert out["reason"] is None
    assert abs(out["scale"] - 0.5) < 1e-9
    assert abs(out["rotation_deg"]) < 1e-6
    assert np.allclose(out["H"], np.diag([0.5, 0.5, 1.0]), atol=1e-9)


def test_derive_missing_tiepoints_names_the_side():
    """Missing metadata must be traceable: the reason names WHICH image
    lost its tiepoints, not just 'failed'."""
    out = derive_tiepoint_coarse_transform(
        _loaded(None, path="src.tif"), _loaded(_SRC_ID, path="ref.tif"))
    assert out["H"] is None
    assert out["reason"] is not None
    assert "source" in out["reason"]
    assert "tiepoints" in out["reason"]
    assert "src.tif" in out["reason"]


def test_derive_degenerate_tiepoints_refused():
    """Collinear tiepoints cannot constrain an affine fit -> refused with
    a reason instead of a garbage transform."""
    collinear = [(5, 0, 0.0, 5.0), (5, 50, 50.0, 5.0),
                 (5, 100, 100.0, 5.0), (5, 150, 150.0, 5.0)]
    out = derive_tiepoint_coarse_transform(_loaded(collinear), _loaded(_SRC_ID))
    assert out["H"] is None
    assert out["reason"] is not None
    assert "degenerate" in out["reason"]


def test_derive_accounts_for_crop_offset():
    """corner_tiepoints are FULL-raster; a window / auto-tile crop moves
    pixels without moving them. derive() shifts them by the KNOWN crop
    origin — the demo always runs windowed or tiled, so withholding would
    make the seed decorative. src cropped at (x=30, y=10): crop-frame map
    is x = u' + 30, y = v' + 10; composed with ref (x = 2u) this gives
    H = 0.5*u' + 15 / 0.5*v' + 5 — NOT the stale 0.5*u' + 0."""
    ref = _loaded([(r, c, 2.0 * c, 2.0 * r) for r, c, _, _ in _SRC_ID])
    out = derive_tiepoint_coarse_transform(
        _loaded(_SRC_ID), ref, src_offset=(30, 10))
    assert out["reason"] is None
    assert abs(out["scale"] - 0.5) < 1e-9
    expected = np.array([[0.5, 0.0, 15.0],
                         [0.0, 0.5, 5.0],
                         [0.0, 0.0, 1.0]])
    assert np.allclose(out["H"], expected, atol=1e-9)


def _result(h, n, provenance="ransac"):
    return HomographyResult(
        H=h,
        inlier_mask=np.ones(n, dtype=bool) if h is not None
        else np.zeros(n, dtype=bool),
        provenance=provenance,
    )


def test_insert_seed_keeps_valid_primary_first():
    """primary_H downstream is results[0].H — a supported seed must never
    shadow a fitted homography, it joins the pool as an extra candidate."""
    primary = _result(np.eye(3), 10)
    seed = _result(np.eye(3), 10, provenance="tiepoint_seed")
    out = insert_seed_hypothesis([primary], seed)
    assert out[0] is primary
    assert out[-1].provenance == "tiepoint_seed"


def test_insert_seed_rescues_failed_primary():
    """When the fitted hypothesis failed (H=None), the seed goes FIRST so
    primary_H is the seed — the rescue case, not a decorative candidate."""
    failed = _result(None, 10)
    seed = _result(np.eye(3), 10, provenance="tiepoint_seed")
    out = insert_seed_hypothesis([failed], seed)
    assert out[0].provenance == "tiepoint_seed"
    assert out[1] is failed


def test_insert_seed_none_is_passthrough():
    """seed_H=None (the default) must leave the estimator's return exactly
    as it was — zero behavior change without metadata."""
    primary = _result(np.eye(3), 10)
    out = insert_seed_hypothesis([primary], None)
    assert len(out) == 1
    assert out[0] is primary


def _translation(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _consistent_matches(seed=5):
    rng = np.random.default_rng(seed)
    pts_src = rng.uniform(10, 190, size=(30, 2)).astype(np.float64)
    H_true = _translation(4.0, -3.0)
    pts_dst = pts_src + H_true[:2, 2]
    return pts_src, pts_dst, H_true


def test_estimator_appends_supported_seed():
    """Matches consistent with the seeded affine -> the seed is evaluated,
    supported (30/30 inliers) and joins the hypothesis pool tagged with
    its provenance."""
    pts_src, pts_dst, H_true = _consistent_matches()
    res = estimate_viewpoint_transform(
        pts_src, pts_dst, get_sensor_config("IIRS"),
        image_shape=(256, 256), cfg=PipelineConfig(), seed_H=H_true,
    )
    results = res if isinstance(res, list) else [res]
    seeds = [r for r in results if r.provenance == "tiepoint_seed"]
    assert len(seeds) == 1
    assert int(seeds[0].inlier_mask.sum()) == len(pts_src)
    assert np.allclose(seeds[0].H, H_true)


def test_estimator_rejects_unsupported_seed():
    """A seed the matches do NOT support (500 px off) must not join the
    pool: bad metadata loses on evidence, it is not trusted."""
    pts_src, pts_dst, _ = _consistent_matches()
    res = estimate_viewpoint_transform(
        pts_src, pts_dst, get_sensor_config("IIRS"),
        image_shape=(256, 256), cfg=PipelineConfig(),
        seed_H=_translation(500.0, 500.0),
    )
    results = res if isinstance(res, list) else [res]
    assert all(r.provenance != "tiepoint_seed" for r in results)


def test_pipeline_summary_reports_tiepoint_coarse(tmp_path):
    """A plain synthetic pair has no georeferencing -> summary must say so
    truthfully: derived False, reason names tiepoints, seed_supported None."""
    import lunar_registration.pipeline as pl

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
    src_path = str(tmp_path / "tc_src.tif")
    ref_path = str(tmp_path / "tc_ref.tif")
    cv2.imwrite(src_path, img)
    cv2.imwrite(ref_path, cv2.warpAffine(img, M, (256, 256)))

    summary = pl.run_pipeline(
        source_path=src_path, reference_path=ref_path,
        out_dir=str(tmp_path / "out_tc"),
        source_sensor="LROC", matcher="pwift",
    )

    tc = summary["tiepoint_coarse"]
    assert tc["derived"] is False
    assert tc["reason"] is not None
    assert "tiepoints" in tc["reason"]
    assert tc["seed_supported"] is None


def test_derive_cross_frame_reprojects_into_reference_crs():
    """The REAL situation on this eval set: source GCPs in lunar lon/lat
    with NO declared CRS, reference as a grid WITH a declared CRS in a
    different frame (the OHRC pair is lon/lat vs Polar Stereographic
    metres — this is the root cause behind Phase 0's "tiepoint warp ~0").
    The source mapping must be reprojected into the reference frame before
    composing; a naive same-frame composition would collapse the seed.

    Oracle: an independent point chain (px -> lonlat -> rasterio.warp ->
    stere -> reference px) must agree with H at the fit corners."""
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import transform as warp_transform

    lonlat_crs = CRS.from_string("+proj=longlat +R=1737400 +no_defs")
    stere_crs = CRS.from_string(
        "+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m +no_defs")

    h, w = 100, 200

    def lonlat(u, v):  # pixel -> lon/lat, tiny extent near the south pole
        return (10.0 + 0.00005 * u, -85.0 - 0.0005 * v)

    corners_px = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    src_gcps = [(v, u, *lonlat(u, v)) for u, v in corners_px]
    lons, lats = zip(*(lonlat(u, v) for u, v in corners_px))
    xs, ys = warp_transform(lonlat_crs, stere_crs, list(lons), list(lats))
    # Reference: north-up axis-aligned grid covering the warped corners.
    a = (max(xs) - min(xs)) / (w - 1)
    e = -(max(ys) - min(ys)) / (h - 1)
    ref_transform = Affine(a, 0.0, min(xs), 0.0, e, max(ys))

    src = LoadedImage(
        data=np.zeros((h, w), np.float32), path="src.tif",
        sensor=get_sensor_config("LROC"), corner_tiepoints=src_gcps)
    ref = LoadedImage(
        data=np.zeros((h, w), np.float32), path="ref.tif",
        sensor=get_sensor_config("LROC"),
        geotransform=ref_transform, crs=stere_crs)

    out = derive_tiepoint_coarse_transform(src, ref)
    assert out["reason"] is None, out["reason"]
    assert out["H"] is not None
    assert 0.01 < out["scale"] < 100.0
    for (u, v), x_st, y_st in zip(corners_px, xs, ys):
        exp = np.array([(x_st - min(xs)) / a, (y_st - max(ys)) / e])
        got = out["H"] @ np.array([u, v, 1.0])
        got = got[:2] / got[2]
        assert np.allclose(got, exp, atol=0.25), (u, v, got, exp)


def test_derive_refuses_cross_frame_without_target_crs():
    """Frames differ AND the projected side declares no CRS -> the
    reprojection target is unknowable; refuse with a reason instead of
    composing a confident lie."""
    h, w = 100, 200
    src_gcps = [(0, 0, 10.0, -85.0), (0, w - 1, 10.01, -85.0),
                (h - 1, 0, 10.0, -85.1), (h - 1, w - 1, 10.01, -85.1)]
    src = LoadedImage(
        data=np.zeros((h, w), np.float32), path="src.tif",
        sensor=get_sensor_config("LROC"), corner_tiepoints=src_gcps)
    ref = LoadedImage(
        data=np.zeros((h, w), np.float32), path="ref.tif",
        sensor=get_sensor_config("LROC"),
        geotransform=(5.0, 0.0, 67338.0, 0.0, -5.0, 150047.0),
        crs=None)

    out = derive_tiepoint_coarse_transform(src, ref)
    assert out["H"] is None
    assert out["reason"] is not None
    assert "cannot reproject" in out["reason"]
    assert "CRS" in out["reason"]
