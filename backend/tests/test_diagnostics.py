"""Tests for the Phase-A alignment diagnostics (`diagnostics.py`).

The diagnostics answer one question before any matcher is trusted: does the
metadata homography actually land the source on the reference's ground? The
tests build a synthetic pair where the source content is known to live inside
the reference at a known homography, plus a deliberately displaced control.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from lunar_registration.diagnostics import (
    alignment_report,
    main,
    render_alignment_overlay,
)

_IMG = (240, 320)      # source canvas (h, w)
_REF = (400, 400)      # reference canvas (h, w)
# Ground-truth src->ref pose: 12 deg rotation, 0.9 scale, translation.
_ROT_DEG = 12.0
_SCALE = 0.9
_TRANS = (40.0, 30.0)


def _unique_texture(size: tuple[int, int], seed: int = 7) -> np.ndarray:
    """Globally unique low-frequency texture (smoothed noise + craters).

    WHY: the diagnostics rank translations by NCC, so the pattern must have a
    single unambiguous peak - a repetitive crater field would not.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    base = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 6.0)
    for _ in range(25):
        cx, cy = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
        r = int(rng.integers(8, 28))
        yy, xx = np.mgrid[0:h, 0:w]
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        base[disk] -= 1.5 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2) ** 2)))[disk]
    base = (base - base.min()) / (base.max() - base.min())
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _pose_h(trans: tuple[float, float] = _TRANS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Homography for the ground-truth pose about the source centre, plus the pair it builds.

    The reference is a full-canvas unique texture; the source is that texture
    pulled through the inverse pose, so warping the source by the pose
    reproduces the reference over the whole source footprint - a warp-path
    control with no artificial content boundary (black padding) to spike the
    gradient maps and skew structural scores.
    """
    c = (_IMG[1] / 2.0, _IMG[0] / 2.0)
    t = np.deg2rad(_ROT_DEG)
    rot = np.array(
        [[_SCALE * np.cos(t), -_SCALE * np.sin(t)],
         [_SCALE * np.sin(t), _SCALE * np.cos(t)]]
    )
    H = np.array(
        [[rot[0, 0], rot[0, 1], trans[0] + c[0] - rot[0, 0] * c[0] - rot[0, 1] * c[1]],
         [rot[1, 0], rot[1, 1], trans[1] + c[1] - rot[1, 0] * c[0] - rot[1, 1] * c[1]],
         [0.0, 0.0, 1.0]]
    )
    ref = _unique_texture(_REF, seed=3)
    src = cv2.warpPerspective(ref, np.linalg.inv(H), (_IMG[1], _IMG[0]))
    return H, src, ref


# Projected lunar CRS shared by the fixture rasters: a declared CRS on both
# sides puts them in the same map frame, so the tiepoint seed composes directly
# with no reprojection involved.
_MOON_EQC_CRS = "+proj=eqc +R=1737400 +units=m +no_defs"
# Constant source pixel->frame shift: `_mapping_from_image` refuses an identity
# geotransform, so the source fixture needs a non-degenerate one too.
_SRC_FRAME_SHIFT = (10.0, 20.0)


def _write_georeferenced_pair(src_arr, ref_arr, H_pose, src_path, ref_path) -> None:
    """Write the synthetic pair as GeoTIFFs whose geotransforms compose to `H_pose`.

    `derive_tiepoint_coarse_transform` computes H = inv(A_ref) @ A_src from each
    raster's pixel->frame affine, so the source gets A_src (identity plus the
    constant shift) and the reference gets A_ref = A_src @ inv(H_pose) - by
    construction the derived seed equals the pose that built the imagery.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    A_src = np.array([
        [1.0, 0.0, _SRC_FRAME_SHIFT[0]],
        [0.0, 1.0, _SRC_FRAME_SHIFT[1]],
        [0.0, 0.0, 1.0],
    ])
    A_ref = A_src @ np.linalg.inv(H_pose)

    for path, arr, A in ((src_path, src_arr, A_src), (ref_path, ref_arr, A_ref)):
        transform = Affine(A[0, 0], A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2])
        with rasterio.open(
            path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
            count=1, dtype="uint8", crs=CRS.from_string(_MOON_EQC_CRS),
            transform=transform,
        ) as dst:
            dst.write((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), 1)


def test_overlay_png_has_two_panels_at_scaled_reference_size(tmp_path: Path):
    H, src, ref = _pose_h()
    out = tmp_path / "overlay.png"

    render_alignment_overlay(src, ref, H, out, max_dim=200)

    assert out.exists() and out.stat().st_size > 0
    img = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert img is not None
    # Two panels (blend | checkerboard) side by side, panel scaled to max_dim.
    scale = 200 / max(_REF)
    expected = (int(_REF[0] * scale), int(2 * _REF[1] * scale))
    assert img.shape[0] == expected[0]
    assert img.shape[1] == expected[1]


def test_alignment_report_confirms_the_metadata_pose(tmp_path: Path):
    H, src, ref = _pose_h()

    report = alignment_report(src, ref, H)

    assert report["structural_ncc_at_pose"] > 0.5
    assert report["lowfreq_best_ncc"] > 0.6
    # The low-frequency search must peak where the metadata says the content is.
    assert report["peak_matches_metadata"] is True
    assert report["content_coverage"] > 0.05


def test_alignment_report_detects_a_displaced_metadata_pose():
    H, src, ref = _pose_h()

    H_wrong = H.copy()
    H_wrong[0, 2] += 180.0
    H_wrong[1, 2] += 60.0

    report = alignment_report(src, ref, H_wrong)

    # Metadata says the content sits 180/60 px away from where it really is,
    # so the search peak must disagree with the metadata placement.
    assert report["peak_matches_metadata"] is False
    assert report["structural_ncc_at_pose"] < report["lowfreq_best_ncc"]


def test_cli_writes_overlays_and_report_for_a_pair_directory(tmp_path: Path):
    H, src, ref = _pose_h()
    src_path = tmp_path / "demo_source_at_5m.tif"
    ref_path = tmp_path / "demo_reference_at_5m.tif"
    _write_georeferenced_pair(src, ref, H, src_path, ref_path)
    out_dir = tmp_path / "out"

    rc = main(["--data-dir", str(tmp_path), "--out", str(out_dir), "--max-dim", "200"])

    assert rc == 0
    assert (out_dir / "demo_overlay.png").exists()
    entries = json.loads((out_dir / "report.json").read_text())
    assert len(entries) == 1
    entry = entries[0]
    assert entry["source"].endswith("demo_source_at_5m.tif")
    assert entry["reference"].endswith("demo_reference_at_5m.tif")
    assert entry["report"]["structural_ncc_at_pose"] > 0.5
    assert entry["rotation_deg"] is not None
