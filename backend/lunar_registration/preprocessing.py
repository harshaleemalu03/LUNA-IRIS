"""
Stage 1 - Preprocessing
=======================

Supports:

- PDS3 .IMG images
- LROC NAC_PHO multi-band .IMG products
- ISIS / GeoTIFF / PNG / JPG
- Pixel-wise NAC_PHO incidence / emission / phase angle maps
- Manual / label / online angle sources
- Image normalization
- Windowed reading/cropping
- GSD resampling

NAC_PHO convention used here:

    Band 1 = calibrated image
    Band 2 = phase angle
    Band 3 = emission angle
    Band 4 = incidence angle

The important addition is that a NAC_PHO .IMG can now be used as the
ACTUAL image input, not merely as an angle source.
"""

from __future__ import annotations

import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import SensorConfig, get_sensor_config

logger = logging.getLogger(__name__)

# Mean lunar radius (m), PDS/IAU convention — turns degree tiepoints into a
# meter-scale GSD when no projected CRS accompanies the image.
R_MOON_M = 1_737_400.0

# Illumination angle tags the sidecar XML may carry; named in "missing"
# logs so a placeholder fallback never happens silently.
SIDECAR_ANGLE_TAGS = (
    "Solar_incidence_angle_in_degree",
    "Sun_elevation_in_degree",
    "Sun_azimuth_in_degree",
)


# ----------------------------------------------------------------------
# Optional dependencies
# ----------------------------------------------------------------------

try:
    import rasterio

    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ----------------------------------------------------------------------
# Loaded image container
# ----------------------------------------------------------------------

@dataclass
class LoadedImage:
    data: np.ndarray
    path: str
    sensor: SensorConfig

    incidence_deg: Optional[np.ndarray] = None
    emission_deg: Optional[np.ndarray] = None
    phase_deg: Optional[np.ndarray] = None

    gsd_m: Optional[float] = None

    geotransform: Optional[tuple] = None
    crs: Optional[object] = None

    # Scalar georeferencing + sidecar XML metadata (Task 2). None = the field
    # was absent in the file set; load time logs WHICH field, so downstream
    # placeholder defaults (scale prior 0.5, routing incidence 30) are traceable.
    pixel_scale_m: Optional[Tuple[float, float]] = None
    corner_tiepoints: Optional[list] = None
    sidecar_path: Optional[str] = None
    sidecar_incidence_deg: Optional[float] = None
    sidecar_sun_elevation_deg: Optional[float] = None
    sidecar_sun_azimuth_deg: Optional[float] = None


# ----------------------------------------------------------------------
# Metadata reader (georeferencing tags + sidecar XML)
# ----------------------------------------------------------------------

class SidecarXmlError(ValueError):
    """An explicitly supplied sidecar XML exists but cannot be used.

    WHY: the caller (API `source_xml` upload / CLI `--source-sidecar-xml`)
    named a specific file, so degrading to the logged placeholder would
    silently reroute registration on a guess — fail loudly instead.
    Subclasses ValueError so generic input-error handlers still classify
    it as bad caller input rather than an internal crash.
    """


def resolve_sidecar_xml(path: str) -> Optional[str]:
    """Find the product sidecar XML for an image.

    One XML exists per product and is shared by its source+reference rasters
    (PROD_source_at_5m.tif / PROD_reference_at_5m.tif -> PROD.xml).
    """
    directory, stem = os.path.split(path)
    stem = os.path.splitext(stem)[0]
    base = re.sub(r"_(source|reference).*$", "", stem)
    for candidate in (os.path.join(directory, base + ".xml"),
                      os.path.join(directory, stem + ".xml")):
        if os.path.exists(candidate):
            return candidate
    return None


def read_sidecar_angles(xml_path: str) -> dict:
    """Return {tag: float} for every illumination angle tag present in the XML."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    angles: dict = {}
    for tag in SIDECAR_ANGLE_TAGS:
        el = tree.find(f".//{tag}")
        if el is not None and el.text:
            try:
                angles[tag] = float(el.text)
            except ValueError:
                logger.warning("metadata: sidecar %s has non-numeric %s=%r",
                               xml_path, tag, el.text)
    return angles


def _geodesic_distance_m(x1: float, y1: float, x2: float, y2: float,
                         geographic: bool) -> float:
    """Great-circle (haversine) distance on the sphere for degree tiepoints,
    plain Euclidean otherwise. Haversine because the cos-formula catastrophically
    cancels for the short (~km) edges inside one image footprint."""
    if not geographic:
        return math.hypot(x2 - x1, y2 - y1)
    lon1, lat1, lon2, lat2 = map(math.radians, (x1, y1, x2, y2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * R_MOON_M * math.asin(min(1.0, math.sqrt(a)))


def _gsd_from_tiepoints(gcps, shape) -> Optional[float]:
    """Meter-per-pixel from corner tiepoints.

    The footprint may be ROTATED in lon/lat space (real OHRC source: lon
    changes along rows as well), so per-axis pitch must use the GCP edge that
    varies in that pixel axis only — never axis-aligned bounding spans.
    """
    if len(gcps) < 2:
        return None
    geographic = (all(abs(g.x) <= 180.0 for g in gcps)
                  and all(abs(g.y) <= 90.0 for g in gcps))

    def _edge(axis: str) -> Optional[float]:
        """Distance/px along 'row' (same col) or 'col' (same row) edges."""
        best = None
        for i, g1 in enumerate(gcps):
            for g2 in gcps[i + 1:]:
                if axis == "row":
                    d_px = abs(g2.row - g1.row)
                    same_line = g2.col == g1.col
                else:
                    d_px = abs(g2.col - g1.col)
                    same_line = g2.row == g1.row
                if same_line and d_px > 0:
                    dist = _geodesic_distance_m(g1.x, g1.y, g2.x, g2.y, geographic)
                    best = dist / d_px if best is None else min(best, dist / d_px)
        return best

    gsd_row = _edge("row")
    gsd_col = _edge("col")
    if gsd_row is None or gsd_col is None:
        return None
    return float((gsd_row + gsd_col) / 2.0)


def _pixel_scale_from_georef(georef: dict) -> Optional[Tuple[float, float]]:
    """(x_res, y_res) in CRS units for a projected, north-up raster."""
    crs = georef.get("crs")
    transform = georef.get("transform")
    if crs is None or transform is None:
        return None
    if getattr(crs, "is_geographic", False):
        return None
    a, b, _, d, e, _ = transform[:6]
    if b or d:
        logger.warning("metadata: rotated geotransform (b=%s d=%s); "
                       "pixel scale not axis-aligned, treating as missing", b, d)
        return None
    res = georef.get("res")
    x_res, y_res = (abs(float(res[0])), abs(float(res[1]))) if res else (abs(float(a)), abs(float(e)))
    return (x_res, y_res)


def attach_file_metadata(
    loaded: LoadedImage,
    georef: Optional[dict] = None,
    sidecar_xml_path: Optional[str] = None,
) -> LoadedImage:
    """Populate georeferencing + sidecar XML scalars on a freshly loaded image.

    Every absent field is logged BY NAME: the pipeline's placeholder defaults
    (scale prior 0.5, routing incidence 30°) must always be traceable to a
    specific missing input, never silent.

    An explicit sidecar_xml_path (API `source_xml` upload / CLI
    `--source-sidecar-xml`) replaces sibling auto-discovery and must be
    usable — broken explicit input raises SidecarXmlError instead of
    degrading, because the caller explicitly sent that file.
    """
    georef = georef or {}
    gcps = georef.get("gcps") or []

    if gcps:
        loaded.corner_tiepoints = [(g.row, g.col, g.x, g.y) for g in gcps]
        loaded.gsd_m = _gsd_from_tiepoints(gcps, loaded.data.shape)
        if loaded.gsd_m is None:
            logger.warning("metadata: %s has corner tiepoints but GSD could "
                           "not be derived (degenerate spans)", loaded.path)
    else:
        pixel_scale = _pixel_scale_from_georef(georef) if georef else None
        if pixel_scale is not None:
            loaded.pixel_scale_m = pixel_scale
            loaded.gsd_m = float(sum(pixel_scale) / 2.0)

    if loaded.gsd_m is None:
        logger.warning(
            "metadata: %s missing georeferencing (no projected pixel scale "
            "and no corner tiepoints) — GSD scale prior falls back to default",
            loaded.path,
        )

    if sidecar_xml_path is not None:
        # Explicit input path: the caller named THIS file, so every failure
        # mode below must raise — silently falling back to placeholders
        # would reroute registration without the metadata the client
        # explicitly sent.
        xml_path = sidecar_xml_path
        if not os.path.isfile(xml_path):
            raise SidecarXmlError(
                f"source sidecar XML not found: {xml_path}"
            )
        try:
            angles = read_sidecar_angles(xml_path)
        except ET.ParseError as exc:
            raise SidecarXmlError(
                f"source sidecar XML is not well-formed: {xml_path} ({exc})"
            ) from exc
        if not angles:
            raise SidecarXmlError(
                f"source sidecar XML {xml_path} contains none of the "
                f"recognized angle tags: {', '.join(SIDECAR_ANGLE_TAGS)}"
            )
    else:
        xml_path = resolve_sidecar_xml(loaded.path)
        if xml_path is None:
            logger.warning(
                "metadata: %s missing sidecar XML (expected a product .xml "
                "next to the image with tags: %s) — illumination angles "
                "unavailable",
                loaded.path, ", ".join(SIDECAR_ANGLE_TAGS),
            )
            return loaded
        angles = read_sidecar_angles(xml_path)

    loaded.sidecar_path = xml_path
    field_by_tag = {
        "Solar_incidence_angle_in_degree": "sidecar_incidence_deg",
        "Sun_elevation_in_degree": "sidecar_sun_elevation_deg",
        "Sun_azimuth_in_degree": "sidecar_sun_azimuth_deg",
    }
    for tag, field in field_by_tag.items():
        if tag in angles:
            setattr(loaded, field, angles[tag])
        else:
            logger.warning("metadata: %s missing sidecar tag %s in %s",
                           loaded.path, tag, xml_path)
    return loaded


# ----------------------------------------------------------------------
# Sensor detection
# ----------------------------------------------------------------------

def detect_sensor(path: str, label_text: str = "") -> SensorConfig:
    """
    Guess sensor from filename + optional label text.
    """

    haystack = (os.path.basename(path) + " " + label_text).upper()

    for name, cfg in {
        "OHRC": get_sensor_config("OHRC"),
        "TMC": get_sensor_config("TMC"),
        "IIRS": get_sensor_config("IIRS"),
        "LROC": get_sensor_config("LROC"),
    }.items():

        for kw in cfg.label_keywords:
            if kw in haystack:
                return cfg

    warnings.warn(
        f"Could not auto-detect sensor for '{path}'. "
        "Defaulting to LROC."
    )

    # Neutral fallback: LROC (the historical reference-image default). OHRC
    # is now detected by its OHRXXD product prefix, so unknown names no
    # longer silently inherit OHRC's placeholder GSD.
    return get_sensor_config("LROC")


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

def _normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image to float32 [0, 1].

    Uses robust 0.5 / 99.5 percentiles.
    """

    img = img.astype(np.float32)

    finite = img[np.isfinite(img)]

    if finite.size == 0:
        return np.zeros_like(img, dtype=np.float32)

    lo, hi = np.percentile(finite, (0.5, 99.5))

    if hi <= lo:
        hi = float(np.max(finite))
        lo = float(np.min(finite))

    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    img = np.clip(
        (img - lo) / max(hi - lo, 1e-6),
        0.0,
        1.0,
    )

    return img.astype(np.float32)


# ----------------------------------------------------------------------
# PDS3 label helpers
# ----------------------------------------------------------------------

def _find_pds3_label_size(path: str) -> int:
    """
    Find PDS3 ASCII label size using:

        RECORD_BYTES * LABEL_RECORDS
    """

    with open(path, "rb") as f:
        header = f.read(20000).decode(
            "latin-1",
            errors="ignore",
        )

    record_match = re.search(
        r"RECORD_BYTES\s*=\s*(\d+)",
        header,
        re.IGNORECASE,
    )

    label_match = re.search(
        r"LABEL_RECORDS\s*=\s*(\d+)",
        header,
        re.IGNORECASE,
    )

    if record_match and label_match:
        return (
            int(record_match.group(1))
            * int(label_match.group(1))
        )

    warnings.warn(
        f"Could not parse PDS3 label size for '{path}'. "
        "Assuming zero offset."
    )

    return 0


def _read_label_text(
    path: str,
    max_bytes: int = 200_000,
) -> str:
    """
    Read the ASCII PDS3 label.
    """

    label_size = _find_pds3_label_size(path)

    read_size = (
        label_size
        if label_size > 0
        else max_bytes
    )

    with open(path, "rb") as f:
        return f.read(read_size).decode(
            "latin-1",
            errors="ignore",
        )


def _find_pds3_image_dimensions(
    path: str,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse:

        LINES
        LINE_SAMPLES

    from a PDS3 label.
    """

    text = _read_label_text(path)

    lines_match = re.search(
        r"\bLINES\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    samples_match = re.search(
        r"\bLINE_SAMPLES\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    lines = (
        int(lines_match.group(1))
        if lines_match
        else None
    )

    samples = (
        int(samples_match.group(1))
        if samples_match
        else None
    )

    return lines, samples


# ----------------------------------------------------------------------
# Standard single-band PDS3 .IMG reader
# ----------------------------------------------------------------------

def _read_raw_img(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Read a simple single-band uint8 PDS3 .IMG.

    This is primarily for the original LROC EDR products such as:

        M129133239RE.IMG
        M150368601RE.IMG

    It is NOT used for NAC_PHO products.
    """

    # --------------------------------------------------------------
    # Try planetaryimage first
    # --------------------------------------------------------------

    try:

        from planetaryimage import PDS3Image

        img = PDS3Image.open(path)

        arr = np.asarray(img.image)

        if arr.ndim > 2:
            arr = np.squeeze(arr)

        if window is not None:
            x, y, w, h = window
            arr = arr[
                y:y + h,
                x:x + w,
            ]

        return arr

    except Exception as e:

        warnings.warn(
            f"planetaryimage failed to load '{path}' "
            f"({type(e).__name__}: {e}). "
            "Attempting raw PDS3 fallback."
        )

    # --------------------------------------------------------------
    # Raw fallback
    # --------------------------------------------------------------

    label_size = _find_pds3_label_size(path)

    lines, samples = _find_pds3_image_dimensions(path)

    if lines is None or samples is None:
        raise ValueError(
            f"Could not determine LINES/LINE_SAMPLES "
            f"for '{path}'."
        )

    with open(path, "rb") as f:

        f.seek(label_size)

        raw = np.fromfile(
            f,
            dtype=np.uint8,
            count=lines * samples,
        )

    expected = lines * samples

    if raw.size < expected:
        raise ValueError(
            f"'{path}' contains only {raw.size} image bytes "
            f"but the label requires {expected}."
        )

    arr = raw.reshape(
        lines,
        samples,
    )

    if window is not None:

        x, y, w, h = window

        arr = arr[
            y:y + h,
            x:x + w,
        ]

    return arr


# ----------------------------------------------------------------------
# NAC_PHO helpers
# ----------------------------------------------------------------------

_NAC_PHO_BAND_NAME_PATTERNS = {

    "phase_deg": re.compile(
        r"phase\s*angle",
        re.IGNORECASE,
    ),

    "emission_deg": re.compile(
        r"emission\s*angle",
        re.IGNORECASE,
    ),

    "incidence_deg": re.compile(
        r"incidence\s*angle",
        re.IGNORECASE,
    ),
}


def _is_nac_pho(path: str) -> bool:
    """
    Detect whether a file is an LROC NAC_PHO product.

    We deliberately use the filename rather than assuming every .IMG
    is NAC_PHO.
    """

    name = os.path.basename(path).upper()

    return (
        "NAC_PHO" in name
        or "_PHO_" in name
    )


def _read_nac_pho_with_rasterio(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
):
    """
    Attempt to read NAC_PHO using rasterio/GDAL.

    IMPORTANT:
    Only the requested window is read, preventing us from loading the
    entire 4-band NAC_PHO product into RAM.

    NAC_PHO convention:
        Band 1 = calibrated image
        Band 2 = phase angle
        Band 3 = emission angle
        Band 4 = incidence angle

    Invalid/no-data angle values are converted to NaN.
    """

    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is not installed."
        )

    with rasterio.open(path) as ds:

        if ds.count < 4:
            raise ValueError(
                f"NAC_PHO product '{path}' has only "
                f"{ds.count} bands. Expected at least 4."
            )

        # ----------------------------------------------------------
        # Rasterio window
        # ----------------------------------------------------------

        rio_window = None

        if window is not None:

            x, y, w, h = window

            rio_window = rasterio.windows.Window(
                col_off=x,
                row_off=y,
                width=w,
                height=h,
            )

        # ----------------------------------------------------------
        # Read Band 1 = image
        # ----------------------------------------------------------

        image = ds.read(
            1,
            window=rio_window,
        )

        # ----------------------------------------------------------
        # Read geometry bands
        # ----------------------------------------------------------

        phase = ds.read(
            2,
            window=rio_window,
        ).astype(np.float32)

        emission = ds.read(
            3,
            window=rio_window,
        ).astype(np.float32)

        incidence = ds.read(
            4,
            window=rio_window,
        ).astype(np.float32)

        # ----------------------------------------------------------
        # Clean invalid NAC_PHO geometry values
        # ----------------------------------------------------------
        #
        # NAC_PHO uses very large negative values such as
        # -3.4028227e+38 to represent invalid/no-data pixels.
        #
        # These are NOT real angles, so convert them to NaN.
        # ----------------------------------------------------------

        for arr in (incidence, emission, phase):

            # Remove NaN/Inf values
            arr[~np.isfinite(arr)] = np.nan

            # Remove PDS3 invalid/no-data values
            arr[arr < -1e20] = np.nan

        # ----------------------------------------------------------
        # Preserve geospatial metadata
        # ----------------------------------------------------------

        geotransform = ds.transform
        crs = ds.crs

    return (
        image,
        incidence,
        emission,
        phase,
        geotransform,
        crs,
    )


def _read_nac_pho_raw(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
):
    """
    Fallback NAC_PHO reader.

    This handles the case where GDAL/rasterio cannot open the PDS3
    NAC_PHO .IMG.

    The reader first parses the PDS3 label to determine:

        LINES
        LINE_SAMPLES
        SAMPLE_BITS
        SAMPLE_TYPE
        BANDS
        BAND_STORAGE_TYPE

    It supports the common BSQ layout used by PDS3 products.

    If the label describes another storage layout, we fail loudly
    instead of silently returning corrupted data.
    """

    text = _read_label_text(path)

    # --------------------------------------------------------------
    # Basic dimensions
    # --------------------------------------------------------------

    lines_match = re.search(
        r"\bLINES\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    samples_match = re.search(
        r"\bLINE_SAMPLES\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    bands_match = re.search(
        r"\bBANDS\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    if not lines_match or not samples_match:
        raise ValueError(
            f"Could not determine NAC_PHO dimensions "
            f"from '{path}'."
        )

    lines = int(lines_match.group(1))
    samples = int(samples_match.group(1))

    bands = (
        int(bands_match.group(1))
        if bands_match
        else 1
    )

    if bands < 4:
        raise ValueError(
            f"NAC_PHO product '{path}' reports "
            f"{bands} band(s), expected at least 4."
        )

    # --------------------------------------------------------------
    # Pixel format
    # --------------------------------------------------------------

    sample_bits_match = re.search(
        r"\bSAMPLE_BITS\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    sample_bits = (
        int(sample_bits_match.group(1))
        if sample_bits_match
        else 32
    )

    sample_type_match = re.search(
        r"\bSAMPLE_TYPE\s*=\s*([A-Z0-9_]+)",
        text,
        re.IGNORECASE,
    )

    sample_type = (
        sample_type_match.group(1).upper()
        if sample_type_match
        else ""
    )

    # --------------------------------------------------------------
    # Determine dtype
    # --------------------------------------------------------------

    if sample_bits == 8:

        dtype = np.uint8

    elif sample_bits == 16:

        if "MSB" in sample_type:

            dtype = np.dtype(">i2")

        elif "UNSIGNED" in sample_type:

            dtype = np.dtype(">u2")

        else:

            dtype = np.dtype(">i2")

    elif sample_bits == 32:

        if "REAL" in sample_type or "FLOAT" in sample_type:

            dtype = np.dtype(">f4")

        elif "UNSIGNED" in sample_type:

            dtype = np.dtype(">u4")

        else:

            dtype = np.dtype(">i4")

    else:

        raise ValueError(
            f"Unsupported NAC_PHO SAMPLE_BITS="
            f"{sample_bits} in '{path}'."
        )

    # --------------------------------------------------------------
    # Storage type
    # --------------------------------------------------------------

    storage_match = re.search(
        r"\bBAND_STORAGE_TYPE\s*=\s*([A-Z0-9_]+)",
        text,
        re.IGNORECASE,
    )

    storage = (
        storage_match.group(1).upper()
        if storage_match
        else "BSQ"
    )

    if storage != "BSQ":

        raise ValueError(
            f"NAC_PHO '{path}' uses "
            f"BAND_STORAGE_TYPE={storage}. "
            "The raw fallback currently supports BSQ only."
        )

    # --------------------------------------------------------------
    # Label offset
    # --------------------------------------------------------------

    label_size = _find_pds3_label_size(path)

    # --------------------------------------------------------------
    # Window
    # --------------------------------------------------------------

    if window is None:

        x = 0
        y = 0
        w = samples
        h = lines

    else:

        x, y, w, h = window

    if x < 0 or y < 0:
        raise ValueError(
            f"Invalid NAC_PHO window {window}."
        )

    if x + w > samples or y + h > lines:

        raise ValueError(
            f"NAC_PHO window {window} exceeds "
            f"image dimensions {(samples, lines)}."
        )

    # --------------------------------------------------------------
    # Bytes per pixel
    # --------------------------------------------------------------

    bytes_per_sample = sample_bits // 8

    # One complete band
    band_bytes = (
        lines
        * samples
        * bytes_per_sample
    )

    # --------------------------------------------------------------
    # Read only the requested region from each band
    # --------------------------------------------------------------

    image = np.empty(
        (h, w),
        dtype=np.float32,
    )

    phase = np.empty(
        (h, w),
        dtype=np.float32,
    )

    emission = np.empty(
        (h, w),
        dtype=np.float32,
    )

    incidence = np.empty(
        (h, w),
        dtype=np.float32,
    )

    # --------------------------------------------------------------
    # Helper for reading one band window
    # --------------------------------------------------------------

    def read_band(
        band_number: int,
    ) -> np.ndarray:

        band_offset = (
            label_size
            + (band_number - 1)
            * band_bytes
        )

        out = np.empty(
            (h, w),
            dtype=dtype,
        )

        row_bytes = (
            samples
            * bytes_per_sample
        )

        wanted_bytes = (
            w
            * bytes_per_sample
        )

        with open(path, "rb") as f:

            for row in range(h):

                source_row = y + row

                offset = (
                    band_offset
                    + source_row * row_bytes
                    + x * bytes_per_sample
                )

                f.seek(offset)

                raw = f.read(wanted_bytes)

                if len(raw) != wanted_bytes:

                    raise IOError(
                        f"Unexpected EOF while reading "
                        f"band {band_number}, row {row} "
                        f"from '{path}'."
                    )

                out[row, :] = np.frombuffer(
                    raw,
                    dtype=dtype,
                    count=w,
                )

        return out

    # --------------------------------------------------------------
    # NAC_PHO bands
    # --------------------------------------------------------------

    image[:, :] = read_band(1)

    phase[:, :] = read_band(2)

    emission[:, :] = read_band(3)

    incidence[:, :] = read_band(4)

    # --------------------------------------------------------------
    # Convert endian / dtype to native float32
    # --------------------------------------------------------------

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    phase = np.asarray(
        phase,
        dtype=np.float32,
    )

    emission = np.asarray(
        emission,
        dtype=np.float32,
    )

    incidence = np.asarray(
        incidence,
        dtype=np.float32,
    )

    return (
        image,
        incidence,
        emission,
        phase,
        None,
        None,
    )


def _read_nac_pho(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
):
    """
    Read an LROC NAC_PHO product.

    Strategy:

        1. Try rasterio/GDAL.
        2. If rasterio cannot open the PDS3 .IMG, use our raw PDS3
           BSQ reader.

    Returns:

        image
        incidence
        emission
        phase
        geotransform
        crs
    """

    # --------------------------------------------------------------
    # First attempt: rasterio
    # --------------------------------------------------------------

    try:

        return _read_nac_pho_with_rasterio(
            path,
            window=window,
        )

    except Exception as e:

        warnings.warn(
            f"rasterio could not open NAC_PHO file "
            f"'{path}' ({type(e).__name__}: {e}). "
            "Falling back to the native PDS3 NAC_PHO reader."
        )

    # --------------------------------------------------------------
    # Second attempt: raw PDS3 reader
    # --------------------------------------------------------------

    return _read_nac_pho_raw(
        path,
        window=window,
    )


# ----------------------------------------------------------------------
# Main image loader
# ----------------------------------------------------------------------

def load_image(
    path: str,
    sensor_hint: Optional[str] = None,
    window: Optional[Tuple[int, int, int, int]] = None,
    angles_from_label: bool = False,
    fetch_angles_online: bool = False,
    manual_incidence_deg: Optional[float] = None,
    manual_emission_deg: Optional[float] = None,
    manual_phase_deg: Optional[float] = None,
    sidecar_xml_path: Optional[str] = None,
    nac_pho_path: Optional[str] = None,
    nac_pho_band_phase: int = 2,
    nac_pho_band_emission: int = 3,
    nac_pho_band_incidence: int = 4,
) -> LoadedImage:
    """
    Load an image.

    IMPORTANT NEW BEHAVIOUR:

    If `path` itself is an LROC NAC_PHO product, Band 1 becomes the
    actual image and Bands 2/3/4 become the pixel-wise geometry maps.

    Therefore this now works:

        load_image(
            "NAC_PHO_E010N0230_M129133239R.IMG",
            sensor_hint="LROC",
            window=(x, y, w, h),
        )

    and gives:

        loaded.data
        loaded.phase_deg
        loaded.emission_deg
        loaded.incidence_deg

    all on the SAME pixel grid.
    """

    ext = os.path.splitext(path)[1].lower()

    geotransform = None
    crs = None
    georef: Optional[dict] = None

    # ==============================================================
    # NAC_PHO DIRECT MODE
    # ==============================================================

    if ext == ".img" and _is_nac_pho(path):

        (
            arr,
            incidence,
            emission,
            phase,
            geotransform,
            crs,
        ) = _read_nac_pho(
            path,
            window=window,
        )

        sensor = (
            get_sensor_config(sensor_hint)
            if sensor_hint
            else get_sensor_config("LROC")
        )

        normalized = _normalize(arr)

        loaded = LoadedImage(
            data=normalized,
            path=path,
            sensor=sensor,
            incidence_deg=incidence,
            emission_deg=emission,
            phase_deg=phase,
            geotransform=geotransform,
            crs=crs,
        )

        # ----------------------------------------------------------
        # If another NAC_PHO path was explicitly supplied, let it
        # override the angle source.
        # ----------------------------------------------------------

        attach_file_metadata(loaded, None, sidecar_xml_path=sidecar_xml_path)

        if nac_pho_path is not None:

            attach_nac_pho_angles(
                loaded,
                nac_pho_path,
                window=window,
                band_phase=nac_pho_band_phase,
                band_emission=nac_pho_band_emission,
                band_incidence=nac_pho_band_incidence,
            )

        return loaded

    # ==============================================================
    # NORMAL IMAGE MODE
    # ==============================================================

    if ext == ".img":

        arr = _read_raw_img(
            path,
            window=window,
        )

    elif (
        ext in (".cub", ".tif", ".tiff")
        and _HAS_RASTERIO
    ):

        with rasterio.open(path) as ds:

            if window is None:

                arr = ds.read(1)

            else:

                x, y, w, h = window

                rio_window = rasterio.windows.Window(
                    col_off=x,
                    row_off=y,
                    width=w,
                    height=h,
                )

                arr = ds.read(
                    1,
                    window=rio_window,
                )

            geotransform = ds.transform
            crs = ds.crs
            georef = {
                "transform": ds.transform,
                "crs": ds.crs,
                "res": ds.res,
                "gcps": ds.gcps[0],
            }

    elif (
        ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        and _HAS_PIL
    ):

        arr = np.array(
            Image.open(path).convert("F")
        )

        if window is not None:

            x, y, w, h = window

            arr = arr[
                y:y + h,
                x:x + w,
            ]

    else:

        raise ValueError(
            f"Unsupported file type '{ext}' or missing dependency "
            f"(rasterio={_HAS_RASTERIO}, PIL={_HAS_PIL})"
        )

    # ==============================================================
    # Construct LoadedImage
    # ==============================================================

    sensor = (
        get_sensor_config(sensor_hint)
        if sensor_hint
        else detect_sensor(path)
    )

    normalized = _normalize(arr)

    loaded = LoadedImage(
        data=normalized,
        path=path,
        sensor=sensor,
        geotransform=geotransform,
        crs=crs,
    )

    attach_file_metadata(loaded, georef, sidecar_xml_path=sidecar_xml_path)

    # ==============================================================
    # Attach angle source
    # ==============================================================

    if nac_pho_path is not None:

        attach_nac_pho_angles(
            loaded,
            nac_pho_path,
            window=window,
            band_phase=nac_pho_band_phase,
            band_emission=nac_pho_band_emission,
            band_incidence=nac_pho_band_incidence,
        )

    elif (
        manual_incidence_deg is not None
        and manual_emission_deg is not None
    ):

        (
            loaded.incidence_deg,
            loaded.emission_deg,
            loaded.phase_deg,
        ) = load_angles_manual(
            manual_incidence_deg,
            manual_emission_deg,
            loaded.data.shape,
            phase_deg=manual_phase_deg,
        )

    elif fetch_angles_online:

        attach_lroc_fetched_angles(loaded)

    elif angles_from_label:

        attach_label_angles(loaded)

    return loaded


# ----------------------------------------------------------------------
# Separate angle-map loader
# ----------------------------------------------------------------------

def load_angle_maps(
    incidence_path: str,
    emission_path: str,
    phase_path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load separate incidence/emission/phase rasters.
    """

    def _load(p):

        if _HAS_RASTERIO:

            with rasterio.open(p) as ds:

                if window is None:

                    a = ds.read(1)

                else:

                    x, y, w, h = window

                    rio_window = rasterio.windows.Window(
                        col_off=x,
                        row_off=y,
                        width=w,
                        height=h,
                    )

                    a = ds.read(
                        1,
                        window=rio_window,
                    )

                return a.astype(np.float32)

        elif _HAS_PIL:

            a = np.array(
                Image.open(p)
            ).astype(np.float32)

            if window is not None:

                x, y, w, h = window

                a = a[
                    y:y + h,
                    x:x + w,
                ]

            return a

        else:

            raise ImportError(
                "Need rasterio or PIL to read angle-map rasters."
            )

    return (
        _load(incidence_path),
        _load(emission_path),
        _load(phase_path),
    )


# ----------------------------------------------------------------------
# NAC_PHO angle loader
# ----------------------------------------------------------------------

def _detect_nac_pho_bands(ds) -> dict:
    """
    Best-effort detection of angle bands from raster metadata.

    Defaults remain:

        phase     = 2
        emission  = 3
        incidence = 4
    """

    found = {}

    try:

        for band_idx in range(
            1,
            ds.count + 1,
        ):

            desc = (
                ds.descriptions[band_idx - 1]
                or ""
            )

            tags = ds.tags(
                band_idx
            ) or {}

            haystack = " ".join(
                [desc]
                + [str(v) for v in tags.values()]
            )

            for key, pattern in (
                _NAC_PHO_BAND_NAME_PATTERNS.items()
            ):

                if (
                    key not in found
                    and pattern.search(haystack)
                ):

                    found[key] = band_idx

    except Exception:

        return {}

    return found


def load_angles_from_nac_pho(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
    band_phase: int = 2,
    band_emission: int = 3,
    band_incidence: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pixel-wise NAC_PHO angle maps.

    Returns:

        incidence
        emission
        phase

    in that order.
    """

    # --------------------------------------------------------------
    # If this is a NAC_PHO PDS3 IMG, use our dedicated reader.
    # --------------------------------------------------------------

    if (
        os.path.splitext(path)[1].lower() == ".img"
        and _is_nac_pho(path)
    ):

        (
            _image,
            incidence,
            emission,
            phase,
            _geotransform,
            _crs,
        ) = _read_nac_pho(
            path,
            window=window,
        )

        return (
            incidence,
            emission,
            phase,
        )

    # --------------------------------------------------------------
    # Otherwise use rasterio.
    # --------------------------------------------------------------

    if not _HAS_RASTERIO:

        raise ImportError(
            "rasterio is required to read this NAC_PHO angle source."
        )

    with rasterio.open(path) as ds:

        required_band = max(
            band_phase,
            band_emission,
            band_incidence,
        )

        if ds.count < required_band:

            raise ValueError(
                f"'{path}' has only {ds.count} band(s), "
                f"but band {required_band} is required."
            )

        detected = _detect_nac_pho_bands(ds)

        b_phase = detected.get(
            "phase_deg",
            band_phase,
        )

        b_emission = detected.get(
            "emission_deg",
            band_emission,
        )

        b_incidence = detected.get(
            "incidence_deg",
            band_incidence,
        )

        if not detected:

            warnings.warn(
                f"'{path}': could not determine band names. "
                f"Using phase={band_phase}, "
                f"emission={band_emission}, "
                f"incidence={band_incidence}."
            )

        # ----------------------------------------------------------
        # Read only requested window
        # ----------------------------------------------------------

        rio_window = None

        if window is not None:

            x, y, w, h = window

            rio_window = rasterio.windows.Window(
                col_off=x,
                row_off=y,
                width=w,
                height=h,
            )

        phase = ds.read(
            b_phase,
            window=rio_window,
        ).astype(np.float32)

        emission = ds.read(
            b_emission,
            window=rio_window,
        ).astype(np.float32)

        incidence = ds.read(
            b_incidence,
            window=rio_window,
        ).astype(np.float32)

    # --------------------------------------------------------------
    # Clean invalid/special pixels
    # --------------------------------------------------------------

    for arr in (
        incidence,
        emission,
        phase,
    ):

        arr[~np.isfinite(arr)] = np.nan

        arr[
            (arr < -1e3)
            | (arr > 1e3)
        ] = np.nan

    return (
        incidence,
        emission,
        phase,
    )


# ----------------------------------------------------------------------
# Attach NAC_PHO angles to LoadedImage
# ----------------------------------------------------------------------

def attach_nac_pho_angles(
    loaded: LoadedImage,
    nac_pho_path: str,
    window: Optional[Tuple[int, int, int, int]] = None,
    band_phase: int = 2,
    band_emission: int = 3,
    band_incidence: int = 4,
) -> LoadedImage:
    """
    Attach NAC_PHO angle maps to an already-loaded image.
    """

    try:

        (
            loaded.incidence_deg,
            loaded.emission_deg,
            loaded.phase_deg,
        ) = load_angles_from_nac_pho(
            nac_pho_path,
            window=window,
            band_phase=band_phase,
            band_emission=band_emission,
            band_incidence=band_incidence,
        )

        # Clean NAC_PHO no-data values
        loaded.incidence_deg = loaded.incidence_deg.astype(np.float32)
        loaded.emission_deg = loaded.emission_deg.astype(np.float32)
        loaded.phase_deg = loaded.phase_deg.astype(np.float32)

        loaded.incidence_deg[loaded.incidence_deg < -1e20] = np.nan
        loaded.emission_deg[loaded.emission_deg < -1e20] = np.nan
        loaded.phase_deg[loaded.phase_deg < -1e20] = np.nan

    except Exception as e:

        warnings.warn(
            f"attach_nac_pho_angles failed for "
            f"'{nac_pho_path}' "
            f"({type(e).__name__}: {e}) - "
            "continuing without photometric weighting."
        )

    return loaded


# ----------------------------------------------------------------------
# Label-based angle shortcut
# ----------------------------------------------------------------------

_ANGLE_FIELD_PATTERNS = {

    "incidence_deg":
        r"INCIDENCE_ANGLE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "emission_deg":
        r"EMISSION_ANGLE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "phase_deg":
        r"PHASE_ANGLE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "sub_solar_azimuth_deg":
        r"SUB_SOLAR_AZIMUTH\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "sub_solar_latitude_deg":
        r"SUB_SOLAR_LATITUDE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",

    "sub_solar_longitude_deg":
        r"SUB_SOLAR_LONGITUDE\s*=\s*([-+]?[0-9]*\.?[0-9]+)",
}


def parse_label_angles(path: str) -> dict:
    """
    Parse incidence/emission/phase angles from PDS3 label.
    """

    text = _read_label_text(path)

    values = {}
    missing = []

    for key, pattern in _ANGLE_FIELD_PATTERNS.items():

        m = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if m:

            values[key] = float(
                m.group(1)
            )

        else:

            missing.append(key)

    required = (
        "incidence_deg",
        "emission_deg",
    )

    missing_required = [
        k
        for k in required
        if k in missing
    ]

    if missing_required:

        raise ValueError(
            f"Label for '{path}' is missing "
            f"required angle field(s): "
            f"{missing_required}. "
            "Fall back to phocube or another geometry source."
        )

    if missing:

        warnings.warn(
            f"Label for '{path}' is missing "
            f"optional field(s): {missing}"
        )

    return values


def load_angles_from_label(
    path: str,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Broadcast scene-level angles to the requested image shape.
    """

    values = parse_label_angles(path)

    incidence = np.full(
        shape,
        values["incidence_deg"],
        dtype=np.float32,
    )

    emission = np.full(
        shape,
        values["emission_deg"],
        dtype=np.float32,
    )

    phase = np.full(
        shape,
        values.get(
            "phase_deg",
            np.nan,
        ),
        dtype=np.float32,
    )

    return (
        incidence,
        emission,
        phase,
    )


def attach_label_angles(
    loaded: LoadedImage,
) -> LoadedImage:
    """
    Attach label-derived angle maps.
    """

    try:

        (
            loaded.incidence_deg,
            loaded.emission_deg,
            loaded.phase_deg,
        ) = load_angles_from_label(
            loaded.path,
            loaded.data.shape,
        )

    except ValueError as e:

        warnings.warn(str(e))

    return loaded


# ----------------------------------------------------------------------
# Manual angles
# ----------------------------------------------------------------------

def load_angles_manual(
    incidence_deg: float,
    emission_deg: float,
    shape: Tuple[int, int],
    phase_deg: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Broadcast manually supplied angles.
    """

    incidence = np.full(
        shape,
        incidence_deg,
        dtype=np.float32,
    )

    emission = np.full(
        shape,
        emission_deg,
        dtype=np.float32,
    )

    phase = np.full(
        shape,
        (
            phase_deg
            if phase_deg is not None
            else np.nan
        ),
        dtype=np.float32,
    )

    return (
        incidence,
        emission,
        phase,
    )


# ----------------------------------------------------------------------
# LROC ODE online angle fetching
# ----------------------------------------------------------------------

def derive_lroc_product_id(
    path: str,
) -> str:
    """
    Example:

        M109080308RE.IMG
        ->
        M109080308RE
    """

    return os.path.splitext(
        os.path.basename(path)
    )[0]


_LROC_ODE_ANGLE_PATTERNS = {

    "incidence_deg":
        r"Incidence angle\s+([-\d.]+)",

    "emission_deg":
        r"Emission angle\s+([-\d.]+)",

    "phase_deg":
        r"Phase angle\s+([-\d.]+)",
}


def fetch_lroc_angles(
    product_id: str,
    dataset: str = "LRO-L-LROC-2-EDR-V1.0",
    timeout: float = 15.0,
) -> dict:
    """
    Fetch angle values from LROC ODE.
    """

    import requests

    url = (
        "https://data.lroc.im-ldi.com/"
        f"lroc/view_lroc/{dataset}/{product_id}"
    )

    resp = requests.get(
        url,
        timeout=timeout,
    )

    resp.raise_for_status()

    clean = re.sub(
        r"<[^>]+>",
        " ",
        resp.text,
    )

    clean = re.sub(
        r"[|]",
        " ",
        clean,
    )

    values = {}
    missing = []

    for key, pattern in (
        _LROC_ODE_ANGLE_PATTERNS.items()
    ):

        m = re.search(
            pattern,
            clean,
            re.IGNORECASE,
        )

        if m:

            values[key] = float(
                m.group(1)
            )

        else:

            missing.append(key)

    if (
        "incidence_deg" in missing
        or "emission_deg" in missing
    ):

        raise ValueError(
            f"Could not parse required angle "
            f"field(s) {missing} from {url}."
        )

    return values


def attach_lroc_fetched_angles(
    loaded: LoadedImage,
    dataset: str = "LRO-L-LROC-2-EDR-V1.0",
) -> LoadedImage:
    """
    Fetch LROC angle values and broadcast them to the image.
    """

    product_id = derive_lroc_product_id(
        loaded.path
    )

    try:

        values = fetch_lroc_angles(
            product_id,
            dataset=dataset,
        )

        (
            loaded.incidence_deg,
            loaded.emission_deg,
            loaded.phase_deg,
        ) = load_angles_manual(
            values["incidence_deg"],
            values["emission_deg"],
            loaded.data.shape,
            phase_deg=values.get(
                "phase_deg"
            ),
        )

    except Exception as e:

        warnings.warn(
            f"fetch_lroc_angles failed for "
            f"'{product_id}' "
            f"({type(e).__name__}: {e}). "
            "Continuing without photometric weighting."
        )

    return loaded


# ----------------------------------------------------------------------
# Tiepoint-derived coarse transform (Stage 1.6, Task 12)
# ----------------------------------------------------------------------

# Assumed source CRS for eval-set rasters that carry lon/lat GCPs but no
# declared CRS (rasterio reports crs=None): lunar lon/lat on the 1737.4 km
# sphere. The reference's declared CRS anchors the composition; a wrong
# assumption yields a seed the matcher's inlier support rejects — the seed
# is evidence, never authority.
ASSUMED_MOON_LONGLAT_CRS = "+proj=longlat +R=1737400 +no_defs"

# Magnitude fallback for crs-less mappings: lunar geographic coordinates
# stay within +/-360 longitude (the eval data mixes -180..180 and 0..360
# conventions) and +/-90 latitude; projected moon grids (polar
# stereographic metres) run to tens of thousands. A wrong read composes
# mixed frames into an absurd seed that fails inlier support downstream.
_GEOGRAPHIC_X_MAX = 360.0
_GEOGRAPHIC_Y_MAX = 90.0

# rasterio returns this when a dataset has no geotransform at all; an
# identity affine maps pixel to pixel and carries no georeferencing.
_IDENTITY_TRANSFORM = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _mapping_from_image(
    image: "LoadedImage", offset: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], list, list, str]:
    """Resolve one image's pixel->frame mapping.

    Returns (A, sample_px, sample_xy, status). Prefers corner tiepoints
    (the label's explicit control points); falls back to the raster's own
    geotransform shifted into the crop frame — eval-set references carry a
    real transform but zero GCPs, sources the reverse. status is "ok" /
    "missing" / "degenerate"; the caller renders it into a reason with
    side + path so refusals stay traceable.
    """
    col_off, row_off = offset
    pts = image.corner_tiepoints
    if pts and len(pts) >= 3:
        sample_px = [(p[1] - col_off, p[0] - row_off) for p in pts]
        sample_xy = [(p[2], p[3]) for p in pts]
        uv1 = np.array([[u, v, 1.0] for u, v in sample_px], dtype=np.float64)
        if np.linalg.matrix_rank(uv1) < 3:
            return None, sample_px, sample_xy, "degenerate"
        fit, *_ = np.linalg.lstsq(uv1, np.asarray(sample_xy), rcond=None)
        A = np.array([
            [fit[0, 0], fit[1, 0], fit[2, 0]],
            [fit[0, 1], fit[1, 1], fit[2, 1]],
            [0.0, 0.0, 1.0],
        ])
        return A, sample_px, sample_xy, "ok"

    gt = image.geotransform
    if gt is not None:
        a, b, c, d, e, f = (float(v) for v in tuple(gt)[:6])
        if (a, b, c, d, e, f) != _IDENTITY_TRANSFORM:
            # Full-raster x = a*col + b*row + c with col = u + x0, row =
            # v + y0 in the crop frame -> shift the origin, exact like the
            # tiepoint path (load_image slices at window=(x, y, w, h)).
            A = np.array([
                [a, b, c + a * col_off + b * row_off],
                [d, e, f + d * col_off + e * row_off],
                [0.0, 0.0, 1.0],
            ])
            h, w = image.data.shape[:2]
            sample_px = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
            sample_xy = [
                (A[0, 0] * u + A[0, 1] * v + A[0, 2],
                 A[1, 0] * u + A[1, 1] * v + A[1, 2])
                for u, v in sample_px
            ]
            return A, sample_px, sample_xy, "ok"
    return None, [], [], "missing"


def _frame_class(sample_xy: list, crs: Optional[object]) -> str:
    """\"geographic\" or \"projected\" for one side's map frame. A declared
    CRS wins; otherwise the coordinate magnitudes decide (see
    _GEOGRAPHIC_* — a heuristic whose mistakes become seeds the matcher's
    inlier support rejects, never trusted guesses)."""
    if crs is not None:
        return "geographic" if getattr(crs, "is_geographic", False) else "projected"
    xs = [abs(p[0]) for p in sample_xy]
    ys = [abs(p[1]) for p in sample_xy]
    if max(xs) <= _GEOGRAPHIC_X_MAX and max(ys) <= _GEOGRAPHIC_Y_MAX:
        return "geographic"
    return "projected"


def _reproject_and_compose(
    mappings: Dict[str, tuple],
    src_frame: str,
    ref_frame: str,
    result: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Compose H after reprojecting the geographic side's frame
    coordinates into the projected side's declared CRS (the eval set's
    real situation: lon/lat GCPs without CRS vs Polar Stereographic
    metres with one — Phase 0's \"tiepoint warp ~0\" root cause).

    On refusal: fills result['reason'] and returns None — a frame we
    cannot reconcile must be reported, never composed into a confident
    lie. The reprojected seed is still only a hypothesis: the estimator
    judges it on matcher inliers like any other."""
    if not _HAS_RASTERIO:
        result["reason"] = "cannot reproject: frames differ but rasterio is unavailable"
        return None
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform

    g_side = "source" if src_frame == "geographic" else "reference"
    p_side = "reference" if g_side == "source" else "source"
    g_A, g_px, g_xy, g_img = mappings[g_side]
    p_A, _, _, p_img = mappings[p_side]

    if p_img.crs is None:
        result["reason"] = (
            f"cannot reproject: frames differ (source {src_frame} vs "
            f"reference {ref_frame}) and the projected side ({p_side} "
            f"'{p_img.path}') declares no CRS"
        )
        return None

    if g_img.crs is not None:
        g_crs = g_img.crs
    else:
        g_crs = CRS.from_string(ASSUMED_MOON_LONGLAT_CRS)
        logger.warning(
            "tiepoint seed: %s '%s' has geographic coordinates but no "
            "declared CRS; assuming %s",
            g_side, g_img.path, ASSUMED_MOON_LONGLAT_CRS,
        )

    try:
        xs_w, ys_w = warp_transform(
            g_crs, p_img.crs, [p[0] for p in g_xy], [p[1] for p in g_xy])
    except Exception as exc:  # PROJ/network/CRS errors -> honest refusal
        result["reason"] = (
            f"cannot reproject {g_side} into the projected frame "
            f"({type(exc).__name__}: {exc})"
        )
        return None

    # Refit the geographic side's pixels against its reprojected frame
    # coordinates, then compose in the projected frame (units cancel there).
    uv1 = np.array([[u, v, 1.0] for u, v in g_px], dtype=np.float64)
    fit, *_ = np.linalg.lstsq(uv1, np.column_stack([xs_w, ys_w]), rcond=None)
    A_g_in_p = np.array([
        [fit[0, 0], fit[1, 0], fit[2, 0]],
        [fit[0, 1], fit[1, 1], fit[2, 1]],
        [0.0, 0.0, 1.0],
    ])
    A_src_p, A_ref_p = ((A_g_in_p, p_A) if g_side == "source"
                        else (p_A, A_g_in_p))
    try:
        return np.linalg.inv(A_ref_p) @ A_src_p
    except np.linalg.LinAlgError:
        result["reason"] = "singular pixel->map fit after reprojection"
        return None


def derive_tiepoint_coarse_transform(
    src: "LoadedImage",
    ref: "LoadedImage",
    src_offset: Tuple[int, int] = (0, 0),
    ref_offset: Tuple[int, int] = (0, 0),
) -> Dict[str, Any]:
    """Coarse source-pixel -> reference-pixel affine from each raster's own
    georeferencing (corner tiepoints, or geotransform) — metadata only.

    WHY: the homography estimator only runs once a matcher produced >= 4
    correspondences, and its RANSAC fit starts from nothing — on weak-texture
    pairs that fit comes out unsupported ("no_transform") even though both
    labels pin the raster corners to map coordinates. The composition
    gives the coarse transform for free; the estimator judges it on
    evidence like any other hypothesis (viewpoint._evaluate_seed_hypothesis),
    so bad metadata loses instead of being believed.

    FRAMES: same numeric frame on both sides (declared CRS, else
    coordinate magnitudes) -> direct composition; frames differ -> the
    geographic side is reprojected into the projected side's CRS first
    (_reproject_and_compose), refused with a reason when that is
    impossible. `src_offset`/`ref_offset` are the (x, y) pixel origin of
    the crop applied to each load (window or auto-tile; (0, 0) = full
    frame) — full-raster tiepoints/transforms are SHIFTED by this known
    origin, exact arithmetic, so windowed and tiled runs still derive.

    Returns {"H": (3, 3) ndarray | None, "rotation_deg": float | None,
    "scale": float | None, "reason": str | None}. On refusal H is None and
    `reason` names the exact cause so the placeholder downstream stays
    traceable, never silent.
    """
    result: Dict[str, Any] = {
        "H": None, "rotation_deg": None, "scale": None, "reason": None,
    }

    mappings: Dict[str, tuple] = {}
    for side, image, offset in (
        ("source", src, src_offset),
        ("reference", ref, ref_offset),
    ):
        A, sample_px, sample_xy, status = _mapping_from_image(image, offset)
        if status == "missing":
            result["reason"] = (
                f"missing or insufficient corner tiepoints and no usable "
                f"geotransform on {side} ('{image.path}', got "
                f"{len(image.corner_tiepoints or [])} tiepoints)"
            )
            return result
        if status == "degenerate":
            result["reason"] = (
                f"degenerate (collinear) tiepoints on {side} ('{image.path}')"
            )
            return result
        mappings[side] = (A, sample_px, sample_xy, image)

    src_A, _, src_xy, src_img = mappings["source"]
    ref_A, _, ref_xy, ref_img = mappings["reference"]
    src_frame = _frame_class(src_xy, src_img.crs)
    ref_frame = _frame_class(ref_xy, ref_img.crs)

    if src_frame == ref_frame:
        # Same numeric frame: units cancel in inv(A_ref) @ A_src.
        try:
            H = np.linalg.inv(ref_A) @ src_A
        except np.linalg.LinAlgError:  # rank-checked above; defensive anyway
            result["reason"] = (
                f"singular pixel->map fit on reference ('{ref.path}')"
            )
            return result
    else:
        H = _reproject_and_compose(mappings, src_frame, ref_frame, result)
        if H is None:
            return result

    result["H"] = H
    result["scale"] = float(np.sqrt(abs(float(np.linalg.det(H[:2, :2])))))
    # Orientation of the source x-axis in reference pixels (0 for two
    # north-up rasters sharing a grid — a diagnostic, not a full decomposition.
    result["rotation_deg"] = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
    return result


# ----------------------------------------------------------------------
# GSD-based scale prior (Stage 1.5)
# ----------------------------------------------------------------------

def estimate_gsd_scale_prior(
    src: "LoadedImage",
    ref: "LoadedImage",
) -> Optional[float]:
    """Returns a prior estimate of src-vs-ref scale ratio from ground
    sampling distance metadata, so scale.select_best_scale's coarse-to-fine
    search (pwift.coarse_to_fine_rotation_scale) can be given a tight band
    around the physically-expected ratio instead of blindly searching the
    sensor's full scale_range (e.g. OHRC's 0.5x-3x) from scratch.

    Priority: each image's own measured `gsd_m` (from its label/metadata,
    when `load_image` was able to populate it) over the sensor's
    `approx_gsd_m` placeholder from config.py. If NEITHER image has a
    usable GSD, returns None - callers should fall back to the existing
    full-range search unchanged.

    Ratio convention matches scale.apply_scale / pwift's scale candidates:
    a return value of 2.0 means the source image needs to be scaled *up*
    2x to match the reference's pixel scale (i.e. src has 2x coarser GSD
    than ref).
    """
    src_gsd = src.gsd_m or getattr(src.sensor, "approx_gsd_m", None)
    ref_gsd = ref.gsd_m or getattr(ref.sensor, "approx_gsd_m", None)

    if not src_gsd or not ref_gsd or src_gsd <= 0 or ref_gsd <= 0:
        warnings.warn(
            "estimate_gsd_scale_prior: no usable GSD for source and/or "
            "reference (checked LoadedImage.gsd_m, then "
            "SensorConfig.approx_gsd_m) - falling back to an unconstrained "
            "scale search over the sensor's full scale_range."
        )
        return None

    return float(src_gsd / ref_gsd
    )