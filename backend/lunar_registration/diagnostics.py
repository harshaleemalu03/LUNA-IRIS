"""Phase-A alignment diagnostics: does the metadata pose land the source on the reference's ground?

WHY: every downstream stage (matching, verification, the gate) only has meaning
once the metadata homography is trusted or refuted. These diagnostics render the
pose for human judgement and score it two ways: a structural NCC *at* the pose,
and a global low-frequency search that reports where the source's kilometre-scale
structure actually lives inside the reference versus where the metadata claims it
lives. On a trustworthy pose both agree; when they disagree the georeferencing is
wrong and no matcher can rescue the run.

Frames: `H` maps source pixels to reference pixels, exactly like the homography
the pipeline reports in `summary.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .preprocessing import derive_tiepoint_coarse_transform, load_image
from .verify import compute_structural_ncc

# Search cell size in px: the global search reasons about kilometre-scale
# structure only (albedo, crater density), which is what survives a sun-angle
# change - fine texture does not.
_LOW_FREQ_DOWNSAMPLE = 4
_LOW_FREQ_BLUR_SIGMA = 2.0
# Agreement tolerance between the metadata placement and the search peak,
# in full-resolution pixels (quantisation = downsample, plus resampling slop).
_PEAK_TOLERANCE_PX = 16.0
# Minimum low-frequency NCC for a peak to count as corroboration: blurred
# kilometre-scale self-similarity on the real SIH pairs tops out around 0.33,
# genuine cross-image agreement starts near 0.6, so 0.5 splits them cleanly.
_MIN_PEAK_NCC = 0.5
# Below this warped-source coverage the pose shows no ground at all.
_MIN_CONTENT_FRACTION = 0.02
_CHECKER_BLOCK_PX = 32
_MAX_DIM_DEFAULT = 2400


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Render-ready uint8 copy: TIFF uint8 passes through, pipeline floats in
    [0, 1] scale up. Inputs are never modified."""
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_alignment_overlay(
    source: np.ndarray,
    reference: np.ndarray,
    H: Optional[np.ndarray],
    out_path: Path,
    max_dim: int = _MAX_DIM_DEFAULT,
) -> Path:
    """Write a two-panel PNG: source warped by `H` blended over the reference, then a checkerboard.

    WHY: numbers cannot answer "is this the same ground"; a human reads the
    blend (do features coincide?) and the checkerboard (do craters continue
    across the seam?) in seconds. `H` of `None` writes the reference alone so
    a missing-metadata pair still leaves visual evidence in the report.
    """
    ref_u8 = _to_uint8(reference)
    ref_h, ref_w = ref_u8.shape[:2]

    if H is None:
        warped = np.zeros_like(ref_u8)
    else:
        warped = cv2.warpPerspective(_to_uint8(source), np.asarray(H, np.float64), (ref_w, ref_h))

    blend = cv2.addWeighted(warped, 0.5, ref_u8, 0.5, 0.0)
    yy, xx = np.mgrid[0:ref_h, 0:ref_w]
    checker_mask = ((yy // _CHECKER_BLOCK_PX + xx // _CHECKER_BLOCK_PX) % 2).astype(bool)
    checker = np.where(checker_mask, warped, ref_u8)

    panel = np.hstack([blend, checker])
    # `max_dim` bounds ONE panel (the composed pair is twice as wide).
    scale = max_dim / max(ref_h, ref_w)
    if scale < 1.0:
        panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), panel):
        raise RuntimeError(f"failed to write overlay PNG: {out_path}")
    return out_path


def alignment_report(
    source: np.ndarray,
    reference: np.ndarray,
    H: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Score the metadata pose: structural NCC at the pose plus a global low-frequency search.

    WHY these two numbers together: structural NCC at the pose says whether the
    two images share structure *where the metadata says they should*, while the
    global search says where the source's coarse structure actually peaks
    anywhere in the reference. `peak_matches_metadata` is true only when a
    peak at or above `_MIN_PEAK_NCC` sits within `_PEAK_TOLERANCE_PX` of the
    metadata placement: an argmax always exists, but on self-similar terrain
    it is noise, so the flag means "placement corroborated", not "argmax
    found".
    """
    ref_h, ref_w = reference.shape[:2]
    result: Dict[str, Any] = {
        "structural_ncc_at_pose": 0.0,
        "content_coverage": 0.0,
        "lowfreq_best_ncc": 0.0,
        "lowfreq_noise_floor": 0.0,
        "metadata_shift_px": None,
        "lowfreq_best_shift_px": None,
        "peak_matches_metadata": False,
    }
    if H is None:
        return result

    H = np.asarray(H, np.float64)
    mask = cv2.warpPerspective(
        np.full(source.shape[:2], 255, np.uint8), H, (ref_w, ref_h)
    )
    coverage = float((mask > 0).mean())
    result["content_coverage"] = coverage

    result["structural_ncc_at_pose"] = float(
        compute_structural_ncc(source, reference, H)
    )
    if coverage < _MIN_CONTENT_FRACTION:
        return result

    # Global low-frequency search: where does the warped source's content
    # sit? Masked NCC via three TM_CCORR passes: a rotated footprint's
    # bounding box contains warp padding the reference does not share, so
    # only coverage-weighted pixels may enter the correlation.
    k = _LOW_FREQ_DOWNSAMPLE
    size = (max(1, ref_w // k), max(1, ref_h // k))
    ref_s = cv2.GaussianBlur(
        cv2.resize(_to_uint8(reference), size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0,
        (0, 0), _LOW_FREQ_BLUR_SIGMA,
    )
    warped_s = cv2.GaussianBlur(
        cv2.resize(
            cv2.warpPerspective(_to_uint8(source), H, (ref_w, ref_h)),
            size, interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0,
        (0, 0), _LOW_FREQ_BLUR_SIGMA,
    )
    weights = cv2.resize(
        mask.astype(np.float32) / 255.0, size, interpolation=cv2.INTER_AREA
    )

    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()) // k, int(xs.max()) // k + 1
    y0, y1 = int(ys.min()) // k, int(ys.max()) // k + 1
    tpl, w = warped_s[y0:y1, x0:x1], weights[y0:y1, x0:x1]
    if tpl.shape[0] > ref_s.shape[0] or tpl.shape[1] > ref_s.shape[1]:
        return result

    # Weighted Pearson r collapses to cross-correlations: with mu_t = sum(wt)/sum(w),
    # r = (A - mu_t B) / sqrt(var_t * (C - B^2/sum(w))) where A, B, C are the
    # TM_CCORR of (tpl*w, ref), (w, ref) and (w, ref^2) respectively.
    m_sum = float(w.sum())
    t_sum = float((tpl * w).sum())
    var_t = float((tpl * tpl * w).sum()) - t_sum * t_sum / m_sum
    a = cv2.matchTemplate(ref_s, tpl * w, cv2.TM_CCORR)
    b = cv2.matchTemplate(ref_s, w, cv2.TM_CCORR)
    c = cv2.matchTemplate(ref_s * ref_s, w, cv2.TM_CCORR)
    num = a - (t_sum / m_sum) * b
    den = np.sqrt(np.maximum(var_t, 1e-12) * np.maximum(c - b * b / m_sum, 1e-12))
    response = num / den

    _, best_ncc, _, best_loc = cv2.minMaxLoc(response)
    noise_floor = float(np.percentile(response, 99.9))

    metadata_shift = (float(x0 * k), float(y0 * k))
    best_shift = (float(best_loc[0] * k), float(best_loc[1] * k))
    offset = float(np.hypot(best_shift[0] - metadata_shift[0], best_shift[1] - metadata_shift[1]))

    result.update({
        "lowfreq_best_ncc": float(best_ncc),
        "lowfreq_noise_floor": noise_floor,
        "metadata_shift_px": list(metadata_shift),
        "lowfreq_best_shift_px": list(best_shift),
        "peak_matches_metadata": bool(best_ncc >= _MIN_PEAK_NCC and offset <= _PEAK_TOLERANCE_PX),
    })
    return result


def discover_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Every `*source*` raster under `data_dir` paired with its `*reference*`
    sibling (a source without a reference is skipped). Sidecar XMLs resolve
    per image at load time (`resolve_sidecar_xml`), so they are not listed."""
    pairs: List[Tuple[Path, Path]] = []
    for src in sorted(Path(data_dir).rglob("*source*.tif")):
        ref = src.parent / src.name.replace("_source", "_reference")
        if ref.exists():
            pairs.append((src, ref))
    return pairs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: render an overlay and score the metadata pose for every pair in a directory."""
    parser = argparse.ArgumentParser(
        prog="lunar_registration.diagnostics",
        description="Render and score the metadata alignment pose for source/reference pairs.",
    )
    parser.add_argument("--data-dir", required=True, help="Directory searched recursively for pairs")
    parser.add_argument("--out", required=True, help="Output directory for overlays and report.json")
    parser.add_argument("--max-dim", type=int, default=_MAX_DIM_DEFAULT,
                        help="Longest edge of each rendered panel (default 2400)")
    args = parser.parse_args(argv)

    data_dir, out_dir = Path(args.data_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(data_dir)
    if not pairs:
        print(f"no source/reference pairs found under {data_dir}", flush=True)
        return 1

    entries: List[Dict[str, Any]] = []
    for src_path, ref_path in pairs:
        src = load_image(str(src_path))
        ref = load_image(str(ref_path))
        coarse = derive_tiepoint_coarse_transform(src, ref, src_offset=(0, 0), ref_offset=(0, 0))
        H = coarse.get("H")

        # Rendered even when the pose is missing (reference-only panels): a
        # failed pair must still leave visual evidence in the report.
        overlay = render_alignment_overlay(
            src.data, ref.data, H,
            out_dir / f"{src_path.name.split('_source')[0]}_overlay.png",
            max_dim=args.max_dim,
        )
        rep = alignment_report(src.data, ref.data, H)
        entries.append({
            "source": str(src_path),
            "reference": str(ref_path),
            "xml": src.sidecar_path,
            "rotation_deg": None if coarse.get("rotation_deg") is None else float(coarse["rotation_deg"]),
            "scale": None if coarse.get("scale") is None else float(coarse["scale"]),
            "reason": coarse.get("reason"),
            "overlay": str(overlay),
            "report": rep,
        })
        if H is None:
            print(f"{src_path.name}: metadata pose unavailable ({coarse.get('reason')})", flush=True)
        else:
            print(
                f"{src_path.name}: ncc@pose={rep['structural_ncc_at_pose']:+.3f} "
                f"lowfreq={rep['lowfreq_best_ncc']:.3f} (floor {rep['lowfreq_noise_floor']:.3f}) "
                f"peak@metadata={rep['peak_matches_metadata']}",
                flush=True,
            )

    (out_dir / "report.json").write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} entries -> {out_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
