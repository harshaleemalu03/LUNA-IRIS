"""
End-to-end orchestrator. Run as:

    python -m lunar_registration.pipeline \\
        --source path/to/ohrc_image.img --source-sensor OHRC \\
        --reference path/to/lroc_nac.cub \\
        --out-dir outputs/run1 \\
        [--source-nac-pho path/to/NAC_PHO_..._source.cub] \\
        [--reference-nac-pho path/to/NAC_PHO_..._reference.cub] \\
        [--angles-from-label | --source-incidence inc.tif --source-emission emi.tif] \\
        [--source-sidecar-xml path/to/PROD.xml] \\
        [--window 2243,298,512,512] \\
        [--no-eloftr]

`--source-nac-pho`/`--reference-nac-pho` read real pixel-wise incidence/
emission/phase angle maps straight out of an LROC NAC_PHO photometry cube's
angle bands (Band 2 = Phase, Band 3 = Local Emission, Band 4 = Local
Incidence) - see preprocessing.py's `load_angles_from_nac_pho`. This is the
preferred angle source whenever you have the NAC_PHO product for an image,
since PWIFT's photometric weighting (paper Sec 3.2) is defined per-pixel;
it takes priority over all the scalar shortcuts below. Both the source and
the reference get their own independent photometric weighting when
supplied - the paper applies this to both images in a pair, not just one.

`--angles-from-label` reads incidence/emission/phase straight from the
source image's PDS3 label (fast, no ISIS) instead of requiring phocube
angle-map .tif files - see preprocessing.py's `load_angles_from_label` for
what it does and when it falls back. Explicit `--source-incidence`/
`--source-emission` paths, if given, always take priority over the label
shortcut. Neither applies to the reference image, which only supports
`--reference-nac-pho` for now.

`--source-sidecar-xml` names the source product's sidecar XML explicitly
(the same file an API client uploads as `source_xml`). It overrides
sibling auto-discovery — use it when the XML does not sit next to the
image — and fails loudly (missing, unparseable, or none of the recognized
angle tags) instead of silently routing at the placeholder. Omit it to
keep filesystem auto-discovery. See preprocessing.py's
`resolve_sidecar_xml`/`attach_file_metadata`.

See README.md for full setup (dependencies, ISIS pre-processing needed for
angle maps, etc).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Optimize PyTorch CUDA allocator to prevent OOM fragmentation on constrained GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import cv2

from .config import PipelineConfig, get_sensor_config
from .preprocessing import (
    load_image, load_angle_maps, LoadedImage, estimate_gsd_scale_prior,
    derive_tiepoint_coarse_transform,
)
from .illumination import apply_illumination_correction
from .scale import select_best_scale, apply_scale, apply_rotation
from .matching import get_matcher, run_pwift_matching, MatchResult, BaseMatcher
from .fusion import fuse_pwift_neural, miho_plus_gcp, assess_pwift_quality
from .verify import compute_structural_ncc, compete_rigid, gate_cheap
from .refine import refine_tile, choose_refiner, compute_texture_energy
from .viewpoint import estimate_viewpoint_transform, HomographyResult
from .georeference import register_image, write_outputs
from .metrics import compute_metrics
from .photometric import normalize_for_matching_cfg
from .pwift import reprojection_cleanup
from .structural import structural_ncc_correspondences


logger = logging.getLogger(__name__)

# Largest image the illumination/matching stages can safely process in memory
# on the CPU-only demo box (~2000x2000). Module-level so tests can shrink it.
MAX_SAFE_PIXELS = 4_000_000

# Bare neural matcher names (never PWIFT). Task 6: these arms receive
# photometrically normalized inputs; Task 7: they always take the fusion
# path (PWIFT runs alongside them), while "pwift" stays single-arm.
NEURAL_MATCHERS: Tuple[str, ...] = ("roma2", "eloftr")


def _parse_window(s: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not s:
        return None
    x, y, w, h = (int(v) for v in s.split(","))
    return x, y, w, h


def _center_tile_window(height: int, width: int, max_pixels: int) -> Tuple[int, int, int, int]:
    """Largest centered crop not exceeding max_pixels, native resolution kept.

    WHY: an explicit crop window is the caller's choice; its absence is not —
    the fallback must preserve GSD (no rescaling) so the scale prior stays valid.
    """
    if height * width <= max_pixels:
        return 0, 0, width, height
    scale = math.sqrt(max_pixels / (height * width))
    nh = max(1, int(round(height * scale)))
    nw = max(1, int(round(width * scale)))
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    return x0, y0, nw, nh


def determine_adaptive_matcher(
    requested_matcher: Optional[str],
    incidence_deg: Optional[float],
    real_time: bool,
    cfg: PipelineConfig,
) -> Tuple[str, Dict[str, Any]]:
    """Determines the matcher and operational regime based on physical & operational conditions.

    Returns:
        (resolved_matcher_name, routing_metadata)
    """
    clean = (requested_matcher or "auto").strip().lower()

    if clean != "auto":
        return clean, {
            "mode": "manual_override",
            "regime": "user_specified",
            "incidence_deg": incidence_deg,
            "real_time_requested": real_time,
            "resolved_matcher": clean,
            "reason": f"Explicit user override: {clean}",
        }

    inc = incidence_deg if incidence_deg is not None else 30.0

    # Condition 1: Extreme Polar / Grazing Sun (i >= 70°)
    if inc >= cfg.polar_incidence_threshold_deg:
        chosen = "hybrid_pwift_eloftr" if real_time else "hybrid_pwift_roma2"
        return chosen, {
            "mode": "adaptive",
            "regime": "polar_grazing",
            "incidence_deg": inc,
            "real_time_requested": real_time,
            "resolved_matcher": chosen,
            "reason": (
                f"Incidence {inc:.1f}° >= {cfg.polar_incidence_threshold_deg:.1f}°: "
                "PWIFT harmonic Akimov masking suppresses migrating shadow boundaries."
            ),
        }

    # Condition 2: Real-Time Operational Constraint (Descent TRN)
    if real_time or cfg.real_time_mode:
        return "eloftr", {
            "mode": "adaptive",
            "regime": "real_time_trn",
            "incidence_deg": inc,
            "real_time_requested": True,
            "resolved_matcher": "eloftr",
            "reason": (
                "Real-time descent navigation constraint requested: "
                "EfficientLoFTR selected for ~88 ms latency and high inlier density."
            ),
        }

    # Condition 3: Sub-Pixel Surface Cartography (Nominal)
    return "roma2", {
        "mode": "adaptive",
        "regime": "subpixel_cartography",
        "incidence_deg": inc,
        "real_time_requested": False,
        "resolved_matcher": "roma2",
        "reason": (
            "Nominal orbital mapping: RoMa v2 selected for sub-pixel precision "
            "(79.9% MMA@1px, 0.10 px corner error)."
        ),
    }


def note_winning_arm_support(
    contingency: Dict[str, Any],
    winner_method: str,
    winner_n_inliers: int,
) -> Dict[str, Any]:
    """Task 11: keep contingency_fallback truthful about the arm that WON.

    WHY: the report saw `triggered: False` while the winning arm carried 0
    geometrically valid inliers — every arm failed to produce verifiable
    support, which is a contingency state even when no alternate arm
    remained to run (fallback_matcher stays None: detected, nothing left to
    try). If the matcher stage already recorded a contingency, the earlier
    reason is kept and this fact is appended — the summary must carry both.
    Mutates and returns `contingency`."""
    if winner_n_inliers > 0:
        return contingency
    fact = (
        f"winning arm '{winner_method}' produced 0 geometrically valid "
        "inliers — matcher output has no verifiable support"
    )
    if contingency.get("triggered"):
        prior = contingency.get("reason")
        contingency["reason"] = f"{prior}; {fact}" if prior else fact
    else:
        contingency["triggered"] = True
        contingency["fallback_matcher"] = None
        contingency["reason"] = fact
    return contingency


def _matcher_inputs(
    src: np.ndarray, ref: np.ndarray, matcher_name: str, cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stage-3 input arrays for `matcher_name`.

    WHY: deep matchers score correspondences from local appearance and
    degrade under the exposure / sun-angle differences between the two
    acquisitions, so `photometric.normalize_for_matching_cfg` strips the
    illumination component before the network sees the images (Task 6).
    PWIFT is the exception — it consumes the Stage-2 illumination maps
    built from the RAW imagery, so its inputs (and any other non-neural
    arm) pass through untouched. Mode "none" short-circuits explicitly,
    handing back the ORIGINAL arrays — identity included — instead of
    normalized copies, so opting out provably changes nothing.
    """
    if matcher_name not in NEURAL_MATCHERS or cfg.neural_input_normalization == "none":
        return src, ref
    return normalize_for_matching_cfg(src, cfg), normalize_for_matching_cfg(ref, cfg)


def _estimate_arm(
    match_result: Optional[MatchResult],
    sensor,
    image_shape: Tuple[int, int],
    cfg: PipelineConfig,
    seed_H: Optional[np.ndarray],
    src_img: np.ndarray,
    ref_img: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Stage-4 estimation for ONE `MatchResult` -> competition candidate.

    WHY (Task 8): the structural-NCC last-resort arm must travel EXACTLY
    the path a matcher arm travels — same viewpoint/RANSAC estimation with
    the tiepoint seed, same Eq. 26-27 reprojection cleanup, same metrics
    and competition fields — so the body of the per-arm evaluation loop
    lives here once and both the matcher loop and the fallback call it.
    Returns None for a result too small to estimate from (the loop's
    `continue`), else the candidate entry; the entry carries
    `tiepoint_seed_supported`, the flag run_pipeline folds into
    summary.tiepoint_coarse.seed_supported.
    """
    if match_result is None or len(match_result.pts_src) < 4:
        return None

    # ---- Stage 4: viewpoint (homography + RANSAC) ----
    hom_result = estimate_viewpoint_transform(
        match_result.pts_src, match_result.pts_dst, sensor,
        image_shape=image_shape, cfg=cfg,
        seed_H=seed_H,
    )
    H_or_local_results = hom_result if isinstance(hom_result, list) else hom_result.H
    _hypotheses = hom_result if isinstance(hom_result, list) else [hom_result]
    tiepoint_seed_supported = any(
        getattr(r, "provenance", "fitted") == "tiepoint_seed" for r in _hypotheses)
    inlier_mask = hom_result.inlier_mask if not isinstance(hom_result, list) else np.zeros(
        len(match_result.pts_src), dtype=bool)
    if isinstance(hom_result, list):
        for r in hom_result:
            inlier_mask |= r.inlier_mask

    # ---- Eq 26-27: explicit homography-based reprojection cleanup ----
    tau_e = cfg.reprojection_cleanup_tau_e_px
    if isinstance(hom_result, list):
        clean_mask = np.zeros(len(match_result.pts_src), dtype=bool)
        for r in hom_result:
            if r.H is None or not np.any(r.inlier_mask):
                continue
            idx = np.nonzero(r.inlier_mask)[0]
            clean = reprojection_cleanup(match_result.pts_src[idx], match_result.pts_dst[idx], r.H, tau_e)
            clean_mask[idx[clean]] = True
        inlier_mask = inlier_mask & clean_mask
    elif hom_result.H is not None:
        clean = reprojection_cleanup(match_result.pts_src, match_result.pts_dst, hom_result.H, tau_e)
        inlier_mask = inlier_mask & clean

    primary_H = hom_result.H if not isinstance(hom_result, list) else (hom_result[0].H if hom_result else None)

    metrics = compute_metrics(
        match_result.method, match_result.pts_src, match_result.pts_dst,
        inlier_mask, H_or_local_results, image_shape=image_shape, grid=cfg.uniformity_grid,
    )

    # Structural fit and non-rigid DOF residual for rigid competition
    s_ncc = compute_structural_ncc(src_img, ref_img, primary_H) if primary_H is not None else 0.0
    coverage = float(metrics.uniformity_score)
    dof_resid = 0.0
    if primary_H is not None and len(match_result.pts_src) > 0:
        proj = cv2.perspectiveTransform(
            match_result.pts_src.reshape(-1, 1, 2).astype(np.float32), primary_H.astype(np.float32)
        ).reshape(-1, 2)
        dof_resid = float(np.median(np.abs(proj - match_result.pts_dst)))

    return {
        "method": match_result.method,
        "match_result": match_result,
        "hom_result": hom_result,
        "inlier_mask": inlier_mask,
        "metrics": metrics,
        "warp": primary_H if primary_H is not None else np.eye(3),
        "fit": s_ncc,
        "coverage": coverage,
        "dof_resid": dof_resid,
        "tiepoint_seed_supported": tiepoint_seed_supported,
    }


def _candidate_has_transform(candidate: Dict[str, Any]) -> bool:
    """Does this candidate carry the homography the fail-closed verdict judges?

    WHY (Task 8): the structural fallback may fire only when NO arm
    produced a transform, and "transform" must mean exactly what the
    verdict below means by it — its primary-H selection over (possibly
    list-valued) hypotheses — so the trigger can never disagree with what
    the gate would evaluate.
    """
    hom = candidate["hom_result"]
    if isinstance(hom, list):
        return bool(hom) and hom[0].H is not None
    return hom.H is not None


def run_pipeline(
    source_path: str, reference_path: str, out_dir: str,
    source_sensor: Optional[str] = None,
    source_incidence_path: Optional[str] = None,
    source_emission_path: Optional[str] = None,
    source_phase_path: Optional[str] = None,
    source_nac_pho_path: Optional[str] = None,
    reference_nac_pho_path: Optional[str] = None,
    nac_pho_band_phase: int = 2,
    nac_pho_band_emission: int = 3,
    nac_pho_band_incidence: int = 4,
    angles_from_label: bool = False,
    fetch_angles_online: bool = False,
    manual_incidence_deg: Optional[float] = None,
    manual_emission_deg: Optional[float] = None,
    manual_phase_deg: Optional[float] = None,
    source_sidecar_xml: Optional[str] = None,
    window: Optional[Tuple[int, int, int, int]] = None,
    source_window: Optional[Tuple[int, int, int, int]] = None,
    reference_window: Optional[Tuple[int, int, int, int]] = None,
    reference_sensor: Optional[str] = None,
    matcher: Optional[str] = None,
    roma2_weights: Optional[str] = None,
    eloftr_checkpoint: Optional[str] = None,
    eloftr_ckpt: Optional[str] = None,
    device: Optional[str] = None,
    use_eloftr: bool = True,
    fuse_pwift_eloftr: bool = False,
    real_time: bool = False,
    enable_orthogonal_gate: Optional[bool] = None,
    polar_incidence_threshold_deg: Optional[float] = None,
    export_gcl_gcps_csv: Optional[bool] = None,
    cfg: Optional[PipelineConfig] = None,
) -> dict:
    cfg = cfg or PipelineConfig()

    # Own the output directory from the start: failure runs never reach
    # write_outputs, but summary.json must still land somewhere.
    os.makedirs(out_dir, exist_ok=True)

    if eloftr_ckpt:
        eloftr_checkpoint = eloftr_ckpt
    if roma2_weights:
        cfg.roma2_weights_path = roma2_weights
    if eloftr_checkpoint:
        cfg.eloftr_checkpoint_path = eloftr_checkpoint
    if device:
        cfg.device = device
    if real_time:
        cfg.real_time_mode = True
    if enable_orthogonal_gate is not None:
        cfg.enable_orthogonal_gate = enable_orthogonal_gate
    if polar_incidence_threshold_deg is not None:
        cfg.polar_incidence_threshold_deg = polar_incidence_threshold_deg
    if export_gcl_gcps_csv is not None:
        cfg.export_gcl_gcps_csv = export_gcl_gcps_csv

    if matcher is None:
        if fuse_pwift_eloftr:
            matcher = "hybrid_pwift_eloftr"
        elif not use_eloftr:
            matcher = "pwift"
        else:
            matcher = getattr(cfg, "matcher_type", "auto")

    if source_window is None:
        source_window = window
    if reference_window is None:
        reference_window = window

    # ---- Stage 1: preprocessing ----
    src: LoadedImage = load_image(
        source_path, sensor_hint=source_sensor, window=source_window,
        angles_from_label=angles_from_label, fetch_angles_online=fetch_angles_online,
        manual_incidence_deg=manual_incidence_deg, manual_emission_deg=manual_emission_deg,
        manual_phase_deg=manual_phase_deg,
        # Explicit product XML (API upload / --source-sidecar-xml); source
        # load only — the reference keeps sibling auto-discovery.
        sidecar_xml_path=source_sidecar_xml,
        nac_pho_path=source_nac_pho_path,
        nac_pho_band_phase=nac_pho_band_phase, nac_pho_band_emission=nac_pho_band_emission,
        nac_pho_band_incidence=nac_pho_band_incidence,
    )
    ref: LoadedImage = load_image(
        reference_path, sensor_hint=reference_sensor, window=reference_window,
        nac_pho_path=reference_nac_pho_path,
        nac_pho_band_phase=nac_pho_band_phase, nac_pho_band_emission=nac_pho_band_emission,
        nac_pho_band_incidence=nac_pho_band_incidence,
    )

    # Size guard: an image with no crop window is auto-center-tiled to
    # MAX_SAFE_PIXELS with a loud warning; an image with an explicit window
    # (shared or per-image) is the caller's choice and proceeds as given.
    # Pixel origin of the crop applied to each load (window or auto-tile);
    # (0, 0) = full frame. Tiepoints stay full-raster coordinates, so the
    # Task-12 seed derivation shifts them by exactly this origin.
    _crop_offsets = {"source": (0, 0), "reference": (0, 0)}
    for _tag, _loaded, _img_window in (
        ("source", src, source_window),
        ("reference", ref, reference_window),
    ):
        if _img_window is not None:
            _crop_offsets[_tag] = (_img_window[0], _img_window[1])
        if _img_window is None and _loaded.data.size > MAX_SAFE_PIXELS:
            _h, _w = _loaded.data.shape[:2]
            _x0, _y0, _nw, _nh = _center_tile_window(_h, _w, MAX_SAFE_PIXELS)
            _crop_offsets[_tag] = (_x0, _y0)
            logger.warning(
                "auto-tile: %s image '%s' is %sx%s (%s px), above the %s px "
                "safety limit and no crop window was given; center-tiling to "
                "window %s,%s,%s,%s. Pass --source-window x,y,w,h and "
                "--reference-window x,y,w,h (API: source_window/reference_window) "
                "to target the geographic overlap region instead.",
                _tag, _loaded.path, _w, _h, f"{_loaded.data.size:,}",
                f"{MAX_SAFE_PIXELS:,}", _x0, _y0, _nw, _nh,
            )
            _rows = np.s_[_y0:_y0 + _nh]
            _cols = np.s_[_x0:_x0 + _nw]
            _loaded.data = _loaded.data[_rows, _cols]
            for _arr_name in ("incidence_deg", "emission_deg", "phase_deg"):
                _arr = getattr(_loaded, _arr_name, None)
                if isinstance(_arr, np.ndarray) and _arr.shape[:2] == (_h, _w):
                    setattr(_loaded, _arr_name, _arr[_rows, _cols])

    # Task 12: metadata-derived coarse transform, wired as a candidate
    # hypothesis for the homography estimator. Crop origins are exact
    # (window=(x, y, w, h) / auto-tile slice), so full-raster tiepoints are
    # SHIFTED rather than withheld — every demo run is windowed or tiled.
    tiepoint_report = derive_tiepoint_coarse_transform(
        src, ref,
        src_offset=_crop_offsets["source"],
        ref_offset=_crop_offsets["reference"],
    )

    src_incidence, src_emission, src_phase = src.incidence_deg, src.emission_deg, src.phase_deg
    if source_incidence_path and source_emission_path:
        src_incidence, src_emission, src_phase = load_angle_maps(
            source_incidence_path, source_emission_path,
            source_phase_path or source_emission_path, window=source_window,
        )
    ref_incidence, ref_emission, ref_phase = ref.incidence_deg, ref.emission_deg, ref.phase_deg

    # Resolve adaptive matcher based on physical and operational conditions
    inc_for_routing = None
    if manual_incidence_deg is not None:
        inc_for_routing = float(manual_incidence_deg)
    elif src_incidence is not None:
        inc_for_routing = float(np.nanmean(src_incidence))
    elif src.incidence_deg is not None:
        inc_for_routing = float(np.nanmean(src.incidence_deg))
    elif src.sidecar_incidence_deg is not None:
        # Scalar from the sidecar XML (e.g. Solar_incidence_angle 84.896724):
        # explicit angles metadata beats any placeholder default.
        inc_for_routing = float(src.sidecar_incidence_deg)

    resolved_matcher, routing_info = determine_adaptive_matcher(
        requested_matcher=matcher,
        incidence_deg=inc_for_routing,
        real_time=real_time or cfg.real_time_mode,
        cfg=cfg,
    )

    # ---- Stage 1.5: GSD-aware scale prior, then coarse-to-fine search ----
    gsd_scale_prior = estimate_gsd_scale_prior(src, ref)
    scale_rot = select_best_scale(
        src.data, ref.data, src.sensor, cfg, prior_scale=gsd_scale_prior,
    )
    scale_search_failure = None
    if scale_rot is None:
        # Task 6: no confident alignment in the coarse search (argmax on a
        # candidate boundary or flat similarity surface). The run fails
        # closed below; continuing with an identity placeholder only so
        # summary.json still records matcher/gate diagnostics for the
        # failure report.
        scale_search_failure = (
            "no_confident_alignment: scale/rotation search returned no "
            "interior peak (argmax on candidate boundary or flat similarity surface)"
        )
        warnings.warn(
            f"{scale_search_failure} - continuing for diagnostics; "
            "the run fails closed")
        best_scale, best_rot = 1.0, 0.0
    else:
        best_scale, best_rot = scale_rot
    src_scaled = apply_scale(src.data, best_scale)
    src_scaled = apply_rotation(src_scaled, best_rot)
    src_incidence_scaled = apply_scale(src_incidence, best_scale) if src_incidence is not None else None
    src_incidence_scaled = apply_rotation(src_incidence_scaled, best_rot) if src_incidence_scaled is not None else None
    src_emission_scaled = apply_scale(src_emission, best_scale) if src_emission is not None else None
    src_emission_scaled = apply_rotation(src_emission_scaled, best_rot) if src_emission_scaled is not None else None
    src_phase_scaled = apply_scale(src_phase, best_scale) if src_phase is not None else None
    src_phase_scaled = apply_rotation(src_phase_scaled, best_rot) if src_phase_scaled is not None else None

    # ---- Stage 2: illumination correction (per-sensor branch) ----
    src_illum = apply_illumination_correction(
        src_scaled, src.sensor, incidence_deg=src_incidence_scaled,
        emission_deg=src_emission_scaled, phase_deg=src_phase_scaled,
        reference=ref.data, n_scales=cfg.pwift_scales, n_orient=cfg.pwift_orientations, cfg=cfg,
    )
    ref_illum = apply_illumination_correction(
        ref.data, ref.sensor, incidence_deg=ref_incidence,
        emission_deg=ref_emission, phase_deg=ref_phase, reference=None,
        n_scales=cfg.pwift_scales, n_orient=cfg.pwift_orientations, cfg=cfg,
    )

    # ---- Stage 3: matching & fusion (Modular, Swappable with Contingency Fallback) ----
    results_to_evaluate: List[MatchResult] = []
    contingency_fallback = {
        "triggered": False,
        "original_matcher": resolved_matcher,
        "fallback_matcher": None,
        "reason": None,
    }

    # Task 7: a bare neural name resolves to the SAME path as its hybrid
    # counterpart — neural arms always run PWIFT alongside and compete in
    # fusion + rigid competition (plan-todos.md architecture decision) —
    # while "pwift" stays single-arm by construction.
    neural_name: Optional[str] = None
    if resolved_matcher.startswith("hybrid_pwift_"):
        neural_name = resolved_matcher.replace("hybrid_pwift_", "")
    elif resolved_matcher in NEURAL_MATCHERS:
        neural_name = resolved_matcher

    if neural_name is not None:
        pw_res = None
        pw_matcher = None
        try:
            pw_matcher = get_matcher("pwift", cfg)
            pw_res = pw_matcher.match(src_scaled, ref.data, src_illum=src_illum, ref_illum=ref_illum)
        except Exception as e:
            # Task 7: a PWIFT failure degrades to the neural arm alone —
            # the mirror image of the neural-failure -> PWIFT fallback
            # below — instead of crashing the whole run with a 500.
            warnings.warn(
                f"PWIFT matcher unavailable ({e}); falling back to neural '{neural_name}' only.")
            contingency_fallback = {
                "triggered": True,
                "original_matcher": resolved_matcher,
                "fallback_matcher": neural_name,
                "reason": str(e),
            }
        finally:
            if pw_matcher is not None:
                del pw_matcher

        neural_res = None
        n_matcher = None
        try:
            n_matcher = get_matcher(neural_name, cfg)
            # Task 6: the neural arm sees photometrically normalized copies;
            # PWIFT above keeps the raw imagery (its Stage-2 illumination
            # maps are built from raw input).
            neural_src, neural_ref = _matcher_inputs(src_scaled, ref.data, neural_name, cfg)
            neural_res = n_matcher.match(neural_src, neural_ref)
        except Exception as e:
            warnings.warn(f"Neural matcher '{neural_name}' unavailable ({e}); falling back to PWIFT only.")
            contingency_fallback = {
                "triggered": True,
                "original_matcher": resolved_matcher,
                "fallback_matcher": "pwift",
                "reason": str(e),
            }
        finally:
            if n_matcher is not None:
                del n_matcher
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

        # Fusion needs BOTH arms; if PWIFT failed, the surviving neural arm
        # alone still enters the competition (contingency recorded above).
        if pw_res is not None and neural_res is not None and len(neural_res.pts_src) >= 4:
            fused_res = fuse_pwift_neural(
                pw_res, neural_res,
                gsd_ref=ref.gsd_m or 1.0,
                alpha=cfg.fusion_alpha,
                beta=cfg.fusion_beta,
                pwift_n_min=cfg.pwift_min_quality_inliers,
                pwift_c_min=cfg.pwift_min_quality_cells,
                pwift_r_min=cfg.pwift_min_quality_ratio,
                pwift_rmse_max_px=cfg.pwift_max_quality_rmse_px,
                src_illum=src_illum,
                ref_illum=ref_illum,
                verify_photometric=cfg.fusion_verify_photometric,
                patch_radius=cfg.fusion_verification_patch_r,
                min_energy_thresh=cfg.fusion_verification_min_energy,
                reject_thresh=cfg.fusion_verification_reject_thresh,
            )
            results_to_evaluate.extend([fused_res, neural_res, pw_res])
        else:
            if neural_res is not None and len(neural_res.pts_src) < 4 and not contingency_fallback["triggered"]:
                contingency_fallback = {
                    "triggered": True,
                    "original_matcher": resolved_matcher,
                    "fallback_matcher": "pwift",
                    "reason": f"Neural matcher '{neural_name}' yielded < 4 matches; degraded to PWIFT only.",
                }
            # Degrade to whichever arm survived: PWIFT alone when the neural
            # arm failed or returned too few matches (the pre-existing
            # fallback), or the neural arm alone when PWIFT was the arm that
            # failed (Task 7).
            results_to_evaluate.extend(
                arm_res for arm_res in (pw_res, neural_res)
                if arm_res is not None and len(arm_res.pts_src) >= 4
            )
    else:
        m = None
        try:
            m = get_matcher(resolved_matcher, cfg)
            arm_src, arm_ref = _matcher_inputs(src_scaled, ref.data, resolved_matcher, cfg)
            res = m.match(arm_src, arm_ref, src_illum=src_illum, ref_illum=ref_illum)
            if (res is None or len(res.pts_src) < 4) and resolved_matcher != "pwift":
                warnings.warn(
                    f"Matcher '{resolved_matcher}' returned insufficient matches "
                    f"({len(res.pts_src) if res is not None else 0} < 4); "
                    "triggering contingency fallback to PWIFT standalone."
                )
                contingency_fallback = {
                    "triggered": True,
                    "original_matcher": resolved_matcher,
                    "fallback_matcher": "pwift",
                    "reason": f"Insufficient matches from {resolved_matcher} (< 4 points)",
                }
                pw_matcher = get_matcher("pwift", cfg)
                try:
                    res = pw_matcher.match(src_scaled, ref.data, src_illum=src_illum, ref_illum=ref_illum)
                finally:
                    del pw_matcher
            if res is not None:
                results_to_evaluate.append(res)
        except Exception as e:
            if resolved_matcher != "pwift":
                warnings.warn(
                    f"Matcher '{resolved_matcher}' failed ({e}); triggering contingency fallback to PWIFT standalone."
                )
                contingency_fallback = {
                    "triggered": True,
                    "original_matcher": resolved_matcher,
                    "fallback_matcher": "pwift",
                    "reason": str(e),
                }
                pw_matcher = get_matcher("pwift", cfg)
                try:
                    res = pw_matcher.match(src_scaled, ref.data, src_illum=src_illum, ref_illum=ref_illum)
                    results_to_evaluate.append(res)
                finally:
                    del pw_matcher
            else:
                raise
        finally:
            if m is not None:
                del m
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

    results_by_method = {}
    competition_pool = []
    # Task 12: did the metadata-seeded hypothesis survive estimation? Used
    # by summary.tiepoint_coarse; provenance tagging comes from viewpoint.
    tiepoint_seed_supported = False

    for match_result in results_to_evaluate:
        candidate_entry = _estimate_arm(
            match_result, sensor=src.sensor, image_shape=src_scaled.shape,
            cfg=cfg, seed_H=tiepoint_report["H"],
            src_img=src_scaled, ref_img=ref.data,
        )
        if candidate_entry is None:
            continue

        tiepoint_seed_supported = (
            tiepoint_seed_supported or candidate_entry["tiepoint_seed_supported"])
        results_by_method[candidate_entry["method"]] = candidate_entry
        competition_pool.append(candidate_entry)

    # ---- Stage 4.5: structural-NCC last-resort arm (Task 8) ----
    # Every run reports this block so summary consumers can rely on one key
    # shape; `attempted` flips only when the switch is on AND no arm already
    # produced a transform — never on a run that already has one.
    structural_report = {
        "attempted": False,
        "n_correspondences": 0,
        "used": False,
        "coverage": 0.0,
    }
    if cfg.structural_fallback_enabled and not any(
        _candidate_has_transform(c) for c in competition_pool
    ):
        structural_report["attempted"] = True
        # src_scaled is already coarse scale/rotation aligned by Stage 3 —
        # exactly the translation-only contract of the generator.
        struct_res = structural_ncc_correspondences(src_scaled, ref.data, cfg=cfg)
        n_struct = int(len(struct_res.pts_src))
        structural_report["n_correspondences"] = n_struct
        # Coverage = fraction of grid cells that produced a kept peak: the
        # generator returns one score per kept peak (<= grid^2 cells, one
        # peak per cell), and an empty return has no observable peaks.
        grid_cells = float(cfg.structural_grid) ** 2
        if struct_res.scores is not None and grid_cells > 0:
            structural_report["coverage"] = float(len(struct_res.scores)) / grid_cells
        # Below the generator's own floor the arm was ATTEMPTED but not
        # USED: nothing else about the run changes (same RuntimeError /
        # fail-closed verdict as if the arm never existed).
        if n_struct >= cfg.structural_min_correspondences:
            struct_candidate = _estimate_arm(
                struct_res, sensor=src.sensor, image_shape=src_scaled.shape,
                cfg=cfg, seed_H=tiepoint_report["H"],
                src_img=src_scaled, ref_img=ref.data,
            )
            if struct_candidate is not None:
                structural_report["used"] = True
                tiepoint_seed_supported = (
                    tiepoint_seed_supported
                    or struct_candidate["tiepoint_seed_supported"])
                results_by_method[struct_candidate["method"]] = struct_candidate
                competition_pool.append(struct_candidate)

    if not results_by_method:
        raise RuntimeError(
            f"No usable matches from {resolved_matcher} (requested: {matcher}). Check input images / thresholds."
        )

    # Rigid-only hypothesis competition (§4)
    comp_res = compete_rigid(
        competition_pool,
        terrain_spacing_px=cfg.rigid_terrain_spacing_px,
        lambda_dof=cfg.rigid_lambda_dof,
        margin_min=cfg.rigid_margin_min,
    )
    best_method = comp_res["winner"]["method"]
    best = results_by_method[best_method]

    # Task 11: the summary must not claim "no contingency" when the arm that
    # won the competition carries zero geometrically valid inliers.
    note_winning_arm_support(
        contingency_fallback, best_method, best["metrics"].n_inliers,
    )

    # ---- Orthogonal Verification Gate (gate_cheap) ----
    primary_H = best["hom_result"].H if not isinstance(best["hom_result"], list) else (
        best["hom_result"][0].H if best["hom_result"] else None
    )
    ortho_eval = None
    if cfg.enable_orthogonal_gate and primary_H is not None:
        ortho_eval = gate_cheap(
            primary_H,
            src_scaled,
            ref.data,
            gsd_src=src.gsd_m or 1.0,
            gsd_ref=ref.gsd_m or 1.0,
            t_struct=cfg.orthogonal_gate_t_struct,
            tau_agree=cfg.orthogonal_gate_tau_agree,
            k_sigma=cfg.orthogonal_gate_k_sigma,
        )

    # Fail-closed verdict (Task 4): no valid transform, or a gate FAIL, is a
    # failed registration — never a "flagged confidence" success. Products are
    # withheld; summary.json still records why (the demo keys off exit codes).
    # Task 6: an unconfident coarse search is an UPSTREAM failure — it
    # outranks whatever the gate reports about the mis-scaled diagnostic run.
    failure_reason = scale_search_failure
    if failure_reason is None:
        if primary_H is None:
            failure_reason = (
                f"no_transform: matcher '{resolved_matcher}' produced no "
                "geometrically valid transform"
            )
        elif ortho_eval is not None and not ortho_eval.get("pass", False):
            failure_reason = f"verification_gate_failed: {ortho_eval.get('reason')}"

    if failure_reason is None:
        # ---- Stage 5: MiHo Piecewise Geometry + 6x6 Gridded GCP Optimizer (§2) ----
        miho_out = miho_plus_gcp(primary_H, best["match_result"], grid_size=cfg.miho_grid_size, target_gcps=cfg.miho_target_gcps)

        if not isinstance(best["hom_result"], list):
            best["hom_result"].Hs_local = miho_out.get("Hs_local")
            best["hom_result"].gcps = miho_out.get("gcps")

        # ---- Stage 5.5: Cause-Branched Subpixel Refinement (§5) ----
        src_inc_mean = float(np.nanmean(src_incidence_scaled)) if src_incidence_scaled is not None else 0.0
        ref_inc_mean = float(np.nanmean(ref_incidence)) if ref_incidence is not None else 0.0
        illum_delta_deg = abs(src_inc_mean - ref_inc_mean)
        subpixel_refine_out = refine_tile(src_scaled, ref.data, illum_delta_deg=illum_delta_deg)
        # Task 8: the refine's inputs (src_scaled vs ref.data) never include
        # the matcher's transform, so its dx/dy are the GROSS
        # source-vs-reference offset — identical for every matcher on this
        # pair (the report observed 123.5054/271.0260 with pwift and roma2
        # alike). That is degenerate as a "subpixel refinement" claim: force
        # low_precision and say why, on top of any value-level objections
        # refine_tile itself raised (weak peak, non-subpixel magnitude).
        structural_reason = (
            "matcher-independent inputs: refine correlates src_scaled vs "
            "reference without the matcher's transform — dx/dy reproduce the "
            "gross source-vs-reference offset (identical for every matcher "
            "on this pair), not a registration residual"
        )
        refine_value_reason = subpixel_refine_out.get("reason")
        subpixel_refine_out["reason"] = (
            f"{refine_value_reason}; {structural_reason}" if refine_value_reason
            else structural_reason
        )
        subpixel_refine_out["low_precision"] = True

        # ---- Stage 6: georeferencing and output ----
        registered = register_image(src_scaled, ref.data.shape, best["hom_result"])
        outputs = write_outputs(
            out_dir, tag=os.path.splitext(os.path.basename(source_path))[0],
            registered_img=registered,
            pts_src=best["match_result"].pts_src, pts_dst=best["match_result"].pts_dst,
            inlier_mask=best["inlier_mask"], method=best_method,
            ref_geotransform=ref.geotransform, ref_crs=ref.crs,
            src_img=src_scaled, ref_img=ref.data,
            gcps=miho_out.get("gcps", []),
            export_gcl_gcps_csv=cfg.export_gcl_gcps_csv,
        )
    else:
        logger.warning(
            "fail-closed: %s — withholding MiHo/GCP/refine stages and all "
            "registered output products (summary.json only)",
            failure_reason,
        )
        miho_out = {"gcps": [], "coverage": 0.0, "Hs_local": None}
        subpixel_refine_out = {
            "method": None, "dx": None, "dy": None, "low_precision": True,
            "reason": f"skipped: {failure_reason}",
        }
        outputs = {}

    summary = {
        "source": source_path, "reference": reference_path,
        "passed": failure_reason is None,
        "failure_reason": failure_reason,
        "sensor": src.sensor.name, "matcher": matcher, "resolved_matcher": resolved_matcher,
        "best_method": best_method,
        # The gate verifies THIS transform, so the report has to carry it:
        # a 3x3 homography from the coarse-aligned source into the reference
        # frame (None when no arm produced one). Consumers decompose it to
        # recover the rotation/scale the run actually committed to.
        "homography": (None if primary_H is None
                       else np.asarray(primary_H, dtype=float).tolist()),
        "condition_routing": routing_info,
        "tiepoint_coarse": {
            "derived": tiepoint_report["H"] is not None,
            "rotation_deg": tiepoint_report["rotation_deg"],
            "scale": tiepoint_report["scale"],
            "reason": tiepoint_report["reason"],
            # None = nothing was derived (nothing to support); True/False =
            # derived and the matcher's correspondences did/didn't back it.
            "seed_supported": (tiepoint_seed_supported
                               if tiepoint_report["H"] is not None else None),
        },
        "contingency_fallback": contingency_fallback,
        # Task 8: last-resort arm report, present on every run (attempted
        # False throughout when the arm never fired).
        "structural_correspondences": structural_report,
        "orthogonal_gate": {
            "enabled": cfg.enable_orthogonal_gate,
            "passed": (ortho_eval.get("pass", False) if ortho_eval is not None
                       else primary_H is not None),
            "reason": (ortho_eval.get("reason") if ortho_eval is not None
                       else ("disabled" if not cfg.enable_orthogonal_gate
                             else "skipped: no transform")),
            "cost_ms": ortho_eval.get("cost_ms", 0.0) if ortho_eval is not None else 0.0,
            "struct_ncc": ortho_eval.get("struct_ncc", 0.0) if ortho_eval is not None else 0.0,
        },
        "provenance": getattr(best["match_result"], "provenance", "direct"),
        "chosen_scale": best_scale, "chosen_rotation_deg": best_rot,
        "gsd_scale_prior": gsd_scale_prior,
        "rigid_competition": {
            "margin": comp_res.get("margin", 0.0),
            "ambiguous": comp_res.get("ambiguous", False),
        },
        "miho_gcps": {
            "count": len(miho_out.get("gcps", [])),
            "coverage": miho_out.get("coverage", 0.0),
            "csv_path": outputs.get("gcl_gcps_csv"),
            "gcps": miho_out.get("gcps", [])[:5],
        },
        "subpixel_refine": {
            "method": subpixel_refine_out.get("method"),
            "dx": subpixel_refine_out.get("dx"),
            "dy": subpixel_refine_out.get("dy"),
            "low_precision": subpixel_refine_out.get("low_precision", False),
            "reason": subpixel_refine_out.get("reason"),
        },
        "metrics": {
            m: {
                "n_matches": r["metrics"].n_matches, "n_inliers": r["metrics"].n_inliers,
                "inlier_ratio": r["metrics"].inlier_ratio, "rmse_px": r["metrics"].rmse_px,
                "uniformity_score": r["metrics"].uniformity_score,
            }
            for m, r in results_by_method.items()
        },
        "outputs": outputs,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="CH2 <-> LROC lunar image registration pipeline")
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-sensor", default=None, choices=["OHRC", "TMC", "IIRS", "LROC"])
    parser.add_argument(
        "--matcher", default="auto",
        choices=["auto", "hybrid_pwift_roma2", "hybrid_pwift_eloftr", "roma2", "eloftr", "pwift"],
        help="Matcher model to use. Defaults to 'auto' for condition-based adaptive dispatch.",
    )
    parser.add_argument(
        "--real-time", "--descent-trn", action="store_true", dest="real_time",
        help="Enable real-time descent TRN mode (routes to EfficientLoFTR).",
    )
    parser.add_argument(
        "--polar-threshold-deg", type=float, default=70.0,
        help="Solar incidence angle threshold (deg) for extreme polar grazing illumination routing.",
    )
    parser.add_argument(
        "--disable-orthogonal-gate", action="store_true",
        help="Disable the orthogonal verification gate (gate_cheap).",
    )
    parser.add_argument("--roma2-weights", default=None, help="Path to fine-tuned RoMa v2 weights")
    parser.add_argument("--eloftr-ckpt", default=None, help="Path to fine-tuned EfficientLoFTR checkpoint")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--source-incidence", default=None)
    parser.add_argument("--source-emission", default=None)
    parser.add_argument("--source-phase", default=None)
    parser.add_argument(
        "--source-sidecar-xml", default=None,
        help="Explicit path to the source product's sidecar XML. Overrides "
             "sibling auto-discovery; a missing/unreadable/untagged file "
             "aborts the run instead of silently routing at the placeholder.",
    )
    parser.add_argument("--source-nac-pho", default=None,
                         help="Path to the source image's LROC NAC_PHO photometry cube.")
    parser.add_argument("--reference-nac-pho", default=None,
                         help="Path to the reference image's LROC NAC_PHO photometry cube.")
    parser.add_argument("--nac-pho-band-phase", type=int, default=2)
    parser.add_argument("--nac-pho-band-emission", type=int, default=3)
    parser.add_argument("--nac-pho-band-incidence", type=int, default=4)
    parser.add_argument("--angles-from-label", action="store_true")
    parser.add_argument("--fetch-lroc-angles", action="store_true",
                         help="Auto-fetch incidence/emission/phase from the LROC ODE page.")
    parser.add_argument("--incidence-deg", type=float, default=None,
                         help="Manually supply the source image's incidence angle in degrees.")
    parser.add_argument("--emission-deg", type=float, default=None,
                         help="Manually supply the source image's emission angle in degrees.")
    parser.add_argument("--phase-deg", type=float, default=None,
                         help="Manually supply the source image's phase angle in degrees.")
    parser.add_argument(
        "--window", default=None,
        help="Backward-compatible shared x,y,w,h crop.",
    )
    parser.add_argument(
        "--source-window", default=None,
        help="Source-image crop x,y,w,h.",
    )
    parser.add_argument(
        "--reference-window", default=None,
        help="Reference-image crop x,y,w,h.",
    )
    parser.add_argument("--no-eloftr", action="store_true", help="Shorthand to run PWIFT only")
    parser.add_argument(
        "--fuse-pwift-eloftr",
        action="store_true",
        help="Shorthand for --matcher hybrid_pwift_eloftr",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        source_path=args.source, reference_path=args.reference, out_dir=args.out_dir,
        source_sensor=args.source_sensor,
        matcher=args.matcher,
        real_time=args.real_time,
        enable_orthogonal_gate=not args.disable_orthogonal_gate,
        polar_incidence_threshold_deg=args.polar_threshold_deg,
        roma2_weights=args.roma2_weights,
        eloftr_checkpoint=args.eloftr_ckpt,
        device=args.device,
        source_incidence_path=args.source_incidence, source_emission_path=args.source_emission,
        source_phase_path=args.source_phase,
        source_nac_pho_path=args.source_nac_pho, reference_nac_pho_path=args.reference_nac_pho,
        nac_pho_band_phase=args.nac_pho_band_phase, nac_pho_band_emission=args.nac_pho_band_emission,
        nac_pho_band_incidence=args.nac_pho_band_incidence,
        angles_from_label=args.angles_from_label,
        fetch_angles_online=args.fetch_lroc_angles,
        manual_incidence_deg=args.incidence_deg, manual_emission_deg=args.emission_deg,
        manual_phase_deg=args.phase_deg,
        source_sidecar_xml=args.source_sidecar_xml,
        window=_parse_window(args.window),
        source_window=_parse_window(args.source_window),
        reference_window=_parse_window(args.reference_window),
        use_eloftr=not args.no_eloftr,
        fuse_pwift_eloftr=args.fuse_pwift_eloftr,
    )
    print(json.dumps(summary, indent=2))

    if not summary.get("passed", False):
        # Fail-closed contract: a registration that failed verification must
        # exit nonzero so demo scripts/CI never key off a green exit alone.
        raise SystemExit(1)


if __name__ == "__main__":
    main()