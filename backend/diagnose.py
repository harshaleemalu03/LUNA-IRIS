"""
Diagnostic: runs each pipeline stage on your real images and prints stats at
every step, so we can see exactly where it goes wrong instead of just getting
"No usable matches" from the full pipeline.

Run with per-image windows (Task 9 — was a single --window for both sides):
    python diagnose.py --source data/S18RAW... --source-sensor OHRC \
        --reference data/M150368601RE... --reference-sensor LROC \
        --source-window 2243,298,512,512 --reference-window 0,0,512,512

`--window x,y,w,h` remains the shorthand that sets BOTH windows when the
per-image flags are absent. The reference sensor defaults to autodetection
(was hardcoded to "LROC", which silently mis-conditioned every non-LROC
reference — Task 9).

WINDOWS / MEMORY (full-image OOM): products in the eval set are 8-25 MP
(OHRC references 15-25 MP, IIRS strips ~8 MP). Phase congruency holds
n_scales x n_orient FFT-sized arrays per image at full size, so a
full-resolution run can exhaust RAM (swap/OOM) on a demo laptop. Prefer
windows. The API has the same exposure and auto-tiles inputs to <= 4 MP
(pipeline.py MAX_SAFE_PIXELS); diagnose.py does NOT auto-tile — you get the
window you asked for (honest what-you-see), so pick windows inside the
geographic overlap of the pair.

STAGE DISPATCH: stage 2 returns a PWIFTMaps bundle for pwift_akimov sensors
(OHRC/LROC) and a plain array otherwise (TMC/IIRS). Stages 3a-3c dispatch
exactly like matching.py: when BOTH sides are PWIFTMaps the paper path runs
(detect_keypoints_pw -> dominant_orientation_pw ->
compute_bichannel_descriptors -> swap_aware_match*); otherwise the generic
path runs on the M_PW response map extracted from any PWIFTMaps side (the
"mixed" case gets a printed note). Keypoint knobs differ by path: the paper
path is tuned with --score-percentile (pwift_keypoint_score_percentile),
the generic path with --keypoint-threshold (default 0.08, matching
matching.run_pwift_matching_generic — PipelineConfig has no
pwift_keypoint_threshold field; Task 9 stopped pretending otherwise).

ANGLE MAPS: src angle maps are loaded and printed but NOT applied here —
diagnose's illumination stage runs unweighted (W=1) and says so; the
pipeline applies angles after the scale/rotation stage. Reproducing a
forced-angles behaviour needs the pipeline, not this tool.

Paste the full printed output back.
"""

import argparse

import numpy as np

from lunar_registration.preprocessing import load_image
from lunar_registration.illumination import apply_illumination_correction
from lunar_registration.scale import select_best_scale, apply_scale, apply_rotation
from lunar_registration.pwift import (
    PWIFTMaps,
    compute_bichannel_descriptors,
    compute_context_descriptor,
    compute_descriptors,
    detect_keypoints_dual_channel,
    detect_keypoints_pw,
    dominant_orientation_pw,
    match_descriptors_swap_aware,
    swap_aware_match,
    swap_aware_match_with_context,
)
from lunar_registration.config import PipelineConfig

# Generic-path keypoint threshold actually used by
# matching.run_pwift_matching_generic (which guards the nonexistent
# cfg.pwift_keypoint_threshold via hasattr and falls back to this value).
GENERIC_KEYPOINT_THRESHOLD = 0.08


def stats(name, arr):
    """Print array stats; PWIFTMaps is a bundle with no .shape (Task 9) —
    summarize its fields instead of crashing on it."""
    if isinstance(arr, PWIFTMaps):
        fields = ", ".join(
            f"{field}={getattr(arr, field).shape}"
            for field in ("M_PW", "m_PW", "MIM", "w", "w_soft", "mask")
        )
        print(f"{name}: PWIFTMaps[{fields}] "
              f"M_PW min={arr.M_PW.min():.4f} max={arr.M_PW.max():.4f} "
              f"mean={arr.M_PW.mean():.4f} std={arr.M_PW.std():.4f} "
              f"mask_true_frac={np.mean(arr.mask):.4f}")
        return
    print(f"{name}: shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}")


def _parse_window(text):
    x, y, w, h = (int(v) for v in text.split(","))
    return (x, y, w, h)


def _pc_map(illum):
    """Generic path wants a single response array; extract M_PW from a
    PWIFTMaps side (mixed src/ref representation case)."""
    return illum.M_PW if isinstance(illum, PWIFTMaps) else illum


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--source-sensor", default=None)
    p.add_argument("--reference-sensor", default=None,
                   help="Sensor hint for the reference image "
                        "(default: autodetected; was hardcoded to LROC).")
    p.add_argument("--angles-from-label", action="store_true")
    p.add_argument("--window", default=None,
                   help="x,y,w,h — shorthand applied to BOTH images when the "
                        "per-image windows are not given.")
    p.add_argument("--source-window", default=None, help="x,y,w,h crop of the source only")
    p.add_argument("--reference-window", default=None, help="x,y,w,h crop of the reference only")
    p.add_argument("--ratio-test", type=float, default=None,
                   help="Override PipelineConfig.pwift_ratio_test (default 0.85). "
                        "Higher = looser = more matches, more false positives.")
    p.add_argument("--keypoint-threshold", type=float, default=None,
                   help="Generic-path (TMC/IIRS) keypoint threshold "
                        f"(default {GENERIC_KEYPOINT_THRESHOLD}, matching "
                        "matching.run_pwift_matching_generic). The paper path "
                        "(OHRC/LROC) has no fixed threshold — tune it with "
                        "--score-percentile instead.")
    p.add_argument("--score-percentile", type=float, default=None,
                   help="Override PipelineConfig.pwift_keypoint_score_percentile "
                        "(default 60.0) — the paper path's adaptive quality-"
                        "screening knob; lower = more keypoints.")
    args = p.parse_args()

    # Task 9: per-image windows; --window stays the both-images shorthand.
    shorthand = _parse_window(args.window) if args.window else None
    src_window = _parse_window(args.source_window) if args.source_window else shorthand
    ref_window = _parse_window(args.reference_window) if args.reference_window else shorthand

    cfg = PipelineConfig()
    if args.ratio_test is not None:
        cfg.pwift_ratio_test = args.ratio_test
    if args.score_percentile is not None:
        cfg.pwift_keypoint_score_percentile = args.score_percentile
    generic_threshold = GENERIC_KEYPOINT_THRESHOLD
    if args.keypoint_threshold is not None:
        generic_threshold = args.keypoint_threshold
    # Task 9: print knobs that actually exist on PipelineConfig — reading
    # cfg.pwift_keypoint_threshold (l.55 before the fix) crashed every run
    # that did not pass --keypoint-threshold.
    print(f"Using pwift_ratio_test={cfg.pwift_ratio_test}, "
          f"pwift_keypoint_score_percentile={cfg.pwift_keypoint_score_percentile} (paper path), "
          f"generic keypoint threshold={generic_threshold} (generic path)")

    print("=== Loading images ===")
    src = load_image(args.source, sensor_hint=args.source_sensor, window=src_window,
                     angles_from_label=args.angles_from_label)
    ref = load_image(args.reference, sensor_hint=args.reference_sensor, window=ref_window)
    stats("src.data (normalized)", src.data)
    stats("ref.data (normalized)", ref.data)
    print(f"src sensor: {src.sensor.name}, ref sensor: {ref.sensor.name}")
    if src.incidence_deg is not None:
        print(f"src incidence_deg: {src.incidence_deg[0, 0]:.2f}, "
              f"emission_deg: {src.emission_deg[0, 0]:.2f}")
        print("note: src angle maps are loaded but NOT applied by diagnose "
              "(illumination runs unweighted, W=1); the pipeline applies them "
              "after the scale/rotation stage.")
    else:
        print("src incidence/emission: None (label didn't have them, or --angles-from-label not set)")

    print("\n=== Stage 5a: coarse scale/rotation search ===")
    scale_rot = select_best_scale(src.data, ref.data, src.sensor, cfg)
    if scale_rot is None:
        # Task 6: no confident alignment — resampling with a garbage scale
        # would only produce misleading diagnostics below.
        raise SystemExit(
            "no confident alignment: scale/rotation search argmax was a "
            "boundary pick or flat-surface tie; refusing to continue"
        )
    best_scale, best_rot = scale_rot
    print(f"chosen_scale={best_scale:.4f}, chosen_rotation_deg={best_rot}")
    src_scaled = apply_scale(src.data, best_scale)
    # Mirror the pipeline: it applies BOTH the chosen scale and rotation
    # before illumination/matching; diagnose used to drop the rotation,
    # which made every downstream diagnosis about a different input.
    src_scaled = apply_rotation(src_scaled, best_rot)
    stats("src_scaled", src_scaled)

    print("\n=== Stage 2: illumination correction (phase congruency) ===")
    src_illum = apply_illumination_correction(
        src_scaled, src.sensor, incidence_deg=None, emission_deg=None,
        n_scales=cfg.pwift_scales, n_orient=cfg.pwift_orientations,
    )
    ref_illum = apply_illumination_correction(
        ref.data, ref.sensor, n_scales=cfg.pwift_scales, n_orient=cfg.pwift_orientations,
    )
    stats("src_illum", src_illum)
    stats("ref_illum", ref_illum)

    # Dispatch exactly like matching.py: BOTH PWIFTMaps -> paper path.
    paper_path = isinstance(src_illum, PWIFTMaps) and isinstance(ref_illum, PWIFTMaps)
    if paper_path:
        print("illumination path: paper (PWIFTMaps -> run_pwift_matching_pw stages)")
    else:
        print("illumination path: generic (M_PW response maps -> "
              "run_pwift_matching_generic stages)")
        if isinstance(src_illum, PWIFTMaps) or isinstance(ref_illum, PWIFTMaps):
            print("  mixed illumination representations — the PWIFTMaps side is "
                  "run through the generic path on its extracted M_PW map")

    print("\n=== Stage 3a: keypoint detection ===")
    if paper_path:
        kp_src = detect_keypoints_pw(
            src_illum.M_PW, src_illum.m_PW, src_illum.w_soft, src_illum.mask,
            min_distance=cfg.pwift_min_keypoint_distance,
            max_keypoints=cfg.pwift_max_keypoints,
            min_retention_ratio=cfg.pwift_min_retention_ratio,
            score_percentile=cfg.pwift_keypoint_score_percentile,
        )
        kp_ref = detect_keypoints_pw(
            ref_illum.M_PW, ref_illum.m_PW, ref_illum.w_soft, ref_illum.mask,
            min_distance=cfg.pwift_min_keypoint_distance,
            max_keypoints=cfg.pwift_max_keypoints,
            min_retention_ratio=cfg.pwift_min_retention_ratio,
            score_percentile=cfg.pwift_keypoint_score_percentile,
        )
        print(f"n_keypoints src={len(kp_src)}, ref={len(kp_ref)} "
              f"(score_percentile={cfg.pwift_keypoint_score_percentile})")
    else:
        src_pc, ref_pc = _pc_map(src_illum), _pc_map(ref_illum)
        kp_src = detect_keypoints_dual_channel(src_pc, threshold=generic_threshold)
        kp_ref = detect_keypoints_dual_channel(ref_pc, threshold=generic_threshold)
        print(f"n_keypoints src={len(kp_src)}, ref={len(kp_ref)} "
              f"(threshold={generic_threshold})")
    if kp_src:
        resp_src = [k.response for k in kp_src]
        print(f"  src response range: {min(resp_src):.4f} - {max(resp_src):.4f}")
    if kp_ref:
        resp_ref = [k.response for k in kp_ref]
        print(f"  ref response range: {min(resp_ref):.4f} - {max(resp_ref):.4f}")

    print("\n=== Stage 3b: descriptors ===")
    if paper_path:
        for kp in kp_src:
            dominant_orientation_pw(kp, src_illum.MIM, src_illum.M_PW,
                                    src_illum.w_soft, K=cfg.pwift_orientations)
        for kp in kp_ref:
            dominant_orientation_pw(kp, ref_illum.MIM, ref_illum.M_PW,
                                    ref_illum.w_soft, K=cfg.pwift_orientations)
        desc_src = compute_bichannel_descriptors(
            kp_src, src_illum.MIM, src_illum.M_PW, src_illum.w_soft, src_illum.w,
            patch_size=cfg.pwift_descriptor_patch, no=cfg.pwift_descriptor_cells,
            nbins=cfg.pwift_orientations, t=cfg.pwift_bright_dark_threshold,
        )
        desc_ref = compute_bichannel_descriptors(
            kp_ref, ref_illum.MIM, ref_illum.M_PW, ref_illum.w_soft, ref_illum.w,
            patch_size=cfg.pwift_descriptor_patch, no=cfg.pwift_descriptor_cells,
            nbins=cfg.pwift_orientations, t=cfg.pwift_bright_dark_threshold,
        )
    else:
        desc_src = compute_descriptors(src_scaled, src_pc, kp_src,
                                       patch_size=cfg.pwift_descriptor_patch)
        desc_ref = compute_descriptors(ref.data, ref_pc, kp_ref,
                                       patch_size=cfg.pwift_descriptor_patch)
    print(f"n_descriptors src={len(desc_src)}, ref={len(desc_ref)} "
          f"(dropped if too close to image edge for patch_size={cfg.pwift_descriptor_patch})")

    print("\n=== Stage 3c: matching ===")
    if paper_path:
        if cfg.pwift_use_context_descriptor:
            context_src = compute_context_descriptor(
                [d.keypoint for d in desc_src], src_illum.M_PW, src_illum.mask,
                n_rings=cfg.pwift_context_rings, n_sectors=cfg.pwift_context_sectors,
                ring_spacing_px=cfg.pwift_context_ring_spacing_px,
            )
            context_ref = compute_context_descriptor(
                [d.keypoint for d in desc_ref], ref_illum.M_PW, ref_illum.mask,
                n_rings=cfg.pwift_context_rings, n_sectors=cfg.pwift_context_sectors,
                ring_spacing_px=cfg.pwift_context_ring_spacing_px,
            )
            matches = swap_aware_match_with_context(
                desc_src, desc_ref, context_src, context_ref,
                no=cfg.pwift_descriptor_cells, nbins=cfg.pwift_orientations,
                ratio_test=cfg.pwift_ratio_test, context_weight=cfg.pwift_context_weight,
            )
        else:
            matches = swap_aware_match(
                desc_src, desc_ref, no=cfg.pwift_descriptor_cells,
                nbins=cfg.pwift_orientations, ratio_test=cfg.pwift_ratio_test,
            )
    else:
        matches = match_descriptors_swap_aware(
            desc_src, desc_ref, ratio_test=cfg.pwift_ratio_test)
    print(f"n_matches={len(matches)} (ratio_test={cfg.pwift_ratio_test})")

    print("\n=== Diagnosis ===")
    if src.data.std() < 0.02 or ref.data.std() < 0.02:
        print("!! Very low pixel variance in src or ref - likely a blank/no-data region. "
              "Try a different window (--source-window/--reference-window) or drop "
              "--window to use the full image (watch RAM — see OOM note above).")
    if len(kp_src) == 0 or len(kp_ref) == 0:
        if paper_path:
            print("!! Zero keypoints detected - the paper path's adaptive screening cut "
                  "everything; try lowering --score-percentile (e.g. to 40) or check the "
                  "M_PW response range printed above.")
        else:
            print(f"!! Zero keypoints detected - the generic threshold {generic_threshold} "
                  "is too high for this image's response range shown above. Try lowering "
                  "--keypoint-threshold (e.g. to 0.03).")
    elif len(desc_src) == 0 or len(desc_ref) == 0:
        print("!! Keypoints found but all too close to the image edge for the descriptor "
              "patch - try a larger window or smaller pwift_descriptor_patch.")
    elif len(matches) == 0:
        print("!! Descriptors computed but none passed the ratio test - try raising "
              "PipelineConfig.pwift_ratio_test (e.g. to 0.95, looser) or check whether "
              "chosen_scale/chosen_rotation above look plausible for your actual image pair.")
    elif len(matches) < 4:
        print(f"!! Only {len(matches)} match(es) - not enough for RANSAC (needs >=4). "
              "The descriptor is finding correspondences (good sign - not zero) but the "
              "ratio test is rejecting most of them. Try: --ratio-test 0.95 (looser) "
              "and/or more keypoints (--keypoint-threshold 0.03 on the generic path, "
              "--score-percentile 40 on the paper path).")
    else:
        print(f"{len(matches)} matches found - the full pipeline should work with these "
              "settings; if it still errors, the issue is downstream (homography/RANSAC).")


if __name__ == "__main__":
    main()
