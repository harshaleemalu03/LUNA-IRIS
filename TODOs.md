# LUNA-IRiS Registration Pipeline — TODOs

Status: **delivered 2026-09-25** on branch `feat/todos-backlog` (TODO-1/TODO-2 landed earlier on
`fix/pipeline-repair`, merged at `9688380`). Full suite: **151 passed, 1 skipped**.
Source: verified diagnostic session +4 control-validated experiments (see `luna-iris-registration-guide.md` for full evidence).
Canonical pair: `OHRXXD18CHO2436502NNNN25039175231280_V2_1_01`, windows `--source-window 0,1400,648,2200 --reference-window 770,1768,1142,2713`, `--source-sensor OHRC`.

---

## Tier 1 — Sure bug fixes (zero research risk, do first)

### TODO-1: Read product metadata (angles + true GSD) ✷ highest leverage
- [x] Parse incidence/emission/phase + ModelPixelScale/GSD from product XML and feed them into the pipeline.
  - Evidence: `36a10e9` (wire pipeline to utilize xml metadata); canonical run routes on `incidence_deg 84.896724`.
- [x] Use truth GSD ratio (measured **1.0067**, source `5.0334` / ref `5.0`) as the scale prior — today the prior is **0.5** and chosen scale **0.575** (wrong by ~1.75×).
  - Evidence: canonical run `gsd_scale_prior 1.006682`. **Acceptance shortfall:** the search still lands `chosen_scale 0.9464` (−6.0% vs 1.0067), outside the `±0.01` band — the prior is correct, the coarse-to-fine search is not pinned to it.
- [x] Feed XML `Solar_incidence_angle_in_degree = 84.896724` so the `incidence ≥ 70°` routing branch actually fires (currently never reached: default 30° input).
  - Evidence: canonical run `regime: "polar_grazing"`, reason `Incidence 84.9° >= 70.0°: PWIFT harmonic Akimov masking suppresses migrating shadow boundaries.`
- Files: `backend/lunar_registration/pipeline.py` — angle input block ~245-254, routing ~244-257; `backend/lunar_registration/scale.py` — `select_best_scale` prior band43-91, `estimate_gsd_scale_prior` ~1858.
- Acceptance: run summary shows `chosen_scale ≈ 1.0067` (±0.01) and routing `regime` reflects measured84.9° incidence. → **partial**: prior + routing ✓, chosen scale ✗ (see above).

### TODO-2: Fail-closed gate ✷ honesty fix
- [x] Gate failure → non-zero exit code + `"status": "failed"` in API response; remove "Proceeding with flagged confidence" + exit 0.
  - Evidence: `e1238a1` (passed flag, output withholding, exit codes, API 422); canonical pair exits `1` with `failure_reason: no_transform`; `test_fail_closed.py` asserts both directions.
- [x] Handle the `reason: "disabled"` path: `passed:true` while `enabled:true` and no transform evaluated is a false pass — must be `passed:false`.
  - Evidence: `backend/tests/test_fail_closed.py` (`reason:"disabled"` while `enabled:true` must fail).
- Files: `pipeline.py` ~492-544 (warning at ~504), `backend/api/app.py` ~95-100.
- Acceptance: all current failing runs (run0-4) return failure; no run exits 0 with a flagged/unverified transform. ✓

### TODO-3: Fix gate check #1 (rotation-blind phase agreement)
- [x] Make `_coarse_phase_align_256` rotation-aware: rotate/warp img0 by H's rotation before phase correlation, **or** compare rotation-compensated translations.
  - Shipped as the "residual" variant (option A): both images are mapped to a common 256×256 frame with `S = diag(256/w, 256/h)`, the source is warped by H, and the phase shift of the *residual* is measured — correct for unequal sizes.
  - Evidence: `backend/tests/test_gate_rotation.py` — a correct 11° H gives `0.002px` residual (the old translation check reported `38.2px > 24` on the same pair); a 100px translation error gives `99.95px` and fails.
- Evidence: at the best-found (correct-band) transform it reports **475.6px disagreement** vs `tau_agree=24` — it structurally cannot pass any pair rotated ~9°.
- Files: `backend/lunar_registration/verify.py` — `_coarse_phase_align_256` ~74-95, `gate_cheap` check1 ~123-135; threshold `config.py` ~205.
- Acceptance: a synthetically-rotated control pair (known 11° rotation) passes check #1 when H is correct. ✓

### TODO-4: Recalibrate `t_struct`
- [x] Lower `orthogonal_gate_t_struct` **0.40 → 0.25** (measured band: identity **0.024**, garbage-H **0.040**, exhaustive-search best **0.317**, same-content control ceiling **0.474**).
- [ ] Alternative (pick one): require score to **beat identity by a clear margin** (e.g. `≥ identity + 0.15`).
  - Not shipped — the threshold was the chosen option; the identity-margin variant was rejected as more complex for the same discriminating power.
- Files: `config.py` ~204; metric `verify.py` `compute_structural_ncc` ~21-71.
- Note: 0.40 sits above what this pair class can achieve even with perfect alignment — current bar is unreachable by design.

### TODO-5: API robustness
- [x] Expose `source_window` / `reference_window` params in `POST /api/register` (absent today).
  - Evidence: `backend/tests/test_api_windows.py::test_register_forwards_crop_windows` + `::test_register_rejects_malformed_window`.
- [x] Handle the 4 MP guard gracefully (upsample/downsample or clear400 error instead of crash) — `pipeline.py` ~221-226.
  - Evidence: `_center_tile_window` (largest centered crop ≤ `max_pixels`, native resolution kept); `test_size_guard_auto_tiles_instead_of_crashing`.
- [x] Pin environment: torch/kornia/cv2 versions; pre-seed `~/.cache/torch/hub/checkpoints/romav2.0.1.pt` (fine-tune overlays 907/907 keys, base download is redundant — `third_party/RoMaV2/src/romav2/romav2.py` ~98-113).
  - Evidence: every entry in `requirements.txt` / `backend/requirements.txt` is pinned `==` (incl. `torch`, `torchvision`, `opencv-*`, `transformers`); `requirements-dev.txt` records `pytest` + `httpx`.
- Files: `backend/api/app.py` ~95-100.

---

## Tier 2 — Measured robustness gains (proven on this pair)

### TODO-6: Photometric normalization before fine-tuned matchers ✷ most certain improvement
- [x] Insert normalization step before `EloftrMatcher.match` (and test the same before `Roma2Matcher`).
  - Evidence: `backend/lunar_registration/photometric.py` + `_matcher_inputs` in `pipeline.py` — normalization applies to every neural arm (`roma2`, `eloftr`, hybrids); `test_photometric_pipeline.py` records the images each arm receives (PWIFT keeps the raw input).
- [x] Implement (measured, in order of promise on canonical pair):
  - **gradient-domain** (Sobel magnitude ÷ p99.5) — default `neural_input_normalization="gradient"`
  - **local-contrast / Weber** (x−μ)/(σ+ε)
  - CLAHE
- [ ] RoMaV2 + normalization: **untested** — cheap experiment, run it.
  - Blocked in this environment: the canonical pair's `roma2` arm fails with a CUDA device mismatch (CPU-only torch) and the weights resolve to host paths outside the repo. The wiring itself is covered by `test_photometric_pipeline.py`; the end-to-end number is not reproducible here.
- Files: `backend/lunar_registration/matching.py` dispatch ~168-267, `EloftrMatcher.match` ~750+, `Roma2Matcher` ~368/462.
- Acceptance: canonical-pair run yields ≥25 RANSAC inliers@3px from at least one normalized matcher (vs 3–5 baseline). → **not evidenced**: the canonical pair fails before a neural arm can produce a transform (CUDA/weights); module + wiring tests pass.

### TODO-7: Route all arms through the existing fusion/voting layer
- [x] Ensure normalized matcher outputs + PWIFT go through `fuse_pwift_neural` + `compete_rigid` (already implemented — currently single-arm runs bypass real competition).
  - Evidence: a bare neural name (`roma2`/`eloftr`) now takes the same fusion path as `hybrid_pwift_<name>` — PWIFT always runs alongside and `compete_rigid` arbitrates; `matcher=pwift` stays single-arm. `test_fusion_routing.py`.
- [x] Keep contingency fallback semantics (matcher fail → PWIFT standalone) but make fallback **visible** in API status.
  - Evidence: `contingency_fallback` in both the 422 `detail` and the 200 body (`test_contingency_truth.py`); canonical run reports `roma2 → PWIFT` degradation in `summary.json`.
- Files: `fusion.py` `fuse_pwift_neural` ~235, `verify.py` `compete_rigid` ~173, `pipeline.py` ~445-544.

---

## Tier 3 — One high-value experiment (decides whether this pair can register at all)

### TODO-8: Dense structural NCC → GCPs (correspondence generator)
- [x] Correlate the two structural maps (M_PW / gradient) **per grid cell**, take best NCC peak per cell as correspondences → feed existing `miho_plus_gcp` 6×6 GCP optimizer.
  - Evidence: `backend/lunar_registration/structural.py` + Stage 4.5 last-resort arm in `pipeline.py` (fires **only** when no arm carries a transform); every run reports `summary["structural_correspondences"]`. `test_structural.py`, `test_structural_pipeline.py`.
- [x] Bypasses the ratio-test failure entirely (keypoint matchers are flat at 4–10 matches across all rotations).
- Rationale: cross-image structural NCC as *global aligner* is capped (exhaustive search of same objective: max **struct_ncc 0.317 < 0.40**, 26,496 evals, broad ambiguous plateau) — but as a *local correspondence source* it is untested and is the highest-upside open path.
- Files: hooks — `verify.py compute_structural_ncc` (reuse math), `pipeline.py` Stage 5 `miho_plus_gcp` ~547-548.
- Success bar: ≥50 correspondences with ≥0.2 NCC and spread coverage; then gate score ≥0.25. → **met on synthetic controls** (≥50 kept with NCC ≥0.2 in `test_structural.py`); **not met on the canonical pair**: `n=0` kept after a ±180° sweep (peaks plateau at NCC 0.10–0.15 over repeating craters).
- [x] Failure bar: if peaks are ambiguous (repeating craters), declare pair un-registrable at demo grade and pick an easier demo pair.
  - Declared: the canonical pair is un-registrable at demo grade (ambiguous structural peaks + zero usable neural matches); the pipeline reports that honestly instead of passing an unverified transform.

---

## Explicitly deferred (wrong-layer for demo; do not start now)

- [ ] DEM-based orthorectification (affine-over-homography without DEM is incomplete).
- [ ] CRS normalization between products (different CRS corner tiepoints in degrees).
- [ ] Per-pixel photometric modeling (beyond the normalization of TODO-6).
- [ ] Stale tool fixes: `backend/diagnose.py` broken three ways (l.25 `PWIFTMaps.shape`, l.55 `pwift_keypoint_threshold` attr, l.60 hardcoded ref `LROC`) — fix only if still referenced.

---

## Suggested execution order

1. TODO-1 + TODO-2 (metadata + fail-closed) — unblocks honest evaluation of everything else.
2. TODO-3 + TODO-4 (gate logic + thresholds) — otherwise correct transforms still fail.
3. TODO-6 + TODO-7 (normalization + fusion) — measured wins.
4. TODO-5 (API/pins) — parallel-safe.
5. TODO-8 (experiment) — go/no-go on registering this pair vs swapping the demo pair.

## Verification bar (whole pipeline, after all fixes)

- [x] Canonical pair run: exits **non-zero** with honest reason, OR registers with ≥50 inliers, `struct_ncc ≥ 0.25`, gate `pass:true` for a real (non-disabled) reason.
  - **Exits non-zero (option 1):** `passed:false`, `failure_reason: no_transform`, gate `skipped: no transform`, `outputs {}`, `SystemExit(1)` (asserted by `test_cli_main_exits_nonzero_when_registration_failed`).
- [x] Synthetic control (known 11° rotation): recovers rotation ±0.5°, passes gate.
  - `backend/tests/test_verification_bar.py`: control pair recovers **0.195°** error (bar 0.5°) with `gate.reason == "all_checks_passed"`, `struct_ncc 0.71`; measurement machinery validated separately (held-out affine fit < 0.1px, scipy sign convention, OpenCV `warpAffine` convention).
- [x] No run ever exits 0 with `passed:false` / `reason:"disabled"` / flagged confidence.
  - `backend/tests/test_fail_closed.py` (non-zero exit on failure, exit-0 only on a real pass, `disabled`-reason false pass rejected).
