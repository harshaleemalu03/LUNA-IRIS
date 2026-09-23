# Task List — LUNA-IRiS Pipeline Repair (scope-locked: no matcher changes)

## Phase 0 — Truthful execution
- [x] Task 1: API accepts crop windows + size-guard fallback (complete: 923bcad, 3 tests)
  - AC: `POST /api/register` accepts `source_window`/`reference_window`; guard at `pipeline.py:221-228` either receives a window or auto-tiles to ≤4 MP with a logged warning
  - Verify: Run 0 scenario (full 16 MP reference, no window) no longer crashes
  - Files: `backend/app.py:95-100`, `backend/lunar_registration/pipeline.py:221-228`
  - Deps: none | Scope: S
- [x] Task 2: Metadata reader — GeoTIFF tags + sidecar XML (complete: 5 tests; real pair ratio 1.0067, incidence 84.896724)
  - AC: reads `ModelPixelScale` / corner tiepoints / CRS and sidecar XML (`Solar_incidence_angle`, `Sun_elevation`, `Sun_azimuth`) into `LoadedImage`; missing fields degrade to placeholder **with a log naming the field**
  - Verify: canonical pair reports GSD ratio ≈ 1.006 and incidence ≈ 84.896724
  - Files: `backend/lunar_registration/preprocessing.py` (new reader near l.369 patterns)
  - Deps: none | Scope: M
- [x] Task 3: Wire metadata → scale prior, routing incidence, sensor conditioning
  - AC: `estimate_gsd_scale_prior` receives computed ratio (prior band [0.855, 1.157] contains truth); routing receives XML/manual incidence so canonical pair resolves `hybrid_pwift_roma2` / `polar_grazing` with no CLI flags; placeholder defaults (`scale.py` 0.5, `pipeline.py:95` 30.0) only used when Task 2 logged a missing field
  - Verify: Run 1 (no flags) now matches Run 2's regime/reason; Run exp-C behavior (chosen scale inside correct band) reproduced automatically
  - Files: `backend/lunar_registration/scale.py:1858`, `backend/lunar_registration/pipeline.py:244-257`
  - Deps: 2 | Scope: S
- [x] Task 4: Fail-closed verification + exit codes
  - AC: (a) `primary_H is None` → `passed: false`, nonzero exit, never emits valid-looking outputs; (b) gate FAIL → nonzero exit, outputs marked invalid; (c) "Proceeding with flagged confidence" removed; (d) `low_precision` surfaced in API response
  - Verify: Runs 2/3/3c scenario exits nonzero; a passing run exits 0 only when gate passed
  - Files: `backend/lunar_registration/pipeline.py:492,506,542-544`, `backend/app.py`
  - Deps: none (but coordinate with frontend response handling) | Scope: M
- [x] Task 5: RoMa fine-tune-only loading + kornia pin
  - AC: (a) `RoMaV2.__init__` no longer unconditionally downloads/loads base `romav2.0.1.pt` (`romav2.py:98-101,113`) — constructor accepts weights from the caller; (b) `matching.py:428-430` loads **only** the fine-tune from `ROMA2_WEIGHTS_PATH`, upgraded to `strict=True`; (c) cold start performs **no GitHub fetch**, base ckpt file not required anywhere on disk; (d) `kornia==0.6.8` pinned explicitly in requirements
  - Verify: unit test asserts `load_state_dict(strict=True)` passes with `ckpt["model"]` (907 keys); pipeline run with `ROMA2_WEIGHTS_PATH` set succeeds in an empty torch cache dir; `grep -r "romav2.0.1" backend/` shows no remaining runtime dependency
  - Files: `backend/third_party/RoMaV2/src/romav2/romav2.py:98-113`, `backend/lunar_registration/matching.py:425-430`, requirements
  - Deps: none | Scope: S

### Checkpoint: Phase 0
- [x] Canonical pair via API: runs, exits nonzero, `passed: false`, honest reason (HTTP 422 + structured detail; server alive after)
- [x] Easy pair: exits 0 only with gate-passed transform (safety property proven; negative finding: NO real zip pair registers — IIRS least-extreme pair fails gate honestly at 52.9px; exit-0 path covered by synthetic gate-pass e2e + CLI unit test; see ledger)
- [x] Prior ≈ 1.006, regime `polar_grazing`, no CLI flags (CLI run: prior 1.006682, polar_grazing → hybrid_pwift_roma2, zero flags)
- [ ] Human review before Phase 1

## Phase 1 — Correctness rot
- [x] Task 6: Peak-quality acceptance in scale/rotation search (complete: 2b07368, 9 tests; RED ImportError -> GREEN 9 passed; suite 62 passed)
  - AC: boundary picks (0.575 / −180 / 3.0 / 15° observed) return "no confident alignment" instead of argmax; interior peak required
  - Verify: unit test with flat/corrupt similarity surface → returns None/no-confidence; good pair → returns interior peak
  - Files: `backend/lunar_registration/scale.py:43-91`, `coarse_to_fine_rotation_scale`
  - Deps: 3 | Scope: S
- [x] Task 7: Akimov guard
  - AC: enabled-without-both-angles → warn + `W≡1` documented; `w_soft` collapsing to all-zero → hard warning + fall back to unweighted path (prevents exp-B annihilation)
  - Verify: forced inc/em run no longer yields 0 source keypoints silently
  - Files: `backend/lunar_registration/pwift.py:155-196,252-255`, `config.py:87`
  - Deps: none | Scope: S
- [x] Task 8: Subpixel-refine honesty
  - AC: matcher-independent identical dx/dy (123.5054…/271.0260…) detected as degenerate → `low_precision: true` with reason, surfaced in summary/API
  - Verify: two different matchers on same pair → flag raised when phase-correlation peak weak or values identical
  - Files: `backend/lunar_registration/pipeline.py` (refine ~l.504) + summary builder
  - Deps: 4 (surfacing) | Scope: S
- [ ] Task 9: Repair `diagnose.py`
  - AC: fixes `pwift_keypoint_threshold` (l.55), `PWIFTMaps.shape` (l.25), hardcoded `sensor_hint="LROC"` (l.60); supports **per-image** windows (shared window cannot express this pair); documented memory note (full-image OOM, exit 137)
  - Verify: `diagnose.py` runs windowed on canonical pair without crashing; prints correct sensor (OHRC)
  - Files: `backend/diagnose.py`
  - Deps: 2 (per-image windows helpers) | Scope: S

### Checkpoint: Phase 1
- [ ] Phase 0 pair set re-run, no regressions
- [ ] `diagnose.py` windowed run clean

## Phase 2 — Cleanup
- [ ] Task 10: Wire or delete `resample_to_gsd` (`scale.py:1805`) — dead either way today
- [ ] Task 11: Contingency-fallback logging — `triggered` reports truthfully when winning arm has 0 inliers
- [ ] Task 12 (stretch): Tiepoint-derived coarse transform seeding the existing estimator (metadata wiring only; needs user go/no-go)
