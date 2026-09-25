# Task List — TODOs.md backlog (Tier 1–3) + Docker

Plan: `tasks/plan-todos.md`. Source list: `TODOs.md`. Each task is bounded, has acceptance criteria
and a verification command; a task is done only when its verification output is shown.

## Phase 0 — Baseline

### Task 1: Restore dev environment + regression baseline — DONE
- [x] AC: `.venv` (py3.12) with `requirements.txt` + test deps installed; baseline recorded
- [x] Verify: `.venv/bin/python -m pytest backend/tests -q` → **105 passed, 1 skipped** (after adding `pytest`, `yacs`, `loguru`, `joblib`, `pytorch-lightning`)
- Deps: none

## Phase 1 — Gate correctness (TODO-3, TODO-4)

### Task 2: Rotation/scale-aware gate check #1
**Description:** Replace the translation-only phase-agreement check with a *residual* phase check:
map source and reference into a common 256×256 frame, warp the source with the candidate homography,
and require the residual phase-correlation shift to be within `tau_agree`. A pair rotated ~9–11° can
now pass when H is correct (today it reports ~475px disagreement and can never pass).

AC:
- [ ] `gate_cheap` reports `phase_agree_px` in its result dict
- [ ] Synthetic control pair (known 11° rotation + known translation), H = ground truth → `phase_agree_px <= tau_agree`
- [ ] Same pair, H with correct rotation but translation off by 100px → `phase_agree_px > tau_agree`
- [ ] Unequal image sizes (e.g. 300×400 vs 512×512) handled without shape errors

Verify: `.venv/bin/python -m pytest backend/tests/test_gate_rotation.py backend/tests/test_verify.py -q`
Deps: none | Files: `backend/lunar_registration/verify.py`, `backend/tests/test_gate_rotation.py` (new) | Scope: S

### Task 3: Recalibrate `orthogonal_gate_t_struct`
Description: `0.40` sits above what this pair class can achieve (measured: identity 0.024, garbage-H
0.040, exhaustive-search best 0.317, same-content control ceiling 0.474). Lower to `0.25`.

AC:
- [ ] `PipelineConfig.orthogonal_gate_t_struct == 0.25` with the measured band documented in the comment
- [ ] `gate_cheap` default matches the config value
- [ ] Correctly-aligned synthetic control with `struct_ncc` in (0.25, 0.40) passes check #3
- [ ] Identity / garbage transform (`struct_ncc ≈ 0.02–0.04`) still fails check #3

Verify: `.venv/bin/python -m pytest backend/tests/test_gate_rotation.py backend/tests/test_verify.py -q`
Deps: none | Files: `backend/lunar_registration/config.py`, `verify.py` | Scope: XS

### Checkpoint: Phase 1
- [ ] Full suite: `.venv/bin/python -m pytest backend/tests -q` → no failures

## Phase 2 — Isolated modules (TODO-6, TODO-8 foundations)

### Task 4: `photometric.py` — matcher-input normalization
Description: One public function, `normalize_for_matching(img, mode)`, implementing the three modes
measured on the canonical pair (gradient-domain Sobel ÷ p99.5, Weber local contrast, CLAHE) plus
`"none"`. Pure module — no pipeline wiring in this task.

AC:
- [ ] Modes `gradient` / `weber` / `clahe` / `none` all return `float32`, same shape, all-finite, without mutating the input
- [ ] `none` returns an unchanged copy
- [ ] Illumination-shift invariance: for two images related by a gamma/exposure change, the normalized pair has higher mutual NCC than the raw pair (gradient mode)
- [ ] `PipelineConfig.neural_input_normalization` selects the mode, default `"gradient"`

Verify: `.venv/bin/python -m pytest backend/tests/test_photometric.py -q`
Deps: 1 | Files: `backend/lunar_registration/photometric.py` (new), `config.py`, `backend/tests/test_photometric.py` (new) | Scope: S

### Task 5: `structural.py` — dense structural-NCC correspondence generator
Description: `structural_ncc_correspondences(src_img, ref_img, scale, rotation_deg, grid, search_radius_px, min_ncc) -> MatchResult`.
Builds an affine prior (scale + rotation from the coarse search, translation by phase correlation
in the rotated/scaled frame), warps the source into the reference frame, then takes the best NCC
peak per grid cell via `cv2.matchTemplate`, mapping each peak back to source coordinates.

AC:
- [ ] On a synthetic pair with known H (≈11° rotation, translation, scale ≈1): ≥ 50 correspondences with NCC ≥ 0.2 and ≥ 50% grid-cell coverage
- [ ] Round-trip accuracy: RANSAC on the returned points recovers H with median reprojection error < 5px
- [ ] Featureless (flat) input yields < `min_ncc` peaks → empty `MatchResult` (caller discards), no exception
- [ ] All public parameters come from `PipelineConfig` (`structural_fallback_enabled`, `structural_grid`, `structural_min_ncc`, `structural_search_radius_px`, `structural_min_correspondences`)

Verify: `.venv/bin/python -m pytest backend/tests/test_structural.py -q`
Deps: 1 | Files: `backend/lunar_registration/structural.py` (new), `config.py`, `backend/tests/test_structural.py` (new) | Scope: M

### Checkpoint: Phase 2
- [ ] Full suite: `.venv/bin/python -m pytest backend/tests -q` → no failures

## Phase 3 — Pipeline integration

### Task 6: Normalize neural matcher inputs (Stage 3)
Description: In `run_pipeline`, hand normalized copies of `src_scaled` / `ref.data` to neural
matchers (`roma2`, `eloftr`, and their hybrid arms) when `cfg.neural_input_normalization != "none"`.
The PWIFT arm keeps the raw images — it consumes the Stage-2 illumination maps built from raw input.

AC:
- [ ] A recording fake neural matcher receives images that differ from the raw inputs when mode is `gradient`
- [ ] The same fake matcher receives the raw inputs when mode is `none`
- [ ] The PWIFT arm always receives the raw images
- [ ] Full suite green (no behavioural regression on existing e2e tests)

Verify: `.venv/bin/python -m pytest backend/tests/test_photometric_pipeline.py -q` + full suite
Deps: 4 | Files: `backend/lunar_registration/pipeline.py`, `backend/tests/test_photometric_pipeline.py` (new) | Scope: S

### Task 7: Route neural arms through fusion + competition; surface contingency in the API
Description: `matcher=roma2|eloftr` must not bypass `fuse_pwift_neural` + `compete_rigid`: run PWIFT
alongside and evaluate `[fused, neural, pwift]`, reusing the existing hybrid semantics (including
contingency degradation if PWIFT throws). Surface `contingency_fallback` in both API responses.

AC:
- [ ] Resolving to a plain neural matcher yields ≥ 2 candidate arms in `compete_rigid` (fused + neural, plus pwift)
- [ ] PWIFT failure during a neural run still degrades to the neural arm with `contingency_fallback.triggered == True`
- [ ] `POST /api/register` 422 detail **and** 200 response both carry `contingency_fallback`
- [ ] `matcher=pwift` unchanged (single arm, no fusion)

Verify: `.venv/bin/python -m pytest backend/tests/test_fusion_routing.py backend/tests/test_api_windows.py backend/tests/test_fail_closed.py -q`
Deps: none | Files: `backend/lunar_registration/pipeline.py`, `backend/api/app.py`, `backend/tests/test_fusion_routing.py` (new) | Scope: M

### Task 8: Structural-NCC arm as last resort (Stage 4)
Description: After the matcher arms have been evaluated, if **no** arm produced a usable transform,
generate structural correspondences and run them through the same estimation path (RANSAC, metrics,
reprojection cleanup, competition, gate). Report the outcome in `summary.json`.

AC:
- [ ] Arm is attempted only when every matcher arm produced `H is None` (or no arm had ≥ 4 points) — never on a run that already has a transform
- [ ] On a synthetic pair with a sabotaged matcher (0 usable points), the run produces a transform from structural correspondences and reaches the gate
- [ ] `summary["structural_correspondences"]` reports `attempted / n_correspondences / used / coverage`
- [ ] If the arm still yields no transform, the run fails closed exactly as before
- [ ] Full suite green

Verify: `.venv/bin/python -m pytest backend/tests/test_structural_pipeline.py -q` + full suite
Deps: 5 | Files: `backend/lunar_registration/pipeline.py`, `backend/tests/test_structural_pipeline.py` (new) | Scope: M

### Checkpoint: Phase 3
- [ ] Full suite: `.venv/bin/python -m pytest backend/tests -q` → no failures
- [ ] Real canonical pair still fails honestly (exit 1, `passed: false`) or registers with a gate-passed transform — never exit 0 with `passed: false`

## Phase 4 — Packaging

### Task 9: Pin runtime dependencies (TODO-5 remainder)
Description: `requirements.txt` currently floats `torch`, `torchvision`, `opencv-*`, `transformers`.
Pin every package that the verified environment resolved so a cold install reproduces the tested stack.

AC:
- [ ] Every entry in `requirements.txt` is pinned `==` (including `torch`, `torchvision`, `opencv-python`, `opencv-python-headless`, `transformers`, `fastapi`, `uvicorn`, `rasterio`, `pillow`, `scipy`, `huggingface_hub`, `einops`, `requests`)
- [ ] Test-only deps recorded in `requirements-dev.txt`
- [ ] `uv pip install --python .venv/bin/python -r requirements.txt` reports nothing new to install

Verify: `.venv/bin/python -m pip check` (via `uv pip check`) + install dry-run
Deps: 1 | Files: `requirements.txt`, `backend/requirements.txt`, `requirements-dev.txt` (new) | Scope: S

### Task 10: Docker Compose stack
Description: Two services — `api` (FastAPI via `uvicorn api.app:app`) and `frontend` (nginx serving
`frontend/`), with a shared volume for outputs/uploads and an optional models mount.

AC:
- [ ] `docker compose config` validates
- [ ] `docker compose up --build -d` → `GET :8000/api/health` returns `{"status": "ok"}`
- [ ] `GET :5173/` (or mapped frontend port) serves the UI; UI talks to the API through a configurable base URL
- [ ] `POST /api/register` with two small synthetic images returns a structured response (200 or fail-closed 422) — never a 500
- [ ] Outputs persist on a named volume; `.dockerignore` keeps data/zip, `.venv`, caches and `__pycache__` out of the build context

Verify: `docker compose config -q && docker compose up --build -d && curl -fsS localhost:8000/api/health`
Deps: 9 | Files: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `frontend/Dockerfile` or nginx conf | Scope: M

### Checkpoint: Complete
- [ ] Full suite green after all changes
- [ ] `TODOs.md` checkboxes updated (delivered items marked, deferred items untouched)
- [ ] Work committed on a feature branch (never directly on `main`)
