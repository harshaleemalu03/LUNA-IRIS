# Implementation Plan: `TODOs.md` backlog (Tier 1–3) + Docker packaging

## Overview

Execute the remaining items of `TODOs.md` (the standalone backlog written 2026-09-24), which the
scope-locked `tasks/plan.md` repair deliberately left alone, and package the system with a Docker
Compose stack. Concretely: make the verification gate rotation-aware (TODO-3), recalibrate the
structural-NCC threshold to the measured band (TODO-4), add photometric normalization in front of the
neural matchers (TODO-6), route neural arms through the existing fusion/competition layer with the
contingency fallback surfaced in the API (TODO-7), add the dense structural-NCC correspondence
generator as a last-resort arm (TODO-8), pin the runtime dependencies (TODO-5 remainder), and ship a
`docker-compose.yml` that runs API + frontend.

`tasks/plan.md` / `tasks/todo.md` from the previous (complete) plan are left untouched; this plan
lives in `tasks/plan-todos.md` with its task list in `tasks/todo-todos.md`.

## Architecture Decisions

- **Gate check #1 measures residual alignment, not translation equality.** `_coarse_phase_align_256`
  is replaced by a phase-residual measurement taken *after* warping source into the reference frame
  with the candidate homography. Both images are first mapped into a common 256×256 frame with the
  similarity `S = diag(256/w, 256/h)` per image, so the warp is dimensionally correct for unequal
  image sizes and costs one 256² warp instead of a full-resolution one. This is the "rotate/warp img0
  by H before phase correlation" option sanctioned by TODO-3.
- **Normalization lives in a new `photometric.py`, not `illumination.py`.** `illumination.py` is the
  sensor-branched Stage-2 representation builder (returns `PWIFTMaps`); matcher-input normalization
  is a different concern with different consumers, so it gets its own module with one public
  function.
- **Structural correspondences live in a new `structural.py`** and return a normal `MatchResult`, so
  they enter Stage 4 through the *same* estimation path as any matcher arm. The prior is a
  translation obtained by phase correlation inside the module — by the time Stage 3/4 run, Stage 3
  has already applied the coarse scale and rotation to the source (`src_scaled`), so translation is
  the only unknown left. The pipeline hook stays a single call.
- **Neural arms always compete against PWIFT.** `matcher=roma2|eloftr` currently short-circuits to a
  single arm, so `compete_rigid` never has anything to arbitrate. Neural resolution now also runs
  PWIFT and fuses (the existing hybrid path); `matcher=pwift` stays single-arm by construction.
  Contingency semantics are unchanged — a PWIFT failure still degrades to the neural arm alone.
- **One config surface.** All new knobs are `PipelineConfig` fields (`config.py`) so CLI, API and
  tests share one place; no module-level constants for tunables.
- **Fail-closed contract is preserved.** Every new arm/threshold only ever *adds* a chance to
  produce a transform; it never converts a failed gate into a pass, and a run that still fails exits
  non-zero / returns 422 exactly as before.

## Task List

> **Status: complete (2026-09-25).** Final suite `151 passed, 1 skipped` (baseline 105). Branch
> `feat/todos-backlog`. Where a `TODOs.md` acceptance criterion could not be evidenced in this
> environment it is annotated as not-evidenced in `TODOs.md` rather than ticked.

### Phase 0 — Baseline
- [x] Task 1: Restore a working dev environment and record the regression baseline
  - AC: `.venv` (py3.12) installs `requirements.txt` + test deps; `pytest backend/tests -q` → **105 passed, 1 skipped** (was 1 failed after a bare install: missing `pytest/yacs/loguru/joblib/pytorch-lightning`).
  - Verify: `.venv/bin/python -m pytest backend/tests -q`
  - Files: `requirements-dev.txt` (new, records the test-only deps)

### Phase 1 — Verification gate correctness (TODO-3, TODO-4)
- [x] Task 2: Rotation/scale-aware gate check #1
- [x] Task 3: Recalibrate `orthogonal_gate_t_struct` to the measured band

### Checkpoint: Phase 1
- [x] `pytest backend/tests/test_gate_rotation.py backend/tests/test_verify.py -q` green → **11 passed**
- [x] Full suite green, no fail-closed regression (every previously failing run still fails)

### Phase 2 — Isolated modules (TODO-6, TODO-8 foundations)
- [x] Task 4: `photometric.py` — matcher-input normalization (gradient / weber / clahe / none)
- [x] Task 5: `structural.py` — dense structural-NCC correspondence generator

### Checkpoint: Phase 2
- [x] New module tests green (`test_photometric.py` + `test_structural.py` → **23 passed**); full suite still green

### Phase 3 — Pipeline integration
- [x] Task 6: Normalize neural matcher inputs in Stage 3
- [x] Task 7: Route neural arms through fusion + competition, surface contingency in the API
- [x] Task 8: Structural-NCC arm as last resort in Stage 4

### Checkpoint: Phase 3
- [x] Full suite green
- [x] Canonical + easy real pair: fail classes unchanged (honest failure), structural arm reported in summary when it fires
  - Canonical run: `passed:false`, `failure_reason: no_transform`, exit **1**, `structural_correspondences {attempted:true, n:0, used:false}` in `summary.json`.

### Phase 4 — Packaging
- [x] Task 9: Pin runtime dependencies (TODO-5 remainder)
- [x] Task 10: Dockerfile + `docker-compose.yml` + `.dockerignore`

### Checkpoint: Complete
- [x] `docker compose config` valid, stack builds, `/api/health` OK, frontend served, one registration request answered
  - `docker compose config -q` clean; API on host **7860** (host 8000 belongs to `searxng-core`), frontend on **5173** with nginx proxying `/api` + `/outputs`; `POST localhost:5173/api/register` → HTTP 200, `passed:true`, `orthogonal_gate_passed:true`.
- [x] `TODOs.md` checkboxes updated to reflect delivered items (deferred section untouched; not-evidenced criteria annotated inline)
- [x] Full suite green (`151 passed, 1 skipped`); everything committed on a feature branch (`feat/todos-backlog`, not pushed)

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Rotation-aware check changes gate behaviour for existing runs | Med | Fail-closed contract is unchanged; tests assert both directions (correct H passes, wrong H fails) |
| Structural arm fires on runs that used to fail honestly | Med | Arm is only attempted when no arm produced a transform; it still passes through RANSAC + the same gate |
| Normalization hurts the matchers it is meant to help | Med | Config-selectable (`none` = opt out); module-level tests prove the intended contrast/illumination invariance |
| Disk headroom for the Docker image build (~6 GB free) | High | `.dockerignore` excludes data/zip/cache; build CPU base image only; reclaim stale build cache if needed |
| Real-pair weights live outside the repo (hardcoded host fallbacks) | Med | `LUNATICS_MODELS_DIR` / `ROMA2_WEIGHTS_PATH` env documented and wired in compose as a volume |

## Open Questions
- None blocking: TODO-7's "single-arm runs bypass real competition" is resolved by always pairing a
  neural arm with PWIFT (documented above); `matcher=pwift` remains the single-arm mode.
