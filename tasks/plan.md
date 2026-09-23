# Implementation Plan: LUNA-IRiS Pipeline Repair (Scope-Locked)

## Overview
Fix the broken wiring of the existing registration pipeline so it runs the canonical
OHRC pair through the API, reads its metadata, routes correctly, and reports failure
honestly. **Scope lock: no matcher-family changes.** PWIFT, RoMaV2, EloFTR, and the
hybrid arms stay exactly as they are; every task is wiring, guard, or truth-telling.

## Architecture Decisions
- **Metadata is the single source of truth**: GSD prior and routing incidence come from
  GeoTIFF tags + sidecar XML, never from placeholders (`scale.py` 0.5 default,
  `pipeline.py:95` 30.0 default are the defects).
- **Verification fails closed**: no transform → failure; gate FAIL → nonzero exit and
  invalid outputs. Exit code is the demo's contract.
- **Akimov branch stays disabled for the demo** (inert by default, zeroing when open —
  exp-B); we only add a guard so it cannot silently annihilate all keypoints if enabled.
- **RoMa uses the fine-tuned checkpoint only** (`ROMA2_WEIGHTS_PATH`); the base
  `romav2.0.1.pt` is removed from the load path entirely — no download, no cache file.
  Justification: §4.8 proof (architecture ≡ base ≡ fine-tune key sets, strict-load
  transitively holds) means the constructor's base load at `romav2.py:98-113` only
  supplies values that `matching.py:430` then overwrites 100%. The vendored
  `third_party/RoMaV2` copy is parameterized so the caller passes the fine-tune state
  dict directly; overlay becomes a single `strict=True` load.
- Expected outcome is stated up front: at 84.9° incidence the canonical pair will still
  not match (report §7b — every matcher family fails even at truth geometry). After this
  work it must **fail truthfully** (nonzero exit, no `passed: true`, no GCPs claimed)
  while pairs within matcher capability register end-to-end.

## Task List

### Phase 0 — Truthful execution (demo-blocking)

## Task 1: API accepts crop windows + size-guard fallback
- [ ] `app.py`, `pipeline.py` — POST /api/register accepts `source_window`/`reference_window`; guard at `pipeline.py:221-228` receives a window or auto-tiling fallback to ≤4 MP with logged warning

## Task 2: Metadata reader — GeoTIFF tags + sidecar XML
- [ ] `preprocessing.py` (new reader) — reads ModelPixelScale / corner tiepoints / CRS + sidecar XML (Solar_incidence_angle, Sun_elevation, Sun_azimuth) into LoadedImage; missing fields degrade to placeholder with a log naming the field

## Task 3: Wire metadata → scale prior, routing incidence, sensor conditioning
- [ ] `scale.py:1858`, `pipeline.py:244-257` — computed GSD ratio feeds estimate_gsd_scale_prior; XML incidence feeds routing so canonical pair resolves hybrid_pwift_roma2 / polar_grazing with no CLI flags; placeholders only used when Task 2 logged a missing field. Deps: Task 2

## Task 4: Fail-closed verification + exit codes
- [ ] `pipeline.py:492,506,542-544`, `app.py` — (a) primary_H None → passed:false, nonzero exit, no valid-looking outputs; (b) gate FAIL → nonzero exit, outputs invalid; (c) "Proceeding with flagged confidence" removed; (d) low_precision surfaced in API response

## Task 5: RoMa fine-tune-only loading + kornia pin
- [ ] `third_party/RoMaV2/src/romav2/romav2.py:98-113`, `matching.py:425-430`, requirements — constructor accepts weights from caller (no unconditional base download); single strict=True load of ROMA2_WEIGHTS_PATH; no GitHub fetch on cold start; kornia==0.6.8 pinned

### Checkpoint: Phase 0
- [ ] Canonical pair runs through `POST /api/register` without crashing
- [ ] Canonical pair exits **nonzero** with `passed: false`, 0 GCPs, reason surfaced
- [ ] An easy pair (low Δillumination) registers end-to-end and exits 0 only if gate passed
- [ ] Scale prior ≈ 1.006 and routing regime `polar_grazing` on the canonical pair

### Phase 1 — Correctness rot

## Task 6: Peak-quality acceptance for scale/rotation search
- [ ] `scale.py:43-91`, `coarse_to_fine_rotation_scale` — boundary picks (0.575 / −180 / 3.0 / 15°) return "no confident alignment" instead of argmax; interior peak required. Deps: Task 3

## Task 7: Akimov guard
- [ ] `pwift.py:155-196,252-255`, `config.py:87` — enabled-without-both-angles → warn + documented W≡1; w_soft collapsing to all-zero → hard warning + fall back to unweighted path (prevents exp-B annihilation)

## Task 8: Subpixel-refine honesty
- [ ] `pipeline.py` refine ~l.504 + summary — matcher-independent identical dx/dy detected as degenerate → low_precision:true with reason, surfaced in summary/API. Deps: Task 4

## Task 9: Repair diagnose.py
- [x] `diagnose.py` — fix pwift_keypoint_threshold (l.55), PWIFTMaps.shape (l.25), hardcoded sensor_hint="LROC" (l.60); support per-image windows; document full-image OOM. Deps: Task 2 — done (commit f701c4c; windowed run EXIT=0 on real IIRS pair)

### Checkpoint: Phase 1
- [x] Phase 0 pair set re-run, no regressions — canonical CLI exit=1 passed=false `no_transform` (was gate-362.9px; class moved by Task 5/7 arm changes, contract intact: prior=1.0066821 polar_grazing/hybrid_pwift_roma2 outputs={} summary-only), canonical API HTTP=422 structured (routing/prior/subpixel_refine_reason present, health 200 before+after), IIRS easy pair exit=1 passed=false `no_confident_alignment` (Task 6 honest class; routing subpixel_cartography/roma2 auto, no flags)
- [x] `diagnose.py` windowed run clean — real IIRS pair, EXIT=0 all stages + honest diagnosis (log: /tmp/opencode/iris_runs/phase1_diagnose_windowed.log)

### Phase 2 — Cleanup (post-demo, out of scope for this run — see ledger scope ruling)

## Task 10: Wire or delete dead resample_to_gsd
- [ ] `scale.py:1805` — dead either way today

## Task 11: Contingency-fallback logging truthfully
- [ ] `triggered` must report truthfully when winning arm has 0 inliers

## Task 12: Tiepoint-derived coarse transform seeding the existing estimator (stretch, needs user go/no-go)
- [ ] metadata wiring only, no new matcher

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Fail-closed exit codes break frontend expectations | Med | Frontend already modified in working tree; surface `passed`/`reason` in response body first, adjust `app.js` display |
| Metadata reader hits unexpected tag layouts across pairs | Med | Reader must degrade gracefully: missing tag → keep placeholder + log which field was missing |
| Canonical pair "failure after fix" looks like a regression to demo audience | Med | Failure report must be explicit: reason string, gate metrics, `low_precision` flag |
| Editing vendored `third_party/RoMaV2` drifts from upstream | Low | Change is additive (optional `weights` param, default preserves current behavior); keep patch minimal and commented |
| Fine-tune strict load fails on unexpected key drift | Low | §4.8 proof: fine-tune key set == base key set (empty diff both ways) and base strict-loads at `romav2.py:113`; assert with a unit test that `strict=True` passes on the actual ckpt |

## Open Questions
- Should Task 12 (tiepoint init) ship before the demo or after? — user decision.
- Which pair is the designated "easy pair" for Checkpoint Phase 0? — pick from eval zip after Task 2 lands (metadata makes Δillumination computable).

## Evidence Ledger (from verification report, repo-re-verified this session)
Guard `pipeline.py:228`; gate `:492`; "Proceeding with flagged confidence" `:506`;
routing default 30.0 `:95`; Akimov gate `pwift.py:252-255`; prior placeholder `scale.py:1858`.
Only frontend files are dirty in the working tree; backend matches the report's citations.
