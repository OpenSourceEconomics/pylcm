# Pro round 0 on PR #486: ingested, F1 and F2 fixed

The parent saved this from the subagent's hand-back. It is condensed but complete in substance. Evidence is in this directory (`SHA256SUMS` over 86 files). The ingestion output is in `~/sciebo/pro-audits/pylcm-pr486/bundle-round0/ingested/`.

## Ingestion
- **Protocol:** 3.2.30, with a consistent shared core.
- **`check_ingestion`:** exit 0, agreeing with `audit-result.json` (393c896b…).
- **Verdict:** `serious_gap` / `implementation_required`.
- **Remaining warning:** F1 lacks a benchmark command. This is immaterial, because F1 closes through RT1, not a benchmark.

## Pro's decisions
- **Helper fold:** neither option. Keep both cold per-shape helpers. The strict-xfail exact-count test stays as a later aspiration.
- **Warm cache:** cache immutable structural blueprints. The key covers sealed program identity, schema, dtype, weak typing, layout, view structure, and compiler and policy fields. Params, grids, operands, owners, liveness, donation and admission are rebound on every solve.
  - This is a separate, measured architecture change that comes after P1 and P2.
  - It needs a structural/binding split in `materialize_core_program`, `_prepare_abstract_program` and `_resolve_output_layouts_and_lowering_keys`.
  - Deferred; it needs the user's go. RT3 (6/6 red at p32; warm calls rebuild 8 or 20 programs) is in the subagent scratchpad and is not committed.

## Findings
| ID | Type | Ownership | Disposition |
|---|---|---|---|
| F1: an aligned selected view was credited as its stored owner in 4 admission predicates | RESOURCE | target (base has no views) | Fixed in 96774125 (P1: predicates test `delivers_stored_buffer`; adds `test_invariant_selected_view_admission.py`) |
| F2: a blocked core with scalar local state and action products was marked DENSE and skipped selected-read planning, giving silently wrong values: (5,6,7) against the Bellman (3,6,9) | SEMANTIC | target | Fixed in 9b36adcb (P2; adds `test_invariant_blocking_scalar.py`; two `direct_flow.py` mutation markers re-pointed at `requires_plan`) |
| R3: blueprint cache | RESOURCE | target-related | Deferred |

No ISSUE_AND_EXIT was proposed.

## Identity
- `6ede6e6b..9b36adcb`: 7 files, +522/−36. Diff sha256 `29d1fd43…c743`, which the parent re-verified.
- Pushed as 9b36adcbb0a7801f0a74ec6f9b3966333b9ffe18.

## Red, green and battery (at 9b36adcb, serial under `cap`; every run exit 0)
- **Red at 6ede6e6b:** 17/17 fail at p32 and at p64.
- **Green:** F1 8/8 and F2 9/9 at both precisions.
- **Blocking files:** 56 tests at p64 and at p32, 4 skipped each (strict xfails).
- **8-device runs:**
  - sharding: 18 tests at p64 and at p32;
  - transfer admission: 3 tests at p64 and at p32.
- **`tests/execution` at p64:** 1560 tests, 27 topology skips.
- **Scheduler, liveness, donation and footprint set:** 165 tests.
- **Distributed / placement / lifetime:** 51 / 72 / 23 tests.
- **Certificate files:** 102 tests.
- **`verify.py --self-test`:** 1148 of 1148 mutations rejected.
- **Certificates:** repin, seals, anchors, verify and `ciw --check` all exit 0 (`cert-f1/`, `cert-f2b/`). The first F2 pass failed on anchors, which led to the marker re-point.
