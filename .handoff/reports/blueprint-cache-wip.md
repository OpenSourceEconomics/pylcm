# Blueprint cache (Pro R3): interim, stopped on 2026-10-02

The local subagent's hand-back, saved by the parent.

The work is local commits on `perf/invariant-structural-blueprint-cache`, on top of 9b36adcb:
- 922651c2: WIP Cache immutable structural blueprints across warm GridSearch solves
- 2a9cdad6: WIP Apply formatter and keyword-only fixes to the blueprint cache

The diff is 4 files, +773/−6, sha256 `ffc308a4…dfddc79`. Both commits are exported as `../wip/blueprint-cache/*.patch`; apply them with `git am` on 9b36adcb. They were not pushed because the pre-push certificate hook fails: about 20 corridors on `backward_induction.py` and `lcm/model.py` need repinning. The pre-commit hooks were skipped for both WIP commits.

## The structure/binding split
- **Where concrete inputs enter:** only through `materialize_core_program`, which calls the GridSearch argument builder with the current space, templates, `flat_params`, period and age.
- **What depends only on the schema:** `_prepare_abstract_program` converts every leaf to `ShapeDtypeStruct` right away. Everything after that depends only on:
  - the abstract schema (shape, dtype, weak typing, commitment, sharding);
  - the declarations;
  - the policy, plus `budget is None`.

  That covers the materialized program, transfer plan, templates, `_CoreFrontier`, candidates, layouts, consumer census and metadata.
- **Dispatch rebuilds its arguments:** `grid_search.py:847` calls the builder again with current arrays on every call.
- **Per-call state:** the `PlannedInputLiveness` ledger, donations, lowering keys (which depend on donations) and the `_LazyCandidateFrontier` dicts. Admission runs after `structural_resolution`.

## Implementation
- **New module `src/_lcm/solution/structural_blueprints.py`:** `StructuralBlueprintCache` (an LRU of 4 entries with a lock and hit/miss counters), `abstract_schema` and `frozen_policy`.
- **`backward_induction.py`:** `_build_structural_blueprint` runs the unchanged loop and returns an immutable `_StructuralBlueprint`. `_bind_structural_blueprint` builds a fresh ledger, donations, lowering keys and frontier on each call.
- **Cache key (`_structural_key`):** the program fingerprint and declaration ids, plus the schemas of `flat_params`, `ages.values`, the value/continuation templates and the base spaces. It also includes the execution fields other than the three budget fields, a budgeted flag, `enable_jit` and x64.
- **What is cached:** only graphs whose builders are all `_GridSearchArgumentBuilder`. Everything else takes the uncached path.
- **`model.py`:** adds `_structural_blueprints` (dropped in `__getstate__`, re-created in `__setstate__`) and passes it to `solve`.

## Tests
`tests/solution/test_invariant_structural_cache.py` has 22 tests: RT3 plus Pro's safety matrix, run at fp32 only.
- **Red at 9b36adcb:** 13 failed, 9 passed. All 6 RT3 cases fail as intended: warm calls rebuild 8 programs unblocked and 20 blocked. The 9 passes are controls that cannot be red before a cache exists.
- **Green attempt:** 15 passed, 7 failed. RT3 passes 6/6, with no warm rebuilds and bytes equal to a fresh model. Plan-equals-fresh passes 6/6. The unblocked budget-change test passes: no rebuild, re-admitted widths equal to a fresh model's.
- **Remaining failures:**
  - `test_a_weak_typing_change_rebuilds_the_structure[False/True]` and `test_a_layout_change_rebuilds_the_structure[False/True]`: the cache hits. `Model.solve` probably normalises params before `flat_params`, so the hit may be correct; check the schema after normalisation and fix the test.
  - `test_a_budget_change_rebinds_and_readmits_without_rebuilding[True]`: an extra narrow-equals-wide assertion fails by 1 ULP at fp32, a width sensitivity unrelated to the cache. Drop that assertion; cached equals fresh narrow.
  - `test_the_cache_is_bounded_and_released_with_its_model`: the fixture's bfloat16 param is rejected by validation. Use float16 or float64.
  - `test_pickling_a_model_drops_its_blueprints`: the test model can't be pickled at all (a mappingproxy in `fixed_params`). Use a model that pickles.

## What's left
1. Fix the 7 failures, then run red and green at fp32 and fp64.
2. Certification: `repin_corridors --changed-source` for `backward_induction.py` and `model.py`, then `check_seals --fix`, manual reviewable-declaration edits (module bindings and imports), `verify.py`, `--self-test` and `generate_ci_workloads --check`. Then prek and ty.
3. Battery on Marvin (intelsr, `-n 96`), the A B A2 CPU host-time ledger, and a Marvin reduced3 ACA driver.
4. Deferred:
   - equal-schema family reuse across type codes;
   - caching narrower candidates bound by refusals;
   - passing the base spaces into the build loop, which currently rebuilds the state-action space per cell.

Overlap with Stage 5A: `lcm/model.py` (one import, one attribute, getstate/setstate, one kwarg).
