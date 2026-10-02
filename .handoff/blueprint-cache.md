# Handoff: structural blueprint cache (Pro round-0 R3)

This is a working directory for passing the work between sessions. Delete it before
anything merges into `main`. The master index, with the plan, the Pro R3 design in
`pro-round0/RAW-REPORT.md` and the pending HPC jobs, is `.handoff/` on
`feat/invariant-state-execution` (#486). Owner:
https://claude.ai/code/session_01Pxm4joxHzgeNTvT5vf4eN9

## State (2026-10-02)
- **Branch and base:** `perf/invariant-structural-blueprint-cache`, on #486's 9b36adcb.
  No PR yet.
- **Design:**
  - Cache immutable structural blueprints, keyed by: the program fingerprint; the
    abstract schema of params, templates and spaces; the execution policy minus its
    budget values; and whether a budget is set.
  - Rebind the liveness ledger, donations, lowering keys and candidate frontier on
    every call. Admission stays fresh.
- **Tests:** `tests/solution/test_invariant_structural_cache.py`, 24/24 at fp64 and
  fp32. It is red 13/24 at 9b36adcb.
- **Marvin full suite at b7c87690:** 2 failures at each precision, both fixed in the
  next commit.
  - `planning_transfer_executed` was not rejected. The transfer-plan resolution had
    moved into the unpinned `_build_structural_blueprint`. Build, bind, key and the
    exact `_StructuralBlueprint` class shape are now pinned.
  - The seam test looked for `materialize_core_program` in its old location. It now
    asserts the seam in the builder, with stricter checks.
- **Rerun of the full suite at the fix commit:** requested from the aca session.

## Open
- **Certificate gap:** `src/_lcm/solution/structural_blueprints.py` (`abstract_schema`
  and the key inputs) is not certified. It would need a new `_parse` obligation and
  a regenerated source inventory.
- **Still to write:**
  - a CPU host-time A B A2 ledger, warm and cold, with and without the cache;
  - a reduced3 ACA GPU driver.
- **Deferred:**
  - equal-schema family reuse across type codes;
  - caching narrower candidates bound by refusals;
  - passing the base spaces into the build loop.
