# Handoff: Stage 7, exact action partitions (GridSearch)

This is a working directory for passing the work between sessions. Delete it before
anything merges into `main`. The master index, with the plan, stage reports and
pending HPC jobs, is `.handoff/` on `feat/invariant-state-execution` (#486). Owner:
https://claude.ai/code/session_01Pxm4joxHzgeNTvT5vf4eN9

## State (2026-10-02)
- **Branch and base:** `feat/invariant-action-partitions`, on #486's 9b36adcb. No PR
  yet.
- **Option:** `ExecutionConfig(action_partitions={"<regime>": n})`.
  - Each device reduces a contiguous run of whole action blocks with
    `HARD_MAX_REDUCTION`, and the accumulators are merged in fixed device order.
  - The mesh is the sharded-state axes × `_lcm_action_partition`.
  - Unsupported uses are refused at construction with `ExecutionPlanningError`.
- **Local results:**
  - kernel and route tests 460/460 at fp64 and fp32, 8-device tests 33/33;
  - the kernel matches a scalar-loop oracle and the unpartitioned stream bit for bit;
  - certificates: verify and self-test (1148/1148) pass; prek passes.
- **Exception to bitwise:** Stage 3 type blocks + states × actions differ by 1 ULP in
  one cell at fp64, the same for n = 2, 4, 8. The test asserts ≤ 8 ULP and says so.
  Accepting this is a user decision.

## Open
- **Marvin battery:** full suite at -n 96, both precisions, plus the 8-device and
  `test_distributed*` files at -n 0. Not yet run.
- **GPU benchmark (plan §12):** 4×A100, comparing four layouts: ordinary
  assets-sharded; type blocks; type blocks + actions; type blocks + states × actions.
  `stage3_arms.py` still needs an `--action-partitions` flag.
- **Certificate gap:** the partitioned route is not its own certified corridor. Its
  sources are sealed, but the candidate-array flow along it is not proved.
- **Admission:** no explicit test that the budget refuses the exchanged accumulators.
- **Deferred:** simulation partitioning, discrete sharded states, folded processes,
  taste shocks and collective reductions (all refused today), and a custom collective.
