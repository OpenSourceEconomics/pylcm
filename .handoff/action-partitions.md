# Handoff: Stage 7, exact action partitions (GridSearch)

This is a working directory for passing the work between sessions. Delete it before
anything merges into `main`. The master index, with the plan, stage reports and
pending HPC jobs, is `.handoff/` on `feat/invariant-state-execution` (#486). Owner:
https://claude.ai/code/session_01Pxm4joxHzgeNTvT5vf4eN9

## State (2026-10-02)
- **Branch and base:** `feat/invariant-action-partitions`, stacked on #486 (merged
  up to 0c3c9d9).
- **Option:** `ExecutionConfig(action_partitions={"<regime>": n})`.
  - Each device reduces a contiguous run of whole action blocks with
    `HARD_MAX_REDUCTION`, and the accumulators are merged in fixed device order.
  - The mesh is the sharded-state axes × `_lcm_action_partition`.
  - Unsupported uses are refused at construction with `ExecutionPlanningError`.
- **Local results:**
  - kernel and route tests 460/460 at fp64 and fp32, 8-device tests 33/33;
  - the kernel matches a scalar-loop oracle and the unpartitioned stream bit for bit;
  - certificates: verify and self-test (1206/1206, 598 direct-flow mutations) pass;
    prek passes.
- **Exception to bitwise (accepted):** Stage 3 type blocks + states × actions differ by
  1 ULP in one cell at fp64, the same for n = 2, 4, 8. The test asserts ≤ 8 ULP and
  says so. User decision (2026-10-02): accepted.
- **Certificate:** the partitioned route is its own certified corridor,
  `singleton_action_partitioned_solve` (`_ACTION_PARTITION_CONTRACTS` in
  `direct_flow.py`). It has 29 mutation controls, each rejected after an independent
  byte reseal (`tests/test_action_partition_certificate.py`), and they are sealed into
  the self-test.

## Marvin evidence (run by the aca session)
Walls are junit `time`, with sacct elapsed in parentheses.

**Full suite at 10f69b8a** (`-n 96`), evidence in `~/marvin-jobs/pylcm-inv-s7/10f69b8a/`:

| Job | Set | Tests | Fail | Err | Skip | Wall |
|---|---|---|---|---|---|---|
| 28031124 | full fp64 | 18674 | 0 | 0 | 333 | 0:20:34 (0:21:06) |
| 28031125 | full fp32 | 18635 | 1 | 0 | 606 | 0:17:36 (0:17:57) |
| 28031126 | topo8 p64 / topo8 p32 / distributed p64, `-n 0` | 54 / 54 / 146 | 0 | 0 | 0 | 0:07:42 elapsed |

The one fp32 failure is `test_compile_requests::test_simulate_host_time_at_progress_is_within_the_bar_of_off[multi_regime]`. It was TIMING_UNSTABLE_HOST under `-n 96` (no steady host in 8 attempts), not a value mismatch. Run in isolation, it passed at this commit and at 9b36adcb.

**Certificate battery at 263189f9**: tests and certificate only since 10f69b8a. It ran `tests/candidate_certificate tests/test_*certificate*.py tests/ci` at `-n 96`, with evidence in `~/marvin-jobs/pylcm-inv-s7/263189f9/`:

| Job | Precision | Tests | Fail | Err | Skip | Wall |
|---|---|---|---|---|---|---|
| 28031615 | fp64 | 1079 | 0 | 0 | 0 | 0:08:13 (0:08:48) |
| 28031616 | fp32 | 1079 | 0 | 0 | 0 | 0:07:24 (0:07:42) |

`test_action_partition_certificate.py` collected 31 tests at each precision, and all passed.

## Open
- **GPU benchmark (plan §12):** 4×A100, comparing four layouts: ordinary
  assets-sharded; type blocks; type blocks + actions; type blocks + states × actions.
  `stage3_arms.py` still needs an `--action-partitions` flag.
- **Admission:** no explicit test that the budget refuses the exchanged accumulators.
- **Deferred:** simulation partitioning, discrete sharded states, folded processes,
  taste shocks and collective reductions (all refused today), and a custom collective.
