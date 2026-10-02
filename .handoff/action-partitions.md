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
- **Admission:** the public CPU boundary is covered by the addendum below.
- **Deferred:** simulation partitioning, discrete sharded states, folded processes,
  taste shocks and collective reductions (all refused today), and a custom collective.

## Explicit action-partition admission acceptance

The public `Model.solve` seam uses `ExecutionConfig` with four action partitions,
four CPU devices, fixed action width 5 and cell width 8. Three separate tests check
refusal one byte below the represented workspace ceiling, bitwise value parity at
the exact ceiling, and terminal values equal to the analytical bequest
`sqrt(linspace(1, 10, 8))`. The existing source already meets this behavior; no
production arithmetic or certificate change is needed.

The ceiling uses independently compiled executable `memory_analysis()` counters
plus fixture-specific retained residency: two unread value vectors in period 0,
one in period 1, terminal owner and transfer scratch in period 2, and neither in
period 3. Diagnostics must charge that independently measured reservation. The
largest compiler reservation is 2524 bytes at fp64 and 2080 at fp32; the largest
resident charge is 592 and 308 bytes, respectively. Exact admission ceilings are
3116 and 2388 bytes; budgets 3115 and 2387 are refused. The three `[8, 4]` exchanged
accumulators occupy 416 and 288 bytes, respectively.

An external negative-control harness removes the exchange charge globally,
including the calibration solve. It is rejected because charged reservations
2108/1792 disagree with the native 2524/2080 bytes. This guards against a lowered
calibration ceiling concealing an accounting omission. The mutation is not
persisted in production source.

Evidence is in `.task-evidence/pylcm-handoff/stage7-admission/` of the enclosing
ACA workspace; `REPORT.md` records commands, source/diff identities and raw paths.
At source base `71c902d266e4eb49d7d92232f8283df302ae8fd2`, the focused device,
route and workspace-budget matrix has these exact JUnit totals:

| Run | Passed | Failed | Errors | Skipped | JUnit wall |
|---|---|---|---|---|---|
| Global omission control, fp64 | 0 | 1 | 0 | 0 | 0:00:01.139 |
| Global omission control, fp32 | 0 | 1 | 0 | 0 | 0:00:01.071 |
| Focused matrix, fp64 | 83 | 0 | 0 | 0 | 0:00:38.055 |
| Focused matrix, fp32 | 83 | 0 | 0 | 0 | 0:00:36.898 |

Both green runs import this isolated worktree's native payload. These are CPU
semantic and admission checks; they provide no GPU performance or physical
process-memory bound. The Stage 7 GPU acceptance remains open.
