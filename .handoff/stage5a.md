# Stage 5A: type-aware simulation

Branch `feat/invariant-type-aware-simulation`, stacked on #486. The design and the
first hand-back are in `reports/stage5a.md` on the original handoff; this note tracks
what changed since.

## History
- 152d0d9, c7abf34: grouped simulation and its docs.
- 9dc2145: merge of #486's F1/F2 fixes; 3 corridors repinned, F2 `requires_plan`
  statements carried over.
- 24649e3: certificate fix. The first Marvin battery (28029942/43) found 4
  `simulation_program:*` mutations surviving a reseal. Cause: 152d0d9 moved the
  decision loop into `_decision_programs`, which no corridor pinned. The fix pins it,
  the `SimulationPrograms.forward_decision` selector and the type-local per-subject
  assignment, and adds 2 controls (`type_local_decision:*`). Supplemental controls go
  from 50 to 52. Tests and certificate only; no `src/` change.

## Local evidence for 24649e3 (CPU, not Marvin)
- `test_simulation_candidate_program_certificate.py`: 310 passed at fp64 and fp32.
- `test_type_grouped_simulation.py`: 23 passed at fp64 and fp32.
- Certificate siblings at fp64: 176 passed.
- `check_seals`, `verify`, `verify --self-test`, `generate_ci_workloads --check`,
  `prek run --all-files`: all exit 0.

## Marvin evidence at 12be1db9 (run by the aca session)
Clone `~/pylcm-inv-5a-r2`, outputs in `~/marvin-jobs/pylcm-inv-5a/12be1db9/`; head
matched and the tree was clean in both jobs.

| Job | Set | Tests | Fail | Err | Skip | Wall (junit / sacct) |
|---|---|---|---|---|---|---|
| 28031402 | main fp64, `-n 96` | 5410 | 0 | 0 | 39 | 0:09:54 / 0:11:01 |
| 28031403 | main fp32, `-n 96` | 5410 | 0 | 0 | 44 | 0:09:02 / 0:09:26 |

All 296 reseal parametrizations passed at both precisions, including the 4 earlier
failures and both `type_local_decision:*` rows.

## Still to do
1. The GPU simulate driver, job 28029945, is pinned to 9dc2145a in `~/pylcm-inv-5a`
   and was pending on priority (estimated start 2026-10-03 03:35). Its source
   differs from this head only in tests and certificate files.
2. Deferred: grouping for taste shocks, gated edges and replay routes.
