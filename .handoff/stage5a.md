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

## Still to do
1. Marvin rerun of `drivers/marvin/battery5a.sbatch`, `SET=main`, `-n 96`, fp64 then
   fp32, on this branch's head. Pass: all 6 reseal parametrizations pass, no other
   failures.
2. The GPU simulate driver, job 28029945 at 9dc2145a, was still pending at the time of
   writing.
3. Then a PR stacked on #486.
4. Deferred: grouping for taste shocks, gated edges and replay routes.
