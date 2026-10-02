# Stage 5B: block-major lifetime (WIP)

Branch `feat/invariant-block-major-lifetime`, stacked on 5A (#491). The design is in
`reports/stage5b-design.md`. The GPU driver is in `drivers/stage5b/`. Evidence for the
Stage 3 ULP finding is in `reports/stage5b-findings/`.

## What it is
- **Opt-in:** `ExecutionConfig(invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR)`.
  The default is `PERIOD_MAJOR`, which is unchanged.
- **Solve:** for each code of the blocked state, the existing backward-induction engine
  runs over all periods. Its input is a copy of the canonical regimes with that state's
  grid and the bound programs narrowed to the code. All codes share one
  `ExecutableCache`, so later codes compile nothing. There is no new `Model`, and the
  functions, params and fingerprints are unchanged.
- **Simulation:** each code's subjects are simulated right after that code is solved.
  Its device buffers are then copied to the host and deleted.
- **Result:** a complete, lazy, host-backed `ValueStore`. Inspecting it reads nothing.
  A coverage manifest refuses a missing or repeated code. A failed code publishes
  nothing and frees its buffers.
- **Refused** (`ExecutionPlanningError`):
  - no blocked state, or a regime without it;
  - ungrouped simulation;
  - budgeted simulation (a budgeted solve works);
  - `log_path`.

## Local evidence (CPU, worker; paths under the session scratchpad `s5b/`)
- **New tests:** `tests/solution/test_block_major_lifetime.py` 40/40 at fp64 and fp32.
- **8-device file:** 28/28 at both precisions.
- **Regressions:** simulation 182 passed; API and docs 62 passed. The solution,
  persistence and wider regression runs were on earlier trees.
- **Certificates:** seals, `repin --check` and `verify` all pass after merging 5A
  (12be1db/68f35d3). The 4 `simulation_program:*` failures the worker saw at 9dc2145a
  are fixed by 5A's 24649e3.
- **Parity:**
  - block-major equals period-major bytewise, for values and panels;
  - it equals unblocked bytewise on the two Stage 3 workloads;
  - on the 5A life-cycle model it is within 8 ULP of unblocked (see below).

## Decisions pending (user)
1. **Plan §1, "no per-type model reconstruction".** Per code, the narrowed regimes go
   through the same engine. This is not `Model.solve`, and nothing about the economic
   model changes. Is that acceptable?
2. **Stage 3 vs unblocked, 5A life-cycle model.** At fp64, Stage 3's blocked `work`
   values differ from unblocked by up to 2 ULP, and one panel value by 1 ULP. This
   predates 5B (5B reproduces Stage 3 exactly). Plan §13 says to investigate before
   widening a gate.
3. **The 5B tests assert 8 ULP against unblocked on that model.** This holds only until
   decision 2 is settled.

## Still to do
- Marvin full-suite battery (fp64, fp32) and topology set.
- The reduced3 GPU driver run.
- Deferred:
  - budgeted block-major simulation;
  - regimes without the blocked state;
  - blocks wider than one code;
  - on-disk fragments;
  - 8A.
