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

## Decisions (user, 2026-10-02)
1. **Plan §1:** per-code narrowing of the canonical regimes through the same engine is
   **accepted**. It reuses the engine, scheduler and executable cache, and leaves the
   economic model and fingerprints unchanged.
2. **Stage 3 vs unblocked, 5A life-cycle model:** Stage 3's blocked `work` values
   differ from unblocked by up to 2 ULP at fp64, and one panel value by 1 ULP. This is
   to be **investigated on #486**: root-cause it, and fix it there if possible. 5B
   keeps its bitwise gate against period-major.
3. **The 5B tests' 8-ULP assertion against unblocked on that model is provisional**
   until decision 2 is resolved.
4. **API and contract changes approved:** `InvariantBlockSchedule`,
   `ExecutionConfig.invariant_block_schedule`, `solve(executable_cache=...)`, lazy
   host-backed entries, and `stored_codes`. Open a draft PR once Marvin is green, then
   run the reduced3 GPU driver, then 8A.

## Marvin battery at 731e9fe8 (aca session; `~/marvin-jobs/pylcm-inv-5b/731e9fe8/`)
| Job | Set | Tests | Fail | Err | Skip | Wall (junit) |
|---|---|---|---|---|---|---|
| 28036892 | full fp64, `-n 96` | 18246 | 1 | 0 | 333 | 0:19:10 |
| 28033548 | full fp32, `-n 96` | 18207 | 0 | 0 | 606 | 0:17:51 |
| 28033549 | topology set (sim8/shard8/grouped2 at both precisions, four-device files at fp64) | all files green | 0 | 0 | 0 | 0:16:51 elapsed |

- **The fp64 failure:** a host-time bar,
  `test_admission_preflight_contract::test_unstubbed_warm_full_call_progress_meets_existing_time_bar[multi_regime]`.
  The measured ratio was 1.53 against a bar of 1.5, under `-n 96`. 5B touches
  `simulate.py`, so an isolated A/B rerun is requested: 5B, then the 5A base
  68f35d39, then 5B again.
- **Your checks:** `test_block_major_lifetime.py` ran 40/40 at both precisions. The
  six new `block_major` tests in the 8-device sharding file all pass.
- **A hang, not a test failure:** the first fp64 job, 28033547, hung on node120 after
  "created: 96/96 workers" and was resubmitted. It's the second such hang today (the
  cache's 28031604 was the first).

- **Isolated timing A/B** (job 28037985, one exclusive node, fp64, `-n 0`, 5
  repetitions per block). Order: 5B 731e9fe8, then 5A 68f35d39, then 5B again.
  - Both parametrizations passed every time: 15/15 `[dissolution]` and 15/15
    `[multi_regime]`.
  - Junit times per repetition are the same across blocks: 10.45–10.70 s on 5B and
    10.41–10.69 s on 5A.
  - The 1.53 seen in the battery is `-n 96` contention, not a 5B regression.

## Still to do
- The reduced3 GPU driver run.
- Deferred:
  - budgeted block-major simulation;
  - regimes without the blocked state;
  - blocks wider than one code;
  - on-disk fragments;
  - 8A.

## Local receipt intake

Queued native arm receipts must be checked with the hardened
`drivers/stage5b/compare_stage5b.py` before acceptance. The comparator requires
three distinct arms, four distinct call labels, valid nonempty matching value
digest keysets, and panel/raw digests on the three simulation calls. It checks
the driver's available source, library, JAX, precision, device, economic-model,
clean-worktree and GPU-exclusivity fields, allowing the intentional execution
configuration difference. Construction refusals remain raw outcomes rather than
completed parity evidence.

Block-major and period-major must have equal byte digests. Unblocked digests are
a populated control; their differences do not establish an ULP bound. Matching
keysets cannot detect the same omitted coordinate in every arm because the
receipt does not contain an independently expected coordinate schema.

The local CLI regression checks use the published driver fields without requiring
unavailable parameter or driver hashes. Local checks do not constitute native
GPU-driver acceptance or reopen the pending unblocked rounding decision.

The comparator CLI acceptance check completed with **49 passed, 0 failed,
0 errors, 0 skipped**, in **0:00:07.554** on hmg-office. The checked comparator
and adjacent test are committed at
`e19da294f4bcfad2cfd0416155c261fe0b301882`; comparator SHA256 is
`fab439f546db5604f36969493534b24a73347169b32a6ef4f8b0e86a8284d083`.
The verbose log, exact JUnit population, red/green sequence, environment identity,
and full commands are recorded locally in
`/home/hmg/econ/aca-dev/.task-evidence/pylcm-handoff/5b-comparator/REPORT.md`.
This is comparator intake acceptance; the queued native GPU gate remains pending.

Reproduce the CLI check with the installed tests-cpu environment and a separate
JUnit output, preserving the archived receipt:

```sh
zsh -ic '
  cd /home/hmg/econ/aca-dev/.codex-worktrees/invariant-5b-comparator &&
  STAGE5B_TEST_MANIFEST=/home/hmg/econ/aca-dev/pylcm/pyproject.toml \
    cap pixi run --as-is \
    --manifest-path /home/hmg/econ/aca-dev/pylcm/pyproject.toml -e tests-cpu \
    pytest .handoff/drivers/stage5b/test_compare_stage5b.py --noconftest -v \
    --junitxml=/tmp/stage5b-comparator-reproduction.xml
'
```
