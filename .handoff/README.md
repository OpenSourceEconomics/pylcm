# Handoff: invariant-state execution (#486 and follow-ups)

This directory is a working space for passing the work between sessions. Delete it
before anything merges into `main`. A `handoff-guard` workflow that fails pull
requests into `main` while it exists is pending: the cloud session's token cannot
write workflow files, so a person has to add it. It is a copy of the original local handoff (2026-10-02), updated by
the cloud session that now owns the work:
https://claude.ai/code/session_01Pxm4joxHzgeNTvT5vf4eN9

Start with `plan.md`; it is authoritative. The user's standing instructions:
- Follow the plan meticulously and hand each chunk to a subagent. Workers return
  changes; the parent commits.
- Explicit permission to push and open PRs.
- Anything on the HPC (Marvin, GPU) is run through the aca session,
  `session_01Hs8mfXPYyEa24KaAp6BwCW`; this session has no cluster access.
- Each pushed work-in-progress branch carries its own `.handoff/`, with this
  branch's copy as the master index.

Approved macOS CI change, prepared for parent publication:
[macos-four-shards-proposal.md](macos-four-shards-proposal.md).
The user approved four macOS shards and that leg's measured-weight refresh.
The isolated preparation retains the reviewed source/config/test bytes;
native runtime acceptance remains outstanding after publication.

Receipt fixture source-identity repair:
[receipt-fixture-identity.md](receipt-fixture-identity.md).

Shared JUnit receipt identity repair: [receipt-junit-identity.md](receipt-junit-identity.md).
This helper-only repair preserves parameter labels. Its original native receipt
observations are historical; it does not revise the approved CI decisions above.

## Stage status (2026-10-02)

| Stage | State | Branch / PR | Head |
|---|---|---|---|
| 0–3 | done | `feat/invariant-state-execution` (#486, draft) | 9b36adcb + this guard |
| 4 | skipped (3 types don't divide a 4×A100 node) | — | — |
| 5A | Marvin battery green (28031402/03); GPU driver pending | `feat/invariant-type-aware-simulation` (#491, draft) | e58b466 |
| Blueprint cache (Pro R3) | Marvin full suite green at 1d8c306f (28033850 / 28031605) | `perf/invariant-structural-blueprint-cache` (#490, draft) | see its `.handoff/blueprint-cache.md` |
| 5B | Marvin battery green (timing row cleared by isolated A/B); GPU driver requested | `feat/invariant-block-major-lifetime` (#494, draft, on #491) | 8691e767 |
| 6 | closed, not warranted (user decision) | — | evidence in #486 comment 5950352511 |
| 7 | Marvin full + certificate batteries green; GPU §12 benchmark pending | `feat/invariant-action-partitions` (#493, draft); see its `.handoff/action-partitions.md` | 71c902d |
| 8A | design starting (after 5B) | — | — |
| 8B | removed | — | — |

## Pending HPC jobs (run by the aca session)

| Jobs | What | Commit |
|---|---|---|
| 28028091 / 28028092 | production pair fp32, blocked vs unblocked | 9b36adcb |
| 28028093 / 28028094 | production pair fp64, blocked vs unblocked | 9b36adcb |
| 28031402 / 28031403 | 5A battery main rerun, fp64 / fp32, `-n 96`: green | 12be1db9 |
| 28029944 | 5A topology set | 9dc2145a |
| 28029945 | 5A GPU simulate driver, reduced3, A100, fp64 | 9dc2145a |
| 28030481 / 28030482 | blueprint-cache full suite, fp64 / fp32, `-n 96` | b7c87690 |

On Marvin:
- the 5A outputs are in `~/marvin-jobs/pylcm-inv-5a/9dc2145a/` and `~/inv5a-out/`;
- the cache outputs are in `~/marvin-jobs/pylcm-inv-cache/b7c87690/`;
- the original handoff tarball is unpacked at `~/inv5a-handoff/`.

Don't touch `~/pylcm-prod-inv` while the production pair runs.

## Recorded decisions
- The user approved the blocked-vs-unblocked ULP contract: see
  `reports/stage3-ulp.md` (FMA contraction; 8 ULP for values, exact structure).
  Numerical implementation and native acceptance are separate from this CI change.
- The authorized endpoint is Stage 8A. This macOS CI preparation makes no
  Stage 8A implementation or acceptance claim.

## Next steps
1. Fold the production-pair numbers into #486.
2. 5A (#491) and the cache (#490) are open as drafts stacked on #486.
3. Then 5B, then 8A (8A depends on 5B).
4. Still to write for the cache: a CPU host-time A B A2 ledger and a reduced3 driver.

## Layout
- `plan.md`, `baseline-revisions.md`: the plan and its frozen revisions.
- `reports/`: stage reports, Pro round 0, and the blueprint-cache work-in-progress
  report.
- `briefs/`: the original subagent handoffs.
- `pro-round0/`: Pro's full reply. `RAW-REPORT.md` holds the R3 cache design, and
  `AUDIT-ARTIFACTS.zip` holds RT3.
- `drivers/`: the 5A GPU driver (`stage3_arms.py` is the shared harness) and the 5A
  Marvin battery sbatch.

## House rules
- **Python:** `pixi run` only.
- **TDD:** red first.
- **Certificates:** follow `docs/development/certification.md`; never loosen a gate.
- **Batteries:** on Marvin, `-n 96`, with `-v --junitxml`.
- **Slurm:** job names `pytask-pylcm-inv-*`; cancel by job ID only, never
  `scancel -u`.
- **ACA:** GPU only.
- **Reporting:** durations as h:mm:ss; report exact junit counts.
