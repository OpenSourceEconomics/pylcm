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

## Current #493 cascade checkpoint (2026-10-03)

The published #486 source `553050fdfa60f2116ee20301a56002670725a80d`
is being merged into #493 target `87126269c862c124916ef67c9f81b77247f8d4eb`
with a plain no-commit merge in an isolated checkout. Incoming approved macOS
four-shard workflow, leg map and provenance are retained; the four target-only
Stage 7 files remain registered and unweighted. Source, certificates, non-CI
tests and [action-partitions.md](action-partitions.md) remain exactly the target.
The maintained generator/check, local 18-case CI smoke and normal all-file hooks
pass. The smoke has zero failures/errors/skips, exact JUnit 0:00:00.330.
Parent commit and committed-tree type/locked-CUDA12/native gates remain before
publication. No current #493 native timing or receipt intake is claimed.

The first three cascade steps are published: #491 `5b898bb5`, #494 `bc05375f`
and #495 `cb62c3d9`. Fresh #491 four macOS XML and20supporting receipts reconcile:
12274distinct cases,11959passed/315skipped/zero failures or errors. Tested merge
`6220d089` has the exact published `5b898bb5` Git tree. Maximum payload/job
0:22:46/0:24:18 meets the 24/30-minute budgets for that one run/source.
The earlier #486 `bdc19260` intake retains its qualified 19/20 source identity
result. The approved value-only eight-ULP contract is on #495 `cc744450` and is
not introduced into this Stage 7 source by the CI cascade; strict structural
and same-program contracts and the existing Stage 7 exception are unchanged.
Stage 8A work is separate: actual three-node, eight-GPU full-model execution and
native acceptance still require the authorized cluster owner/access.

## Historical handoff and proposal (2026-10-02)

Approved macOS CI change, then prepared for parent publication:
[macos-four-shards-proposal.md](macos-four-shards-proposal.md).
The user approved four macOS shards and that leg's measured-weight refresh.
The isolated preparation retains the reviewed source/config/test bytes;
native runtime acceptance remains outstanding after publication.

Receipt fixture source-identity repair:
[receipt-fixture-identity.md](receipt-fixture-identity.md).

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
