# Handoff: invariant-state execution (#486 and follow-ups)

This directory is a working space for passing the work between sessions. Delete it
before anything merges into `main`. Historical cloud-session record: a
`handoff-guard` workflow was pending because that session's token could not write
workflow files. This directory derives from the original local handoff
(2026-10-02), subsequently updated by that cloud session; its ownership/token
statement is historical, and the current checkpoint below governs:
https://claude.ai/code/session_01Pxm4joxHzgeNTvT5vf4eN9

Start with `plan.md`; it is authoritative. The user's standing instructions:
- Follow the plan meticulously and hand each chunk to a subagent. Workers return
  changes; the parent commits.
- Explicit permission to push and open PRs.
- The user authorized direct SSH to Marvin, superseding the historical
  aca-session-only route. Production submission waits for passing checks.
- Each pushed work-in-progress branch carries its own `.handoff/`, with this
  branch's copy as the master index.

Approved macOS CI change, historical preparation:
[macos-four-shards-proposal.md](macos-four-shards-proposal.md).
The user approved four macOS shards and that leg's measured-weight refresh.
The isolated preparation retains the reviewed source/config/test bytes.
Its prepared-state receipt remains historical; current cascade status follows.

Receipt fixture source-identity repair:
[receipt-fixture-identity.md](receipt-fixture-identity.md).

## Current #497 checkpoint (2026-10-03)

#491 is published at `3a214d11988b1c88cbf18667bfb43aa580781450`, incorporating
#494, #495 and the complete #493 history. #493 is closed as incorporated:
GitHub refused retargeting because no new commits remained against #491.
#496 is merged into #486 at `b1dc9d077ccd1289c41010615132be11df40a53d`.
The plain #491 merge into this branch carries the shared
[JUnit identity note](receipt-junit-identity.md) and this checkpoint; source,
tests, certificates, workflow, manifest and lock match the qualified production
driver commit `d9da850a96261a698c8ffdc52c1e485335b2dc78` exactly. It introduces
no numerical implementation change or native acceptance.

The user authorized consolidation by merges and pushes. #495 was fast-forwarded
into #494, then #494 into #491, both at the already tested commit
`cb62c3d9351b9d4aa2acdad885b6aaab93f8a16c`; normal push hooks passed. GitHub
records both PRs merged and automatically retargeted #497 to #491. No main
merge occurred. The source/check records below distinguish original core
validation from subsequent CI refresh and production-driver validation.

- Stage8A core is published as draft [PR #497](https://github.com/OpenSourceEconomics/pylcm/pull/497)
  at `3028a022e8156880d71c6733a47c43f20828283b`, based on published #495
  `cb62c3d9351b9d4aa2acdad885b6aaab93f8a16c`. Normal publication hooks and
  committed source-binding checks passed. The approved general value-only
  eight-ULP contract is published on #495; structural and same-program byte
  gates remain exact. Stage8A implementation,195 local numerical cases per
  precision, later cast/generated-version qualifications and completed local
  certificate/CI gates are detailed in [stage8a.md](stage8a.md) and its
  [numerical report](reports/Stage8A/FINAL-NUMERICAL-REPORT.md) and
  [gate report](reports/Stage8A/FINAL-GATES-REPORT.md).
- Exact #497 Mac intake reconciled12,305 selected/executed cases across four
  lanes (11,990pass/315skip/0fail/error) at tested merge
  `1717cb235b3f73740080c3ad7614f370c1d3e920`, whose tree equals the published
  head. This is the general `not slow and not manual` subset: the new71-case
  component-job module has **zero Mac selection/execution** because its
  solution directory is slow. Mac3 payload25:29 misses24minutes; its whole
  job26:53 fits30. Passing jobs do not establish all timing budgets or new
  Stage8A macOS coverage.
- The approved one-weight Mac refresh has local validation: one measured
  module790.597seconds/count150, maintained four-shard regeneration, and
  prediction20.13minutes per lane. Existing129 contract cases pass,
  0fail/error/skip,0:00:07.783; all other weights, budgets, coverage and source
  gates are preserved. Publication identity and fresh measured Mac24/30 timing
  are tracked in PR #497 and the publication ledger. See [refresh report](reports/CI/MAC-ONE-WEIGHT-REFRESH.md)
  and [matching CSV](reports/CI/mac-one-weight-refresh.csv).
- Published `0019faab26766a0bbffb24d786f873e05a16aa67` has completed refreshed
  Mac intake: 12,305 cases, 11,990 pass/315 skip/zero failure or error, with
  exactly the prior whole population and outcomes. Maximum payload12:32 and
  whole-job13:43 meet24/30minutes for this sample. All four canonical and16
  worker receipts bind tested merge `7635f58c47bca10054a245c856226b99d1755008`,
  whose whole tree equals0019. This general subset still selects none of the
  new71 component-job cases; it establishes neither native acceptance nor a
  causal speedup. See [Mac intake](reports/CI/MAC-0019-INTAKE.md).
- The production driver is prepared on this branch, with its bounded
  qualification in [stage8a-production-driver.md](stage8a-production-driver.md).
  Actual simultaneous
  **three nodes × eight GPUs each on mlgpu_short** with full canonicalACA inputs,
  matched fp32 reference and native correctness/performance/resource receipts
  is unsubmitted. The user authorized any available direct SSH connection;
  access through the existing desktop agent succeeds. Isolated frozen sources
  and production inputs are staged; native installation and admission remain
  required before submission.
  Driver/local and source-review receipts do not establish this native gate.

## Historical stage status (2026-10-02)

Live integration and rounding-contract acceptance are tracked in
[`stages0-7-integration.md`](stages0-7-integration.md) and
[`rounding-contract.md`](rounding-contract.md). The user approved value-only eight
ULP for general blocked/unblocked comparisons and the macOS CI proposal; exact
structural and same-program byte contracts remain binding. Stage 8A is required;
the optional blueprint cache is deferred. The table below is the historical
cloud-session checkpoint, not a fresh native execution receipt.

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
- [`stages0-7-integration.md`](stages0-7-integration.md): the isolated 5A/5B + 7
  source composition, fresh CPU gates, certificate controls and pending native
  acceptance decisions. The enclosing reconciled handoff governs this chunk.
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
