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

Approved macOS CI change, historical preparation:
[macos-four-shards-proposal.md](macos-four-shards-proposal.md).
The user approved four macOS shards and that leg's measured-weight refresh.
The isolated preparation retains the reviewed source/config/test bytes.
Its prepared-state receipt remains historical; current cascade status follows.

Receipt fixture source-identity repair:
[receipt-fixture-identity.md](receipt-fixture-identity.md).

## Consolidated #491 checkpoint (2026-10-03)

The user authorized merging and pushing. #495 and#494 are merged into#491 at
`cb62c3d9351b9d4aa2acdad885b6aaab93f8a16c`; #497 is based on#491. #496 is
merged into#486 at `b1dc9d077ccd1289c41010615132be11df40a53d`. This root
cascade preserves #491 source/tests/certificates/workflows/manifest/lock exactly
atcb62; its only payload is the JUnit identity handoff note and link. The
original #495 qualification below is historical. No new numerical or native
acceptance is inferred from this documentation merge.

Shared JUnit repair: [receipt-junit-identity.md](receipt-junit-identity.md).
Direct SSH is authorized, superseding the historical ACA-owner-only route;
production submission still waits for passing checks and authenticated access.

## Historical #495 checkpoint (2026-10-03)

- Incoming CI: published #494 `bc05375f8318d0a5522a46300258bac53744c829`,
  carrying #491 `5b898bb554ccca70dd621c01510639453749055b` and #486
  `553050fdfa60f2116ee20301a56002670725a80d`. The isolated #495 merge preserves
  production, certificates and the ordered-distance test contract exactly from
  its pre-merge head `cc744450d30d4b12cc5b1b9ab6b923226f8ed340`.
  The production source tree remains `babc7fa21e75dccee1894fec46c6f7089b3a4810`.
  Parent publication is pending.
- Local CI acceptance for this #495 merge: maintained manifest generation/check
  and the 18-case workflow/manifest/parser/source-identity smoke passed, with
  no failures, errors or skips, exact JUnit duration **0:00:00.340**. Normal
  all-file hooks passed; parent publication remains pending. Earlier #494
  receipts are not this step's acceptance. Fresh native #495 timing and
  complete receipt identity remain required; the earlier #486 `bdc19260`
  macOS result retains its supporting-worker source-identity qualification.
- The general value-only eight-ULP contract is approved; its tests/documentation
  are published on #495 at `cc744450d30d4b12cc5b1b9ab6b923226f8ed340`.
  Structural and same-program byte gates remain exact.
- Stage 8A has a four-case fp64 CPU receipt on a separate, uncommitted local
  branch. The updated driver has three binding/plan cases only. These are
  inherited local receipts, not native three-node execution or a current
  Stage 8A certificate claim. Actual three nodes × eight full-model GPUs,
  native correctness/performance/resource gates and authorized owner access
  remain required. Cluster execution stays with the ACA owner; this CI
  cascade has no cluster access or Stage 8A acceptance claim.

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
