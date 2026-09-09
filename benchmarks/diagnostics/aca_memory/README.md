# Bounded ACA allocating-core diagnostic

The diagnostic is prepared and locally verified. Remote execution has not started.

## Source and inputs

- PyLCM: `c710f9b522ec6041830211d9eaf2536f1c842826`, archived into `pylcm-source/`.
- ACA-model: `b941507c14932b85b482d77ed8d7f3034c5dbfee`, the clean public snapshot at
  `../aca-model-b941507/`.
- Caller: the pinned `benchmarks.asv.bench_aca_baseline._build()` and first
  `model.simulate(...)`, with logging off, 1,000 subjects, seed 0, two benchmark
  preference types and unchanged benchmark economic grids.
- Parameters: tracked ACA `src/aca_model/_benchmark_data/benchmark_params.pkl`, SHA256
  `fc1232211980ab6e5a28de16a84e8384c5345295b6575f28ee708021389c0fc5`.
- `sources.json` seals all relevant Python sources and the frozen pickle. A generated
  version-only shim is separately declared. Algorithmic source files are unchanged.
- Production ACA caller repositories and private data are unnecessary for this
  benchmark. Cancelled production trials remain stopped.

## Execution contract

One fresh process executes the same automatic-solve entry route. After
`_solve_from_flat_params` completes, a private sentinel stops the call before forward
simulation. On failure the original exception propagates. On success this establishes
completion of the observed cold solve, with no numerical recovery or performance claim.

Initial IAME treatment: V100 16 GB, explicit fp64, default allocator-derived planning
budget, preallocation disabled, autotune disabled, empty compilation cache, unchanged
grids. `JAX_PLATFORMS=cuda,cpu` retains host storage support. Both memory-fraction
environment spellings are cleared; the resulting allocator limit is recorded. This
explicit environment treatment must accompany comparison with historical CI, whose
complete environment was not captured.

The first diagnostic uses `--mode synchronize`: adapter inputs and transfers complete
before the actual executable call, and outputs complete immediately afterward. This
changes scheduling and may remove an overlap-dependent failure. `--mode observe`
preserves asynchronous output dispatch but has host metadata/pointer-observation
overhead. It labels completion as unconfirmed. A passing synchronous result cannot close
the original asynchronous OOM.

Bound: 900 seconds, 32 GiB summed process-tree RSS, one process, no warm solve and no
forward simulation. The external supervisor writes launch/completion/deadline receipts
and terminates only its own child process group. RSS summation can double-count shared
pages, so the host cap is conservative. GPU work is bounded by unchanged benchmark size
and the native budget; diagnostics do not tune widths to force a pass.

## Evidence

`receipts.jsonl` is flushed and fsynced before each actual compiled dispatch. It
records:

- Selected workspace widths, compiler identity, compiler peak/residency estimates,
  budget and scalar residency inventory.
- Preferred, fallback and actually selected runtime executable IDs and donation
  decisions.
- Period, regime, core, selected widths, transfers and compiler memory-analysis fields.
- Live allocator statistics and a transient census of live array buffers, joined to
  named retained owners through device/pointer identity.
- Transfer, earlier-pending-work, input-wait, execute and output-wait failure stages.
- GPU/process snapshot on failure, when the driver remains responsive.

Frame, weakref and array references are transient. Receipts retain metadata only.
Registry weak links are identified separately from strong owners. Pointer deduplication
cannot completely account for partial aliases, allocator cache, executable constants,
fragmentation or temporary execution workspace. No collector runs garbage collection,
clears a cache, copies values or reduces numerical arrays.

Compiler/pointer census, formatting and file I/O lie outside recorded execute/wait
intervals. These are diagnostic intervals; synchronization, host gaps and metadata
access affect the run, so none are acceptance timings.

## Verified locally

- The final four-case JAX CPU check passed in 2.904 seconds, exit 0, under a 90-second
  watchdog. It covers both observation modes, compiled-signature rejection, and the
  automatic solve route. The scalar fp32 addition returns exactly 4.0, with four-byte
  argument and output memory reports. The two-period model returns acting values
  `[2.5, 4.0]` and terminal values `[1.0, 2.0]`; workspace/runtime compiler identities
  join actual dispatch receipts, all eight hooks restore, and the sentinel excludes
  forward simulation.
- The final 29 non-numerical checks passed: 11 source verification, three CI launch,
  four supervisor lifecycle and 11 workflow-routing cases. These cover source drift,
  fixed arguments and child exit propagation, unchanged ordinary benchmark routing,
  cancellation during launch, a TERM-ignoring descendant, handler restoration and
  compilation-cache exclusion from uploads.
- Source-only checks verified 355 files in complete checkouts and 354 files through the
  installed-package layout, explicitly omitting ACA packaging metadata. The local CPU
  environment has no ACA installation; the runner verifies its actual installed package
  before numerical imports.
- The observer's nine focused tests passed for receipt durability, original-error
  preservation, wait-stage attribution and result release. Red/green development
  receipts are retained in the coordinating task's evidence directory.

These checks establish the diagnostic plumbing. Full ACA reproduction on IAME is
pending. Synchronization and pointer observation affect scheduling, so a successful
observed solve cannot establish that the original asynchronous OOM is fixed.

## Runner-owned launch

The isolated branch `codex/aca-memory-diagnostic-c710` adds the optional boolean
`aca_memory_diagnostic` input to the existing `benchmark-pr.yml`. Default false
preserves normal PR and manual benchmark runs. Diagnostic mode reuses setup,
`benchmarks-cuda12`, native-payload verification and the single `gpu-benchmark` runner,
then skips ASV and PR comments. The manual diagnostic step has a 20-minute outer
deadline and always uploads available receipts. Compilation-cache files are excluded
from the artifact.

After coordinating review and branch push with the main PR440 task, dispatch:

```sh
gh workflow run benchmark-pr.yml --repo OpenSourceEconomics/pylcm \
  --ref codex/aca-memory-diagnostic-c710 -f aca_memory_diagnostic=true
```

The entry is `benchmarks/diagnostics/aca_memory/run_ci.py`, executed by the existing
benchmark environment. It discovers the installed ACA package without importing it,
records actual checkout SHA separately from the c710 algorithmic source base, and
launches exactly one supervised worker. Python and pickle bytes are verified before
numerical imports. A full ACA checkout also verifies packaging; the installed package
receipt explicitly lists `pyproject.toml` as unavailable packaging metadata. Generated
PyLCM version metadata is recorded separately from the numerical source seal.

Receipts live under `reports/aca-memory-diagnostic/RUN_ID-ATTEMPT/` with `launch.jsonl`,
`supervisor/` and `worker/` output. No arbitrary command, precision, grid or iteration
input is exposed by the workflow.

## Scheduling boundary

Preserve automatic IAME run `34370742673`, job `102531003007`, at c710. It remains
running at the latest check. No diagnostic dispatch before it finishes. The coordinating
task owns review and scheduling; no competing PR branch push or manual SSH GPU process
is part of this plan.

One runner serializes its CI jobs. The existing point-in-time GPU-idle check cannot
reserve the device against independently launched workloads. A durable prevention change
requires all GPU launchers to participate in one scheduler or common machine-level
reservation held throughout execution. This diagnostic patch does not install that
machine-wide protocol and makes no exclusivity claim for external GPU users. It captures
GPU/process state to support interpretation.

No remote numerical command or diagnostic branch push has been performed.
