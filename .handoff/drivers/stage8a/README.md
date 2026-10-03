# Full-production ACA component jobs

`stage8a_production.py` runs the maintained component-job engine through the
existing ACA production builder. Local tests cover a tiny CPU campaign and
explicit runtime recorders. They do not establish full ACA construction,
native GPU readiness, memory fit or a submitted production job. The required
owner-run signal remains simultaneous **three nodes, eight GPUs per node, on
mlgpu**, following a separate matched one-node reference.

The final local matrix passes **124 cases** in exact JUnit time **0:00:29.697**.
Explicit Ruff, Ty, format, keyword-only and shell checks pass for the new
driver. The inherited builder has no new diagnostics relative to its baseline;
its existing 43 Ruff and 2 Ty diagnostics remain disclosed separately.

## Fixed campaign

The builder is `../stage5b/stage3_arms.py::_build(workload="production")`.
Its optional `production_execution_config` keyword selects width-one
`pref_type` BLOCK_MAJOR execution with `device_memory_bytes=None` before the
single Model construction. The default helper path remains available to its
other callers. Do not reconstruct a Model or modify its private execution
fields.

The campaign retains the frozen ACA model source
`ad38653696ec366e318ac61b9a81b597a4ecb700`, `PolicyVariant.ACA`, brute-force
solver, complete economic parameters, all original codes `(0, 1, 2)`, and the
full canonical `nvidia_a100_sxm4_80gb` grid. This grid choice is deliberate even
on A40 nodes. There is no reduced-grid or population-size option.

The inherited builder applies `triple_initdist_by_pref_type` and
`replicate_for_draws(N_DRAWS_PER_INDIVIDUAL)`. Every phase uses the frozen
owner's `select_admissible_starts`; rows outside the model's entry-age domain
are the owner's canonical exclusion, rather than a driver crop. Keep every
admitted row in its original order. Use one dense positional population and
production seed **20260903** for planner, every worker and reference. Workers
receive that complete population; the engine selects their original code.
The immutable original-ID array independently authenticates the external IDs;
the engine's initial-array digest does not hash a DataFrame index.

The intended built-in singleton GridSearch route retains complete values and
has empty auxiliary/replay/continuation/diagnostic channels and omissions.
The driver does not claim general persistence support for other solver targets.

## CLI

All phases require `--plan-directory`, `--out` and `--aca-slurm-src`.

- `plan` creates metadata with assignment `((0,), (1,), (2,))` and publishes an
  atomic planning receipt plus `original_ids.npy`.
- `run` additionally requires `--job-from-slurm-procid`, `--planning-receipt`
  and `--planning-receipt-sha256`. Exact `SLURM_PROCID` values `0`, `1`, `2`
  choose the original-code job. Receipts live in distinct `out/job-NNNN` paths.
- `reference` creates its own metadata-only plan and runs the existing
  `simulate_with_dense_index` adapter on the same complete canonical population.
- `collect` requires the planning receipt/hash, `--reference`,
  `--reference-receipt-sha256` and `--worker-receipts`. It collects through the
  public engine and
  compares the authenticated complete reference before publishing success.

Hashes use exactly 64 lowercase hexadecimal characters. CLI parsing imports
neither JAX nor ACA. Source/setup failures are recorded for a fresh attempt;
completed or interrupted output attempts must remain immutable. New attempts
use new dedicated output paths.

Reference and collection save the complete `SolutionResult` before the
consuming `SimulationResult.save`, then publish an atomic receipt. Public
archives keep their maintained formats. The receipt preserves original value,
raw-coordinate and per-field key order independently of storage's canonical
mapping order. Every persisted payload is authenticated before deserialization;
comparison covers complete ordered keys, shapes, dtypes, numeric bytes,
panel/index/category schema and original-ID mapping. Framed numeric checks
also distinguish signed zero. After exact positional checks, both panels use
the frozen owner's `restore_subject_ids` and compare their external IDs again.
Runtime source labels and per-instance Model
UUIDs do not replace durable Model/parameter/campaign identity.

## Owner installation and allocation

Pin one isolated, clean pylcm checkout containing this driver and its reviewed
helper delta; use the exact frozen ACA model and an explicitly compatible
owner-selected ACA Slurm commit. Diagnostic Slurm source
`b650981593437799ae3235d1d912ea2cd9d5feda` is not a chosen native production pin.
Freeze the manifest/lock and all eleven maintained production input files.

Install the locked native environment **once before submission**, in that
checkout's own `.pixi/envs/benchmarks-cuda12` prefix. Refresh/verify installed
pylcm metadata against its actual generated/source version, then run the
maintained native probe on that exact source root. Require READY and record
the installed library/manifest hashes, source imports, compiler/environment
identity and actual eight-A40 fp32 devices. The simultaneous workers cannot
perform the first shared-prefix build. Runtime recipes use `pixi run --as-is`;
there is no mutable environment symlink or runtime installation fallback.

Owner-provided absolute paths/pins are required:

```sh
export PYLCM_DIR=/absolute/isolated/pylcm
export PYLCM_COMMIT=exact_reviewed_commit
export ACA_MODEL_DIR=/absolute/frozen/aca-model
export ACA_MODEL_COMMIT=ad38653696ec366e318ac61b9a81b597a4ecb700
export ACA_SLURM_DIR=/absolute/compatible/aca-slurm
export ACA_SLURM_COMMIT=exact_owner_selected_commit
export PIXI_LOCK_SHA256=exact_lock_sha256
export DRIVER_SHA256=exact_driver_sha256
export BENCHMARK_HELPER_SHA256=exact_helper_sha256
export OUT_ROOT=/absolute/existing/owner/Lustre/workspace
```

After owner resource/limit verification, the reference recipe requests one
node × eight GPUs; production requests one allocation of three distinct nodes,
one task per node and eight GPUs each. Both literally request **mlgpu**.
The proposed 96 CPUs/task, 120G/node and eight-hour envelope require owner
validation and are not measured fit evidence. Per-phase/per-rank JAX caches
are node-local `/tmp` and set before heavy imports. Heavy outputs use Lustre,
rather than home NFS.

Run the separate reference first, retain its immutable receipt and bind its
absolute directory/hash to the production recipe:

```sh
# Only after owner admission; these submissions have not happened.
sbatch --export=ALL stage8a_reference.sbatch
export REFERENCE_DIR=/absolute/completed/reference
export REFERENCE_RECEIPT_SHA256=exact_completed_reference_receipt_sha256
sbatch --export=ALL stage8a_production.sbatch
```

The production recipe plans on the first node, launches the three workers in
one concurrent `srun --nodes=3 --ntasks=3 --ntasks-per-node=1` step, and collects
only after every worker succeeds. Source/lock/helper guards repeat before and
after the steps. Worker receipts and timestamped engine intervals must prove
three distinct hosts, one allocation/step, ranks 0/1/2, 24 distinct GPU UUIDs,
and actual numerical overlap. A parallel launch directive alone is insufficient.

## Observations and remaining native acceptance

Reuse the existing compile-request counter, phase parser, timestamped logs,
allocator handler and NVML sampler. Counts include persistent-cache requests;
they are not cold-compilation counts. Worker-call wall includes fragment I/O.
Keep collection, comparison and archive/publication walls separate. Host RSS
is the process high-water in Linux KiB, rather than a reset per-call peak.
Close observers/samplers before archive inventory and immutable hashing.
Debug-only plan fields are unavailable at matched `progress` logging; do not
change numerical logging policy to manufacture them.

The local full-production construction diagnostic stopped at the owner's
unsupported local GPU hardware selection before Model construction or native
probe. There is no bypass, fake hardware, model pruning or full-ACA CPU solve.
Owner access is unavailable and neither reference nor the required 24-GPU job
has been submitted. Native construction/compatibility, all input/source proofs,
physical GPU exclusivity/overlap, memory fit and full matched output comparison
remain acceptance gates. One production signal does not establish repeated
paired medians or speedup. Exact local XML outcomes, hashes, commands and
qualified failures are kept in the task evidence, not reconstructed here.
