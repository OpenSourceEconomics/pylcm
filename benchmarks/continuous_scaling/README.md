# Continuous-sharding scaling measurement

This task measures the fixed finite 3 × 3 × 24 witness on 3, 4, 6 and 8 physical A40
GPUs, with an unsharded one-GPU reference. It uses both precisions, two fresh processes
per arm, and one cold plus three synchronized warm solves per process. Separate semantic
tests retain the eight-ULP contract, exact decisions, RNG, transfer ownership and
compiler-derived admission checks. This is tiny-workload strong scaling; it does not
establish ACA production performance or close the remaining baseline/main resource
comparisons.

Commit and push this directory with the source tree. On Marvin, pull that exact revision
into a dedicated clean Git checkout, then run its normal frozen Pixi installation for
`tests-cuda13`. Never populate production imports or native payloads from source
archives, copied version modules or another checkout's installation. Set
`NVCCFLAGS=-arch=sm_86` consistently for installation and execution on A40. Set
`PYLCM_SCALING_EXPECTED_HEAD` to the full pushed commit, and run:

```bash
pixi install --frozen -e tests-cuda13
JAX_PLATFORMS=cpu pixi run --frozen --no-install -e tests-cuda13 \
  python benchmarks/continuous_scaling/check_install.py \
  --output reports/scaling-install-login.json
```

The controller needs an existing environment with pytask and pytask-slurm. Set
`SCALING_CONTROLLER_MANIFEST` to that environment's manifest. This environment is used
only for orchestration; the task explicitly invokes this checkout's own Pixi manifest
for all measurement subprocesses. Collect first and require exactly one task. Run from
the dedicated pulled checkout:

```bash
pixi run --manifest-path "$SCALING_CONTROLLER_MANIFEST" --frozen --no-install \
  -e cuda13 python -m pytask collect \
  --config benchmarks/continuous_scaling/pytask.toml \
  benchmarks/continuous_scaling/task_scaling.py
pixi run --manifest-path "$SCALING_CONTROLLER_MANIFEST" --frozen --no-install \
  -e cuda13 python -m pytask build \
  --config benchmarks/continuous_scaling/pytask.toml \
  benchmarks/continuous_scaling/task_scaling.py --slurm
```

The task requests one allocation: `mlgpu_short`, 8 A40 GPUs, 16 CPUs, 128 GiB, 5 hours;
its internal watchdog is 17,700 seconds. It refuses a nonempty results directory. Record
the controller log, generated batch script and exact job ID; never restart the
controller to harvest a failed job, since that can resubmit it. No development-partition
companion is submitted.

The worker verifies Git, installed versions, distribution and native paths, then checks
the native payload against the compute node's actual build inputs and FFI registration.
As in GPU CI, a proven absent/stale native payload permits one normal
`pixi reinstall --frozen -e tests-cuda13 pylcm`, with its own retained log and a second
strict probe. Other failures stop immediately. No native files are copied.

Results and raw logs live in `reports/continuous-scaling-rerun/`. Compiler duration
categories overlap; backend events count compile-or-cache requests. Warm requests must
be zero and cold positive controls must fire. GPU peaks are cumulative live allocator
high-water marks within each process, including initialization and the retained first
solution; they are not phase-reset peaks or reserved-pool estimates. The source-sealed
fixture patches change physical topology and derived accounting, not economic
definitions or tolerances. A completed scheduler job is not scientific acceptance:
inspect every semantic XML and measurement record before disposition.
