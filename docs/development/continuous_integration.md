---
title: Continuous integration
---

# Continuous integration

PyLCM assigns tests by what they require and what coverage they provide. File location
is not a proxy for hardware or cost, and a GPU job does not replay every CPU test.

The workload manifest, the shard split, the timing-witness lane, the files that CI pins
by path, and the candidate certificate are described in
[Certification and preflight](certification.md).

The ordinary pull-request policy is:

```console
pixi run test -- --ci-policy=pr
```

The canonical exhaustive switch for the current machine is:

```console
pixi run test -- --full-suite
```

`--full-suite` includes every policy tier the current machine can truthfully execute. It
does not emulate missing hardware: a CPU full run cannot discharge a GPU or multi-GPU
obligation. Precision legs and tests requiring a fresh process run in separate pytest
children because JAX precision, XLA state, native registration, and compilation caches
are process-global.

Useful explicit choices are:

```console
pixi run test -- --ci-policy=pr --hardware-profile=cpu --precision=64
pixi run test -- --full-suite --hardware-profile=gpu-large --precision=auto
pixi run test -- --ci-policy=nightly tests/test_models/test_ds_app2_housing_builds.py
```

## Declaring a test contract

Four independent markers describe a test, and the policy reads exactly the arguments
listed here:

- `requires(device, min_devices)` states hard capabilities. `device` is `any`, `cpu` or
  `gpu`; `min_devices` above one selects the multi-GPU profile.
- `coverage(backends, precisions)` assigns routine backend and precision coverage. An
  unmarked test is owned by CPU at its representative precision; GPU CI selects explicit
  GPU obligations only.
- `isolation(process)` declares a fresh-process boundary. The launcher schedules such
  tests in their own pytest child.
- `ci(tier)` assigns the bounded tier: `pr`, `relevant`, `extended`, or `nightly`.

For example, a GPU witness that is excluded from ordinary pull requests but remains
available under nightly and full policies is:

```python
@pytest.mark.requires(device="gpu")
@pytest.mark.coverage(backends=("gpu-small", "gpu-large"), precisions="both")
@pytest.mark.isolation(process="fresh")
@pytest.mark.ci(tier="nightly")
def test_production_case(): ...
```

Marker arguments are validated during collection, and an argument outside the list above
is a collection error rather than a silently ignored hint. The selection report records
every collected node as selected, policy-deselected, matrix-deselected,
isolation-deselected, or capability-skipped.

`pyproject.toml` also registers a
`resources(wall, wall_seconds, host_mem_gb, gpu_mem_gb, cpu_cores, compile)` marker. It
documents a measured resource contract for a human reader; the execution policy does not
read it, and it selects nothing.

## Python line coverage

Every lane that runs under coverage uploads its own `coverage-*` artifact. A single
`coverage` job downloads all of them, checks them against the `coverage_contributors`
list in `tests/ci/ci-workloads.json` with `tests/ci/check_coverage_manifest.py`, and
uploads one combined report under the `cpu-python` flag. Reports are not carried across
commits.

Completeness is enforced by that checker, not by Codecov: the combine step refuses to
proceed unless every recorded lane delivered a non-empty report, and refuses an
unrecorded extra artifact too. Codecov itself publishes on the single upload
(`after_n_builds: 1`), so a missing lane would otherwise read as a coverage drop in the
code under test rather than as a lane that never ran.

This percentage measures Python lines exercised on CPU. The fp32 and GPU jobs do not
repeat coverage instrumentation; they test dtype and hardware behavior directly, while
native C++ and CUDA kernels remain outside `coverage.py`. Benchmark jobs never run under
coverage instrumentation.

## Native payload setup

The exact-affine C++/CUDA libraries are part of the installed pylcm payload, including
editable installs. CI restores the Pixi environment once and then runs a strict probe:

- an absent or stale payload triggers one `pixi reinstall pylcm`;
- a present but unloadable payload fails immediately;
- a matching, loadable payload proceeds without compiling.

The fingerprint covers maintained sources, Python ABI, operating system and machine,
JAX/jaxlib, compiler identities and versions, CUDA compiler, and `NVCCFLAGS`. GPU CI
queries the runner's compute capability and builds only that architecture.

## Large GPUs

Large-GPU execution starts serially. A larger device does not imply that simultaneous
JAX compilation is safe: host memory, compile CPU, device memory, and writable caches
are separate constraints. Concurrency above one is enabled only after measurements at
candidate counts 1 through 4 show:

- no correctness or resource failures;
- at least 10% throughput improvement;
- no more than 25% p95 per-test slowdown; and
- sufficient host- and device-memory reserves.

Fresh, exclusive, compilation-heavy, and multi-device groups remain serial. Missing or
inconsistent telemetry falls back to one process.
