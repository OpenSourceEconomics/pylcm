---
title: Continuous integration
---

# Continuous integration

PyLCM assigns tests by what they require and what coverage they provide. File location
is not a proxy for hardware or cost, and a GPU job does not replay every CPU test.

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

Five independent markers describe a test:

- `requires(...)` states hard capabilities such as a GPU, native kernel, platform, or
  minimum device memory.
- `coverage(...)` assigns routine backend and precision coverage. An unmarked test is
  owned by CPU at its representative precision; GPU CI selects explicit GPU obligations
  only.
- `resources(...)` records measured wall time, host/device memory, CPU demand, and
  compile intensity.
- `isolation(...)` declares fresh-process, exclusive-device, cache, or environment
  boundaries.
- `ci(...)` assigns the bounded tier: `pr`, `relevant`, `extended`, or `nightly`.

For example, a production-scale GPU witness that is excluded from ordinary pull requests
but remains available under nightly and full policies is:

```python
@pytest.mark.requires(device="gpu", native=("exact_affine",))
@pytest.mark.coverage(backends=("gpu-small", "gpu-large"), precisions="both")
@pytest.mark.resources(wall="production", gpu_mem_gb=16, compile="heavy")
@pytest.mark.isolation(process="fresh", gpu="exclusive", cache="isolated")
@pytest.mark.ci(tier="nightly")
def test_production_case(): ...
```

Marker arguments are validated during collection. The selection report records every
collected node as selected, policy-deselected, matrix-deselected, isolation-deselected,
or capability-skipped.

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
