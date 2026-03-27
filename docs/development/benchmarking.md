---
title: Benchmarking
---

# Benchmarking

pylcm uses [ASV (Airspeed Velocity)](https://asv.readthedocs.io/) to track performance
across commits. Benchmarks run locally on GPU hardware and results are published to a
[dashboard](https://open-econ.org/pylcm-benchmarks/).

## Machine Setup

On first use, register your machine with ASV:

```bash
asv machine --yes
```

This creates `.asv/results/<machine-name>/machine.json` with hardware metadata.

## Running Benchmarks

There are two primary workflows depending on whether you are working on main or a PR
branch:

**PR branches** — run benchmarks and post a comparison comment on your PR:

```bash
pixi run -e tests-cuda13 asv-run-and-pr-comment
```

**Main branch** — run benchmarks and publish results to the dashboard:

```bash
pixi run -e tests-cuda13 asv-run-and-publish-main
```

Both workflows run `asv-run` (which requires a clean worktree) followed by their
respective post-processing step.

Individual tasks are also available:

```bash
# Run all benchmarks (GPU required)
pixi run -e tests-cuda13 asv-run

# Quick smoke test (not saved)
pixi run -e tests-cuda13 asv-quick

# Post benchmark comment to current PR (no GPU needed)
pixi run asv-pr-comment

# Compare two commits (no GPU needed)
pixi run asv-compare HEAD~1 HEAD

# Preview dashboard locally (no GPU needed)
pixi run asv-preview

# Publish results to dashboard (no GPU needed)
pixi run asv-publish
```

The `asv-run` and `asv-quick` tasks set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.3` automatically to prevent JAX from grabbing all GPU
memory.

## Benchmark Scenarios

| File                             | What it benchmarks                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `bench_precautionary_savings.py` | Solve (varying grid sizes), simulate (varying subjects), solve+simulate, lin vs irreg grid comparison |
| `bench_mortality.py`             | Mortality model — solve + simulate                                                                    |
| `bench_mahler_yum.py`            | Mahler & Yum (2024) replication (GPU only)                                                            |

Each benchmark tracks three metrics:

- **`time_*`** — execution time (after JIT warmup)
- **`peakmem_*`** — peak memory usage
- **`track_warmup`** — JIT compilation time

## Publishing Results

After running benchmarks on the main branch, publish them to the dashboard:

```bash
pixi run asv-publish
```

This generates the ASV HTML dashboard and pushes results to the
[OpenSourceEconomics.github.io](https://github.com/OpenSourceEconomics/OpenSourceEconomics.github.io)
repo under `pylcm-benchmarks/`. A persistent clone is kept in `.benchmark-site/`
(gitignored) to avoid re-cloning on every publish.

## CI Check

The `benchmark-check` workflow runs on every pull request. It looks for a PR comment
with the `<!-- benchmark-check -->` marker and checks the embedded commit hash against
the PR's HEAD:

- **Passes** if the comment's commit hash matches the PR HEAD
- **Passes with warning** if a benchmark comment exists but for an older commit
- **Fails** if no benchmark comment is found

To satisfy the check:

```bash
pixi run -e tests-cuda13 asv-run-and-pr-comment
```

## Adding New Benchmarks

Create a new `bench_*.py` file in the `benchmarks/` directory. Benchmarks use ASV's
class-based API:

```python
import gc
import time


class TimeMyModel:
    timeout = 600

    def setup(self):
        # Lazy imports — JAX must not be imported at module level
        import jax.numpy as jnp
        from lcm_examples import my_model

        self.model = my_model.get_model()
        self.model_params = my_model.get_params()
        self.initial_conditions = {
            "wealth": jnp.full(1_000, 5.0),
            "regime": jnp.zeros(1_000, dtype=jnp.int32),
        }

        # JIT warmup (timed separately)
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
        )
        self._warmup_time = time.perf_counter() - start

    def time_solve(self):
        self.model.solve(params=self.model_params, log_level="off")

    def teardown(self):
        import jax

        jax.clear_caches()
        gc.collect()

    def track_warmup(self):
        return self._warmup_time

    track_warmup.unit = "seconds"
```

Key points:

- **Lazy imports**: All imports of JAX and lcm must happen inside `setup()`, not at
  module level, to avoid `os.fork()` conflicts with JAX's GPU runtime.
- **`teardown()`**: Clear JAX caches and run garbage collection between benchmarks.
- **`track_warmup`**: Measure JIT compilation time separately from steady-state
  execution time.
- Use the `params` and `param_names` class attributes to vary grid sizes or other
  parameters.
