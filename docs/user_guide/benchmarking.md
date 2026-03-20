---
title: Benchmarking
---

# Benchmarking Your Models

JAX JIT-compiles functions on first call, so the first invocation of `solve()` or
`simulate()` is much slower than subsequent calls. When tuning grid sizes, adding regimes,
or upgrading pylcm, you want to know whether steady-state performance changed — not just
whether the first call got slower. [ASV (Airspeed Velocity)](https://asv.readthedocs.io/)
is a benchmarking framework that tracks metrics across commits, making regressions visible
immediately.

This page shows how to set up ASV in your own project to benchmark your own models.

## Setup

### Install ASV

```bash
pixi add asv
```

### Create `asv.conf.json`

In your project root, create a minimal ASV configuration. The key setting is
`"environment_type": "existing:python"`, which tells ASV to use the current Python
environment (managed by pixi) instead of creating its own virtualenvs:

```json
{
    "version": 1,
    "project": "my-project",
    "project_url": "",
    "repo": ".",
    "environment_type": "existing:python",
    "show_commit_url": "",
    "branches": ["main"],
    "benchmark_dir": "benchmarks",
    "results_dir": ".asv/results",
    "html_dir": ".asv/html"
}
```

### Register your machine

```bash
asv machine --yes
```

### Create the benchmarks directory

```bash
mkdir -p benchmarks
touch benchmarks/__init__.py
```

## Writing a Benchmark

ASV discovers benchmark classes in `bench_*.py` files inside the `benchmarks/` directory.
Here is a full annotated example:

```python
# benchmarks/bench_my_model.py
import gc
import time


class TimeSolveSimulate:
    """Benchmark solve and simulate for my model."""

    # ASV gives up if a single benchmark method exceeds this (seconds).
    timeout = 600

    def setup(self):
        # --- Lazy imports ------------------------------------------------
        # JAX must NOT be imported at module level. ASV discovers benchmarks
        # by importing the module in the main process, then forks a child
        # for each run. If JAX initialises its GPU runtime before the fork,
        # the child inherits a broken CUDA context. Importing inside setup()
        # avoids this because setup() runs in the child.
        import jax.numpy as jnp

        from my_project.models import retirement

        # --- Build model and params --------------------------------------
        self.model = retirement.get_model()
        self.model_params = retirement.get_params()  # not self.params — ASV reserves that
        self.initial_conditions = {
            "age": jnp.full(500, 25.0),
            "wealth": jnp.full(500, 5.0),
            "regime": jnp.zeros(500, dtype=jnp.int32),
        }

        # --- JAX warmup --------------------------------------------------
        # The first call triggers JIT compilation. Time it separately so
        # track_warmup reports compilation cost while time_* methods measure
        # only post-compilation performance.
        start = time.perf_counter()
        self._V = self.model.solve(params=self.model_params, log_level="off")
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=self._V,
            log_level="off",
        )
        self._warmup_time = time.perf_counter() - start

    def time_solve(self):
        """Steady-state solve time (after JIT warmup)."""
        self.model.solve(params=self.model_params, log_level="off")

    def time_simulate(self):
        """Steady-state simulate time (after JIT warmup)."""
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
            period_to_regime_to_V_arr=self._V,
            log_level="off",
        )

    def peakmem_solve(self):
        """Peak memory during solve."""
        self.model.solve(params=self.model_params, log_level="off")

    def track_warmup(self):
        """JIT compilation time (first call)."""
        return self._warmup_time

    track_warmup.unit = "seconds"

    def teardown(self):
        import jax

        jax.clear_caches()
        gc.collect()
```

### Method naming conventions

ASV uses the method prefix to decide what to measure:

| Prefix | What ASV measures |
|---|---|
| `time_*` | Wall-clock time (repeated, reports statistics) |
| `peakmem_*` | Peak memory usage during execution |
| `track_*` | An arbitrary scalar you return (e.g. warmup time) |

### Parameterised benchmarks

Vary grid sizes or number of subjects by adding `params` and `param_names` to the class:

```python
class TimeSolveGrid:
    params = [[50, 100, 200]]
    param_names = ["n_wealth_points"]
    timeout = 600

    def setup(self, n_wealth_points):
        ...  # build model with n_wealth_points grid points

    def time_solve(self, n_wealth_points):
        self.model.solve(params=self.model_params, log_level="off")
```

## Running Benchmarks

```bash
# Run all benchmarks against the current commit
asv run

# Compare two commits
asv compare HEAD~1 HEAD

# Generate and preview an HTML dashboard
asv publish
asv preview
```

`asv run` requires a **clean git worktree** (no uncommitted changes). Commit or stash
before running.

## JAX-Specific Tips

### Memory allocation

By default, JAX pre-allocates most GPU memory at startup. This can cause out-of-memory
errors when ASV forks child processes. Set these environment variables before running:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
```

Or wrap them in a pixi task so you don't forget:

```toml
# pixi.toml
[feature.bench.tasks]
asv-run = { cmd = "asv run", env = { XLA_PYTHON_CLIENT_PREALLOCATE = "false", XLA_PYTHON_CLIENT_MEM_FRACTION = "0.3" } }
```

### Reproducibility

For comparable results across runs:

- Always benchmark on the **same machine** with the same GPU.
- Use a **clean worktree** so ASV can tag results with the exact commit hash.
- Call `jax.clear_caches()` in `teardown()` to prevent trace caching from leaking between
  benchmarks.

## See Also

- [ASV documentation](https://asv.readthedocs.io/) — full reference for configuration
  and CLI
- [Development: Benchmarking](../development/benchmarking.md) — how pylcm's own
  benchmarks are structured and published
