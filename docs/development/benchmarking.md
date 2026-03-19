---
title: Benchmarking
---

# Benchmarking

pylcm uses [ASV (Airspeed Velocity)](https://asv.readthedocs.io/) to track
performance across commits. Benchmarks run locally on GPU hardware and results are
published to a [dashboard](https://open-econ.org/pylcm-benchmarks/).

## Running Benchmarks

```bash
# Run all benchmarks (GPU recommended)
pixi run -e tests-cuda13 asv-run

# Quick smoke test (not saved)
pixi run -e tests-cuda13 asv-quick

# Compare two commits
pixi run -e tests-cuda13 asv-compare HEAD~1 HEAD

# Preview dashboard locally
pixi run -e tests-cuda13 asv-preview
```

The `asv-run` task guards against dirty worktrees — commit or stash changes first.

## Benchmark Scenarios

| File | What it benchmarks |
|---|---|
| `bench_precautionary_savings.py` | Solve (varying grid sizes), simulate (varying subjects), solve+simulate, lin vs irreg grid comparison |
| `bench_mortality.py` | Mortality model — solve + simulate |
| `bench_mahler_yum.py` | Mahler & Yum (2024) replication (GPU only) |

Each benchmark tracks three metrics:

- **`time_*`** — execution time (after JIT warmup)
- **`peakmem_*`** — peak memory usage
- **`track_warmup`** — JIT compilation time

## Publishing Results

After running benchmarks, publish them to the dashboard:

```bash
pixi run -e tests-cuda13 asv-publish
```

This generates the ASV HTML dashboard and pushes results to the
[OpenSourceEconomics.github.io](https://github.com/OpenSourceEconomics/OpenSourceEconomics.github.io)
repo under `pylcm-benchmarks/`. A persistent clone is kept in `.benchmark-site/`
(gitignored) to avoid re-cloning on every publish.

## Profiling

ASV includes built-in profiling support for drilling into bottlenecks:

```bash
# Profile a specific benchmark
asv profile "bench_precautionary_savings.TimeSolve.time_solve" HEAD^!
```

This runs the benchmark under cProfile and opens the results in snakeviz.

## CI Check

The `benchmark-check` workflow runs on every pull request. It verifies that benchmark
results exist for at least one commit in the PR branch.

- **Passes** if results exist for the HEAD commit
- **Passes with warning** if results exist for an older PR commit (a PR comment lists
  commits since the last benchmark)
- **Fails** if no commit in the PR has benchmark results

To satisfy the check:

```bash
pixi run -e tests-cuda13 asv-run
pixi run -e tests-cuda13 asv-publish
```

## Adding New Benchmarks

Create a new `bench_*.py` file in the `benchmarks/` directory. Benchmarks use ASV's
class-based API:

```python
import time


class TimeMyModel:
    timeout = 600

    def setup(self):
        self.model = ...
        self.model_params = ...
        # JIT warmup (timed separately)
        start = time.perf_counter()
        self.model.solve(self.model_params, log_level="off")
        self._warmup_time = time.perf_counter() - start

    def time_solve(self):
        self.model.solve(self.model_params, log_level="off")

    def peakmem_solve(self):
        self.model.solve(self.model_params, log_level="off")

    def track_warmup(self):
        return self._warmup_time

    track_warmup.unit = "seconds"
```

Use the `params` and `param_names` class attributes to vary grid sizes or other
parameters.
