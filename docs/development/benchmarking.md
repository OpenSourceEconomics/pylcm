---
title: Benchmarking
---

# Benchmarking

pylcm uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) to track
performance across commits. Benchmarks run locally on GPU hardware and results are
published to a [dashboard](https://open-econ.org/pylcm-benchmarks/).

## Running Benchmarks

```bash
# Run all benchmarks (GPU recommended)
pixi run -e tests-cuda13 benchmarks

# Run and save results for later comparison
pixi run -e tests-cuda13 benchmarks-save

# Compare against the last saved run
pixi run -e tests-cuda13 benchmarks-compare
```

Benchmarks are excluded from normal test runs (`pixi run tests`) via
`--benchmark-disable` in `pyproject.toml`.

## Benchmark Scenarios

| File | What it benchmarks |
|---|---|
| `bench_precautionary_savings.py` | Solve (varying grid sizes), simulate (varying subjects), solve+simulate, lin vs irreg grid comparison |
| `bench_mortality.py` | Mortality model — solve + simulate |
| `bench_mahler_yum.py` | Mahler & Yum (2024) replication (GPU only) |

Each benchmark warms up JIT compilation before timing.

## Publishing Results

After saving benchmarks, publish them to the dashboard:

```bash
pixi run -e tests-cuda13 benchmarks-publish
```

This pushes results to the
[OpenSourceEconomics.github.io](https://github.com/OpenSourceEconomics/OpenSourceEconomics.github.io)
repo under `pylcm-benchmarks/`. A persistent clone is kept in `.benchmark-site/`
(gitignored) to avoid re-cloning on every publish.

## CI Check

The `benchmark-check` workflow runs on every pull request. It verifies that benchmark
results exist for at least one commit in the PR branch.

- **Passes** if results exist for the HEAD commit
- **Passes with warning** if results exist for an older PR commit (a PR comment lists
  commits since the last benchmark)
- **Fails** if no commit in the PR has benchmark results

To satisfy the check:

```bash
pixi run -e tests-cuda13 benchmarks-save
pixi run -e tests-cuda13 benchmarks-publish
```

The required machine(s) are configured in `benchmark-config.json` on the org site repo.

## Adding New Benchmarks

Create a new `bench_*.py` file in the `benchmarks/` directory. Each benchmark is a
regular pytest function that receives the `benchmark` fixture:

```python
def test_solve_my_model(benchmark):
    model = ...
    params = ...
    # Warm up JIT
    model.solve(params, log_level="off")
    benchmark(model.solve, params, log_level="off")
```

Use `@pytest.mark.parametrize` to vary grid sizes or other parameters.
