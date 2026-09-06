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
pixi run -e benchmarks-cuda12 asv-run-and-pr-comment
```

**Main branch** — run benchmarks and publish results to the dashboard:

```bash
pixi run -e benchmarks-cuda12 asv-run-and-publish-main
```

Both workflows run `asv-run` (which requires a clean worktree) followed by their
respective post-processing step.

Individual tasks are also available:

```bash
# Run all benchmarks (GPU required)
pixi run -e benchmarks-cuda12 asv-run

# Quick smoke test (not saved)
pixi run -e benchmarks-cuda12 asv-quick

# Post benchmark comment to current PR (no GPU needed)
pixi run asv-pr-comment

# Compare two commits (no GPU needed)
pixi run asv-compare HEAD~1 HEAD

# Preview dashboard locally (no GPU needed)
pixi run asv-preview

# Publish results to dashboard (no GPU needed)
pixi run asv-publish
```

The `asv-run` and `asv-quick` tasks set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
automatically so JAX allocates GPU memory on demand rather than grabbing it all up
front.

### Exact paired GridSearch measurements

Use the separate paired harness when a GridSearch execution change needs evidence
against one exact base revision. Run it from a clean checkout of the head revision and
provide separate, clean checkouts of both revisions:

```bash
pixi run -e benchmarks-cuda12 grid-search-pair \
    --base-checkout /path/to/base \
    --base-revision <full-base-commit> \
    --head-checkout /path/to/head \
    --head-revision <full-head-commit> \
    --harness-revision <full-harness-commit> \
    --output /path/outside/both/checkouts/grid-search-pair \
    --precision 32 \
    --backend gpu \
    --repeats 4
```

All three revisions must be full 40-character commit hashes. The checkout you run the
task from supplies the harness and is validated against `--harness-revision`, so it need
not be the head checkout; all three checkouts must agree on the lock file and
scenario-owned sources. The output directory must not already exist and must lie outside
the harness, base, and head checkouts.

The harness measures eleven named scenarios. Five are GridSearch routes: singleton hard
max, singleton EV1, collective GridSearch with value-dependent inputs, distributed
co-map, and folded singleton hard max. The other six are the full 18-regime ACA baseline
at each combination of 3 or 6 asset points and 16, 64, or 256 consumption points, named
`aca-a<assets>-c<consumption>`. Pass `--scenario` once per name to measure a subset;
omitting it measures all eleven. The distributed row always uses exactly four CPU
devices; the other rows use the requested backend. Use at least two repeats so execution
alternates base--head on even repeats and head--base on odd repeats; four repeats are
the balanced evidence profile.

Every pair must pass numerical value parity and exact output shape, dtype, and sharding
parity. The harness also requires non-empty HLO and compiler-memory status for every
compiled core, host high-water-mark evidence, and measured peak device memory on GPU
(with an explicit not-applicable record on CPU). It verifies the native execution
declaration for every named target: singleton hard max, distributed co-map, and folded
hard max are planned and streamed; EV1 is deliberately dense to preserve canonical
reduction order; and collective GridSearch is deliberately dense because streaming is
resource-adverse. It also rejects communication collectives in the distributed co-map
head. Raw timings, compiler and device memory, HLO files and digests, layout manifests,
and the base/head ratios are retained under the output directory; `summary.json` is the
entry point for review.

## Benchmark Scenarios

| File                             | What it benchmarks                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `bench_precautionary_savings.py` | Solve (varying grid sizes), simulate (varying subjects), solve+simulate, lin vs irreg grid comparison |
| `bench_mahler_yum.py`            | Mahler & Yum (2024) replication (GPU only)                                                            |
| `bench_aca_baseline.py`          | ACA baseline (18 regimes) end-to-end simulate on benchmark-sized grids (GPU only)                     |

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
pixi run -e benchmarks-cuda12 asv-run-and-pr-comment
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
            "regime_id": jnp.zeros(1_000, dtype=jnp.int32),
        }

        # JIT warmup (timed separately)
        start = time.perf_counter()
        self.model.simulate(
            params=self.model_params,
            initial_conditions=self.initial_conditions,
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
