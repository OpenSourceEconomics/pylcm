---
title: Performance and memory tuning
---

# Performance and memory tuning

Tune a measured model, not an abstract solver. Correctness comes first: choose a solver
whose assumptions represent the economics, run the model at `log_level="debug"`, and
compare a reduced problem with grid search before optimizing it.

The performance workflow has four measurements:

1. cold environment and cold compilation cache;
1. cold model compilation in an installed environment;
1. warm execution in the same process;
1. peak host and device memory.

Record model size, precision, device, solver configuration, and whether the compilation
cache was warm. Without those fields, two timings are not comparable.

## First locate the limiting axis

List the sizes of:

- the state-grid product;
- the action-grid product;
- discrete branches;
- stochastic nodes;
- outer candidates in a nested solve;
- periods and active regime shapes;
- simulated subjects.

The [scaling discussion](../methods/performance_scaling.md) explains how those axes
enter each solver family. The largest declared grid is not necessarily the largest
intermediate: a product or envelope matrix can dominate.

## Reduce discretization only when accuracy permits

Fewer nodes reduce work and memory, but the right grid is an economic approximation
choice. Inspect value and policy changes as grids are refined. Put resolution near
curvature, boundaries, or regions visited frequently in simulation.

`PiecewiseLinSpacedGrid` and `PiecewiseLogSpacedGrid` control density around known
locations. They do not declare a budget kink or cliff to NBEGM; use the structured
[budget declarations](../methods/nonconvex_budgets.md) for that.

## Stream work to reduce peak memory

Grid `batch_size=k` processes an axis in chunks. Solver-specific controls stream
stochastic nodes, cells, branches, envelope segments, intervals, or outer candidates.
`subject_batch_size` does the same for simulated people.

These controls are **value-invariant memory knobs**, not time-neutral promises. Smaller
chunks retain fewer intermediates but add loop or dispatch overhead and may underfill an
accelerator. Use the largest chunk that fits and remeasure wall time.

Typical order:

1. leave batching at zero for the baseline;
1. identify the intermediate or axis causing the memory limit;
1. split that axis into the fewest chunks that fit;
1. verify values are unchanged;
1. measure the new compile and execution time.

Exact solver fields are in [Solvers and capabilities](../reference/solvers.md),
[Upper envelopes](../reference/envelopes.md), and
[Outer search](../reference/outer_search.md).

## Distribute independent discrete state work

`distributed=True` shards a supported discrete grid over visible devices. Continuous
grids reject distribution because their interpolation needs the full coordinate axis. A
grid cannot be both batched and distributed; if a shard remains too large, batch a
different axis.

Before solving, verify the resources actually visible to JAX:

```python
import jax

assert jax.device_count() == expected_devices
```

A larger GPU can run larger chunks and may benefit from more concurrent independent
work. It does not automatically shorten a workload made of small sequential kernels.
Measure occupancy and memory rather than extrapolating from device memory alone.

## Batch forward simulation

`model.simulate(subject_batch_size=k, ...)` bounds the subject workspace and offloads
completed chunks to host. Random keys are assigned by global subject index, so changing
the batch size does not change simulated draws.

If `Model(n_subjects=n)` was constructed, a matching first simulation can compile for
that population/chunk shape ahead of execution and cache it. Reuse requires stable
parameter shapes and dtypes.

## Reuse compilation

pylcm enables a persistent JAX compilation cache by default. Check:

```python
import jax

print(jax.config.jax_compilation_cache_dir)
```

A `None` value means no persistent cache. `JAX_COMPILATION_CACHE_DIR` chooses the full
path; `LCM_COMPILATION_CACHE_NAME` chooses a project leaf under the default root.
Compilation keys still change when program shapes or the software environment change.

Do not create JAX device arrays at module import solely for constants. Keep tables as
NumPy arrays and convert inside traced functions; this avoids initializing a device in
processes that only import the model.

Runtime environment controls are listed in
[Runtime, results, and persistence](../reference/runtime_and_results.md).

## Benchmark the decision you face

For a solver comparison, hold the economic model and accuracy target fixed. Report:

- cold compile and warm execution separately;
- peak host and device memory;
- value and policy discrepancies;
- precision and hardware;
- grid, envelope, batching, and outer-search settings.

Use the external
[LCM solver benchmarks](https://github.com/OpenSourceEconomics/lcm-solver-benchmarks)
for evolving shared evidence. The pylcm package's own regression benchmarks belong to
the [Development](../development/benchmarking.md) chapter.

## Checklist

- Validate at `log_level="debug"` before tuning.
- Make solver choice an economic-representation decision.
- Refine grids against an accuracy target.
- Use the fewest chunks that meet the memory budget.
- Shard only supported discrete axes and verify device visibility.
- Measure compilation separately from execution.
- Treat large-GPU speedups and solver break-even points as empirical.
- Record the complete configuration with every timing.
