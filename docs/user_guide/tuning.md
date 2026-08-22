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

## Stream work—or change only the commit stride

Some controls reduce live intermediates. Grid `batch_size`,
`stochastic_node_batch_size`, `envelope_segment_block_size`, `subject_batch_size`, and
any solver field whose Reference contract explicitly says it streams an evaluation axis
can lower temporary workspace. The exact effect still depends on retained banks and
downstream folds; for example, `NEGM.outer_batch_size` can lower temporary evaluation
memory without capping the retained candidate bank.

NBEGM's `interval_batch_size`, `cell_block_size`, and `branch_batch_size` are different.
Their production kernels evaluate a fixed profile window on every loop iteration. The
fields select how many completed rows are committed before the next iteration; they do
not change the compiled vector width or establish a peak-memory ceiling. `0` selects the
largest admitted commit stride, capped by the fixed profile window, and is a
one-iteration whole-axis pass only when the axis fits in that window. A smaller stride
can increase iteration count without reducing the fixed per-iteration workspace.

First identify which contract the field has in Reference. For a true streaming control,
choose the largest chunk that meets the measured memory target and verify values. For a
commit-stride control, tune only after measuring scheduling and runtime; do not use it
as a memory budget.

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
- Distinguish true streaming controls, retained banks, and fixed-window commit strides
  before tuning a memory limit.
- Shard only supported discrete axes and verify device visibility.
- Measure compilation separately from execution.
- Treat large-GPU speedups and solver break-even points as empirical.
- Record the complete configuration with every timing.
