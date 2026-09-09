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

## Fix a planner axis width

A solver declares its execution axes by name. `ExecutionConfig(axis_widths=...)` fixes
the width used by each applicable compiled program or host-dispatch loop:

```python
from lcm import ExecutionConfig, Model

model = Model(
    regimes=regimes,
    ages=ages,
    regime_id_class=RegimeId,
    execution_config=ExecutionConfig(axis_widths={"action_product": 8}),
)
```

A width smaller than the axis bounds how many entries are processed together; a width at
or above the extent selects the whole axis in one chunk. Compiled axes vectorize that
chunk; a host-dispatch axis bounds pending node calls. Lower values can reduce live
intermediates at the cost of more sequential execution. They do not cap surrounding
arrays, retained candidate banks, compilation memory, or total device memory.

Which axis names exist depends on which solver a regime uses; each solver's section in
[Solvers and capabilities](../reference/solvers.md) names the axes it declares.

| axis              | declared by                                                                                          |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `action_product`  | the flattened Cartesian action product of a streamed `GridSearch` core                               |
| `stochastic_node` | child stochastic nodes folded by `DCEGM`, ride-along `NBEGM`, and their nested solvers               |
| `cell`            | independent state cells tiled by `GridSearch`, `DCEGM`, ride-along `NBEGM`, and their nested solvers |
| `interval`        | liquid intervals folded by ride-along `NBEGM`, including eligible `NNBEGM` inner programs            |
| `branch`          | discrete-action subproblems mapped by ride-along `NBEGM`, including eligible `NNBEGM` inner programs |
| `savings_point`   | exogenous savings nodes tiled by `DCEGM`, including `NEGM` inner programs                            |
| `euler_point`     | Euler nodes tiled by `DCEGM`, including `NEGM` inner programs                                        |
| `envelope_cell`   | independent envelope cells tiled by `DCEGM`, including `NEGM` inner programs                         |
| `outer_candidate` | the exogenous outer post-decision nodes of a nested outer search                                     |

These are solver capability names; a particular program declares only its applicable
axes. Plain `EGM` declares no execution-width axis. Nested solvers inherit the relevant
inner axes; their outer search determines how `outer_candidate` is used. See the
generated capability table in [Solvers and capabilities](../reference/solvers.md).

Choose the largest width that meets the measured memory target, then verify values and
runtime against the whole-axis setting on the model and backend you will use. Leaving an
axis out of the mapping lets the planner choose, which is what a device-memory budget
asks it to do.

## Choose execution widths independently of grids

Grids specify economic support and interpolation. Set execution widths on the model with
`ExecutionConfig(axis_widths=...)`. The effect depends on retained arrays and downstream
folds; for example, `outer_candidate` can lower a nested solver's temporary evaluation
memory without capping its retained candidate bank.

For `GridSearch`, `cell` tiles the flattened product of the states evaluated inside each
sharded slice. Sharded states stay outside this product. `action_product` separately
streams eligible hard-max action reductions; EV1 and collective models keep their
canonical dense action reductions while tiling state cells. `DCEGM` uses the applicable
axes listed above.

NBEGM also owns no compiled-width fields. Use its applicable axes in the table above:
`interval` streams the continuation read together with the stable-identity candidate
fold, `branch` maps the discrete subproblems before their maximum, and `cell` tiles
independent ride cells inside each co-mapped carry slice. A route declares only the
nontrivial meshes it consumes. The interval stream has no separate segment-width loop.
Smaller widths can reduce live intermediates at the cost of sequential execution; they
do not cap surrounding arrays, retained candidate banks, compilation memory, or total
device memory.

Exact solver fields are in [Solvers and capabilities](../reference/solvers.md),
[Upper envelopes](../reference/envelopes.md), and
[Outer search](../reference/outer_search.md).

## Distribute independent discrete state work

Declare a discrete state at model level, then name it in
`ExecutionConfig(sharded_states=("preference",))` to shard its axis. Select devices with
`ExecutionConfig(devices=(0, 1, 2, 3), ...)`; omitting `devices` uses all devices
visible to JAX. Continuous states cannot be sharded because interpolation reads their
full coordinate axis. Solver-specific restrictions also apply; see
[Solvers and capabilities](../reference/solvers.md).

If a shard remains too large, reduce an applicable execution width. For example,
`ExecutionConfig(sharded_states=("preference",), axis_widths={"cell": 32})` keeps the
preference axis sharded and tiles the remaining GridSearch state product.

Before solving, verify the resources actually visible to JAX:

```python
import jax

assert jax.device_count() == expected_devices
```

A larger GPU can run larger chunks and may benefit from more concurrent independent
work. It does not automatically shorten a workload made of small sequential kernels.
Measure occupancy and memory rather than extrapolating from device memory alone.

## Batch forward simulation

Set `ExecutionConfig(axis_widths={"subject": k})` on the model to process subjects in
chunks. The inner compiled subject width remains fixed at `k`; the outer chunk is
clamped to the population and rounded up to a multiple of the subject-device count. For
example, width three uses outer chunks of four on four devices while retaining an inner
width of three. Completed chunks are offloaded to host. Random keys retain their
original population and global subject indices, so changing the width does not change
simulated draws.

With a device-memory budget and no fixed subject width, simulation selects the widest
outer candidate whose complete retained storage and compiled stages fit. It tries the
available inner widths before shrinking a chunk. The bound includes the retained
solution, full-population inputs and RNG workspace, pending period owners, published
results, padding, and final assembly. It is conservative about future aliases; it does
not estimate an allocator optimum. On CPU, retaining all chunks and assembling the final
result can impose a floor that narrower chunks cannot remove. CPU assembly after GPU
offload uses host RAM outside the GPU ceiling.

Parameterized grids and user entry laws can still allocate eagerly while shared inputs
are completed. Their resulting arrays are counted, but this entry work is not
pre-admitted by the chunk profiles. Budgeted host replay routes remain refused until
their complete stages and owners have profiles.

The same fixed `seed` also gives the same EV1 taste-shock choices in lazy and
ahead-of-time simulation. Subject chunking and `Model(n_subjects=...)` change
compilation and workspace shape, not which per-subject Gumbel key is used. Keep the
seed, parameters, initial conditions, and model fixed when using that invariance as a
regression check.

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
- Distinguish true streaming controls, retained banks, inert requests, and active
  admitted branch strides before tuning.
- Shard only supported discrete axes and verify device visibility.
- Measure compilation separately from execution.
- Treat large-GPU speedups and solver break-even points as empirical.
- Record the complete configuration with every timing.
