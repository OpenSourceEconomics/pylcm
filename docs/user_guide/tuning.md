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

### Fix it in one regime only

A bare integer applies to every regime declaring the axis, which is the wrong instrument
when two regimes have opposite shapes: a width that keeps a 16-cell, 2048-action regime
inside memory takes a 4096-cell, two-action regime from eight chunks to 512. Give the
axis a mapping from regime name to width instead, and only the regimes it names are
fixed:

```python
ExecutionConfig(axis_widths={"cell": {"heavy": 8}})
```

Regimes the mapping does not name keep the width the planner chooses for them, per
regime and per period. The two forms may be mixed across axes. Because only the solve
phase plans per regime, an axis that only simulation programs declare — `subject`, for
one — takes the bare-integer form; a per-regime width for it is refused at model build,
as is one naming a regime the model does not declare.

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

## Distribute state work

Declare a discrete state at model level, then name it in
`ExecutionConfig(sharded_states=("preference",))` to shard its axis. Select devices with
`ExecutionConfig(devices=(0, 1, 2, 3), ...)`; omitting `devices` uses all devices
visible to JAX. Solver-specific restrictions also apply; see
[Solvers and capabilities](../reference/solvers.md).

Sharding is resolved per regime, from the state each regime actually carries. A regime
whose DAG reads the state carries its grid axis and runs on the submesh that axis
defines. A regime that never reads it --- so DAG pruning drops it there --- publishes a
value without that axis and runs on a single device; the planner moves the values that
cross between the two placements. That holds for a non-terminal regime as much as for a
terminal one, so a shock read through working life and dropped in retirement shards the
working-life regimes alone. Only a state *every* regime prunes is refused, because then
no axis is left to spread. The stricter continuous route additionally requires the state
in every regime.

Ordinary singleton hard-max `GridSearch` also supports one model-level `LinSpacedGrid`
as the sole sharded state. Every regime must retain that same static grid. Unsharded
states may include:

- Concrete discrete grids.
- At most one static `PiecewiseLinSpacedGrid` interpolation coordinate per regime.
- Fully specified, unconditioned `RouwenhorstAR1Process` nodes and unfolded
  `NormalIIDProcess` nodes with Gauss-Hermite quadrature.
- Carried states declared with a `LinSpacedGrid` simulation domain. Their solve
  imputation does not add a value-array dimension.

Runtime state grids, folded processes, other process families, mixed solvers, collective
or gated routes, same-period references and taste shocks remain outside this route.
Runtime action grids keep their ordinary GridSearch semantics. Process nodes retain the
ordinary index-based continuation read; sharding does not enable process-aware off-grid
interpolation. Additional continuous coordinates keep their named positions, so the
sharded axis need not be trailing.

For example, a 24-point assets axis can use eight selected devices, with three assets
coordinates on each device. Logical value-axis order stays unchanged. Interpolation
still needs the complete target value: the planner explicitly replicates each needed
continuation on its consumer mesh before the numerical core runs. Retained shards,
replicas and transfer workspace all count toward admission. Budgeted continuous sharding
currently requires every regime and transfer to use the complete selected device set.
Transfer workspace is reserved conservatively across the period, even when individual
copies finish earlier. These copies can limit capacity and add communication time; eight
shards do not imply an eightfold memory reduction or speedup. Reducing a cell width
cannot remove the full-continuation storage requirement.

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

With an explicit memory budget, solve execution waits for earlier compiled work that
uses any of the next core's execution or transfer devices. Work on disjoint device sets
can remain asynchronous. A shared source array can therefore serialize cores assigned to
different regime submeshes. The wait covers returned auxiliary arrays and declared
copies as well as values; it does not bound memory used by compiler autotuning.

(set-a-device-memory-budget)=

## Set a device-memory budget

`ExecutionConfig(device_memory_bytes=...)` declares a per-device ceiling for the
compiler's reservation plus the accounted live residency, and turns on budget admission:
a plan that does not fit is refused with `ExecutionPlanningError` rather than left to
the allocator. The default `None` omits admission entirely. Pass a positive integer
resolved for the hardware you will run on --- the population count is not a memory
budget.

You may pass a device's whole allocator pool limit. The model does not plan against all
of it: `device_memory_headroom_fraction` (default `0.15`) is the share of each selected
device's pool kept outside the ceiling, an operational margin for storage the
represented accounting does not see --- collective-communication buffers, library
workspaces such as cuBLAS, the driver context, allocator fragmentation. The effective
ceiling is the smaller of your request and every selected device's pool limit less that
fraction. The fraction is a policy awaiting workload validation, not a measured
requirement: if you have measured your own envelope, set it to `0.0` and plan against
the whole pool. It must be an exact float in `[0, 1)`, and it moves only the ceiling ---
every reservation and residency figure compared against that ceiling is unchanged.

```python
execution_config = ExecutionConfig(
    devices=(0, 1),
    device_memory_bytes=40 * 2**30,
    device_memory_headroom_fraction=0.0,  # Only with a measured envelope.
)
```

Resolving the budget is logged, so you can see which of the two bounds actually bound:

- The devices did not cap the request: one summary line at `log_level="progress"` and
  `"debug"`, naming the request, the fraction, each selected device's pool limit and the
  effective ceiling.
- The devices capped the request: the same line as a **warning**, visible from
  `log_level="warning"` up. Read it --- the plan was admitted against less memory than
  you asked for.

Every admission refusal repeats both budgets: when the devices capped the request, the
message appends the effective bytes, the requested bytes, and the headroom fraction that
separates them, so a refusal is never ambiguous about which ceiling it was measured
against. Field-by-field contracts are in
[Runtime, results, and persistence](../reference/runtime_and_results.md).

## Batch forward simulation

Set `ExecutionConfig(axis_widths={"subject": k})` on the model to process subjects in
chunks. The inner compiled subject width remains fixed at `k`; the outer chunk is
clamped to the population and rounded up to a multiple of the subject-device count. For
example, width three uses outer chunks of four on four devices while retaining an inner
width of three. Completed chunks are offloaded to host. Random keys retain their
original population and global subject indices, so changing the width does not change
simulated draws.

With a positive `device_memory_bytes` budget, every call is planned by the same
top-first outer-cohort search, whether or not `axis_widths["subject"]` is pinned.
Unsupported budgeted replay routes still refuse execution.

A pinned `axis_widths["subject"]` fixes the anchor width the frontier doubles from.
Without a pin, the existing single-axis workspace search proposes the widest
representable subject width the budget admits as that anchor; the frontier and its map
selection then proceed identically either way. The anchor only seeds the search — it is
not a memory verdict, and the admitted extent may use a different inner map than the
anchor if a fallback runs.

The planner constructs device-aligned outer sizes by repeatedly doubling the anchor
width, clamping to the population, and suppressing duplicates. The largest size covers
the population. It derives the preferred full inner-width map from the anchor's axis
declarations, without compiling or admitting the anchor size first. It then profiles the
largest size. If that complete profile fits, it returns immediately: one whole-chunk
profile, no smaller-shape sweep, and no predicted memory verdict.

After a largest-size refusal, it selects the original full/bootstrap map at the anchor
size. The full map is tried first; a distinct conservative bootstrap map is tried only
if the full anchor fails. That complete admitted map then stays frozen while the
remaining outer sizes are tried largest first. The first fitting descending candidate
wins; otherwise the admitted anchor is retained. An already rejected extent/map pair is
not repeated. If fallback changes the map, the largest size must receive its own new
profile. If neither anchor map fits, the search refuses; unsearched configurations may
still fit. There is no outer-by-inner width sweep or assumption of monotone memory use.

For K distinct sizes, a fitting preferred full-population candidate uses one complete
profile. If the full map remains selected, the worst case is K profiles. A distinct
bootstrap fallback uses at most K + 2 profiles (two for a singleton frontier). This
trade-off favors populations that fit whole; models fitting only small cohorts can pay
more planning cost than an ascending search. Compiler exceptions are not caught as
memory refusals. Compilation and its host-memory cost are not bounded by the simulated
payload's device budget.

For example, 226,848 subjects, inner width 2,048 and eight subject devices give the
geometric sizes 2,048, 4,096, 8,192, 16,384, 32,768, 65,536, 131,072 and 226,848. The
last size is profiled first. Success selects one cohort immediately, with the inner map
still fixed. Every attempted size uses its own compiled requirements, retained owners,
padding, and output/assembly reservations. Different outer shapes are not
interchangeable executables. Selection is largest-first feasibility, not a measured
speed optimum.

Receipts use frontier_version=3. Their candidates remain in ascending geometric order;
attempts record actual profile order, including refusals and map fallback. profile_count
counts complete profiles, not backend compilations. Current inputs and resident owners
are checked on each call, and live admission is still checked before chunk execution.

The bound includes the retained solution, full-population inputs and RNG workspace,
pending period owners, published results, padding, and final assembly. It is
conservative about future aliases; it does not estimate an allocator optimum. On CPU,
retaining all chunks and assembling the final result can impose a floor that narrower
chunks cannot remove. CPU assembly after GPU offload uses host RAM outside the GPU
ceiling.

Finite NNBEGM replay includes candidate preparation, diagnostics and canonical ranking
in this bound. The prepared bank has one row per subject in the outer chunk and keeps
the full published candidate extent. A smaller inner width does not remove that bank's
storage requirement. Addressed policy copies and their transfer overlap are counted
alongside retained originals.

Parameterized grids and user entry laws can still allocate eagerly while shared inputs
are completed. Their resulting arrays are counted, but this entry work is not
pre-admitted by the chunk profiles. Budgeted host replay routes remain refused until
their complete stages and owners have profiles.

The same fixed `seed` gives the same per-subject EV1 taste-shock choices across subject
chunk widths and repeated calls. Keep the seed, parameters, initial conditions, and
model fixed when checking that invariance.

Each `simulate(...)` call takes its population from the initial conditions. A model can
simulate different populations; runtime executors reuse compiled programs when their
shapes and other compilation inputs match. Parameter and support validation still run
for every call.

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
- Shard only supported state axes and verify actual shard indices and device use.
- Measure compilation separately from execution.
- Treat large-GPU speedups and solver break-even points as empirical.
- Record the complete configuration with every timing.
