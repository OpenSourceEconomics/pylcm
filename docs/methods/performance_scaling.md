---
title: Scaling, memory, and hardware
---

# Scaling, memory, and hardware

Correctness narrows the feasible solver set; computational feasibility can still decide
which member to use. Compare compilation, execution, peak memory, and accuracy on the
hardware that will run the production model.

## Sources of growth

A useful first inventory is:

| Axis                 | Why it matters                                            |
| -------------------- | --------------------------------------------------------- |
| State-grid product   | Number of value-function cells                            |
| Action-grid product  | Grid-search candidates per state cell                     |
| Discrete branches    | Separate candidate surfaces and envelope work             |
| Continuation classes | Continuation reads per cell in a branch-carrying NB-EGM   |
| Stochastic nodes     | Continuation evaluations before expectation               |
| Outer candidates     | Complete inner solves in a nested method                  |
| Refinement budget    | Adaptive outer search, where node count is data-dependent |
| Period/regime shapes | Number of distinct programs JAX may compile               |
| Subjects             | Simulation batch size and compiled shape                  |

A discrete branch inside `NBEGM` costs a candidate surface and its share of the
envelope, but not necessarily a continuation read: branches that agree on every action
reaching the continuation (the regime transition, a law of motion, a child's resources,
the discount factor, or, on a per-interval read, a schedule variable) share one read, so
a budget-only action adds branches without adding continuation classes.

For grid search, continuous action grids multiply. For a nested solver, outer candidates
multiply the inner solve — a declared count under a finite outer grid, and a
budget-bounded one under an adaptive mesh. For EGM, a savings-grid construction replaces
one current state-by-action search but interpolation and envelope work remain.

These expressions predict direction, not wall time. Kernel fusion, memory traffic,
compile reuse, and device occupancy can reverse a simple operation-count ranking.

## CPU and GPU tendencies

| Work shape                    | CPU tendency      | GPU tendency                |
| ----------------------------- | ----------------- | --------------------------- |
| Dense static map-reduce       | Viable            | Usually strong              |
| Sequential topology scan      | Often strong      | Often poor                  |
| Query-side segmented envelope | Viable            | Often preferable            |
| Many small compiled shapes    | Lower launch cost | Can be compile/launch bound |
| Large independent batches     | Core parallelism  | Strong if memory permits    |

`GridSearch` is therefore not merely a slow fallback: a modest dense action grid can be
excellent on a GPU. Conversely, an EGM method with sequential scans or many small shapes
can lose despite lower arithmetic complexity.

Eligible JIT GridSearch solve kernels evaluate the canonical action product in bounded
blocks. Each period kernel names every continuation and reference value it reads as an
exact logical target artifact paired with the source core's argument channel and tree
path. Same-period value references, gated-target continuations, and edge-reference
values and parameters therefore remain ordinary dynamic inputs to those blocks, so a
value-dependent declaration does not by itself force dense action materialization.
Ordinary co-mapped state routes preserve device-local continuation reads while streaming
actions. Eligible singleton folded-state routes also stream their action product at each
shock node and then apply the unchanged quadrature reduction; the fold-node axis itself
is still evaluated and reduced in full. The classifier deliberately keeps co-map
intersections with separate same-period or edge-reference channels, trivial action
products, JIT-disabled and raw execution, and all simulation-policy construction dense.
Simulation lies outside the solve classifier. Collective EV1, EV1 with a fold,
collective hard max with a fold, and EV1 without a discrete action are unsupported by
the streamed program. See the canonical
[GridSearch route matrix](../reference/solvers.md#gridsearch-jit-route-matrix). The
planner selects a private width keyword outside the complete model argument namespace,
so model names never force a dense fallback. Blockwise action evaluation preserves the
full represented support; its runtime and memory effect remains empirical.

Before lowering a planned GridSearch core, the engine resolves every declared value read
to one of two private transfer operations. `ALIGNED_LOCAL` passes an array through when
it is already resident on the source core's mesh, retaining the value's own
rank-specific partitioning. `COPY_TO_SOURCE_LAYOUT` makes an explicit copy into the
supported layout on the source core's mesh. The exact same resolved plan transforms the
lowering arguments and the runtime arguments; unexpected shape, dtype, sharding,
address, or conversion combinations fail closed.

Each planned read also authenticates its declared
`(source_regime, source_period, core_key)` against the actual compiled core before its
channel and argument-tree path are resolved. Agreement among declarations cannot make a
different source node authoritative.

The exact declarations also support conservative remaining-consumer accounting. A
logical artifact is counted once per planned dispatch and its count is committed only
after that dispatch returns successfully. Reaching zero means only that an unpinned
artifact is eligible for a later scheduler decision. The current planner does not
physically release, donate, or offload arrays. Dense compatibility routes and consumers
without a complete plan remain pinned, so this bookkeeping is not a graph-wide
peak-memory claim.

A large GPU should not be treated as a faster small GPU automatically. Independent
tests, regimes, branches, subjects, or candidate chunks can improve occupancy, but only
if scheduling keeps aggregate device memory bounded. pylcm's current public controls
mainly stream individual numerical axes; cross-test concurrency remains a workflow
decision.

## Compilation and execution are different costs

JAX compiles programs for concrete shapes. Long lifecycle models with changing active
regimes may produce many programs even when each executes quickly. pylcm enables a
persistent compilation cache by default; repeated processes can reuse entries when the
model and environment produce the same compilation key.

Measure at least:

1. cold environment and cold compilation cache;
1. warm environment and cold model compilation;
1. warm compilation cache;
1. repeated execution in one process;
1. peak host and device memory.

A speed claim that reports only the fourth quantity does not answer whether the model is
practical in estimation or CI.

### Maintainer-only compile-only memory analysis of a captured period

The private period capture can be lowered without being executed. The analyzer compiles
every production core of the captured regime-period and reads the compiler's memory
analysis for each executable: argument, output, temporary, and peak bytes. For a
ride-along NB-EGM period the report covers both retention-scoped programs, so the
values-only program's peak can be compared against the replay program's.

A period capture preserves logical pytrees, values, and production array shapes, but its
pickle round trip does not preserve device sharding. The analyzer therefore lowers every
core using default backend placement after the capture round trip and states so in its
report. The resulting compiler byte counts are not production-layout memory
measurements. Runtime and peak-memory claims require a representative execution with the
intended production placement.

## What is known and what remains empirical

The candidate-growth relationships above are structural. Exact break-even grid sizes,
the fastest envelope backend, useful batch sizes, and benefits from a larger GPU depend
on model topology and hardware. The documentation should not manufacture universal
thresholds.

Use the external
[LCM solver benchmarks](https://github.com/OpenSourceEconomics/lcm-solver-benchmarks)
for evolving comparisons, and benchmark your own model with the workflow in
[Performance and memory tuning](../user_guide/tuning.md). Package maintainers should use
the distinct [development benchmark suite](../development/benchmarking.md).
