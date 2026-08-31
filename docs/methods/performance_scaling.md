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
| Stochastic nodes     | Continuation evaluations before expectation               |
| Outer candidates     | Complete inner solves in a nested method                  |
| Refinement budget    | Adaptive outer search, where node count is data-dependent |
| Period/regime shapes | Number of distinct programs JAX may compile               |
| Subjects             | Simulation batch size and compiled shape                  |

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

### Maintainer-only fused NB-EGM replay

The private fused NB-EGM period replay is a compile-only architecture experiment. It
compares the current split cores with their existing full continuation-to-envelope
calculation placed inside one JIT boundary. Its question is narrowly whether that
boundary changes the full stacks' compiler-visible lifetime. It is not a tile-local
solver implementation or an architecture-completion claim.

A period capture preserves logical pytrees, values, and production array shapes, but its
pickle round trip does not preserve device sharding. The analyzer therefore lowers both
forms using default backend placement after the capture round trip. Its report records
this as machine-readable shape provenance and layout fidelity, and explicitly states
that production sharding was not preserved. The resulting compiler byte counts are not
production-layout memory measurements. Runtime and peak-memory claims require a
representative execution with the intended production placement.

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
