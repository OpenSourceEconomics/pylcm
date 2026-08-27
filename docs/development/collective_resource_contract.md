---
title: Resource contract for collective and gated models
---

# Resource contract for collective and gated models

A collective regime and a value-dependent transition each buy something a singleton
model does not have, and each costs something a singleton model does not pay. This page
names the workloads those costs are measured on, says which axis each one is allowed to
grow along, and states what a regression is.

It is a contract, not a report: the numbers live in the ASV history, and what is written
down here is the *shape* each cost is allowed to have. A change that moves a level is
reviewed against the history; a change that moves an **order** is a defect whatever the
level.

## The workloads

All of them are in `benchmarks/asv/bench_collective_household.py`, over the marriage
market of `lcm_examples.collective_household`: two singles who marry under mutual
consent, a household with a participation constraint on each partner, and a dissolution
edge keyed by the continuing household.

| Workload                    | Class                                                         | What it isolates                                                                                                                |
| --------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Model construction          | `CollectiveHouseholdConstruct`                                | The phase scan, the lowering of the collective declarations, and per-edge parameter discovery, with nothing traced or compiled. |
| First-solve assembly        | `CollectiveHouseholdSolve.track_compilation_time`             | Tracing, lowering, and compiling-or-loading every regime's kernels plus one fold per gated edge. See the note below.            |
| Warm solve                  | `CollectiveHouseholdSolve.time_execution`                     | What an estimation loop pays per parameter vector.                                                                              |
| Host memory, solve          | `CollectiveHouseholdSolve.peakmem_execution`                  | Resident peak while backward induction runs.                                                                                    |
| Device memory, solve        | `CollectiveHouseholdSolveGpuPeakMem`                          | Device peak on the same workload.                                                                                               |
| Simulation over cohort size | `CollectiveHouseholdSimulate`, `n_subjects ∈ {1e3, 1e4, 1e5}` | Routing: one gate evaluation per edge per period over the whole population.                                                     |
| Transitive reference depth  | `ReferenceChainSolve`, `depth ∈ {1, 2, 4, 8}`                 | The closure a value constraint opens: link `k` reads link `k-1` in the same period.                                             |

## The budgets

Each budget is an **order**, because that is what a benchmark suite can defend across
machines and backends. A level is defended by the ASV history on one machine.

- **Model construction** is `O(regimes × phases × declarations)` and involves no device
  work. It is allowed to grow with the number of declarations a model makes and with
  nothing else. Construction that grew with a *grid* size would mean a grid was
  materialized during the scan.
- **First-solve assembly** is `O(regimes × periods)` programs plus
  `O(edges × period groups)` folds and gate evaluators. Gate evaluators are compiled
  ahead of time together with the decision kernels when `Model(n_subjects=N)` is set, so
  a fixed batch size pays for them once, before the first simulated period, rather than
  in it.
- **Warm solve** is `O(periods × regimes × cells)`, the same order as a singleton model
  of the same total grid size. A collective regime multiplies the cell count by its
  stakeholder count; a gated edge adds one fold over the target's grid per period. It
  may not grow with the *number of subjects*, which appears nowhere in the solve.
- **Simulation** is `O(periods × (regimes + edges) × subjects)` and therefore linear in
  the cohort. The three cohort sizes exist to make a super-linear term visible; a slope
  above one between adjacent points is the regression this workload is for.
- **Memory**, host and device, is `O(largest single V array + working set)` and does not
  accumulate across periods. Backward induction frees each period's intermediates, so a
  peak that grew with the *number of periods* would mean it stopped.
- **The shape cache** is bounded by the model, not by the run. A gate evaluator's
  population call is keyed on `(callable, cohort size)` and every other program on
  `(callable, dedup key)`, all of which are properties of the model and its declared
  batch size. Repeated `solve()` / `simulate()` calls at one cohort size may not add
  entries; a cache that grew per call would recompile per call.

## What the first-solve number is, and is not

pylcm enables JAX's persistent compilation cache, and the benchmark suite shares one, so
`track_compilation_time` is not a compilation time. It covers tracing and lowering, and
then *either* compiling each program or loading it from the cache — and which of those
two it did is invisible in the number.

Measured on this workload, on one GPU backend, with the cache directory asserted empty
before the run and populated after: the first solve costs about twice what the same
solve costs once its programs are cached, so a cache hit covers a little under half of
it. The saving grows with the model, because compilation grows with program size while
tracing and lowering grow with the number of programs.

Two things follow, and both are about how to *read* a change in this line:

- A movement here is a movement in assembly cost, not in compiler work, unless the cache
  state is known to have been the same on both sides. Pin the cache — point
  `JAX_COMPILATION_CACHE_DIR` at a fresh directory — for any comparison meant to be
  about compilation.
- A number from one machine is not comparable to a number from another whose cache holds
  a different set of programs. The ASV history is per machine for this reason.

## Where a pointwise reoptimization mode would sit

`off_grid="pointwise"` reads the operands at the landing point and gates them there, in
both phases, using the kernels the model already has. A future reoptimization mode —
recomputing the target's own optimum at the realized point — is a different kernel and
is kept statically separate from the default one: it must be selectable per edge, must
not appear in a model that did not ask for it, and carries its own entries in this table
before it is offered. Its cost is `O(subjects × target action grid)` per edge per
period, which is a different order from the default's `O(subjects)`, so the two may not
share a budget line.

## Running them

```bash
pixi run -e benchmarks-cuda12 asv-quick     # one repetition, for a smoke check
pixi run -e benchmarks-cuda12 asv-run       # the tracked run; refuses a dirty worktree
pixi run -e benchmarks-cuda12 asv-compare   # against a previous commit
```

The GPU peak-memory companions need a CUDA environment; on a CPU-only machine they are
skipped and the host `peakmem_*` numbers stand alone.
