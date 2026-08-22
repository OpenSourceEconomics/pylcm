---
title: Solvers and capabilities
---

# Solvers and capabilities

All solver-related names live under `lcm.solvers`:

```python
from lcm.solvers import DCEGM, EGM, GridSearch, NBEGM, NEGM, NNBEGM
```

A solver object contains numerical configuration. Specialized economic roles are
declared on the regime.

## Regime pairing

| Regime declaration               | Accepted solver family                                     |
| -------------------------------- | ---------------------------------------------------------- |
| `Regime`                         | `GridSearch` or another non-margin `Solver`                |
| `ConsumptionSavingsRegime`       | `GridSearch` or `OneMarginSolver`: `EGM`, `DCEGM`, `NBEGM` |
| `NestedConsumptionSavingsRegime` | `GridSearch` or `TwoMarginSolver`: `NEGM`, `NNBEGM`        |

Collective regimes currently require `GridSearch`. `NBEGM` and `NNBEGM` reject EV1 taste
shocks. Model construction validates the concrete solver's remaining prerequisites.

## Constructors

### `GridSearch`

```python
GridSearch()
```

Evaluates the complete state-action product and applies constraints directly. It is the
broadest route and the default solver on `Regime`.

### `EGM`

```python
EGM(savings_grid=...)
```

Plain one-margin EGM. The liquid state is cash-on-hand and the post-decision function
must implement the validated identity between liquid state, consumption, and savings. No
upper envelope is taken.

### `DCEGM`

```python
DCEGM(
    savings_grid=...,
    envelope=ExactEnvelope(),
    refined_grid_factor=2.0,
    n_constrained_points=20,
    stochastic_node_batch_size=0,
)
```

Solves a liquid margin conditional on discrete choices and takes an upper envelope.
`refined_grid_factor` controls policy read-out density, `n_constrained_points` the
borrowing-corner segment, and `stochastic_node_batch_size` bounds stochastic-node
workspace. Envelope options are documented in [Upper envelopes](envelopes.md).

### `NEGM`

```python
NEGM(inner=..., outer_grid=..., outer_batch_size=0)
```

Runs the bound `DCEGM` inner solve for every finite outer-grid node and includes the
keeper. A positive `outer_batch_size` folds candidates in chunks; zero processes the
full outer grid at once.

### `NBEGM`

```python
NBEGM(
    savings_grid=...,
    jump_read="one_sided",
    stochastic_node_batch_size=0,
    envelope_segment_block_size=0,
    envelope_arithmetic="certified",
    interval_batch_size=0,
    cell_block_size=0,
    branch_batch_size=0,
    probe_failure="reject",
)
```

Solves supported declared non-convex budgets and smooth discrete branches. `jump_read`
selects one-sided topology-preserving continuation reads or a faster bridged finite-grid
read. The batching fields stream distinct numerical axes without changing the result.
`envelope_arithmetic` selects certified or ordinary candidate ownership.
`probe_failure="assume_declared"` turns an unexecutable structural probe into an author
assertion and warning; it does not relax the mathematical prerequisite.

### `NNBEGM`

```python
NNBEGM(inner=..., outer_search=..., branch_aggregator=...)
```

Nests an `NBEGM` liquid solve inside a configurable outer search. The inner solver must
use a bridged carry compatible with the outer fold. See
[Outer search and branch aggregation](outer_search.md).

## Capability is validated, not inferred from class names

The selected solver inspects finalized declarations before numerical lowering. A model
that violates its state/action count, budget form, constraint route, continuation
layout, shock, or boundary assumptions fails during `Model(...)` or the first
parameter-dependent validation. There is no supported “try EGM and see whether it runs”
workflow.

Start with [Choosing a solver](../user_guide/choosing_a_solver.md). The mathematical map
is [Solver families](../methods/solver_families.md).
