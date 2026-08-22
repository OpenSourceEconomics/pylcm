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

## Capability table

| Solver       | Required declaration                                           | Problem shape                                                                 | Hard prerequisites and supported constraints                                                                                                                                                                                                                                                          | Main tradeoff                                                                                    |
| ------------ | -------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `GridSearch` | `Regime` or a specialized regime                               | General discrete-continuous action product                                    | Ordinary callable constraints; no EGM structure required                                                                                                                                                                                                                                              | Broadest representation; work and memory grow with the action product                            |
| `EGM`        | `ConsumptionSavingsRegime` with one `LiquidMargin`             | Smooth, concave one-state/one-action cash-on-hand problem                     | Exactly one continuous state and action; no discrete/process states or actions; resources equal the liquid state; post-decision state equals state minus action; utility does not read the liquid state; default Koopmans aggregator; only a provable post-decision lower bound as a solve constraint | Narrowest contract and no upper envelope                                                         |
| `DCEGM`      | `ConsumptionSavingsRegime` with one `LiquidMargin`             | One liquid Euler margin with competing discrete-continuous branches           | Valid liquid resources and post-decision roles; declared lower bound; solver-supported discrete/passive dimensions and continuation layout                                                                                                                                                            | Adds constrained candidates and an upper envelope; simulation may re-optimize on the action grid |
| `NBEGM`      | `ConsumptionSavingsRegime` with one `LiquidMargin`             | Supported declared kinks, jumps, hard boundaries, or smooth discrete branches | Supported case-piece or piecewise-affine declaration; solver-proven constraint routes; no EV1 taste shocks                                                                                                                                                                                            | Preserves declared topology; structural probes and candidate geometry add cost                   |
| `NEGM`       | `NestedConsumptionSavingsRegime` with liquid and outer margins | A `DCEGM` inner solve conditional on a finite outer grid                      | Full inner `DCEGM` contract plus outer state, action, post-decision, no-adjustment, and cost roles                                                                                                                                                                                                    | Exact relative to the outer candidate set; candidate retention can dominate memory               |
| `NNBEGM`     | `NestedConsumptionSavingsRegime` with liquid and outer margins | An `NBEGM` inner solve inside a finite or adaptive outer search               | Full inner `NBEGM` contract plus a compatible outer search and branch aggregator                                                                                                                                                                                                                      | Most expressive EGM route and the highest structural/computational burden                        |

A valid declaration is part of the model, not a hint. Start with
[Authoring for EGM-family solvers](../user_guide/authoring_specialized_solvers.md).

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

Plain one-margin EGM. It validates the exact cash-on-hand identity summarized in the
capability table and takes no upper envelope.

### `DCEGM`

```python
from lcm.solvers import DCEGM, LTMEnvelope

solver = DCEGM(
    savings_grid=...,
    envelope=LTMEnvelope(),
    refined_grid_factor=2.0,
    n_constrained_points=20,
    stochastic_node_batch_size=0,
)
```

The supported typed envelope configurations are `ExactEnvelope`, `FUESEnvelope`,
`RFCEnvelope`, `LTMEnvelope`, and `MSSEnvelope`. String selectors are not part of the
public API. See [Upper envelopes](envelopes.md) for their distinct contracts.

`refined_grid_factor` controls policy read-out density, `n_constrained_points` the
borrowing-corner segment, and `stochastic_node_batch_size` the stochastic-node
workspace.

::::\{important} Solved and simulated continuous actions A solve can expose an off-grid
DCEGM policy for inspection when `return_simulation_policy=True`. No envelope shipped
with pylcm currently passes the conservative off-grid policy-read gate, so ordinary
simulation recomputes the action argmax on the regime's declared action grid. Simulation
also uses that gridded argmax when supplied value arrays carry no policy.

The simulated continuous action can therefore differ from the off-grid solve policy.
With taste shocks, simulated choice frequencies follow the grid-restricted
choice-specific values rather than necessarily matching the solve's off-grid choice
probabilities. The intrinsic budget is still applied as a simulation feasibility mask.
::::

### `NEGM`

```python
NEGM(inner=..., outer_grid=..., outer_batch_size=0)
```

Runs the bound `DCEGM` inner solve for every finite outer-grid node and includes the
keeper. `outer_batch_size` limits how many candidate values are evaluated at once. It
can reduce temporary evaluation memory, but it does not in general cap the size of the
candidate bank retained for later envelope or ordered-fold operations. Peak memory can
therefore continue to grow with the full candidate set. Measure both temporary and
retained arrays for the exact model and solver profile.

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

`jump_read` selects topology-preserving one-sided continuation reads or a faster bridged
finite-grid read. `probe_failure="assume_declared"` turns an unexecutable structural
probe into an author assertion and warning; it does not relax the mathematical
prerequisite.

With `envelope_arithmetic="certified"`, candidate ownership is ordered from the stored
floating-point operands using fixed-width integer arithmetic. Exact affine value comes
first, followed by deterministic geometric and stable-index tie-breaks. NaNs and other
non-finite or invalid geometry are not ordinary ordered values: the query is rejected
and remains uncovered/NaN so runtime validation can surface it. `"ordinary"` compares in
the working floating format and requires model-specific validation near crossings.

**Native capability.** `"certified"` uses pylcm's installed exact-affine CPU/CUDA
payload; it is not a pure-JAX numerical option. A compatible payload must exist for the
active JAX backend. NBEGM never silently falls back: if the payload is absent or
unloadable, certified mode raises `ExactAffineKernelUnavailableError` before returning a
certified result. Select `envelope_arithmetic="ordinary"` only when working-format
ownership is acceptable under model-specific crossing checks. The same requirement
applies when this NBEGM is the inner solver of `NNBEGM`.

The memory controls do not all mean “compiled batch width”:

- `stochastic_node_batch_size` and `envelope_segment_block_size` stream their named
  intermediate axes;
- `interval_batch_size` and `cell_block_size` are accepted fixed-window compatibility
  requests. In the current 256-row ride geometry, every nonnegative request admits the
  same stride of 256, so neither field changes iteration count, scheduling, compiled
  width, or workspace;
- `branch_batch_size` is an active commit-stride request only when the branch axis
  exceeds one four-row microtile. Positive requests round up to a multiple of four and
  are capped by the axis's static window (at most 64 rows); `0` selects the largest
  admitted stride. On an axis of four or fewer branches, every request admits the same
  four-row stride. Distinct admitted strides change scheduling and iteration count, not
  compiled width or fixed workspace.

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
