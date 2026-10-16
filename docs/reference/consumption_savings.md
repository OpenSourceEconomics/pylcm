---
title: Consumption-saving regimes and margins
---

# Consumption-saving regimes and margins

EGM-family solvers need economic role names that a general DAG cannot infer. Specialized
regimes own those names; solver objects own only numerical configuration.

Import these declarations from `lcm` or `lcm.consumption_savings_regime`. Import every
solver from `lcm.solvers`.

(api-liquid-margin)=

## `LiquidMargin`

```python
import lcm

liquid = lcm.LiquidMargin(
    state="wealth",
    action="consumption",
    resources="wealth",
    post_decision_state="savings",
)
```

Fields:

- `state`: the liquid continuous state;
- `action`: the continuous action paid from resources;
- `resources`: the liquid state itself, a named resources function, or
  `NetOfAdjustmentCost`;
- `post_decision_state`: the named post-decision liquid state.

When resources are exactly the state, name that state directly. Do not create an
identity function merely to rename it.

`NetOfAdjustmentCost(output, before_cost, cost)` asks pylcm to compose
`output = before_cost - cost`. This keeps the accounting relation explicit without an
extra wrapper whose only job is subtraction.

(api-consumption-savings-regime)=

## `ConsumptionSavingsRegime`

`ConsumptionSavingsRegime(liquid=..., solver=...)` accepts `GridSearch` or a one-margin
solver: `EGM`, `DCEGM`, or `NBEGM`. It otherwise carries the same economic slots as
`Regime`.

The declaration is useful even while validating against grid search: the economic role
names remain attached to the model while the numerical method changes.

(api-outer-continuous-margin)=

## `OuterContinuousMargin`

```python
outer = lcm.OuterContinuousMargin(
    state="durable",
    action="next_durable",
    post_decision_state="durable_after_choice",
    no_adjustment=lcm.outer_unchanged,
    adjustment_cost=None,
)
```

`adjustment_cost` declares how the keeper and adjuster branches combine at this margin.
`None` means adjusting is free at the margin and the branches combine by the
deterministic maximum; `UniformObservedFixedCost(...)` declares an i.i.d. uniform cost
observed before the branch choice and integrates it analytically. See
[Outer search and branch aggregation](outer_search.md).

`no_adjustment` is either a named DAG function or the `outer_unchanged` sentinel when
the outer stock literally remains unchanged. Use the sentinel instead of an identity
helper.

(api-nested-consumption-savings-regime)=

## `NestedConsumptionSavingsRegime`

`NestedConsumptionSavingsRegime(liquid=..., outer_continuous=..., solver=...)` accepts
`GridSearch` or a two-margin solver: `NEGM` or `NNBEGM`.

The inner liquid and outer roles must be distinct and resolve in the assembled regime.
Nested EGM represents a conditional one-dimensional liquid problem around an outer
search; it does not infer or solve a general coupled two-dimensional Euler system.

(api-post-decision-lower-bounds)=

## Post-decision lower bounds

`post_decision_lower_bound(margin=liquid, lower=0.0)` returns an ordinary constraint
callable for

$$
\text{post-decision state} \ge \text{lower}.
$$

The lower bound can also be a callable DAG producer:

```python
from lcm.typing import ScalarFloat


def saving_floor(*, limit_level: ScalarFloat) -> ScalarFloat:
    return limit_level


borrowing_limit = lcm.post_decision_lower_bound(margin=liquid, lower=saving_floor)
```

A callable returns one finite real floating scalar for each supported fixed-state
cell. It may read parameters, time and the fixed states admitted by the selected
solver. Plain EGM admits parameter/time floors; DC-EGM and NB-EGM also admit their
supported discrete, process and passive states. The floor cannot read the liquid
state, any chosen action, or a helper depending on them. In a nested problem the
current outer state is fixed information; the outer choice is an action.

For a callable floor b and declared savings nodes g, the solver evaluates
`b + (g - g[0])` before Euler inversion, starting exactly at b in the working dtype.
The shifted nodes must remain finite and strictly increasing. Callable floors use
the declared grid's offsets and cannot accompany a runtime-supplied savings grid.
A literal lower bound must equal the declared grid's first node. Existing solver
state, schedule and utility restrictions still apply.

The constraint retains its relationship to the margin. Grid search and simulation
evaluate it using actual candidate inputs. Parameters discovered through a callable
floor appear in the model's ordinary parameter template.

Use this form for a borrowing limit on the declared post-decision state. Use a general
[structured Condition](conditions.md) for other retained comparisons and an ordinary
callable when no solver needs their named structure.

Method background: [The endogenous-grid method](../methods/egm_foundations.md) and
[Nested endogenous-grid methods](../methods/nested_egm.md).
