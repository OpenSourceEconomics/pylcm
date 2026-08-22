---
title: Consumption-saving regimes and margins
---

# Consumption-saving regimes and margins

EGM-family solvers need economic role names that a general DAG cannot infer. Specialized
regimes own those names; solver objects own only numerical configuration.

Import these declarations from `lcm` or `lcm.consumption_savings_regime`. Import every
solver from `lcm.solvers`.

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

## `ConsumptionSavingsRegime`

`ConsumptionSavingsRegime(liquid=..., solver=...)` accepts `GridSearch` or a one-margin
solver: `EGM`, `DCEGM`, or `NBEGM`. It otherwise carries the same economic slots as
`Regime`.

The declaration is useful even while validating against grid search: the economic role
names remain attached to the model while the numerical method changes.

## `OuterContinuousMargin`

```python
outer = lcm.OuterContinuousMargin(
    state="durable",
    action="next_durable",
    post_decision_state="durable_after_choice",
    no_adjustment=lcm.outer_unchanged,
)
```

`no_adjustment` is either a named DAG function or the `outer_unchanged` sentinel when
the outer stock literally remains unchanged. Use the sentinel instead of an identity
helper.

## `NestedConsumptionSavingsRegime`

`NestedConsumptionSavingsRegime(liquid=..., outer_continuous=..., solver=...)` accepts
`GridSearch` or a two-margin solver: `NEGM` or `NNBEGM`.

The inner liquid and outer roles must be distinct and resolve in the assembled regime.
Nested EGM represents a conditional one-dimensional liquid problem around an outer
search; it does not infer or solve a general coupled two-dimensional Euler system.

## Post-decision lower bounds

`post_decision_lower_bound(margin=liquid, lower=0.0)` returns an ordinary constraint
callable for

$$
\text{post-decision state} \ge \text{lower}.
$$

It also retains the relationship to the margin. An EGM solver can prove the restriction
from the matching savings grid and rejects a different lower endpoint during model
construction. Grid search and simulation simply evaluate the callable.

Use this form for a borrowing limit on the declared post-decision state. Use a general
[structured Condition](conditions.md) for other retained comparisons and an ordinary
callable when no solver needs their named structure.

Method background: [The endogenous-grid method](../methods/egm_foundations.md) and
[Nested endogenous-grid methods](../methods/nested_egm.md).
