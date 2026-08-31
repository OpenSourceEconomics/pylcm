---
title: Authoring for EGM-family solvers
---

# Authoring for EGM-family solvers

EGM-family solvers are not drop-in replacements for `GridSearch`. They solve models
whose economic roles are declared in a specialized regime. Choose the declaration first,
then write the model around that contract.

This page gives three complete, executable starting points. Keep the roles and
structural constraints visible while replacing the toy economics with your own.

## One liquid margin

The first model has liquid wealth, consumption, and end-of-period savings. Plain `EGM`
is valid because the problem is smooth and concave, has one continuous state and action,
and has no discrete choice.

```python
from lcm import AgeGrid, Model
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    post_decision_lower_bound,
)
from lcm.regime import Regime
from lcm.solvers import EGM
from lcm_examples.specialized_consumption_savings import (
    CONSUMPTION_GRID,
    ONE_MARGIN,
    RegimeId,
    SAVINGS_GRID,
    WEALTH_GRID,
    example_initial_conditions,
    example_params,
    next_regime,
    next_wealth,
    savings,
    terminal_utility,
    utility,
)

working = ConsumptionSavingsRegime(
    transition=next_regime,
    states={"wealth": WEALTH_GRID},
    actions={"consumption": CONSUMPTION_GRID},
    state_transitions={"wealth": next_wealth},
    functions={"utility": utility, "savings": savings},
    constraints={
        "borrowing_limit": post_decision_lower_bound(
            margin=ONE_MARGIN,
            lower=0.0,
        )
    },
    liquid=ONE_MARGIN,
    solver=EGM(savings_grid=SAVINGS_GRID),
    active=lambda age: age == 0,
)

dead = Regime(
    transition=None,
    states={"wealth": WEALTH_GRID},
    functions={"utility": terminal_utility},
    active=lambda age: age == 1,
)

model = Model(
    regimes={"working": working, "dead": dead},
    regime_id_class=RegimeId,
    ages=AgeGrid(start=0, stop=1, step="Y"),
)

params = example_params()
solution = model.solve(params=params, log_level="debug")
result = model.simulate(
    params=params,
    initial_conditions=example_initial_conditions(),
    period_to_regime_to_V_arr=solution,
    log_level="debug",
)
df = result.to_dataframe(additional_targets=["savings"])
```

`ONE_MARGIN` names the four roles the solver consumes:

```python
LiquidMargin(
    state="wealth",
    action="consumption",
    resources="wealth",
    post_decision_state="savings",
)
```

Naming the state directly as `resources` avoids an identity function such as
`def resources(wealth): return wealth`. The lower-bound declaration says exactly which
constraint the savings grid enforces. Other constraints may make plain `EGM` invalid;
check the [solver capability table](../reference/solvers.md#capability-table).

## A kinked tax schedule

Plain `EGM` becomes ineligible as soon as cash on hand is a genuine function rather than
the liquid state itself. If that function has a declared kink, write the model for
`NBEGM` from the outset. This complete two-period model taxes liquid wealth above an
exemption:

```python
import jax.numpy as jnp

import lcm
from lcm import AgeGrid, Model
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    post_decision_lower_bound,
)
from lcm.regime import Regime
from lcm.solvers import NBEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from lcm_examples.specialized_consumption_savings import (
    CONSUMPTION_GRID,
    RegimeId,
    SAVINGS_GRID,
    WEALTH_GRID as LIQUID_GRID,
    inverse_marginal_utility,
    next_regime,
    utility,
)


@lcm.piecewise_affine(
    output="tax",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint(
            threshold="tax_exemption",
            kind="continuous_kink",
        ),
    ),
)
def tax(
    liquid: ContinuousState,
    tax_rate: float,
    tax_exemption: float,
) -> FloatND:
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


def resources(liquid: ContinuousState, tax: FloatND, income: float) -> FloatND:
    return liquid + income - tax


def savings(resources: FloatND, consumption: ContinuousAction) -> ContinuousState:
    return resources - consumption


def next_liquid(savings: ContinuousState) -> ContinuousState:
    return savings


def terminal_utility(liquid: ContinuousState) -> FloatND:
    return jnp.log1p(liquid)


margin = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)

working = ConsumptionSavingsRegime(
    transition=next_regime,
    states={"liquid": LIQUID_GRID},
    actions={"consumption": CONSUMPTION_GRID},
    state_transitions={"liquid": next_liquid},
    functions={
        "utility": utility,
        "tax": tax,
        "resources": resources,
        "savings": savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    },
    constraints={
        "borrowing_limit": post_decision_lower_bound(margin=margin, lower=0.0)
    },
    liquid=margin,
    solver=NBEGM(savings_grid=SAVINGS_GRID),
    active=lambda age: age == 0,
)

dead = Regime(
    transition=None,
    states={"liquid": LIQUID_GRID},
    functions={"utility": terminal_utility},
    active=lambda age: age == 1,
)

model = Model(
    regimes={"working": working, "dead": dead},
    regime_id_class=RegimeId,
    ages=AgeGrid(start=0, stop=1, step="Y"),
)

params = {
    "working": {
        "koopmans_aggregator": {"discount_factor": 0.95},
        "tax": {"tax_rate": 0.2, "tax_exemption": 7.0},
        "resources": {"income": 2.0},
    },
    "dead": {},
}
initial_conditions = {
    "age": jnp.array([0, 0]),
    "liquid": jnp.array([5.0, 10.0]),
    "regime_id": jnp.array([RegimeId.working, RegimeId.working]),
}

solution = model.solve(params=params, log_level="debug")
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=solution,
    log_level="debug",
)
df = result.to_dataframe(additional_targets=["tax", "resources", "savings"])
```

The `piecewise_affine` declaration tells NBEGM where the slope changes; the decorated
`tax` function remains ordinary executable economics. The margin identifies the genuine
`resources` node and the lower-bound constraint certifies the first savings-grid point.
The tested source version is `build_kinked_tax_model()` in
`lcm_examples.specialized_consumption_savings`. See
[Piecewise-affine schedules](../reference/piecewise_affine.md) for multiple brackets,
jumps, indexed thresholds, and the exact validation contract.

## A liquid margin nested inside an outer choice

The third model adds an illiquid stock and an investment action. Conditional on the
post-decision illiquid stock, `NEGM` runs a one-dimensional `DCEGM` liquid solve and
then compares the outer candidates.

```python
from lcm import AgeGrid, Model
from lcm.consumption_savings_regime import (
    NestedConsumptionSavingsRegime,
    post_decision_lower_bound,
)
from lcm.regime import Regime
from lcm.solvers import DCEGM, LTMEnvelope, NEGM
from lcm_examples.specialized_consumption_savings import (
    CONSUMPTION_GRID,
    ILLIQUID_GRID,
    INVESTMENT_GRID,
    NESTED_LIQUID_MARGIN,
    OUTER_MARGIN,
    RegimeId,
    SAVINGS_GRID,
    WEALTH_GRID,
    adjustment_cost,
    example_initial_conditions,
    example_params,
    inverse_marginal_utility,
    liquid_savings,
    nested_terminal_utility,
    nested_utility,
    new_illiquid,
    next_illiquid,
    next_regime,
    next_wealth_from_liquid_savings,
    resources_before_cost,
)

working = NestedConsumptionSavingsRegime(
    transition=next_regime,
    states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
    actions={
        "consumption": CONSUMPTION_GRID,
        "illiquid_investment": INVESTMENT_GRID,
    },
    state_transitions={
        "wealth": next_wealth_from_liquid_savings,
        "illiquid": next_illiquid,
    },
    functions={
        "utility": nested_utility,
        "new_illiquid": new_illiquid,
        "adjustment_cost": adjustment_cost,
        "resources_before_cost": resources_before_cost,
        "liquid_savings": liquid_savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    },
    constraints={
        "borrowing_limit": post_decision_lower_bound(
            margin=NESTED_LIQUID_MARGIN,
            lower=0.0,
        )
    },
    liquid=NESTED_LIQUID_MARGIN,
    outer_continuous=OUTER_MARGIN,
    solver=NEGM(
        inner=DCEGM(savings_grid=SAVINGS_GRID, envelope=LTMEnvelope()),
        outer_grid=ILLIQUID_GRID,
    ),
    active=lambda age: age == 0,
)

dead = Regime(
    transition=None,
    states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
    functions={"utility": nested_terminal_utility},
    active=lambda age: age == 1,
)

model = Model(
    regimes={"working": working, "dead": dead},
    regime_id_class=RegimeId,
    ages=AgeGrid(start=0, stop=1, step="Y"),
)

params = example_params()
solution = model.solve(params=params, log_level="debug")
result = model.simulate(
    params=params,
    initial_conditions=example_initial_conditions(nested=True),
    period_to_regime_to_V_arr=solution,
    log_level="debug",
)
df = result.to_dataframe(additional_targets=["liquid_savings"])
```

The two demonstration subjects choose `illiquid_investment` values 2 and 0, so the
example contains both the adjuster and the no-adjustment keeper. State and action
values, including `consumption` and `illiquid_investment`, are already ordinary result
columns; `additional_targets` is only for derived DAG outputs such as `liquid_savings`.

For a worked `NNBEGM` declaration — an `NBEGM` liquid solve nested inside an outer
search — see `lcm_examples.mahler_yum_2024.paper`, which builds one at production scale.

The liquid resources declaration also makes the adjustment-cost composition explicit.
The `adjustment_cost` named here is a resources-composition node: a DAG function
subtracted from resources before the liquid solve. It is unrelated to
`OuterContinuousMargin.adjustment_cost`, which is a branch-aggregation declaration
saying how keeper and adjuster values combine.

```python
NESTED_LIQUID_MARGIN = LiquidMargin(
    state="wealth",
    action="consumption",
    resources=NetOfAdjustmentCost(
        output="resources",
        before_cost="resources_before_cost",
        cost="adjustment_cost",
    ),
    post_decision_state="liquid_savings",
)

OUTER_MARGIN = OuterContinuousMargin(
    state="illiquid",
    action="illiquid_investment",
    post_decision_state="new_illiquid",
    no_adjustment=outer_unchanged,
)
```

## Where to go next

- **Guide:** [Choosing a solver](choosing_a_solver.md) maps problem shapes to
  declarations and solvers.
- **Methods:** [EGM foundations](../methods/egm_foundations.md) and
  [Nested endogenous-grid methods](../methods/nested_egm.md) explain the algorithms.
- **Examples:** [Iskhakov et al. (2017)](../examples/iskhakov_et_al_2017.md) uses
  `DCEGM`; [Mahler & Yum (2024)](../examples/mahler_yum_2024.md) uses a nested
  EGM-family declaration.
- **Reference:**
  [Consumption-saving regimes and margins](../reference/consumption_savings.md) and
  [Solvers and capabilities](../reference/solvers.md) state the exact contracts.
