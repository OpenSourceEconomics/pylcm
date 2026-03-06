---
title: Consumption-Saving Model
---

# Consumption-Saving Model

This example walks through a consumption-savings model with health and exercise —
richer than the [tiny example](../getting_started/tiny_example.ipynb) but still compact
enough to read in one sitting. It demonstrates multiple states, multiple continuous
actions, auxiliary functions, constraints, and regime transitions.

:::{note}
The parameterization is chosen to showcase pylcm's features, not to match any empirical
calibration. Do not read economic content into the specific functional forms.
:::

## Overview

An agent lives from age 18 to 24. During working life (ages 18–23) the agent chooses:

- **whether to work** (discrete: working / retired)
- **how much to consume** (continuous)
- **how much to exercise** (continuous)

Two continuous states evolve over time:

- **wealth** — increases with labor income and interest, decreases with consumption
- **health** — improves with exercise, deteriorates with work

At age 24 the agent enters a terminal **retirement** regime where utility depends only on
remaining wealth and health (no choices).

## Categorical Variables

The model uses two `@categorical` classes. `WorkingStatus` labels the discrete work
action; `RegimeId` maps regime names to integer codes for the regime transition function.

```python
from lcm import categorical


@categorical
class WorkingStatus:
    retired: int
    working: int


@categorical
class RegimeId:
    working: int
    retirement: int
```

See [Grids — Discrete Grids](../user_guide/grids.md) for more on `@categorical` and
`DiscreteGrid`.

## Economic Functions

### Utility

The working-life utility function combines log consumption, a disutility of work
(offset by health), and an exercise cost:

```python
def utility(consumption, working, health, exercise, disutility_of_work):
    return jnp.log(consumption) - (disutility_of_work - health) * working - exercise
```

`disutility_of_work` is a **parameter** — it appears in the function signature but is
not a state or action, so pylcm will look for it in the params dict.

The retirement utility is simpler — just log wealth scaled by health:

```python
def utility_retired(wealth, health):
    return jnp.log(wealth) * health
```

### Auxiliary Functions

`wage` computes an age-dependent wage and `labor_income` multiplies it by the work
decision. These are registered in `functions` alongside `utility`, so pylcm can
automatically wire them into other functions that depend on `labor_income`:

```python
def labor_income(wage, working):
    return wage * working


def wage(age):
    return 1 + 0.1 * age
```

See [Writing Economics](../user_guide/write_economics.ipynb) for how pylcm resolves
function dependencies via a DAG.

### State Transitions

State transitions are attached to grids via the `transition` parameter. Each transition
function returns the next-period value of its state:

```python
def next_wealth(wealth, consumption, labor_income, interest_rate):
    return (1 + interest_rate) * (wealth + labor_income - consumption)


def next_health(health, exercise, working):
    return health * (1 + exercise - working / 2)
```

`interest_rate` is a parameter (resolved from the params dict).

### Regime Transition

The regime transition function determines which regime the agent enters next period. Here
the agent stays in the working regime until the second-to-last period, then transitions
to retirement:

```python
def next_regime(period, n_periods):
    certain_retirement = period >= n_periods - 2
    return jnp.where(certain_retirement, RegimeId.retirement, RegimeId.working)
```

`n_periods` is supplied as a parameter at solve time.

### Constraints

A borrowing constraint ensures consumption does not exceed available resources:

```python
def borrowing_constraint(consumption, wealth, labor_income):
    return consumption <= wealth + labor_income
```

pylcm filters out state-action combinations that violate constraints before evaluating
utility.

## Regime Assembly

### Working Regime

The working regime has two continuous states (wealth, health), three actions (work
decision, consumption, exercise), three functions (utility plus auxiliaries), and one
constraint:

```python
from lcm import DiscreteGrid, LinSpacedGrid, Regime

working = Regime(
    transition=next_regime,
    active=lambda age: age < RETIREMENT_AGE,
    states={
        "wealth": LinSpacedGrid(
            start=1, stop=100, n_points=100, transition=next_wealth,
        ),
        "health": LinSpacedGrid(
            start=0, stop=1, n_points=100, transition=next_health,
        ),
    },
    actions={
        "working": DiscreteGrid(WorkingStatus),
        "consumption": LinSpacedGrid(start=1, stop=100, n_points=100),
        "exercise": LinSpacedGrid(start=0, stop=1, n_points=200),
    },
    functions={
        "utility": utility,
        "labor_income": labor_income,
        "wage": wage,
    },
    constraints={"borrowing_constraint": borrowing_constraint},
)
```

See [Regimes](../user_guide/regimes.ipynb) for a detailed guide to `Regime`.

### Retirement Regime

The terminal regime has `transition=None`, no actions, and states with `transition=None`
(fixed — no evolution needed since there is no next period):

```python
retired = Regime(
    transition=None,
    active=lambda age: age >= RETIREMENT_AGE,
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=100, transition=None),
        "health": LinSpacedGrid(start=0, stop=1, n_points=100, transition=None),
    },
    functions={"utility": utility_retired},
)
```

## Model and Parameters

```python
from lcm import AgeGrid, Model

model = Model(
    regimes={"working": working, "retirement": retired},
    ages=AgeGrid(start=18, stop=RETIREMENT_AGE, step="Y"),
    regime_id_class=RegimeId,
)
```

The params dict follows the template from `model.params_template` — a top-level
`discount_factor` and regime-specific nested dicts:

```python
params = {
    "discount_factor": 0.95,
    "working": {
        "utility": {"disutility_of_work": 0.05},
        "next_wealth": {"interest_rate": 0.05},
        "next_regime": {"n_periods": model.n_periods},
    },
    "retirement": {},
}
```

See [Parameters](../user_guide/parameters.md) for how to inspect and construct the
params dict.

## Solving and Simulating

```python
import jax.numpy as jnp

result = model.solve_and_simulate(
    params=params,
    initial_regimes=["working"] * 1_000,
    initial_states={
        "age": jnp.full(1_000, model.ages.values[0]),
        "wealth": jnp.full(1_000, 1.0),
        "health": jnp.full(1_000, 1.0),
    },
)

df = result.to_dataframe(additional_targets="all")
```

See [Solving and Simulating](../user_guide/solving_and_simulating.md) for the full API.

## See Also

- [A Tiny Example](../getting_started/tiny_example.ipynb) — simpler three-period model
- [Writing Economics](../user_guide/write_economics.ipynb) — function DAGs and regime
  design
- [Grids](../user_guide/grids.md) — grid types and transitions
- [Parameters](../user_guide/parameters.md) — constructing the params dict
