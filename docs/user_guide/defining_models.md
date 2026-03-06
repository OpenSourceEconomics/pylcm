---
title: Defining Models
---

# Defining Models

A `Model` ties together regimes, an age grid, and a regime ID class into a solvable
lifecycle model.

## The Model Constructor

```python
from lcm import Model

model = Model(
    regimes=regimes,             # dict mapping names to Regime instances
    ages=ages,                   # AgeGrid defining the lifecycle timeline
    regime_id_class=RegimeId,    # @categorical dataclass mapping names to int indices
    enable_jit=True,             # controls JAX compilation (default: True)
    fixed_params={},             # optional params baked in at init time
    description="",              # optional description string
)
```

All arguments are keyword-only. The three required arguments are `regimes`, `ages`, and
`regime_id_class`.

## Regime ID Classes

The `regime_id_class` maps regime names to integer indices. Use the `@categorical`
decorator to create it:

```python
from lcm import categorical

@categorical
class RegimeId:
    retired: int
    working: int
```

Rules:

- Fields must match the keys of the `regimes` dict exactly (sorted alphabetically).
- Values are auto-assigned as consecutive integers starting from 0.
- Use `RegimeId.working` (class attribute access) to reference regime IDs in transition
  functions.

## Age Grids

The `ages` argument defines the lifecycle timeline. There are two construction modes:

### Range-based

```python
from lcm import AgeGrid

ages = AgeGrid(start=25, stop=75, step="Y")  # annual steps, ages 25 to 75
```

Step formats:
- `"Y"` — 1 year
- `"2Y"` — 2 years
- `"Q"` — quarter (0.25 years)
- `"M"` — month (1/12 year)
- `"3M"` — 3 months

The `stop` value is inclusive if `(stop - start)` is exactly divisible by the step size.

### Exact values

```python
ages = AgeGrid(precise_values=[25, 35, 45, 55, 65, 75])
```

Use this for irregular age spacing.

### Key properties

- `ages.values` — JAX array of ages, indexed by period
- `ages.n_periods` — number of periods
- `ages.step_size` — step size in years (or `None` for exact values)
- `ages.period_to_age(period)` — convert period index to age
- `ages.get_periods_where(predicate)` — get periods matching a condition

## Model Validation Rules

The `Model` constructor validates:

- At least one terminal regime and one non-terminal regime must be provided.
- Regime names cannot contain `__` (reserved separator).
- `regime_id_class` fields must exactly match the `regimes` dict keys.
- All states and actions must be used by at least one function (utility, constraints, or
  transitions).
- The age grid must have at least 2 periods.

## Inspecting a Model

After construction, the model exposes several useful attributes:

```python
model.regimes             # immutable mapping of user Regime objects
model.internal_regimes    # processed internal representations
model.n_periods           # number of periods
model.regime_names_to_ids # name -> integer mapping
model.params_template     # see Parameters page
```

Use `model.get_params_template()` to get a mutable copy of the parameter template — see
[Parameters](parameters.md).

## Complete Example

```python
import jax.numpy as jnp
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical


@categorical
class RegimeId:
    retired: int
    working: int


@categorical
class WorkChoice:
    no: int
    yes: int


def next_wealth(wealth, consumption, interest_rate):
    return (wealth - consumption) * (1 + interest_rate)


def next_regime(work):
    return jnp.where(work == WorkChoice.yes, RegimeId.working, RegimeId.retired)


def utility(consumption, work, disutility_of_work):
    return jnp.log(consumption) - disutility_of_work * work


def terminal_utility(wealth):
    return jnp.log(wealth)


working = Regime(
    transition=next_regime,
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=50, transition=next_wealth),
    },
    actions={
        "consumption": LinSpacedGrid(start=1, stop=50, n_points=30),
        "work": DiscreteGrid(WorkChoice),
    },
    functions={"utility": utility},
)

retired = Regime(
    transition=None,
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=50),
    },
    functions={"utility": terminal_utility},
)

model = Model(
    regimes={"working": working, "retired": retired},
    ages=AgeGrid(start=25, stop=75, step="Y"),
    regime_id_class=RegimeId,
)
```

## See Also

- [Writing Economics](write_economics.ipynb) — function DAGs and regime design
- [Regimes](regimes.ipynb) — detailed guide to defining regimes
- [Parameters](parameters.md) — constructing the params dict
- [Solving and Simulating](solving_and_simulating.md) — running the model
