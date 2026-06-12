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
    regimes=regimes,  # dict mapping names to Regime instances
    ages=ages,  # AgeGrid defining the lifecycle timeline
    regime_id_class=RegimeId,  # @categorical dataclass mapping names to ScalarInt indices
    enable_jit=True,  # controls JAX compilation (default: True)
    fixed_params={},  # optional params baked in at init time
    description="",  # optional description string
)
```

All arguments are keyword-only. The three required arguments are `regimes`, `ages`, and
`regime_id_class`. The finalized regimes are stored as `model.user_regimes` (plain
`Regime` instances in user vocabulary); the processed canonical form is the
engine-internal `model._regimes`.

## Model-Level Regime Slots

When several regimes share functions, states, or actions, declare the shared structure
once at the model level instead of repeating it per regime — a lifecycle model with a
couple of dozen shared functions and a handful of shared states shrinks to one
declaration site:

```python
model = Model(
    regimes={"working": working, "retired": retired, "dead": dead},
    ages=ages,
    regime_id_class=RegimeId,
    functions={"taxes": taxes, "net_income": net_income},
    constraints={"budget": budget_constraint},
    states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=50)},
    state_transitions={"wealth": next_wealth},
    actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=30)},
)
```

Each model-level slot accepts exactly what the regime-level slot accepts — including
`Phased`, stochastic processes, per-target dicts, and `fixed_transition`. The entries
are merged into every regime under three rules:

- **Exactly one level.** Each name (function, constraint, state, state transition,
  action) may be defined at the model level or at the regime level, never both —
  defining it at both raises an ambiguity error at model build, exactly like supplying a
  parameter at two levels of the params dict. The same rule applies uniformly to every
  slot, including `derived_categoricals`.
- **`None` masks.** A regime opts out of a model-level entry by setting that name to
  `None` at the regime level (the *mask*) — the entry is removed for that regime.
  Masking a state also drops its broadcast law of motion, and masking a name that has no
  model-level entry behind it is an error.
- **DAG pruning.** A model-level (broadcast) state or action survives in a given regime
  only if some root computation of that regime — utility, `H`, a constraint, a derived
  categorical, the regime transition, or a law of motion toward a reachable target that
  carries the state — transitively reads it. Because "a law toward a reachable target
  that carries the state" refers to *other* regimes' carried states, pruning one
  variable in regime B can make a variable in regime A newly dead, so the pruning
  iterates across all regimes until nothing more can be dropped (a cross-regime fixed
  point). It runs separately on the solve slice and the simulate slice of each regime; a
  variable is dropped only when dead in **both** phases. Regime-level declarations are
  never pruned. `model.pruned_variables` records the outcome per regime.

Pruning means a model-level state costs nothing in regimes that never touch it — the
grid axis simply does not appear there. Two restrictions keep the device layout
coherent: `distributed=True` (sharding) is legal only on model-level states, and a
sharded state pruned from a non-terminal regime is an error (unshard it or make the
regime use it).

## Regime ID Classes

The `regime_id_class` maps regime names to integer indices. Use the `@categorical`
decorator to create it:

```python
from lcm import categorical
from lcm.typing import ScalarInt


@categorical(ordered=False)
class RegimeId:
    retired: ScalarInt
    working: ScalarInt
```

Rules:

- Fields must be annotated as `ScalarInt` — the 0-d `jnp.int32` scalar pylcm produces
  for category codes. Other annotations raise `CategoricalDefinitionError` at decoration
  time.
- Fields must match the keys of the `regimes` dict exactly (sorted alphabetically).
- Values are auto-assigned as consecutive `jnp.int32` scalars starting from 0.
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
ages = AgeGrid(exact_values=[25, 35, 45, 55, 65, 75])
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
model.user_regimes  # immutable mapping of finalized `Regime` objects
model.pruned_variables  # per regime, the broadcast names pruned by DAG reachability
model.n_periods  # number of periods
model.regime_names_to_ids  # name -> integer mapping
model.get_params_template()  # mutable copy of the parameter template
```

Use `model.get_params_template()` to get a mutable copy of the parameter template — see
[Parameters](parameters.md).

## Complete Example

```python
import jax.numpy as jnp
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import ScalarInt


@categorical(ordered=False)
class RegimeId:
    retired: ScalarInt
    working: ScalarInt


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    work: ScalarInt


def next_wealth(wealth, consumption, interest_rate):
    return (wealth - consumption) * (1 + interest_rate)


def next_regime(labor_supply):
    return jnp.where(
        labor_supply == LaborSupply.work, RegimeId.working, RegimeId.retired
    )


def utility(consumption, labor_supply, disutility_of_work):
    return jnp.log(consumption) - disutility_of_work * labor_supply


def terminal_utility(wealth):
    return jnp.log(wealth)


working = Regime(
    transition=next_regime,
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=50),
    },
    state_transitions={
        "wealth": next_wealth,
    },
    actions={
        "consumption": LinSpacedGrid(start=1, stop=50, n_points=30),
        "labor_supply": DiscreteGrid(LaborSupply),
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
