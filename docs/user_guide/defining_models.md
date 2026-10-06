---
title: Defining Models
---

# Defining Models

A `Model` ties together regimes, an age grid, and a regime ID class into a solvable
lifecycle model.

> **Choose the solver-facing regime first.** The general `Regime` used below is the
> `GridSearch` baseline. A model intended for EGM should start with
> `ConsumptionSavingsRegime`; a nested liquid/outer problem should start with
> `NestedConsumptionSavingsRegime`. See
> [Choose your starting declaration](../getting_started/next_steps.md).

## The Model Constructor

```python
from lcm import Model

model = Model(
    regimes=regimes,  # dict mapping names to Regime instances
    ages=ages,  # AgeGrid defining the lifecycle timeline
    regime_id_class=RegimeId,  # @categorical dataclass mapping names to ScalarInt indices
    edges=edges,  # source → destination → source ages
    initial_nodes=((25, "working"),),  # admissible starting pairs
    enable_jit=True,  # controls JAX compilation (default: True)
    fixed_params={},  # optional params baked in at init time
    description="",  # optional description string
)
```

All arguments are keyword-only. The five required arguments are `regimes`, `ages`,
`regime_id_class`, `edges` and `initial_nodes`. `edges` maps source regimes to
destinations and their source-age selectors, and declares every regime transition. A
source with one destination at each source age needs nothing more: the graph is its law.
A source with several destinations at some age is declared as
`Transition(targets={target: source_ages, ...}, law=...)`, whose law picks one. A regime
with no outgoing edges is terminal. Prefer explicit initial pairs such as
`((25, "working"),)`; selector-to-name mappings remain a convenience. There is no
default. The solved problems are derived from these roots, see
[Age-indexed regimes](dated_regime_graph.md). The finalized regimes are stored as
`model.user_regimes` (plain `Regime` instances in user vocabulary); the processed
canonical form is the engine-internal `model._regimes`.

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
    edges=edges,
    initial_nodes=initial_nodes,
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
  only if some root computation of that regime — utility, the Koopmans aggregator, a
  constraint, a derived categorical, the regime transition, or a law of motion toward a
  reachable target that carries the state — transitively reads it. Because "a law toward
  a reachable target that carries the state" refers to *other* regimes' carried states,
  pruning one variable in regime B can make a variable in regime A newly dead, so the
  pruning iterates across all regimes until nothing more can be dropped (a cross-regime
  fixed point). The solve slice and the simulate slice of each regime are closed
  **jointly** to a single such fixed point, not one application of each in turn: a
  target that keeps a state only because its simulate slice reads it makes the
  solve-side law of motion toward that target a root as well, and whatever that law
  reads then survives in the source regime. Both operators only ever add names to a
  finite pool, so the alternation reaches the least common fixed point and the result
  does not depend on which phase is closed first. Regime-level declarations are never
  pruned. `model.pruned_variables` records the outcome per regime.

Pruning means a model-level state costs nothing in regimes that never touch it — the
grid axis simply does not appear there. To spread a state's grid axis over the devices,
declare the state in `ExecutionConfig.sharded_states` on the `Model`. Sharding is legal
only on model-level states, and it follows pruning: the regimes that read the state
carry its device axis, while the regimes that prune it publish a value without that axis
and run on a single device. Placement is resolved per regime, so a sharded state may be
absent from a non-terminal regime as readily as from a terminal one, and the planner
moves the values that cross between the two placements. Only a state every regime prunes
is an error, because then no axis is left to spread.

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

ages = AgeGrid(start=25, inclusive_stop=75, step="Y")  # annual steps, ages 25 to 75
```

Step formats:

- `"Y"` — 1 year
- `"2Y"` — 2 years
- `"Q"` — quarter (0.25 years)
- `"M"` — month (1/12 year)
- `"3M"` — 3 months

The `inclusive_stop` value belongs to the grid; `(inclusive_stop - start)` must be
exactly divisible by the step size.

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

- At least one terminal regime must be provided; terminal-only starts need no
  non-terminal regime.
- Regime names cannot contain `__` (reserved separator).
- `regime_id_class` fields must exactly match the `regimes` dict keys.
- All states and actions must be used by at least one function (utility, constraints, or
  transitions).
- The age grid must be nonempty; every outgoing edge must have a next grid coordinate.
- Required `edges` and `initial_nodes` must name known regimes and exact admissible
  ages.

## Inspecting a Model

After construction, the model exposes several useful attributes:

```python
model.graph.edges.solve  # declared perceived source-age support
model.graph.solution  # effective perceived graph, indexed by period
model.graph.nodes  # valued age–regime pairs
model.graph.visited_nodes  # realized reachable age–regime pairs
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
from lcm import (
    AgeGrid,
    AgeRange,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    Transition,
    categorical,
)
from lcm.typing import ScalarInt


@categorical(ordered=False)
class RegimeId:
    retired: ScalarInt
    working: ScalarInt


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    work: ScalarInt


def next_wealth(*, wealth, consumption, interest_rate):
    return (wealth - consumption) * (1 + interest_rate)


def next_regime(*, labor_supply, age):
    return jnp.where(
        (age < 74) & (labor_supply == LaborSupply.work),
        RegimeId.working,
        RegimeId.retired,
    )


def utility(*, consumption, labor_supply, disutility_of_work):
    return jnp.log(consumption) - disutility_of_work * labor_supply


def terminal_utility(wealth):
    return jnp.log(wealth)


working = Regime(
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=50),
    },
    state_transitions={
        "wealth": next_wealth,
    },
    actions={
        "consumption": LinSpacedGrid(start=1, stop=50, n_points=30),
        "labor_supply": DiscreteGrid(category_class=LaborSupply),
    },
    functions={"utility": utility},
)

retired = Regime(
    states={
        "wealth": LinSpacedGrid(start=1, stop=100, n_points=50),
    },
    functions={"utility": terminal_utility},
)

model = Model(
    regimes={"working": working, "retired": retired},
    ages=AgeGrid(start=25, inclusive_stop=75, step="Y"),
    regime_id_class=RegimeId,
    edges={
        "working": Transition(
            targets={
                "working": AgeRange(start=25, exclusive_stop=74),
                "retired": AgeRange(start=25, exclusive_stop=75),
            },
            law=next_regime,
        )
    },
    initial_nodes=((25, "working"),),
)
```

`working` can stay or retire at every age before 74, so its edges carry `next_regime` as
their law; at 74 retirement is the only edge. `retired` has no outgoing edges and is
terminal.

## Correlated state transitions

Use `JointTransition` when several next states must share one stochastic draw. It owns a
joint support, one probability function, and one output projection per affected state.
Do not also declare separate transition laws for those outputs.

See [Transitions](transitions.ipynb) for the workflow and
[Transitions and phase specialization](../reference/transitions.md#api-joint-transitions)
for the exact support and output contract.

## See Also

- [Writing Economics](write_economics.ipynb) — function DAGs and regime design
- [Regimes](regimes.ipynb) — detailed guide to defining regimes
- [Parameters](parameters.md) — constructing the params dict
- [Solving and Simulating](solving_and_simulating.md) — running the model
