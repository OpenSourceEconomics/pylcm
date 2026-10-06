---
title: Age-indexed regime graphs
---

# Age-indexed regime graphs

A model separates three declarations: admissible initial nodes, graph edges, and
numerical transition laws. `Model` requires `initial_nodes` and `edges`. Neither is
inferred from probability functions or from the availability of an age-indexed law.

## Declare starts and edges

```python
import jax.numpy as jnp

from lcm import AgeGrid, ByAge, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def feasible(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption < wealth


wealth_grid = LinSpacedGrid(start=1.0, stop=100.0, n_points=25)
consumption_grid = LinSpacedGrid(start=0.5, stop=50.0, n_points=25)
working = Regime(
    regime_transitions=ByAge.until(
        start_age_inclusive=60,
        stop_age_exclusive=63,
        law="working",
        then="retired",
    ),
    states={"wealth": wealth_grid},
    actions={"consumption": consumption_grid},
    state_transitions={"wealth": next_wealth},
    functions={"utility": utility},
    constraints={"feasible": feasible},
)
retired = Regime(
    regime_transitions="dead",
    states={"wealth": wealth_grid},
    actions={"consumption": consumption_grid},
    state_transitions={"wealth": next_wealth},
    functions={"utility": utility},
    constraints={"feasible": feasible},
)
dead = Regime(
    regime_transitions=None,
    states={"wealth": wealth_grid},
    functions={"utility": bequest},
)
ages = AgeGrid(start=60, inclusive_stop=65, step="Y")
edges = {
    "working": {"working": (60, 61), "retired": 62},
    "retired": {"dead": (63, 64)},
}
model = Model(
    regimes={"working": working, "retired": retired, "dead": dead},
    ages=ages,
    regime_id_class=RegimeId,
    edges=edges,
    initial_nodes=((60, "working"),),
)
```

Each source maps destinations to their permitted **source ages**. In this example,
working at 62 leads to retired at 63, then dead at 64. The retired-to-dead edge at 64
also permits a retired start at 64 if that pair is added to `initial_nodes`. Declared
edges need not all be reached from the chosen starts.

The final age of `AgeGrid` is inclusive. Every effective edge lands at the next grid
coordinate, which need not be one calendar year later. Declared selectors may include
the final age as dormant metadata; the effective graph never has an outgoing edge there.
Names in dormant declarations are still validated. A terminal regime has
`regime_transitions=None` and no outgoing declared edges.

Prefer an explicit tuple of age–regime pairs. A selector mapping remains convenient when
many starts share regimes:

```python
# Equivalent admissible starts.
initial_nodes = ((60, "working"), (61, "working"))
initial_nodes = {(60, 61): "working"}
```

Starts are admissibility declarations, not population weights. Simulation still receives
subjects and their states through `InitialConditions` or `Population`. There is no
default start and no automatic inference of roots from a graph.

## Select laws separately

The working regime above can use:

```python
# Fragment: this is the regime_transitions argument of working.
working_law = ByAge.until(
    stop_age_exclusive=63,
    start_age_inclusive=60,
    law="working",
    then="retired",
)
retired_law = "dead"
```

`ByAge.until` assigns `then` to the last selected source age before its exclusive stop;
other selected ages use `law`. `ByAge(cases=..., default=...)` can select unrelated
functions or probability mappings. These declarations choose numerical behavior;
`Model.edges` remains the sole declaration of structural support.

Plain functions are deterministic. Explicit wrappers and decorator syntax work for both
regime and state laws:

```python
@deterministic_transition()
def destination(age: ScalarInt) -> ScalarInt:
    return jnp.where(age < 62, RegimeId.working, RegimeId.retired)


@stochastic_transition()
def death_probability(mortality: ScalarFloat) -> ScalarFloat:
    return mortality
```

A deterministic regime function returns a global regime code supported at that source
age. A full-vector `StochasticTransition(func=...)` returns probabilities in full global
regime-code order and must be zero outside graph support. Per-target scalar probability
mappings provide the probabilities for graph-selected destinations. Their keys do not
independently define edges. Public wrappers and decorators have no `targets` argument.

Source-age selectors can be exact ages, nonempty tuples, integer ranges, or
`AgeRange(start=..., exclusive_stop=...)`. A half-open selector excludes its stop:

```python
edges = {"working": {"working": AgeRange(start=60, exclusive_stop=62), "retired": 62}}
```

`AgeRange` selects existing grid coordinates; it does not create intermediate ages.
Transitions between regimes with different state spaces require explicit handoffs for
each retained destination state. For example, health may be remapped from three working
categories to two retired categories, while death may receive a separate assets law and
omit health entirely. Per-target state laws and entry declarations remain on `Regime`;
the graph owns their structural destination and age restrictions.

## Perceived and realized edges

Use `Phased` on `edges` when beliefs and realized transitions differ:

```python
# Fragment: the corresponding numerical laws must agree with each phase's support.
edges = Phased(
    solve={"working": {"working": 60}},
    simulate={"working": {"retired": 60}},
)
working_law = Phased(solve="working", simulate="retired")
```

Every realized visit needs a local solve value and its recursive perceived continuation
values. Additional nodes needed only for valuation do not create realized visits. Graph
support, numerical laws, and state handoffs are validated together at construction.

## Inspect declared and effective graphs

`model.graph` is immutable and read-only. It is a result of constructing `Model`, not a
separate public graph constructor:

```python
declared_solve = model.graph.edges.solve
declared_simulate = model.graph.edges.simulate
perceived = model.graph.solution
realized = model.graph.simulation
valued_nodes = model.graph.nodes
visited_nodes = model.graph.visited_nodes
removed_solve_edges = model.graph.pruned_edges["solve"]
```

Declared edge selectors resolve to exact source-age sets. The effective phase graphs are
indexed by model period. `.nodes` contains valued age–regime pairs; `.visited_nodes`
contains physically reachable pairs. Pruning records use `(source_age, source, target)`
keys and the reason `"fixed_zero_probability"`.

An ordinary scalar probability cell is pruned only when its complete dependency DAG
contains construction-fixed leaves and produces exactly represented zero. States,
actions, age, period, free parameters and coordinate-indexed fixed series retain edges.
Runtime zeros cannot change topology. Positive subnormals remain live; invalid and
all-zero laws retain validation. Pruning never renormalizes probability mass, and the
declared graph retains removed edges for inspection.

State laws and joint lotteries that hand states across a removed edge leave with it, but
the authored model keeps its meaning. A removed joint lottery is still checked for its
target-state ownership. A source state whose only law was that lottery stays a valid
state of the source with the empty per-target law `{}`, exactly as in the model declared
without the removed edge, and a state or action read only by a removed law still counts
as used. Every target that keeps an edge must still receive each state it carries.

Large applications can build regimes and the matching edge mapping from one internal
edge catalog, then reuse that topology across policy variants that change only economic
functions. This keeps health remapping, target-specific assets laws and entry-state
handoffs near their economic definitions while making shared connectivity inspectable.
