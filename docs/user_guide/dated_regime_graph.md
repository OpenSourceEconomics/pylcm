---
title: Age-indexed regime graphs
---

# Age-indexed regime graphs

A model declares its admissible initial nodes and its graph. `Model` requires
`initial_nodes` and `edges`. `Model(edges=...)` is the only place regime transitions are
declared, structure and law alike: a source with one destination per age needs no law,
and a `Transition` carries the law wherever a source age has several destinations. A
`Regime` carries no law; `model.graph.laws` holds each regime's law as the model binds
it from its edges.

## Declare starts and edges

```python
import jax.numpy as jnp

from lcm import AgeGrid, InitialNodes, LinSpacedGrid, Model, Regime, categorical
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
    states={"wealth": wealth_grid},
    actions={"consumption": consumption_grid},
    state_transitions={"wealth": next_wealth},
    functions={"utility": utility},
    constraints={"feasible": feasible},
)
retired = Regime(
    states={"wealth": wealth_grid},
    actions={"consumption": consumption_grid},
    state_transitions={"wealth": next_wealth},
    functions={"utility": utility},
    constraints={"feasible": feasible},
)
dead = Regime(
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
    initial_nodes=InitialNodes(by_age={60: "working"}),
)
```

Each source maps destinations to their permitted **source ages**. In this example,
working at 62 leads to retired at 63, then dead at 64. Every source age has exactly one
outgoing edge, so the graph is the whole law: no regime declares a transition, and
`dead`, which no edge leaves, is terminal. The retired-to-dead edge at 64 also permits a
retired start at 64 if that pair is added to `initial_nodes`. Declared edges need not
all be reached from the chosen starts.

The final age of `AgeGrid` is inclusive. Every effective edge lands at the next grid
coordinate, which need not be one calendar year later. Declared selectors may include
the final age as dormant metadata; the effective graph never has an outgoing edge there.
Names in dormant declarations are still validated. A regime with no outgoing declared
edges is terminal.

Use `InitialNodes(by_age=...)` to identify starting coordinates explicitly. Each
selector maps to a regime name or a nonempty sequence or set of names:

```python
# Equivalent admissible starts.
initial_nodes = InitialNodes(by_age={60: "working", 61: "working"})
initial_nodes = InitialNodes(by_age={(60, 61): "working"})
```

Overlapping selectors contribute their union; repeated pairs are harmless. `AgeRange`
selects existing grid ages in its half-open interval. Explicit off-grid ages, selectors
that select no age, and unknown regimes raise errors. `ByAge` selects transition laws;
it does not declare admissible starts.

The declaration copies its mapping and regime collections. The model publishes a
normalized `InitialNodes`: `model.initial_nodes.by_age` maps exact grid ages to sorted,
unique regime tuples. Pass `model.initial_nodes` back to `Model` to reconstruct the same
starts. Use `model.graph.initial_nodes` when you need expanded `(age, regime)` pairs.
Legacy pair collections and bare selector mappings remain accepted as inputs.

Starts are admissibility declarations, not population weights. Simulation still receives
subjects and their states through `InitialConditions` or `Population`. There is no
default start and no automatic inference of roots from a graph.

## Declare a law where a source has several destinations

A source age with more than one outgoing edge needs a law that chooses among them. The
source is then declared as a `Transition`. A law that names its destinations —
per-target mappings and regime names, possibly selected by `ByAge` — is enough on its
own; the destination-to-age mapping is derived from it:

```python
# Fragment: survive and die are scalar probability functions.
edges = {
    "working": Transition(
        law=ByAge(
            cases={
                (60, 61): {
                    "working": StochasticTransition(func=survive),
                    "dead": StochasticTransition(func=die),
                },
                62: "retired",
            }
        ),
    ),
    "retired": {"dead": (63, 64)},
}
```

This derives `{"working": (60, 61), "dead": (60, 61), "retired": 62}`. Passing that
mapping as `targets=` as well is allowed; a `targets` that differs from the derived one
is refused. A law over all targets — a function or a full-vector `StochasticTransition`
— names none, so it needs `targets` spelled out.

The law can be

- a per-target mapping of `StochasticTransition` probabilities, keyed by destination;
- a plain function or `DeterministicTransition` returning a global regime code, which is
  how a discrete choice between regimes is written;
- a full-vector `StochasticTransition`;
- a regime name;
- `ByAge(cases=..., default=...)` selecting one of these per source age, or
  `ByAge.until(...)`;
- `Phased(solve=..., simulate=...)` giving each phase its own.

A `ByAge` law must select every source age with several outgoing edges. A `ByAge` with a
case over all targets, which comes with an explicit `targets`, need not select an age
with a single outgoing edge; that edge is the law there. A law that names its targets
declares every edge it has, so it selects such an age too, as the regime name at age 62
above does. Several outgoing edges without a law are rejected. A law that does select an
age with a single outgoing edge is evaluated there and must put unit mass on that edge.

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
mappings provide the probabilities for their destinations, and their keys are the
destinations `Transition` derives when `targets` is omitted. Public wrappers and
decorators take no `targets` argument; passing one raises a `TypeError`. Destinations
are declared only in `Model(edges=...)`.

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

Use `Phased` on `edges` when beliefs and realized transitions differ. Each phase's
mapping declares that phase's edges, and a `Transition` in it carries that phase's law:

```python
# Fragment: perceived and realized survival probabilities differ.
edges = Phased(
    solve={
        "working": Transition(targets=targets, law=perceived_survival),
        "retired": {"dead": (63, 64)},
    },
    simulate={
        "working": Transition(targets=targets, law=realized_survival),
        "retired": {"dead": (63, 64)},
    },
)
```

A law-free source follows its single edge in each phase, so
`Phased(solve={"working": {"working": 60}}, simulate={"working": {"retired": 60}})`
believes in staying while realizing retirement. Where one phase has a single edge at a
source age and the other phase's `Transition` law is a per-target probability mapping
there, the lone edge counts as a probability-one cell for its destination, so both
phases carry the same form.

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

Pruning removes as much as it can up front: the pruned model is the model declared
without the removed edges. State laws and joint lotteries that hand states across a
removed edge leave with it, unchecked: a model whose declarations toward a fixed-zero
target would conflict over a target-state cell builds exactly as the model without that
edge, while every edge that stays keeps its full target-state ownership checks,
including one whose free probability is zero at runtime. A source state whose only law
was that lottery stays a state of the source with the empty per-target law `{}`, exactly
as an author would declare it without the edge. A state or action read only across a
removed edge is unused, and the model is rejected just as the edge-free model would be;
the error names the removed edge. Every target that keeps an edge must still receive
each state it carries.

Large applications can build regimes and the matching edge mapping from one internal
edge catalog, then reuse that topology across policy variants that change only economic
functions. This keeps health remapping, target-specific assets laws and entry-state
handoffs near their economic definitions while making shared connectivity inspectable.
