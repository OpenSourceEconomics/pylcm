---
title: Migrating regime graph declarations
---

# Migrating regime graph declarations

The model owns every regime transition through a required `edges` argument, both its
structure and its law. Initial age–regime pairs are required explicitly through
`initial_nodes`.

## Move transitions onto Model

`Regime` takes no regime transition law. The removed form declared the law on the source
regime and now raises a `TypeError`:

```text
# Removed: raises TypeError. Shown only to identify code that needs migrating.
working = Regime(
    functions={"utility": utility},
    states={"assets": assets_grid},
    state_transitions={"assets": next_assets},
    regime_transitions=ByAge.until(
        stop_age_exclusive=62, law="working", then="retired"
    ),
)
```

Delete that argument from every `Regime` and declare the destinations in
`Model(edges=...)`. Where each source age has a single destination, the edges alone are
the law:

```python
# Current declaration fragment: working stays until 62, then retires.
working = Regime(
    functions={"utility": utility},
    states={"assets": assets_grid},
    state_transitions={"assets": next_assets},
)
edges = {
    "working": {
        "working": AgeRange(start=60, exclusive_stop=62),
        "retired": 62,
    },
}
model = Model(
    regimes={"working": working, "retired": retired},
    regime_id_class=RegimeId,
    ages=AgeGrid(start=60, inclusive_stop=63, step="Y"),
    edges=edges,
    initial_nodes=((60, "working"),),
)
```

This replaces laws such as `"dead"`, `ByAge.until(law="working", then="retired")` or a
selector that only ever returns the one available destination. A regime whose removed
law was `None` simply has no outgoing edges.

Code that read a regime's law back reads it from the model:

| Removed                                 | Read instead                                   |
| --------------------------------------- | ---------------------------------------------- |
| `Regime.terminal`                       | `model.graph.laws[name].terminal`              |
| `Regime.gated_edges`                    | `model.graph.laws[name].gated_edges`           |
| `Regime.decomposed_transition`          | `model.graph.laws[name].decomposed_transition` |
| the law passed as `regime_transitions=` | `model.edges[name].law`                        |

`model.edges` holds the edges exactly as declared, so `model.edges[name].law` is the law
of a source declared as a `Transition` (under `model.edges.solve` and
`model.edges.simulate` when the whole mapping is `Phased`). `model.graph.laws[name]` is
the law the solver and simulator evaluate: bound to the graph, pruned of fixed-zero
cells and lowered to the ages the starts demand. A regime is terminal when it has no
outgoing edge in `Model(edges=...)`, and `model.graph.laws[name].terminal` says so:

```python
assert model.graph.laws["dead"].terminal
assert not model.graph.laws["working"].terminal
```

The bound law carries no destinations; a source's declared targets are the keys of
`model.edges[name]`, or `model.edges[name].targets` for a `Transition`, and the exact
source ages per phase are `model.graph.edges.solve[name]` and
`model.graph.edges.simulate[name]`.

Where a source age has several destinations, move the former law unchanged into a
`Transition` that replaces the source's destination mapping:

```python
# Fragment: a choice between continuing to work and retiring.
edges = {
    "working": Transition(
        targets={"working": AgeRange(start=60, exclusive_stop=63), "retired": (61, 62)},
        law=DeterministicTransition(func=destination),
    ),
}
```

A `ByAge` law may leave single-destination ages unselected. A law that does reach such
an age is evaluated there and must put unit mass on its one destination; see
[Laws at every horizon](#laws-at-every-horizon).

Public `DeterministicTransition` and `StochasticTransition` have no `targets` argument;
passing one raises a `TypeError`. Declare destinations in `Model(edges=...)`, the only
place regime transitions are declared. Their targetless decorator factories are
`@deterministic_transition()` and `@stochastic_transition()`. Plain functions remain
deterministic for both state and regime laws. Full-vector stochastic laws retain global
regime-code ordering; probabilities outside graph support must be zero.

Per-target probability mappings still supply scalar probability laws. Move their
structural destination and age restrictions into `edges`, and keep target-specific state
handoffs on the source regime. `ByAge` selects complete laws, not topology or solved
coverage. Replace age activity predicates with graph selectors and explicit starts.

(migrating-edge-parameters)=

## Move transition parameters under `edges`

The regime-transition law, its gates, gate references and route fallbacks are declared
in `Model(edges=...)`, and their parameters live there too: the parameter path of an
edge-declared callable is its declaration path under `params["edges"]`. The source
regime owns none of them, so its branch of the parameters has no `next_regime` or `gate`
entry. Per-target state laws (`state_transitions={state: {target: law}}`) belong to the
source regime and keep their paths, `params[source][target]["next_<state>"]`.

| Key under the source regime                                       | Path under `params["edges"]`                                            |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `[source]["next_regime"][arg]` (a law over all targets)           | `[source][arg]`                                                         |
| `[source][target]["next_regime"][arg]` (a per-target cell)        | `[source][target][arg]`                                                 |
| the same, for a `ValueDependentTransition` cell                   | `[source][target]["probability"][arg]`                                  |
| `[source][target]["gate"][arg]`                                   | `[source][target]["gate"][arg]`                                         |
| `[source][target]["gate_ref_<reference>_<state>"][arg]`           | `[source][target]["gate_references"][reference][state][arg]`            |
| `[source][target]["leg_fallback_<regime>_<state>"][arg]`          | `[source][target]["routes"][route]["fallback"][state][arg]`             |
| the same, solve side of a `Phased` fallback                       | `[source][target]["routes"][route]["fallback"]["solve"][state][arg]`    |
| `[source][target]["simulate_leg_fallback_<regime>_<state>"][arg]` | `[source][target]["routes"][route]["fallback"]["simulate"][state][arg]` |
| `[source][arg]`, where the law or a gate reads it                 | `[source][arg]`, or the model level                                     |

A law over all targets is a plain function, `DeterministicTransition`, a full-vector
`StochasticTransition`, or `ByAge` / `Phased` around one; it has no `law` segment
because the law is the only callable of a `Transition`. The `ByAge` cases and `Phased`
sides of a law share its slot, and their arguments are unioned. A route fallback is
keyed by its `routes` key, not by the regime it falls back to. A source declared by its
edges alone has no `params["edges"]` entry, and `get_params_template()` has an `edges`
branch only when some edge-declared callable takes a parameter:

```python
params = {
    "discount_factor": 0.95,
    "working": {
        "utility": {"disutility_of_work": 0.05},
        "next_wealth": {"interest_rate": 0.05},
    },
    "edges": {"working": {"retirement_age": 62}},
}
```

### Where an edge parameter's value may come from

For one parameter of an edge-declared callable, the candidate levels are, most specific
first:

1. its declaration path, e.g. `params["edges"]["working"]["dead"]["survival_rate"]`;
1. `params["edges"][source][arg]`, which covers every callable below the source: the law
   over all targets, each target's cell, gates, gate references and route fallbacks;
1. the model level, `params[arg]`, which also feeds regime functions.

There is no `params["edges"][arg]` level. A regime-level value never reaches an edge
callable: `params[source][arg]` feeds only the source regime's own functions, and when
none of them reads it, it is an unknown key: `InvalidParamsError` names its
`params["edges"]` path. A value supplied at two levels for one parameter raises
`InvalidNameError`. `fixed_params` take the same paths and levels. A `next_regime` or
`gate` key under a regime raises `InvalidParamsError` naming the new path. Regime
functions keep their levels: function, regime, model.

### Names that become path segments

Every user-chosen name that becomes a parameter-path segment contains no `__` and does
not start or end with `_`: regime (source and target), state, action, function and
constraint names, `gate_references` keys, `routes` keys, stakeholder names, and the
argument names of model functions, which become parameter names. A violation raises when
the regime or model is built, naming the name and its kind. `edges` is reserved: no
regime, function or function argument may take that name. A law argument may not share
its name with a regime.

(laws-at-every-horizon)=

### Laws at every horizon

A declared law is evaluated at every source age with outgoing edges, including ages
where only one destination is declared or left after fixed-zero pruning; there it must
put unit mass on that destination. Its parameter slots are read off the declared
`Transition`, so they do not depend on the number of periods, on age windows or on fixed
values: a law that is never decisive, for instance on a two-period grid, still has
required parameters.

Declare the `Transition` at every horizon and write its law horizon-aware, or let
`ByAge` (e.g. `ByAge.until`) leave single-destination ages unselected. Delete branches
such as `Transition(targets=..., law=...) if len(targets) > 1 else targets` in a model
and the matching branch in its parameter helper:

```python
def retire(*, age: float, retirement_age: float) -> ScalarInt:
    return jnp.where(age < retirement_age, RegimeId.working, RegimeId.retired)


edges = {"working": Transition(targets=working_targets, law=retire)}
params = {"discount_factor": 0.95, "edges": {"working": {"retirement_age": 62}}}
```

A law short of unit mass at such an age is caught by the probability check at
`log_level="debug"`, which names the cells dropped for lack of an edge. Run a model at
that level at least once.

## Name age bounds explicitly

`AgeGrid(start=..., inclusive_stop=..., step=...)` includes its final age.
`AgeRange(start=..., exclusive_stop=...)` excludes its stop. The old `stop` keywords are
removed. `ByAge.until(stop_age_exclusive=...)` continues to use its existing explicit
bound name. Every edge refers to source ages and lands at the next grid coordinate,
including on irregular or multi-year grids.

## Declare initial nodes

Replace `initial_regimes` with `initial_nodes`. Prefer explicit pairs:

```python
initial_nodes = ((60, "working"), (62, "retired"))
```

Selector mappings such as `{60: "working", 62: "retired"}` remain accepted convenience.
There is no default. These declarations govern admissible starts; subject weights and
initial states still belong to `Population` and `InitialConditions`.

## Migrate phase differences and inspection

A shared edge mapping applies to both phases. A former `Phased(solve=..., simulate=...)`
regime law stays a `Phased`, inside the law of one `Transition` whose targets both
phases share:

```python
ALIVE_AGES = AgeRange(start=60, exclusive_stop=75)
edges = {
    "alive": Transition(
        targets={"alive": ALIVE_AGES, "dead": ALIVE_AGES},
        law=Phased(solve=perceived_survival, simulate=realized_survival),
    ),
}
```

Only perceived versus realized connectivity, destinations that differ between the
phases, wraps the whole mapping: `Model(edges=Phased(solve={...}, simulate={...}))`,
each phase's mapping declaring its own `Transition`. A source mapped to
`Phased(solve=Transition(...), simulate=Transition(...))` is refused with the supported
form. State handoffs stay phase-specific on the source regime. Physically visited nodes
require solve values plus their perceived dependencies; extra valued nodes do not imply
realized visits.

Inspect `model.graph.edges.solve` and `.simulate` for declared exact source ages,
`.solution` and `.simulation` for effective period-indexed graphs, `.nodes` for valued
pairs and `.visited_nodes` for realized pairs. `.pruned_edges["solve"]` and
`["simulate"]` map `(source_age, source, target)` to `"fixed_zero_probability"`.
Construction-fixed scalar zero proofs can prune effective edges; runtime probability
zeros cannot. Declared edges remain inspectable and probability mass is never
renormalized.
