---
title: Migrating regime graph declarations
---

# Migrating regime graph declarations

The model owns every regime transition through a required `edges` argument, both its
structure and its law. Initial age–regime pairs are required explicitly through
`initial_nodes`.

## Make starting coordinates explicit

Use `initial_nodes=InitialNodes(by_age={60: "working"})`. Legacy tuples such as
`((60, "working"),)` and bare selector mappings remain accepted as constructor inputs.
`model.initial_nodes` always returns a normalized immutable `InitialNodes`, so replace
loops over it with loops over `model.graph.initial_nodes` for exact `(age, regime)`
pairs, or inspect `model.initial_nodes.by_age` for an age-to-regime-tuples mapping.
Passing `model.initial_nodes` into another model remains supported.

For annotations accepting all constructor forms, use `lcm.typing.UserInitialNodes`.
`InitialNodes`, including its re-export from `lcm.typing`, now names the declaration
class.

Declarations snapshot their containers: changing source edge dictionaries, nested law
mappings, or starting-regime collections cannot change a constructed model's published
configuration. To change declarations, construct a new model.

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
| value-reading `Regime.gated_edges`      | explicit `Transition(arrivals=...)`            |
| `Regime.decomposed_transition`          | `model.graph.laws[name].decomposed_transition` |
| the law passed as `regime_transitions=` | `model.declared_transitions[phase][name].law`  |

`model.declared_transitions["solve"][name]` and
`model.declared_transitions["simulate"][name]` are the `Transition` a source declared,
exactly as declared: its `law`, its `arrivals` and its declared or derived `targets`.
Edges declared for both phases give both phases the same `Transition`; a `Phased`
`edges` mapping gives each phase its own. A source without a `Transition` in a phase is
absent from that phase's mapping. `model.graph.laws[name]` is the law the solver and
simulator evaluate: bound to the graph, pruned of fixed-zero cells and lowered to the
ages the starts demand. A regime is terminal when it has no outgoing edge in
`Model(edges=...)`, and `model.graph.laws[name].terminal` says so:

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

With declared `targets`, a `ByAge` law may leave single-destination ages unselected. A
law that does reach such an age is evaluated there and must put unit mass on its one
destination; see [Laws at every horizon](#laws-at-every-horizon).

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

A law that names its destinations — a per-target mapping, a regime name, or a `ByAge` /
`Phased` of those — makes `Transition.targets` optional: the destinations are derived
from the law's keys at the non-final source ages each case covers, plus each arrival's
route fallback regimes at the source ages of its scheduled target. A `targets` supplied
with such a law must equal the derived mapping exactly, or `Model(...)` raises a
`ModelInitializationError` listing both. Where a per-target law reaches more ages than
the declared `targets`, wrap it in `ByAge(cases={ages: law})` restricted to those ages.
A law over all targets still requires `targets`.

When the targets are derived from a `ByAge` law, every source age with an edge needs a
case: an age no case selects has no edge. At an age with a single certain destination,
the case is that regime's bare name, not a per-target cell holding a probability
function. Per-target cases and bare-name cases mix freely in one `ByAge`:

```python
edges = {
    "worker": Transition(
        law=ByAge(
            cases={
                AgeRange(exclusive_stop=64): {
                    "worker": StochasticTransition(func=survive),
                    "dead": StochasticTransition(func=die),
                },
                64: "retiree",
            }
        ),
    ),
    "retiree": {"dead": AgeRange(start=65)},
}
```

Here `worker` reaches `worker` and `dead` at every age below 64 and `retiree` at 64.

(current-shocks-and-exits)=

## Declare current shocks independently of states

Declare current IID laws in `Model(shocks=...)` and name each consuming group in
`Regime(shocks=...)`. The physical shock names are function arguments; they are not
persistent state grids or stored value axes. For example, these declaration fragments
use a runtime normal scale:

```python
from lcm import Regime
from lcm.shocks import MonteCarlo, Normal, ShockGroup

shocks = {
    "risk": ShockGroup(
        shocks={"eps": Normal(mu=1, sigma=None)},
        nodes=MonteCarlo(2, seed=7),
    )
}
regime = Regime(
    functions={"utility": utility},
    actions=actions,
    shocks=("risk",),
)
params = {"risk": {"eps__sigma": 2.0}}
```

Pass `shocks` to the model alongside its regimes, lifecycle and edges. A function
reading `eps` receives its physical realization. The finite `nodes` rule approximates
expectations during solving; it does not restrict simulation to those nodes. Simulation
uses fresh draws from the declared law. Keep genuinely persistent processes and their
state transitions separate from this current-shock declaration.

For tagged simulation inputs, use `BaseInnovations(values=...)` for innovations to
transform through the current law, or `PhysicalValues(values=...)` for supplied physical
realizations. A history is indexed by original subject row and computational period.
Supplying a value does not declare it observed by another decision.

Partial observation of a correlated group and partial correlated physical overrides are
unsupported. A wholly hidden group retains its joint law; supported overrides use a
complete physical frame or named base innovations. Independent hidden shocks remain
available.

A group read by an endogenous-grid regime (`DCEGM`, `NBEGM`, `NEGM`, `NNBEGM`) is solved
on the node-state axis, so the migrated model solves every shape its node-state version
did: several states, discrete states and actions, and chains of endogenous-grid regimes.
Published values integrate the shock's nodes, and simulation draws and overrides follow
the rules above. The supported laws and rules are listed in
[Named current-IID shocks in endogenous-grid regimes](../reference/solvers.md#current-iid-egm).

(migrating-gates)=

## Preserve the economic date of a value-dependent decision

Use `Regime(exits=...)` when a subject chooses whether to leave **in the current
period**, before its final action. The exit target is a same-date Participating policy
problem at the declared projection. This singleton fragment prices a target without
observing its offer:

```python
from lcm import Exit, ExitTarget

exits = {
    "leave": Exit(
        targets={
            "self": ExitTarget(
                regime="target",
                projection={},
                utility_adjustment=adjustment,
            )
        },
        rule="unilateral",
    )
}
```

Declare the target's group on its own `Regime(shocks=...)`. Add `observes=("offer",)` to
the `Exit` only when that physical target offer is visible at the exit choice. Otherwise
pricing integrates the hidden offer; after the irrevocable move the target acts with its
actual realization. An exit's utility adjustment affects its benchmark and is paid once
on settlement. It reads the exit source query's state, helper and observed-information
context; `ExitTarget.projection` does not remap its inputs. When that source is itself a
projected household query, its mapped household state is available to the adjustment. A
lagged marital-status indicator there can make the adjustment zero for newlyweds.

An old edge decision after a scheduled transition belongs in `Transition(arrivals=...)`:
ordinary transition laws and state laws first produce the next-period landing, then its
`ArrivalDecision` evaluates, then the selected regime acts. Moving it into a current
`Exit` would change its economic date.

This fragment, taken from the shipped collective example, reads the continuing
household's policy support instead of a stored dissolution flag:

```python
from lcm import ArrivalDecision, ArrivalRead

arrival = ArrivalDecision(
    predicate=has_policy_support,
    reads={
        "target_support": ArrivalRead(
            regime="married_with_participation",
            stakeholder="f",
            kind="Participating",
            field="support",
            projection={"wage": identity_wage},
        )
    },
    routes=stakeholder_routes,
)
```

Place it under `Transition(arrivals={"married_with_participation": arrival}, ...)`. Each
source stakeholder's route declares its continuing role and its own projected fallback.
The decision owns only subjects arriving on that incoming edge; unrelated ordinary draws
and initial occupants do not borrow its fallback.

`ArrivalRead(kind="Raw")` reads the unconstrained-by-participation action problem;
`kind="Participating"` reads the actions kept by its exit rule. `field="support"`
returns Boolean policy support rather than a numeric value. `kind="IntegratedFull"`
explicitly reads the stored shock-integrated Full value and exposes neither policy
support nor realized physical information. Name the reads in the predicate's signature;
`V_target` is not injected implicitly, and `D_target` is unsupported.

The arrival predicate must return a scalar Boolean. Its ordinary inputs and projections
use the landed target's state and helper vocabulary. An explicit named read binds its
own value operand, even when a target state or helper has the same name. Declare
physical information with `ArrivalDecision.observes` or the individual
`ArrivalRead.observes`; a supplied simulation path never enlarges that information.
Preserve the original comparison, including its tie rule, target role, projection and
fallback, when replacing a value-reading `Gate`.

Custom current exit rules return `RuleResult` with a declared alternative identifier,
and declare whether they read outside values through `reads_outside_values`. They do not
return a Boolean arrival verdict. A custom source rule combined with source EV1 tastes
is explicitly refused; this does not refuse a custom rule pricing a hidden EV1 target.

See the executable declaration examples in `lcm_examples.collective_regimes` and
`tests/simulation/test_current_exits.py`. The uncertainty and collective reference pages
still describe parts of the former state-grid/value-gate interface; their rewrite is
pending. Use the declarations above for this migration.

(migrating-edge-parameters)=

## Move transition parameters under `edges`

The regime-transition law, its arrival decisions, reads and route fallbacks are declared
in `Model(edges=...)`, and their parameters live there too: the parameter path of an
edge-declared callable is its declaration path under `params["edges"]`. The source
regime owns none of them, so its branch of the parameters has no `next_regime` or `gate`
entry. Per-target state laws (`state_transitions={state: {target: law}}`) belong to the
source regime and keep their paths, `params[source][target]["next_<state>"]`.

| Key under the source regime                                       | Path under `params["edges"]`                                            |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `[source]["next_regime"][arg]` (a law over all targets)           | `[source][arg]`                                                         |
| `[source][target]["next_regime"][arg]` (a per-target cell)        | `[source][target][arg]`                                                 |
| the same, for an arrival target's cell                            | `[source][target][arg]`                                                 |
| `[source][target]["leg_fallback_<regime>_<state>"][arg]`          | `[source][target]["routes"][route]["fallback"][state][arg]`             |
| the same, solve side of a `Phased` fallback                       | `[source][target]["routes"][route]["fallback"]["solve"][state][arg]`    |
| `[source][target]["simulate_leg_fallback_<regime>_<state>"][arg]` | `[source][target]["routes"][route]["fallback"]["simulate"][state][arg]` |
| `[source][arg]`, where the law or an arrival reads it             | `[source][arg]`, or the model level                                     |

Use `get_params_template()` for the exact predicate, read-projection and route paths of
arrival decisions. Current `Exit` declarations belong to the regime, so their parameters
stay in that regime's parameter branch.

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
   over all targets, each target's cell, arrival predicates, read projections and route
   fallbacks;
1. the model level, `params[arg]`, which also feeds regime functions.

`get_params_template()` lists every slot at its declaration path, the most specific
level; supply each slot at any one of the three. An argument several callables below one
source read is supplied once at the source level. With per-target cells into `working`
and `dead` that both read `survival_rate`, the template lists
`params["edges"]["working"]["working"]["survival_rate"]` and
`params["edges"]["working"]["dead"]["survival_rate"]`, and one value covers both:

```python
params = {"edges": {"working": {"survival_rate": 0.98}}}
```

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
constraint names, arrival `reads` keys, `routes` keys, stakeholder names, and the
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

Declare the `Transition` at every horizon and write its law horizon-aware, or, with
declared `targets`, let `ByAge` (e.g. `ByAge.until`) leave single-destination ages
unselected. Delete branches such as
`Transition(targets=..., law=...) if len(targets) > 1 else targets` in a model and the
matching branch in its parameter helper:

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
