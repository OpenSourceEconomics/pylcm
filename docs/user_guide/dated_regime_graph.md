(dated-regime-graph)=

# Age-indexed regimes: starting problems, laws and demand

A regime is a local economic problem. Its age-indexed instances are that problem at
particular ages, identified by **age–regime pairs**. A model declares two different
things:

- **which problems a history may start in**, through the required `initial_regimes`;
- **which law governs each regime at each age**, through `regime_transitions`.

The engine derives every problem it has to solve from the first declaration and the
laws. A transition law never makes a problem solved just by being present.

Code blocks marked *fragment* are declaration fragments whose economic functions, grids
and regime registry come from the surrounding model; the complete example below runs as
written.

## Explicit, required starting problems

`Model(..., initial_regimes=...)` is a required keyword-only argument. There is no
default: omitting it, passing `None`, or passing a bare regime name raises. Its elements
are the **admissible roots**: the age–regime pairs where a solved problem may be started
and where simulation may admit a history.

```python
import jax.numpy as jnp

from lcm import AgeGrid, AgeRange, ByAge, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction, interest_rate: float
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption)


def budget(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return consumption <= wealth


wealth_grid = LinSpacedGrid(start=1, stop=100, n_points=20)
consumption_grid = LinSpacedGrid(start=0.5, stop=100, n_points=40)

working = Regime(
    regime_transitions=ByAge.until(
        stop_age_exclusive=63, law="working", then="retired"
    ),
    states={"wealth": wealth_grid},
    state_transitions={"wealth": next_wealth},
    actions={"consumption": consumption_grid},
    constraints={"budget": budget},
    functions={"utility": utility},
)
retired = Regime(
    regime_transitions="dead",
    states={"wealth": wealth_grid},
    state_transitions={"wealth": next_wealth},
    actions={"consumption": consumption_grid},
    constraints={"budget": budget},
    functions={"utility": utility},
)
dead = Regime(
    regime_transitions=None,
    states={"wealth": wealth_grid},
    functions={"utility": bequest},
)

model = Model(
    regimes={"working": working, "retired": retired, "dead": dead},
    ages=AgeGrid(start=60, stop=65, step="Y"),
    regime_id_class=RegimeId,
    initial_regimes={60: "working"},
)
model.initial_nodes  # frozenset({(60, "working")})
```

`model.initial_nodes` is the normalized, immutable set of roots. It carries no
probabilities or population counts. A caller who only solves still declares the problems
it wants values for and simply never calls `simulate`.

Declare roots for the problems a history is intended to start in, not for every
intermediate value. Continuations and value references are followed automatically. A
model whose intended starts span several ages says so — a **late root** is an ordinary
rule:

```python
late_model = Model(
    regimes={"working": working, "retired": retired, "dead": dead},
    ages=AgeGrid(start=60, stop=65, step="Y"),
    regime_id_class=RegimeId,
    initial_regimes={
        60: "working",
        AgeRange(start=63, stop=65): "retired",
    },
)
# initial_nodes: (60, "working"), (63, "retired"), (64, "retired")
```

## Mapping broadcast grammar

`initial_regimes` is a mapping from an age selector to a regime selection:

- **age selector** — an exact grid age, a tuple of exact ages, a Python `range`, or an
  `AgeRange(start=..., stop=...)`;
- **regime selection** — one regime name, or a nonempty sequence of names. A string
  names one regime; it is never iterated as characters.

Each rule contributes the Cartesian product of its selected grid ages and names; rules
are unioned, and duplicate pairs are harmless. A tuple of ages is a set of exact
coordinates, not an interval. `AgeRange` is half-open and selects existing grid points.
`range` selects its integer coordinates, not fractional ages between them.

Model construction rejects an empty mapping, an empty selection, an unknown regime name,
a malformed selector, and an explicit age that is not on the grid (it is never rounded
onto the clock). A root must also be a valid problem: a nonterminal root at the last
grid age, or a root at an age where its regime has no law, raises.

## Laws select behavior; demand selects what is solved

A plain nonterminal law supplies the same law whenever its regime is queried at any age.
`ByAge` supplies the law its case selects at an age, or none. Neither creates a solved
node by being present.

The solved nodes are derived from **demand**: starting from each root at its own age,
the engine follows every declared physical successor and every declared value read, and
solves exactly the age–regime problems they require. A regime whose law exists at an age
but that nothing demands there is not solved there. Changing the roots can therefore
change the set of solved problems; it never changes the value of a problem that stays in
the set.

`ByAge(cases={}, default=law)` is an explicit fallback **law**, filling every unmatched
grid age including the last. It is not a default set of roots. A law at the last age is
only checked for validity if a nonterminal problem is actually required there.

## Keyword-only age selectors and exact boundaries

All declaration constructors take keyword arguments only:
`ByAge(cases=..., default=...)`, `AgeRange(start=..., stop=...)`,
`Choose(func=..., targets=...)`, `MarkovTransition(func=..., targets=...)`, and

```python
# signature of the classmethod ByAge.until
def until(
    *,
    stop_age_exclusive: UserAge | float,
    law: object,
    then: object,
    start_age_inclusive: UserAge | float | None = None,
) -> ByAge: ...
```

Both bounds refer to **source ages**. The helper supplies laws for source grid ages in
`[start_age_inclusive, stop_age_exclusive)`. It selects `law` at the earlier of those
sources and **`then` at the last source grid point below `stop_age_exclusive`**, so the
exit lands at `stop_age_exclusive`, the next grid age. It supplies no law at the stop
age itself or outside the interval.

```python
# fragment
regime_transitions = ByAge.until(
    start_age_inclusive=25,
    stop_age_exclusive=62,
    law=work_law,
    then="early_retirement",
)
```

This does **not** run `work_law` at every source age below 62: the last source uses
`then`.

| Clock near the boundary                         | Last source using `law` | Source using `then` | Destination age of that exit | Source law at 62 from this helper |
| ----------------------------------------------- | ----------------------: | ------------------: | ---------------------------: | --------------------------------- |
| Annual                                          |                      60 |                  61 |                           62 | Absent                            |
| Quarterly                                       |                    61.5 |               61.75 |                           62 | Absent                            |
| Irregular, with final local points 60, 61.5, 62 |                      60 |                61.5 |                           62 | Absent                            |

The stop, and an explicit start, must be exact grid ages with the start before the stop.
An omitted `start_age_inclusive` selects the first clock age; that is a law convenience,
not a root. `then` is a complete nonterminal law and may name a terminal destination,
but it is never `None`. No `age - 1` arithmetic is involved: resolution uses exact grid
positions. A changed time grid still needs its probabilities, payoffs and discounting
recalibrated; selector resolution does not do that.

For a general partition:

```python
# fragment
regime_transitions = ByAge(
    cases={
        AgeRange(start=25, stop=60): before_60,
        AgeRange(start=60, stop=65): from_60_to_65,
    },
)
```

`AgeRange.start` is inclusive and `AgeRange.stop` is exclusive; ordinary `AgeRange`
bounds need not be grid points. Overlapping cases, off-grid explicit ages and empty
cases raise. Inspect a schedule without executing economic functions:

```python
# fragment
regime_transitions.resolve(ages=ages).at(age=60)
```

The ordinary forms are: a regime name for a fixed move; `Choose(func=..., targets=...)`
returning the global regime code; `MarkovTransition(func=..., targets=...)` returning
the full registered probability vector, zero outside `targets`; and a per-target mapping
of `MarkovTransition(func=...)` cells. Every selected law must carry unit probability
mass; a runtime-zero probability is still a structural edge.

## Terminality

A regime is terminal exactly when `regime_transitions is None`:

```python
# fragment
dead = Regime(regime_transitions=None, functions={"utility": bequest})
```

A terminal template is available at every age but evaluated only where demanded. A
string transition into it, `regime_transitions="dead"`, is **nonterminal**: the source
decides, evolves its states, and enters `dead` at the next age. `None` is not allowed
inside `ByAge`, its `default` or `then`, or a `Phased` transition. A required
nonterminal problem at the last grid age is an error even if its survival probability
would be zero. A terminal root is valid, including a one-period model made only of
terminal roots:

```python
terminal_roots = Model(
    regimes={"working": working, "retired": retired, "dead": dead},
    ages=AgeGrid(start=60, stop=65, step="Y"),
    regime_id_class=RegimeId,
    initial_regimes={AgeRange(start=60, stop=66): "dead"},
)
```

## Physical versus value-only dependencies

Demand has two roles:

- **physical** — a history can be in that problem: roots and the realized successors of
  physically reached problems;
- **value-only** — some required program reads that problem's value: perceived
  continuations, attempted targets and gate references of a `ValueDependentTransition`,
  and same-period outside options.

A value-only problem is solved together with its own continuation and references, but
its simulation-only successors are not followed, and it does not become a root. A
declared local outside option is the typical case:

```python
# fragment
constraints = {
    "participation_f": ValueDependentConstraint(
        predicate=participation_f,
        references={
            "V_single_f_ref": ProjectedRegimeValue(
                regime="single_f",
                projection={"wage": identity_wage},
            )
        },
    ),
}
```

With the couple as the only root, `single_f` is solved at every age where the couple's
constraint reads it, although no history starts there. Do not add it to
`initial_regimes` to make the value appear; roots never stand in for prerequisites the
engine infers. See [Collective regimes](../examples/collective_regimes.md) for the full
model.

## Empirical admission

A simulated row is admitted only if its starting age–regime pair is in
`model.initial_nodes`. Solving a value at a pair does not authorize starts there: a
model whose only roots are living regimes rejects initial conditions in `dead` at every
age, including ages where `dead` is solved as a continuation. The roots are
admissibility, not weights; the population recipe owns the distribution and any
age/regime correlation. An explicit counterfactual root also authorizes simulation to
start there.

## Parameters and identity

Roots add no parameter level and do not affect parameter paths, regime IDs, the full
regime-vector axis, period indexing or period-keyed solution access
(`solution[period][regime_name]`). Schedules add no case level to the parameter tree:
requirements are unioned over all selected cases and phases at existing paths, and
incompatible uses of one path raise. `Model.reachability` reports the derived problems;
reading a problem outside the derived set raises rather than returning a filled value.
See [the migration guide](migrating_dated_regimes.md).
