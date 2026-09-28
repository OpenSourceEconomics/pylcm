(dated-regime-graph)=

# Declare when a regime's transition law applies

```{important}
This page documents the proposed v3 interface. The new schedule/support declarations
and membership views require the upstream implementation; they are not available
in unchanged PR #474. Code snippets are declaration fragments whose economic
functions, grids and regime registry come from the surrounding model.
```

A regime is a local economic problem. Its dated instances are that same problem at
particular ages. Define the ages of a nonterminal regime through its transition schedule
instead of keeping an `active` predicate consistent with a separate law. pylcm checks
that every declared destination exists next period. It never removes an invalid target
to make a probability row fit.

All declared problems are solved, including values that a caller wants to compare but a
simulation never visits. Simulation entry does not select the solve domain.

## Terminality is unchanged

```python
dead = Regime(transition=None, functions={"utility": bequest})
```

`transition=None` means the regime is terminal. Its payoff is available at every model
age, and a history entering it ends after evaluating that payoff. Terminal is not a
synonym for last age.

```python
transition = "dead"
```

This is a **nonterminal** law: take the current decision, evolve the states, then enter
the dead regime at the next age. Replacing it with `None` removes continuation and
changes the model. There is no new terminal flag or terminal schedule. Do not put `None`
inside any transition wrapper, including a schedule's default or exit.

Deleting an old terminal activity restriction may add terminal values at earlier ages.
It does not add transitions from living regimes to them. Review that output key change
explicitly; terminal primitives must be defined where evaluated. A one-period model
consisting only of terminal problems is a valid special case.

## A stage exit without a second switch parameter

```python
from lcm import ByAge

transition = ByAge.until(
    62,
    law=work_transition,
    then="early_retirement",
    start=25,
)
```

Here 62 is the **destination age of the exit**. The work regime covers sources from 25
up to, but not including, 62. Its ordinary law applies before the last of those sources;
the exit applies at the grid predecessor of 62. On an annual grid that is age 61, on a
quarterly grid 61.75. It declares no work problem at 62. The early-retirement regime
must have coverage there.

No `AGE - 1` arithmetic or new age-type API is needed. Resolution uses the existing
`AgeGrid.exact_values`. Boundary and start must be grid points. Changing time units
still requires appropriate economic recalibration; the schedule does not annualize or
rescale probabilities and flows for you.

For a general partition:

```python
from lcm import AgeRange, ByAge

transition = ByAge(
    {
        AgeRange(25, 60): before_60,
        AgeRange(60, 65): from_60_to_65,
    }
)
```

`AgeRange` is half-open and selects existing grid points. Scalars/tuples select exact
coordinates; Python `range` literally selects integers, not fractional dates between
birthdays. Case overlap, off-grid explicit points and empty cases raise. `default=law`
fills remaining non-final source positions; no default leaves them undeclared. A plain
nonterminal law similarly broadcasts to non-final positions. An explicit nonterminal law
at the last age is an error, not automatic termination.

A self-loop at the last non-final source still needs an explicit exit: otherwise it
would target a living problem with no final-age law. Numeric survival zero is not a
declaration that this structural edge is absent.

Inspect a schedule without executing economic functions:

```python
selected = transition.resolve(ages).at(60)
```

## Keep ordinary transition forms

A fixed target is a string. A state/action-dependent deterministic selector is:

```python
from lcm import Choose

transition = Choose(choose_regime, targets=("working", "retired"))
```

The function returns the existing global regime code, not an index into `targets`. It
keeps the deterministic execution path and adds no categorical draw.

A full probability vector retains its current output layout:

```python
from lcm import MarkovTransition

transition = MarkovTransition(
    regime_probabilities,
    targets=("working", "retired", "dead"),
)
```

The vector still has one cell per model regime in existing ID order. All cells outside
declared support must be exactly zero. Targets remain structural possibilities even when
their current probability is zero.

Per-target functions keep their current wrappers:

```python
transition = {
    "working": MarkovTransition(p_work),
    "retired": MarkovTransition(p_retire),
    "dead": MarkovTransition(p_die),
}
```

Mapping keys declare support; each function returns one probability. Keep these
wrappers, including for a constant probability. This migration does not add literal
numeric cells or bare probability callables to the ordinary mapping form. A one-target
Markov mapping remains stochastic; use a string to declare a fixed deterministic route.
Existing `ValueDependentTransition` cells retain their probability convenience, gates,
references and stakeholder routes.

Every selected law must be complete, with unit probability mass. Regime-selection checks
are not disabled by log verbosity. Two targets sharing a function are not necessarily
invalid (both may be 0.5); actual rows are checked, not callable identity. State/joint
transitions retain their existing validation as well.

## Optional entry permissions, ordinary solution access

Most calls need no extra entry declaration:

```python
model = Model(ages=ages, regimes=regimes, regime_id_class=RegimeId)
```

The following are optional arguments to that constructor:

```python
initial_regimes = {25: ("single", "couple")}  # only those two pairs
initial_regimes = "single"  # all covered ages of single
initial_regimes = {}  # no external entries; solve unchanged
```

A bare name/sequence selects the named regimes' covered nodes. Explicit mapping rules
form Cartesian pairs and union across rules; each pair must be covered. Entry overlap is
harmless set union, unlike ambiguous overlapping law cases. Each real simulation row is
checked pairwise before padding and user-law calls. State, role, categorical, shape and
feasibility checks still apply independently.

Use two immutable membership views:

```python
model.reachability.nodes  # every declared (age, regime) problem
model.initial_nodes  # admissible external entry pairs
```

Keep existing phase reachability queries and period-indexed solution access:

```python
model.reachability.solution.targets(period=0, source="working")
solution[period][regime_name]
```

These target queries retain their existing direct/attempted-target meaning; gate-closed
landings are routes, not extra cells in the selection lottery. A missing solution cell
raises using existing error behavior. There is no second graph API, new result wrapper
or required cached history set.

An entry restriction never deletes a counterfactual value. A population recipe still
owns the distribution and any age/regime correlation; permissions are not weights and
must not renormalize the recipe. The shared drawer checks `initial_nodes` instead of
calling `.active`. Model-dependent draw grids can be constructed later.

## Preserve phases and value dependencies

Keep `Phased`'s existing economic meaning and put it inside age cases:

```python
from lcm import Phased

transition = ByAge.until(
    100,
    law=Phased(solve=perceived_mortality, simulate=realized_mortality),
    then="dead",
    start=66,
)
```

Existing form/support/gate compatibility rules remain; both sides declare the same
direct support. Do not wrap separate schedules inside the two sides of `Phased`. Carried
states, current utility and continuation beliefs are not reinterpreted.

A local same-period outside-option read orders local problems. A gate on a transition is
instead a requirement of that **source edge**, evaluated at the next-age landing point.
If a schedule removes the gate at an age, its edge fold and references are not needed
there. Reciprocal marriage/divorce edges must not become a false local same-period
cycle.

## Parameters, solvers and reuse

Schedules add no parameter-tree case level. Requirements are unioned over all selected
cases/phases at existing paths. Reusing a path shares a parameter; incompatible
schemas/namespace roles raise. Use distinct names or existing period-indexed arrays for
different values. Late-only arguments and states survive pruning. State-law keys alone
never declare regime edges.

A schedule normalizes before ordinary solver validation. Wrapping an unchanged supported
law does not itself make a solver unsupported. Genuine unsupported variation raises
explicitly; it must not silently narrow actions or swap solvers. GridSearch is the first
acceptance target for this migration.

Reuse identical numerical functions across cases. Different destination grids may still
need different continuation programs. Measure model build, backend compile and warm
runtime; neither fewer Python lines nor more visible node metadata proves a speedup. See
[the migration guide](migrating_dated_regimes.md).
