---
title: Age-specialized functions and grids
---

# Age-specialized functions and grids

Use age specialization when a function implementation or continuous-state grid changes
with age. If an ordinary function can simply take `age`, prefer that simpler form.

A typical use is a cohort moving through different policy years: the net-income function
at age 58 closes over a different tax system from the function at age 63.

## Age-specialized functions

`AgeSpecializedFunction(build, signature)` resolves one concrete function per active age
when the model is built.

```python
from lcm import AgeSpecializedFunction


def make_net_income(age):
    policy = load_policy_environment(year_for(age))

    def net_income(gross_income):
        return policy.apply(gross_income)

    return net_income


functions = {
    "net_income": AgeSpecializedFunction(
        build=make_net_income,
        signature=lambda age: year_for(age),
    )
}
```

The three user obligations are:

1. every function returned by `build(age)` has the same call signature;
1. equal `signature(age)` values imply identical closure behavior;
1. `build(age)` is deterministic and side-effect-free, because model construction may
   resolve the same age multiple times.

The signature is a correctness precondition and a compilation-reuse key. Include every
varying ingredient when in doubt.

A policy-dependent law of motion remains a plain transition and reads an age-specialized
helper:

```python
functions = {
    "new_pension_points": AgeSpecializedFunction(
        build=make_points,
        signature=lambda age: policy_year(age),
    )
}
state_transitions = {
    "pension_points": lambda pension_points, new_pension_points: (
        pension_points + new_pension_points
    )
}
```

Do not place the marker directly in `state_transitions`.

## Age-specialized grids

`AgeSpecializedGrid(build, signature)` varies a continuous state's numerical grid with
age:

```python
from lcm import AgeSpecializedGrid, LinSpacedGrid

states = {
    "assets": AgeSpecializedGrid(
        build=lambda age: LinSpacedGrid(
            start=borrowing_floor(age),
            stop=ASSET_MAX,
            n_points=40,
        ),
        signature=lambda age: borrowing_floor(age),
    )
}
```

The grid class, number of nodes, shape, and dtype remain fixed; bounds or node values
may change. Equal signatures must resolve to identical grids.

Use an age-specialized grid only for a continuous state whose points are known at model
construction. Runtime-supplied points use `IrregSpacedGrid(n_points=...)` instead.

## Gate references and leg fallbacks on an age-specialized regime

A `ValueDependentTransition` reads other regimes' values within one period — its
`gate_references` read a reference regime's value function, and a shut gate reads the
route's `fallback` regime's value at a projected coordinate. Either regime may hold its
states on an `AgeSpecializedGrid`.

Every such read is measured against the grid of **the period whose value is being
folded**, not against some other age at which that regime is also active. This is worth
stating explicitly because nothing in the arrays would reveal a mistake: `n_points` is
fixed across ages while the bounds move, so every period's value array has the same
shape, and a read against another age's nodes lands on a different point of an otherwise
correctly shaped array.

A route's `fallback` projection owes one coordinate function per state the fallback
regime carries **in simulation**, and an age-specialized state is one of those like any
other:

```python
StakeholderRoute(
    fallback=ProjectedRegimeValue(
        regime="annuity",
        # `annuity` holds `principal` on an `AgeSpecializedGrid`. The projection
        # owes a coordinate on it exactly as it would on a plain grid state.
        projection={"principal": principal_from_balance},
    )
)
```

A `gate_references` projection instead owes one coordinate per state of the reference
regime's *value function*, i.e. its solve states. The two sets differ only by the states
a regime carries in simulation alone.

## Before using it

Age specialization creates period-specific programs. Use it only when the economic
implementation changes, and keep the number of distinct signatures as small as
correctness permits. If the regime uses a specialized solver, verify that every
age-specific function and grid still satisfies that solver's assumptions.

Exact placement rules—including terminal regimes, `Phased`, transition restrictions,
DataFrame targets, and age-specific projections—are in
[Transitions and phase specialization](../reference/transitions.md). See
[Scaling, memory, and hardware](../methods/performance_scaling.md) for the compilation
consequences.
