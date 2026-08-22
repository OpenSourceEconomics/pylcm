---
title: Constraints and structured Conditions
---

# Constraints and structured Conditions

Most constraints do **not** need `Condition`. The choice depends on what pylcm must know
in order to express the restriction along the selected solver's candidate route.

## Use an ordinary callable when the Boolean result is enough

```python
import jax.numpy as jnp


def feasible_consumption(consumption, resources):
    return (consumption > 0) & (consumption <= resources) & jnp.isfinite(consumption)
```

`GridSearch` evaluates this predicate on complete state-action candidates. A specialized
solver can also evaluate it at any route stage where every argument exists.

## Use `Condition` when the comparison structure must survive

A callable exposes inputs and an output, but not the fact that it means `savings >= 0`
or `wealth < asset_limit`. Retain that structure when a specialized solver must:

- prove that its construction already enforces the constraint;
- compile a named boundary into its candidate partition;
- determine strict versus inclusive boundary ownership;
- explain precisely why the constraint cannot be evaluated on its route.

You may also choose `Condition` simply because its declarative form is clearer. It does
not expand the mathematical set of Booleans available to grid search, and it does not
make an unsupported constraint supported by an EGM solver.

## Syntax

`ref(name)` refers to a state, action, DAG output, parameter, or declared margin role.

```python
import lcm

# Named value and literal.
nonnegative_savings = lcm.ref("savings") >= 0.0

# Named value and named value.
below_asset_limit = lcm.ref("wealth") < lcm.ref("asset_limit")

# Intersection and union.
working_age = (lcm.ref("age") >= 18) & (lcm.ref("age") < 67)
insured_or_eligible = (lcm.ref("insured") == 1) | (
    lcm.ref("income") <= lcm.ref("eligibility_limit")
)

# Complement.
not_retired = ~(lcm.ref("retired") == 1)

# Conditional requirement.
hours_if_working = lcm.implies(
    premise=lcm.ref("working") == 1,
    consequent=lcm.ref("hours") <= 40,
)
```

Use `&`, `|`, and `~`, not Python's `and`, `or`, and `not`. Comparisons may use `<`,
`<=`, `==`, `!=`, `>=`, and `>`. Strictness determines which side owns an exact boundary
point.

## When a particular solver needs retained structure

| Restriction                                             | Broad callable evaluation                                         | Why retained structure may be needed                                |
| ------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------- |
| Consumption is finite and below resources               | Complete grid-search candidate                                    | Usually not needed                                                  |
| Savings lower bound                                     | Savings may be constructed rather than exposed at every EGM stage | Solver proves it from its grid                                      |
| Liquid-state interval                                   | NBEGM partitions work by declared liquid boundaries               | Compiler needs names, thresholds, and strictness                    |
| Arbitrary Boolean using unavailable intermediate values | Grid search may expose them                                       | Structure cannot help unless the solver implements a proof/compiler |

The last row is important: `Condition` is information, not magic.

## Prefer a margin-specific lower bound

For the borrowing constraint of a declared liquid margin, prefer:

```python
borrowing_limit = lcm.post_decision_lower_bound(
    margin=liquid,
    lower=0.0,
)
```

It prevents the name in the condition from drifting away from the
`LiquidMargin.post_decision_state` name and lets the solver compare the lower bound with
its savings grid.

Budget kinks and cliffs are not ordinary feasibility constraints. Declare them with
[case pieces](case_pieces.md) or a [piecewise-affine schedule](piecewise_affine.md).
