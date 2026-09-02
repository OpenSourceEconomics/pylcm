---
title: Collective regimes
---

# Collective regimes

Two small household models. The first isolates the shared argmax: one action, two
members who value it differently. The second adds what makes a household more than a
joint utility function — a participation constraint reading each partner's outside
option in the *same* period, and a transition routing each partner somewhere different
when the household ends. Both use two- and three-point grids and one binary action, so
every value below can be recomputed by hand; the grids are curated for exposition and
tests, not calibrated.

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/collective_regimes.py)

## Shared decision: one action, two values

The household chooses work or leisure at a wage of 8 or 40. Stakeholder `f` gains 30
from leisure and consumes her wage when working; stakeholder `m` values consumption
twice as strongly and gets nothing from leisure. One `CollectiveUtility` declares both,
and its keys are the regime's stakeholders in the order written:

```python
couple = Regime(
    transition=to_couple_terminal,
    active=lambda age: age < 1,
    states={"wage": LinSpacedGrid(start=8.0, stop=40.0, n_points=2)},
    state_transitions={"wage": next_wage},
    actions={"work": DiscreteGrid(category_class=Work)},
    functions={
        "utility": CollectiveUtility(
            utilities={"f": utility_f, "m": utility_m},
        )
    },
)
```

No `objective` is declared, so the stakeholders carry equal weight.

```python
from lcm_examples.collective_regimes import get_params, get_shared_decision_model

model = get_shared_decision_model()
solution = model.solve(params=get_params(), log_level="debug")

solution.values[0]["couple"]
```

```text
[[ 46.  92.]
 [ 78. 156.]]
```

Rows are the two wage nodes, columns `f` and `m` in the order the `utilities` keys were
written. Both entries of a row are read at one household choice, so they are not two
separate optimizations. At the low wage node the household works — working lifts next
period's wage to 40, where `m`'s consumption value dominates — and `f` is carried to 46
even though the leisure branch would have been worth 58.5 to her.

## Participation and dissolution

The second model runs three ages. `married` decides at age 0;
`married_with_participation` is the couple at age 1, where either partner may walk out;
`married_terminal` closes the household at age 2. `single_f` and `single_m` are the
outside options, terminal from age 1 on. The wage is fixed at 1, 2, or 3 and never
moves, so the wage node alone decides the outcome. `f` alone is worth 5.5 at the middle
node and 1.5 elsewhere; `m` alone is worth 1 everywhere. Each partner compares their own
action value against that same-period outside option through a
`ValueDependentConstraint`:

```python
def participation_f(*, Q_f, V_single_f_ref):
    return Q_f >= V_single_f_ref - 0.5


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
    # ... and the mirror image for m against single_m
}
```

At wages 1 and 3 some action clears both constraints. At the middle node `f`'s outside
option jumps to 5.5, so her constraint demands 5.0 and the best the household can offer
her is 4 — no action clears both, and the cell holds no viable household. That is a
structural outcome rather than a number: the cell carries the `-inf` sentinel, and the
regime publishes a dissolution flag.

```python
from lcm_examples.collective_regimes import get_dissolution_model

model = get_dissolution_model()
solution = model.solve(
    params=get_params(),
    log_level="debug",
)

from lcm.solver_api import DISSOLUTION_FLAG

dissolution_flags = solution.replay_artifacts.project(DISSOLUTION_FLAG)
dissolution_flags[1]["married_with_participation"]
# Array([False,  True, False], dtype=bool)
```

The solve logs `Inf in V_arr for regime 'married_with_participation'` at that age, which
is the sentinel doing its job rather than a defect.

`married` reaches age 1 through a `ValueDependentTransition` keyed by
`married_with_participation` — the branch where the gate is **open** and the couple
keeps going — with the gate reading the target's dissolution flag. Each stakeholder's
route names the role they take inside the surviving couple and the singleton value they
fall back to when the gate shuts:

```python
transition = {
    "married_with_participation": ValueDependentTransition(
        probability=MarkovTransition(probability_one),
        gate=lambda D_target: ~D_target,
        routes={
            "f": StakeholderRoute(
                target_stakeholder="f",
                fallback=ProjectedRegimeValue(
                    regime="single_f",
                    projection={"wage": identity_wage},
                ),
            ),
            # ... and the mirror image for m into single_m
        },
    )
}
```

Keying this edge by `single_f` would send both partners into `single_f` whenever the
household survives, which is why the key is the couple; see
[the gate-open rule](../user_guide/collective_regimes.md#the-key-is-always-the-gate-open-target).
`married` produces no flow utility of its own, so its value is the routed continuation
discounted at 0.95: `(1.9, 0.95)` at wage 1 and `(5.7, 2.85)` at wage 3, against `f`'s
fallback `0.95 × 5.5 = 5.225` at the middle node.

## Simulating one role at a time

Stakeholder identity is per subject. Every subject here starts inside a collective
regime, so each declares a role in `initial_conditions`, and that role decides which
fallback its row takes if the household dissolves:

```python
import jax.numpy as jnp

initial_conditions = {
    "wage": jnp.array([1.0, 2.0, 3.0]),
    "age": jnp.zeros(3),
    "regime_id": jnp.full(3, model.regime_names_to_ids["married"], dtype=jnp.int32),
    "own_stakeholder": jnp.full(
        3, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
    ),
}

result = model.simulate(
    params=get_params(),
    initial_conditions=initial_conditions,
    solution=solution,
    log_level="debug",
    seed=0,
)

result.to_dataframe()
```

```text
   subject_id  period                 regime_name own_stakeholder  value  value_f  value_m  wage     work  age
0           0       0                     married               f    NaN    1.900     0.95   1.0  leisure    0
1           0       1  married_with_participation               f    NaN    2.000     1.00   1.0     work    1
2           0       2            married_terminal               f    NaN    0.000     0.00   1.0  leisure    2
3           1       0                     married               f    NaN    5.225     0.95   2.0  leisure    0
4           1       1                    single_f             NaN    5.5      NaN      NaN   2.0      NaN    1
5           2       0                     married               f    NaN    5.700     2.85   3.0  leisure    0
6           2       1  married_with_participation               f    NaN    6.000     3.00   3.0     work    1
7           2       2            married_terminal               f    NaN    0.000     0.00   3.0  leisure    2
```

The middle-wage subject leaves for `single_f` at age 1; the other two stay. Collective
rows publish one value column per stakeholder and carry their role in `own_stakeholder`;
singleton rows publish the scalar `value` and hold no role. A cohort has a fixed size,
so a dissolving row does not split into two linked rows — it follows the route belonging
to the role it carries. Seed a second cohort with `model.stakeholder_names_to_ids["m"]`
to watch the same wage nodes from the other side.

## Structure

- **Shared-decision model**: 2 ages; regimes `couple`, `couple_terminal`; one two-point
  wage state; one binary action; `f` and `m` as the shared `CollectiveUtility` keys.
- **Dissolution model**: 3 ages; regimes `married`, `married_with_participation`,
  `married_terminal`, `single_f`, `single_m`; one three-point wage state held fixed by
  `fixed_transition`; one binary action; `f` and `m` in the three collective regimes,
  `single_f` and `single_m` singleton.
- **Solver**: grid search, which is what collective regimes support and what this size
  makes cheap.

## See also

- [Households and value-dependent choice](../user_guide/collective_regimes.md) — what
  each declaration means and what its parts may read
- [Collective regimes](../reference/collective_regimes.md) — the exact contracts
- [Regimes](../user_guide/regimes.ipynb) — the general model workflow
- [Collective resource contract](../development/collective_resource_contract.md) — the
  cost of these declarations at scale
