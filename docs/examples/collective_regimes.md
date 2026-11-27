---
title: Collective regimes
---

# Collective regimes

Two small models of a couple. The first isolates the shared argmax: one action, two
members who value it differently. The second adds what makes a couple more than a joint
utility function. It declares a composition action whose rule reads the value of each
partner's outside option in the *same* period. When the couple separates, the
composition action sends each partner to a different destination. Both models use two-
and three-point grids and one binary action, so every value below can be recomputed by
hand. The grids are curated for exposition and tests, not calibrated.

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/collective_regimes.py)

## Shared decision: one action, two values

The couple chooses work or leisure at a wage of 8 or 40. Member `f` gains 30 from
leisure and consumes her wage when working; member `m` values consumption twice as
strongly and gets nothing from leisure. One `CollectiveUtility` declares both, and its
keys are the regime's members in the order written:

```python
couple = Regime(
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

No `objective` is declared, so the members carry equal Pareto weights. The couple's only
outgoing edge, `edges={"couple": {"couple_terminal": 0}}`, is its whole transition law.

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
written. Both entries of a row are read at one common choice, so they are not two
separate optimizations. At the low wage node the couple works — working lifts next
period's wage to 40, where `m`'s consumption value dominates — and `f` is carried to 46
even though the leisure branch would have been worth 58.5 to her.

## Participation and dissolution

The second model runs three ages. `married` decides at age 0;
`married_with_participation` is the couple at age 1, where either partner may leave;
`married_terminal` is the couple at age 2, the last age. `single_f` and `single_m` are
the outside options, terminal from age 1 on. The wage is fixed at 1, 2, or 3 and never
changes, so the wage node alone decides the outcome. `f` alone is worth 5.5 at the
middle node and 1.5 elsewhere; `m` alone is worth 1 everywhere. At age 1 each partner
compares their own action value with the value of that same-period outside option, net
of a divorce cost of 0.5 for `f` and 0.2 for `m`. The regime's `CompositionAction` names
where each partner goes and the rule that decides, here unilateral divorce with
participation constraints on actions:

```python
composition_action = CompositionAction(
    destinations={
        "f": Destination(
            regime="single_f",
            projection={"wage": identity_wage},
            utility_adjustment=divorce_cost_f,  # -0.5
        ),
        "m": Destination(
            regime="single_m",
            projection={"wage": identity_wage},
            utility_adjustment=divorce_cost_m,  # -0.2
        ),
    },
    rule=PCOnActions(tie="leave"),  # stay only if both strictly prefer it
)
```

At wages 1 and 3 some action beats both outside options. At the middle node `f`'s
outside option jumps to 5.5 - 0.5 = 5.0, and the best the couple can offer her is 4, so
the couple separates and each partner gets their single value net of the cost:

```python
from lcm_examples.collective_regimes import get_dissolution_model

model = get_dissolution_model()
solution = model.solve(
    params=get_params(),
    log_level="debug",
)

solution.values[1]["married_with_participation"]
```

```text
[[2.  1. ]
 [5.  0.8]
 [6.  3. ]]
```

`married` reaches `married_with_participation` at age 1 through a plain edge. The
decision to stay or to separate is that regime's own composition action:

```python
edges = {
    "married": {"married_with_participation": 0},
    "married_with_participation": {"married_terminal": 1},
}
```

In `married` and `married_terminal` work pays nothing and costs each partner 0.25, so
the couple takes leisure and its flow utility is zero. The value of `married` is
therefore the continuation discounted at 0.95: `(1.9, 0.95)` at wage 1, `(5.7, 2.85)` at
wage 3, and at the middle node the values after separation net of the divorce cost,
`(0.95 × 5.0, 0.95 × 0.8) = (4.75, 0.76)`.

## Simulating one role at a time

Member identity is per agent. Every agent here starts inside a collective regime, so
each declares a role in `initial_conditions`. That role decides which destination its
row takes if the couple separates:

```python
import jax.numpy as jnp

initial_conditions = {
    "wage": jnp.array([1.0, 2.0, 3.0]),
    "age": jnp.zeros(3),
    "regime_id": jnp.full(3, model.regime_names_to_ids["married"], dtype=jnp.int32),
    "own_member": jnp.full(3, model.member_names_to_ids["f"], dtype=jnp.int32),
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
   agent_id  period                 regime_name own_member  value  value_f  value_m  wage     work  age
0           0       0                     married               f    NaN    1.900     0.95   1.0  leisure    0
1           0       1  married_with_participation               f    NaN    2.000     1.00   1.0     work    1
2           0       2            married_terminal               f    NaN    0.000     0.00   1.0  leisure    2
3           1       0                     married               f    NaN    4.750     0.76   2.0  leisure    0
4           1       1                    single_f             NaN    5.0      NaN      NaN   2.0      NaN    1
5           2       0                     married               f    NaN    5.700     2.85   3.0  leisure    0
6           2       1  married_with_participation               f    NaN    6.000     3.00   3.0     work    1
7           2       2            married_terminal               f    NaN    0.000     0.00   3.0  leisure    2
```

The middle-wage agent leaves for `single_f` at age 1; the other two stay. Collective
rows publish one value column per member and carry their role in `own_member`; singleton
rows publish the scalar `value` and hold no role. A cohort has a fixed size, so a
separating row does not split into two linked rows. It follows the destination of the
role it carries. Seed a second cohort with `model.member_names_to_ids["m"]` to watch the
same wage nodes from the other side.

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

- [Several members and the decision on the composition](../user_guide/collective_regimes.md)
  — what each declaration means and what its parts may read
- [Collective regimes](../reference/collective_regimes.md) — the exact contracts
- [Regimes](../user_guide/regimes.ipynb) — the general model workflow
- [Collective resource contract](../development/collective_resource_contract.md) — the
  cost of these declarations at scale
