---
title: Collective regimes
---

# Collective regimes

This example asks two related questions:

1. How does a household choose one action when its members value outcomes differently?
1. What happens when no shared action satisfies both members' participation constraints?

The importable definitions live in
[`lcm_examples.collective_regimes`](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/collective_regimes.py).
Both models use deliberately tiny grids and `GridSearch` so the values and routing can
be checked by hand. Those grids are curated for exposition and tests; they are not
production recommendations.

## Stage 1: one action, two values

The household chooses between work and leisure. Stakeholder `f` values own consumption
and leisure; stakeholder `m` values consumption more strongly. The regime declares two
stakeholders and one utility function for each:

```python
couple = Regime(
    transition=...,
    stakeholders=("f", "m"),
    states={"wage": wage_grid},
    actions={"work": DiscreteGrid(Work)},
    functions={
        "utility_f": utility_f,
        "utility_m": utility_m,
    },
)
```

With omitted `weights`, both stakeholders receive weight one half. pylcm maximizes the
weighted household objective once and stores both stakeholder values at that shared
argmax.

```python
from lcm_examples.collective_regimes import (
    get_params,
    get_shared_decision_model,
)

model = get_shared_decision_model()
solution = model.solve(params=get_params(), log_level="debug")

solution[0]["couple"]
```

The two wage rows are

```text
[[ 46.  92.]
 [ 78. 156.]]
```

The columns are the `f` and `m` values. They are not two independent optimizations: both
are read at the household's common work choice.

## Stage 2: same-period outside options

The extended model has a middle-period collective regime named
`married_with_participation`. Each partner compares their action value with the value of
being single in that same period:

```python
def participation_f(Q_f, V_single_f_ref):
    return Q_f >= V_single_f_ref - 0.5


married_with_participation = Regime(
    transition=...,
    stakeholders=("f", "m"),
    value_constraints={"participation_f": participation_f},
    same_period_refs={
        "V_single_f_ref": SamePeriodRef(
            regime="single_f",
            projection={"wage": identity_wage},
        ),
    },
)
```

The abbreviated declaration shows one partner; the importable model declares both.
`SamePeriodRef` is needed because a continuation value from period $t+1$ is the wrong
outside option. pylcm orders the same-period reference regimes before the collective
regime and interpolates their values at the declared projections.

At wages one and three, at least one shared action satisfies both partners. At wage two,
neither action does. The solve exposes that structural outcome separately from numeric
values:

```python
from lcm_examples.collective_regimes import get_dissolution_model

model = get_dissolution_model()
solution, dissolution_flags = model.solve(
    params=get_params(),
    log_level="debug",
    return_dissolution_flags=True,
)

dissolution_flags[1]["married_with_participation"]
# Array([False, True, False], dtype=bool)
```

## Stage 3: route the dissolution

The preceding period transitions toward the participation regime through a `GatedEdge`.
When the target's dissolution flag is false, each stakeholder takes their component of
the collective target value. When it is true, each follows their own projected singleton
value:

```python
GatedEdge(
    gate=lambda D_target: ~D_target,
    legs={
        "f": EdgeLeg(
            target_stakeholder="f",
            fallback=SamePeriodRef(
                regime="single_f",
                projection={"wage": identity_wage},
            ),
        ),
        "m": EdgeLeg(
            target_stakeholder="m",
            fallback=SamePeriodRef(
                regime="single_m",
                projection={"wage": identity_wage},
            ),
        ),
    },
)
```

To simulate a collective source, select the role carried by that cohort:

```python
result = model.simulate(
    params=get_params(),
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=solution,
    period_to_regime_to_dissolution_flags=dissolution_flags,
    own_stakeholder="f",
    log_level="debug",
    seed=0,
)
```

The wage-two row enters `single_f`; the wage-one and wage-three rows remain in the
collective target.

## Scope and scale

This model has one three-point continuous state, one binary action, two stakeholders,
and five regimes. It is designed for understanding and testing, not calibration.
Collective regimes currently require `GridSearch`, which is inexpensive here.

Simulation represents one fixed-size cohort. A dissolved row does not split into two
linked rows; `own_stakeholder` chooses which fallback leg governs the cohort. Simulate a
second cohort with `own_stakeholder="m"` when both roles are needed.

Exact contracts: [Collective regimes](../reference/collective_regimes.md). General model
workflow: [Regimes](../user_guide/regimes.ipynb).
