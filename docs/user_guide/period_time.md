---
title: Periods and temporal parameters
---

# Periods and temporal parameters

Use `Model(n_periods=...)` when periods are the natural model clock. Keep
`Model(ages=AgeGrid(...))` when actual ages define that clock. Supply exactly one. A
model that takes several decisions in one period declares them as `stages`.

## Model time

```python
from lcm import InitialNodes, Model, PeriodRange, Periods

model = Model(
    n_periods=3,
    regimes=regimes,
    regime_id_class=RegimeId,
    initial_nodes=InitialNodes(by_period={0: "working"}),
    edges={
        "working": {
            "working": PeriodRange(exclusive_stop=1),
            "terminal": Periods(values=(1,)),
        },
    },
)
```

This horizon contains slots 0, 1 and 2. An edge selects its **source period** and lands
in the next slot: the edge from period 1 reaches the terminal regime in period 2.
`n_periods` counts terminal slots too. It must be a positive integer; booleans are
invalid.

`model.ages` is `None` in period mode. Functions may request `period`. An unresolved
`age` argument is an error. Declare a separately named `biological_age` or
`economic_quarter` function when the economics needs it. A period does not determine
discounting, survival, interest accrual or how flows are aggregated.

Admissible starts use `InitialNodes(by_period={0: "working"})`; age models use
`InitialNodes(by_age={25: "working"})`. Exactly one nonempty mapping is required, and a
declaration for the other clock is rejected. Both forms copy their containers and
normalize regime collections. `model.initial_nodes` can be passed directly into another
model; `model.graph.initial_nodes` exposes expanded pairs labelled by
`graph.coordinate_kind`. See
[InitialNodes](../reference/model_and_regime.md#api-initial-nodes) for selector and
grid-membership rules.

Period graph declarations require `PeriodRange` or `Periods`; a bare integer or tuple
cannot silently change from an age into a period. `ByPeriod(cases={0: ..., 1: ...})`
declares its coordinate kind explicitly, so integer case keys are sufficient. As with
`ByAge`, it selects regime-transition laws on `Transition(law=...)`.
`PeriodSpecializedFunction` and `PeriodSpecializedGrid` take integer periods in their
`build` and `signature` callbacks and retain the age markers' determinism, deduplication
and shape contracts. Age declarations belong to age models; period declarations belong
to period models.

Simulation mappings carry `period` arrays and `regime_id` codes; DataFrames carry
`period` and `regime_name`. Starts must be integer periods within the horizon and among
the declared initial nodes. Wrong coordinate names, both names together, booleans,
fractional values and out-of-range starts fail even with `log_level="off"`. Validation
and feasibility methods use the same rules. Results always identify periods; only age
models publish an `age` column, including after a result is saved and loaded.

## Temporal parameters

```python
import jax.numpy as jnp
from lcm import TimeVarying, time_varying_params
from lcm.typing import FloatND


@time_varying_params("wage")
def earnings(*, hours: FloatND, wage: FloatND) -> FloatND:
    return hours * wage


params = {
    "working": {
        "earnings": {
            "wage": TimeVarying(
                values=jnp.array([22.0, 20.0, 999.0]),
                periods=(1, 0, 8),
            ),
        },
    },
}
```

The function receives wage 20 in period 0 and wage 22 in period 1. It receives the
selected scalar or remaining tensor, with the time axis removed. It may also request
`period` for other logic. A scalar parameter is constant over time. A `TimeVarying`
input requires a declared temporal slot; an unlabelled array cannot fill one. Names in
the decorator must be parameter arguments, never states or other DAG nodes.

Use `ages=(...)` for an age model. A Series with a named `age` or `period` index is an
equivalent labelled input. MultiIndex data retain named categorical dimensions; names
and category labels are checked and reordered to the consumer's indexing order.

Alignment follows this order:

1. Check coordinate kind, names, types and axis lengths, including surplus rows.
1. **Silently discard coordinates outside the model grid**, for ages and periods alike.
1. Reject duplicate selected keys. Repeated time labels for different categorical
   combinations are ordinary data. Duplicates only outside the grid are discarded.
1. Require every selected combination the consumer can structurally read.
1. Reorder by labels into period-addressed numeric values.

Missing required rows are errors, never NaN filling, interpolation or clipping. An
inactive or unused terminal slot needs no invented observation. Negative integer labels
in a wider period table are surplus rows; negative simulation starts are invalid.
Changing an ignored value has no effect on canonical consumed parameters.

A transition law consumes its **source period**. A function evaluated at a target node
consumes that node's period. If a table describes destinations, its author must realign
it to the source labels before supplying it to a transition law.

Fixed and runtime temporal inputs have the same alignment checks. A fixed slot is bound
at model construction and removed from the runtime template; rebuild the model to change
it. Supplying that slot at solve time is rejected by the ordinary parameter grammar.

`get_params_template()` marks temporal slots with `TimeVarying`. A parameter shared by
two phase variants or schedule cases must have the same managed-time declaration in both
consumers. Supply a separate parameter name when they need different axis meanings.

## Manual indexing

Labelled Series also work with the existing manual `table[period]` route. **Do not use
`table[age]` on these inputs: labelled arrays are normalized to period order, so an age
is not an array position.** Known direct uses are rejected.

In an age model, a known time-indexed raw array emits `UnlabelledTimeParameterWarning`,
independently of logging settings, including an explicit `table[age - start_age]`
mapping. The raw array's indexing convention remains the author's responsibility. In a
period model, that input is rejected. Use managed temporal selection or supply a
labelled Series.

Source inspection cannot prove the meaning of arbitrary helpers, closures or indexing
arithmetic. Managed selection removes the time axis before the consumer sees the value;
it is the route that prevents a consumer from confusing an age with a period position.

## Several decisions in one period

A household may first decide whether to stay together and then how much to consume, or
two players may move in turn. Declare the decisions of one period as an ordered mapping
from stage names to the regimes of each stage:

```python
model = Model(
    n_periods=3,
    stages={"work": ("working",), "spend": ("spending", "final")},
    regimes=regimes,
    regime_id_class=RegimeId,
    initial_nodes=InitialNodes(by_period={0: "working"}),
    edges={
        "working": {
            "spending": PeriodRange(exclusive_stop=2),
            "final": Periods(values=(2,)),
        },
        "spending": {"working": PeriodRange(exclusive_stop=2)},
    },
)
```

Every period has the same stages, and every regime is in exactly one of them. A regime
that decides in two stages of one period is two declared regimes.

- An edge lands at its target's next occurrence: the first step after the current one
  that belongs to the target's stage. A later stage of the same period is reached
  without discounting; crossing into the next period discounts once, however many stages
  are skipped. An edge into a regime of the source's own stage lands in the next period.
  Its selector names the source's period, as in a model without stages.
- An exit enters a regime of its own stage or of a later stage of its period. In a later
  stage its outside value is the destination's value at its occurrence later in the
  period, at the landing state, without discounting. The exit cannot observe a shock the
  destination draws in its own stage, which is not drawn yet: draw it in the earlier
  stage and carry it as a state.
- Subjects start in their regime's own stage, so `initial_nodes` and initial conditions
  may name a regime of any stage.
- The stages of a period are linked without discounting: where a regime's edges land in
  its period, it adds their certainty equivalent to its utility. Its edges at one age
  all land in its period or all cross into the next one; a regime whose edges land in
  its period at some age declares no Koopmans aggregator and uses no endogenous-grid
  solver, and if its edges cross at other ages the model's aggregator is
  `LinearAggregator`, which discounts at those ages only.

A regime with one decision per period, such as a single among couples who first decide
whether to stay together and then how much to consume, is declared once, in the stage
where it decides. An edge into it reads its value at its next occurrence, and a subject
in it has no row and no transition at a step it sits out. The parameter template,
`model.regime_names_to_ids`, `model.initial_nodes`, the solution's values, the raw
simulation results and the simulated frame list only declared regimes, so they have no
entry for a step the regime sits out. A gated edge does not skip a stage, a regime with
a continuous stochastic process does not sit out a stage, and an edge whose target's
next occurrence lies beyond the last period is refused.

- `period` and `age` are the period's wherever the model reads them: in functions,
  constraints and conditions, laws, gates, exits and settlements, Pareto weights and
  shock laws. A condition that reads `period` then offers a solver no structure, as one
  built with `Condition.from_callable` does. `TimeVarying` and other time-indexed
  parameters are indexed by period, and the simulated frame has a `stage` column next to
  `period`.
- A regime decides in its own stage of every period, so a selector that names only the
  final period is refused for a regime of the last stage, where no transition happens.
  The regime finder does not analyse a model with stages.

Two players who move in turn are one household regime per stage, each with both players
as members and a `ParetoObjective` that puts all weight on the player who moves: the
mover maximizes its own value, and the other player's value follows the mover's choice.

The engine solves one step per pair of a period and a stage: `model.n_periods` counts
steps, and the solution's values and the raw simulation results are keyed by step,
`2 * period` for `work` and `2 * period + 1` for `spend` above. Without `stages` every
period has one stage, and a step is a period.

## Economic profiles

For a labelled economic profile, use an explicit mapping:

```python
profile = TimeVarying.from_profile(
    values=jnp.array([20.0, 10.0, 999.0]),
    labels=("2031Q2", "2031Q1", "outside"),
    period_to_label={0: "2031Q1", 1: "2031Q1", 2: "2031Q2", 3: "2031Q2"},
)
```

This returns period-labelled values `[10, 10, 20, 20]`: the two periods of each quarter
deliberately share a source value. A missing mapped source label or a duplicate selected
source label is an error; unused source rows are ignored. The numeric gather preserves
derivatives with respect to the profile's values. The helper assigns neither elapsed
duration nor biological age to the resulting periods.

`period_to_label` returns a period-labelled table, which an age model rejects. In an age
model, map each model age to its source label with `age_to_label` instead. The result is
labelled by age, and the age keys must be finite numbers. Give exactly one of the two
mappings.

A parameter the model reads at the next age, with the last age reading its own row:

```python
ages = tuple(range(25, 86))
next_income = TimeVarying.from_profile(
    values=income_profile.to_numpy(),
    labels=tuple(income_profile.index),
    age_to_label={age: min(age + 1, ages[-1]) for age in ages},
)
```

A profile lagged by one year maps each age to the previous year's row,
`age_to_label={age: age - 1 for age in ages}`, provided the profile has a row for the
year before the first age.

Both mappings follow the same rules. Several ages may select the same source label, as
the last two ages do above. A source label that appears more than once in `labels` is an
error when it is selected and ignored otherwise.
