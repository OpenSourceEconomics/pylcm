---
title: Periods and temporal parameters
---

# Periods and temporal parameters

Use `Model(n_periods=...)` when computational stages are the natural model clock. Keep
`Model(ages=AgeGrid(...))` when actual ages define that clock. Supply exactly one.

## Model time

```python
from lcm import InitialNode, Model, PeriodRange, Periods

model = Model(
    n_periods=3,
    regimes=regimes,
    regime_id_class=RegimeId,
    initial_nodes=(InitialNode(period=0, regime="working"),),
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
`economic_quarter` function when the economics needs it. A computational stage does not
determine discounting, survival, interest accrual or how flows are aggregated.

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

## Stages within economic time

A quarterly model may have several computational stages per quarter. Declare one
paper-owned mapping from period to economic quarter and use it to gather profiles by
their source labels. Two stages may deliberately read the same quarter. Neither adding a
stage nor repeating a biological age implies that time has elapsed. Keep discounting and
transition timing in the model's equations.

For a labelled economic profile, use an explicit mapping:

```python
profile = TimeVarying.from_profile(
    values=jnp.array([20.0, 10.0, 999.0]),
    labels=("2031Q2", "2031Q1", "outside"),
    period_to_label={0: "2031Q1", 1: "2031Q1", 2: "2031Q2", 3: "2031Q2"},
)
```

This returns period-labelled values `[10, 10, 20, 20]`. The two stages in each quarter
deliberately share a source value. A missing mapped source label or a duplicate selected
source label is an error; unused source rows are ignored. The numeric gather preserves
derivatives with respect to the profile's values. The helper assigns neither elapsed
duration nor biological age to the resulting periods.
