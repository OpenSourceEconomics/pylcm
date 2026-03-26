---
title: Working with DataFrames and Series
---

# Working with DataFrames and Series

pylcm accepts initial conditions as a pandas DataFrame — the natural format when your
data comes from a survey, an external dataset, or a scenario table. Simulation results
come back as a DataFrame too, so the typical workflow is DataFrame in, DataFrame out.

## Initial Conditions from a DataFrame

Convert a pandas DataFrame into the `initial_conditions` dict expected by
`model.simulate()`. This is the standard way to supply initial conditions. The returned
dict includes all state arrays plus a `"regime"` array with integer codes.

```python
from lcm import initial_conditions_from_dataframe

df = pd.DataFrame({
    "regime": ["working", "working", "retired"],
    "wealth": [10.0, 50.0, 30.0],
    "health": ["good", "bad", "good"],
    "age": [25.0, 25.0, 25.0],
})

initial_conditions = initial_conditions_from_dataframe(df=df, model=model)

result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=None,
)
```

The function requires a `"regime"` column with valid regime names. All other columns are
treated as state variables. Discrete states (those backed by a `DiscreteGrid`) are mapped
from string labels to integer codes automatically — you write `"good"` instead of `1`.
Continuous states are passed through as-is.

Columns with pandas `Categorical` dtype are also supported and converted to codes via the
same label mapping.

## Parameters from Pandas

When parameters include array values — transition probabilities, wage profiles, or any
array indexed by states — it is natural to prepare them as labeled `pd.Series` with a
named `MultiIndex`. `params_from_pandas` converts an entire params dict in one call,
replacing every `pd.Series` with the correctly shaped JAX array:

```python
from lcm import params_from_pandas

params = {
    "discount_factor": 0.95,
    "working": {
        "next_health": {
            "probs_array": health_probs_series,  # pd.Series with MultiIndex
        },
        "utility": {"risk_aversion": 1.5},
    },
}

converted = params_from_pandas(params=params, model=model)
model.simulate(params=converted, ...)
```

The function broadcasts params at any nesting level (model / regime / function) against
the model's params template — the same resolution rules as `model.solve(params=...)`.
Scalars, existing arrays, and `MappingLeaf` / `SequenceLeaf` values pass through
unchanged; only `pd.Series` values are converted.

Each Series must have a named `MultiIndex` (or named `Index` for 1-D arrays) whose level
names match the function's indexing parameters. Use `"age"` with actual age values for
the age dimension, not `"period"`. Levels are reordered automatically, so you don't need
to worry about getting the order right. For transition functions (`next_*`), include the
outcome level too (`"next_health"` for state transitions, `"next_regime"` for regime
transitions).

## Under the Hood: `array_from_series`

`params_from_pandas` calls `array_from_series` for each Series value. If you need
fine-grained control over a single parameter — or want to inspect the conversion
step by step — you can call it directly:

```python
from lcm.pandas_utils import array_from_series

probs = pd.Series(
    [0.9, 0.1, 0.3, 0.7, 0.8, 0.2, 0.4, 0.6],
    index=pd.MultiIndex.from_tuples(
        [
            (25, "good", "good"),
            (25, "good", "bad"),
            (25, "bad", "good"),
            (25, "bad", "bad"),
            (35, "good", "good"),
            (35, "good", "bad"),
            (35, "bad", "good"),
            (35, "bad", "bad"),
        ],
        names=["age", "health", "next_health"],
    ),
)

health_probs = array_from_series(
    sr=probs,
    model=model,
    param_path=("working", "next_health", "probs_array"),
)
```

`param_path` is a 1-to-3 element tuple identifying the parameter in the model:
`(param,)`, `(func, param)`, or `(regime, func, param)`. When the path points to a
`next_*` function, the outcome axis is appended automatically.

Discrete state and action labels are mapped to integer codes using the same grids defined
in the model. Age values outside the model's `AgeGrid` are silently dropped; missing grid
points are filled with NaN.

## Validating Transition Probabilities

Check that a transition probability array has the correct shape, values in $[0, 1]$, and
rows that sum to 1 before passing it to `model.solve()`.

```python
from lcm import validate_transition_probs

validate_transition_probs(
    probs=health_probs,
    model=model,
    regime_name="working",
    state_name="health",
)
```

Raises `ValueError` if:

- The array shape doesn't match the expected dimensions (indexing parameters + outcome
  axis)
- Any value is outside $[0, 1]$
- Any row (slice along the last axis) doesn't sum to 1

Call this after `params_from_pandas` or after manual array construction to catch
mistakes early.

## See Also

- [Solving and Simulating](solving_and_simulating.md) — the `initial_conditions` format
- [Parameters](parameters.md) — where transition probability arrays go in the params dict
- [Regimes](regimes.ipynb) — defining `MarkovTransition` state transitions
