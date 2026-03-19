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

initial_conditions = initial_conditions_from_dataframe(df, model=model)

result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    V_arr_dict=None,
)
```

The function requires a `"regime"` column with valid regime names. All other columns are
treated as state variables. Discrete states (those backed by a `DiscreteGrid`) are mapped
from string labels to integer codes automatically — you write `"good"` instead of `1`.
Continuous states are passed through as-is.

Columns with pandas `Categorical` dtype are also supported and converted to codes via the
same label mapping.

## Transition Probabilities from a Series

Build a transition probability array from a pandas Series with a named `MultiIndex`,
replacing manual array construction where axis ordering is error-prone.

```python
from lcm import transition_probs_from_series

# Series with named MultiIndex levels — use "age" (not "period")
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

health_probs = transition_probs_from_series(
    series=probs,
    model=model,
    regime_name="working",
)
```

The transition type is inferred from the `"next_*"` level in the MultiIndex:
`"next_health"` means a state transition on `"health"`, while `"next_regime"` means a
regime transition. The `regime_name` can also be omitted when inference is unambiguous:

```python
regime_probs = transition_probs_from_series(
    series=regime_series,
    model=model,
)
```

The MultiIndex level names must match the indexing parameters of the transition function
(in any order) plus the outcome level (`"next_{state_name}"` for state transitions,
`"next_regime"` for regime transitions). Use `"age"` with actual age values from the
model's `AgeGrid` for the age dimension (not `"period"`). The function reorders levels to
match the declaration order automatically, so you don't need to worry about getting the
level order right.

Discrete state and action labels are mapped to integer codes using the same grids defined
in the model. Age values are converted to period indices automatically.

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

Call this after `transition_probs_from_series` or after manual array construction to catch
mistakes early.

## See Also

- [Solving and Simulating](solving_and_simulating.md) — the `initial_conditions` format
- [Parameters](parameters.md) — where transition probability arrays go in the params dict
- [Regimes](regimes.ipynb) — defining `MarkovTransition` state transitions
