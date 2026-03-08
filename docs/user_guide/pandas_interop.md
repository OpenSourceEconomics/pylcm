---
title: Pandas Interop
---

# Pandas Interop

pylcm works with JAX arrays internally, but real-world data often lives in pandas. The
`lcm.pandas_utils` module provides utilities that bridge the gap — converting DataFrames
to initial conditions and labeled Series to transition probability arrays.

## Initial States from a DataFrame

Convert a pandas DataFrame into the `initial_states` dict and `initial_regimes` list
expected by `model.simulate()` and `model.solve_and_simulate()`.

```python
from lcm import initial_states_from_dataframe

df = pd.DataFrame({
    "regime": ["working", "working", "retired"],
    "wealth": [10.0, 50.0, 30.0],
    "health": ["good", "bad", "good"],
    "age": [25.0, 25.0, 25.0],
})

initial_states, initial_regimes = initial_states_from_dataframe(df, model=model)

result = model.solve_and_simulate(
    params=params,
    initial_states=initial_states,
    initial_regimes=initial_regimes,
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

# Series with named MultiIndex levels matching the transition function signature
probs = pd.Series(
    [0.9, 0.1, 0.3, 0.7, 0.8, 0.2, 0.4, 0.6],
    index=pd.MultiIndex.from_tuples(
        [
            (0, "good", "good"),
            (0, "good", "bad"),
            (0, "bad", "good"),
            (0, "bad", "bad"),
            (1, "good", "good"),
            (1, "good", "bad"),
            (1, "bad", "good"),
            (1, "bad", "bad"),
        ],
        names=["period", "health", "next_health"],
    ),
)

health_probs = transition_probs_from_series(
    probs,
    model=model,
    regime_name="working",
    state_name="health",
)
```

The MultiIndex level names must match the indexing parameters of the transition function
(in any order) plus `"next_{state_name}"` for the outcome axis. The function reorders
levels to match the declaration order automatically, so you don't need to worry about
getting the level order right.

Discrete state and action labels are mapped to integer codes using the same grids defined
in the model. The `"period"` level uses integer values directly.

## Validating Transition Probabilities

Check that a transition probability array has the correct shape, values in $[0, 1]$, and
rows that sum to 1 before passing it to `model.solve()`.

```python
from lcm import validate_transition_probs

validate_transition_probs(
    health_probs,
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

- [Solving and Simulating](solving_and_simulating.md) — the `initial_states` and
  `initial_regimes` format
- [Parameters](parameters.md) — where transition probability arrays go in the params dict
- [Regimes](regimes.ipynb) — defining `MarkovTransition` state transitions
