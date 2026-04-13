---
title: Working with DataFrames and Series
---

# Working with DataFrames and Series

`solve()` and `simulate()` accept pandas objects directly. Initial conditions can be a
DataFrame, and parameters can contain `pd.Series` values with labeled indices.
Simulation results come back as a DataFrame via `.to_dataframe()`. The typical workflow
is DataFrame in, DataFrame out.

## Initial Conditions as a DataFrame

Pass a pandas DataFrame directly to `simulate()` as `initial_conditions`. One row per
agent, one column per state variable, plus a `"regime"` column:

```python
df = pd.DataFrame(
    {
        "regime": ["working", "working", "retired"],
        "wealth": [10.0, 50.0, 30.0],
        "health": ["good", "bad", "good"],
        "age": [25.0, 25.0, 25.0],
    }
)

result = model.simulate(
    params=params,
    initial_conditions=df,
    period_to_regime_to_V_arr=None,
)
```

- `"regime"` column is required. Use regime names as strings (e.g., `"working"`).
- Discrete states use string labels from the model's categorical classes (e.g., `"good"`
  instead of `0`). Labels are validated and mapped to integer codes automatically.
- Continuous states pass through as-is.
- `Categorical` dtype columns are also supported.

You can also pass initial conditions as a plain dict of JAX arrays (see
[Solving and Simulating](solving_and_simulating.md#as-jax-arrays)).

## Parameters with `pd.Series`

When parameters include array values — transition probabilities, wage profiles, or any
array indexed by model variables — prepare them as labeled `pd.Series` with a named
`MultiIndex`. Pass them directly in the params dict; `solve()` and `simulate()` convert
them automatically:

```python
params = {
    "discount_factor": 0.95,
    "working": {
        "next_health": {
            "probs_array": health_probs_series,  # pd.Series with MultiIndex
        },
        "utility": {"risk_aversion": 1.5},
    },
}

# Series values are converted to JAX arrays transparently
result = model.simulate(
    params=params,
    initial_conditions=df,
    period_to_regime_to_V_arr=None,
)
```

Scalars and existing JAX arrays pass through unchanged — only `pd.Series` values trigger
conversion.

### Series format

Each `pd.Series` must have:

- A **named** `MultiIndex` (or named `Index` for 1-D arrays). Level names must match the
  function's indexing parameters.
- **String labels** for discrete variables, matching the model's categorical classes.
- **`"age"`** (not `"period"`) for the age dimension, with actual age values from the
  model's `AgeGrid`.
- For **transition functions** (`next_*`): an additional outcome level (`"next_health"`
  for state transitions, `"next_regime"` for regime transitions).

Level order does not matter — levels are reordered to match the function signature
automatically.

### What happens during conversion

Your model functions work with plain JAX arrays and integer indexing — nothing about
pandas enters the model at runtime. The Series is purely an input convenience. Before
any model code runs, the conversion inspects the function signature to determine which
dimensions the array is indexed over, maps each label to an integer position using the
model's grids (e.g., `"good"` → `0`, `"bad"` → `1`), and scatters the Series values into
a JAX array of the correct shape. The function receives a normal `jnp.ndarray` and never
sees pandas.

## Why Labeled Indices Matter

Every discrete variable axis must use string labels from the model's categorical
classes, not raw integer codes. This is a deliberate design choice.

The conversion step validates every label against the model's grids before building the
array. If a label is misspelled, a category is missing, or axes are swapped, you get a
clear error *before* the array enters JAX. Without this validation, a wrong index would
silently produce a misshapen array. JAX would then vmap that array over millions of
simulated agents — producing garbage results with no error message and no way to trace
the problem back to the input.

Labeled indices turn silent data corruption into loud, early errors with actionable
messages.

## `derived_categoricals`

When a function indexes its array parameter by a variable that is *not* a state or
action in the model — typically a DAG function output — the model has no grid to
validate labels against. You will see an error like:

```
Unrecognised indexing parameter 'employment_type'. Expected 'age' or a
discrete grid name (['health', 'partner']). If 'employment_type' is a DAG
function output, add derived_categoricals={"employment_type": DiscreteGrid(EmploymentType)}
to the Regime or Model constructor.
```

Fix this by declaring the grid on the `Regime` that uses it:

```python
working = Regime(
    # ... other fields ...
    derived_categoricals={"employment_type": DiscreteGrid(EmploymentType)},
)
```

If the variable has different categories in different regimes, each regime declares its
own grid:

```python
working = Regime(
    # ... other fields ...
    derived_categoricals={"employment_type": DiscreteGrid(FullEmploymentType)},
)
retired = Regime(
    # ... other fields ...
    derived_categoricals={"employment_type": DiscreteGrid(RetiredEmploymentType)},
)
```

For convenience, model-level `derived_categoricals` are broadcast to all regimes:

```python
Model(
    regimes={"working": working, "retired": retired},
    derived_categoricals={"employment_type": DiscreteGrid(EmploymentType)},
    # ... other fields ...
)
```

### Integer return types required

Functions used as derived categoricals must return **integer** values, not booleans. JAX
cannot use boolean values as array indices inside JIT-compiled code
(`NonConcreteBooleanIndexError`). If your derived categorical compares states:

```python
# Wrong — returns bool, fails inside JIT
def is_good_health(health: DiscreteState) -> BoolND:
    return health == Health.good


# Correct — returns int32
def is_good_health(health: DiscreteState) -> IntND:
    return jnp.int32(health == Health.good)
```

## Validating Transition Probabilities

Check that a transition probability array has the correct shape, values in $[0, 1]$, and
rows that sum to 1:

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

Call this after building the array to catch mistakes early.

## Under the Hood

Internally, `solve()` and `simulate()` call `convert_series_in_params` (in
`lcm.pandas_utils`) to walk the already-broadcast params and convert each `pd.Series`
via `array_from_series`. For initial conditions, `initial_conditions_from_dataframe`
handles the DataFrame-to-dict conversion. Both are internal helpers — you don't need to
call them directly.

## See Also

- [Solving and Simulating](solving_and_simulating.md) — full `solve()` / `simulate()`
  API
- [Parameters](parameters.md) — where transition probability arrays go in the params
  dict
- [Regimes](regimes.ipynb) — defining `MarkovTransition` state transitions
