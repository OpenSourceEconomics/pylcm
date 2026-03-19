---
title: Solving and Simulating
---

# Solving and Simulating

Once you have defined a `Model` and prepared your parameters, pylcm solves via backward
induction and simulates forward.

## Solving

```python
V_arr_dict = model.solve(params=params)
```

Performs backward induction using dynamic programming. Returns an immutable mapping of
`period -> regime_name -> value_function_array`.

### Log levels

Control console output and snapshot persistence with `log_level`:

```python
# Default: progress + timing
V_arr_dict = model.solve(params=params)

# Silent
V_arr_dict = model.solve(params=params, log_level="off")

# Full diagnostics + disk snapshots
V_arr_dict = model.solve(params=params, log_level="debug", log_path="./debug/")
```

See [Debugging](debugging.md) for details on log levels and debug snapshots.

## Simulating

```python
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    V_arr_dict=V_arr_dict,
)
```

Forward simulation using solved value functions. Each agent starts from the given initial
conditions and makes optimal decisions at each period. Returns a `SimulationResult`
object.

## Simulate without pre-solving

When `V_arr_dict=None`, `simulate()` solves the model automatically before simulating.
Use this when you don't need the raw value function arrays:

```python
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    V_arr_dict=None,
)
```

## Initial Conditions

### From a DataFrame

The standard way to supply initial conditions is as a pandas DataFrame with one row per
agent. Use `initial_conditions_from_dataframe` to convert it to the format expected by
`simulate()`:

```python
import pandas as pd
from lcm import initial_conditions_from_dataframe

df = pd.DataFrame({
    "regime": ["working_life", "working_life", "retirement", "working_life"],
    "age": [25.0, 25.0, 25.0, 25.0],
    "wealth": [1.0, 5.0, 10.0, 20.0],
    "health": ["good", "bad", "bad", "good"],  # string labels, auto-converted
})

initial_conditions = initial_conditions_from_dataframe(df, model=model)
```

Discrete states (those backed by a `DiscreteGrid`) are mapped from string labels to
integer codes automatically. See [Working with DataFrames and Series](pandas_interop.md) for
details.

### As JAX arrays

You can also pass initial conditions directly as JAX arrays — useful for programmatic
setups like grid searches or tests:

```python
initial_conditions = {
    "age": jnp.array([25.0, 25.0, 25.0, 25.0]),
    "wealth": jnp.array([1.0, 5.0, 10.0, 20.0]),
    "health": jnp.array([0, 1, 1, 0]),  # integer codes for discrete states
    "regime": jnp.array([
        RegimeId.working_life, RegimeId.working_life,
        RegimeId.retirement, RegimeId.working_life,
    ]),
}
```

- Every non-shock state must have an entry.
- `"regime"` must be included, with integer codes from the `regime_id_class`.
- All arrays must have the same length (= number of agents).
- Shock states are drawn automatically.

### Optional arguments

- `check_initial_conditions=True`: Validates that initial states are on-grid and regimes
  are valid. Set to `False` to skip validation.
- `seed=None`: Random seed for stochastic simulations (int).
- `log_level="progress"`: Controls logging verbosity (same options as `solve()`).
- `log_path=None`: Directory for debug snapshots (when `log_level="debug"`).
- `log_keep_n_latest=3`: Maximum snapshot directories to retain.

### Heterogeneous initial ages

`"age"` must always be provided in `initial_conditions`. Each value must be a valid point on
the model's `AgeGrid`, and each subject's initial regime must be active at their starting
age. The most common case is that all subjects start at the initial age — just pass a
constant array.

Subjects can start at different ages:

```python
initial_conditions = {
    "age": jnp.array([40.0, 60.0]),
    "wealth": jnp.array([50.0, 50.0]),
    "regime": jnp.array([
        model.regime_names_to_ids["working_life"],
        model.regime_names_to_ids["working_life"],
    ]),
}
```

In the resulting DataFrame, each subject appears only from their starting age onward —
earlier periods are omitted, not filled with placeholders.

## Working with SimulationResult

### Converting to DataFrame

```python
df = result.to_dataframe()
```

Returns a pandas DataFrame with columns: `subject_id`, `period`, `age`, `regime`,
`value`, plus all states and actions. Discrete variables are pandas Categorical with
string labels.

### Additional targets

Compute functions and constraints alongside the standard output:

```python
# Specific targets
df = result.to_dataframe(additional_targets=["utility", "consumption"])

# All available targets
df = result.to_dataframe(additional_targets="all")

# See what's available
result.available_targets  # ['consumption', 'earnings', 'utility', ...]
```

Each target is computed for regimes where it exists; rows from other regimes get NaN.

### Integer codes instead of labels

```python
df = result.to_dataframe(use_labels=False)
```

Returns discrete variables as raw integer codes instead of categorical labels.

### Metadata

```python
result.regime_names   # ['retirement', 'working_life']
result.state_names    # ['health', 'wealth']
result.action_names   # ['consumption', 'work']
result.n_periods      # 50
result.n_subjects     # 1000
```

### Serialization

Save and load results (requires `cloudpickle`):

```python
# Save
result.to_pickle("my_results.pkl")

# Load
from lcm.simulation.result import SimulationResult
loaded = SimulationResult.from_pickle("my_results.pkl")
```

### Raw data (advanced)

```python
result.raw_results      # regime -> period -> PeriodRegimeSimulationData
result.internal_params  # processed parameter object
result.V_arr_dict       # value function arrays from solve()
```

## Typical Workflow

```python
import numpy as np
import pandas as pd
from lcm import Model, initial_conditions_from_dataframe

# 1. Define model (see previous pages)
model = Model(regimes={...}, ages=..., regime_id_class=...)

# 2. Set parameters
params = {
    "discount_factor": 0.95,
    "interest_rate": 0.03,
    ...
}

# 3. Prepare initial conditions
initial_df = pd.DataFrame({
    "regime": "working_life",
    "age": model.ages.values[0],
    "wealth": np.linspace(1, 50, 100),
})
initial_conditions = initial_conditions_from_dataframe(initial_df, model=model)

# 4. Simulate (solves automatically when V_arr_dict=None)
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    V_arr_dict=None,
)

# 5. Analyze
df = result.to_dataframe(additional_targets="all")
df.groupby("period")["wealth"].mean()
```

## See Also

- [Defining Models](defining_models.md) — constructing the `Model`
- [Parameters](parameters.md) — preparing the params dict
- [Working with DataFrames and Series](pandas_interop.md) — DataFrame conversion
  utilities
- [A Tiny Example](tiny_example.ipynb) — complete walkthrough
- [Examples](../examples/index.md) — full worked examples
