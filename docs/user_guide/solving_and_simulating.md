---
title: Solving and Simulating
---

# Solving and Simulating

Once you have defined a `Model` and prepared your parameters, pylcm solves via backward
induction and simulates forward.

## Solving

```python
V_arr_dict = model.solve(params)
```

Performs backward induction using dynamic programming. Returns an immutable mapping of
`period -> regime_name -> value_function_array`.

### Debug mode

- `debug_mode=True` (default): Runs without JIT compilation for better error messages.
  Use this while developing.
- `debug_mode=False`: Full JIT compilation for speed. Use for production runs.

```python
V_arr_dict = model.solve(params, debug_mode=False)
```

## Simulating

```python
result = model.simulate(
    params=params,
    initial_states=initial_states,
    initial_regimes=initial_regimes,
    V_arr_dict=V_arr_dict,
)
```

Forward simulation using solved value functions. Each agent starts from the given initial
conditions and makes optimal decisions at each period. Returns a `SimulationResult`
object.

## Solve and Simulate (combined)

```python
result = model.solve_and_simulate(
    params=params,
    initial_states=initial_states,
    initial_regimes=initial_regimes,
)
```

Convenience method combining both steps. Use when you don't need the raw value function
arrays.

## Initial Conditions

### Initial states

A flat dictionary mapping state names to JAX arrays, one value per agent:

```python
initial_states = {
    "wealth": jnp.array([1.0, 5.0, 10.0, 20.0]),
    "health": jnp.array([0, 1, 1, 0]),  # integer codes for discrete states
}
```

- Every non-shock state must have an entry.
- All arrays must have the same length (= number of agents).
- Shock states are drawn automatically.

### Initial regimes

A list of regime names, one per agent:

```python
initial_regimes = ["working", "working", "retired", "working"]
```

### Optional arguments

- `check_initial_conditions=True`: Validates that initial states are on-grid and regimes
  are valid. Set to `False` to skip validation.
- `seed=None`: Random seed for stochastic simulations (int).
- `debug_mode=True`: Same as for `solve()`.

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
result.regime_names   # ['retired', 'working']
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
import jax.numpy as jnp
from lcm import Model

# 1. Define model (see previous pages)
model = Model(regimes={...}, ages=..., regime_id_class=...)

# 2. Set parameters
params = {
    "discount_factor": 0.95,
    "interest_rate": 0.03,
    ...
}

# 3. Solve and simulate
result = model.solve_and_simulate(
    params=params,
    initial_states={"wealth": jnp.linspace(1, 50, 100)},
    initial_regimes=["working"] * 100,
)

# 4. Analyze
df = result.to_dataframe(additional_targets="all")
df.groupby("period")["wealth"].mean()
```

## See Also

- [Defining Models](defining_models.md) — constructing the `Model`
- [Parameters](parameters.md) — preparing the params dict
- [A Tiny Example](../getting_started/tiny_example.ipynb) — complete walkthrough
- [Examples](../examples/index.md) — full worked examples
