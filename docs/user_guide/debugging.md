---
title: Debugging
---

# Debugging

Dynamic programming models are complex, and most computation happens inside JIT-compiled
functions. This page covers practical strategies for diagnosing problems.

## Disable JIT for readable tracebacks

By default, pylcm JIT-compiles internal functions for performance. When something goes
wrong inside a JIT-compiled function, the traceback is often unhelpful. Disable JIT at
model creation time to get standard Python tracebacks:

```python
model = Model(
    regimes={...},
    ages=ages,
    regime_id_class=RegimeId,
    enable_jit=False,  # readable tracebacks, but slower
)
```

This does not affect correctness --- the same functions run, just without compilation.
Re-enable JIT once the issue is resolved.

## Auto-persist intermediate results

When `debug=True` (the default) and a `debug_path` is provided, pylcm automatically
saves intermediate results to disk:

```python
# Solve: saves value function arrays
V_arr_dict = model.solve(params, debug=True, debug_path="./debug/")
# Creates: ./debug/solution_001.pkl

# Simulate: saves the SimulationResult
result = model.simulate(
    params, initial_states, initial_regimes, V_arr_dict,
    debug=True, debug_path="./debug/",
)
# Creates: ./debug/simulation_result_001.pkl

# solve_and_simulate: saves both
result = model.solve_and_simulate(
    params, initial_states, initial_regimes,
    debug=True, debug_path="./debug/",
)
# Creates: ./debug/solution_001.pkl and ./debug/simulation_result_001.pkl
```

### Loading persisted results

```python
from lcm import load_solution, SimulationResult

# Load value function arrays
V_arr_dict = load_solution(path="./debug/solution_001.pkl")

# Load simulation result
result = SimulationResult.from_pickle("./debug/simulation_result_001.pkl")
df = result.to_dataframe()
```

### File retention

When running inside a numerical optimization loop, debug files can accumulate quickly.
The `keep_n_latest` parameter (default 3) limits how many snapshots are kept:

```python
V_arr_dict = model.solve(
    params, debug=True, debug_path="./debug/", keep_n_latest=5
)
```

After each write, the oldest files beyond the limit are deleted automatically.

## Recipe: Debugging NaN in parameter estimation with optimagic

A common scenario: you are estimating model parameters with optimagic, and at some
iteration the criterion function returns NaN. Here is how to diagnose the problem.

### 1. Enable optimagic logging

```python
import optimagic as om

result = om.minimize(
    fun=criterion,
    params=start_params,
    algorithm="scipy_lbfgsb",
    logging="my_log.db",
)
```

### 2. Find the problematic parameters

```python
reader = om.SQLiteLogReader("my_log.db")
history = reader.read_history()

# history["fun"] contains criterion values, history["params"] the parameter vectors
import numpy as np

fun_values = history["fun"]
nan_mask = np.isnan(fun_values)
if nan_mask.any():
    first_nan_idx = np.argmax(nan_mask)
    bad_params = history["params"].iloc[first_nan_idx]
    print(f"First NaN at iteration {first_nan_idx}")
    print(f"Parameters: {bad_params}")
```

### 3. Re-run with JIT disabled

```python
# Re-create the model without JIT
model = Model(
    regimes={...},
    ages=ages,
    regime_id_class=RegimeId,
    enable_jit=False,
)

# Call solve with the bad parameters --- the traceback will be readable
V_arr_dict = model.solve(bad_params)
```

The traceback now points to the exact line in your user-defined functions where the
NaN originates.

## Inspecting value function arrays

The solution `V_arr_dict` is a nested mapping: `period -> regime_name -> array`. You can
iterate over it to check shapes, look for NaN/inf, or plot slices:

```python
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

V_arr_dict = model.solve(params)

# Check for issues
for period, regimes in V_arr_dict.items():
    for regime_name, V_arr in regimes.items():
        n_nan = int(jnp.sum(jnp.isnan(V_arr)))
        n_inf = int(jnp.sum(jnp.isinf(V_arr)))
        if n_nan > 0 or n_inf > 0:
            print(f"Period {period}, regime '{regime_name}': "
                  f"shape={V_arr.shape}, NaN={n_nan}, Inf={n_inf}")

# Plot a 1D slice (e.g. value over wealth grid for first period)
period = 0
regime_name = "working"
V_arr = V_arr_dict[period][regime_name]

fig = go.Figure()
fig.add_trace(go.Scatter(y=V_arr.tolist(), mode="lines", name="V(wealth)"))
fig.update_layout(title=f"Value function, period {period}, regime '{regime_name}'")
fig.show()
```

## Understanding error messages

pylcm raises specific exceptions to help you diagnose problems:

- **`InvalidValueFunctionError`**: The value function array contains NaN at a given age
  and regime. The message reports the regime name and how many values are NaN (e.g.
  "3 of 100 values are NaN"). Common causes: utility function returning NaN for some
  state-action combinations, or impossible regime transitions.

- **`InvalidRegimeTransitionProbabilitiesError`**: Regime transition probabilities are
  non-finite, outside [0, 1], don't sum to 1, or assign positive probability to an
  inactive regime. The message includes the source regime, age range, and a table of
  failing entries.

- **`ModelInitializationError`**: Something is wrong with the model definition
  (mismatched regime names, unused variables, etc.). Read the message carefully --- it
  usually lists all issues found.
