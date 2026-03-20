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

## Log levels

The `log_level` parameter controls both console output and disk persistence:

| Level | Output | Persistence |
|-------|--------|-------------|
| `"off"` | Nothing (good for HPC batch jobs) | No |
| `"warning"` | NaN/Inf warnings in value functions | No |
| `"progress"` (default) | Progress and timing per period, total elapsed time | No |
| `"debug"` | All above + V_arr statistics per regime, regime transition counts | Yes, requires `log_path` |

```python
# Silent — no console output at all
period_to_regime_to_V_arr = model.solve(params=params, log_level="off")

# Warnings only — alerts on NaN/Inf but no progress output
period_to_regime_to_V_arr = model.solve(params=params, log_level="warning")

# Progress (default) — timing per period
period_to_regime_to_V_arr = model.solve(params=params)  # log_level="progress"

# Debug — full diagnostics + snapshot persistence
period_to_regime_to_V_arr = model.solve(params=params, log_level="debug", log_path="./debug/")
```

Using `log_level="debug"` without providing `log_path` raises a `ValueError`.

## Debug snapshots

When `log_level="debug"` and `log_path` is provided, pylcm saves a **snapshot
directory** containing all inputs and outputs. This lets you reconstruct a failed run on
a different machine.

### What's saved

Each snapshot is a directory (e.g. `solve_snapshot_001/`) containing:

| File | Contents |
|------|----------|
| `arrays.h5` | Value function arrays in HDF5 (datasets at `/V_arr/{period}/{regime}`) |
| `model.pkl` | The Model instance (cloudpickle) |
| `params.pkl` | User parameters (cloudpickle) |
| `initial_states.pkl` | Initial state arrays (simulate only) |
| `initial_regimes.pkl` | Initial regime assignments (simulate only) |
| `result.pkl` | SimulationResult (simulate only) |
| `metadata.json` | Snapshot type, platform string, field manifest |
| `pixi.lock` | Lock file from the project root |
| `pyproject.toml` | Project file from the project root |
| `REPRODUCE.md` | Step-by-step reconstruction recipe |

### Creating snapshots

```python
# Solve snapshot
period_to_regime_to_V_arr = model.solve(
    params=params, log_level="debug", log_path="./debug/"
)
# Creates: ./debug/solve_snapshot_001/

# Simulate snapshot (with pre-solved value functions)
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=period_to_regime_to_V_arr,
    log_level="debug",
    log_path="./debug/",
)
# Creates: ./debug/simulate_snapshot_001/

# Simulate snapshot (solving automatically)
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=None,
    log_level="debug",
    log_path="./debug/",
)
# Creates: ./debug/simulate_snapshot_001/
```

### Loading snapshots

```python
from lcm import load_snapshot

# Load the full snapshot
snapshot = load_snapshot("./debug/solve_snapshot_001")
snapshot.model       # the Model instance
snapshot.params      # the user parameters
snapshot.period_to_regime_to_V_arr  # value function arrays (loaded from HDF5)

# Re-run the solve to reproduce the result
period_to_regime_to_V_arr = snapshot.model.solve(params=snapshot.params)
```

For large snapshots, skip fields you don't need:

```python
# Load without the (potentially large) value function arrays
snapshot = load_snapshot("./debug/solve_snapshot_001", exclude=["period_to_regime_to_V_arr"])
snapshot.period_to_regime_to_V_arr  # None
snapshot.model       # still available
```

### Platform mismatch

Each snapshot records the platform it was created on (e.g. `x86_64-Linux`).
When loading on a different platform, a warning is emitted:

```text
WARNING  Snapshot created on x86_64-Linux but loading on arm64-Darwin
         — environment may not match
```

To reproduce the environment exactly, use the bundled lock file:

```bash
cp ./debug/solve_snapshot_001/pixi.lock .
cp ./debug/solve_snapshot_001/pyproject.toml .
pixi install --frozen
```

## Snapshot retention

Snapshots accumulate when running inside an optimization loop. The `log_keep_n_latest`
parameter (default 3) limits how many snapshot directories are kept per type:

```python
period_to_regime_to_V_arr = model.solve(
    params=params, log_level="debug", log_path="./debug/", log_keep_n_latest=5
)
```

After each write, the oldest directories beyond the limit are deleted automatically.

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
period_to_regime_to_V_arr = model.solve(params=bad_params)
```

The traceback now points to the exact line in your user-defined functions where the
NaN originates.

## Inspecting value function arrays

The solution `period_to_regime_to_V_arr` is a nested mapping: `period -> regime_name -> array`. You can
iterate over it to check shapes, look for NaN/inf, or plot slices:

```python
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

period_to_regime_to_V_arr = model.solve(params=params)

# Check for issues
for period, regimes in period_to_regime_to_V_arr.items():
    for regime_name, V_arr in regimes.items():
        n_nan = int(jnp.sum(jnp.isnan(V_arr)))
        n_inf = int(jnp.sum(jnp.isinf(V_arr)))
        if n_nan > 0 or n_inf > 0:
            print(f"Period {period}, regime '{regime_name}': "
                  f"shape={V_arr.shape}, NaN={n_nan}, Inf={n_inf}")

# Plot a 1D slice (e.g. value over wealth grid for first period)
period = 0
regime_name = "working"
V_arr = period_to_regime_to_V_arr[period][regime_name]

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
