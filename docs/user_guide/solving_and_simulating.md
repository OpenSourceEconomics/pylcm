---
title: Solving and Simulating
---

# Solving and Simulating

Once you have defined a `Model` and prepared your parameters, pylcm solves via backward
induction and simulates forward.

## Solving

```python
period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")
```

Performs backward induction using dynamic programming. Returns an immutable mapping of
`period -> regime_name -> value_function_array`.

### Log levels and runtime validation

`log_level` is a required argument: it controls both console verbosity *and* the
runtime-validation policy — how `solve()` / `simulate()` react to an invalid
transition-probability ensemble or a NaN value function. Start every project at
`"debug"` (validation runs and raises); ease to `"warning"` / `"off"` once the model is
trusted.

```python
# Debug — validation runs and raises on the first failure
period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

# Silent — no logging, no validation
period_to_regime_to_V_arr = model.solve(params=params, log_level="off")

# Validation runs but only warns; the run continues
period_to_regime_to_V_arr = model.solve(params=params, log_level="warning")

# Diagnostics + disk snapshots
period_to_regime_to_V_arr = model.solve(
    params=params, log_level="debug", log_path="./debug/"
)
```

The full behaviour of every `log_level` × `log_path` combination:

| `log_level`           | `log_path` | Runtime validation        | Console output                  | Snapshots to disk                                         |
| --------------------- | ---------- | ------------------------- | ------------------------------- | --------------------------------------------------------- |
| `"off"`               | (ignored)  | not run                   | silent                          | none                                                      |
| `"warning"`           | `None`     | runs → failures **warn**  | warnings                        | none                                                      |
| `"warning"`           | set        | runs → failures **warn**  | warnings                        | one per warned failure, capped at `log_keep_n_latest`     |
| `"progress"`          | `None`     | runs → failures **warn**  | warnings + timing               | none                                                      |
| `"progress"`          | set        | runs → failures **warn**  | warnings + timing               | one per warned failure, capped at `log_keep_n_latest`     |
| `"debug"` *(default)* | `None`     | runs → failures **raise** | warnings + timing + V_arr stats | none                                                      |
| `"debug"` *(default)* | set        | runs → failures **raise** | warnings + timing + V_arr stats | one per solve and on raise, capped at `log_keep_n_latest` |

`log_path` is optional at every level — snapshots are written only when it is set. In
`"warning"` / `"progress"` mode, an invalid model produces warnings and a numerically
meaningless result rather than an exception; use this to keep an estimation loop
running, but read the warnings.

See [Debugging](debugging.md) for details on snapshots.

## Simulating

```python
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=period_to_regime_to_V_arr,
    log_level="debug",
)
```

Forward simulation using solved value functions. Each agent starts from the given
initial conditions and makes optimal decisions at each period. Returns a
`SimulationResult` object.

## Simulate without pre-solving

When `period_to_regime_to_V_arr=None`, `simulate()` solves the model automatically
before simulating. Use this when you don't need the raw value function arrays:

```python
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=None,
    log_level="debug",
)
```

## Initial Conditions

### From a DataFrame

The standard way to supply initial conditions is as a pandas DataFrame with one row per
agent. Pass it directly to `simulate()`:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "regime_name": ["working_life", "working_life", "retirement", "working_life"],
        "age": [25.0, 25.0, 25.0, 25.0],
        "wealth": [1.0, 5.0, 10.0, 20.0],
        "health": ["good", "bad", "bad", "good"],  # string labels, auto-converted
    }
)

result = model.simulate(
    params=params,
    initial_conditions=df,
    period_to_regime_to_V_arr=None,
    log_level="debug",
)
```

Discrete states (those backed by a `DiscreteGrid`) are mapped from string labels to
integer codes automatically. See [Working with DataFrames and Series](pandas_interop.md)
for details.

### As JAX arrays

You can also pass initial conditions directly as JAX arrays — useful for programmatic
setups like grid searches or tests:

```python
initial_conditions = {
    "age": jnp.array([25.0, 25.0, 25.0, 25.0]),
    "wealth": jnp.array([1.0, 5.0, 10.0, 20.0]),
    "health": jnp.array([0, 1, 1, 0]),  # integer codes for discrete states
    "regime_id": jnp.array(
        [
            RegimeId.working_life,
            RegimeId.working_life,
            RegimeId.retirement,
            RegimeId.working_life,
        ]
    ),
}
```

- Every non-shock state must have an entry.
- `"regime_id"` must be included, with integer codes from the `regime_id_class`.
- All arrays must have the same length (= number of agents).
- Shock states are drawn automatically.

### Further arguments

- `log_level`: Required. Console verbosity and runtime-validation policy (same options
  and table as `solve()`); start at `"debug"`. Initial-condition validation (states
  on-grid, regimes valid) follows this policy too — `"off"` skips it.
- `seed=None`: Random seed for stochastic simulations (int).
- `log_path=None`: Directory for diagnostic snapshots; optional at every level.
- `log_keep_n_latest=3`: Maximum snapshot directories to retain.

### Heterogeneous initial ages

`"age"` must always be provided in `initial_conditions`. Each value must be a valid
point on the model's `AgeGrid`, and each subject's initial regime must be active at
their starting age. The most common case is that all subjects start at the initial age —
just pass a constant array.

Subjects can start at different ages:

```python
initial_conditions = {
    "age": jnp.array([40.0, 60.0]),
    "wealth": jnp.array([50.0, 50.0]),
    "regime_id": jnp.array(
        [
            model.regime_names_to_ids["working_life"],
            model.regime_names_to_ids["working_life"],
        ]
    ),
}
```

In the resulting DataFrame, each subject appears only from their starting age onward —
earlier periods are omitted, not filled with placeholders.

## Working with SimulationResult

### Converting to DataFrame

```python
df = result.to_dataframe()
```

Returns a pandas DataFrame with columns: `subject_id`, `period`, `age`, `regime_name`,
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
result.regime_names  # ['retirement', 'working_life']
result.state_names  # ['health', 'wealth']
result.action_names  # ['consumption', 'work']
result.n_periods  # 50
result.n_subjects  # 1000
```

### Persistence

`SimulationResult.save(directory=...)` writes three sibling artifacts:

- `arrays/` — orbax checkpoint of every JAX array (per-shard, no gathering of sharded
  V-arrays to a single device).
- `metadata.pkl` — `cloudpickle` of regimes, ages, and the parameter scaffold.
- `simulated_data.arrow` — a `feather` dump of `to_dataframe`, ready for downstream
  consumers that want the flat per-subject view without re-instantiating a
  `SimulationResult`.

```python
# Save
result.save(directory="my_results/")

# Load (reads arrays + metadata; the arrow file is for downstream consumers)
from lcm import SimulationResult

loaded = SimulationResult.load(directory="my_results/")
```

### Raw data (advanced)

```python
result.raw_results  # regime -> period -> PeriodRegimeSimulationData
result.flat_params  # processed parameter object
result.period_to_regime_to_V_arr  # value function arrays from solve()
```

## Typical Workflow

```python
import numpy as np
import pandas as pd
from lcm import Model

# 1. Define model (see previous pages)
model = Model(regimes={...}, ages=..., regime_id_class=...)

# 2. Set parameters
params = {
    "discount_factor": 0.95,
    "interest_rate": 0.03,
    ...
}

# 3. Prepare initial conditions as a DataFrame
initial_df = pd.DataFrame({
    "regime_name": "working_life",
    "age": model.ages.values[0],
    "wealth": np.linspace(1, 50, 100),
})

# 4. Simulate (solves automatically when period_to_regime_to_V_arr=None)
result = model.simulate(
    params=params,
    initial_conditions=initial_df,
    period_to_regime_to_V_arr=None,
    log_level="debug",
)

# 5. Analyze
df = result.to_dataframe(additional_targets="all")
df.groupby("period")["wealth"].mean()
```

## Float32 GPU Reproducibility

```{note}
At float32 precision, GPU simulation results are **not reproducible across process
invocations**. XLA compiles different fused kernels each time a process starts, changing
float32 accumulation order and producing ~1e-3 value function differences for large
models. This is a property of XLA's GPU compiler, not a pylcm bug.

**Key facts:**

- Float64 results are reproducible across processes.
- Float32 results are deterministic *within* a single process (repeated calls give
  identical output).
- `--xla_gpu_deterministic_ops=true` does **not** help — it only guarantees determinism
  within a single compiled program, not across separate compilations.
- Smaller models may be reproducible at float32 because XLA chooses the same kernel
  fusion strategy; larger models are not.

If you need bitwise-reproducible results for testing or validation, use float64 precision
(`jax.config.update("jax_enable_x64", True)`).
```

## See Also

- [Defining Models](defining_models.md) — constructing the `Model`
- [Parameters](parameters.md) — preparing the params dict
- [Working with DataFrames and Series](pandas_interop.md) — DataFrame conversion
  utilities
- [A Tiny Example](tiny_example.ipynb) — complete walkthrough
- [Examples](../examples/index.md) — full worked examples
