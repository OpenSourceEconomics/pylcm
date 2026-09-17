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

`log_level` controls console verbosity *and* the runtime-validation policy — how
`solve()` / `simulate()` react to an invalid transition-probability ensemble or a NaN
value function. See [Solving and Simulating](solving_and_simulating.md) for the full
`log_level` × `log_path` behaviour table.

```python
# Silent — no console output, no validation
solution = model.solve(params=params, log_level="off")

# Warnings only — invalid input is logged, the run continues
solution = model.solve(params=params, log_level="warning")

# Debug — validation raises, full diagnostics
solution = model.solve(params=params, log_level="debug")
value_functions = solution.values

# Debug + snapshot persistence
solution = model.solve(params=params, log_level="debug", log_path="./debug/")
```

`log_path` is optional at every level — including `"debug"`.

### Run every production model at `"debug"` at least once

`"off"` skips runtime validation entirely, and some of what it skips cannot be
reconstructed from the output afterwards. The clearest case is regime-transition
probabilities that do not represent unit mass.

Every aggregation route normalizes the continuation by the mass it actually receives. So
if a regime transition puts probability on a target that is not active in the next
period, that target is dropped from the continuation and the remaining targets are
renormalized. Left alone, that would make the solved value function come back finite,
plausible, and *independent of the missing mass*: a model whose survival probability
ranges from 0.999999 to 0.000001 would produce bit-identical values, because in each
case the surviving branch renormalizes to one.

The arithmetic itself carries a backstop against that, at every log level, `"off"`
included: a represented regime mass more than `1e-3` away from one turns the
continuation into NaN, so a grossly misspecified transition cannot return a plausible
number. The tolerance is deliberately loose — it catches a wrong *model*, not a
numerical inaccuracy. Smaller mass errors still pass silently, and at the top of the
validator's own tolerance they are already large enough to reverse the optimal action.

A NaN is also not a diagnosis. It tells you the model is wrong; it does not tell you
which regime, which target, or which age. Run the model once at `log_level="debug"`,
with the parameters you intend to use, and the transition check reports the offending
`(source regime, target regime, age)` directly.

```python
# Do this once per model and parameter regime, before trusting any output.
solution = model.solve(params=params, log_level="debug")
```

After that run passes, `"off"` is a reasonable choice for an estimation loop that
re-solves the same model at many parameter vectors — validation costs roughly a fifth of
a warm solve, so skipping it is worth real time. It is only safe because the structural
question was already answered.

### Never diagnose a failure below `"debug"`

Unless the cause is **very** obvious, do not reason about a failure observed at `"off"`,
`"warning"`, or `"progress"`. Reproduce it at `log_level="debug"` first and diagnose
from that run. The lower levels are for models you already trust; the moment one
misbehaves, the setting that made it cheap is also the setting that removed the
information you need.

The mass check above is the example to keep in mind. At `"off"` a non-unit regime mass
reaches you as NaN in the value function and nothing else — no source regime, no target,
no age, and no indication that transition probabilities are involved at all. From that
observation the natural hypotheses are the ones you can see: the utility function at the
edge of its domain, a constraint that admits no action, an interpolation running off the
grid. Every one of them is wrong, and each is expensive to rule out. The same run at
`"debug"` names the offending `(source regime, target regime, age)` in the exception
message.

The general form: `"off"` and `"warning"` change which failures are *visible* and how
much of the failure survives into what you can inspect. A hypothesis formed from a
degraded observation is a hypothesis about the log level as much as about the model.

## Debug snapshots

When `log_path` is provided, pylcm saves a **snapshot directory** containing all inputs
and outputs, so you can reconstruct a failed run on a different machine. In `"debug"`
mode a snapshot is written on every solve and on a raised failure; in `"warning"` /
`"progress"` mode one is written whenever a warned failure leaves NaN in the value
function.

### What's saved

Each snapshot is a directory (e.g. `solve_snapshot_001/`) containing:

| File                     | Contents                                                               |
| ------------------------ | ---------------------------------------------------------------------- |
| `arrays.h5`              | Value function arrays in HDF5 (datasets at `/V_arr/{period}/{regime}`) |
| `model.pkl`              | The Model instance (cloudpickle)                                       |
| `params.pkl`             | User parameters (cloudpickle)                                          |
| `initial_conditions.pkl` | Initial state arrays and regime codes (simulate only)                  |
| `result.pkl`             | SimulationResult (simulate only)                                       |
| `metadata.json`          | Snapshot type, platform string, field manifest                         |
| `pixi.lock`              | Lock file from the project root                                        |
| `pyproject.toml`         | Project file from the project root                                     |
| `REPRODUCE.md`           | Step-by-step reconstruction recipe                                     |

### Creating snapshots

```python
# Solve snapshot
solution = model.solve(params=params, log_level="debug", log_path="./debug/")
# Creates: ./debug/solve_snapshot_001/

# Simulate snapshot (with a pre-solved complete result)
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    solution=solution,
    log_level="debug",
    log_path="./debug/",
)
# Creates: ./debug/simulate_snapshot_001/

# Simulate snapshot (solving automatically)
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
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
snapshot.model  # the Model instance
snapshot.params  # the user parameters
snapshot.period_to_regime_to_V_arr  # value function arrays (loaded from HDF5)

# Re-run the solve to reproduce the result
solution = snapshot.model.solve(params=snapshot.params, log_level="debug")
```

For large snapshots, skip fields you don't need:

```python
# Load without the (potentially large) value function arrays
snapshot = load_snapshot(
    "./debug/solve_snapshot_001", exclude=["period_to_regime_to_V_arr"]
)
snapshot.period_to_regime_to_V_arr  # None
snapshot.model  # still available
```

### Platform mismatch

Each snapshot records the platform it was created on (e.g. `x86_64-Linux`). When loading
on a different platform, a warning is emitted:

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
solution = model.solve(
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
solution = model.solve(params=bad_params, log_level="debug")
```

The traceback now points to the exact line in your user-defined functions where the NaN
originates.

## Inspecting value function arrays

The `values` field of `SolutionResult` is a nested mapping:
`period -> regime_name -> array`. You can iterate over it to check shapes, look for
NaN/inf, or plot slices:

```python
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

solution = model.solve(params=params, log_level="debug")
value_functions = solution.values

# Check for issues
for period, regimes in value_functions.items():
    for regime_name, V_arr in regimes.items():
        n_nan = int(jnp.sum(jnp.isnan(V_arr)))
        n_inf = int(jnp.sum(jnp.isinf(V_arr)))
        if n_nan > 0 or n_inf > 0:
            print(
                f"Period {period}, regime '{regime_name}': "
                f"shape={V_arr.shape}, NaN={n_nan}, Inf={n_inf}"
            )

# Plot a 1D slice (e.g. value over wealth grid for first period)
period = 0
regime_name = "working"
V_arr = value_functions[period][regime_name]

fig = go.Figure()
fig.add_trace(go.Scatter(y=V_arr.tolist(), mode="lines", name="V(wealth)"))
fig.update_layout(title=f"Value function, period {period}, regime '{regime_name}'")
fig.show()
```

## Failure snapshots

When `log_path` is set and `solve()` raises `InvalidValueFunctionError` (in `"debug"`
mode), a snapshot is saved automatically. This lets you inspect the partial solution
(value functions for periods that completed before the error) on another machine.

```python
# log_path is enough to get a failure snapshot
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    log_level="debug",
    log_path="./debug/",
)
```

## NaN diagnostics

When the solver detects NaN in the value function, it reports which intermediate is the
source. The error message includes a diagnostic summary like:

```text
Diagnostics for regime 'working' at age 55:
  F: 0.9500 feasible
  Among feasible state-action pairs:  U: 0.0000 NaN  |  E[V]: 0.3200 NaN
  Regime probs: working: 0.8500 | retired: 0.1500
  E[V] NaN fraction by state (among feasible state-action pairs):
    wealth                   [0.00, 0.00, 0.12, 0.45, 0.80, 0.95, 1.00, 1.00, 1.00, 1.00]
    health                   [0.00, 0.64]
```

This tells you:

- **F: 0.9500 feasible** --- 95% of state-action combinations satisfy all constraints.
- **U: 0.0000 NaN** (among feasible) --- utility is clean in every feasible cell; the
  problem is not in the utility function.
- **E\[V\]: 0.3200 NaN** (among feasible) --- 32% of E[V] values in feasible cells are
  NaN. The NaN comes from the continuation value, not from utility. Infeasible cells are
  excluded because the solver masks them out before taking the max, so a NaN there would
  not propagate to `V_arr`.
- **Regime probs** --- how much weight the failing cell places on each reachable target
  regime.
- **By-state breakdown** --- NaN concentrates at high wealth levels and in the second
  health state. This points to the regime transition function or next-period value
  interpolation for those states.

The diagnostic functions are compiled lazily --- only when NaN is detected. There is no
compilation overhead in the normal (no-NaN) solve path.

## Understanding error messages

pylcm raises specific exceptions to help you diagnose problems. Every one of them lives
in `lcm.exceptions` and derives from `lcm.exceptions.PyLCMError`, so `except PyLCMError`
catches anything pylcm itself raises:

```python
from lcm.exceptions import ExecutionPlanningError, PyLCMError
```

### Model definition

- **`ModelInitializationError`**: Something is wrong with the model definition
  (mismatched regime names, unused variables, etc.). Read the message carefully --- it
  usually lists all issues found.

- **`RegimeInitializationError`**: A single regime is invalid. A regime is validated
  both at its own construction and again when a model finalizes it, so the same defect
  surfaces from either call. It subclasses `ModelInitializationError`, so catching that
  catches both.

- **`GridInitializationError`**: A grid declaration is invalid. The same exception
  covers the age grid and the category class a discrete grid is built from.

- **`CategoricalDefinitionError`**: An `@categorical`-decorated class violates the
  contract that every field is annotated `ScalarInt`. Raised at decoration time, before
  any grid, regime, or derived-categorical mapping is built.

- **`InvalidNameError`**: Names are invalid --- a name contains the reserved separator,
  or two name sets that must be disjoint overlap. A parameter written at two levels of
  the params dict also lands here.

- **`NBEGMCaseError`**: An NBEGM case-boundary or formula-piece declaration is invalid
  --- a malformed boundary or piece, hidden branching caught by the smoothness gate, or
  a declaration outside the supported case-piece scope.

- **`ModelSealError`**: A name one of the model's callables reads was rebound after the
  model was built. The model captures its callables together with the globals and
  closure cells they read, and refuses to solve or simulate against a rebinding rather
  than produce a result its durable identity would accept for the wrong model.

### Parameters, inputs, and results

- **`InvalidParamsError`**: The params structure does not match the params template.

- **`InvalidInitialConditionsError`**: The initial states or regime codes handed to
  `simulate` are invalid --- a wrongly shaped or wrongly typed array, a value off its
  grid, an invalid discrete or regime code, or a missing `own_stakeholder` entry where
  roles are required.

- **`InvalidAdditionalTargetsError`**: A requested additional DAG target is not
  available on the result.

- **`InvalidSimulationInputError`**: Caller-supplied solve artifacts cannot drive
  simulation --- a missing value-function array for a continuation target, or a missing
  or mismatched replay policy where the solver's decision cannot be reconstructed from
  value functions alone.

- **`UnsupportedOperationError`**: A valid model requests a runtime operation pylcm does
  not support, such as simulating a regime whose solver declares its decision
  irreproducible.

### Planning and execution

- **`ExecutionPlanningError`**: The requested execution policy cannot produce a valid
  plan. This covers every device-memory budget, axis-width, device-selection and
  sharding refusal, so it is the exception a tuning run meets most often. When the
  selected devices capped the requested budget, the message names both the requested and
  the effective bytes together with the headroom fraction that separates them; see
  [Performance and memory tuning](tuning.md#set-a-device-memory-budget).

- **`FunctionDispatchError`**: A function cannot be dispatched over the variables it is
  asked to map --- a positional-only parameter, or a requested variable absent from the
  signature.

### Numerical validation

- **`InvalidValueFunctionError`**: The value function array contains NaN at a given age
  and regime. The message lists common causes and a diagnostic summary showing NaN
  fractions per intermediate (U, E[V], Q) and per state dimension. A debug snapshot is
  saved automatically when `log_path` is set.

- **`InvalidRegimeTransitionProbabilitiesError`**: Regime transition probabilities are
  non-finite, outside [0, 1], don't sum to 1, or assign positive probability to an
  inactive regime. The message includes the source regime, age range, and a table of
  failing entries.

- **`InvalidStateTransitionProbabilitiesError`**: A `MarkovTransition` produces an
  output with the wrong outcome-axis size, values outside [0, 1], rows that don't sum to
  1, or `probs_array[…]` subscripts that don't match the signature parameter order.

- **`OuterSearchConvergenceError`**: An adaptive outer mesh reached its node or round
  budget while validation-marked intervals remained. Inference-grade continuous-outer
  solves fail closed rather than silently return a degraded solution.

- **`ScaledLotteryDifferentiationError`**: A lottery is differentiated with respect to
  probabilities too small to represent as ordinary floating-point numbers. Returning
  zero would suggest a locally flat objective to an optimizer, so pylcm raises instead.

- **`UnrepresentableOuterCandidateError`**: Replay cannot reconstruct an outer candidate
  the solve kept, because the recovered action reaches a stock outside the outer state's
  declared domain. Such candidates are dropped from that subject's choice set, and the
  message reports how many.

- **`ExactAffineKernelUnavailableError`**: A certified exact-affine operation
  (`ExactEnvelope`, or NBEGM's `"certified"` ownership mode) found no loadable compiled
  payload for the active JAX backend. pylcm raises rather than fall back to approximate
  floating-point comparisons.

### Persistence

- **`SolutionIntegrityError`**: A persisted solution archive failed an integrity check.

- **`IncompatibleSolutionError`**: A solution uses an unsupported schema or plugin
  version. pylcm rejects a mismatch rather than migrating it silently.

## Inspect the resolved execution plan

`model.simulate(...)` reports the execution plan it actually dispatched: the forward
route, the ordered subject devices and their backend, the resolved planner axis widths
by regime, the outer chunk count and admitted chunk widths, and the budget mode with its
effective device-memory bytes. A one-line summary logs at `log_level="progress"` and
`"debug"`, and the complete record at `"debug"` only.

Read it when a run is slower or larger than expected but raises nothing: it is the only
statement of which route engaged, at which widths, and against which budget. The same
record is available on the result as `SimulationResult.plan_summary`, so a batch job can
keep it without keeping the log. It is `None` for a result read back with
`SimulationResult.load`. See
[Runtime, results, and persistence](../reference/runtime_and_results.md) for the exact
contract.

## See also

- [Solving and simulating](solving_and_simulating.md) for the common runtime workflow.
- [Runtime, results, and persistence](../reference/runtime_and_results.md) for exact
  logging and snapshot contracts.
- [Performance and memory tuning](tuning.md) after the model is correct.
