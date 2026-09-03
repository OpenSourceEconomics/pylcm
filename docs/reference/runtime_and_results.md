---
title: Runtime, results, and persistence
---

# Runtime, results, and persistence

## Solving

`model.solve(params=..., log_level=...)` returns one immutable `SolutionResult` for
every built-in solver. Its `values` mapping is indexed as
`period -> regime -> value array`; replay and collective-dissolution data stay in the
addressed artifact stores instead of changing the return type.

Optional arguments:

- `max_compilation_workers` caps parallel XLA compilation;
- `log_path` and `log_keep_n_latest` control diagnostic snapshots;
- `retention` selects which post-solve artifacts remain available.

There are no flag-selected tuple returns. Pass the complete result to
`model.simulate(solution=...)`; omitting `solution` asks simulation to solve first.

(api-solution-result)=

### Solution results

`model.solve(...)` keeps values, metadata, replay artifacts, diagnostics, and explicit
omission reasons in one `SolutionResult`:

```python
from lcm.solver_api import ResultRetention

solution = model.solve(
    params=params,
    log_level="debug",
    retention=ResultRetention.VALUES_AND_REPLAY,
)

V_working = solution.value(period=0, regime="working")
```

The result and its supporting types live in the lightweight `lcm.solver_api` submodule;
they are not re-exported from the top-level `lcm` namespace. The retention modes are:

| Mode                        | Retention-controlled result data                                             |
| --------------------------- | ---------------------------------------------------------------------------- |
| `VALUES`                    | Values; no replay artifacts                                                  |
| `VALUES_AND_REPLAY`         | Values plus applicable simulation-policy and dissolution artifacts (default) |
| `ALL_PERSISTABLE_ARTIFACTS` | All currently persistable artifacts; presently the same replay set           |

Artifacts are addressed by an `ArtifactRef(period=..., regime=..., key=...)` and kept in
immutable `ArtifactStore` instances. Built-in key identities include
`SIMULATION_POLICY`, `DISSOLUTION_FLAG`, `EGM_CONTINUATION`, and `SOLVER_DIAGNOSTICS`.
The `omissions` mapping distinguishes an artifact that is not applicable, not requested,
unsupported, or not persisted.

Metadata binds the result to the exact in-memory `Model` instance and to a SHA-256
digest of the canonical flat parameters used for the solve. The instance token survives
when that model is pickled and restored, but it is deliberately not a durable model
fingerprint. Each `(period, regime)` value also has a lightweight `ValueArraySchema`
recording its exact shape, dtype, and canonical named axes. That record is descriptive:
it does not authenticate the value. Simulation uses an immutable descriptor owned by the
model, built from the canonical model and parameters plus private solve-side facts for
data-dependent adaptive axes, then checks the value and its repeated schema
independently against that description.

Solver diagnostics follow `log_level`, independently of retention. Continuation
artifacts used during backward induction are not currently retainable: their absence is
recorded as `NOT_REQUESTED`, or as `UNSUPPORTED` when `ALL_PERSISTABLE_ARTIFACTS`
requests everything the boundary can describe.

`SolutionResult` persistence is not implemented yet. `save_solution()` persists only a
standalone value-function mapping; it does not serialize the labelled metadata or
artifact stores. This boundary is also not yet a stable contract for out-of-tree solver
implementations.

## Simulation

`model.simulate(...)` accepts parameters, initial conditions, an optional complete
`SolutionResult` as `solution=...`, and a required `log_level`. Omitting `solution`
solves first. Bare value mappings and separate policy or dissolution-flag inputs are not
accepted.

Before consuming a `SolutionResult`, simulation checks its in-memory model identity,
canonical-parameter digest, schema versions, period count, regime order, solver types,
and exact active period/regime value coverage. It unconditionally checks every value and
its descriptive schema independently against the model-owned shape, canonical dtype, and
named axes, including at `log_level="off"`. All artifact stores and omission records
must address active result cells with the exact key version and channel; one reference
cannot appear in multiple stores or be both present and recorded as omitted. Simulation
also checks required EGM/NNBEGM replay policies against the model-owned payload type,
canonical dtypes, complete ordered axes, action roles, categorical code domains, and
consuming route. Collective dissolution flags are checked against the model-owned
Boolean dtype and exact collective state shape. A values-only result is therefore
sufficient for a model whose decisions are fully recoverable from values, but fails
closed before forward simulation when a required replay artifact is absent or invalid.
Use the default `ResultRetention.VALUES_AND_REPLAY` when the model may require such
artifacts.

`subject_batch_size` streams subjects without changing results. `seed` controls random
draws. A collective model may require an addressed dissolution replay artifact and
`own_stakeholder`; see [Collective regimes](collective_regimes.md).

Initial conditions are a mapping of state names plus `regime_id` to equal-length arrays,
or a DataFrame with a `regime_name` column.

## Validation and logging

`log_level` controls both output and runtime validation:

| Level        | Behavior                                                      |
| ------------ | ------------------------------------------------------------- |
| `"off"`      | Silent; runtime probability and non-finite checks skipped     |
| `"warning"`  | Validate, warn, continue                                      |
| `"progress"` | Warning behavior plus timings                                 |
| `"debug"`    | Validate and raise at first failure; include value statistics |

Start model development at `"debug"`. Reduce validation only after the model is trusted
and the cost matters.

pylcm enables a persistent JAX compilation cache by default. Set
`JAX_COMPILATION_CACHE_DIR` to choose the full directory or `LCM_COMPILATION_CACHE_NAME`
to choose the project-specific leaf. Set `XLA_PYTHON_CLIENT_PREALLOCATE=true` before
importing pylcm to restore JAX's device preallocation; pylcm otherwise requests
on-demand allocation.

(api-simulation-result)=

## `SimulationResult`

`to_dataframe(additional_targets=None, use_labels=True, terminal_rows="first")`
materializes a flat DataFrame. `additional_targets` accepts selected DAG outputs or
`"all"`. `terminal_rows="all"` retains every frozen absorbing row; the default keeps
only terminal entry.

Inspection properties include `regime_names`, `state_names`, `action_names`,
`n_periods`, `n_subjects`, `available_targets`, `raw_results`, `flat_params`, and
`period_to_regime_to_V_arr`.

`SimulationResult.save(directory=...)` writes array checkpoints, value functions,
metadata, and a Feather table. `SimulationResult.load(directory=...)` restores it.

(api-standalone-persistence)=

## Standalone persistence

- `save_solution(period_to_regime_to_V_arr=solution.values, path=...)` and
  `load_solution(path=...)` persist value functions only.
- `SolveSnapshot` and `SimulateSnapshot` describe diagnostic snapshots.
- `load_snapshot(path, exclude=...)` loads a snapshot, with optional components omitted.

Workflow: [Solving and simulating](../user_guide/solving_and_simulating.md),
[DataFrame interoperability](../user_guide/pandas_interop.md), and
[Debugging](../user_guide/debugging.md).
