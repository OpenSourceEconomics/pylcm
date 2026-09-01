---
title: Runtime, results, and persistence
---

# Runtime, results, and persistence

## Solving

`model.solve(params=..., log_level=...)` returns an immutable
`period -> regime -> value array` mapping.

Optional arguments:

- `max_compilation_workers` caps parallel XLA compilation;
- `log_path` and `log_keep_n_latest` control diagnostic snapshots;
- `return_simulation_policy=True` also returns published off-grid policy artifacts;
- `return_dissolution_flags=True` also returns collective dissolution masks.

The existing `solve()` contract is unchanged: depending on those two flags, it returns
either the value mapping or the historical tuple of parallel mappings.

(api-solution-result)=

### Experimental labelled solution results

`model.solve_result(...)` is an experimental, migration-oriented alternative that keeps
values, metadata, replay artifacts, diagnostics, and explicit omission reasons in one
`SolutionResult`:

```python
from lcm.solver_api import ResultRetention

solution = model.solve_result(
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
recording its exact shape, dtype, and canonical named axes.

Solver diagnostics follow `log_level`, independently of retention. Continuation
artifacts used during backward induction are not currently retainable: their absence is
recorded as `NOT_REQUESTED`, or as `UNSUPPORTED` when `ALL_PERSISTABLE_ARTIFACTS`
requests everything the boundary can describe.

`SolutionResult` persistence is not implemented yet. `save_solution()` persists only the
legacy value-function mapping; it does not serialize the labelled metadata or artifact
stores. This boundary is also not yet a stable contract for out-of-tree solver
implementations.

## Simulation

`model.simulate(...)` accepts parameters, initial conditions, a value-function mapping
or `None`, and a required `log_level`. Passing `None` solves first. Alternatively, pass
the complete labelled result as `solution=solution`; do not combine it with the legacy
value, policy, or dissolution-flag arguments.

Before consuming a `SolutionResult`, simulation checks its in-memory model identity,
canonical-parameter digest, schema versions, period count, regime order, solver types,
and exact active period/regime value coverage. It unconditionally checks every value's
shape, dtype, and named axes, including at `log_level="off"`. All artifact stores and
omission records must address active result cells; one reference cannot appear in
multiple stores or be both present and recorded as omitted. Simulation also validates
the structure of every required EGM/NNBEGM replay policy and checks collective
dissolution flags for Boolean dtype and the exact collective state shape. A values-only
result is therefore sufficient for a model whose decisions are fully recoverable from
values, but fails closed before forward simulation when a required replay artifact is
absent or invalid. Use the default `ResultRetention.VALUES_AND_REPLAY` when the model
may require such artifacts.

`subject_batch_size` streams subjects without changing results. `seed` controls random
draws. A collective model may require `period_to_regime_to_dissolution_flags` and
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

- `save_solution(period_to_regime_to_V_arr=..., path=...)` and `load_solution(path=...)`
  persist value functions.
- `SolveSnapshot` and `SimulateSnapshot` describe diagnostic snapshots.
- `load_snapshot(path, exclude=...)` loads a snapshot, with optional components omitted.

Workflow: [Solving and simulating](../user_guide/solving_and_simulating.md),
[DataFrame interoperability](../user_guide/pandas_interop.md), and
[Debugging](../user_guide/debugging.md).
