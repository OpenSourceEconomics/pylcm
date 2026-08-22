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

## Simulation

`model.simulate(...)` accepts parameters, initial conditions, a value-function mapping
or `None`, and a required `log_level`. Passing `None` solves first.

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

## Standalone persistence

- `save_solution(solution, path)` and `load_solution(path)` persist value functions.
- `SolveSnapshot` and `SimulateSnapshot` describe diagnostic snapshots.
- `load_snapshot(path, exclude=...)` loads a snapshot, with optional components omitted.

Workflow: [Solving and simulating](../user_guide/solving_and_simulating.md),
[DataFrame interoperability](../user_guide/pandas_interop.md), and
[Debugging](../user_guide/debugging.md).
