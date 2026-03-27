# Migration Mapping: Module Reorganization (Issue #296)

Detailed old → new path mapping for tracing git history across the restructuring. Use
`git log --follow <new-path>` to trace a file's history through the rename.

## File moves

| Old path                                                    | New path                                   |
| ----------------------------------------------------------- | ------------------------------------------ |
| `src/lcm/input_processing/regime_processing.py`             | `src/lcm/regime_building/processing.py`    |
| `src/lcm/input_processing/util.py`                          | `src/lcm/regime_building/variable_info.py` |
| `src/lcm/input_processing/params_processing.py`             | `src/lcm/params/processing.py`             |
| `src/lcm/input_processing/create_regime_params_template.py` | `src/lcm/params/regime_template.py`        |
| `src/lcm/Q_and_F.py`                                        | `src/lcm/regime_building/Q_and_F.py`       |
| `src/lcm/next_state.py`                                     | `src/lcm/regime_building/next_state.py`    |
| `src/lcm/max_Q_over_a.py`                                   | `src/lcm/regime_building/max_Q_over_a.py`  |
| `src/lcm/function_representation.py`                        | `src/lcm/regime_building/V.py`             |
| `src/lcm/argmax.py`                                         | `src/lcm/regime_building/argmax.py`        |
| `src/lcm/ndimage.py`                                        | `src/lcm/regime_building/ndimage.py`       |
| `src/lcm/random.py`                                         | `src/lcm/simulation/random.py`             |
| `src/lcm/grid_helpers.py`                                   | `src/lcm/grids/coordinates.py`             |
| `src/lcm/functools.py`                                      | `src/lcm/utils/functools.py`               |
| `src/lcm/dispatchers.py`                                    | `src/lcm/utils/dispatchers.py`             |
| `src/lcm/state_action_space.py`                             | `src/lcm/state_action_space.py`            |
| `src/lcm/error_handling.py`                                 | `src/lcm/utils/error_handling.py`          |
| `src/lcm/logging.py`                                        | `src/lcm/utils/logging.py`                 |
| `src/lcm/simulation/utils.py`                               | `src/lcm/simulation/transitions.py`        |

## File splits

| Old path           | New paths                      | What went where                                                                                                                                                                                                     |
| ------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lcm/grids.py` | `src/lcm/grids/categorical.py` | `categorical()`, `validate_category_class()`, `_validate_discrete_grid()`                                                                                                                                           |
|                    | `src/lcm/grids/continuous.py`  | `Grid`, `ContinuousGrid`, `UniformContinuousGrid`, `LinSpacedGrid`, `LogSpacedGrid`, `IrregSpacedGrid`, `_validate_continuous_grid`, `_validate_irreg_spaced_grid`                                                  |
|                    | `src/lcm/grids/piecewise.py`   | `Piece`, `PiecewiseLinSpacedGrid`, `PiecewiseLogSpacedGrid`, `_parse_interval`, `_get_effective_bounds`, `_init_piecewise_grid_cache`, `_validate_piecewise_lin_spaced_grid`, `_validate_piecewise_log_spaced_grid` |
|                    | `src/lcm/grids/discrete.py`    | `_DiscreteGridBase`, `DiscreteGrid`                                                                                                                                                                                 |
|                    | `src/lcm/grids/__init__.py`    | Re-exports all public names                                                                                                                                                                                         |
| `src/lcm/utils.py` | `src/lcm/utils/containers.py`  | `Unset`, `ensure_containers_are_immutable`, `ensure_containers_are_mutable`, `_make_immutable`, `_make_mutable`, `find_duplicates`, `get_field_names_and_values`, `first_non_none`                                  |
|                    | `src/lcm/utils/namespace.py`   | `flatten_regime_namespace`, `unflatten_regime_namespace`                                                                                                                                                            |

## File merges (regime_components.py eliminated)

| Function                                  | Old location                                    | New location                                                              |
| ----------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------- |
| `build_Q_and_F_functions`                 | `src/lcm/input_processing/regime_components.py` | `src/lcm/regime_building/Q_and_F.py`                                      |
| `build_max_Q_over_a_functions`            | same                                            | `src/lcm/regime_building/max_Q_over_a.py`                                 |
| `_build_max_Q_over_a_function`            | same                                            | `src/lcm/regime_building/max_Q_over_a.py`                                 |
| `build_argmax_and_max_Q_over_a_functions` | same                                            | `src/lcm/regime_building/max_Q_over_a.py`                                 |
| `_build_argmax_and_max_Q_over_a_function` | same                                            | `src/lcm/regime_building/max_Q_over_a.py`                                 |
| `_get_vmap_params`                        | same                                            | `src/lcm/regime_building/max_Q_over_a.py`                                 |
| `build_next_state_simulation_functions`   | same                                            | `src/lcm/regime_building/next_state.py`                                   |
| `build_regime_transition_probs_functions` | same                                            | `src/lcm/regime_building/processing.py`                                   |
| `_wrap_regime_transition_probs`           | same                                            | `src/lcm/regime_building/processing.py`                                   |
| `_wrap_deterministic_regime_transition`   | same                                            | `src/lcm/regime_building/processing.py`                                   |
| `create_v_interpolation_info`             | `src/lcm/input_processing/regime_processing.py` | `src/lcm/regime_building/V.py` (renamed to `create_v_interpolation_info`) |

## Deleted files

| Path                                                        | Reason                                                  |
| ----------------------------------------------------------- | ------------------------------------------------------- |
| `src/lcm/input_processing/__init__.py`                      | Directory eliminated                                    |
| `src/lcm/input_processing/regime_components.py`             | Functions merged into target modules                    |
| `src/lcm/input_processing/regime_processing.py`             | Moved to `regime_building/processing.py`                |
| `src/lcm/input_processing/util.py`                          | Moved to `regime_building/variable_info.py`             |
| `src/lcm/input_processing/params_processing.py`             | Moved to `params/processing.py`                         |
| `src/lcm/input_processing/create_regime_params_template.py` | Moved to `params/regime_template.py`                    |
| `src/lcm/grids.py`                                          | Split into `grids/` package                             |
| `src/lcm/utils.py`                                          | Split into `utils/containers.py` + `utils/namespace.py` |

## Test moves

| Old path                                                       | New path                                                      |
| -------------------------------------------------------------- | ------------------------------------------------------------- |
| `tests/input_processing/`                                      | `tests/regime_building/`                                      |
| `tests/input_processing/__init__.py`                           | `tests/regime_building/__init__.py`                           |
| `tests/input_processing/test_create_regime_params_template.py` | `tests/regime_building/test_create_regime_params_template.py` |
| `tests/input_processing/test_process_params.py`                | `tests/regime_building/test_process_params.py`                |
| `tests/input_processing/test_regime_processing.py`             | `tests/regime_building/test_regime_processing.py`             |
