"""Utilities for converting between pandas and LCM data structures."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dags.tree import qname_from_tree_path, tree_path_from_qname
from jax import Array

from lcm.ages import AgeGrid
from lcm.error_handling import (
    _get_func_indexing_params,
)
from lcm.grids import DiscreteGrid, IrregSpacedGrid
from lcm.input_processing.params_processing import broadcast_to_template
from lcm.model import Model
from lcm.params import MappingLeaf
from lcm.params.sequence_leaf import SequenceLeaf
from lcm.regime import Regime
from lcm.shocks import _ShockGrid

type _PandasLeaf = bool | float | Array | pd.Series | MappingLeaf | SequenceLeaf
type PandasUserParams = Mapping[
    str,
    _PandasLeaf | Mapping[str, _PandasLeaf | Mapping[str, _PandasLeaf]],
]


def initial_conditions_from_dataframe(
    *,
    df: pd.DataFrame,
    model: Model,
) -> dict[str, Array]:
    """Convert a DataFrame of initial conditions to LCM initial conditions format.

    Args:
        df: DataFrame with columns for states and a "regime" column.
        model: The LCM Model instance.

    Returns:
        Dict mapping state names (plus `"regime"`) to JAX arrays. The
        `"regime"` entry contains integer codes derived from the `"regime"`
        column via `model.regime_names_to_ids`.

    Raises:
        ValueError: If the DataFrame is empty, the "regime" column is missing,
            contains invalid regime names, has unknown columns, is missing required
            states, or categorical columns contain invalid labels.

    """
    if "regime" not in df.columns:
        msg = "DataFrame must contain a 'regime' column."
        raise ValueError(msg)

    if len(df) == 0:
        msg = "DataFrame must not be empty."
        raise ValueError(msg)

    # Validate regime names
    valid_regimes = set(model.regime_names_to_ids.keys())
    invalid = set(df["regime"]) - valid_regimes
    if invalid:
        msg = (
            f"Invalid regime names in 'regime' column: {sorted(invalid)}. "
            f"Valid regimes: {sorted(valid_regimes)}."
        )
        raise ValueError(msg)

    state_columns = {col for col in df.columns if col != "regime"}
    _validate_state_columns(
        state_columns=state_columns, regimes=model.regimes, initial_regimes=df["regime"]
    )

    discrete_lookup = _build_discrete_grid_lookup(model.regimes)

    initial_conditions: dict[str, Array] = {}
    for col in df.columns:
        if col == "regime":
            continue

        if col in discrete_lookup:
            grid = discrete_lookup[col]
            label_to_code = dict(zip(grid.categories, grid.codes, strict=True))

            values = df[col]
            # Convert categorical dtype to string values
            if hasattr(values, "cat"):
                values = values.astype(str)

            invalid_labels = set(values) - set(grid.categories)
            if invalid_labels:
                msg = (
                    f"Invalid labels for discrete state '{col}': "
                    f"{sorted(invalid_labels)}. "
                    f"Valid labels: {list(grid.categories)}."
                )
                raise ValueError(msg)

            initial_conditions[col] = jnp.array(values.map(label_to_code).to_numpy())
        else:
            initial_conditions[col] = jnp.array(df[col].to_numpy())

    # Convert regime names to integer codes
    initial_conditions["regime"] = jnp.array(
        df["regime"].map(dict(model.regime_names_to_ids)).to_numpy()
    )

    return initial_conditions


def params_from_pandas(
    *,
    params: PandasUserParams,
    model: Model,
    categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]] | None = None,
) -> dict:
    """Convert a params dict, replacing `pd.Series` values with JAX arrays.

    Broadcast `params` (which may use model, regime, or function-level
    nesting) against the model's params template, then convert any
    `pd.Series` leaf values via `array_from_series`. `MappingLeaf` and
    `SequenceLeaf` values are traversed and any Series inside are
    converted. Other values (scalars, existing arrays) pass through
    unchanged.

    The output is template-shaped (`{regime: {func__param: value}}`).
    Pass it to `model.solve(params=...)` or `model.simulate(params=...)`.

    Args:
        params: User params dict with `pd.Series` in place of array values.
        model: The LCM Model instance.
        categoricals: Extra categorical mappings (level name to grid) for
            derived variables not in the model's state/action grids.

    Returns:
        Dict with template structure, Series replaced by JAX arrays.

    Raises:
        ValueError: If a param name cannot be resolved in the model's
            template.

    """
    resolved = broadcast_to_template(
        params=params,
        template=model.get_params_template(),
        required=False,
    )

    result: dict[str, dict[str, object]] = {}
    for regime_name, regime_params in resolved.items():
        regime = model.regimes[regime_name]
        all_funcs = regime.get_all_functions()
        converted_regime: dict[str, object] = {}
        for func_param, value in regime_params.items():
            parts = tree_path_from_qname(func_param)
            template_func_name = parts[0] if len(parts) > 1 else func_param
            param_name = parts[-1]

            # Runtime grid/shock params are scalar — no AST inspection
            if _is_runtime_grid_param(template_func_name, regime):
                converted_regime[func_param] = _convert_param_value(
                    value=value,
                    func=None,
                    param_name=param_name,
                    func_name=template_func_name,
                    model=model,
                    regime_name=regime_name,
                    categoricals=categoricals,
                )
                continue

            # Resolve per-target template key before function lookup
            resolved_func_name = (
                _resolve_per_target_template_key(
                    func_name=template_func_name, regime=regime
                )
                or template_func_name
            )
            func = all_funcs[resolved_func_name]
            converted_regime[func_param] = _convert_param_value(
                value=value,
                func=func,
                param_name=param_name,
                func_name=resolved_func_name,
                model=model,
                regime_name=regime_name,
                categoricals=categoricals,
            )
        result[regime_name] = converted_regime
    return result


def _convert_param_value(
    *,
    value: object,
    func: Callable | None,
    param_name: str,
    func_name: str,
    model: Model,
    regime_name: str | None,
    categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]] | None = None,
) -> object:
    """Convert a single param value, dispatching on type.

    Args:
        value: The parameter value (Series, MappingLeaf, or passthrough).
        func: The function that uses this parameter (`None` for runtime
            grid params — triggers scalar passthrough).
        param_name: Parameter name in the function.
        func_name: Function name (for `next_*` outcome axis detection).
        model: The LCM Model instance.
        regime_name: Regime name for action grid lookup.
        categoricals: Extra categorical mappings (level name to grid).

    Returns:
        Converted value: JAX array for Series, MappingLeaf with converted
        Series entries, or the original value unchanged.

    """

    def _recurse(inner_value: object) -> object:
        return _convert_param_value(
            value=inner_value,
            func=func,
            param_name=param_name,
            func_name=func_name,
            model=model,
            regime_name=regime_name,
            categoricals=categoricals,
        )

    if isinstance(value, pd.Series):
        return array_from_series(
            sr=value,
            func=func,
            param_name=param_name,
            func_name=func_name,
            model=model,
            regime_name=regime_name,
            categoricals=categoricals,
        )
    if isinstance(value, MappingLeaf):
        return MappingLeaf({k: _recurse(v) for k, v in value.data.items()})
    if isinstance(value, SequenceLeaf):
        return SequenceLeaf(tuple(_recurse(v) for v in value.data))
    return value


def array_from_series(
    *,
    sr: pd.Series,
    func: Callable | None,
    param_name: str,
    func_name: str,
    model: Model,
    regime_name: str | None = None,
    categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]] | None = None,
) -> Array:
    """Convert a pandas Series to a JAX array.

    Inspect `func` to determine indexing dimensions (states, actions,
    period) and scatter the labeled Series into an N-dimensional array.

    The Series MultiIndex must use `"age"` (with actual age values) for the
    age/period dimension, not `"period"`.

    Missing grid points are filled with NaN. Extra ages outside the model's
    `AgeGrid` are silently dropped.

    Args:
        sr: Labeled pandas Series.
        func: The function that uses this array parameter. `None` for
            runtime grid/shock params (triggers scalar passthrough).
        param_name: The array parameter name in `func`.
        func_name: Function name (for `next_*` outcome axis detection).
        model: The LCM Model instance.
        regime_name: Regime for action grid lookup.
        categoricals: Extra categorical mappings (level name to grid) for
            derived variables not in the model's state/action grids.

    Returns:
        JAX array with axes corresponding to the indexing parameters in
        declaration order.

    Raises:
        ValueError: If level names don't match or labels are invalid.

    """
    if func is None:
        return jnp.array(sr.to_numpy(), dtype=float)

    indexing_params = _get_func_indexing_params(func, param_name)

    if not indexing_params and not func_name.startswith("next_"):
        return jnp.array(sr.to_numpy(), dtype=float)

    grids = _resolve_categoricals(
        model=model, regime_name=regime_name, categoricals=categoricals
    )

    # Replace internal "period" with user-facing "age"
    display_params = ["age" if p == "period" else p for p in indexing_params]

    level_mappings = _build_level_mappings_for_param(
        indexing_params=display_params, all_grids=grids, ages=model.ages
    )

    # Append outcome axis for transition functions (next_* functions)
    if func_name.startswith("next_"):
        outcome_mapping = _build_outcome_mapping(
            func_name=func_name, all_grids=grids, model=model
        )
        level_mappings = (*level_mappings, outcome_mapping)

    if "age" in display_params:
        _fail_if_period_level(sr)
        sr = _filter_to_grid_ages(series=sr, ages=model.ages)

    return _scatter_series(series=sr, level_mappings=level_mappings)


def _resolve_categoricals(
    *,
    model: Model,
    regime_name: str | None,
    categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]] | None,
) -> dict[str, DiscreteGrid]:
    """Build combined categorical lookup from model grids and explicit overrides.

    Categoricals can be provided at two levels:

    - Model-level: `{"var": grid}` — applies to all regimes.
    - Regime-level: `{"var": {"regime_a": grid_a, "regime_b": grid_b}}` —
      the grid for `regime_name` is selected.

    Args:
        model: The LCM Model instance.
        regime_name: Regime for action grid discovery and regime-level
            categorical resolution.
        categoricals: Explicit categorical mappings. Values are either a
            `DiscreteGrid` (model-level) or a `Mapping` from regime names
            to `DiscreteGrid` (regime-level).

    Returns:
        Dict mapping variable names to `DiscreteGrid` instances.

    Raises:
        ValueError: If a key in `categoricals` already exists in the model
            grids with different categories.

    """
    grids: dict[str, DiscreteGrid] = {}
    grids.update(_build_discrete_grid_lookup(model.regimes))
    if regime_name is not None:
        grids.update(_build_discrete_action_lookup(model.regimes[regime_name]))
    if categoricals is not None:
        for name, entry in categoricals.items():
            grid = _resolve_categorical_entry(
                name=name, entry=entry, regime_name=regime_name
            )
            if grid is None:
                continue
            if name in grids:
                if grids[name].categories != grid.categories:
                    msg = (
                        f"Explicit categorical '{name}' conflicts with "
                        f"model grid: {grid.categories} vs "
                        f"{grids[name].categories}."
                    )
                    raise ValueError(msg)
            else:
                grids[name] = grid
    return grids


def _resolve_categorical_entry(
    *,
    name: str,
    entry: DiscreteGrid | Mapping[str, DiscreteGrid],
    regime_name: str | None,
) -> DiscreteGrid | None:
    """Resolve a single categoricals entry to a grid.

    Args:
        name: Variable name.
        entry: Either a `DiscreteGrid` (model-level) or a `Mapping` from
            regime names to `DiscreteGrid` (regime-level).
        regime_name: Current regime name for regime-level resolution.

    Returns:
        The resolved `DiscreteGrid`, or `None` if the regime-level entry
        doesn't have a grid for the current regime.

    """
    if isinstance(entry, DiscreteGrid):
        return entry
    if isinstance(entry, Mapping):
        if regime_name is None:
            msg = (
                f"Regime-level categorical '{name}' requires a resolved "
                f"regime_name, but regime_name is None. Use a fully "
                f"qualified 3-part param_path."
            )
            raise ValueError(msg)
        return entry.get(regime_name)
    msg = (
        f"Categorical '{name}' must be a DiscreteGrid or a Mapping "
        f"from regime names to DiscreteGrid. Got {type(entry).__name__}."
    )
    raise TypeError(msg)


def _resolve_per_target_template_key(
    *,
    func_name: str,
    regime: Regime,
) -> str | None:
    """Translate a per-target template key to `get_all_functions()` format.

    Template uses `to_{target}_{next_state}` (e.g., `to_working_next_health`).
    `get_all_functions()` uses `next_{state}__{target}` (e.g.,
    `next_health__working`). Return the translated key if `func_name`
    matches the template pattern, or `None` if it doesn't.

    Args:
        func_name: The function name from the template
            (e.g., `to_working_next_health`).
        regime: The regime to check for per-target transitions.

    Returns:
        The `get_all_functions()` key, or `None` if not a per-target key.

    """
    if not func_name.startswith("to_"):
        return None

    # Per-target template keys have the form "to_{target}_{next_state}".
    # Find matching per-target transitions in the regime.
    for state_name, raw in regime.state_transitions.items():
        if isinstance(raw, Mapping):
            for target_name in raw:
                target = str(target_name)
                next_state = f"next_{state_name}"
                template_key = f"to_{target}_{next_state}"
                if template_key == func_name:
                    return qname_from_tree_path((next_state, target))

    return None


def _is_runtime_grid_param(func_name: str, regime: Regime) -> bool:
    """Check if a template function key refers to a runtime grid param."""
    if func_name not in regime.states:
        return False
    grid = regime.states[func_name]
    return (isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime) or (
        isinstance(grid, _ShockGrid) and bool(grid.params_to_pass_at_runtime)
    )


def _fail_if_period_level(sr: pd.Series) -> None:
    """Raise if the Series has a 'period' level instead of 'age'."""
    if "period" in sr.index.names:
        msg = (
            "Use 'age' (with actual age values) as the MultiIndex level name "
            "instead of 'period'."
        )
        raise ValueError(msg)


def _filter_to_grid_ages(
    *,
    series: pd.Series,
    ages: AgeGrid,
) -> pd.Series:
    """Keep only rows whose `"age"` level value is on the `AgeGrid`.

    Args:
        series: Series with an `"age"` MultiIndex level.
        ages: The model's `AgeGrid`.

    Returns:
        Filtered Series containing only rows with valid grid ages.

    """
    grid_ages = {float(v) for v in ages.exact_values}
    age_values = series.index.get_level_values("age").astype(float)
    return series.loc[age_values.isin(grid_ages)]


@dataclass(frozen=True)
class _LevelMapping:
    """Specification for mapping one MultiIndex level to array indices."""

    name: str
    """Level name in the MultiIndex (e.g., `"age"`, `"health"`, `"next_health"`)."""

    size: int
    """Number of positions along this axis."""

    label_to_index: Callable[[object], int]
    """Map a single label value to its integer index."""

    valid_labels: tuple[str, ...] = ()
    """Valid label names, for error messages. Empty for age levels."""


def _age_level_mapping(ages: AgeGrid) -> _LevelMapping:
    """Create a `_LevelMapping` for the age dimension."""
    return _LevelMapping(
        name="age",
        size=ages.n_periods,
        label_to_index=ages.age_to_period,  # ty: ignore[invalid-argument-type]
    )


def _grid_level_mapping(*, name: str, grid: DiscreteGrid) -> _LevelMapping:
    """Create a `_LevelMapping` for a categorical dimension.

    Args:
        name: Level name in the MultiIndex.
        grid: The `DiscreteGrid` defining valid categories.

    Returns:
        `_LevelMapping` mapping category labels to integer codes.

    """
    label_to_code = dict(zip(grid.categories, grid.codes, strict=True))
    return _LevelMapping(
        name=name,
        size=len(grid.categories),
        label_to_index=label_to_code.__getitem__,
        valid_labels=grid.categories,
    )


def _build_level_mappings_for_param(
    *,
    indexing_params: list[str],
    all_grids: dict[str, DiscreteGrid],
    ages: AgeGrid,
) -> tuple[_LevelMapping, ...]:
    """Build level mappings for `array_from_series` from indexing params.

    Args:
        indexing_params: Parameter names in output axis order, with
            `"period"` already replaced by `"age"`.
        all_grids: Categorical grid lookup.
        ages: The model's `AgeGrid`.

    Returns:
        Tuple of `_LevelMapping` instances.

    """
    mappings: list[_LevelMapping] = []
    for param in indexing_params:
        if param == "age":
            mappings.append(_age_level_mapping(ages))
        elif param in all_grids:
            mappings.append(_grid_level_mapping(name=param, grid=all_grids[param]))
        else:
            msg = (
                f"Unrecognised indexing parameter '{param}'. Expected 'age', "
                f"or a discrete grid name ({sorted(all_grids)})."
            )
            raise ValueError(msg)
    return tuple(mappings)


def _build_outcome_mapping(
    *,
    func_name: str,
    all_grids: dict[str, DiscreteGrid],
    model: Model,
) -> _LevelMapping:
    """Build a `_LevelMapping` for the outcome axis of a `next_*` function.

    For state transitions (e.g. `"next_partner"`), look up the state grid.
    For regime transitions (`"next_regime"`), use `model.regime_names_to_ids`.

    Args:
        func_name: Function name starting with `"next_"`.
        all_grids: Categorical grid lookup.
        model: The LCM Model instance.

    Returns:
        `_LevelMapping` for the outcome (last) axis.

    """
    if func_name == "next_regime":
        regime_ids = dict(model.regime_names_to_ids)
        return _LevelMapping(
            name="next_regime",
            size=len(regime_ids),
            label_to_index=regime_ids.__getitem__,  # ty: ignore[invalid-argument-type]
            valid_labels=tuple(regime_ids),
        )

    path = tree_path_from_qname(func_name)
    state_name = path[0].removeprefix("next_")
    return _grid_level_mapping(name=f"next_{state_name}", grid=all_grids[state_name])


def _scatter_series(
    *,
    series: pd.Series,
    level_mappings: tuple[_LevelMapping, ...],
    fill_value: float = np.nan,
) -> Array:
    """Scatter a MultiIndex Series into an N-dimensional JAX array.

    Each `_LevelMapping` defines one axis: its size, and how to map labels from
    the corresponding MultiIndex level to integer indices. Positions not covered
    by the Series are filled with `fill_value`.

    Args:
        series: Series with a named MultiIndex.
        level_mappings: One mapping per axis, in output axis order.
        fill_value: Value for positions not present in the Series.

    Returns:
        JAX array with shape `[m.size for m in level_mappings]`.

    """
    expected_levels = [m.name for m in level_mappings]
    series = _validate_and_reorder_levels(
        series=series, expected_levels=expected_levels
    )

    shape = [m.size for m in level_mappings]

    if len(series) == 0:
        return jnp.full(shape, fill_value)

    index_arrays = [
        _map_level(
            mapping=mapping, level_values=series.index.get_level_values(mapping.name)
        )
        for mapping in level_mappings
    ]

    result = np.full(shape, fill_value)
    result[tuple(index_arrays)] = series.to_numpy()
    return jnp.array(result)


def _map_level(*, mapping: _LevelMapping, level_values: pd.Index) -> np.ndarray:
    """Map label values from one MultiIndex level to integer indices.

    Args:
        mapping: The `_LevelMapping` for this level.
        level_values: Index values from the Series MultiIndex level.

    Returns:
        NumPy array of integer indices.

    Raises:
        ValueError: If any label is not valid for the mapping.

    """
    try:
        return np.array([mapping.label_to_index(v) for v in level_values])
    except ValueError:
        # Age levels: age_to_period raises ValueError with a good message
        raise
    except KeyError:
        # Categorical levels: collect all invalid labels
        invalid = sorted(set(level_values) - set(mapping.valid_labels))
        msg = (
            f"Invalid labels for level '{mapping.name}': {invalid}. "
            f"Valid labels: {sorted(mapping.valid_labels)}."
        )
        raise ValueError(msg) from None


def _validate_and_reorder_levels(
    *,
    series: pd.Series,
    expected_levels: list[str],
) -> pd.Series:
    """Validate MultiIndex level names and reorder to match expected order."""
    actual_names = list(series.index.names)

    if "period" in actual_names:
        msg = (
            "Use 'age' (with actual age values) as the MultiIndex level name "
            "instead of 'period'."
        )
        raise ValueError(msg)

    if len(actual_names) != len(set(actual_names)):
        msg = (
            f"Series MultiIndex has duplicate level names: {actual_names}. "
            f"All level names must be unique."
        )
        raise ValueError(msg)

    if set(actual_names) != set(expected_levels):
        msg = (
            f"Series MultiIndex level names must be {expected_levels}, "
            f"but got {actual_names}."
        )
        raise ValueError(msg)

    if actual_names != expected_levels:
        series = series.reorder_levels(expected_levels)  # ty: ignore[invalid-argument-type]

    return series


def _validate_state_columns(
    *,
    state_columns: set[str],
    regimes: Mapping[str, Regime],
    initial_regimes: pd.Series,
) -> None:
    """Validate that DataFrame columns match model states."""
    all_states = _collect_all_state_names(
        regimes=regimes, initial_regimes=initial_regimes
    )

    unknown = state_columns - all_states
    if unknown:
        msg = (
            f"Unknown columns not matching any model state: {sorted(unknown)}. "
            f"Expected states: {sorted(all_states)}."
        )
        raise ValueError(msg)

    missing = all_states - state_columns
    if missing:
        msg = (
            f"Missing required state columns: {sorted(missing)}. "
            f"All non-shock states must be provided."
        )
        raise ValueError(msg)


def _collect_all_state_names(
    *,
    regimes: Mapping[str, Regime],
    initial_regimes: pd.Series,
) -> set[str]:
    """Collect all non-shock state names from regimes present in initial_regimes."""
    state_names: set[str] = set()
    for regime_name in initial_regimes.unique():
        regime = regimes[regime_name]
        for name, grid in regime.states.items():
            if not isinstance(grid, _ShockGrid):
                state_names.add(name)
    # Always include age
    state_names.add("age")
    return state_names


def _build_discrete_grid_lookup(
    regimes: Mapping[str, Regime],
) -> dict[str, DiscreteGrid]:
    """Collect all DiscreteGrid instances across regimes, verifying consistency.

    Args:
        regimes: Mapping of regime names to Regime instances.

    Returns:
        dict mapping state name to DiscreteGrid.

    Raises:
        ValueError: If two regimes define the same state with different categories.

    """
    lookup: dict[str, DiscreteGrid] = {}
    for regime_name, regime in regimes.items():
        for state_name, grid in regime.states.items():
            if isinstance(grid, DiscreteGrid):
                if state_name in lookup:
                    if lookup[state_name].categories != grid.categories:
                        msg = (
                            f"Inconsistent DiscreteGrid for state '{state_name}': "
                            f"regime '{regime_name}' has categories "
                            f"{grid.categories}, but a previous regime has "
                            f"{lookup[state_name].categories}."
                        )
                        raise ValueError(msg)
                else:
                    lookup[state_name] = grid
    return lookup


def _build_discrete_action_lookup(regime: Regime) -> dict[str, DiscreteGrid]:
    """Collect DiscreteGrid instances from a regime's actions.

    Args:
        regime: The Regime instance.

    Returns:
        dict mapping action name to DiscreteGrid.

    """
    return {
        name: grid
        for name, grid in regime.actions.items()
        if isinstance(grid, DiscreteGrid)
    }
