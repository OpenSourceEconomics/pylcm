"""Utilities for converting between pandas and LCM data structures."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
from dags.tree import qname_from_tree_path, tree_path_from_qname
from jax import Array

from lcm.ages import PSEUDO_STATE_NAMES, AgeGrid
from lcm.grids import DiscreteGrid, IrregSpacedGrid
from lcm.params import MappingLeaf
from lcm.params.sequence_leaf import SequenceLeaf
from lcm.regime import Regime
from lcm.shocks import _ShockGrid
from lcm.simulation.initial_conditions import MISSING_CAT_CODE
from lcm.typing import InternalParams, RegimeName, RegimeNamesToIds
from lcm.utils.error_handling import (
    _get_func_indexing_params,
)


def has_series(params: Mapping) -> bool:
    """Check if any leaf value in a params mapping is a pd.Series."""
    for value in params.values():
        if isinstance(value, pd.Series):
            return True
        if isinstance(value, Mapping) and has_series(value):
            return True
        if isinstance(value, (MappingLeaf, SequenceLeaf)):
            items = (
                value.data.values() if isinstance(value, MappingLeaf) else value.data
            )
            if any(isinstance(v, pd.Series) for v in items):
                return True
    return False


def initial_conditions_from_dataframe(  # noqa: C901
    *,
    df: pd.DataFrame,
    regimes: Mapping[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
) -> dict[str, Array]:
    """Convert a DataFrame of initial conditions to LCM initial conditions format.

    Args:
        df: DataFrame with columns for states and a "regime" column.
        regimes: Mapping of regime names to user Regime instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.

    Returns:
        Dict mapping state names (plus `"regime"`) to JAX arrays. The
        `"regime"` entry contains integer codes derived from the `"regime"`
        column via `regime_names_to_ids`.

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
    valid_regimes = set(regime_names_to_ids.keys())
    invalid_regimes = set(df["regime"]) - valid_regimes
    if invalid_regimes:
        msg = (
            f"Invalid regime names in 'regime' column: {sorted(invalid_regimes)}. "
            f"Valid regimes: {sorted(valid_regimes)}."
        )
        raise ValueError(msg)

    state_columns = {col for col in df.columns if col != "regime"}
    _validate_state_columns(
        state_columns=state_columns,
        regimes=regimes,
        initial_regimes=df["regime"].tolist(),
    )

    n_subjects = len(df)
    state_cols = [col for col in df.columns if col != "regime"]

    # Pre-allocate result arrays (NaN default surfaces bugs for missing states)
    result_arrays: dict[str, np.ndarray] = {
        col: np.full(n_subjects, np.nan) for col in state_cols
    }
    discrete_state_names: set[str] = set()

    # Process per regime group (vectorised .map() within each group)
    for regime_name, group in df.groupby("regime"):
        regime = regimes[str(regime_name)]
        idx = group.index
        discrete_grids = {
            name: grid
            for name, grid in regime.states.items()
            if isinstance(grid, DiscreteGrid)
        }
        discrete_state_names |= discrete_grids.keys()

        regime_state_names = set(regime.states.keys()) | PSEUDO_STATE_NAMES

        for col in state_cols:
            if col not in regime_state_names:
                continue

            values = group[col]
            if hasattr(values, "cat"):
                values = values.astype(str)

            if col in discrete_grids:
                _map_discrete_labels(
                    values=values,
                    grid=discrete_grids[col],
                    result_array=result_arrays[col],
                    idx=idx,
                    col=col,
                    regime_name=str(regime_name),
                )
            else:
                result_arrays[col][idx] = values.to_numpy(dtype=float)

    # Replace remaining NaN in discrete columns with an explicit int sentinel
    # before casting to int32. This avoids platform-undefined NaN→int behavior
    # and the associated RuntimeWarning.
    for col in discrete_state_names:
        if col in result_arrays:
            nan_mask = np.isnan(result_arrays[col])
            result_arrays[col][nan_mask] = MISSING_CAT_CODE

    initial_conditions: dict[str, Array] = {
        col: jnp.array(arr, dtype=jnp.int32)
        if col in discrete_state_names
        else jnp.array(arr)
        for col, arr in result_arrays.items()
    }
    initial_conditions["regime"] = jnp.array(
        df["regime"].map(dict(regime_names_to_ids)).to_numpy()
    )

    return initial_conditions


def _map_discrete_labels(
    *,
    values: pd.Series,
    grid: DiscreteGrid,
    result_array: np.ndarray,
    idx: pd.Index,
    col: str,
    regime_name: RegimeName,
) -> None:
    """Map string labels to integer codes for a discrete state column in-place."""
    label_to_code = dict(zip(grid.categories, grid.codes, strict=True))
    mapped = values.map(label_to_code)
    unmapped = mapped.isna() & values.notna()
    if unmapped.any():
        bad = set(values[unmapped])
        msg = (
            f"Invalid labels for state '{col}' in regime "
            f"'{regime_name}': {sorted(bad)}. "
            f"Valid: {list(grid.categories)}."
        )
        raise ValueError(msg)
    result_array[idx] = mapped.to_numpy()


def convert_series_in_params(
    *,
    internal_params: Mapping[RegimeName, Mapping[str, object]],
    ages: AgeGrid,
    regimes: Mapping[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
) -> InternalParams:
    """Convert pd.Series leaves in already-broadcast internal params to JAX arrays.

    Iterate over the template-shaped `internal_params` (produced by
    `process_params`) and convert any `pd.Series` leaf values via
    `array_from_series`. `MappingLeaf` and `SequenceLeaf` values are
    traversed and any Series inside are converted. Other values (scalars,
    existing arrays) pass through unchanged.

    Each regime's `derived_categoricals` field is used to resolve index
    levels that correspond to DAG function outputs (not states/actions).

    Args:
        internal_params: Already-broadcast params in template shape
            (`{regime: {func__param: value}}`).
        ages: Age grid for the model.
        regimes: Mapping of regime names to user Regime instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.

    Returns:
        Immutable mapping with the same structure, Series replaced by JAX
        arrays.

    """
    result: dict[RegimeName, dict[str, object]] = {}
    for regime_name, regime_params in internal_params.items():
        regime = regimes[regime_name]
        all_funcs = regime.get_all_functions()
        converted_regime: dict[str, object] = {}
        for func_param, value in regime_params.items():
            parts = tree_path_from_qname(func_param)
            template_func_name = parts[0] if len(parts) > 1 else func_param
            param_name = parts[-1]

            # Runtime grid/shock params are scalar — no AST inspection
            if _is_runtime_grid_param(func_name=template_func_name, regime=regime):
                converted_regime[func_param] = _convert_param_value(
                    value=value,
                    func=None,
                    param_name=param_name,
                    func_name=template_func_name,
                    ages=ages,
                    regimes=regimes,
                    regime_names_to_ids=regime_names_to_ids,
                    regime_name=regime_name,
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
                ages=ages,
                regimes=regimes,
                regime_names_to_ids=regime_names_to_ids,
                regime_name=regime_name,
            )
        result[regime_name] = converted_regime
    return cast(
        "InternalParams",
        MappingProxyType({k: MappingProxyType(v) for k, v in result.items()}),
    )


def _convert_param_value(
    *,
    value: object,
    func: Callable | None,
    param_name: str,
    func_name: str,
    ages: AgeGrid,
    regimes: Mapping[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
    regime_name: RegimeName | None,
) -> object:
    """Convert a single param value, dispatching on type.

    Args:
        value: The parameter value (Series, MappingLeaf, or passthrough).
        func: The function that uses this parameter (`None` for runtime
            grid params — triggers scalar passthrough).
        param_name: Parameter name in the function.
        func_name: Function name (for `next_*` outcome axis detection).
        ages: Age grid for the model.
        regimes: Mapping of regime names to user Regime instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        regime_name: Regime name for action grid lookup.

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
            ages=ages,
            regimes=regimes,
            regime_names_to_ids=regime_names_to_ids,
            regime_name=regime_name,
        )

    if isinstance(value, pd.Series):
        return array_from_series(
            sr=value,
            func=func,
            param_name=param_name,
            func_name=func_name,
            ages=ages,
            regimes=regimes,
            regime_names_to_ids=regime_names_to_ids,
            regime_name=regime_name,
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
    ages: AgeGrid,
    regimes: Mapping[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
    regime_name: RegimeName | None = None,
) -> Array:
    """Convert a pandas Series to a JAX array.

    Inspect `func` to determine indexing dimensions (states, actions,
    period) and scatter the labeled Series into an N-dimensional array.

    The Series MultiIndex must use `"age"` (with actual age values) for the
    age/period dimension, not `"period"`.

    Missing grid points are filled with NaN. Extra ages outside the model's
    `AgeGrid` are silently dropped.

    Derived categoricals are read from `regimes[regime_name].derived_categoricals`
    when `regime_name` is not None.

    Args:
        sr: Labeled pandas Series.
        func: The function that uses this array parameter. `None` for
            runtime grid/shock params (triggers scalar passthrough).
        param_name: The array parameter name in `func`.
        func_name: Function name (for `next_*` outcome axis detection).
        ages: Age grid for the model.
        regimes: Mapping of regime names to user Regime instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        regime_name: Regime for grid and derived categorical lookup.

    Returns:
        JAX array with axes corresponding to the indexing parameters in
        declaration order.

    Raises:
        ValueError: If level names don't match or labels are invalid.

    """
    if func is None:
        return jnp.array(sr.to_numpy(), dtype=float)

    indexing_params = _get_func_indexing_params(func=func, array_param_name=param_name)

    if not indexing_params:
        return jnp.array(sr.to_numpy(), dtype=float)

    grids = _resolve_categoricals(
        regimes=regimes,
        regime_name=regime_name,
    )

    # Replace internal "period" with user-facing "age"
    display_params = ["age" if p == "period" else p for p in indexing_params]

    level_mappings = _build_level_mappings_for_param(
        indexing_params=display_params, grids=grids, ages=ages
    )

    # Append outcome axis for transition probability arrays (next_* functions
    # where the Series has a next_* level in its MultiIndex)
    if func_name.startswith("next_") and isinstance(sr.index, pd.MultiIndex):
        next_levels = [
            n for n in sr.index.names if isinstance(n, str) and n.startswith("next_")
        ]
        if next_levels:
            outcome_mapping = _build_outcome_mapping(
                func_name=func_name,
                grids=grids,
                regimes=regimes,
                regime_names_to_ids=regime_names_to_ids,
            )
            level_mappings = (*level_mappings, outcome_mapping)

    if "age" in display_params:
        _fail_if_period_level(sr)
        sr = _filter_to_grid_ages(series=sr, ages=ages)

    return _scatter_series(series=sr, level_mappings=level_mappings)


def _resolve_categoricals(
    *,
    regimes: Mapping[RegimeName, Regime],
    regime_name: RegimeName | None,
) -> dict[str, DiscreteGrid]:
    """Build combined categorical lookup from model grids and regime overrides.

    Collect discrete state and action grids, then merge in the regime's
    `derived_categoricals` (grids for DAG function outputs).

    Args:
        regimes: Mapping of regime names to user Regime instances.
        regime_name: Regime for grid discovery. When `None`, grids from
            all regimes are merged.

    Returns:
        Dict mapping variable names to `DiscreteGrid` instances.

    Raises:
        ValueError: If a derived categorical conflicts with a model grid.

    """
    grids: dict[str, DiscreteGrid] = {}
    if regime_name is not None:
        regime = regimes[regime_name]
        for grids_mapping in (regime.states, regime.actions):
            grids.update(
                {n: g for n, g in grids_mapping.items() if isinstance(g, DiscreteGrid)}
            )
        for name, grid in regime.derived_categoricals.items():
            if name in grids and grids[name].categories != grid.categories:
                msg = (
                    f"Derived categorical '{name}' conflicts with "
                    f"model grid: {grid.categories} vs "
                    f"{grids[name].categories}."
                )
                raise ValueError(msg)
            grids[name] = grid
    else:
        grids.update(_build_discrete_grid_lookup(regimes))
        for regime in regimes.values():
            for name, grid in regime.derived_categoricals.items():
                if name in grids and grids[name].categories != grid.categories:
                    msg = (
                        f"Derived categorical '{name}' conflicts with "
                        f"model grid: {grid.categories} vs "
                        f"{grids[name].categories}."
                    )
                    raise ValueError(msg)
                grids[name] = grid
    return grids


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


def _is_runtime_grid_param(*, func_name: str, regime: Regime) -> bool:
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

    get_code_from_label: Callable[[str], int]
    """Return the integer code for a label."""

    valid_labels: tuple[str, ...] = ()
    """Valid label names, for error messages. Empty for age levels."""


def _age_level_mapping(ages: AgeGrid) -> _LevelMapping:
    """Create a `_LevelMapping` for the age dimension."""
    return _LevelMapping(
        name="age",
        size=ages.n_periods,
        get_code_from_label=ages.age_to_period,  # ty: ignore[invalid-argument-type]
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
        get_code_from_label=label_to_code.__getitem__,
        valid_labels=grid.categories,
    )


def _build_level_mappings_for_param(
    *,
    indexing_params: list[str],
    grids: dict[str, DiscreteGrid],
    ages: AgeGrid,
) -> tuple[_LevelMapping, ...]:
    """Build level mappings for `array_from_series` from indexing params.

    Args:
        indexing_params: Parameter names in output axis order, with
            `"period"` already replaced by `"age"`.
        grids: Categorical grid lookup.
        ages: The model's `AgeGrid`.

    Returns:
        Tuple of `_LevelMapping` instances.

    """
    mappings: list[_LevelMapping] = []
    for param in indexing_params:
        if param == "age":
            mappings.append(_age_level_mapping(ages))
        elif param in grids:
            mappings.append(_grid_level_mapping(name=param, grid=grids[param]))
        else:
            msg = (
                f"Unrecognised indexing parameter '{param}'. Expected 'age' "
                f"or a discrete grid name ({sorted(grids)}). If "
                f"'{param}' is a DAG function output, add "
                f'derived_categoricals={{"{param}": DiscreteGrid(...)}} '
                f"to the Regime or Model constructor."
            )
            raise ValueError(msg)
    return tuple(mappings)


def _build_outcome_mapping(
    *,
    func_name: str,
    grids: dict[str, DiscreteGrid],
    regimes: Mapping[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
) -> _LevelMapping:
    """Build a `_LevelMapping` for the outcome axis of a `next_*` function.

    For state transitions (e.g. `"next_partner"`), look up the state grid.
    For per-target transitions (e.g. `"next_health__post65"`), use the target
    regime's grid for the outcome axis.
    For regime transitions (`"next_regime"`), use `regime_names_to_ids`.

    Args:
        func_name: Function name starting with `"next_"`.
        grids: Categorical grid lookup.
        regimes: Mapping of regime names to user Regime instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.

    Returns:
        `_LevelMapping` for the outcome (last) axis.

    """
    if func_name == "next_regime":
        return _LevelMapping(
            name="next_regime",
            size=len(regime_names_to_ids),
            get_code_from_label=regime_names_to_ids.__getitem__,
            valid_labels=tuple(regime_names_to_ids),
        )

    path = tree_path_from_qname(func_name)
    state_name = path[0].removeprefix("next_")

    # Per-target transitions (e.g. "next_health__post65") must use the TARGET
    # regime's grid for the outcome axis, not the source regime's grid.
    if len(path) > 1:
        target_regime_name = path[1]
        target_regime = regimes.get(target_regime_name)
        if target_regime is not None and state_name in target_regime.states:
            target_grid = target_regime.states[state_name]
            if isinstance(target_grid, DiscreteGrid):
                return _grid_level_mapping(name=f"next_{state_name}", grid=target_grid)

    return _grid_level_mapping(name=f"next_{state_name}", grid=grids[state_name])


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
    # Categorical levels must use string labels matching grid category names.
    # Reject integer labels early with a clear message instead of a cryptic KeyError.
    if mapping.valid_labels and any(not isinstance(v, str) for v in level_values):
        non_str_types = sorted(
            {type(v).__name__ for v in level_values if not isinstance(v, str)}
        )
        msg = (
            f"Series index level '{mapping.name}' uses non-string labels "
            f"(types: {non_str_types}) but the DiscreteGrid expects string "
            f"category names. Use string labels matching: "
            f"{sorted(mapping.valid_labels)}."
        )
        raise ValueError(msg)

    try:
        return np.array([mapping.get_code_from_label(v) for v in level_values])
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
    regimes: Mapping[RegimeName, Regime],
    initial_regimes: list[RegimeName],
) -> None:
    """Validate that DataFrame columns match model states."""
    expected = _collect_state_names(regimes=regimes, initial_regimes=initial_regimes)

    unknown = state_columns - expected
    if unknown:
        msg = (
            f"Unknown columns not matching any state of an initial regime: "
            f"{sorted(unknown)}. "
            f"Expected states: {sorted(expected)}."
        )
        raise ValueError(msg)

    missing = expected - state_columns
    if missing:
        required_by: dict[str, list[str]] = {name: [] for name in missing}
        for regime_name in set(initial_regimes):
            for name in regimes[regime_name].states:
                if name in required_by:
                    required_by[name].append(regime_name)
        details = ", ".join(
            _format_missing_state_detail(name=name, required_by=required_by[name])
            for name in sorted(missing)
        )
        msg = f"Missing required state columns: {details}."
        raise ValueError(msg)


def _format_missing_state_detail(*, name: str, required_by: list[str]) -> str:
    if name in PSEUDO_STATE_NAMES:
        return f"'{name}' (required for every subject)"
    if required_by:
        return f"'{name}' (required by {sorted(required_by)})"
    return f"'{name}' (required by an initial regime)"


def _collect_state_names(
    *,
    regimes: Mapping[RegimeName, Regime],
    initial_regimes: list[RegimeName],
) -> set[str]:
    """Collect all state names (including shock grids) from initial regimes.

    Returns:
        Set of all state names from the initial regimes, plus the pseudo-state
        names from `PSEUDO_STATE_NAMES` (always required).

    """
    names: set[str] = set(PSEUDO_STATE_NAMES)
    for regime_name in set(initial_regimes):
        names.update(regimes[regime_name].states.keys())
    return names


def _build_discrete_grid_lookup(
    regimes: Mapping[RegimeName, Regime],
) -> dict[str, DiscreteGrid]:
    """Collect all DiscreteGrid instances from states and actions across regimes.

    Args:
        regimes: Mapping of regime names to Regime instances.

    Returns:
        Dict mapping variable name to DiscreteGrid.

    Raises:
        ValueError: If two regimes define the same variable with different categories.

    """
    lookup: dict[str, DiscreteGrid] = {}
    for regime_name, regime in regimes.items():
        for grids_mapping in (regime.states, regime.actions):
            for var_name, grid in grids_mapping.items():
                if isinstance(grid, DiscreteGrid):
                    if var_name in lookup:
                        if lookup[var_name].categories != grid.categories:
                            msg = (
                                f"Inconsistent DiscreteGrid for '{var_name}': "
                                f"regime '{regime_name}' has categories "
                                f"{grid.categories}, but a previous regime has "
                                f"{lookup[var_name].categories}."
                            )
                            raise ValueError(msg)
                    else:
                        lookup[var_name] = grid
    return lookup
