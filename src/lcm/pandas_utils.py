"""Utilities for converting between pandas and LCM data structures."""

import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from lcm.grids import DiscreteGrid
from lcm.regime import MarkovTransition, Regime

if TYPE_CHECKING:
    from lcm.model import Model


def initial_states_from_dataframe(
    df: pd.DataFrame,
    *,
    model: Model,
) -> tuple[dict[str, Array], list[str]]:
    """Convert a DataFrame of initial conditions to LCM initial states format.

    Args:
        df: DataFrame with columns for states and a "regime" column.
        model: The LCM Model instance.

    Returns:
        Tuple of (initial_states dict mapping state names to JAX arrays,
        initial_regimes list of regime name strings).

    Raises:
        ValueError: If the "regime" column is missing, contains invalid regime names,
            or categorical columns contain invalid labels.

    """
    if "regime" not in df.columns:
        msg = "DataFrame must contain a 'regime' column."
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

    initial_regimes = df["regime"].tolist()

    discrete_lookup = _build_discrete_grid_lookup(model.regimes)

    initial_states: dict[str, Array] = {}
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

            initial_states[col] = jnp.array([label_to_code[v] for v in values])
        else:
            initial_states[col] = jnp.array(df[col].values)

    return initial_states, initial_regimes


def transition_probs_from_series(
    series: pd.Series,
    *,
    model: Model,
    regime_name: str,
    state_name: str,
) -> Array:
    """Convert a labeled pandas Series to a transition probability array.

    Build a transition probability array from a Series with a named MultiIndex,
    eliminating manual array construction with opaque axis ordering.

    Args:
        series: Series with a named MultiIndex. Level names must match the
            indexing parameters of the transition function plus
            `"next_{state_name}"` for the outcome level.
        model: The LCM Model instance.
        regime_name: Name of the regime containing the state transition.
        state_name: Name of the state with a `MarkovTransition`.

    Returns:
        JAX array with axes corresponding to the indexing parameters in
        declaration order, followed by the outcome axis.

    Raises:
        TypeError: If the state transition is not a `MarkovTransition`.
        ValueError: If level names don't match or labels are invalid.

    """
    regime = model.regimes[regime_name]

    # 1. Look up the MarkovTransition
    raw_transition = regime.state_transitions[state_name]
    if not isinstance(raw_transition, MarkovTransition):
        msg = (
            f"State '{state_name}' in regime '{regime_name}' is not a "
            f"MarkovTransition. Got {type(raw_transition).__name__}."
        )
        raise TypeError(msg)

    func = raw_transition.func

    # 2. Extract indexing params and identify axes
    indexing_params = _get_indexing_params(func)
    outcome_level = f"next_{state_name}"
    expected_levels = [*indexing_params, outcome_level]

    # 3. Validate and reorder MultiIndex level names
    series = _validate_and_reorder_levels(series, expected_levels)

    # 4. Build grid lookup for label→code mapping
    discrete_lookup = _build_discrete_grid_lookup(model.regimes)
    action_lookup = _build_discrete_action_lookup(regime)
    all_grids = {**discrete_lookup, **action_lookup}

    # 5. Compute shape and map labels to integer codes
    shape = _compute_shape(
        expected_levels,
        outcome_level,
        state_name,
        all_grids,
        model,
        series,
    )
    index_arrays = _map_labels_to_codes(
        expected_levels,
        outcome_level,
        state_name,
        all_grids,
        series,
    )

    # 6. Place values into the n-dim array
    result = np.empty(shape, dtype=float)
    result[tuple(index_arrays)] = series.to_numpy()

    return jnp.array(result)


def validate_transition_probs(
    array: Array,
    *,
    model: Model,
    regime_name: str,
    state_name: str,
) -> None:
    """Validate a transition probability array for shape, values, and row sums.

    Args:
        array: The transition probability array to validate.
        model: The LCM Model instance.
        regime_name: Name of the regime containing the state transition.
        state_name: Name of the state with a `MarkovTransition`.

    Raises:
        TypeError: If the state transition is not a `MarkovTransition`.
        ValueError: If the shape is wrong, values are outside [0, 1], or rows
            don't sum to 1.

    """
    regime = model.regimes[regime_name]

    raw_transition = regime.state_transitions[state_name]
    if not isinstance(raw_transition, MarkovTransition):
        msg = (
            f"State '{state_name}' in regime '{regime_name}' is not a "
            f"MarkovTransition. Got {type(raw_transition).__name__}."
        )
        raise TypeError(msg)

    func = raw_transition.func
    indexing_params = _get_indexing_params(func)

    # Build expected shape
    discrete_lookup = _build_discrete_grid_lookup(model.regimes)
    action_lookup = _build_discrete_action_lookup(regime)
    all_grids = {**discrete_lookup, **action_lookup}

    expected_shape: list[int] = []
    for param_name in indexing_params:
        if param_name == "period":
            expected_shape.append(model.n_periods)
        elif param_name in all_grids:
            expected_shape.append(len(all_grids[param_name].categories))
        else:
            msg = (
                f"Cannot determine expected size for parameter '{param_name}'. "
                f"It is not 'period' and not a DiscreteGrid state or action."
            )
            raise ValueError(msg)

    # Add outcome axis
    state_grid = all_grids[state_name]
    expected_shape.append(len(state_grid.categories))

    expected_tuple = tuple(expected_shape)
    if array.shape != expected_tuple:
        msg = f"Expected shape {expected_tuple} but got {array.shape}."
        raise ValueError(msg)

    # Check values in [0, 1]
    if jnp.any(array < 0) or jnp.any(array > 1):
        msg = "All values must be in [0, 1]."
        raise ValueError(msg)

    # Check rows sum to 1
    row_sums = jnp.sum(array, axis=-1)
    if not jnp.allclose(row_sums, 1.0, atol=1e-6):
        msg = "Rows must sum to 1 along the last axis."
        raise ValueError(msg)


def _validate_and_reorder_levels(
    series: pd.Series,
    expected_levels: list[str],
) -> pd.Series:
    """Validate MultiIndex level names and reorder to match expected order."""
    if series.index.names is None or set(series.index.names) != set(expected_levels):
        msg = (
            f"Series MultiIndex level names must be {expected_levels}, "
            f"but got {list(series.index.names)}."
        )
        raise ValueError(msg)

    if list(series.index.names) != expected_levels:
        series = series.reorder_levels(expected_levels)  # ty: ignore[invalid-argument-type]

    return series


def _compute_shape(
    expected_levels: list[str],
    outcome_level: str,
    state_name: str,
    all_grids: dict[str, DiscreteGrid],
    model: Model,
    series: pd.Series,
) -> list[int]:
    """Compute the expected array shape from grid sizes."""
    shape: list[int] = []
    for level_name in expected_levels:
        if level_name == "period":
            shape.append(model.n_periods)
        elif level_name == outcome_level:
            shape.append(len(all_grids[state_name].categories))
        elif level_name in all_grids:
            shape.append(len(all_grids[level_name].categories))
        else:
            shape.append(len(series.index.get_level_values(level_name).unique()))
    return shape


def _map_labels_to_codes(
    expected_levels: list[str],
    outcome_level: str,
    state_name: str,
    all_grids: dict[str, DiscreteGrid],
    series: pd.Series,
) -> list[np.ndarray]:
    """Map string labels in MultiIndex levels to integer codes."""
    index_arrays: list[np.ndarray] = []
    for level_name in expected_levels:
        level_values = series.index.get_level_values(level_name)
        grid_name = state_name if level_name == outcome_level else level_name
        if grid_name in all_grids:
            grid = all_grids[grid_name]
            label_to_code = dict(zip(grid.categories, grid.codes, strict=True))
            try:
                mapped = np.array([label_to_code[v] for v in level_values])
            except KeyError:
                invalid = set(level_values) - set(grid.categories)
                msg = (
                    f"Invalid labels for level '{level_name}': "
                    f"{sorted(invalid)}. "
                    f"Valid labels: {list(grid.categories)}."
                )
                raise ValueError(msg) from None
            index_arrays.append(mapped)
        else:
            index_arrays.append(np.array(level_values, dtype=int))
    return index_arrays


def _get_indexing_params(func: Callable) -> list[str]:
    """Return indexing parameter names from a transition function signature.

    All parameters except `probs_array` are indexing params, returned in
    declaration order.

    Args:
        func: The transition function.

    Returns:
        List of indexing parameter names.

    Raises:
        ValueError: If `probs_array` is not found in the signature.

    """
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    if "probs_array" not in param_names:
        msg = (
            f"Transition function '{func.__name__}' must have a parameter named "  # ty: ignore[unresolved-attribute]
            f"'probs_array'. Found parameters: {param_names}."
        )
        raise ValueError(msg)
    return [name for name in param_names if name != "probs_array"]


def _build_discrete_grid_lookup(
    regimes: Mapping[str, Regime],
) -> dict[str, DiscreteGrid]:
    """Collect all DiscreteGrid instances across regimes, verifying consistency.

    Args:
        regimes: Mapping of regime names to Regime instances.

    Returns:
        Mapping from state name to DiscreteGrid.

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
        Mapping from action name to DiscreteGrid.

    """
    return {
        name: grid
        for name, grid in regime.actions.items()
        if isinstance(grid, DiscreteGrid)
    }
