"""Utilities for converting between pandas and LCM data structures."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import overload

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from lcm.error_handling import _get_indexing_params
from lcm.grids import DiscreteGrid
from lcm.model import Model
from lcm.regime import MarkovTransition, Regime


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


@overload
def transition_probs_from_series(
    *,
    series: pd.Series,
    model: Model,
    regime_name: str,
    state_name: str,
) -> Array: ...


@overload
def transition_probs_from_series(
    *,
    series: pd.Series,
    model: Model,
    regime_name: str,
) -> Array: ...


def transition_probs_from_series(
    *,
    series: pd.Series,
    model: Model,
    regime_name: str,
    state_name: str | None = None,
) -> Array:
    """Convert a labeled pandas Series to a transition probability array.

    Build a transition probability array from a Series with a named MultiIndex,
    eliminating manual array construction with opaque axis ordering. Works for
    both state transitions (pass `state_name`) and regime transitions (omit
    `state_name`).

    Args:
        series: Series with a named MultiIndex. Level names must match the
            indexing parameters of the transition function plus the outcome
            level (`"next_{state_name}"` or `"next_regime"`).
        model: The LCM Model instance.
        regime_name: Name of the regime containing the transition.
        state_name: Name of the state with a `MarkovTransition`. Omit for
            regime transitions.

    Returns:
        JAX array with axes corresponding to the indexing parameters in
        declaration order, followed by the outcome axis.

    Raises:
        TypeError: If the transition is not a `MarkovTransition`.
        ValueError: If level names don't match or labels are invalid.

    """
    regime = model.regimes[regime_name]
    discrete_lookup = _build_discrete_grid_lookup(model.regimes)
    action_lookup = _build_discrete_action_lookup(regime)
    all_grids = {**discrete_lookup, **action_lookup}

    if state_name is not None:
        raw_transition = regime.state_transitions[state_name]
        if not isinstance(raw_transition, MarkovTransition):
            msg = (
                f"State '{state_name}' in regime '{regime_name}' is not a "
                f"MarkovTransition. Got {type(raw_transition).__name__}."
            )
            raise TypeError(msg)

        func = raw_transition.func
        state_grid = all_grids[state_name]
        outcome = _OutcomeMapping(
            level_name=f"next_{state_name}",
            label_to_code=MappingProxyType(
                dict(zip(state_grid.categories, state_grid.codes, strict=True))
            ),
            n_outcomes=len(state_grid.categories),
        )
    else:
        if not isinstance(regime.transition, MarkovTransition):
            msg = (
                f"Regime '{regime_name}' does not have a stochastic regime "
                f"transition. Got {type(regime.transition).__name__}."
            )
            raise TypeError(msg)

        func = regime.transition.func
        outcome = _OutcomeMapping(
            level_name="next_regime",
            label_to_code=MappingProxyType(dict(model.regime_names_to_ids)),
            n_outcomes=len(model.regime_names_to_ids),
        )

    return _build_probs_array(func, outcome, all_grids, model, series)


@dataclass(frozen=True)
class _OutcomeMapping:
    """Metadata for the outcome level of a transition probability array."""

    level_name: str
    """Level name in the MultiIndex (e.g., ``"next_health"`` or ``"next_regime"``)."""

    label_to_code: MappingProxyType[str, int]
    """Immutable mapping from string labels to integer codes."""

    n_outcomes: int
    """Number of outcome categories."""


def _build_probs_array(
    func: Callable,
    outcome: _OutcomeMapping,
    all_grids: dict[str, DiscreteGrid],
    model: Model,
    series: pd.Series,
) -> Array:
    """Build a probability array from a transition function and labeled Series."""
    indexing_params = _get_indexing_params(func)
    expected_levels = [*indexing_params, outcome.level_name]

    series = _validate_and_reorder_levels(series, expected_levels)

    shape = _compute_shape(expected_levels, outcome, all_grids, model, series)
    index_arrays = _map_labels_to_codes(expected_levels, outcome, all_grids, series)

    result = np.zeros(shape, dtype=float)
    result[tuple(index_arrays)] = series.to_numpy()

    return jnp.array(result)


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
    outcome: _OutcomeMapping,
    all_grids: dict[str, DiscreteGrid],
    model: Model,
    series: pd.Series,
) -> list[int]:
    """Compute the expected array shape from grid sizes."""
    shape: list[int] = []
    for level_name in expected_levels:
        if level_name == "period":
            shape.append(model.n_periods)
        elif level_name == outcome.level_name:
            shape.append(outcome.n_outcomes)
        elif level_name in all_grids:
            shape.append(len(all_grids[level_name].categories))
        else:
            shape.append(len(series.index.get_level_values(level_name).unique()))
    return shape


def _map_labels_to_codes(
    expected_levels: list[str],
    outcome: _OutcomeMapping,
    all_grids: dict[str, DiscreteGrid],
    series: pd.Series,
) -> list[np.ndarray]:
    """Map string labels in MultiIndex levels to integer codes."""
    index_arrays: list[np.ndarray] = []
    for level_name in expected_levels:
        level_values = series.index.get_level_values(level_name)

        if level_name == outcome.level_name:
            label_to_code = outcome.label_to_code
        elif level_name in all_grids:
            grid = all_grids[level_name]
            label_to_code = dict(zip(grid.categories, grid.codes, strict=True))
        else:
            unique_sorted = sorted(set(level_values))
            label_to_code = {v: i for i, v in enumerate(unique_sorted)}

        try:
            mapped = np.array([label_to_code[v] for v in level_values])
        except KeyError:
            invalid = set(level_values) - set(label_to_code)
            msg = (
                f"Invalid labels for level '{level_name}': "
                f"{sorted(invalid)}. "
                f"Valid labels: {sorted(label_to_code)}."
            )
            raise ValueError(msg) from None
        index_arrays.append(mapped)
    return index_arrays


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
