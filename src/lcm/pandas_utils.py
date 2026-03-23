"""Utilities for converting between pandas and LCM data structures."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import overload

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from lcm.error_handling import _extract_markov_transition, _get_indexing_params
from lcm.grids import DiscreteGrid
from lcm.model import Model
from lcm.regime import MarkovTransition, Regime
from lcm.shocks import _ShockGrid


def initial_conditions_from_dataframe(
    df: pd.DataFrame,
    *,
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

    regime_names = df["regime"].tolist()

    state_columns = {col for col in df.columns if col != "regime"}
    _validate_state_columns(state_columns, model.regimes, regime_names)

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

            initial_conditions[col] = jnp.array([label_to_code[v] for v in values])
        else:
            initial_conditions[col] = jnp.array(df[col].values)

    # Convert regime names to integer codes
    initial_conditions["regime"] = jnp.array(
        [model.regime_names_to_ids[name] for name in regime_names]
    )

    return initial_conditions


@overload
def transition_probs_from_series(
    *,
    series: pd.Series,
    model: Model,
    regime_name: str | None = ...,
) -> Array: ...


@overload
def transition_probs_from_series(
    *,
    series: pd.Series,
    model: Model,
    regime_name: str | None = ...,
    target_regime_name: str,
) -> Array: ...


def transition_probs_from_series(
    *,
    series: pd.Series,
    model: Model,
    regime_name: str | None = None,
    target_regime_name: str | None = None,
) -> Array:
    """Convert a labeled pandas Series to a transition probability array.

    Build a transition probability array from a Series with a named MultiIndex,
    eliminating manual array construction with opaque axis ordering. The
    transition type (state vs. regime) is inferred from the `"next_*"` level in
    the MultiIndex: `"next_health"` means a state transition on `"health"`,
    while `"next_regime"` means a regime transition.

    The MultiIndex must use `"age"` (with actual age values from the model's
    `AgeGrid`) for the age/period dimension — not `"period"`.

    For per-target state transitions (where ``state_transitions[state_name]`` is a
    dict mapping target regime names to `MarkovTransition` instances), pass
    ``target_regime_name`` to select the specific transition.

    Args:
        series: Series with a named MultiIndex. Level names must match the
            indexing parameters of the transition function (with `"period"`
            replaced by `"age"`) plus the outcome level
            (`"next_{state_name}"` or `"next_regime"`).
        model: The LCM Model instance.
        regime_name: Name of the regime containing the transition. If ``None``,
            inferred by scanning regimes for a matching `MarkovTransition`.
        target_regime_name: Target regime name for per-target state transitions.
            Required when the state transition is a per-target dict.

    Returns:
        JAX array with axes corresponding to the indexing parameters in
        declaration order, followed by the outcome axis.

    Raises:
        TypeError: If the transition is not a `MarkovTransition`.
        ValueError: If level names don't match, labels are invalid, or
            inference fails.

    """
    state_name = _infer_transition_target(series)
    if regime_name is None:
        regime_name = _infer_regime_name(
            model=model,
            state_name=state_name,
            target_regime_name=target_regime_name,
        )

    regime = model.regimes[regime_name]
    discrete_lookup = _build_discrete_grid_lookup(model.regimes)
    action_lookup = _build_discrete_action_lookup(regime)
    all_grids = {**discrete_lookup, **action_lookup}

    if state_name is not None:
        raw_transition = regime.state_transitions[state_name]
        markov = _extract_markov_transition(
            raw_transition,
            state_name=state_name,
            regime_name=regime_name,
            target_regime_name=target_regime_name,
        )
        func = markov.func
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


def _infer_transition_target(series: pd.Series) -> str | None:
    """Find the ``next_*`` level in the MultiIndex.

    Args:
        series: Series whose MultiIndex must contain exactly one ``next_*`` level.

    Returns:
        The state name (e.g. ``"health"`` for ``"next_health"``), or ``None``
        for regime transitions (``"next_regime"``).

    Raises:
        ValueError: If no ``next_*`` level is found.

    """
    next_levels = [
        name
        for name in series.index.names
        if isinstance(name, str) and name.startswith("next_")
    ]
    if not next_levels:
        msg = (
            "No 'next_*' level found in the Series MultiIndex. "
            "Expected a level like 'next_health' or 'next_regime'."
        )
        raise ValueError(msg)
    suffix = next_levels[0].removeprefix("next_")
    return None if suffix == "regime" else suffix


def _infer_regime_name(
    *,
    model: Model,
    state_name: str | None,
    target_regime_name: str | None,
) -> str:
    """Infer the regime name by scanning for a matching MarkovTransition.

    Args:
        model: The LCM Model instance.
        state_name: Name of the state variable, or ``None`` for regime
            transitions.
        target_regime_name: Target regime name for per-target dicts.

    Returns:
        The inferred regime name.

    Raises:
        TypeError: If a candidate regime uses a per-target dict for the state.
        ValueError: If no matching regime is found or multiple regimes match
            with different transition function signatures.

    """
    if state_name is None:
        candidates = [
            name
            for name, regime in model.regimes.items()
            if isinstance(regime.transition, MarkovTransition)
        ]
    else:
        candidates = [
            name
            for name, regime in model.regimes.items()
            if _has_markov_on_state(regime, state_name)
        ]

    if not candidates:
        if state_name is None:
            msg = "No regime with a stochastic regime transition found."
        else:
            msg = f"No regime with a MarkovTransition on state '{state_name}' found."
        raise ValueError(msg)

    _fail_if_per_target_candidate(candidates, state_name, model)

    if len(candidates) == 1:
        return candidates[0]

    return _pick_among_multiple_candidates(
        candidates, state_name, target_regime_name, model
    )


def _fail_if_per_target_candidate(
    candidates: list[str],
    state_name: str | None,
    model: Model,
) -> None:
    """Raise if any candidate uses a per-target dict for `state_name`."""
    if state_name is None:
        return
    for cand_name in candidates:
        raw = model.regimes[cand_name].state_transitions[state_name]
        if isinstance(raw, Mapping):
            msg = (
                f"State '{state_name}' in regime '{cand_name}' uses "
                f"per-target transitions. Pass `regime_name` and "
                f"`target_regime_name` explicitly."
            )
            raise TypeError(msg)


def _pick_among_multiple_candidates(
    candidates: list[str],
    state_name: str | None,
    target_regime_name: str | None,
    model: Model,
) -> str:
    """Pick a regime when multiple candidates have a matching MarkovTransition.

    Return the first candidate if all have identical indexing params.
    Raise if signatures differ.

    """
    param_sets: set[tuple[str, ...]] = set()
    for cand_name in candidates:
        regime = model.regimes[cand_name]
        if state_name is not None:
            raw = regime.state_transitions[state_name]
            markov = _extract_markov_transition(
                raw,
                state_name=state_name,
                regime_name=cand_name,
                target_regime_name=target_regime_name,
            )
            func = markov.func
        else:
            func = regime.transition.func  # ty: ignore[unresolved-attribute]
        param_sets.add(tuple(_get_indexing_params(func)))

    if len(param_sets) == 1:
        return candidates[0]

    msg = (
        f"Multiple regimes have a MarkovTransition that matches, but with "
        f"different signatures: {candidates}. Pass regime_name explicitly."
    )
    raise ValueError(msg)


def _has_markov_on_state(regime: Regime, state_name: str) -> bool:
    """Return True if `regime` has a MarkovTransition on `state_name`."""
    if state_name not in regime.state_transitions:
        return False
    raw = regime.state_transitions[state_name]
    if isinstance(raw, MarkovTransition):
        return True
    return isinstance(raw, Mapping) and any(
        isinstance(v, MarkovTransition) for v in raw.values()
    )


@dataclass(frozen=True)
class _OutcomeMapping:
    """Metadata for the outcome level of a transition probability array."""

    level_name: str
    """Level name in the MultiIndex (e.g., `"next_health"` or `"next_regime"`)."""

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
    """Build a probability array from a transition function and labeled Series.

    Args:
        func: The transition function whose signature defines the axis order.
        outcome: Metadata for the outcome (last) axis.
        all_grids: dict of discrete grid names to `DiscreteGrid` instances.
        model: The LCM Model instance (used for age-to-period mapping).
        series: Series with a named MultiIndex containing the probability values.

    Returns:
        JAX array with axes matching the function's indexing parameters (in
        declaration order) followed by the outcome axis.

    """
    indexing_params = _get_indexing_params(func)
    # Replace internal "period" param with user-facing "age" level name
    expected_levels = [
        "age" if param == "period" else param for param in indexing_params
    ]
    expected_levels.append(outcome.level_name)

    series = _validate_and_reorder_levels(series, expected_levels)

    shape = _compute_shape(expected_levels, outcome, all_grids, model)
    index_arrays = _map_labels_to_codes(
        expected_levels, outcome, all_grids, model, series
    )

    result = np.zeros(shape, dtype=float)
    result[tuple(index_arrays)] = series.to_numpy()

    return jnp.array(result)


def _validate_and_reorder_levels(
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


def _compute_shape(
    expected_levels: list[str],
    outcome: _OutcomeMapping,
    all_grids: dict[str, DiscreteGrid],
    model: Model,
) -> list[int]:
    """Compute the expected array shape from grid sizes."""
    shape: list[int] = []
    for level_name in expected_levels:
        if level_name == "age":
            shape.append(model.n_periods)
        elif level_name == outcome.level_name:
            shape.append(outcome.n_outcomes)
        elif level_name in all_grids:
            shape.append(len(all_grids[level_name].categories))
        else:
            msg = (
                f"Unrecognised level name '{level_name}'. Expected 'age', "
                f"a discrete grid name ({sorted(all_grids)}), or the outcome "
                f"level '{outcome.level_name}'."
            )
            raise ValueError(msg)
    return shape


def _map_labels_to_codes(
    expected_levels: list[str],
    outcome: _OutcomeMapping,
    all_grids: dict[str, DiscreteGrid],
    model: Model,
    series: pd.Series,
) -> list[np.ndarray]:
    """Map string labels in MultiIndex levels to integer codes."""
    index_arrays: list[np.ndarray] = []
    for level_name in expected_levels:
        level_values = series.index.get_level_values(level_name)

        if level_name == "age":
            try:
                mapped = np.array([model.ages.age_to_period(v) for v in level_values])
            except ValueError:
                valid_ages = sorted(float(v) for v in model.ages.exact_values)
                invalid = {float(v) for v in level_values} - set(valid_ages)
                msg = (
                    f"Invalid age values: {sorted(invalid)}. Valid ages: {valid_ages}."
                )
                raise ValueError(msg) from None
            index_arrays.append(mapped)
            continue

        if level_name == outcome.level_name:
            label_to_code = outcome.label_to_code
        elif level_name in all_grids:
            grid = all_grids[level_name]
            label_to_code = dict(zip(grid.categories, grid.codes, strict=True))
        else:
            msg = (
                f"Unrecognised level name '{level_name}'. Expected 'age', "
                f"a discrete grid name ({sorted(all_grids)}), or the outcome "
                f"level '{outcome.level_name}'."
            )
            raise ValueError(msg)

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


def _validate_state_columns(
    state_columns: set[str],
    regimes: Mapping[str, Regime],
    initial_regimes: list[str],
) -> None:
    """Validate that DataFrame columns match model states."""
    all_states = _collect_all_state_names(regimes, initial_regimes)

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
    regimes: Mapping[str, Regime],
    initial_regimes: list[str],
) -> set[str]:
    """Collect all non-shock state names from regimes present in initial_regimes."""
    state_names: set[str] = set()
    for regime_name in set(initial_regimes):
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
