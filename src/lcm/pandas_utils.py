"""Utilities for converting between pandas and LCM data structures."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import overload

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from lcm.ages import AgeGrid
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
    categoricals: dict[str, DiscreteGrid] | None = ...,
) -> Array: ...


@overload
def transition_probs_from_series(
    *,
    series: pd.Series,
    model: Model,
    regime_name: str | None = ...,
    target_regime_name: str,
    categoricals: dict[str, DiscreteGrid] | None = ...,
) -> Array: ...


def transition_probs_from_series(
    *,
    series: pd.Series,
    model: Model,
    regime_name: str | None = None,
    target_regime_name: str | None = None,
    categoricals: dict[str, DiscreteGrid] | None = None,
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
        categoricals: Extra categorical mappings (level name → grid) for
            derived variables not in the model's state/action grids.

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
    all_grids = _resolve_categoricals(
        model=model, regime_name=regime_name, categoricals=categoricals
    )

    if state_name is not None:
        raw_transition = regime.state_transitions[state_name]
        markov = _extract_markov_transition(
            raw_transition,
            state_name=state_name,
            regime_name=regime_name,
            target_regime_name=target_regime_name,
        )
        func = markov.func
        outcome_mapping = _grid_level_mapping(
            f"next_{state_name}", all_grids[state_name]
        )
    else:
        if not isinstance(regime.transition, MarkovTransition):
            msg = (
                f"Regime '{regime_name}' does not have a stochastic regime "
                f"transition. Got {type(regime.transition).__name__}."
            )
            raise TypeError(msg)

        func = regime.transition.func
        regime_ids = dict(model.regime_names_to_ids)
        outcome_mapping = _LevelMapping(
            name="next_regime",
            size=len(regime_ids),
            label_to_index=regime_ids.__getitem__,  # ty: ignore[invalid-argument-type]
            valid_labels=tuple(regime_ids),
        )

    return _build_probs_array(func, outcome_mapping, all_grids, model, series)


def array_from_series(
    *,
    data: pd.Series,
    ages: AgeGrid | None = None,
    model: Model | None = None,
    regime_name: str | None = None,
    categoricals: dict[str, DiscreteGrid] | None = None,
    expected_levels: tuple[str, ...] | None = None,
) -> Array:
    """Convert a pandas Series to a JAX array.

    When ``expected_levels`` is provided, the Series must have a MultiIndex
    whose level names match exactly (after ``"period"`` → ``"age"`` rename).
    Levels are reordered to match, and the result is an N-dimensional array
    with one axis per level. This mode supports any number of categorical
    levels.

    When ``expected_levels`` is ``None``, behavior is inferred from the index:

    1. ``ages`` + simple index → 1D ``[n_periods]``, NaN for missing ages
    2. ``ages`` + MultiIndex (age + 1 categorical) → 2D ``[n_periods, n_cats]``
    3. No ``ages``, categorical index → 1D ``[n_cats]``
    4. No ``ages``, no categoricals → 1D from raw values

    Categorical levels are resolved from ``model`` state/action grids (auto)
    and/or explicit ``categoricals`` (which take precedence). Every non-age
    MultiIndex level **must** have a corresponding categorical or ValueError
    is raised.

    Missing grid points are filled with NaN.

    Args:
        data: pandas Series.
        ages: ``AgeGrid`` for age-to-period alignment.
        model: ``Model`` for auto-discovering state/action categoricals.
        regime_name: Regime for action grid discovery (requires ``model``).
        categoricals: Explicit categorical mappings (level name → grid).
        expected_levels: Exact MultiIndex level names in output axis order.
            Enables strict validation and N-categorical support.

    Returns:
        JAX array.

    Raises:
        ValueError: If a non-age index level has no categorical mapping, or
            if ``expected_levels`` doesn't match the Series MultiIndex.

    """
    grids = _resolve_categoricals(
        model=model, regime_name=regime_name, categoricals=categoricals
    )

    if expected_levels is not None:
        level_mappings = _build_level_mappings_from_grids(
            expected_levels, grids=grids, ages=ages
        )
        return _scatter_series(data, level_mappings)

    if isinstance(data.index, pd.MultiIndex):
        return _multiindex_series_to_array(data, ages=ages, grids=grids)

    # Simple index — check if it's age-aligned or categorical
    if ages is not None:
        return _age_series_to_array(data, ages=ages)

    # Check if index name matches a categorical
    idx_name = str(data.index.name) if data.index.name is not None else None
    if idx_name is not None and idx_name in grids:
        return _categorical_series_to_array(data, grid=grids[idx_name])

    return jnp.array(data.to_numpy(), dtype=float)


def _build_level_mappings_from_grids(
    expected_levels: tuple[str, ...],
    *,
    grids: dict[str, DiscreteGrid],
    ages: AgeGrid | None,
) -> tuple[_LevelMapping, ...]:
    """Build level mappings for `array_from_series` from level names and grids.

    Args:
        expected_levels: Level names in output axis order.
        grids: Categorical grid lookup.
        ages: ``AgeGrid`` for the age dimension (required if ``"age"`` is in
            ``expected_levels``).

    Returns:
        Tuple of `_LevelMapping` instances.

    """
    mappings: list[_LevelMapping] = []
    for level_name in expected_levels:
        if level_name == "age":
            if ages is None:
                msg = "expected_levels contains 'age' but no AgeGrid was provided."
                raise ValueError(msg)
            mappings.append(_age_level_mapping(ages))
        elif level_name in grids:
            mappings.append(_grid_level_mapping(level_name, grids[level_name]))
        else:
            msg = (
                f"No categorical mapping for level {level_name!r}. "
                f"Provide it via `categoricals` or `model`. "
                f"Available: {sorted(grids)}."
            )
            raise ValueError(msg)
    return tuple(mappings)


def array_mapping_from_dataframe(
    *,
    data: pd.DataFrame,
    ages: AgeGrid | None = None,
    model: Model | None = None,
    regime_name: str | None = None,
    categoricals: dict[str, DiscreteGrid] | None = None,
) -> dict[str, Array]:
    """Convert each DataFrame column to a JAX array via `array_from_series`.

    Args:
        data: pandas DataFrame.
        ages: ``AgeGrid`` for age-to-period alignment.
        model: ``Model`` for auto-discovering state/action categoricals.
        regime_name: Regime for action grid discovery (requires ``model``).
        categoricals: Explicit categorical mappings (level name → grid).

    Returns:
        Dict mapping column names to JAX arrays.

    """
    return {
        col: array_from_series(
            data=data[col],
            ages=ages,
            model=model,
            regime_name=regime_name,
            categoricals=categoricals,
        )
        for col in data.columns
    }


def _resolve_categoricals(
    *,
    model: Model | None,
    regime_name: str | None,
    categoricals: dict[str, DiscreteGrid] | None,
) -> dict[str, DiscreteGrid]:
    """Build combined categorical lookup from model + explicit overrides."""
    grids: dict[str, DiscreteGrid] = {}
    if model is not None:
        grids.update(_build_discrete_grid_lookup(model.regimes))
        if regime_name is not None:
            grids.update(_build_discrete_action_lookup(model.regimes[regime_name]))
    if categoricals is not None:
        grids.update(categoricals)
    return grids


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


def _grid_level_mapping(name: str, grid: DiscreteGrid) -> _LevelMapping:
    """Create a `_LevelMapping` for a categorical dimension."""
    label_to_code = dict(zip(grid.categories, grid.codes, strict=True))
    return _LevelMapping(
        name=name,
        size=len(grid.categories),
        label_to_index=label_to_code.__getitem__,
        valid_labels=grid.categories,
    )


def _scatter_series(
    series: pd.Series,
    level_mappings: tuple[_LevelMapping, ...],
    *,
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
        JAX array with shape ``[m.size for m in level_mappings]``.

    """
    expected_levels = [m.name for m in level_mappings]
    series = _validate_and_reorder_levels(series, expected_levels)

    shape = [m.size for m in level_mappings]
    index_arrays = [
        _map_level(mapping, series.index.get_level_values(mapping.name))
        for mapping in level_mappings
    ]

    result = np.full(shape, fill_value)
    result[tuple(index_arrays)] = series.to_numpy()
    return jnp.array(result)


def _map_level(mapping: _LevelMapping, level_values: pd.Index) -> np.ndarray:
    """Map label values from one MultiIndex level to integer indices."""
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


def _age_series_to_array(series: pd.Series, *, ages: AgeGrid) -> Array:
    """Convert a Series with age index to a 1D JAX array, NaN-filling gaps."""
    grid_ages = {float(v) for v in ages.exact_values}
    filtered = series.loc[[a for a in series.index if float(a) in grid_ages]]
    period_indices = np.array([ages.age_to_period(float(a)) for a in filtered.index])

    result = np.full(ages.n_periods, np.nan)
    result[period_indices] = filtered.to_numpy()
    return jnp.array(result)


def _categorical_series_to_array(series: pd.Series, *, grid: DiscreteGrid) -> Array:
    """Convert a Series indexed by categorical labels to a 1D JAX array."""
    label_to_code = dict(zip(grid.categories, grid.codes, strict=True))
    result = np.full(len(grid.categories), np.nan)
    for label, value in series.items():
        if label not in label_to_code:
            msg = (
                f"Invalid label {label!r} for categorical '{series.index.name}'. "
                f"Valid labels: {list(grid.categories)}."
            )
            raise ValueError(msg)
        result[label_to_code[label]] = float(value)
    return jnp.array(result)


def _multiindex_series_to_array(
    series: pd.Series,
    *,
    ages: AgeGrid | None,
    grids: dict[str, DiscreteGrid],
) -> Array:
    """Convert a MultiIndex Series to a JAX array using age + categorical mapping."""
    index: pd.MultiIndex = series.index  # ty: ignore[invalid-assignment]
    level_names = list(index.names)

    has_age = "age" in level_names
    cat_levels = [name for name in level_names if name != "age"]

    # Validate: every non-age level must have a categorical
    for level in cat_levels:
        if level not in grids:
            msg = (
                f"No categorical mapping for index level {level!r}. "
                f"Provide it via `categoricals` or `model`. "
                f"Available: {sorted(grids)}."
            )
            raise ValueError(msg)

    if len(cat_levels) != 1:
        msg = f"Expected exactly one non-age index level, but got {cat_levels}."
        raise ValueError(msg)

    cat_level = str(cat_levels[0])
    cat_grid = grids[cat_level]
    label_to_code = dict(zip(cat_grid.categories, cat_grid.codes, strict=True))
    n_cats = len(cat_grid.categories)

    if has_age and ages is not None:
        grid_ages = {float(v) for v in ages.exact_values}
        result = np.full((ages.n_periods, n_cats), np.nan)

        age_pos = level_names.index("age")
        cat_pos = 1 - age_pos  # the other position
        for idx, value in series.items():
            tup = idx if isinstance(idx, tuple) else (idx,)
            age_f = float(str(tup[age_pos]))
            cat_label = str(tup[cat_pos])
            if age_f not in grid_ages:
                continue
            if cat_label not in label_to_code:
                msg = (
                    f"Invalid label {cat_label!r} for level {cat_level!r}. "
                    f"Valid: {list(cat_grid.categories)}."
                )
                raise ValueError(msg)
            period = ages.age_to_period(age_f)
            result[period, label_to_code[cat_label]] = float(value)
    else:
        # No age dimension — just categorical MultiIndex
        result = np.full(n_cats, np.nan)
        for idx, value in series.items():
            tup = idx if isinstance(idx, tuple) else (idx,)
            cat_label = str(tup[0])
            if cat_label not in label_to_code:
                msg = (
                    f"Invalid label {cat_label!r} for level {cat_level!r}. "
                    f"Valid: {list(cat_grid.categories)}."
                )
                raise ValueError(msg)
            result[label_to_code[cat_label]] = float(value)

    return jnp.array(result)


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


def _build_probs_array(
    func: Callable,
    outcome_mapping: _LevelMapping,
    all_grids: dict[str, DiscreteGrid],
    model: Model,
    series: pd.Series,
) -> Array:
    """Build a probability array from a transition function and labeled Series.

    Args:
        func: The transition function whose signature defines the axis order.
        outcome_mapping: `_LevelMapping` for the outcome (last) axis.
        all_grids: dict of discrete grid names to `DiscreteGrid` instances.
        model: The LCM Model instance (used for age-to-period mapping).
        series: Series with a named MultiIndex containing the probability values.

    Returns:
        JAX array with axes matching the function's indexing parameters (in
        declaration order) followed by the outcome axis.

    """
    indexing_params = _get_indexing_params(func)
    level_mappings = _build_level_mappings(
        indexing_params, all_grids, model.ages, outcome_mapping
    )
    return _scatter_series(series, level_mappings)


def _build_level_mappings(
    indexing_params: list[str],
    all_grids: dict[str, DiscreteGrid],
    ages: AgeGrid,
    outcome_mapping: _LevelMapping,
) -> tuple[_LevelMapping, ...]:
    """Build level mappings from function indexing params + outcome.

    Replaces the internal ``"period"`` parameter name with ``"age"`` and
    constructs a `_LevelMapping` for each level.

    """
    mappings: list[_LevelMapping] = []
    for param in indexing_params:
        if param == "period":
            mappings.append(_age_level_mapping(ages))
        elif param in all_grids:
            mappings.append(_grid_level_mapping(param, all_grids[param]))
        else:
            msg = (
                f"Unrecognised indexing parameter '{param}'. Expected 'period', "
                f"or a discrete grid name ({sorted(all_grids)})."
            )
            raise ValueError(msg)
    mappings.append(outcome_mapping)
    return tuple(mappings)


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
