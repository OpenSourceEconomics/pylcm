"""Simulation result object with deferred DataFrame computation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import pandas as pd
from dags import concatenate_functions

from lcm.dispatchers import vmap_1d
from lcm.exceptions import InvalidAdditionalTargetsError
from lcm.grids import DiscreteGrid

if TYPE_CHECKING:
    from jax import Array

    from lcm.interfaces import InternalRegime, PeriodRegimeData
    from lcm.typing import FloatND, ParamsDict, RegimeName


# ======================================================================================
# Main result class
# ======================================================================================


class SimulationResult:
    """Result object from model simulation with deferred DataFrame computation."""

    def __init__(
        self,
        raw_results: dict[str, dict[int, PeriodRegimeData]],
        internal_regimes: dict[RegimeName, InternalRegime],
        params: ParamsDict,
        V_arr_dict: dict[int, dict[RegimeName, FloatND]],
    ) -> None:
        self._raw_results = raw_results
        self._internal_regimes = internal_regimes
        self._params = params
        self._V_arr_dict = V_arr_dict
        self._metadata = _compute_metadata(internal_regimes, raw_results)

    # ----------------------------------------------------------------------------------
    # Public properties for advanced users
    # ----------------------------------------------------------------------------------

    @property
    def raw_results(self) -> dict[str, dict[int, PeriodRegimeData]]:
        """Raw simulation results by regime and period."""
        return self._raw_results

    @property
    def params(self) -> ParamsDict:
        """Model parameters used in simulation."""
        return self._params

    @property
    def V_arr_dict(self) -> dict[int, dict[RegimeName, FloatND]]:
        """Value function arrays from the solution."""
        return self._V_arr_dict

    # ----------------------------------------------------------------------------------
    # Metadata properties (delegated to _metadata)
    # ----------------------------------------------------------------------------------

    @property
    def regime_names(self) -> tuple[str, ...]:
        """Names of all regimes."""
        return self._metadata.regime_names

    @property
    def state_names(self) -> tuple[str, ...]:
        """Names of all state variables (union across regimes)."""
        return self._metadata.state_names

    @property
    def action_names(self) -> tuple[str, ...]:
        """Names of all action variables (union across regimes)."""
        return self._metadata.action_names

    @property
    def n_periods(self) -> int:
        """Number of periods in the simulation."""
        return self._metadata.n_periods

    @property
    def n_subjects(self) -> int:
        """Number of subjects simulated."""
        return self._metadata.n_subjects

    # ----------------------------------------------------------------------------------
    # Main methods
    # ----------------------------------------------------------------------------------

    def to_dataframe(
        self,
        additional_targets: list[str] | None = None,
        *,
        use_labels: bool = True,
    ) -> pd.DataFrame:
        """Convert simulation results to a flat pandas DataFrame.

        Args:
            additional_targets: Optional list of target names to compute. Targets
                can be any function defined in a regime. Each target is computed for the
                regimes where it exists; rows from regimes without that target will have
                NaN.
            use_labels: If True (default), discrete variables (states, actions, and
                regime) are returned as pandas Categorical dtype with string labels.
                If False, discrete variables are returned as integer codes.

        Returns:
            DataFrame with simulation results.

        """
        if additional_targets is not None:
            _validate_targets(additional_targets, self._internal_regimes)

        df = _create_flat_dataframe(
            raw_results=self._raw_results,
            internal_regimes=self._internal_regimes,
            params=self._params,
            metadata=self._metadata,
            additional_targets=additional_targets,
        )

        if use_labels:
            df = _convert_to_categorical(df, self._metadata)

        return df

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  n_periods={self.n_periods},\n"
            f"  n_subjects={self.n_subjects},\n"
            f"  regime_names={self.regime_names},\n"
            f"  state_names={self.state_names},\n"
            f"  action_names={self.action_names}\n"
            f")"
        )


# ======================================================================================
# Metadata
# ======================================================================================


@dataclass(frozen=True)
class SimulationMetadata:
    """Pre-computed metadata about the simulation."""

    regime_names: tuple[str, ...]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    n_periods: int
    n_subjects: int
    regime_to_states: dict[str, tuple[str, ...]]
    regime_to_actions: dict[str, tuple[str, ...]]
    discrete_categories: dict[str, tuple[str, ...]]


def _compute_metadata(
    internal_regimes: dict[RegimeName, InternalRegime],
    raw_results: dict[RegimeName, dict[int, PeriodRegimeData]],
) -> SimulationMetadata:
    """Compute metadata from internal regimes and raw results."""
    regime_names = tuple(internal_regimes.keys())

    all_states: set[str] = set()
    all_actions: set[str] = set()
    regime_to_states: dict[str, tuple[str, ...]] = {}
    regime_to_actions: dict[str, tuple[str, ...]] = {}
    discrete_categories: dict[str, tuple[str, ...]] = {}

    for regime_name, regime in internal_regimes.items():
        vi = regime.variable_info
        states = tuple(vi.query("is_state").index.tolist())
        actions = tuple(vi.query("is_action").index.tolist())
        regime_to_states[regime_name] = states
        regime_to_actions[regime_name] = actions
        all_states.update(states)
        all_actions.update(actions)

        # Extract categories from discrete grids
        for var_name, grid in regime.gridspecs.items():
            if isinstance(grid, DiscreteGrid) and var_name not in discrete_categories:
                discrete_categories[var_name] = grid.categories

    n_periods = len(raw_results[regime_names[0]])
    n_subjects = _get_n_subjects(raw_results)

    return SimulationMetadata(
        regime_names=regime_names,
        state_names=tuple(sorted(all_states)),
        action_names=tuple(sorted(all_actions)),
        n_periods=n_periods,
        n_subjects=n_subjects,
        regime_to_states=regime_to_states,
        regime_to_actions=regime_to_actions,
        discrete_categories=discrete_categories,
    )


def _get_n_subjects(raw_results: dict[RegimeName, dict[int, PeriodRegimeData]]) -> int:
    """Extract number of subjects from raw results."""
    for regime_results in raw_results.values():
        if regime_results:
            first_result = next(iter(regime_results.values()))
            return len(first_result.in_regime)
    return 0


# ======================================================================================
# Target validation
# ======================================================================================


def _validate_targets(
    targets: list[str],
    internal_regimes: dict[RegimeName, InternalRegime],
) -> None:
    """Validate that each target exists in at least one regime."""
    all_available = _collect_all_available_targets(internal_regimes)
    invalid = set(targets) - all_available
    if invalid:
        raise InvalidAdditionalTargetsError(
            f"Targets {invalid} not found in any regime. "
            f"Available targets: {sorted(all_available)}"
        )


def _collect_all_available_targets(
    internal_regimes: dict[RegimeName, InternalRegime],
) -> set[str]:
    """Collect all available target names across all regimes."""
    all_targets: set[str] = set()
    for regime in internal_regimes.values():
        all_targets.update(_get_available_targets_for_regime(regime))
    return all_targets


def _get_available_targets_for_regime(regime: InternalRegime) -> set[str]:
    """Get available target names for a single regime."""
    targets = {"utility"}
    targets.update(regime.functions.keys())
    targets.update(regime.constraints.keys())
    return targets


# ======================================================================================
# DataFrame creation
# ======================================================================================


def _create_flat_dataframe(
    raw_results: dict[str, dict[int, PeriodRegimeData]],
    internal_regimes: dict[RegimeName, InternalRegime],
    params: ParamsDict,
    metadata: SimulationMetadata,
    additional_targets: list[str] | None,
) -> pd.DataFrame:
    """Create a single flat DataFrame from all regime results."""
    regime_dfs = [
        _process_regime(
            internal_regime=internal_regimes[name],
            regime_results=raw_results[name],
            regime_states=metadata.regime_to_states[name],
            regime_actions=metadata.regime_to_actions[name],
            params=params[name],
            additional_targets=additional_targets,
        )
        for name in metadata.regime_names
        if raw_results[name]
    ]

    return _assemble_dataframe(
        regime_dfs=regime_dfs,
        state_names=metadata.state_names,
        action_names=metadata.action_names,
    )


def _process_regime(
    internal_regime: InternalRegime,
    regime_results: dict[int, PeriodRegimeData],
    regime_states: tuple[str, ...],
    regime_actions: tuple[str, ...],
    params: ParamsDict,
    additional_targets: list[str] | None,
) -> pd.DataFrame:
    """Process results for a single regime into a DataFrame."""
    # Build period data
    period_dicts = [
        _extract_period_data(result, period, regime_states, regime_actions)
        for period, result in regime_results.items()
    ]

    # Concatenate and filter to in-regime subjects
    data = _concatenate_and_filter(period_dicts)

    # Add regime name
    data["regime"] = [internal_regime.name] * len(data["period"])

    # Compute additional targets
    if additional_targets:
        targets_for_regime = _filter_targets_for_regime(
            additional_targets, internal_regime
        )
        if targets_for_regime:
            target_values = _compute_targets(
                data, targets_for_regime, internal_regime, params
            )
            data.update(target_values)

    return pd.DataFrame(data)


def _extract_period_data(
    result: PeriodRegimeData,
    period: int,
    regime_states: tuple[str, ...],
    regime_actions: tuple[str, ...],
) -> dict[str, Array]:
    """Extract data from a single period's simulation results."""
    data: dict[str, Array] = {
        "period": jnp.full_like(result.in_regime, period, dtype=jnp.int32),
        "subject_id": jnp.arange(len(result.in_regime)),
        "_in_regime": result.in_regime,
        "value": result.V_arr,
    }

    for name in regime_states:
        if name in result.states:
            data[name] = result.states[name]

    for name in regime_actions:
        if name in result.actions:
            data[name] = result.actions[name]

    return data


def _concatenate_and_filter(period_dicts: list[dict[str, Array]]) -> dict[str, Any]:
    """Concatenate period data and filter to in-regime subjects."""
    keys = [k for k in period_dicts[0] if k != "_in_regime"]

    concatenated = {
        key: jnp.concatenate([d[key] for d in period_dicts]) for key in period_dicts[0]
    }

    mask = concatenated["_in_regime"].astype(bool)
    return {key: concatenated[key][mask] for key in keys}


def _filter_targets_for_regime(
    targets: list[str],
    internal_regime: InternalRegime,
) -> list[str]:
    """Filter targets to only those available in this regime."""
    available = _get_available_targets_for_regime(internal_regime)
    return [t for t in targets if t in available]


def _assemble_dataframe(
    regime_dfs: list[pd.DataFrame],
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> pd.DataFrame:
    """Combine regime DataFrames, add missing columns, reorder, and sort."""
    if not regime_dfs:
        return _empty_dataframe(state_names, action_names)

    df = pd.concat(regime_dfs, ignore_index=True)
    df = _add_missing_columns(df, state_names, action_names)
    df = _reorder_columns(df, state_names, action_names)
    return df.sort_values(["subject_id", "period"]).reset_index(drop=True)


def _empty_dataframe(
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> pd.DataFrame:
    """Create empty DataFrame with correct columns."""
    columns = ["period", "subject_id", "regime", "value"]
    columns.extend(state_names)
    columns.extend(action_names)
    return pd.DataFrame(columns=columns)


def _add_missing_columns(
    df: pd.DataFrame,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> pd.DataFrame:
    """Add NaN columns for states/actions not present in DataFrame."""
    for name in state_names:
        if name not in df.columns:
            df[name] = jnp.nan
    for name in action_names:
        if name not in df.columns:
            df[name] = jnp.nan
    return df


def _reorder_columns(
    df: pd.DataFrame,
    state_names: tuple[str, ...],
    action_names: tuple[str, ...],
) -> pd.DataFrame:
    """Reorder columns: period, subject_id, regime, value, states, actions, rest."""
    base = ["period", "subject_id", "regime", "value"]
    known = set(base) | set(state_names) | set(action_names)
    rest = [c for c in df.columns if c not in known]
    return df[base + list(state_names) + list(action_names) + rest]


# ======================================================================================
# Categorical conversion
# ======================================================================================


def _convert_to_categorical(
    df: pd.DataFrame,
    metadata: SimulationMetadata,
) -> pd.DataFrame:
    """Convert discrete columns to pandas Categorical dtype with string labels.

    Converts:
    - regime column: uses regime_names as categories
    - discrete state/action columns: uses categories from DiscreteGrid

    """
    df = df.copy()

    # Convert regime column
    df["regime"] = pd.Categorical(df["regime"], categories=metadata.regime_names)

    # Convert discrete state and action columns
    for var_name, categories in metadata.discrete_categories.items():
        if var_name in df.columns:
            df[var_name] = _codes_to_categorical(df[var_name], categories)

    return df


def _codes_to_categorical(
    codes: pd.Series,
    categories: tuple[str, ...],
) -> pd.Categorical | pd.Series:
    """Convert integer codes to Categorical, handling NaN and out-of-range values.

    If values are outside the valid category range, returns the original series
    unchanged to avoid data loss.

    """
    codes_array = codes.to_numpy()
    has_nan = pd.isna(codes_array)
    n_categories = len(categories)

    # Check for out-of-range values (excluding NaN)
    valid_values = codes_array[~has_nan]
    if len(valid_values) > 0:
        int_values = valid_values.astype(int)
        if int_values.min() < 0 or int_values.max() >= n_categories:
            # Values outside valid range - return original series unchanged
            return codes

    if has_nan.any():
        # Use -1 for NaN positions (will become NaN in Categorical)
        int_codes = pd.array(
            [-1 if pd.isna(c) else int(c) for c in codes_array],
            dtype="Int64",
        )
        return pd.Categorical.from_codes(
            int_codes,  # type: ignore[arg-type]
            categories=list(categories),  # type: ignore[arg-type]
        )

    return pd.Categorical.from_codes(
        codes_array.astype(int),  # type: ignore[arg-type]
        categories=list(categories),  # type: ignore[arg-type]
    )


# ======================================================================================
# Target computation
# ======================================================================================


def _compute_targets(
    data: dict[str, Any],
    targets: list[str],
    internal_regime: InternalRegime,
    params: ParamsDict,
) -> dict[str, Array]:
    """Compute additional targets for a regime."""
    functions_pool = _build_functions_pool(internal_regime)
    target_func = _create_target_function(functions_pool, targets)
    variables = _get_function_variables(target_func)
    vectorized_func = vmap_1d(target_func, variables=variables)
    kwargs = {k: jnp.asarray(v) for k, v in data.items() if k in variables}
    result = vectorized_func(params=params, **kwargs)
    # Squeeze any (n, 1) shaped arrays to (n,)
    return {k: jnp.squeeze(v) for k, v in result.items()}


def _build_functions_pool(internal_regime: InternalRegime) -> dict[str, Any]:
    """Build pool of available functions for target computation."""
    pool: dict[str, Any] = {
        **internal_regime.functions,
        **internal_regime.constraints,
        "utility": internal_regime.utility,
    }
    if internal_regime.regime_transition_probs is not None:
        pool["regime_transition_probs"] = (
            internal_regime.regime_transition_probs.simulate
        )
    return pool


def _create_target_function(
    functions_pool: dict[str, Any],
    targets: list[str],
) -> Any:  # noqa: ANN401
    """Create combined function for computing targets."""
    return concatenate_functions(
        functions=functions_pool,
        targets=targets,
        return_type="dict",
        set_annotations=True,
    )


def _get_function_variables(func: Any) -> tuple[str, ...]:  # noqa: ANN401
    """Get variable names from function signature, excluding 'params'."""
    return tuple(p for p in inspect.signature(func).parameters if p != "params")
