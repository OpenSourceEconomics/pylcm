"""Simulation result object with deferred DataFrame computation."""

import inspect
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import cloudpickle
import jax.numpy as jnp
import pandas as pd
from dags import concatenate_functions
from dags.tree import tree_path_from_qname
from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import InvalidAdditionalTargetsError
from lcm.grids import DiscreteGrid
from lcm.interfaces import InternalRegime, PeriodRegimeSimulationData
from lcm.persistence import atomic_dump
from lcm.regime import Regime
from lcm.regime_building.processing import compute_merged_discrete_categories
from lcm.typing import (
    FlatRegimeParams,
    FloatND,
    InternalParams,
    RegimeName,
    UserFunction,
)
from lcm.utils.dispatchers import vmap_1d
from lcm.utils.namespace import flatten_regime_namespace

CLOUDPICKLE_IMPORT_ERROR_MSG = (
    "Pickling SimulationResult objects requires the optional dependency 'cloudpickle'. "
    "Install it with: `pixi/uv add cloudpickle` (or add it to your project deps)."
)


class SimulationResult:
    """Result object from model simulation with deferred DataFrame computation."""

    def __init__(
        self,
        *,
        raw_results: MappingProxyType[
            str, MappingProxyType[int, PeriodRegimeSimulationData]
        ],
        internal_regimes: MappingProxyType[RegimeName, InternalRegime],
        internal_params: InternalParams,
        period_to_regime_to_V_arr: MappingProxyType[
            int, MappingProxyType[RegimeName, FloatND]
        ],
        ages: AgeGrid,
        simulation_output_dtypes: Mapping[str, pd.CategoricalDtype],
    ) -> None:
        self._raw_results = raw_results
        self._internal_regimes = internal_regimes
        self._internal_params = internal_params
        self._period_to_regime_to_V_arr = period_to_regime_to_V_arr
        self._ages = ages
        self._metadata = _compute_metadata(
            internal_regimes=internal_regimes,
            raw_results=raw_results,
            simulation_output_dtypes=simulation_output_dtypes,
            ages=ages,
        )
        self._available_targets = sorted(
            _collect_all_available_targets(internal_regimes)
        )

    @property
    def raw_results(
        self,
    ) -> MappingProxyType[str, MappingProxyType[int, PeriodRegimeSimulationData]]:
        """Raw simulation results by regime and period."""
        return self._raw_results

    @property
    def internal_params(self) -> InternalParams:
        """Model parameters used in simulation."""
        return self._internal_params

    @property
    def period_to_regime_to_V_arr(
        self,
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
        """Value function arrays from the solution."""
        return self._period_to_regime_to_V_arr

    @property
    def regime_names(self) -> list[str]:
        """Names of all regimes."""
        return self._metadata.regime_names

    @property
    def state_names(self) -> list[str]:
        """Names of all state variables (union across regimes)."""
        return self._metadata.state_names

    @property
    def action_names(self) -> list[str]:
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

    @property
    def available_targets(self) -> list[str]:
        """Names of all available additional targets.

        These can be passed to `to_dataframe(additional_targets=...)`. Includes utility
        functions, auxiliary functions, and constraints from all regimes.

        """
        return self._available_targets

    def to_dataframe(
        self,
        additional_targets: list[str] | Literal["all"] | None = None,
        *,
        use_labels: bool = True,
    ) -> pd.DataFrame:
        """Convert simulation results to a flat pandas DataFrame.

        Args:
            additional_targets: Targets to compute. Can be:
                - None (default): No additional targets
                - list[str]: Specific target names to compute
                - "all": Compute all available targets (see `available_targets`)
                Targets can be any function defined in a regime. Each target is
                computed for the regimes where it exists; rows from regimes without
                that target will have NaN.
            use_labels: If True (default), discrete variables (states, actions, and
                regime) are returned as pandas Categorical dtype with string labels.
                If False, discrete variables are returned as integer codes.

        Returns:
            DataFrame with simulation results.

        """
        resolved_targets = _resolve_targets(
            additional_targets=additional_targets,
            available_targets=self.available_targets,
        )

        df = _create_flat_dataframe(
            raw_results=self._raw_results,
            internal_regimes=self._internal_regimes,
            internal_params=self._internal_params,
            metadata=self._metadata,
            additional_targets=resolved_targets,
            ages=self._ages,
        )

        if use_labels:
            return _convert_to_categorical(df=df, metadata=self._metadata)

        return df

    def to_pickle(
        self,
        path: str | Path,
        *,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        """Serialize the SimulationResult to a file.

        Args:
            path: File path to save the pickle.
            protocol: Int which indicates which protocol should be used by the pickler,
                default HIGHEST_PROTOCOL. The possible values are 0, 1, 2, 3, 4, 5. See
                https://docs.python.org/3/library/pickle.html.

        Returns:
            The path where the object was saved.

        """
        return atomic_dump(self, path, protocol=protocol)

    @classmethod
    def from_pickle(cls, path: str | Path) -> SimulationResult:
        """Deserialize a SimulationResult from a pickle file.

        Args:
            path: File path to read the pickle from.

        Returns:
            The unpickled SimulationResult object.

        """
        p = Path(path)
        with p.open("rb") as f:
            obj = cloudpickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickle at {p} is {type(obj).__name__}, expected {cls.__name__}"
            )
        return obj

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


def get_simulation_output_dtypes(
    regimes: Mapping[str, Regime],
    regime_names_to_ids: Mapping[str, int],
) -> MappingProxyType[str, pd.CategoricalDtype]:
    """Compute pandas CategoricalDtype for all discrete output columns.

    Merge ordered categories across regimes via topological sort. This must be
    called after model validation (which guarantees merges succeed).

    Args:
        regimes: Mapping of regime names to Regime instances.
        regime_names_to_ids: Mapping of regime names to integer IDs.

    Returns:
        Immutable mapping of variable name to `pd.CategoricalDtype`. Includes
        all discrete state/action variables plus the `"regime"` column.

    """
    merged_categories, ordered_flags = compute_merged_discrete_categories(regimes)

    dtypes: dict[str, pd.CategoricalDtype] = {}
    for var_name, categories in merged_categories.items():
        dtypes[var_name] = pd.CategoricalDtype(
            categories=list(categories),
            ordered=ordered_flags[var_name],
        )

    dtypes["regime"] = pd.CategoricalDtype(
        categories=list(regime_names_to_ids.keys()),
        ordered=False,
    )

    return MappingProxyType(dtypes)


@dataclass(frozen=True)
class SimulationMetadata:
    """Pre-computed metadata about the simulation."""

    regime_names: list[str]
    """Names of all regimes in the model."""

    state_names: list[str]
    """Sorted union of state variable names across all regimes."""

    action_names: list[str]
    """Sorted union of action variable names across all regimes."""

    n_periods: int
    """Number of periods in the simulation."""

    n_subjects: int
    """Number of subjects simulated."""

    regime_to_states: MappingProxyType[str, tuple[str, ...]]
    """Immutable mapping of regime names to their state variable names."""

    regime_to_actions: MappingProxyType[str, tuple[str, ...]]
    """Immutable mapping of regime names to their action variable names."""

    discrete_categories: MappingProxyType[str, tuple[str, ...]]
    """Immutable mapping of discrete variable names to their category labels."""

    discrete_ordered: MappingProxyType[str, bool]
    """Immutable mapping of discrete variable names to their ordered flag."""

    regime_discrete_categories: MappingProxyType[tuple[str, str], tuple[str, ...]]
    """Immutable mapping of (regime_name, var_name) to per-regime categories."""


def _compute_metadata(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    raw_results: MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
    simulation_output_dtypes: Mapping[str, pd.CategoricalDtype],
    ages: AgeGrid,
) -> SimulationMetadata:
    """Compute metadata from internal regimes, raw results, and output dtypes."""
    regime_names = list(internal_regimes.keys())

    all_states: set[str] = set()
    all_actions: set[str] = set()
    regime_to_states: dict[str, tuple[str, ...]] = {}
    regime_to_actions: dict[str, tuple[str, ...]] = {}

    for regime_name, regime in internal_regimes.items():
        vi = regime.variable_info
        states = tuple(vi.query("is_state").index.tolist())
        actions = tuple(vi.query("is_action").index.tolist())
        regime_to_states[regime_name] = states
        regime_to_actions[regime_name] = actions
        all_states.update(states)
        all_actions.update(actions)

    # Extract categories and ordered flags from simulation_output_dtypes
    discrete_categories: dict[str, tuple[str, ...]] = {}
    discrete_ordered: dict[str, bool] = {}
    for var_name, dtype in simulation_output_dtypes.items():
        if var_name == "regime":
            continue
        discrete_categories[var_name] = tuple(dtype.categories)
        discrete_ordered[var_name] = bool(dtype.ordered)

    # Per-regime discrete categories for correct code→label mapping
    regime_discrete_categories: dict[tuple[str, str], tuple[str, ...]] = {}
    for regime_name, regime in internal_regimes.items():
        for var_name, grid in regime.grids.items():
            if isinstance(grid, DiscreteGrid):
                regime_discrete_categories[(regime_name, var_name)] = grid.categories

    n_periods = ages.n_periods
    n_subjects = _get_n_subjects(raw_results)

    return SimulationMetadata(
        regime_names=regime_names,
        state_names=sorted(all_states),
        action_names=sorted(all_actions),
        n_periods=n_periods,
        n_subjects=n_subjects,
        regime_to_states=MappingProxyType(regime_to_states),
        regime_to_actions=MappingProxyType(regime_to_actions),
        discrete_categories=MappingProxyType(discrete_categories),
        discrete_ordered=MappingProxyType(discrete_ordered),
        regime_discrete_categories=MappingProxyType(regime_discrete_categories),
    )


def _get_n_subjects(
    raw_results: MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
) -> int:
    """Extract number of subjects from raw results."""
    for regime_results in raw_results.values():
        if regime_results:
            first_result = next(iter(regime_results.values()))
            return len(first_result.in_regime)
    return 0


def _resolve_targets(
    *,
    additional_targets: list[str] | Literal["all"] | None,
    available_targets: list[str],
) -> list[str] | None:
    """Resolve and validate additional targets.

    Args:
        additional_targets: User-provided targets specification.
        available_targets: List of all available target names.

    Returns:
        Resolved list of target names, or None if no targets requested.

    Raises:
        InvalidAdditionalTargetsError: If any target is not available.

    """
    if additional_targets is None:
        return None
    if additional_targets == "all":
        return available_targets

    # Validate user-provided targets
    invalid = set(additional_targets) - set(available_targets)
    if invalid:
        raise InvalidAdditionalTargetsError(
            f"Targets {invalid} not found in any regime. "
            f"Available targets: {available_targets}"
        )

    return additional_targets


def _collect_all_available_targets(
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
) -> set[str]:
    """Collect all available target names across all regimes."""
    all_targets: set[str] = set()
    for regime in internal_regimes.values():
        all_targets.update(_get_available_targets_for_regime(regime))
    return all_targets


def _get_available_targets_for_regime(regime: InternalRegime) -> set[str]:
    """Get available target names for a single regime."""
    excluded = {"H"} | _get_stochastic_weight_function_names(regime)
    sim = regime.simulate_functions
    return {
        name for name in sim.functions if name not in excluded
    } | sim.constraints.keys()


def _get_stochastic_weight_function_names(regime: InternalRegime) -> set[str]:
    """Get names of internal stochastic weight functions.

    These are functions named `weight_{transition_name}` that return probability arrays
    for stochastic state transitions. They should not be exposed as available targets.
    """
    stochastic_transition_names = regime.simulate_functions.stochastic_transition_names
    flat_transitions = flatten_regime_namespace(regime.simulate_functions.transitions)
    return {
        f"weight_{name}"
        for name in flat_transitions
        if tree_path_from_qname(name)[-1] in stochastic_transition_names
    }


def _create_flat_dataframe(
    *,
    raw_results: MappingProxyType[
        str, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    metadata: SimulationMetadata,
    additional_targets: list[str] | None,
    ages: AgeGrid,
) -> pd.DataFrame:
    """Create a single flat DataFrame from all regime results."""
    regime_dfs = [
        _process_regime(
            internal_regime=internal_regimes[name],
            regime_results=raw_results[name],
            regime_states=metadata.regime_to_states[name],
            regime_actions=metadata.regime_to_actions[name],
            regime_params=internal_params[name],
            additional_targets=additional_targets,
            ages=ages,
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
    *,
    internal_regime: InternalRegime,
    regime_results: MappingProxyType[int, PeriodRegimeSimulationData],
    regime_states: tuple[str, ...],
    regime_actions: tuple[str, ...],
    regime_params: FlatRegimeParams,
    additional_targets: list[str] | None,
    ages: AgeGrid,
) -> pd.DataFrame:
    """Process results for a single regime into a DataFrame."""
    # Build period data
    period_dicts = [
        _extract_period_data(
            result=result,
            period=period,
            regime_states=regime_states,
            regime_actions=regime_actions,
        )
        for period, result in regime_results.items()
    ]

    # Concatenate and filter to in-regime subjects
    data: dict[str, Array | Sequence[str]] = _concatenate_and_filter(period_dicts)  # ty: ignore[invalid-assignment]

    # Add age column (computed from period using ages grid)
    data["age"] = ages.values[data["period"]]  # noqa: PD011

    # Add regime name
    data["regime"] = [internal_regime.name] * len(data["period"])

    # Compute additional targets
    if additional_targets:
        targets_for_regime = _filter_targets_for_regime(
            targets=additional_targets, internal_regime=internal_regime
        )
        if targets_for_regime:
            target_values = _compute_targets(
                data=data,
                targets=targets_for_regime,
                internal_regime=internal_regime,
                regime_params=regime_params,
            )
            data.update(target_values)

    return pd.DataFrame(data)


def _extract_period_data(
    *,
    result: PeriodRegimeSimulationData,
    period: int,
    regime_states: tuple[str, ...],
    regime_actions: tuple[str, ...],
) -> dict[str, Array]:
    """Extract data from a single period's simulation results."""
    data: dict[str, Array] = {
        "subject_id": jnp.arange(len(result.in_regime)),
        "period": jnp.full_like(result.in_regime, period, dtype=jnp.int32),
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


def _concatenate_and_filter(period_dicts: list[dict[str, Array]]) -> dict[str, Array]:
    """Concatenate period data and filter to in-regime subjects."""
    keys = [k for k in period_dicts[0] if k != "_in_regime"]

    concatenated = {
        key: jnp.concatenate([d[key] for d in period_dicts]) for key in period_dicts[0]
    }

    mask = concatenated["_in_regime"].astype(bool)
    return {key: concatenated[key][mask] for key in keys}


def _filter_targets_for_regime(
    *,
    targets: list[str],
    internal_regime: InternalRegime,
) -> list[str]:
    """Filter targets to only those available in this regime."""
    available = _get_available_targets_for_regime(internal_regime)
    return [t for t in targets if t in available]


def _assemble_dataframe(
    *,
    regime_dfs: list[pd.DataFrame],
    state_names: list[str],
    action_names: list[str],
) -> pd.DataFrame:
    """Combine regime DataFrames, add missing columns, reorder, and sort."""
    if not regime_dfs:
        return _empty_dataframe(state_names=state_names, action_names=action_names)

    df = pd.concat(regime_dfs, ignore_index=True)
    df = _add_missing_columns(df=df, state_names=state_names, action_names=action_names)
    df = _reorder_columns(df=df, state_names=state_names, action_names=action_names)
    return df.sort_values(["subject_id", "period"]).reset_index(drop=True)


def _empty_dataframe(
    *,
    state_names: list[str],
    action_names: list[str],
) -> pd.DataFrame:
    """Create empty DataFrame with correct columns."""
    columns = ["subject_id", "period", "regime", "value"]
    columns.extend(state_names)
    columns.extend(action_names)
    return pd.DataFrame(columns=pd.Index(columns))


def _add_missing_columns(
    *,
    df: pd.DataFrame,
    state_names: list[str],
    action_names: list[str],
) -> pd.DataFrame:
    """Add NaN columns for states/actions not present in DataFrame."""
    for name in state_names:
        if name not in df.columns:
            df[name] = float("nan")
    for name in action_names:
        if name not in df.columns:
            df[name] = float("nan")
    return df


def _reorder_columns(
    *,
    df: pd.DataFrame,
    state_names: list[str],
    action_names: list[str],
) -> pd.DataFrame:
    """Reorder columns: subject_id, period, regime, value, states, actions, rest."""
    base = ["subject_id", "period", "regime", "value"]
    known = set(base) | set(state_names) | set(action_names)
    rest = [c for c in df.columns if c not in known]
    return df[base + state_names + action_names + rest]


def _convert_to_categorical(
    *,
    df: pd.DataFrame,
    metadata: SimulationMetadata,
) -> pd.DataFrame:
    """Convert discrete columns to pandas Categorical dtype with string labels.

    Converts:
    - regime column: uses regime_names as categories
    - discrete state/action columns: uses categories from simulation metadata

    """
    df = df.copy()

    # Convert regime column
    df["regime"] = pd.Categorical(df["regime"], categories=metadata.regime_names)

    # Convert discrete state and action columns
    for var_name, merged_categories in metadata.discrete_categories.items():
        if var_name not in df.columns:
            continue

        # Check if any regime has different categories than merged
        needs_remap = any(
            metadata.regime_discrete_categories.get((rn, var_name)) != merged_categories
            for rn in metadata.regime_names
            if (rn, var_name) in metadata.regime_discrete_categories
        )

        if needs_remap:
            df[var_name] = _remap_codes_per_regime(
                df=df,
                var_name=var_name,
                merged_categories=merged_categories,
                ordered=metadata.discrete_ordered[var_name],
                metadata=metadata,
            )
        else:
            df[var_name] = _codes_to_categorical(
                codes=df[var_name],
                categories=merged_categories,
                ordered=metadata.discrete_ordered[var_name],
            )

    return df


def _remap_codes_per_regime(
    *,
    df: pd.DataFrame,
    var_name: str,
    merged_categories: tuple[str, ...],
    ordered: bool,
    metadata: SimulationMetadata,
) -> pd.Categorical:
    """Map per-regime integer codes to labels, then build a merged Categorical.

    When regimes define different categories for the same variable, the raw integer
    codes in the DataFrame correspond to each regime's own category ordering. This
    function converts per-regime codes to string labels, then wraps them in a
    Categorical with the merged category set.

    """
    labels = pd.array(  # ty: ignore[no-matching-overload]
        [pd.NA] * len(df), dtype="string"
    )

    for regime_name in metadata.regime_names:
        regime_cats = metadata.regime_discrete_categories.get((regime_name, var_name))
        if regime_cats is None:
            continue

        mask = df["regime"] == regime_name
        if not mask.any():
            continue

        codes_in_regime = df.loc[mask, var_name]
        valid = codes_in_regime.notna()
        int_codes = codes_in_regime[valid].astype(int)
        mapped = int_codes.map(dict(enumerate(regime_cats))).to_numpy()
        labels[mask & valid] = mapped

    return pd.Categorical(labels, categories=list(merged_categories), ordered=ordered)


def _codes_to_categorical(
    *,
    codes: pd.Series,
    categories: tuple[str, ...],
    ordered: bool = False,
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
        int_codes = [-1 if pd.isna(c) else int(c) for c in codes_array]
        return pd.Categorical.from_codes(
            int_codes,
            categories=pd.Index(categories),
            ordered=ordered,
        )

    return pd.Categorical.from_codes(
        codes_array.astype(int),
        categories=pd.Index(categories),
        ordered=ordered,
    )


def _compute_targets(
    *,
    data: dict[str, Array | Sequence[str]],
    targets: list[str],
    internal_regime: InternalRegime,
    regime_params: FlatRegimeParams,
) -> dict[str, Array]:
    """Compute additional targets for a regime."""
    functions_pool = _build_functions_pool(internal_regime)
    target_func = _create_target_function(
        functions_pool=functions_pool, targets=targets
    )
    # Merge resolved fixed params with runtime params so that the target
    # function (built from raw user functions) receives all needed arguments.
    all_params = {**internal_regime.resolved_fixed_params, **regime_params}
    flat_param_names = frozenset(all_params.keys())
    variables = _get_function_variables(func=target_func, param_names=flat_param_names)
    vectorized_func = vmap_1d(func=target_func, variables=variables)
    kwargs = {k: jnp.asarray(v) for k, v in data.items() if k in variables}
    result = vectorized_func(**all_params, **kwargs)
    # Squeeze any (n, 1) shaped arrays to (n,)
    return {k: jnp.squeeze(v) for k, v in result.items()}


def _build_functions_pool(internal_regime: InternalRegime) -> dict[str, UserFunction]:
    """Build pool of available functions for target computation."""
    sim = internal_regime.simulate_functions
    pool: dict[str, UserFunction] = {
        **{k: v for k, v in sim.functions.items() if k != "H"},
        **sim.constraints,
    }
    if sim.compute_regime_transition_probs is not None:
        pool["regime_transition_probs"] = sim.compute_regime_transition_probs
    return pool


def _create_target_function(
    *,
    functions_pool: dict[str, UserFunction],
    targets: list[str],
) -> UserFunction:
    """Create combined function for computing targets."""
    return concatenate_functions(
        functions=functions_pool,
        targets=targets,
        return_type="dict",
        set_annotations=True,
    )


def _get_function_variables(
    *,
    func: Callable[..., Any],
    param_names: frozenset[str],
) -> tuple[str, ...]:
    """Get variable names from signature, excluding flat param names."""
    return tuple(p for p in inspect.signature(func).parameters if p not in param_names)
