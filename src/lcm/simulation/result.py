"""Simulation result object with deferred DataFrame computation."""

import contextlib
import inspect
import pickle
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import jax.numpy as jnp
import pandas as pd
from dags import concatenate_functions
from jax import Array

from lcm.ages import AgeGrid
from lcm.dispatchers import vmap_1d
from lcm.exceptions import InvalidAdditionalTargetsError
from lcm.grids import DiscreteGrid
from lcm.input_processing.util import is_stochastic_transition
from lcm.interfaces import InternalRegime, PeriodRegimeSimulationData
from lcm.typing import FlatRegimeParams, FloatND, InternalParams, RegimeName
from lcm.utils import flatten_regime_namespace

CLOUDPICKLE_IMPORT_ERROR_MSG = (
    "Pickling SimulationResult objects requires the optional dependency 'cloudpickle'. "
    "Install it with: `pixi/uv add cloudpickle` (or add it to your project deps)."
)


# ======================================================================================
# Main result class
# ======================================================================================


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
        V_arr_dict: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
        ages: AgeGrid,
    ) -> None:
        self._raw_results = raw_results
        self._internal_regimes = internal_regimes
        self._internal_params = internal_params
        self._V_arr_dict = V_arr_dict
        self._ages = ages
        self._metadata = _compute_metadata(
            internal_regimes=internal_regimes, raw_results=raw_results
        )
        self._available_targets = sorted(
            _collect_all_available_targets(internal_regimes)
        )

    # ----------------------------------------------------------------------------------
    # Public properties for advanced users
    # ----------------------------------------------------------------------------------

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
    def V_arr_dict(
        self,
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
        """Value function arrays from the solution."""
        return self._V_arr_dict

    # ----------------------------------------------------------------------------------
    # Metadata properties (delegated to _metadata)
    # ----------------------------------------------------------------------------------

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

    # ----------------------------------------------------------------------------------
    # Main methods
    # ----------------------------------------------------------------------------------

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

        Note: This requires the optional dependency 'cloudpickle'.

        Args:
            path: File path to save the pickle.
            protocol: Int which indicates which protocol should be used by the pickler,
                default HIGHEST_PROTOCOL. The possible values are 0, 1, 2, 3, 4, 5. See
                https://docs.python.org/3/library/pickle.html.

        Returns:
            The path where the object was saved.

        """
        return _atomic_dump(self, path, protocol=protocol)

    @classmethod
    def from_pickle(cls, path: str | Path) -> SimulationResult:
        """Deserialize a SimulationResult from a pickle file.

        Note: This requires the optional dependency 'cloudpickle'.

        Args:
            path: File path to read the pickle from.

        Returns:
            The unpickled SimulationResult object.

        """
        try:
            import cloudpickle  # noqa: PLC0415
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(CLOUDPICKLE_IMPORT_ERROR_MSG) from e

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


# ======================================================================================
# Metadata
# ======================================================================================


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


def _compute_metadata(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    raw_results: MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
) -> SimulationMetadata:
    """Compute metadata from internal regimes and raw results."""
    regime_names = list(internal_regimes.keys())

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
        state_names=sorted(all_states),
        action_names=sorted(all_actions),
        n_periods=n_periods,
        n_subjects=n_subjects,
        regime_to_states=MappingProxyType(regime_to_states),
        regime_to_actions=MappingProxyType(regime_to_actions),
        discrete_categories=MappingProxyType(discrete_categories),
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


# ======================================================================================
# Target resolution and validation
# ======================================================================================


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
    return {
        name for name in regime.functions if name not in excluded
    } | regime.constraints.keys()


def _get_stochastic_weight_function_names(regime: InternalRegime) -> set[str]:
    """Get names of internal stochastic weight functions.

    These are functions named `weight_{transition_name}` that return probability arrays
    for stochastic state transitions. They should not be exposed as available targets.
    """
    flat_transitions = flatten_regime_namespace(regime.transitions)
    return {
        f"weight_{name}"
        for name, fn in flat_transitions.items()
        if is_stochastic_transition(fn)
    }


# ======================================================================================
# DataFrame creation
# ======================================================================================


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
    data = _concatenate_and_filter(period_dicts)

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


def _concatenate_and_filter(period_dicts: list[dict[str, Array]]) -> dict[str, Any]:
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


# ======================================================================================
# Categorical conversion
# ======================================================================================


def _convert_to_categorical(
    *,
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
            df[var_name] = _codes_to_categorical(
                codes=df[var_name], categories=categories
            )

    return df


def _codes_to_categorical(
    *,
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
        int_codes = [-1 if pd.isna(c) else int(c) for c in codes_array]
        return pd.Categorical.from_codes(
            int_codes,
            categories=pd.Index(categories),
        )

    return pd.Categorical.from_codes(
        codes_array.astype(int),
        categories=pd.Index(categories),
    )


# ======================================================================================
# Target computation
# ======================================================================================


def _compute_targets(
    *,
    data: dict[str, Any],
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


def _build_functions_pool(internal_regime: InternalRegime) -> dict[str, Any]:
    """Build pool of available functions for target computation."""
    pool: dict[str, Any] = {
        **{k: v for k, v in internal_regime.functions.items() if k != "H"},
        **internal_regime.constraints,
    }
    if internal_regime.regime_transition_probs is not None:
        pool["regime_transition_probs"] = (
            internal_regime.regime_transition_probs.simulate
        )
    return pool


def _create_target_function(
    *,
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


def _get_function_variables(
    *,
    func: Callable[..., Any],
    param_names: frozenset[str],
) -> tuple[str, ...]:
    """Get variable names from signature, excluding flat param names."""
    return tuple(p for p in inspect.signature(func).parameters if p not in param_names)


# ======================================================================================
# IO operations
# ======================================================================================


def _atomic_dump(obj: Any, path: str | Path, *, protocol: int) -> Path:  # noqa: ANN401
    """Serialize `obj` to `path` in an atomic (all-or-nothing) way.

    Args:
        obj: Object to serialize.
        path: File path to save the pickle.
        protocol: Int which indicates which protocol should be used by the pickler.
            The possible values are 0, 1, 2, 3, 4, 5. See
            https://docs.python.org/3/library/pickle.html.

    Returns:
        The path where the object was saved.

    """
    try:
        import cloudpickle  # noqa: PLC0415
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(CLOUDPICKLE_IMPORT_ERROR_MSG) from e

    p = Path(path)
    if not p.parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {p.parent}")

    tmp: Path | None = None
    try:
        # Write to a uniquely-named temp file in the *same directory* as the target.
        with tempfile.NamedTemporaryFile(mode="wb", dir=p.parent, delete=False) as f:
            tmp = Path(f.name)
            cloudpickle.dump(obj, f, protocol=protocol)

        # Atomic replace: after this line, readers either see the old file or the new
        # one, never a partially-written file. (Temp file is closed already, which
        # matters on Windows.)
        tmp.replace(p)
        return p
    finally:
        # If anything failed before the replace succeeded, delete the temp file. We used
        # delete=False so we can close the file before replacing (needed on Windows), so
        # the context manager will not auto-delete it for us.
        if tmp is not None:
            with contextlib.suppress(OSError):
                tmp.unlink()
