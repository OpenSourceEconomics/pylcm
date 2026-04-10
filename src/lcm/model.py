"""Collection of classes that are used by the user to define the model and grids."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import InvalidParamsError, InvalidValueFunctionError
from lcm.grids import DiscreteGrid
from lcm.model_processing import (
    build_regimes_and_template,
    validate_model_inputs,
)
from lcm.pandas_utils import (
    convert_series_in_params,
    has_series,
    initial_conditions_from_dataframe,
)
from lcm.params import MappingLeaf, SequenceLeaf
from lcm.params.processing import (
    process_params,
)
from lcm.persistence import (
    save_simulate_snapshot,
    save_solve_snapshot,
)
from lcm.regime import Regime
from lcm.regime_building.processing import InternalRegime
from lcm.simulation.initial_conditions import validate_initial_conditions
from lcm.simulation.result import SimulationResult, get_simulation_output_dtypes
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve
from lcm.typing import (
    FloatND,
    InternalParams,
    ParamsTemplate,
    RegimeName,
    RegimeNamesToIds,
    UserFacingParamsTemplate,
    UserParams,
)
from lcm.utils.containers import (
    ensure_containers_are_immutable,
    ensure_containers_are_mutable,
    get_field_names_and_values,
)
from lcm.utils.error_handling import validate_regime_transitions_all_periods
from lcm.utils.logging import LogLevel, get_logger


class Model:
    """A model which is created from a regime.

    Upon initialization, internal regimes will be created which contain all
    the functions needed to solve and simulate the model.

    """

    description: str | None = None
    """Description of the model."""

    ages: AgeGrid
    """Age grid for the model."""

    n_periods: int
    """Number of periods in the model."""

    regime_names_to_ids: RegimeNamesToIds
    """Immutable mapping from regime names to integer indices."""

    regimes: MappingProxyType[str, Regime]
    """Immutable mapping of regime names to user `Regime` instances."""

    internal_regimes: MappingProxyType[RegimeName, InternalRegime]
    """Immutable mapping of regime names to internal regime instances."""

    enable_jit: bool = True
    """Whether to JIT-compile the functions of the internal regime."""

    fixed_params: UserParams
    """Parameters fixed at model initialization."""

    _params_template: ParamsTemplate
    """Template for the model parameters."""

    def __init__(
        self,
        *,
        description: str = "",
        ages: AgeGrid,
        regimes: Mapping[str, Regime],
        regime_id_class: type,
        enable_jit: bool = True,
        fixed_params: UserParams = MappingProxyType({}),
    ) -> None:
        """Initialize the Model.

        Args:
            regimes: Mapping of regime names to Regime instances.
            ages: Age grid for the model.
            description: Description of the model.
            regime_id_class: Dataclass mapping regime names to integer indices.
            enable_jit: Whether to jit the functions of the internal regime.
            fixed_params: Parameters that can be fixed at model initialization.

        """
        self.description = description
        self.ages = ages
        self.n_periods = ages.n_periods
        self.fixed_params = ensure_containers_are_immutable(fixed_params)

        validate_model_inputs(
            n_periods=self.n_periods,
            regimes=regimes,
            regime_id_class=regime_id_class,
        )
        self.regime_names_to_ids = MappingProxyType(
            dict(
                sorted(
                    get_field_names_and_values(regime_id_class).items(),
                    key=lambda x: x[1],
                )
            )
        )
        self.regimes = MappingProxyType(dict(regimes))

        def _convert_and_validate_fixed(
            internal_params: InternalParams,
        ) -> InternalParams:
            converted = _maybe_convert_series(
                internal_params, model=self, derived_categoricals=None
            )
            _validate_param_types(converted)
            return converted

        self.internal_regimes, self._params_template = build_regimes_and_template(
            regimes=regimes,
            ages=self.ages,
            regime_names_to_ids=self.regime_names_to_ids,
            enable_jit=enable_jit,
            fixed_params=self.fixed_params,
            convert_fixed_params=_convert_and_validate_fixed,
        )
        self.enable_jit = enable_jit
        self.simulation_output_dtypes = get_simulation_output_dtypes(
            regimes=self.regimes,
            regime_names_to_ids=self.regime_names_to_ids,
        )

    def get_params_template(self) -> UserFacingParamsTemplate:
        """Get a human-readable params template.

        Return a nested dict showing which parameters each function in each
        regime expects.

        """
        mutable = ensure_containers_are_mutable(self._params_template)
        return {
            regime: {
                func: {
                    param: getattr(typ, "__name__", str(typ))
                    for param, typ in params.items()
                }
                for func, params in funcs.items()
            }
            for regime, funcs in mutable.items()
        }

    def solve(
        self,
        *,
        params: UserParams,
        derived_categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]]
        | None = None,
        max_compilation_workers: int | None = None,
        log_level: LogLevel = "progress",
        log_path: str | Path | None = None,
        log_keep_n_latest: int = 3,
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
        """Solve the model using the pre-computed functions.

        Args:
            params: Model parameters compatible with `get_params_template()`.
                Parameters can be provided at exactly one of three levels:
                - Model level: {"arg_0": 0.0} - propagates to all functions needing
                  arg_0
                - Regime level: {"regime_0": {"arg_0": 0.0}} - propagates within
                  regime_0
                - Function level: {"regime_0": {"func": {"arg_0": 0.0}}} - direct
                  specification
                Values may be `pd.Series` with labeled indices; they are
                auto-converted to JAX arrays.
            derived_categoricals: Extra categorical mappings (level name to
                `DiscreteGrid`) for derived variables not in the model's
                state/action grids. Pass per-regime mappings as
                `{"var": {"regime_a": grid_a, ...}}`.
            max_compilation_workers: Maximum number of threads for parallel XLA
                compilation. Defaults to `os.cpu_count()`. Lower this on machines
                with limited RAM, as each concurrent compilation holds an XLA HLO
                graph in memory.
            log_level: Logging verbosity. `"off"` suppresses output, `"warning"` shows
                NaN/Inf warnings, `"progress"` adds timing, `"debug"` adds stats and
                requires `log_path`.
            log_path: Directory for persisting debug snapshots. Required when
                `log_level="debug"`.
            log_keep_n_latest: Maximum number of debug snapshots to keep on disk.

        Returns:
            Immutable mapping of period to a value function array for each regime.

        """
        _validate_log_args(log_level=log_level, log_path=log_path)
        internal_params = process_params(
            params=params, params_template=self._params_template
        )
        internal_params = _maybe_convert_series(
            internal_params, model=self, derived_categoricals=derived_categoricals
        )
        _validate_param_types(internal_params)
        validate_regime_transitions_all_periods(
            internal_regimes=self.internal_regimes,
            internal_params=internal_params,
            ages=self.ages,
        )
        try:
            period_to_regime_to_V_arr = solve(
                internal_params=internal_params,
                ages=self.ages,
                internal_regimes=self.internal_regimes,
                logger=get_logger(log_level=log_level),
                max_compilation_workers=max_compilation_workers,
                regimes=self.regimes,
                regime_id_class=self.regime_id_class,
                fixed_params=self.fixed_params,
                enable_jit=self.enable_jit,
            )
        except InvalidValueFunctionError as exc:
            if log_path is not None and exc.partial_solution is not None:
                save_solve_snapshot(
                    model=self,
                    params=params,
                    period_to_regime_to_V_arr=exc.partial_solution,  # ty: ignore[invalid-argument-type]
                    log_path=Path(log_path),
                    log_keep_n_latest=log_keep_n_latest,
                )
            raise
        if log_level == "debug" and log_path is not None:
            save_solve_snapshot(
                model=self,
                params=params,
                period_to_regime_to_V_arr=period_to_regime_to_V_arr,
                log_path=Path(log_path),
                log_keep_n_latest=log_keep_n_latest,
            )
        return period_to_regime_to_V_arr

    def simulate(
        self,
        *,
        params: UserParams,
        derived_categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]]
        | None = None,
        initial_conditions: Mapping[str, Array],
        period_to_regime_to_V_arr: MappingProxyType[
            int, MappingProxyType[RegimeName, FloatND]
        ]
        | None,
        check_initial_conditions: bool = True,
        seed: int | None = None,
        log_level: LogLevel = "progress",
        log_path: str | Path | None = None,
        log_keep_n_latest: int = 3,
    ) -> SimulationResult:
        """Simulate the model forward, optionally solving first.

        When `period_to_regime_to_V_arr` is `None`, the model is solved before
        simulating. Pass pre-computed value functions from `solve()` to skip the
        solve step.

        Args:
            params: Model parameters compatible with `get_params_template()`.
                Parameters can be provided at exactly one of three levels:
                - Model level: {"arg_0": 0.0} - propagates to all functions needing
                  arg_0
                - Regime level: {"regime_0": {"arg_0": 0.0}} - propagates within
                  regime_0
                - Function level: {"regime_0": {"func": {"arg_0": 0.0}}} - direct
                  specification
                Values may be `pd.Series` with labeled indices; they are
                auto-converted to JAX arrays.
            derived_categoricals: Extra categorical mappings (level name to
                `DiscreteGrid`) for derived variables not in the model's
                state/action grids. Pass per-regime mappings as
                `{"var": {"regime_a": grid_a, ...}}`.
            initial_conditions: Mapping of state names (plus `"regime"`) to arrays.
                All arrays must have the same length (number of subjects). The
                `"regime"` entry must contain integer regime codes (from
                `model.regime_names_to_ids`). May also be a `pd.DataFrame`
                with a `"regime"` column (auto-converted).
            period_to_regime_to_V_arr: Value function arrays from `solve()`.
                When `None`, the model is solved automatically before simulating.
            check_initial_conditions: Whether to validate initial conditions.
            seed: Random seed.
            log_level: Logging verbosity. `"off"` suppresses output, `"warning"` shows
                NaN/Inf warnings, `"progress"` adds timing, `"debug"` adds stats and
                requires `log_path`.
            log_path: Directory for persisting debug snapshots. Required when
                `log_level="debug"`.
            log_keep_n_latest: Maximum number of debug snapshots to keep on disk.

        Returns:
            SimulationResult object. Call .to_dataframe() to get a pandas DataFrame,
            optionally with additional_targets.

        """
        _validate_log_args(log_level=log_level, log_path=log_path)
        initial_conditions = _maybe_convert_dataframe(initial_conditions, model=self)
        internal_params = process_params(
            params=params, params_template=self._params_template
        )
        internal_params = _maybe_convert_series(
            internal_params, model=self, derived_categoricals=derived_categoricals
        )
        _validate_param_types(internal_params)
        if check_initial_conditions:
            validate_initial_conditions(
                initial_conditions=initial_conditions,
                internal_regimes=self.internal_regimes,
                regime_names_to_ids=self.regime_names_to_ids,
                internal_params=internal_params,
                ages=self.ages,
            )
        validate_regime_transitions_all_periods(
            internal_regimes=self.internal_regimes,
            internal_params=internal_params,
            ages=self.ages,
        )
        log = get_logger(log_level=log_level)
        if period_to_regime_to_V_arr is None:
            try:
                period_to_regime_to_V_arr = solve(
                    internal_params=internal_params,
                    ages=self.ages,
                    internal_regimes=self.internal_regimes,
                    logger=log,
                )
            except InvalidValueFunctionError as exc:
                if log_path is not None and exc.partial_solution is not None:
                    save_solve_snapshot(
                        model=self,
                        params=params,
                        period_to_regime_to_V_arr=exc.partial_solution,  # ty: ignore[invalid-argument-type]
                        log_path=Path(log_path),
                        log_keep_n_latest=log_keep_n_latest,
                    )
                raise
        result = simulate(
            internal_params=internal_params,
            initial_conditions=initial_conditions,
            internal_regimes=self.internal_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            logger=log,
            period_to_regime_to_V_arr=period_to_regime_to_V_arr,
            ages=self.ages,
            simulation_output_dtypes=self.simulation_output_dtypes,
            seed=seed,
        )
        if log_level == "debug" and log_path is not None:
            save_simulate_snapshot(
                model=self,
                params=params,
                initial_conditions=initial_conditions,
                period_to_regime_to_V_arr=period_to_regime_to_V_arr,
                result=result,
                log_path=Path(log_path),
                log_keep_n_latest=log_keep_n_latest,
            )
        return result


def _maybe_convert_series(
    internal_params: InternalParams,
    *,
    model: Model,
    derived_categoricals: Mapping[str, DiscreteGrid | Mapping[str, DiscreteGrid]]
    | None,
) -> InternalParams:
    """Convert pd.Series leaves in params to JAX arrays if any are present."""
    if derived_categoricals is not None or has_series(internal_params):
        return convert_series_in_params(
            internal_params=internal_params,
            model=model,
            derived_categoricals=derived_categoricals,
        )
    return internal_params


def _validate_param_types(internal_params: InternalParams) -> None:
    """Raise if any param leaf is not a Python scalar or JAX array.

    After processing, every leaf value (including inside MappingLeaf /
    SequenceLeaf containers) must be a Python scalar (float, int, bool) or a
    JAX array. Notably, numpy arrays and pandas Series are not accepted.
    """
    for regime_name, regime_params in internal_params.items():
        for key, value in regime_params.items():
            _check_leaf(value, f"{regime_name}__{key}")


def _check_leaf(value: object, path: str) -> None:
    """Check a single leaf value, recursing into MappingLeaf/SequenceLeaf."""
    if isinstance(value, MappingLeaf):
        for k, v in value.data.items():
            _check_leaf(v, f"{path}.{k}")
        return
    if isinstance(value, SequenceLeaf):
        for i, v in enumerate(value.data):
            _check_leaf(v, f"{path}[{i}]")
        return
    if isinstance(value, (float, int, bool)):
        return
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        if isinstance(value, Array):
            return
        type_name = type(value).__module__ + "." + type(value).__name__
        msg = (
            f"Parameter '{path}' is a {type_name} (shape {value.shape}). "
            f"Use jnp.array() or pass a pd.Series with a named index."
        )
        raise InvalidParamsError(msg)
    type_name = type(value).__module__ + "." + type(value).__name__
    msg = f"Parameter '{path}' has unexpected type {type_name}."
    raise InvalidParamsError(msg)


def _maybe_convert_dataframe(
    initial_conditions: Mapping[str, Array],
    *,
    model: Model,
) -> Mapping[str, Array]:
    """Convert a DataFrame to initial_conditions dict if needed."""
    if isinstance(initial_conditions, pd.DataFrame):
        return initial_conditions_from_dataframe(df=initial_conditions, model=model)
    return initial_conditions


def _validate_log_args(*, log_level: LogLevel, log_path: str | Path | None) -> None:
    """Raise ValueError if log_level='debug' but log_path is not set."""
    if log_level == "debug" and log_path is None:
        msg = "log_path is required when log_level='debug'"
        raise ValueError(msg)
