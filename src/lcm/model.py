"""Collection of classes that are used by the user to define the model and grids."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import InvalidValueFunctionError, ModelInitializationError
from lcm.grids import DiscreteGrid
from lcm.model_processing import (
    _validate_param_types,
    build_regimes_and_template,
    validate_model_inputs,
)
from lcm.pandas_utils import (
    convert_series_in_params,
    has_series,
    initial_conditions_from_dataframe,
)
from lcm.params.processing import (
    process_params,
)
from lcm.persistence import (
    save_simulate_snapshot,
    save_solve_snapshot,
)
from lcm.regime import Regime
from lcm.regime_building.partitions import (
    inject_partition_scalars,
    iterate_partition_points,
    stack_partition_V_arrays,
)
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
    """Whether to JIT-compile the functions of the internal regimes."""

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
        derived_categoricals: Mapping[str, DiscreteGrid] = MappingProxyType({}),
    ) -> None:
        """Initialize the Model.

        Args:
            regimes: Mapping of regime names to Regime instances.
            ages: Age grid for the model.
            description: Description of the model.
            regime_id_class: Dataclass mapping regime names to integer indices.
            enable_jit: Whether to JIT-compile the functions of the internal
                regimes.
            fixed_params: Parameters that can be fixed at model initialization.
            derived_categoricals: Categorical grids for DAG function outputs
                not in states/actions. Broadcast to all regimes (merged with
                each regime's own `derived_categoricals`). Raises if a regime
                already has a conflicting entry.

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
        self.regimes = _merge_derived_categoricals(regimes, derived_categoricals)
        self.internal_regimes, self._params_template = build_regimes_and_template(
            ages=self.ages,
            regimes=self.regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            enable_jit=enable_jit,
            fixed_params=self.fixed_params,
        )
        self._partition_grid = _build_partition_grid(self.internal_regimes)
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
            max_compilation_workers: Maximum number of threads for parallel XLA
                compilation. Defaults to the number of physical CPU cores.
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
        internal_params = self._process_params(params)
        validate_regime_transitions_all_periods(
            internal_regimes=self.internal_regimes,
            internal_params=internal_params,
            ages=self.ages,
        )
        logger = get_logger(log_level=log_level)
        sub_V_arrays: list[
            MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]
        ] = []
        try:
            for partition_point in iterate_partition_points(self._partition_grid):
                partition_params = inject_partition_scalars(
                    internal_params=internal_params, partition_point=partition_point
                )
                sub_V_arrays.append(
                    solve(
                        internal_params=partition_params,
                        ages=self.ages,
                        internal_regimes=self.internal_regimes,
                        logger=logger,
                        enable_jit=self.enable_jit,
                        max_compilation_workers=max_compilation_workers,
                    )
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
        period_to_regime_to_V_arr = stack_partition_V_arrays(
            sub_V_arrays=sub_V_arrays, partition_grid=self._partition_grid
        )
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
        max_compilation_workers: int | None = None,
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
            max_compilation_workers: Maximum number of threads for parallel XLA
                compilation. Only used when `period_to_regime_to_V_arr` is `None`
                (i.e. when solve runs automatically). Defaults to the number of
                physical CPU cores.

        Returns:
            SimulationResult object. Call .to_dataframe() to get a pandas DataFrame,
            optionally with additional_targets.

        """
        _validate_log_args(log_level=log_level, log_path=log_path)
        if isinstance(initial_conditions, pd.DataFrame):
            initial_conditions = initial_conditions_from_dataframe(
                df=initial_conditions,
                regimes=self.regimes,
                regime_names_to_ids=self.regime_names_to_ids,
            )
        internal_params = self._process_params(params)
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
                    enable_jit=self.enable_jit,
                    max_compilation_workers=max_compilation_workers,
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

    def _process_params(self, params: UserParams) -> InternalParams:
        """Broadcast, convert Series, and validate user params."""
        internal_params = process_params(
            params=params, params_template=self._params_template
        )
        if has_series(internal_params):
            internal_params = convert_series_in_params(
                internal_params=internal_params,
                ages=self.ages,
                regimes=self.regimes,
                regime_names_to_ids=self.regime_names_to_ids,
            )
        _validate_param_types(internal_params)
        return internal_params


def _merge_derived_categoricals(
    regimes: Mapping[str, Regime],
    derived_categoricals: Mapping[str, DiscreteGrid],
) -> MappingProxyType[str, Regime]:
    """Merge model-level derived_categoricals into each regime.

    Args:
        regimes: Mapping of regime names to Regime instances.
        derived_categoricals: Model-level categorical grids to broadcast.

    Returns:
        Immutable mapping of regime names to (possibly updated) Regime instances.

    Raises:
        ModelInitializationError: If a regime already has a conflicting entry
            (same key, different categories).

    """
    if not derived_categoricals:
        return MappingProxyType(dict(regimes))
    result = {}
    for name, regime in regimes.items():
        merged = dict(regime.derived_categoricals)
        for var, grid in derived_categoricals.items():
            existing = merged.get(var)
            if existing is not None and existing.categories != grid.categories:
                msg = (
                    f"Model-level derived_categoricals['{var}'] conflicts "
                    f"with regime '{name}': {grid.categories} vs "
                    f"{existing.categories}."
                )
                raise ModelInitializationError(msg)
            merged[var] = grid
        result[name] = dataclasses.replace(
            regime, derived_categoricals=MappingProxyType(merged)
        )
    return MappingProxyType(result)


def _validate_log_args(*, log_level: LogLevel, log_path: str | Path | None) -> None:
    """Raise ValueError if log_level='debug' but log_path is not set."""
    if log_level == "debug" and log_path is None:
        msg = "log_path is required when log_level='debug'"
        raise ValueError(msg)


def _build_partition_grid(
    internal_regimes: Mapping[RegimeName, InternalRegime],
) -> MappingProxyType[str, DiscreteGrid]:
    """Aggregate `InternalRegime.partitions` into a single model-level grid.

    Every regime that declares a partition name must agree on its grid
    (same categories). A partition name declared in one regime and absent
    from another is allowed — the outer partition loop simply iterates
    identical sub-solves for regimes that do not reference the partition.

    Args:
        internal_regimes: Mapping of regime names to `InternalRegime` instances.

    Returns:
        Immutable mapping of partition name to `DiscreteGrid`. Empty when
        no regime declares any partition.

    Raises:
        ModelInitializationError: If two regimes declare the same partition
            name with different categories.

    """
    grid: dict[str, DiscreteGrid] = {}
    for regime_name, internal_regime in internal_regimes.items():
        for name, spec in internal_regime.partitions.items():
            existing = grid.get(name)
            if existing is None:
                grid[name] = spec
            elif existing.categories != spec.categories:
                msg = (
                    f"Partition dimension '{name}' has inconsistent categories "
                    f"across regimes: {existing.categories} vs {spec.categories} "
                    f"(regime '{regime_name}'). All regimes that declare a "
                    f"partition must agree on its grid."
                )
                raise ModelInitializationError(msg)
    return MappingProxyType(grid)
