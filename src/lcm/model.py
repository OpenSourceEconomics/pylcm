"""Collection of classes that are used by the user to define the model and grids."""

import dataclasses
import logging
import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
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
from lcm.regime_building.processing import InternalRegime
from lcm.simulation.compile import compile_all_simulate_functions
from lcm.simulation.initial_conditions import validate_initial_conditions
from lcm.simulation.result import SimulationResult, get_simulation_output_dtypes
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve
from lcm.typing import (
    FloatND,
    FunctionName,
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

    regimes: MappingProxyType[RegimeName, Regime]
    """Immutable mapping of regime names to user `Regime` instances."""

    internal_regimes: MappingProxyType[RegimeName, InternalRegime]
    """Immutable mapping of regime names to internal regime instances."""

    enable_jit: bool = True
    """Whether to JIT-compile the functions of the internal regimes."""

    fixed_params: UserParams
    """Parameters fixed at model initialization."""

    n_subjects: int | None = None
    """Expected simulate batch size; enables AOT compile of simulate functions.

    Dispatch by call shape:

    - `None`: purely lazy behaviour, no AOT.
    - First `simulate(...)` with `actual_n == n_subjects`: AOT-compiles all
      simulate functions for that batch shape in parallel and caches them.
    - Subsequent `simulate(...)` with the same matching size: reuses the
      cached compiled programs.
    - `simulate(...)` with a mismatching size: warns once per size and falls
      back to the runtime-traced path.

    Param-shape contract: the cache is keyed only on `n_subjects`. The shapes
    and dtypes of `internal_params` leaves at the first matching call become
    part of the AOT signature; subsequent calls must keep them stable. MSM-
    style estimation (varying values, fixed shapes) is the target use case;
    construct a fresh `Model` whenever a param array's shape or dtype changes.
    """

    _params_template: ParamsTemplate
    """Template for the model parameters."""

    _simulate_compile_cache: dict[int, MappingProxyType[RegimeName, InternalRegime]]
    """AOT-compiled `internal_regimes` per matching `n_subjects`."""

    _warned_n_subjects: set[int]
    """Mismatching `actual_n_subjects` already warned about (one warning each)."""

    _simulate_compile_lock: threading.Lock
    """Serialises mutations of `_simulate_compile_cache` and
    `_warned_n_subjects`.

    The check-then-set on each container is held under this lock. The
    consequent `log.warning` call sits outside the lock so concurrent
    simulate() calls don't serialise on logging I/O.
    """

    def __init__(
        self,
        *,
        description: str = "",
        ages: AgeGrid,
        regimes: Mapping[RegimeName, Regime],
        regime_id_class: type,
        enable_jit: bool = True,
        fixed_params: UserParams = MappingProxyType({}),
        derived_categoricals: Mapping[FunctionName, DiscreteGrid] = MappingProxyType(
            {}
        ),
        n_subjects: int | None = None,
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
            n_subjects: Expected simulate batch size; if set, the first matching
                `simulate(...)` call AOT-compiles all simulate functions for
                batch shape `n_subjects` in parallel. `None` keeps the purely
                lazy behaviour.

        """
        self.description = description
        self.ages = ages
        self.n_periods = ages.n_periods
        self.fixed_params = ensure_containers_are_immutable(fixed_params)
        self.n_subjects = n_subjects
        self._simulate_compile_cache = {}
        self._warned_n_subjects = set()
        self._simulate_compile_lock = threading.Lock()

        validate_model_inputs(
            n_periods=self.n_periods,
            regimes=regimes,
            regime_id_class=regime_id_class,
            n_subjects=n_subjects,
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
        self.enable_jit = enable_jit
        self.simulation_output_dtypes = get_simulation_output_dtypes(
            regimes=self.regimes,
            regime_names_to_ids=self.regime_names_to_ids,
        )

    def __getstate__(self) -> dict[str, object]:
        """Return a copy of `__dict__` with per-process AOT compile state removed.

        Drops `_simulate_compile_lock` (a `threading.Lock`, not pickleable),
        `_simulate_compile_cache` (compiled XLA programs that can't survive
        a process boundary), and `_warned_n_subjects` (its companion set).
        `__setstate__` restores all three to their fresh state.
        """
        state = self.__dict__.copy()
        state.pop("_simulate_compile_lock", None)
        state.pop("_simulate_compile_cache", None)
        state.pop("_warned_n_subjects", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore AOT compile state to a fresh empty cache."""
        self.__dict__.update(state)
        self._simulate_compile_cache = {}
        self._warned_n_subjects = set()
        self._simulate_compile_lock = threading.Lock()

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
        return self._solve_compiled(
            internal_params=internal_params,
            params=params,
            log=get_logger(log_level=log_level),
            log_level=log_level,
            log_path=log_path,
            log_keep_n_latest=log_keep_n_latest,
            max_compilation_workers=max_compilation_workers,
        )

    def _solve_compiled(
        self,
        *,
        internal_params: InternalParams,
        params: UserParams,
        log: logging.Logger,
        log_level: LogLevel,
        log_path: str | Path | None,
        log_keep_n_latest: int,
        max_compilation_workers: int | None,
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
        """Run backward induction, persisting a snapshot on debug or NaN failure."""
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
                snap_dir = save_solve_snapshot(
                    model=self,
                    params=params,
                    period_to_regime_to_V_arr=exc.partial_solution,  # ty: ignore[invalid-argument-type]
                    log_path=Path(log_path),
                    log_keep_n_latest=log_keep_n_latest,
                )
                exc.add_note(f"Snapshot saved to {snap_dir}")
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

    def _spawn_simulate_compile(
        self,
        *,
        n_subjects: int,
        internal_params: InternalParams,
        max_compilation_workers: int | None,
        logger: logging.Logger,
    ) -> Future[MappingProxyType[RegimeName, InternalRegime]]:
        """Submit `compile_all_simulate_functions` to a single-thread executor.

        Caller decides whether to spawn (`n_subjects` set, batch shape
        matches, no cache hit). The returned `Future` runs in parallel with
        whatever the caller does next — typically `_solve_compiled(...)`.
        """
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lcm-simulate-compile"
        )
        future = executor.submit(
            compile_all_simulate_functions,
            internal_regimes=self.internal_regimes,
            internal_params=internal_params,
            ages=self.ages,
            n_subjects=n_subjects,
            max_compilation_workers=max_compilation_workers,
            logger=logger,
        )
        executor.shutdown(wait=False)
        return future

    def _resolve_simulate_internal_regimes(
        self,
        *,
        compile_future: Future[MappingProxyType[RegimeName, InternalRegime]] | None,
        actual_n_subjects: int,
        log: logging.Logger,
    ) -> MappingProxyType[RegimeName, InternalRegime]:
        """Return internal_regimes to use for simulate; AOT cache when matching.

        Dispatch by `n_subjects` and batch-shape match:

        - `n_subjects is None`: return the original `internal_regimes`
          (purely lazy path).
        - `actual_n_subjects != n_subjects`: warn once per mismatching size,
          return the original `internal_regimes`.
        - `actual_n_subjects == n_subjects`, `compile_future is not None`:
          await it and cache the result.
        - `actual_n_subjects == n_subjects`, `compile_future is None`: cache
          must already hold the entry (caller spawned only on cache miss);
          return the cached compiled regimes.
        """
        if self.n_subjects is None:
            return self.internal_regimes
        if actual_n_subjects != self.n_subjects:
            with self._simulate_compile_lock:
                already_warned = actual_n_subjects in self._warned_n_subjects
                if not already_warned:
                    self._warned_n_subjects.add(actual_n_subjects)
            if not already_warned:
                log.warning(
                    "simulate called with n_subjects=%d but model declared "
                    "n_subjects=%d; falling back to runtime compile.",
                    actual_n_subjects,
                    self.n_subjects,
                )
            return self.internal_regimes
        if compile_future is not None:
            compiled = compile_future.result()
            with self._simulate_compile_lock:
                self._simulate_compile_cache[self.n_subjects] = compiled
            return compiled
        with self._simulate_compile_lock:
            return self._simulate_compile_cache[self.n_subjects]

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
        actual_n_subjects = len(next(iter(initial_conditions.values())))
        n_subjects = self.n_subjects
        compile_future: Future[MappingProxyType[RegimeName, InternalRegime]] | None = (
            None
        )
        if n_subjects is not None and n_subjects == actual_n_subjects:
            with self._simulate_compile_lock:
                needs_compile = n_subjects not in self._simulate_compile_cache
            if needs_compile:
                compile_future = self._spawn_simulate_compile(
                    n_subjects=n_subjects,
                    internal_params=internal_params,
                    max_compilation_workers=max_compilation_workers,
                    logger=log,
                )
        if period_to_regime_to_V_arr is None:
            period_to_regime_to_V_arr = self._solve_compiled(
                internal_params=internal_params,
                params=params,
                log=log,
                log_level=log_level,
                log_path=log_path,
                log_keep_n_latest=log_keep_n_latest,
                max_compilation_workers=max_compilation_workers,
            )
        simulate_internal_regimes = self._resolve_simulate_internal_regimes(
            compile_future=compile_future,
            actual_n_subjects=actual_n_subjects,
            log=log,
        )
        result = simulate(
            internal_params=internal_params,
            initial_conditions=initial_conditions,
            internal_regimes=simulate_internal_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            logger=log,
            period_to_regime_to_V_arr=period_to_regime_to_V_arr,
            ages=self.ages,
            simulation_output_dtypes=self.simulation_output_dtypes,
            seed=seed,
        )
        # AOT-compiled regimes carry `jax.stages.Compiled` callables that
        # wrap an unpicklable `LoadedExecutable`. `to_dataframe` only reads
        # the lazy DAG functions / constraints / transitions on
        # `simulate_functions`, never the compiled callables — so swap in
        # the lazy regimes to keep the result cloudpickle-safe.
        if simulate_internal_regimes is not self.internal_regimes:
            result._internal_regimes = self.internal_regimes  # noqa: SLF001
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
    regimes: Mapping[RegimeName, Regime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> MappingProxyType[RegimeName, Regime]:
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
    result: dict[RegimeName, Regime] = {}
    for regime_name, regime in regimes.items():
        merged = dict(regime.derived_categoricals)
        for var, grid in derived_categoricals.items():
            existing = merged.get(var)
            if existing is not None and existing.categories != grid.categories:
                msg = (
                    f"Model-level derived_categoricals['{var}'] conflicts "
                    f"with regime '{regime_name}': {grid.categories} vs "
                    f"{existing.categories}."
                )
                raise ModelInitializationError(msg)
            merged[var] = grid
        result[regime_name] = dataclasses.replace(
            regime, derived_categoricals=MappingProxyType(merged)
        )
    return MappingProxyType(result)


def _validate_log_args(*, log_level: LogLevel, log_path: str | Path | None) -> None:
    """Raise ValueError if log_level='debug' but log_path is not set."""
    if log_level == "debug" and log_path is None:
        msg = "log_path is required when log_level='debug'"
        raise ValueError(msg)
