"""Collection of classes that are used by the user to define the model and grids."""

import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from beartype import beartype

from _lcm.beartype_conf import MODEL_CONF, PARAMS_CONF
from _lcm.grids import DiscreteGrid
from _lcm.model_processing import (
    _validate_param_types,
    build_regimes_and_template,
    merge_derived_categoricals,
    validate_model_inputs,
)
from _lcm.pandas_utils import (
    convert_series_in_params,
    has_series,
    initial_conditions_from_dataframe,
)
from _lcm.params.processing import (
    broadcast_to_template,
    cast_params_to_canonical_dtypes,
)
from _lcm.persistence.snapshots import (
    _save_simulate_snapshot,
    _save_solve_snapshot,
)
from _lcm.regime_building.processing import Regime
from _lcm.simulation.compile import compile_all_simulate_functions
from _lcm.simulation.initial_conditions import (
    canonicalize_initial_conditions,
    validate_initial_conditions,
)
from _lcm.simulation.result_metadata import _get_output_dtypes
from _lcm.simulation.simulate import simulate
from _lcm.solution.solve_brute import solve
from _lcm.solution.validate_V import contains_nan
from _lcm.transition_checks import validate_transitions
from _lcm.typing import (
    FlatParams,
    FunctionName,
    ParamsTemplate,
    PeriodToRegimeToVArr,
    RegimeName,
    RegimeNamesToIds,
)
from _lcm.utils.containers import (
    ensure_containers_are_immutable,
    ensure_containers_are_mutable,
    get_field_names_and_values,
)
from _lcm.utils.logging import (
    LogLevel,
    get_logger,
    raise_or_warn,
    validation_enabled,
    validation_raises,
)
from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidValueFunctionError,
)
from lcm.regime import Regime as UserRegime
from lcm.result import SimulationResult
from lcm.typing import UserFacingParamsTemplate, UserInitialConditions, UserParams


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

    user_regimes: MappingProxyType[RegimeName, UserRegime]
    """Boundary-form snapshot of regimes as supplied by the user."""

    _regimes: MappingProxyType[RegimeName, Regime]
    """Canonical, processed regimes used by solve and simulate.

    Private: the canonical form is engine-internal. User code should read
    `user_regimes` (the boundary form supplied to the constructor).
    """

    enable_jit: bool = True
    """Whether to JIT-compile the functions of the internal regimes."""

    fixed_params: UserParams
    """Parameters fixed at model initialization."""

    n_subjects: int | None = None
    """Expected simulate batch size; enables AOT compile of simulate functions.

    Dispatch by call shape:

    - `None`: purely lazy behaviour, no AOT.
    - First `simulate(...)` with `actual_n == n_subjects`: AOT-compiles all
      simulate functions for that batch shape (blocks before solve runs)
      and caches them.
    - Subsequent `simulate(...)` with the same matching size: reuses the
      cached compiled programs.
    - `simulate(...)` with a mismatching size: warns once per size and falls
      back to the runtime-traced path.

    Param-shape contract: the cache is keyed only on `n_subjects`. The shapes
    and dtypes of `flat_params` leaves at the first matching call become
    part of the AOT signature; subsequent calls must keep them stable. MSM-
    style estimation (varying values, fixed shapes) is the target use case;
    construct a fresh `Model` whenever a param array's shape or dtype changes.
    """

    _params_template: ParamsTemplate
    """Template for the model parameters."""

    _simulate_compile_cache: dict[int, MappingProxyType[RegimeName, Regime]]
    """AOT-compiled `regimes` per matching `n_subjects`."""

    _warned_n_subjects: set[int]
    """Mismatching `actual_n_subjects` already warned about (one warning each)."""

    _simulate_compile_lock: threading.Lock
    """Serialises mutations of `_simulate_compile_cache` and
    `_warned_n_subjects`.

    The check-then-set on each container is held under this lock. The
    consequent `log.warning` call sits outside the lock so concurrent
    simulate() calls don't serialise on logging I/O.
    """

    @beartype(conf=MODEL_CONF)
    def __init__(
        self,
        *,
        description: str = "",
        ages: AgeGrid,
        regimes: Mapping[RegimeName, UserRegime],
        regime_id_class: type,
        enable_jit: bool = True,
        fixed_params: UserParams = MappingProxyType({}),
        derived_categoricals: Mapping[FunctionName, DiscreteGrid] = MappingProxyType(
            {}
        ),
        n_subjects: int | None = None,
        subjects_batch_size: int = 0,
    ) -> None:
        """Initialize the Model.

        Args:
            regimes: Mapping of regime names to user-provided `Regime`
                instances. Stored as `self.user_regimes` after merging in
                any model-level `derived_categoricals`; the canonical
                processed form is exposed as `self._regimes`.
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
                batch shape `n_subjects` before backward induction starts.
                `None` keeps the purely lazy behaviour.
            subjects_batch_size: Per-device chunk size for the simulate-side
                per-subject dispatch. `0` (default) keeps the single big vmap
                over the entire subjects axis. `>0` iterates each device's
                local shard in chunks of this size via `jax.lax.map` — shrinks
                the per-iteration intermediate by roughly
                `n_subjects_per_device / subjects_batch_size`, which lets
                production grid sizes fit in per-device HBM budgets. JAX
                handles non-divisible remainders.

        """
        if subjects_batch_size < 0:
            msg = (
                f"subjects_batch_size must be >= 0, got {subjects_batch_size}. "
                "Use 0 for no chunking; pick a positive value to bound the "
                "per-device intermediate at simulate dispatch."
            )
            raise ValueError(msg)
        self.description = description
        self.ages = ages
        self.n_periods = ages.n_periods
        self.fixed_params = ensure_containers_are_immutable(fixed_params)
        self.n_subjects = n_subjects
        self.subjects_batch_size = subjects_batch_size
        self._simulate_compile_cache = {}
        self._warned_n_subjects = set()
        self._simulate_compile_lock = threading.Lock()

        validate_model_inputs(
            n_periods=self.n_periods,
            user_regimes=regimes,
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
        self.user_regimes = merge_derived_categoricals(
            user_regimes=regimes,
            derived_categoricals=derived_categoricals,
        )
        self._regimes, self._params_template = build_regimes_and_template(
            ages=self.ages,
            user_regimes=self.user_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            enable_jit=enable_jit,
            fixed_params=self.fixed_params,
            subjects_batch_size=self.subjects_batch_size,
        )
        self.enable_jit = enable_jit
        self.simulation_output_dtypes = _get_output_dtypes(
            user_regimes=self.user_regimes,
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

    @beartype(conf=PARAMS_CONF)
    def solve(
        self,
        *,
        params: UserParams,
        log_level: LogLevel,
        max_compilation_workers: int | None = None,
        log_path: str | Path | None = None,
        log_keep_n_latest: int = 3,
    ) -> PeriodToRegimeToVArr:
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
            log_level: Verbosity, and the runtime-validation policy it implies.
                Required — pick deliberately for the situation:
                - `"off"` — silent; transition-probability and NaN checks skipped.
                - `"warning"` — validation runs, failures logged as warnings,
                  the run continues.
                - `"progress"` — as `"warning"`, plus timing.
                - `"debug"` — validation runs and **raises** on the first
                  failure; adds value-function stats.
                Start every project at `"debug"`: fail early and gather maximum
                diagnostics. Ease to `"warning"` / `"off"` only once the model
                is trusted and you need the speed or the non-raising behaviour
                for an estimation loop.
            max_compilation_workers: Maximum number of threads for parallel XLA
                compilation. Defaults to the number of physical CPU cores.
            log_path: Directory for persisting diagnostic snapshots. Optional at
                every level; snapshots are written only when it is set.
            log_keep_n_latest: Maximum number of snapshots to retain on disk.

        Returns:
            Immutable mapping of period to a value function array for each regime.

        """
        log = get_logger(log_level=log_level)
        flat_params = self._process_params(params)
        validate_transitions(
            regimes=self._regimes,
            flat_params=flat_params,
            ages=self.ages,
            logger=log,
        )
        return self._solve_compiled(
            flat_params=flat_params,
            params=params,
            log=log,
            log_path=log_path,
            log_keep_n_latest=log_keep_n_latest,
            max_compilation_workers=max_compilation_workers,
        )

    def _solve_compiled(
        self,
        *,
        flat_params: FlatParams,
        params: UserParams,
        log: logging.Logger,
        log_path: str | Path | None,
        log_keep_n_latest: int,
        max_compilation_workers: int | None,
    ) -> PeriodToRegimeToVArr:
        """Run backward induction, persisting a diagnostic snapshot when warranted.

        With `log_path` set, a snapshot is written at `log_level="debug"`
        (every solve) and at `"warning"` / `"progress"` whenever the returned
        solution contains NaN. `_enforce_retention` caps the snapshot count at
        `log_keep_n_latest`.
        """
        try:
            period_to_regime_to_V_arr = solve(
                flat_params=flat_params,
                ages=self.ages,
                regimes=self._regimes,
                logger=log,
                enable_jit=self.enable_jit,
                max_compilation_workers=max_compilation_workers,
            )
        except InvalidValueFunctionError as exc:
            if log_path is not None and exc.partial_solution is not None:
                snap_dir = _save_solve_snapshot(
                    model=self,
                    params=params,
                    period_to_regime_to_V_arr=exc.partial_solution,  # ty: ignore[invalid-argument-type]
                    log_path=Path(log_path),
                    log_keep_n_latest=log_keep_n_latest,
                )
                exc.add_note(f"Snapshot saved to {snap_dir}")
            raise
        if (
            log_path is not None
            and validation_enabled(log)
            and (validation_raises(log) or contains_nan(period_to_regime_to_V_arr))
        ):
            _save_solve_snapshot(
                model=self,
                params=params,
                period_to_regime_to_V_arr=period_to_regime_to_V_arr,
                log_path=Path(log_path),
                log_keep_n_latest=log_keep_n_latest,
            )
        return period_to_regime_to_V_arr

    def _resolve_simulate_regimes(
        self,
        *,
        actual_n_subjects: int,
        log: logging.Logger,
    ) -> MappingProxyType[RegimeName, Regime]:
        """Return regimes to use for simulate; AOT cache when matching.

        Dispatch by `n_subjects` and batch-shape match:

        - `n_subjects is None`: return the original `regimes`
          (purely lazy path).
        - `actual_n_subjects != n_subjects`: warn once per mismatching size,
          return the original `regimes`.
        - `actual_n_subjects == n_subjects`: return the cached compiled
          regimes (caller must have populated the cache before calling).
        """
        if self.n_subjects is None:
            return self._regimes
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
            return self._regimes
        with self._simulate_compile_lock:
            return self._simulate_compile_cache[self.n_subjects]

    @beartype(conf=PARAMS_CONF)
    def simulate(
        self,
        *,
        params: UserParams,
        initial_conditions: UserInitialConditions | pd.DataFrame,
        period_to_regime_to_V_arr: PeriodToRegimeToVArr | None,
        log_level: LogLevel,
        seed: int | None = None,
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
            initial_conditions: Mapping of state names (plus `"regime_id"`) to arrays.
                All arrays must have the same length (number of subjects). The
                `"regime_id"` entry must contain integer regime codes (from
                `model.regime_names_to_ids`). May also be a `pd.DataFrame`
                with a `"regime_name"` column carrying regime label strings
                (auto-converted via `initial_conditions_from_dataframe`).
            period_to_regime_to_V_arr: Value function arrays from `solve()`.
                When `None`, the model is solved automatically before simulating.
            seed: Random seed.
            log_level: Verbosity, and the runtime-validation policy it implies.
                Required — pick deliberately for the situation:
                - `"off"` — silent; initial-condition, transition-probability,
                  and NaN checks skipped.
                - `"warning"` — validation runs, failures logged as warnings,
                  the run continues.
                - `"progress"` — as `"warning"`, plus timing.
                - `"debug"` — validation runs and **raises** on the first
                  failure; adds value-function stats.
                Start every project at `"debug"`: fail early and gather maximum
                diagnostics. Ease to `"warning"` / `"off"` only once the model
                is trusted and you need the speed or the non-raising behaviour
                for an estimation loop.
            log_path: Directory for persisting diagnostic snapshots. Optional at
                every level; snapshots are written only when it is set.
            log_keep_n_latest: Maximum number of snapshots to retain on disk.
            max_compilation_workers: Maximum number of threads for parallel XLA
                compilation. Only used when `period_to_regime_to_V_arr` is `None`
                (i.e. when solve runs automatically). Defaults to the number of
                physical CPU cores.

        Returns:
            SimulationResult object. Call .to_dataframe() to get a pandas DataFrame,
            optionally with additional_targets.

        """
        log = get_logger(log_level=log_level)
        if isinstance(initial_conditions, pd.DataFrame):
            initial_conditions = initial_conditions_from_dataframe(
                df=initial_conditions,
                user_regimes=self.user_regimes,
                regime_names_to_ids=self.regime_names_to_ids,
            )
        initial_conditions = canonicalize_initial_conditions(
            initial_conditions=initial_conditions,
            regimes=self._regimes,
        )
        flat_params = self._process_params(params)
        if validation_enabled(log):
            try:
                validate_initial_conditions(
                    initial_conditions=initial_conditions,
                    regimes=self._regimes,
                    regime_names_to_ids=self.regime_names_to_ids,
                    flat_params=flat_params,
                    ages=self.ages,
                )
            except InvalidInitialConditionsError as error:
                raise_or_warn(logger=log, error=error)
        validate_transitions(
            regimes=self._regimes,
            flat_params=flat_params,
            ages=self.ages,
            logger=log,
        )
        actual_n_subjects = len(next(iter(initial_conditions.values())))
        n_subjects = self.n_subjects
        if n_subjects is not None and n_subjects == actual_n_subjects:
            with self._simulate_compile_lock:
                needs_compile = n_subjects not in self._simulate_compile_cache
            if needs_compile:
                compiled = compile_all_simulate_functions(
                    regimes=self._regimes,
                    flat_params=flat_params,
                    ages=self.ages,
                    n_subjects=n_subjects,
                    max_compilation_workers=max_compilation_workers,
                    logger=log,
                )
                with self._simulate_compile_lock:
                    self._simulate_compile_cache[n_subjects] = compiled
        if period_to_regime_to_V_arr is None:
            period_to_regime_to_V_arr = self._solve_compiled(
                flat_params=flat_params,
                params=params,
                log=log,
                log_path=log_path,
                log_keep_n_latest=log_keep_n_latest,
                max_compilation_workers=max_compilation_workers,
            )
        simulate_regimes = self._resolve_simulate_regimes(
            actual_n_subjects=actual_n_subjects,
            log=log,
        )
        result = simulate(
            flat_params=flat_params,
            initial_conditions=initial_conditions,
            regimes=simulate_regimes,
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
        if simulate_regimes is not self._regimes:
            result._regimes = self._regimes  # noqa: SLF001
        if log_path is not None and validation_raises(log):
            _save_simulate_snapshot(
                model=self,
                params=params,
                initial_conditions=initial_conditions,
                period_to_regime_to_V_arr=period_to_regime_to_V_arr,
                result=result,
                log_path=Path(log_path),
                log_keep_n_latest=log_keep_n_latest,
            )
        return result

    def _process_params(self, params: UserParams) -> FlatParams:
        """Broadcast, convert Series, dtype-cast, and validate user params.

        Step order matters: `convert_series_in_params` runs *between*
        `broadcast_to_template` and `cast_params_to_canonical_dtypes` so
        the dtype cast walks a uniform tree (no `pd.Series` to special-
        case).
        """
        flat_params = broadcast_to_template(
            params=params, template=self._params_template, required=True
        )
        if has_series(flat_params):
            flat_params = convert_series_in_params(
                flat_params=flat_params,
                ages=self.ages,
                user_regimes=self.user_regimes,
                regime_names_to_ids=self.regime_names_to_ids,
            )
        flat_params = cast_params_to_canonical_dtypes(flat_params)
        _validate_param_types(flat_params)
        return flat_params
