"""Collection of classes that are used by the user to define the model and grids."""

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from dags import get_ancestors
from dags.tree import QNAME_DELIMITER, qname_from_tree_path
from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import InvalidParamsError, ModelInitializationError, format_messages
from lcm.grids import DiscreteGrid
from lcm.pandas_utils import (
    convert_series_in_params,
    has_series,
    initial_conditions_from_dataframe,
)
from lcm.params import MappingLeaf, SequenceLeaf
from lcm.params.processing import (
    broadcast_to_template,
    create_params_template,
    process_params,
)
from lcm.persistence import (
    save_simulate_snapshot,
    save_solve_snapshot,
)
from lcm.regime import Regime
from lcm.regime_building.processing import (
    InternalRegime,
    process_regimes,
)
from lcm.simulation.result import SimulationResult, get_simulation_output_dtypes
from lcm.simulation.simulate import simulate
from lcm.simulation.validation import validate_initial_conditions
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

        _validate_model_inputs(
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
        self.internal_regimes, self._params_template = _build_regimes_and_template(
            regimes=regimes,
            ages=self.ages,
            regime_names_to_ids=self.regime_names_to_ids,
            enable_jit=enable_jit,
            fixed_params=self.fixed_params,
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
        period_to_regime_to_V_arr = solve(
            internal_params=internal_params,
            ages=self.ages,
            internal_regimes=self.internal_regimes,
            logger=get_logger(log_level=log_level),
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
            period_to_regime_to_V_arr = solve(
                internal_params=internal_params,
                ages=self.ages,
                internal_regimes=self.internal_regimes,
                logger=log,
            )
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
        if isinstance(value, np.ndarray):
            msg = (
                f"Parameter '{path}' is a numpy array (shape {value.shape}). "
                f"Use jax.numpy.array() or pass a pd.Series with a named index."
            )
            raise InvalidParamsError(msg)
        return  # JAX array
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


def _build_regimes_and_template(
    *,
    regimes: Mapping[str, Regime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
    fixed_params: UserParams,
) -> tuple[MappingProxyType[RegimeName, InternalRegime], ParamsTemplate]:
    """Build internal regimes and params template in a single pass.

    Composes regime processing, template creation, and optional fixed-param partialling
    so that each result is computed exactly once.

    """
    internal_regimes = process_regimes(
        regimes=regimes,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=enable_jit,
    )
    params_template = create_params_template(internal_regimes)

    if fixed_params:
        fixed_internal = _resolve_fixed_params(
            fixed_params=dict(fixed_params), template=params_template
        )
        if any(v for v in fixed_internal.values()):
            internal_regimes = _partial_fixed_params_into_regimes(
                internal_regimes=internal_regimes, fixed_internal=fixed_internal
            )
            params_template = _remove_fixed_from_template(
                template=params_template, fixed_internal=fixed_internal
            )

    return internal_regimes, params_template


def _validate_model_inputs(
    *,
    n_periods: int,
    regimes: Mapping[str, Regime],
    regime_id_class: type,
) -> None:
    # Early exit if regimes are not lcm.Regime instances
    if not all(isinstance(regime, Regime) for regime in regimes.values()):
        raise ModelInitializationError(
            "All items in regimes must be instances of lcm.Regime."
        )

    error_messages: list[str] = []

    if not isinstance(n_periods, int):
        error_messages.append("n_periods must be an integer.")
    elif n_periods <= 1:
        error_messages.append("n_periods must be at least 2.")

    if not regimes:
        error_messages.append(
            "At least one non-terminal and one terminal regime must be provided."
        )

    # Validate regime names don't contain separator
    invalid_names = [name for name in regimes if QNAME_DELIMITER in name]
    if invalid_names:
        error_messages.append(
            f"Regime names cannot contain the separator character "
            f"'{QNAME_DELIMITER}'. The following names are invalid: {invalid_names}."
        )

    # Assume all items in regimes are lcm.Regime instances beyond this point
    terminal_regimes = [name for name, r in regimes.items() if r.terminal]
    if len(terminal_regimes) < 1:
        error_messages.append("lcm.Model must have at least one terminal regime.")

    non_terminal_regimes = {name: r for name, r in regimes.items() if not r.terminal}
    if len(non_terminal_regimes) < 1:
        error_messages.append("lcm.Model must have at least one non-terminal regime.")

    regime_id_fields = sorted(get_field_names_and_values(regime_id_class).keys())
    regime_names = sorted(regimes.keys())
    if regime_id_fields != regime_names:
        error_messages.append(
            f"regime_id_cls fields must match regime names.\nGot:\n"
            "regime_id_cls fields:\n"
            f"    {regime_id_fields}\n"
            "regime names:\n"
            f"    {regime_names}."
        )
    error_messages.extend(_validate_all_variables_used(regimes))

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitializationError(msg)


def _validate_all_variables_used(regimes: Mapping[str, Regime]) -> list[str]:
    """Validate that all states and actions are used somewhere in each regime.

    Each state or action must appear in at least one of:
    - The concurrent valuation (utility or constraints)
    - A transition function

    Args:
        regimes: Mapping of regime names to regimes to validate.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages = []

    for regime_name, regime in regimes.items():
        variable_names = set(regime.states) | set(regime.actions)
        user_functions = dict(regime.get_all_functions(phase="solve"))

        targets = [
            "utility",
            *list(regime.constraints),
            *(
                name
                for name in user_functions
                if name.startswith("next_")
                and not getattr(user_functions[name], "_is_auto_identity", False)
            ),
        ]
        reachable = get_ancestors(
            user_functions, targets=targets, include_targets=False
        )
        unused_variables = sorted(variable_names - reachable)

        if unused_variables:
            unused_states = [v for v in unused_variables if v in regime.states]
            unused_actions = [v for v in unused_variables if v in regime.actions]

            msg_parts = []
            if unused_states:
                state_word = "state" if len(unused_states) == 1 else "states"
                msg_parts.append(f"{state_word} {unused_states}")
            if unused_actions:
                action_word = "action" if len(unused_actions) == 1 else "actions"
                msg_parts.append(f"{action_word} {unused_actions}")

            error_messages.append(
                f"The following variables are defined but never used in regime "
                f"'{regime_name}': {' and '.join(msg_parts)}. "
                f"Each state and action must be used in at least one of: "
                f"utility, constraints, or transition functions."
            )

    return error_messages


def _resolve_fixed_params(
    *,
    fixed_params: dict[str, object],
    template: ParamsTemplate,
) -> InternalParams:
    """Resolve fixed_params against the params template.

    Like `process_params`, support model/regime/function level specification, but
    do NOT require all template keys to be present — only match what's provided.

    """
    return broadcast_to_template(
        params=fixed_params,
        template=template,
        required=False,
    )


def _remove_fixed_from_template(
    *,
    template: ParamsTemplate,
    fixed_internal: InternalParams,
) -> ParamsTemplate:
    """Remove fixed params from the params template.

    After partialling fixed params into compiled functions, remove them from the
    template so users don't need to supply them at solve/simulate time.

    """
    result: dict[str, dict[str, dict[str, str]]] = {}
    for regime_name, regime_template in template.items():
        regime_fixed = fixed_internal.get(regime_name, MappingProxyType({}))
        new_regime: dict[str, dict[str, str]] = {}
        for func_name, func_params in regime_template.items():
            new_func_params = {
                param_name: param_type
                for param_name, param_type in func_params.items()
                if qname_from_tree_path((func_name, param_name)) not in regime_fixed
            }
            if new_func_params:
                new_regime[func_name] = new_func_params
        if new_regime:
            result[regime_name] = new_regime
        else:
            # Keep regime key even if empty (needed by process_params)
            result[regime_name] = {}
    return MappingProxyType(
        {
            regime_name: MappingProxyType(
                {
                    func_name: MappingProxyType(func_params)
                    for func_name, func_params in regime.items()
                }
            )
            for regime_name, regime in result.items()
        }
    )


def _partial_fixed_params_into_regimes(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    fixed_internal: InternalParams,
) -> MappingProxyType[RegimeName, InternalRegime]:
    """Partial fixed params into all compiled functions on each InternalRegime."""
    result = {}
    for name, regime in internal_regimes.items():
        regime_fixed = dict(fixed_internal.get(name, MappingProxyType({})))
        if not regime_fixed:
            result[name] = regime
            continue

        # Build new solve_functions with partialled functions
        solve_funcs = regime.solve_functions
        new_solve = dataclasses.replace(
            solve_funcs,
            max_Q_over_a=MappingProxyType(
                {
                    period: functools.partial(func, **regime_fixed)
                    for period, func in solve_funcs.max_Q_over_a.items()
                }
            ),
            compute_regime_transition_probs=(
                functools.partial(
                    solve_funcs.compute_regime_transition_probs,
                    **_filter_kwargs_for_func(
                        func=solve_funcs.compute_regime_transition_probs,
                        kwargs=regime_fixed,
                    ),
                )
                if solve_funcs.compute_regime_transition_probs is not None
                else None
            ),
        )

        # Build new simulate_functions with partialled functions
        simulate_funcs = regime.simulate_functions
        new_simulate = dataclasses.replace(
            simulate_funcs,
            argmax_and_max_Q_over_a=MappingProxyType(
                {
                    period: functools.partial(func, **regime_fixed)
                    for period, func in simulate_funcs.argmax_and_max_Q_over_a.items()
                }
            ),
            next_state=functools.partial(simulate_funcs.next_state, **regime_fixed),
            compute_regime_transition_probs=(
                functools.partial(
                    simulate_funcs.compute_regime_transition_probs,
                    **_filter_kwargs_for_func(
                        func=simulate_funcs.compute_regime_transition_probs,
                        kwargs=regime_fixed,
                    ),
                )
                if simulate_funcs.compute_regime_transition_probs is not None
                else None
            ),
        )

        result[name] = dataclasses.replace(
            regime,
            solve_functions=new_solve,
            simulate_functions=new_simulate,
            resolved_fixed_params=MappingProxyType(regime_fixed),
        )
    return MappingProxyType(result)


def _filter_kwargs_for_func(
    *, func: Callable, kwargs: Mapping[str, object]
) -> Mapping[str, object]:
    """Filter kwargs to only those accepted by func's signature."""

    try:
        sig = inspect.signature(func)
    except ValueError, TypeError:
        # If we can't inspect the signature, pass all kwargs through
        return kwargs
    params = sig.parameters
    # If the function accepts **kwargs, pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}
