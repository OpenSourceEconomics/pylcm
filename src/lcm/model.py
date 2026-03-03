"""Collection of classes that are used by the user to define the model and grids."""

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType

from dags.tree import QNAME_DELIMITER, flatten_to_qnames
from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidNameError,
    InvalidParamsError,
    ModelInitializationError,
    format_messages,
)
from lcm.input_processing.params_processing import collapse_pair_keys, process_params
from lcm.input_processing.regime_processing import InternalRegime, process_regimes
from lcm.input_processing.util import get_variable_info
from lcm.interfaces import PhaseVariantContainer
from lcm.logging import get_logger
from lcm.regime import Regime
from lcm.simulation.result import SimulationResult
from lcm.simulation.simulate import simulate
from lcm.simulation.validation import validate_initial_conditions
from lcm.solution.solve_brute import solve
from lcm.typing import (
    REGIME_PAIR_SEPARATOR,
    FloatND,
    InternalParams,
    MutableParamsTemplate,
    ParamsTemplate,
    RegimeName,
    RegimeNamesToIds,
    UserFunction,
    UserParams,
)
from lcm.utils import (
    ensure_containers_are_immutable,
    ensure_containers_are_mutable,
    get_field_names_and_values,
)


class Model:
    """A model which is created from a regime.

    Upon initialization, an internal regime will be created which contains all
    the functions needed to solve and simulate the model.

    Attributes:
        description: Description of the model.
        n_periods: Number of periods in the model.
        enable_jit: Whether to jit the functions of the internal regime.
        regime_names_to_ids: Mapping from regime names to integer indices.
        regimes: The user provided regimes that contain the information
            about the model's regimes.
        internal_regimes: The internal regime instances created by LCM, which allow
            to solve and simulate the model.
        params_template: Template for the model parameters.

    """

    description: str | None = None
    ages: AgeGrid
    n_periods: int
    regime_names_to_ids: RegimeNamesToIds
    regimes: MappingProxyType[str, Regime]
    internal_regimes: MappingProxyType[RegimeName, InternalRegime]
    enable_jit: bool = True
    fixed_params: UserParams
    params_template: ParamsTemplate

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
        (
            self.internal_regimes,
            self._internal_params_template,
            self.params_template,
            self._collapsed_pair_keys,
        ) = _build_regimes_and_template(
            regimes=regimes,
            ages=self.ages,
            regime_names_to_ids=self.regime_names_to_ids,
            enable_jit=enable_jit,
            fixed_params=self.fixed_params,
        )
        self.enable_jit = enable_jit

    def get_params_template(self) -> MutableParamsTemplate:
        """Get a mutable copy of the params template.

        Returns a deep copy of the params_template where all immutable containers
        (MappingProxyType, tuple, frozenset) are converted to their mutable
        equivalents (dict, list, set).

        Returns:
            A mutable nested dict with the same structure as params_template.

        """
        return ensure_containers_are_mutable(  # ty: ignore[invalid-return-type]
            self.params_template
        )

    def get_all_functions(
        self,
        regime_name: str,
    ) -> MappingProxyType[str, UserFunction]:
        """Get all functions for a regime, including boundary-encoded transitions.

        Args:
            regime_name: Name of the regime.

        Returns:
            Read-only mapping of all regime functions.

        """
        from lcm.input_processing.process_transitions import (  # noqa: PLC0415
            collect_regime_functions,
        )

        all_funcs, _ = collect_regime_functions(
            regime_name=regime_name,
            regimes=self.regimes,
        )
        return MappingProxyType(all_funcs)

    def _process_params(self, params: UserParams) -> InternalParams:
        """Process user params against the internal template with user-facing errors.

        Wraps `process_params` to translate internal pair-key references (e.g.,
        `working_to_retired`) back to their collapsed source-regime form (e.g.,
        `working`) in error messages, so errors match what users see in
        `params_template`.

        """
        try:
            return process_params(
                params=params, params_template=self._internal_params_template
            )
        except (InvalidParamsError, InvalidNameError) as e:
            msg = str(e)
            for pair_key, source in self._collapsed_pair_keys.items():
                msg = msg.replace(pair_key, source)
            raise type(e)(msg) from None

    def solve(
        self,
        params: UserParams,
        *,
        debug_mode: bool = True,
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
        """Solve the model using the pre-computed functions.

        Args:
            params: Model parameters matching the template from self.params_template
                Parameters can be provided at exactly one of three levels:
                - Model level: {"arg_0": 0.0} - propagates to all functions needing
                  arg_0
                - Regime level: {"regime_0": {"arg_0": 0.0}} - propagates within
                  regime_0
                - Function level: {"regime_0": {"func": {"arg_0": 0.0}}} - direct
                  specification
            debug_mode: Whether to enable debug logging

        Returns:
            Immutable mapping of period to a value function array for each regime.
        """
        internal_params = self._process_params(params)
        return solve(
            internal_params=internal_params,
            ages=self.ages,
            internal_regimes=self.internal_regimes,
            logger=get_logger(debug_mode=debug_mode),
        )

    def simulate(
        self,
        params: UserParams,
        initial_states: Mapping[str, Array],
        initial_regimes: list[RegimeName],
        V_arr_dict: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
        *,
        check_initial_conditions: bool = True,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> SimulationResult:
        """Simulate the model forward using pre-computed value functions.

        Args:
            params: Model parameters matching the template from self.params_template.
                Parameters can be provided at exactly one of three levels:
                - Model level: {"arg_0": 0.0} - propagates to all functions needing
                  arg_0
                - Regime level: {"regime_0": {"arg_0": 0.0}} - propagates within
                  regime_0
                - Function level: {"regime_0": {"func": {"arg_0": 0.0}}} - direct
                  specification
            initial_states: Mapping of state names to arrays. All arrays must have the
                same length (number of subjects). Each state name should correspond to a
                state variable defined in at least one regime.
            initial_regimes: List of regime names the subjects start in.
            V_arr_dict: Value function arrays from solve().
            check_initial_conditions: Whether to validate initial states and regimes.
            seed: Random seed.
            debug_mode: Whether to enable debug logging.

        Returns:
            SimulationResult object. Call .to_dataframe() to get a pandas DataFrame,
            optionally with additional_targets.

        """
        internal_params = self._process_params(params)
        if check_initial_conditions:
            validate_initial_conditions(
                initial_states=initial_states,
                initial_regimes=initial_regimes,
                internal_regimes=self.internal_regimes,
                internal_params=internal_params,
                ages=self.ages,
            )
        return simulate(
            internal_params=internal_params,
            initial_states=initial_states,
            initial_regimes=initial_regimes,
            internal_regimes=self.internal_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            logger=get_logger(debug_mode=debug_mode),
            V_arr_dict=V_arr_dict,
            ages=self.ages,
            seed=seed,
        )

    def solve_and_simulate(
        self,
        params: UserParams,
        initial_states: Mapping[str, Array],
        initial_regimes: list[RegimeName],
        *,
        check_initial_conditions: bool = True,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> SimulationResult:
        """Solve and then simulate the model in one call.

        Args:
            params: Model parameters matching the template from self.params_template.
                Parameters can be provided at exactly one of three levels:
                - Model level: {"arg_0": 0.0} - propagates to all functions needing
                  arg_0
                - Regime level: {"regime_0": {"arg_0": 0.0}} - propagates within
                  regime_0
                - Function level: {"regime_0": {"func": {"arg_0": 0.0}}} - direct
                  specification
            initial_states: Mapping of state names to arrays. All arrays must have the
                same length (number of subjects). Each state name should correspond to a
                state variable defined in at least one regime.
            initial_regimes: List of regime names the subjects start in.
            check_initial_conditions: Whether to validate initial states and regimes.
            seed: Random seed.
            debug_mode: Whether to enable debug logging.

        Returns:
            SimulationResult object. Call .to_dataframe() to get a pandas DataFrame,
            optionally with additional_targets.

        """
        internal_params = self._process_params(params)
        if check_initial_conditions:
            validate_initial_conditions(
                initial_states=initial_states,
                initial_regimes=initial_regimes,
                internal_regimes=self.internal_regimes,
                internal_params=internal_params,
                ages=self.ages,
            )
        V_arr_dict = solve(
            internal_params=internal_params,
            ages=self.ages,
            internal_regimes=self.internal_regimes,
            logger=get_logger(debug_mode=debug_mode),
        )
        return simulate(
            internal_params=internal_params,
            initial_states=initial_states,
            initial_regimes=initial_regimes,
            internal_regimes=self.internal_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            logger=get_logger(debug_mode=debug_mode),
            V_arr_dict=V_arr_dict,
            ages=self.ages,
            seed=seed,
        )


def _build_regimes_and_template(
    *,
    regimes: Mapping[str, Regime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
    fixed_params: UserParams,
) -> tuple[
    MappingProxyType[RegimeName, InternalRegime],
    ParamsTemplate,
    ParamsTemplate,
    MappingProxyType[str, str],
]:
    """Build internal regimes and params template in a single pass.

    Composes regime processing, template creation, and optional fixed-param partialling
    so that each result is computed exactly once.

    Returns:
        Tuple of (internal_regimes, internal_params_template, user_params_template,
        collapsed_pair_keys). The internal template always uses pair keys; the
        user-facing template collapses pair keys when all boundaries from a source
        regime are structurally identical. `collapsed_pair_keys` maps each collapsed
        pair key to its source regime name, for translating error messages.

    """
    internal_regimes, internal_template = process_regimes(
        regimes=regimes,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=enable_jit,
    )

    if fixed_params:
        fixed_internal = _resolve_fixed_params(
            fixed_params=dict(fixed_params), template=internal_template
        )
        if fixed_internal:
            internal_regimes = _partial_fixed_params_into_regimes(
                internal_regimes=internal_regimes, fixed_internal=fixed_internal
            )
            internal_template = _remove_fixed_from_template(
                template=internal_template, fixed_internal=fixed_internal
            )

    user_template = collapse_pair_keys(internal_template)

    # Build mapping of collapsed pair keys to source regimes
    collapsed = {
        pk: pk.split(REGIME_PAIR_SEPARATOR, 1)[0]
        for pk in internal_template
        if REGIME_PAIR_SEPARATOR in pk and pk not in user_template
    }

    return (
        internal_regimes,
        internal_template,
        user_template,
        MappingProxyType(collapsed),
    )


def _validate_model_inputs(  # noqa: C901
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

    # Validate regime names don't contain the boundary pair separator
    invalid_boundary = [name for name in regimes if REGIME_PAIR_SEPARATOR in name]
    if invalid_boundary:
        error_messages.append(
            f"Regime names cannot contain '{REGIME_PAIR_SEPARATOR}' "
            f"(reserved for boundary pairs). Invalid: {invalid_boundary}."
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
    from lcm.input_processing.process_transitions import (  # noqa: PLC0415
        collect_regime_functions,
    )

    error_messages = []

    for regime_name, regime in regimes.items():
        all_funcs, _ = collect_regime_functions(
            regime_name=regime_name,
            regimes=regimes,
        )
        variable_info = get_variable_info(regime, user_functions=all_funcs)
        is_used = (
            variable_info["enters_concurrent_valuation"]
            | variable_info["enters_transition"]
        )
        unused_variables = variable_info.index[~is_used].tolist()

        if unused_variables:
            unused_states = [
                v for v in unused_variables if variable_info.loc[v, "is_state"]
            ]
            unused_actions = [
                v for v in unused_variables if variable_info.loc[v, "is_action"]
            ]

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


def _find_candidates(
    *,
    key: str,
    params_flat: Mapping[str, object],
) -> list[str]:
    """Find candidate matches for a template key.

    Checks exact, pair, source-regime, and model levels. For pair keys
    (e.g., `working_to_retired__next_wealth__rate`), also checks
    source-regime matches for backward compatibility.

    """
    parts = key.split(QNAME_DELIMITER)
    param_name = parts[-1]
    candidates: list[str] = []

    if key in params_flat:
        candidates.append(key)

    if len(parts) == 3:  # noqa: PLR2004
        top_key = parts[0]
        top_level_key = f"{top_key}{QNAME_DELIMITER}{param_name}"
        if top_level_key in params_flat:
            candidates.append(top_level_key)

        # Source-regime matching for pair keys (e.g., "working_to_retired")
        if REGIME_PAIR_SEPARATOR in top_key:
            source = top_key.split(REGIME_PAIR_SEPARATOR, 1)[0]
            func_name = parts[1]
            source_func_key = (
                f"{source}{QNAME_DELIMITER}{func_name}{QNAME_DELIMITER}{param_name}"
            )
            if source_func_key in params_flat:
                candidates.append(source_func_key)
            source_regime_key = f"{source}{QNAME_DELIMITER}{param_name}"
            if source_regime_key in params_flat:
                candidates.append(source_regime_key)

    if param_name in params_flat:
        candidates.append(param_name)

    return candidates


def _resolve_fixed_params(
    *,
    fixed_params: dict[str, object],
    template: ParamsTemplate,
) -> InternalParams:
    """Resolve fixed_params against the params template.

    Like process_params, supports model/regime/function level specification, but
    does NOT require all template keys to be present — only matches what's provided.

    Returns a flat model-level mapping with regime-prefixed keys.

    """
    template_flat = flatten_to_qnames(template)
    params_flat = flatten_to_qnames(fixed_params)

    result_flat: dict[str, object] = {}
    used_keys: set[str] = set()

    for key in template_flat:
        candidates = _find_candidates(key=key, params_flat=params_flat)

        if len(candidates) > 1:
            raise ModelInitializationError(
                f"Ambiguous fixed_params specification for {key!r}. "
                f"Found values at: {candidates}"
            )
        if candidates:
            result_flat[key] = params_flat[candidates[0]]
            used_keys.add(candidates[0])

    unknown = set(params_flat) - used_keys
    if unknown:
        raise ModelInitializationError(
            f"Unknown keys in fixed_params: {sorted(unknown)}"
        )

    return ensure_containers_are_immutable(result_flat)  # ty: ignore[invalid-return-type]


def _remove_fixed_from_template(
    *,
    template: ParamsTemplate,
    fixed_internal: InternalParams,
) -> ParamsTemplate:
    """Remove fixed params from the params template.

    After partialling fixed params into compiled functions, remove them from the
    template so users don't need to supply them at solve/simulate time.

    """
    result: dict[str, dict[str, dict[str, type | tuple[int, ...]]]] = {}
    for regime_name, regime_template in template.items():
        new_regime: dict[str, dict[str, type | tuple[int, ...]]] = {}
        for func_name, func_params in regime_template.items():
            new_func_params = {
                param_name: param_type
                for param_name, param_type in func_params.items()
                if QNAME_DELIMITER.join((regime_name, func_name, param_name))
                not in fixed_internal
            }
            if new_func_params:
                new_regime[func_name] = new_func_params
        if new_regime:
            result[regime_name] = new_regime
        else:
            # Keep regime key even if empty (needed by process_params)
            result[regime_name] = {}
    return ensure_containers_are_immutable(result)  # ty: ignore[invalid-return-type]


def _partial_fixed_params_into_regimes(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    fixed_internal: InternalParams,
) -> MappingProxyType[RegimeName, InternalRegime]:
    """Partial fixed params into all compiled functions on each InternalRegime.

    Partials the full model-level fixed params dict into every compiled function.
    Q-functions and next-state functions use `allow_only_kwargs(enforce=False)`
    wrappers that silently ignore extra kwargs, so the full unfiltered dict is
    safe. `regime_transition_probs` lacks this wrapper, so we filter to matching
    kwargs to avoid signature mismatches in `inspect.signature`.

    """
    model_fixed = dict(fixed_internal)
    if not model_fixed:
        return internal_regimes

    result = {}
    for name, regime in internal_regimes.items():
        # Partial into per-period solve functions
        new_max_Q = {
            period: functools.partial(func, **model_fixed)
            for period, func in regime.max_Q_over_a_functions.items()
        }

        # Partial into per-period simulate functions
        new_argmax_max_Q = {
            period: functools.partial(func, **model_fixed)
            for period, func in regime.argmax_and_max_Q_over_a_functions.items()
        }

        # Partial into next-state simulation function
        new_next_state = functools.partial(
            regime.next_state_simulation_function, **model_fixed
        )

        # Partial into regime transition probs — only include params that the
        # function actually accepts to avoid signature mismatches during
        # inspect.signature (used by dags.concatenate_functions in to_dataframe).
        if regime.regime_transition_probs is not None:
            new_regime_tp = PhaseVariantContainer(
                solve=functools.partial(
                    regime.regime_transition_probs.solve,
                    **_filter_kwargs_for_func(
                        func=regime.regime_transition_probs.solve, kwargs=model_fixed
                    ),
                ),
                simulate=functools.partial(
                    regime.regime_transition_probs.simulate,
                    **_filter_kwargs_for_func(
                        func=regime.regime_transition_probs.simulate,
                        kwargs=model_fixed,
                    ),
                ),
            )
        else:
            new_regime_tp = None

        # Also update the nested internal_functions so simulation code
        # (which reads from internal_functions.regime_transition_probs) sees
        # the partialled version.
        new_internal_functions = dataclasses.replace(
            regime.internal_functions,
            regime_transition_probs=new_regime_tp,
        )

        result[name] = dataclasses.replace(
            regime,
            max_Q_over_a_functions=MappingProxyType(new_max_Q),
            argmax_and_max_Q_over_a_functions=MappingProxyType(new_argmax_max_Q),
            next_state_simulation_function=new_next_state,
            regime_transition_probs=new_regime_tp,
            internal_functions=new_internal_functions,
            resolved_fixed_params=MappingProxyType(model_fixed),
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
