"""Model initialization helpers: validation, template creation, fixed-param handling.

Extracted from `model.py` to keep the `Model` class focused on its public API.

"""

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType

from dags import get_ancestors
from dags.tree import QNAME_DELIMITER, qname_from_tree_path
from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import InvalidParamsError, ModelInitializationError, format_messages
from lcm.pandas_utils import convert_series_in_params, has_series
from lcm.params import MappingLeaf
from lcm.params.processing import (
    broadcast_to_template,
    create_params_template,
)
from lcm.params.sequence_leaf import SequenceLeaf
from lcm.regime import Regime
from lcm.regime_building.h_dag import get_dag_targets_consumed_by_H
from lcm.regime_building.processing import (
    InternalRegime,
    process_regimes,
)
from lcm.typing import (
    FunctionName,
    InternalParams,
    ParamsTemplate,
    RegimeName,
    RegimeNamesToIds,
    UserParams,
)
from lcm.utils.containers import get_field_names_and_values


def build_regimes_and_template(
    *,
    ages: AgeGrid,
    regimes: Mapping[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
    fixed_params: UserParams,
) -> tuple[MappingProxyType[RegimeName, InternalRegime], ParamsTemplate]:
    """Build internal regimes and params template in a single pass.

    Compose regime processing, template creation, and optional fixed-param partialling
    so that each result is computed exactly once.

    Args:
        ages: Age grid for the model.
        regimes: Mapping of regime names to Regime instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        enable_jit: Whether to JIT-compile regime functions.
        fixed_params: Parameters to fix at model initialization.

    Returns:
        Tuple of (internal_regimes, params_template).

    """
    if not fixed_params:
        internal_regimes = process_regimes(
            ages=ages,
            regimes=regimes,
            regime_names_to_ids=regime_names_to_ids,
            enable_jit=enable_jit,
        )
        params_template = create_params_template(internal_regimes)
    else:
        internal_regimes, params_template = (
            _build_regimes_and_template_with_fixed_params(
                ages=ages,
                regimes=regimes,
                regime_names_to_ids=regime_names_to_ids,
                enable_jit=enable_jit,
                fixed_params=fixed_params,
            )
        )

    return internal_regimes, params_template


def _build_regimes_and_template_with_fixed_params(
    *,
    ages: AgeGrid,
    regimes: Mapping[RegimeName, Regime],
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
    fixed_params: UserParams,
) -> tuple[MappingProxyType[RegimeName, InternalRegime], ParamsTemplate]:
    """Build internal regimes and template, then partial in fixed params.

    Args:
        ages: Age grid for the model.
        regimes: Mapping of regime names to Regime instances.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        enable_jit: Whether to JIT-compile regime functions.
        fixed_params: Parameters to fix at model initialization.

    Returns:
        Tuple of internal_regimes and params_template with fixed params
        partialled in.

    """
    raw_internal_regimes = process_regimes(
        ages=ages,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=enable_jit,
    )
    raw_params_template = create_params_template(raw_internal_regimes)

    fixed_internal = _resolve_fixed_params(
        fixed_params=dict(fixed_params), template=raw_params_template
    )
    if has_series(fixed_internal):
        fixed_internal = convert_series_in_params(
            internal_params=fixed_internal,
            ages=ages,
            regimes=regimes,
            regime_names_to_ids=regime_names_to_ids,
        )
    _validate_param_types(fixed_internal)

    return (
        _partial_fixed_params_into_regimes(
            internal_regimes=raw_internal_regimes,
            fixed_internal=fixed_internal,
        ),
        _remove_fixed_params_from_template(
            template=raw_params_template,
            fixed_internal=fixed_internal,
        ),
    )


def validate_model_inputs(
    *,
    n_periods: int,
    regimes: Mapping[RegimeName, Regime],
    regime_id_class: type,
    n_subjects: int | None = None,
) -> None:
    """Validate model constructor inputs."""
    _fail_if_invalid_n_subjects(n_subjects=n_subjects)

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


def _fail_if_invalid_n_subjects(*, n_subjects: int | None) -> None:
    """Raise TypeError if non-int, ValueError if non-positive."""
    if n_subjects is None:
        return
    # `bool` is a subclass of `int`; reject explicitly so True/False don't slip through.
    if not isinstance(n_subjects, int) or isinstance(n_subjects, bool):
        msg = f"n_subjects must be an int or None, got {type(n_subjects).__name__}."
        raise TypeError(msg)
    if n_subjects <= 0:
        msg = f"n_subjects must be a positive integer, got {n_subjects}."
        raise ValueError(msg)


def _validate_all_variables_used(regimes: Mapping[RegimeName, Regime]) -> list[str]:
    """Validate that all states and actions are used somewhere in each regime.

    Each state or action must appear in at least one of:
    - The concurrent valuation (utility or constraints)
    - A transition function
    - A regime function whose output H consumes at the Bellman step

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
            *get_dag_targets_consumed_by_H(user_functions),
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


def _remove_fixed_params_from_template(
    *,
    template: ParamsTemplate,
    fixed_internal: InternalParams,
) -> ParamsTemplate:
    """Remove fixed params from the params template.

    After partialling fixed params into compiled functions, remove them from the
    template so users don't need to supply them at solve/simulate time.

    """
    result: dict[RegimeName, dict[FunctionName, dict[str, str]]] = {}
    for regime_name, regime_template in template.items():
        regime_fixed = fixed_internal.get(regime_name, MappingProxyType({}))
        new_regime: dict[FunctionName, dict[str, str]] = {}
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
    result: dict[RegimeName, InternalRegime] = {}
    for regime_name, regime in internal_regimes.items():
        regime_fixed = dict(fixed_internal.get(regime_name, MappingProxyType({})))
        if not regime_fixed:
            result[regime_name] = regime
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

        result[regime_name] = dataclasses.replace(
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
