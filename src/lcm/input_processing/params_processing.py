"""Process user-provided params into internal params."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from dags.tree import QNAME_DELIMITER

from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.interfaces import InternalRegime
from lcm.typing import (
    InternalParams,
    ParamsTemplate,
    RegimeName,
    RegimeParamsTemplate,
    UserParams,
)
from lcm.utils import (
    ensure_containers_are_immutable,
    flatten_regime_namespace,
)

_NUM_PARTS_FUNCTION_PARAM = 3


def process_params(
    *,
    params: UserParams,
    params_template: ParamsTemplate,
) -> InternalParams:
    """Process user-provided params into internal params.

    Users can provide parameters at exactly one of three levels:

    - Model level: {"arg_0": 0.0} - propagates to all functions needing arg_0
    - Regime level: {"regime_0": {"arg_0": 0.0}} - propagates within regime_0
    - Function level: {"regime_0": {"func": {"arg_0": 0.0}}} - direct specification

    The output always matches the params_template skeleton.

    Args:
        params: User-provided parameters dictionary.
        params_template: Template from model.params_template.

    Returns:
        internal_params as an immutable MappingProxyType with the same structure
        as params_template.

    Raises:
        InvalidParamsError: If params contains unexpected keys or type mismatches.
        InvalidNameError: If the same parameter is specified at multiple levels.

    """
    template_flat = flatten_regime_namespace(params_template)
    params_flat = flatten_regime_namespace(params)

    result_flat = {}
    used_keys: set[str] = set()

    for key in template_flat:
        parts = key.split(QNAME_DELIMITER)
        param_name = parts[-1]

        candidates = []

        # 1. Exact match (e.g. regime__function__param or regime__param)
        if key in params_flat:
            candidates.append(key)

        # 2. Regime level (if key is function level: regime__function__param)
        # We want to check for regime__param
        if len(parts) == _NUM_PARTS_FUNCTION_PARAM:
            regime = parts[0]
            regime_level_key = f"{regime}{QNAME_DELIMITER}{param_name}"
            # Check if this regime-level key was provided in params
            if regime_level_key in params_flat:
                candidates.append(regime_level_key)

        # 3. Model level (Global: param)
        if param_name in params_flat:
            candidates.append(param_name)

        # Check for ambiguity
        if len(candidates) > 1:
            raise InvalidNameError(
                f"Ambiguous parameter specification for {key!r}. "
                f"Found values at: {candidates}"
            )

        if not candidates:
            raise InvalidParamsError(f"Missing required parameter: {key!r}")

        chosen_key = candidates[0]
        result_flat[key] = params_flat[chosen_key]
        used_keys.add(chosen_key)

    # Check for unknown keys
    # Keys in params that were not used to satisfy any template requirement
    unknown_keys = set(params_flat.keys()) - used_keys
    if unknown_keys:
        raise InvalidParamsError(f"Unknown keys: {sorted(unknown_keys)}")

    # Split flat keys into per-regime dicts and ensure all regimes are present
    result = {name: {} for name in params_template}
    for key, value in result_flat.items():
        regime_name, remainder = key.split(QNAME_DELIMITER, 1)
        result[regime_name][remainder] = value

    _validate_param_shapes(result, params_template)

    return cast("InternalParams", ensure_containers_are_immutable(result))


def create_params_template(  # noqa: C901
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
) -> ParamsTemplate:
    """Create params_template from internal regimes and validate name uniqueness.

    This function validates that regime names, function names, and argument names
    are disjoint sets to enable unambiguous parameter propagation.

    Args:
        internal_regimes: Mapping of regime names to InternalRegime instances.

    Returns:
        The parameter template.

    Raises:
        InvalidNameError: If names are not disjoint or contain the separator.

    """
    template: dict[str, Any] = {}
    regime_names: set[str] = set()
    function_names: set[str] = set()
    arg_names: set[str] = set()

    for name, regime in internal_regimes.items():
        regime_names.add(name)
        regime_template = dict(regime.regime_params_template)
        template[name] = regime_template

        for key, val in regime_template.items():
            if isinstance(val, (dict, Mapping)):
                function_names.add(key)
                for arg_name in val:
                    # Check for separator in argument names
                    if QNAME_DELIMITER in arg_name:
                        raise InvalidNameError(
                            f"Argument name {arg_name!r} in function {key!r} "
                            f"cannot contain the separator '{QNAME_DELIMITER}'"
                        )
                    arg_names.add(arg_name)
            else:
                raise InvalidNameError(
                    f"Parameter {key!r} in regime {name!r} must be nested under "
                    f"a function name, e.g., {{'function_name': {{'{key}': type}}}}"
                )

    # Check for separator in regime names
    for name in regime_names:
        if QNAME_DELIMITER in name:
            raise InvalidNameError(
                f"Regime name {name!r} cannot contain the separator '{QNAME_DELIMITER}'"
            )

    # Check for separator in function names
    for name in function_names:
        if QNAME_DELIMITER in name:
            raise InvalidNameError(
                f"Function name {name!r} cannot contain the separator "
                f"'{QNAME_DELIMITER}'"
            )

    # Check that names are disjoint
    regime_func_overlap = regime_names & function_names
    if regime_func_overlap:
        raise InvalidNameError(
            f"Regime names and function names must be disjoint. "
            f"Overlap: {sorted(regime_func_overlap)}"
        )

    regime_arg_overlap = regime_names & arg_names
    if regime_arg_overlap:
        raise InvalidNameError(
            f"Regime names and argument names must be disjoint. "
            f"Overlap: {sorted(regime_arg_overlap)}"
        )

    # Note: Function names CAN overlap with argument names across regimes.
    # This happens when a function output in one regime is a parameter in another.
    # E.g., labor_income is a function in 'working' but a param in 'retired'.

    return ensure_containers_are_immutable(template)


def _validate_param_shapes(
    resolved: dict[str, dict[str, Any]],
    template: ParamsTemplate,
) -> None:
    """Validate that array parameters match expected shape suffixes.

    When a template value is a tuple of ints (e.g. (2, 2) for a transition matrix),
    check that the corresponding resolved value has a matching shape suffix.

    """
    for regime_name, regime_template in template.items():
        for func_name, func_params in regime_template.items():
            if not isinstance(func_params, Mapping):
                continue
            for param_name, expected in func_params.items():
                if not (
                    isinstance(expected, tuple)
                    and all(isinstance(d, int) for d in expected)
                ):
                    continue
                flat_key = f"{func_name}{QNAME_DELIMITER}{param_name}"
                value = resolved.get(regime_name, {}).get(flat_key)
                if value is None or not hasattr(value, "shape"):
                    continue
                n = len(expected)
                if len(value.shape) < n or value.shape[-n:] != expected:
                    raise InvalidParamsError(
                        f"Parameter '{regime_name}/{func_name}/{param_name}' "
                        f"has shape {value.shape}, expected suffix {expected}."
                    )


def get_flat_param_names(regime_params_template: RegimeParamsTemplate) -> set[str]:
    """Get all flat parameter names from a regime params template.

    Converts nested template entries like {"utility": {"risk_aversion": type}} to
    flat names like "utility__risk_aversion".

    """
    result = set()
    for key, value in regime_params_template.items():
        if isinstance(value, Mapping):
            for param_name in value:
                result.add(f"{key}{QNAME_DELIMITER}{param_name}")
    return result
