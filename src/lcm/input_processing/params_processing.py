"""Process user-provided params into internal params."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.interfaces import InternalRegime
from lcm.typing import InternalParams, ParamsTemplate, RegimeName, UserParams
from lcm.utils import (
    REGIME_SEPARATOR,
    ensure_containers_are_immutable,
    flatten_regime_namespace,
    unflatten_regime_namespace,
)

_NUM_PARTS_FUNCTION_PARAM = 3


def process_params(  # noqa: C901
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
        parts = key.split(REGIME_SEPARATOR)
        param_name = parts[-1]

        candidates = []

        # 1. Exact match (e.g. regime__function__param or regime__param)
        if key in params_flat:
            candidates.append(key)

        # 2. Regime level (if key is function level: regime__function__param)
        # We want to check for regime__param
        if len(parts) == _NUM_PARTS_FUNCTION_PARAM:
            regime = parts[0]
            regime_level_key = f"{regime}{REGIME_SEPARATOR}{param_name}"
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

    result = unflatten_regime_namespace(result_flat)

    # Ensure all regimes from the template are present in the result
    # (even if they have no parameters)
    for regime_name in params_template:
        if regime_name not in result:
            result[regime_name] = {}

    return ensure_containers_are_immutable(result)


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
    argument_names: set[str] = set()

    for name, regime in internal_regimes.items():
        regime_names.add(name)
        regime_template = dict(regime.params_template)
        template[name] = regime_template

        for key, val in regime_template.items():
            if isinstance(val, (dict, Mapping)):
                function_names.add(key)
                for arg_name in val:
                    # Check for separator in argument names
                    if REGIME_SEPARATOR in arg_name:
                        raise InvalidNameError(
                            f"Argument name {arg_name!r} in function {key!r} "
                            f"cannot contain the separator '{REGIME_SEPARATOR}'"
                        )
                    argument_names.add(arg_name)
            else:
                # Top-level param like discount_factor
                argument_names.add(key)

    # Check for separator in regime names
    for name in regime_names:
        if REGIME_SEPARATOR in name:
            raise InvalidNameError(
                f"Regime name {name!r} cannot contain the separator "
                f"'{REGIME_SEPARATOR}'"
            )

    # Check for separator in function names
    for name in function_names:
        if REGIME_SEPARATOR in name:
            raise InvalidNameError(
                f"Function name {name!r} cannot contain the separator "
                f"'{REGIME_SEPARATOR}'"
            )

    # Check that names are disjoint
    regime_func_overlap = regime_names & function_names
    if regime_func_overlap:
        raise InvalidNameError(
            f"Regime names and function names must be disjoint. "
            f"Overlap: {sorted(regime_func_overlap)}"
        )

    regime_arg_overlap = regime_names & argument_names
    if regime_arg_overlap:
        raise InvalidNameError(
            f"Regime names and argument names must be disjoint. "
            f"Overlap: {sorted(regime_arg_overlap)}"
        )

    # Note: Function names CAN overlap with argument names across regimes.
    # This happens when a function output in one regime is a parameter in another.
    # E.g., labor_income is a function in 'working' but a param in 'retired'.

    return ensure_containers_are_immutable(template)
