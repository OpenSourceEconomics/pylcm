"""Process user-provided params into internal params."""

from collections.abc import Mapping
from typing import cast

from dags.tree import QNAME_DELIMITER, flatten_to_qnames

from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.typing import (
    InternalParams,
    ParamsTemplate,
    UserParams,
)
from lcm.utils import ensure_containers_are_immutable

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
    template_flat = flatten_to_qnames(params_template)
    params_flat = flatten_to_qnames(params)

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

    return cast("InternalParams", ensure_containers_are_immutable(result_flat))


def validate_params_template(params_template: ParamsTemplate) -> None:  # noqa: C901
    """Validate regime parameter templates for uniqueness and naming rules.

    Validate that regime names, function names, and argument names
    are disjoint sets to enable unambiguous parameter propagation.

    Args:
        all_regime_params_templates: Immutable mapping of regime names to their
            parameter templates.

    Raises:
        InvalidNameError: If names are not disjoint or contain the separator.

    """
    function_names: set[str] = set()
    arg_names: set[str] = set()

    for name, regime_template in params_template.items():
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
    for name in params_template:
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
    regime_func_overlap = set(params_template) & function_names
    if regime_func_overlap:
        raise InvalidNameError(
            f"Regime names and function names must be disjoint. "
            f"Overlap: {sorted(regime_func_overlap)}"
        )

    regime_arg_overlap = set(params_template) & arg_names
    if regime_arg_overlap:
        raise InvalidNameError(
            f"Regime names and argument names must be disjoint. "
            f"Overlap: {sorted(regime_arg_overlap)}"
        )
