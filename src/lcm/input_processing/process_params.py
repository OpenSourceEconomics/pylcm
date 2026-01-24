"""Process user-provided params into internal params."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.typing import InternalParams
from lcm.utils import REGIME_SEPARATOR


def process_params(
    params: Mapping[str, Any],
    params_template: InternalParams,
) -> InternalParams:
    """Process user-provided params into internal params.

    Users can provide parameters at different levels:
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
    template = _to_dict(params_template)

    # Collect all known names at each level from template
    regime_names = set(template.keys())
    function_names: set[str] = set()
    argument_names: set[str] = set()

    for regime_template in template.values():
        for key, val in regime_template.items():
            if isinstance(val, (dict, Mapping)):
                function_names.add(key)
                argument_names.update(val.keys())
            else:
                # Top-level param like discount_factor, treat as argument
                argument_names.add(key)

    # Process params with propagation
    errors: list[str] = []
    result = _process_with_propagation(
        params=_to_dict(params),
        template=template,
        regime_names=regime_names,
        function_names=function_names,
        argument_names=argument_names,
        errors=errors,
    )

    if errors:
        raise InvalidParamsError("\n".join(errors))

    return _to_mapping_proxy(result)


def _process_with_propagation(
    params: dict[str, Any],
    template: dict[str, Any],
    regime_names: set[str],
    function_names: set[str],
    argument_names: set[str],
    errors: list[str],
) -> dict[str, Any]:
    """Process params with propagation from higher levels.

    Args:
        params: User-provided params dict.
        template: The params template.
        regime_names: Set of all regime names.
        function_names: Set of all function names.
        argument_names: Set of all argument names.
        errors: List to collect error messages.

    Returns:
        Processed params matching template structure.

    """
    # Separate params into model-level args and regime-level entries
    model_level_args: dict[str, Any] = {}
    regime_entries: dict[str, Any] = {}

    for key, val in params.items():
        if key in regime_names:
            regime_entries[key] = val
        elif key in argument_names:
            model_level_args[key] = val
        else:
            errors.append(f"Unknown key at model level: {key!r}")

    # Process each regime
    result: dict[str, Any] = {}
    for regime_name, regime_template in template.items():
        regime_params = regime_entries.get(regime_name, {})
        if not isinstance(regime_params, (dict, Mapping)):
            got = type(regime_params).__name__
            errors.append(f"Expected dict for regime {regime_name!r}, got {got}")
            regime_params = {}

        result[regime_name] = _process_regime(
            regime_name=regime_name,
            regime_params=dict(regime_params),
            regime_template=regime_template,
            model_level_args=model_level_args,
            function_names=function_names,
            argument_names=argument_names,
            errors=errors,
        )

    return result


def _process_regime(  # noqa: C901
    regime_name: str,
    regime_params: dict[str, Any],
    regime_template: dict[str, Any],
    model_level_args: dict[str, Any],
    function_names: set[str],
    argument_names: set[str],
    errors: list[str],
) -> dict[str, Any]:
    """Process a single regime's params with propagation.

    Args:
        regime_name: Name of the regime.
        regime_params: User params for this regime.
        regime_template: Template for this regime.
        model_level_args: Arguments specified at model level.
        function_names: Set of all function names.
        argument_names: Set of all argument names.
        errors: List to collect error messages.

    Returns:
        Processed regime params matching template structure.

    """
    # Separate regime_params into regime-level args and function-level entries
    regime_level_args: dict[str, Any] = {}
    function_entries: dict[str, Any] = {}

    for key, val in regime_params.items():
        if key in function_names:
            function_entries[key] = val
        elif key in argument_names:
            regime_level_args[key] = val
        else:
            errors.append(f"Unknown key in regime {regime_name!r}: {key!r}")

    # Check for ambiguity between model and regime level
    ambiguous = set(model_level_args.keys()) & set(regime_level_args.keys())
    if ambiguous:
        raise InvalidNameError(
            f"Ambiguous parameter specification in regime {regime_name!r}: "
            f"{sorted(ambiguous)} specified at both model and regime level"
        )

    # Merge model and regime level args (regime takes precedence if no ambiguity)
    inherited_args = {**model_level_args, **regime_level_args}

    # Process each function in the template
    result: dict[str, Any] = {}
    for key, template_val in regime_template.items():
        if isinstance(template_val, (dict, Mapping)):
            # It's a function with params
            func_params = function_entries.get(key, {})
            if not isinstance(func_params, (dict, Mapping)):
                errors.append(
                    f"Expected dict for {regime_name}[{key!r}], "
                    f"got {type(func_params).__name__}"
                )
                func_params = {}

            result[key] = _process_function(
                regime_name=regime_name,
                func_name=key,
                func_params=dict(func_params),
                func_template=dict(template_val),
                inherited_args=inherited_args,
                errors=errors,
            )
        # It's a top-level param like discount_factor
        elif key in regime_level_args:
            # Check ambiguity with model level
            if key in model_level_args:
                raise InvalidNameError(
                    f"Ambiguous parameter specification: {key!r} specified at "
                    f"both model level and in regime {regime_name!r}"
                )
            result[key] = regime_level_args[key]
        elif key in model_level_args:
            result[key] = model_level_args[key]
        # If key not in either, it's not provided - OK for optional params

    return result


def _process_function(
    regime_name: str,
    func_name: str,
    func_params: dict[str, Any],
    func_template: dict[str, type],
    inherited_args: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """Process a single function's params with propagation.

    Args:
        regime_name: Name of the regime.
        func_name: Name of the function.
        func_params: User params for this function.
        func_template: Template for this function (arg_name -> type).
        inherited_args: Arguments inherited from model/regime level.
        errors: List to collect error messages.

    Returns:
        Processed function params matching template structure.

    """
    # Check for ambiguity between inherited and function level
    ambiguous = set(inherited_args.keys()) & set(func_params.keys())
    if ambiguous:
        raise InvalidNameError(
            f"Ambiguous parameter specification in {regime_name}[{func_name!r}]: "
            f"{sorted(ambiguous)} specified at multiple levels"
        )

    # Check for unexpected keys in func_params
    expected_keys = set(func_template.keys())
    extra = set(func_params.keys()) - expected_keys
    if extra:
        errors.append(
            f"Unexpected keys in {regime_name}[{func_name!r}]: {sorted(extra)}"
        )

    # Build result: use func_params if provided, else inherited, else skip
    result: dict[str, Any] = {}
    for arg_name in func_template:
        if arg_name in func_params:
            result[arg_name] = func_params[arg_name]
        elif arg_name in inherited_args:
            result[arg_name] = inherited_args[arg_name]
        # If not in either, leave it out - it's optional

    return result


def create_params_template(  # noqa: C901
    internal_regimes: Mapping[str, Any],
) -> InternalParams:
    """Create params_template from internal regimes and validate name uniqueness.

    This function validates that regime names, function names, and argument names
    are disjoint sets to enable unambiguous parameter propagation.

    Args:
        internal_regimes: Mapping of regime names to InternalRegime instances.

    Returns:
        The params_template as an immutable MappingProxyType.

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

    return _to_mapping_proxy(template)


def _to_dict(m: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively convert a Mapping to a dict."""
    result = {}
    for k, v in m.items():
        if isinstance(v, Mapping):
            result[k] = _to_dict(v)
        else:
            result[k] = v
    return result


def _to_mapping_proxy(d: dict[str, Any]) -> MappingProxyType[str, Any]:
    """Recursively convert a dict to MappingProxyType."""
    return MappingProxyType(
        {k: _to_mapping_proxy(v) if isinstance(v, dict) else v for k, v in d.items()}
    )
