from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from lcm.exceptions import InvalidParamsError
from lcm.typing import InternalParams


def process_params(
    params: Mapping[str, Any],
    params_template: InternalParams,
) -> InternalParams:
    """Process user-provided params into internal params.

    Validates that the user-provided params don't contain unexpected keys, then
    ensures the internal_params has the same nested dict structure as params_template
    (filling in empty dicts where needed for missing function parameters).

    Args:
        params: User-provided parameters dictionary.
        params_template: Template from model.params_template.

    Returns:
        internal_params as an immutable MappingProxyType with the same structure
        as params_template.

    Raises:
        InvalidParamsError: If params contains unexpected keys or type mismatches.

    """
    errors: list[str] = []
    internal_params = _merge_with_template(
        dict(params), dict(params_template), "params", errors
    )

    if errors:
        raise InvalidParamsError("\n".join(errors))

    return _to_mapping_proxy(internal_params)


def _merge_with_template(
    params: dict,
    template: dict,
    path: str,
    errors: list[str],
) -> dict:
    """Recursively merge params with template, filling in missing empty dicts."""
    result = {}

    params_keys = set(params.keys())
    template_keys = set(template.keys())

    # Check for unexpected keys in params
    extra = params_keys - template_keys
    if extra:
        errors.append(f"Unexpected keys in {path}: {sorted(extra)}")

    for key in template_keys:
        template_val = template[key]
        template_is_dict = isinstance(template_val, (dict, Mapping))

        if key in params:
            params_val = params[key]
            params_is_dict = isinstance(params_val, (dict, Mapping))

            if params_is_dict and template_is_dict:
                # Both are dicts, recurse
                result[key] = _merge_with_template(
                    dict(params_val), dict(template_val), f"{path}[{key!r}]", errors
                )
            elif not params_is_dict and not template_is_dict:
                # Both are leaves, use the user's value
                result[key] = params_val
            else:
                # Type mismatch
                if params_is_dict:
                    errors.append(
                        f"Type mismatch at {path}[{key!r}]: expected leaf, got dict"
                    )
                else:
                    errors.append(
                        f"Type mismatch at {path}[{key!r}]: expected dict, got leaf"
                    )
                result[key] = params_val  # Still include to avoid KeyError
        # Key missing in params - fill in from template structure
        # For nested dicts, create empty structure; for leaves, skip them
        # (users are allowed to omit params they don't need, like discount_factor
        # for terminal regimes)
        elif template_is_dict:
            result[key] = _create_empty_structure(dict(template_val))
            # else: skip - user didn't provide this leaf value and that's OK

    return result


def _create_empty_structure(template: dict) -> dict:
    """Recursively create an empty nested dict structure matching the template."""
    result = {}
    for key, val in template.items():
        if isinstance(val, (dict, Mapping)):
            result[key] = _create_empty_structure(dict(val))
        # Skip leaf values - don't copy type annotations
    return result


def _to_mapping_proxy(d: dict) -> MappingProxyType:
    """Recursively convert a dict to MappingProxyType."""
    return MappingProxyType(
        {k: _to_mapping_proxy(v) if isinstance(v, dict) else v for k, v in d.items()}
    )
