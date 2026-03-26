"""Process user-provided params into internal params."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from dags.tree import QNAME_DELIMITER, qname_from_tree_path, tree_path_from_qname

from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.interfaces import InternalRegime
from lcm.typing import (
    InternalParams,
    ParamsTemplate,
    RegimeName,
    RegimeParamsTemplate,
)
from lcm.utils import (
    ensure_containers_are_immutable,
    flatten_regime_namespace,
)

_NUM_PARTS_FUNCTION_PARAM = 3


def process_params(
    *,
    params: Mapping,
    params_template: ParamsTemplate,
) -> InternalParams:
    """Process user-provided params into internal params.

    Users can provide parameters at exactly one of three levels:

    - Model level: `{"arg_0": 0.0}` — propagates to all functions needing arg_0
    - Regime level: `{"regime_0": {"arg_0": 0.0}}` — propagates within regime_0
    - Function level: `{"regime_0": {"func": {"arg_0": 0.0}}}` — direct specification

    The output always matches the params_template skeleton.

    Args:
        params: User-provided parameters dictionary.
        params_template: Template from `model.get_params_template()`.

    Returns:
        Immutable mapping with the same structure as params_template.

    Raises:
        InvalidParamsError: If params contains unexpected keys or type mismatches.
        InvalidNameError: If the same parameter is specified at multiple levels.

    """
    return broadcast_to_template(params=params, template=params_template, required=True)


def broadcast_to_template(
    *,
    params: Mapping,
    template: Mapping[str, Mapping],
    required: bool = True,
) -> InternalParams:
    """Broadcast user params to template shape via 3-level resolution.

    For each template qname, search for a matching user value at:

    1. Exact match: `regime__function__param`
    2. Regime level: `regime__param`
    3. Model level: `param`

    Args:
        params: User-provided values at any nesting depth.
        template: Target structure defining all valid 3-part keys.
        required: If True, raise when any template key has no match.

    Returns:
        Immutable mapping from regime name to mapping of `func__param`
        keys to resolved values. All regime keys from the template are
        present (possibly with empty inner mappings when `required` is
        False).

    Raises:
        InvalidParamsError: On missing required keys or unknown user keys.
        InvalidNameError: On ambiguous multi-level specification.

    """
    template_flat = flatten_regime_namespace(template)
    params_flat = flatten_regime_namespace(params)

    result: dict[str, dict[str, object]] = {name: {} for name in template}
    used_keys: set[str] = set()

    for qname in template_flat:
        candidates = _find_candidates(qname=qname, params_flat=params_flat)

        if len(candidates) > 1:
            raise InvalidNameError(
                f"Ambiguous parameter specification for {qname!r}. "
                f"Found values at: {candidates}"
            )

        if candidates:
            chosen = candidates[0]
            path = tree_path_from_qname(qname)
            regime = path[0]
            remainder = qname_from_tree_path(path[1:])
            result[regime][remainder] = params_flat[chosen]
            used_keys.add(chosen)
        elif required:
            raise InvalidParamsError(f"Missing required parameter: {qname!r}")

    unknown = set(params_flat) - used_keys
    if unknown:
        raise InvalidParamsError(f"Unknown keys: {sorted(unknown)}")

    return cast(
        "InternalParams",
        MappingProxyType({k: MappingProxyType(v) for k, v in result.items()}),
    )


def _find_candidates(
    *,
    qname: str,
    params_flat: Mapping[str, object],
) -> list[str]:
    """Find candidate matches for a template qname at exact / regime / model levels."""
    tree_path = tree_path_from_qname(qname)
    param_name = tree_path[-1]
    candidates: list[str] = []

    if qname in params_flat:
        candidates.append(qname)

    if len(tree_path) == _NUM_PARTS_FUNCTION_PARAM:
        regime_level_qname = qname_from_tree_path((tree_path[0], param_name))
        if regime_level_qname in params_flat:
            candidates.append(regime_level_qname)

    if param_name in params_flat:
        candidates.append(param_name)

    return candidates


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


def get_flat_param_names(regime_params_template: RegimeParamsTemplate) -> set[str]:
    """Get all flat parameter names from a regime params template.

    Converts nested template entries like {"utility": {"risk_aversion": type}} to
    flat names like "utility__risk_aversion".

    """
    result = set()
    for key, value in regime_params_template.items():
        if isinstance(value, Mapping):
            for param_name in value:
                result.add(qname_from_tree_path((key, param_name)))
    return result
