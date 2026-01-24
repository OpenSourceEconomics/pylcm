from types import MappingProxyType
from typing import Any

import dags.tree as dt

from lcm.regime import Regime


def create_params_template(
    regime: Regime,
    default_params: dict[str, type] = {"discount_factor": float},  # noqa: B006
) -> MappingProxyType[str, Any]:
    """Create parameter template from a regime specification.

    Args:
        regime: The regime as provided by the user.
        default_params: A dictionary of default parameters with their type annotations.
            Default is {"discount_factor": float}. For other lifetime reward objectives,
            additional parameters may be required, for example
            {"discount_factor": float, "short_run_discount_factor": float} for
            beta-delta discounting.

    Returns:
        The regime parameter template with type annotations as values.

    """
    function_params = _create_function_params(regime)

    return MappingProxyType(default_params | function_params)


def _create_function_params(
    regime: Regime,
) -> dict[str, dict[str, Any]]:
    """Get function parameters from a regime specification using dags.tree.

    Uses dags.tree.create_tree_with_input_types() to discover parameters and their
    type annotations from function signatures. Parameters are identified as function
    arguments that are not states, actions, auxiliary functions, or special variables
    (period, age).

    Args:
        regime: The regime as provided by the user.

    Returns:
        A dictionary for each regime function, containing the parameters required in the
        regime functions with their type annotations as values. If no annotation exists,
        the value is "no_annotation_found" (dags.tree default).

    """
    # Collect all regime variables: actions, states, special variables (period, age),
    # and auxiliary variables (regime function names). These are NOT parameters.
    variables = {
        *regime.functions,
        *regime.actions,
        *regime.states,
        "period",
        "age",
    }

    function_params = {}
    # Use dags.tree to discover parameters and their type annotations for each function.
    for name, func in regime.get_all_functions().items():
        tree = dt.create_tree_with_input_types({name: func})
        # Filter out variables to get only the parameters
        params = {k: v for k, v in sorted(tree.items()) if k not in variables}
        function_params[name] = params

    return function_params
