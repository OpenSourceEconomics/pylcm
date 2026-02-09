import dags.tree as dt

from lcm.regime import Regime
from lcm.typing import RegimeParamsTemplate
from lcm.utils import ensure_containers_are_immutable

# The namespace for aggregation (Bellman equation) parameters like discount_factor.
# Matches the mathematical notation H(U, v) = U + β·v used in the docstrings.
AGGREGATION_FUNCTION_NAME = "_H"


def create_regime_params_template(
    regime: Regime,
    aggregation_params: dict[str, type] = {"discount_factor": float},  # noqa: B006
) -> RegimeParamsTemplate:
    """Create parameter template from a regime specification.

    Uses dags.tree.create_tree_with_input_types() to discover parameters and their
    type annotations from function signatures. Parameters are identified as function
    arguments that are not states, actions, auxiliary functions, or special variables
    (period, age).

    Args:
        regime: The regime as provided by the user.
        aggregation_params: Parameters for the Bellman aggregation function H(U, v).
            Default is {"discount_factor": float}. For other lifetime reward objectives,
            additional parameters may be required, for example
            {"discount_factor": float, "short_run_discount_factor": float} for
            beta-delta discounting.

    Returns:
        The regime parameter template with type annotations as values. Contains
        aggregation_params under the "_H" namespace, plus a dictionary for each
        regime function containing the parameters required by that function.

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

    base = {AGGREGATION_FUNCTION_NAME: aggregation_params} if aggregation_params else {}
    return ensure_containers_are_immutable(base | function_params)
