import dags.tree as dt

from lcm.exceptions import InvalidNameError
from lcm.regime import Regime
from lcm.typing import RegimeParamsTemplate
from lcm.utils import ensure_containers_are_immutable


def create_regime_params_template(
    regime: Regime,
) -> RegimeParamsTemplate:
    """Create parameter template from a regime specification.

    Uses dags.tree.create_tree_with_input_types() to discover parameters and their
    type annotations from function signatures. Parameters are identified as function
    arguments that are not states, actions, auxiliary functions, or special variables
    (period, age, utility, continuation_value).

    Args:
        regime: The regime as provided by the user.

    Returns:
        The regime parameter template with type annotations as values. Contains a
        dictionary for each regime function containing the parameters required by that
        function.

    """
    # Collect all variables that H may receive: regime functions, special variables
    # (period, age) and continuation_value.
    H_variables = {*regime.functions, "period", "age", "continuation_value"}
    # Other functions may receive states/actions, too.
    variables = H_variables | {
        *regime.actions,
        *regime.states,
    }

    function_params = {}
    # Use dags.tree to discover parameters and their type annotations for each function.
    for name, func in regime.get_all_functions().items():
        tree = dt.create_tree_with_input_types({name: func})
        excl = H_variables if name == "H" else variables
        # Filter out variables to get only the parameters
        params = {k: v for k, v in sorted(tree.items()) if k not in excl}
        function_params[name] = params

    # Validate that no discovered H parameter shadows a state or action name.
    state_action_names = set(regime.states) | set(regime.actions)
    for func_name, params in function_params.items():
        shadows = set(params) & state_action_names
        if shadows:
            raise InvalidNameError(
                f"Function '{func_name}' has parameter(s) {sorted(shadows)} that "
                f"shadow state/action variable(s) with the same name. Please rename "
                f"the parameter(s) or the state(s)/action(s) to avoid ambiguity."
            )

    return ensure_containers_are_immutable(function_params)
