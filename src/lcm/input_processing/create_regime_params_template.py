from collections.abc import Mapping

import dags.tree as dt
from jax import Array

from lcm.exceptions import InvalidNameError
from lcm.grids import IrregSpacedGrid
from lcm.regime import Regime
from lcm.shocks import _ShockGrid
from lcm.typing import RegimeParamsTemplate
from lcm.utils import ensure_containers_are_immutable


def create_regime_params_template(
    regime: Regime,
) -> RegimeParamsTemplate:
    """Create parameter template from a regime specification.

    Uses dags.tree.create_tree_with_input_types() to discover parameters and their
    type annotations from function signatures. Parameters are identified as function
    arguments that are not states, actions, other regime functions, or special variables
    (period, age, continuation_value).

    Grids with runtime-supplied values (IrregSpacedGrid without points, _ShockGrid
    without full shock_params) add entries to the template under pseudo-function keys
    matching the state name.

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

    _discover_mapping_transition_params(regime, variables, function_params)

    # Validate that no discovered parameter shadows a state or action name.
    # In practice, only H can trigger this since other functions already exclude
    # states/actions from their parameter discovery.
    state_action_names = set(regime.states) | set(regime.actions)
    for func_name, params in function_params.items():
        shadows = set(params) & state_action_names
        if shadows:
            raise InvalidNameError(
                f"Function '{func_name}' has parameter(s) {sorted(shadows)} that "
                f"shadow state/action variable(s) with the same name. Please rename "
                f"the parameter(s) or the state(s)/action(s) to avoid ambiguity."
            )

    _add_runtime_grid_params(regime, function_params)

    return ensure_containers_are_immutable(function_params)


def _discover_mapping_transition_params(
    regime: Regime,
    variables: set[str],
    function_params: dict[str, dict[str, type]],
) -> None:
    """Discover parameters from per-boundary mapping transition callables.

    When a grid has a mapping transition `{(src, tgt): func}`, the callable may
    have parameters that belong to this (target) regime's template but are not
    visible through `get_all_functions()` (which returns an identity placeholder).

    """
    for state_name, grid in regime.states.items():
        trans = getattr(grid, "transition", None)
        if not isinstance(trans, Mapping):
            continue
        next_name = f"next_{state_name}"
        for func in trans.values():
            if func is None or not callable(func):
                continue
            tree = dt.create_tree_with_input_types({next_name: func})
            params = {k: v for k, v in sorted(tree.items()) if k not in variables}
            if params:
                existing = dict(function_params.get(next_name, {}))
                existing.update(params)
                function_params[next_name] = existing


def _add_runtime_grid_params(
    regime: Regime,
    function_params: dict[str, dict[str, type]],
) -> None:
    """Add entries for grids whose points/params are supplied at runtime."""
    for state_name, grid in regime.states.items():
        if isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime:
            if state_name in function_params:
                raise InvalidNameError(
                    f"IrregSpacedGrid state '{state_name}' (with runtime-supplied "
                    f"points) conflicts with a function of the same name in the "
                    f"regime."
                )
            function_params[state_name] = {"points": Array}
        elif isinstance(grid, _ShockGrid) and grid.params_to_pass_at_runtime:
            if state_name in function_params:
                raise InvalidNameError(
                    f"_ShockGrid state '{state_name}' (with runtime-supplied "
                    f"params) conflicts with a function of the same name in the "
                    f"regime."
                )
            function_params[state_name] = dict.fromkeys(
                grid.params_to_pass_at_runtime, float
            )
