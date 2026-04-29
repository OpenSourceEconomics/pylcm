from types import MappingProxyType

import dags.tree as dt
from dags.tree import tree_path_from_qname

from lcm.exceptions import InvalidNameError
from lcm.grids import IrregSpacedGrid
from lcm.interfaces import SolveSimulateFunctionPair
from lcm.regime import Regime
from lcm.regime_building.validation import collect_state_transitions
from lcm.shocks import _ShockGrid
from lcm.typing import (
    FunctionName,
    RegimeParamsTemplate,
    TransitionFunctionName,
    UserFunction,
)


def create_regime_params_template(
    regime: Regime,
) -> RegimeParamsTemplate:
    """Create parameter template from a regime specification.

    Discover parameters from function signatures via `dags.tree`. Parameters are
    function arguments that are not states, actions, other regime functions, or
    special variables (period, age, E_next_V).

    For `SolveSimulateFunctionPair` entries, the template contains the **union**
    of both variants' parameters so the user can provide a single flat params
    dict that satisfies both phases.

    Grids with runtime-supplied values (IrregSpacedGrid without points,
    `_ShockGrid` without full shock_params) add entries to the template under
    pseudo-function keys matching the state name.

    Args:
        regime: The regime as provided by the user.

    Returns:
        The regime parameter template with type annotations as values.

    """
    H_variables = {*regime.functions, "period", "age", "E_next_V"}
    variables = H_variables | set(regime.actions) | set(regime.states)

    function_params: dict[FunctionName, dict[str, str]] = {}

    for name, func in _collect_all_functions_for_template(regime).items():
        if isinstance(func, SolveSimulateFunctionPair):
            tree_solve = dt.create_tree_with_input_types({name: func.solve})
            tree_sim = dt.create_tree_with_input_types({name: func.simulate})
            tree = dict(tree_solve) | dict(tree_sim)
        else:
            tree = dt.create_tree_with_input_types({name: func})

        # H is exempt from param-template extraction for state/action names
        # that appear in its signature: pylcm wires those values through
        # `states_actions_params` at call time, so they must not surface as
        # user-facing params in the template.
        params = {k: v for k, v in sorted(tree.items()) if k not in variables}

        path = tree_path_from_qname(name)
        template_key = f"to_{path[1]}_{path[0]}" if len(path) > 1 else name

        if template_key in function_params:
            function_params[template_key] |= params
        else:
            function_params[template_key] = params

    _validate_no_shadowing(function_params, regime)

    _add_runtime_grid_params(function_params, regime)

    return MappingProxyType(
        {k: MappingProxyType(v) for k, v in function_params.items()}
    )


def _add_runtime_grid_params(
    function_params: dict[FunctionName, dict[str, str]],
    regime: Regime,
) -> None:
    """Add runtime-supplied state/action grid params to the template in place."""
    for state_name, grid in regime.states.items():
        if isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime:
            _fail_if_runtime_grid_shadows_function(
                function_params=function_params, name=state_name, kind="state"
            )
            function_params[state_name] = {"points": "Float1D"}
        elif isinstance(grid, _ShockGrid) and grid.params_to_pass_at_runtime:
            _fail_if_runtime_grid_shadows_function(
                function_params=function_params,
                name=state_name,
                kind="_ShockGrid state",
            )
            function_params[state_name] = dict.fromkeys(
                grid.params_to_pass_at_runtime, "float"
            )

    for action_name, grid in regime.actions.items():
        if isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime:
            _fail_if_runtime_grid_shadows_function(
                function_params=function_params, name=action_name, kind="action"
            )
            function_params[action_name] = {"points": "Float1D"}


def _fail_if_runtime_grid_shadows_function(
    *,
    function_params: dict[FunctionName, dict[str, str]],
    name: str,
    kind: str,
) -> None:
    if name in function_params:
        raise InvalidNameError(
            f"IrregSpacedGrid {kind} '{name}' (with runtime-supplied "
            f"points/params) conflicts with a function of the same name in the regime."
        )


def _collect_all_functions_for_template(
    regime: Regime,
) -> dict[
    FunctionName | TransitionFunctionName, UserFunction | SolveSimulateFunctionPair
]:
    """Collect all regime functions, preserving `SolveSimulateFunctionPair` entries.

    Unlike `regime.get_all_functions(phase=...)` which resolves pairs to a single
    variant, this returns pairs as-is so the caller can union both variants'
    parameters.
    """
    result: dict[
        FunctionName | TransitionFunctionName,
        UserFunction | SolveSimulateFunctionPair,
    ] = dict(regime.functions)
    result |= dict(regime.constraints)
    if callable(regime.transition):
        result |= collect_state_transitions(regime.states, regime.state_transitions)
        result["next_regime"] = regime.transition
    return result


def _validate_no_shadowing(
    function_params: dict[FunctionName, dict[str, str]],
    regime: Regime,
) -> None:
    """Raise if any discovered parameter shadows a state or action name."""
    state_action_names = set(regime.states) | set(regime.actions)
    for func_name, params in function_params.items():
        shadows = set(params) & state_action_names
        if shadows:
            raise InvalidNameError(
                f"Function '{func_name}' has parameter(s) {sorted(shadows)} that "
                f"shadow state/action variable(s) with the same name. Please rename "
                f"the parameter(s) or the state(s)/action(s) to avoid ambiguity."
            )
