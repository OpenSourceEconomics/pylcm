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
    function arguments that are not states, actions, other regime functions,
    `next_<state>` transition outputs, or special variables (period, age,
    E_next_V).

    The `next_<state>` exemption lets a state transition consume the output of
    another state transition: dags resolves the chain at evaluation time
    (`get_next_state_function_for_solution` merges all transitions and DAG
    functions into a single dict before calling `concatenate_functions`), so
    these names must not surface as user-facing fixed_params.

    For `SolveSimulateFunctionPair` entries, the template contains the **union**
    of both variants' parameters so the user can provide a single flat params
    dict that satisfies both phases.

    Grids with runtime-supplied values (IrregSpacedGrid without points,
    `_ShockGrid` without full shock_params) add entries to the template under
    pseudo-function keys matching the state or action name.

    Args:
        regime: The regime as provided by the user.

    Returns:
        The regime parameter template with type annotations as values.

    """
    H_variables = {*regime.functions, "period", "age", "E_next_V"}
    next_state_names = {f"next_{name}" for name in regime.states}
    constraint_names = set(regime.constraints)
    variables = (
        H_variables | set(regime.actions) | set(regime.states) | next_state_names
    )

    function_params: dict[FunctionName, dict[str, str]] = {}

    for name, func in _collect_all_functions_for_template(regime).items():
        if isinstance(func, SolveSimulateFunctionPair):
            tree_solve = dt.create_tree_with_input_types({name: func.solve})
            tree_sim = dt.create_tree_with_input_types({name: func.simulate})
            tree = dict(tree_solve) | dict(tree_sim)
        else:
            tree = dt.create_tree_with_input_types({name: func})

        path = tree_path_from_qname(name)

        _fail_if_non_transition_consumes_next_state(
            func_name=name,
            path=path,
            param_names=set(tree),
            next_state_names=next_state_names,
            constraint_names=constraint_names,
        )

        # H is exempt from param-template extraction for state/action names
        # that appear in its signature: pylcm wires those values through
        # `states_actions_params` at call time, so they must not surface as
        # user-facing params in the template.
        params = {k: v for k, v in sorted(tree.items()) if k not in variables}

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
    """Raise if a runtime grid name collides with an existing function name.

    Runtime-supplied state and action grids contribute pseudo-function entries
    to the params template (keyed by the state or action name). Letting such a
    pseudo-function entry shadow a real regime function would silently break
    parameter resolution, so we reject it at template-construction time.

    Args:
        function_params: Template entries collected so far, keyed by
            (pseudo-)function name.
        name: State or action name being added.
        kind: `"state"` or `"action"`, surfaced in the error message.

    Raises:
        InvalidNameError: If `name` already exists in `function_params`.

    """
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


def _fail_if_non_transition_consumes_next_state(
    *,
    func_name: str,
    path: tuple[str, ...],
    param_names: set[str],
    next_state_names: set[str],
    constraint_names: set[str],
) -> None:
    """Reject `next_<state>` parameters on regular DAG functions.

    The `next_<state>` exemption from fixed_param extraction lets state
    transitions consume each other's outputs (dags resolves the chain at
    evaluation time). Constraints also legitimately depend on transition
    outputs (e.g. `borrowing_constraint(next_assets)` — see issue #230).
    For everything else (utility, helpers, custom H), a `next_<state>`
    parameter name is almost always a typo where the user meant the
    current-period `<state>`. Without this guard, the typo'd param would
    be silently filtered from the template and wired to the transition
    output via dags, yielding a wrong result with no error. Catch the
    mistake at template-construction time.
    """
    head = path[0]
    if head == "next_regime" or head in next_state_names or head in constraint_names:
        return
    typos = sorted(param_names & next_state_names)
    if typos:
        raise InvalidNameError(
            f"Function {func_name!r} has parameter(s) {typos} matching "
            f"reserved `next_<state>` transition-output names. Drop the "
            f"'next_' prefix to use the current-period state, or move the "
            f"logic into a state transition (or constraint) if the "
            f"next-period value is genuinely needed."
        )


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
