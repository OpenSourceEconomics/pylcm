from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import dags.tree as dt
from dags.tree import tree_path_from_qname

from _lcm.grids import IrregSpacedGrid
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.typing import (
    FunctionName,
    RegimeName,
    RegimeParamsTemplate,
    TransitionFunctionName,
)
from lcm.exceptions import InvalidNameError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.typing import UserFunction


def create_regime_params_template(user_regime: UserRegime) -> RegimeParamsTemplate:
    """Create parameter template from a regime specification.

    Discover parameters from function signatures via `dags.tree`. Parameters
    are function arguments that are not states, actions, regime functions,
    `next_<state>` outputs, or special variables (`period`, `age`, `E_next_V`).

    For `Phased` entries, the template contains the **union** of both
    variants' parameters so the user can provide a single flat params dict
    that satisfies both phases.

    Grids with runtime-supplied values (`IrregSpacedGrid` without points,
    `_ContinuousStochasticProcess` without full distribution params) add
    entries to the template under pseudo-function keys matching the state or
    action name.

    Args:
        user_regime: User-form `Regime` instance.

    Returns:
        The regime parameter template with type annotations as values.

    """
    variables = {
        *set(user_regime.states),
        *set(user_regime.actions),
        *user_regime.functions,
        *(f"next_{name}" for name in user_regime.states),
        "period",
        "age",
        "E_next_V",
    }

    function_params: dict[FunctionName, dict[str, str]] = {}

    for name, func in _collect_all_functions_for_template(user_regime).items():
        if isinstance(func, Phased):
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

    _validate_no_shadowing(function_params, user_regime)

    _add_runtime_grid_params(function_params, user_regime)

    return MappingProxyType(
        {k: MappingProxyType(v) for k, v in function_params.items()}
    )


def _add_runtime_grid_params(
    function_params: dict[FunctionName, dict[str, str]],
    user_regime: UserRegime,
) -> None:
    """Add runtime-supplied state/action grid params to the template in place."""
    for state_name, grid in user_regime.states.items():
        if isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime:
            _fail_if_runtime_grid_shadows_function(
                function_params=function_params, name=state_name, kind="state"
            )
            function_params[state_name] = {"points": "Float1D"}
        elif (
            isinstance(grid, _ContinuousStochasticProcess)
            and grid.params_to_pass_at_runtime
        ):
            _fail_if_runtime_grid_shadows_function(
                function_params=function_params,
                name=state_name,
                kind="_ContinuousStochasticProcess state",
            )
            function_params[state_name] = dict.fromkeys(
                grid.params_to_pass_at_runtime, "float"
            )

    for action_name, grid in user_regime.actions.items():
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
    user_regime: UserRegime,
) -> dict[FunctionName | TransitionFunctionName, UserFunction | Phased]:
    """Collect all regime functions, preserving phase-variant entries.

    Unlike `user_regime.get_all_functions(phase=...)` which resolves `Phased`
    entries to a single variant, this returns them as-is so the caller can
    union both variants' parameters.
    """
    result: dict[FunctionName | TransitionFunctionName, UserFunction | Phased] = dict(
        user_regime.functions
    )
    result |= dict(user_regime.constraints)
    # A carried state contributes its `solve` variant as a derived function
    # under the state's name (solve-phase imputation), so its parameters
    # surface in the template. Its law of motion is its regular
    # `state_transitions` entry (keyed `next_<name>`), collected below.
    for name, spec in user_regime.states.items():
        if isinstance(spec, Phased):
            result[name] = cast("UserFunction", spec.solve)
    if user_regime.transition is not None:
        result |= collect_state_transitions(
            user_regime.states, user_regime.state_transitions
        )
        result |= _regime_transition_entries(user_regime.transition)
    return result


def _regime_transition_entries(
    transition: object,
) -> dict[TransitionFunctionName, UserFunction | Phased]:
    """Key the regime transition for parameter discovery.

    - coarse forms ⇒ one `next_regime` entry
    - a per-target dict ⇒ one `next_regime__<target>` entry per cell, so each
      cell's parameters surface under `to_<target>_next_regime`
    - `Phased` per-target dicts (identical key sets) ⇒ per-cell `Phased`
      entries, so both phases' parameters are unioned per target

    """
    if isinstance(transition, Phased) and isinstance(transition.solve, Mapping):
        solve_cells = cast("Mapping[RegimeName, UserFunction]", transition.solve)
        simulate_cells = cast("Mapping[RegimeName, UserFunction]", transition.simulate)
        return {
            f"next_regime__{target_name}": Phased(
                solve=solve_cells[target_name],
                simulate=simulate_cells[target_name],
            )
            for target_name in solve_cells
        }
    if isinstance(transition, Mapping):
        cells = cast("Mapping[RegimeName, UserFunction]", transition)
        return {
            f"next_regime__{target_name}": cell for target_name, cell in cells.items()
        }
    return {"next_regime": cast("UserFunction | Phased", transition)}


def _validate_no_shadowing(
    function_params: dict[FunctionName, dict[str, str]],
    user_regime: UserRegime,
) -> None:
    """Raise if any discovered parameter shadows a state or action name."""
    state_action_names = set(user_regime.states) | set(user_regime.actions)
    for func_name, params in function_params.items():
        shadows = set(params) & state_action_names
        if shadows:
            raise InvalidNameError(
                f"Function '{func_name}' has parameter(s) {sorted(shadows)} that "
                f"shadow state/action variable(s) with the same name. Please rename "
                f"the parameter(s) or the state(s)/action(s) to avoid ambiguity."
            )
