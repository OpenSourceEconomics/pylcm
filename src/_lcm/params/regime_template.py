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
from lcm.transition import AgeSpecialized
from lcm.typing import UserFunction


def create_regime_params_template(
    user_regime: UserRegime, *, representative_age: float | None = None
) -> RegimeParamsTemplate:
    """Create parameter template from a regime specification.

    Discover parameters from function signatures via `dags.tree`. Parameters
    are function arguments that are not states, actions, regime functions,
    `next_<state>` outputs, or special variables (`period`, `age`, `E_next_V`).

    `AgeSpecialized` nodes carry a `(*args, **kwargs)` wrapper signature, so the
    template is read off a **representative** concrete resolution `build(age)` at
    `representative_age` (the first active age). The `AgeSpecialized` contract makes
    the call signature age-invariant, so the representative's template is every
    age's template. `representative_age` is required whenever the regime contains
    an `AgeSpecialized` node.

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
    per_target_params: dict[RegimeName, dict[FunctionName, dict[str, str]]] = {}

    for name, func in _collect_all_functions_for_template(
        user_regime, representative_age=representative_age
    ).items():
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

        # Per-target entries (`<func>__<target>`) nest under the target — the
        # target is a genuine tree level, mirroring the canonical transition
        # bundles, so param qnames parallel engine function qnames.
        path = tree_path_from_qname(name)
        if len(path) > 1:
            func_name, target_regime_name = path[0], path[1]
            target_branch = per_target_params.setdefault(target_regime_name, {})
            if func_name in target_branch:
                target_branch[func_name] |= params
            else:
                target_branch[func_name] = params
        elif name in function_params:
            function_params[name] |= params
        else:
            function_params[name] = params

    _validate_no_shadowing(
        {**function_params, **{k: {} for k in per_target_params}}, user_regime
    )

    _add_runtime_grid_params(function_params, user_regime)

    if user_regime.taste_shocks is not None:
        if "taste_shocks" in function_params:
            raise InvalidNameError(
                "The regime declares `taste_shocks`, whose scale parameter lives "
                "under the pseudo-function name 'taste_shocks' in the params — "
                "this conflicts with a regime function of the same name."
            )
        function_params["taste_shocks"] = {"scale": "float"}

    _add_certainty_equivalent_params(function_params, user_regime)

    top_level_collisions = set(function_params) & set(per_target_params)
    if top_level_collisions:
        raise InvalidNameError(
            f"Name(s) {sorted(top_level_collisions)} are used both as a "
            f"target regime of a per-target transition and as a function, "
            f"state, or action in the regime. Rename one of the two."
        )

    return MappingProxyType(
        {
            **{k: MappingProxyType(v) for k, v in function_params.items()},
            **{
                target_regime_name: MappingProxyType(
                    {k: MappingProxyType(v) for k, v in target_params.items()}
                )
                for target_regime_name, target_params in per_target_params.items()
            },
        }
    )


def _add_certainty_equivalent_params(
    function_params: dict[FunctionName, dict[str, str]],
    user_regime: UserRegime,
) -> None:
    """Add the certainty equivalent's params under its pseudo-function name in place.

    The transform parameters surface in the template under the reserved
    key `certainty_equivalent`; a regime function of that name collides
    and is rejected.
    """
    if user_regime.certainty_equivalent is None:
        return
    if "certainty_equivalent" in function_params:
        raise InvalidNameError(
            "The regime declares `certainty_equivalent`, whose parameters "
            "live under the pseudo-function name 'certainty_equivalent' in "
            "the params — this conflicts with a regime function of the "
            "same name."
        )
    function_params["certainty_equivalent"] = dict.fromkeys(
        sorted(user_regime.certainty_equivalent.param_names), "float"
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
    user_regime: UserRegime, *, representative_age: float | None = None
) -> dict[FunctionName | TransitionFunctionName, UserFunction | Phased]:
    """Collect all regime functions, preserving phase-variant entries.

    Unlike `user_regime.get_all_functions(phase=...)` which resolves `Phased`
    entries to a single variant, this returns them as-is so the caller can
    union both variants' parameters.
    """
    # The template reads the finalized regime, where `None` masks are
    # already resolved; the filters narrow the type.
    result: dict[FunctionName | TransitionFunctionName, UserFunction | Phased] = {
        name: func for name, func in user_regime.functions.items() if func is not None
    }
    result |= {
        name: func for name, func in user_regime.constraints.items() if func is not None
    }
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
    return _resolve_age_specialized(result, representative_age)


def _resolve_age_specialized(
    collected: dict[FunctionName | TransitionFunctionName, UserFunction | Phased],
    representative_age: float | None,
) -> dict[FunctionName | TransitionFunctionName, UserFunction | Phased]:
    """Replace every `AgeSpecialized` leaf with its representative resolution.

    Descends into both sides of a `Phased` entry, so phase-split specialized
    functions also surface their concrete parameters. The template only needs
    each node's (age-invariant) call signature, so a single
    `build(representative_age)` per node suffices. Raise if the regime carries
    an `AgeSpecialized` node but no representative age was supplied.
    """

    def _has_marker(value: object) -> bool:
        if isinstance(value, Phased):
            return _has_marker(value.solve) or _has_marker(value.simulate)
        return isinstance(value, AgeSpecialized)

    if not any(_has_marker(func) for func in collected.values()):
        return collected
    if representative_age is None:
        raise ValueError(
            "The regime contains an `AgeSpecialized` node, so `representative_age` "
            "is required to read its concrete function's parameters."
        )

    def _resolve(value: UserFunction | Phased) -> UserFunction | Phased:
        if isinstance(value, Phased):
            return Phased(
                solve=_resolve(value.solve),  # ty: ignore[invalid-argument-type]
                simulate=_resolve(value.simulate),  # ty: ignore[invalid-argument-type]
            )
        if isinstance(value, AgeSpecialized):
            return value.build(representative_age)
        return value

    return {name: _resolve(func) for name, func in collected.items()}


def _regime_transition_entries(
    transition: object,
) -> dict[TransitionFunctionName, UserFunction | Phased]:
    """Key the regime transition for parameter discovery.

    - coarse forms ⇒ one `next_regime` entry
    - a per-target dict ⇒ one `next_regime__<target>` entry per cell, so each
      cell's parameters nest under the target (`template[target_regime]["next_regime"]`)
    - `Phased` per-target dicts (identical key sets) ⇒ per-cell `Phased`
      entries, so both phases' parameters are unioned per target

    """
    if isinstance(transition, Phased) and isinstance(transition.solve, Mapping):
        solve_cells = cast("Mapping[RegimeName, UserFunction]", transition.solve)
        simulate_cells = cast("Mapping[RegimeName, UserFunction]", transition.simulate)
        return {
            f"next_regime__{target_regime_name}": Phased(
                solve=solve_cells[target_regime_name],
                simulate=simulate_cells[target_regime_name],
            )
            for target_regime_name in solve_cells
        }
    if isinstance(transition, Mapping):
        cells = cast("Mapping[RegimeName, UserFunction]", transition)
        return {
            f"next_regime__{target_regime_name}": cell
            for target_regime_name, cell in cells.items()
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
