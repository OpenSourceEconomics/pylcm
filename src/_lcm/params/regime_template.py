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
    StateName,
    TransitionFunctionName,
)
from lcm.exceptions import InvalidNameError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import MarkovTransition
from lcm.typing import UserFunction


def create_regime_params_template(
    user_regime: UserRegime,
    *,
    other_regime_state_names: frozenset[StateName] = frozenset(),
) -> RegimeParamsTemplate:
    """Create parameter template from a regime specification.

    Discover parameters from function signatures via `dags.tree`. Parameters
    are function arguments that are not states, actions, regime functions,
    `next_<state>` outputs, or special variables (`period`, `age`, `CE`).

    `next_<state>` is reserved vocabulary for a transition's output, so an
    argument of that shape is never a parameter — not even when this regime
    neither carries the state nor declares a law for it. Such an argument names
    a value belonging to a target regime, which is either produced by a declared
    entry or is a draw with no realized value; the transition invariants decide
    which and say so. Classifying it as a parameter instead would ask the user
    to supply a scalar in place of a random draw.

    Age specialization is already resolved by `normalize_age_specialization`
    before this runs: the regime passed here carries concrete first-active-age
    functions in place of any `AgeSpecializedFunction` marker, so the template is
    read directly off those concrete signatures (the call signature is
    age-invariant by contract, so the first-active resolution is every age's
    template) and this module is unaware of age markers.

    For `Phased` entries, the template contains the **union** of both
    variants' parameters so the user can provide a single flat params dict
    that satisfies both phases.

    Grids with runtime-supplied values (`IrregSpacedGrid` without points,
    `_ContinuousStochasticProcess` without full distribution params) add
    entries to the template under pseudo-function keys matching the state or
    action name.

    Args:
        user_regime: User-form `Regime` instance.
        other_regime_state_names: State names declared by any other regime of the
            model. Their `next_<state>` forms are withheld from the parameter
            namespace so that a law reading one is adjudicated as a transition
            value rather than silently rebound to a parameter.

    Returns:
        The regime parameter template with type annotations as values.

    """
    # Every state the regime declares a law for produces a `next_<state>` in the
    # target's bundle, so a law reading one reads a DAG value, not a parameter.
    # Reading `state_transitions` and not only `states` is what makes that hold
    # for a state the source hands over without carrying — a declared entry into
    # a target's process is the case, and its physical value is available to its
    # neighbours exactly as any other transition's output is.
    variables = {
        *set(user_regime.states),
        *set(user_regime.actions),
        *user_regime.functions,
        *(f"next_{name}" for name in user_regime.states),
        *(f"next_{name}" for name in user_regime.state_transitions),
        "period",
        "age",
        "CE",
    }
    # A target's state is next-period vocabulary only where next-period values
    # exist: inside a transition, and inside whatever feeds one. `utility` and
    # `constraints` are evaluated at this period's states, so `next_<state>`
    # there names an ordinary parameter and stays one.
    transition_role = _function_names_in_transition_role(user_regime)
    variables_in_transition_role = variables | {
        f"next_{name}" for name in other_regime_state_names
    }

    function_params: dict[FunctionName, dict[str, str]] = {}
    per_target_params: dict[RegimeName, dict[FunctionName, dict[str, str]]] = {}

    for name, func in _collect_all_functions_for_template(user_regime).items():
        if isinstance(func, Phased):
            tree_solve = dt.create_tree_with_input_types({name: func.solve})
            tree_sim = dt.create_tree_with_input_types({name: func.simulate})
            tree = dict(tree_solve) | dict(tree_sim)
        else:
            tree = dt.create_tree_with_input_types({name: func})

        # State and action names appearing in a function's signature are
        # exempt from param-template extraction: pylcm wires those values
        # through `states_actions_params` at call time, so they must not
        # surface as user-facing params in the template.
        non_params = (
            variables_in_transition_role
            if tree_path_from_qname(name)[0] in transition_role
            else variables
        )
        params = {k: v for k, v in sorted(tree.items()) if k not in non_params}

        _drop_engine_provided_args(name=name, params=params, user_regime=user_regime)

        # A dotted qname (`<func>__<target>`) marks a per-target function — a
        # transition cell whose parameters must nest under the target regime
        # (`template[target][func]`), so each target's cell keeps its own params.
        # A bare name is a plain regime-level function whose params sit at the
        # top level.
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

    _add_koopmans_aggregator_params(function_params, user_regime)
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


def _add_koopmans_aggregator_params(
    function_params: dict[FunctionName, dict[str, str]],
    user_regime: UserRegime,
) -> None:
    """Add the Koopmans aggregator's params under its pseudo-function name in place.

    The aggregator's parameters beyond `utility` and `CE` surface in the
    template under the reserved key `koopmans_aggregator`; a regime function
    of that name collides and is rejected. Parameters that are states,
    actions, or regime functions are wired at call time and never surface.
    """
    if user_regime.koopmans_aggregator is None:
        return
    if "koopmans_aggregator" in function_params:
        raise InvalidNameError(
            "The regime declares `koopmans_aggregator`, whose parameters live "
            "under the pseudo-function name 'koopmans_aggregator' in the "
            "params — this conflicts with a regime function of the same name."
        )
    aggregator = user_regime.koopmans_aggregator
    if isinstance(aggregator, Phased):
        tree = dict(
            dt.create_tree_with_input_types({"koopmans_aggregator": aggregator.solve})
        ) | dict(
            dt.create_tree_with_input_types(
                {"koopmans_aggregator": aggregator.simulate}
            )
        )
    else:
        tree = dt.create_tree_with_input_types({"koopmans_aggregator": aggregator})
    variables = {
        *set(user_regime.states),
        *set(user_regime.actions),
        *user_regime.functions,
        *(f"next_{name}" for name in user_regime.states),
        "period",
        "age",
        "utility",
        "CE",
    }
    function_params["koopmans_aggregator"] = {
        k: v for k, v in sorted(tree.items()) if k not in variables
    }


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


def _function_names_in_transition_role(user_regime: UserRegime) -> frozenset[str]:
    """Return the names of functions that compute, or feed, a state transition.

    Args:
        user_regime: User-form `Regime` instance.

    Returns:
        Frozenset of the regime's transition function names together with every
        regime function they read, directly or through other regime functions.

    """
    roots: list[UserFunction] = []
    frontier_laws: list[object] = list(user_regime.state_transitions.values())
    while frontier_laws:
        law = frontier_laws.pop()
        if isinstance(law, Phased):
            frontier_laws.extend([law.solve, law.simulate])
        elif isinstance(law, Mapping) and not isinstance(law, MarkovTransition):
            frontier_laws.extend(law.values())
        elif law is not None:
            roots.append(cast("UserFunction", law))

    functions = user_regime.functions
    feeders: set[str] = set()
    frontier = list(roots)
    while frontier:
        func = frontier.pop()
        for arg in dt.create_tree_with_input_types({"_": func}):
            arg_name = tree_path_from_qname(arg)[-1]
            if arg_name in feeders or arg_name not in functions:
                continue
            feeders.add(arg_name)
            frontier.append(cast("UserFunction", functions[arg_name]))

    return frozenset(
        {f"next_{name}" for name in user_regime.state_transitions}
        | {"next_regime"}
        | feeders
    )


def _collect_all_functions_for_template(
    user_regime: UserRegime,
) -> dict[FunctionName | TransitionFunctionName, UserFunction | Phased]:
    """Collect all regime functions, preserving phase-variant entries.

    Unlike `user_regime.get_all_functions(phase=...)` which resolves `Phased`
    entries to a single variant, this returns them as-is so the caller can
    union both variants' parameters. Age markers are already resolved to concrete
    functions upstream, so no marker handling is needed here.
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
    return result


def _drop_engine_provided_args(
    *, name: FunctionName, params: dict[str, str], user_regime: UserRegime
) -> None:
    """Remove a function's engine-supplied arguments from its discovered params.

    In a continuation-based (Euler-inversion) regime the inversion function
    `inverse_marginal_utility` receives `marginal_continuation` from the EGM
    kernel (in a regime whose solver reads no continuation, a function of that
    name is ordinary). This must not surface as a user-facing param, so it is
    popped in place. Gated on the `requires_continuation` capability, not the
    concrete solver type, so every Euler-inversion solver is covered.
    """
    if name == "inverse_marginal_utility" and user_regime.solver.requires_continuation:
        params.pop("marginal_continuation", None)


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
