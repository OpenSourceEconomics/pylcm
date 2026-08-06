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
    variables = {
        *set(user_regime.states),
        *set(user_regime.actions),
        *user_regime.functions,
        "period",
        "age",
        "CE",
    }
    # `next_<state>` names a value, never a parameter. Whether a consumer may
    # read it is a question of whether that value exists where the consumer runs.
    #
    # `next_<state>` names a value a transition produces, and only a transition —
    # or a function feeding one — is evaluated where that value exists. Three
    # families of name are transition outputs there:
    #
    # - `next_<state>` for a state this regime carries;
    # - `next_<state>` for a state it declares a law for — which covers a state
    #   handed over without being carried, a declared entry into a target's
    #   process being the case;
    # - `next_<state>` for a state some other regime declares, since the value
    #   belongs to a target and is produced by the entry or is a draw.
    #
    # Anywhere else the name has no value behind it, and admitting it as a
    # parameter would answer a next-period question with a constant the user
    # supplies. That is rejected rather than classified.
    transition_role = _function_names_in_transition_role(user_regime)
    variables_in_transition_role = variables | {
        f"next_{name}"
        for name in (
            *user_regime.states,
            *user_regime.state_transitions,
            *other_regime_state_names,
        )
    }
    # A within-period consumer may read a next state the chosen action
    # determines -- the NEGM/DC-EGM durable pattern, where the service flow
    # accrues from the newly chosen stock and the budget constraint bounds it.
    # The decision evaluation resolves such a name from the regime's own law,
    # so it is that law's output rather than a parameter.
    #
    # Only a law that is both deterministic and target-independent qualifies:
    #
    # - a `MarkovTransition` names a draw not yet realised when the decision is
    #   evaluated, so it produces nothing there;
    # - a per-target law is a handover value that is not well defined until the
    #   destination is known, which is the same ground on which the decision
    #   evaluation rejects reading a `next_<state>` whose law differs across
    #   targets.
    #
    # A `next_<state>` this regime does not move at all -- a target's process,
    # say -- is a parameter here, whatever it is spelled.
    variables_from_own_within_period_laws = variables | {
        f"next_{state_name}"
        for state_name, law in user_regime.state_transitions.items()
        if _is_within_period_law(law)
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
            else variables_from_own_within_period_laws
        )
        params = {k: v for k, v in sorted(tree.items()) if k not in non_params}
        _fail_if_a_param_is_next_prefixed(consumer_name=name, params=params)

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


def _fail_if_a_param_is_next_prefixed(
    *, consumer_name: FunctionName, params: Mapping[str, str]
) -> None:
    """Check that no argument left over as a parameter claims the reserved prefix.

    `next_<name>` names a value, never a parameter. Every argument of that shape
    denoting a value this consumer can see has already been removed by the time
    this runs — a state the regime carries or declares a law for, and, for a
    transition, a state belonging to a target. What remains would be handed to the
    user as a parameter under a name that says it is a next-period value, in a
    place where that value does not exist.

    Args:
        consumer_name: Function whose parameters are being checked, named in the
            message. `koopmans_aggregator` and `certainty_equivalent` enter under
            their pseudo-function names.
        params: Mapping of the consumer's parameter names to their annotations.

    Raises:
        InvalidNameError: If any parameter name starts with `next_`.

    """
    reserved = sorted(name for name in params if name.startswith("next_"))
    if not reserved:
        return
    raise InvalidNameError(
        f"'{consumer_name}' takes {reserved} as arguments, but the 'next_' prefix "
        f"names the output of a state transition and is never a parameter. "
        f"'{consumer_name}' is evaluated before this period's transitions run, so "
        f"there is no next-period value for {reserved} to mean here — accepting "
        f"them would answer a next-period question with a constant supplied at "
        f"solve time. Restate the quantity in this period's states and actions "
        f"(a constraint on next-period assets is a constraint on assets minus "
        f"consumption), read it inside a transition law, or rename the argument "
        f"if a parameter was meant."
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
        "period",
        "age",
        "utility",
        "CE",
    }
    params = {k: v for k, v in sorted(tree.items()) if k not in variables}
    _fail_if_a_param_is_next_prefixed(
        consumer_name="koopmans_aggregator", params=params
    )
    function_params["koopmans_aggregator"] = params


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


def _callables_in(value: object) -> list[UserFunction]:
    """Return the callables a regime slot value stands for.

    A law and a regime function accept the same shapes, so one traversal serves
    both and they cannot come to disagree about what is walkable:

    - a plain callable stands for itself;
    - a `Phased` entry is two implementations rather than one, so anything
      reading a signature descends into both;
    - a per-target dict holds one law per target, each walked in turn;
    - `None` masks a model-level entry and stands for no implementation at all,
      so there is no signature to read.

    Args:
        value: A `state_transitions` or `functions` entry.

    Returns:
        List of the callables it stands for, empty when it stands for none.

    """
    if value is None:
        return []
    if isinstance(value, Phased):
        return [*_callables_in(value.solve), *_callables_in(value.simulate)]
    if isinstance(value, Mapping) and not isinstance(value, MarkovTransition):
        return [
            callable_
            for member in value.values()
            for callable_ in _callables_in(member)
        ]
    return [cast("UserFunction", value)]


def _is_within_period_law(law: object) -> bool:
    """Return whether a state's law produces a value the decision can read.

    Two things disqualify a law. A `MarkovTransition` names a draw that has not
    been realised when the within-period decision is evaluated. A per-target
    dict names a handover value that is not well defined until the destination
    is known -- one law per reachable target, and nothing says they agree.

    What is left is a bare, target-independent, deterministic law: the chosen
    next durable stock, and anything else this period's states and actions
    determine on their own.
    """
    if isinstance(law, MarkovTransition | Mapping):
        return False
    if isinstance(law, Phased):
        return _is_within_period_law(law.solve)
    return True


def _function_names_in_transition_role(user_regime: UserRegime) -> frozenset[str]:
    """Return the names of functions that compute, or feed, a state transition.

    Args:
        user_regime: User-form `Regime` instance.

    Returns:
        Frozenset of the regime's transition function names together with every
        regime function they read, directly or through other regime functions.

    """
    functions = user_regime.functions
    feeders: set[str] = set()
    frontier = [
        variant
        for law in user_regime.state_transitions.values()
        for variant in _callables_in(law)
    ]
    while frontier:
        func = frontier.pop()
        for arg in dt.create_tree_with_input_types({"_": func}):
            arg_name = tree_path_from_qname(arg)[-1]
            if arg_name in feeders or arg_name not in functions:
                continue
            feeders.add(arg_name)
            frontier.extend(_callables_in(functions[arg_name]))

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
