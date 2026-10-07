"""Derive one regime's params template from what its callables actually read.

`create_regime_params_template` inspects the finalized user regime — its
functions, constraints, state transitions, Koopmans aggregator, certainty
equivalent and Pareto objective — and returns the nested template of parameters
a caller must supply. It fails closed on a name that shadows a function and on a
transition-local node read from outside its transition.

`create_edge_params_template` reads the callables a source's `Model(edges=...)`
declaration holds — its regime-transition law, gates, gate references and route
fallbacks — and returns that source's branch of `params["edges"]`.
"""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Literal, cast

import dags.tree as dt
from dags.tree import qname_from_tree_path, tree_path_from_qname

from _lcm.grids import IrregSpacedGrid
from _lcm.params.edges import (
    EDGES,
    FALLBACK,
    GATE,
    GATE_REFERENCES,
    PROBABILITY,
    ROUTES,
    user_path,
)
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.collective import PARETO_OBJECTIVE_ENTRY
from _lcm.regime_building.gated_edges import is_target_value_operand
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.regime_law import RegimeLaw, RegimeLawDeclaration
from _lcm.typing import (
    EdgeParamsTemplate,
    FunctionName,
    RegimeName,
    RegimeParamsTemplate,
    StateName,
    TransitionFunctionName,
)
from _lcm.utils.error_messages import path_segment_name_errors
from _lcm.utils.functools import get_union_of_args
from lcm.collective import ValueDependentTransition
from lcm.exceptions import InvalidNameError, ModelInitializationError
from lcm.phased import Phased
from lcm.regime import ProjectedRegimeValue
from lcm.regime import Regime as UserRegime
from lcm.transition import (
    ByAge,
    DeterministicTransition,
    JointTransition,
    StochasticTransition,
)
from lcm.typing import UserFunction


def create_regime_params_template(
    *,
    user_regime: UserRegime,
    law: RegimeLaw,
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

    The regime-transition law and a gated edge's gate, gate references and
    route fallbacks are read here only for the checks on what they may read.
    Their parameters belong to the source's branch of `params["edges"]`, which
    `create_edge_params_template` builds from the declared law.

    Args:
        user_regime: User-form `Regime` instance.
        law: The regime's law, bound from `Model(edges=...)`.
        other_regime_state_names: State names declared by any other regime of the
            model. Their `next_<state>` forms are withheld from the parameter
            namespace so that a law reading one is adjudicated as a transition
            value rather than silently rebound to a parameter.

    Returns:
        The regime parameter template with type annotations as values.

    """
    variables = _wired_names(user_regime)

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
    edge_entries = _gated_edge_entries(law)
    template_functions = _collect_all_functions_for_template(
        user_regime, law=law, edge_entries=edge_entries
    )
    _fail_if_a_next_name_is_read_outside_a_transition(
        user_regime, template_functions=template_functions
    )
    _fail_if_a_joint_node_is_read_outside_its_transition(
        user_regime, template_functions=template_functions
    )

    # Every illegitimate read is already rejected, so the subtraction below only
    # has to be permissive enough for the legitimate ones: a name in transition
    # role in either phase keeps its `next_` arguments out of the template.
    transition_role = _function_names_in_transition_role(
        user_regime=user_regime, phase="solve"
    ) | _function_names_in_transition_role(user_regime=user_regime, phase="simulate")
    variables_in_transition_role = (
        variables
        | _joint_transition_node_names(user_regime)
        | {
            f"next_{name}"
            for name in (
                *user_regime.states,
                *user_regime.state_transitions,
                *other_regime_state_names,
            )
        }
    )
    function_params: dict[FunctionName, dict[str, str]] = {}
    per_target_params: dict[RegimeName, dict[str, Any]] = {}

    # The law and the gated-edge callables join the checks above, but their
    # parameters belong to the edge namespace (`create_edge_params_template`).
    edge_entry_names = set(edge_entries) | (
        set()
        if law.terminal
        else set(_regime_transition_entries(law.decomposed_transition))
    )

    for name, func in template_functions.items():
        if name in edge_entry_names:
            continue
        # State and action names appearing in a function's signature are
        # exempt from param-template extraction: pylcm wires those values
        # through `states_actions_params` at call time, so they must not
        # surface as user-facing params in the template.
        non_params = (
            variables_in_transition_role
            if tree_path_from_qname(name)[0] in transition_role
            else variables
        )
        params = _discovered_params(
            name=name,
            func=func,
            non_params=non_params,
            strip_target_value_operands=False,
        )

        _drop_engine_provided_args(name=name, params=params, user_regime=user_regime)

        _record_params(
            name=name,
            params=params,
            function_params=function_params,
            per_target_params=per_target_params,
        )

    _add_joint_transition_params(
        per_target_params=per_target_params,
        user_regime=user_regime,
        variables=variables,
        other_regime_state_names=other_regime_state_names,
    )

    _validate_no_shadowing(
        function_params={**function_params, **{k: {} for k in per_target_params}},
        user_regime=user_regime,
    )

    _add_runtime_grid_params(function_params=function_params, user_regime=user_regime)

    if user_regime.taste_shocks is not None:
        if "taste_shocks" in function_params:
            raise InvalidNameError(
                "The regime declares `taste_shocks`, whose scale parameter lives "
                "under the pseudo-function name 'taste_shocks' in the params — "
                "this conflicts with a regime function of the same name."
            )
        function_params["taste_shocks"] = {"scale": "float"}

    _add_koopmans_aggregator_params(
        function_params=function_params, user_regime=user_regime
    )
    _add_certainty_equivalent_params(
        function_params=function_params, user_regime=user_regime
    )
    _add_pareto_objective_params(
        function_params=function_params, user_regime=user_regime
    )

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
                    {k: _freeze_template_node(v) for k, v in target_params.items()}
                )
                for target_regime_name, target_params in per_target_params.items()
            },
        }
    )


def create_edge_params_template(
    *,
    source: RegimeName,
    user_regime: UserRegime,
    declared_laws: tuple[RegimeLawDeclaration, ...],
    state_names_by_regime: Mapping[RegimeName, frozenset[StateName]],
) -> EdgeParamsTemplate:
    """Create a source regime's branch of the `edges` parameter template.

    Every parameter sits at its declaration path below `params["edges"][source]`:

    - a law over all targets ⇒ `[<param>]`;
    - a per-target cell ⇒ `[<target>][<param>]`;
    - a value-dependent cell's probability ⇒ `[<target>]["probability"][<param>]`;
    - its gate ⇒ `[<target>]["gate"][<param>]`;
    - a gate reference's projection of one reference-regime state ⇒
      `[<target>]["gate_references"][<reference>][<state>][<param>]`;
    - a route fallback's projection ⇒
      `[<target>]["routes"][<route>]["fallback"][<state>][<param>]`, with a
      `"solve"` / `"simulate"` level after `"fallback"` when it is `Phased`.

    The template is read off the declared laws, every `ByAge` case and both
    phases included, so its slots do not depend on the horizon or on which cells
    a fixed zero prunes. A law reads the source regime's variables; a gate and a
    projection run on the target regime's grid and also read its states, its
    `V_target` value components, `D_target` and the cell's gate-reference keys.
    None of those is a parameter.

    Args:
        source: The source regime's name.
        user_regime: The source regime, whose engine-wired names the law reads.
        declared_laws: The `Transition` laws declared for the source, one per
            phase of `Model(edges=...)`.
        state_names_by_regime: State names declared by each regime of the model.

    Returns:
        The nested template, empty for a source whose edges declare no parameter.

    Raises:
        InvalidNameError: If one path is both a parameter and a branch, e.g. a
            law over all targets reading an argument named like a target whose
            cell has parameters.
        ModelInitializationError: If the source's laws read parameters both over
            all targets and per target.

    """
    variables = _wired_names(user_regime)
    params_by_path: list[tuple[tuple[str, ...], dict[str, str]]] = []
    for law in declared_laws:
        for path, func, cell in iter_edge_callables(law=law, path=()):
            non_params = (
                variables
                if cell is None
                else variables
                | _gated_edge_wired_names(
                    gate_reference_names=frozenset(cell.gate_references),
                    target_state_names=state_names_by_regime.get(path[0], frozenset()),
                )
            )
            params_by_path.append(
                (
                    path,
                    _discovered_params(
                        name="edge",
                        func=func,
                        non_params=non_params,
                        strip_target_value_operands=cell is not None,
                        kind=(
                            "Edge-callable (at "
                            f"{user_path(path=(EDGES, source, *path))}) argument"
                        ),
                    ),
                )
            )
    _fail_if_coarse_law_params_meet_per_target_cells(
        source=source, params_by_path=params_by_path
    )
    template: dict[str, Any] = {}
    for path, params in params_by_path:
        for param_name, annotation in params.items():
            _insert_edge_slot(
                template=template, path=(*path, param_name), annotation=annotation
            )
    return cast("EdgeParamsTemplate", _freeze_template_node(template))


def _fail_if_coarse_law_params_meet_per_target_cells(
    *,
    source: RegimeName,
    params_by_path: list[tuple[tuple[str, ...], dict[str, str]]],
) -> None:
    """Reject a source whose laws read parameters both over all targets and per target.

    A law over all targets files its parameters at `params["edges"][source][arg]`,
    which is also the level that broadcasts into every per-target slot below
    `params["edges"][source]`. A source mixing the two forms — a `ByAge` with a
    `DeterministicTransition` case and a per-target case, say — would make one
    path both a slot and a broadcast level.
    """
    coarse = sorted(
        {name for path, params in params_by_path if not path for name in params}
    )
    if coarse and any(path for path, _ in params_by_path):
        raise ModelInitializationError(
            f"The law of '{source}' mixes a law over all targets reading "
            f"parameters {coarse} with per-target cases. Its parameters would sit "
            f"both at params['edges']['{source}'][<arg>] and below each target. "
            "Write every case in one form: the law over all targets as a "
            "per-target mapping (`{target: StochasticTransition(...)}`, a "
            "deterministic choice as indicator probabilities), or every case as "
            "a law over all targets."
        )


def iter_edge_callables(
    *, law: object, path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], UserFunction, ValueDependentTransition | None]]:
    """Yield every callable a declared law holds, at its declaration path.

    Args:
        law: A declared law, or one of its cases, phases or cells.
        path: The declaration path of `law` below the source's edge branch.

    Yields:
        Triples of the path, the callable, and the value-dependent cell when the
        callable runs on that cell's target grid (a gate or a projection).

    Raises:
        TypeError: If the law, a case, a phase or a cell is none of the law
            forms.

    """
    if law is None or isinstance(law, str):
        return
    if isinstance(law, ByAge):
        for case in law.laws:
            yield from iter_edge_callables(law=case, path=path)
    elif isinstance(law, Phased):
        yield from iter_edge_callables(law=law.solve, path=path)
        yield from iter_edge_callables(law=law.simulate, path=path)
    elif isinstance(law, ValueDependentTransition):
        yield from _value_dependent_callables(cell=law, path=path)
    elif isinstance(law, DeterministicTransition | StochasticTransition):
        yield path, cast("UserFunction", law.func), None
    elif isinstance(law, Mapping):
        for target_regime_name, cell in law.items():
            yield from iter_edge_callables(law=cell, path=(*path, target_regime_name))
    elif callable(law):
        yield path, cast("UserFunction", law), None
    else:
        msg = (
            f"A declared law holds a {type(law).__name__!r} at "
            f"{list(path)} below its source's edges, which is not a law form."
        )
        raise TypeError(msg)


def _value_dependent_callables(
    *, cell: ValueDependentTransition, path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], UserFunction, ValueDependentTransition | None]]:
    """Yield a value-dependent cell's callables at their declaration paths."""
    probability = cell.probability
    yield (
        (*path, PROBABILITY),
        cast(
            "UserFunction",
            probability.func
            if isinstance(probability, StochasticTransition)
            else probability,
        ),
        None,
    )
    yield (*path, GATE), cell.gate, cell
    for ref_name, ref in cell.gate_references.items():
        for state_name, projection in ref.projection.items():
            yield (*path, GATE_REFERENCES, ref_name, state_name), projection, cell
    for route_name, route in cell.routes.items():
        for phase, ref in _fallbacks_by_phase(
            solve=route.solve_fallback,
            simulate=route.simulate_fallback,
            is_phased=route.fallback_is_phased,
        ):
            prefix = (
                *path,
                ROUTES,
                route_name,
                FALLBACK,
                *(() if phase is None else (phase,)),
            )
            for state_name, projection in ref.projection.items():
                yield (*prefix, state_name), projection, cell


def _insert_edge_slot(
    *, template: dict[str, Any], path: tuple[str, ...], annotation: str
) -> None:
    """File one parameter at its path, in place, refusing a leaf/branch clash."""
    branch = template
    for depth, segment in enumerate(path[:-1]):
        node = branch.setdefault(segment, {})
        if not isinstance(node, dict):
            raise InvalidNameError(_edge_slot_clash(path=path[: depth + 1]))
        branch = node
    if isinstance(branch.get(path[-1]), dict):
        raise InvalidNameError(_edge_slot_clash(path=path))
    branch[path[-1]] = annotation


# The entries a value-dependent cell's parameters nest under.
_CELL_ENTRIES = (PROBABILITY, GATE, GATE_REFERENCES, ROUTES)


def _edge_slot_clash(*, path: tuple[str, ...]) -> str:
    """Name an edge path that is both a parameter and a branch."""
    spelled = "".join(f"[{segment!r}]" for segment in path)
    return (
        f"The edge parameter path {spelled} below a source's `params['edges']` "
        "branch is both a parameter and a branch of further parameters: a "
        "law's argument has the name of a target or of a cell entry "
        f"({', '.join(repr(entry) for entry in _CELL_ENTRIES)}). Rename the argument."
    )


def _wired_names(user_regime: UserRegime) -> set[str]:
    """Return the names a regime's own callables read that the engine binds.

    Args:
        user_regime: User-form `Regime` instance.

    Returns:
        Set of the regime's states, actions and functions, `period`, `age` and
        `CE`, plus a collective regime's engine-wired aggregator inputs.

    """
    variables = {
        *set(user_regime.states),
        *set(user_regime.actions),
        *user_regime.decomposed_functions,
        "period",
        "age",
        "CE",
    }
    if user_regime.stakeholders is not None:
        # A collective regime carries per-stakeholder
        # `utility_<s>` functions instead of a singleton `utility`, but the
        # Bellman aggregator H still takes a `utility` argument — engine-wired
        # (the stacked per-stakeholder utilities), exactly like `E_next_V` —
        # so the name must not surface as a user-facing param.
        variables.add("utility")
        # Value-constraint predicates read the
        # engine-computed per-stakeholder action values `Q_<s>` and the
        # interpolated same-period reference values (keyed by the
        # `same_period_refs` names) as named arguments — engine-wired, never
        # user-facing params.
        variables.update(f"Q_{s}" for s in user_regime.stakeholders)
        variables.update(user_regime.same_period_refs)
    return variables


def _record_params(
    *,
    name: FunctionName | TransitionFunctionName,
    params: dict[str, str],
    function_params: dict[FunctionName, dict[str, str]],
    per_target_params: dict[RegimeName, dict[str, Any]],
) -> None:
    """File one entry's parameters under the branch its key names, in place.

    A dotted qname (`<func>__<target>`) marks a per-target entry — a state
    transition cell — whose parameters nest under the target
    regime (`template[target][func]`), so each target keeps its own. A bare name
    is a plain regime-level function whose parameters sit at the top level.
    Either way an entry met twice unions with what is already filed.

    Args:
        name: Template key the entry is collected under.
        params: The entry's discovered parameters.
        function_params: Top-level entries collected so far.
        per_target_params: Per-target-regime entries collected so far.

    """
    path = tree_path_from_qname(name)
    if len(path) > 1:
        func_name, target_regime_name = path[0], path[1]
        target_branch = per_target_params.setdefault(target_regime_name, {})
        target_branch[func_name] = target_branch.get(func_name, {}) | params
    else:
        function_params[name] = function_params.get(name, {}) | params


def _freeze_template_node(value: Any) -> Any:  # noqa: ANN401
    """Recursively freeze a parameter-template branch."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {name: _freeze_template_node(member) for name, member in value.items()}
        )
    return value


def _add_joint_transition_params(
    *,
    per_target_params: dict[RegimeName, dict[str, Any]],
    user_regime: UserRegime,
    variables: set[str],
    other_regime_state_names: frozenset[StateName],
) -> None:
    """File joint support, probability, and output params under their owners."""
    joint_transitions = getattr(user_regime, "joint_transitions", {})
    joint_node_names = set(_joint_transition_node_names(user_regime))
    next_state_names = {
        f"next_{name}"
        for name in (
            *user_regime.states,
            *user_regime.state_transitions,
            *other_regime_state_names,
            *(
                output
                for kernels in joint_transitions.values()
                for raw in kernels.values()
                for kernel in _joint_variants(raw)
                for output in kernel.outputs
            ),
        )
    }
    output_non_params = variables | joint_node_names | next_state_names
    probability_non_params = variables | joint_node_names | next_state_names
    support_non_params = {"period", "age", *joint_node_names}

    for target_name, kernels in joint_transitions.items():
        target_branch = per_target_params.setdefault(target_name, {})
        for kernel_name, raw in kernels.items():
            variants = _joint_variants(raw)
            support_functions = [
                kernel.support for kernel in variants if callable(kernel.support)
            ]
            probability_functions = [kernel.probabilities for kernel in variants]
            kernel_branch = target_branch.setdefault(kernel_name, {})
            kernel_branch["support"] = _union_callable_params(
                functions=support_functions, non_params=support_non_params
            )
            kernel_branch["probabilities"] = _union_callable_params(
                functions=probability_functions, non_params=probability_non_params
            )

            for output_name in variants[0].outputs:
                params = _union_callable_params(
                    functions=[kernel.outputs[output_name] for kernel in variants],
                    non_params=output_non_params,
                )
                transition_name = f"next_{output_name}"
                target_branch[transition_name] = (
                    target_branch.get(transition_name, {}) | params
                )


def _joint_variants(raw: JointTransition | Phased) -> tuple[JointTransition, ...]:
    """Return one or both phase variants of a joint declaration."""
    if isinstance(raw, Phased):
        return cast(
            "tuple[JointTransition, JointTransition]", (raw.solve, raw.simulate)
        )
    return (raw,)


def _joint_variant_for_phase(
    *, raw: JointTransition | Phased, phase: Literal["solve", "simulate"]
) -> JointTransition:
    """Return the concrete joint declaration used in one phase."""
    if isinstance(raw, Phased):
        return cast("JointTransition", raw.solve if phase == "solve" else raw.simulate)
    return raw


def _joint_transition_node_names(user_regime: UserRegime) -> frozenset[str]:
    """Return every public transition-local node name declared by a regime."""
    return frozenset(
        kernel_name
        for kernels in user_regime.joint_transitions.values()
        for kernel_name in kernels
    )


def _union_callable_params(
    *, functions: list[UserFunction], non_params: set[str]
) -> dict[str, str]:
    """Union signature-derived parameters over one role's phase variants."""
    tree: dict[str, str] = {}
    for index, func in enumerate(functions):
        tree |= dict(dt.create_tree_with_input_types({f"role_{index}": func}))
    return {
        arg_name: annotation
        for arg_name, annotation in sorted(tree.items())
        if arg_name not in non_params
    }


def _discovered_params(
    *,
    name: FunctionName | TransitionFunctionName,
    func: UserFunction | Phased,
    non_params: set[str],
    strip_target_value_operands: bool,
    kind: str | None = None,
) -> dict[str, str]:
    """Return the parameters one collected template entry contributes.

    Args:
        name: Template key the entry is collected under.
        func: The entry's callable, or the `Phased` pair whose two variants'
            parameters are unioned so one flat params dict satisfies both phases.
        non_params: Names the engine wires at call time, which never surface as
            user-facing parameters.
        strip_target_value_operands: Whether the reserved `V_target` vocabulary
            is engine-wired here too, which holds for a gated edge's callables.
        kind: How an invalid argument name's error introduces the arguments; the
            template key `name` when omitted.

    Returns:
        Dictionary of parameter name to type annotation, in name order.

    Raises:
        InvalidNameError: If a parameter's name is not a valid parameter-path
            segment.

    """
    if isinstance(func, Phased):
        tree = {
            arg_name: annotation
            for variant in _callables_in(value=func)
            for arg_name, annotation in dt.create_tree_with_input_types(
                {name: variant}
            ).items()
        }
    else:
        tree = dict(dt.create_tree_with_input_types({name: func}))
    # `dags` nests a `__`-joined argument into a subtree, so the argument names
    # are read back from the flattened tree.
    if errors := path_segment_name_errors(
        kind=f"{name!r} argument" if kind is None else kind,
        names=[arg for arg in dt.flatten_to_qnames(tree) if arg not in non_params],
    ):
        raise InvalidNameError(errors[0])
    return {
        arg_name: annotation
        for arg_name, annotation in sorted(tree.items())
        if arg_name not in non_params
        and not (strip_target_value_operands and is_target_value_operand(arg_name))
    }


# keyword-only-exempt: primary-argument=user_regime
def _fail_if_a_joint_node_is_read_outside_its_transition(
    user_regime: UserRegime,
    *,
    template_functions: Mapping[str, object],
) -> None:
    """Reject transition-local nodes outside target-output evaluation.

    A joint node is engine-wired only while one target edge's output DAG is
    evaluated. It is not a parameter, a current-period payoff input, or an input
    to support/probability construction. Helpers inherit that permission only on
    a path feeding a transition output. Target-specific scope is checked again
    after canonicalization, when the target plans are available.
    """
    joint_node_names = _joint_transition_node_names(user_regime)
    if not joint_node_names:
        return

    for phase in ("solve", "simulate"):
        transition_role = _function_names_in_transition_role(
            user_regime=user_regime, phase=phase
        )
        consumers: dict[str, object] = {
            name: func
            for name, func in template_functions.items()
            if tree_path_from_qname(name)[0] not in transition_role
        }
        if user_regime.koopmans_aggregator is not None:
            consumers["koopmans_aggregator"] = user_regime.koopmans_aggregator
        for name, func in consumers.items():
            _fail_if_a_joint_node_is_read(
                consumer_name=name,
                reserved=_joint_nodes_reachable_from(
                    func=func,
                    functions=user_regime.decomposed_functions,
                    phase=phase,
                    joint_node_names=joint_node_names,
                ),
                allowed_context="a target transition output and the helpers feeding it",
            )

        for target, kernels in user_regime.joint_transitions.items():
            for kernel_name, raw in kernels.items():
                kernel = _joint_variant_for_phase(raw=raw, phase=phase)
                roles: dict[str, object] = {
                    "probabilities": kernel.probabilities,
                }
                if callable(kernel.support):
                    roles["support"] = kernel.support
                for role, func in roles.items():
                    consumer_name = (
                        f"joint transition '{kernel_name}' {role} on target '{target}'"
                    )
                    _fail_if_a_joint_node_is_read(
                        consumer_name=consumer_name,
                        reserved=_joint_nodes_reachable_from(
                            func=func,
                            functions=user_regime.decomposed_functions,
                            phase=phase,
                            joint_node_names=joint_node_names,
                        ),
                        allowed_context=(
                            "neither support nor probability construction; express "
                            "correlation through one joint support"
                        ),
                    )
                    _fail_if_joint_role_reads_a_next_output(
                        consumer_name=consumer_name,
                        reserved=_next_names_reachable_from(
                            func=func,
                            functions=user_regime.decomposed_functions,
                            phase=phase,
                        ),
                    )
                    if role == "support":
                        _fail_if_joint_support_reads_runtime_names(
                            consumer_name=consumer_name,
                            func=func,
                            phase=phase,
                            user_regime=user_regime,
                        )


def _fail_if_joint_role_reads_a_next_output(
    *,
    consumer_name: str,
    reserved: Mapping[str, tuple[FunctionName, ...]],
) -> None:
    """Reject support or probabilities conditioned on a realized output."""
    if not reserved:
        return
    routes = [
        f"'{name}' through {' -> '.join(repr(step) for step in chain)}"
        if chain
        else f"'{name}'"
        for name, chain in sorted(reserved.items())
    ]
    raise InvalidNameError(
        f"{consumer_name} reads next-period output(s) {', '.join(routes)}. "
        "Joint support and probabilities are formed before any target output "
        "is realized, so they cannot condition on a `next_<state>` value."
    )


def _fail_if_joint_support_reads_runtime_names(
    *,
    consumer_name: str,
    func: object,
    phase: Literal["solve", "simulate"],
    user_regime: UserRegime,
) -> None:
    """Reject source states, actions, and helpers in a support provider."""
    forbidden = {
        *user_regime.states,
        *user_regime.actions,
        *user_regime.decomposed_functions,
        *user_regime.constraints,
        *user_regime.value_constraints,
        "CE",
    }
    inputs = {
        tree_path_from_qname(arg)[-1]
        for variant in _callables_in(value=func, phase=phase)
        for arg in dt.create_tree_with_input_types({"_": variant})
    }
    invalid = sorted(inputs & forbidden)
    if not invalid:
        return
    raise InvalidNameError(
        f"{consumer_name} reads source runtime name(s) {invalid}. A callable "
        "JointTransition support may read only `period`, `age`, and parameters; "
        "source states, actions, helpers, constraints, and realized transition "
        "values are unavailable because support is hoisted outside source-cell "
        "evaluation."
    )


def _joint_nodes_reachable_from(
    *,
    func: object,
    functions: Mapping[FunctionName, UserFunction | Phased | None],
    phase: Literal["solve", "simulate"],
    joint_node_names: frozenset[str],
) -> dict[str, tuple[FunctionName, ...]]:
    """Return joint-node names reachable from one consumer, with helper routes."""
    reached: dict[str, tuple[FunctionName, ...]] = {}
    walked: set[FunctionName] = set()
    frontier: list[tuple[UserFunction, tuple[FunctionName, ...]]] = [
        (variant, ()) for variant in _callables_in(value=func, phase=phase)
    ]
    while frontier:
        current, chain = frontier.pop()
        for arg in dt.create_tree_with_input_types({"_": current}):
            arg_name = tree_path_from_qname(arg)[-1]
            if arg_name in joint_node_names:
                reached.setdefault(arg_name, chain)
            elif arg_name in functions and arg_name not in walked:
                walked.add(arg_name)
                frontier.extend(
                    (variant, (*chain, arg_name))
                    for variant in _callables_in(value=functions[arg_name], phase=phase)
                )
    return reached


def _fail_if_a_joint_node_is_read(
    *,
    consumer_name: str,
    reserved: Mapping[str, tuple[FunctionName, ...]],
    allowed_context: str,
) -> None:
    """Reject a consumer that would reclassify a joint node as a parameter."""
    if not reserved:
        return
    routes = [
        f"'{name}' through {' -> '.join(repr(step) for step in chain)}"
        if chain
        else f"'{name}'"
        for name, chain in sorted(reserved.items())
    ]
    raise InvalidNameError(
        f"'{consumer_name}' reads transition-local joint node(s) "
        f"{', '.join(routes)}. A joint node is engine-wired only while evaluating "
        f"{allowed_context}; it is never a user parameter."
    )


# keyword-only-exempt: primary-argument=user_regime
def _fail_if_a_next_name_is_read_outside_a_transition(
    user_regime: UserRegime, *, template_functions: Mapping[str, object]
) -> None:
    """Check that only a state transition, or a function feeding one, reads `next_`.

    Whether a consumer may name a next-period value is a question about *when*
    that consumer runs, so the check is indexed by consumer and by phase rather
    than by name:

    - a state transition, and every regime function it feeds, runs where
      next-period values exist and may read them;
    - utility, constraints, derived categoricals, the Koopmans aggregator and the
      certainty equivalent are this period's payoff and see none;
    - the regime transition selects the target *before* that target's state laws
      run, so it sees none either.

    The reach through helpers is what makes this a walk rather than a signature
    test. A helper feeding both a law and utility is legitimate on the law's path
    and not on utility's, and permission granted for the one would otherwise
    cover the other silently.

    Args:
        user_regime: User-form `Regime` instance.
        template_functions: Every callable the template collects for the regime,
            from `_collect_all_functions_for_template`.

    Raises:
        InvalidNameError: If a consumer that runs before next-period values exist
            reads one, directly or through a regime function.

    """
    for phase in ("solve", "simulate"):
        transition_role = _function_names_in_transition_role(
            user_regime=user_regime, phase=phase
        )
        consumers: dict[str, object] = {
            name: func
            for name, func in template_functions.items()
            if tree_path_from_qname(name)[0] not in transition_role
        }
        if user_regime.koopmans_aggregator is not None:
            consumers["koopmans_aggregator"] = user_regime.koopmans_aggregator
        for name, func in consumers.items():
            _fail_if_a_next_name_is_read(
                consumer_name=name,
                reserved=_next_names_reachable_from(
                    func=func, functions=user_regime.decomposed_functions, phase=phase
                ),
            )

    # A certainty equivalent declares parameter names rather than a signature to
    # walk, so its public names get the same test the walk applies to arguments.
    if user_regime.certainty_equivalent is not None:
        _fail_if_a_next_name_is_read(
            consumer_name="certainty_equivalent",
            reserved={
                param_name: ()
                for param_name in user_regime.certainty_equivalent.param_names
                if param_name.startswith("next_")
            },
        )


def _next_names_reachable_from(
    *,
    func: object,
    functions: Mapping[FunctionName, UserFunction | Phased | None],
    phase: Literal["solve", "simulate"],
) -> dict[str, tuple[FunctionName, ...]]:
    """Return every `next_`-prefixed argument reachable from `func`, with its route.

    Args:
        func: Regime slot value to walk — a callable, a `Phased` entry, or a
            per-target dict.
        functions: The regime's functions, which the walk descends into.
        phase: Phase whose variant is taken from each `Phased` entry.

    Returns:
        Mapping of each reachable `next_<name>` to the chain of regime functions
        leading to it, empty for one named in `func`'s own signature.

    """
    reached: dict[str, tuple[FunctionName, ...]] = {}
    walked: set[FunctionName] = set()
    frontier: list[tuple[UserFunction, tuple[FunctionName, ...]]] = [
        (variant, ()) for variant in _callables_in(value=func, phase=phase)
    ]
    while frontier:
        current, chain = frontier.pop()
        for arg in dt.create_tree_with_input_types({"_": current}):
            arg_name = tree_path_from_qname(arg)[-1]
            if arg_name.startswith("next_"):
                reached.setdefault(arg_name, chain)
            elif arg_name in functions and arg_name not in walked:
                walked.add(arg_name)
                frontier.extend(
                    (variant, (*chain, arg_name))
                    for variant in _callables_in(value=functions[arg_name], phase=phase)
                )
    return reached


def _fail_if_a_next_name_is_read(
    *, consumer_name: FunctionName, reserved: Mapping[str, tuple[FunctionName, ...]]
) -> None:
    """Reject a consumer that names a next-period value where none exists.

    `next_<name>` names a value, never a parameter. Admitting it here would hand
    the user a parameter slot under a name that says it is a next-period value,
    and answer a next-period question with a constant supplied at solve time.

    The prefix stays reserved outside a transition even for a quantity that is in
    fact determined within the period, such as a post-decision stock. What
    `next_<name>` denotes depends on where the value is going: with one law per
    target it is target-specific, and with a stochastic law it is a distribution
    rather than a number. Admitting the name wherever the declaration happens to
    make it single-valued would make a utility function's legality depend on the
    regime's transition topology, so that adding a second target invalidates a
    payoff that never mentioned targets. A quantity a payoff or a constraint needs
    is an ordinary function of this period's states and actions; the law reads that
    function, and so does everything else.

    Args:
        consumer_name: Consumer being checked, named in the message.
            `koopmans_aggregator` and `certainty_equivalent` enter under their
            pseudo-function names.
        reserved: Mapping of each reserved name the consumer reads to the chain of
            regime functions leading to it, empty for a direct read.

    Raises:
        InvalidNameError: If `reserved` is non-empty.

    """
    if not reserved:
        return
    names = sorted(reserved)
    routes = [
        f"'{name}' (read by '{consumer_name}' through "
        f"{' -> '.join(repr(step) for step in reserved[name])})"
        if reserved[name]
        else f"'{name}'"
        for name in names
    ]
    example = names[0].removeprefix("next_")
    raise InvalidNameError(
        f"'{consumer_name}' reads {', '.join(routes)}, but the 'next_' prefix "
        f"names the output of a state transition and is never a parameter. Only a "
        f"transition law, and what feeds one, may read a next-period value; "
        f"elsewhere the name would be bound to a constant supplied at solve time, "
        f"answering a next-period question with a number that has nothing to do "
        f"with next period. If the quantity is determined within this period — a "
        f"post-decision stock, or next period's assets as this period's assets "
        f"minus consumption — give it its own name as an ordinary function of this "
        f"period's states and actions, and have both '{consumer_name}' and the law "
        f"read that function: define `new_{example}(...)`, then declare "
        f"`state_transitions={{'{example}': lambda new_{example}: new_{example}}}`. "
        f"If a parameter was meant, rename the argument."
    )


def _add_koopmans_aggregator_params(
    *, function_params: dict[FunctionName, dict[str, str]], user_regime: UserRegime
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
        *user_regime.decomposed_functions,
        "period",
        "age",
        "utility",
        "CE",
    }
    params = {k: v for k, v in sorted(tree.items()) if k not in variables}
    function_params["koopmans_aggregator"] = params


def _add_pareto_objective_params(
    *, function_params: dict[FunctionName, dict[str, str]], user_regime: UserRegime
) -> None:
    """Add the Pareto weights' free params under their pseudo-function name.

    Every stakeholder's weight reads one shared namespace, so a weight named
    in two of them is one parameter and appears once. A weight argument that
    names a state, or the engine context, is wired at call time and never
    surfaces.
    """
    objective = user_regime.pareto_objective
    if objective is None:
        return
    if PARETO_OBJECTIVE_ENTRY in function_params:
        raise InvalidNameError(
            "The regime declares `pareto_objective`, whose parameters live "
            f"under the pseudo-function name '{PARETO_OBJECTIVE_ENTRY}' in the "
            "params — this conflicts with a regime function of the same name."
        )
    wired = {*user_regime.states, "period", "age"}
    params = {
        arg: "float"
        for weight in objective.weights.values()
        if callable(weight)
        for arg in get_union_of_args([weight])
        if arg not in wired
    }
    if params:
        function_params[PARETO_OBJECTIVE_ENTRY] = dict(sorted(params.items()))


def _add_certainty_equivalent_params(
    *, function_params: dict[FunctionName, dict[str, str]], user_regime: UserRegime
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
    *, function_params: dict[FunctionName, dict[str, str]], user_regime: UserRegime
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


def _callables_in(
    *, value: object, phase: Literal["solve", "simulate"] | None = None
) -> list[UserFunction]:
    """Return the callables a regime slot value stands for.

    A law and a regime function accept the same shapes, so one traversal serves
    both and they cannot come to disagree about what is walkable:

    - a plain callable stands for itself;
    - a `Phased` entry is two implementations rather than one, so a caller
      naming a phase gets that variant and a caller naming none gets both;
    - a per-target dict holds one law per target, each walked in turn;
    - `None` masks a model-level entry and stands for no implementation at all,
      so there is no signature to read.

    Args:
        value: A `state_transitions` or `functions` entry.
        phase: Phase whose variant to take from a `Phased` entry. `None` takes
            both, which is what the parameter template wants — it unions the two
            phases' parameters. Anything deciding what a consumer may *read*
            names its phase, since a variant reaching a law in one phase says
            nothing about the other.

    Returns:
        List of the callables it stands for, empty when it stands for none.

    """
    if value is None:
        return []
    if isinstance(value, Phased):
        if phase == "solve":
            return _callables_in(value=value.solve, phase=phase)
        if phase == "simulate":
            return _callables_in(value=value.simulate, phase=phase)
        return [*_callables_in(value=value.solve), *_callables_in(value=value.simulate)]
    if isinstance(value, Mapping) and not isinstance(value, StochasticTransition):
        return [
            callable_
            for member in value.values()
            for callable_ in _callables_in(value=member, phase=phase)
        ]
    return [cast("UserFunction", value)]


def _function_names_in_transition_role(
    *, user_regime: UserRegime, phase: Literal["solve", "simulate"]
) -> frozenset[str]:
    """Return the names of functions that compute, or feed, a state transition.

    A state transition is the only consumer evaluated where next-period values
    exist, so it and the regime functions feeding it are the only ones that may
    read a `next_<name>`. The regime transition is deliberately absent: it
    selects the target and runs *before* that target's state laws, so a regime
    probability naming a next-period state names something not yet computed.

    Args:
        user_regime: User-form `Regime` instance.
        phase: Phase whose variants are walked. A `Phased` helper feeding a law
            in one phase carries no permission into the other.

    Returns:
        Frozenset of the regime's state-transition function names together with
        every regime function they read, directly or through other regime
        functions.

    """
    # A law binds an argument to a DAG node, so the names to resolve against
    # are the engine's, not the author's.
    functions = user_regime.decomposed_functions
    feeders: set[str] = set()
    frontier = [
        variant
        for law in user_regime.state_transitions.values()
        for variant in _callables_in(value=law, phase=phase)
    ]
    frontier.extend(
        output
        for kernels in user_regime.joint_transitions.values()
        for raw in kernels.values()
        for output in _joint_variant_for_phase(raw=raw, phase=phase).outputs.values()
    )
    while frontier:
        func = frontier.pop()
        for arg in dt.create_tree_with_input_types({"_": func}):
            arg_name = tree_path_from_qname(arg)[-1]
            if arg_name in feeders or arg_name not in functions:
                continue
            feeders.add(arg_name)
            frontier.extend(_callables_in(value=functions[arg_name], phase=phase))

    joint_output_names = {
        f"next_{state_name}"
        for kernels in user_regime.joint_transitions.values()
        for raw in kernels.values()
        for state_name in _joint_variant_for_phase(raw=raw, phase=phase).outputs
    }
    return frozenset(
        {f"next_{name}" for name in user_regime.state_transitions}
        | joint_output_names
        | feeders
    )


# keyword-only-exempt: primary-argument=user_regime
def _collect_all_functions_for_template(
    user_regime: UserRegime,
    *,
    law: RegimeLaw,
    edge_entries: Mapping[FunctionName, UserFunction],
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
        name: func
        for name, func in user_regime.decomposed_functions.items()
        if func is not None
    }
    result |= {
        name: func
        for name, func in user_regime.decomposed_constraints.items()
        if func is not None
    }
    # Value-constraint predicates carry user params exactly like ordinary
    # constraints; their engine-wired `Q_<s>` / reference-value arguments are
    # excluded via `variables`.
    result |= dict(user_regime.value_constraints)
    # A carried state contributes its `solve` variant as a derived function
    # under the state's name (solve-phase imputation), so its parameters
    # surface in the template. Its law of motion is its regular
    # `state_transitions` entry (keyed `next_<name>`), collected below.
    for name, spec in user_regime.states.items():
        if isinstance(spec, Phased):
            result[name] = cast("UserFunction", spec.solve)
    if not law.terminal:
        joint_output_names = {
            output_name
            for kernels in user_regime.joint_transitions.values()
            for raw in kernels.values()
            for joint in _joint_variants(raw)
            for output_name in joint.outputs
        }
        result |= collect_state_transitions(
            states=user_regime.states,
            state_transitions=user_regime.state_transitions,
            joint_output_names=joint_output_names,
        )
        result |= _regime_transition_entries(law.decomposed_transition)
    result |= edge_entries
    return result


def _gated_edge_entries(law: RegimeLaw) -> dict[FunctionName, UserFunction]:
    """Key every gated-edge callable of a regime for the read checks.

    A gated edge is declared with three kinds of user callable, each an ordinary
    DAG function:

    - the `gate` predicate;
    - one projection per state of each gate reference's reference regime;
    - one projection per state of each route fallback's reference regime, per
      phase when the fallback is `Phased`.

    Every key ends in `__<target>` and is unique per callable; the keys serve
    only to tell the callables apart.

    Args:
        law: The source regime's law, whose gated edges are keyed.

    Returns:
        Dictionary of a per-callable key to the callable, empty for a regime
        declaring no gated edge.

    """
    entries: dict[FunctionName, UserFunction] = {}
    for target_regime_name, edge in law.gated_edges.items():
        entries[qname_from_tree_path((GATE, target_regime_name))] = edge.gate
        for ref_name, ref in edge.gate_refs.items():
            for state_name, projection in ref.projection.items():
                key = f"{GATE_REFERENCES}_{ref_name}_{state_name}"
                entries[qname_from_tree_path((key, target_regime_name))] = projection
        for leg_name, leg in edge.legs.items():
            for phase, ref in _fallbacks_by_phase(
                solve=leg.solve_fallback,
                simulate=leg.simulate_fallback,
                is_phased=leg.fallback_is_phased,
            ):
                for state_name, projection in ref.projection.items():
                    key = f"{phase or 'both'}_{ROUTES}_{leg_name}_{state_name}"
                    entries[qname_from_tree_path((key, target_regime_name))] = (
                        projection
                    )
    return entries


def _fallbacks_by_phase(
    *,
    solve: ProjectedRegimeValue,
    simulate: ProjectedRegimeValue,
    is_phased: bool,
) -> tuple[tuple[Literal["solve", "simulate"] | None, ProjectedRegimeValue], ...]:
    """Return a route fallback's references, named by phase when `Phased`.

    A `Phased` fallback is two sets of callables with two parameter sets, one per
    phase; a fallback declaring one reference for both is one set, named by no
    phase.
    """
    if is_phased:
        return (("solve", solve), ("simulate", simulate))
    return ((None, solve),)


def _gated_edge_wired_names(
    *,
    gate_reference_names: frozenset[str],
    target_state_names: frozenset[StateName],
) -> set[str]:
    """Return the names ONE edge's callables read that the engine binds itself.

    A gate and a projection are evaluated on that edge's TARGET regime's grid, so
    beyond the source regime's own vocabulary they read names no user supplies:

    - the target regime's states, which the fold binds from that regime's grids;
    - `D_target`, the target's dissolution flag;
    - each key of THIS edge's `gate_references`, bound to that reference's
      interpolated value.

    The set is per edge because that is what makes the answer right. A name that
    is engine-bound on one edge — the target's own state, another edge's
    reference key, or any state of a regime this edge never reaches — is an
    ordinary parameter here, and widening the set to a model-wide union drops it
    from the template, leaving the gate reading a value nothing can supply.

    The target's own value components enter under the reserved `V_target`
    vocabulary and are recognised by `is_target_value_operand` instead, since
    their per-stakeholder spellings depend on the target regime rather than on
    anything the source declares.

    Args:
        gate_reference_names: The keys of the edge's gate references.
        target_state_names: State names declared by that edge's target regime.

    Returns:
        Set of the names this edge's callables may read without them being
        parameters.

    """
    return {"D_target", *target_state_names, *gate_reference_names}


def _drop_engine_provided_args(
    *, name: FunctionName, params: dict[str, str], user_regime: UserRegime
) -> None:
    """Remove a function's engine-supplied arguments from its discovered params.

    In a continuation-based (Euler-inversion) regime the inversion function
    `inverse_marginal_utility` receives `marginal_continuation` from the EGM
    kernel (in a regime whose solver reads no continuation, a function of that
    name is ordinary). This must not surface as a user-facing param, so it is
    popped in place. Gated on the declared continuation demand, not the
    concrete solver type, so every Euler-inversion solver is covered.
    """
    if (
        name == "inverse_marginal_utility"
        and user_regime.solver.required_continuation_keys
    ):
        params.pop("marginal_continuation", None)


def _regime_transition_entries(
    transition: object,
) -> dict[TransitionFunctionName, UserFunction | Phased]:
    """Key the regime transition for the read checks.

    The entries let the checks walk what the law reads and keep it out of the
    regime's own template; its parameters live in the edge namespace, at
    `params["edges"][source]` for a coarse law and at
    `params["edges"][source][target]` for a per-target cell
    (`create_edge_params_template`).

    - coarse forms ⇒ one `next_regime` entry
    - a per-target dict ⇒ one `next_regime__<target>` entry per cell
    - `Phased` per-target dicts ⇒ per-cell `Phased` entries, so both phases'
      callables are walked per target; an absent phase contributes none

    """
    if isinstance(transition, Phased) and isinstance(transition.solve, Mapping):
        solve_cells = cast("Mapping[RegimeName, UserFunction]", transition.solve)
        simulate_cells = cast("Mapping[RegimeName, UserFunction]", transition.simulate)
        return {
            f"next_regime__{target_regime_name}": Phased(
                solve=solve_cells.get(target_regime_name),
                simulate=simulate_cells.get(target_regime_name),
            )
            for target_regime_name in dict.fromkeys((*solve_cells, *simulate_cells))
        }
    if isinstance(transition, Mapping):
        cells = cast("Mapping[RegimeName, UserFunction]", transition)
        return {
            f"next_regime__{target_regime_name}": cell
            for target_regime_name, cell in cells.items()
        }
    return {"next_regime": cast("UserFunction | Phased", transition)}


def _validate_no_shadowing(
    *, function_params: dict[FunctionName, dict[str, str]], user_regime: UserRegime
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
