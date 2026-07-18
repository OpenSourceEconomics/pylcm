import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags import concatenate_functions, get_ancestors, with_signature

from _lcm.certainty_equivalent import CertaintyEquivalent, resolve_certainty_equivalent
from _lcm.regime_building.h_dag import _get_build_H_kwargs
from _lcm.regime_building.next_state import (
    get_next_state_function_for_solution,
    get_next_stochastic_weights_function,
)
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.typing import (
    ConstraintFunction,
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
    _ParamsLeaf,
)
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import get_union_of_args
from lcm.typing import BoolND, Float1D, FloatND


def get_Q_and_F(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    co_map_state_names: tuple[StateName, ...] = (),
    certainty_equivalent: CertaintyEquivalent | None = None,
    continuation_functions: EconFunctionsMapping | None = None,
    flow_transitions: TransitionFunctionsMapping | None = None,
    flow_stochastic_transition_names: frozenset[TransitionFunctionName] | None = None,
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    `age` and `period` are runtime arguments (via `**states_actions_params`),
    not closure constants. This allows periods with the same target
    configuration to share a single JIT-compiled function.

    Q mixes two phases when it is built for the simulate phase: the *current* flow
    (utility, feasibility, `H`) is simulate-phase, while the *continuation* is priced
    under the agent's perceived law — the solve phase. The flow is *now*, so it is
    realized under the true law; the belief is about the *future*, so it prices only
    the continuation.

    Each of the two sub-DAGs must be **phase-closed**: a transition law is a DAG node
    like any other, and `dags` resolves its argument names against a function pool
    transitively, so a law that depends on a `Phased` helper picks up whichever variant
    that pool holds. It therefore takes a matched (transitions, functions) pair per
    role:

    - flow: `flow_transitions` + `functions`,
    - continuation: `transitions` + `continuation_functions`.

    Mixing them across roles — e.g. a solve outer `next_<state>` resolving its helpers
    from the simulate pool — yields a sub-DAG that is neither phase and can reverse the
    argmax. The same `next_<state>` name legitimately resolves to *different* callables
    in the two roles; that is the phase split, not an inconsistency.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
            Supplies the current-period flow (utility, feasibility, `H`).
        constraints: Immutable mapping of constraint names to internal user functions.
        period_targets: Target regimes whose continuation enters E[V]
            this period (reachable, with state laws, active next period).
        transitions: Immutable mapping of transition names to transition functions.
        stochastic_transition_names: Frozenset of stochastic transition function names.
        compute_regime_transition_probs: Regime transition probability function
            for solve.
        regime_to_v_interpolation_info: Mapping of regime names to V-interpolation
            info.
        co_map_state_names: Tuple of state names co-mapped with the continuation V —
            their axes are sliced off each `next_V_arr` leaf by the backward-induction
            co-map, so their coordinates are dropped from the interpolation. Only fixed
            (never-transitioning) distributed states qualify.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.
        continuation_functions: Function pool the continuation sub-DAG (the state
            transitions and the stochastic weights) is resolved against. Defaults to
            `functions`, which is correct in the solve phase, where both pools are the
            solve pool. The simulate phase must pass the SOLVE pool here so the agent
            compares actions under its perceived law while the world is realized under
            the true one.
        flow_transitions: Transition bundle the *flow* `next_<state>` nodes are taken
            from — the ones a within-period utility or feasibility may read (the NEGM
            service-flow pattern). Defaults to `transitions`, which is correct in the
            solve phase. The simulate phase must pass the SIMULATE transitions, so that
            the flow sub-DAG is closed under the simulate pool supplied as `functions`.
        flow_stochastic_transition_names: Stochastic names to exclude when merging
            `flow_transitions`. Defaults to `stochastic_transition_names`. It is a
            separate argument because a state may be stochastic in one phase and
            deterministic in the other.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    # In the solve phase the two roles coincide; only simulate passes them apart.
    continuation_pool = (
        functions if continuation_functions is None else continuation_functions
    )
    flow_pool = transitions if flow_transitions is None else flow_transitions
    flow_stochastic_names = (
        stochastic_transition_names
        if flow_stochastic_transition_names is None
        else flow_stochastic_transition_names
    )
    # The flow's `next_<state>` nodes pair with `functions`; the continuation's pair
    # with `continuation_pool`. Keeping the two merges separate is what makes each
    # sub-DAG phase-closed.
    deterministic_transitions, conflicting_deterministic_transition_names = (
        _get_deterministic_transitions(
            transitions=flow_pool,
            stochastic_transition_names=flow_stochastic_names,
        )
    )
    U_and_F = _get_U_and_F(
        functions=functions,
        constraints=constraints,
        deterministic_transitions=deterministic_transitions,
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
        stochastic_transition_names=flow_stochastic_names,
    )
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[RegimeName, frozenset[str]] = {}

    for target_regime_name in period_targets:
        # Transitions from the current regime to the target regime
        bundle = transitions[target_regime_name]

        # Functions required to calculate the expected continuation values. These read
        # `continuation_pool`, NOT `functions`: the continuation is priced under the
        # perceived (solve-phase) law, helpers included.
        state_transitions[target_regime_name] = get_next_state_function_for_solution(
            functions=continuation_pool,
            transitions=bundle,
        )
        next_stochastic_states_weights[target_regime_name] = (
            get_next_stochastic_weights_function(
                functions=continuation_pool,
                transitions=bundle,
                stochastic_transition_names=stochastic_transition_names,
                regime_name=target_regime_name,
            )
        )
        joint_weights_from_marginals[target_regime_name] = _get_joint_weights_function(
            transitions=bundle,
            stochastic_transition_names=stochastic_transition_names,
            regime_name=target_regime_name,
        )
        V_arr_name = "next_V_arr"
        next_V_interpolator = get_V_interpolator(
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
            state_prefix="next_",
            V_arr_name=V_arr_name,
            co_map_state_names=co_map_state_names,
        )
        # Determine extra kwargs needed by next_V beyond next_states and next_V_arr
        # (e.g. wealth__points for IrregSpacedGrid with runtime-supplied points).
        next_V_extra_param_names[target_regime_name] = frozenset(
            get_union_of_args([next_V_interpolator]) - set(bundle) - {V_arr_name}
        )
        stochastic_variables = tuple(
            key for key in bundle if key in stochastic_transition_names
        )
        next_V[target_regime_name] = productmap(
            func=next_V_interpolator,
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    # ----------------------------------------------------------------------------------
    # Create the state-action value and feasibility function
    # ----------------------------------------------------------------------------------

    _build_H_kwargs = _get_build_H_kwargs(functions)
    ce, ce_transform_flat_names, ce_inverse_flat_names = resolve_certainty_equivalent(
        certainty_equivalent
    )

    # Co-mapped states are sliced off each `next_V_arr` leaf by the backward-
    # induction co-map, so their `next_`-prefixed coordinates are not passed to
    # the interpolator (which no longer indexes those axes).
    _co_map_next_names = frozenset(f"next_{name}" for name in co_map_state_names)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[
            U_and_F,
            compute_regime_transition_probs,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
        ],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: FloatND,
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action value and feasibility for a non-terminal period.

        Args:
            next_regime_to_V_arr: The next period's value function array.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        U_arr, F_arr = U_and_F(**states_actions_params)
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in period_targets}
        )

        E_next_V = jnp.zeros_like(U_arr)
        for target_regime_name in period_targets:
            next_states = state_transitions[target_regime_name](
                **states_actions_params,
            )
            marginal_next_stochastic_states_weights = next_stochastic_states_weights[
                target_regime_name
            ](**states_actions_params)
            joint_next_stochastic_states_weights = joint_weights_from_marginals[
                target_regime_name
            ](**marginal_next_stochastic_states_weights)

            # As we productmap'd the value function over the stochastic variables, the
            # resulting next value function gets a new dimension for each stochastic
            # variable.
            extra_kw = {
                k: states_actions_params[k]
                for k in next_V_extra_param_names[target_regime_name]
            }
            next_V_at_stochastic_states_arr = next_V[target_regime_name](
                **{
                    name: val
                    for name, val in next_states.items()
                    if name not in _co_map_next_names
                },
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )
            if ce is not None:
                next_V_at_stochastic_states_arr = ce.transform(
                    value=next_V_at_stochastic_states_arr,
                    **{
                        arg: states_actions_params[flat_name]
                        for arg, flat_name in ce_transform_flat_names.items()
                    },
                )

            # We then take the weighted average of the next value function at the
            # stochastic states to get the expected next value function.
            next_V_expected_arr = jnp.average(
                next_V_at_stochastic_states_arr,
                weights=joint_next_stochastic_states_weights,
            )
            E_next_V = (
                E_next_V + active_regime_probs[target_regime_name] * next_V_expected_arr
            )

        if ce is not None:
            E_next_V = ce.inverse(
                value=E_next_V,
                **{
                    arg: states_actions_params[flat_name]
                    for arg, flat_name in ce_inverse_flat_names.items()
                },
            )

        Q_arr = functions["H"](
            utility=U_arr,
            E_next_V=E_next_V,
            **_build_H_kwargs(states_actions_params),
        )

        # Handle cases when there is only one state.
        # In that case, Q_arr and F_arr are scalars, but we require arrays as output.
        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_compute_intermediates(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    certainty_equivalent: CertaintyEquivalent | None = None,
) -> Callable:
    """Build a closure that computes Q_and_F intermediates for diagnostics.

    Mirrors `get_Q_and_F` but returns all intermediates instead of just
    `(Q, F)`. The caller productmaps and JIT-compiles the closure; it runs
    only in the error path when `validate_V` detects NaN. `age` and `period`
    are runtime arguments (passed via `states_actions_params`) so that
    periods sharing the same target configuration share a single
    JIT-compiled function.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to constraint functions.
        period_targets: Target regimes whose continuation enters E[V]
            this period (reachable, with state laws, active next period).
        transitions: Immutable mapping of target regime names to state transition
            functions.
        stochastic_transition_names: Frozenset of stochastic transition function
            names.
        compute_regime_transition_probs: Callable returning regime transition
            probabilities for the current regime.
        regime_to_v_interpolation_info: Immutable mapping of regime names to
            V-interpolation info.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.

    Returns:
        Closure returning `(U_arr, F_arr, E_next_V, Q_arr, active_regime_probs)`.

    """
    deterministic_transitions, conflicting_deterministic_transition_names = (
        _get_deterministic_transitions(
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
        )
    )
    U_and_F = _get_U_and_F(
        functions=functions,
        constraints=constraints,
        deterministic_transitions=deterministic_transitions,
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
        stochastic_transition_names=stochastic_transition_names,
    )
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[RegimeName, frozenset[str]] = {}

    for target_regime_name in period_targets:
        bundle = transitions[target_regime_name]
        state_transitions[target_regime_name] = get_next_state_function_for_solution(
            functions=functions,
            transitions=bundle,
        )
        next_stochastic_states_weights[target_regime_name] = (
            get_next_stochastic_weights_function(
                functions=functions,
                transitions=bundle,
                stochastic_transition_names=stochastic_transition_names,
                regime_name=target_regime_name,
            )
        )
        joint_weights_from_marginals[target_regime_name] = _get_joint_weights_function(
            transitions=bundle,
            stochastic_transition_names=stochastic_transition_names,
            regime_name=target_regime_name,
        )
        V_arr_name = "next_V_arr"
        next_V_interpolator = get_V_interpolator(
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
            state_prefix="next_",
            V_arr_name=V_arr_name,
        )
        next_V_extra_param_names[target_regime_name] = frozenset(
            get_union_of_args([next_V_interpolator]) - set(bundle) - {V_arr_name}
        )
        stochastic_variables = tuple(
            key for key in bundle if key in stochastic_transition_names
        )
        next_V[target_regime_name] = productmap(
            func=next_V_interpolator,
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    ce, ce_transform_flat_names, ce_inverse_flat_names = resolve_certainty_equivalent(
        certainty_equivalent
    )

    arg_names_of_compute_intermediates = _get_arg_names_of_Q_and_F(
        deps=[
            U_and_F,
            compute_regime_transition_probs,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
        ],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_compute_intermediates,
        return_annotation=(
            "tuple[FloatND, FloatND, FloatND, FloatND, "
            "MappingProxyType[RegimeName, FloatND]]"
        ),
    )
    def compute_intermediates(
        next_regime_to_V_arr: FloatND,
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[
        FloatND, FloatND, FloatND, FloatND, MappingProxyType[RegimeName, FloatND]
    ]:
        """Compute all Q_and_F intermediates."""
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        U_arr, F_arr = U_and_F(**states_actions_params)
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in period_targets}
        )

        E_next_V = jnp.zeros_like(U_arr)
        for target_regime_name in period_targets:
            next_states = state_transitions[target_regime_name](
                **states_actions_params,
            )
            marginal = next_stochastic_states_weights[target_regime_name](
                **states_actions_params,
            )
            joint = joint_weights_from_marginals[target_regime_name](**marginal)
            extra_kw = {
                k: states_actions_params[k]
                for k in next_V_extra_param_names[target_regime_name]
            }
            next_V_stoch = next_V[target_regime_name](
                **next_states,
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )
            if ce is not None:
                next_V_stoch = ce.transform(
                    value=next_V_stoch,
                    **{
                        arg: states_actions_params[flat_name]
                        for arg, flat_name in ce_transform_flat_names.items()
                    },
                )
            contribution = jnp.average(next_V_stoch, weights=joint)
            E_next_V = E_next_V + active_regime_probs[target_regime_name] * contribution

        if ce is not None:
            E_next_V = ce.inverse(
                value=E_next_V,
                **{
                    arg: states_actions_params[flat_name]
                    for arg, flat_name in ce_inverse_flat_names.items()
                },
            )

        Q_arr = functions["H"](
            utility=U_arr,
            E_next_V=E_next_V,
            **_get_build_H_kwargs(functions)(states_actions_params),
        )

        return U_arr, F_arr, E_next_V, Q_arr, active_regime_probs

    return compute_intermediates


def get_Q_and_F_terminal(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a terminal period.

    `age` and `period` are runtime arguments (via `**states_actions_params`).

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a terminal period.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[U_and_F],
        # While the terminal period does not depend on the value function array, we
        # include it in the signature, such that we can treat all periods uniformly
        # during the solution and simulation.
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: FloatND,  # noqa: ARG001
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action values and feasibilities for a terminal period.

        Args:
            next_regime_to_V_arr: Unused in the terminal period; accepted so that
                solve and simulate treat all periods uniformly.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple of the state-action value array (Q) and the feasibility
            mask (F).

        """
        U_arr, F_arr = U_and_F(**states_actions_params)
        return jnp.asarray(U_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_period_targets(
    *,
    period: int,
    transitions: TransitionFunctionsMapping,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
) -> tuple[RegimeName, ...]:
    """Return the target regimes whose continuation enters E[V] this period.

    The canonical transition bundles (`transitions` keys) carry exactly the
    reachable targets with at least one state law; the period filter keeps
    those active in the next period. A reachable target absent from the
    bundles has no states (its V is identically zero) and contributes
    nothing to the continuation.

    Args:
        period: The period to enumerate targets for.
        transitions: Immutable mapping of target regime names to their
            state transition functions.
        regimes_to_active_periods: Immutable mapping of regime names to
            their active period tuples.

    Returns:
        Tuple of this period's target regime names.

    """
    return tuple(
        regime_name
        for regime_name in transitions
        if period + 1 in regimes_to_active_periods.get(regime_name, ())
    )


def _get_arg_names_of_Q_and_F(
    *,
    deps: list[Callable[..., Any]],
    include: frozenset[str] = frozenset(),
    exclude: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Get the argument names of the dependencies.

    Args:
        deps: List of dependencies.
        include: Set of argument names to include.
        exclude: Set of argument names to exclude.

    Returns:
        The union of the argument names in deps and include, except for those in
        exclude.

    """
    return tuple((get_union_of_args(deps) | include) - exclude)


def _get_joint_weights_function(
    *,
    transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
    stochastic_transition_names: frozenset[TransitionFunctionName],
    regime_name: RegimeName,
) -> Callable[..., FloatND]:
    """Get function that calculates the joint weights.

    This function takes the weights of the individual stochastic variables and
    multiplies them together to get the joint weights on the product space of the
    stochastic variables.

    Args:
        transitions: Transitions of the target regime.
        stochastic_transition_names: Frozenset of stochastic transition function names.
        regime_name: Name of the target regime.

    Returns:
        A function that computes the outer product of the weights of the stochastic
        variables.

    """
    arg_names = [
        f"weight_{regime_name}__{key}"
        for key in transitions
        if key in stochastic_transition_names
    ]

    @with_signature(args=arg_names)
    def _outer(**kwargs: Float1D) -> FloatND:
        weights = jnp.array(list(kwargs.values()))
        return jnp.prod(weights)

    variables = tuple(arg_names)
    return productmap(
        func=_outer, variables=variables, batch_sizes=dict.fromkeys(variables, 0)
    )


def _get_deterministic_transitions(
    *,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
) -> tuple[
    Mapping[TransitionFunctionName, TransitionFunction],
    frozenset[TransitionFunctionName],
]:
    """Merge the deterministic `next_<state>` transitions across all targets.

    Iterates every target bundle, not just this period's targets: the within-
    period durable law (`next_<durable>`) lives in the source regime's own
    self-transition bundle and is needed even in periods bound for a terminal
    target that does not carry it. Own-regime within-period laws are
    target-independent, so the first occurrence of each `next_<state>` name is
    kept. Stochastic transitions are excluded — a within-period utility or
    constraint cannot read an unrealised stochastic next state.

    Returns the merged mapping and the set of `next_<state>` names that appear in
    more than one target bundle with non-identical implementations. The merge
    keeps one of them, so a within-period utility or constraint reading such a
    name would silently bind one target's law; the caller rejects the model if a
    conflicting name is actually read by the decision evaluation.

    Returns:
        Tuple of the immutable merged `next_<state>` mapping and the frozenset of
        conflicting `next_<state>` names.
    """
    merged: dict[TransitionFunctionName, TransitionFunction] = {}
    conflicting: set[TransitionFunctionName] = set()
    for bundle in transitions.values():
        for name, func in bundle.items():
            if name in stochastic_transition_names:
                continue
            if name in merged and _law_source(merged[name]) is not _law_source(func):
                conflicting.add(name)
            merged.setdefault(name, func)
    return MappingProxyType(merged), frozenset(conflicting)


def _law_source(func: TransitionFunction) -> object:
    """The raw user law behind a processed transition, for provenance comparison.

    A single BARE (coarse) state law carried by several targets is canonicalized
    into one cell per target, and `_rename_params_to_qnames` then renames each with
    its own qualified parameter names — producing DISTINCT wrapper objects that are
    nonetheless the same user law. `rename_arguments` sets `__wrapped__`, so
    `inspect.unwrap` recovers that shared source; two wrappers of one coarse law
    unwrap to the same object, while genuinely different per-target laws unwrap to
    different ones. Comparing processed wrapper identity (`is`) instead would flag
    the coarse-broadcast case as a spurious target-dependent conflict.
    """
    return inspect.unwrap(func)


def _get_U_and_F(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction] = (
        MappingProxyType({})
    ),
    conflicting_deterministic_transition_names: frozenset[
        TransitionFunctionName
    ] = frozenset(),
    stochastic_transition_names: frozenset[TransitionFunctionName] = frozenset(),
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Get the instantaneous utility and feasibility function.

    Note:
    -----
    U may depend on all kinds of other functions (taxes, transfers, ...), which will be
    executed if they matter for the value of U.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        deterministic_transitions: Mapping of `next_<state>` names to deterministic
            own-regime transition functions, made available so within-period utility
            or feasibility that reads a chosen next state (the NEGM service-flow
            `next_<durable>`, or a budget constraint reading it) resolves it from the
            current states and actions. Pruned away when unread, so the grid-search
            path is unchanged.
        conflicting_deterministic_transition_names: Frozenset of `next_<state>`
            names whose deterministic law differs across target bundles. A model is
            rejected if any of them is read by the within-period decision (utility
            or feasibility), because the merged law would disagree with the
            simulate state-update.

    Returns:
        The instantaneous utility and feasibility function.

    """
    combined = {
        "feasibility": _get_feasibility(
            functions=functions,
            constraints=constraints,
            deterministic_transitions=deterministic_transitions,
        ),
        **dict(deterministic_transitions),
        **{k: v for k, v in functions.items() if k != "H"},
    }
    _fail_if_conflicting_transition_is_read(
        combined=combined,
        targets=["utility", "feasibility"],
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
    )
    _fail_if_stochastic_transition_is_read(
        combined=combined,
        targets=["utility", "feasibility"],
        stochastic_transition_names=stochastic_transition_names,
    )
    return concatenate_functions(
        functions=combined,
        targets=["utility", "feasibility"],
        enforce_signature=False,
        set_annotations=True,
    )


def _fail_if_conflicting_transition_is_read(
    *,
    combined: Mapping[str, Callable[..., Any]],
    targets: list[str],
    conflicting_deterministic_transition_names: frozenset[TransitionFunctionName],
) -> None:
    """Reject a model whose decision reads a target-dependent `next_<state>` law.

    A `next_<state>` whose deterministic law differs across target bundles is
    merged down to one implementation; binding it into the decision DAG while the
    simulate state-update uses the per-target law produces a silent disagreement.
    Raise naming each such state actually read by `targets`.

    Args:
        combined: Mapping of function names to the functions assembled for the
            decision DAG.
        targets: List of target function names the decision evaluates.
        conflicting_deterministic_transition_names: Frozenset of `next_<state>`
            names with non-identical implementations across target bundles.
    """
    if not conflicting_deterministic_transition_names:
        return
    read_names = get_ancestors(combined, targets, include_targets=True)
    offending = sorted(conflicting_deterministic_transition_names & read_names)
    if offending:
        names = ", ".join(offending)
        msg = (
            "Within-period utility or feasibility reads a target-dependent "
            f"deterministic state law ({names}), but its implementation differs "
            "across target regimes. The decision DAG would bind one target's law "
            "while the simulate state-update uses the right one, so they would "
            "disagree silently. Make the law identical across all targets that "
            "carry the state, or stop reading the chosen next state in the "
            "within-period utility/feasibility."
        )
        raise ValueError(msg)


def _fail_if_stochastic_transition_is_read(
    *,
    combined: Mapping[str, Callable[..., Any]],
    targets: list[str],
    stochastic_transition_names: frozenset[TransitionFunctionName],
) -> None:
    """Reject a decision that reads an unrealised stochastic next state.

    A within-period utility or feasibility cannot read a `next_<state>` that is
    stochastic in this phase: its value is not known when the action is chosen,
    so `_get_deterministic_transitions` deliberately omits it from the flow DAG.
    `dags` then leaves that `next_<state>` an unresolved external argument of the
    decision, which fails much later with a confusing missing-argument error
    (and only in the phase where the law is stochastic). Fail early and clearly,
    naming each such state actually read by `targets`.

    Mixed stochasticity makes the phase matter: a state that is deterministic in
    one phase and stochastic in the other is readable in the deterministic phase
    and rejected here in the stochastic one -- so `stochastic_transition_names`
    is the *flow phase's* set, not a phase-invariant one.

    Args:
        combined: Mapping of function names assembled for the decision DAG.
        targets: The decision target names (`utility`, `feasibility`).
        stochastic_transition_names: `next_<state>` names stochastic in the flow
            phase.
    """
    if not stochastic_transition_names:
        return
    read_names = get_ancestors(combined, targets, include_targets=True)
    offending = sorted(stochastic_transition_names & read_names)
    if offending:
        names = ", ".join(offending)
        msg = (
            "Within-period utility or feasibility reads a stochastic state "
            f"transition ({names}). The value of an unrealised stochastic next "
            "state is not known when the action is chosen, so it cannot enter "
            "the within-period decision. Read the CURRENT state instead, or make "
            "this transition deterministic in the phase where utility or "
            "feasibility reads it."
        )
        raise ValueError(msg)


def _get_feasibility(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction] = (
        MappingProxyType({})
    ),
) -> ConstraintFunction:
    """Create a function that combines all constraint functions into a single one.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        deterministic_transitions: Mapping of `next_<state>` names to deterministic
            transition functions, so a constraint reading a chosen next state (the
            NEGM budget constraint reading `next_<durable>`) resolves it. Pruned when
            unread.

    Returns:
        The combined constraint function (feasibility).

    """
    if constraints:
        combined_constraint = concatenate_functions(
            functions=dict(deterministic_transitions)
            | dict(constraints)
            | dict(functions),
            targets=list(constraints),
            aggregator=jnp.logical_and,
            aggregator_return_type="Feasibility",
            set_annotations=True,
        )

    else:

        def combined_constraint() -> bool:
            """Dummy feasibility function that always returns True."""
            return True

    return cast("ConstraintFunction", combined_constraint)
