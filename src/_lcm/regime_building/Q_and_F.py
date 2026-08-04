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
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    co_map_state_names: tuple[StateName, ...] = (),
    certainty_equivalent: CertaintyEquivalent | None = None,
    continuation_functions: EconFunctionsMapping | None = None,
    flow_transitions: TransitionFunctionsMapping | None = None,
    flow_stochastic_transition_names: frozenset[TransitionFunctionName] | None = None,
    next_state_names: frozenset[TransitionFunctionName] = frozenset(),
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
        period_targets: Carry targets — reachable, active next period, and
            carrying at least one state, so their continuation is read at the
            next states their laws produce.
        scalar_targets: Graph targets active next period that carry no state.
            Their value function is rank-zero, so it enters `E[V]` as a single
            degenerate lottery node weighted only by the regime transition
            probability.
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
        next_state_names=next_state_names,
    )
    compute_E_next_V, continuation_deps = _get_compute_E_next_V(
        # `continuation_pool`, NOT `functions`: the continuation is priced under
        # the perceived (solve-phase) law, helpers included. In the solve phase the
        # two are the same object; only simulate passes them apart.
        functions=continuation_pool,
        period_targets=period_targets,
        scalar_targets=scalar_targets,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
    )
    _build_H_kwargs = _get_build_H_kwargs(functions)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[U_and_F, *continuation_deps],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
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
        U_arr, F_arr = U_and_F(**states_actions_params)
        E_next_V, _ = compute_E_next_V(
            next_regime_to_V_arr=next_regime_to_V_arr,
            zero=jnp.zeros_like(U_arr),
            states_actions_params=states_actions_params,
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
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    certainty_equivalent: CertaintyEquivalent | None = None,
    next_state_names: frozenset[TransitionFunctionName] = frozenset(),
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
        period_targets: Carry targets — reachable, active next period, and
            carrying at least one state.
        scalar_targets: Graph stateless targets active next period, whose
            rank-zero value enters `E[V]` weighted only by the regime transition
            probability. Must match what `get_Q_and_F` was built with, or the
            diagnostics disagree with the solve they explain.
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
        next_state_names=next_state_names,
    )
    compute_E_next_V, continuation_deps = _get_compute_E_next_V(
        functions=functions,
        period_targets=period_targets,
        scalar_targets=scalar_targets,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        certainty_equivalent=certainty_equivalent,
    )
    _build_H_kwargs = _get_build_H_kwargs(functions)

    arg_names_of_compute_intermediates = _get_arg_names_of_Q_and_F(
        deps=[U_and_F, *continuation_deps],
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
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[
        FloatND, FloatND, FloatND, FloatND, MappingProxyType[RegimeName, FloatND]
    ]:
        """Compute all Q_and_F intermediates."""
        U_arr, F_arr = U_and_F(**states_actions_params)
        E_next_V, active_regime_probs = compute_E_next_V(
            next_regime_to_V_arr=next_regime_to_V_arr,
            zero=jnp.zeros_like(U_arr),
            states_actions_params=states_actions_params,
        )

        Q_arr = functions["H"](
            utility=U_arr,
            E_next_V=E_next_V,
            **_build_H_kwargs(states_actions_params),
        )

        return U_arr, F_arr, E_next_V, Q_arr, active_regime_probs

    return compute_intermediates


def get_Q_and_F_terminal(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    next_state_names: frozenset[TransitionFunctionName] = frozenset(),
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
    U_and_F = _get_U_and_F(
        functions=functions,
        constraints=constraints,
        next_state_names=next_state_names,
    )

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
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],  # noqa: ARG001
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


def partition_continuation_targets(
    *,
    targets: tuple[RegimeName, ...],
    regime_to_v_interpolation_info: Mapping[RegimeName, VInterpolationInfo],
) -> tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]]:
    """Partition canonical graph targets into stateful and stateless tuples.

    Membership comes entirely from `targets`. Interpolation metadata classifies how
    each continuation is read without adding or removing graph edges.

    Args:
        targets: Canonical graph targets for one source and period.
        regime_to_v_interpolation_info: Mapping of regime names to interpolation
            metadata whose state names determine the continuation representation.

    Returns:
        Tuple of `(stateful_targets, scalar_targets)` preserving graph order.

    """
    stateful_targets = tuple(
        target
        for target in targets
        if regime_to_v_interpolation_info[target].state_names
    )
    scalar_targets = tuple(
        target
        for target in targets
        if not regime_to_v_interpolation_info[target].state_names
    )
    return stateful_targets, scalar_targets


def _get_compute_E_next_V(
    *,
    functions: EconFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...] = (),
) -> tuple[
    Callable[..., tuple[FloatND, MappingProxyType[RegimeName, FloatND]]],
    tuple[Callable[..., Any], ...],
]:
    """Build the closure that aggregates next period's value into `E[V']`.

    The single continuation-aggregation site of the engine: both the Bellman
    `Q` and the NaN diagnostics call the closure this returns, so they cannot
    disagree. The continuation is a lottery over the stochastic nodes of every
    reachable target regime, weighted by that target's regime-transition
    probability:

    - Without a certainty equivalent, it is aggregated as the linear
      expectation `Σ_r p_r · E_w[V'_r]`.
    - With one, the whole joint lottery is handed to
      `CertaintyEquivalent.aggregate` in one piece, which applies the
      transform before every expectation and inverts exactly once. Flattening
      before aggregating is what lets `PowerMean` anchor the transform, which
      a per-target `transform -> reduce -> inverse` decomposition cannot.

    A target carrying no state joins that same lottery as a single degenerate
    node, so it is transformed with every other node rather than on its own.

    Args:
        functions: Immutable mapping of function names to internal user functions.
            This is the CONTINUATION pool, not the flow pool: every use of it here
            builds a next-period object, which is priced under the perceived
            (solve-phase) law. Callers pass `continuation_pool`. Do not add a use
            that needs the flow pool without taking it as its own argument.
        period_targets: Carry targets whose continuation is read at the next
            states their laws produce.
        scalar_targets: Targets carrying no state, whose rank-zero value enters
            as one degenerate lottery node.
        transitions: Immutable mapping of transition names to transition functions.
        stochastic_transition_names: Frozenset of stochastic transition function names.
        compute_regime_transition_probs: Regime transition probability function
            for solve.
        regime_to_v_interpolation_info: Immutable mapping of regime names to
            V-interpolation info.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.
        co_map_state_names: Tuple of state names co-mapped with the continuation V.

    Returns:
        Tuple of the closure returning `(E_next_V, active_regime_probs)` and the
        dependencies whose arguments must enter the calling closure's signature.

    """
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[RegimeName, frozenset[str]] = {}
    next_V_has_stochastic_states: dict[RegimeName, bool] = {}

    for target_regime_name in period_targets:
        # Transitions from the current regime to the target regime
        bundle = transitions.get(target_regime_name, MappingProxyType({}))

        # Functions required to calculate the expected continuation values
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
        next_V_has_stochastic_states[target_regime_name] = bool(stochastic_variables)
        next_V[target_regime_name] = productmap(
            func=next_V_interpolator,
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    ce, ce_flat_param_names = resolve_certainty_equivalent(certainty_equivalent)

    # Co-mapped states are sliced off each `next_V_arr` leaf by the backward-
    # induction co-map, so their `next_`-prefixed coordinates are not passed to
    # the interpolator (which no longer indexes those axes).
    co_map_next_names = frozenset(f"next_{name}" for name in co_map_state_names)

    def compute_E_next_V(
        *,
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        zero: FloatND,
        states_actions_params: Mapping[str, Any],
    ) -> tuple[FloatND, MappingProxyType[RegimeName, FloatND]]:
        """Aggregate the continuation lottery into `E[V']` at one state-action point.

        Args:
            next_regime_to_V_arr: Immutable mapping of target regime names to
                next period's value function arrays.
            zero: Zero at the shape and dtype of the value being built up.
            states_actions_params: Mapping of states, actions, age, period, and
                flat regime params. Forwarded verbatim to the transition and
                probability functions, so it carries whatever the caller
                supplies — including params that never passed through
                `cast_params_to_canonical_dtypes`.

        Returns:
            Tuple of the aggregated continuation value and the regime transition
            probabilities of the reachable targets.

        """
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in (*period_targets, *scalar_targets)}
        )

        E_next_V, lottery_values, lottery_weights = _scalar_target_contribution(
            scalar_targets=scalar_targets,
            next_regime_to_V_arr=next_regime_to_V_arr,
            active_regime_probs=active_regime_probs,
            as_lottery=ce is not None,
            zero=zero,
        )
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
                    if name not in co_map_next_names
                },
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )

            if ce is None:
                # We then take the weighted average of the next value function at the
                # stochastic states to get the expected next value function.
                if next_V_has_stochastic_states[target_regime_name]:
                    next_V_expected_arr = jnp.average(
                        next_V_at_stochastic_states_arr,
                        weights=joint_next_stochastic_states_weights,
                    )
                else:
                    next_V_expected_arr = jnp.average(next_V_at_stochastic_states_arr)
                E_next_V = (
                    E_next_V
                    + active_regime_probs[target_regime_name] * next_V_expected_arr
                )
            else:
                values, node_weights = _as_lottery(
                    values=next_V_at_stochastic_states_arr,
                    weights=joint_next_stochastic_states_weights,
                    has_stochastic_states=next_V_has_stochastic_states[
                        target_regime_name
                    ],
                )
                lottery_values.append(values)
                lottery_weights.append(
                    active_regime_probs[target_regime_name] * node_weights
                )

        if ce is not None and lottery_values:
            E_next_V = ce.aggregate(
                values=jnp.concatenate(lottery_values),
                weights=jnp.concatenate(lottery_weights),
                # The params template types every certainty-equivalent
                # parameter as a float, so its runtime values are float arrays.
                params=cast(
                    "Mapping[str, FloatND]",
                    {
                        arg: states_actions_params[flat_name]
                        for arg, flat_name in ce_flat_param_names.items()
                    },
                ),
            )

        return E_next_V, active_regime_probs

    deps = (
        compute_regime_transition_probs,
        *state_transitions.values(),
        *next_stochastic_states_weights.values(),
    )
    return compute_E_next_V, deps


def _scalar_target_contribution(
    *,
    scalar_targets: tuple[RegimeName, ...],
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    active_regime_probs: Mapping[RegimeName, FloatND],
    as_lottery: bool,
    zero: FloatND,
) -> tuple[FloatND, list[FloatND], list[FloatND]]:
    """Seed the continuation accumulators with the stateless targets.

    A target carrying no state has a rank-zero value function: there is no next
    state to evaluate it at and no stochastic node to average over, so it
    contributes exactly one node whose only weight is the probability of going
    there. Under a certainty equivalent that node joins the joint lottery, so it
    is transformed together with every other target's nodes rather than on its
    own; under the linear expectation it is added straight to `E[V']`.

    Args:
        scalar_targets: Targets active next period that carry no state.
        next_regime_to_V_arr: Mapping of target regime names to next period's
            value function arrays.
        active_regime_probs: Mapping of target regime names to their regime
            transition probabilities.
        as_lottery: Whether a nonlinear certainty equivalent aggregates the
            continuation, so the nodes must be handed over unaggregated.
        zero: Zero at the shape and dtype of the value being built up.

    Returns:
        Tuple of the seeded `E[V']`, the lottery values, and their weights.

    """
    E_next_V = zero
    values: list[FloatND] = []
    weights: list[FloatND] = []
    for target_regime_name in scalar_targets:
        scalar_V = next_regime_to_V_arr[target_regime_name]
        prob = active_regime_probs[target_regime_name]
        if as_lottery:
            node = jnp.ravel(scalar_V)
            values.append(node)
            weights.append(prob * jnp.ones_like(node))
        else:
            E_next_V = E_next_V + prob * scalar_V
    return E_next_V, values, weights


def _as_lottery(
    *,
    values: FloatND,
    weights: FloatND,
    has_stochastic_states: bool,
) -> tuple[Float1D, Float1D]:
    """Flatten one target regime's continuation into a unit-mass lottery.

    Args:
        values: Next period's value at this target's stochastic nodes.
        weights: Joint weights over those nodes; ignored when the target has
            no stochastic states.
        has_stochastic_states: Whether the target's transition draws stochastic
            states.

    Returns:
        Tuple of the flattened values and their probabilities, which sum to one.

    """
    flat_values = jnp.ravel(values)
    if has_stochastic_states:
        flat_weights = jnp.ravel(weights)
        return flat_values, flat_weights / jnp.sum(flat_weights)
    uniform = jnp.full(
        flat_values.shape, 1.0 / flat_values.size, dtype=flat_values.dtype
    )
    return flat_values, uniform


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

    Non-identity is tested by object identity (`is not`), not structural
    equality. This is a conservative proxy that relies on the canonicalization
    pipeline installing the *same* function object for a target-independent
    own-regime within-period law across every bundle: a shared reference is
    correctly seen as non-conflicting, and a distinct object genuinely signals a
    different target's law. Two behaviourally-equal but distinct objects would be
    over-reported as conflicting — harmless, since the conflict set only matters
    for names the decision evaluation actually reads.

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
            if name in merged and _law_sources_differ(merged[name], func):
                conflicting.add(name)
            merged.setdefault(name, func)
    return MappingProxyType(merged), frozenset(conflicting)


# Attribute stamped by `_rename_params_to_qnames` onto an engine-renamed
# transition cell as `(user_law, qualified_param_location)`. See `_law_sources_differ`.
LAW_SOURCE_ATTR = "_lcm_law_source"


def _law_sources_differ(a: TransitionFunction, b: TransitionFunction) -> bool:
    """Whether two processed cells of one `next_<state>` name wrap different user laws.

    Compared WITHOUT invoking user-defined equality: the base user law is compared by
    object IDENTITY (`is`) and the parameter LOCATION by string equality. A user law
    may be an array-backed callable whose `==`/`!=` builds an array or raises, so a
    value comparison of the whole token is unsafe (an array-backed callable's `!=`
    yields a non-bool). Identity on the base plus string equality on the location is
    the exact distinction the token encodes and touches no user `__eq__`.

    The engine STAMPS every parameterized cell it renames with
    `(user_law, qualified_param_location)`:

    - A COARSE law binds ONE shared parameter branch across its target cells, so every
      cell carries the SAME base object and the SAME (bare) location — the cells merge.
    - A PER-TARGET dict binds a TARGET-QUALIFIED branch per cell, so cells carry
      DIFFERENT locations even when the user reuses the SAME callable object across
      targets — the reused-callable case raw identity missed.

    A parameter-free law receives no engine wrapper (and no stamp): its cell's own
    object identity separates one coarse law (the same object broadcast to every
    target) from distinct per-target laws, and a reused parameter-free callable is
    genuinely identical (no parameter can differ), so shared identity is correct there.
    When either cell is unstamped, fall back to object identity of the cells themselves.
    """
    src_a = getattr(a, LAW_SOURCE_ATTR, None)
    src_b = getattr(b, LAW_SOURCE_ATTR, None)
    if src_a is None or src_b is None:
        # Engine-generated identity laws (`fixed_transition`) are parameter-free and
        # carry no stamp, but canonicalization rebuilds a FRESH `_IdentityTransition`
        # per target cell, so object identity would wrongly flag two identities for the
        # SAME state as differing. They are extensionally equal (next value = the same
        # current state), so merge them. Duck-typed on `_is_auto_identity` to avoid an
        # import cycle.
        if _both_auto_identity_for_same_state(a, b):
            return False
        return a is not b
    base_a, location_a = src_a
    base_b, location_b = src_b
    return base_a is not base_b or location_a != location_b


def _both_auto_identity_for_same_state(
    a: TransitionFunction, b: TransitionFunction
) -> bool:
    """Whether `a` and `b` are engine identity laws for the same state (and annotation).

    `_IdentityTransition` (backing `lcm.fixed_transition`) sets `_is_auto_identity` and
    `_state_name`; the collector rebuilds one per target with the state's grid-matched
    annotation. Two such laws for the same state compute the identical next value, so a
    within-period read of them must NOT be treated as a target-dependent conflict.
    """
    if not (
        getattr(a, "_is_auto_identity", False)
        and getattr(b, "_is_auto_identity", False)
    ):
        return False
    same_state = getattr(a, "_state_name", object()) == getattr(
        b, "_state_name", object()
    )
    ann_a = getattr(a, "__annotations__", {}).get("return")
    ann_b = getattr(b, "__annotations__", {}).get("return")
    return same_state and ann_a == ann_b


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
    next_state_names: frozenset[TransitionFunctionName] = frozenset(),
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
    # Run the conflict/stochastic guards on the RAW decision graph -- utility plus
    # the INDIVIDUAL constraints -- BEFORE `_get_feasibility` concatenates them.
    # `_get_feasibility` resolves a chosen `next_<state>` *into* the compiled
    # feasibility callable, erasing it from that callable's external ancestry; a
    # conflict or stochastic read reached only through a constraint would then be
    # invisible to a guard that inspects the compiled `feasibility`. The raw graph
    # keeps every `next_<state>` visible in the constraints' own ancestry.
    raw_decision_graph = {
        **dict(deterministic_transitions),
        **dict(constraints),
        **{k: v for k, v in functions.items() if k != "H"},
    }
    guard_targets = ["utility", *constraints]
    _fail_if_conflicting_transition_is_read(
        combined=raw_decision_graph,
        targets=guard_targets,
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
    )
    _fail_if_stochastic_transition_is_read(
        combined=raw_decision_graph,
        targets=guard_targets,
        stochastic_transition_names=stochastic_transition_names,
    )
    _fail_if_unproduced_next_state_is_read(
        combined=raw_decision_graph,
        targets=guard_targets,
        next_state_names=next_state_names,
    )
    combined = {
        "feasibility": _get_feasibility(
            functions=functions,
            constraints=constraints,
            deterministic_transitions=deterministic_transitions,
        ),
        **dict(deterministic_transitions),
        **{k: v for k, v in functions.items() if k != "H"},
    }
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


def _fail_if_unproduced_next_state_is_read(
    *,
    combined: Mapping[str, Callable[..., Any]],
    targets: list[str],
    next_state_names: frozenset[TransitionFunctionName],
) -> None:
    """Reject a within-period read of a `next_<state>` with no producer this phase.

    A within-period utility or feasibility may legitimately read a chosen deterministic
    next state (the NEGM service-flow `next_<durable>`, or a budget constraint reading
    it). That read resolves only if THIS phase's flow supplies a producer for the
    unqualified `next_<state>` — i.e. some reachable target carries the state and
    contributes its law to the merged deterministic transitions
    (`_get_deterministic_transitions`). When no reachable target carries it in this
    phase (a target-only handover whose carrier does not grid it here, or a carried
    state imputed rather than gridded in the solve phase), the name is left an
    unresolved external argument that fails much later with a cryptic missing-argument
    error — and only in the phase that lacks the producer. Fail early, naming each such
    state.

    Producer availability is read off `combined`: a produced `next_<state>` is a KEY
    (its merged transition function); a read-but-unproduced one is an ancestor that is
    not a key. Stochastic next-states are excluded from the flow and guarded separately
    (`_fail_if_stochastic_transition_is_read`, run first), so any remaining unproduced
    `next_*` ancestor is a genuine deterministic no-producer read.

    Being phase-local — it runs on each phase's own flow DAG — this catches a
    simulate-only read whose producer exists only in the solve phase, and does NOT
    over-reject a read whose producer a reachable ordinary target does supply.

    A `next_<state>` node exists only for a name in `next_state_names` — the engine's
    declared transition-output names for this regime (own or target-only states). A user
    may LEGALLY name a current state or action `next_stock` (only FUNCTION names reserve
    the `next_` prefix); such a variable is an ordinary decision input, not a next-state
    node — its own transition is `next_next_stock` — so it must not be flagged. Hence
    the offending set intersects the declared next-state names, not a raw string prefix.

    Args:
        combined: The raw decision graph — deterministic transitions (the producers),
            constraints, and functions — keyed by name.
        targets: The decision target names the graph evaluates (`utility` and the
            individual constraints).
        next_state_names: The engine's declared next-state node names for this regime
            (`next_<state>` for every own and target-only state). Only these can be a
            genuine unproduced next-state read.
    """
    read_names = get_ancestors(combined, targets, include_targets=True)
    offending = sorted(
        name for name in read_names & next_state_names if name not in combined
    )
    if offending:
        names = ", ".join(offending)
        msg = (
            f"Within-period utility or feasibility reads the next value of state(s) "
            f"({names}), but this phase's flow has no producer for them. A "
            f"`next_<state>` is produced only where a reachable target carries the "
            f"state in this phase; a target-only handover whose carrier does not grid "
            f"it here — or a carried state imputed rather than gridded in the solve "
            f"phase — leaves the read unsupplied. Grid the state in a reachable target "
            f"(or in this regime) if the decision genuinely depends on its next value, "
            f"or remove the `next_<state>` read from the within-period function."
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
