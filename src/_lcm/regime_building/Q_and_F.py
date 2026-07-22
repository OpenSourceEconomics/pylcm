from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags import concatenate_functions, get_ancestors, with_signature

from _lcm.certainty_equivalent import CertaintyEquivalent, resolve_certainty_equivalent
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.h_dag import _get_build_H_kwargs
from _lcm.regime_building.next_state import (
    get_next_state_function_for_solution,
    get_next_stochastic_weights_function,
)
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.regime_building.zero_safe import zero_safe_average, zero_safe_weighted_term
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


def _sum_regime_mixture(
    mixture_terms: list[tuple[RegimeName, FloatND, FloatND]], *, like: FloatND
) -> FloatND:
    """Reduce the regime mixture ``E[V']=Σ p_r·V_r`` as ONE zero-safe contraction.

    ``mixture_terms`` is a list of ``(target_name, prob_r, expected_V_r)`` — the
    UNMULTIPLIED per-target probability and expected continuation. The per-target
    probabilities and continuations are stacked along a new leading (target) axis and
    multiplied ONCE inside a single ``zero_safe_weighted_term``; the resulting
    per-target contributions ``p_r·V_r`` are then reduced by a VALUE-ORDERED
    ``jnp.sum`` — the contributions are ``jnp.sort``-ed along the target axis
    before the sum. Two
    properties this buys over the earlier sequential left-fold
    ``E = 0; for r: E += zero_safe_weighted_term(p_r, V_r)`` (round-8/round-10 external
    re-review, both MEASURED reproduce-first):

    - **Accuracy.** Stacking the OPERANDS and multiplying once inside the reduction —
      NOT stacking the already-formed products — lands on the exact-policy side of the
      round-8 pinned 5-target fixture (``> alternative`` bits ...843) where the
      left-fold and ``jnp.sum(jnp.stack(products))`` both land on the wrong side
      (bits ...842). It
      is still NOT correctly-rounded: under cancellation (Σ|p_r·V_r| ≫ |Σ p_r·V_r|) the
      error scales with Σ|p_r·V_r|, hundreds of result-ULP, so a genuine knife-edge
      argmax can still resolve either way. Deterministic resolution AT a genuine
      knife-edge would need compensated/exact summation, which is not implemented (a
      value-sorted Neumaier compensated sum WAS measured and, on the round-10
      counterexample, landed on the WRONG side of the competing action while the plain
      value-sorted reduction landed exact-side, so it was NOT adopted).
    - **Reproducibility (label-independence).** The reduction ORDER is a deterministic
      function of the contribution VALUES — economically meaningful — and NEVER of the
      arbitrary regime NAMES. The pre-round-10 code ``sorted(mixture_terms, key=name)``
      removed the transition-mapping ITERATION-ORDER dependence but made the float64
      summation order a function of the user's regime LABELS: a pure ALPHA-RENAMING of
      the regimes (same probabilities, same continuations, only the dict keys change)
      reordered the non-associative float64 sum and, MEASURED, moved the result across
      37 distinct outputs over the 120 name bijections of a valid 5-target float64
      mixture — reversing a non-tied household argmax on the round-10 counterexample.
      Sorting the CONTRIBUTION MULTISET (``jnp.sort`` along the target axis) makes the
      sum provably invariant to alpha-renaming: the multiset ``{p_r·V_r}`` is unchanged
      by relabeling, and the sorted order (hence the summation order and its bits) is a
      function of that multiset alone. The stacking order of ``mixture_terms`` is
      therefore irrelevant (the sort canonicalises it), so no name-sort is needed.

    Zero-mass safety is preserved (a zero ``p_r`` beside an admissible ``±inf`` V_r is
    masked to exactly 0 by ``zero_safe_weighted_term`` BEFORE the sort, so a zero-mass
    ``-inf`` contributes 0 and never survives the sort as ``-inf``). Cost: the K
    per-target contributions are materialised together and sorted along the (small)
    target axis, an O(K log K) sort on a tiny axis, rather than folded one at a time — K
    is the number of active next-period targets. ``mixture_terms`` is empty in a
    terminal period with no active target; the mixture is then exactly ``zeros_like``.
    """
    if not mixture_terms:
        return jnp.zeros_like(like)
    probs = jnp.stack([prob for _, prob, _ in mixture_terms], axis=0)
    values = jnp.stack([value for _, _, value in mixture_terms], axis=0)
    # Right-pad the probability rank to the value rank so the per-target weight
    # broadcasts over the TARGET axis (leading, axis 0) and is constant across any
    # trailing value-only axes. The collective site carries a trailing stakeholder
    # axis on the continuation (`values` is (K, *cell, S)) that the scalar regime
    # probability (K, *cell) does not: without this alignment `zero_safe_weighted_
    # term` right-aligns and weights the STAKEHOLDER axis instead of the target axis
    # -- silently reversing a household action when K==S, leaking a zero-mass -inf,
    # or raising when K!=S. A no-op at the scalar/singleton sites (equal ranks).
    if probs.ndim < values.ndim:
        probs = probs.reshape(probs.shape + (1,) * (values.ndim - probs.ndim))
    # Reduce in VALUE order, not label order. `zero_safe_weighted_term` forms the
    # zero-mass-safe per-target contributions `p_r*V_r` (masking a zero-mass `+-inf`
    # to 0); sorting them along the target axis (axis 0) before `jnp.sum` makes the
    # non-associative float64 reduction order a deterministic function of the
    # contribution multiset -- provably invariant to an economically-inert
    # alpha-renaming of the regimes -- where the previous name-sort made the bits
    # (and a non-tied argmax) depend on the arbitrary regime labels. See the docstring.
    contributions = zero_safe_weighted_term(probs, values)
    return jnp.sum(jnp.sort(contributions, axis=0), axis=0)


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
        next_state_names=next_state_names,
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
        # COLLECTIVE-REGIMES (E2): F_arr is built here, before and independently
        # of Q (it never reads E_next_V). A value-aware mask cannot stay here:
        # it needs per-stakeholder Q^s, so E2 splits this into (i) build the
        # state-independent F here, (ii) compute Q^s, (iii) `mask = F ∧ g(...)`
        # applied in max_Q_over_a. This site also returns the explicit dissolution
        # flag D = 1[mask empty], distinct from a numeric -inf. See design doc
        # §2 (E2) / §3.
        U_arr, F_arr = U_and_F(**states_actions_params)
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in period_targets}
        )

        mixture_terms: list[tuple[RegimeName, FloatND, FloatND]] = []
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
            # Zero-safe: a zero-probability stochastic node next to an
            # admissible on-path -inf must not turn the average into a nan.
            next_V_expected_arr = zero_safe_average(
                next_V_at_stochastic_states_arr,
                weights=joint_next_stochastic_states_weights,
            )
            # Collect the UNMULTIPLIED (prob, expected-V) per target; the mixture is
            # reduced once by `_sum_regime_mixture` (stack operands, one zero-safe
            # contraction, canonical target order) — see that helper for why this is
            # more accurate and order-independent than a sequential left-fold (round-8).
            mixture_terms.append(
                (
                    target_regime_name,
                    active_regime_probs[target_regime_name],
                    next_V_expected_arr,
                )
            )
        E_next_V = _sum_regime_mixture(mixture_terms, like=U_arr)

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
        next_state_names=next_state_names,
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

        mixture_terms: list[tuple[RegimeName, FloatND, FloatND]] = []
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
            # Zero-safe, mirroring `get_Q_and_F` above: see the guards there.
            contribution = zero_safe_average(next_V_stoch, weights=joint)
            mixture_terms.append(
                (
                    target_regime_name,
                    active_regime_probs[target_regime_name],
                    contribution,
                )
            )
        E_next_V = _sum_regime_mixture(mixture_terms, like=U_arr)

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


def get_Q_and_F_terminal_collective(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    stakeholders: tuple[str, ...],
) -> QAndFFunction:
    """Terminal (Q, F) for a collective regime — stacked per-stakeholder U + shared F.

    COLLECTIVE-REGIMES (E1). Separate from `get_Q_and_F_terminal` so the singleton
    terminal path (shared with the simulate / compute-intermediates machinery) is
    byte-identical; this builder is used only at the collective solve site.

    Builds one `U^s`-and-`F` closure per stakeholder from its own `utility_<s>`
    DAG target (feasibility is regime-level, so it is identical across
    stakeholders — the first one is kept). The returned `Q_and_F` stacks the
    per-stakeholder utilities on a trailing stakeholder axis: for a scalar
    (state, action) cell it returns `U` of shape `(n_stakeholders,)` and a scalar
    `F`. After the action product-map in `get_max_Q_over_a`, `U` has shape
    `(*action_axes, n_stakeholders)` and `F` `(*action_axes,)`; the stakeholder
    branch there splits `U` by stakeholder and calls `collective_readout`.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions;
            carries `utility_<s>` for each stakeholder in place of `utility`.
        constraints: Immutable mapping of constraint names to internal user functions.
        stakeholders: Ordered stakeholder names; fixes the trailing-axis order.

    Returns:
        A function computing the stacked per-stakeholder utilities (Q) and the
        shared feasibility mask (F) for a terminal collective period.

    """
    U_and_F_by_stakeholder = {
        stakeholder: _get_U_and_F(
            functions=functions,
            constraints=constraints,
            utility_name=f"utility_{stakeholder}",
        )
        for stakeholder in stakeholders
    }

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=list(U_and_F_by_stakeholder.values()),
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
        """Stacked per-stakeholder utilities and the shared feasibility mask.

        Args:
            next_regime_to_V_arr: Unused in a terminal period; accepted so solve
                treats all periods uniformly.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple of the stacked per-stakeholder utility array (trailing
            stakeholder axis) and the shared feasibility mask.

        """
        U_arrays: list[FloatND] = []
        F_arr: BoolND | None = None
        for u_and_f in U_and_F_by_stakeholder.values():
            U_s, F_arr = u_and_f(**states_actions_params)
            U_arrays.append(jnp.asarray(U_s))
        U_stack = jnp.stack(U_arrays, axis=-1)
        return U_stack, jnp.asarray(F_arr)

    return Q_and_F


# COLLECTIVE-REGIMES (E2): the name under which the mapping of same-period
# reference regimes to their current-period V arrays enters the kernel
# signature. Only regimes declaring `same_period_refs` carry it.
SAME_PERIOD_V_ARG = "same_period_regime_to_V_arr"

# COLLECTIVE-REGIMES (E2, F4 fix): the name under which the mapping of
# same-period reference regimes to THEIR OWN flat params enters the kernel
# signature, alongside `SAME_PERIOD_V_ARG`. Carried by every reader built by
# `_build_same_period_ref_reader`, and hence by every regime that reads another
# regime's same-period V.
#
# A reference reader interpolates the REFERENCE regime's V over the REFERENCE
# regime's grid, so the interpolator's runtime grid helpers (an
# `IrregSpacedGrid(pass_points_at_runtime=True)` reference state's `points`, via
# `V._get_coordinate_finder`) are parameters of the REFERENCE regime: they live
# in `flat_params[ref.regime]`, never in the READING regime's own namespace.
# Before this argument existed the reader exposed those helpers as extra outer
# arguments named after the PREFIXED coordinate variable
# (`__same_period_ref__x__points`), which no caller supplies and no params
# template ever emits (`_lcm.params.regime_template._add_runtime_grid_params`
# emits `x__points`, in the reference regime's own template): all four consumers
# of `_build_same_period_ref_reader` — ordinary E2 same-period refs, solve-side
# gate refs, solve-side leg-fallback value readers, and simulate-side gate refs
# — raised a missing-argument error the moment a reference regime declared a
# runtime irregular grid. Coordinate VARIABLES stay prefixed (internal wiring
# that must not collide with the reading regime's own state names); PARAMETER
# qnames are separated from them and resolved against the reference regime's
# explicit namespace through this mapping instead.
SAME_PERIOD_PARAMS_ARG = "same_period_regime_to_params"

# Internal argument names of the same-period reference interpolation; never
# surfaced in the kernel signature.
_REF_STATE_PREFIX = "__same_period_ref__"
_REF_V_ARR_NAME = "__same_period_ref_V_arr__"


@dataclass(frozen=True, kw_only=True)
class ResolvedSamePeriodRef:
    """Engine-side form of a user `SamePeriodRef`, resolved at model processing.

    COLLECTIVE-REGIMES (E2). The user declaration names a stakeholder; the
    engine resolves it to the index on the reference regime's trailing
    stakeholder axis (`None` for a singleton reference, whose V has no such
    axis).
    """

    regime: RegimeName
    """Name of the reference regime whose same-period V is read."""

    projection: Mapping[StateName, Callable[..., Any]]
    """Per-reference-state projection functions (user vocabulary, DAG-resolved)."""

    stakeholder_index: int | None
    """Index into the reference V's trailing stakeholder axis, or `None`."""


def _build_same_period_ref_reader(
    *,
    ref: ResolvedSamePeriodRef,
    v_interpolation_info: VInterpolationInfo,
    functions: EconFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction],
    conflicting_deterministic_transition_names: frozenset[
        TransitionFunctionName
    ] = frozenset(),
) -> Callable[..., FloatND]:
    """Build the reader of one same-period reference value at a (state, action) cell.

    COLLECTIVE-REGIMES (E2). Each projection entry is concatenated with the
    regime's function DAG (so it may read states, actions, helper functions,
    and the merged deterministic `next_<state>` laws), producing one coordinate
    per reference state; the reference regime's CURRENT-period V array — passed
    per solve step under `SAME_PERIOD_V_ARG` — is then interpolated at those
    coordinates with the ordinary V-interpolation machinery
    (`get_V_interpolator`), sliced to the named stakeholder first when the
    reference is collective. The returned callable's signature carries only
    user-level names (states / actions / params reached by the projections,
    plus `SAME_PERIOD_V_ARG` and `SAME_PERIOD_PARAMS_ARG`), so the kernel
    signature stays clean.

    The projections are expressed in the READING regime's vocabulary and their
    free parameters are bound from the reading regime's own params (every caller
    passes exactly that); the INTERPOLATION helpers instead belong to the
    REFERENCE regime's grid and are resolved against `SAME_PERIOD_PARAMS_ARG`
    (F4 fix — see that constant). The two provenances are separated here rather
    than merged into one namespace, because a runtime irregular grid names its
    helper after the STATE alone (`x__points`), so a reading regime that happens
    to declare an identically named state would otherwise silently supply its
    OWN grid points for the reference regime's interpolation.

    A projection produces a genuine VALUE for every reference state
    (interpolation-worthy, possibly off-grid) — unlike the ordinary
    continuation-value path, which always feeds a process axis its exact
    on-grid Markov-chain index. When the reference regime carries a
    non-folded process state (`_ContinuousStochasticProcess`, classified
    `discrete_states` for the Markov-chain solve path but read here as a
    genuine value), `get_V_interpolator`'s process-aware mode
    (`interpolate_process_axes=True`) is used so that axis is linearly
    interpolated instead of integer-looked-up; a reference regime without a
    process state is unaffected (`interpolate_process_axes=False`, the
    ordinary path, byte-identical).

    Args:
        ref: Resolved same-period reference declaration.
        v_interpolation_info: V-interpolation info of the reference regime.
        functions: Immutable mapping of function names to internal user
            functions.
        deterministic_transitions: Mapping of `next_<state>` names to merged
            deterministic own-regime transition functions, made available to
            the projection like an ordinary same-period-ref read.
        conflicting_deterministic_transition_names: Frozenset of `next_<state>`
            names whose deterministic law differs across target bundles (see
            `_get_deterministic_transitions`). Rejected exactly like an
            ordinary utility/feasibility read
            (`_fail_if_conflicting_transition_is_read`) if a projection
            actually reads one of them, since the merged law would silently
            disagree with the simulate state-update there too.
    """
    _reference_has_process_axis = any(
        isinstance(grid, _ContinuousStochasticProcess)
        for grid in v_interpolation_info.discrete_states.values()
    )
    interpolator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        state_prefix=_REF_STATE_PREFIX,
        V_arr_name=_REF_V_ARR_NAME,
        interpolate_process_axes=_reference_has_process_axis,
    )
    dag_pool = {
        **dict(deterministic_transitions),
        **{k: v for k, v in functions.items() if k != "H"},
    }
    _fail_if_conflicting_transition_is_read(
        combined={
            **dag_pool,
            **{
                f"{_REF_STATE_PREFIX}{state_name}": projection
                for state_name, projection in ref.projection.items()
            },
        },
        targets=[
            f"{_REF_STATE_PREFIX}{state_name}"
            for state_name in v_interpolation_info.state_names
        ],
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
    )
    projection_funcs: dict[StateName, Callable[..., FloatND]] = {}
    projection_args: dict[StateName, tuple[str, ...]] = {}
    for state_name in v_interpolation_info.state_names:
        target = f"{_REF_STATE_PREFIX}{state_name}"
        projection_funcs[state_name] = concatenate_functions(
            functions={**dag_pool, target: ref.projection[state_name]},
            targets=target,
            enforce_signature=False,
            set_annotations=True,
        )
        projection_args[state_name] = tuple(
            get_union_of_args([projection_funcs[state_name]])
        )
    coordinate_names = {
        f"{_REF_STATE_PREFIX}{state}" for state in v_interpolation_info.state_names
    }
    # Extra interpolator inputs beyond the coordinates and the V array (e.g.
    # runtime-supplied irregular-grid points). F4 fix: these are the REFERENCE
    # regime's own parameters, so they are NOT exposed as outer arguments of
    # this reader (the reading regime's caller has no such param, and the
    # prefixed name they carried was unsatisfiable by anyone) — they are looked
    # up per call in `SAME_PERIOD_PARAMS_ARG[ref.regime]` under their qname in
    # the reference regime's OWN namespace.
    interpolator_extra_qnames = _reference_interpolator_param_qnames(
        extra_args=get_union_of_args([interpolator])
        - coordinate_names
        - {_REF_V_ARR_NAME},
        ref=ref,
    )
    arg_names = sorted(
        {arg for args in projection_args.values() for arg in args}
        | {SAME_PERIOD_V_ARG, SAME_PERIOD_PARAMS_ARG}
    )

    @with_signature(args=arg_names, return_annotation="FloatND")
    def read_reference_value(**kwargs: _ParamsLeaf) -> FloatND:
        same_period_V = cast("Mapping[RegimeName, FloatND]", kwargs[SAME_PERIOD_V_ARG])
        V_ref = same_period_V[ref.regime]
        if ref.stakeholder_index is not None:
            # A collective reference V carries a trailing stakeholder axis;
            # read the declared stakeholder's slice (state axes only remain).
            V_ref = V_ref[..., ref.stakeholder_index]
        coordinates = {
            f"{_REF_STATE_PREFIX}{state}": projection_funcs[state](
                **{arg: kwargs[arg] for arg in projection_args[state]}
            )
            for state in v_interpolation_info.state_names
        }
        return interpolator(
            **coordinates,
            **_lookup_reference_params(
                qnames=interpolator_extra_qnames,
                regime_to_params=kwargs[SAME_PERIOD_PARAMS_ARG],
                ref_regime=ref.regime,
            ),
            **{_REF_V_ARR_NAME: V_ref},
        )

    return read_reference_value


def _reference_interpolator_param_qnames(
    *,
    extra_args: set[str],
    ref: ResolvedSamePeriodRef,
) -> MappingProxyType[str, str]:
    """Map each extra interpolator input to its qname in the REFERENCE namespace.

    COLLECTIVE-REGIMES (E2, F4 fix). `get_V_interpolator` derives its runtime
    grid-helper names from the COORDINATE VARIABLE it was given
    (`_get_coordinate_finder`: `qname_from_tree_path((in_name.removeprefix(
    "next_"), "points"))`), so with `state_prefix=_REF_STATE_PREFIX` the helper
    for reference state `x` is called `__same_period_ref__x__points` while the
    reference regime's params template calls the very same quantity `x__points`.
    Stripping the coordinate prefix is exactly the inverse of the prefixing
    `get_V_interpolator` applied, and recovers the reference regime's own qname.

    Any extra input that does NOT carry the prefix cannot be attributed to a
    reference state this way; rather than bind it from an arbitrary namespace
    (the defect class this whole mechanism exists to end), fail loudly at build
    time.

    Raises:
        NotImplementedError: An interpolator input could not be attributed to a
            prefixed reference coordinate.
    """
    qnames: dict[str, str] = {}
    for arg in sorted(extra_args):
        if not arg.startswith(_REF_STATE_PREFIX):
            msg = (
                f"The same-period reference reader for regime '{ref.regime}' "
                f"needs an interpolation helper argument '{arg}' that does not "
                f"derive from a prefixed reference coordinate "
                f"('{_REF_STATE_PREFIX}...'), so pylcm cannot tell which "
                "regime's parameter namespace it belongs to. Binding it from a "
                "guessed namespace would silently read another regime's "
                "parameter; this is not supported."
            )
            raise NotImplementedError(msg)
        qnames[arg] = arg.removeprefix(_REF_STATE_PREFIX)
    return MappingProxyType(qnames)


def _lookup_reference_params(
    *,
    qnames: Mapping[str, str],
    regime_to_params: object,
    ref_regime: RegimeName,
) -> dict[str, _ParamsLeaf]:
    """Resolve a reader's interpolation helpers in the REFERENCE regime's params.

    COLLECTIVE-REGIMES (E2, F4 fix). See `SAME_PERIOD_PARAMS_ARG`.

    Raises:
        KeyError: The reference regime's params are missing from the mapping, or
            do not carry a helper the reference regime's own grid needs.
    """
    if not qnames:
        return {}
    params_per_regime = cast(
        "Mapping[RegimeName, Mapping[str, _ParamsLeaf]]", regime_to_params
    )
    if ref_regime not in params_per_regime:
        msg = (
            f"Reading regime '{ref_regime}''s same-period V requires that "
            f"regime's own params (it declares runtime grid points), but "
            f"'{ref_regime}' is missing from '{SAME_PERIOD_PARAMS_ARG}' "
            f"(present: {sorted(params_per_regime)})."
        )
        raise KeyError(msg)
    ref_params = params_per_regime[ref_regime]
    resolved: dict[str, _ParamsLeaf] = {}
    for arg, qname in qnames.items():
        if qname not in ref_params:
            msg = (
                f"Interpolating regime '{ref_regime}''s same-period V needs its "
                f"parameter '{qname}', which is not in flat_params"
                f"['{ref_regime}'] (present: {sorted(ref_params)})."
            )
            raise KeyError(msg)
        resolved[arg] = ref_params[qname]
    return resolved


def get_Q_and_F_collective(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    stakeholders: tuple[str, ...],
    co_map_state_names: tuple[StateName, ...] = (),
    value_constraints: ConstraintFunctionsMapping = MappingProxyType({}),
    same_period_refs: Mapping[str, ResolvedSamePeriodRef] = MappingProxyType({}),
) -> QAndFFunction:
    """Non-terminal (Q, F) for a collective regime — per-stakeholder continuation.

    COLLECTIVE-REGIMES (E1, slice 2). Separate from `get_Q_and_F` so the
    singleton path is byte-identical; this builder is used only at the
    collective solve site.

    Per stakeholder `s`, computes `Q^s = H(u^s, E[V'^s])` with the shared
    Bellman aggregator `H` (the default `H_linear` applies `u + beta * E[V']`
    elementwise, so every stakeholder is discounted with the SAME beta). Each
    transition target must itself be a collective regime with the identical
    `stakeholders` tuple (validated at model processing), so its
    `next_V_arr` leaf carries the trailing stakeholder axis. The continuation
    interpolates the target's V over STATE axes only: the interpolator is
    evaluated once per stakeholder on the leaf's slice `next_V_arr[..., s]` and
    the results are re-stacked on a trailing axis, so the stakeholder axis
    provably rides through the stochastic-node product-map (which stacks its
    mapped axes at the front) as the last axis. For a scalar (state, action)
    cell the returned `Q` has shape `(n_stakeholders,)` while `F` is scalar;
    after the action product-map in `get_max_Q_over_a`, `Q` is
    `(*action_axes, n_stakeholders)` and `F` `(*action_axes,)` — exactly what
    the stakeholder branch there (`collective_readout`) consumes.

    No taste shocks and no nonlinear certainty equivalent: both are rejected at
    regime construction for collective regimes.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user
            functions; carries `utility_<s>` for each stakeholder in place of
            `utility`, plus the shared `H`.
        constraints: Immutable mapping of constraint names to internal user
            functions.
        period_targets: Target regimes whose continuation enters E[V^s] this
            period (all collective with the identical stakeholder tuple).
        transitions: Immutable mapping of transition names to transition
            functions.
        stochastic_transition_names: Frozenset of stochastic transition function
            names.
        compute_regime_transition_probs: Regime transition probability function
            for solve (stakeholder-independent — per-stakeholder gates are E3').
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info (state axes only; the stakeholder axis is not
            an interpolation axis).
        stakeholders: Ordered stakeholder names; fixes the trailing-axis order.
        co_map_state_names: Tuple of state names co-mapped with the continuation
            V (see `get_Q_and_F`).
        value_constraints: Immutable mapping of value-constraint names to
            predicates (params already renamed to qnames). COLLECTIVE-REGIMES
            (E2): evaluated AFTER the per-stakeholder `Q^s`, each predicate may
            read `Q_<s>` per stakeholder, the `same_period_refs` reference
            values, and ordinary states / actions / functions / params via the
            DAG; the results are ANDed into the feasibility mask, so the
            household argmax runs over `F ∧ g(Q^s, V_ref, ...)` and an
            all-infeasible cell publishes the dissolution flag `D` downstream.
        same_period_refs: Immutable mapping of reference-value names to resolved
            same-period reference declarations. When non-empty, the returned
            `Q_and_F` carries the extra argument `SAME_PERIOD_V_ARG` — the
            mapping of reference regime names to their CURRENT-period V arrays,
            supplied per period by the solve loop (which orders the period's
            regimes so references are solved first).

    Returns:
        A function computing the stacked per-stakeholder state-action values
        (trailing stakeholder axis) and the shared feasibility mask for a
        non-terminal collective period.

    """
    deterministic_transitions, conflicting_deterministic_transition_names = (
        _get_deterministic_transitions(
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
        )
    )
    U_and_F_by_stakeholder = {
        stakeholder: _get_U_and_F(
            functions=functions,
            constraints=constraints,
            deterministic_transitions=deterministic_transitions,
            conflicting_deterministic_transition_names=(
                conflicting_deterministic_transition_names
            ),
            utility_name=f"utility_{stakeholder}",
        )
        for stakeholder in stakeholders
    }
    n_stakeholders = len(stakeholders)

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
            co_map_state_names=co_map_state_names,
        )
        next_V_extra_param_names[target_regime_name] = frozenset(
            get_union_of_args([next_V_interpolator]) - set(bundle) - {V_arr_name}
        )
        stochastic_variables = tuple(
            key for key in bundle if key in stochastic_transition_names
        )
        next_V[target_regime_name] = productmap(
            func=_get_stakeholder_sliced_interpolator(
                base_interpolator=next_V_interpolator,
                V_arr_name=V_arr_name,
                n_stakeholders=n_stakeholders,
            ),
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    _build_H_kwargs = _get_build_H_kwargs(functions)
    _co_map_next_names = frozenset(f"next_{name}" for name in co_map_state_names)

    # COLLECTIVE-REGIMES (E2): build the same-period reference readers and the
    # value-constraint evaluators once; their engine-supplied arguments —
    # `Q_<s>` and the reference-value names — are excluded from the kernel
    # signature and bound per (state, action) cell inside `Q_and_F`.
    value_constraint_machinery = _build_value_constraint_machinery(
        value_constraints=value_constraints,
        same_period_refs=same_period_refs,
        stakeholders=stakeholders,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        functions=functions,
        deterministic_transitions=deterministic_transitions,
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
    )

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[
            *list(U_and_F_by_stakeholder.values()),
            compute_regime_transition_probs,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
            *list(value_constraint_machinery.evaluators.values()),
            *list(value_constraint_machinery.reference_readers.values()),
        ],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=value_constraint_machinery.engine_supplied_names,
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: FloatND,
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[FloatND, BoolND]:
        """Per-stakeholder state-action values and the shared feasibility mask.

        Args:
            next_regime_to_V_arr: The next period's value function arrays, each
                target leaf carrying a trailing stakeholder axis.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple of the stacked per-stakeholder state-action value array
            (trailing stakeholder axis) and the shared feasibility mask.

        """
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        U_arrays: list[FloatND] = []
        F_arr: BoolND | None = None
        for u_and_f in U_and_F_by_stakeholder.values():
            U_s, F_arr = u_and_f(**states_actions_params)
            U_arrays.append(jnp.asarray(U_s))
        U_stack = jnp.stack(U_arrays, axis=-1)
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in period_targets}
        )

        mixture_terms: list[tuple[RegimeName, FloatND, FloatND]] = []
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

            extra_kw = {
                k: states_actions_params[k]
                for k in next_V_extra_param_names[target_regime_name]
            }
            # Shape (*stochastic_axes, n_stakeholders): the product-map stacks
            # the stochastic-node axes at the front, the stakeholder axis stays
            # trailing.
            next_V_at_stochastic_states_arr = next_V[target_regime_name](
                **{
                    name: val
                    for name, val in next_states.items()
                    if name not in _co_map_next_names
                },
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )

            # Per-stakeholder weighted average over the stochastic nodes only —
            # never over the trailing stakeholder axis. Zero-safe: see the
            # guards in `get_Q_and_F` above.
            next_V_expected_arr = zero_safe_average(
                next_V_at_stochastic_states_arr.reshape(-1, n_stakeholders),
                axis=0,
                weights=jnp.asarray(joint_next_stochastic_states_weights).reshape(-1),
            )
            mixture_terms.append(
                (
                    target_regime_name,
                    active_regime_probs[target_regime_name],
                    next_V_expected_arr,
                )
            )
        E_next_V = _sum_regime_mixture(mixture_terms, like=U_stack)

        # H applied on the stacked arrays is H per stakeholder: `utility` and
        # `E_next_V` share the trailing stakeholder axis and H's parameters
        # (e.g. the default `H_linear`'s discount factor) are shared across
        # stakeholders, so the elementwise aggregation is exactly
        # Q^s = H(u^s, E[V'^s], beta) with the same beta for every s.
        Q_arr = functions["H"](
            utility=U_stack,
            E_next_V=E_next_V,
            **_build_H_kwargs(states_actions_params),
        )

        # COLLECTIVE-REGIMES (E2): value-aware feasibility. Evaluated AFTER
        # Q^s — this is the reorder the singleton path never needs (there,
        # F is built before and independently of Q). Interpolate each declared
        # same-period reference value at the projected coordinates, then AND
        # every predicate — reading its own `Q_<s>` gathers, the reference
        # values, and ordinary cell kwargs — into the mask. The household
        # argmax downstream runs over the masked set; an all-infeasible cell
        # sets the dissolution flag D there (`collective_readout`).
        if value_constraint_machinery.evaluators:
            F_arr = _apply_value_constraints(
                machinery=value_constraint_machinery,
                Q_arr=jnp.asarray(Q_arr),
                # A constraint-less regime's F is the Python `True` scalar.
                F_arr=jnp.asarray(F_arr),
                states_actions_params=states_actions_params,
            )

        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


@dataclass(frozen=True, kw_only=True)
class _ValueConstraintMachinery:
    """Prebuilt E2 evaluation machinery closed over by a collective `Q_and_F`."""

    reference_readers: Mapping[str, Callable[..., FloatND]]
    """Per reference-value name, the same-period reference reader."""

    reference_reader_args: Mapping[str, tuple[str, ...]]
    """Each reader's argument names (fetched off the cell kwargs)."""

    evaluators: Mapping[str, Callable[..., BoolND]]
    """Per value-constraint name, the DAG-concatenated predicate."""

    evaluator_args: Mapping[str, tuple[str, ...]]
    """Each evaluator's argument names (split engine-supplied vs cell kwargs)."""

    q_value_index: Mapping[str, int]
    """`Q_<s>` argument name -> index on the trailing stakeholder axis."""

    engine_supplied_names: frozenset[str]
    """Names bound by the engine per cell — excluded from the kernel signature."""


def _build_value_constraint_machinery(
    *,
    value_constraints: ConstraintFunctionsMapping,
    same_period_refs: Mapping[str, ResolvedSamePeriodRef],
    stakeholders: tuple[str, ...],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    functions: EconFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction],
    conflicting_deterministic_transition_names: frozenset[
        TransitionFunctionName
    ] = frozenset(),
) -> _ValueConstraintMachinery:
    """Build the E2 reference readers and value-constraint evaluators once.

    COLLECTIVE-REGIMES (E2). Each evaluator is the predicate concatenated with
    the regime's function DAG (so it may read helper functions and the merged
    deterministic `next_<state>` laws, exactly like ordinary constraints); its
    engine-supplied arguments — `Q_<s>` and the reference-value names — are
    bound per (state, action) cell by `_apply_value_constraints`.

    `conflicting_deterministic_transition_names` is threaded to the reference
    readers and enforced on the value-constraint predicates themselves
    (`_fail_if_conflicting_transition_is_read`), exactly as for the ordinary
    utility/feasibility read in `_get_U_and_F`: an E2 predicate or projection
    reading a target-dependent `next_<state>` law would silently bind one
    target's law while the simulate state-update uses the per-target one.
    """
    reference_readers: dict[str, Callable[..., FloatND]] = {}
    reference_reader_args: dict[str, tuple[str, ...]] = {}
    for ref_name, ref in same_period_refs.items():
        reader = _build_same_period_ref_reader(
            ref=ref,
            v_interpolation_info=regime_to_v_interpolation_info[ref.regime],
            functions=functions,
            deterministic_transitions=deterministic_transitions,
            conflicting_deterministic_transition_names=(
                conflicting_deterministic_transition_names
            ),
        )
        reference_readers[ref_name] = reader
        reference_reader_args[ref_name] = tuple(get_union_of_args([reader]))

    dag_pool = {
        **dict(deterministic_transitions),
        **{k: v for k, v in functions.items() if k != "H"},
    }
    evaluators: dict[str, Callable[..., BoolND]] = {}
    evaluator_args: dict[str, tuple[str, ...]] = {}
    for constraint_name, predicate in value_constraints.items():
        combined = {**dag_pool, constraint_name: predicate}
        _fail_if_conflicting_transition_is_read(
            combined=combined,
            targets=[constraint_name],
            conflicting_deterministic_transition_names=(
                conflicting_deterministic_transition_names
            ),
        )
        evaluator = concatenate_functions(
            functions=combined,
            targets=constraint_name,
            enforce_signature=False,
            set_annotations=True,
        )
        evaluators[constraint_name] = evaluator
        evaluator_args[constraint_name] = tuple(get_union_of_args([evaluator]))

    q_value_index = {f"Q_{s}": index for index, s in enumerate(stakeholders)}
    return _ValueConstraintMachinery(
        reference_readers=MappingProxyType(reference_readers),
        reference_reader_args=MappingProxyType(reference_reader_args),
        evaluators=MappingProxyType(evaluators),
        evaluator_args=MappingProxyType(evaluator_args),
        q_value_index=MappingProxyType(q_value_index),
        engine_supplied_names=(frozenset(q_value_index) | frozenset(reference_readers)),
    )


def _apply_value_constraints(
    *,
    machinery: _ValueConstraintMachinery,
    Q_arr: FloatND,
    F_arr: BoolND,
    # `object` values: besides ordinary `_ParamsLeaf` leaves, the cell kwargs
    # carry the same-period V mapping under `SAME_PERIOD_V_ARG`.
    states_actions_params: Mapping[str, object],
) -> BoolND:
    """AND every value constraint into the feasibility of one (state, action) cell.

    COLLECTIVE-REGIMES (E2). Reads each declared same-period reference value at
    the projected coordinates (the readers pull the current-period reference V
    arrays off `states_actions_params[SAME_PERIOD_V_ARG]`), then evaluates each
    predicate with its `Q_<s>` arguments gathered from the trailing stakeholder
    axis of `Q_arr`, its reference-value arguments, and its remaining arguments
    from the cell kwargs.
    """
    reference_values = {
        ref_name: reader(
            **{
                arg: states_actions_params[arg]
                for arg in machinery.reference_reader_args[ref_name]
            }
        )
        for ref_name, reader in machinery.reference_readers.items()
    }
    for constraint_name, evaluate in machinery.evaluators.items():
        predicate_kwargs: dict[str, object] = {}
        for arg in machinery.evaluator_args[constraint_name]:
            if arg in machinery.q_value_index:
                predicate_kwargs[arg] = Q_arr[..., machinery.q_value_index[arg]]
            elif arg in reference_values:
                predicate_kwargs[arg] = reference_values[arg]
            else:
                predicate_kwargs[arg] = states_actions_params[arg]
        F_arr = jnp.logical_and(F_arr, evaluate(**predicate_kwargs))
    return F_arr


def _get_stakeholder_sliced_interpolator(
    *,
    base_interpolator: Callable[..., FloatND],
    V_arr_name: str,
    n_stakeholders: int,
) -> Callable[..., FloatND]:
    """Evaluate a V-interpolator per stakeholder slice of a stacked V array.

    COLLECTIVE-REGIMES (E1, slice 2). The target regime's `next_V_arr` leaf has
    shape `(*target_state_axes, n_stakeholders)`; the base interpolator
    interpolates over the state axes of a plain `(*target_state_axes,)` array.
    Calling it once per stakeholder on the slice `next_V_arr[..., s]` and
    re-stacking on a trailing axis keeps the interpolation semantics untouched
    and puts the stakeholder axis last by construction — no axis bookkeeping
    can reorder it. The wrapper carries the base interpolator's exact argument
    names so the stochastic-variable product-map and the extra-param discovery
    treat it like the singleton interpolator.

    Args:
        base_interpolator: The singleton V-interpolator from
            `get_V_interpolator` (state axes only).
        V_arr_name: Name of the interpolator's value-array argument.
        n_stakeholders: Number of stakeholder slices on the trailing axis.

    Returns:
        A callable with the base interpolator's signature returning the
        per-stakeholder interpolated values, stakeholder axis trailing.

    """
    arg_names = tuple(get_union_of_args([base_interpolator]))

    @with_signature(args=arg_names, return_annotation="FloatND")
    def next_V_per_stakeholder(**kwargs: _ParamsLeaf) -> FloatND:
        stacked_V_arr = cast("FloatND", kwargs.pop(V_arr_name))
        return jnp.stack(
            [
                base_interpolator(**kwargs, **{V_arr_name: stacked_V_arr[..., s]})
                for s in range(n_stakeholders)
            ],
            axis=-1,
        )

    return next_V_per_stakeholder


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
    bundles is ASSUMED to have no states — its V identically zero, so it
    contributes nothing to the continuation.

    That assumption is load-bearing, and only as good as the bundle
    construction in `_process_regime_core`: a target that DOES declare states
    yet ends up with an empty bundle is dropped from E[V] SILENTLY — no error,
    and its value is not in fact zero. Known remaining hole (fold-review F2,
    deliberately NOT closed in this slice): a target declaring a state the
    SOURCE does not also declare gets nothing wired for it —
    `target_process_grids` intersects each target's grids with the SOURCE
    regime's own `process_names`, and an ordinary state reached only via the
    regime transition has no auto-wiring at all — so such a target vanishes
    from the continuation and its actions are valued as if it were worthless.
    Closing it needs an unconditional-marginal transition primitive (a
    process's weight/next functions currently take the source's realized value
    of that same state as an argument, which a source that never declares it
    cannot supply); that is a separate feature, not a fold concern.

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
    utility_name: str = "utility",
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
        utility_name: DAG target name of the felicity function. `"utility"` (the
            default) is the singleton case; a collective regime passes a
            per-stakeholder `"utility_<s>"` so this builder returns that
            stakeholder's own `U^s` alongside the shared feasibility.

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
    guard_targets = [utility_name, *constraints]
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
    _fail_if_conflicting_transition_is_read(
        combined=combined,
        targets=[utility_name, "feasibility"],
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
    )
    return concatenate_functions(
        functions=combined,
        targets=[utility_name, "feasibility"],
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
            f"deterministic state law ({names}), but the targets that carry the "
            "state supply DIFFERENT callable OBJECTS for it (the conflict test is "
            "Python object identity, `merged[name] is not func` -- see "
            "`_get_deterministic_transitions`; two separately-declared but "
            "extensionally-equal functions are conservatively treated as a "
            "conflict, never merged silently). The decision DAG would bind one "
            "target's law while the simulate state-update uses the right one, so "
            "they would disagree silently. Target-independent laws must currently "
            "REUSE ONE callable object across every target that carries the state "
            "(assign the function once and reference it, do not redefine it per "
            "target); or stop reading the chosen next state in the within-period "
            "utility/feasibility. A registry-backed law identity that accepts "
            "extensionally-equal declarations is a future refinement."
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
