"""The per-(period, regime) DC-EGM solve kernel.

`build_egm_step_functions` turns a regime's processed functions, transitions,
and `DCEGM` solver configuration into per-period kernels that replace the
brute-force `max_Q_over_a` during backward induction. Each kernel runs the
concave EGM step on the exogenous savings grid:

1. compute child states at every savings node (the post-decision function is
   removed from the DAG so the savings node enters as an external input),
2. map the child state into the child's resources space and interpolate the
   child's carry rows there,
3. take the regime-transition-probability-weighted expectation of the
   marginal continuation, multiplying by the composed-gradient factor
   $\\partial R'/\\partial A$ of the map $A \\mapsto R'(\\mathcal{T}(A))$,
4. invert the Euler equation per savings node (with the degenerate-inversion
   guard) to obtain the optimal action and the endogenous resources grid,
5. add the closed-form credit-constrained segment as additional candidates,
6. refine the candidate correspondence through the configured upper-envelope
   backend,
7. publish the value function on the regime's exogenous state grid and
   assemble the carry for the regime's parents.

Scope: regimes (and their carry targets) without discrete states, discrete
actions, or process states; the expectation covers deterministic child-state
transitions. Configurations outside this scope build kernels that raise
`NotImplementedError` at solve time, so `Model` construction always succeeds
for a validated DC-EGM regime.
"""

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
from dags import concatenate_functions

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EgmCarry, build_template_egm_carry
from _lcm.egm.euler import invert_euler
from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope import get_upper_envelope
from _lcm.regime_building.h_dag import _get_build_H_kwargs
from _lcm.regime_building.next_state import get_next_state_function_for_solution
from _lcm.regime_building.Q_and_F import get_complete_targets
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import (
    EconFunctionsMapping,
    EgmStepFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from _lcm.utils.functools import get_union_of_args
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import Float1D, FloatND, ScalarFloat, ScalarInt, UserFunction

# Smallest constrained-segment action as a fraction of the segment's span.
# The constrained candidates are geometrically spaced from this offset toward
# the borrowing limit (additive, so a non-positive limit stays well-defined).
CONSTRAINED_OFFSET_FRACTION = 1e-4


def build_egm_step_functions(
    *,
    solver: DCEGM,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    flat_param_names: frozenset[str],
) -> tuple[MappingProxyType[int, EgmStepFunction], EgmCarry]:
    """Build per-period DC-EGM kernels and the regime's carry template.

    Periods sharing the same continuation-target configuration share one
    kernel (and hence one compiled program), mirroring the per-period
    grouping of the Q-and-F builders.

    Args:
        solver: The regime's DC-EGM solver configuration.
        regime_name: Name of the regime the kernels solve.
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances (the carry targets' resources functions are read from
            here).
        functions: The regime's processed functions (params renamed to
            qualified names).
        transitions: Immutable mapping of target regime names to their state
            transition functions.
        stochastic_transition_names: Frozenset of stochastic transition
            function names.
        compute_regime_transition_probs: Regime transition probability
            function for solve.
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info.
        regimes_to_active_periods: Immutable mapping of regime names to their
            active period tuples.
        flat_param_names: Frozenset of flat parameter names for the regime.

    Returns:
        Tuple of the per-period kernel mapping and the regime's all-finite
        carry template.

    """
    n_pad = compute_egm_carry_length(solver=solver)
    carry_template = build_template_egm_carry(n_rows=n_pad)

    configs: dict[tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]], list[int]] = {}
    for period in regimes_to_active_periods[regime_name]:
        target_split = get_egm_continuation_targets(
            period=period,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )
        configs.setdefault(target_split, []).append(period)

    built: dict[
        tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]], EgmStepFunction
    ] = {}
    for carry_targets, scalar_targets in configs:
        unsupported = _find_unsupported_feature(
            solver=solver,
            regime_name=regime_name,
            user_regimes=user_regimes,
            functions=functions,
            carry_targets=carry_targets,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            flat_param_names=flat_param_names,
        )
        if unsupported is not None:
            kernel = _get_raising_egm_step(reason=unsupported)
        else:
            kernel = _get_egm_step(
                solver=solver,
                user_regimes=user_regimes,
                functions=functions,
                transitions=transitions,
                compute_regime_transition_probs=compute_regime_transition_probs,
                carry_targets=carry_targets,
                scalar_targets=scalar_targets,
                n_pad=n_pad,
            )
        built[(carry_targets, scalar_targets)] = kernel

    result: dict[int, EgmStepFunction] = {}
    for target_split, periods in configs.items():
        for period in periods:
            result[period] = built[target_split]

    return MappingProxyType(dict(sorted(result.items()))), carry_template


def get_egm_continuation_targets(
    *,
    period: int,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]]:
    """Split next-period-active targets into carry-interpolated and scalar ones.

    This adapter is the single place where the EGM step derives "which target
    regimes / which transition functions" from the engine regime's
    transitions; changes to the transition representation swap this body
    without touching the kernel.

    - *Carry targets* have state-transition entries; their continuation is
      interpolated from their `EgmCarry` rows.
    - *Scalar targets* are stateless (no transition entries, no states; e.g.
      a `dead` regime); their continuation is the constant value of their
      carry rows and their marginal continuation is zero.

    Args:
        period: The period the kernel solves.
        transitions: Immutable mapping of target regime names to their state
            transition functions.
        stochastic_transition_names: Frozenset of stochastic transition
            function names.
        regimes_to_active_periods: Immutable mapping of regime names to their
            active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info.

    Returns:
        Tuple of carry-target names and scalar-target names.

    """
    carry_targets = get_complete_targets(
        period=period,
        transitions=transitions,
        regimes_to_active_periods=regimes_to_active_periods,
        stochastic_transition_names=stochastic_transition_names,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
    )
    scalar_targets = tuple(
        name
        for name in regime_to_v_interpolation_info
        if period + 1 in regimes_to_active_periods.get(name, ())
        and not regime_to_v_interpolation_info[name].state_names
        and name not in carry_targets
    )
    return carry_targets, scalar_targets


def compute_egm_carry_length(*, solver: DCEGM) -> int:
    """Static carry-row length for a DC-EGM regime.

    Covers the Euler candidates (one per savings node), the closed-form
    constrained candidates, and the headroom factor for envelope-kink
    insertions (each costs two slots).

    Args:
        solver: The regime's DC-EGM solver configuration.

    Returns:
        Number of slots in the regime's carry rows.

    """
    n_savings = int(solver.savings_grid.to_jax().shape[0])
    n_candidates = n_savings + solver.n_constrained_points
    return math.ceil(solver.refined_grid_factor * n_candidates)


def _get_egm_step(
    *,
    solver: DCEGM,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    compute_regime_transition_probs: RegimeTransitionFunction,
    carry_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...],
    n_pad: int,
) -> EgmStepFunction:
    """Build the EGM kernel for one continuation-target configuration."""
    euler_state_name = solver.continuous_state
    action_name = solver.continuous_action
    post_decision_name = solver.post_decision_function

    savings_nodes = jnp.asarray(
        solver.savings_grid.to_jax(), dtype=canonical_float_dtype()
    )
    borrowing_limit = savings_nodes[0]
    n_constrained = solver.n_constrained_points
    # Static geometric ratio: the constrained actions run from
    # `span * CONSTRAINED_OFFSET_FRACTION` up to `span`, so the ratio depends
    # only on the offset fraction and the point count.
    constrained_ratio = (1.0 / CONSTRAINED_OFFSET_FRACTION) ** (
        1.0 / max(n_constrained - 1, 1)
    )

    next_state_funcs, child_resources_funcs, child_next_state_keys = (
        _build_target_closures(
            user_regimes=user_regimes,
            functions=functions,
            transitions=transitions,
            carry_targets=carry_targets,
            post_decision_name=post_decision_name,
        )
    )
    utility_func, inverse_marginal_utility_func, own_resources_func = (
        _concatenate_regime_function(functions=functions, target="utility"),
        _concatenate_regime_function(
            functions=functions, target="inverse_marginal_utility"
        ),
        _concatenate_regime_function(functions=functions, target=solver.resources),
    )
    build_H_kwargs = _get_build_H_kwargs(functions)
    refine = get_upper_envelope(solver=solver, n_refined=n_pad)

    def egm_step(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],  # noqa: ARG001
        next_regime_to_egm_carry: MappingProxyType[RegimeName, EgmCarry],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[FloatND, EgmCarry]:
        """Run the concave EGM step and publish V on the exogenous grid.

        Args:
            next_regime_to_V_arr: The next period's value-function arrays;
                accepted so solve treats all kernels uniformly (continuation
                values come from the carries).
            next_regime_to_egm_carry: The next period's EGM carries.
            **kwargs: The regime's state grids, flat params, `period`, and
                `age`.

        Returns:
            Tuple of the value-function array on the exogenous state grid
            and the regime's carry.

        """
        dtype = canonical_float_dtype()
        pool = {k: v for k, v in kwargs.items() if k != euler_state_name}
        state_grid = jnp.asarray(kwargs[euler_state_name], dtype=dtype)

        regime_transition_probs = compute_regime_transition_probs(**pool)
        # Validation pins the default Bellman aggregator, whose single
        # non-(utility, E_next_V) parameter is the discount factor.
        (discount_factor,) = tuple(build_H_kwargs(pool).values())

        def utility_of_action(action_value: ScalarFloat) -> ScalarFloat:
            return utility_func(**{action_name: action_value}, **pool)

        def inverse_marginal_utility(
            marginal_continuation: ScalarFloat,
        ) -> ScalarFloat:
            return inverse_marginal_utility_func(
                marginal_continuation=marginal_continuation, **pool
            )

        def compute_node(
            savings_value: ScalarFloat,
        ) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]:
            """Euler-invert one savings node against the continuation."""
            expected_marginal = jnp.asarray(0.0, dtype=dtype)
            expected_value = jnp.asarray(0.0, dtype=dtype)
            for target in carry_targets:
                carry = next_regime_to_egm_carry[target]

                def composed_resources(
                    savings: ScalarFloat, *, target: RegimeName = target
                ) -> ScalarFloat:
                    """Map a savings node into the child's resources space."""
                    next_states = next_state_funcs[target](
                        **pool, **{post_decision_name: savings}
                    )
                    # The solution-phase next-state function returns a flat
                    # mapping of `next_<state>` names to scalars; the shared
                    # protocol's nested return type is the simulation form.
                    next_state_value = cast(
                        "ScalarFloat", next_states[child_next_state_keys[target]]
                    )
                    return child_resources_funcs[target](next_state_value)

                child_resources, child_dr_da = jax.value_and_grad(composed_resources)(
                    savings_value
                )
                value_at_child = interp_on_padded_grid(
                    x_query=child_resources, xp=carry.endog_grid, fp=carry.value
                )
                marginal_at_child = interp_on_padded_grid(
                    x_query=child_resources,
                    xp=carry.endog_grid,
                    fp=carry.marginal_utility,
                )
                prob = regime_transition_probs[target]
                # Zero unreachable-target contributions on the results, never
                # by multiplying into a possibly non-finite value.
                expected_marginal = expected_marginal + jnp.where(
                    prob > 0.0, prob * marginal_at_child * child_dr_da, 0.0
                )
                expected_value = expected_value + jnp.where(
                    prob > 0.0, prob * value_at_child, 0.0
                )
            for target in scalar_targets:
                prob = regime_transition_probs[target]
                constant_value = next_regime_to_egm_carry[target].value[0]
                expected_value = expected_value + jnp.where(
                    prob > 0.0, prob * constant_value, 0.0
                )

            action = invert_euler(
                expected_marginal_continuation=expected_marginal,
                discount_factor=discount_factor,
                inverse_marginal_utility=inverse_marginal_utility,
            )
            endog_point = savings_value + action
            value = utility_of_action(action) + discount_factor * expected_value
            return action, endog_point, value, expected_value

        actions, endog_grid, values, expected_values = jax.vmap(compute_node)(
            savings_nodes
        )

        def own_resources_of_state(state_value: ScalarFloat) -> ScalarFloat:
            return own_resources_func(**{euler_state_name: state_value}, **pool)

        publish_resources = jax.vmap(own_resources_of_state)(state_grid)

        constrained_actions, constrained_values = _compute_constrained_candidates(
            first_endogenous_point=endog_grid[0],
            publish_resources=publish_resources,
            borrowing_limit=borrowing_limit,
            n_constrained=n_constrained,
            constrained_ratio=constrained_ratio,
            utility_of_action=utility_of_action,
            discounted_expected_value_at_limit=discount_factor * expected_values[0],
        )

        refined_grid, refined_policy, refined_value, n_kept = refine(
            endog_grid=jnp.concatenate(
                [borrowing_limit + constrained_actions, endog_grid]
            ),
            policy=jnp.concatenate([constrained_actions, actions]),
            value=jnp.concatenate([constrained_values, values]),
        )

        return _publish_V_and_assemble_carry(
            refined_grid=refined_grid,
            refined_policy=refined_policy,
            refined_value=refined_value,
            n_kept=n_kept,
            n_pad=n_pad,
            publish_resources=publish_resources,
            borrowing_limit=borrowing_limit,
            utility_of_action=utility_of_action,
            discounted_expected_value_at_limit=discount_factor * expected_values[0],
        )

    return egm_step


def _build_target_closures(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    carry_targets: tuple[RegimeName, ...],
    post_decision_name: str,
) -> tuple[
    dict[RegimeName, Callable[..., Any]],
    dict[RegimeName, Callable[[ScalarFloat], ScalarFloat]],
    dict[RegimeName, str],
]:
    """Build the per-carry-target closures of the EGM kernel.

    The post-decision function is removed from the DAG, so its output (the
    savings node) becomes an external input of the next-state functions.
    Passing the full `functions` mapping would let the DAG compute savings
    internally from the (unknown) state and action leaves — it runs, but is
    silently wrong.

    Returns:
        Tuple of three dicts keyed by carry-target name: the next-state
        function, the closed-over child resources map, and the child's
        `next_<state>` key.

    """
    functions_without_post = MappingProxyType(
        {name: func for name, func in functions.items() if name != post_decision_name}
    )
    next_state_funcs: dict[RegimeName, Callable[..., Any]] = {
        target: get_next_state_function_for_solution(
            transitions=transitions[target],
            functions=functions_without_post,
        )
        for target in carry_targets
    }
    child_resources_funcs = {
        target: _get_child_resources_function(user_regime=user_regimes[target])
        for target in carry_targets
    }
    child_next_state_keys = {
        target: f"next_{_get_child_state_name(user_regime=user_regimes[target])}"
        for target in carry_targets
    }
    return next_state_funcs, child_resources_funcs, child_next_state_keys


def _concatenate_regime_function(
    *,
    functions: EconFunctionsMapping,
    target: str,
) -> UserFunction:
    """Concatenate one regime-function target from the H-free DAG."""
    return concatenate_functions(
        functions={name: func for name, func in functions.items() if name != "H"},
        targets=target,
        enforce_signature=False,
        set_annotations=True,
    )


def _compute_constrained_candidates(
    *,
    first_endogenous_point: ScalarFloat,
    publish_resources: FloatND,
    borrowing_limit: ScalarFloat,
    n_constrained: int,
    constrained_ratio: float,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
) -> tuple[FloatND, FloatND]:
    """Compute the closed-form credit-constrained candidate points.

    Below the first endogenous point the borrowing limit binds: the action is
    `R - borrowing_limit` and the continuation is the limit-node expectation.
    The candidate actions are geometrically spaced toward the limit; the span
    is capped at the publish range so a degenerate (huge) first endogenous
    point cannot stretch the sample beyond where queries land.

    Args:
        first_endogenous_point: The endogenous resources point of the lowest
            savings node.
        publish_resources: Resources at the regime's exogenous state grid.
        borrowing_limit: Lower bound of the savings grid.
        n_constrained: Number of constrained candidate points.
        constrained_ratio: Static geometric spacing ratio of the candidates.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of constrained candidate actions and values.

    """
    dtype = publish_resources.dtype
    span = jnp.maximum(
        jnp.minimum(first_endogenous_point, jnp.max(publish_resources))
        - borrowing_limit,
        jnp.finfo(dtype).tiny,
    )
    constrained_actions = (
        span
        * CONSTRAINED_OFFSET_FRACTION
        * constrained_ratio ** jnp.arange(n_constrained, dtype=dtype)
    )
    constrained_values = (
        jax.vmap(utility_of_action)(constrained_actions)
        + discounted_expected_value_at_limit
    )
    return constrained_actions, constrained_values


def _publish_V_and_assemble_carry(
    *,
    refined_grid: Float1D,
    refined_policy: Float1D,
    refined_value: Float1D,
    n_kept: ScalarInt,
    n_pad: int,
    publish_resources: FloatND,
    borrowing_limit: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
) -> tuple[FloatND, EgmCarry]:
    """Interpolate V onto the exogenous grid and assemble the carry.

    The published value is the maximum of the interpolated refined envelope
    and the closed-form constrained value: the latter is the exact value of a
    feasible policy (save exactly the borrowing limit), so the maximum is
    exact where the constraint binds and a valid lower bound everywhere else.

    Envelope overflow is not silent: the outputs are NaN-poisoned so the
    solve loop's NaN diagnostics surface the offending (regime, period).

    Args:
        refined_grid: Refined endogenous grid row from the envelope backend.
        refined_policy: Refined policy row.
        refined_value: Refined value row.
        n_kept: Number of envelope points the backend kept.
        n_pad: Static length of the refined rows.
        publish_resources: Resources at the regime's exogenous state grid.
        borrowing_limit: Lower bound of the savings grid.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of the value-function array and the regime's carry.

    """
    dtype = publish_resources.dtype
    overflowed = n_kept > n_pad

    value_interpolated = interp_on_padded_grid(
        x_query=publish_resources, xp=refined_grid, fp=refined_value
    )
    closed_form_actions = publish_resources - borrowing_limit
    value_constrained = jnp.where(
        closed_form_actions > 0.0,
        jax.vmap(utility_of_action)(
            jnp.maximum(closed_form_actions, jnp.finfo(dtype).tiny)
        )
        + discounted_expected_value_at_limit,
        -jnp.inf,
    )
    V_arr = jnp.maximum(value_interpolated, value_constrained)
    V_arr = jnp.where(overflowed, jnp.nan, V_arr).astype(dtype)

    marginal_utility = jax.vmap(jax.grad(utility_of_action))(
        jnp.where(jnp.isnan(refined_policy), 1.0, refined_policy)
    )
    marginal_utility = jnp.where(jnp.isnan(refined_policy), jnp.nan, marginal_utility)
    marginal_utility = jnp.where(jnp.isneginf(refined_value), 0.0, marginal_utility)

    carry = EgmCarry(
        endog_grid=jnp.where(overflowed, jnp.nan, refined_grid).astype(dtype),
        policy=refined_policy.astype(dtype),
        value=jnp.where(overflowed, jnp.nan, refined_value).astype(dtype),
        marginal_utility=marginal_utility.astype(dtype),
        taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
    )
    return V_arr, carry


def _get_raising_egm_step(*, reason: str) -> EgmStepFunction:
    """Build a kernel that raises at solve time for unsupported configurations.

    `Model` construction with a validated DC-EGM regime always succeeds;
    features the kernel does not cover yet surface as `NotImplementedError`
    when the model is solved.
    """

    def raising_egm_step(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        next_regime_to_egm_carry: MappingProxyType[RegimeName, EgmCarry],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[FloatND, EgmCarry]:
        raise NotImplementedError(reason)

    return raising_egm_step


def _find_unsupported_feature(
    *,
    solver: DCEGM,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    carry_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    flat_param_names: frozenset[str],
) -> str | None:
    """Return a message naming the first feature outside the kernel's scope.

    Returns `None` when the configuration is fully supported.
    """
    own_extra_actions = [
        name
        for name in user_regimes[regime_name].actions
        if name != solver.continuous_action
    ]
    message: str | None = None
    if own_extra_actions:
        message = (
            f"it has actions {own_extra_actions} besides the continuous "
            f"action '{solver.continuous_action}'."
        )
    elif regime_to_v_interpolation_info[regime_name].discrete_states:
        discrete = list(regime_to_v_interpolation_info[regime_name].discrete_states)
        message = f"it has discrete or process states {discrete}."

    for target in carry_targets:
        if message is not None:
            break
        message = _find_unsupported_target_feature(
            target=target,
            user_regimes=user_regimes,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )

    if message is None:
        message = _find_unsupported_function_args(
            solver=solver,
            functions=functions,
            compute_regime_transition_probs=compute_regime_transition_probs,
            flat_param_names=flat_param_names,
        )

    if message is None:
        return None
    return (
        f"The DC-EGM solver cannot solve regime '{regime_name}' yet: {message} "
        "Support arrives with the discrete-choice DC-EGM step."
    )


def _find_unsupported_target_feature(
    *,
    target: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> str | None:
    """Return a message naming the first unsupported feature of one target."""
    target_info = regime_to_v_interpolation_info[target]
    if target_info.discrete_states:
        return (
            f"its target regime '{target}' has discrete or process "
            f"states {list(target_info.discrete_states)}."
        )
    if len(target_info.state_names) != 1:
        return (
            f"its target regime '{target}' has states "
            f"{list(target_info.state_names)}; exactly one continuous state "
            "is supported."
        )
    if user_regimes[target].terminal and user_regimes[target].actions:
        return (
            f"its terminal target regime '{target}' has actions "
            f"{list(user_regimes[target].actions)}, so its carry is not "
            "its utility on the state grid."
        )
    stochastic = sorted(set(transitions[target]) & stochastic_transition_names)
    if stochastic:
        return f"the transitions {stochastic} into regime '{target}' are stochastic."
    child_state_name = _get_child_state_name(user_regime=user_regimes[target])
    extra_resources_args = sorted(
        _get_child_resources_arg_names(user_regime=user_regimes[target])
        - {child_state_name}
    )
    if extra_resources_args:
        return (
            f"the resources function of target regime '{target}' depends on "
            f"{extra_resources_args} in addition to '{child_state_name}'."
        )
    return None


def _find_unsupported_function_args(
    *,
    solver: DCEGM,
    functions: EconFunctionsMapping,
    compute_regime_transition_probs: RegimeTransitionFunction,
    flat_param_names: frozenset[str],
) -> str | None:
    """Return a message naming the first function with out-of-scope arguments."""
    allowed_params = flat_param_names | {"age", "period"}
    utility_func = _concatenate_regime_function(functions=functions, target="utility")
    arg_requirements: list[tuple[str, frozenset[str], set[str]]] = [
        (
            "the utility function",
            frozenset(get_union_of_args([utility_func])),
            {solver.continuous_action} | allowed_params,
        ),
        (
            "the regime transition probability function",
            frozenset(get_union_of_args([compute_regime_transition_probs])),
            set(allowed_params),
        ),
    ]
    for label, needed, allowed in arg_requirements:
        extra = sorted(needed - allowed)
        if extra:
            return f"{label} depends on {extra}."
    return None


def _get_child_state_name(*, user_regime: UserRegime) -> StateName:
    """Name of a carry target's single continuous state.

    For a DC-EGM target this is its configured Euler state; for a terminal
    target it is its only state (uniqueness is checked by
    `_find_unsupported_feature`).
    """
    if isinstance(user_regime.solver, DCEGM):
        return user_regime.solver.continuous_state
    return next(iter(user_regime.states))


def _get_child_resources_function(
    *, user_regime: UserRegime
) -> Callable[[ScalarFloat], ScalarFloat]:
    """Build the closed-over resources map of one carry target.

    For a DC-EGM target the map is its declared resources function (resolved
    to the solve-phase variant); for a terminal target the carry lives in
    M-space and the map is the identity. The returned callable takes the
    child's state value positionally so the kernel can compose it with the
    state transition and differentiate the composition.
    """
    if isinstance(user_regime.solver, DCEGM):
        child_state_name = user_regime.solver.continuous_state
        resources_func = _concatenate_child_resources(user_regime=user_regime)

        def child_resources(state_value: ScalarFloat) -> ScalarFloat:
            return resources_func(**{child_state_name: state_value})

        return child_resources

    def identity_resources(state_value: ScalarFloat) -> ScalarFloat:
        return state_value

    return identity_resources


def _get_child_resources_arg_names(*, user_regime: UserRegime) -> set[str]:
    """Argument names of a carry target's resources map."""
    if isinstance(user_regime.solver, DCEGM):
        return set(
            get_union_of_args([_concatenate_child_resources(user_regime=user_regime)])
        )
    return {next(iter(user_regime.states))}


def _concatenate_child_resources(*, user_regime: UserRegime) -> UserFunction:
    """Concatenate a DC-EGM target's resources function from its user DAG."""
    solver = cast("DCEGM", user_regime.solver)
    resolved: dict[str, UserFunction] = {}
    for name, func in user_regime.functions.items():
        if isinstance(func, Phased):
            resolved[name] = cast("UserFunction", func.solve)
        else:
            resolved[name] = func
    return concatenate_functions(
        functions=resolved,
        targets=solver.resources,
        enforce_signature=False,
        set_annotations=True,
    )
