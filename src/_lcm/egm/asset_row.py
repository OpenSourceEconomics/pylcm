"""Asset-row DC-EGM: the per-Euler-node solve for Euler-state-dependent stages.

When any savings-stage function reads the current Euler state — the state's own
law, the regime-transition probabilities, a transition weight, or a passive
state's law — the kernel solves once per exogenous asset node instead of once
per combo. Conditional on a node, every such read is a per-combo constant, so
`step_core`'s single-post-state pipeline is exact within the node's row, and the
row publishes only its own node (brute-force-equivalent by construction). This
module maps that core pipeline over the Euler grid, differentiates the
continuation in the Euler slot for the per-node marginal $dV/dR$, and publishes
each node's scalar value and optimal action at its single resources query. The
per-combo carry holds the per-node published points, not the envelope workspace.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import (
    bind_continuation,
)
from _lcm.egm.interp import (
    _interp_between_nodes,
    interp_on_padded_grid,
)
from _lcm.egm.step_core import (
    _candidate_supgradient,
    _compute_constrained_candidates,
    _compute_nodes_over_savings,
    _EgmKernelPieces,
    _get_compute_node,
)
from _lcm.egm.upper_envelope.fues import (
    QueryBracket,
)
from _lcm.typing import (
    RegimeName,
    StateName,
)
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    ScalarFloat,
    ScalarInt,
)


def _get_solve_one_combo_asset_rows(
    *,
    pieces: _EgmKernelPieces,
    pool: dict[str, Any],
    state_grid: Float1D,
    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
    euler_batch_size: int,
    savings_batch_size: int,
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
) -> Callable[
    [tuple[ScalarInt | ScalarFloat, ...]],
    tuple[Float1D, Float1D, Float1D, Float1D, Float1D],
]:
    """Build the per-combo EGM computation solving per exogenous asset node.

    Used when any savings-stage function (the Euler state's law, the
    regime-transition probabilities, a stochastic transition weight, a
    passive state's law) reads the current Euler state: conditional on one
    node of the Euler grid every such read is a per-combo constant, so the
    single-post-state pipeline (Euler inversion over the savings nodes,
    constrained candidates, upper-envelope refinement) is exact within the
    node's row, and each row publishes only its own node — exactly where
    the brute-force oracle evaluates the same decision-time functions. The
    per-combo carry row holds the per-node published points: abscissa the
    node resources (strictly increasing by the resources monotonicity check),
    value the published V, and marginal the corrected
    $dV/dR = u'(c^*) + \\beta\\, (\\partial W/\\partial a)|_{A^*} / R'(a)$,
    NaN-padded to the carry length. The Euler-state gradient
    $\\partial W/\\partial a$ rebuilds the full continuation closure from
    the traced node value, so it carries every direct channel: the law's
    residual, $\\sum \\partial P/\\partial a \\cdot EV$ from the
    probabilities, the weights' and passive laws' derivatives.
    """
    dtype = state_grid.dtype
    n_state = int(state_grid.shape[0])

    def solve_one_combo(
        combo_values: tuple[ScalarInt | ScalarFloat, ...],
    ) -> tuple[Float1D, Float1D, Float1D, Float1D, Float1D]:
        """Run the per-asset-node EGM step for one (discrete x passive) combo.

        Takes the combo's values (discrete codes and passive node values)
        positionally so `jax.vmap` can batch over flattened combo arrays.

        Returns:
            Tuple of the combo's value row on the exogenous state grid and
            its per-node endogenous grid, the published consumption policy on
            that grid, and the value and marginal-utility carry
            rows.

        """
        combo_pool = {
            **pool,
            **dict(zip(pieces.combo_names, combo_values, strict=True)),
        }
        # Validation pins the default Bellman aggregator, whose single
        # non-(utility, E_next_V) parameter is the discount factor.
        (discount_factor,) = tuple(pieces.build_H_kwargs(combo_pool).values())

        def own_resources_of_state(state_value: ScalarFloat) -> ScalarFloat:
            return pieces.own_resources_func(
                **{pieces.euler_state_name: state_value}, **combo_pool
            )

        def continuation_of_euler_state(
            state_value: ScalarFloat, savings_value: ScalarFloat
        ) -> ScalarFloat:
            """Expected continuation with the Euler slot as the grad argument.

            Rebuilds the per-combo continuation closure with the Euler slot
            of the combo pool bound to `state_value` (positional, so
            `jax.grad` differentiates the law's direct Euler-state channel)
            and evaluates it at fixed savings.
            """
            node_pool = {**combo_pool, pieces.euler_state_name: state_value}
            expected_continuation = _get_expected_continuation_value(
                pieces=pieces,
                combo_pool=node_pool,
                next_regime_to_continuation=next_regime_to_continuation,
                dtype=dtype,
                resolved_process_grids=resolved_process_grids,
            )
            return expected_continuation(savings_value)

        def solve_one_node(
            node_value: ScalarFloat,
        ) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat]:
            """Run the single-post-state pipeline conditional on one node."""
            node_pool = {**combo_pool, pieces.euler_state_name: node_value}

            def utility_of_action(action_value: ScalarFloat) -> ScalarFloat:
                return pieces.utility_func(
                    **{pieces.action_name: action_value}, **node_pool
                )

            compute_node = _get_compute_node(
                pieces=pieces,
                combo_pool=node_pool,
                discount_factor=discount_factor,
                utility_of_action=utility_of_action,
                next_regime_to_continuation=next_regime_to_continuation,
                dtype=dtype,
                resolved_process_grids=resolved_process_grids,
            )
            actions, endog_grid, values, expected_values = _compute_nodes_over_savings(
                compute_node=compute_node,
                savings_nodes=pieces.savings_nodes,
                savings_batch_size=savings_batch_size,
            )

            resources_at_node, resources_gradient = jax.value_and_grad(
                own_resources_of_state
            )(node_value)

            constrained_actions, constrained_values = _compute_constrained_candidates(
                first_endogenous_point=endog_grid[0],
                publish_resources=resources_at_node,
                borrowing_limit=pieces.borrowing_limit,
                n_constrained=pieces.n_constrained,
                constrained_ratio=pieces.constrained_ratio,
                utility_of_action=utility_of_action,
                discounted_expected_value_at_limit=discount_factor * expected_values[0],
            )

            candidate_grid = jnp.concatenate(
                [pieces.borrowing_limit + constrained_actions, endog_grid]
            )
            candidate_policy = jnp.concatenate([constrained_actions, actions])
            candidate_value = jnp.concatenate([constrained_values, values])
            # Exogenous source savings per candidate: the savings node for each
            # Euler candidate (`endog_grid = savings_node + action`, so the
            # implied `endog_grid - policy` equals it in exact arithmetic), the
            # borrowing limit for the constrained candidates (their savings is
            # pinned there). FUES compares these pristine sources exactly.
            candidate_savings = jnp.concatenate(
                [
                    jnp.full_like(constrained_actions, pieces.borrowing_limit),
                    pieces.savings_nodes,
                ]
            )
            # Same `-inf` masking as the default per-combo computation: dead
            # candidates become the envelope scan's absent form (NaN).
            candidate_dead = jnp.isneginf(candidate_value)
            candidate_marginal = _candidate_supgradient(
                policy=candidate_policy,
                dead=candidate_dead,
                utility_of_action=utility_of_action,
            )
            # The node reads its refined envelope at one query
            # (`resources_at_node`): every finder materializes the full refined
            # envelope row and locates the bracketing pair, so the published
            # `(V, policy)` is a full-envelope-then-interpolate. A sub-`n_pad`
            # streamed finder is future work for all backends.
            bracket = pieces.refine_to_bracket(
                endog_grid=jnp.where(candidate_dead, jnp.nan, candidate_grid),
                policy=jnp.where(candidate_dead, jnp.nan, candidate_policy),
                value=jnp.where(candidate_dead, jnp.nan, candidate_value),
                marginal_utility=candidate_marginal,
                savings=jnp.where(candidate_dead, jnp.nan, candidate_savings),
                x_query=resources_at_node,
            )

            V_node, policy_node = publish_node_from_bracket(
                bracket=bracket,
                n_pad=pieces.n_pad,
                resources_at_node=resources_at_node,
                borrowing_limit=pieces.borrowing_limit,
                utility_of_action=utility_of_action,
                discounted_expected_value_at_limit=discount_factor * expected_values[0],
            )

            # The carry marginal at this node:
            # dV/dR = u'(c*) + discount_factor * (dW/da at fixed A*) / R'(a),
            # with A* = R(a) - c*(a). The envelope term u'(c*) covers the
            # savings channel; the second term is the law's direct
            # Euler-state channel through the continuation, mapped into
            # resources space by the resources slope.
            marginal_utility_node = jax.grad(utility_of_action)(
                jnp.where(jnp.isnan(policy_node), 1.0, policy_node)
            )
            savings_at_optimum = resources_at_node - policy_node
            continuation_gradient = jax.grad(continuation_of_euler_state)(
                node_value, savings_at_optimum
            )
            mu_node = (
                marginal_utility_node
                + discount_factor * continuation_gradient / resources_gradient
            )
            V_node, mu_node = _finalize_asset_row_node(
                V_node=V_node,
                mu_node=mu_node,
                policy_node=policy_node,
                candidate_dead=candidate_dead,
                resources_gradient=resources_gradient,
            )
            return V_node, policy_node, mu_node

        # Splay the per-asset-node solve into `lax.map` blocks of
        # `euler_batch_size` to shed peak working-set memory; `0` (or a size
        # covering the whole grid) keeps the fused vmap. The two are
        # numerically identical — only the schedule differs.
        if 0 < euler_batch_size < n_state:
            V_vec, policy_vec, mu_vec = jax.lax.map(
                solve_one_node, state_grid, batch_size=euler_batch_size
            )
        else:
            V_vec, policy_vec, mu_vec = jax.vmap(solve_one_node)(state_grid)
        publish_resources = jax.vmap(own_resources_of_state)(state_grid)

        pad = jnp.full((pieces.n_carry_rows - n_state,), jnp.nan, dtype=dtype)
        grid_row = jnp.concatenate([publish_resources.astype(dtype), pad])
        policy_row = jnp.concatenate([policy_vec.astype(dtype), pad])
        value_row = jnp.concatenate([V_vec.astype(dtype), pad])
        marginal_row = jnp.concatenate([mu_vec.astype(dtype), pad])

        if pieces.feasibility_func is not None:
            # Infeasible discrete combos: -inf value rows so they win no
            # maximum and carry zero choice probability; exactly-zero
            # marginal utility so probability-weighted sums stay finite.
            feasible = pieces.feasibility_func(**combo_pool)
            V_vec = jnp.where(feasible, V_vec, -jnp.inf)
            value_row = jnp.where(feasible, value_row, -jnp.inf)
            marginal_row = jnp.where(feasible, marginal_row, 0.0)

        return (
            V_vec.astype(dtype),
            grid_row,
            policy_row,
            value_row,
            marginal_row,
        )

    return solve_one_combo


def _finalize_asset_row_node(
    *,
    V_node: ScalarFloat,
    mu_node: ScalarFloat,
    policy_node: ScalarFloat,
    candidate_dead: BoolND,
    resources_gradient: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Apply the no-live-candidate and resources-validity guards to one node.

    A node with no live candidate (its entire continuation is `-inf`) is worth
    `-inf`, like an infeasible combo, with an exactly-zero marginal so
    probability-weighted sums stay finite. Separately, the published marginal
    divides the direct continuation channel by the resources slope `dR/da`, which
    the method requires strictly positive (the carry abscissae are the node
    resources). The build-time strict-monotonicity check skips a resources map that
    reads a parameter, so a non-positive slope would otherwise publish a
    finite value/marginal with the wrong orientation. Fail loud: NaN the value
    *and* the marginal so the solve's NaN diagnostics (which scan the value array)
    name the offending regime/period rather than returning silently-wrong numbers.
    """
    mu_node = jnp.where(jnp.isnan(policy_node), jnp.nan, mu_node)
    no_live_candidate = jnp.all(candidate_dead)
    V_node = jnp.where(no_live_candidate, -jnp.inf, V_node)
    mu_node = jnp.where(jnp.isneginf(V_node) | no_live_candidate, 0.0, mu_node)
    invalid_resources = resources_gradient <= 0.0
    V_node = jnp.where(invalid_resources, jnp.nan, V_node)
    mu_node = jnp.where(invalid_resources, jnp.nan, mu_node)
    return V_node, mu_node


def _get_expected_continuation_value(
    *,
    pieces: _EgmKernelPieces,
    combo_pool: dict[str, Any],
    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
    dtype: Any,  # noqa: ANN401
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
) -> Callable[[ScalarFloat], ScalarFloat]:
    """Build the expected-continuation map $W(A)$ for one combo pool.

    Mirrors the expected-value aggregation of the per-savings-node Euler
    inversion: per-target smoothed carry reads, regime-transition-probability
    weighted, plus the scalar targets' constant values. The asset-row mode
    differentiates this map in the Euler slot of the combo pool to obtain
    the direct Euler-state channel $\\partial W/\\partial a$. The
    probabilities, transition weights, and child next-state reads are all
    evaluated from the combo pool *inside* this builder, so when the pool's
    Euler slot is a traced value the gradient carries their first-order
    terms (e.g. $\\sum \\partial P/\\partial a \\cdot EV$) — precomputing
    them outside the differentiated closure would silently drop those
    terms (Danskin does not cancel them: the probabilities are not the
    softmax of the values they weight).
    """
    continuation = bind_continuation(
        plan=pieces.continuation_plan,
        combo_pool=combo_pool,
        next_regime_to_continuation=next_regime_to_continuation,
        dtype=dtype,
        resolved_process_grids=resolved_process_grids,
    )

    def expected_continuation(savings_value: ScalarFloat) -> ScalarFloat:
        expected_value, _ = continuation(savings_value)
        return expected_value

    return expected_continuation


def _publish_node_V_and_policy(
    *,
    refined_grid: Float1D,
    refined_policy: Float1D,
    refined_value: Float1D,
    n_kept: ScalarInt,
    n_pad: int,
    resources_at_node: ScalarFloat,
    borrowing_limit: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Publish one asset node's value and optimal action at its single query.

    The scalar-query counterpart of `_publish_V_and_carry_rows`: below the
    lowest refined envelope point the closed-form constrained value is
    published outright; everywhere else the Hermite read of the refined
    envelope is floored at the constrained value, which remains a
    feasible-policy lower bound. The published action follows the winning
    branch — the closed-form action `R - borrowing_limit` where the
    constrained value wins, the interpolated refined policy otherwise.
    Envelope overflow NaN-poisons both outputs so the solve loop's NaN
    diagnostics surface the offending (regime, period).

    The production asset-row solve publishes via `_publish_V_and_carry_rows`,
    which likewise materializes the full refined row and slices the bracket (no
    streaming); this single-query form is the readable reference the row-path
    equivalence tests (`test_egm_refine_to_query.py`) check that publisher
    against.

    Args:
        refined_grid: Refined endogenous grid row from the envelope backend.
        refined_policy: Refined policy row.
        refined_value: Refined value row.
        n_kept: Number of envelope points the backend kept.
        n_pad: Static length of the refined rows.
        resources_at_node: Resources at this exogenous Euler node (the row's
            single publish query).
        borrowing_limit: Lower bound of the savings grid.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of the node's published value and published optimal action.

    """
    dtype = resources_at_node.dtype
    overflowed = n_kept > n_pad

    marginal_utility = jax.vmap(jax.grad(utility_of_action))(
        jnp.where(jnp.isnan(refined_policy), 1.0, refined_policy)
    )
    marginal_utility = jnp.where(jnp.isnan(refined_policy), jnp.nan, marginal_utility)
    marginal_utility = jnp.where(jnp.isneginf(refined_value), 0.0, marginal_utility)

    value_interpolated = interp_on_padded_grid(
        x_query=resources_at_node,
        xp=refined_grid,
        fp=refined_value,
        fp_slopes=marginal_utility,
    )
    policy_interpolated = interp_on_padded_grid(
        x_query=resources_at_node,
        xp=refined_grid,
        fp=refined_policy,
    )
    closed_form_action = resources_at_node - borrowing_limit
    value_constrained = jnp.where(
        closed_form_action > 0.0,
        utility_of_action(jnp.maximum(closed_form_action, jnp.finfo(dtype).tiny))
        + discounted_expected_value_at_limit,
        -jnp.inf,
    )
    below_refined = (closed_form_action > 0.0) & (resources_at_node <= refined_grid[0])
    constrained_wins = below_refined | (value_constrained >= value_interpolated)
    V_node = jnp.where(
        below_refined,
        value_constrained,
        jnp.maximum(value_interpolated, value_constrained),
    )
    policy_node = jnp.where(constrained_wins, closed_form_action, policy_interpolated)
    V_node = jnp.where(overflowed, jnp.nan, V_node).astype(dtype)
    policy_node = jnp.where(overflowed, jnp.nan, policy_node).astype(dtype)
    return V_node, policy_node


def publish_node_from_bracket(
    *,
    bracket: QueryBracket,
    n_pad: int,
    resources_at_node: ScalarFloat,
    borrowing_limit: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Publish one asset node's value and optimal action from its query bracket.

    The single-bracket counterpart of `_publish_node_V_and_policy`: it consumes
    the two envelope nodes `refine_to_bracket` sliced around `resources_at_node`
    from the full refined row. The published economics are identical:

    - The value is the cubic-Hermite read of the envelope between the two
      bracket nodes (the value slope at each node is `grad(utility_of_action)`
      at that node's policy, the envelope-theorem marginal), floored at the
      closed-form constrained value, which is a feasible-policy lower bound.
    - Below the lowest envelope node the closed-form constrained value is
      published outright; the winning branch sets the action (the closed-form
      `R - borrowing_limit` where constrained wins, the interpolated policy
      otherwise).
    - Envelope overflow (`n_kept > n_pad`) NaN-poisons both outputs, identical
      to the row path, so the solve loop's NaN diagnostics surface the offending
      (regime, period).

    Because the value and policy arithmetic is the shared `_interp_between_nodes`
    primitive — the same one the row path reaches through
    `interp_on_padded_grid`, reading the same refined row — the bracket publish
    cannot diverge from row-then-interpolate.

    Args:
        bracket: The query bracket from `refine_to_bracket`.
        n_pad: Static length of the envelope-refinement workspace (the overflow
            threshold).
        resources_at_node: Resources at this exogenous Euler node (the row's
            single publish query).
        borrowing_limit: Lower bound of the savings grid.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of the node's published value and published optimal action.

    """
    dtype = resources_at_node.dtype
    overflowed = bracket.n_kept > n_pad

    # The value Hermite slope is the envelope-theorem marginal `u'(c*)`, masked
    # exactly as the row path: NaN where the node's policy is NaN (a padded
    # slot), 0.0 where the node's value is `-inf` (an infeasible endpoint), so
    # `_interp_between_nodes` falls back to the linear rule on those brackets.
    slope_lower = _node_value_slope(
        policy=bracket.lower_policy,
        value=bracket.lower_value,
        utility_of_action=utility_of_action,
    )
    slope_upper = _node_value_slope(
        policy=bracket.upper_policy,
        value=bracket.upper_value,
        utility_of_action=utility_of_action,
    )

    value_interpolated = _interp_between_nodes(
        x_query=resources_at_node,
        xp_lower=bracket.lower_grid,
        xp_upper=bracket.upper_grid,
        fp_lower=bracket.lower_value,
        fp_upper=bracket.upper_value,
        slope_lower=slope_lower,
        slope_upper=slope_upper,
    )
    policy_interpolated = _interp_between_nodes(
        x_query=resources_at_node,
        xp_lower=bracket.lower_grid,
        xp_upper=bracket.upper_grid,
        fp_lower=bracket.lower_policy,
        fp_upper=bracket.upper_policy,
    )
    # Degenerate envelopes mirror the row read's contract exactly
    # (`interp_on_prepared_grid`): a single kept node — always the bracket's
    # lower slot, the upper is a NaN pad — is a constant clamp on both sides,
    # and an empty envelope reads NaN.
    single_node = bracket.n_kept == 1
    empty_envelope = bracket.n_kept == 0
    value_interpolated = jnp.where(single_node, bracket.lower_value, value_interpolated)
    value_interpolated = jnp.where(empty_envelope, jnp.nan, value_interpolated)
    policy_interpolated = jnp.where(
        single_node, bracket.lower_policy, policy_interpolated
    )
    policy_interpolated = jnp.where(empty_envelope, jnp.nan, policy_interpolated)

    closed_form_action = resources_at_node - borrowing_limit
    value_constrained = jnp.where(
        closed_form_action > 0.0,
        utility_of_action(jnp.maximum(closed_form_action, jnp.finfo(dtype).tiny))
        + discounted_expected_value_at_limit,
        -jnp.inf,
    )
    below_refined = (closed_form_action > 0.0) & (
        resources_at_node <= bracket.first_grid
    )
    constrained_wins = below_refined | (value_constrained >= value_interpolated)
    V_node = jnp.where(
        below_refined,
        value_constrained,
        jnp.maximum(value_interpolated, value_constrained),
    )
    policy_node = jnp.where(constrained_wins, closed_form_action, policy_interpolated)
    V_node = jnp.where(overflowed, jnp.nan, V_node).astype(dtype)
    policy_node = jnp.where(overflowed, jnp.nan, policy_node).astype(dtype)
    return V_node, policy_node


def _node_value_slope(
    *,
    policy: ScalarFloat,
    value: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
) -> ScalarFloat:
    """Envelope-theorem value slope at one bracket node, masked like the row.

    The slope is `grad(utility_of_action)` at the node's policy — NaN where the
    policy is a padded NaN slot, 0.0 where the node value is `-inf` — matching
    the per-node masking the row path applies before interpolating.
    """
    slope = jax.grad(utility_of_action)(jnp.where(jnp.isnan(policy), 1.0, policy))
    slope = jnp.where(jnp.isnan(policy), jnp.nan, slope)
    return jnp.where(jnp.isneginf(value), 0.0, slope)
