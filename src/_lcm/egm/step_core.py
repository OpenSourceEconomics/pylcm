"""The textbook DC-EGM step: Euler inversion to the upper-envelope value row.

The single-post-state algorithm a reviewer can read top to bottom. For one
discrete-choice combo it Euler-inverts the continuation over the savings grid,
appends the closed-form credit-constrained candidates, refines the candidate
value correspondence to its upper envelope, and publishes the value function and
the next period's carry rows on the regime's exogenous state grid. It reads the
expected continuation through `continuation`'s per-target carry reader and knows
nothing about asset-row mode, the multi-target machinery itself, or the kernel
orchestration that maps it over the discrete-combo product.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import (
    ContinuationPlan,
    bind_continuation,
)
from _lcm.egm.euler import invert_euler
from _lcm.egm.interp import (
    interp_on_padded_grid,
)
from _lcm.egm.numeric_inverse import numeric_inverse_marginal_utility
from _lcm.egm.upper_envelope.fues import QueryBracket
from _lcm.typing import (
    ActionName,
    RegimeName,
    StateName,
)
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    UserFunction,
)

# Smallest constrained-segment action as a fraction of the segment's span.
# The constrained candidates are geometrically spaced from this offset toward
# the borrowing limit (additive, so a non-positive limit stays well-defined).
CONSTRAINED_OFFSET_FRACTION = 1e-4


@dataclass(frozen=True, kw_only=True)
class _EgmKernelPieces:
    """Build-time statics shared by every per-combo EGM computation."""

    euler_state_name: StateName
    """Name of the regime's continuous (Euler) state."""

    action_name: ActionName
    """Name of the regime's continuous action."""

    savings_nodes: Float1D
    """The exogenous end-of-period savings grid."""

    borrowing_limit: ScalarFloat
    """Lower bound of the savings grid."""

    n_constrained: int
    """Number of closed-form credit-constrained candidate points."""

    constrained_ratio: float
    """Static geometric spacing ratio of the constrained candidates."""

    n_pad: int
    """Static length of the envelope-refinement workspace (FUES/RFC, overflow)."""

    n_carry_rows: int
    """Static length of the persisted carry rows.

    Equals `n_pad` in the single-post-state kernel (the carry is the refined
    envelope) and `n_euler_nodes` in asset-row mode (the carry is one published
    point per exogenous Euler node, with no envelope-workspace padding)."""

    combo_names: tuple[StateName | ActionName, ...]
    """Discrete-state, passive-state, then discrete-action names (carry-axis order)."""

    euler_axis_in_V: int
    """Canonical axis of the Euler state in the published value-function array."""

    utility_func: UserFunction
    """The regime's concatenated utility function."""

    inverse_marginal_utility_func: UserFunction | None
    """The regime's concatenated inverse-marginal-utility function.

    `None` when the regime supplies no analytic inverse: EGM then inverts `u'`
    numerically (the iEGM path), deriving the marginal utility from `utility_func`
    at the call site.
    """

    own_resources_func: UserFunction
    """The regime's concatenated resources function."""

    feasibility_func: Callable[..., ScalarBool] | None
    """Discrete-feasibility predicate of a combo, or `None`."""

    build_H_kwargs: Callable[[Mapping[str, Any]], dict[str, Any]]
    """Closure assembling the Bellman aggregator's keyword arguments."""

    refine: Callable[..., tuple[Float1D, Float1D, Float1D, ScalarInt, ScalarBool]]
    """The configured upper-envelope backend (single-post-state carry)."""

    refine_to_bracket: Callable[..., QueryBracket]
    """The single-query bracket finder for the asset-row publish (builds the full
    refined row and slices the bracketing node pair)."""

    continuation_plan: ContinuationPlan
    """Build-time statics of the per-savings-node continuation aggregation."""


def _get_solve_one_combo(
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
    tuple[Float1D, Float1D, Float1D, Float1D, Float1D, ScalarBool],
]:
    """Build the per-combo EGM computation for one kernel invocation.

    `euler_batch_size` is accepted for a uniform builder signature but unused:
    the single-post-state kernel solves once per combo, with no per-asset-node
    axis to splay (only the asset-row kernel honors it).

    `resolved_process_grids` maps each runtime-resolved process state to its
    solve-time grid, threaded into the continuation so a resources function
    reading a process node integrates over the resolved nodes.
    """
    del euler_batch_size
    dtype = state_grid.dtype

    def solve_one_combo(
        combo_values: tuple[ScalarInt | ScalarFloat, ...],
    ) -> tuple[Float1D, Float1D, Float1D, Float1D, Float1D, ScalarBool]:
        """Run the EGM step for one (discrete x passive-node) combo.

        Takes the combo's values (discrete codes and passive node values)
        positionally so `jax.vmap` can batch over flattened combo arrays.

        Returns:
            Tuple of the combo's value row on the exogenous state grid, its
            refined endogenous grid, the published consumption policy on that
            grid, the value and marginal-utility carry rows, and the
            backend's read-support verdict (whether the row's linear span is
            free of compacted coverage gaps, so the off-grid simulation read
            may consume it).

        """
        combo_pool = {
            **pool,
            **dict(zip(pieces.combo_names, combo_values, strict=True)),
        }
        # Validation pins the default Bellman aggregator, whose single
        # non-(utility, E_next_V) parameter is the discount factor.
        (discount_factor,) = tuple(pieces.build_H_kwargs(combo_pool).values())

        def utility_of_action(action_value: ScalarFloat) -> ScalarFloat:
            return pieces.utility_func(
                **{pieces.action_name: action_value}, **combo_pool
            )

        compute_node = _get_compute_node(
            pieces=pieces,
            combo_pool=combo_pool,
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

        def own_resources_of_state(state_value: ScalarFloat) -> ScalarFloat:
            return pieces.own_resources_func(
                **{pieces.euler_state_name: state_value}, **combo_pool
            )

        publish_resources = jax.vmap(own_resources_of_state)(state_grid)

        constrained_actions, constrained_values = _compute_constrained_candidates(
            first_endogenous_point=endog_grid[0],
            publish_resources=publish_resources,
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
        # Exogenous source savings per candidate: the savings node for each Euler
        # candidate (`endog_grid = savings_node + action`, so the implied
        # `endog_grid - policy` equals it in exact arithmetic), the borrowing
        # limit for the constrained candidates (their savings is pinned there).
        # FUES compares these pristine sources instead of the lossy implied
        # difference, so a same-source tie is never dropped by rounding.
        candidate_savings = jnp.concatenate(
            [
                jnp.full_like(constrained_actions, pieces.borrowing_limit),
                pieces.savings_nodes,
            ]
        )
        # A `-inf`-valued candidate (e.g. a corner whose continuation is
        # `-inf`) is dominated by every finite candidate and would inject
        # `-inf - (-inf) = NaN` into the envelope scan's gradient arithmetic.
        # Mask the triple to NaN — the scan's absent form. NaN-valued
        # candidates stay as they are: genuine poison must keep propagating.
        candidate_dead = jnp.isneginf(candidate_value)
        candidate_marginal = _candidate_supgradient(
            policy=candidate_policy,
            dead=candidate_dead,
            utility_of_action=utility_of_action,
        )
        # No explicit `segment_id` is passed: the candidate chain is a single
        # connected branch (the constrained segment joins the unconstrained Euler
        # branch continuously at the borrowing limit). Within-period discrete
        # choices are maximized across combos in the V array, not stacked here, so
        # `refine` never sees two unrelated branches that could be bridged. A
        # future-kink-induced fold is data-emergent (the inverted endogenous grid
        # decreases) rather than known a priori, so it is the backend's geometry —
        # not an explicit label — that resolves it. Explicit topology is the
        # oracle/test path (LTM/MSS/query accept `segment_id`); production has none
        # to emit.
        refined_grid, refined_policy, refined_value, n_kept, read_supported = (
            pieces.refine(
                endog_grid=jnp.where(candidate_dead, jnp.nan, candidate_grid),
                policy=jnp.where(candidate_dead, jnp.nan, candidate_policy),
                value=jnp.where(candidate_dead, jnp.nan, candidate_value),
                marginal_utility=candidate_marginal,
            )
        )

        V_row, value_row, marginal_utility_row = _publish_V_and_carry_rows(
            refined_grid=refined_grid,
            refined_policy=refined_policy,
            refined_value=refined_value,
            n_kept=n_kept,
            n_pad=pieces.n_pad,
            publish_resources=publish_resources,
            borrowing_limit=pieces.borrowing_limit,
            utility_of_action=utility_of_action,
            discounted_expected_value_at_limit=discount_factor * expected_values[0],
        )

        # A combo with no live candidate (its entire continuation is `-inf`)
        # is worth `-inf` everywhere, like an infeasible combo.
        no_live_candidate = jnp.all(candidate_dead)
        V_row = jnp.where(no_live_candidate, -jnp.inf, V_row)
        value_row = jnp.where(no_live_candidate, -jnp.inf, value_row)
        marginal_utility_row = jnp.where(no_live_candidate, 0.0, marginal_utility_row)

        if pieces.feasibility_func is not None:
            # Infeasible discrete combos: -inf value rows so they win no
            # maximum and carry zero choice probability; exactly-zero
            # marginal utility so probability-weighted sums stay finite.
            feasible = pieces.feasibility_func(**combo_pool)
            V_row = jnp.where(feasible, V_row, -jnp.inf)
            value_row = jnp.where(feasible, value_row, -jnp.inf)
            marginal_utility_row = jnp.where(feasible, marginal_utility_row, 0.0)

        return (
            V_row,
            refined_grid.astype(dtype),
            refined_policy.astype(dtype),
            value_row,
            marginal_utility_row,
            read_supported,
        )

    return solve_one_combo


def _compute_nodes_over_savings(
    *,
    compute_node: Callable,
    savings_nodes: Float1D,
    savings_batch_size: int,
) -> tuple[FloatND, FloatND, FloatND, FloatND]:
    """Run `compute_node` over every savings node, optionally splayed.

    A positive `savings_batch_size` below the grid length splays the
    per-savings-node continuation computation — the dominant egm_step working
    buffer (savings nodes by the child stochastic mesh by the combo block) — into
    `lax.map` blocks, shedding peak memory; 0 (or a size covering the whole
    grid) keeps the fused vmap. The output is identical either way: the per-node
    `(action, endogenous resources, value, expected continuation)` candidates
    stacked along the savings axis, which the constrained-region assembly and
    the upper envelope then consume on the full grid.
    """
    n_savings = savings_nodes.shape[0]
    if 0 < savings_batch_size < n_savings:
        return jax.lax.map(compute_node, savings_nodes, batch_size=savings_batch_size)
    return jax.vmap(compute_node)(savings_nodes)


def _get_compute_node(
    *,
    pieces: _EgmKernelPieces,
    combo_pool: dict[str, Any],
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
    dtype: Any,  # noqa: ANN401
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
) -> Callable[[ScalarFloat], tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]]:
    """Build the per-savings-node Euler inversion for one discrete combo."""
    continuation = bind_continuation(
        plan=pieces.continuation_plan,
        combo_pool=combo_pool,
        next_regime_to_continuation=next_regime_to_continuation,
        dtype=dtype,
        resolved_process_grids=resolved_process_grids,
    )

    analytic_inverse = pieces.inverse_marginal_utility_func
    if analytic_inverse is not None:

        def inverse_marginal_utility(
            marginal_continuation: ScalarFloat,
        ) -> ScalarFloat:
            return analytic_inverse(
                marginal_continuation=marginal_continuation, **combo_pool
            )
    else:
        # iEGM: no analytic inverse supplied, so invert `u'` numerically. The
        # marginal utility is the action-derivative of the combo-bound utility;
        # the bracket spans a small floor to a generous multiple of the savings
        # grid's top node (interior optima lie within resources, the savings
        # scale; a clamped near-zero-marginal corner whose root exceeds the
        # bracket lands far to the right and is discarded by the envelope, exactly
        # as the analytic path's large value is).
        #
        # `action_upper` must dominate every feasible consumption. It is derived
        # from the savings-grid scale (the model's own resources scale) rather than
        # the resources bound directly, since the per-savings-node kernel has no
        # single current-state resources value here. A model whose optimal
        # consumption can genuinely exceed ~1000x the top savings node (income far
        # above the savings scale) is mis-scaled for this grid and would clamp a
        # real interior root to the bound (reported with `dc/dm = 0`, like a binding
        # corner); widen the savings grid or supply an analytic inverse in that case.
        marginal_utility_of_action = jax.grad(utility_of_action)
        action_upper = pieces.savings_nodes[-1] * 1000.0 + 1000.0
        action_lower = jnp.asarray(1e-8, dtype=action_upper.dtype)

        def inverse_marginal_utility(
            marginal_continuation: ScalarFloat,
        ) -> ScalarFloat:
            return numeric_inverse_marginal_utility(
                marginal_continuation=marginal_continuation,
                marginal_utility=marginal_utility_of_action,
                c_lower=action_lower,
                c_upper=action_upper,
            )

    def compute_node(
        savings_value: ScalarFloat,
    ) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]:
        """Euler-invert one savings node against the continuation."""
        expected_value, expected_marginal = continuation(savings_value)
        action = invert_euler(
            expected_marginal_continuation=expected_marginal,
            discount_factor=discount_factor,
            inverse_marginal_utility=inverse_marginal_utility,
        )
        endog_point = savings_value + action
        value = utility_of_action(action) + discount_factor * expected_value
        return action, endog_point, value, expected_value

    return compute_node


def _candidate_supgradient(
    *,
    policy: Float1D,
    dead: BoolND,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
) -> Float1D:
    """Compute the candidate value-row supgradient $\\mu = \\partial v / \\partial R$.

    By the envelope theorem the value row's slope in resources is the marginal
    utility of consumption $u'(c)$ at the candidate's optimal action — the
    same exact slope the published carry row carries. The upper-envelope
    backend reads it as the tangent gradient at each candidate; FUES ignores
    it. Dead candidates get $0$ (they are dropped before the dominance test).

    Args:
        policy: Candidate policy (consumption) values.
        dead: Indicator of dead candidates (their continuation is `-inf`).
        utility_of_action: Utility with everything but the continuous action
            bound.

    Returns:
        Supgradient per candidate.

    """
    safe_policy = jnp.where(dead, 1.0, policy)
    marginal = jax.vmap(jax.grad(utility_of_action))(safe_policy)
    return jnp.where(dead, 0.0, marginal)


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


def _publish_V_and_carry_rows(
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
) -> tuple[FloatND, Float1D, Float1D]:
    """Interpolate V onto the exogenous grid and finish the carry value rows.

    Below the lowest refined envelope point the interpolant edge-clamps, so
    the published value there is the closed-form constrained value — the
    exact value of saving exactly the borrowing limit. Above it, the refined
    envelope is read with the cubic Hermite interpolant (the marginal-utility
    row is the value row's exact slope by the envelope theorem), floored at
    the constrained value, which remains a feasible-policy lower bound
    everywhere.

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
        Tuple of the value row on the exogenous state grid, the carry value
        row, and the carry marginal-utility row.

    """
    dtype = publish_resources.dtype
    overflowed = n_kept > n_pad

    marginal_utility = jax.vmap(jax.grad(utility_of_action))(
        jnp.where(jnp.isnan(refined_policy), 1.0, refined_policy)
    )
    marginal_utility = jnp.where(jnp.isnan(refined_policy), jnp.nan, marginal_utility)
    marginal_utility = jnp.where(jnp.isneginf(refined_value), 0.0, marginal_utility)

    value_interpolated = interp_on_padded_grid(
        x_query=publish_resources,
        xp=refined_grid,
        fp=refined_value,
        fp_slopes=marginal_utility,
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
    # Below the lowest refined point the interpolant edge-clamps, so the
    # closed-form constrained value (the exact value of saving exactly the
    # borrowing limit) is published outright there. Everywhere else it is a
    # feasible-policy floor under the Hermite read of the refined envelope:
    # forcing it further up — e.g. to the first Euler point — would discard
    # envelope information wherever degenerate inversions push that point
    # right (a zero-ish marginal continuation makes it ~1/eps), and the
    # constrained candidates inside the envelope already carry exact slopes.
    refined_grid_start = refined_grid[0]
    V_row = jnp.where(
        (closed_form_actions > 0.0) & (publish_resources <= refined_grid_start),
        value_constrained,
        jnp.maximum(value_interpolated, value_constrained),
    )
    V_row = jnp.where(overflowed, jnp.nan, V_row).astype(dtype)

    # The carry value row is the refined upper envelope itself — the correct
    # object for a parent to interpolate. The constrained floor applies only to
    # the published `V_row` on the exogenous grid; flooring the carry would lift
    # finite envelope nodes above the true envelope wherever the closed-form
    # constrained value exceeds them, overstating the parent's continuation.
    value_row = jnp.where(overflowed, jnp.nan, refined_value).astype(dtype)
    # The carry marginal-utility row is the parent's Hermite slope. On overflow
    # the backend still returns a finite truncated prefix, so — like the value
    # rows — it must be NaN-poisoned too, or an overflowed period would feed the
    # parent a wrong-but-finite slope past the NaN diagnostics.
    marginal_row = jnp.where(overflowed, jnp.nan, marginal_utility).astype(dtype)
    return V_row, value_row, marginal_row
