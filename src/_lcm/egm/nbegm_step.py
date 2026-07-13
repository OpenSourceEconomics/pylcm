"""One-period NBEGM steps for a 1-D consumption--saving regime.

The step family, and where the solver dispatches each:

- `nbegm_one_asset_step` — a binary *jump* case boundary on the liquid state
  (`liquid < asset_limit`) shifting an additive subsidy into cash-on-hand.
  Dispatched for the direct case-piece split and for a schedule whose single
  breakpoint is a jump.
- `nbegm_multi_interval_step` — a *continuous* piecewise-affine budget (every
  breakpoint a kink or hard-constraint floor, no jump), where `coh(liquid)`
  stays continuous and monotone, so one EGM pass in `coh` space and an inversion
  of `coh(liquid)` recover the whole liquid grid with no interval seam.
- `nbegm_unified_step` — a schedule mixing kinds (jumps and kinks together):
  each continuous run of intervals solves by `coh` inversion and the cases are
  masked across the jumps.
- `nbegm_recurring_jump_step` — a schedule of multiple jump breakpoints (N
  cliffs), resolving every jump with boundary-targeting and jump-aware
  continuation reads.
- `nbegm_multi_interval_step_savings` / `nbegm_unified_step_savings` —
  savings-space variants for the ride-along path, consuming a continuation
  supplied on the exogenous savings grid instead of the next liquid grid.
- `nbegm_per_interval_continuation_step_savings` — the ride-along cell whose
  next-state law reads the liquid state, so the continuation is constant only
  within each declared interval and the step consumes one continuation row per
  interval.
- `nbegm_discrete_envelope_step` — a discrete action shifting a smooth budget:
  the continuous subproblem solves per discrete-action value and the discrete
  choice is taken by the upper envelope over the branch values.

Shared mechanics: within each case/interval the budget is smooth, so the period
solves by ordinary 1-D EGM on cash-on-hand; the recovered endogenous state is the
budget inverse. A case's value is the upper envelope over its candidate branches
(the Euler interior path, boundary-targeting saves that land exactly on the
eligible side of a jump, and the hard borrowing corner). Cases are NaN-dead
masked to the region where their predicate is consistent with the recovered
state and merged by the branch-aware upper envelope; the boundary's
`equality_owner` fixes which side owns the exact boundary point through the
strict/non-strict comparison split.
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from _lcm.egm.euler import invert_euler
from _lcm.egm.ez_kernel import (
    ez_consumption_from_euler,
    ez_marginal_of_resource,
    ez_period_value,
)
from _lcm.egm.nbegm_segments import mask_dead_candidates, segment_ids_from_folds
from _lcm.egm.upper_envelope.query import envelope_at_query
from lcm.case_piece import EqualityOwner
from lcm.typing import BoolND, Float1D, FloatND, IntND, ScalarFloat, ScalarInt

# Below this marginal value of liquid the continuation is treated as flat, so the
# Euler inversion is degenerate (consumption diverges) and the candidate is dropped.
_DEGENERATE_MARGINAL_TOL = 1e-10

# Below this |xi| = |phi (1-rho) - 1| the Euler equation is treated as constant in
# consumption: the closed-form inversion `c = x^(1/xi)` is undefined at xi = 0, so
# the exponent is NaN-poisoned and the solve's NaN fail-fast reports it.
_DEGENERATE_EULER_EXPONENT_TOL = 1e-8

# A budget interval is flat when the consumption floor binds: cash-on-hand is constant
# in the liquid state, so the Euler inversion onto the coh-endogenous grid is degenerate
# and the interval is solved by a dense savings search at the constant budget, not EGM.
# Flatness is classified by the *relative* span of cash-on-hand across the grid — its
# range as a fraction of its magnitude — rather than an absolute slope threshold, so a
# dollar-scaled budget whose slope is tiny but non-zero (the coh grid collapses to a
# constant at working precision) is still recognised as flat.
_FLAT_SPAN_REL_TOL = 1e-6

# Intervals of the per-interval continuation step solve in chunks of this many at a
# time: parallel (vmap) within a chunk, sequential (lax.map) across chunks. Larger
# chunks run more intervals in parallel — trading peak memory (a chunk's intermediates
# materialize together) for a shallower sequential loop. A small value keeps the
# vmap memory bounded while still cutting the sequential depth to `n_intervals /
# _CHUNK_SIZE`.
_CHUNK_SIZE = 4


def nbegm_multi_interval_step(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    gross_return: ScalarFloat | float,
    income: ScalarFloat | float,
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    flat_interval_mask: tuple[bool, ...] | None = None,
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one period of a piecewise-affine, continuous-budget regime by EGM.

    The breakpoints partition the liquid axis into intervals on which cash-on-hand
    is affine, `coh = slope_i * liquid + intercept_i`. With a continuous budget
    (every breakpoint a kink, no jump), `coh(liquid)` is continuous and monotone, so
    its inverse is a single continuous map. EGM therefore runs once in `coh` space —
    `coh = consumption + savings` — and the endogenous liquid is recovered by
    inverting `coh(liquid)` on the grid, leaving no interval seam to under-cover. The
    marginal value of liquid scales by the active interval's slope (the envelope
    theorem through the affine budget).

    A slope-0 interval is a hard-constraint floor: cash-on-hand is pinned at the
    floor for every liquid in it, so the value is constant — the floor's own
    optimum. The coh inversion is degenerate there, so a flat interval's interior
    candidates are pulled onto its crossing breakpoint (where they link to the
    rising interior just above) and a flat corner segment carries the constant value
    across the interval below. A savings-node corner chain per post-decision node
    competes over the whole grid (the `s = 0` chain is the hard borrowing corner),
    and the interior path, flat corners, and node chains merge by the branch-aware
    upper envelope.

    Args:
        next_value: Next period's value on `liquid_grid`.
        next_marginal: Next period's marginal value of liquid on `liquid_grid`.
        liquid_grid: Regular liquid-state grid (ascending).
        savings_grid: Post-decision savings grid `s = coh - consumption` (>= 0).
        discount_factor: Discount factor.
        crra: Coefficient of relative risk aversion.
        gross_return: Gross liquid return `1 + r`.
        income: Deterministic income added to next-period liquid.
        coh_slopes: Per-interval cash-on-hand slope in liquid, length N+1.
        coh_intercepts: Per-interval cash-on-hand intercept, length N+1.
        breakpoints: Sorted ascending liquid breakpoints, length N.
        flat_interval_mask: Static per-interval flag, length N+1, marking the
            hard-constraint (slope-0) floor intervals. `None` means no floor.

    Returns:
        Tuple of this period's value, marginal value of liquid, and consumption
        policy, each on `liquid_grid`.

    """
    interval_of_grid = jnp.searchsorted(breakpoints, liquid_grid, side="right")
    coh_grid = (
        coh_slopes[interval_of_grid] * liquid_grid + coh_intercepts[interval_of_grid]
    )

    next_liquid = gross_return * savings_grid + income
    value_next = jnp.interp(next_liquid, liquid_grid, next_value)
    marginal_next = jnp.interp(next_liquid, liquid_grid, next_marginal)

    consumption = (discount_factor * gross_return * marginal_next) ** (-1.0 / crra)
    coh_endog = consumption + savings_grid
    # An endogenous cash-on-hand beyond the grid's coh range inverts to a liquid
    # beyond the grid; the boundary segments' slopes continue the inversion there,
    # so the branch's last live link still brackets the boundary query points.
    liquid_endog = _invert_coh_with_linear_extension(
        coh_endog=coh_endog, coh_case_grid=coh_grid, liquid_grid=liquid_grid
    )
    # Marginal value of liquid = u'(c) * d coh / d liquid; the slope is the active
    # interval's at the recovered liquid (envelope theorem through the budget).
    endog_interval = jnp.searchsorted(breakpoints, liquid_endog, side="right")
    slope_endog = coh_slopes[endog_interval]
    value_endog = _crra_utility(consumption, crra) + discount_factor * value_next
    marginal_endog = slope_endog * consumption ** (-crra)

    upper_edges = jnp.concatenate([breakpoints, liquid_grid[-1:]])
    lower_edges = jnp.concatenate([liquid_grid[:1], breakpoints])
    flat_indices = _flat_interval_indices(
        flat_interval_mask, n_intervals=coh_slopes.shape[0]
    )
    # Pull each flat interval's degenerate interior candidates onto its crossing
    # breakpoint at the floor's own optimum, where they link to the rising interior
    # just above; a flat corner below carries the constant value across the interval.
    # The floor's optimum is read by a dense consumption search at `coh = floor` —
    # robust to a recurring flat continuation, where the Euler inversion is degenerate.
    flat_corners: list[tuple[Float1D, Float1D, Float1D, Float1D]] = []
    for i in flat_indices:
        floor_value, floor_policy = _floor_optimum(
            floor_coh=coh_intercepts[i],
            liquid_grid=liquid_grid,
            next_value=next_value,
            discount_factor=discount_factor,
            crra=crra,
            gross_return=gross_return,
            income=income,
        )
        in_flat = endog_interval == i
        liquid_endog = jnp.where(in_flat, upper_edges[i], liquid_endog)
        value_endog = jnp.where(in_flat, floor_value, value_endog)
        consumption = jnp.where(in_flat, floor_policy, consumption)
        marginal_endog = jnp.where(in_flat, 0.0, marginal_endog)
        flat_corners.append(
            (
                jnp.stack([lower_edges[i], upper_edges[i]]),
                jnp.full((2,), floor_value),
                jnp.full((2,), floor_policy),
                jnp.zeros((2,)),
            )
        )
    # Degenerate-inversion guard: where the continuation is flat (a recurring floor
    # gives zero marginal value of liquid), the Euler inversion sends consumption to
    # infinity, so those interior candidates are spurious. Drop the ones not already
    # pulled onto a floor crossing; the floor corner and the s=0 corner cover them.
    degenerate = marginal_next <= _DEGENERATE_MARGINAL_TOL
    in_any_flat = jnp.isin(endog_interval, jnp.asarray(flat_indices, dtype=jnp.int32))
    liquid_endog = jnp.where(degenerate & ~in_any_flat, jnp.nan, liquid_endog)
    # A non-concave (convex-kinked) budget can fold the interior path back, so keep
    # its monotone runs apart for the upper envelope.
    interior_segment = segment_ids_from_folds(endog_grid=liquid_endog)
    next_segment = jnp.nanmax(interior_segment) + 1.0

    endog_parts: list[Float1D] = [liquid_endog]
    value_parts: list[Float1D] = [value_endog]
    policy_parts: list[Float1D] = [consumption]
    marginal_parts: list[Float1D] = [marginal_endog]
    segment_parts: list[Float1D] = [interior_segment]

    # Hard borrowing corner: save nothing, consume all of this point's cash-on-hand,
    # land next-period liquid at `income`. A candidate over the whole grid, since the
    # constraint binds wherever the no-save corner beats the Euler path.
    value_at_income = jnp.interp(jnp.asarray(income), liquid_grid, next_value)
    endog_parts.append(liquid_grid)
    value_parts.append(
        _crra_utility(coh_grid, crra) + discount_factor * value_at_income
    )
    policy_parts.append(coh_grid)
    marginal_parts.append(coh_slopes[interval_of_grid] * coh_grid ** (-crra))
    segment_parts.append(jnp.full_like(liquid_grid, next_segment))

    for offset, (edges, values, policies, marginals) in enumerate(flat_corners):
        endog_parts.append(edges)
        value_parts.append(values)
        policy_parts.append(policies)
        marginal_parts.append(marginals)
        segment_parts.append(jnp.full((2,), next_segment + 1.0 + float(offset)))

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.concatenate(endog_parts),
        policy=jnp.concatenate(policy_parts),
        value=jnp.concatenate(value_parts),
        marginal=jnp.concatenate(marginal_parts),
        segment_id=jnp.concatenate(segment_parts),
        x_query=liquid_grid,
    )
    return value, marginal, policy


def _invert_coh_with_linear_extension(
    *, coh_endog: Float1D, coh_case_grid: Float1D, liquid_grid: Float1D
) -> Float1D:
    """Invert the case's monotone coh map, extending linearly past its range.

    `jnp.interp` clips an endogenous coh beyond the sampled range onto the
    boundary grid node, which strands the branch's last live link inside the
    grid span and leaves the boundary query points unbracketed — the envelope
    then falls back to a corner there. Continuing the boundary segments'
    slopes keeps the branch extending past the first/last grid point; the
    per-case interval mask still bounds where its candidates may live.
    """
    inner = jnp.interp(coh_endog, coh_case_grid, liquid_grid)
    lower_width = coh_case_grid[1] - coh_case_grid[0]
    upper_width = coh_case_grid[-1] - coh_case_grid[-2]
    lower_slope = (liquid_grid[1] - liquid_grid[0]) / jnp.where(
        lower_width > 0.0, lower_width, 1.0
    )
    upper_slope = (liquid_grid[-1] - liquid_grid[-2]) / jnp.where(
        upper_width > 0.0, upper_width, 1.0
    )
    below = liquid_grid[0] + (coh_endog - coh_case_grid[0]) * lower_slope
    above = liquid_grid[-1] + (coh_endog - coh_case_grid[-1]) * upper_slope
    below = jnp.where(lower_width > 0.0, below, inner)
    above = jnp.where(upper_width > 0.0, above, inner)
    return jnp.where(
        coh_endog < coh_case_grid[0],
        below,
        jnp.where(coh_endog > coh_case_grid[-1], above, inner),
    )


def _invert_euler_over_savings(
    *,
    cont_marginal: Float1D,
    discount_factor: ScalarFloat,
    inverse_marginal_utility: Callable[[ScalarFloat], ScalarFloat],
) -> Float1D:
    """Recover the consumption action at every savings node via the Euler equation.

    The expected marginal continuation is already in savings space (it carries the
    gross-return factor `dR/ds`), so each node inverts the regime's marginal utility
    at the discounted expected marginal continuation — `invert_euler` applies the
    degenerate-inversion clamp before calling `inverse_marginal_utility`.
    """

    def invert_one(node_marginal: ScalarFloat) -> ScalarFloat:
        return invert_euler(
            expected_marginal_continuation=node_marginal,
            discount_factor=discount_factor,
            inverse_marginal_utility=inverse_marginal_utility,
        )

    return jax.vmap(invert_one)(cont_marginal)


def _ez_flow_power_structure(
    *,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    inverse_eis: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Log-domain Euler-inversion coefficients for a single-power period flow.

    The Epstein-Zin closed-form Euler inversion needs the flow's marginal as a
    single power of consumption, `q^(-rho) q_c = kappa * c^flow_exponent`. For
    any constant-elasticity flow `q(c) = A c^phi` — the basic good `q = c`, and
    the fixed-service Cobb-Douglas `q = c^phi s^(1-phi)` with `s` the outer
    durable fixed per outer node — read `A = q(1)` and the elasticity
    `phi = q'(1) / A` from the regime's own flow, then
    `log(kappa) = (1-rho) log(A) + log(phi)` and
    `flow_exponent = phi (1-rho) - 1`. The coefficient stays in log form
    throughout: the raw `A^(1-rho)` leaves the dtype's range long before the
    inverted consumption does. Reduces to `(0, -rho)` for the basic flow
    (`A = phi = 1`). A nonpositive `A` or `phi` (excluded by the build probe,
    but runtime parameters can recreate it) reads NaN and poisons the solve.

    Valid only for a flow with a constant consumption elasticity (a single power of
    `c`); a flow whose elasticity varies with `c` (e.g. a nested CES) needs a
    numeric inverse instead.
    """
    one = jnp.asarray(1.0)
    flow_scale = utility_of_action(one)
    flow_power = jax.grad(utility_of_action)(one) / flow_scale
    one_minus_rho = 1.0 - inverse_eis
    log_flow_coefficient = one_minus_rho * jnp.log(flow_scale) + jnp.log(flow_power)
    raw_exponent = flow_power * one_minus_rho - 1.0
    # At `xi = phi (1-rho) - 1 = 0` the Euler equation is constant in
    # consumption and the closed-form inversion `c = x^(1/xi)` is undefined.
    # Both `phi` and `rho` are runtime parameters, so the degenerate
    # combination cannot be rejected at model build; poison the exponent with
    # NaN so the solve's NaN fail-fast surfaces the (regime, period) instead
    # of the inversion computing a finite but meaningless consumption.
    flow_exponent = jnp.where(
        jnp.abs(raw_exponent) < _DEGENERATE_EULER_EXPONENT_TOL,
        jnp.nan,
        raw_exponent,
    )
    return log_flow_coefficient, flow_exponent


def nbegm_multi_interval_step_savings(
    *,
    cont_value: Float1D,
    cont_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    inverse_marginal_utility: Callable[[ScalarFloat], ScalarFloat],
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    inverse_eis: ScalarFloat | None = None,
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve a continuous piecewise-affine regime against savings-space continuation.

    Same EGM geometry as `nbegm_multi_interval_step`, but the continuation is
    supplied as the expected value and expected marginal continuation already
    evaluated at each post-decision savings node — the transition-aware
    continuation reader has integrated the next-period regime transition,
    stochastic shocks, and ride-along co-state transitions and returned both in
    savings space. The expected marginal therefore already carries the
    gross-return factor `dR/ds`, so the Euler inversion reads it directly with no
    explicit return term.

    The Euler inversion, the period value, and the marginal value of liquid all
    read the regime's own utility: the consumption action solving the savings-node
    Euler equation comes from `inverse_marginal_utility` (the regime's analytic
    inverse, or a numeric inversion of `u'` built from the utility), the value adds
    `utility_of_action(consumption)`, and the marginal value of liquid is the
    cash-on-hand slope times `u'(consumption)` by the envelope theorem.

    The hard-borrowing corner (save nothing) lands on the lowest savings node, so
    its continuation value is `cont_value[0]` (the expectation at `savings = 0`).
    The flat-interval (hard-constraint floor) case is not covered here; a regime
    with a ride-along co-state and a floor breakpoint is a later slice.

    Args:
        cont_value: Expected continuation value at each savings node.
        cont_marginal: Expected marginal continuation (savings space) at each
            savings node.
        liquid_grid: Regular liquid-state grid (ascending).
        savings_grid: Post-decision savings grid `s = coh - consumption` (>= 0,
            with `savings_grid[0] == 0` the no-save corner).
        discount_factor: Discount factor.
        utility_of_action: The regime's period utility as a function of consumption,
            with the ride-along cell's states and the utility params already bound.
        inverse_marginal_utility: The regime's inverse marginal utility as a function
            of the discounted expected marginal continuation, with the cell already
            bound — `invert_euler` calls it to recover the consumption action.
        coh_slopes: Per-interval cash-on-hand slope in liquid, length N+1.
        coh_intercepts: Per-interval cash-on-hand intercept, length N+1.
        breakpoints: Sorted ascending liquid breakpoints, length N.

    Returns:
        Tuple of this period's value, marginal value of liquid, and consumption
        policy, each on `liquid_grid`.

    """
    interval_of_grid = jnp.searchsorted(breakpoints, liquid_grid, side="right")
    coh_grid = (
        coh_slopes[interval_of_grid] * liquid_grid + coh_intercepts[interval_of_grid]
    )

    # The expected marginal is already in savings space (it carries `dR/ds`), so
    # the Euler inversion reads it directly — no explicit gross-return factor.
    # Under Epstein-Zin the continuation pair is `(nu, dnu/ds)` and the inversion,
    # period value, and envelope marginal read the recursive aggregator instead of
    # the additive `u + beta E[V']` form.
    marginal_utility = jax.grad(utility_of_action)
    if inverse_eis is not None:
        log_flow_coefficient, flow_exponent = _ez_flow_power_structure(
            utility_of_action=utility_of_action, inverse_eis=inverse_eis
        )
        consumption = ez_consumption_from_euler(
            nu=cont_value,
            dnu_ds=cont_marginal,
            discount_factor=discount_factor,
            inverse_eis=inverse_eis,
            log_flow_coefficient=log_flow_coefficient,
            flow_exponent=flow_exponent,
        )
    else:
        consumption = _invert_euler_over_savings(
            cont_marginal=cont_marginal,
            discount_factor=discount_factor,
            inverse_marginal_utility=inverse_marginal_utility,
        )
    coh_endog = consumption + savings_grid
    liquid_endog = _invert_coh_with_linear_extension(
        coh_endog=coh_endog, coh_case_grid=coh_grid, liquid_grid=liquid_grid
    )
    endog_interval = jnp.searchsorted(breakpoints, liquid_endog, side="right")
    slope_endog = coh_slopes[endog_interval]
    if inverse_eis is not None:
        value_endog = ez_period_value(
            flow=jax.vmap(utility_of_action)(consumption),
            nu=cont_value,
            discount_factor=discount_factor,
            inverse_eis=inverse_eis,
        )
        marginal_endog = slope_endog * ez_marginal_of_resource(
            log_flow_marginal=log_flow_coefficient
            + flow_exponent * jnp.log(consumption),
            value=value_endog,
            discount_factor=discount_factor,
            inverse_eis=inverse_eis,
        )
    else:
        value_endog = (
            jax.vmap(utility_of_action)(consumption) + discount_factor * cont_value
        )
        marginal_endog = slope_endog * jax.vmap(marginal_utility)(consumption)

    # Where the continuation is flat (zero marginal value of liquid), the Euler
    # inversion diverges; drop those interior candidates.
    degenerate = cont_marginal <= _DEGENERATE_MARGINAL_TOL
    liquid_endog = jnp.where(degenerate, jnp.nan, liquid_endog)
    interior_segment = segment_ids_from_folds(endog_grid=liquid_endog)
    next_segment = jnp.nanmax(interior_segment) + 1.0

    # Savings-node corner chains: for every post-decision node `s_i`, consume
    # `coh - s_i` at each liquid grid point and earn that node's continuation. The
    # family is a dense Bellman floor on the savings grid at every query point, so
    # a continuation kink between Euler roots (a child-value interpolation node,
    # where the inversion has no root) still gets a candidate; the `s = 0` chain
    # is the hard borrowing corner. One segment id per node keeps chains apart.
    node_consumption = coh_grid[None, :] - savings_grid[:, None]
    node_feasible = node_consumption > 0.0
    node_consumption_safe = jnp.where(node_feasible, node_consumption, 1.0)
    if inverse_eis is not None:
        node_value_safe = ez_period_value(
            flow=jax.vmap(jax.vmap(utility_of_action))(node_consumption_safe),
            nu=cont_value[:, None],
            discount_factor=discount_factor,
            inverse_eis=inverse_eis,
        )
        node_marginal_safe = ez_marginal_of_resource(
            log_flow_marginal=log_flow_coefficient
            + flow_exponent * jnp.log(node_consumption_safe),
            value=node_value_safe,
            discount_factor=discount_factor,
            inverse_eis=inverse_eis,
        )
    else:
        node_value_safe = (
            jax.vmap(jax.vmap(utility_of_action))(node_consumption_safe)
            + discount_factor * cont_value[:, None]
        )
        node_marginal_safe = jax.vmap(jax.vmap(marginal_utility))(node_consumption_safe)
    node_value = jnp.where(node_feasible, node_value_safe, jnp.nan)
    node_endog = jnp.where(
        node_feasible,
        jnp.broadcast_to(liquid_grid, node_consumption.shape),
        jnp.nan,
    )
    node_marginal = coh_slopes[interval_of_grid][None, :] * node_marginal_safe
    node_segment = jnp.broadcast_to(
        next_segment
        + jnp.arange(savings_grid.shape[0], dtype=liquid_grid.dtype)[:, None],
        node_consumption.shape,
    )
    endog_parts = [liquid_endog, node_endog.ravel()]
    value_parts = [value_endog, node_value.ravel()]
    policy_parts = [consumption, node_consumption_safe.ravel()]
    marginal_parts = [marginal_endog, node_marginal.ravel()]
    segment_parts = [interior_segment, node_segment.ravel()]

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.concatenate(endog_parts),
        policy=jnp.concatenate(policy_parts),
        value=jnp.concatenate(value_parts),
        marginal=jnp.concatenate(marginal_parts),
        segment_id=jnp.concatenate(segment_parts),
        x_query=liquid_grid,
    )
    return value, marginal, policy


def _interval_corner_candidates(
    *,
    corner_coh_grid: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    lower: ScalarFloat,
    upper: ScalarFloat,
    flat: BoolND,
    value_at_no_save: ScalarFloat,
    interval_value: Float1D,
    coh_slope: ScalarFloat,
    coh_intercept: ScalarFloat,
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    marginal_utility: Callable[[ScalarFloat], ScalarFloat],
    base: ScalarFloat,
    next_segment: ScalarFloat,
) -> tuple[Float1D, ...]:
    """Build one interval's no-save and upper-savings corner candidates.

    Both corners consume `corner_coh_grid` — the true per-grid-point cash-on-hand — so
    a corner is always a real feasible action's value, robust to an interval whose
    recovered affine budget extrapolates below zero where a kink binds only in part of
    the interval.

    Both corners span the interval's liquid range and are each duplicated into an
    interleaved self-segment pair, so a corner in an interval that holds a single liquid
    grid point stays visible to the link-only envelope: the zero-width pair `(p, p)` is
    a segment the envelope brackets exactly at `q == p`. Distinct segment ids keep the
    two corners from cross-linking.

    Returns the flattened `(endog, value, policy, marginal, segment)` columns of the
    no-save corner followed by those of the upper-savings corner.
    """
    in_interval = (liquid_grid >= lower) & (liquid_grid < upper)

    # No-save / floor corner (`s = 0`). Where the budget is flat the consumption floor
    # binds, cash-on-hand is the constant `coh_intercept` for every liquid, so the value
    # is the dense Bellman max over savings at the constant budget, robust to the
    # degenerate Euler inversion. Where the budget slopes it is the ordinary no-save
    # candidate `u(coh(liquid)) + beta * V'(0)`.
    floor_consumption = coh_intercept - savings_grid
    floor_feasible = floor_consumption > 0.0
    floor_node_value = jnp.where(
        floor_feasible,
        jax.vmap(utility_of_action)(jnp.where(floor_feasible, floor_consumption, 1.0))
        + discount_factor * interval_value,
        -jnp.inf,
    )
    best_floor = jnp.argmax(floor_node_value)
    # A no-save corner consumes the whole cash-on-hand. `corner_coh_grid` is the true
    # per-grid-point cash-on-hand, so consumption is feasible (positive) at every grid
    # point where the budget is defined; the positivity guard drops any point where an
    # undeclared kink still leaves it non-positive rather than letting `u(<=0)` = NaN
    # leak into the envelope as a live candidate.
    s0_consumption_safe = jnp.where(corner_coh_grid > 0.0, corner_coh_grid, 1.0)
    s0 = mask_dead_candidates(
        endog_grid=liquid_grid,
        value=jnp.where(
            flat,
            floor_node_value[best_floor],
            jax.vmap(utility_of_action)(s0_consumption_safe)
            + discount_factor * value_at_no_save,
        ),
        policy=jnp.where(flat, floor_consumption[best_floor], corner_coh_grid),
        marginal=coh_slope * jax.vmap(marginal_utility)(s0_consumption_safe),
        valid=in_interval & (flat | (corner_coh_grid > 0.0)),
    )

    # Upper-savings corner (`s = savings_grid[-1]`). With a finite savings grid the
    # finite-domain Bellman optimum at high cash-on-hand can be to save the grid maximum
    # rather than at an interior Euler point, above where the recovered endogenous grid
    # reaches. Feasible only where residual consumption is positive; redundant on a flat
    # interval, whose dense floor search already spans the whole savings grid.
    smax_consumption = corner_coh_grid - savings_grid[-1]
    smax_feasible = (smax_consumption > 0.0) & (~flat) & in_interval
    smax_consumption_safe = jnp.where(smax_feasible, smax_consumption, 1.0)
    smax = mask_dead_candidates(
        endog_grid=liquid_grid,
        value=jax.vmap(utility_of_action)(smax_consumption_safe)
        + discount_factor * interval_value[-1],
        policy=smax_consumption,
        marginal=coh_slope * jax.vmap(marginal_utility)(smax_consumption_safe),
        valid=smax_feasible,
    )

    s0_pair = tuple(jnp.repeat(channel, 2) for channel in s0)
    smax_pair = tuple(jnp.repeat(channel, 2) for channel in smax)
    s0_segment = jnp.repeat(jnp.full_like(liquid_grid, base + next_segment), 2)
    smax_segment = jnp.repeat(jnp.full_like(liquid_grid, base + next_segment + 1.0), 2)
    return (*s0_pair, s0_segment, *smax_pair, smax_segment)


def nbegm_per_interval_continuation_step_savings(
    *,
    cont_value: FloatND,
    cont_marginal: FloatND,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    inverse_marginal_utility: Callable[[ScalarFloat], ScalarFloat],
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    coh_grid: Float1D | None = None,
    envelope_segment_block_size: int = 0,
    extra_savings: FloatND | None = None,
    extra_cont_value: FloatND | None = None,
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve a budget whose continuation differs per liquid interval.

    When the next-period state law carries a current-asset boundary — a transfer,
    eligibility scale, or tax-rate term that is piecewise-constant in the current
    liquid state through the declared cliffs — the expected continuation is itself
    piecewise-constant-shifted across intervals: within interval `k` the boundary
    term is the constant the cliff partition fixes, so `next_liquid(savings)` and
    hence the continuation `E[V'(next_liquid)]` take interval `k`'s form. The caller
    therefore supplies one continuation row per interval (the boundary term bound to
    that interval's value), and each interval is solved as its own case: the EGM
    inverts interval `k`'s continuous `coh(liquid)` against interval `k`'s
    continuation, masks to `[breakpoint_{k-1}, breakpoint_k)`, and adds its
    hard-borrowing corner. The cases plus corners merge by the branch-aware upper
    envelope.

    Args:
        cont_value: Expected continuation value per interval and savings node,
            shaped `(n_intervals, n_savings)`.
        cont_marginal: Expected marginal continuation (savings space) per interval
            and savings node, shaped `(n_intervals, n_savings)`.
        liquid_grid: Regular liquid-state grid (ascending).
        savings_grid: Post-decision savings grid `s = coh - consumption` (>= 0,
            with `savings_grid[0] == 0` the no-save corner).
        discount_factor: Discount factor.
        utility_of_action: The regime's period utility as a function of consumption,
            with the ride-along cell's states and the utility params already bound.
        inverse_marginal_utility: The regime's inverse marginal utility, cell-bound.
        coh_slopes: Per-interval cash-on-hand slope in liquid, length N+1.
        coh_intercepts: Per-interval cash-on-hand intercept, length N+1.
        breakpoints: Sorted ascending liquid breakpoints, length N.
        coh_grid: True cash-on-hand at each liquid grid point, length equal to
            `liquid_grid`. The corners consume this instead of the recovered affine
            budget, so a corner stays a real feasible action where an undeclared kink
            (a consumption floor binding only in part of an interval) makes the affine
            cash-on-hand extrapolate below zero. When `None`, the corners fall back to
            the per-interval affine budget — exact whenever the budget is smooth across
            the whole interval.
        envelope_segment_block_size: Streams the merged upper envelope over
            candidate-segment blocks of this size instead of materialising the full
            `(n_query, n_segment)` bracket matrix; `0` keeps the one-shot dense
            envelope. The result is identical either way — the knob trades peak
            memory against a sequential scan.

    Returns:
        Tuple of this period's value, marginal value of liquid, and consumption
        policy, each on `liquid_grid`.

    """
    n_intervals = coh_slopes.shape[0]
    marginal_utility = jax.grad(utility_of_action)
    interval_stride = 4 * (savings_grid.shape[0] + liquid_grid.shape[0])

    # Each interval's lower/upper liquid bound, with the open ends at the extremes.
    edge = jnp.array([jnp.inf], dtype=breakpoints.dtype)
    lowers = jnp.concatenate([-edge, breakpoints])
    uppers = jnp.concatenate([breakpoints, edge])

    def solve_interval(
        interval_index: ScalarInt,
        interval_value: Float1D,
        interval_marginal: Float1D,
        coh_slope: ScalarFloat,
        coh_intercept: ScalarFloat,
        lower: ScalarFloat,
        upper: ScalarFloat,
    ) -> tuple[Float1D, ...]:
        """Solve one interval's EGM case and its no-save corner.

        Returns the interior candidate (endog grid, value, policy, marginal,
        segment id) followed by the s=0 corner candidate, each on its own grid.
        The segment ids are offset by `interval_index * interval_stride` so the
        per-interval folds stay disjoint when the cases merge under one envelope.
        """
        base = interval_index.astype(jnp.result_type(float)) * interval_stride
        consumption = _invert_euler_over_savings(
            cont_marginal=interval_marginal,
            discount_factor=discount_factor,
            inverse_marginal_utility=inverse_marginal_utility,
        )
        coh_endog = consumption + savings_grid
        interp_value = (
            jax.vmap(utility_of_action)(consumption) + discount_factor * interval_value
        )
        value_at_no_save = interval_value[0]
        degenerate = interval_marginal <= _DEGENERATE_MARGINAL_TOL

        coh_case_grid = coh_slope * liquid_grid + coh_intercept
        coh_scale = jnp.maximum(1.0, jnp.max(jnp.abs(coh_case_grid)))
        flat = (
            jnp.max(coh_case_grid) - jnp.min(coh_case_grid)
        ) <= _FLAT_SPAN_REL_TOL * coh_scale

        # A flat interval's `coh_case_grid` is constant, and `jnp.interp` needs a
        # strictly increasing `xp`; guard the recovered liquid to a finite in-interval
        # value there so no NaN endog leaks into the envelope (interior killed below).
        liquid_endog = jnp.where(
            flat,
            jnp.full_like(coh_endog, lower),
            _invert_coh_with_linear_extension(
                coh_endog=coh_endog,
                coh_case_grid=coh_case_grid,
                liquid_grid=liquid_grid,
            ),
        )
        marginal_endog = coh_slope * jax.vmap(marginal_utility)(consumption)
        in_case = (
            (liquid_endog >= lower) & (liquid_endog < upper) & (~degenerate) & (~flat)
        )
        interior = mask_dead_candidates(
            endog_grid=liquid_endog,
            value=interp_value,
            policy=consumption,
            marginal=marginal_endog,
            valid=in_case,
        )
        segment = segment_ids_from_folds(endog_grid=interior[0])
        # A flat interval kills every interior candidate, so `segment` is all-NaN and
        # `nanmax` would be NaN; fall back to `0` there so the live corners still get a
        # finite segment id.
        segment_max = jnp.nanmax(segment)
        next_segment = jnp.where(jnp.isnan(segment_max), 0.0, segment_max) + 1.0

        # Corners consume the true cash-on-hand at each grid point when supplied, so a
        # no-save/upper-savings corner stays feasible where the interval's affine budget
        # extrapolates below zero; without it they fall back to the affine budget.
        corner_coh_grid = coh_case_grid if coh_grid is None else coh_grid
        corners = _interval_corner_candidates(
            corner_coh_grid=corner_coh_grid,
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            lower=lower,
            upper=upper,
            flat=flat,
            value_at_no_save=value_at_no_save,
            interval_value=interval_value,
            coh_slope=coh_slope,
            coh_intercept=coh_intercept,
            discount_factor=discount_factor,
            utility_of_action=utility_of_action,
            marginal_utility=marginal_utility,
            base=base,
            next_segment=next_segment,
        )
        return (
            interior[0],
            interior[1],
            interior[2],
            interior[3],
            segment + base,
            *corners,
        )

    # Solve the intervals in chunks of `_CHUNK_SIZE`: `lax.map` runs the chunks
    # sequentially (so at most one chunk's intermediates materialize at once,
    # bounding peak memory), while a `vmap` inside each chunk solves its intervals in
    # parallel. The chunk body traces once, keeping the HLO small, and the sequential
    # depth drops from `n_intervals` to `ceil(n_intervals / _CHUNK_SIZE)`.
    #
    # `n_intervals` need not divide `_CHUNK_SIZE`, so the inputs are padded up to a
    # whole number of chunks. Each padding lane carries a global interval index of
    # `n_intervals` or above (a unique segment offset) and `lower == upper == +inf`,
    # so both its `in_case` and `s0_valid` masks are all-False and every one of its
    # candidates is NaN-dead — the padding contributes nothing live to the envelope.
    n_chunks = -(-n_intervals // _CHUNK_SIZE)
    n_padded = n_chunks * _CHUNK_SIZE
    pad = n_padded - n_intervals

    interval_indices = jnp.arange(n_padded, dtype=jnp.int32)
    edge = jnp.array([jnp.inf], dtype=breakpoints.dtype)
    padded_cont_value = jnp.concatenate(
        [cont_value, jnp.zeros((pad, cont_value.shape[1]), dtype=cont_value.dtype)]
    )
    padded_cont_marginal = jnp.concatenate(
        [
            cont_marginal,
            jnp.zeros((pad, cont_marginal.shape[1]), dtype=cont_marginal.dtype),
        ]
    )
    padded_coh_slopes = jnp.concatenate(
        [coh_slopes, jnp.ones((pad,), dtype=coh_slopes.dtype)]
    )
    padded_coh_intercepts = jnp.concatenate(
        [coh_intercepts, jnp.zeros((pad,), dtype=coh_intercepts.dtype)]
    )
    padded_lowers = jnp.concatenate([lowers, jnp.broadcast_to(edge, (pad,))])
    padded_uppers = jnp.concatenate([uppers, jnp.broadcast_to(edge, (pad,))])

    def solve_chunk(packed: tuple[IntND | FloatND, ...]) -> tuple[FloatND, ...]:
        """Solve one chunk of intervals in parallel with `vmap`."""
        return jax.vmap(solve_interval)(*packed)

    def to_chunks(array: IntND | FloatND) -> IntND | FloatND:
        return array.reshape((n_chunks, _CHUNK_SIZE, *array.shape[1:]))

    (
        int_endog,
        int_value,
        int_policy,
        int_marginal,
        int_segment,
        s0_endog,
        s0_value,
        s0_policy,
        s0_marginal,
        s0_segment,
        smax_endog,
        smax_value,
        smax_policy,
        smax_marginal,
        smax_segment,
    ) = jax.lax.map(
        solve_chunk,
        (
            to_chunks(interval_indices),
            to_chunks(padded_cont_value),
            to_chunks(padded_cont_marginal),
            to_chunks(padded_coh_slopes),
            to_chunks(padded_coh_intercepts),
            to_chunks(padded_lowers),
            to_chunks(padded_uppers),
        ),
    )

    node_endog, node_value, node_policy, node_marginal, node_segment = (
        _savings_node_point_candidates(
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            cont_value=cont_value,
            discount_factor=discount_factor,
            utility_of_action=utility_of_action,
            marginal_utility=marginal_utility,
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=breakpoints,
            coh_grid=coh_grid,
            segment_base=float(n_padded * interval_stride),
        )
    )

    # Save-to-cliff point candidates: interval-dependent one-sided savings
    # targets with the continuation's exact one-sided values, offered at every
    # liquid grid point next to the dense savings-node floor.
    if extra_savings is not None and extra_cont_value is not None:
        cliff_parts = _savings_node_point_candidates(
            liquid_grid=liquid_grid,
            savings_grid=extra_savings,
            cont_value=extra_cont_value,
            discount_factor=discount_factor,
            utility_of_action=utility_of_action,
            marginal_utility=marginal_utility,
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=breakpoints,
            coh_grid=coh_grid,
            segment_base=float((n_padded + 1) * interval_stride),
        )
    else:
        cliff_parts = (jnp.empty(0),) * 5

    def stack(*parts: FloatND) -> Float1D:
        return jnp.concatenate([part.reshape(-1) for part in parts])

    value, policy, marginal = envelope_at_query(
        endog_grid=stack(int_endog, s0_endog, smax_endog, node_endog, cliff_parts[0]),
        policy=stack(int_policy, s0_policy, smax_policy, node_policy, cliff_parts[2]),
        value=stack(int_value, s0_value, smax_value, node_value, cliff_parts[1]),
        marginal=stack(
            int_marginal, s0_marginal, smax_marginal, node_marginal, cliff_parts[3]
        ),
        segment_id=stack(
            int_segment, s0_segment, smax_segment, node_segment, cliff_parts[4]
        ),
        x_query=liquid_grid,
        segment_block_size=envelope_segment_block_size,
    )
    return value, marginal, policy


def _savings_node_point_candidates(
    *,
    liquid_grid: Float1D,
    savings_grid: FloatND,
    cont_value: FloatND,
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    marginal_utility: Callable[[ScalarFloat], ScalarFloat],
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    coh_grid: Float1D | None,
    segment_base: float,
) -> tuple[Float1D, ...]:
    """Build the savings-node point candidates of the per-interval merge.

    At every liquid grid point, one zero-width candidate pair per post-decision
    node: consume that point's cash-on-hand minus the node and earn the node's
    own-interval continuation. The family is a dense Bellman floor at every
    query point — an optimum at a continuation kink between Euler roots (where
    the inversion has no root) is still carried — and each pair is a segment
    the envelope brackets exactly at its own grid point, so visibility does not
    depend on how many grid points the interval holds. Infeasible entries
    (consumption non-positive) are NaN-dead.

    Returns:
        Tuple of the flattened pair-interleaved `(endog, value, policy,
        marginal, segment)` columns.

    """
    interval_of_grid = jnp.searchsorted(breakpoints, liquid_grid, side="right")
    point_coh = (
        coh_slopes[interval_of_grid] * liquid_grid + coh_intercepts[interval_of_grid]
        if coh_grid is None
        else coh_grid
    )
    # A 2-D `(n_intervals, n_nodes)` savings input holds interval-dependent
    # entries (save-to-cliff targets under an interval-bound liquid law); each
    # liquid grid point consumes its own interval's row.
    per_interval_savings = savings_grid.ndim > 1
    savings_at_grid = (
        savings_grid[interval_of_grid, :]
        if per_interval_savings
        else savings_grid[None, :]
    )
    node_consumption = point_coh[:, None] - savings_at_grid
    node_feasible = node_consumption > 0.0
    node_consumption_safe = jnp.where(node_feasible, node_consumption, 1.0)
    node_utility = jax.vmap(jax.vmap(utility_of_action))(node_consumption_safe)
    node_value = jnp.where(
        node_feasible,
        node_utility + discount_factor * cont_value[interval_of_grid],
        jnp.nan,
    )
    node_endog = jnp.where(
        node_feasible,
        jnp.broadcast_to(liquid_grid[:, None], node_consumption.shape),
        jnp.nan,
    )
    node_marginal = coh_slopes[interval_of_grid][:, None] * jax.vmap(
        jax.vmap(marginal_utility)
    )(node_consumption_safe)
    node_segment = segment_base + jnp.arange(
        node_consumption.size, dtype=jnp.result_type(float)
    ).reshape(node_consumption.shape)

    def as_pairs(entries: FloatND) -> Float1D:
        return jnp.stack([entries, entries], axis=-1).reshape(-1)

    return (
        as_pairs(node_endog),
        as_pairs(node_value),
        as_pairs(node_consumption_safe),
        as_pairs(node_marginal),
        as_pairs(node_segment),
    )


def nbegm_unified_step_savings(
    *,
    cont_value: Float1D,
    cont_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    inverse_marginal_utility: Callable[[ScalarFloat], ScalarFloat],
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    jump_positions: tuple[Any, ...],
    extra_savings: Float1D | None = None,
    extra_cont_value: Float1D | None = None,
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve a mixed jump-and-kink piecewise-affine budget against savings continuation.

    The savings-space analogue of `nbegm_unified_step`: the continuation is the
    expected value and expected marginal already evaluated at each post-decision
    savings node (the transition-aware reader has integrated the regime
    transition, stochastic shocks, and ride-along co-state transition), so the
    Euler inversion reads the marginal directly with no explicit gross-return
    term, and the no-save corner reads `cont_value[0]`.

    The jumps partition the liquid axis into cases on each of which cash-on-hand
    is continuous (kinks only); within a case the EGM inverts the case's
    continuous `coh(liquid)` — recovered by clamping the interval index to the
    case's range — and the case is masked to its liquid interval. Each case adds
    its hard-borrowing corner. The cases plus corners merge by the branch-aware
    upper envelope. The pure-kink budget (no jump) reduces to the same answer as
    `nbegm_multi_interval_step_savings`.

    Args:
        cont_value: Expected continuation value at each savings node.
        cont_marginal: Expected marginal continuation (savings space) at each node.
        liquid_grid: Regular liquid-state grid (ascending).
        savings_grid: Post-decision savings grid `s = coh - consumption` (>= 0,
            with `savings_grid[0] == 0` the no-save corner).
        discount_factor: Discount factor.
        utility_of_action: The regime's period utility as a function of consumption,
            with the ride-along cell's states and the utility params already bound.
        inverse_marginal_utility: The regime's inverse marginal utility as a function
            of the discounted expected marginal continuation, with the cell already
            bound — `invert_euler` calls it to recover the consumption action.
        coh_slopes: Per-interval cash-on-hand slope in liquid, length N+1.
        coh_intercepts: Per-interval cash-on-hand intercept, length N+1.
        breakpoints: Sorted ascending liquid breakpoints, length N.
        jump_positions: Indices (into the sorted breakpoints) of the jump
            breakpoints, length J. Static for a single variable; a per-cell traced
            array when breakpoints declared on several variables reorder per cell.

    Returns:
        Tuple of this period's value, marginal value of liquid, and consumption
        policy, each on `liquid_grid`.

    """
    last_interval = coh_slopes.shape[0] - 1
    n_cases = len(jump_positions) + 1
    case_starts = (0, *(position + 1 for position in jump_positions))
    case_ends = (*jump_positions, last_interval)
    case_stride = 4 * (savings_grid.shape[0] + liquid_grid.shape[0])

    # The expected marginal already carries `dR/ds`, so the Euler inversion reads
    # it directly. The continuation does not depend on the current-period jump, so
    # the same consumption schedule serves every subsidy case.
    marginal_utility = jax.grad(utility_of_action)
    consumption = _invert_euler_over_savings(
        cont_marginal=cont_marginal,
        discount_factor=discount_factor,
        inverse_marginal_utility=inverse_marginal_utility,
    )
    coh_endog = consumption + savings_grid
    interp_value = (
        jax.vmap(utility_of_action)(consumption) + discount_factor * cont_value
    )
    value_at_no_save = cont_value[0]
    degenerate = cont_marginal <= _DEGENERATE_MARGINAL_TOL
    grid_interval = jnp.searchsorted(breakpoints, liquid_grid, side="right")

    endog_parts: list[Float1D] = []
    value_parts: list[Float1D] = []
    policy_parts: list[Float1D] = []
    marginal_parts: list[Float1D] = []
    segment_parts: list[Float1D] = []
    for case, (start, end) in enumerate(zip(case_starts, case_ends, strict=True)):
        case_grid_interval = jnp.clip(grid_interval, start, end)
        coh_case_grid = (
            coh_slopes[case_grid_interval] * liquid_grid
            + coh_intercepts[case_grid_interval]
        )
        liquid_endog = _invert_coh_with_linear_extension(
            coh_endog=coh_endog,
            coh_case_grid=coh_case_grid,
            liquid_grid=liquid_grid,
        )
        endog_interval = jnp.clip(
            jnp.searchsorted(breakpoints, liquid_endog, side="right"), start, end
        )
        marginal_endog = coh_slopes[endog_interval] * jax.vmap(marginal_utility)(
            consumption
        )
        # The first case opens at the lower grid edge and the last closes at the
        # upper edge; interior case edges are the adjacent breakpoints, gathered at
        # the (possibly per-cell traced) jump positions `start - 1` and `end`.
        lower = -jnp.inf if case == 0 else breakpoints[start - 1]
        upper = jnp.inf if case == n_cases - 1 else breakpoints[end]
        in_case = (liquid_endog >= lower) & (liquid_endog < upper) & (~degenerate)
        interior = mask_dead_candidates(
            endog_grid=liquid_endog,
            value=interp_value,
            policy=consumption,
            marginal=marginal_endog,
            valid=in_case,
        )
        segment = segment_ids_from_folds(endog_grid=interior[0])
        next_segment = jnp.nanmax(segment) + 1.0
        endog_parts.append(interior[0])
        value_parts.append(interior[1])
        policy_parts.append(interior[2])
        marginal_parts.append(interior[3])
        segment_parts.append(segment + float(case) * case_stride)

        # Hard borrowing corner (save nothing) over this case's liquid range.
        s0_consumption = coh_case_grid
        s0_valid = (liquid_grid >= lower) & (liquid_grid < upper)
        s0 = mask_dead_candidates(
            endog_grid=liquid_grid,
            value=jax.vmap(utility_of_action)(s0_consumption)
            + discount_factor * value_at_no_save,
            policy=s0_consumption,
            marginal=coh_slopes[case_grid_interval]
            * jax.vmap(marginal_utility)(s0_consumption),
            valid=s0_valid,
        )
        endog_parts.append(s0[0])
        value_parts.append(s0[1])
        policy_parts.append(s0[2])
        marginal_parts.append(s0[3])
        segment_parts.append(
            jnp.full_like(liquid_grid, float(case) * case_stride + next_segment)
        )

    # Save-to-cliff point candidates: the continuation's one-sided values at
    # the child cliffs' savings targets, offered at every liquid grid point.
    # The continuation is interval-independent here, so the extra values
    # broadcast across the intervals for the shared point-candidate builder.
    if extra_savings is not None and extra_cont_value is not None:
        cliff = _savings_node_point_candidates(
            liquid_grid=liquid_grid,
            savings_grid=extra_savings,
            cont_value=jnp.broadcast_to(
                extra_cont_value[None, :],
                (coh_slopes.shape[0], extra_cont_value.shape[0]),
            ),
            discount_factor=discount_factor,
            utility_of_action=utility_of_action,
            marginal_utility=marginal_utility,
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=breakpoints,
            coh_grid=None,
            segment_base=float(n_cases * case_stride),
        )
        endog_parts.append(cliff[0])
        value_parts.append(cliff[1])
        policy_parts.append(cliff[2])
        marginal_parts.append(cliff[3])
        segment_parts.append(cliff[4])

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.concatenate(endog_parts),
        policy=jnp.concatenate(policy_parts),
        value=jnp.concatenate(value_parts),
        marginal=jnp.concatenate(marginal_parts),
        segment_id=jnp.concatenate(segment_parts),
        x_query=liquid_grid,
    )
    return value, marginal, policy


def _flat_interval_indices(
    flat_interval_mask: tuple[bool, ...] | None, *, n_intervals: int
) -> tuple[int, ...]:
    """Return the indices of the hard-constraint (slope-0) floor intervals."""
    if flat_interval_mask is None:
        return ()
    return tuple(i for i in range(n_intervals) if flat_interval_mask[i])


def _floor_optimum(
    *,
    floor_coh: ScalarFloat,
    liquid_grid: Float1D,
    next_value: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    gross_return: ScalarFloat | float,
    income: ScalarFloat | float,
    n_dense: int = 512,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Find the value and policy at a fixed floor cash-on-hand by a dense search.

    Where the floor binds, cash-on-hand equals `floor_coh` for every liquid, so the
    value is the single-point Bellman max over consumption. A dense consumption
    search evaluates it directly — convention-free and robust to a recurring flat
    continuation, where the Euler inversion is degenerate.
    """
    fractions = jnp.linspace(1e-4, 1.0, n_dense)
    consumption = fractions * floor_coh
    next_liquid = gross_return * (floor_coh - consumption) + income
    value = _crra_utility(consumption, crra) + discount_factor * jnp.interp(
        next_liquid, liquid_grid, next_value
    )
    best = jnp.argmax(value)
    return value[best], consumption[best]


def nbegm_discrete_envelope_step(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    gross_return: ScalarFloat | float,
    income: ScalarFloat | float,
    choices: tuple[Mapping[str, Float1D], ...],
    taste_shock_scale: float = 0.0,
) -> tuple[Float1D, Float1D, Float1D, IntND]:
    """Compose per-discrete-choice NBEGM solves into a discrete upper envelope.

    Each discrete choice (e.g. buy private insurance or not) shifts cash-on-hand
    differently. NBEGM solves the continuous consumption/savings subproblem inside
    each branch; the discrete choice is then taken by the upper envelope over the
    branch values — the `NBEGM ∘ DC-EGM` composition with the discrete envelope
    outside.

    - With no taste shocks (`taste_shock_scale == 0`) the envelope is the hard
      maximum, so by Danskin's theorem the winning branch's marginal value and
      policy carry through.
    - With an EV1 taste-shock scale the smoothed value is the scaled logsum and the
      smoothed marginal is the choice-probability-weighted branch marginal; the
      reported policy and choice are the modal branch's.

    Args:
        next_value: Next period's value on `liquid_grid`.
        next_marginal: Next period's marginal value of liquid on `liquid_grid`.
        liquid_grid: Regular liquid-state grid (ascending).
        savings_grid: Post-decision savings grid `s = coh - consumption` (>= 0).
        discount_factor: Discount factor.
        crra: Coefficient of relative risk aversion.
        gross_return: Gross liquid return `1 + r`.
        income: Deterministic income added to next-period liquid.
        choices: Per-discrete-choice budgets, each a mapping with `coh_slopes`,
            `coh_intercepts`, and `breakpoints` for `nbegm_multi_interval_step`.
        taste_shock_scale: EV1 taste-shock scale; `0` is the hard maximum.

    Returns:
        Tuple of this period's value, marginal value of liquid, consumption policy,
        and the modal discrete-choice index, each on `liquid_grid`.

    """
    values: list[Float1D] = []
    marginals: list[Float1D] = []
    policies: list[Float1D] = []
    for choice in choices:
        value, marginal, policy = nbegm_multi_interval_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            crra=crra,
            gross_return=gross_return,
            income=income,
            coh_slopes=choice["coh_slopes"],
            coh_intercepts=choice["coh_intercepts"],
            breakpoints=choice["breakpoints"],
        )
        values.append(value)
        marginals.append(marginal)
        policies.append(policy)

    value_stack = jnp.stack(values)
    marginal_stack = jnp.stack(marginals)
    policy_stack = jnp.stack(policies)
    modal = jnp.argmax(value_stack, axis=0).astype(jnp.int32)
    index = jnp.arange(liquid_grid.shape[0])
    if taste_shock_scale == 0.0:
        return (
            value_stack[modal, index],
            marginal_stack[modal, index],
            policy_stack[modal, index],
            modal,
        )
    scaled = value_stack / taste_shock_scale
    probabilities = jax.nn.softmax(scaled, axis=0)
    smoothed_value = taste_shock_scale * jax.scipy.special.logsumexp(scaled, axis=0)
    smoothed_marginal = jnp.sum(probabilities * marginal_stack, axis=0)
    return (
        smoothed_value,
        smoothed_marginal,
        policy_stack[modal, index],
        modal,
    )


def nbegm_unified_step(  # noqa: PLR0915
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    gross_return: ScalarFloat | float,
    income: ScalarFloat | float,
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    jump_mask: tuple[bool, ...],
    equality_owner: EqualityOwner = "otherwise",
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one period of a mixed jump-and-kink piecewise-affine budget by EGM.

    The breakpoints split the liquid axis into intervals; `jump_mask` marks which
    breakpoints are discontinuities (jumps) versus continuous kinks. The jumps
    partition the axis into cases on each of which cash-on-hand is continuous (kinks
    only); within a case the EGM runs in `coh` space and inverts the case's
    continuous `coh(liquid)` — recovered by clamping the interval index to the
    case's interval range, which is exactly the affine extension of the case's
    segments. Each case is masked to its liquid range, reads the continuation
    jump-aware at every jump, and competes a boundary-targeting candidate per jump
    plus the hard borrowing corner; all merge by the branch-aware upper envelope.
    The pure-kink (no jump) and pure-jump (slope-1) budgets are special cases.

    Args:
        next_value: Next period's value on `liquid_grid`.
        next_marginal: Next period's marginal value of liquid on `liquid_grid`.
        liquid_grid: Regular liquid-state grid (ascending).
        savings_grid: Post-decision savings grid `s = coh - consumption` (>= 0).
        discount_factor: Discount factor.
        crra: Coefficient of relative risk aversion.
        gross_return: Gross liquid return `1 + r`.
        income: Deterministic income added to next-period liquid.
        coh_slopes: Per-interval cash-on-hand slope in liquid, length N+1.
        coh_intercepts: Per-interval cash-on-hand intercept, length N+1.
        breakpoints: Sorted ascending liquid breakpoints, length N.
        jump_mask: Static per-breakpoint flag, length N, `True` for a jump.
        equality_owner: Side owning each exact jump point (`when` or `otherwise`).

    Returns:
        Tuple of this period's value, marginal value of liquid, and consumption
        policy, each on `liquid_grid`.

    """
    n_breakpoints = breakpoints.shape[0]
    last_interval = coh_slopes.shape[0] - 1
    jump_positions = tuple(j for j in range(n_breakpoints) if jump_mask[j])
    jump_breakpoints = (
        breakpoints[jnp.asarray(jump_positions)]
        if jump_positions
        else jnp.zeros((0,), dtype=breakpoints.dtype)
    )
    case_starts = (0, *(p + 1 for p in jump_positions))
    case_ends = (*jump_positions, last_interval)
    case_stride = 4 * (savings_grid.shape[0] + liquid_grid.shape[0])

    next_liquid = gross_return * savings_grid + income
    value_next = _jump_aware_interp(
        next_liquid, liquid_grid, next_value, jump_breakpoints, equality_owner
    )
    marginal_next = _jump_aware_interp(
        next_liquid, liquid_grid, next_marginal, jump_breakpoints, equality_owner
    )
    consumption = (discount_factor * gross_return * marginal_next) ** (-1.0 / crra)
    coh_endog = consumption + savings_grid
    interp_value = _crra_utility(consumption, crra) + discount_factor * value_next
    value_at_income = _jump_aware_interp(
        jnp.asarray(income), liquid_grid, next_value, jump_breakpoints, equality_owner
    )
    grid_interval = jnp.searchsorted(breakpoints, liquid_grid, side="right")

    endog_parts: list[Float1D] = []
    value_parts: list[Float1D] = []
    policy_parts: list[Float1D] = []
    marginal_parts: list[Float1D] = []
    segment_parts: list[Float1D] = []
    for case, (start, end) in enumerate(zip(case_starts, case_ends, strict=True)):
        case_grid_interval = jnp.clip(grid_interval, start, end)
        coh_case_grid = (
            coh_slopes[case_grid_interval] * liquid_grid
            + coh_intercepts[case_grid_interval]
        )
        liquid_endog = _invert_coh_with_linear_extension(
            coh_endog=coh_endog,
            coh_case_grid=coh_case_grid,
            liquid_grid=liquid_grid,
        )
        endog_interval = jnp.clip(
            jnp.searchsorted(breakpoints, liquid_endog, side="right"), start, end
        )
        marginal_endog = coh_slopes[endog_interval] * consumption ** (-crra)
        lower = -jnp.inf if start == 0 else breakpoints[start - 1]
        upper = jnp.inf if end == last_interval else breakpoints[end]
        in_case = (liquid_endog >= lower) & (liquid_endog < upper)
        interior = mask_dead_candidates(
            endog_grid=liquid_endog,
            value=interp_value,
            policy=consumption,
            marginal=marginal_endog,
            valid=in_case,
        )
        segment = segment_ids_from_folds(endog_grid=interior[0])
        next_segment = jnp.nanmax(segment) + 1.0
        endog_parts.append(interior[0])
        value_parts.append(interior[1])
        policy_parts.append(interior[2])
        marginal_parts.append(interior[3])
        segment_parts.append(segment + float(case) * case_stride)

        # Hard borrowing corner over this case's liquid range.
        s0_consumption = coh_case_grid
        s0_valid = (liquid_grid >= lower) & (liquid_grid < upper)
        s0 = mask_dead_candidates(
            endog_grid=liquid_grid,
            value=_crra_utility(s0_consumption, crra)
            + discount_factor * value_at_income,
            policy=s0_consumption,
            marginal=coh_slopes[case_grid_interval] * s0_consumption ** (-crra),
            valid=s0_valid,
        )
        endog_parts.append(s0[0])
        value_parts.append(s0[1])
        policy_parts.append(s0[2])
        marginal_parts.append(s0[3])
        segment_parts.append(
            jnp.full_like(liquid_grid, float(case) * case_stride + next_segment)
        )

        # Boundary-targeting at each jump: save to land just inside its eligible
        # side for the higher continuation, consuming this case's cash-on-hand.
        for offset, jump_idx in enumerate(jump_positions):
            cliff = breakpoints[jump_idx]
            # Bound the below-side continuation read to the segment between this
            # cliff and the breakpoint below it, so close cliffs don't bridge.
            prev_limit = (
                breakpoints[jump_idx - 1] if jump_idx > 0 else liquid_grid[0] - 1.0
            )
            kink = _boundary_targeting_coh(
                liquid_grid=liquid_grid,
                coh_case_grid=coh_case_grid,
                next_value=next_value,
                discount_factor=discount_factor,
                crra=crra,
                gross_return=gross_return,
                income=income,
                asset_limit=cliff,
                prev_limit=prev_limit,
                coh_slope=coh_slopes[case_grid_interval],
                valid=s0_valid,
            )
            endog_parts.append(kink[0])
            value_parts.append(kink[1])
            policy_parts.append(kink[2])
            marginal_parts.append(kink[3])
            segment_parts.append(
                jnp.where(
                    jnp.isnan(kink[0]),
                    jnp.nan,
                    float(case) * case_stride + next_segment + 1.0 + float(offset),
                )
            )

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.concatenate(endog_parts),
        policy=jnp.concatenate(policy_parts),
        value=jnp.concatenate(value_parts),
        marginal=jnp.concatenate(marginal_parts),
        segment_id=jnp.concatenate(segment_parts),
        x_query=liquid_grid,
    )
    return value, marginal, policy


def _boundary_targeting_coh(
    *,
    liquid_grid: Float1D,
    coh_case_grid: Float1D,
    next_value: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    gross_return: ScalarFloat | float,
    income: ScalarFloat | float,
    asset_limit: ScalarFloat | float,
    prev_limit: ScalarFloat | float,
    coh_slope: Float1D,
    valid: BoolND,
) -> tuple[Float1D, Float1D, Float1D, Float1D]:
    """Save to land next-period liquid just inside a cliff's eligible side.

    The case's cash-on-hand `coh_case_grid` funds consumption `coh - s_kink` where
    `s_kink` lands next-period liquid one ulp below the cliff (the eligible side),
    paired with that side's continuation so policy and value stay consistent. The
    below-side continuation limit reads only nodes in `(prev_limit, asset_limit)`,
    so with cliffs close together it does not bridge the neighbouring jump.
    """
    limit_minus = jnp.nextafter(
        jnp.asarray(asset_limit, dtype=liquid_grid.dtype),
        jnp.asarray(-jnp.inf, dtype=liquid_grid.dtype),
    )
    s_kink = (limit_minus - income) / gross_return
    value_limit_minus = _bounded_limit_below(
        liquid_grid,
        next_value,
        limit=asset_limit,
        prev_limit=prev_limit,
        n=liquid_grid.shape[0],
    )
    kink_consumption = coh_case_grid - s_kink
    kink_value = (
        _crra_utility(kink_consumption, crra) + discount_factor * value_limit_minus
    )
    # The targeted saving is fixed to the cliff, so consumption moves with the
    # case's affine cash-on-hand: `dc/da = coh_slope`, and the marginal value of
    # liquid is `coh_slope * c**(-crra)`, matching the interior and corner
    # candidates.
    kink_marginal = coh_slope * kink_consumption ** (-crra)
    kink_valid = valid & (kink_consumption > 0.0) & (s_kink >= 0.0)
    return mask_dead_candidates(
        endog_grid=liquid_grid,
        value=kink_value,
        policy=kink_consumption,
        marginal=kink_marginal,
        valid=kink_valid,
    )


def nbegm_recurring_jump_step(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    gross_return: ScalarFloat | float,
    income: ScalarFloat | float,
    subsidy_levels: Float1D,
    jump_breakpoints: Float1D,
    equality_owner: EqualityOwner = "otherwise",
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one period of an N-cliff budget with a recurring jumped continuation.

    The N-cliff generalization of the binary `nbegm_one_asset_step`. Each of the
    N+1 subsidy levels is a case whose budget is smooth; its 1-D EGM reads the
    continuation jump-aware at every cliff (no bridging across a value jump) and
    competes, per cliff, a boundary-targeting candidate that saves to land
    next-period liquid just inside the cliff's eligible side for its higher
    continuation. Each case is masked to its liquid range and all cases merge by the
    branch-aware upper envelope, so the solve is exact through every recurring jump,
    not only at a terminal-adjacent period.

    Args:
        next_value: Next period's value on `liquid_grid`.
        next_marginal: Next period's marginal value of liquid on `liquid_grid`.
        liquid_grid: Regular liquid-state grid (ascending).
        savings_grid: Post-decision savings grid `s = coh - consumption` (>= 0).
        discount_factor: Discount factor.
        crra: Coefficient of relative risk aversion.
        gross_return: Gross liquid return `1 + r`.
        income: Deterministic income added to next-period liquid.
        subsidy_levels: Additive subsidy per case, length N+1, in liquid order.
        jump_breakpoints: Sorted ascending liquid cliffs, length N.
        equality_owner: Side owning each exact cliff point (`when` or `otherwise`).

    Returns:
        Tuple of this period's value, marginal value of liquid, and consumption
        policy, each on `liquid_grid`.

    """
    lower_edges = jnp.concatenate(
        [jnp.asarray([-jnp.inf], dtype=liquid_grid.dtype), jump_breakpoints]
    )
    upper_edges = jnp.concatenate(
        [jump_breakpoints, jnp.asarray([jnp.inf], dtype=liquid_grid.dtype)]
    )
    n_cases = subsidy_levels.shape[0]
    case_stride = 4 * savings_grid.shape[0]

    endog_parts: list[Float1D] = []
    value_parts: list[Float1D] = []
    policy_parts: list[Float1D] = []
    marginal_parts: list[Float1D] = []
    segment_parts: list[Float1D] = []
    for k in range(n_cases):
        case_value, case_policy, case_marginal, case_endog, case_segment = (
            _recurring_jump_case(
                next_value=next_value,
                next_marginal=next_marginal,
                liquid_grid=liquid_grid,
                savings_grid=savings_grid,
                discount_factor=discount_factor,
                crra=crra,
                gross_return=gross_return,
                income=income,
                subsidy=subsidy_levels[k],
                jump_breakpoints=jump_breakpoints,
                equality_owner=equality_owner,
            )
        )
        in_case = (case_endog >= lower_edges[k]) & (case_endog < upper_edges[k])
        masked = mask_dead_candidates(
            endog_grid=case_endog,
            value=case_value,
            policy=case_policy,
            marginal=case_marginal,
            valid=in_case,
        )
        endog_parts.append(masked[0])
        value_parts.append(masked[1])
        policy_parts.append(masked[2])
        marginal_parts.append(masked[3])
        segment_parts.append(case_segment + float(k) * case_stride)

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.concatenate(endog_parts),
        policy=jnp.concatenate(policy_parts),
        value=jnp.concatenate(value_parts),
        marginal=jnp.concatenate(marginal_parts),
        segment_id=jnp.concatenate(segment_parts),
        x_query=liquid_grid,
    )
    return value, marginal, policy


def _recurring_jump_case(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    gross_return: ScalarFloat | float,
    income: ScalarFloat | float,
    subsidy: ScalarFloat | float,
    jump_breakpoints: Float1D,
    equality_owner: EqualityOwner,
) -> tuple[Float1D, Float1D, Float1D, Float1D, Float1D]:
    """Build one subsidy case's candidate path: Euler, per-cliff target, s=0 corner.

    `coh = liquid + subsidy`, so the endogenous liquid recovered from the Euler
    inversion is `consumption + savings - subsidy`. The continuation is read
    jump-aware at every cliff. A jumped continuation is nonconcave, so the case
    contributes, beyond the Euler path, a boundary-targeting candidate per cliff
    (save to land just inside its eligible side) and the hard borrowing corner, all
    over the liquid grid.
    """
    next_liquid = gross_return * savings_grid + income
    value_next = _jump_aware_interp(
        next_liquid, liquid_grid, next_value, jump_breakpoints, equality_owner
    )
    marginal_next = _jump_aware_interp(
        next_liquid, liquid_grid, next_marginal, jump_breakpoints, equality_owner
    )
    consumption = (discount_factor * gross_return * marginal_next) ** (-1.0 / crra)
    liquid_endog = consumption + savings_grid - subsidy
    value_endog = _crra_utility(consumption, crra) + discount_factor * value_next
    marginal_endog = consumption ** (-crra)
    interior_segment = segment_ids_from_folds(endog_grid=liquid_endog)
    next_segment = jnp.nanmax(interior_segment) + 1.0

    endog_parts: list[Float1D] = [liquid_endog]
    value_parts: list[Float1D] = [value_endog]
    policy_parts: list[Float1D] = [consumption]
    marginal_parts: list[Float1D] = [marginal_endog]
    segment_parts: list[Float1D] = [interior_segment]

    n_cliffs = jump_breakpoints.shape[0]
    for j in range(n_cliffs):
        cliff = jump_breakpoints[j]
        prev_cliff = jump_breakpoints[j - 1] if j > 0 else liquid_grid[0] - 1.0
        kink = _boundary_targeting_branch(
            liquid_grid=liquid_grid,
            next_value=next_value,
            discount_factor=discount_factor,
            crra=crra,
            gross_return=gross_return,
            income=income,
            subsidy=subsidy,
            asset_limit=cliff,
            prev_limit=prev_cliff,
        )
        endog_parts.append(kink[0])
        value_parts.append(kink[1])
        policy_parts.append(kink[2])
        marginal_parts.append(kink[3])
        segment_parts.append(
            jnp.where(jnp.isnan(kink[0]), jnp.nan, next_segment + float(j))
        )

    value_at_income = _jump_aware_interp(
        jnp.asarray(income), liquid_grid, next_value, jump_breakpoints, equality_owner
    )
    s0_consumption = liquid_grid + subsidy
    endog_parts.append(liquid_grid)
    value_parts.append(
        _crra_utility(s0_consumption, crra) + discount_factor * value_at_income
    )
    policy_parts.append(s0_consumption)
    marginal_parts.append(s0_consumption ** (-crra))
    segment_parts.append(jnp.full_like(liquid_grid, next_segment + float(n_cliffs)))

    return (
        jnp.concatenate(value_parts),
        jnp.concatenate(policy_parts),
        jnp.concatenate(marginal_parts),
        jnp.concatenate(endog_parts),
        jnp.concatenate(segment_parts),
    )


def _jump_aware_interp(
    query: FloatND,
    grid: Float1D,
    values: Float1D,
    breakpoints: Float1D,
    equality_owner: EqualityOwner,
) -> FloatND:
    """Interpolate `values` on `grid` without bridging the jumps at `breakpoints`.

    The continuation carries a value jump at every cliff. Two split abscissae per
    cliff carry the below-side and above-side values, each linearly extrapolated
    from the two nearest same-side nodes; the owning side's node sits exactly on the
    cliff. A query then interpolates within one side of every cliff and never bridges
    a discontinuity — the N-cliff generalization of `_kink_aware_interp`.
    """
    n = grid.shape[0]
    n_bp = breakpoints.shape[0]
    below_nodes: list[Float1D] = []
    below_values: list[Float1D] = []
    above_nodes: list[Float1D] = []
    above_values: list[Float1D] = []
    for j in range(n_bp):
        limit = breakpoints[j]
        limit_at = jnp.asarray(limit, dtype=grid.dtype)
        # Bound each side's stencil to the interval between adjacent cliffs, so a
        # one-sided limit never extrapolates across a neighbouring jump.
        prev_limit = breakpoints[j - 1] if j > 0 else grid[0] - 1.0
        next_limit = breakpoints[j + 1] if j < n_bp - 1 else grid[-1] + 1.0
        left_at = _bounded_limit_below(
            grid, values, limit=limit_at, prev_limit=prev_limit, n=n
        )
        right_at = _bounded_limit_above(
            grid, values, limit=limit_at, next_limit=next_limit, n=n
        )
        below = jnp.nextafter(limit_at, jnp.asarray(-jnp.inf, dtype=grid.dtype))
        above = jnp.nextafter(limit_at, jnp.asarray(jnp.inf, dtype=grid.dtype))
        if equality_owner == "otherwise":
            left_node, right_node = below, limit_at
        else:
            left_node, right_node = limit_at, above
        below_nodes.append(left_node)
        below_values.append(left_at)
        above_nodes.append(right_node)
        above_values.append(right_at)
    aug_grid = jnp.concatenate([grid, jnp.stack(below_nodes), jnp.stack(above_nodes)])
    aug_values = jnp.concatenate(
        [values, jnp.stack(below_values), jnp.stack(above_values)]
    )
    order = jnp.argsort(aug_grid)
    return jnp.interp(query, aug_grid[order], aug_values[order])


def nbegm_one_asset_step(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    return_liquid: ScalarFloat | float,
    income: ScalarFloat | float,
    subsidy_when: ScalarFloat | float,
    subsidy_otherwise: ScalarFloat | float,
    asset_limit: ScalarFloat | float,
    equality_owner: EqualityOwner,
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one period of the Medicaid one-asset toy by case-piece EGM.

    Args:
        next_value: Next period's value on `liquid_grid`.
        next_marginal: Next period's marginal value of liquid on `liquid_grid`.
        liquid_grid: Regular liquid-state grid (ascending).
        savings_grid: Post-decision savings grid `s = coh - consumption` (>= 0).
        discount_factor: Discount factor.
        crra: Coefficient of relative risk aversion.
        return_liquid: Liquid net return.
        income: Deterministic income added to next-period liquid.
        subsidy_when: Subsidy into cash-on-hand where the predicate holds.
        subsidy_otherwise: Subsidy where the predicate fails.
        asset_limit: Medicaid asset limit; the predicate is `liquid < asset_limit`.
        equality_owner: Predicate side owning the exact boundary point.

    Returns:
        Tuple of this period's value, marginal value of liquid, and consumption
        policy, each on `liquid_grid`.

    """
    when_value, when_marginal, when_policy = _case_step(
        next_value=next_value,
        next_marginal=next_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        crra=crra,
        return_liquid=return_liquid,
        income=income,
        subsidy=subsidy_when,
        asset_limit=asset_limit,
        equality_owner=equality_owner,
    )
    otherwise_value, otherwise_marginal, otherwise_policy = _case_step(
        next_value=next_value,
        next_marginal=next_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        crra=crra,
        return_liquid=return_liquid,
        income=income,
        subsidy=subsidy_otherwise,
        asset_limit=asset_limit,
        equality_owner=equality_owner,
    )

    # The owning side keeps the exact boundary point: `equality_owner="otherwise"`
    # gives it to the otherwise case through the strict `<` / non-strict `>=`
    # split; `equality_owner="when"` mirrors the split the other way.
    if equality_owner == "otherwise":
        when_valid = liquid_grid < asset_limit
        otherwise_valid = liquid_grid >= asset_limit
    else:
        when_valid = liquid_grid <= asset_limit
        otherwise_valid = liquid_grid > asset_limit
    when = mask_dead_candidates(
        endog_grid=liquid_grid,
        value=when_value,
        policy=when_policy,
        marginal=when_marginal,
        valid=when_valid,
    )
    otherwise = mask_dead_candidates(
        endog_grid=liquid_grid,
        value=otherwise_value,
        policy=otherwise_policy,
        marginal=otherwise_marginal,
        valid=otherwise_valid,
    )

    n_grid = liquid_grid.shape[0]
    endog_grid = jnp.concatenate([when[0], otherwise[0]])
    value = jnp.concatenate([when[1], otherwise[1]])
    policy = jnp.concatenate([when[2], otherwise[2]])
    marginal = jnp.concatenate([when[3], otherwise[3]])
    segment_id = jnp.concatenate(
        [
            jnp.zeros(n_grid, dtype=liquid_grid.dtype),
            jnp.ones(n_grid, dtype=liquid_grid.dtype),
        ]
    )
    env_value, env_policy, env_marginal = envelope_at_query(
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal=marginal,
        segment_id=segment_id,
        x_query=liquid_grid,
    )
    return env_value, env_marginal, env_policy


def _case_step(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    return_liquid: ScalarFloat | float,
    income: ScalarFloat | float,
    subsidy: ScalarFloat | float,
    asset_limit: ScalarFloat | float,
    equality_owner: EqualityOwner,
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one case's 1-D consumption--saving sub-problem as an upper envelope.

    `coh = liquid + subsidy`, so the endogenous state recovered from the Euler
    inversion is `liquid = consumption + savings - subsidy`. A jumped continuation
    is nonconcave, so the case value is the upper envelope over three candidate
    branches rather than the bare Euler path:

    - the Euler interior path;
    - the boundary-targeting branch that saves to land just inside the eligible
      side of the boundary (`next_liquid -> asset_limit` from below) for the higher
      eligible continuation — a corner the Euler equation never produces;
    - the hard borrowing corner `s = 0`, consuming all cash-on-hand.

    The continuation is read kink-aware at `asset_limit`: a query landing in the
    boundary cell reads the equality-owning side rather than a bridged average.
    """
    gross_return = 1.0 + return_liquid
    next_liquid = gross_return * savings_grid + income
    value_next = _kink_aware_interp(
        next_liquid, liquid_grid, next_value, asset_limit, equality_owner
    )
    marginal_next = _kink_aware_interp(
        next_liquid, liquid_grid, next_marginal, asset_limit, equality_owner
    )

    consumption = (discount_factor * gross_return * marginal_next) ** (-1.0 / crra)
    liquid_endog = consumption + savings_grid - subsidy
    value_endog = _crra_utility(consumption, crra) + discount_factor * value_next
    marginal_endog = consumption ** (-crra)
    # A kinked continuation folds `liquid_endog` back (the DC-EGM secondary kink),
    # so the interior path may carry several monotone segments.
    interior_segment = segment_ids_from_folds(endog_grid=liquid_endog)

    # A single jump has no neighbouring cliff below; the sentinel leaves the
    # below-side continuation stencil unbounded on the left.
    kink_grid, kink_value, kink_consumption, kink_marginal = _boundary_targeting_branch(
        liquid_grid=liquid_grid,
        next_value=next_value,
        discount_factor=discount_factor,
        crra=crra,
        gross_return=gross_return,
        income=income,
        subsidy=subsidy,
        asset_limit=asset_limit,
        prev_limit=liquid_grid[0] - 1.0,
    )
    kink_segment = jnp.where(
        jnp.isnan(kink_grid), jnp.nan, jnp.nanmax(interior_segment) + 1.0
    )

    # Hard borrowing corner: saving nothing consumes all cash-on-hand and lands
    # next-period liquid at `income`. Because a jumped continuation is nonconcave,
    # this corner can dominate the Euler path even where an Euler segment brackets
    # the query, so it is an envelope candidate over the whole grid — not only the
    # below-grid constrained tail a concave EGM shortcut would assume.
    value_at_income = _kink_aware_interp(
        jnp.asarray(income), liquid_grid, next_value, asset_limit, equality_owner
    )
    s0_consumption = liquid_grid + subsidy
    s0_value = _crra_utility(s0_consumption, crra) + discount_factor * value_at_income
    s0_marginal = s0_consumption ** (-crra)
    s0_segment = jnp.full_like(liquid_grid, jnp.nanmax(interior_segment) + 2.0)

    value, consumption_on_grid, marginal = envelope_at_query(
        endog_grid=jnp.concatenate([liquid_endog, kink_grid, liquid_grid]),
        policy=jnp.concatenate([consumption, kink_consumption, s0_consumption]),
        value=jnp.concatenate([value_endog, kink_value, s0_value]),
        marginal=jnp.concatenate([marginal_endog, kink_marginal, s0_marginal]),
        segment_id=jnp.concatenate([interior_segment, kink_segment, s0_segment]),
        x_query=liquid_grid,
    )
    return value, marginal, consumption_on_grid


def _boundary_targeting_branch(
    *,
    liquid_grid: Float1D,
    next_value: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    gross_return: ScalarFloat | float,
    income: ScalarFloat | float,
    subsidy: ScalarFloat | float,
    asset_limit: ScalarFloat | float,
    prev_limit: ScalarFloat | float,
) -> tuple[Float1D, Float1D, Float1D, Float1D]:
    """Build the save-to-the-boundary candidate as a masked grid-aligned branch.

    Saving exactly to the limit lands `next_liquid == asset_limit`, which the
    otherwise side owns, so it earns the lower continuation. To earn the higher
    eligible continuation the branch targets the open left limit `asset_limit⁻`
    (one ulp below), so the reported policy and the eligible-side value it is
    paired with are mutually consistent rather than a supremum dressed as a
    maximum. Saving the fixed amount maps current liquid to itself
    (`endog == liquid`), so the branch is the curve `c = coh - s_kink`. The
    below-side continuation limit reads only nodes in `(prev_limit, asset_limit)`,
    so with cliffs close together it does not bridge the neighbouring jump.
    """
    limit_minus = jnp.nextafter(
        jnp.asarray(asset_limit, dtype=liquid_grid.dtype),
        jnp.asarray(-jnp.inf, dtype=liquid_grid.dtype),
    )
    s_kink = (limit_minus - income) / gross_return
    value_limit_minus = _bounded_limit_below(
        liquid_grid,
        next_value,
        limit=asset_limit,
        prev_limit=prev_limit,
        n=liquid_grid.shape[0],
    )
    kink_consumption = liquid_grid + subsidy - s_kink
    kink_value = (
        _crra_utility(kink_consumption, crra) + discount_factor * value_limit_minus
    )
    kink_marginal = kink_consumption ** (-crra)
    kink_valid = (kink_consumption > 0.0) & (s_kink >= 0.0)
    return mask_dead_candidates(
        endog_grid=liquid_grid,
        value=kink_value,
        policy=kink_consumption,
        marginal=kink_marginal,
        valid=kink_valid,
    )


def _kink_aware_interp(
    query: FloatND,
    grid: Float1D,
    values: Float1D,
    limit: ScalarFloat | float,
    equality_owner: EqualityOwner,
) -> FloatND:
    """Interpolate `values` on `grid` without bridging the jump at `limit`.

    The continuation carries a value jump at the case boundary: the grid node
    just below `limit` holds the `when`-side value and the node just above holds
    the `otherwise`-side value, so a plain linear interpolation across that cell
    returns a meaningless average. Two extra abscissae split the jump at `limit`,
    each carrying its side's value linearly extrapolated from the two nearest
    same-side nodes. The owning side's node sits exactly on `limit`, so a query at
    exactly the boundary reads the owning side:

    - `equality_owner="otherwise"`: the otherwise (upper) value sits on `limit`
      and the `when` value one ulp below;
    - `equality_owner="when"`: the `when` (lower) value sits on `limit` and the
      otherwise value one ulp above.

    A query strictly below `limit` then interpolates within the `when` branch, a
    query strictly above within the `otherwise` branch, and neither bridges the
    discontinuity.
    """
    n = grid.shape[0]
    last_below = jnp.clip(jnp.sum(grid < limit) - 1, 1, n - 3).astype(jnp.int32)
    left_at_limit = _extrapolate(grid, values, last_below - 1, last_below, limit)
    right_at_limit = _extrapolate(grid, values, last_below + 1, last_below + 2, limit)

    limit_at = jnp.asarray(limit, dtype=grid.dtype)
    below = jnp.nextafter(limit_at, jnp.asarray(-jnp.inf, dtype=grid.dtype))
    above = jnp.nextafter(limit_at, jnp.asarray(jnp.inf, dtype=grid.dtype))
    if equality_owner == "otherwise":
        left_node, right_node = below, limit_at
    else:
        left_node, right_node = limit_at, above
    aug_grid = jnp.concatenate([grid, jnp.stack([left_node, right_node])])
    aug_values = jnp.concatenate([values, jnp.stack([left_at_limit, right_at_limit])])
    order = jnp.argsort(aug_grid)
    return jnp.interp(query, aug_grid[order], aug_values[order])


def _extrapolate(
    grid: Float1D,
    values: Float1D,
    lower: IntND,
    upper: IntND,
    target: ScalarFloat | float,
) -> ScalarFloat:
    """Linearly extrapolate `values` to `target` through nodes `lower`, `upper`."""
    g0, g1 = grid[lower], grid[upper]
    v0, v1 = values[lower], values[upper]
    slope = (v1 - v0) / (g1 - g0)
    return v1 + slope * (target - g1)


def _bounded_limit_below(
    grid: Float1D,
    values: Float1D,
    *,
    limit: ScalarFloat | float,
    prev_limit: ScalarFloat | float,
    n: int,
) -> ScalarFloat:
    """One-sided limit approaching `limit` from below, using only nodes strictly
    inside `(prev_limit, limit)` so the stencil never crosses the previous cliff.

    Falls back to the nearest in-interval node's value when fewer than two such
    nodes exist, rather than extrapolating across the neighbouring discontinuity.
    """
    hi = jnp.sum(grid < limit) - 1
    floor = jnp.sum(grid <= prev_limit)
    lo = jnp.clip(jnp.maximum(hi - 1, floor), 0, n - 1).astype(jnp.int32)
    hi = jnp.clip(jnp.maximum(hi, floor), 0, n - 1).astype(jnp.int32)
    return jnp.where(lo == hi, values[hi], _extrapolate(grid, values, lo, hi, limit))


def _bounded_limit_above(
    grid: Float1D,
    values: Float1D,
    *,
    limit: ScalarFloat | float,
    next_limit: ScalarFloat | float,
    n: int,
) -> ScalarFloat:
    """One-sided limit approaching `limit` from above, using only nodes strictly
    inside `(limit, next_limit)` so the stencil never crosses the next cliff.

    Falls back to the nearest in-interval node's value when fewer than two such
    nodes exist, rather than extrapolating across the neighbouring discontinuity.
    """
    lo = jnp.sum(grid <= limit)
    ceil = jnp.sum(grid < next_limit) - 1
    hi = jnp.clip(jnp.minimum(lo + 1, ceil), 0, n - 1).astype(jnp.int32)
    lo = jnp.clip(jnp.minimum(lo, ceil), 0, n - 1).astype(jnp.int32)
    return jnp.where(lo == hi, values[lo], _extrapolate(grid, values, lo, hi, limit))


def _crra_utility(consumption: Float1D, crra: ScalarFloat | float) -> Float1D:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )
