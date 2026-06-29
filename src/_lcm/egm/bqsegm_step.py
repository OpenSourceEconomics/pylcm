"""One-period BQSEGM steps for a 1-D consumption--saving regime.

Two regime shapes are covered:

- A *continuous* piecewise-affine budget (`bqsegm_multi_interval_step`) — every
  breakpoint a kink, no jump — where `coh(liquid)` stays continuous and monotone,
  so one EGM pass in `coh` space and an inversion of `coh(liquid)` recover the
  whole liquid grid with no interval seam.
- A binary *jump* case boundary (`bqsegm_one_asset_step`) where cash-on-hand jumps
  across the boundary, requiring two masked cases merged through the value
  discontinuity.

The binary jump case: a case boundary on the liquid state (`liquid < asset_limit`)
shifts an additive subsidy into cash-on-hand. Within each case the budget is
smooth, so the period solves by ordinary 1-D EGM on `coh = liquid + subsidy`; the
recovered endogenous state is `liquid = coh - subsidy`. Each case's value is the
upper envelope over three candidate branches on the liquid grid:

- the Euler interior path from the EGM inversion;
- the boundary-targeting branch that saves just enough to land on the eligible
  side of the boundary and earn its higher continuation;
- the hard borrowing corner that saves nothing and consumes all cash-on-hand.

The two cases are then merged by the branch-aware upper envelope after NaN-dead
masking each case to its consistent region — the `when` case where the predicate
holds, the `otherwise` case where it fails. The `equality_owner` of the boundary
fixes which side owns the exact boundary point: `equality_owner="otherwise"` gives
the otherwise side ownership through the strict `<` / non-strict `>=` split.
"""

from collections.abc import Mapping

import jax.numpy as jnp

from _lcm.egm.bqsegm_segments import mask_dead_candidates, segment_ids_from_folds
from _lcm.egm.upper_envelope.query import envelope_at_query
from lcm.case_piece import EqualityOwner
from lcm.typing import BoolND, Float1D, FloatND, IntND, ScalarFloat


def bqsegm_multi_interval_step(
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
    across the interval below. The hard borrowing corner competes over the whole
    grid, and the interior path, flat corners, and borrowing corner merge by the
    branch-aware upper envelope.

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
    liquid_endog = jnp.interp(coh_endog, coh_grid, liquid_grid)
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
    flat_corners: list[tuple[Float1D, Float1D, Float1D, Float1D]] = []
    for i in flat_indices:
        floor_value = jnp.interp(coh_intercepts[i], coh_endog, value_endog)
        floor_policy = jnp.interp(coh_intercepts[i], coh_endog, consumption)
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


def _flat_interval_indices(
    flat_interval_mask: tuple[bool, ...] | None, *, n_intervals: int
) -> tuple[int, ...]:
    """Return the indices of the hard-constraint (slope-0) floor intervals."""
    if flat_interval_mask is None:
        return ()
    return tuple(i for i in range(n_intervals) if flat_interval_mask[i])


def bqsegm_discrete_envelope_step(
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
) -> tuple[Float1D, Float1D, Float1D, IntND]:
    """Compose per-discrete-choice BQSEGM solves into a discrete upper envelope.

    Each discrete choice (e.g. buy private insurance or not) shifts cash-on-hand
    differently. BQSEGM solves the continuous consumption/savings subproblem inside
    each branch; the discrete choice is then taken by the upper envelope over the
    branch values — the `BQSEGM ∘ DC-EGM` composition with the discrete envelope
    outside. With no taste shocks the envelope is the hard maximum, so by Danskin's
    theorem the winning branch's marginal value and policy carry through.

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
            `coh_intercepts`, and `breakpoints` for `bqsegm_multi_interval_step`.

    Returns:
        Tuple of this period's value, marginal value of liquid, consumption policy,
        and the winning discrete-choice index, each on `liquid_grid`.

    """
    values: list[Float1D] = []
    marginals: list[Float1D] = []
    policies: list[Float1D] = []
    for choice in choices:
        value, marginal, policy = bqsegm_multi_interval_step(
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
    winning = jnp.argmax(value_stack, axis=0).astype(jnp.int32)
    index = jnp.arange(liquid_grid.shape[0])
    return (
        value_stack[winning, index],
        jnp.stack(marginals)[winning, index],
        jnp.stack(policies)[winning, index],
        winning,
    )


def bqsegm_unified_step(  # noqa: PLR0915
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
    n = liquid_grid.shape[0]

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
        liquid_endog = jnp.interp(coh_endog, coh_case_grid, liquid_grid)
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
            last_below = jnp.clip(jnp.sum(liquid_grid < cliff) - 1, 1, n - 3).astype(
                jnp.int32
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
                last_below=last_below,
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
    last_below: IntND,
    valid: BoolND,
) -> tuple[Float1D, Float1D, Float1D, Float1D]:
    """Save to land next-period liquid just inside a cliff's eligible side.

    The case's cash-on-hand `coh_case_grid` funds consumption `coh - s_kink` where
    `s_kink` lands next-period liquid one ulp below the cliff (the eligible side),
    paired with that side's continuation so policy and value stay consistent.
    """
    limit_minus = jnp.nextafter(
        jnp.asarray(asset_limit, dtype=liquid_grid.dtype),
        jnp.asarray(-jnp.inf, dtype=liquid_grid.dtype),
    )
    s_kink = (limit_minus - income) / gross_return
    value_limit_minus = _extrapolate(
        liquid_grid, next_value, last_below - 1, last_below, asset_limit
    )
    kink_consumption = coh_case_grid - s_kink
    kink_value = (
        _crra_utility(kink_consumption, crra) + discount_factor * value_limit_minus
    )
    kink_marginal = kink_consumption ** (-crra)
    kink_valid = valid & (kink_consumption > 0.0) & (s_kink >= 0.0)
    return mask_dead_candidates(
        endog_grid=liquid_grid,
        value=kink_value,
        policy=kink_consumption,
        marginal=kink_marginal,
        valid=kink_valid,
    )


def bqsegm_recurring_jump_step(
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

    The N-cliff generalization of the binary `bqsegm_one_asset_step`. Each of the
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

    n = liquid_grid.shape[0]
    n_cliffs = jump_breakpoints.shape[0]
    for j in range(n_cliffs):
        cliff = jump_breakpoints[j]
        last_below = jnp.clip(jnp.sum(liquid_grid < cliff) - 1, 1, n - 3).astype(
            jnp.int32
        )
        kink = _boundary_targeting_branch(
            liquid_grid=liquid_grid,
            next_value=next_value,
            discount_factor=discount_factor,
            crra=crra,
            gross_return=gross_return,
            income=income,
            subsidy=subsidy,
            asset_limit=cliff,
            last_below=last_below,
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
        last_below = jnp.clip(jnp.sum(grid < limit) - 1, 1, n - 3).astype(jnp.int32)
        left_at = _extrapolate(grid, values, last_below - 1, last_below, limit)
        right_at = _extrapolate(grid, values, last_below + 1, last_below + 2, limit)
        limit_at = jnp.asarray(limit, dtype=grid.dtype)
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


def bqsegm_jump_step(
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
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one period of a budget with N additive subsidy cliffs by EGM.

    Each cliff drops the additive subsidy into cash-on-hand, so `coh(liquid)` jumps
    down as liquid crosses it and the value function jumps there too. The continuous
    coh inversion cannot cross a discontinuity, so each of the N+1 subsidy levels is
    a separate case: `coh = liquid + subsidy_k` is smooth within case `k`, solved by
    ordinary 1-D EGM whose endogenous liquid is `consumption + savings - subsidy_k`.
    Each case is masked to its own liquid range `[cliff_{k-1}, cliff_k)`, the hard
    borrowing corner competes over the whole grid, and all candidates merge by the
    branch-aware upper envelope — the N-cliff generalization of the binary case
    merge.

    The continuation is read by plain interpolation, which is exact when it is
    smooth (a terminal-adjacent period). A recurring jumped continuation additionally
    needs the boundary-targeting candidate and jump-aware continuation read of the
    binary `bqsegm_one_asset_step`, generalized per cliff.

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

    Returns:
        Tuple of this period's value, marginal value of liquid, and consumption
        policy, each on `liquid_grid`.

    """
    next_liquid = gross_return * savings_grid + income
    value_next = jnp.interp(next_liquid, liquid_grid, next_value)
    marginal_next = jnp.interp(next_liquid, liquid_grid, next_marginal)

    consumption = (discount_factor * gross_return * marginal_next) ** (-1.0 / crra)
    value_endog = _crra_utility(consumption, crra) + discount_factor * value_next
    marginal_endog = consumption ** (-crra)

    lower_edges = jnp.concatenate(
        [jnp.asarray([-jnp.inf], dtype=liquid_grid.dtype), jump_breakpoints]
    )
    upper_edges = jnp.concatenate(
        [jump_breakpoints, jnp.asarray([jnp.inf], dtype=liquid_grid.dtype)]
    )
    n_cases = subsidy_levels.shape[0]
    stride = savings_grid.shape[0] + 1

    endog_parts: list[Float1D] = []
    value_parts: list[Float1D] = []
    policy_parts: list[Float1D] = []
    marginal_parts: list[Float1D] = []
    segment_parts: list[Float1D] = []
    for k in range(n_cases):
        liquid_endog = consumption + savings_grid - subsidy_levels[k]
        in_case = (liquid_endog >= lower_edges[k]) & (liquid_endog < upper_edges[k])
        segment = segment_ids_from_folds(endog_grid=liquid_endog) + float(k) * stride
        masked = mask_dead_candidates(
            endog_grid=liquid_endog,
            value=value_endog,
            policy=consumption,
            marginal=marginal_endog,
            valid=in_case,
        )
        endog_parts.append(masked[0])
        value_parts.append(masked[1])
        policy_parts.append(masked[2])
        marginal_parts.append(masked[3])
        segment_parts.append(segment)

    # Hard borrowing corner: save nothing, consume all of this point's cash-on-hand.
    interval_of_grid = jnp.searchsorted(jump_breakpoints, liquid_grid, side="right")
    coh_grid = liquid_grid + subsidy_levels[interval_of_grid]
    value_at_income = jnp.interp(jnp.asarray(income), liquid_grid, next_value)
    endog_parts.append(liquid_grid)
    value_parts.append(
        _crra_utility(coh_grid, crra) + discount_factor * value_at_income
    )
    policy_parts.append(coh_grid)
    marginal_parts.append(coh_grid ** (-crra))
    segment_parts.append(jnp.full_like(liquid_grid, float(n_cases) * stride))

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.concatenate(endog_parts),
        policy=jnp.concatenate(policy_parts),
        value=jnp.concatenate(value_parts),
        marginal=jnp.concatenate(marginal_parts),
        segment_id=jnp.concatenate(segment_parts),
        x_query=liquid_grid,
    )
    return value, marginal, policy


def bqsegm_one_asset_step(
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

    n = liquid_grid.shape[0]
    last_below = jnp.clip(jnp.sum(liquid_grid < asset_limit) - 1, 1, n - 3).astype(
        jnp.int32
    )
    kink_grid, kink_value, kink_consumption, kink_marginal = _boundary_targeting_branch(
        liquid_grid=liquid_grid,
        next_value=next_value,
        discount_factor=discount_factor,
        crra=crra,
        gross_return=gross_return,
        income=income,
        subsidy=subsidy,
        asset_limit=asset_limit,
        last_below=last_below,
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
    last_below: IntND,
) -> tuple[Float1D, Float1D, Float1D, Float1D]:
    """Build the save-to-the-boundary candidate as a masked grid-aligned branch.

    Saving exactly to the limit lands `next_liquid == asset_limit`, which the
    otherwise side owns, so it earns the lower continuation. To earn the higher
    eligible continuation the branch targets the open left limit `asset_limit⁻`
    (one ulp below), so the reported policy and the eligible-side value it is
    paired with are mutually consistent rather than a supremum dressed as a
    maximum. Saving the fixed amount maps current liquid to itself
    (`endog == liquid`), so the branch is the curve `c = coh - s_kink`.
    """
    limit_minus = jnp.nextafter(
        jnp.asarray(asset_limit, dtype=liquid_grid.dtype),
        jnp.asarray(-jnp.inf, dtype=liquid_grid.dtype),
    )
    s_kink = (limit_minus - income) / gross_return
    value_limit_minus = _extrapolate(
        liquid_grid, next_value, last_below - 1, last_below, asset_limit
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


def _crra_utility(consumption: Float1D, crra: ScalarFloat | float) -> Float1D:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )
