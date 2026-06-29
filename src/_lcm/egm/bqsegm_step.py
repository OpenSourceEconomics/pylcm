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

import jax.numpy as jnp

from _lcm.egm.bqsegm_segments import mask_dead_candidates, segment_ids_from_folds
from _lcm.egm.upper_envelope.query import envelope_at_query
from lcm.case_piece import EqualityOwner
from lcm.typing import Float1D, FloatND, IntND, ScalarFloat


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
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one period of a piecewise-affine, continuous-budget regime by EGM.

    The breakpoints partition the liquid axis into intervals on which cash-on-hand
    is affine, `coh = slope_i * liquid + intercept_i`. With a continuous budget
    (every breakpoint a kink, no jump), `coh(liquid)` is continuous and monotone, so
    its inverse is a single continuous map. EGM therefore runs once in `coh` space —
    `coh = consumption + savings` — and the endogenous liquid is recovered by
    inverting `coh(liquid)` on the grid, leaving no interval seam to under-cover. The
    marginal value of liquid scales by the active interval's slope (the envelope
    theorem through the affine budget). The hard borrowing corner competes over the
    whole grid, and the interior path and corner merge by the branch-aware upper
    envelope.

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

    Returns:
        Tuple of this period's value, marginal value of liquid, and consumption
        policy, each on `liquid_grid`.

    """
    # Cash-on-hand is continuous and monotone in liquid, so its inverse is a single
    # continuous map: EGM runs once in coh space and the endogenous liquid is read
    # by inverting `coh(liquid)` on the grid, with no interval seams to leave gaps.
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
    slope_endog = coh_slopes[jnp.searchsorted(breakpoints, liquid_endog, side="right")]
    value_endog = _crra_utility(consumption, crra) + discount_factor * value_next
    marginal_endog = slope_endog * consumption ** (-crra)
    # A non-concave (convex-kinked) budget can fold the interior path back, so keep
    # its monotone runs apart for the upper envelope.
    interior_segment = segment_ids_from_folds(endog_grid=liquid_endog)

    # Hard borrowing corner: save nothing, consume all of this point's cash-on-hand,
    # land next-period liquid at `income`. A candidate over the whole grid, since the
    # constraint binds wherever the no-save corner beats the Euler path.
    value_at_income = jnp.interp(jnp.asarray(income), liquid_grid, next_value)
    s0_value = _crra_utility(coh_grid, crra) + discount_factor * value_at_income
    s0_marginal = coh_slopes[interval_of_grid] * coh_grid ** (-crra)
    s0_segment = jnp.full_like(liquid_grid, jnp.nanmax(interior_segment) + 1.0)

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.concatenate([liquid_endog, liquid_grid]),
        policy=jnp.concatenate([consumption, coh_grid]),
        value=jnp.concatenate([value_endog, s0_value]),
        marginal=jnp.concatenate([marginal_endog, s0_marginal]),
        segment_id=jnp.concatenate([interior_segment, s0_segment]),
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
