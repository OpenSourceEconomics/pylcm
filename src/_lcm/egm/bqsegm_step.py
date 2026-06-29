"""One-period BQSEGM step for a 1-D consumption--saving regime with a case split.

A binary case boundary on the liquid state (`liquid < asset_limit`) shifts an
additive subsidy into cash-on-hand. Within each case the budget is smooth, so the
period solves by ordinary 1-D EGM on `coh = liquid + subsidy`; the recovered
endogenous state is `liquid = coh - subsidy`. The two cases are then merged on the
liquid grid by the branch-aware upper envelope after NaN-dead masking each case to
its consistent region — the `when` case where the predicate holds, the `otherwise`
case where it fails. The strict `<` / non-strict `>=` split gives the `otherwise`
side ownership of the exact boundary, matching `equality="otherwise"`.
"""

import jax.numpy as jnp

from _lcm.egm.bqsegm_segments import mask_dead_candidates
from _lcm.egm.upper_envelope.query import envelope_at_query
from lcm.typing import Float1D, FloatND, IntND, ScalarFloat


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
    )

    when_valid = liquid_grid < asset_limit
    otherwise_valid = liquid_grid >= asset_limit
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
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one case's smooth 1-D consumption--saving sub-problem by EGM.

    `coh = liquid + subsidy`, so the endogenous state recovered from the Euler
    inversion is `liquid = consumption + savings - subsidy`. Below the smallest
    endogenous liquid the borrowing constraint binds: the agent consumes all
    cash-on-hand and saves nothing. The continuation is read kink-aware at
    `asset_limit`: the next-period value and marginal jump there, so a query
    landing in the boundary cell reads the correct one-sided branch rather than a
    bridged average (the topology-preserving continuation read).
    """
    gross_return = 1.0 + return_liquid
    next_liquid = gross_return * savings_grid + income
    value_next = _kink_aware_interp(next_liquid, liquid_grid, next_value, asset_limit)
    marginal_next = _kink_aware_interp(
        next_liquid, liquid_grid, next_marginal, asset_limit
    )

    consumption = (discount_factor * gross_return * marginal_next) ** (-1.0 / crra)
    liquid_endog = consumption + savings_grid - subsidy
    value_endog = _crra_utility(consumption, crra) + discount_factor * value_next

    interior_value = jnp.interp(liquid_grid, liquid_endog, value_endog)
    interior_consumption = jnp.interp(liquid_grid, liquid_endog, consumption)

    constrained = liquid_grid < liquid_endog[0]
    value_at_zero_savings = _kink_aware_interp(
        jnp.asarray(income), liquid_grid, next_value, asset_limit
    )
    constrained_consumption = liquid_grid + subsidy
    constrained_value = (
        _crra_utility(constrained_consumption, crra)
        + discount_factor * value_at_zero_savings
    )
    consumption_on_grid = jnp.where(
        constrained, constrained_consumption, interior_consumption
    )
    value = jnp.where(constrained, constrained_value, interior_value)
    marginal = consumption_on_grid ** (-crra)
    return value, marginal, consumption_on_grid


def _kink_aware_interp(
    query: FloatND,
    grid: Float1D,
    values: Float1D,
    limit: ScalarFloat | float,
) -> FloatND:
    """Interpolate `values` on `grid` without bridging the jump at `limit`.

    The continuation carries a value jump at the case boundary: the grid node
    just below `limit` holds the `when`-side value and the node just above holds
    the `otherwise`-side value, so a plain linear interpolation across that cell
    returns a meaningless average. Two extra abscissae are inserted at `limit`
    (split by a negligible epsilon) carrying each side's value, linearly
    extrapolated from the two nearest same-side nodes. A query below `limit` then
    interpolates within the `when` branch, a query above within the `otherwise`
    branch, and neither bridges the discontinuity.
    """
    n = grid.shape[0]
    last_below = jnp.clip(jnp.sum(grid < limit) - 1, 1, n - 3).astype(jnp.int32)
    left_at_limit = _extrapolate(grid, values, last_below - 1, last_below, limit)
    right_at_limit = _extrapolate(grid, values, last_below + 1, last_below + 2, limit)

    eps = jnp.asarray(1e-9, dtype=grid.dtype)
    aug_grid = jnp.concatenate([grid, jnp.stack([limit - eps, limit + eps])])
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
