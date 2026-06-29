"""Multi-interval BQSEGM step against a dense single-period brute oracle.

A piecewise-affine budget partitions the liquid axis into intervals on which
cash-on-hand is affine. The multi-interval step runs EGM per interval, masks each
candidate to its interval, and merges by the branch-aware upper envelope. A
continuous-kink budget (a tax bracket with no jump) produces a continuous,
kinked-but-monotone cash-on-hand; the step must reproduce the dense-consumption
Bellman max where both are exact.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.bqsegm_step import bqsegm_multi_interval_step


def _crra(consumption, crra):
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )


def _dense_brute_value(
    *,
    liquid_grid,
    coh_of_liquid,
    next_value_of_liquid,
    crra,
    discount_factor,
    gross_return,
    income,
    n_consumption=4000,
):
    """Bellman max over a dense consumption grid — the convention-free oracle."""
    coh = coh_of_liquid(liquid_grid)
    fractions = jnp.linspace(1e-4, 1.0, n_consumption)
    # consumption = fraction * coh, broadcast over the liquid grid.
    consumption = fractions[:, None] * coh[None, :]
    savings = coh[None, :] - consumption
    next_liquid = gross_return * savings + income
    value = _crra(consumption, crra) + discount_factor * next_value_of_liquid(
        next_liquid
    )
    return jnp.max(value, axis=0)


def test_multi_interval_step_matches_brute_through_a_continuous_tax_kink():
    """A continuous tax-bracket kink in the budget reproduces the dense oracle.

    Cash-on-hand is `liquid + base` below an exemption and taxed above it, so its
    slope drops from 1 to 1 - rate at the bracket edge — a continuous kink, no
    jump. The two-interval EGM merge must track the dense-consumption Bellman max
    across the asset interior.
    """
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    base = 2.0
    rate = 0.3
    exemption = 12.0

    liquid_grid = jnp.linspace(0.1, 30.0, 160)
    savings_grid = jnp.linspace(0.0, 28.0, 200)

    def coh_of_liquid(liquid):
        return liquid + base - rate * jnp.maximum(liquid - exemption, 0.0)

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

    # Below the kink: coh = liquid + base (slope 1). Above: coh = (1-rate)*liquid +
    # base + rate*exemption.
    coh_slopes = jnp.asarray([1.0, 1.0 - rate])
    coh_intercepts = jnp.asarray([base, base + rate * exemption])
    breakpoints = jnp.asarray([exemption])

    value, _marginal, _policy = bqsegm_multi_interval_step(
        next_value=next_value_of_liquid(liquid_grid),
        next_marginal=next_marginal_of_liquid(liquid_grid),
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        crra=crra,
        gross_return=gross_return,
        income=income,
        coh_slopes=coh_slopes,
        coh_intercepts=coh_intercepts,
        breakpoints=breakpoints,
    )

    brute = _dense_brute_value(
        liquid_grid=liquid_grid,
        coh_of_liquid=coh_of_liquid,
        next_value_of_liquid=next_value_of_liquid,
        crra=crra,
        discount_factor=discount_factor,
        gross_return=gross_return,
        income=income,
    )

    interior = (np.asarray(liquid_grid) > 1.0) & (np.asarray(liquid_grid) < 28.0)
    np.testing.assert_allclose(
        np.asarray(value)[interior], np.asarray(brute)[interior], atol=2e-2, rtol=5e-3
    )


def test_multi_interval_step_matches_brute_through_a_convex_kink():
    """A convex budget kink (slope rising with liquid) still tracks the oracle.

    A subsidy that phases in above a threshold makes cash-on-hand convex in liquid,
    so the value function can be non-concave there — the case the branch-aware
    upper envelope exists for. The step must still reproduce the dense Bellman max.
    """
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    base = 2.0
    slope_low = 0.7
    slope_high = 1.3
    knot = 10.0

    liquid_grid = jnp.linspace(0.1, 30.0, 160)
    savings_grid = jnp.linspace(0.0, 28.0, 200)

    def coh_of_liquid(liquid):
        return (
            base
            + slope_low * jnp.minimum(liquid, knot)
            + slope_high * jnp.maximum(liquid - knot, 0.0)
        )

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

    coh_slopes = jnp.asarray([slope_low, slope_high])
    coh_intercepts = jnp.asarray([base, base + (slope_low - slope_high) * knot])
    breakpoints = jnp.asarray([knot])

    value, _marginal, _policy = bqsegm_multi_interval_step(
        next_value=next_value_of_liquid(liquid_grid),
        next_marginal=next_marginal_of_liquid(liquid_grid),
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        crra=crra,
        gross_return=gross_return,
        income=income,
        coh_slopes=coh_slopes,
        coh_intercepts=coh_intercepts,
        breakpoints=breakpoints,
    )
    brute = _dense_brute_value(
        liquid_grid=liquid_grid,
        coh_of_liquid=coh_of_liquid,
        next_value_of_liquid=next_value_of_liquid,
        crra=crra,
        discount_factor=discount_factor,
        gross_return=gross_return,
        income=income,
    )
    interior = (np.asarray(liquid_grid) > 1.0) & (np.asarray(liquid_grid) < 28.0)
    np.testing.assert_allclose(
        np.asarray(value)[interior], np.asarray(brute)[interior], atol=2e-2, rtol=5e-3
    )


def test_multi_interval_step_matches_brute_through_two_continuous_kinks():
    """Two bracket edges (three affine segments) still reproduce the dense oracle.

    A two-bracket tax bends cash-on-hand twice, partitioning the liquid axis into
    three affine intervals. The single coh-space EGM inversion tracks the dense
    Bellman max across all three, with no seam at either kink.
    """
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    base = 2.0
    rate_low = 0.2
    rate_high = 0.45
    knot_low = 8.0
    knot_high = 18.0

    liquid_grid = jnp.linspace(0.1, 30.0, 160)
    savings_grid = jnp.linspace(0.0, 28.0, 200)

    def coh_of_liquid(liquid):
        return (
            liquid
            + base
            - rate_low * jnp.clip(liquid - knot_low, 0.0, knot_high - knot_low)
            - rate_high * jnp.maximum(liquid - knot_high, 0.0)
        )

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

    # Segment slopes: 1, 1-rate_low, 1-rate_low-rate_high; intercepts continuous.
    coh_slopes = jnp.asarray([1.0, 1.0 - rate_low, 1.0 - rate_low - rate_high])
    coh_intercepts = jnp.asarray(
        [
            base,
            base + rate_low * knot_low,
            base + rate_low * knot_low + rate_high * knot_high,
        ]
    )
    breakpoints = jnp.asarray([knot_low, knot_high])

    value, _marginal, _policy = bqsegm_multi_interval_step(
        next_value=next_value_of_liquid(liquid_grid),
        next_marginal=next_marginal_of_liquid(liquid_grid),
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        crra=crra,
        gross_return=gross_return,
        income=income,
        coh_slopes=coh_slopes,
        coh_intercepts=coh_intercepts,
        breakpoints=breakpoints,
    )
    brute = _dense_brute_value(
        liquid_grid=liquid_grid,
        coh_of_liquid=coh_of_liquid,
        next_value_of_liquid=next_value_of_liquid,
        crra=crra,
        discount_factor=discount_factor,
        gross_return=gross_return,
        income=income,
    )
    interior = (np.asarray(liquid_grid) > 1.0) & (np.asarray(liquid_grid) < 28.0)
    np.testing.assert_allclose(
        np.asarray(value)[interior], np.asarray(brute)[interior], atol=2e-2, rtol=5e-3
    )
