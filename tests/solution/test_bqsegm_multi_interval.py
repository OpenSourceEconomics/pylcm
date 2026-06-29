"""Multi-interval BQSEGM step against a dense single-period brute oracle.

A piecewise-affine budget partitions the liquid axis into intervals on which
cash-on-hand is affine. The multi-interval step runs EGM per interval, masks each
candidate to its interval, and merges by the branch-aware upper envelope. A
continuous-kink budget (a tax bracket with no jump) produces a continuous,
kinked-but-monotone cash-on-hand; the step must reproduce the dense-consumption
Bellman max where both are exact.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.bqsegm_step import (
    bqsegm_discrete_envelope_step,
    bqsegm_jump_step,
    bqsegm_multi_interval_step,
    bqsegm_unified_step,
)


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


def test_multi_interval_step_matches_brute_through_a_hard_constraint_floor():
    """A cash-on-hand floor (a slope-0 segment) yields a flat value where it binds.

    Cash-on-hand is topped up to a floor below a threshold liquid level, so its
    slope is zero there and the value is constant across the floored region. The
    step's flat-corner handling reproduces the dense Bellman max both where the
    floor binds and on the unconstrained segment above it.
    """
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    base = 2.0
    floor = 5.0
    knot = floor - base

    liquid_grid = jnp.linspace(0.1, 30.0, 160)
    savings_grid = jnp.linspace(0.0, 28.0, 200)

    def coh_of_liquid(liquid):
        return jnp.maximum(liquid + base, floor)

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

    coh_slopes = jnp.asarray([0.0, 1.0])
    coh_intercepts = jnp.asarray([floor, base])
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
        flat_interval_mask=(True, False),
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
    interior = (np.asarray(liquid_grid) > 0.5) & (np.asarray(liquid_grid) < 28.0)
    np.testing.assert_allclose(
        np.asarray(value)[interior], np.asarray(brute)[interior], atol=2e-2, rtol=5e-3
    )


def test_jump_step_matches_brute_through_two_additive_cliffs():
    """Two downward subsidy cliffs (three levels) reproduce the dense oracle.

    Cash-on-hand jumps down as liquid crosses each asset cliff (the subsidy drops),
    so the budget is discontinuous and the value function jumps there. The masked
    three-case merge must track the dense Bellman max on each side of both cliffs.
    """
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    subsidy_levels = jnp.asarray([3.0, 1.5, 0.5])
    jump_breakpoints = jnp.asarray([5.0, 12.0])

    liquid_grid = jnp.linspace(0.1, 30.0, 200)
    savings_grid = jnp.linspace(0.0, 28.0, 220)

    def coh_of_liquid(liquid):
        interval = jnp.searchsorted(jump_breakpoints, liquid, side="right")
        return liquid + subsidy_levels[interval]

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

    value, _marginal, _policy = bqsegm_jump_step(
        next_value=next_value_of_liquid(liquid_grid),
        next_marginal=next_marginal_of_liquid(liquid_grid),
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        crra=crra,
        gross_return=gross_return,
        income=income,
        subsidy_levels=subsidy_levels,
        jump_breakpoints=jump_breakpoints,
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
    # Exclude a one-cell neighbourhood of each cliff, where the grid straddles the
    # discontinuity, as the recurring-jump agreement test does.
    liquid = np.asarray(liquid_grid)
    near_cliff = (np.abs(liquid - 5.0) < 0.3) | (np.abs(liquid - 12.0) < 0.3)
    interior = (liquid > 1.0) & (liquid < 27.0) & ~near_cliff
    np.testing.assert_allclose(
        np.asarray(value)[interior], np.asarray(brute)[interior], atol=3e-2, rtol=8e-3
    )


def test_unified_step_matches_brute_through_a_jump_then_a_kink():
    """A budget with one jump and one continuous kink reproduces the dense oracle.

    Cash-on-hand jumps down at a subsidy cliff and bends at a tax bracket above it,
    so the budget mixes a discontinuity and a continuous kink. The unified step
    solves each continuous case by coh inversion and masks across the jump, tracking
    the dense Bellman max on both sides of the cliff and through the kink.
    """
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    sub_low = 0.5
    sub_high = 3.0
    cliff = 6.0
    rate = 0.3
    exemption = 14.0

    liquid_grid = jnp.linspace(0.1, 30.0, 200)
    savings_grid = jnp.linspace(0.0, 28.0, 220)

    def coh_of_liquid(liquid):
        subsidy = jnp.where(liquid < cliff, sub_high, sub_low)
        tax = rate * jnp.maximum(liquid - exemption, 0.0)
        return liquid + subsidy - tax

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

    # Intervals: [0,cliff): coh=liquid+sub_high; [cliff,exemption): liquid+sub_low;
    # [exemption,inf): (1-rate)*liquid + sub_low + rate*exemption.
    coh_slopes = jnp.asarray([1.0, 1.0, 1.0 - rate])
    coh_intercepts = jnp.asarray([sub_high, sub_low, sub_low + rate * exemption])
    breakpoints = jnp.asarray([cliff, exemption])
    jump_mask = (True, False)

    value, _marginal, _policy = bqsegm_unified_step(
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
        jump_mask=jump_mask,
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
    liquid = np.asarray(liquid_grid)
    near = (np.abs(liquid - cliff) < 0.3) | (np.abs(liquid - exemption) < 0.3)
    interior = (liquid > 1.0) & (liquid < 27.0) & ~near
    np.testing.assert_allclose(
        np.asarray(value)[interior], np.asarray(brute)[interior], atol=3e-2, rtol=8e-3
    )


def test_discrete_envelope_step_matches_brute_over_a_binary_choice():
    """A binary discrete choice over two budgets reproduces the dense oracle.

    Two discrete options (e.g. buy private insurance or not) each shift cash-on-hand
    differently; the value is the upper envelope over the two BQSEGM solves. The
    discrete-envelope step must equal a dense brute that maximises over both the
    discrete choice and consumption.
    """
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    base_yes = 4.0  # higher cash-on-hand level when buying
    base_no = 1.0

    liquid_grid = jnp.linspace(0.1, 30.0, 160)
    savings_grid = jnp.linspace(0.0, 28.0, 200)

    def coh_yes(liquid):
        return liquid + base_yes

    def coh_no(liquid):
        return liquid + base_no

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

    choices = (
        {
            "coh_slopes": jnp.asarray([1.0]),
            "coh_intercepts": jnp.asarray([base_yes]),
            "breakpoints": jnp.zeros((0,)),
        },
        {
            "coh_slopes": jnp.asarray([1.0]),
            "coh_intercepts": jnp.asarray([base_no]),
            "breakpoints": jnp.zeros((0,)),
        },
    )

    value, _marginal, _policy, choice = bqsegm_discrete_envelope_step(
        next_value=next_value_of_liquid(liquid_grid),
        next_marginal=next_marginal_of_liquid(liquid_grid),
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        crra=crra,
        gross_return=gross_return,
        income=income,
        choices=choices,
    )

    # Dense brute over both the discrete choice and consumption.
    def brute_choice(coh_of_liquid):
        coh = coh_of_liquid(liquid_grid)
        fractions = jnp.linspace(1e-4, 1.0, 4000)
        consumption = fractions[:, None] * coh[None, :]
        savings = coh[None, :] - consumption
        next_liquid = gross_return * savings + income
        val = _crra(consumption, crra) + discount_factor * next_value_of_liquid(
            next_liquid
        )
        return jnp.max(val, axis=0)

    brute = jnp.maximum(brute_choice(coh_yes), brute_choice(coh_no))
    interior = (np.asarray(liquid_grid) > 1.0) & (np.asarray(liquid_grid) < 28.0)
    np.testing.assert_allclose(
        np.asarray(value)[interior], np.asarray(brute)[interior], atol=2e-2, rtol=5e-3
    )
    # The higher-cash-on-hand option is always at least as good here.
    assert np.all(np.asarray(choice)[interior] == 0)


def test_discrete_envelope_step_smooths_with_an_ev1_taste_shock():
    """With a taste-shock scale the value is the scaled logsum over the choices.

    EV1 taste shocks smooth the discrete choice: the value is `scale * logsumexp`
    over the per-choice BQSEGM values, matching a dense brute that takes the same
    logsum over each choice's Bellman max.
    """
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    base_yes = 2.5
    base_no = 2.0
    scale = 0.5

    liquid_grid = jnp.linspace(0.1, 30.0, 160)
    savings_grid = jnp.linspace(0.0, 28.0, 200)

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

    choices = (
        {
            "coh_slopes": jnp.asarray([1.0]),
            "coh_intercepts": jnp.asarray([base_yes]),
            "breakpoints": jnp.zeros((0,)),
        },
        {
            "coh_slopes": jnp.asarray([1.0]),
            "coh_intercepts": jnp.asarray([base_no]),
            "breakpoints": jnp.zeros((0,)),
        },
    )

    value, _marginal, _policy, _choice = bqsegm_discrete_envelope_step(
        next_value=next_value_of_liquid(liquid_grid),
        next_marginal=next_marginal_of_liquid(liquid_grid),
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        crra=crra,
        gross_return=gross_return,
        income=income,
        choices=choices,
        taste_shock_scale=scale,
    )

    def brute_choice(base):
        coh = liquid_grid + base
        fractions = jnp.linspace(1e-4, 1.0, 4000)
        consumption = fractions[:, None] * coh[None, :]
        savings = coh[None, :] - consumption
        next_liquid = gross_return * savings + income
        val = _crra(consumption, crra) + discount_factor * next_value_of_liquid(
            next_liquid
        )
        return jnp.max(val, axis=0)

    stacked = jnp.stack([brute_choice(base_yes), brute_choice(base_no)])
    brute = scale * jax.scipy.special.logsumexp(stacked / scale, axis=0)
    interior = (np.asarray(liquid_grid) > 1.0) & (np.asarray(liquid_grid) < 28.0)
    np.testing.assert_allclose(
        np.asarray(value)[interior], np.asarray(brute)[interior], atol=2e-2, rtol=5e-3
    )
