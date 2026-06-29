"""BQSEGM coh-space inversion must drop off-grid endogenous points.

The Euler inversion recovers a current cash-on-hand `coh_endog` for each
post-decision savings node. A high savings node can imply a `coh_endog` above the
largest cash-on-hand the liquid grid can fund (`coh_grid[-1]`). That endogenous
point is off-grid: the agent would need more current liquid than the grid holds.
It must be dropped, not clipped onto the top liquid node — clipping admits an
infeasible high-consumption candidate that inflates the value through the
upper-asset region. Because the EGM value is a max over feasible candidates, it
must never exceed the dense feasible Bellman optimum; a value above brute is the
fingerprint of an admitted off-grid candidate.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.bqsegm_step import bqsegm_multi_interval_step, bqsegm_unified_step


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
    """Bellman max over a dense feasible consumption grid — convention-free oracle."""
    coh = coh_of_liquid(liquid_grid)
    fractions = jnp.linspace(1e-4, 1.0, n_consumption)
    consumption = fractions[:, None] * coh[None, :]
    savings = coh[None, :] - consumption
    next_liquid = gross_return * savings + income
    value = _crra(consumption, crra) + discount_factor * next_value_of_liquid(
        next_liquid
    )
    return jnp.max(value, axis=0)


def test_multi_interval_step_drops_off_grid_inverse_at_the_upper_boundary():
    """A continuous-kink budget never values above the dense feasible optimum, even
    where many savings nodes imply an off-grid endogenous cash-on-hand."""
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    base = 2.0
    rate = 0.3
    exemption = 2.4

    # A small liquid grid and a far-reaching savings grid: many savings nodes imply
    # `coh_endog` well above `coh_grid[-1]`, so the off-grid inverse, if clipped to
    # the top liquid node, admits a clearly infeasible candidate.
    liquid_grid = jnp.linspace(0.1, 6.0, 160)
    savings_grid = jnp.linspace(0.0, 200.0, 240)

    def coh_of_liquid(liquid):
        return liquid + base - rate * jnp.maximum(liquid - exemption, 0.0)

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

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
    # The EGM value is a max over feasible candidates, so it can never exceed the
    # dense feasible Bellman optimum by more than interpolation tolerance. A clipped
    # off-grid (infeasible high-consumption) candidate is the only thing that can
    # push it above brute — the F5 signature, concentrated at the top liquid node.
    excess = np.asarray(value) - np.asarray(brute)
    assert np.max(excess) <= 5e-3, f"max(step - brute) = {np.max(excess)}"


def test_unified_step_drops_off_grid_inverse_at_the_upper_boundary():
    """The mixed jump-and-kink step's top case must also drop off-grid inverses.

    The top case has an unbounded upper edge, so a clipped off-grid endogenous
    point lands inside the case and would otherwise be admitted. The value must
    never exceed the dense feasible Bellman optimum across the subsidy cliff.
    """
    crra = 2.0
    discount_factor = 0.95
    gross_return = 1.03
    income = 1.0
    base = 2.0
    subsidy = 1.0
    cliff = 2.4

    liquid_grid = jnp.linspace(0.1, 6.0, 160)
    savings_grid = jnp.linspace(0.0, 200.0, 240)

    def coh_of_liquid(liquid):
        # A downward subsidy cliff at `cliff`: coh = liquid + base + subsidy below,
        # liquid + base above (a jump down of `subsidy`).
        return liquid + base + subsidy * (liquid < cliff)

    def next_value_of_liquid(liquid):
        return _crra(liquid, crra)

    def next_marginal_of_liquid(liquid):
        return liquid ** (-crra)

    coh_slopes = jnp.asarray([1.0, 1.0])
    coh_intercepts = jnp.asarray([base + subsidy, base])
    breakpoints = jnp.asarray([cliff])

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
        jump_mask=(True,),
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
    excess = np.asarray(value) - np.asarray(brute)
    assert np.max(excess) <= 5e-3, f"max(step - brute) = {np.max(excess)}"
