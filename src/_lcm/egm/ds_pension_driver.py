"""Backward-induction driver solving the DS pension model by the G2EGM method.

Assembles the endogenous-grid steps into a full lifecycle solve of the DS pension model:

- the terminal period is the CRRA bequest on the liquid grid;
- retired periods are the 1-D consumption--saving EGM step (`egm_one_asset_step`),
  carrying the marginal value of liquid backward;
- the single working->retired boundary period is the four-segment G2EGM step reading the
  1-D retired continuation through the lump-sum payout (`g2egm_retiring_step`);
- earlier working periods are the four-segment G2EGM step reading the 2-D working
  continuation (`g2egm_step`).

The working utility carries an additive work disutility the generic envelope objective
omits; it is an additive constant (it shifts the value level without changing the
policy), so the driver subtracts it from each working period's value.

This standalone driver is the numerical core validated against the brute solve before it
is wrapped as a prime-time `Solver`. It threads raw grids and scalar parameters rather
than the engine's regime machinery.
"""

import jax.numpy as jnp

from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from _lcm.egm.two_asset_g2egm_step import g2egm_retiring_step, g2egm_step
from lcm.typing import Float1D, FloatND, RegimeName


def solve_ds_pension_g2egm(
    *,
    n_periods: int,
    retirement_period: int,
    liquid_grid: Float1D,
    pension_grid: Float1D,
    a_grid: Float1D,
    b_grid: Float1D,
    consumption_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: float,
    crra: float,
    work_disutility: float,
    match_rate: float,
    return_liquid: float,
    return_pension: float,
    wage: float,
    retirement_income: float,
    pension_payout_return: float,
    threshold: float = 0.25,
) -> dict[int, dict[RegimeName, FloatND]]:
    """Solve the DS pension model by backward induction with the G2EGM steps.

    Args:
        n_periods: Number of lifecycle periods (the last is the terminal dead period).
        retirement_period: First retired period; working spans `0..retirement_period-1`.
        liquid_grid: Regular liquid-state grid, shared by working and retired (the
            working `m` grid and the retired state grid).
        pension_grid: Regular working pension-state grid (the working `n` grid).
        a_grid: Liquid post-decision grid for the working `ucon`/`dcon` segments.
        b_grid: Pension post-decision grid for the working segments.
        consumption_grid: Consumption sweep for the working `acon`/`con` segments.
        savings_grid: Post-decision savings grid for the retired EGM step.
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.
        work_disutility: Additive disutility of work, subtracted from working values.
        match_rate: Pension employer-match coefficient `chi`.
        return_liquid: Liquid net return `r^a`.
        return_pension: Pension net return `r^b`.
        wage: Deterministic labor income while working.
        retirement_income: Retirement income added to the retired liquid state.
        pension_payout_return: Factor the pension is paid out at on retirement.
        threshold: Barycentric extrapolation tolerance for the working envelope.

    Returns:
        Mapping of period to a mapping of regime name to this model's value array:
        `working` (2-D on `(liquid, pension)`) for `0..retirement_period-1`, `retired`
        (1-D on `liquid`) for `retirement_period..n_periods-2`, and `dead` (the terminal
        bequest) for `n_periods-1`.

    """
    bequest_value = _crra_utility(liquid_grid, crra)
    bequest_marginal = liquid_grid ** (-crra)

    solution: dict[int, dict[RegimeName, FloatND]] = {
        n_periods - 1: {"dead": bequest_value}
    }

    # Retired periods, backward to the first retired period. The continuation is the
    # terminal bequest at the last retired period, else next period's retired value.
    retired_marginal = bequest_marginal
    next_retired_value = bequest_value
    for period in range(n_periods - 2, retirement_period - 1, -1):
        value, retired_marginal = egm_one_asset_step(
            next_value=next_retired_value,
            next_marginal=retired_marginal,
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            crra=crra,
            return_liquid=return_liquid,
            income=retirement_income,
        )
        solution[period] = {"retired": value}
        next_retired_value = value

    # Working->retired boundary period: read the 1-D retired continuation through the
    # lump-sum payout.
    boundary_value = g2egm_retiring_step(
        next_value_retired=next_retired_value,
        next_marginal_retired=retired_marginal,
        liquid_grid=liquid_grid,
        m_grid=liquid_grid,
        n_grid=pension_grid,
        a_grid=a_grid,
        b_grid=b_grid,
        consumption_grid=consumption_grid,
        discount_factor=discount_factor,
        crra=crra,
        match_rate=match_rate,
        return_liquid=return_liquid,
        pension_payout_return=pension_payout_return,
        retirement_income=retirement_income,
        threshold=threshold,
    )
    next_working_value = boundary_value - work_disutility
    solution[retirement_period - 1] = {"working": next_working_value}

    # Earlier working periods: read the 2-D working continuation.
    for period in range(retirement_period - 2, -1, -1):
        value = g2egm_step(
            next_value=next_working_value,
            m_grid=liquid_grid,
            n_grid=pension_grid,
            a_grid=a_grid,
            b_grid=b_grid,
            consumption_grid=consumption_grid,
            discount_factor=discount_factor,
            crra=crra,
            match_rate=match_rate,
            return_liquid=return_liquid,
            return_pension=return_pension,
            wage=wage,
            threshold=threshold,
        )
        next_working_value = value - work_disutility
        solution[period] = {"working": next_working_value}

    return solution


def _crra_utility(consumption: Float1D, crra: float) -> Float1D:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )
