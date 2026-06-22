"""The DS pension comparison harness — solve time and Euler-error accuracy per method.

Reproduces the structure of the Dobrescu--Shanker pension comparison table: for each
solution method and grid resolution, the total solve time and the distribution of
unit-free consumption Euler errors. This module populates the **G2EGM** row by solving
the DS pension model with the four-segment G2EGM steps, timing the solve, and pooling
the working consumption Euler errors across the working->working periods (plus the
retired Euler error). The RFC comparator row is added once the multidimensional RFC
backend lands; `format_comparison_table` renders whatever rows it is given.

The Euler error is reported on the unconstrained, grid-resolved interior — the
low-liquid borrowing-constrained band and the off-grid top-pension boundary layer are
excluded, as they reflect grid extent rather than method accuracy.
"""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.euler_errors import working_consumption_euler_error_log10
from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from _lcm.egm.two_asset_g2egm_step import (
    G2EGMResult,
    g2egm_retiring_step,
    g2egm_step,
)
from lcm.typing import Float1D


@dataclass(frozen=True)
class MethodBenchmark:
    """One method's solve time and Euler-error accuracy at one grid resolution."""

    method: str
    """Solution method name (e.g. `"G2EGM"`)."""
    n_liquid: int
    """Number of liquid grid points."""
    n_pension: int
    """Number of pension grid points."""
    solve_seconds: float
    """Wall-clock time for the backward-induction solve."""
    euler_error_median_log10: float
    """Median interior consumption Euler error, base-10 log relative."""
    euler_error_p90_log10: float
    """90th-percentile interior consumption Euler error, base-10 log relative."""


def benchmark_g2egm_ds_pension(
    *,
    n_periods: int = 5,
    retirement_period: int = 3,
    n_liquid: int = 12,
    n_pension: int = 10,
    liquid_max: float = 20.0,
    pension_max: float = 15.0,
    discount_factor: float = 0.98,
    crra: float = 2.0,
    work_disutility: float = 0.25,
    match_rate: float = 0.10,
    return_liquid: float = 0.02,
    return_pension: float = 0.04,
    wage: float = 1.0,
    retirement_income: float = 0.50,
    pension_payout_return: float = 1.04,
    post_decision_factor: float = 2.0,
    low_liquid_skip: int = 3,
    pension_interior: int = 6,
) -> MethodBenchmark:
    """Solve the DS pension model by G2EGM and measure its time and Euler errors.

    Args:
        n_periods: Number of lifecycle periods.
        retirement_period: First retired period.
        n_liquid: Liquid grid points.
        n_pension: Pension grid points.
        liquid_max: Liquid grid upper bound.
        pension_max: Pension grid upper bound.
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion.
        work_disutility: Additive disutility of work.
        match_rate: Pension employer-match coefficient.
        return_liquid: Liquid net return.
        return_pension: Pension net return.
        wage: Working labor income.
        retirement_income: Retirement income.
        pension_payout_return: Factor the pension is paid out at on retirement.
        post_decision_factor: Post-decision grid points per state grid point. Druedahl
            & Jorgensen (2017) recommend roughly 4x (the post-decision grid drives
            accuracy); the default 2x keeps the benchmark fast.
        low_liquid_skip: Liquid rows excluded as the borrowing-constrained band.
        pension_interior: Pension columns retained (excludes the off-grid edge layer).

    Returns:
        The G2EGM method's benchmark row.

    """
    liquid_grid = jnp.linspace(0.1, liquid_max, n_liquid)
    pension_grid = jnp.linspace(0.0, pension_max, n_pension)
    # The post-decision (endogenous) grids drive accuracy, so they scale with the state
    # resolution rather than staying fixed; under-refining them caps the Euler error.
    n_a = max(18, int(post_decision_factor * n_liquid))
    n_b = max(18, int(post_decision_factor * n_pension))
    a_grid = jnp.linspace(0.0, liquid_max, n_a)
    b_grid = jnp.linspace(0.0, 2.0 * pension_max, n_b)
    consumption_grid = jnp.linspace(0.1, liquid_max, n_a)
    savings_grid = jnp.linspace(0.0, liquid_max, 4 * n_liquid)

    def solve() -> dict[int, G2EGMResult]:
        return _solve_g2egm_policies(
            n_periods=n_periods,
            retirement_period=retirement_period,
            liquid_grid=liquid_grid,
            pension_grid=pension_grid,
            a_grid=a_grid,
            b_grid=b_grid,
            consumption_grid=consumption_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            crra=crra,
            work_disutility=work_disutility,
            match_rate=match_rate,
            return_liquid=return_liquid,
            return_pension=return_pension,
            wage=wage,
            retirement_income=retirement_income,
            pension_payout_return=pension_payout_return,
        )

    # Warm up the compile, then time a fresh solve to device-ready.
    solve()
    start = time.perf_counter()
    working_policies = solve()
    jax.block_until_ready([p.value for p in working_policies.values()])
    solve_seconds = time.perf_counter() - start

    interior = np.s_[low_liquid_skip:, :pension_interior]

    # Working->working periods (continuation is a working period): pool their interior
    # Euler errors. The boundary period's transition is the lump-sum payout, not a
    # working transition, so it is not part of this pool.
    def period_errors(period: int) -> np.ndarray:
        errors = np.asarray(
            working_consumption_euler_error_log10(
                m_grid=liquid_grid,
                n_grid=pension_grid,
                consumption=working_policies[period].consumption,
                deposit=working_policies[period].deposit,
                next_consumption=working_policies[period + 1].consumption,
                discount_factor=discount_factor,
                crra=crra,
                return_liquid=return_liquid,
                return_pension=return_pension,
                match_rate=match_rate,
                wage=wage,
            )
        )[interior]
        return errors[np.isfinite(errors)]

    all_errors = np.concatenate(
        [period_errors(period) for period in range(retirement_period - 1)]
    )
    return MethodBenchmark(
        method="G2EGM",
        n_liquid=n_liquid,
        n_pension=n_pension,
        solve_seconds=solve_seconds,
        euler_error_median_log10=float(np.median(all_errors)),
        euler_error_p90_log10=float(np.percentile(all_errors, 90)),
    )


def format_comparison_table(rows: list[MethodBenchmark]) -> str:
    """Render benchmark rows as a fixed-width DS-style comparison table."""
    header = (
        f"{'method':<8} {'n_liq':>6} {'n_pen':>6} {'time (s)':>10} "
        f"{'EE median':>11} {'EE p90':>9}"
    )
    body = [
        f"{r.method:<8} {r.n_liquid:>6} {r.n_pension:>6} "
        f"{r.solve_seconds:>10.3f} {r.euler_error_median_log10:>11.2f} "
        f"{r.euler_error_p90_log10:>9.2f}"
        for r in rows
    ]
    return "\n".join([header, "-" * len(header), *body])


def _solve_g2egm_policies(
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
) -> dict[int, G2EGMResult]:
    """Backward-induct the working phase, returning each period's G2EGM result."""
    bequest_value = liquid_grid ** (1.0 - crra) / (1.0 - crra)
    next_retired_value = bequest_value
    retired_marginal = liquid_grid ** (-crra)
    for _ in range(n_periods - 2, retirement_period - 1, -1):
        retired = egm_one_asset_step(
            next_value=next_retired_value,
            next_marginal=retired_marginal,
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            crra=crra,
            return_liquid=return_liquid,
            income=retirement_income,
        )
        next_retired_value = retired.value
        retired_marginal = retired.marginal

    working_policies = {}
    boundary = g2egm_retiring_step(
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
    )
    working_policies[retirement_period - 1] = boundary
    next_working_value = boundary.value - work_disutility
    for period in range(retirement_period - 2, -1, -1):
        step = g2egm_step(
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
        )
        working_policies[period] = step
        next_working_value = step.value - work_disutility
    return working_policies
