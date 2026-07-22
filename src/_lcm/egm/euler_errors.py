"""Unit-free consumption Euler errors — a brute-free solution-accuracy metric.

The accuracy column of the DS comparison tables is the Euler error: at an interior
(unconstrained) consumption--saving optimum the Euler equation
`u'(c) = beta*(1+r)*u'(c_next)` holds exactly, so the relative gap between the chosen
consumption and the consumption the equation implies measures how well a method nulls
the first-order condition — independent of any reference solve. It is reported as the
base-10 logarithm of the relative consumption error, so `-3` reads as a 0.1% error.

The metric is meaningful only on the unconstrained interior: where the borrowing
constraint binds the Euler equation holds with a positive multiplier, and the residual
there reflects the constraint, not solution error.
"""

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from lcm.typing import Float1D, Float2D


def consumption_euler_error_log10(
    *,
    liquid_grid: Float1D,
    consumption: Float1D,
    next_consumption: Float1D,
    discount_factor: float,
    crra: float,
    return_liquid: float,
    income: float,
) -> Float1D:
    """Compute the log10 unit-free consumption Euler error at each liquid grid point.

    For chosen consumption `c` the next-period liquid state is
    `(1 + r)*(liquid - c) + income`, and the Euler equation implies
    `c_euler = (beta*(1+r)*u'(c_next))**(-1/crra)`, with `c_next` the next-period
    consumption policy interpolated at the next-period liquid state. The error is
    `log10(|c_euler / c - 1|)`.

    Args:
        liquid_grid: Regular liquid-state grid (ascending) the policy is defined on.
        consumption: Chosen consumption policy on `liquid_grid`.
        next_consumption: Next period's consumption policy on `liquid_grid`. Pass the
            identity `liquid_grid` for a terminal bequest (all wealth consumed).
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.
        return_liquid: Liquid net return `r`.
        income: Deterministic income added to next-period liquid.

    Returns:
        The base-10 log relative consumption error at each liquid grid point, shape
        `(len(liquid_grid),)`.

    """
    next_liquid = (1.0 + return_liquid) * (liquid_grid - consumption) + income
    consumption_next = jnp.interp(next_liquid, liquid_grid, next_consumption)
    marginal_next = consumption_next ** (-crra)
    consumption_euler = (discount_factor * (1.0 + return_liquid) * marginal_next) ** (
        -1.0 / crra
    )
    relative_error = jnp.abs(consumption_euler / consumption - 1.0)
    return jnp.log10(relative_error)


def working_consumption_euler_error_log10(
    *,
    m_grid: Float1D,
    n_grid: Float1D,
    consumption: Float2D,
    deposit: Float2D,
    next_consumption: Float2D,
    discount_factor: float,
    crra: float,
    return_liquid: float,
    return_pension: float,
    match_rate: float,
    wage: float,
) -> Float2D:
    """Compute the log10 consumption Euler error at each working `(m, n)` grid point.

    The liquid-margin intertemporal first-order condition for the two-asset working
    problem is `u'(c_t) = beta*(1+r^a)*u'(c_{t+1})`, with the next working state
    `m' = (1+r^a)*(m - c - d) + wage`, `n' = (1+r^b)*(n + d + chi*log(1+d))` reached
    under the chosen policy and `c_{t+1}` the next period's working consumption policy
    bilinearly interpolated there. The error is `log10(|c_euler / c - 1|)`.

    Args:
        m_grid: Regular working liquid-state grid (ascending, evenly spaced).
        n_grid: Regular working pension-state grid (ascending, evenly spaced).
        consumption: This period's consumption policy on the `(m, n)` grid.
        deposit: This period's deposit policy on the `(m, n)` grid.
        next_consumption: Next period's working consumption policy on the `(m, n)` grid.
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.
        return_liquid: Liquid net return `r^a`.
        return_pension: Pension net return `r^b`.
        match_rate: Pension employer-match coefficient `chi`.
        wage: Deterministic labor income.

    Returns:
        The base-10 log relative consumption error per `(m, n)` grid point.

    """
    m_mesh, n_mesh = jnp.meshgrid(m_grid, n_grid, indexing="ij")
    liquid_next = (1.0 + return_liquid) * (m_mesh - consumption - deposit) + wage
    pension_next = (1.0 + return_pension) * (
        n_mesh + deposit + match_rate * jnp.log1p(deposit)
    )
    m_index = (liquid_next - m_grid[0]) / (m_grid[1] - m_grid[0])
    n_index = (pension_next - n_grid[0]) / (n_grid[1] - n_grid[0])
    consumption_next = map_coordinates(
        next_consumption, [m_index, n_index], order=1, mode="nearest"
    )
    marginal_next = consumption_next ** (-crra)
    consumption_euler = (discount_factor * (1.0 + return_liquid) * marginal_next) ** (
        -1.0 / crra
    )
    relative_error = jnp.abs(consumption_euler / consumption - 1.0)
    return jnp.log10(relative_error)
