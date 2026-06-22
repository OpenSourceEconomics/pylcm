"""The unit-free consumption Euler error measures a solution's accuracy.

For an interior (unconstrained) consumption--saving optimum the Euler equation
`u'(c) = beta*(1+r)*u'(c_next)` holds exactly, so the relative gap between the chosen
consumption and the consumption the Euler equation implies is a brute-free accuracy
metric. It is reported as `log10` of the relative consumption error (e.g. `-3` is a
0.1% error). The endogenous grid method nulls the interior Euler residual by
construction, so a correct retired solution has tiny interior Euler errors.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.euler_errors import (
    consumption_euler_error_log10,
    working_consumption_euler_error_log10,
)
from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from _lcm.egm.two_asset_g2egm_step import g2egm_retiring_step, g2egm_step

_LIQUID_GRID = jnp.linspace(0.1, 20.0, 12)
_SAVINGS_GRID = jnp.linspace(0.0, 20.0, 40)
_DISCOUNT, _CRRA, _RETURN, _INCOME = 0.98, 2.0, 0.02, 0.50
# The lowest liquid points are borrowing-constrained: the unconstrained Euler equation
# does not hold there (it holds with a constraint multiplier), so the metric is reported
# on the unconstrained interior.
_INTERIOR = np.s_[2:]


def test_constrained_consume_all_solution_has_an_exact_euler_residual():
    """A hand-built unconstrained interior point has its Euler error reproduce the gap.

    With `next_consumption = next_liquid` (consume everything next period) the implied
    consumption is `c_euler = (beta*(1+r))**(-1/crra) * next_liquid`; planting a policy
    exactly equal to `c_euler` drives the Euler error to `-inf` (zero residual), and a
    10% overconsumption gives `log10(0.1)`.
    """
    liquid = jnp.array([10.0])
    next_liquid_if = lambda c: (1.0 + _RETURN) * (liquid - c) + _INCOME  # noqa: E731
    # Solve the fixed point c = (beta*(1+r))**(-1/crra) * ((1+r)(liquid-c)+income).
    k = (_DISCOUNT * (1.0 + _RETURN)) ** (-1.0 / _CRRA)
    c_star = k * ((1.0 + _RETURN) * liquid + _INCOME) / (1.0 + k * (1.0 + _RETURN))
    err = consumption_euler_error_log10(
        liquid_grid=liquid,
        consumption=c_star,
        next_consumption=next_liquid_if(c_star),
        discount_factor=_DISCOUNT,
        crra=_CRRA,
        return_liquid=_RETURN,
        income=_INCOME,
    )
    assert float(err[0]) < -8.0  # essentially zero residual


def _retired_median_euler_error(*, n_liquid):
    """Median interior Euler error of a one-step retired solve at a grid resolution."""
    liquid_grid = jnp.linspace(0.1, 20.0, n_liquid)
    savings_grid = jnp.linspace(0.0, 20.0, 4 * n_liquid)
    step = egm_one_asset_step(
        next_value=liquid_grid ** (1.0 - _CRRA) / (1.0 - _CRRA),
        next_marginal=liquid_grid ** (-_CRRA),
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
        return_liquid=_RETURN,
        income=_INCOME,
    )
    # The continuation is the terminal bequest: at death all wealth is consumed, so the
    # next-period consumption policy is the identity in liquid.
    errors = np.asarray(
        consumption_euler_error_log10(
            liquid_grid=liquid_grid,
            consumption=step.consumption,
            next_consumption=liquid_grid,
            discount_factor=_DISCOUNT,
            crra=_CRRA,
            return_liquid=_RETURN,
            income=_INCOME,
        )
    )
    return np.median(errors[_INTERIOR])


def test_retired_euler_error_converges_under_grid_refinement():
    """The interior Euler error shrinks as the liquid grid refines.

    The endogenous grid method nulls the Euler residual at the endogenous nodes;
    interpolating the policy back onto a coarse regular grid reintroduces it, so the
    residual is a resolution diagnostic that converges to zero. Refining the grid four
    times over drives the median interior error well below a percent.
    """
    coarse = _retired_median_euler_error(n_liquid=12)
    fine = _retired_median_euler_error(n_liquid=48)
    assert coarse < -1.5  # ~2% at the coarse oracle grid
    assert fine < -3.0  # below 0.1% once resolved
    assert fine < coarse - 1.0  # at least an order of magnitude better


_W = {
    "discount_factor": 0.98,
    "crra": 2.0,
    "match_rate": 0.10,
    "return_liquid": 0.02,
    "return_pension": 0.04,
    "wage": 1.0,
}
_WORK_DISUTILITY, _RET_INCOME, _PAYOUT = 0.25, 0.50, 1.04


def _working_first_period_euler_error(*, n_liquid):
    """Median interior working consumption Euler error two steps before retirement.

    Solves the DS pension working chain (retired -> boundary -> p1 -> p0) keeping the
    consumption policy at each step, then evaluates the working consumption Euler error
    at the first period from its own and the next period's policy.
    """
    m_grid = jnp.linspace(0.1, 20.0, n_liquid)
    n_grid = jnp.linspace(0.0, 15.0, 10)
    a_grid = jnp.linspace(0.0, 20.0, max(18, n_liquid + 2))
    b_grid = jnp.linspace(0.0, 30.0, 16)
    consumption_grid = jnp.linspace(0.1, 20.0, max(18, n_liquid + 2))

    def working(next_value):
        return g2egm_step(
            next_value=next_value,
            m_grid=m_grid,
            n_grid=n_grid,
            a_grid=a_grid,
            b_grid=b_grid,
            consumption_grid=consumption_grid,
            **_W,
        )

    v_dead = m_grid ** (1.0 - _W["crra"]) / (1.0 - _W["crra"])
    retired = egm_one_asset_step(
        next_value=v_dead,
        next_marginal=m_grid ** (-_W["crra"]),
        liquid_grid=m_grid,
        savings_grid=jnp.linspace(0.0, 20.0, 60),
        discount_factor=_W["discount_factor"],
        crra=_W["crra"],
        return_liquid=_W["return_liquid"],
        income=_RET_INCOME,
    )
    boundary = g2egm_retiring_step(
        next_value_retired=retired.value,
        next_marginal_retired=retired.marginal,
        liquid_grid=m_grid,
        m_grid=m_grid,
        n_grid=n_grid,
        a_grid=a_grid,
        b_grid=b_grid,
        consumption_grid=consumption_grid,
        discount_factor=_W["discount_factor"],
        crra=_W["crra"],
        match_rate=_W["match_rate"],
        return_liquid=_W["return_liquid"],
        pension_payout_return=_PAYOUT,
        retirement_income=_RET_INCOME,
    )
    period1 = working(boundary.value - _WORK_DISUTILITY)
    period0 = working(period1.value - _WORK_DISUTILITY)
    errors = np.asarray(
        working_consumption_euler_error_log10(
            m_grid=m_grid,
            n_grid=n_grid,
            consumption=period0.consumption,
            deposit=period0.deposit,
            next_consumption=period1.consumption,
            **_W,
        )
    )
    # Exclude the unresolved low-liquid band and the off-grid top-pension hole layer.
    return np.median(errors[3:, :7])


def test_working_consumption_euler_error_converges_under_grid_refinement():
    """The working consumption Euler error shrinks as the liquid grid refines."""
    coarse = _working_first_period_euler_error(n_liquid=12)
    fine = _working_first_period_euler_error(n_liquid=24)
    assert coarse < -1.0  # the chained two-asset solve is coarser than the 1-D retired
    assert fine < coarse - 0.3  # refinement improves it
