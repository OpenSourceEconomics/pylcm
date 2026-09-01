"""The unit-free consumption Euler error measures a solution's accuracy.

For an interior (unconstrained) consumption--saving optimum the Euler equation
`u'(c) = beta*g'(A)*u'(c_next)` holds exactly, where `g` is the regime's own law of
motion and `A = liquid - c` the savings it is evaluated at. The relative gap between
the chosen consumption and the consumption the Euler equation implies is a brute-free
accuracy metric. It is reported as `log10` of the relative consumption error (e.g.
`-3` is a 0.1% error). The endogenous grid method nulls the interior Euler residual by
construction, so a correct retired solution has tiny interior Euler errors.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.euler_errors import consumption_euler_error_log10
from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from tests.conftest import X64_ENABLED
from tests.solution._crra_preferences import crra_preferences

# The "essentially zero" log10 Euler residual floor is set by float eps:
# roughly -15 at 64-bit, -7 at 32-bit; -8/-6 leave margin for rounding.
_EXACT_RESIDUAL_LOG10_BOUND = -8.0 if X64_ENABLED else -6.0

_LIQUID_GRID = jnp.linspace(0.1, 20.0, 12)
_SAVINGS_GRID = jnp.linspace(0.0, 20.0, 40)
_DISCOUNT, _CRRA, _RETURN, _INCOME = 0.98, 2.0, 0.02, 0.50


def _conventional_law(savings):
    """The conventional law of motion `(1 + r)*A + income`."""
    return (1.0 + _RETURN) * savings + _INCOME


def _law_readings(*, law, savings):
    """Evaluate a law of motion and its slope at each savings level.

    The same pair the solver reads off a regime's declared transition: where a level
    of savings lands next period, and how that landing point moves when savings move.
    """
    return jax.vmap(jax.value_and_grad(law))(savings)


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
    # Solve the fixed point c = (beta*(1+r))**(-1/crra) * ((1+r)(liquid-c)+income).
    k = (_DISCOUNT * (1.0 + _RETURN)) ** (-1.0 / _CRRA)
    c_star = k * ((1.0 + _RETURN) * liquid + _INCOME) / (1.0 + k * (1.0 + _RETURN))
    next_liquid, marginal_return = _law_readings(
        law=_conventional_law, savings=liquid - c_star
    )
    err = consumption_euler_error_log10(
        liquid_grid=liquid,
        consumption=c_star,
        next_consumption=next_liquid,
        discount_factor=_DISCOUNT,
        preferences=crra_preferences(crra=_CRRA),
        next_liquid=next_liquid,
        marginal_return=marginal_return,
    )
    assert float(err[0]) < _EXACT_RESIDUAL_LOG10_BOUND  # essentially zero residual


def _retired_median_euler_error(*, n_liquid):
    """Median interior Euler error of a one-step retired solve at a grid resolution."""
    liquid_grid = jnp.linspace(0.1, 20.0, n_liquid)
    savings_grid = jnp.linspace(0.0, 20.0, 4 * n_liquid)
    step = egm_one_asset_step(
        next_value=liquid_grid ** (1.0 - _CRRA) / (1.0 - _CRRA),
        next_marginal=liquid_grid ** (-_CRRA),
        liquid_grid=liquid_grid,
        next_liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=_DISCOUNT,
        preferences=crra_preferences(crra=_CRRA),
        next_liquid=(1.0 + _RETURN) * savings_grid + _INCOME,
        marginal_return=jnp.full_like(savings_grid, 1.0 + _RETURN),
    )
    next_liquid, marginal_return = _law_readings(
        law=_conventional_law, savings=liquid_grid - step.consumption
    )
    # The continuation is the terminal bequest: at death all wealth is consumed, so the
    # next-period consumption policy is the identity in liquid.
    errors = np.asarray(
        consumption_euler_error_log10(
            liquid_grid=liquid_grid,
            consumption=step.consumption,
            next_consumption=liquid_grid,
            discount_factor=_DISCOUNT,
            preferences=crra_preferences(crra=_CRRA),
            next_liquid=next_liquid,
            marginal_return=marginal_return,
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


# A per-period fixed cost is a term no rearrangement of `(1+r)*A + income` expresses.
# The income is large enough relative to the cost that every landing point stays well
# inside the grid: a landing point outside it is clamped by the interpolation, which
# would mask a wrong law rather than expose it.
_FIXED_COST = 1.5
_FIXED_COST_INCOME = 5.0
_FIXED_COST_LIQUID_GRID = jnp.linspace(0.5, 20.0, 24)


def _fixed_cost_law(savings):
    """A law carrying a per-period fixed cost: `(1 + r)*A + income - kappa`."""
    return (1.0 + _RETURN) * savings + _FIXED_COST_INCOME - _FIXED_COST


def test_euler_error_vanishes_for_a_policy_optimal_under_a_fixed_cost_law():
    """A policy solving the Euler equation of the declared law has no Euler error.

    The metric reads the law's landing point and slope rather than rebuilding a gross
    return and an income, so a law carrying a per-period fixed cost is scored against
    itself. Against a terminal consume-all continuation, CRRA gives the exactly
    Euler-optimal policy in closed form, whose residual is therefore essentially zero.
    """
    liquid_grid = _FIXED_COST_LIQUID_GRID
    # Solve c = (beta*(1+r))**(-1/crra) * ((1+r)(liquid-c) + income - kappa).
    k = (_DISCOUNT * (1.0 + _RETURN)) ** (-1.0 / _CRRA)
    c_star = (
        k
        * ((1.0 + _RETURN) * liquid_grid + _FIXED_COST_INCOME - _FIXED_COST)
        / (1.0 + k * (1.0 + _RETURN))
    )
    next_liquid, marginal_return = _law_readings(
        law=_fixed_cost_law, savings=liquid_grid - c_star
    )
    assert bool(
        jnp.all(next_liquid > liquid_grid.min())
        and jnp.all(next_liquid < liquid_grid.max())
    ), "witness requires interior landing points; the interpolation clamps outside"

    errors = consumption_euler_error_log10(
        liquid_grid=liquid_grid,
        consumption=c_star,
        next_consumption=liquid_grid,
        discount_factor=_DISCOUNT,
        preferences=crra_preferences(crra=_CRRA),
        next_liquid=next_liquid,
        marginal_return=marginal_return,
    )

    assert float(jnp.max(errors)) < _EXACT_RESIDUAL_LOG10_BOUND
