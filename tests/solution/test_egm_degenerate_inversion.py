"""A zero marginal value of saving leaves the 1-D EGM step's solution finite.

The Euler target $\\beta (1+r) V'$ is zero whenever saving carries no marginal
value — a terminal continuation with no bequest motive, or a zero discount
factor. The mathematical inverse $(u')^{-1}(0)$ is then unbounded, so the step
clamps the target before inverting and lets the closed-form
borrowing-constrained candidate represent the consume-everything corner.

Without the clamp the failure depends on how the regime supplies its inverse: an
analytic one returns an infinity, a numerically inverted one leaves its bracket
and returns NaN. Both are numerical artifacts of a corner the step already knows
how to represent.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.numeric_inverse import numeric_inverse_marginal_utility
from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from _lcm.egm.preferences import Preferences
from lcm.typing import FloatND
from tests.conftest import DECIMAL_PRECISION
from tests.solution._crra_preferences import crra_preferences

_CRRA = 2.0
_LIQUID_GRID = jnp.linspace(0.5, 3.0, 5)
_NEXT_LIQUID_GRID = jnp.linspace(0.5, 5.0, 6)
_SAVINGS_GRID = jnp.linspace(0.0, 2.0, 6)
_DISCOUNT_FACTOR = 0.98
_RETURN_LIQUID = 0.02
_INCOME = 1.0
_NEXT_VALUE_LEVEL = 3.0


def _numerically_inverted_crra() -> Preferences:
    """CRRA felicity whose inverse marginal is solved for, not written down.

    A regime that declares no `inverse_marginal_utility` gets this path, and it
    is the one with no infinity to fall back on: the bracketed Newton solve
    works in log-consumption, so an unbracketed target leaves the bracket.
    """
    analytic = crra_preferences(_CRRA)

    def scalar_inverse(marginal_continuation: FloatND) -> FloatND:
        return numeric_inverse_marginal_utility(
            marginal_continuation=marginal_continuation,
            marginal_utility=jax.grad(analytic.utility),
            c_lower=jnp.asarray(1e-8),
            c_upper=jnp.asarray(100.0),
        )

    return Preferences(
        utility=analytic.utility,
        marginal_utility=analytic.marginal_utility,
        inverse_marginal_utility=jax.vmap(scalar_inverse),
    )


@pytest.mark.parametrize(
    "preferences",
    [crra_preferences(_CRRA), _numerically_inverted_crra()],
    ids=["analytic_inverse", "numeric_inverse"],
)
def test_a_zero_marginal_continuation_publishes_the_consume_everything_corner(
    preferences: Preferences,
) -> None:
    """With no marginal value of saving the agent consumes all liquid.

    Saving nothing is optimal at every liquid level, so consumption is the state
    itself and the value is this period's felicity plus the discounted flat
    continuation reached with zero savings.
    """
    result = egm_one_asset_step(
        next_value=jnp.full_like(_NEXT_LIQUID_GRID, _NEXT_VALUE_LEVEL),
        next_marginal=jnp.zeros_like(_NEXT_LIQUID_GRID),
        liquid_grid=_LIQUID_GRID,
        next_liquid_grid=_NEXT_LIQUID_GRID,
        savings_grid=_SAVINGS_GRID,
        discount_factor=_DISCOUNT_FACTOR,
        preferences=preferences,
        next_liquid=(1.0 + _RETURN_LIQUID) * _SAVINGS_GRID + _INCOME,
        marginal_return=jnp.full_like(_SAVINGS_GRID, 1.0 + _RETURN_LIQUID),
    )
    expected_value = (
        preferences.utility(_LIQUID_GRID) + _DISCOUNT_FACTOR * _NEXT_VALUE_LEVEL
    )
    np.testing.assert_array_almost_equal(
        result.consumption, _LIQUID_GRID, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.value, expected_value, decimal=DECIMAL_PRECISION
    )
