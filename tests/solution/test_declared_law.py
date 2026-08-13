"""The declared law of motion is read from the model, not assumed.

A solver needs two things from the budget constraint: where a level of savings
lands next period, and how that landing point moves when savings move. Both are
read off the law the regime declares, so a term outside the conventional
`return x balance + income` form reaches the Euler inversion like any other.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.declared_law import (
    build_declared_liquid_law,
    fail_if_declared_law_is_not_increasing,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousState, FloatND
from tests.conftest import DECIMAL_PRECISION

_SAVINGS = jnp.linspace(0.0, 10.0, 11)


def savings(wealth: ContinuousState, consumption: FloatND) -> FloatND:
    """The post-decision variable the laws below are written through."""
    return wealth - consumption


def next_wealth_conventional(
    savings: FloatND, return_liquid: float, retirement_income: float
) -> ContinuousState:
    return (1.0 + return_liquid) * savings + retirement_income


def next_wealth_with_a_fixed_cost(
    savings: FloatND,
    return_liquid: float,
    retirement_income: float,
    fixed_cost: float,
) -> ContinuousState:
    return (1.0 + return_liquid) * savings + retirement_income - fixed_cost


def next_wealth_falling_in_savings(savings: FloatND) -> ContinuousState:
    """A law a saver is punished by — outside what the method solves."""
    return 10.0 - savings


def next_wealth_reaching_past_the_post_decision(
    wealth: ContinuousState,
    consumption: FloatND,
    return_liquid: float,
) -> ContinuousState:
    """A law written straight from the state and the action, bypassing savings."""
    return (1.0 + return_liquid) * (wealth - consumption)


def _law_for(func):
    """Build the declared-law reader for a single-target regime using `func`."""
    return build_declared_liquid_law(
        transitions=MappingProxyType(
            {"retired": MappingProxyType({"next_wealth": func})}
        ),
        # `EconFunction` describes the post-processing engine signature; these
        # are plain user functions, which is what the DAG composes.
        functions=MappingProxyType({"savings": savings}),  # ty: ignore[invalid-argument-type]
        post_decision_name="savings",
        target="retired",
        target_state="wealth",
        variable_names=frozenset({"wealth", "consumption"}),
    )


def test_a_law_that_bypasses_the_post_decision_node_is_rejected_by_name():
    """A law reaching the state or action directly names them and says why.

    The method reads where a level of savings lands and how that landing point
    moves; both are only defined if the law is a function of savings. A law
    written straight from the state and the action is not, and is refused when
    the law is composed rather than failing later inside the composition.
    """
    with pytest.raises(
        RegimeInitializationError, match=r"reads \['consumption', 'wealth'\]"
    ):
        _law_for(next_wealth_reaching_past_the_post_decision)


def test_the_conventional_law_yields_the_gross_return_as_its_slope():
    """A law of the assumed form reproduces what the assumption computed."""
    law = _law_for(next_wealth_conventional)

    next_liquid, marginal_return = law(
        savings_grid=_SAVINGS, return_liquid=0.03, retirement_income=2.0
    )

    np.testing.assert_array_almost_equal(
        np.asarray(next_liquid),
        1.03 * np.asarray(_SAVINGS) + 2.0,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(marginal_return),
        np.full(_SAVINGS.shape, 1.03),
        decimal=DECIMAL_PRECISION,
    )


def test_a_fixed_cost_moves_the_landing_points_and_not_the_slope():
    """A term outside the assumed form reaches the level but not the derivative."""
    law = _law_for(next_wealth_with_a_fixed_cost)

    next_liquid, marginal_return = law(
        savings_grid=_SAVINGS,
        return_liquid=0.03,
        retirement_income=2.0,
        fixed_cost=0.5,
    )

    np.testing.assert_array_almost_equal(
        np.asarray(next_liquid),
        1.03 * np.asarray(_SAVINGS) + 1.5,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(marginal_return),
        np.full(_SAVINGS.shape, 1.03),
        decimal=DECIMAL_PRECISION,
    )


def test_a_law_falling_in_savings_is_rejected_with_an_ordering_explanation():
    """The endogenous grid is read back by interpolation, which needs ascent."""
    law = _law_for(next_wealth_falling_in_savings)
    next_liquid, _marginal_return = law(savings_grid=_SAVINGS)

    with pytest.raises(RegimeInitializationError, match="falls as savings rise"):
        fail_if_declared_law_is_not_increasing(
            next_liquid=next_liquid, regime_name="working", target="retired"
        )


def test_a_law_flat_over_a_band_is_rejected_as_flat_rather_than_falling():
    """A means test clawing a transfer back one-for-one is flat, not backwards.

    Such a law is still outside the method — several savings levels reach the
    same landing point, so the level behind a given one is not unique — but a
    modeller reading `falls as savings rise` would look for a sign error that
    is not there.
    """
    clawed_back = jnp.concatenate(
        [jnp.linspace(0.0, 3.0, 4), jnp.full((4,), 3.0), jnp.linspace(4.0, 7.0, 3)]
    )

    with pytest.raises(RegimeInitializationError, match="is flat over a range"):
        fail_if_declared_law_is_not_increasing(
            next_liquid=clawed_back, regime_name="working", target="retired"
        )
