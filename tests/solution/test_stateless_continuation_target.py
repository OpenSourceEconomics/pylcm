"""A reachable regime with no states still contributes its value to its parent.

A regime need not carry states: an absorbing "gone" regime whose payoff is a bare
bequest is a scalar-valued regime, and its value is whatever its utility returns —
not zero. A parent that transitions into it must therefore add that value to its
continuation, exactly as it would for a stateful target.

The oracle is arithmetic. With `max_c log(c) = log(1) = 0` on a consumption grid
whose top node is `1.0`, a parent leaving for the stateless regime is worth exactly
`0 + discount_factor * bequest`, which is checked directly rather than against
another solve.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import ScalarInt
from tests.conftest import DECIMAL_PRECISION

_DISCOUNT = 0.95
_LEAVE_AT_WEALTH = 3.0
_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=5.0, n_points=4)
_LAST_AGE = 22


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    gone: ScalarInt


def _utility(consumption):
    return jnp.log(consumption)


def _next_wealth(wealth, consumption):
    return wealth - consumption


def _next_regime(wealth, age):
    # Wealth at or above the threshold leaves for the stateless regime; everyone
    # leaves after the last age at which `alive` is active, so no probability mass
    # is ever sent to an inactive regime.
    leaves = (wealth >= _LEAVE_AT_WEALTH) | (age >= _LAST_AGE - 1)
    return jnp.where(leaves, RegimeId.gone, RegimeId.alive)


def _solve_with_bequest(bequest: float):
    """Solve a two-regime model whose terminal regime carries no state."""
    alive = Regime(
        transition=_next_regime,
        active=lambda age: age < _LAST_AGE,
        states={"wealth": _WEALTH_GRID},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=4)},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility},
    )
    gone = Regime(transition=None, functions={"utility": lambda: jnp.array(bequest)})
    model = Model(
        regimes={"alive": alive, "gone": gone},
        ages=AgeGrid(start=20, stop=_LAST_AGE, step="Y"),
        regime_id_class=RegimeId,
    )
    params = {
        "alive": {
            "utility": {},
            "H": {"discount_factor": _DISCOUNT},
            "next_wealth": {},
            "next_regime": {},
        },
        "gone": {"utility": {}},
    }
    return model.solve(params=params, log_level="debug")


def test_a_stateless_regime_carries_its_own_value():
    """The stateless regime's value is its utility, not zero."""
    solution = _solve_with_bequest(10.0)
    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["gone"]).ravel(), [10.0], decimal=DECIMAL_PRECISION
    )


@pytest.mark.xfail(
    reason="continuation targets are enumerated from state-law bundle keys, so a "
    "stateless regime never appears as a target and its value is dropped",
    strict=False,
)
@pytest.mark.parametrize("bequest", [0.0, 4.0, 10.0])
def test_a_parent_leaving_for_a_stateless_regime_gets_its_value(bequest):
    """At wealth levels that leave, the parent's value is `beta * bequest`."""
    solution = _solve_with_bequest(bequest)
    wealth = np.asarray(_WEALTH_GRID.to_jax())
    leaving = wealth >= _LEAVE_AT_WEALTH

    alive = np.asarray(solution[0]["alive"])
    np.testing.assert_array_almost_equal(
        alive[leaving],
        np.full(int(leaving.sum()), _DISCOUNT * bequest),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.xfail(
    reason="the stateless target's value never reaches the parent's continuation",
    strict=True,
)
def test_the_stateless_bequest_moves_the_parent():
    """Changing the bequest changes the parent's value function.

    Without this, the checks above would still pass if the continuation were
    dropped and the arithmetic happened to agree at one bequest.
    """
    poor = np.asarray(_solve_with_bequest(0.0)[0]["alive"])
    rich = np.asarray(_solve_with_bequest(10.0)[0]["alive"])
    assert not np.allclose(poor, rich)
