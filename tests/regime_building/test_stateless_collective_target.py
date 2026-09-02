"""The continuation value of a target regime that carries no state.

A collective regime may transition into a target whose value function has no
state axis at all — a household whose terminal payoff depends on nothing it
carries forward. There is no next state to interpolate at, but the transition
still carries probability mass and the target's value still belongs in the
certainty equivalent.
"""

import numpy as np

from tests.collective_fixtures import (
    STATELESS_TARGET_V_PERIOD_0,
    make_stateless_collective_target_model,
)
from tests.conftest import DECIMAL_PRECISION


def test_stateless_collective_target_enters_the_continuation():
    """A target regime carrying no state contributes `beta * V_target`.

    The `couple` regime chooses work or leisure over a two-point wage grid and
    transitions with probability one into a collective terminal regime whose
    household value is `(10, 0)` over the stakeholders `("f", "m")`. With
    `beta = 0.95`, each stakeholder's action value is
    `Q^s = u^s + 0.95 * V_terminal^s`, so leisure wins at the low wage and work
    at the high wage, and the period-0 value function over
    `(wage, stakeholder)` is `((39.5, 0), (49.5, 80))`.
    """
    model, params = make_stateless_collective_target_model()

    solution = model.solve(params=params, log_level="debug").values

    np.testing.assert_array_almost_equal(
        solution[0]["couple"],
        np.asarray(STATELESS_TARGET_V_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )
