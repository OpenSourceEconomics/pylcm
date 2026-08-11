"""A target reached with zero probability contributes nothing to the continuation.

A regime transition may place zero mass on a target for part of the state space. Where
it does, that target's value is not consulted, so an entry law that would be nonsense
there — off the target's support, undefined, non-finite — never reaches the result.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.typing import ScalarFloat, ScalarInt

_WEALTH = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt
    other: ScalarInt


def _p_target(wealth: ScalarFloat) -> ScalarFloat:
    """All mass on `target` at or below wealth 2, none above it."""
    return jnp.where(wealth <= 2.0, jnp.float32(1), jnp.float32(0))


def _p_other(wealth: ScalarFloat) -> ScalarFloat:
    return jnp.where(wealth <= 2.0, jnp.float32(0), jnp.float32(1))


def _enter_shock(wealth: ScalarFloat) -> ScalarFloat:
    """Enter the target's shock at the source's wealth, off support above 2."""
    return wealth


def _keep_wealth(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _wealth_utility(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _shock_utility(shock: ScalarFloat) -> ScalarFloat:
    return shock


@pytest.fixture
def model() -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "target": MarkovTransition(_p_target),
                    "other": MarkovTransition(_p_other),
                },
                active=lambda age: age < 22,
                states={"wealth": _WEALTH},
                state_transitions={
                    "shock": {"target": _enter_shock},
                    "wealth": {"other": _keep_wealth},
                },
                functions={"utility": _wealth_utility},
            ),
            "target": Regime(
                transition=None,
                states={
                    "shock": NormalIIDProcess(
                        n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0
                    )
                },
                functions={"utility": _shock_utility},
            ),
            "other": Regime(
                transition=None,
                states={"wealth": _WEALTH},
                functions={"utility": _wealth_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


def test_zero_mass_target_does_not_contaminate_the_value_function(model: Model) -> None:
    """`V = 2 * wealth` across the whole grid, including where `target` has no mass.

    Utility is `wealth` and the discount factor is one. Below wealth 2 the source
    enters `target`, whose terminal utility is the shock it is entered at, so the
    continuation is `wealth`. Above 2 it enters `other`, whose terminal utility is the
    wealth handed over, so the continuation is `wealth` there too.
    """
    V = model.solve(
        params={"source": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        log_level="off",
    )

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).ravel(),
        np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        atol=1e-5,
    )
