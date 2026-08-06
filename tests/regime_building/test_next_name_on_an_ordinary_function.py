"""`next_<state>` on a function no transition reads is an ordinary parameter.

`next_<state>` is next-period vocabulary only where a next-period value exists: inside
a transition, and inside whatever feeds one. `utility` and `constraints` are evaluated
at this period's states, so an argument of that spelling there is a parameter the user
supplies, and it reaches the template and the solution as one.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.typing import ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _to_target() -> ScalarFloat:
    return jnp.float32(1)


def _shock_utility(shock: ScalarFloat) -> ScalarFloat:
    return shock


def _utility(next_shock: ScalarFloat) -> ScalarFloat:
    """Utility of a parameter that merely happens to be spelled `next_shock`."""
    return next_shock


@pytest.fixture
def model() -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                functions={"utility": _utility},
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
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


def test_the_parameter_appears_in_the_template(model: Model) -> None:
    """The template asks for it under the function that reads it."""
    assert model.get_params_template()["source"]["utility"] == {
        "next_shock": "ScalarFloat"
    }


def test_the_supplied_value_reaches_the_value_function(model: Model) -> None:
    """`V = next_shock + E[shock]`: five against a mean of one gives six."""
    V = model.solve(
        params={
            "source": {
                "koopmans_aggregator": {"discount_factor": 1.0},
                "utility": {"next_shock": 5.0},
            }
        },
        log_level="off",
    )

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).ravel(), np.array([6.0]), atol=1e-5
    )
