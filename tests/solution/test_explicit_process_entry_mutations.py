"""An explicit entry law is priced at the value it names, across process shapes.

A declared entry law into a process the source does not carry names a physical
value, while the target holds its value function on the process's nodes. These
cases sweep the shapes where confusing the two would show: odd Gauss-Hermite
orders, which put a node exactly at the mean, and translated and scaled
supports, which separate the value from its position on the axis.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, MarkovTransition, Model, NormalIIDProcess, Regime, categorical
from lcm.typing import FloatND, ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def one_probability() -> ScalarFloat:
    return jnp.float32(1)


def zero() -> ScalarFloat:
    return jnp.float32(0)


def square(shock: ScalarFloat) -> FloatND:
    return shock**2


def active(age: float) -> bool:
    return age < 22


def source_value(model: Model) -> float:
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    period = max(p for p, regimes in solution.items() if "source" in regimes)
    return float(np.asarray(solution[period]["source"]).ravel()[0])


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("n_points", [3, 5])
@pytest.mark.parametrize("mu", [-1.5, 0.0, 2.25])
@pytest.mark.parametrize("sigma", [0.3, 1.2])
def test_entry_at_the_mean_is_priced_at_the_mean_not_its_position(
    enable_jit: bool,  # noqa: FBT001
    n_points: int,
    mu: float,
    sigma: float,
) -> None:
    """Entering at the mean under payoff `shock**2` is worth `mu**2`.

    An odd-order Gauss-Hermite grid holds the mean exactly, so the declared
    value names a support point and the continuation is that point's payoff
    alone — not the intrinsic lottery over every node, and not the payoff of
    the node's index.
    """

    def enter_at_mean() -> ScalarFloat:
        return jnp.asarray(mu)

    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(one_probability)},
                active=active,
                state_transitions={"shock": {"target": enter_at_mean}},
                functions={"utility": zero},
            ),
            "target": Regime(
                transition=None,
                states={
                    "shock": NormalIIDProcess(
                        n_points=n_points,
                        gauss_hermite=True,
                        mu=mu,
                        sigma=sigma,
                    )
                },
                functions={"utility": square},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )

    got = source_value(model)
    assert np.isfinite(got)
    np.testing.assert_allclose(got, mu**2, rtol=5e-5, atol=5e-5)
