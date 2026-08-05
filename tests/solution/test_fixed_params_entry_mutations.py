"""A law fixed through `fixed_params` prices an entered process like any other.

`Model(fixed_params=...)` pins a parameter at model initialization, so a process
whose missing fields arrive that way is fixed at construction and can be entered.
These cases sweep the ways the entry is built — granular and coarse regime
transitions, eager and compiled — against a Gauss-Hermite oracle that shares no
code with the grid construction under test.
"""

import math
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LogNormalIIDProcess,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.typing import FloatND, ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def one_probability() -> ScalarFloat:
    return jnp.float32(1)


def target_id() -> ScalarInt:
    return RegimeId.target


def active(age: float) -> bool:
    return age < 22


def zero() -> ScalarFloat:
    return jnp.float32(0)


def square(shock: ScalarFloat) -> FloatND:
    return shock**2


def identity(shock: ScalarFloat) -> FloatND:
    return shock


def source_value(model: Model, params: dict) -> float:
    solution = model.solve(params=params, log_level="debug")
    p = max(period for period, regimes in solution.items() if "source" in regimes)
    return float(np.asarray(solution[p]["source"]).ravel()[0])


def gh_expectation(transform: Callable[[np.ndarray], np.ndarray]) -> float:
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(3)
    nodes = math.sqrt(2.0) * raw_nodes
    probabilities = raw_weights / math.sqrt(math.pi)
    return float(np.dot(probabilities, transform(nodes)))


@pytest.mark.parametrize("coarse", [False, True], ids=["granular", "coarse"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    ("process", "utility", "expected"),
    [
        (
            NormalIIDProcess(n_points=3, gauss_hermite=True),
            square,
            gh_expectation(lambda x: x**2),
        ),
        (
            LogNormalIIDProcess(n_points=3, gauss_hermite=True),
            identity,
            gh_expectation(np.exp),
        ),
    ],
    ids=["normal-square", "lognormal-linear"],
)
def test_a_law_from_fixed_params_prices_an_entered_process(
    coarse: bool,  # noqa: FBT001
    enable_jit: bool,  # noqa: FBT001
    process: NormalIIDProcess | LogNormalIIDProcess,
    utility: Callable[..., FloatND],
    expected: float,
) -> None:
    """The source reads the process's own law, and never its parameters.

    The law reaches the process through `fixed_params`, so the source's solve
    takes only `discount_factor` — the target's parameters are gone from the
    runtime template — and still prices the continuation at the law's own
    weighted expectation rather than an unweighted node average.
    """
    transition = target_id if coarse else {"target": MarkovTransition(one_probability)}
    model = Model(
        regimes={
            "source": Regime(
                transition=transition,
                active=active,
                functions={"utility": zero},
            ),
            "target": Regime(
                transition=None,
                states={"shock": process},
                functions={"utility": utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        fixed_params={"target": {"shock": {"mu": 0.0, "sigma": 1.0}}},
        enable_jit=enable_jit,
    )
    np.testing.assert_allclose(
        source_value(model, {"discount_factor": 1.0}),
        expected,
        rtol=5e-5,
        atol=5e-5,
    )
