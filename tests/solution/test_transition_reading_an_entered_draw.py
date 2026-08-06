"""A transition may read an entered draw, evaluated inside the process's own axis.

A deterministic law of a target may depend on a stochastic draw of that same target.
The continuation is then the expectation over the target's existing node axis,

```{math}
\\sum_j p_j \\, V\\!\\left(g(\\varepsilon_j),\\, \\varepsilon_j\\right),
```

taken on the nodes the process already carries — not by inventing a parameter for the
draw, and not by adding a second axis for it.
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
from lcm.exceptions import ModelInitializationError
from lcm.typing import ContinuousState, ScalarFloat, ScalarInt

# `mu=1, sigma=0.5, n_std=2` at three points puts symmetric nodes on `(0, 1, 2)`,
# so the draw has mean one whatever weights the discretization assigns them.
_SHOCK = NormalIIDProcess(n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0)
# Holds `2 * eps` for every node exactly, so no interpolation error enters.
_WEALTH = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _to_target() -> ScalarFloat:
    return jnp.float32(1)


def _no_utility() -> ScalarFloat:
    return jnp.float32(0)


def _shock_utility(shock: ScalarFloat) -> ScalarFloat:
    return shock


def _wealth_plus_shock(wealth: ScalarFloat, shock: ScalarFloat) -> ScalarFloat:
    return wealth + shock


def _next_wealth_from_draw(next_shock: ContinuousState) -> ScalarFloat:
    return 2.0 * next_shock


def _scaled(next_shock: ContinuousState) -> ScalarFloat:
    return 2.0 * next_shock


def _next_wealth_via_helper(scaled: ScalarFloat) -> ScalarFloat:
    return scaled


def _build(functions, next_wealth) -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                state_transitions={"wealth": {"target": next_wealth}},
                functions=functions,
            ),
            "target": Regime(
                transition=None,
                states={"wealth": _WEALTH, "shock": _SHOCK},
                functions={"utility": _wealth_plus_shock},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}


@pytest.fixture(params=["direct", "through_a_helper"])
def model(request: pytest.FixtureRequest) -> Model:
    if request.param == "direct":
        return _build({"utility": _no_utility}, _next_wealth_from_draw)
    return _build({"utility": _no_utility, "scaled": _scaled}, _next_wealth_via_helper)


def test_the_continuation_is_the_expectation_over_the_draw(model: Model) -> None:
    """`V = E[2*eps + eps] = 3 * E[eps] = 3`, since the draw has mean one.

    The target pays `wealth + shock` and wealth arrives as `2 * shock`, so the
    integrand is `3 * eps` on every node. Its expectation is three for any weights
    that make the symmetric node set average to one.
    """
    V = model.solve(params=_PARAMS, log_level="off")

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).ravel(), np.array([3.0]), atol=1e-5
    )


def test_the_draw_is_not_asked_of_the_user_as_a_parameter(model: Model) -> None:
    """No parameter slot appears for the draw the model resolves itself."""
    template = model.get_params_template()["source"]

    assert "next_shock" not in str(template)


_RUNTIME_SHOCK = NormalIIDProcess(n_points=3, gauss_hermite=False)


def _build_reading_a_runtime_draw() -> Model:
    """Both regimes carry the same process, whose law arrives at runtime."""
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                states={"shock": _RUNTIME_SHOCK},
                state_transitions={"wealth": {"target": _next_wealth_from_draw}},
                functions={"utility": _shock_utility},
            ),
            "target": Regime(
                transition=None,
                states={"wealth": _WEALTH, "shock": _RUNTIME_SHOCK},
                functions={"utility": _wealth_plus_shock},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


def test_reading_a_draw_whose_nodes_arrive_at_runtime_is_rejected() -> None:
    """A law cannot read a draw whose support is unknown while the model builds.

    Resolving the law on the node axis needs the nodes themselves, and a process
    parameterized at runtime has none yet — so the model says so rather than
    resolving against a support that does not exist.
    """
    with pytest.raises(ModelInitializationError, match="runtime"):
        _build_reading_a_runtime_draw()
