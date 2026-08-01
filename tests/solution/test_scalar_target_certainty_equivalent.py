"""A nonlinear certainty equivalent transforms stateless targets like any other.

`QuasiArithmeticMean` is defined as `CE = g⁻¹(Σ_r p_r · E_w[g(V'_r)])`: the
transform `g` applies to next-period values before *every* expectation — over
stochastic state transitions and over regime transitions alike — and the inverse
`g⁻¹` applies once, after the probability-weighted sum. A target that carries no
state is still a target, so its value belongs inside `g` too.

The oracle is arithmetic rather than another solve. With `risk_aversion = 1` the
power mean is the geometric mean, so an even lottery over stateless payoffs `1`
and `9` is worth exactly `sqrt(1 * 9) = 3`. Weighting the raw values instead and
inverting afterwards would give `exp(0.5 · 1 + 0.5 · 9) = exp(5) ≈ 148.4`.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    PowerMean,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import ScalarInt
from tests.conftest import DECIMAL_PRECISION

_LOW_PAYOFF = 1.0
_HIGH_PAYOFF = 9.0
_GEOMETRIC_MEAN = 3.0
_LAST_AGE = 21


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    low: ScalarInt
    high: ScalarInt


def _no_flow_payoff(wealth):
    """Return zero flow utility, reading `wealth` so the state counts as used."""
    return jnp.zeros_like(wealth)


def _solve_with_geometric_certainty_equivalent():
    """Solve a model whose only continuation is an even stateless lottery."""
    alive = Regime(
        transition={
            "low": MarkovTransition(lambda: jnp.array(0.5)),
            "high": MarkovTransition(lambda: jnp.array(0.5)),
        },
        active=lambda age: age < _LAST_AGE,
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
        # No flow payoff and no discounting, so the regime's value *is* the
        # certainty equivalent of the continuation lottery. Utility reads
        # `wealth` only because every declared state must be used somewhere.
        functions={"utility": _no_flow_payoff},
        state_transitions={"wealth": fixed_transition("wealth")},
        certainty_equivalent=PowerMean(),
    )
    low = Regime(transition=None, functions={"utility": lambda: jnp.array(_LOW_PAYOFF)})
    high = Regime(
        transition=None, functions={"utility": lambda: jnp.array(_HIGH_PAYOFF)}
    )
    model = Model(
        regimes={"alive": alive, "low": low, "high": high},
        ages=AgeGrid(start=20, stop=_LAST_AGE, step="Y"),
        regime_id_class=RegimeId,
    )
    params = {
        "alive": {
            "utility": {},
            "H": {"discount_factor": 1.0},
            "next_regime": {"low": {}, "high": {}},
            "certainty_equivalent": {"risk_aversion": 1.0},
        },
        "low": {"utility": {}},
        "high": {"utility": {}},
    }
    return model.solve(params=params, log_level="debug")


def test_stateless_targets_enter_the_certainty_equivalent_transformed():
    """An even geometric lottery over stateless payoffs 1 and 9 is worth 3."""
    solution = _solve_with_geometric_certainty_equivalent()
    alive = np.asarray(solution[0]["alive"])
    np.testing.assert_array_almost_equal(
        alive,
        np.full(alive.shape, _GEOMETRIC_MEAN),
        decimal=DECIMAL_PRECISION,
    )
