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

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    TauchenAR1Process,
    categorical,
)
from lcm.typing import FloatND, ScalarInt
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


def _enter_shock() -> FloatND:
    """Enter the target's process at its middle node.

    An entry law names a value on the target's support, not a position in it.
    The Tauchen nodes are symmetric about the unconditional mean, which is zero
    here, so the middle node is `0.0`.
    """
    return jnp.asarray(0.0)


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
            "koopmans_aggregator": {"discount_factor": _DISCOUNT},
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


def test_the_stateless_bequest_moves_the_parent():
    """Changing the bequest changes the parent's value function.

    Without this, the checks above would still pass if the continuation were
    dropped and the arithmetic happened to agree at one bequest.
    """
    poor = np.asarray(_solve_with_bequest(0.0)[0]["alive"])
    rich = np.asarray(_solve_with_bequest(10.0)[0]["alive"])
    assert not np.allclose(poor, rich)


@categorical(ordered=False)
class _ThreeRegimeId:
    alive: ScalarInt
    gone: ScalarInt
    limbo: ScalarInt


def _solve_with_an_unreachable_stateless_regime(limbo_bequest: float):
    """Solve a model holding a stateless regime the parent cannot reach.

    `alive`'s regime transition is a per-target mapping naming only `alive` and
    `gone`, so `limbo` is structurally unreachable from it even though `limbo` is
    a perfectly ordinary active stateless regime with a large payoff.
    """

    def _leaves(wealth, age):
        return (wealth >= _LEAVE_AT_WEALTH) | (age >= _LAST_AGE - 1)

    alive = Regime(
        transition={
            "alive": MarkovTransition(lambda wealth, age: 1.0 - _leaves(wealth, age)),
            "gone": MarkovTransition(lambda wealth, age: 1.0 * _leaves(wealth, age)),
        },
        active=lambda age: age < _LAST_AGE,
        states={"wealth": _WEALTH_GRID},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=4)},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility},
    )
    gone = Regime(transition=None, functions={"utility": lambda: jnp.array(10.0)})
    limbo = Regime(
        transition=None, functions={"utility": lambda: jnp.array(limbo_bequest)}
    )
    model = Model(
        regimes={"alive": alive, "gone": gone, "limbo": limbo},
        ages=AgeGrid(start=20, stop=_LAST_AGE, step="Y"),
        regime_id_class=_ThreeRegimeId,
    )
    params = {
        "alive": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": _DISCOUNT},
            "next_wealth": {},
            "next_regime": {"alive": {}, "gone": {}},
        },
        "gone": {"utility": {}},
        "limbo": {"utility": {}},
    }
    return model.solve(params=params, log_level="debug")


def test_an_unreachable_stateless_regime_stays_out_of_the_continuation():
    """A stateless regime the transition never names contributes nothing.

    The discriminator between the two candidate rules. *Declared reachability ∩
    activity* excludes `limbo`, because `alive`'s per-target transition does not
    name it. The weaker rule "every regime active next period is a target" would
    include it, and the parent's value would move with a payoff it can never
    collect. Only a model where the two rules disagree can tell them apart, which
    is why `limbo` is active and richly paid rather than merely absent.
    """
    poor = np.asarray(_solve_with_an_unreachable_stateless_regime(0.0)[0]["alive"])
    rich = np.asarray(_solve_with_an_unreachable_stateless_regime(1000.0)[0]["alive"])
    np.testing.assert_array_almost_equal(poor, rich, decimal=DECIMAL_PRECISION)


def _solve_with_process_only_target(level: float):
    """Solve a model whose terminal regime's only state is a stochastic process.

    The source supplies an explicit entry law because it does not carry the
    target's process. Once entered, the process carries its own intrinsic law.
    """
    alive = Regime(
        transition=_next_regime,
        active=lambda age: age < _LAST_AGE,
        states={"wealth": _WEALTH_GRID},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=4)},
        state_transitions={
            "wealth": _next_wealth,
            "shock": {"gone": _enter_shock},
        },
        functions={"utility": _utility},
    )
    retired = Regime(
        transition=None,
        # Fixed at construction, not passed at runtime: the entry law places a
        # value on this process's own support, and that support has to exist
        # before the source's laws are built.
        states={
            "shock": TauchenAR1Process(
                n_points=3, gauss_hermite=False, rho=0.9, sigma=1.0, mu=0.0, n_std=2
            )
        },
        functions={"utility": lambda shock: shock + level},
    )
    model = Model(
        regimes={"alive": alive, "gone": retired},
        ages=AgeGrid(start=20, stop=_LAST_AGE, step="Y"),
        regime_id_class=RegimeId,
    )
    params = {
        "alive": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": _DISCOUNT},
            "next_wealth": {},
            "next_shock": {},
            "next_regime": {},
        },
        "gone": {"utility": {}},
    }
    return model.solve(params=params, log_level="debug")


def test_a_process_only_regime_carries_its_own_value():
    """The process-only regime's value varies with the shock, centred on the level."""
    solution = _solve_with_process_only_target(10.0)
    retired = np.asarray(solution[0]["gone"]).ravel()
    # Tauchen nodes are symmetric about the mean, so the middle node is the level.
    np.testing.assert_array_almost_equal(retired[1], 10.0, decimal=DECIMAL_PRECISION)
    assert retired[0] < retired[1] < retired[2]


def test_the_process_only_level_moves_the_parent():
    """Shifting the process-only regime's level changes the parent's value.

    A symmetric shock alone cannot show this: its expectation is zero either way, so
    a dropped continuation is indistinguishable from a correct one. The level shift
    makes `E[V]` unambiguously nonzero.
    """
    poor = np.asarray(_solve_with_process_only_target(0.0)[0]["alive"])
    rich = np.asarray(_solve_with_process_only_target(10.0)[0]["alive"])
    assert not np.allclose(poor, rich)
