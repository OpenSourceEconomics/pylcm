"""Two continuation paths that must not silently publish zero.

Both cases below share a failure shape: a finite continuation is replaced by
zero, no error is raised, and the Bellman argmax can reverse. Neither is
detectable from a value that merely looks plausible, so each is pinned against
an arithmetic oracle rather than against another solve.

- **A target-only process, entered coarsely.** A regime whose only state is a
  stochastic process cannot be priced by a parent that neither carries the
  process nor says how it is entered: there is no next-period value to read. The
  build refuses such a model and names the three ways out. What it must never do
  is accept it and contribute zero, which is indistinguishable from a target
  worth nothing.
- **A tiny-valued lottery under a power-mean certainty equivalent.** With
  `risk_aversion > 1`, `v ** (1 - risk_aversion)` leaves the dtype's range for
  small `v` long before the certainty equivalent does, so an aggregation that
  raises before it averages returns zero for a lottery whose exact value is
  positive.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.certainty_equivalent import PowerMean
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_DISCOUNT = 0.95
_LAST_AGE = 22
_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=5.0, n_points=4)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    gone: ScalarInt


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def _next_regime(age: FloatND) -> ScalarInt:
    """A coarse transition: every regime is reachable, none is named per target."""
    return jnp.where(age >= _LAST_AGE - 1, RegimeId.gone, RegimeId.alive)


def _enter_shock() -> ScalarInt:
    """Enter the target's process at its middle node."""
    return jnp.int32(1)


def _solve_coarse_into_process_only_target(level: float):
    """Build a coarse-transition model whose target's only state is a process.

    The parent carries no `shock` and declares no entry law for it, while its
    coarse transition leaves the target reachable, so the build has no
    next-period value to read there. This does not solve; it is the input to the
    rejection test below.
    """
    alive = Regime(
        transition=_next_regime,
        active=lambda age: age < _LAST_AGE,
        states={"wealth": _WEALTH_GRID},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=4)},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility},
    )
    gone = Regime(
        transition=None,
        states={"shock": NormalIIDProcess(n_points=3, gauss_hermite=False)},
        functions={"utility": lambda shock: shock + level},
    )
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
        "gone": {
            "utility": {},
            "shock": {"mu": 0.0, "sigma": 1.0, "n_std": 2},
        },
    }
    return model.solve(params=params, log_level="debug")


def test_a_coarse_transition_into_a_process_only_target_is_refused():
    """A parent that neither carries the process nor enters it is rejected.

    The alternative is to contribute zero for that target, which reads as a
    regime worth nothing rather than as a model the solver cannot price. The
    message names the state and every way to resolve it, so the rejection is
    actionable rather than merely safe.
    """
    with pytest.raises(ModelInitializationError) as excinfo:
        _solve_coarse_into_process_only_target(10.0)
    message = str(excinfo.value)
    assert "shock" in message
    assert "entry law" in message


def _solve_with_entry_law(level: float):
    """The same model, with the parent declaring how the target is entered."""
    alive = Regime(
        transition=_next_regime,
        active=lambda age: age < _LAST_AGE,
        states={"wealth": _WEALTH_GRID},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=4)},
        state_transitions={"wealth": _next_wealth, "shock": {"gone": _enter_shock}},
        functions={"utility": _utility},
    )
    gone = Regime(
        transition=None,
        states={"shock": NormalIIDProcess(n_points=3, gauss_hermite=False)},
        functions={"utility": lambda shock: shock + level},
    )
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
            "next_shock": {},
            "next_regime": {},
        },
        "gone": {
            "utility": {},
            "shock": {"mu": 0.0, "sigma": 1.0, "n_std": 2},
        },
    }
    return model.solve(params=params, log_level="debug")


def test_an_entered_process_only_target_reaches_its_parent():
    """Shifting the target's level moves the parent that declared the entry.

    A symmetric shock alone cannot show this -- its expectation is zero either
    way, so a dropped continuation is indistinguishable from a correct one. The
    level shift makes the contribution unambiguously nonzero.
    """
    poor = np.asarray(_solve_with_entry_law(0.0)[0]["alive"])
    rich = np.asarray(_solve_with_entry_law(10.0)[0]["alive"])
    assert not np.allclose(poor, rich)


def test_an_entered_process_only_target_is_priced_at_its_entry_node():
    """The parent's last living value is `max_c log(c) + beta * (node + level)`.

    The entry law hands over the middle node of a mean-zero symmetric grid,
    which is zero, and the top consumption node is `1.0`. Both sides are
    arithmetic rather than a second solve.
    """
    level = 10.0
    solution = _solve_with_entry_law(level)
    last_living = max(period for period in solution if "alive" in solution[period])
    got = np.asarray(solution[last_living]["alive"])
    expected = np.log(1.0) + _DISCOUNT * level
    np.testing.assert_array_almost_equal(
        got, np.full_like(got, expected), decimal=DECIMAL_PRECISION
    )


@categorical(ordered=False)
class _TinyRegimeId:
    alive: ScalarInt
    dead: ScalarInt


_TINY_WEALTH = LinSpacedGrid(start=1.0, stop=5.0, n_points=5)
_TINY_CONSUMPTION = LinSpacedGrid(start=1.0, stop=5.0, n_points=5)
# The scale at which `v ** (1 - risk_aversion)` leaves float32's range: at
# `risk_aversion = 8` the intermediate is `v ** -7`, which overflows around
# `v = 1e-6` and takes the certainty equivalent to zero with it.
_TINY = 1e-8
_RISK_AVERSION = 8.0


def _tiny_terminal_utility(wealth: ContinuousState) -> FloatND:
    """Strictly positive, tiny, and decreasing in wealth.

    Positivity is what the power mean requires. Decreasing is what makes the
    decision discriminating: consuming is optimal, so the correct policy is the
    *last* consumption node, while a continuation collapsed to a constant zero
    leaves every action tied and selects the first.
    """
    return _TINY * (2.0 - wealth / 10.0)


def _tiny_alive_utility() -> FloatND:
    """No flow payoff, so the choice is driven only by the continuation."""
    return jnp.asarray(0.0)


def _tiny_budget(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _tiny_next_regime() -> ScalarInt:
    return _TinyRegimeId.dead


def _solve_tiny_certainty_equivalent(*, risk_aversion: float = _RISK_AVERSION):
    """Solve a model whose entire continuation sits at the `1e-8` scale."""
    alive = Regime(
        transition=_tiny_next_regime,
        active=lambda age: age < 41,
        states={"wealth": _TINY_WEALTH},
        actions={"consumption": _TINY_CONSUMPTION},
        state_transitions={"wealth": _next_wealth},
        constraints={"budget": _tiny_budget},
        functions={"utility": _tiny_alive_utility},
        certainty_equivalent=PowerMean(),
    )
    dead = Regime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=0.0, stop=5.0, n_points=5)},
        functions={"utility": _tiny_terminal_utility},
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=40, stop=41, step="Y"),
        regime_id_class=_TinyRegimeId,
    )
    params = {
        "alive": {
            "utility": {},
            "H": {"discount_factor": 1.0},
            "next_wealth": {},
            "next_regime": {},
            "certainty_equivalent": {"risk_aversion": risk_aversion},
        },
        "dead": {"utility": {}},
    }
    return model.solve(params=params, log_level="debug")


def test_a_tiny_continuation_survives_the_power_mean():
    """A continuation at `1e-8` reaches the parent instead of collapsing to zero.

    Raising before averaging sends `v ** (1 - risk_aversion)` out of range and
    returns zero for every state; the published value must instead stay at the
    continuation's own scale.
    """
    solution = _solve_tiny_certainty_equivalent()
    alive = np.asarray(solution[0]["alive"])
    assert np.isfinite(alive).all()
    assert (alive > 0.0).all()
    # Every reachable continuation lies in `[1e-8, 2e-8]`, so the value does
    # too. The upper comparison carries a relative tolerance because the bound
    # is attained exactly: the largest terminal utility *is* `2e-8`, and a
    # float32 round trip through the aggregation lands a few ulp above it.
    assert (alive >= _TINY).all()
    assert (alive <= 2.0 * _TINY * (1.0 + 1e-5)).all()


def test_the_tiny_continuation_still_decides_the_policy():
    """The parent's value tracks consuming, not the tie a collapsed value leaves.

    With no flow payoff and a terminal value decreasing in wealth, the optimal
    action spends down to the lowest reachable next wealth. The resulting value
    is the terminal utility at that wealth, computed directly.
    """
    solution = _solve_tiny_certainty_equivalent()
    alive = np.asarray(solution[0]["alive"])
    wealth_nodes = np.asarray(_TINY_WEALTH.to_jax())
    consumption_nodes = np.asarray(_TINY_CONSUMPTION.to_jax())
    expected = []
    for wealth in wealth_nodes:
        feasible = consumption_nodes[consumption_nodes <= wealth]
        next_wealth = wealth - feasible
        expected.append(np.max(_TINY * (2.0 - next_wealth / 10.0)))
    np.testing.assert_allclose(alive, np.asarray(expected), rtol=1e-4)


@pytest.mark.parametrize("risk_aversion", [2.0, 8.0, 20.0])
def test_the_tiny_continuation_survives_across_risk_aversions(risk_aversion):
    """The collapse threshold moves with `risk_aversion`; none of them collapses.

    The overflowing intermediate is `v ** (1 - risk_aversion)`, so a single risk
    aversion could pass by sitting inside the dtype's range rather than by
    aggregating stably.
    """
    alive = np.asarray(
        _solve_tiny_certainty_equivalent(risk_aversion=risk_aversion)[0]["alive"]
    )
    assert np.isfinite(alive).all()
    assert (alive >= _TINY).all()
