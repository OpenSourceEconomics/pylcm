"""Two continuation paths that must not silently publish zero.

Both cases below share a failure shape: a finite continuation is replaced by
zero, no error is raised, and the Bellman argmax can reverse. Neither is
detectable from a value that merely looks plausible, so each is pinned against
an arithmetic oracle rather than against another solve.

- **A target-only process, entered coarsely.** Whether a parent can price a
  target whose only state is a stochastic process depends on the process. An IID
  process draws independently of its previous value, so its entry distribution is
  its own unconditional law and the parent needs nothing from the source; it is
  priced. An AR(1) draw depends on a previous value the source does not have, so
  there is no next-period value to read and the build refuses, naming the ways
  out. What neither case may do is accept the model and contribute zero, which is
  indistinguishable from a target worth nothing.
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
    TauchenAR1Process,
    UniformIIDProcess,
    categorical,
)
from lcm.certainty_equivalent import PowerMean
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_DISCOUNT = 0.95
_LAST_AGE = 22
_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=5.0, n_points=4)

# Equal weights on `(0, 1, 2, 3)`, so the unconditional expectation is the plain
# mean of the nodes and needs no quadrature to state. The law is fixed at
# construction because an entered process is priced inside the source's Bellman
# equation, which reads only the source's own parameters.
_UNIFORM_NODES = (0.0, 1.0, 2.0, 3.0)
_UNIFORM_SHOCK = UniformIIDProcess(
    n_points=4, start=_UNIFORM_NODES[0], stop=_UNIFORM_NODES[-1]
)


def _process_params(
    process: UniformIIDProcess | TauchenAR1Process,
) -> dict[str, float | int]:
    """Return the runtime parameters the process's grid reads."""
    if isinstance(process, UniformIIDProcess):
        return {}
    return {"rho": 0.9, "sigma": 1.0, "mu": 0.0, "n_std": 2}


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


def _enter_shock() -> FloatND:
    """Enter the target's process at its middle node.

    An entry law names a value on the target's support, not a position in it.
    The support here is `(-2, 0, 2)`, so its middle node is `0.0`.
    """
    return jnp.asarray(0.0)


def _solve_coarse_into_process_only_target(
    level: float,
    *,
    process: UniformIIDProcess | TauchenAR1Process | None = None,
):
    """Build a coarse-transition model whose target's only state is a process.

    The parent carries no `shock` and declares no entry law for it, while its
    coarse transition leaves the target reachable. Whether that is solvable is
    the process's business: an IID process supplies its own entry distribution,
    an AR(1) process cannot without a previous value.
    """
    process = _UNIFORM_SHOCK if process is None else process
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
        states={"shock": process},
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
            "koopmans_aggregator": {"discount_factor": _DISCOUNT},
            "next_wealth": {},
            "next_regime": {},
        },
        "gone": {"utility": {}, "shock": _process_params(process)},
    }
    return model.solve(params=params, log_level="debug")


def test_a_coarse_transition_into_an_iid_target_is_priced_at_its_own_law():
    """The parent reads `E[shock]` over the process's unconditional law.

    An IID draw does not depend on its previous value, so the source has nothing
    to hand over and the entry distribution is the process's own. On nodes
    `(0, 1, 2, 3)` at equal weight that expectation is `1.5`, which no single
    node equals -- so this distinguishes pricing at the law from pricing at any
    one node, and from dropping the target and publishing `log(1.0) = 0`.
    """
    solution = _solve_coarse_into_process_only_target(0.0)
    last_living = max(period for period in solution if "alive" in solution[period])
    got = np.asarray(solution[last_living]["alive"])
    expected = np.log(1.0) + _DISCOUNT * np.mean(_UNIFORM_NODES)
    np.testing.assert_array_almost_equal(
        got, np.full_like(got, expected), decimal=DECIMAL_PRECISION
    )


def test_a_coarse_transition_into_an_ar1_target_is_refused():
    """An AR(1) target the source cannot seed is rejected, not priced at zero.

    Its next draw depends on a previous value that the source neither carries
    nor supplies, so no next-period value exists. The message names the state,
    the parameters that block it, and every way to resolve it, so the rejection
    is actionable rather than merely safe.

    Supplying an entry law is not among those ways: the support the law would
    place a value on is itself built from parameters this process only supplies
    at runtime, so the law is rejected for the same reason. The routes out are
    to fix those parameters, to carry the state on the source, or to stop the
    transition from reaching this target at all.
    """
    with pytest.raises(ModelInitializationError) as excinfo:
        _solve_coarse_into_process_only_target(
            10.0,
            process=TauchenAR1Process(n_points=3, gauss_hermite=False),
        )
    message = str(excinfo.value)
    assert "shock" in message
    assert "'mu', 'n_std', 'rho', 'sigma'" in message
    assert "at construction" in message
    assert "declare it on 'alive'" in message
    assert "narrow the transition's static target support" in message


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
        # Fixed at construction, not passed at runtime: the entry law places a
        # value on this process's own support, and that support has to exist
        # before the source's laws are built.
        states={
            "shock": NormalIIDProcess(
                n_points=3, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2
            )
        },
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
            "koopmans_aggregator": {"discount_factor": _DISCOUNT},
            "next_wealth": {},
            "next_shock": {},
            "next_regime": {},
        },
        "gone": {"utility": {}},
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
            "koopmans_aggregator": {"discount_factor": 1.0},
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
