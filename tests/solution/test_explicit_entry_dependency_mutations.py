"""Every consumer of a declared process entry reads its physical value.

Three kinds of law can consume a `next_<state>` output: another declared entry,
an ordinary deterministic transition, and a stochastic law's probability
function. All three are covered here, in both declaration orders — resolution is
topological, so which law is written first must not matter.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.typing import DiscreteState, FloatND, ScalarFloat, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


@categorical(ordered=True)
class Good:
    bad: ScalarInt
    good: ScalarInt


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _one_probability() -> FloatND:
    return jnp.asarray(1.0)


def _source_is_early(age: float) -> bool:
    return age < 22


def _process() -> NormalIIDProcess:
    # Binned nodes are exactly (0, 1, 2), so entry at 1.5 is off-node but on-support.
    return NormalIIDProcess(
        n_points=3,
        gauss_hermite=False,
        mu=1.0,
        sigma=0.5,
        n_std=2.0,
    )


def _enter_shock() -> ScalarFloat:
    return jnp.asarray(1.5)


def _enter_other(next_shock: ScalarFloat) -> ScalarFloat:
    return next_shock


def _double_the_entry(next_shock: ScalarFloat) -> ScalarFloat:
    return 2.0 * next_shock


def _good_probs(next_shock: ScalarFloat) -> FloatND:
    """Put all mass on `good` above 1.0, which the entry value 1.5 clears."""
    return jnp.where(
        next_shock >= 1.0,
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([1.0, 0.0]),
    )


def _two_process_utility(shock: ScalarFloat, other: ScalarFloat) -> FloatND:
    return shock + 10.0 * other


def _good_utility(good: DiscreteState, shock: ScalarFloat) -> FloatND:
    return good + 0.0 * shock


def _wealth_utility(wealth: ScalarFloat, shock: ScalarFloat) -> FloatND:
    return wealth + 0.0 * shock


def _source_value(model: Model, params: dict) -> float:
    solution = model.solve(params=params, log_level="debug")
    period = max(p for p in solution if "source" in solution[p])
    return float(np.asarray(solution[period]["source"]).ravel()[0])


def _ordered[T](items: list[tuple[str, T]], *, reverse: bool) -> dict[str, T]:
    return dict(reversed(items)) if reverse else dict(items)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["producer_first", "consumer_first"]
)
def test_explicit_entry_feeds_another_explicit_entry(
    enable_jit: bool,  # noqa: FBT001
    reverse: bool,  # noqa: FBT001
) -> None:
    """Two chained entries at 1.5 price the affine target at 16.5, in either order."""
    state_transitions = _ordered(
        [
            ("shock", {"target": _enter_shock}),
            ("other", {"target": _enter_other}),
        ],
        reverse=reverse,
    )
    target_states = _ordered(
        [("shock", _process()), ("other", _process())], reverse=reverse
    )
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions=state_transitions,
                functions={"utility": _zero_utility},
                certainty_equivalent=PowerMean(),
            ),
            "target": Regime(
                transition=None,
                states=target_states,
                functions={"utility": _two_process_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )
    params = {
        "source": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": 1.0},
            "certainty_equivalent": {"risk_aversion": 2.0},
            "target": {"next_regime": {}, "next_shock": {}, "next_other": {}},
        },
        "target": {"utility": {}},
    }
    np.testing.assert_almost_equal(
        _source_value(model, params), 16.5, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["producer_first", "consumer_first"]
)
def test_explicit_entry_feeds_stochastic_weight_law(
    enable_jit: bool,  # noqa: FBT001
    reverse: bool,  # noqa: FBT001
) -> None:
    """A Markov law reading the entry sees 1.5, so all mass lands on `good` (1.0)."""
    state_transitions = _ordered(
        [
            ("shock", {"target": _enter_shock}),
            ("good", {"target": MarkovTransition(_good_probs)}),
        ],
        reverse=reverse,
    )
    target_states = _ordered(
        [("shock", _process()), ("good", DiscreteGrid(Good))], reverse=reverse
    )
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions=state_transitions,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states=target_states,
                functions={"utility": _good_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )
    params = {
        "source": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": 1.0},
            "target": {"next_regime": {}, "next_shock": {}, "next_good": {}},
        },
        "target": {"utility": {}},
    }
    np.testing.assert_almost_equal(
        _source_value(model, params), 1.0, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["producer_first", "consumer_first"]
)
def test_explicit_entry_feeds_an_ordinary_deterministic_law(
    enable_jit: bool,  # noqa: FBT001
    reverse: bool,  # noqa: FBT001
) -> None:
    """`next_wealth = 2 * next_shock` receives 1.5, so the target enters at wealth 3.0.

    The node vector the entry is interpolated over is `(0, 1, 2)`; had the
    consumer received that instead of the physical value it would have entered at
    `(0, 2, 4)`, none of which is 3.0.
    """
    state_transitions = _ordered(
        [
            ("shock", {"target": _enter_shock}),
            ("wealth", {"target": _double_the_entry}),
        ],
        reverse=reverse,
    )
    target_states = _ordered(
        [
            ("shock", _process()),
            ("wealth", LinSpacedGrid(start=0.0, stop=10.0, n_points=11)),
        ],
        reverse=reverse,
    )
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions=state_transitions,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states=target_states,
                functions={"utility": _wealth_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )
    params = {
        "source": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": 1.0},
            "target": {"next_regime": {}, "next_shock": {}, "next_wealth": {}},
        },
        "target": {"utility": {}},
    }
    np.testing.assert_almost_equal(
        _source_value(model, params), 3.0, decimal=DECIMAL_PRECISION
    )
