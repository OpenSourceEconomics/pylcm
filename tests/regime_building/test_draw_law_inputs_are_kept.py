"""A transition-local Markov draw keeps every state its probability law reads.

The source carries a binary Markov state `shock` and binary conditioning states.
Its law toward the terminal `dead` regime sets `next_wealth = next_shock`, and
`dead` values wealth linearly while not carrying `shock`, so the draw of
`next_shock` is transition-local. The probability of `next_shock = 1` depends on
the conditioning states, directly or through a helper function, so the source's
value is that probability: it varies with the conditioning states and is constant
along the lagged shock. A broadcast state nothing reads is still pruned, and a
Markov law whose draw nothing reads keeps none of its inputs.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    StochasticTransition,
    categorical,
)
from lcm.regime import Regime
from lcm.typing import ContinuousState, DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class _Binary:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    dead: ScalarInt


def _shock_probs_direct(*, driver: DiscreteState) -> FloatND:
    high = 0.25 + 0.5 * driver
    return jnp.stack((1.0 - high, high))


def _high_probability(*, driver: DiscreteState) -> FloatND:
    return 0.25 + 0.5 * driver


def _shock_probs_through_helper(*, high_probability: FloatND) -> FloatND:
    return jnp.stack((1.0 - high_probability, high_probability))


def _shock_probs_two_drivers(
    *, driver: DiscreteState, other_driver: DiscreteState
) -> FloatND:
    high = 0.125 + 0.5 * driver + 0.25 * other_driver
    return jnp.stack((1.0 - high, high))


def _wealth_from_draw(*, next_shock: DiscreteState) -> ContinuousState:
    return jnp.asarray(next_shock, dtype=float)


def _wealth_constant() -> ContinuousState:
    return jnp.asarray(0.5)


def _identity_driver(*, driver: DiscreteState) -> DiscreteState:
    return driver


def _identity_other_driver(*, other_driver: DiscreteState) -> DiscreteState:
    return other_driver


def _identity_unread(*, unread: DiscreteState) -> DiscreteState:
    return unread


def _certain() -> FloatND:
    return jnp.asarray(1.0)


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _wealth_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


type _Law = Literal["direct", "helper", "two_drivers"]
type _Level = Literal["regime", "model"]


def _model(*, law: _Law, declared_at: _Level, draw_read: bool = True) -> Model:
    drivers = ("driver", "other_driver") if law == "two_drivers" else ("driver",)
    identities = {"driver": _identity_driver, "other_driver": _identity_other_driver}
    probabilities = {
        "direct": _shock_probs_direct,
        "helper": _shock_probs_through_helper,
        "two_drivers": _shock_probs_two_drivers,
    }[law]
    states = {
        name: DiscreteGrid(category_class=_Binary) for name in ("shock", *drivers)
    }
    laws = {
        "wealth": _wealth_from_draw if draw_read else _wealth_constant,
        "shock": StochasticTransition(func=probabilities),
        **{name: identities[name] for name in drivers},
    }
    source = Regime(
        regime_transitions={"dead": StochasticTransition(func=_certain)},
        states=states if declared_at == "regime" else {},
        state_transitions=laws
        if declared_at == "regime"
        else {"wealth": laws["wealth"]},
        functions={"utility": _zero}
        | ({"high_probability": _high_probability} if law == "helper" else {}),
    )
    dead = Regime(
        regime_transitions=None,
        states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _wealth_utility},
    )
    model_slots = (
        {
            "states": states | {"unread": DiscreteGrid(category_class=_Binary)},
            "state_transitions": {
                name: law for name, law in laws.items() if name != "wealth"
            }
            | {"unread": _identity_unread},
        }
        if declared_at == "model"
        else {}
    )
    return Model(
        regimes={"source": source, "dead": dead},
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"source": {"dead": 0}},
        initial_nodes=((0, "source"),),
        enable_jit=False,
        **model_slots,
    )


def _source_values(*, model: Model, axes: tuple[str, ...]) -> np.ndarray:
    """Return the source's values with their axes ordered as `axes`."""
    names = model.state_names(regime_name="source")
    values = model.solve(params={"discount_factor": 1.0}, log_level="off").values
    return np.transpose(
        np.asarray(values[0]["source"]), [names.index(name) for name in axes]
    )


_CASES = [
    pytest.param(law, declared_at, id=f"{law}-{declared_at}")
    for law in ("direct", "helper")
    for declared_at in ("regime", "model")
]


@pytest.mark.parametrize(("law", "declared_at"), _CASES)
def test_draw_law_input_is_a_carried_state(*, law: _Law, declared_at: _Level) -> None:
    """The source carries the Markov state and the state its draw law reads."""
    model = _model(law=law, declared_at=declared_at)
    assert set(model.state_names(regime_name="source")) == {"shock", "driver"}


@pytest.mark.parametrize(("law", "declared_at"), _CASES)
def test_draw_law_input_conditions_the_value(*, law: _Law, declared_at: _Level) -> None:
    """The source's value is 1/4 at `driver = 0` and 3/4 at `driver = 1`."""
    model = _model(law=law, declared_at=declared_at)
    np.testing.assert_array_equal(
        _source_values(model=model, axes=("driver", "shock")),
        np.asarray([[0.25, 0.25], [0.75, 0.75]]),
    )


@pytest.mark.parametrize("declared_at", ["regime", "model"])
def test_draw_law_reading_several_states_keeps_each(*, declared_at: _Level) -> None:
    """A draw law reading two states conditions the value on both of them."""
    model = _model(law="two_drivers", declared_at=declared_at)
    np.testing.assert_array_equal(
        _source_values(model=model, axes=("driver", "other_driver", "shock")),
        np.asarray(
            [
                [[0.125, 0.125], [0.375, 0.375]],
                [[0.625, 0.625], [0.875, 0.875]],
            ]
        ),
    )


@pytest.mark.parametrize("law", ["direct", "helper", "two_drivers"])
def test_state_no_draw_law_reads_is_pruned(*, law: _Law) -> None:
    """A broadcast state read by no law, utility or draw is pruned."""
    model = _model(law=law, declared_at="model")
    assert model.pruned_variables["source"] == frozenset({"unread"})


@pytest.mark.parametrize("law", ["direct", "helper", "two_drivers"])
def test_unread_draw_keeps_neither_its_state_nor_its_law_inputs(*, law: _Law) -> None:
    """A Markov law whose draw nothing reads roots none of its inputs."""
    model = _model(law=law, declared_at="model", draw_read=False)
    assert model.state_names(regime_name="source") == ()
