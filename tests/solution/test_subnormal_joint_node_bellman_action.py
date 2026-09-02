"""A joint node too small for the dtype still decides which action is optimal.

Two independent binary Markov states, each reaching its low value with a
probability the dtype holds comfortably, meet at a joint node whose probability
it cannot: `2**-64` squared in float32, `2**-512` squared in float64. A payoff
near the top of the range stands at that node, so its contribution is exactly a
quarter — an order-one quantity assembled entirely out of numbers at the two
ends of the format.

The safe action pays an eighth for certain. The risky one pays nothing now and
reaches that node. A quarter beats an eighth, so the risky action is optimal,
and an implementation that lets the joint probability round to zero prices the
risky action at nothing and picks the safe one. The disagreement is a reversed
discrete choice, not a tolerance.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    DiscreteGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.typing import DiscreteAction, DiscreteState, FloatND, ScalarFloat, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _Level:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _Bet:
    safe: ScalarInt
    risky: ScalarInt


_SAFE_PAYOFF = 0.125
_RISKY_CONTINUATION = 0.25


def _active_dtype() -> np.dtype:
    """The precision the suite is running at."""
    return np.dtype(jnp.zeros(()).dtype)


def _factor() -> ScalarFloat:
    """A probability the dtype holds, whose square it does not."""
    dtype = _active_dtype()
    exponent = -512 if dtype == np.float64 else -64
    return jnp.ldexp(jnp.asarray(1.0, dtype=dtype), exponent)


def _low_low_payoff() -> ScalarFloat:
    """A payoff whose product with the joint probability is a quarter."""
    dtype = _active_dtype()
    exponent = 1022 if dtype == np.float64 else 126
    return jnp.ldexp(jnp.asarray(1.0, dtype=dtype), exponent)


def _certain() -> ScalarFloat:
    return jnp.asarray(1.0, dtype=_active_dtype())


def _bet_payoff(bet: DiscreteAction) -> FloatND:
    """The safe action's certain payoff; the risky one pays only through the node."""
    return jnp.where(
        bet == _Bet.safe,
        jnp.asarray(_SAFE_PAYOFF, dtype=_active_dtype()),
        jnp.asarray(0.0, dtype=_active_dtype()),
    )


def _row(*, bet: DiscreteAction, level: DiscreteState, reachable: bool) -> FloatND:
    """The next-level distribution: low only under the risky action, and rarely.

    The row is the same from either current level — the two states are
    independent draws — but it is written as a function of the level so the
    state it moves is one the regime reads.
    """
    dtype = _active_dtype()
    low = jnp.where(
        (bet == _Bet.risky) & reachable, _factor(), jnp.asarray(0.0, dtype=dtype)
    )
    low = low + jnp.asarray(0.0, dtype=dtype) * level
    return jnp.stack([low, 1.0 - low])


def _health_probs(*, bet: DiscreteAction, health: DiscreteState) -> FloatND:
    return _row(bet=bet, level=health, reachable=True)


def _mood_probs(*, bet: DiscreteAction, mood: DiscreteState) -> FloatND:
    return _row(bet=bet, level=mood, reachable=True)


def _unreachable_health_probs(*, bet: DiscreteAction, health: DiscreteState) -> FloatND:
    return _row(bet=bet, level=health, reachable=False)


def _unreachable_mood_probs(*, bet: DiscreteAction, mood: DiscreteState) -> FloatND:
    return _row(bet=bet, level=mood, reachable=False)


def _terminal_payoff(*, health: DiscreteState, mood: DiscreteState) -> FloatND:
    """A payoff near the top of the range, standing only at the joint low node."""
    at_the_node = (health == _Level.low) & (mood == _Level.low)
    return jnp.where(
        at_the_node, _low_low_payoff(), jnp.asarray(0.0, dtype=_active_dtype())
    )


def _model(*, node_is_reachable: bool = True) -> Model:
    levels = DiscreteGrid(category_class=_Level)
    health_probs = _health_probs if node_is_reachable else _unreachable_health_probs
    mood_probs = _mood_probs if node_is_reachable else _unreachable_mood_probs
    return Model(
        regimes={
            "alive": Regime(
                transition={"dead": MarkovTransition(_certain)},
                active=lambda age: age < 21,
                actions={"bet": DiscreteGrid(category_class=_Bet)},
                states={"health": levels, "mood": levels},
                state_transitions={
                    "health": MarkovTransition(health_probs),
                    "mood": MarkovTransition(mood_probs),
                },
                functions={"utility": _bet_payoff},
            ),
            "dead": Regime(
                transition=None,
                states={"health": levels, "mood": levels},
                functions={"utility": _terminal_payoff},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=_RegimeId,
    )


def _solve(*, node_is_reachable: bool) -> FloatND:
    V = (
        _model(node_is_reachable=node_is_reachable)
        .solve(
            params={"alive": {"koopmans_aggregator": {"discount_factor": 1.0}}},
            log_level="off",
        )
        .values
    )
    return jnp.asarray(V[0]["alive"])


def test_the_witness_rests_on_a_joint_node_below_the_normal_range() -> None:
    """The premise: both factors are normal and their product is not.

    Whether the product then arrives as a subnormal or as a flushed zero is a
    property of the executing backend, so what is asserted is the part that
    holds on any of them.
    """
    dtype = _active_dtype()
    factor = float(_factor())
    tiny = float(np.finfo(dtype).tiny)

    assert factor >= tiny
    assert float(jnp.asarray(_factor()) * jnp.asarray(_factor())) < tiny


def test_a_joint_node_the_dtype_cannot_hold_still_wins_the_bellman_choice() -> None:
    """The risky action is worth a quarter, so it beats the safe eighth."""
    np.testing.assert_array_equal(
        np.asarray(_solve(node_is_reachable=True)),
        np.full((2, 2), _RISKY_CONTINUATION, dtype=_active_dtype()),
    )


def test_the_safe_action_stands_where_that_node_cannot_be_reached() -> None:
    """The control: make the node impossible and the safe payoff is the answer.

    The same model with both Markov rows placing no mass on the low level has
    nothing for the risky action to reach, so an eighth wins. The quarter above
    is therefore the joint node's contribution rather than an artefact of how
    the model is put together.
    """
    np.testing.assert_array_equal(
        np.asarray(_solve(node_is_reachable=False)),
        np.full((2, 2), _SAFE_PAYOFF, dtype=_active_dtype()),
    )
