"""An event that can occur may not be priced as one that cannot.

A joint node's probability is the product of one factor per stochastic axis.
Each factor can sit comfortably inside the dtype's normal range while the
product falls below it: in float32, `sqrt(tiny)/2` squared is subnormal. The
hardware then delivers exactly zero, and a zero weight is how this engine
spells "this cannot happen" — so an event with strictly positive probability
would be dropped, silently, and a `-inf` standing at that node (the ordinary
value of a state where no action is feasible) would never reach the answer.

The representation cannot carry such a weight, so the engine refuses it rather
than rounding it to impossible: the continuation is `NaN`. That is arithmetic
rather than validation, so it holds at every log level.

A factor of exactly zero is a different thing entirely — a genuine null event —
and still contributes nothing, whatever value stands at it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.zero_safe import joint_weight_or_nan
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.typing import DiscreteState, FloatND, ScalarFloat, ScalarInt

_WEALTH = LinSpacedGrid(start=1.0, stop=4.0, n_points=4)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _Binary:
    low: ScalarInt
    high: ScalarInt


def _certain() -> ScalarFloat:
    return jnp.float32(1)


def _no_utility(health: DiscreteState, mood: DiscreteState) -> FloatND:
    return jnp.asarray(0.0) + 0.0 * health + 0.0 * mood


def _wealth_utility(
    wealth: ScalarFloat, health: DiscreteState, mood: DiscreteState
) -> FloatND:
    return wealth + 0.0 * health + 0.0 * mood


def _keep_wealth(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


@pytest.mark.parametrize("n_factors", [2, 3, 4])
def test_normal_factors_with_an_unrepresentable_product_give_nan(
    n_factors: int,
) -> None:
    """Factors the dtype holds, with a product it does not, refuse to be zero."""
    factor = float(np.finfo(np.float32).tiny ** (1.0 / n_factors) / 2)
    factors = jnp.asarray([factor] * n_factors, dtype=jnp.float32)
    assert bool(jnp.all(factors >= np.finfo(np.float32).tiny))

    assert bool(jnp.isnan(joint_weight_or_nan(factors)))


def test_a_factor_of_exactly_zero_is_a_null_event_not_a_refusal() -> None:
    """An impossible event keeps its zero weight rather than becoming `NaN`."""
    factors = jnp.asarray([0.0, 0.5], dtype=jnp.float32)

    assert float(joint_weight_or_nan(factors)) == 0.0


def test_ordinary_factors_multiply_as_they_always_did() -> None:
    """A representable product is returned unchanged."""
    factors = jnp.asarray([0.25, 0.5, 0.5], dtype=jnp.float32)

    np.testing.assert_allclose(float(joint_weight_or_nan(factors)), 0.0625, rtol=0)


def test_a_directly_supplied_subnormal_factor_is_refused() -> None:
    """A single subnormal probability is unrepresentable, not impossible.

    It compares equal to zero under every arithmetic test, so nothing short of
    reading its bits can tell it from a null event.
    """
    subnormal = float(
        np.nextafter(np.float32(np.finfo(np.float32).tiny), np.float32(0))
    )
    factors = jnp.asarray([subnormal], dtype=jnp.float32)

    assert bool(jnp.isnan(joint_weight_or_nan(factors)))


def test_a_negative_factor_stays_visible() -> None:
    """A negative probability is malformed, and is not rescued into a refusal."""
    factors = jnp.asarray([-0.5, 0.5], dtype=jnp.float32)

    assert float(joint_weight_or_nan(factors)) == -0.25


def test_the_refusal_survives_jit() -> None:
    """The check is arithmetic, so it holds inside a compiled program."""
    factor = float(np.finfo(np.float32).tiny) ** 0.5 / 2
    factors = jnp.asarray([factor, factor], dtype=jnp.float32)

    assert bool(jnp.isnan(jax.jit(joint_weight_or_nan)(factors)))


@pytest.mark.parametrize("n_factors", [2, 3, 4])
def test_the_refusal_holds_at_the_active_precision(n_factors: int) -> None:
    """Whatever precision the suite runs at, an unrepresentable product refuses.

    float64 needs far smaller factors than float32 before a product underflows,
    so the witness is derived from the active dtype rather than fixed.
    """
    active = jnp.zeros(()).dtype
    tiny = float(jnp.finfo(active).tiny)
    factor = tiny ** (1.0 / n_factors) / 2
    factors = jnp.asarray([factor] * n_factors, dtype=active)
    assert bool(jnp.all(factors >= tiny))

    assert bool(jnp.isnan(joint_weight_or_nan(factors)))


def test_an_ordinary_lottery_is_untouched_at_the_active_precision() -> None:
    """Representable factors keep their exact product."""
    active = jnp.zeros(()).dtype
    factors = jnp.asarray([0.5, 0.25], dtype=active)

    np.testing.assert_allclose(float(joint_weight_or_nan(factors)), 0.125, rtol=0)


def test_a_model_whose_joint_node_underflows_solves_to_nan() -> None:
    """Two ordinary Markov rows can still make a joint node the dtype cannot hold.

    Each row is a valid distribution of normal probabilities, so every
    input-level check passes. Their joint node is the product of the two small
    entries, which underflows. The solve refuses rather than dropping the node.
    """
    active = jnp.zeros(()).dtype
    small = float(jnp.finfo(active).tiny) ** 0.5 / 2

    def _health_probs() -> FloatND:
        return jnp.asarray([small, 1.0 - small], dtype=active)

    def _mood_probs() -> FloatND:
        return jnp.asarray([small, 1.0 - small], dtype=active)

    model = Model(
        regimes={
            "alive": Regime(
                transition={"dead": MarkovTransition(_certain)},
                active=lambda age: age < 21,
                states={
                    "wealth": _WEALTH,
                    "health": DiscreteGrid(_Binary),
                    "mood": DiscreteGrid(_Binary),
                },
                state_transitions={
                    "wealth": _keep_wealth,
                    "health": MarkovTransition(_health_probs),
                    "mood": MarkovTransition(_mood_probs),
                },
                functions={"utility": _no_utility},
            ),
            "dead": Regime(
                transition=None,
                states={
                    "wealth": _WEALTH,
                    "health": DiscreteGrid(_Binary),
                    "mood": DiscreteGrid(_Binary),
                },
                functions={"utility": _wealth_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=_RegimeId,
    )

    V = model.solve(
        params={"alive": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        log_level="off",
    )

    assert bool(jnp.all(jnp.isnan(jnp.asarray(V[0]["alive"]))))
