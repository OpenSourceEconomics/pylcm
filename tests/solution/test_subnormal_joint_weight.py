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

from _lcm.zero_safe import joint_weight
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
_WEALTH_VALUES = np.array([1.0, 2.0, 3.0, 4.0])


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


def _no_utility(*, health: DiscreteState, mood: DiscreteState) -> FloatND:
    return jnp.asarray(0.0) + 0.0 * health + 0.0 * mood


def _wealth_utility(
    *, wealth: ScalarFloat, health: DiscreteState, mood: DiscreteState
) -> FloatND:
    return wealth + 0.0 * health + 0.0 * mood


def _keep_wealth(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _assert_carries_an_unrepresentable_probability(
    *, weight: float, exact_product: float, tiny: float
) -> None:
    """Assert the three properties a vanished product must satisfy on any backend.

    Whether the substitution fires at all is a property of the executing
    backend, not of the contract: XLA:CPU flushes such a product and receives
    the substitute, while CUDA represents it and keeps its own magnitude. Both
    satisfy the same three statements, so those are what is asserted.

    The agreement bound is two-sided. A product can underflow *below* the
    smallest representable magnitude — two float32 factors of `1e-30` multiply
    to `1e-60` — and the substitute then overstates it. What is bounded is the
    absolute difference, by that same smallest magnitude, which is far below
    every declared tolerance.
    """
    smallest = abs(exact_product) * 0.0 + float(np.finfo(np.float32).smallest_subnormal)
    assert weight != 0.0, (
        "a node that can occur must stay distinguishable from one that cannot"
    )
    assert abs(abs(weight) - abs(exact_product)) <= max(abs(exact_product), smallest), (
        "the weight agrees with its node to within the smallest representable magnitude"
    )
    assert abs(weight) < tiny, "the weight is below the normal range either way"


@pytest.mark.parametrize("n_factors", [2, 3, 4])
def test_normal_factors_with_an_unrepresentable_product_stay_nonzero(
    n_factors: int,
) -> None:
    """Factors the dtype holds, with a product it does not, do not become zero.

    Zero is reserved for an event that cannot occur, so such a product stays
    nonzero, never exceeds its true value, and stays below the normal range.
    """
    factor = float(np.finfo(np.float32).tiny ** (1.0 / n_factors) / 2)
    factors = jnp.asarray([factor] * n_factors, dtype=jnp.float32)
    assert bool(jnp.all(factors >= np.finfo(np.float32).tiny))

    _assert_carries_an_unrepresentable_probability(
        weight=float(joint_weight(factors)),
        exact_product=factor**n_factors,
        tiny=float(np.finfo(np.float32).tiny),
    )


def test_a_product_below_the_smallest_magnitude_is_overstated_but_bounded() -> None:
    """A product too small even to substitute for is bounded, not faithful.

    Two float32 factors of `1e-30` multiply to `1e-60`, far below the smallest
    representable magnitude. The substitute is therefore larger than the true
    probability, and the guarantee is the absolute bound rather than a direction:
    the node is still priced below every tolerance, and still distinguishable
    from one that cannot occur.
    """
    factors = jnp.asarray([1e-30, 1e-30], dtype=jnp.float32)
    smallest = float(np.finfo(np.float32).smallest_subnormal)

    weight = float(joint_weight(factors))

    assert weight != 0.0
    assert weight > 1e-60, "the substitute is larger than the true product here"
    assert weight <= smallest, (
        "and never larger than the smallest representable magnitude"
    )


def test_a_factor_of_exactly_zero_is_a_null_event() -> None:
    """An impossible event keeps its zero weight rather than being raised."""
    factors = jnp.asarray([0.0, 0.5], dtype=jnp.float32)

    assert float(joint_weight(factors)) == 0.0


def test_ordinary_factors_multiply_as_they_always_did() -> None:
    """A representable product is returned unchanged."""
    factors = jnp.asarray([0.25, 0.5, 0.5], dtype=jnp.float32)

    np.testing.assert_allclose(float(joint_weight(factors)), 0.0625, rtol=0)


def test_a_directly_supplied_subnormal_factor_keeps_its_own_magnitude() -> None:
    """A subnormal that arrives already representable is passed through as it is.

    Its own magnitude is more informative than any substitute, and enlarging
    it would overstate a node the format merely cannot multiply.
    """
    subnormal = float(
        np.nextafter(np.float32(np.finfo(np.float32).tiny), np.float32(0))
    )
    factors = jnp.asarray([subnormal], dtype=jnp.float32)

    assert float(joint_weight(factors)) == float(subnormal)


def test_a_negative_factor_stays_visible() -> None:
    """A negative probability is malformed, and is not rescued into a refusal."""
    factors = jnp.asarray([-0.5, 0.5], dtype=jnp.float32)

    assert float(joint_weight(factors)) == -0.25


def test_the_rule_survives_jit() -> None:
    """The rule is arithmetic, so it holds inside a compiled program."""
    factor = float(np.finfo(np.float32).tiny) ** 0.5 / 2
    factors = jnp.asarray([factor, factor], dtype=jnp.float32)

    _assert_carries_an_unrepresentable_probability(
        weight=float(jax.jit(joint_weight)(factors)),
        exact_product=factor**2,
        tiny=float(np.finfo(np.float32).tiny),
    )


@pytest.mark.parametrize("n_factors", [2, 3, 4])
def test_the_rule_holds_at_the_active_precision(n_factors: int) -> None:
    """Whatever precision the suite runs at, an unrepresentable product stays nonzero.

    float64 needs far smaller factors than float32 before a product underflows,
    so the witness is derived from the active dtype rather than fixed.
    """
    active = jnp.zeros(()).dtype
    tiny = float(jnp.finfo(active).tiny)
    factor = tiny ** (1.0 / n_factors) / 2
    factors = jnp.asarray([factor] * n_factors, dtype=active)
    assert bool(jnp.all(factors >= tiny))

    _assert_carries_an_unrepresentable_probability(
        weight=float(joint_weight(factors)), exact_product=factor**n_factors, tiny=tiny
    )


def test_an_ordinary_lottery_is_untouched_at_the_active_precision() -> None:
    """Representable factors keep their exact product."""
    active = jnp.zeros(()).dtype
    factors = jnp.asarray([0.5, 0.25], dtype=active)

    np.testing.assert_allclose(float(joint_weight(factors)), 0.125, rtol=0)


def test_a_model_whose_joint_node_underflows_still_solves() -> None:
    """Two ordinary Markov rows can still make a joint node the dtype cannot hold.

    Each row is a valid distribution of normal probabilities, so every
    input-level check passes. Their joint node is the product of the two small
    entries, which underflows. Its value is finite, so its contribution is below
    the last bit of the continuation and the solve answers as though the node
    were absent — which it is, to every tolerance the model declares.
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
                    "health": DiscreteGrid(category_class=_Binary),
                    "mood": DiscreteGrid(category_class=_Binary),
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
                    "health": DiscreteGrid(category_class=_Binary),
                    "mood": DiscreteGrid(category_class=_Binary),
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

    np.testing.assert_allclose(
        np.asarray(V[0]["alive"]), np.broadcast_to(_WEALTH_VALUES, (2, 2, 4))
    )
