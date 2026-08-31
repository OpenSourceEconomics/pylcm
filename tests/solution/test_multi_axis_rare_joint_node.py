"""A joint node reached only when every stochastic axis is rare stays in the model.

Several independent draws each have a rare branch, so the node where all of them
take it carries a probability far below the dtype's normal range — further below
than one base-two scale shared by the whole lottery can express. Under a
nonlinear certainty equivalent that node is not negligible: at power exponent
`p` it enters through a `1/p`-th root, so an arbitrarily small probability moves
the continuation by a first-order amount, and a household offered a safe
alternative takes it.

The oracle is exact decimal arithmetic on the same lottery, which shares no code
with the engine.
"""

from decimal import Decimal, getcontext

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    MarkovTransition,
    Model,
    PowerMean,
    Regime,
    categorical,
)
from lcm.typing import DiscreteState, FloatND, ScalarFloat, ScalarInt

# Leaves the power mean at exponent `1 - risk_aversion`, so `-3`.
_RISK_AVERSION = 4.0

# What the alternative to the lottery is worth.
_SAFE_VALUE = 0.5


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    lottery: ScalarInt
    safe: ScalarInt


@categorical(ordered=False)
class LotteryOnlyRegimeId:
    source: ScalarInt
    lottery: ScalarInt


@categorical(ordered=False)
class Draw:
    common: ScalarInt
    rare: ScalarInt


@categorical(ordered=False)
class Plan:
    gamble: ScalarInt
    take_the_safe_value: ScalarInt


def _active_dtype() -> np.dtype:
    """The precision the suite is running at."""
    return np.dtype(jnp.zeros(()).dtype)


def _rare_probability_exponent() -> int:
    """The base-two exponent of one axis's rare branch."""
    return -100 if _active_dtype() == np.float32 else -700


def _rare_value_exponent() -> int:
    """The base-two exponent of the value standing at the all-rare node."""
    return -126 if _active_dtype() == np.float32 else -1022


def _n_axes() -> int:
    """How many draws have to be rare at once.

    Enough that the joint probability falls below the dtype's normal range by
    more than its exponent field can absorb, which is the case no single scale
    shared across the lottery can carry.
    """
    return 3 if _active_dtype() == np.float32 else 4


def _exact_continuation() -> float:
    """The lottery's certainty equivalent, in exact decimal arithmetic."""
    getcontext().prec = 200
    rare = Decimal(2) ** _rare_probability_exponent()
    total_mass = (Decimal(1) + rare) ** _n_axes()
    rare_mass = rare ** _n_axes()
    rare_value = Decimal(2) ** _rare_value_exponent()
    exponent = Decimal(1) - Decimal(_RISK_AVERSION)
    moment = ((total_mass - rare_mass) + rare_mass * rare_value**exponent) / total_mass
    return float((moment.ln() / exponent).exp())


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _safe_utility() -> FloatND:
    return jnp.asarray(_SAFE_VALUE, dtype=_active_dtype())


def _draw_probabilities() -> FloatND:
    """One axis's law: its rare branch, and everything else."""
    rare = jnp.asarray(2.0 ** _rare_probability_exponent(), dtype=_active_dtype())
    return jnp.stack([1.0 - rare, rare])


def _value_at(every_draw_is_rare: FloatND) -> FloatND:
    return jnp.where(
        every_draw_is_rare,
        jnp.asarray(2.0 ** _rare_value_exponent(), dtype=_active_dtype()),
        jnp.asarray(1.0, dtype=_active_dtype()),
    )


def _utility_of_three_draws(
    *, draw_0: DiscreteState, draw_1: DiscreteState, draw_2: DiscreteState
) -> FloatND:
    """The catastrophic value where every draw is rare, and one elsewhere."""
    return _value_at(
        (draw_0 == Draw.rare) & (draw_1 == Draw.rare) & (draw_2 == Draw.rare)
    )


def _utility_of_four_draws(
    *,
    draw_0: DiscreteState,
    draw_1: DiscreteState,
    draw_2: DiscreteState,
    draw_3: DiscreteState,
) -> FloatND:
    """The catastrophic value where every draw is rare, and one elsewhere."""
    return _value_at(
        (draw_0 == Draw.rare)
        & (draw_1 == Draw.rare)
        & (draw_2 == Draw.rare)
        & (draw_3 == Draw.rare)
    )


def _certain() -> FloatND:
    return jnp.asarray(1.0, dtype=_active_dtype())


def _to_lottery(plan: DiscreteState) -> FloatND:
    return jnp.where(plan == Plan.gamble, 1.0, 0.0)


def _to_safe(plan: DiscreteState) -> FloatND:
    return jnp.where(plan == Plan.gamble, 0.0, 1.0)


def _build_model(*, with_a_safe_alternative: bool, enable_jit: bool) -> Model:
    """A source whose target is entered at several independent rare draws."""
    axis_names = tuple(f"draw_{index}" for index in range(_n_axes()))
    lottery_utility = (
        _utility_of_three_draws if _n_axes() == 3 else _utility_of_four_draws
    )
    regimes = {
        "source": Regime(
            transition=(
                {
                    "lottery": MarkovTransition(_to_lottery),
                    "safe": MarkovTransition(_to_safe),
                }
                if with_a_safe_alternative
                else {"lottery": MarkovTransition(_certain)}
            ),
            active=lambda age: age < 21,
            actions={"plan": DiscreteGrid(category_class=Plan)}
            if with_a_safe_alternative
            else {},
            state_transitions={
                name: {"lottery": MarkovTransition(_draw_probabilities)}
                for name in axis_names
            },
            functions={"utility": _zero_utility},
            certainty_equivalent=PowerMean(),
        ),
        "lottery": Regime(
            transition=None,
            states={name: DiscreteGrid(category_class=Draw) for name in axis_names},
            functions={"utility": lottery_utility},
        ),
    }
    if with_a_safe_alternative:
        regimes["safe"] = Regime(transition=None, functions={"utility": _safe_utility})
    return Model(
        regimes=regimes,
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=RegimeId if with_a_safe_alternative else LotteryOnlyRegimeId,
        enable_jit=enable_jit,
    )


_PARAMS = {
    "source": {
        "koopmans_aggregator": {"discount_factor": 1.0},
        "certainty_equivalent": {"risk_aversion": _RISK_AVERSION},
    }
}


def _source_value(model: Model) -> ScalarFloat:
    return jnp.max(
        jnp.asarray(model.solve(params=_PARAMS, log_level="off")[0]["source"])
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_the_all_rare_node_prices_the_continuation(*, enable_jit: bool) -> None:
    """The continuation equals the exact certainty equivalent of the full lottery."""
    model = _build_model(with_a_safe_alternative=False, enable_jit=enable_jit)

    value = float(_source_value(model))

    rtol = 1e-5 if _active_dtype() == np.float32 else 1e-12
    np.testing.assert_allclose(value, _exact_continuation(), rtol=rtol)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_the_safe_alternative_beats_the_lottery(*, enable_jit: bool) -> None:
    """A household offered `0.5` declines a lottery worth far less than that."""
    model = _build_model(with_a_safe_alternative=True, enable_jit=enable_jit)

    value = float(_source_value(model))

    np.testing.assert_allclose(value, _SAFE_VALUE, rtol=1e-6)
