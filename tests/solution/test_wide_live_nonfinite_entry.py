"""A node too rare to weigh is still a node that can occur.

An entry law names a value for every node of the draws it reads, and a value
outside the target's support is a misspecified model rather than a number to be
averaged. Whether the engine says so must not depend on how small the node's
probability is: a joint node reached only when several independent draws are all
rare carries a probability far below the dtype's normal range, and the consumer
that reduces the lottery as ordinary numbers cannot name that weight at all.

An entry it cannot weigh is not an entry it may drop. The weight is understated,
which the contract allows, but the node stays live, so a `NaN` standing there
survives to the value function and the model fails loudly.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    MarkovTransition,
    Model,
    Regime,
    UniformIIDProcess,
    categorical,
)
from lcm.typing import DiscreteState, FloatND, ScalarFloat, ScalarInt

_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}

# Income lives on `[0, 2]`, so the first is outside the target's support and the
# second is an ordinary value inside it.
_OFF_SUPPORT_ENTRY = 10.0
_IN_SUPPORT_ENTRY = 2.0

# What every node other than the all-rare one enters at.
_ORDINARY_ENTRY = 1.0


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


@categorical(ordered=False)
class Draw:
    common: ScalarInt
    rare: ScalarInt


def _active_dtype() -> np.dtype:
    """The precision the suite is running at."""
    return np.dtype(jnp.zeros(()).dtype)


def _rare_probability_exponent() -> int:
    """The base-two exponent of one axis's rare branch."""
    return -100 if _active_dtype() == np.float32 else -700


def _n_axes() -> int:
    """How many draws have to be rare at once.

    Enough that the joint node's probability falls below the likeliest node's by
    more than the format can express, so that lowering the two onto one base-two
    scale sends this one to zero.
    """
    return 3 if _active_dtype() == np.float32 else 4


def _certain() -> FloatND:
    return jnp.asarray(1.0, dtype=_active_dtype())


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _income_utility_of_three_draws(
    income: ScalarFloat,
    draw_0: DiscreteState,
    draw_1: DiscreteState,
    draw_2: DiscreteState,
) -> FloatND:
    """Income, reading the draws it is indexed by so none of them is idle."""
    return income + 0.0 * (draw_0 + draw_1 + draw_2)


def _income_utility_of_four_draws(
    income: ScalarFloat,
    draw_0: DiscreteState,
    draw_1: DiscreteState,
    draw_2: DiscreteState,
    draw_3: DiscreteState,
) -> FloatND:
    """Income, reading the draws it is indexed by so none of them is idle."""
    return income + 0.0 * (draw_0 + draw_1 + draw_2 + draw_3)


def _draw_probabilities() -> FloatND:
    """One axis's law: its rare branch, and everything else."""
    rare = jnp.asarray(2.0 ** _rare_probability_exponent(), dtype=_active_dtype())
    return jnp.stack([1.0 - rare, rare])


def _entry_at(*, every_draw_is_rare: FloatND, rare_entry: float) -> FloatND:
    return jnp.where(
        every_draw_is_rare,
        jnp.asarray(rare_entry, dtype=_active_dtype()),
        jnp.asarray(_ORDINARY_ENTRY, dtype=_active_dtype()),
    )


def _make_entry(rare_entry: float):
    """Build the entry law for however many draws this precision needs."""
    if _n_axes() == 3:

        def entry_income(
            next_draw_0: DiscreteState,
            next_draw_1: DiscreteState,
            next_draw_2: DiscreteState,
        ) -> FloatND:
            return _entry_at(
                every_draw_is_rare=(next_draw_0 == Draw.rare)
                & (next_draw_1 == Draw.rare)
                & (next_draw_2 == Draw.rare),
                rare_entry=rare_entry,
            )

        return entry_income

    def entry_income(
        next_draw_0: DiscreteState,
        next_draw_1: DiscreteState,
        next_draw_2: DiscreteState,
        next_draw_3: DiscreteState,
    ) -> FloatND:
        return _entry_at(
            every_draw_is_rare=(next_draw_0 == Draw.rare)
            & (next_draw_1 == Draw.rare)
            & (next_draw_2 == Draw.rare)
            & (next_draw_3 == Draw.rare),
            rare_entry=rare_entry,
        )

    return entry_income


def _build_model(*, rare_entry: float, enable_jit: bool) -> Model:
    """A source entering the target's income process at several rare draws."""
    axis_names = tuple(f"draw_{index}" for index in range(_n_axes()))
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_certain)},
                active=lambda age: age < 21,
                state_transitions={
                    "income": {"target": _make_entry(rare_entry)},
                    **{
                        name: {"target": MarkovTransition(_draw_probabilities)}
                        for name in axis_names
                    },
                },
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states={
                    "income": UniformIIDProcess(start=0.0, stop=2.0, n_points=3),
                    **{name: DiscreteGrid(Draw) for name in axis_names},
                },
                functions={
                    "utility": (
                        _income_utility_of_three_draws
                        if _n_axes() == 3
                        else _income_utility_of_four_draws
                    )
                },
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


def _source_value(model: Model) -> FloatND:
    return jnp.asarray(model.solve(params=_PARAMS, log_level="off")[0]["source"])


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_an_off_support_entry_at_a_rare_node_still_fails_loudly(
    *, enable_jit: bool
) -> None:
    """A node reachable only when every draw is rare can still occur.

    The entry names a value outside the target's support there, so the model is
    misspecified and the continuation is `NaN` rather than the average over the
    nodes whose weights the format happens to be able to name.
    """
    model = _build_model(rare_entry=_OFF_SUPPORT_ENTRY, enable_jit=enable_jit)

    assert bool(jnp.all(jnp.isnan(_source_value(model))))


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_an_in_support_entry_at_the_same_rare_node_is_priced_as_usual(
    *, enable_jit: bool
) -> None:
    """The same node carrying an ordinary value gives the ordinary mean.

    Its probability is far below what the mean can resolve, so the continuation
    is the common nodes' entry. This is the control for the loud failure above:
    what makes that one `NaN` is the value outside support, not the rarity.
    """
    model = _build_model(rare_entry=_IN_SUPPORT_ENTRY, enable_jit=enable_jit)

    value = float(jnp.max(_source_value(model)))

    np.testing.assert_allclose(value, _ORDINARY_ENTRY, rtol=1e-6)
