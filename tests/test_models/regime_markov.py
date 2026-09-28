"""Model with MarkovTransition on regime transitions."""

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import DiscreteState, FloatND, Period, ScalarInt
from tests.test_models.schedules import until_exit


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _next_regime_probs(
    *,
    period: Period,
    health: DiscreteState,
    probs_array: FloatND,
) -> FloatND:
    return probs_array[period, health]


alive = UserRegime(
    regime_transitions=until_exit(
        62,
        law=MarkovTransition(func=_next_regime_probs, targets=("alive", "dead")),
        exits=("dead",),
    ),
    states={
        "health": DiscreteGrid(category_class=Health),
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
    },
    state_transitions={
        "health": fixed_transition("health"),
        "wealth": lambda wealth: wealth,
    },
    functions={"utility": lambda wealth, health: wealth + health},
)

dead = UserRegime(
    regime_transitions=None,
    functions={"utility": lambda: 0.0},
)


def get_model() -> Model:
    """Create a model with MarkovTransition on regime transitions."""
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=60, stop=62, step="Y"),
        regime_id_class=RegimeId,
    )
