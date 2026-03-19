"""Model with MarkovTransition on regime transitions."""

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.typing import DiscreteState, FloatND, Period


@categorical(ordered=True)
class Health:
    bad: int
    good: int


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


def _next_regime_probs(
    period: Period,
    health: DiscreteState,
    probs_array: FloatND,
) -> FloatND:
    return probs_array[period, health]


alive = Regime(
    transition=MarkovTransition(_next_regime_probs),
    states={
        "health": DiscreteGrid(Health),
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
    },
    state_transitions={
        "health": None,
        "wealth": lambda wealth: wealth,
    },
    functions={"utility": lambda wealth, health: wealth + health},
    active=lambda age: age < 62,
)

dead = Regime(
    transition=None,
    functions={"utility": lambda: 0.0},
)


def get_model() -> Model:
    """Create a model with MarkovTransition on regime transitions."""
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=60, stop=62, step="Y"),
        regime_id_class=RegimeId,
    )
