"""Basic model with discrete + continuous states, no stochastic transitions."""

from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import ScalarInt


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working_life: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def _next_regime() -> ScalarInt:
    return RegimeId.dead


working_life = UserRegime(
    regime_transitions=DeterministicTransition(func=_next_regime),
    states={
        "health": DiscreteGrid(category_class=Health),
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=10),
    },
    state_transitions={
        "health": fixed_transition("health"),
        "wealth": lambda wealth: wealth,
    },
    functions={"utility": lambda wealth, health: wealth + health},
)

retirement = UserRegime(
    regime_transitions=DeterministicTransition(func=_next_regime),
    states={
        "health": DiscreteGrid(category_class=Health),
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=10),
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
    """Create a minimal model with discrete + continuous states and two regimes."""
    return Model(
        edges={
            "working_life": {"dead": (25, 35, 45, 55, 65)},
            "retirement": {"dead": (25, 35, 45, 55, 65)},
        },
        regimes={
            "working_life": working_life,
            "retirement": retirement,
            "dead": dead,
        },
        ages=AgeGrid(start=25, inclusive_stop=75, step="10Y"),
        regime_id_class=RegimeId,
        initial_nodes={25: "working_life"},
    )
