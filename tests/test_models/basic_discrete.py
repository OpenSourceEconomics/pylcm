"""Basic model with discrete + continuous states, no stochastic transitions."""

from lcm import (
    AgeGrid,
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


working_life = UserRegime(
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
    functions={"utility": lambda: 0.0},
)


def get_model() -> Model:
    """Create a minimal model with discrete + continuous states and three regimes.

    `working_life` and `retirement` are the two economic regimes; both lead to
    the terminal `dead` regime.
    """
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
