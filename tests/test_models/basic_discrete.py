"""Basic model with discrete + continuous states, no stochastic transitions."""

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.api.regime import Regime as UserRegime
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
    transition=_next_regime,
    states={
        "health": DiscreteGrid(Health),
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=10),
    },
    state_transitions={
        "health": None,
        "wealth": lambda wealth: wealth,
    },
    functions={"utility": lambda wealth, health: wealth + health},
)

retirement = UserRegime(
    transition=_next_regime,
    states={
        "health": DiscreteGrid(Health),
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=10),
    },
    state_transitions={
        "health": None,
        "wealth": lambda wealth: wealth,
    },
    functions={"utility": lambda wealth, health: wealth + health},
)

dead = UserRegime(
    transition=None,
    functions={"utility": lambda: 0.0},
)


def get_model() -> Model:
    """Create a minimal model with discrete + continuous states and two regimes."""
    return Model(
        regimes={
            "working_life": working_life,
            "retirement": retirement,
            "dead": dead,
        },
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=RegimeId,
    )
