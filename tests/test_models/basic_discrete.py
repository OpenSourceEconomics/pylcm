"""Basic model with discrete + continuous states, no stochastic transitions."""

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical


@categorical(ordered=True)
class Health:
    bad: int
    good: int


@categorical(ordered=False)
class RegimeId:
    working_life: int
    retirement: int
    dead: int


def _next_regime() -> int:
    return RegimeId.dead


working_life = Regime(
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

retirement = Regime(
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

dead = Regime(
    transition=None,
    functions={"utility": lambda: 0.0},
)


def get_model() -> Model:
    """Create a minimal model with discrete + continuous states and two regimes."""
    return Model(
        regimes={"working_life": working_life, "retirement": retirement, "dead": dead},
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=RegimeId,
    )
