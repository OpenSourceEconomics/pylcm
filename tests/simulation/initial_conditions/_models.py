"""Small unsolved models for initial-condition validation and feasibility tests."""

import jax.numpy as jnp

from lcm import DiscreteGrid, ExecutionConfig, LinSpacedGrid, Model, categorical
from lcm.ages import AgeGrid
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


def make_minimal_model() -> Model:
    """Minimal model with two states (wealth, health) for initial states tests."""

    @categorical(ordered=False)
    class Health:
        healthy: ScalarInt
        sick: ScalarInt

    @categorical(ordered=False)
    class RegimeId:
        active: ScalarInt
        terminal: ScalarInt

    def utility(*, wealth: ContinuousState, health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array(0.0)

    def next_regime(period: int) -> ScalarInt:
        return jnp.where(
            period + 1 >= 2,
            RegimeId.terminal,
            RegimeId.active,
        )

    n_periods = 2
    ages = AgeGrid(start=0, stop=n_periods, step="Y")

    alive = UserRegime(
        functions={"utility": utility},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "health": DiscreteGrid(category_class=Health),
        },
        state_transitions={
            "wealth": lambda wealth: wealth,
            "health": lambda health: health,
        },
        transition=next_regime,
        active=lambda age: age < n_periods - 1,
    )

    dead = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= n_periods - 1,
    )

    return Model(
        regimes={"active": alive, "terminal": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 2.0


def make_constraint_model(wealth_grid) -> Model:
    """Create a constraint model with the given wealth grid."""
    final_age = 1

    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        dead: ScalarInt

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def borrowing_constraint(
        *, consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    def next_regime(*, age: float, final_age_alive: float) -> ScalarInt:
        return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.working_life)

    working_regime = UserRegime(
        actions={
            "consumption": LinSpacedGrid(start=0.5, stop=10, n_points=20),
        },
        states={"wealth": wealth_grid},
        state_transitions={"wealth": _next_wealth},
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime,
        functions={"utility": utility},
        active=lambda age: age <= final_age,
    )
    dead_regime = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age > final_age,
    )
    return Model(
        regimes={"working_life": working_regime, "dead": dead_regime},
        ages=AgeGrid(start=0, stop=final_age + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def make_constrained_asymmetric_model() -> Model:
    """Create a constrained asymmetric model.

    Alive has a consumption action with `consumption <= wealth` constraint;
    dead has no actions and no constraints. Consumption grid starts at 51, so
    wealth <= 50 makes all actions infeasible in alive.
    Ages 0, 1, 2. alive is active for age < 2, dead is active for age >= 2.
    """

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def borrowing_constraint(
        *, consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    def next_wealth(
        *, wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption + 60.0

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 1, RegimeId.dead, RegimeId.alive)

    alive = UserRegime(
        functions={"utility": utility},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
        state_transitions={
            "wealth": next_wealth,
        },
        actions={
            "consumption": LinSpacedGrid(start=51, stop=100, n_points=10),
        },
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime,
        active=lambda age: age < 2,
    )

    dead = UserRegime(
        transition=None,
        functions={"utility": lambda wealth: wealth},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
        active=lambda age: age >= 2,
    )

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=RegimeId,
    )


def make_asymmetric_state_model() -> Model:
    """Create a model where alive has 2 states (wealth + health), dead has 1 (wealth).

    Ages 0, 1, 2.  alive is active for age < 2, dead is active for age >= 2.
    """

    @categorical(ordered=False)
    class Health:
        healthy: ScalarInt
        sick: ScalarInt

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    def utility(
        *,
        wealth: ContinuousState,
        health: DiscreteState,  # noqa: ARG001
    ) -> FloatND:
        return wealth

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 1, RegimeId.dead, RegimeId.alive)

    alive = UserRegime(
        functions={"utility": utility},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "health": DiscreteGrid(category_class=Health),
        },
        state_transitions={
            "wealth": lambda wealth: wealth,
            "health": lambda health: health,
        },
        transition=next_regime,
        active=lambda age: age < 2,
    )

    dead = UserRegime(
        transition=None,
        functions={"utility": lambda wealth: wealth},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
        active=lambda age: age >= 2,
    )

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=RegimeId,
    )


def make_state_only_constraint_model(
    *, device_memory_bytes: int | None = None
) -> Model:
    """Create a model whose action-free `dead` regime carries a state-only constraint.

    `alive` chooses consumption subject to `consumption <= wealth`; `dead` has no
    actions but requires `wealth > 0`. Ages 0, 1, 2; `alive` is active for age < 2,
    `dead` for age >= 2.
    """

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def borrowing_constraint(
        *, consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    def solvent(wealth: ContinuousState) -> BoolND:
        return wealth > 0

    def dead_utility(wealth: ContinuousState) -> FloatND:
        return wealth

    def next_wealth(
        *, wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption + 60.0

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 1, RegimeId.dead, RegimeId.alive)

    alive = UserRegime(
        functions={"utility": utility},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        state_transitions={"wealth": next_wealth},
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime,
        active=lambda age: age < 2,
    )
    dead = UserRegime(
        transition=None,
        functions={"utility": dead_utility},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        constraints={"solvent": solvent},
        active=lambda age: age >= 2,
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=device_memory_bytes),
    )
