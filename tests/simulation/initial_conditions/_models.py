"""Small unsolved models for initial-condition validation and feasibility tests."""

import jax.numpy as jnp

from lcm import (
    DeterministicTransition,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    Period,
    ScalarFloat,
    ScalarInt,
)
from tests.test_models.basic_discrete import Health


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
    ages = AgeGrid(start=0, inclusive_stop=n_periods, step="Y")

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
        regime_transitions=DeterministicTransition(func=next_regime),
    )

    dead = UserRegime(
        regime_transitions=None,
        functions={"utility": lambda: 0.0},
    )

    return Model(
        regimes={"active": alive, "terminal": dead},
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "active"},
        edges={"active": {"terminal": 0}},
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
        regime_transitions=DeterministicTransition(func=next_regime),
        functions={"utility": utility},
    )
    dead_regime = UserRegime(
        regime_transitions=None,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"working_life": working_regime, "dead": dead_regime},
        ages=AgeGrid(start=0, inclusive_stop=final_age + 1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "working_life"},
        edges={"working_life": {"working_life": 0, "dead": (0, 1)}},
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
        regime_transitions=DeterministicTransition(func=next_regime),
    )

    dead = UserRegime(
        regime_transitions=None,
        functions={"utility": lambda wealth: wealth},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
    )

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"alive": 0, "dead": (0, 1)}},
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
        regime_transitions=DeterministicTransition(func=next_regime),
    )

    dead = UserRegime(
        regime_transitions=None,
        functions={"utility": lambda wealth: wealth},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
    )

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "alive", 1: "alive", 2: "dead"},
        edges={"alive": {"alive": 0, "dead": (0, 1)}},
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
        regime_transitions=DeterministicTransition(func=next_regime),
    )
    dead = UserRegime(
        regime_transitions=None,
        functions={"utility": dead_utility},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        constraints={"solvent": solvent},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=device_memory_bytes),
        initial_nodes={0: "alive", 1: "alive", 2: "dead"},
        edges={"alive": {"alive": 0, "dead": (0, 1)}},
    )


def make_period_constraint_model() -> Model:
    """Create a model whose constraint reads `period`, a fixed and a runtime scalar.

    `alive` chooses consumption on a grid from 1 to 10 subject to
    `consumption <= wealth + period * cap - floor`, where `cap` is a runtime
    parameter and `floor` is fixed at model build. Ages 0, 1, 2 map to periods
    0, 1, 2; `dead` is active from age 3.
    """

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def affordable(
        *,
        consumption: ContinuousAction,
        wealth: ContinuousState,
        period: Period,
        cap: ScalarFloat,
        floor: ScalarFloat,
    ) -> BoolND:
        return consumption <= wealth + period * cap - floor

    def next_wealth(
        *, wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption + 5.0

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 2, RegimeId.dead, RegimeId.alive)

    alive = UserRegime(
        functions={"utility": utility},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        state_transitions={"wealth": next_wealth},
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=10)},
        constraints={"affordable": affordable},
        regime_transitions=DeterministicTransition(func=next_regime),
    )
    dead = UserRegime(
        regime_transitions=None,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=4, step="Y"),
        regime_id_class=RegimeId,
        fixed_params={"alive": {"affordable": {"floor": 0.5}}},
        initial_nodes={0: "alive", 1: "alive", 2: "alive"},
        edges={"alive": {"alive": (0, 1), "dead": (0, 1, 2)}},
    )


def make_joint_constraint_model() -> Model:
    """Create a model with two constraints that can conflict only jointly.

    `at_least` requires `consumption >= lower` and `borrowing` requires
    `consumption <= wealth`, with consumption on a grid from 1 to 10. Each is
    satisfiable alone at any wealth in the grid; together they fail exactly when
    `wealth < lower`.
    """

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def at_least(*, consumption: ContinuousAction, lower: ScalarFloat) -> BoolND:
        return consumption >= lower

    def borrowing(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
        return consumption <= wealth

    def next_wealth(
        *, wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption + 5.0

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 1, RegimeId.dead, RegimeId.alive)

    alive = UserRegime(
        functions={"utility": utility},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        state_transitions={"wealth": next_wealth},
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=10)},
        constraints={"at_least": at_least, "borrowing": borrowing},
        regime_transitions=DeterministicTransition(func=next_regime),
    )
    dead = UserRegime(
        regime_transitions=None,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "alive", 1: "alive"},
        edges={"alive": {"alive": 0, "dead": (0, 1)}},
    )


@categorical(ordered=True)
class HealthWithDisability:
    disabled: ScalarInt
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class HetRegimeId:
    pre65: ScalarInt
    post65: ScalarInt
    dead: ScalarInt


def _het_next_regime() -> ScalarInt:
    return HetRegimeId.dead


def _het_utility(*, wealth: float, health: int, bonus: float) -> float:
    return wealth + health + bonus


def _het_next_wealth(wealth: float) -> float:
    return wealth


def _het_dead_utility() -> float:
    return 0.0


def make_heterogeneous_health_model() -> Model:
    """Create a model whose `health` state has different categories per regime.

    `pre65` uses `HealthWithDisability` (three labels), `post65` uses `Health` (two),
    and `dead` has no states; ages 50, 60, 70 in steps of ten years.
    """
    pre65 = UserRegime(
        regime_transitions=DeterministicTransition(func=_het_next_regime),
        states={
            "health": DiscreteGrid(category_class=HealthWithDisability),
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
        },
        state_transitions={
            "health": fixed_transition("health"),
            "wealth": _het_next_wealth,
        },
        functions={"utility": _het_utility},
    )
    post65 = UserRegime(
        regime_transitions=DeterministicTransition(func=_het_next_regime),
        states={
            "health": DiscreteGrid(category_class=Health),
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
        },
        state_transitions={
            "health": fixed_transition("health"),
            "wealth": _het_next_wealth,
        },
        functions={"utility": _het_utility},
    )
    dead = UserRegime(
        regime_transitions=None,
        functions={"utility": _het_dead_utility},
    )
    return Model(
        regimes={"pre65": pre65, "post65": post65, "dead": dead},
        ages=AgeGrid(start=50, inclusive_stop=80, step="10Y"),
        regime_id_class=HetRegimeId,
        initial_nodes={50: "pre65", 70: "post65"},
        edges={"pre65": {"dead": (50, 60)}, "post65": {"dead": 70}},
    )
