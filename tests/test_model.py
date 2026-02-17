import jax.numpy as jnp
import pytest

import lcm
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


def test_regime_invalid_states():
    """Regime rejects non-dict states argument."""
    with pytest.raises(RegimeInitializationError, match="states must be a mapping"):
        Regime(
            transition=lambda: 0,
            states="health",  # ty: ignore[invalid-argument-type]
            actions={},
            functions={"utility": lambda: 0},
            active=lambda age: age < 5,
        )


def test_regime_invalid_actions():
    """Regime rejects non-dict actions argument."""
    with pytest.raises(RegimeInitializationError, match="actions must be a mapping"):
        Regime(
            transition=lambda: 0,
            states={},
            actions="exercise",  # ty: ignore[invalid-argument-type]
            functions={"utility": lambda: 0},
            active=lambda age: age < 5,
        )


def test_regime_invalid_functions():
    """Regime rejects non-dict functions argument."""
    with pytest.raises(
        RegimeInitializationError, match="functions must each be a mapping"
    ):
        Regime(
            transition=lambda: 0,
            states={},
            actions={},
            functions="utility",  # ty: ignore[invalid-argument-type]
            active=lambda age: age < 5,
        )


def test_regime_invalid_functions_values():
    """Regime rejects non-callable function values."""
    with pytest.raises(
        RegimeInitializationError,
        match=r"function values must be a callable, but is 0.",
    ):
        Regime(
            states={},
            actions={},
            transition=lambda: 0,
            functions={"utility": lambda: 0, "function": 0},  # ty: ignore[invalid-argument-type]
            active=lambda age: age < 5,
        )


def test_regime_invalid_functions_keys():
    """Regime rejects non-string function keys."""
    with pytest.raises(
        RegimeInitializationError, match=r"function keys must be a strings, but is 0."
    ):
        Regime(
            states={},
            actions={},
            transition=lambda: 0,
            functions={"utility": lambda: 0, 0: lambda: 0},  # ty: ignore[invalid-argument-type]
            active=lambda age: age < 5,
        )


def test_regime_invalid_actions_values():
    """Regime rejects non-grid action values."""
    with pytest.raises(
        RegimeInitializationError, match=r"actions value 0 must be an LCM grid."
    ):
        Regime(
            states={},
            actions={"exercise": 0},  # ty: ignore[invalid-argument-type]
            functions={"utility": lambda: 0},
            transition=lambda: 0,
            active=lambda age: age < 5,
        )


def test_regime_invalid_states_values():
    """Regime rejects non-grid state values."""
    with pytest.raises(
        RegimeInitializationError, match=r"states value 0 must be an LCM grid."
    ):
        Regime(
            states={"health": 0},  # ty: ignore[invalid-argument-type]
            actions={},
            functions={"utility": lambda: 0},
            transition=lambda: 0,
            active=lambda age: age < 5,
        )


def test_regime_invalid_utility():
    """Regime rejects non-callable utility argument."""
    with pytest.raises(
        RegimeInitializationError,
        match=(r"function values must be a callable, but is 0"),
    ):
        Regime(
            states={},
            actions={},
            functions={"utility": 0},  # ty: ignore[invalid-argument-type]
            transition=lambda: 0,
            active=lambda age: age < 5,
        )


def test_regime_overlapping_states_actions(binary_category_class):
    """Regime rejects overlapping state and action names."""
    with pytest.raises(
        RegimeInitializationError,
        match=r"States and actions cannot have overlapping names.",
    ):
        Regime(
            states={"health": DiscreteGrid(binary_category_class)},
            actions={"health": DiscreteGrid(binary_category_class)},
            functions={"utility": lambda: 0},
            transition=lambda: 0,
            active=lambda age: age < 5,
        )


def test_regime_transition_must_be_callable():
    """Regime rejects non-callable transition."""
    with pytest.raises(
        RegimeInitializationError,
        match="transition must be a callable or None",
    ):
        Regime(
            states={},
            actions={},
            functions={"utility": lambda: 0},
            transition=42,  # ty: ignore[invalid-argument-type]
            active=lambda age: age < 5,
        )


def test_model_requires_terminal_regime(binary_category_class):
    """Model must have at least one terminal regime."""

    @categorical
    class RegimeId:
        test: int

    regime = Regime(
        states={
            "health": DiscreteGrid(
                binary_category_class, transition=lambda health: health
            ),
        },
        actions={},
        functions={"utility": lambda health: health},
        transition=lcm.mark.stochastic(lambda: jnp.array([1.0])),
        active=lambda age: age < 1,
    )
    with pytest.raises(ModelInitializationError, match="at least one terminal regime"):
        Model(
            regimes={"test": regime},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=RegimeId,
        )


def test_model_requires_non_terminal_regime(binary_category_class):
    """Model must have at least one non-terminal regime."""

    @categorical
    class RegimeId:
        dead: int

    dead = Regime(
        transition=None,
        states={"health": DiscreteGrid(binary_category_class)},
        functions={"utility": lambda health: health * 0},
        active=lambda age: age >= 1,
    )
    with pytest.raises(ModelInitializationError, match="at least one non-terminal"):
        Model(
            regimes={"dead": dead},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=RegimeId,
        )


def test_model_keyword_only():
    """Model requires keyword arguments only."""
    # Positional arguments should raise TypeError
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        Model({}, AgeGrid(start=0, stop=2, step="Y"))  # ty: ignore[missing-argument,too-many-positional-arguments]


def test_model_accepts_multiple_terminal_regimes(binary_category_class):
    """Model can have multiple terminal regimes."""

    @categorical
    class RegimeId:
        alive: int
        dead1: int
        dead2: int

    alive = Regime(
        states={
            "health": DiscreteGrid(
                binary_category_class, transition=lambda health: health
            ),
        },
        functions={"utility": lambda health: health},
        transition=lcm.mark.stochastic(lambda: jnp.array([0.8, 0.1, 0.1])),
        active=lambda age: age < 1,
    )
    dead1 = Regime(
        transition=None,
        states={"health": DiscreteGrid(binary_category_class)},
        functions={"utility": lambda health: health * 0},
        active=lambda age: age >= 1,
    )
    dead2 = Regime(
        transition=None,
        states={"health": DiscreteGrid(binary_category_class)},
        functions={"utility": lambda health: health * 0},
        active=lambda age: age >= 1,
    )
    # Should not raise - multiple terminal regimes are allowed
    model = Model(
        regimes={"alive": alive, "dead1": dead1, "dead2": dead2},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )
    assert model.internal_regimes is not None


def test_model_regime_id_mapping_created_from_dict_keys(binary_category_class):
    """Model creates regime id mapping from dict keys in order."""

    @categorical
    class RegimeId:
        alive: int
        dead: int

    alive = Regime(
        states={
            "health": DiscreteGrid(
                binary_category_class, transition=lambda health: health
            ),
        },
        functions={"utility": lambda health: health},
        transition=lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        active=lambda age: age < 1,
    )
    dead = Regime(
        transition=None,
        states={"health": DiscreteGrid(binary_category_class)},
        functions={"utility": lambda health: health * 0},
        active=lambda age: age >= 1,
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )
    # regime id should be created from dict keys in order
    assert model.regime_names_to_ids["alive"] == 0
    assert model.regime_names_to_ids["dead"] == 1


def test_model_regime_name_validation(binary_category_class):
    """Model validates regime names don't contain the separator."""

    @categorical
    class RegimeId:
        alive__bad: int
        dead: int

    alive = Regime(
        states={
            "health": DiscreteGrid(
                binary_category_class, transition=lambda health: health
            ),
        },
        functions={"utility": lambda health: health},
        transition=lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        active=lambda age: age < 1,
    )
    dead = Regime(
        transition=None,
        states={"health": DiscreteGrid(binary_category_class)},
        functions={"utility": lambda health: health * 0},
        active=lambda age: age >= 1,
    )
    # Using separator in regime name should raise error
    with pytest.raises(ModelInitializationError, match="separator character"):
        Model(
            regimes={"alive__bad": alive, "dead": dead},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=RegimeId,
        )


def test_unused_state_raises_error():
    """Model raises error when a state is defined but never used."""

    @categorical
    class RegimeId:
        working: int
        retired: int

    @categorical
    class UnusedState:
        low: int
        medium: int
        high: int

    # Define a regime where 'unused_state' is not used in any function
    working = Regime(
        functions={
            "utility": lambda wealth, consumption: (
                jnp.log(consumption) + wealth * 0.001
            ),
        },
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
                transition=lambda wealth, consumption: wealth - consumption,
            ),
            "unused_state": DiscreteGrid(UnusedState),  # Not used anywhere!
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lcm.mark.stochastic(lambda: jnp.array([0.9, 0.1])),
        active=lambda age: age < 5,
    )

    retired = Regime(
        transition=None,
        functions={"utility": lambda wealth: wealth * 0.5},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "unused_state": DiscreteGrid(UnusedState),
        },
        active=lambda age: age >= 5,
    )

    # Should raise error about unused_state
    with pytest.raises(ModelInitializationError, match="unused_state"):
        Model(
            regimes={"working": working, "retired": retired},
            ages=AgeGrid(start=0, stop=5, step="Y"),
            regime_id_class=RegimeId,
        )


def test_unused_action_raises_error():
    """Model raises error when an action is defined but never used."""

    @categorical
    class RegimeId:
        working: int
        retired: int

    @categorical
    class UnusedAction:
        option_a: int
        option_b: int

    working = Regime(
        functions={
            "utility": lambda wealth, consumption: (
                jnp.log(consumption) + wealth * 0.001
            ),
        },
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
                transition=lambda wealth, consumption: wealth - consumption,
            ),
        },
        actions={
            "consumption": LinSpacedGrid(start=1, stop=50, n_points=10),
            "unused_action": DiscreteGrid(UnusedAction),  # Not used anywhere!
        },
        transition=lcm.mark.stochastic(lambda: jnp.array([0.9, 0.1])),
        active=lambda age: age < 5,
    )

    retired = Regime(
        transition=None,
        functions={"utility": lambda wealth: wealth * 0.5},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        active=lambda age: age >= 5,
    )

    # Should raise error about unused_action
    with pytest.raises(ModelInitializationError, match="unused_action"):
        Model(
            regimes={"working": working, "retired": retired},
            ages=AgeGrid(start=0, stop=5, step="Y"),
            regime_id_class=RegimeId,
        )


# ======================================================================================
# Reproducers for GitHub issue #230
# ======================================================================================


def test_constraint_depending_on_transition_output():
    """Test that constraints can depend on transition outputs like next_assets.

    Previously this worked, but now fails with:
    ValueError: list.index(x): x not in list

    The workaround is to rewrite the constraint to use raw states/actions instead
    of transition outputs.
    """

    @categorical
    class RegimeId:
        alive: int
        dead: int

    @categorical
    class EmploymentLastPeriod:
        unemployed: int
        employed: int

    @categorical
    class EmploymentStatus:
        not_employed: int
        employed: int

    def next_regime(age: float, model_end_age: int) -> ScalarInt:
        return jnp.where(age == model_end_age, RegimeId.dead, RegimeId.alive)

    def model_end_age(value: int) -> int:
        return value

    def utility(
        consumption_q: ContinuousAction, lagged_employment: DiscreteState
    ) -> FloatND:
        return jnp.log(consumption_q + lagged_employment * 0.001)

    def dead_utility() -> float:
        return 0.0

    def next_assets(
        assets: ContinuousState, consumption_q: ContinuousAction
    ) -> ContinuousState:
        return assets - consumption_q

    def next_lagged_employment(employment: DiscreteState) -> DiscreteState:
        return jnp.where(
            employment == EmploymentStatus.employed,
            EmploymentLastPeriod.employed,
            EmploymentLastPeriod.unemployed,
        )

    # This constraint depends on transition output - used to work, now fails
    def borrowing_constraint(next_assets: ContinuousState) -> BoolND:
        return next_assets >= 0.0

    alive_regime = Regime(
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime,
        functions={"utility": utility, "model_end_age": model_end_age},
        actions={
            "consumption_q": LinSpacedGrid(start=1, stop=10, n_points=5),
            "employment": DiscreteGrid(EmploymentStatus),
        },
        states={
            "assets": LinSpacedGrid(
                start=10, stop=100, n_points=5, transition=next_assets
            ),
            "lagged_employment": DiscreteGrid(
                EmploymentLastPeriod, transition=next_lagged_employment
            ),
        },
    )

    dead_regime = Regime(
        transition=None,
        functions={"utility": dead_utility},
    )

    # This should work but currently raises ValueError
    Model(
        regimes={"alive": alive_regime, "dead": dead_regime},
        ages=AgeGrid(start=59, stop=61, step="Y"),
        regime_id_class=RegimeId,
    )


def test_state_only_used_in_transitions():
    """Test that states can be used only in transitions, not in utility/constraints.

    Previously this worked, but now fails with:
    ValueError: list.index(x): x not in list

    The state 'assets' is only used in the next_assets transition, not directly
    in utility or constraints.
    """

    @categorical
    class RegimeId:
        alive: int
        dead: int

    @categorical
    class EmploymentLastPeriod:
        unemployed: int
        employed: int

    @categorical
    class EmploymentStatus:
        not_employed: int
        employed: int

    def next_regime(age: float, model_end_age: int) -> ScalarInt:
        return jnp.where(age == model_end_age, RegimeId.dead, RegimeId.alive)

    def model_end_age(value: int) -> int:
        return value

    # Utility does NOT use assets directly
    def utility(
        consumption_q: ContinuousAction, lagged_employment: DiscreteState
    ) -> FloatND:
        return jnp.log(consumption_q + lagged_employment * 0.001)

    def dead_utility() -> float:
        return 0.0

    # Assets is used in transition but not in utility
    def next_assets(
        assets: ContinuousState, consumption_q: ContinuousAction
    ) -> ContinuousState:
        return assets - consumption_q

    def next_lagged_employment(employment: DiscreteState) -> DiscreteState:
        return jnp.where(
            employment == EmploymentStatus.employed,
            EmploymentLastPeriod.employed,
            EmploymentLastPeriod.unemployed,
        )

    alive_regime = Regime(
        transition=next_regime,
        functions={"utility": utility, "model_end_age": model_end_age},
        actions={
            "consumption_q": LinSpacedGrid(start=1, stop=10, n_points=5),
            "employment": DiscreteGrid(EmploymentStatus),
        },
        states={
            "assets": LinSpacedGrid(
                start=10, stop=100, n_points=5, transition=next_assets
            ),
            "lagged_employment": DiscreteGrid(
                EmploymentLastPeriod, transition=next_lagged_employment
            ),
        },
    )

    dead_regime = Regime(
        transition=None,
        functions={"utility": dead_utility},
    )

    # This should work but currently raises ValueError
    Model(
        regimes={"alive": alive_regime, "dead": dead_regime},
        ages=AgeGrid(start=59, stop=61, step="Y"),
        regime_id_class=RegimeId,
    )


# ======================================================================================
# Reproducer for GitHub issue #236
# ======================================================================================


def test_state_only_in_transitions_with_terminal_regime():
    """State used only in transitions causes ValueError at terminal-transition period.

    When a state variable appears only in transition functions (not in utility or
    constraints), and the regime transitions to a terminal regime, the Q_and_F
    function's signature at that period does not include the state variable. But
    simulation_spacemap still passes all regime states to vmap_1d, causing
    ``ValueError: list.index(x): x not in list``.

    See: https://github.com/OpenSourceEconomics/pylcm/issues/236
    """

    @categorical
    class RegimeId:
        alive: int
        dead: int

    @categorical
    class TypeVar:
        low: int
        high: int

    def utility(consumption, wealth):
        return jnp.log(consumption) + 0.01 * wealth

    def dead_utility():
        return 0.0

    def next_wealth(wealth, consumption, type_var):
        """type_var affects wealth transition but does NOT appear in utility."""
        return (1 + 0.05 * type_var) * (wealth - consumption)

    def next_regime(age):
        return jnp.where(age >= 2, RegimeId.dead, RegimeId.alive)

    ages = AgeGrid(start=0, stop=3, step="Y")

    alive = Regime(
        functions={"utility": utility},
        states={
            "wealth": LinSpacedGrid(
                start=1, stop=100, n_points=10, transition=next_wealth
            ),
            "type_var": DiscreteGrid(TypeVar),  # Fixed state (no transition)
        },
        actions={
            "consumption": LinSpacedGrid(start=1, stop=50, n_points=10),
        },
        transition=next_regime,
        active=lambda age: age <= 2,
    )

    dead = Regime(transition=None, functions={"utility": dead_utility})

    Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )
