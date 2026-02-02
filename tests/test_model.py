import jax.numpy as jnp
import pytest

import lcm
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.grids import ShockGrid


def test_regime_invalid_states():
    """Regime rejects non-dict states argument."""
    with pytest.raises(RegimeInitializationError, match="states must be a dictionary"):
        Regime(
            states="health",  # ty: ignore[invalid-argument-type]
            actions={},
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=lambda age: age < 5,
        )


def test_regime_invalid_actions():
    """Regime rejects non-dict actions argument."""
    with pytest.raises(RegimeInitializationError, match="actions must be a dictionary"):
        Regime(
            states={},
            actions="exercise",  # ty: ignore[invalid-argument-type]
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=lambda age: age < 5,
        )


def test_regime_invalid_functions():
    """Regime rejects non-dict functions argument."""
    with pytest.raises(
        RegimeInitializationError, match="functions must each be a dictionary"
    ):
        Regime(
            states={},
            actions={},
            transitions={"next_health": lambda: 0},
            utility=lambda: 0,
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
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            functions={"function": 0},  # ty: ignore[invalid-argument-type]
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
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            functions={0: lambda: 0},  # ty: ignore[invalid-argument-type]
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
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
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
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=lambda age: age < 5,
        )


def test_regime_missing_next_func(binary_category_class):
    """Regime rejects states without corresponding transition functions."""
    with pytest.raises(
        RegimeInitializationError,
        match=r"Each state must have a corresponding transition function.",
    ):
        Regime(
            states={
                "health": DiscreteGrid(binary_category_class),
                "wealth": DiscreteGrid(binary_category_class),
            },
            actions={"exercise": DiscreteGrid(binary_category_class)},
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=lambda age: age < 5,
        )


def test_regime_invalid_utility():
    """Regime rejects non-callable utility argument."""
    with pytest.raises(
        RegimeInitializationError,
        match=(r"utility must be a callable."),
    ):
        Regime(
            states={},
            actions={},
            functions={},
            utility=0,  # ty: ignore[invalid-argument-type]
            transitions={"next_health": lambda: 0},
            active=lambda age: age < 5,
        )


def test_regime_invalid_transition_names():
    """Regime rejects transition names not starting with 'next_'."""
    with pytest.raises(
        RegimeInitializationError,
        match=(r"Each transitions name must start with 'next_'."),
    ):
        Regime(
            states={},
            actions={},
            functions={},
            utility=lambda: 0,
            transitions={"invalid_name": lambda: 0},
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
            utility=lambda: 0,
            transitions={"next_health": lambda: 0},
            active=lambda age: age < 5,
        )


def test_model_requires_terminal_regime(binary_category_class):
    """Model must have at least one terminal regime."""

    @categorical
    class RegimeId:
        test: int

    regime = Regime(
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([1.0])),
        },
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
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health * 0,
        terminal=True,
        active=lambda age: age >= 1,
    )
    with pytest.raises(ModelInitializationError, match="at least one non-terminal"):
        Model(
            regimes={"dead": dead},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=RegimeId,
        )


def test_multi_regime_without_next_regime_raises(binary_category_class):
    """Multi-regime models must have next_regime in each regime."""

    @categorical
    class RegimeId:
        regime1: int
        regime2: int

    regime1 = Regime(
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            # Missing next_regime
        },
        active=lambda age: age < 1,
    )
    regime2 = Regime(
        states={"health": DiscreteGrid(binary_category_class)},
        actions={},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
        active=lambda age: age < 1,
    )
    with pytest.raises(ModelInitializationError, match="next_regime"):
        Model(
            regimes={"regime1": regime1, "regime2": regime2},
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
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.8, 0.1, 0.1])),
        },
        active=lambda age: age < 1,
    )
    dead1 = Regime(
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health * 0,
        terminal=True,
        active=lambda age: age >= 1,
    )
    dead2 = Regime(
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health * 0,
        terminal=True,
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
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
        active=lambda age: age < 1,
    )
    dead = Regime(
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health * 0,
        terminal=True,
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
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
        active=lambda age: age < 1,
    )
    dead = Regime(
        states={"health": DiscreteGrid(binary_category_class)},
        utility=lambda health: health * 0,
        terminal=True,
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
        utility=lambda wealth, consumption: jnp.log(consumption) + wealth * 0.001,
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "unused_state": DiscreteGrid(UnusedState),  # Not used anywhere!
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transitions={
            "next_wealth": lambda wealth, consumption: wealth - consumption,
            "next_unused_state": lambda unused_state: unused_state,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.9, 0.1])),
        },
        active=lambda age: age < 5,
    )

    retired = Regime(
        utility=lambda wealth: wealth * 0.5,
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "unused_state": DiscreteGrid(UnusedState),
        },
        terminal=True,
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
        utility=lambda wealth, consumption: jnp.log(consumption) + wealth * 0.001,
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        actions={
            "consumption": LinSpacedGrid(start=1, stop=50, n_points=10),
            "unused_action": DiscreteGrid(UnusedAction),  # Not used anywhere!
        },
        transitions={
            "next_wealth": lambda wealth, consumption: wealth - consumption,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.9, 0.1])),
        },
        active=lambda age: age < 5,
    )

    retired = Regime(
        utility=lambda wealth: wealth * 0.5,
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        terminal=True,
        active=lambda age: age >= 5,
    )

    # Should raise error about unused_action
    with pytest.raises(ModelInitializationError, match="unused_action"):
        Model(
            regimes={"working": working, "retired": retired},
            ages=AgeGrid(start=0, stop=5, step="Y"),
            regime_id_class=RegimeId,
        )


def test_missing_transition_for_other_regime_state_raises_error():
    """Non-terminal regimes must have transitions for all states across all regimes."""

    @categorical
    class RegimeId:
        working: int
        retired: int

    # Working regime only has 'wealth', but retired has 'wealth' AND 'pension'.
    # Working must define next_pension since it can transition to retired.
    working = Regime(
        utility=lambda wealth: wealth,
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        transitions={
            "next_wealth": lambda wealth: wealth,
            # Missing next_pension!
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
    )

    retired = Regime(
        utility=lambda wealth, pension: wealth + pension,
        states={
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=5),
            "pension": LinSpacedGrid(start=0, stop=5, n_points=3),
        },
        terminal=True,
    )

    with pytest.raises(ModelInitializationError, match="next_pension"):
        Model(
            regimes={"working": working, "retired": retired},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=RegimeId,
        )


def test_fixed_params_validation():
    """Model validates that fixed params exist when are shocks used."""

    @categorical
    class RegimeId:
        alive: int
        dead: int

    alive = Regime(
        states={"health": ShockGrid(distribution_type="tauchen", n_points=2)},
        utility=lambda health: health,
        transitions={
            "next_health": lambda health: health,
            "next_regime": lcm.mark.stochastic(lambda: jnp.array([0.5, 0.5])),
        },
        active=lambda age: age < 1,
    )
    dead = Regime(
        utility=lambda: 0,
        terminal=True,
        active=lambda age: age >= 1,
    )

    # Using separator in regime name should raise error
    with pytest.raises(ModelInitializationError, match="is missing fixed params"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=RegimeId,
            fixed_params={},
        )
