from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_law import bind_regime_law
from _lcm.user_regime_validation import validate_regime_law
from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    LinearAggregator,
    LinearExpectation,
    LinSpacedGrid,
    Model,
    Phased,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidNameError,
    InvalidParamsError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    FunctionName,
    ScalarInt,
    UserFunction,
)
from tests.conftest import bind_laws


def test_regime_invalid_states():
    """Regime rejects non-dict states argument."""
    with pytest.raises(RegimeInitializationError, match="states"):
        UserRegime(
            states="health",  # ty: ignore[invalid-argument-type]
            actions={},
            functions={"utility": lambda: 0},
        )


def test_regime_invalid_actions():
    """Regime rejects non-dict actions argument."""
    with pytest.raises(RegimeInitializationError, match="actions"):
        UserRegime(
            states={},
            actions="exercise",  # ty: ignore[invalid-argument-type]
            functions={"utility": lambda: 0},
        )


def test_regime_invalid_functions():
    """Regime rejects non-dict functions argument."""
    with pytest.raises(RegimeInitializationError, match="functions"):
        UserRegime(
            states={},
            actions={},
            functions="utility",  # ty: ignore[invalid-argument-type]
        )


def test_regime_invalid_functions_values():
    """Regime rejects non-callable function values."""
    with pytest.raises(RegimeInitializationError, match="functions"):
        UserRegime(
            states={},
            actions={},
            functions={"utility": lambda: 0, "function": 0},  # ty: ignore[invalid-argument-type]
        )


def test_regime_invalid_functions_keys():
    """Regime rejects non-string function keys."""
    with pytest.raises(RegimeInitializationError, match="functions"):
        UserRegime(
            states={},
            actions={},
            functions={"utility": lambda: 0, 0: lambda: 0},  # ty: ignore[invalid-argument-type]
        )


def test_regime_invalid_actions_values():
    """Regime rejects non-grid action values."""
    with pytest.raises(RegimeInitializationError, match="actions"):
        UserRegime(
            states={},
            actions={"exercise": 0},  # ty: ignore[invalid-argument-type]
            functions={"utility": lambda: 0},
        )


def test_regime_invalid_states_values():
    """Regime rejects non-grid state values."""
    with pytest.raises(RegimeInitializationError, match="states"):
        UserRegime(
            states={"health": 0},  # ty: ignore[invalid-argument-type]
            actions={},
            functions={"utility": lambda: 0},
        )


def test_regime_invalid_utility():
    """Regime rejects non-callable utility argument."""
    with pytest.raises(RegimeInitializationError, match="functions"):
        UserRegime(
            states={},
            actions={},
            functions={"utility": 0},  # ty: ignore[invalid-argument-type]
        )


def test_regime_overlapping_states_actions(binary_category_class):
    """Regime finalization rejects overlapping state and action names."""
    regime = UserRegime(
        states={
            "health": DiscreteGrid(category_class=binary_category_class),
        },
        state_transitions={"health": fixed_transition("health")},
        actions={"health": DiscreteGrid(category_class=binary_category_class)},
        functions={"utility": lambda: 0},
    )
    with pytest.raises(
        RegimeInitializationError,
        match=r"States and actions cannot have overlapping names.",
    ):
        finalize_regimes(
            user_regimes={"regime": regime},
            laws=bind_laws({"regime": lambda: 0}),
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
        )


def test_regime_transition_must_be_callable():
    """Binding a non-callable regime transition law is rejected."""
    regime = UserRegime(states={}, actions={}, functions={"utility": lambda: 0})
    with pytest.raises(RegimeInitializationError, match="transition"):
        validate_regime_law(regime, law=bind_regime_law(42))  # ty: ignore[invalid-argument-type]


def test_model_requires_terminal_regime(binary_category_class):
    """A model without a terminal regime demands a problem at the last age."""

    @categorical(ordered=False)
    class RegimeId:
        test: ScalarInt

    regime = UserRegime(
        states={
            "health": DiscreteGrid(category_class=binary_category_class),
        },
        state_transitions={"health": lambda health: health},
        actions={},
        functions={"utility": lambda health: health},
    )
    with pytest.raises(ModelInitializationError, match="nonterminal at the last age"):
        Model(
            regimes={"test": regime},
            ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
            regime_id_class=RegimeId,
            initial_nodes={0: "test"},
            edges={"test": {"test": (0, 1)}},
        )


def test_model_keyword_only():
    """Model requires keyword arguments only."""
    # Positional arguments should raise TypeError
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        Model({}, AgeGrid(start=0, inclusive_stop=2, step="Y"))  # ty: ignore[missing-argument,too-many-positional-arguments]


def test_model_accepts_multiple_terminal_regimes(binary_category_class):
    """Model can have multiple terminal regimes."""

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead1: ScalarInt
        dead2: ScalarInt

    alive = UserRegime(
        states={
            "health": DiscreteGrid(category_class=binary_category_class),
        },
        state_transitions={"health": lambda health: health},
        functions={"utility": lambda health: health},
    )
    dead1 = UserRegime(
        states={
            "health": DiscreteGrid(category_class=binary_category_class),
        },
        functions={"utility": lambda health: health * 0},
    )
    dead2 = UserRegime(
        states={
            "health": DiscreteGrid(category_class=binary_category_class),
        },
        functions={"utility": lambda health: health * 0},
    )
    # Should not raise - multiple terminal regimes are allowed
    model = Model(
        regimes={"alive": alive, "dead1": dead1, "dead2": dead2},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
        edges={
            "alive": Transition(
                targets={"dead1": 0, "dead2": 0},
                law=StochasticTransition(func=lambda: jnp.array([0.8, 0.1, 0.1])),
            )
        },
    )
    assert model._regimes is not None


def test_model_regime_id_mapping_created_from_dict_keys(binary_category_class):
    """Model creates regime id mapping from dict keys in order."""

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    alive = UserRegime(
        states={
            "health": DiscreteGrid(category_class=binary_category_class),
        },
        state_transitions={"health": lambda health: health},
        functions={"utility": lambda health: health},
    )
    dead = UserRegime(
        states={
            "health": DiscreteGrid(category_class=binary_category_class),
        },
        functions={"utility": lambda health: health * 0},
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"dead": 0}},
    )
    # regime id should be created from dict keys in order
    assert model.regime_names_to_ids["alive"] == 0
    assert model.regime_names_to_ids["dead"] == 1


def test_model_rejects_a_regime_id_class_whose_codes_are_not_scalar_ints(
    binary_category_class,
):
    """A regime id class whose codes are not `ScalarInt`s is a definition error."""

    @dataclass(frozen=True)
    class RegimeId:
        alive: int = 0
        dead: str = "one"

    alive = UserRegime(
        states={"health": DiscreteGrid(category_class=binary_category_class)},
        state_transitions={"health": lambda health: health},
        functions={"utility": lambda health: health},
    )
    dead = UserRegime(
        states={"health": DiscreteGrid(category_class=binary_category_class)},
        functions={"utility": lambda health: health * 0},
    )

    with pytest.raises(
        ModelInitializationError,
        match=(
            r"regime_id_class is not a valid category class\. Field values of the "
            r"category_class must be `ScalarInt` \(0-d int32 jax scalars\)\. The "
            r"values to the following fields are not: \['alive', 'dead'\]"
        ),
    ):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
            regime_id_class=RegimeId,
            initial_nodes={0: "alive"},
            edges={"alive": {"dead": 0}},
        )


def test_model_regime_name_validation(binary_category_class):
    """Model validates regime names don't contain the separator."""

    @categorical(ordered=False)
    class RegimeId:
        alive__bad: ScalarInt
        dead: ScalarInt

    alive = UserRegime(
        states={
            "health": DiscreteGrid(category_class=binary_category_class),
        },
        state_transitions={"health": lambda health: health},
        functions={"utility": lambda health: health},
    )
    dead = UserRegime(
        states={
            "health": DiscreteGrid(category_class=binary_category_class),
        },
        functions={"utility": lambda health: health * 0},
    )
    # Using separator in regime name should raise error
    with pytest.raises(ModelInitializationError, match="separator character"):
        Model(
            regimes={"alive__bad": alive, "dead": dead},
            ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
            regime_id_class=RegimeId,
            initial_nodes={0: "alive__bad"},
            edges={"alive__bad": {"dead": 0}},
        )


def test_unused_state_raises_error():
    """Model raises error when a state is defined but never used."""

    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    @categorical(ordered=False)
    class UnusedState:
        low: ScalarInt
        medium: ScalarInt
        high: ScalarInt

    # Define a regime where 'unused_state' is not used in any function
    working_life = UserRegime(
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
            ),
            "unused_state": DiscreteGrid(category_class=UnusedState),
        },
        state_transitions={
            "wealth": lambda wealth, consumption: wealth - consumption,
            "unused_state": fixed_transition("unused_state"),
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
    )

    retirement = UserRegime(
        functions={"utility": lambda wealth: wealth * 0.5},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "unused_state": DiscreteGrid(category_class=UnusedState),
        },
    )

    # Should raise error about unused_state
    with pytest.raises(ModelInitializationError, match="unused_state"):
        Model(
            regimes={"working_life": working_life, "retirement": retirement},
            ages=AgeGrid(start=0, inclusive_stop=5, step="Y"),
            regime_id_class=RegimeId,
            initial_nodes={0: "working_life"},
            edges={
                "working_life": Transition(
                    targets={
                        "working_life": (0, 1, 2, 3),
                        "retirement": (0, 1, 2, 3, 4),
                    },
                    law=StochasticTransition(func=lambda: jnp.array([0.9, 0.1])),
                )
            },
        )


def test_unused_action_raises_error():
    """Model raises error when an action is defined but never used."""

    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    @categorical(ordered=False)
    class UnusedAction:
        option_a: ScalarInt
        option_b: ScalarInt

    working_life = UserRegime(
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
            ),
        },
        state_transitions={
            "wealth": lambda wealth, consumption: wealth - consumption,
        },
        actions={
            "consumption": LinSpacedGrid(start=1, stop=50, n_points=10),
            "unused_action": DiscreteGrid(
                category_class=UnusedAction
            ),  # Not used anywhere!
        },
    )

    retirement = UserRegime(
        functions={"utility": lambda wealth: wealth * 0.5},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
    )

    # Should raise error about unused_action
    with pytest.raises(ModelInitializationError, match="unused_action"):
        Model(
            regimes={"working_life": working_life, "retirement": retirement},
            ages=AgeGrid(start=0, inclusive_stop=5, step="Y"),
            regime_id_class=RegimeId,
            initial_nodes={0: "working_life"},
            edges={
                "working_life": Transition(
                    targets={
                        "working_life": (0, 1, 2, 3),
                        "retirement": (0, 1, 2, 3, 4),
                    },
                    law=StochasticTransition(func=lambda: jnp.array([0.9, 0.1])),
                )
            },
        )


def test_constraint_naming_a_transition_output_is_rejected():
    """A constraint is evaluated ahead of the transitions, so it cannot read one.

    `next_assets` is computed after the state-action grid has been scored, so a
    constraint naming it has no value to read. The model says so and points at the
    equivalent statement in this period's variables — `assets - consumption >= 0`
    — rather than binding the name to a parameter.
    """

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    @categorical(ordered=False)
    class EmploymentLastPeriod:
        unemployed: ScalarInt
        employed: ScalarInt

    @categorical(ordered=False)
    class EmploymentStatus:
        not_employed: ScalarInt
        employed: ScalarInt

    def next_regime(*, age: float, model_end_age: int) -> ScalarInt:
        return jnp.where(age == model_end_age, RegimeId.dead, RegimeId.alive)

    def model_end_age(value: int) -> int:
        return value

    def utility(
        *, consumption_q: ContinuousAction, lagged_employment: DiscreteState
    ) -> FloatND:
        return jnp.log(consumption_q + lagged_employment * 0.001)

    def dead_utility() -> float:
        return 0.0

    def next_assets(
        *, assets: ContinuousState, consumption_q: ContinuousAction
    ) -> ContinuousState:
        return assets - consumption_q

    def next_lagged_employment(employment: DiscreteState) -> DiscreteState:
        return jnp.where(
            employment == EmploymentStatus.employed,
            EmploymentLastPeriod.employed,
            EmploymentLastPeriod.unemployed,
        )

    def borrowing_constraint(next_assets: ContinuousState) -> BoolND:
        return next_assets >= 0.0

    alive_regime = UserRegime(
        constraints={"borrowing_constraint": borrowing_constraint},
        functions={"utility": utility, "model_end_age": model_end_age},
        actions={
            "consumption_q": LinSpacedGrid(start=1, stop=10, n_points=5),
            "employment": DiscreteGrid(category_class=EmploymentStatus),
        },
        states={
            "assets": LinSpacedGrid(start=10, stop=100, n_points=5),
            "lagged_employment": DiscreteGrid(category_class=EmploymentLastPeriod),
        },
        state_transitions={
            "assets": next_assets,
            "lagged_employment": next_lagged_employment,
        },
    )

    dead_regime = UserRegime(
        functions={"utility": dead_utility},
    )

    with pytest.raises(InvalidNameError, match="next_assets"):
        Model(
            regimes={"alive": alive_regime, "dead": dead_regime},
            ages=AgeGrid(start=59, inclusive_stop=61, step="Y"),
            regime_id_class=RegimeId,
            initial_nodes={59: "alive"},
            edges={
                "alive": Transition(
                    targets={"alive": 59, "dead": (59, 60)},
                    law=DeterministicTransition(func=next_regime),
                )
            },
        )


def test_state_only_used_in_transitions():
    """A state used only in transitions, not in utility or constraints, builds.

    Such a state must not trip the state-usage walk.
    """

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    @categorical(ordered=False)
    class EmploymentLastPeriod:
        unemployed: ScalarInt
        employed: ScalarInt

    @categorical(ordered=False)
    class EmploymentStatus:
        not_employed: ScalarInt
        employed: ScalarInt

    def next_regime(*, age: float, model_end_age: int) -> ScalarInt:
        return jnp.where(age == model_end_age, RegimeId.dead, RegimeId.alive)

    def model_end_age(value: int) -> int:
        return value

    # Utility does NOT use assets directly
    def utility(
        *, consumption_q: ContinuousAction, lagged_employment: DiscreteState
    ) -> FloatND:
        return jnp.log(consumption_q + lagged_employment * 0.001)

    def dead_utility() -> float:
        return 0.0

    # Assets is used in transition but not in utility
    def next_assets(
        *, assets: ContinuousState, consumption_q: ContinuousAction
    ) -> ContinuousState:
        return assets - consumption_q

    def next_lagged_employment(employment: DiscreteState) -> DiscreteState:
        return jnp.where(
            employment == EmploymentStatus.employed,
            EmploymentLastPeriod.employed,
            EmploymentLastPeriod.unemployed,
        )

    alive_regime = UserRegime(
        functions={"utility": utility, "model_end_age": model_end_age},
        actions={
            "consumption_q": LinSpacedGrid(start=1, stop=10, n_points=5),
            "employment": DiscreteGrid(category_class=EmploymentStatus),
        },
        states={
            "assets": LinSpacedGrid(start=10, stop=100, n_points=5),
            "lagged_employment": DiscreteGrid(category_class=EmploymentLastPeriod),
        },
        state_transitions={
            "assets": next_assets,
            "lagged_employment": next_lagged_employment,
        },
    )

    dead_regime = UserRegime(
        functions={"utility": dead_utility},
    )

    Model(
        regimes={"alive": alive_regime, "dead": dead_regime},
        ages=AgeGrid(start=59, inclusive_stop=61, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={59: "alive"},
        edges={
            "alive": Transition(
                targets={"alive": 59, "dead": (59, 60)},
                law=DeterministicTransition(func=next_regime),
            )
        },
    )


def test_state_only_in_transitions_with_terminal_regime():
    """State used only in transitions causes ValueError at terminal-transition period.

    When a state variable appears only in transition functions (not in utility or
    constraints), and the regime transitions to a terminal regime, the Q_and_F
    function's signature at that period does not include the state variable. But
    simulation_spacemap still passes all regime states to vmap_1d, causing
    `ValueError: list.index(x): x not in list`.

    See: https://github.com/OpenSourceEconomics/pylcm/issues/236
    """

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    @categorical(ordered=False)
    class TypeVar:
        low: ScalarInt
        high: ScalarInt

    def utility(*, consumption, wealth):
        return jnp.log(consumption) + 0.01 * wealth

    def dead_utility():
        return 0.0

    def next_wealth(*, wealth, consumption, type_var: DiscreteState):
        """type_var affects wealth transition but does NOT appear in utility."""
        return (1 + 0.05 * type_var) * (wealth - consumption)

    def next_regime(age):
        return jnp.where(age >= 2, RegimeId.dead, RegimeId.alive)

    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")

    alive = UserRegime(
        functions={"utility": utility},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "type_var": DiscreteGrid(category_class=TypeVar),
        },
        state_transitions={
            "wealth": next_wealth,
            "type_var": fixed_transition("type_var"),
        },
        actions={
            "consumption": LinSpacedGrid(start=1, stop=50, n_points=10),
        },
    )

    dead = UserRegime(functions={"utility": dead_utility})

    Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "alive"},
        edges={
            "alive": Transition(
                targets={"alive": (0, 1), "dead": (0, 1, 2)},
                law=DeterministicTransition(func=next_regime),
            )
        },
    )


@categorical(ordered=False)
class _OutputRegimeId:
    alive: ScalarInt


def _output_utility(wealth: FloatND) -> FloatND:
    return wealth


def _output_report(*, wealth: FloatND, report_scale: float) -> FloatND:
    return wealth * report_scale


def _output_identity(report: FloatND) -> FloatND:
    return report


def _output_from_helper(helper: FloatND) -> FloatND:
    return helper


def _output_model(
    *,
    functions: Mapping[FunctionName, UserFunction | Phased],
    model_output: tuple[FunctionName, ...] = (),
    regime_output: tuple[FunctionName, ...] = (),
) -> Model:
    return Model(
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_OutputRegimeId,
        regimes={
            "alive": UserRegime(
                states={"wealth": LinSpacedGrid(start=1.0, stop=3.0, n_points=3)},
                functions=functions,
                **({"output_only_functions": regime_output} if regime_output else {}),
            )
        },
        initial_nodes={0: "alive"},
        edges={},
        **({"output_only_functions": model_output} if model_output else {}),
    )


def test_unread_functions_name_regime_and_every_function() -> None:
    """Unconsumed functions require an explicit output declaration."""
    with pytest.raises(RegimeInitializationError) as error:
        _output_model(
            functions={
                "utility": _output_utility,
                "report": _output_report,
                "unused": lambda: 1.0,
            }
        )
    message = str(error.value)
    assert all(name in message for name in ("alive", "report", "unused"))
    assert "remove" in message.lower()
    assert "output_only_functions=" in message


@pytest.mark.parametrize("level", ["model", "regime"])
def test_output_only_functions_keep_runtime_params_and_report(level: str) -> None:
    """Output-only targets accept ordinary runtime params and report their values."""
    model = _output_model(
        functions={"utility": _output_utility, "report": _output_report},
        model_output=("report",) if level == "model" else (),
        regime_output=("report",) if level == "regime" else (),
    )
    assert model.user_regimes["alive"].output_only_functions == ("report",)
    assert "report_scale" in model.get_params_template()["alive"]["report"]
    with pytest.raises(InvalidParamsError, match=r"Missing required parameter\(s\)"):
        model.solve(params={}, log_level="off")
    result = model.simulate(
        log_level="off",
        params={"alive": {"report": {"report_scale": 4.0}}},
        initial_conditions={
            "age": jnp.array([0.0, 0.0]),
            "wealth": jnp.array([1.0, 2.0]),
            "regime_id": jnp.array([0, 0]),
        },
    )
    frame = result.to_dataframe(additional_targets=["report", "utility"])
    assert frame["report"].tolist() == [4.0, 8.0]
    assert frame["utility"].tolist() == [1.0, 2.0]
    assert result.to_dataframe(additional_targets="all")["report"].tolist() == [
        4.0,
        8.0,
    ]


def test_read_helper_is_reportable_without_output_only_declaration() -> None:
    """A function consumed by utility remains an available reporting target."""
    model = _output_model(
        functions={
            "utility": _output_identity,
            "report": _output_utility,
        }
    )
    result = model.simulate(
        log_level="off",
        params={},
        initial_conditions={
            "age": jnp.array([0.0]),
            "wealth": jnp.array([2.0]),
            "regime_id": jnp.array([0]),
        },
    )
    assert result.to_dataframe(additional_targets=["report"])["report"].tolist() == [
        2.0
    ]


@pytest.mark.parametrize(
    ("model_output", "regime_output", "message"),
    [
        (("report",), ("report",), "both|model level.*regime level"),
        (("missing",), (), "missing"),
        ((), ("missing",), "missing"),
        ((), ("utility",), "utility.*read|read.*utility"),
        (("utility",), (), "utility.*read|read.*utility"),
    ],
)
def test_output_only_declarations_reject_conflicts_missing_and_read_names(
    *,
    model_output: tuple[FunctionName, ...],
    regime_output: tuple[FunctionName, ...],
    message: str,
) -> None:
    """Output declarations name existing unread functions at exactly one level."""
    with pytest.raises(
        (ModelInitializationError, RegimeInitializationError), match=message
    ):
        _output_model(
            functions={"utility": _output_utility, "report": _output_report},
            model_output=model_output,
            regime_output=regime_output,
        )


def test_output_only_dependency_is_an_ordinary_read() -> None:
    """Output-only functions may use helpers without declaring those helpers."""
    model = _output_model(
        functions={
            "utility": _output_utility,
            "report": _output_from_helper,
            "helper": _output_utility,
        },
        regime_output=("report",),
    )
    assert set(model.user_regimes["alive"].functions) == {"utility", "report", "helper"}


def test_output_only_dependency_cannot_be_declared_for_output() -> None:
    """A helper consumed by a reporting function is already read."""
    with pytest.raises(RegimeInitializationError, match=r"reads.*helper"):
        _output_model(
            functions={
                "utility": _output_utility,
                "report": _output_from_helper,
                "helper": _output_utility,
            },
            regime_output=("report", "helper"),
        )


def test_simulation_phase_function_consumers_count_as_reads() -> None:
    """A helper used only by a simulation-phase utility is read by the model."""
    model = _output_model(
        functions={
            "utility": Phased(solve=_output_utility, simulate=_output_identity),
            "report": _output_utility,
        }
    )
    assert "report" in model.user_regimes["alive"].functions


def test_output_only_reporting_retains_broadcast_inputs_by_reachability() -> None:
    """An explicitly retained reporting target consumes its broadcast inputs."""
    model = Model(
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_OutputRegimeId,
        regimes={"alive": UserRegime(functions={"utility": lambda: 0.0})},
        states={"wealth": LinSpacedGrid(start=1.0, stop=3.0, n_points=3)},
        functions={"report": _output_report},
        output_only_functions=("report",),
        initial_nodes={0: "alive"},
        edges={},
    )
    assert "wealth" in model.user_regimes["alive"].states
    assert "wealth" not in model.get_params_template()["alive"]["report"]


@pytest.mark.parametrize("mask", [False, True])
def test_model_output_only_names_apply_where_defined_and_respect_masks(
    *, mask: bool
) -> None:
    """A shared reporting declaration applies only where the function survives."""

    @categorical(ordered=False)
    class RegimeId:
        alive: ScalarInt
        dead: ScalarInt

    model = Model(
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        regimes={
            "alive": UserRegime(
                functions={
                    "utility": lambda: 0.0,
                    **({} if mask else {"report": lambda: 2.0}),
                }
            ),
            "dead": UserRegime(
                functions={"utility": lambda: 0.0, **({"report": None} if mask else {})}
            ),
        },
        functions={"report": lambda: 2.0} if mask else {},
        output_only_functions=("report",),
        initial_nodes={0: "alive"},
        edges={},
    )
    assert model.user_regimes["alive"].output_only_functions == ("report",)
    assert model.user_regimes["dead"].output_only_functions == ()
    assert "report" not in model.user_regimes["dead"].functions


def _output_positive(report: FloatND) -> BoolND:
    return report > 0.0


@pytest.mark.parametrize("declare_constraint", [False, True])
def test_constraints_read_their_helpers_and_need_no_output_declaration(
    *,
    declare_constraint: bool,
) -> None:
    """Constraints are engine consumers of the functions their predicates read."""

    def build() -> Model:
        return Model(
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=_OutputRegimeId,
            regimes={
                "alive": UserRegime(
                    states={"wealth": LinSpacedGrid(start=1.0, stop=3.0, n_points=3)},
                    functions={"utility": _output_utility, "report": _output_utility},
                    constraints={"allowed": _output_positive},
                    output_only_functions=("allowed",) if declare_constraint else (),
                )
            },
            initial_nodes={0: "alive"},
            edges={},
        )

    if declare_constraint:
        with pytest.raises(RegimeInitializationError, match=r"reads.*allowed"):
            build()
    else:
        assert "report" in build().user_regimes["alive"].functions
