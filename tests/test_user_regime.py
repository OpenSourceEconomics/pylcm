"""Test Regime class validation."""

import inspect
import re
from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
import pytest
from dags import rename_arguments
from dags.tree import QNAME_DELIMITER

from _lcm.grids import IrregSpacedGrid
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.transitions import (
    _IdentityTransition,
    collect_state_transitions,
)
from _lcm.regime_law import bind_regime_law
from _lcm.user_regime_validation import validate_regime_law
from _lcm.utils.error_messages import path_segment_name_errors
from lcm import (
    AgeRange,
    CollectiveUtility,
    DeterministicTransition,
    DiscreteGrid,
    LinearAggregator,
    LinearExpectation,
    LinSpacedGrid,
    Model,
    Phased,
    ProjectedRegimeValue,
    StakeholderRoute,
    Transition,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidNameError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.regime import Regime as UserRegime
from lcm.regime import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


def utility(consumption):
    return consumption


def next_wealth(*, wealth, consumption):
    return wealth - consumption


WEALTH_GRID = LinSpacedGrid(start=1, stop=10, n_points=5)
CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=5, n_points=5)


def test_regime_name_does_not_contain_separator():
    """Regime name validation happens at Model level, not Regime level."""

    @categorical(ordered=False)
    class RegimeId:
        work__test: ScalarInt  # Contains separator — validated at Model level
        dead: ScalarInt

    working = UserRegime(
        functions={"utility": utility},
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": fixed_transition("wealth")},
    )
    dead = UserRegime(
        functions={"utility": lambda: 0},
    )
    ages = AgeGrid(start=0, inclusive_stop=5, step="Y")

    # Regime name containing separator should raise at Model creation
    with pytest.raises(ModelInitializationError, match=QNAME_DELIMITER):
        Model(
            regimes={f"work{QNAME_DELIMITER}test": working, "dead": dead},
            ages=ages,
            regime_id_class=RegimeId,
            initial_nodes={ages.exact_values[0]: "work__test"},
            edges={f"work{QNAME_DELIMITER}test": {"dead": AgeRange(exclusive_stop=5)}},
        )


def test_function_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=QNAME_DELIMITER):
        UserRegime(
            states={"wealth": WEALTH_GRID},
            actions={f"consumption{QNAME_DELIMITER}action": CONSUMPTION_GRID},
            functions={"utility": utility, f"helper{QNAME_DELIMITER}func": lambda: 1},
            state_transitions={"wealth": fixed_transition("wealth")},
        )


def test_state_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=QNAME_DELIMITER):
        UserRegime(
            functions={"utility": utility},
            states={f"my{QNAME_DELIMITER}wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            state_transitions={
                f"my{QNAME_DELIMITER}wealth": fixed_transition(
                    f"my{QNAME_DELIMITER}wealth"
                )
            },
        )


INVALID_SEGMENT_NAMES = [
    pytest.param("my__name", id="separator"),
    pytest.param("_name", id="leading-underscore"),
    pytest.param("name_", id="trailing-underscore"),
]


def _invalid_names_match(*, kind: str, name: str) -> str:
    """Pattern of the error naming `name` as an invalid name of this kind."""
    return rf"{kind} names cannot contain.*{re.escape(repr(name))}"


@pytest.mark.parametrize("name", INVALID_SEGMENT_NAMES)
def test_regime_name_must_be_a_valid_path_segment(name):
    """A regime name with `__` or a leading/trailing `_` is rejected by `Model`."""
    regime_id = categorical(ordered=False)(
        type("RegimeId", (), {"__annotations__": {name: ScalarInt, "dead": ScalarInt}})
    )
    working = UserRegime(
        functions={"utility": utility},
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": fixed_transition("wealth")},
    )
    dead = UserRegime(functions={"utility": lambda: 0})
    ages = AgeGrid(start=0, inclusive_stop=5, step="Y")
    with pytest.raises(
        ModelInitializationError, match=_invalid_names_match(kind="Regime", name=name)
    ):
        Model(
            regimes={name: working, "dead": dead},
            ages=ages,
            regime_id_class=regime_id,
            initial_nodes={ages.exact_values[0]: name},
            edges={name: {"dead": AgeRange(exclusive_stop=5)}},
        )


@pytest.mark.parametrize("name", INVALID_SEGMENT_NAMES)
def test_state_name_must_be_a_valid_path_segment(name):
    """A state name with `__` or a leading/trailing `_` is rejected by `Regime`."""
    with pytest.raises(
        RegimeInitializationError,
        match=_invalid_names_match(kind="State and action", name=name),
    ):
        UserRegime(
            functions={"utility": utility},
            states={name: WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            state_transitions={name: fixed_transition(name)},
        )


@pytest.mark.parametrize("name", INVALID_SEGMENT_NAMES)
def test_action_name_must_be_a_valid_path_segment(name):
    """An action name with `__` or a leading/trailing `_` is rejected by `Regime`."""
    with pytest.raises(
        RegimeInitializationError,
        match=_invalid_names_match(kind="State and action", name=name),
    ):
        UserRegime(
            functions={"utility": lambda wealth: wealth},
            states={"wealth": WEALTH_GRID},
            actions={name: CONSUMPTION_GRID},
            state_transitions={"wealth": fixed_transition("wealth")},
        )


@pytest.mark.parametrize("name", INVALID_SEGMENT_NAMES)
def test_function_name_must_be_a_valid_path_segment(name):
    """A function name with `__` or a leading/trailing `_` is rejected by `Regime`."""
    with pytest.raises(
        RegimeInitializationError,
        match=_invalid_names_match(kind="Function and constraint", name=name),
    ):
        UserRegime(
            functions={"utility": utility, name: lambda: 1.0},
            states={"wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            state_transitions={"wealth": fixed_transition("wealth")},
        )


@pytest.mark.parametrize("name", INVALID_SEGMENT_NAMES)
def test_constraint_name_must_be_a_valid_path_segment(name):
    """A constraint name with `__` or a leading/trailing `_` is rejected by `Regime`."""
    with pytest.raises(
        RegimeInitializationError,
        match=_invalid_names_match(kind="Function and constraint", name=name),
    ):
        UserRegime(
            functions={"utility": utility},
            constraints={name: lambda consumption: consumption > 0},
            states={"wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            state_transitions={"wealth": fixed_transition("wealth")},
        )


@pytest.mark.parametrize("name", INVALID_SEGMENT_NAMES)
def test_stakeholder_name_must_be_a_valid_path_segment(name):
    """A stakeholder with `__` or a leading/trailing `_` is rejected by `Regime`."""
    with pytest.raises(
        RegimeInitializationError,
        match=_invalid_names_match(kind="Stakeholder", name=name),
    ):
        UserRegime(
            functions={
                "utility": CollectiveUtility(
                    utilities={name: utility, "partner": utility}
                )
            },
            states={"wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
        )


@pytest.mark.parametrize("name", INVALID_SEGMENT_NAMES)
def test_route_key_must_be_a_valid_path_segment(name):
    """A route key with `__` or a leading/trailing `_` is rejected by `Model`."""
    with pytest.raises(
        RegimeInitializationError,
        match=_invalid_names_match(kind="route", name=name),
    ):
        _gated_edge_model(route_key=name)


@pytest.mark.parametrize("name", INVALID_SEGMENT_NAMES)
def test_gate_reference_key_must_be_a_valid_path_segment(name):
    """A gate-reference key with `__` or a leading/trailing `_` is rejected."""
    with pytest.raises(
        RegimeInitializationError,
        match=_invalid_names_match(kind="gate-reference", name=name),
    ):
        _gated_edge_model(gate_reference_key=name)


@pytest.mark.parametrize(
    "callable_kind",
    [
        "utility",
        "regime_transition_law",
        "probability",
        "gate",
        "gate_reference_projection",
        "fallback_projection",
    ],
)
@pytest.mark.parametrize("name", INVALID_SEGMENT_NAMES)
def test_parameter_argument_name_must_be_a_valid_path_segment(
    *, name: str, callable_kind: str
) -> None:
    """A model function's parameter named with `__` or `_` at an end is rejected.

    The argument becomes a parameter name, so `Model` refuses it whether it sits
    on a regime function or on a callable declared in `edges`.
    """
    func, build_model = _PARAMETER_CARRIERS[callable_kind]
    renamed = rename_arguments(func, mapper={"scale": name})
    with pytest.raises(
        InvalidNameError, match=_invalid_names_match(kind="argument", name=name)
    ):
        build_model(renamed)


@pytest.mark.parametrize(
    "name",
    [
        "probability",
        "gate",
        "gate_references",
        "routes",
        "fallback",
        "solve",
        "simulate",
    ],
)
def test_entry_names_of_edge_declarations_are_valid_path_segments(name):
    """The field names pylcm puts into a parameter path obey the user naming rule."""
    assert path_segment_name_errors(kind="Entry", names=[name]) == []


@categorical(ordered=False)
class _GatedRegimeId:
    source: ScalarInt
    target: ScalarInt
    outside: ScalarInt


_WAGE_GRID = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)


def _wage_utility(*, wage: ContinuousState, scale: float) -> FloatND:
    return scale * wage


def _probability_one(*, age: float, scale: float) -> FloatND:
    return jnp.ones_like(age, dtype=float) * scale


def _gate_open(*, V_target: FloatND, scale: float) -> BoolND:
    return V_target >= scale


def _project_wage(*, wage: ContinuousState, scale: float) -> ContinuousState:
    return scale * wage


def _gated_edge_model(
    *,
    route_key: str = "only",
    gate_reference_key: str = "outside_value",
    utility: Callable = _wage_utility,
    probability: Callable = _probability_one,
    gate: Callable = _gate_open,
    gate_reference_projection: Callable = _project_wage,
    fallback_projection: Callable = _project_wage,
) -> Model:
    """Build a singleton source whose edge into `target` is gated."""
    source = UserRegime(
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        functions={"utility": utility},
    )
    target = UserRegime(
        states={"wage": _WAGE_GRID}, functions={"utility": _wage_utility}
    )
    outside = UserRegime(
        states={"wage": _WAGE_GRID}, functions={"utility": _wage_utility}
    )
    return Model(
        regimes={"source": source, "target": target, "outside": outside},
        edges={
            "source": Transition(
                targets={"target": 0, "outside": 0},
                law={
                    "target": ValueDependentTransition(
                        probability=StochasticTransition(func=probability),
                        gate=gate,
                        gate_references={
                            gate_reference_key: ProjectedRegimeValue(
                                regime="outside",
                                projection={"wage": gate_reference_projection},
                            )
                        },
                        routes={
                            route_key: StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="outside",
                                    projection={"wage": fallback_projection},
                                )
                            )
                        },
                    )
                },
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_GatedRegimeId,
        initial_nodes={0: "source"},
    )


@categorical(ordered=False)
class _LawRegimeId:
    working: ScalarInt
    retired: ScalarInt


def _next_regime(*, age: float, scale: float) -> ScalarInt:
    return jnp.where(age >= scale, _LawRegimeId.retired, _LawRegimeId.working)


def _law_model(*, law: Callable) -> Model:
    """Build a working regime whose regime-transition law is `law`."""
    working = UserRegime(
        functions={"utility": utility},
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": next_wealth},
    )
    retired = UserRegime(functions={"utility": lambda wealth: wealth})
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    return Model(
        regimes={"working": working, "retired": retired},
        edges={
            "working": Transition(
                targets={"working": (0, 1), "retired": (0, 1, 2)},
                law=DeterministicTransition(func=law),
            )
        },
        ages=ages,
        regime_id_class=_LawRegimeId,
        initial_nodes={0: "working"},
    )


_PARAMETER_CARRIERS: dict[str, tuple[Callable, Callable[[Callable], Model]]] = {
    "utility": (_wage_utility, lambda func: _gated_edge_model(utility=func)),
    "regime_transition_law": (_next_regime, lambda func: _law_model(law=func)),
    "probability": (
        _probability_one,
        lambda func: _gated_edge_model(probability=func),
    ),
    "gate": (_gate_open, lambda func: _gated_edge_model(gate=func)),
    "gate_reference_projection": (
        _project_wage,
        lambda func: _gated_edge_model(gate_reference_projection=func),
    ),
    "fallback_projection": (
        _project_wage,
        lambda func: _gated_edge_model(fallback_projection=func),
    ),
}


def _aggregator(*, utility, CE, discount_factor):
    return utility + discount_factor * CE


def test_phased_aggregator_missing_a_phase_is_rejected_once_the_regime_has_a_law():
    """A non-terminal regime needs a Koopmans aggregator in both phases.

    Whether the regime is terminal comes from its law, so the regime itself
    constructs and the check fires when the law is bound.
    """
    regime = UserRegime(
        functions={"utility": utility},
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": fixed_transition("wealth")},
        koopmans_aggregator=Phased(solve=None, simulate=_aggregator),
    )
    with pytest.raises(
        RegimeInitializationError,
        match=r"`koopmans_aggregator` is `Phased\(\.\.\.\)` with `solve=None`",
    ):
        validate_regime_law(regime, law=bind_regime_law(lambda: 0))


@pytest.mark.parametrize(
    ("regime_kwargs", "match"),
    [
        (
            {"functions": {"utility": utility, "next_helper": lambda: 1}},
            r"must not start with 'next_'.*\['next_helper'\]",
        ),
        (
            {
                "functions": {
                    "utility": utility,
                    "helper": Phased(solve=lambda: 1, simulate=3),
                },
            },
            r"functions\['helper'\] simulate variant must be a callable, got 3",
        ),
        (
            {
                "functions": {"utility": utility},
                "states": {"wealth": WEALTH_GRID},
                "state_transitions": {"wealth": fixed_transition("savings")},
            },
            r"`fixed_transition\('savings'\)` is assigned to state 'wealth'",
        ),
    ],
    ids=[
        "next-prefixed function",
        "non-callable phase variant",
        "mismatched fixed transition",
    ],
)
def test_regime_local_declaration_errors_raise_at_construction(*, regime_kwargs, match):
    """Errors that need no law are raised when the regime is constructed."""
    with pytest.raises(RegimeInitializationError, match=match):
        UserRegime(**regime_kwargs)


def test_terminal_regime_creation():
    """A regime bound to no outgoing law is terminal and can have states and utility."""
    regime = UserRegime(
        functions={"utility": lambda wealth: wealth * 0.5},
        states={"wealth": WEALTH_GRID},
    )
    law = bind_regime_law(None)
    validate_regime_law(regime, law=law)
    assert law.terminal is True


def test_terminal_regime_with_actions():
    """Terminal regime can have actions for final decisions."""
    regime = UserRegime(
        functions={"utility": lambda wealth, bequest_share: wealth * bequest_share},
        states={"wealth": WEALTH_GRID},
        actions={"bequest_share": LinSpacedGrid(start=0, stop=1, n_points=11)},
    )
    law = bind_regime_law(None)
    validate_regime_law(regime, law=law)
    assert law.terminal is True
    assert "bequest_share" in regime.actions


def test_non_terminal_regime_has_transition():
    """A regime bound to a transition law is non-terminal."""
    regime = UserRegime(
        functions={"utility": utility},
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": fixed_transition("wealth")},
    )
    law = bind_regime_law(next_wealth)
    validate_regime_law(regime, law=law)
    assert law.terminal is False


def test_terminal_regime_can_be_created_without_states():
    """Terminal regime can be created without states (e.g., death state)."""
    regime = UserRegime(
        functions={"utility": lambda: 0},
        states={},
    )
    law = bind_regime_law(None)
    validate_regime_law(regime, law=law)
    assert law.terminal is True
    assert regime.states == {}


def test_regime_has_no_activity_argument():
    """Where a regime is solved comes only from the model's edges."""
    with pytest.raises(TypeError, match="active"):
        UserRegime(
            functions={"utility": utility},
            active=lambda age: age < 5,  # ty: ignore[unknown-argument]
        )


# keyword-only-exempt: primary-argument=regime
def _finalize(regime: UserRegime, *, law: object) -> UserRegime:
    """Run the completeness validation the model applies to a regime and its law."""
    return finalize_regimes(
        user_regimes={"regime": regime},
        laws={"regime": bind_regime_law(law)},
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )["regime"]


def test_regime_requires_utility_in_functions():
    """Regime must have 'utility' in the functions dict."""
    regime = UserRegime(
        functions={"helper": lambda: 1},
        states={"wealth": WEALTH_GRID},
    )
    with pytest.raises(RegimeInitializationError, match=r"utility.*must be provided"):
        _finalize(regime, law=None)


def test_markov_transition_rejects_non_callable():
    with pytest.raises(RegimeInitializationError, match="func"):
        StochasticTransition(func=42)  # ty: ignore[invalid-argument-type]


def test_identity_transition_call():
    """Identity transition returns the state value unchanged."""
    identity = _IdentityTransition(state_name="wealth", annotation=ContinuousState)
    result = identity(wealth=jnp.array(42.0))
    assert result == jnp.array(42.0)


def test_identity_transition_discrete():
    """Identity transition works for discrete states."""
    identity = _IdentityTransition(state_name="education", annotation=DiscreteState)
    result = identity(education=jnp.array(1, dtype=jnp.int32))
    assert result == jnp.array(1, dtype=jnp.int32)


def test_identity_transition_name():
    """Identity transition has the correct __name__."""
    identity = _IdentityTransition(state_name="wealth", annotation=ContinuousState)
    assert identity.__name__ == "next_wealth"


def test_identity_transition_signature():
    """Identity transition has a proper signature with annotation."""
    identity = _IdentityTransition(state_name="wealth", annotation=ContinuousState)
    sig = inspect.signature(identity)
    assert list(sig.parameters) == ["wealth"]
    assert sig.parameters["wealth"].annotation is ContinuousState
    assert sig.return_annotation is ContinuousState


def test_identity_transition_annotations():
    """Identity transition exposes __annotations__ for dags discovery."""
    identity = _IdentityTransition(state_name="education", annotation=DiscreteState)
    assert identity.__annotations__ == {
        "education": DiscreteState,
        "return": DiscreteState,
    }


def test_identity_transition_is_auto_identity():
    """Identity transition is flagged as auto-generated."""
    identity = _IdentityTransition(state_name="x", annotation=ContinuousState)
    assert identity._is_auto_identity is True


def test_get_all_functions_includes_identity_for_fixed_discrete_state():
    """Fixed discrete states get identity transitions with DiscreteState annotation."""

    @categorical(ordered=False)
    class Edu:
        low: ScalarInt
        high: ScalarInt

    regime = UserRegime(
        functions={"utility": lambda education: education},
        states={"education": DiscreteGrid(category_class=Edu)},
        state_transitions={"education": fixed_transition("education")},
    )
    all_funcs = regime.get_all_functions(law=bind_regime_law(lambda: 0))
    identity_func = all_funcs["next_education"]
    assert isinstance(identity_func, _IdentityTransition)
    assert identity_func.__annotations__["education"] is DiscreteState
    assert identity_func.__annotations__["return"] is DiscreteState


def test_get_all_functions_includes_identity_for_fixed_continuous_state():
    """Fixed continuous states get identity transitions with correct annotation."""
    regime = UserRegime(
        functions={"utility": lambda wealth: wealth},
        states={"wealth": LinSpacedGrid(start=0, stop=10, n_points=5)},
        state_transitions={"wealth": fixed_transition("wealth")},
    )
    all_funcs = regime.get_all_functions(law=bind_regime_law(lambda: 0))
    identity_func = all_funcs["next_wealth"]
    assert isinstance(identity_func, _IdentityTransition)
    assert identity_func.__annotations__["wealth"] is ContinuousState
    assert identity_func.__annotations__["return"] is ContinuousState


def test_state_grid_without_explicit_transition_raises():
    """Non-terminal regime with a state missing from state_transitions is rejected."""
    regime = UserRegime(
        functions={"utility": utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        actions={"consumption": CONSUMPTION_GRID},
    )
    with pytest.raises(
        RegimeInitializationError, match="must have an entry in state_transitions"
    ):
        _finalize(regime, law=lambda: 0)


def test_state_grid_with_fixed_transition_is_accepted():
    """A fixed state declared via `fixed_transition` is valid."""
    regime = UserRegime(
        functions={"utility": utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": fixed_transition("wealth")},
    )
    validate_regime_law(regime, law=bind_regime_law(lambda: 0))
    assert "wealth" in regime.states


def test_state_grid_with_transition_callable_is_accepted():
    """State with a transition function in state_transitions is valid."""
    regime = UserRegime(
        functions={"utility": utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": next_wealth},
    )
    validate_regime_law(regime, law=bind_regime_law(lambda: 0))
    assert "wealth" in regime.states


def test_action_grid_without_transition_is_accepted():
    """Action grid with default UNSET transition is valid."""
    regime = UserRegime(
        functions={"utility": utility},
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": fixed_transition("wealth")},
    )
    validate_regime_law(regime, law=bind_regime_law(lambda: 0))
    assert "consumption" in regime.actions


@pytest.mark.parametrize(
    "grid_cls",
    [LinSpacedGrid, IrregSpacedGrid],
    ids=["LinSpacedGrid", "IrregSpacedGrid"],
)
def test_state_grid_unset_error_with_different_grid_types(grid_cls):
    """Missing state_transitions entry error works for various grid types."""
    if grid_cls is LinSpacedGrid:
        grid = LinSpacedGrid(start=1, stop=10, n_points=5)
    else:
        grid = IrregSpacedGrid(points=(1.0, 5.0, 10.0))

    regime = UserRegime(
        functions={"utility": utility},
        states={"wealth": grid},
    )
    with pytest.raises(
        RegimeInitializationError, match="must have an entry in state_transitions"
    ):
        _finalize(regime, law=lambda: 0)


def test_discrete_state_grid_without_explicit_transition_raises():
    """Discrete state grid missing from state_transitions is rejected."""

    @categorical(ordered=False)
    class Status:
        low: ScalarInt
        high: ScalarInt

    regime = UserRegime(
        functions={"utility": lambda status: status},
        states={"status": DiscreteGrid(category_class=Status)},
    )
    with pytest.raises(
        RegimeInitializationError, match="must have an entry in state_transitions"
    ):
        _finalize(regime, law=lambda: 0)


def test_collect_state_transitions_missing_state_raises():
    """collect_state_transitions raises RegimeInitializationError for missing state."""

    states = MappingProxyType({"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)})
    with pytest.raises(
        RegimeInitializationError, match="has no entry in state_transitions"
    ):
        collect_state_transitions(states=states, state_transitions={})


def test_regime_with_fixed_states_only():
    """A regime whose every state is fixed (all `fixed_transition` laws) builds."""

    @categorical(ordered=False)
    class FixedRegimeId:
        working_life: ScalarInt
        dead: ScalarInt

    def fixed_utility(
        *, consumption: ContinuousAction, wealth: ContinuousState
    ) -> FloatND:
        return jnp.log(consumption) + 0.01 * wealth

    def fixed_borrowing(
        *, consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    def fixed_next_regime(*, age: float, final_age_alive: float) -> ScalarInt:
        dead = FixedRegimeId.dead
        working = FixedRegimeId.working_life
        return jnp.where(age >= final_age_alive, dead, working)

    final_age = 1

    working_regime = UserRegime(
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=20)},
        states={
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=15),
        },
        constraints={"borrowing": fixed_borrowing},
        functions={"utility": fixed_utility},
        state_transitions={"wealth": fixed_transition("wealth")},
    )
    dead_regime = UserRegime(
        functions={"utility": lambda: 0.0},
    )
    model = Model(
        regimes={"working_life": working_regime, "dead": dead_regime},
        ages=AgeGrid(start=0, inclusive_stop=final_age + 1, step="Y"),
        regime_id_class=FixedRegimeId,
        initial_nodes={0: "working_life"},
        edges={
            "working_life": Transition(
                targets={
                    "working_life": AgeRange(exclusive_stop=final_age),
                    "dead": AgeRange(exclusive_stop=final_age + 1),
                },
                law=DeterministicTransition(func=fixed_next_regime),
            )
        },
    )
    V = model.solve(
        log_level="debug",
        params={"discount_factor": 0.95, "final_age_alive": final_age},
    ).values
    assert all(
        jnp.all(jnp.isfinite(V[p]["working_life"])) for p in V if "working_life" in V[p]
    )
