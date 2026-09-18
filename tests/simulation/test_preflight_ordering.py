"""Independent legacy-order controls for the two-summary preflight change.

These use real canonical validators and known invalid inputs. No packed flags or
future implementation helper serves as the expected-result oracle.
"""

import dataclasses
import logging
from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
import pytest

import _lcm.simulation.initial_conditions as initial_module
from _lcm.params.processing import process_params
from _lcm.simulation.initial_conditions import validate_simulation_inputs
from _lcm.transition_checks import validate_transitions
from _lcm.typing import (
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    InitialConditions,
)
from _lcm.utils.logging import LogLevel, get_logger
from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidRegimeTransitionProbabilitiesError,
    InvalidStateTransitionProbabilitiesError,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.simulation.test_initial_conditions import (
    _make_asymmetric_state_model,
    _make_constrained_asymmetric_model,
)
from tests.test_transition_checks import _model_with_state_probs


def _validate(*, model: Model, initial: InitialConditions) -> None:
    """Reach the same canonical initial validator with concrete processed params."""
    validate_simulation_inputs(
        initial_conditions=initial,
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        flat_params=process_params(
            params={"discount_factor": 0.95}, params_template=model._params_template
        ),
        ages=model.ages,
        logger=get_logger(log_level="debug"),
    )


@pytest.mark.parametrize(
    ("regime_id", "age", "health", "missing", "expected"),
    [
        (999, 99.0, 999, True, "Invalid regime IDs [999]. Valid IDs: [0, 1]"),
        (0, 99.0, 999, True, "Missing model states: ['wealth']."),
        (0, 99.0, 999, False, "Invalid age values [99.0]"),
        (1, 0.0, 999, False, "not active"),
        (0, 0.0, 999, False, "Invalid values [999] for discrete state 'health'"),
    ],
)
def test_earlier_initial_errors_prevent_every_feasibility_build(
    *,
    monkeypatch: pytest.MonkeyPatch,
    regime_id: int,
    age: float,
    health: int,
    missing: bool,
    expected: str,
) -> None:
    """Invalid IDs, names, ages and codes retain their established precedence."""
    model = _make_asymmetric_state_model()
    initial = {
        "regime_id": jnp.array([regime_id], dtype=jnp.int32),
        "age": jnp.array([age]),
        "wealth": jnp.array([40.0]),
        "health": jnp.array([health], dtype=jnp.int32),
    }
    if missing:
        del initial["wealth"]

    def forbidden(**_kwargs: object) -> None:
        raise AssertionError("A feasibility builder was reached before validation.")

    monkeypatch.setattr(initial_module, "_get_feasibility", forbidden)
    with pytest.raises(InvalidInitialConditionsError) as caught:
        _validate(model=model, initial=initial)
    assert expected in str(caught.value)


def test_name_errors_and_lengths_are_reported_in_the_existing_order() -> None:
    """One initial exception retains all structural messages before age checks."""
    model = _make_asymmetric_state_model()
    with pytest.raises(InvalidInitialConditionsError) as caught:
        _validate(
            model=model,
            initial={
                "regime_id": jnp.array([0, 0], dtype=jnp.int32),
                "age": jnp.array([99.0, 99.0]),
                "health": jnp.array([999], dtype=jnp.int32),
                "extra": jnp.array([0.0]),
            },
        )
    message = str(caught.value)
    fragments = (
        "Missing model states: ['wealth'].",
        "Unknown initial states: ['extra'].",
        "Got lengths: {'age': 2, 'health': 1, 'extra': 1}",
    )
    offsets = tuple(message.index(fragment) for fragment in fragments)
    assert offsets == tuple(sorted(offsets))
    assert "Invalid age values" not in message
    assert "Invalid values" not in message


def test_unused_regime_feasibility_is_neither_built_nor_traced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unused action regime is skipped before even composing its constraints."""
    model = _make_constrained_asymmetric_model()
    unused = model._regimes["alive"].simulation.constraints
    original = initial_module._get_feasibility
    visited = []

    def observe(
        *, functions: EconFunctionsMapping, constraints: ConstraintFunctionsMapping
    ) -> Callable[..., object]:
        assert constraints is not unused, "Unused-regime constraints were composed."
        visited.append(constraints)
        return original(functions=functions, constraints=constraints)

    monkeypatch.setattr(initial_module, "_get_feasibility", observe)
    _validate(
        model=model,
        initial={
            "regime_id": jnp.array([1], dtype=jnp.int32),
            "age": jnp.array([2.0]),
            "wealth": jnp.array([40.0]),
        },
    )
    assert len(visited) == 1


def test_a_discrete_code_in_an_unrelated_cohort_is_ignored() -> None:
    """A supplied state of an absent regime is allowed and not validated there."""
    model = _make_asymmetric_state_model()
    _validate(
        model=model,
        initial={
            "regime_id": jnp.array([1], dtype=jnp.int32),
            "age": jnp.array([2.0]),
            "wealth": jnp.array([40.0]),
            "health": jnp.array([999], dtype=jnp.int32),
        },
    )


def test_changed_initial_values_are_validated_again_on_the_same_model() -> None:
    """A successful first call cannot cache a validity bit for later arrays."""
    model = _make_asymmetric_state_model()
    initial = {
        "regime_id": jnp.array([0], dtype=jnp.int32),
        "age": jnp.array([0.0]),
        "wealth": jnp.array([40.0]),
        "health": jnp.array([0], dtype=jnp.int32),
    }
    _validate(model=model, initial=initial)
    changed = {**initial, "health": jnp.array([999], dtype=jnp.int32)}
    with pytest.raises(InvalidInitialConditionsError) as caught:
        _validate(model=model, initial=changed)
    assert str(caught.value) == (
        "Invalid values [999] for discrete state 'health'. Valid codes are: [0, 1]"
    )


def test_changed_transition_params_are_validated_again_on_the_same_model() -> None:
    """The same canonical producer must consume this call's probability operand."""

    def state_law(health: DiscreteState) -> FloatND:
        del health
        return jnp.array([0.5, 0.5])

    model = _model_with_state_probs(state_law)
    regime = model._regimes["alive"]

    def bound_probs(
        *, probability: FloatND, age: object, period: object
    ) -> MappingProxyType:
        del age, period
        return MappingProxyType({"terminal": probability})

    regimes = MappingProxyType(
        {
            **model._regimes,
            "alive": dataclasses.replace(
                regime,
                solution=dataclasses.replace(
                    regime.solution, validation_regime_transition_probs=bound_probs
                ),
            ),
        }
    )
    params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    valid = MappingProxyType(
        {
            **params,
            "alive": MappingProxyType(
                {**params["alive"], "probability": jnp.array(1.0)}
            ),
        }
    )
    invalid = MappingProxyType(
        {
            **params,
            "alive": MappingProxyType(
                {**params["alive"], "probability": jnp.array(-1.0)}
            ),
        }
    )
    logger = get_logger(log_level="debug")
    validate_transitions(
        regimes=regimes, flat_params=valid, ages=model.ages, logger=logger
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError) as caught:
        validate_transitions(
            regimes=regimes, flat_params=invalid, ages=model.ages, logger=logger
        )
    assert "contain values outside [0, 1]" in str(caught.value)


@pytest.mark.parametrize("family", ["initial", "transition"])
def test_skipping_either_validation_family_breaks_a_real_refusal(
    *, family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each seeded bypass makes its independent invalid-input oracle fail."""
    if family == "initial":
        model = _make_constrained_asymmetric_model()
        initial = {
            "age": jnp.array([0.0]),
            "wealth": jnp.array([40.0]),
            "regime_id": jnp.array([0], dtype=jnp.int32),
        }
        error_type = InvalidInitialConditionsError
        target = "_collect_feasibility_errors"
    else:

        def invalid_probabilities(health: DiscreteState) -> FloatND:
            return jnp.where(health == 0, jnp.array([-0.1, 1.1]), jnp.array([0.0, 1.0]))

        model = _model_with_state_probs(invalid_probabilities)
        initial = {
            "age": jnp.array([0.0]),
            "wealth": jnp.array([5.0]),
            "health": jnp.array([0], dtype=jnp.int32),
            "regime_id": jnp.array([0], dtype=jnp.int32),
        }
        error_type = InvalidStateTransitionProbabilitiesError
        target = "validate_transitions"

    def assert_rejected() -> None:
        with pytest.raises(error_type):
            _validate(model=model, initial=initial)

    assert_rejected()
    monkeypatch.setattr(initial_module, target, lambda **_kwargs: None)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        assert_rejected()


def test_a_later_feasibility_typeerror_overrides_earlier_aggregated_failures() -> None:
    """Feasibility messages are only raised after the complete used-regime sweep."""

    @categorical(ordered=False)
    class RegimeId:
        first: ScalarInt
        second: ScalarInt
        dead: ScalarInt

    armed = False

    def first_constraint(
        *, consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    def second_constraint(
        *, consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        if armed:
            raise TypeError("later feasibility sentinel")
        return consumption <= wealth

    def utility(consumption: ContinuousAction) -> FloatND:
        return consumption

    def next_regime() -> ScalarInt:
        return RegimeId.dead

    def terminal_utility(wealth: ContinuousState) -> FloatND:
        return wealth

    states = {"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)}
    actions = {"consumption": LinSpacedGrid(start=2, stop=3, n_points=2)}
    regimes = {
        name: UserRegime(
            transition=next_regime,
            active=lambda age: age == 0,
            states=states,
            actions=actions,
            state_transitions={"wealth": lambda wealth: wealth},
            functions={"utility": utility},
            constraints={"budget": constraint},
        )
        for name, constraint in (
            ("first", first_constraint),
            ("second", second_constraint),
        )
    }
    regimes["dead"] = UserRegime(
        transition=None,
        active=lambda age: age == 1,
        states=states,
        functions={"utility": terminal_utility},
    )
    model = Model(
        regimes=regimes,
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )
    armed = True
    with pytest.raises(InvalidInitialConditionsError) as caught:
        _validate(
            model=model,
            initial={
                "regime_id": jnp.array([0, 1], dtype=jnp.int32),
                "age": jnp.array([0.0, 0.0]),
                "wealth": jnp.array([1.0, 1.0]),
            },
        )
    assert str(caught.value) == (
        "TypeError in feasibility check for regime 'second': later feasibility sentinel"
    )


@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_earlier_transition_failure_precedes_a_later_python_exception(
    *, log_level: LogLevel, caplog: pytest.LogCaptureFixture
) -> None:
    """Transition errors publish per item, unlike initial feasibility aggregation."""
    armed = False

    def state_law(health: DiscreteState) -> FloatND:
        del health
        if armed:
            raise TypeError("later transition sentinel")
        return jnp.array([0.5, 0.5])

    model = _model_with_state_probs(state_law)
    regime = model._regimes["alive"]

    def invalid_regime_probs(*, age: object, period: object) -> MappingProxyType:
        del age, period
        return MappingProxyType({"terminal": jnp.array(-1.0)})

    # Inject a known malformed canonical probability array at its real producer.
    # The actual validator, error classes and policy dispatch remain unchanged.
    regimes = MappingProxyType(
        {
            **model._regimes,
            "alive": dataclasses.replace(
                regime,
                solution=dataclasses.replace(
                    regime.solution,
                    validation_regime_transition_probs=invalid_regime_probs,
                ),
            ),
        }
    )
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    armed = True
    logger = get_logger(log_level=log_level)
    caplog.clear()
    if log_level == "off":
        validate_transitions(
            regimes=regimes, flat_params=flat_params, ages=model.ages, logger=logger
        )
        assert not caplog.records
        return
    expected = (
        InvalidRegimeTransitionProbabilitiesError if log_level == "debug" else TypeError
    )
    with pytest.raises(expected) as caught:
        validate_transitions(
            regimes=regimes, flat_params=flat_params, ages=model.ages, logger=logger
        )
    if log_level == "debug":
        assert "contain values outside [0, 1]" in str(caught.value)
        assert not caplog.records
    else:
        assert str(caught.value) == "later transition sentinel"
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert "contain values outside [0, 1]" in warnings[0]
