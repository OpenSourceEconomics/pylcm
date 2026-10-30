"""A `Gate` routes on target states and params, never on a value.

A predicate naming a target value component or the dissolution flag is refused
when the gate is declared, with a message pointing to the two declarations that
express a value-driven move: a source `Exit` in `Regime(exits=...)` for a move
decided in the current period, and an `ArrivalDecision` for a comparison made on
landing.
"""

import pytest

from lcm import Gate, ProjectedRegimeValue, StakeholderRoute
from lcm.exceptions import GateReadsValueError, RegimeInitializationError
from lcm.typing import BoolND, ContinuousState, FloatND, UserFunction

_ROUTES = {
    "only": StakeholderRoute(
        fallback=ProjectedRegimeValue(regime="fallback", projection={})
    ),
}


def _reads_V_target(V_target: FloatND) -> BoolND:
    return V_target > 0.0


def _reads_stakeholder_component(*, V_target_f: FloatND, x: ContinuousState) -> BoolND:
    return V_target_f > x


def _reads_dissolution_flag(D_target: BoolND) -> BoolND:
    return ~D_target


def _reads_state(*, x: ContinuousState, threshold: float) -> BoolND:
    return x > threshold


@pytest.mark.parametrize(
    ("predicate", "operand"),
    [
        (_reads_V_target, "V_target"),
        (_reads_stakeholder_component, "V_target_f"),
        (_reads_dissolution_flag, "D_target"),
    ],
)
def test_a_value_reading_gate_predicate_is_refused(
    *, predicate: UserFunction, operand: str
) -> None:
    """Each reserved value operand names itself and both replacements."""
    with pytest.raises(GateReadsValueError) as excinfo:
        Gate(predicate=predicate, routes=_ROUTES)

    message = str(excinfo.value)
    assert operand in message
    assert "exits=" in message
    assert "ArrivalDecision" in message


def test_gate_reads_value_error_is_a_regime_initialization_error() -> None:
    """Code catching declaration errors also catches the refusal."""
    assert issubclass(GateReadsValueError, RegimeInitializationError)


def test_a_gate_declares_no_references() -> None:
    """Projected references are read by an `ArrivalDecision`, not by a gate."""
    with pytest.raises(TypeError, match="references"):
        Gate(
            predicate=_reads_state,
            routes=_ROUTES,
            references={  # ty: ignore[unknown-argument]
                "V_ref": ProjectedRegimeValue(regime="fallback", projection={})
            },
        )


def test_a_state_reading_gate_predicate_is_accepted() -> None:
    """A gate on target states and a free parameter is declared as written."""
    gate = Gate(predicate=_reads_state, routes=_ROUTES)

    assert gate.predicate is _reads_state
