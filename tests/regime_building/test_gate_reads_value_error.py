"""A `Gate` routes on target states and params, never on a value.

A predicate naming a target value component or the dissolution flag is refused when the
gate is declared. The message points to the two declarations of a decision on the
composition that compares values, each a `CompositionAction` of the regime that
decides: formation under a `Consent` rule on the source regime, and separation as the
own `CompositionAction` of the regime with several members.
"""

import pytest

from lcm import Gate, MemberRoute, ProjectedRegimeValue
from lcm.exceptions import GateReadsValueError, RegimeInitializationError
from lcm.typing import BoolND, ContinuousState, FloatND, UserFunction

_ROUTES = {
    "only": MemberRoute(
        fallback=ProjectedRegimeValue(regime="fallback", projection={})
    ),
}


def _reads_V_target(V_target: FloatND) -> BoolND:
    return V_target > 0.0


def _reads_member_component(*, V_target_f: FloatND, x: ContinuousState) -> BoolND:
    return V_target_f > x


def _reads_dissolution_flag(D_target: BoolND) -> BoolND:
    return ~D_target


def _reads_state(*, x: ContinuousState, threshold: float) -> BoolND:
    return x > threshold


@pytest.mark.parametrize(
    ("predicate", "operand"),
    [
        (_reads_V_target, "V_target"),
        (_reads_member_component, "V_target_f"),
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
    assert "`CompositionAction` with a `Consent` rule on the source regime" in message
    assert "own `CompositionAction` of the regime with several members" in message


def test_gate_reads_value_error_is_a_regime_initialization_error() -> None:
    """Code catching declaration errors also catches the refusal."""
    assert issubclass(GateReadsValueError, RegimeInitializationError)


def test_a_gate_declares_no_references() -> None:
    """A gate routes on target states and params; it declares no value reference."""
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
