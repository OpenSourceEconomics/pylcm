"""A gated edge's gate must return a Boolean.

The gate selects a branch with a strict `where`, and JAX treats every nonzero
value as true there. A gate returning `0.25` would therefore open the edge for
every row rather than representing a 25% chance or failing, silently changing
both the solved continuation and the simulated route. A return annotation does
not constrain what a user function actually returns, so the realized output
dtype is checked where the gate is evaluated, in both phases.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    EdgeLeg,
    GatedEdge,
    LinSpacedGrid,
    Model,
    Regime,
    SamePeriodRef,
    categorical,
    fixed_transition,
)
from lcm.exceptions import PyLCMError
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt

_AGES = AgeGrid(start=40, stop=50, step="5Y")
_X = LinSpacedGrid(start=0.0, stop=2.0, n_points=2)


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _utility_source(x: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(x) * work


def _utility_target(x: ContinuousState) -> FloatND:
    return 10.0 + jnp.zeros_like(x)


def _utility_fallback(x: ContinuousState) -> FloatND:
    return jnp.zeros_like(x)


def _identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _numeric_gate(x: ContinuousState) -> BoolND:
    """Annotated Boolean, but actually returns a float — the defect under test."""
    return jnp.full_like(x, 0.25)


def _boolean_gate(x: ContinuousState) -> BoolND:
    return x > 1.0


def _make_model(*, gate) -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_prob_one)},
                active=lambda age: age < 45,
                states={"x": _X},
                state_transitions={"x": fixed_transition("x")},
                actions={"work": DiscreteGrid(Work)},
                functions={"utility": _utility_source},
                gated_edges={
                    "target": GatedEdge(
                        gate=gate,
                        legs={
                            "only": EdgeLeg(
                                fallback=SamePeriodRef(
                                    regime="fallback",
                                    projection={"x": _identity_x},
                                )
                            )
                        },
                    )
                },
            ),
            "target": Regime(
                transition=None,
                active=lambda age: age >= 45,
                states={"x": _X},
                functions={"utility": _utility_target},
            ),
            "fallback": Regime(
                transition=None,
                active=lambda age: age >= 45,
                states={"x": _X},
                functions={"utility": _utility_fallback},
            ),
        },
        ages=_AGES,
        regime_id_class=RegimeId,
    )


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "source": {"koopmans_aggregator": {"discount_factor": 0.5}},
        "target": {},
        "fallback": {},
    }


def test_solve_rejects_a_gate_returning_a_non_boolean() -> None:
    """Solving names the edge and the offending dtype rather than opening the edge."""
    model = _make_model(gate=_numeric_gate)

    with pytest.raises(PyLCMError) as excinfo:
        model.solve(params=_params(), log_level="debug")

    message = str(excinfo.value)
    assert "target" in message
    assert "float" in message.lower()


def test_solve_accepts_a_boolean_gate() -> None:
    """The check is specific to the dtype: an ordinary Boolean gate still solves."""
    model = _make_model(gate=_boolean_gate)

    solution = model.solve(params=_params(), log_level="debug")

    assert set(solution[1]) == {"target", "fallback"}
