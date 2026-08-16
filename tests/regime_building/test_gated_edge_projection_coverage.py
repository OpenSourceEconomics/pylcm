"""Which states a gated edge's projections must supply a coordinate for.

A `SamePeriodRef` on a gated edge is read by two different consumers, and they
need different coordinates of the regime it names:

- a LEG FALLBACK sends a subject INTO that regime, so forward simulation writes
  a per-subject row there. The row occupies every state the regime carries in
  simulation, carried states (`Phased(solve=..., simulate=Grid)`) included, so
  the projection owes a landing point on each of them.
- a GATE REFERENCE only READS that regime's value function. Its axes are the
  regime's solve states, and a carried state is none of them, so a coordinate
  on one would have nothing to index.

Both properties are pinned on the same reference regime, `fallback`, which
carries `career` in simulation but solves on `wage` alone: the projection that
the leg owes is exactly the one the gate reference may not declare.
"""

import re

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Phased,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt

_AGES = AgeGrid(start=0, stop=3, step="Y")
_WAGE = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
_CAREER = LinSpacedGrid(start=0.0, stop=10.0, n_points=3)
_GATE_THRESHOLD = 1.5


@categorical(ordered=True)
class Work:
    """The binary labor-supply action every regime here offers."""

    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class RegimeId:
    """Regime ids of the source-and-fallback model in this module."""

    src: ScalarInt
    src_exit: ScalarInt
    fallback: ScalarInt
    fallback_exit: ScalarInt


def test_leg_fallback_projection_must_cover_a_carried_state_of_its_regime():
    """A leg's fallback projection owes a coordinate on the regime's `career`.

    `fallback` carries `career` per subject once simulation runs, so a leg
    routing a subject into it must say where on that axis the subject lands.
    A projection covering the solve state alone is rejected at model build,
    naming both states it has to supply.
    """
    with pytest.raises(
        ModelInitializationError, match=re.escape("(['career', 'wage']); got ['wage']")
    ):
        _build_model(fallback_projects_career=False, gate_ref_projects_career=False)


def test_gate_ref_projection_must_cover_only_the_reference_value_functions_axes():
    """A gate reference may not project a state its regime does not solve on.

    The gate reads `fallback`'s value function, whose only axis is `wage` —
    `career` has no axis there to index. Supplying a coordinate for it is
    rejected at model build, naming `wage` alone as what the reference needs.
    """
    with pytest.raises(
        ModelInitializationError,
        match=re.escape("(['wage']); got ['career', 'wage']"),
    ):
        _build_model(fallback_projects_career=True, gate_ref_projects_career=True)


def _build_model(
    *, fallback_projects_career: bool, gate_ref_projects_career: bool
) -> Model:
    """Build the source-and-fallback model at the given projection coverage.

    Args:
        fallback_projects_career: Whether the leg's fallback projection supplies
            a coordinate for the carried `career` state.
        gate_ref_projects_career: Whether the gate reference's projection does.

    Returns:
        The built model.

    """
    fallback_projection = {"wage": _identity_wage}
    if fallback_projects_career:
        fallback_projection["career"] = _project_career
    gate_ref_projection = {"wage": _identity_wage}
    if gate_ref_projects_career:
        gate_ref_projection["career"] = _project_career

    src = Regime(
        transition={"src_exit": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_src},
        gated_edges={
            "src_exit": GatedEdge(
                gate=_gate,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="fallback", projection=fallback_projection
                        )
                    )
                },
                gate_refs={
                    "V_fallback_ref": SamePeriodRef(
                        regime="fallback", projection=gate_ref_projection
                    )
                },
            )
        },
    )
    src_exit = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _utility_no_payoff},
    )
    fallback = Regime(
        transition={"fallback_exit": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={
            "wage": _WAGE,
            "career": Phased(solve=_impute_career, simulate=_CAREER),
        },
        state_transitions={"wage": fixed_transition("wage"), "career": _next_career},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_fallback},
    )
    fallback_exit = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE},
        functions={"utility": _utility_no_payoff},
    )
    return Model(
        regimes={
            "src": src,
            "src_exit": src_exit,
            "fallback": fallback,
            "fallback_exit": fallback_exit,
        },
        ages=_AGES,
        regime_id_class=RegimeId,
    )


def _prob_one(age: FloatND) -> FloatND:
    """Regime transition taken with certainty."""
    return jnp.ones_like(age, dtype=float)


def _gate(V_fallback_ref: FloatND, wage: ContinuousState) -> BoolND:
    """The edge stays open above a wage, reading the fallback's value too."""
    return (wage > _GATE_THRESHOLD) | (V_fallback_ref < 0.0)


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    """A subject keeps its wage on entering the fallback regime."""
    return wage


def _project_career(wage: ContinuousState) -> FloatND:
    """The career a subject starts the fallback regime at."""
    return 2.0 * wage


def _impute_career() -> FloatND:
    """Career while `fallback` is solved: a constant, never a grid axis."""
    return jnp.asarray(0.0)


def _next_career(career: FloatND) -> FloatND:
    """Career law once simulation runs: one more year per period."""
    return career + 1.0


def _utility_src(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Payoff in the source regime."""
    return wage * work


def _utility_fallback(
    wage: ContinuousState, work: DiscreteAction, career: FloatND
) -> FloatND:
    """Payoff in the fallback regime, reading its carried career."""
    return wage * work + 0.0 * career


def _utility_no_payoff(wage: ContinuousState) -> FloatND:
    """Terminal payoff: nothing, so the terminal period adds no value."""
    return 0.0 * wage
