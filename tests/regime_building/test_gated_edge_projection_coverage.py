"""Which states a gated edge's projections must supply a coordinate for.

A `ProjectedRegimeValue` on a gated edge is read by two different consumers, and they
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

A state whose grid is declared with `AgeSpecializedGrid` is a state its regime
carries in simulation like any other — the marker says only that the grid's
bounds move with age. A second reference regime, `annuity`, pins both halves of
that: the projection is owed on such a state, and supplying it builds.
"""

import re
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.regime_building.gated_edges import build_fallback_state_projector
from _lcm.regime_building.Q_and_F import (
    ResolvedProjectedRegimeValue,
    _build_same_period_ref_reader,
)
from _lcm.regime_building.V import VInterpolationInfo
from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt

_AGES = AgeGrid(start=0, stop=3, step="Y")
_WAGE = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
_CAREER = LinSpacedGrid(start=0.0, stop=10.0, n_points=3)
_GATE_THRESHOLD = 1.5

# The annuity's principal grid keeps three nodes at every age and moves its
# ceiling once, which is the whole point of the marker: nothing about the shape
# of the regime's arrays records that its states are age-specialized.
_PRINCIPAL_CEILING_EARLY = 8.0
_PRINCIPAL_CEILING_LATE = 6.0


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


@categorical(ordered=False)
class AgeSpecializedRegimeId:
    """Regime ids of the model whose fallback holds an age-varying grid."""

    src: ScalarInt
    src_exit: ScalarInt
    annuity: ScalarInt


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


def test_leg_fallback_projection_must_cover_an_age_specialized_state_of_its_regime():
    """A leg's fallback projection owes a coordinate on an age-varying state.

    `annuity` holds `principal` on a grid whose ceiling moves with age. That
    makes it no less a state the regime carries in simulation, so a leg routing
    a subject there must say where on that axis it lands. An empty projection is
    rejected at model build, naming the state it has to supply.
    """
    with pytest.raises(
        ModelInitializationError, match=re.escape("(['principal']); got []")
    ):
        _build_age_specialized_model(fallback_projects_principal=False)


def test_leg_fallback_projection_covering_an_age_specialized_state_builds():
    """A projection onto an age-varying state is a complete one.

    Supplying the coordinate the previous test says is owed leaves nothing
    outstanding, so the model builds.
    """
    model = _build_age_specialized_model(fallback_projects_principal=True)

    assert set(model.regime_names_to_ids) == {"src", "src_exit", "annuity"}


def test_same_period_ref_reader_names_the_state_a_short_projection_omits():
    """A reader built below `Model` refuses a short projection by name.

    `Model` rejects an incomplete projection before any kernel is built, so a
    caller reaching the reader directly is the only one that can still hand it
    one. It gets the same account of what is missing — which reference regime,
    which state, and what the projection does supply — rather than a bare
    lookup failure naming a state and nothing around it.
    """
    with pytest.raises(
        ModelInitializationError,
        match=re.escape(
            "The projection onto reference regime 'annuity' supplies no "
            "coordinate function for state 'principal'. It supplies []."
        ),
    ):
        _build_same_period_ref_reader(
            ref=_short_ref(),
            v_interpolation_info=VInterpolationInfo(
                state_names=("principal",),
                discrete_states=MappingProxyType({}),
                continuous_states=MappingProxyType({"principal": _principal_grid(0.0)}),
            ),
            functions=MappingProxyType({}),
        )


def test_fallback_state_projector_names_the_state_a_short_projection_omits():
    """A projector built below `Model` refuses a short projection by name.

    The simulate-side projector owes a coordinate on every state a routed row
    occupies, and answers a missing one the same way its solve-side companion
    does.
    """
    with pytest.raises(
        ModelInitializationError,
        match=re.escape(
            "The projection onto reference regime 'annuity' supplies no "
            "coordinate function for state 'principal'. It supplies []."
        ),
    ):
        build_fallback_state_projector(
            ref=_short_ref(),
            fallback_simulate_state_names=("principal",),
            target_regime_name="src_exit",
            target_state_names=("wage",),
            target_functions=MappingProxyType({}),
            target_deterministic_transitions=MappingProxyType({}),
        )


def _short_ref() -> ResolvedProjectedRegimeValue:
    """A reference onto `annuity` whose projection supplies no coordinate."""
    return ResolvedProjectedRegimeValue(
        regime="annuity",
        projection=MappingProxyType({}),
        stakeholder_index=None,
    )


def _build_age_specialized_model(*, fallback_projects_principal: bool) -> Model:
    """Build a source whose leg fallback holds an age-varying grid.

    Args:
        fallback_projects_principal: Whether the leg's fallback projection
            supplies a coordinate for the annuity's age-specialized state.

    Returns:
        The built model.

    """
    projection = (
        {"principal": _project_principal} if fallback_projects_principal else {}
    )
    src = Regime(
        transition={
            "src_exit": ValueDependentTransition(
                probability=MarkovTransition(_prob_one),
                gate=_wage_gate,
                routes={
                    "only": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="annuity", projection=projection
                        )
                    )
                },
            )
        },
        active=lambda age: age < 1,
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_src},
    )
    src_exit = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _utility_no_payoff},
    )
    annuity = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={
            "principal": AgeSpecializedGrid(
                build=_principal_grid, signature=_principal_ceiling
            )
        },
        functions={"utility": _utility_annuity},
    )
    return Model(
        regimes={"src": src, "src_exit": src_exit, "annuity": annuity},
        ages=_AGES,
        regime_id_class=AgeSpecializedRegimeId,
    )


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
        transition={
            "src_exit": ValueDependentTransition(
                probability=MarkovTransition(_prob_one),
                gate=_gate,
                routes={
                    "only": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="fallback", projection=fallback_projection
                        )
                    )
                },
                gate_references={
                    "V_fallback_ref": ProjectedRegimeValue(
                        regime="fallback", projection=gate_ref_projection
                    )
                },
            )
        },
        active=lambda age: age < 1,
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_src},
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


def _wage_gate(wage: ContinuousState) -> BoolND:
    """The edge stays open above a wage, reading no other regime's value."""
    return wage > _GATE_THRESHOLD


def _principal_ceiling(age: float) -> float:
    """The highest principal the annuity's grid reaches at this age."""
    return _PRINCIPAL_CEILING_EARLY if age <= 1 else _PRINCIPAL_CEILING_LATE


def _principal_grid(age: float) -> LinSpacedGrid:
    """The annuity's principal grid: zero to the age's ceiling, on three nodes."""
    return LinSpacedGrid(start=0.0, stop=_principal_ceiling(age), n_points=3)


def _project_principal(wage: ContinuousState) -> FloatND:
    """The principal a subject rolls into the annuity on entering it."""
    return 2.0 * wage


def _utility_annuity(principal: ContinuousState) -> FloatND:
    """The annuity pays out its principal."""
    return principal


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
