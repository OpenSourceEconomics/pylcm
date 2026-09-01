"""Contracts the collective-regime and gated-edge surface holds at its edges.

Each test pins one property a user meets through the documented API: a
declaration that stays immutable, a mapping every regime-shaped object carries,
a saved artifact that fails by naming what it lacks, a kernel that does not
collapse an array when it has nothing to reduce, a coordinate that never blends
two categories, a declaration rejected where the user can still act on it, and
the two names an edge cannot be declared without.
"""

from pathlib import Path
from types import MappingProxyType

import cloudpickle
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

import lcm
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.collective import _gather_along_actions
from _lcm.regime_building.V import _get_identity_coordinate
from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    ParetoObjective,
    Regime,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.collective import ProjectedRegimeValue, StakeholderRoute
from lcm.exceptions import PyLCMError
from lcm.result import SimulationResult
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.collective_fixtures import (
    Work,
    make_couple_initial_conditions,
    make_two_stakeholder_model,
)
from tests.mock_regime import MockRegime

# Lifecycle of the gated-edge model: the source is active at age 0, both
# terminal regimes from age 1 on.
GATE_AGES = AgeGrid(start=0, stop=2, step="Y")

# The gated-edge model's only state.
GATE_WAGE_GRID = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)


@categorical(ordered=False)
class GateRegimeId:
    """Regime ids of the model whose gate reads a dissolution flag."""

    source: ScalarInt  # code 0
    target: ScalarInt  # code 1
    fallback: ScalarInt  # code 2


def test_regime_weights_keep_the_values_they_were_declared_with():
    """A regime's Pareto weights are its own; the declaring dict cannot rewrite them."""
    declared_weights = {"f": 0.25, "m": 0.75}
    regime = Regime(
        transition=None,
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _wife_payoff, "m": _husband_payoff},
                objective=ParetoObjective(weights=declared_weights),
            )
        },
    )

    declared_weights["f"] = 0.9

    objective = regime.pareto_objective
    assert objective is not None
    assert objective.weights == {"f": 0.25, "m": 0.75}


def test_mock_regime_carries_an_empty_gated_edges_mapping():
    """A regime-shaped object with no edges declared reads as declaring none."""
    assert dict(MockRegime().gated_edges) == {}


def test_saved_collective_result_lacking_stakeholder_metadata_names_the_field(
    tmp_path: Path,
):
    """Reading a saved collective result fails by naming the metadata it lacks.

    A result written without the per-regime stakeholder mapping cannot say
    which columns a collective regime's value splits into, so reading it back
    reports `regime_to_stakeholders` as the missing field.
    """
    model, params = make_two_stakeholder_model()
    result = model.simulate(
        params=params,
        initial_conditions=make_couple_initial_conditions(n_subjects=2),
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    directory = tmp_path / "result"
    result.save(directory=directory)
    _drop_stakeholder_metadata(directory=directory)

    with pytest.raises(PyLCMError, match="regime_to_stakeholders"):
        SimulationResult.load(directory=directory).to_dataframe()


def test_gather_at_the_household_argmax_over_no_action_axis_keeps_every_cell():
    """With no action axis to reduce, every cell's value is its own gathered value."""
    q = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    argmax_flat, _ = argmax_and_max(a=q, axis=(), initial=-jnp.inf)

    gathered = _gather_along_actions(q=q, argmax_flat=argmax_flat, action_axes=())

    assert_array_equal(gathered, q)


def test_non_integral_categorical_coordinate_is_rejected():
    """A value between two categories is not a state, so reading V there fails.

    A genuine categorical axis holds one value per category, and a coordinate
    halfway between two of them names no category — interpolating there would
    report a blend of two categories as if it were a state's value.
    """
    find_coordinate = _get_identity_coordinate(in_name="health")

    with pytest.raises((PyLCMError, ValueError), match="health"):
        find_coordinate(health=jnp.asarray(0.5))


def test_gate_reading_a_dissolution_flag_on_a_singleton_target_is_rejected_at_build():
    """A dissolution gate is rejected when its target cannot dissolve.

    A gate reading `D_target` names the target's dissolution flag, which only a
    collective regime has. A singleton target publishes none, and no argument
    to `solve` or `simulate` can supply one, so the declaration is rejected
    while the model is being built.
    """
    with pytest.raises((PyLCMError, NotImplementedError), match="D_target"):
        Model(
            regimes=_make_singleton_target_dissolution_gate_regimes(),
            ages=GATE_AGES,
            regime_id_class=GateRegimeId,
        )


def test_the_route_type_is_exported_and_the_edge_type_is_not():
    """A route is written; an edge is derived, so only the route is public.

    The comparison needs the defining module's object on one side, which is why
    this module imports `StakeholderRoute` from `lcm.collective`: reading both
    sides off `lcm` would compare the name against itself and pin nothing.
    """
    assert getattr(lcm, "StakeholderRoute", None) is StakeholderRoute
    assert not hasattr(lcm, "GatedEdge")


def _drop_stakeholder_metadata(*, directory: Path) -> None:
    """Rewrite a saved result's metadata payload without its stakeholder mapping.

    Args:
        directory: Directory `SimulationResult.save` wrote to.

    """
    path = directory / "metadata.pkl"
    with path.open("rb") as fh:
        metadata = cloudpickle.load(fh)
    del metadata.result_metadata.__dict__["regime_to_stakeholders"]
    with path.open("wb") as fh:
        cloudpickle.dump(metadata, fh)


def _make_singleton_target_dissolution_gate_regimes() -> MappingProxyType[str, Regime]:
    """Build regimes whose gate reads the dissolution flag of a singleton target.

    Returns:
        Immutable mapping of the three regime names to regimes: the gated
        `source`, the singleton `target` its edge routes into, and the
        `fallback` the closed branch reads.

    """
    source = Regime(
        transition={
            "target": ValueDependentTransition(
                probability=MarkovTransition(_enters_target),
                gate=_no_dissolution,
                routes={
                    "only": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="fallback",
                            projection={"wage": _identity_wage},
                        ),
                    )
                },
            ),
            "fallback": MarkovTransition(_never_entered),
        },
        active=lambda age: age < 1,
        states={"wage": GATE_WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _wage_utility},
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": GATE_WAGE_GRID},
        functions={"utility": _terminal_wage_utility},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": GATE_WAGE_GRID},
        functions={"utility": _fallback_wage_utility},
    )
    return MappingProxyType({"source": source, "target": target, "fallback": fallback})


def _wife_payoff(work: DiscreteAction) -> FloatND:
    """Wife's payoff: 10 for working, nothing otherwise."""
    return 10.0 * work


def _husband_payoff(work: DiscreteAction) -> FloatND:
    """Husband's payoff: 5 for her leisure, nothing otherwise."""
    return 5.0 * (1.0 - work)


def _no_dissolution(D_target: BoolND) -> BoolND:
    """Gate open exactly where the target regime has not dissolved."""
    return ~D_target


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    """Projection onto the fallback regime's own wage state."""
    return wage


def _wage_utility(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Source payoff: the wage when working, nothing otherwise."""
    return wage * work


def _terminal_wage_utility(wage: ContinuousState) -> FloatND:
    """Target payoff: the wage itself."""
    return wage


def _fallback_wage_utility(wage: ContinuousState) -> FloatND:
    """Fallback payoff: a tenth of the wage."""
    return 0.1 * wage


def _enters_target() -> FloatND:
    """Regime transition: the source enters `target` with probability one."""
    return jnp.asarray(1.0)


def _never_entered() -> FloatND:
    """Regime transition: `fallback` is reached through the edge, never directly."""
    return jnp.asarray(0.0)
