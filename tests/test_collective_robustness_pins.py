"""Contracts the collective-regime and gated-edge surface holds at its edges.

Each test pins one property a user meets through the documented API: a
declaration that stays immutable, a mapping every regime-shaped object carries,
a saved artifact that fails by naming what it lacks, a kernel that does not
collapse an array when it has nothing to reduce, a coordinate that never blends
two categories, and the two names an edge cannot be declared without.
"""

from pathlib import Path

import cloudpickle
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

import lcm
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.collective import _gather_along_actions
from _lcm.regime_building.V import _get_identity_coordinate
from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    ParetoObjective,
    Regime,
)
from lcm.collective import MemberRoute
from lcm.exceptions import PyLCMError
from lcm.result import SimulationResult
from lcm.typing import DiscreteAction, FloatND
from tests.collective_fixtures import (
    Work,
    make_couple_initial_conditions,
    make_two_member_model,
)
from tests.mock_regime import MockRegime


def test_regime_weights_keep_the_values_they_were_declared_with() -> None:
    """A regime's Pareto weights are its own; the declaring dict cannot rewrite them."""
    declared_weights = {"f": 0.25, "m": 0.75}
    regime = Regime(
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


def test_mock_regime_carries_no_gated_edges() -> None:
    """A regime-shaped object carries no gated edges: they live on the source's law."""
    assert not hasattr(MockRegime(), "gated_edges")


def test_saved_collective_result_lacking_member_metadata_names_the_field(
    tmp_path: Path,
) -> None:
    """Reading a saved collective result fails by naming the metadata it lacks.

    A result written without the per-regime member mapping cannot say
    which columns a collective regime's value splits into, so reading it back
    reports `regime_to_member_axis` as the missing field.
    """
    model, params = make_two_member_model()
    result = model.simulate(
        params=params,
        initial_conditions=make_couple_initial_conditions(n_subjects=2),
        log_level="off",
    )
    directory = tmp_path / "result"
    result.save(directory=directory)
    _drop_member_metadata(directory=directory)

    with pytest.raises(PyLCMError, match="regime_to_member_axis"):
        SimulationResult.load(directory=directory).to_dataframe()


def test_gather_at_the_household_argmax_over_no_action_axis_keeps_every_cell() -> None:
    """With no action axis to reduce, every cell's value is its own gathered value."""
    q = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    argmax_flat, _ = argmax_and_max(a=q, axis=(), initial=-jnp.inf)

    gathered = _gather_along_actions(q=q, argmax_flat=argmax_flat, action_axes=())

    assert_array_equal(gathered, q)


def test_non_integral_categorical_coordinate_is_rejected() -> None:
    """A value between two categories is not a state, so reading V there fails.

    A genuine categorical axis holds one value per category, and a coordinate
    halfway between two of them names no category — interpolating there would
    report a blend of two categories as if it were a state's value.
    """
    find_coordinate = _get_identity_coordinate(in_name="health")

    with pytest.raises((PyLCMError, ValueError), match="health"):
        find_coordinate(health=jnp.asarray(0.5))


def test_the_route_type_is_exported_and_the_edge_type_is_not() -> None:
    """A route is written; an edge is derived, so only the route is public.

    The comparison needs the defining module's object on one side, which is why
    this module imports `MemberRoute` from `lcm.collective`: reading both
    sides off `lcm` would compare the name against itself and pin nothing.
    """
    assert getattr(lcm, "MemberRoute", None) is MemberRoute
    assert not hasattr(lcm, "GatedEdge")


def _drop_member_metadata(*, directory: Path) -> None:
    """Rewrite a saved result's metadata payload without its member mapping.

    Args:
        directory: Directory `SimulationResult.save` wrote to.

    """
    path = directory / "metadata.pkl"
    with path.open("rb") as fh:
        metadata = cloudpickle.load(fh)
    del metadata.result_metadata.__dict__["regime_to_member_axis"]
    with path.open("wb") as fh:
        cloudpickle.dump(metadata, fh)


def _wife_payoff(work: DiscreteAction) -> FloatND:
    """Wife's payoff: 10 for working, nothing otherwise."""
    return 10.0 * work


def _husband_payoff(work: DiscreteAction) -> FloatND:
    """Husband's payoff: 5 for her leisure, nothing otherwise."""
    return 5.0 * (1.0 - work)
