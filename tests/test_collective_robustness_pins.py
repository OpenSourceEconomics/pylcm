"""Contracts the collective-regime and gated-edge surface holds at its edges.

Each test pins one property a user meets through the documented API: a
declaration that stays immutable, a mapping every regime-shaped object carries,
a saved artifact that fails by naming what it lacks, a tie the household breaks
by its members rather than by its action grid, a coordinate that never blends two
categories, and the two names an edge cannot be declared without.
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

import lcm
from _lcm.regime_building.V import _get_identity_coordinate
from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    ParetoObjective,
    Regime,
)
from lcm.collective import MemberRoute
from lcm.exceptions import PyLCMError
from lcm.typing import DiscreteAction, FloatND
from tests.collective_fixtures import (
    Work,
    make_tied_shares_model,
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


@pytest.mark.parametrize("action_order", [("a", "b"), ("b", "a")])
def test_tied_household_solve_and_simulation_follow_the_members_not_the_grid(
    *, action_order: tuple[str, str]
) -> None:
    """A tie in the objective goes to the first member's best action.

    Every feasible split of the unit ties at objective `1/2`; `f` is declared
    first and values `a`, so the household takes `a = 1` and publishes
    `V = (1, 0)`, whichever of the two actions is declared first.
    """
    model = make_tied_shares_model(action_order=action_order)

    solution = model.solve(params={}, log_level="off")
    simulated = model.simulate(
        params={},
        initial_conditions={
            "period": jnp.zeros(1, dtype=jnp.int32),
            "regime_id": jnp.zeros(1, dtype=jnp.int32),
        },
        log_level="off",
    ).to_dataframe()

    assert_array_equal(solution.values[0]["couple"], [1.0, 0.0])
    assert_array_equal(simulated[["a", "b", "value_f", "value_m"]], [[1, 0, 1, 0]])


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


def _wife_payoff(work: DiscreteAction) -> FloatND:
    """Wife's payoff: 10 for working, nothing otherwise."""
    return 10.0 * work


def _husband_payoff(work: DiscreteAction) -> FloatND:
    """Husband's payoff: 5 for her leisure, nothing otherwise."""
    return 5.0 * (1.0 - work)
