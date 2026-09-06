"""A NEGM model may use any legal public name for its outer post-decision node.

The compiled outer sweep carries the keeper's outputs and its own inputs next to
the inner adjuster's arguments under engine-only keys, so no name a model can
declare is reserved by the solver.
"""

import numpy as np
import pytest
from dags import rename_arguments

from lcm import AgeGrid, Model
from lcm.consumption_savings_regime import (
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    outer_unchanged,
)
from tests.test_models import negm_kinked_toy as toy

_PARAMS = {"discount_factor": 0.95, "alive": {}}
_FINAL_AGE_ALIVE = 20 + (toy.N_PERIODS - 2) * 5


def _alive_regime_with_outer_node_named(
    *, outer_node: str
) -> NestedConsumptionSavingsRegime:
    """The kinked toy's alive regime with its outer post-decision node renamed."""
    return NestedConsumptionSavingsRegime(
        active=lambda age, n=_FINAL_AGE_ALIVE: age <= n,
        states={"wealth": toy.WEALTH_GRID, "illiquid": toy.ILLIQUID_GRID},
        state_transitions={
            "wealth": toy.next_wealth,
            "illiquid": rename_arguments(
                func=toy.durable_transition, mapper={"new_durable": outer_node}
            ),
        },
        actions={
            "consumption": toy.CONSUMPTION_GRID,
            "illiquid_investment": toy.ILLIQUID_INVESTMENT_GRID,
        },
        transition=toy.next_regime,
        functions={
            "utility": toy.utility,
            outer_node: toy.new_durable,
            "resources_before_outer_cost": toy.resources_before_outer_cost,
            "liquid_savings": toy.liquid_savings,
            "credited": rename_arguments(
                func=toy.credited, mapper={"new_durable": outer_node}
            ),
            "inverse_marginal_utility": toy.inverse_marginal_utility,
        },
        solver=toy.NEGM_SOLVER,
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources=NetOfAdjustmentCost(
                output="resources",
                before_cost="resources_before_outer_cost",
                cost="credited",
            ),
            post_decision_state="liquid_savings",
        ),
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="illiquid_investment",
            post_decision_state=outer_node,
            no_adjustment=outer_unchanged,
        ),
    )


def _model_with_outer_node_named(*, outer_node: str) -> Model:
    return Model(
        regimes={
            "alive": _alive_regime_with_outer_node_named(outer_node=outer_node),
            "dead": toy.build_dead_regime(),
        },
        regime_id_class=toy.RegimeId,
        ages=AgeGrid(start=20, stop=20 + (toy.N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": _FINAL_AGE_ALIVE},
    )


@pytest.mark.parametrize(
    "outer_node", ["outer_nodes", "coh_shifts", "keeper_value", "keeper_carry"]
)
def test_any_legal_name_for_the_outer_post_decision_node_solves_identically(
    *, outer_node: str
) -> None:
    """Renaming the outer post-decision node changes nothing but the name."""
    expected = toy.build_model().solve(params=_PARAMS, log_level="off").values
    got = (
        _model_with_outer_node_named(outer_node=outer_node)
        .solve(params=_PARAMS, log_level="off")
        .values
    )

    assert got.keys() == expected.keys()
    for period in expected:
        assert got[period].keys() == expected[period].keys()
        for regime in expected[period]:
            np.testing.assert_array_equal(
                np.asarray(got[period][regime]),
                np.asarray(expected[period][regime]),
                err_msg=f"period {period}, regime {regime}",
            )
