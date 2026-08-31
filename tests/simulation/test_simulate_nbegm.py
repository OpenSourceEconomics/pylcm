"""NBEGM simulation evaluates phase-resolved post-decision constraints."""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    Phased,
    categorical,
    post_decision_lower_bound,
    ref,
)
from lcm.regime import Regime
from lcm.solvers import NBEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

pytestmark = pytest.mark.slow

_ACTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=41)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=41)
_MARGIN = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="liquid",
    post_decision_state="savings",
)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def terminal_utility(liquid: ContinuousState) -> FloatND:
    return 0.0 * liquid


def solve_savings(*, liquid: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return liquid - consumption


def simulate_savings(
    *, liquid: ContinuousState, consumption: ContinuousAction
) -> FloatND:
    return liquid - 2.0 * consumption


def next_liquid(savings: FloatND) -> ContinuousState:
    return savings


def next_regime(age: int) -> ScalarInt:
    return jnp.where(age >= 1, RegimeId.dead, RegimeId.alive)


def _model(*, hand_written: bool = False) -> Model:
    borrowing_limit = (
        ref("savings") >= 0.0
        if hand_written
        else post_decision_lower_bound(margin=_MARGIN, lower=0.0)
    )
    alive = ConsumptionSavingsRegime(
        actions={"consumption": _ACTION_GRID},
        states={"liquid": _ACTION_GRID},
        state_transitions={"liquid": {"alive": next_liquid, "dead": next_liquid}},
        constraints={"borrowing_limit": borrowing_limit},
        transition=next_regime,
        functions={
            "utility": utility,
            "savings": Phased(
                solve=solve_savings,
                simulate=simulate_savings,
            ),
        },
        active=lambda age: age < 2,
        solver=NBEGM(savings_grid=_SAVINGS_GRID),
        liquid=_MARGIN,
    )
    dead = Regime(
        transition=None,
        states={"liquid": _ACTION_GRID},
        functions={"utility": terminal_utility},
        active=lambda age: age == 2,
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


def _filled_params(model: Model) -> dict:
    def fill(node: object) -> object:
        if isinstance(node, dict):
            return {key: fill(value) for key, value in node.items()}
        if isinstance(node, bool) or not isinstance(node, int | float):
            return 0.95
        return node

    return fill(model.get_params_template())  # ty: ignore[invalid-return-type]


@pytest.mark.parametrize("declaration_kind", ["factory", "hand_written"])
def test_nbegm_simulation_evaluates_a_phase_resolved_savings_bound(
    declaration_kind: str,
) -> None:
    """The chosen action satisfies the simulation post-decision function."""
    model = _model(hand_written=declaration_kind == "hand_written")
    params = _filled_params(model)
    solved = model.solve(params=params, log_level="off")
    zeroed = MappingProxyType(
        {
            period: MappingProxyType(
                {name: jnp.zeros_like(arr) for name, arr in regime_to_V.items()}
            )
            for period, regime_to_V in solved.items()
        }
    )

    result = model.simulate(
        params=params,
        initial_conditions={
            "liquid": jnp.array([10.0]),
            "age": jnp.array([0.0]),
            "regime_id": jnp.array([RegimeId.alive], dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=zeroed,
        log_level="off",
        seed=1,
    )

    period_0 = result.to_dataframe(additional_targets=["savings"]).query(
        "regime_name == 'alive' and period == 0"
    )
    consumption = float(period_0["consumption"].iloc[0])
    savings = float(period_0["savings"].iloc[0])
    action_nodes = np.asarray(_ACTION_GRID.to_jax())
    expected = action_nodes[2.0 * action_nodes <= 10.0].max()
    assert consumption == pytest.approx(expected)
    assert savings == pytest.approx(10.0 - 2.0 * expected)
