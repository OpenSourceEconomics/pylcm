"""Plain EGM reads literal DC corners within existing target-family guards."""

import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
    post_decision_lower_bound,
)
from lcm.exceptions import ModelInitializationError
from lcm.solvers import DCEGM, EGM
from lcm.typing import ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_egm_floor import (
    inverse_marginal_utility,
    next_wealth,
    savings,
    terminal_utility,
    utility,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


def _to_child(savings: FloatND) -> ContinuousState:
    return savings + 1.0


def _model(*, plain_parent: bool, floor: float) -> Model:
    margin = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="wealth",
        post_decision_state="savings",
    )
    parent_solver = (
        EGM(savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0])))
        if plain_parent
        else DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0])),
            n_constrained_points=4,
        )
    )
    nodes = tuple(floor + offset for offset in (0.0, 0.5, 1.0, 1.5))
    child_solver = (
        DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array(nodes)), n_constrained_points=4
        )
        if plain_parent
        else EGM(savings_grid=IrregSpacedGrid(points=np.array(nodes)))
    )
    return Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={"wealth": IrregSpacedGrid(points=np.array([0.125, 0.25]))},
                actions={
                    "consumption": IrregSpacedGrid(points=np.array([0.125, 0.25, 0.5]))
                },
                functions={
                    "utility": utility,
                    "savings": savings,
                    "inverse_marginal_utility": inverse_marginal_utility,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=0.0)
                },
                state_transitions={"wealth": _to_child},
                solver=parent_solver,
            ),
            "child": ConsumptionSavingsRegime(
                liquid=margin,
                states={"wealth": IrregSpacedGrid(points=np.array([0.75, 1.0]))},
                actions={
                    "consumption": IrregSpacedGrid(
                        points=np.array([0.125, 0.25, 0.5, 1.0])
                    )
                },
                functions={
                    "utility": utility,
                    "savings": savings,
                    "inverse_marginal_utility": inverse_marginal_utility,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(margin=margin, lower=floor)
                },
                state_transitions={"wealth": next_wealth},
                solver=child_solver,
            ),
            "done": Regime(
                states={
                    "wealth": IrregSpacedGrid(
                        points=np.array([4.0, 5.5, 7.0, 8.5, 10.0])
                    )
                },
                functions={"utility": terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
    )


@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_plain_egm_parent_reads_literal_dc_binding_corner(*, floor: float) -> None:
    """A plain EGM parent consumes an exactly represented DC child corner."""
    model = _model(plain_parent=True, floor=floor)
    solution = model.solve(
        params={
            "parent": {"koopmans_aggregator": {"discount_factor": 1.0}},
            "child": {"koopmans_aggregator": {"discount_factor": 1.0}},
            "done": {},
        },
        log_level="off",
    )
    child_value = np.log(1.0 - floor) + np.log(3.0 * floor + 4.0)
    np.testing.assert_allclose(
        solution.values[1]["child"][-1],
        child_value,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_allclose(
        solution.values[0]["parent"][0],
        np.log(0.125) + child_value,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_dc_parent_preserves_its_nonterminal_target_family_guard(
    *, floor: float
) -> None:
    """A DC parent refuses nonterminal plain EGM targets at model construction."""
    with pytest.raises(
        ModelInitializationError, match="must itself use the DCEGM solver"
    ):
        _model(plain_parent=False, floor=floor)
