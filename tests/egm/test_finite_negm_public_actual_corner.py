"""Public NEGM source production and an observed reached finite query.

The scalar witness lives in `_finite_negm_actual_corner_witness.py`.
The fixture uses the shipped public declarations and the installed query method.
The observer calls that method unchanged and records its actual output; it does
not replace banks, change arithmetic, suppress status, or provide a fallback.
Authored and statically reviewed only; this audit did not execute this file.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.dcegm_query import DCEGMQuery, FiniteNEGMQueryResult
from _lcm.egm.query_data import QueryNodeKind
from _lcm.solution.negm import _NEGMPeriodKernel
from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    Regime,
    categorical,
    outer_unchanged,
    post_decision_lower_bound,
)
from lcm.solvers import DCEGM, NEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_negm_actual_corner_witness import (
    actual_corner,
    source_production_facts,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


def _child_gross(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return liquid + 1.0 + 8.0 * stock


def _parent_gross(*, liquid: ContinuousState) -> FloatND:
    return liquid + 20.0


def _cost(*, stock: ContinuousState, new_stock: ContinuousState) -> FloatND:
    return (new_stock - stock) ** 2 / 4.0


def _new_stock(
    *, stock: ContinuousState, investment: ContinuousAction
) -> ContinuousState:
    return stock + investment


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _inverse_marginal_utility(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _floor(*, stock: ContinuousState) -> FloatND:
    return 4.0 * stock * (1.0 - stock)


def _next_liquid(*, savings: FloatND) -> ContinuousState:
    return savings


def _parent_next_liquid(*, savings: FloatND) -> ContinuousState:
    return savings + 1.0


def _next_stock(*, new_stock: ContinuousState) -> ContinuousState:
    return new_stock


def _terminal(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return liquid / 8.0 + 0.0 * stock


def make_model(
    *,
    stock: float = 0.5,
    enable_jit: bool = True,
    child_utility: UserFunction = _utility,
    child_inverse_marginal_utility: UserFunction = _inverse_marginal_utility,
    child_consumption_points: tuple[float, ...] = (0.125, 0.5, 1.0, 2.0, 4.0, 16.0),
    child_floor: UserFunction = _floor,
) -> Model:
    """Reach the child's passive interior through a literal finite NEGM parent.

    The child's utility, inverse marginal utility, consumption grid and
    callable floor can be replaced so a second public witness reuses the same
    parent, transitions and terminal law.
    """
    liquid = IrregSpacedGrid(points=(0.0, 1.0, 2.0))
    child_stock = IrregSpacedGrid(points=(0.0, 1.0))
    parent_stock = IrregSpacedGrid(points=(0.0, stock, 1.0))
    savings_grid = IrregSpacedGrid(points=(0.0, 0.25, 1.0, 2.0))
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources=NetOfAdjustmentCost(
            output="resources", before_cost="gross", cost="cost"
        ),
        post_decision_state="savings",
    )

    def nested(*, child: bool):
        return NestedConsumptionSavingsRegime(
            states={"liquid": liquid, "stock": child_stock if child else parent_stock},
            actions={
                "consumption": IrregSpacedGrid(
                    points=child_consumption_points
                    if child
                    else (0.125, 0.5, 1.0, 2.0, 4.0, 16.0)
                ),
                "investment": IrregSpacedGrid(points=(-1.0, -0.5, 0.0, 0.5, 1.0)),
            },
            functions={
                "gross": _child_gross if child else _parent_gross,
                "cost": _cost,
                "new_stock": _new_stock,
                "savings": _savings,
                "utility": child_utility if child else _utility,
                "inverse_marginal_utility": (
                    child_inverse_marginal_utility
                    if child
                    else _inverse_marginal_utility
                ),
            },
            constraints={
                "borrowing": post_decision_lower_bound(
                    margin=margin, lower=child_floor if child else 0.0
                )
            },
            state_transitions={
                "liquid": _next_liquid if child else _parent_next_liquid,
                "stock": _next_stock,
            },
            liquid=margin,
            outer_continuous=OuterContinuousMargin(
                state="stock",
                action="investment",
                post_decision_state="new_stock",
                no_adjustment=outer_unchanged,
            ),
            solver=NEGM(
                inner=DCEGM(savings_grid=savings_grid),
                outer_grid=IrregSpacedGrid(points=(0.0, 1.0)),
            ),
        )

    return Model(
        regimes={
            "parent": nested(child=False),
            "child": nested(child=True),
            "done": Regime(
                states={"liquid": liquid, "stock": child_stock},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


def observe_public_corner(
    *,
    stock: float,
    runtime_checks: bool,
    enable_jit: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Read native source banks through an unchanged reached public solve."""
    source_production_facts()
    model = make_model(stock=stock, enable_jit=enable_jit)
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None
    assert authority.bound.kind == "dag"
    assert authority.outer_mode == "finite_max"
    assert authority.state_names == ("stock",)
    assert authority.action_names == ()
    assert authority.finite_outer is not None
    assert authority.finite_outer.outer_nodes == (0.0, 1.0)
    assert authority.corner_coordinate_kind == "compact_interval"
    assert authority.outer_capacity == 3
    assert len(authority.interval_ids) == len(authority.interval_side_ids) == 1
    assert isinstance(
        model._regimes["parent"].solution.period_kernels[0], _NEGMPeriodKernel
    )
    original = DCEGMQuery.read_choices
    observed = []

    def record(
        *,
        liquid,
        passive,
        value,
        marginal,
        coverage,
        nonfinite_floor,
        invalid_grid,
        grid,
        savings,
        kind,
        floors,
        corners,
        corner_kind,
        corner_interval,
        corner_side,
        coordinates,
        gradients,
        euler_gradient,
    ):
        if float(np.asarray(liquid)) == 1.0 and float(np.asarray(passive)) == stock:
            observed.append(
                tuple(
                    np.array(x)
                    for x in (
                        value,
                        marginal,
                        coverage,
                        nonfinite_floor,
                        invalid_grid,
                        grid,
                        savings,
                        kind,
                        floors,
                        corners,
                        corner_kind,
                        corner_interval,
                        corner_side,
                        coordinates,
                        gradients,
                        euler_gradient,
                    )
                )
            )

    def observed_read(self, **kwargs):
        result = original(self, **kwargs)
        if self.model.authority is authority:
            pair = result if isinstance(result, FiniteNEGMQueryResult) else result[0]
            assert isinstance(pair, FiniteNEGMQueryResult)
            data = kwargs["data"]
            assert data.corner_continuation is not None
            assert data.corner_kind is not None
            assert data.corner_interval_id is not None
            assert data.corner_side_id is not None
            assert kwargs["grid"].shape == (2, 3, authority.row_capacity)
            assert data.savings.shape == data.node_kind.shape == kwargs["grid"].shape
            assert data.floor_used.shape == (2,)
            assert data.corner_continuation.shape == data.corner_kind.shape == (2, 3, 1)
            assert (
                data.corner_interval_id.shape == data.corner_side_id.shape == (2, 3, 1)
            )
            assert kwargs["row_queries"].shape == kwargs["row_gradients"].shape == (2,)
            jax.debug.callback(
                record,
                liquid=kwargs["pool"]["liquid"],
                passive=kwargs["pool"]["stock"],
                value=pair.value,
                marginal=pair.marginal,
                coverage=pair.coverage_missing,
                nonfinite_floor=pair.bound_status.nonfinite_floor,
                invalid_grid=pair.bound_status.invalid_shifted_grid,
                grid=kwargs["grid"],
                savings=data.savings,
                kind=data.node_kind,
                floors=data.floor_used,
                corners=data.corner_continuation,
                corner_kind=data.corner_kind,
                corner_interval=data.corner_interval_id - authority.interval_ids[0],
                corner_side=data.corner_side_id - authority.interval_side_ids[0],
                coordinates=kwargs["row_queries"],
                gradients=kwargs["row_gradients"],
                euler_gradient=kwargs["euler_gradient"],
            )
        return result

    with monkeypatch.context() as patch:
        patch.setattr(DCEGMQuery, "read_choices", observed_read)
        solution = model.solve(
            params={"discount_factor": 0.5},
            log_level="off",
            runtime_checks=runtime_checks,
        )
        jax.block_until_ready(solution.values[0]["parent"])
        jax.effects_barrier()
    assert observed, "Public NEGM solve did not reach child liquid1/stock target."
    return observed


def assert_actual_corner(*, observed, stock: float) -> None:
    """Check source production before comparing the represented corner pair."""
    expected = actual_corner(stock=stock)
    tolerance = 10.0 ** (-DECIMAL_PRECISION)
    for item in observed:
        (
            value,
            marginal,
            coverage,
            nonfinite_floor,
            invalid_grid,
            grid,
            savings,
            kind,
            floors,
            corners,
            corner_kind,
            corner_interval,
            corner_side,
            coordinates,
            gradients,
            euler_gradient,
        ) = item
        assert not np.any(coverage | nonfinite_floor | invalid_grid)
        np.testing.assert_array_equal(floors, np.zeros_like(floors))
        np.testing.assert_array_equal(corners, np.zeros_like(corners))
        np.testing.assert_array_equal(
            corner_kind, np.full_like(corner_kind, QueryNodeKind.FINITE)
        )
        np.testing.assert_array_equal(corner_interval, np.zeros_like(corner_interval))
        np.testing.assert_array_equal(corner_side, np.zeros_like(corner_side))
        np.testing.assert_array_equal(coordinates, np.asarray([2.0, 10.0]))
        np.testing.assert_array_equal(gradients, np.ones_like(gradients))
        np.testing.assert_array_equal(euler_gradient, 1.0)
        assert grid.shape[:2] == (2, 3)
        for row in range(2):
            for outer in range(3):
                live = np.flatnonzero(kind[row, outer] == QueryNodeKind.FINITE)
                assert live.size >= 2
                x = grid[row, outer, live]
                position = np.searchsorted(x, coordinates[row], side="right")
                lower = max(0, min(position - 1, live.size - 2))
                bracket = live[lower : lower + 2]
                assert x[lower] <= coordinates[row] <= x[lower + 1]
                np.testing.assert_array_equal(savings[row, outer, bracket], [0.0, 0.0])
                assert np.all(savings[row, outer, bracket] < expected.savings)
        np.testing.assert_allclose(
            value, [expected.value], rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            marginal, [expected.marginal], rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("runtime_checks", [False, True])
def test_public_finite_negm_recovers_the_actual_corner(*, runtime_checks, monkeypatch):
    observed = observe_public_corner(
        stock=0.5,
        runtime_checks=runtime_checks,
        enable_jit=True,
        monkeypatch=monkeypatch,
    )
    assert_actual_corner(observed=observed, stock=0.5)
