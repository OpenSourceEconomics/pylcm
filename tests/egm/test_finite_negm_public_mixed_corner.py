"""Public NEGM mixed ordinary/corner witness through an observed finite query.

The model is the public actual-corner route with a callable zero floor and a
quadratic child utility. The observer calls the installed query unchanged and
authenticates the actual source records before comparing with the scalar
witness in `_finite_negm_mixed_corner_witness.py`. Constructor, source or
authority failures are prerequisite failures, not a semantic disagreement.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.dcegm_query import DCEGMQuery, FiniteNEGMQueryResult
from _lcm.egm.query_data import QueryNodeKind
from _lcm.solution.negm import _NEGMPeriodKernel
from lcm import Model
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_negm_mixed_corner_witness import blend_then_choose
from tests.egm.test_finite_negm_public_actual_corner import make_model as make_seed

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def quadratic(*, consumption: ContinuousAction) -> FloatND:
    return consumption - consumption**2 / 32.0


def inverse_quadratic(*, marginal_continuation: FloatND) -> FloatND:
    return 16.0 * (1.0 - marginal_continuation)


def zero_floor(*, stock: ContinuousState) -> FloatND:
    return 0.0 * stock


def make_model(*, stock: float, enable_jit: bool) -> Model:
    """Reuse the public actual-corner route with a quadratic child and zero floor.

    Endpoint controls keep the interior parent node at 0.5 so the public parent
    grid stays free of duplicates; the parent then reaches the child at the
    source node itself.
    """
    return make_seed(
        stock=stock if 0.0 < stock < 1.0 else 0.5,
        enable_jit=enable_jit,
        child_utility=quadratic,
        child_inverse_marginal_utility=inverse_quadratic,
        child_consumption_points=(0.125, 0.5, 1.0, 2.0, 4.0, 15.0),
        child_floor=zero_floor,
    )


def observe_mixed(*, stock: float, runtime_checks: bool, enable_jit: bool, monkeypatch):
    model = make_model(stock=stock, enable_jit=enable_jit)
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None
    assert authority.bound.kind == "dag"
    assert authority.outer_mode == "finite_max"
    assert authority.outer_capacity == 3
    assert authority.state_names == ("stock",)
    assert authority.action_names == ()
    assert authority.finite_outer is not None
    assert authority.finite_outer.outer_nodes == (0.0, 1.0)
    assert authority.corner_coordinate_kind == "compact_interval"
    assert len(authority.interval_ids) == len(authority.interval_side_ids) == 1
    assert isinstance(
        model._regimes["parent"].solution.period_kernels[0], _NEGMPeriodKernel
    )
    original = DCEGMQuery.read_choices
    observed = []

    def record(**item):
        if (
            float(np.asarray(item.pop("liquid"))) == 1.0
            and float(np.asarray(item.pop("stock"))) == stock
        ):
            observed.append({name: np.array(value) for name, value in item.items()})

    def observed_read(self, **kwargs):
        result = original(self, **kwargs)
        if self.model.authority is authority:
            pair = result if isinstance(result, FiniteNEGMQueryResult) else result[0]
            assert isinstance(pair, FiniteNEGMQueryResult)
            data = kwargs["data"]
            assert (
                kwargs["grid"].shape
                == data.savings.shape
                == data.node_kind.shape
                == (2, 3, authority.row_capacity)
            )
            assert data.floor_used.shape == (2,)
            assert data.corner_continuation is not None
            assert data.corner_kind is not None
            assert data.corner_interval_id is not None
            assert data.corner_side_id is not None
            assert data.corner_continuation.shape == data.corner_kind.shape == (2, 3, 1)
            assert kwargs["row_queries"].shape == kwargs["row_gradients"].shape == (2,)
            count = None if isinstance(result, FiniteNEGMQueryResult) else result[1]
            assert (count is not None) == kwargs.get(
                "collect_continuation_read_counts", False
            )
            jax.debug.callback(
                record,
                liquid=kwargs["pool"]["liquid"],
                stock=kwargs["pool"]["stock"],
                value=pair.value,
                marginal=pair.marginal,
                coverage=pair.coverage_missing,
                floor_bad=pair.bound_status.nonfinite_floor,
                grid_bad=pair.bound_status.invalid_shifted_grid,
                grid=kwargs["grid"],
                raw_value=kwargs["value"],
                raw_marginal=kwargs["marginal"],
                savings=data.savings,
                kind=data.node_kind,
                links=data.link_valid,
                owners=data.owner_id,
                sides=data.side_id,
                floors=data.floor_used,
                corner=data.corner_continuation,
                corner_kind=data.corner_kind,
                corner_interval=data.corner_interval_id - authority.interval_ids[0],
                corner_side=data.corner_side_id - authority.interval_side_ids[0],
                coordinate=kwargs["row_queries"],
                gradient=kwargs["row_gradients"],
                euler_gradient=kwargs["euler_gradient"],
                counted=jnp.asarray(count is not None),
                counts=jnp.asarray((-1, -1))
                if count is None
                else jnp.stack(tuple(count)),
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
    assert observed, (
        "The unchanged installed child query was not reached at the intended target."
    )
    return observed


def assert_blended_mixed(*, observed, stock: float, runtime_checks: bool):
    tolerance = 10.0 ** (-DECIMAL_PRECISION)
    reference = blend_then_choose(stock=Fraction(stock))
    for item in observed:
        assert not np.any(item["coverage"] | item["floor_bad"] | item["grid_bad"])
        np.testing.assert_array_equal(item["coordinate"], [2.0, 10.0])
        np.testing.assert_array_equal(item["gradient"], [1.0, 1.0])
        np.testing.assert_array_equal(item["euler_gradient"], 1.0)
        np.testing.assert_array_equal(item["floors"], [0.0, 0.0])
        np.testing.assert_array_equal(item["corner"], np.zeros_like(item["corner"]))
        np.testing.assert_array_equal(
            item["corner_kind"], np.full_like(item["corner_kind"], QueryNodeKind.FINITE)
        )
        np.testing.assert_array_equal(
            item["corner_interval"], np.zeros_like(item["corner_interval"])
        )
        np.testing.assert_array_equal(
            item["corner_side"], np.zeros_like(item["corner_side"])
        )
        for row, source_stock in enumerate((0.0, 1.0)):
            for outer, identity in enumerate((source_stock, 0.0, 1.0)):
                known = item["kind"][row, outer] == QueryNodeKind.FINITE
                live = np.flatnonzero(known)
                assert live.size >= 2
                grid = item["grid"][row, outer]
                coordinate = item["coordinate"][row]
                position = np.searchsorted(grid[live], coordinate, side="right")
                lower = max(0, min(position - 1, live.size - 2))
                left, right = live[lower : lower + 2]
                assert right == left + 1
                assert grid[left] <= coordinate <= grid[right]
                assert grid[left] < grid[right]
                assert item["links"][row, outer, left]
                assert (
                    item["owners"][row, outer, left]
                    == item["owners"][row, outer, right]
                    >= 0
                )
                np.testing.assert_array_equal(
                    item["savings"][row, outer, [left, right]], [0.0, 0.0]
                )
                cost = (identity - source_stock) ** 2 / 4.0
                consumption = grid[[left, right]] - cost
                assert np.all((consumption > 0.0) & (consumption <= 15.0))
                expected_value = consumption - consumption**2 / 32.0
                expected_marginal = 1.0 - consumption / 16.0
                np.testing.assert_allclose(
                    item["raw_value"][row, outer, [left, right]],
                    expected_value,
                    rtol=tolerance,
                    atol=tolerance,
                )
                np.testing.assert_allclose(
                    item["raw_marginal"][row, outer, [left, right]],
                    expected_marginal,
                    rtol=tolerance,
                    atol=tolerance,
                )
                # For this concave quadratic the native 3*secant limiter is
                # inactive: both positive endpoint slopes are <=2*secant.
                secant = (expected_value[1] - expected_value[0]) / (
                    grid[right] - grid[left]
                )
                assert np.all(expected_marginal > 0.0)
                assert np.all(expected_marginal < 3.0 * secant)
        assert bool(item["counted"]) == runtime_checks
        if runtime_checks:
            active_rows = 2 if 0.0 < stock < 1.0 else 1
            np.testing.assert_array_equal(item["counts"], [0, 3 * active_rows])
        np.testing.assert_allclose(
            item["value"], [float(reference.value)], rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            item["marginal"],
            [float(reference.marginal)],
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.parametrize("runtime_checks", [False, True])
def test_public_mixed_corner_blends_each_kind_before_choosing(
    *, runtime_checks, monkeypatch
):
    observed = observe_mixed(
        stock=0.5,
        runtime_checks=runtime_checks,
        enable_jit=True,
        monkeypatch=monkeypatch,
    )
    assert_blended_mixed(observed=observed, stock=0.5, runtime_checks=runtime_checks)
