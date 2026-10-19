"""Scalar NB values preserve their linear owned-row representation."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.dcegm_query import _BlendedRow
from _lcm.egm.nbegm_query import NBEGMQuery
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.solution.feasibility import ContinuationReadCounts
from lcm.typing import Float1D, ScalarFloat
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_nbegm_counted_original_rows import _canonical_query

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _data(
    *,
    savings: Float1D,
    kinds: tuple[int, int, int],
    links: tuple[bool, bool] = (True, True),
    sides: tuple[int, int, int] = (0, 0, 0),
) -> EGMQueryData:
    """Supply analytic numeric rows to the genuine three-node scalar authority."""
    return EGMQueryData(
        savings=savings,
        node_kind=jnp.asarray(kinds, dtype=jnp.int32),
        link_valid=jnp.asarray(links),
        owner_id=jnp.zeros(3, dtype=jnp.int32),
        side_id=jnp.asarray(sides, dtype=jnp.int32),
        floor_used=jnp.asarray(0.5),
        corner_continuation=jnp.asarray([-100.0]),
        corner_kind=jnp.asarray([QueryNodeKind.FINITE], dtype=jnp.int32),
        corner_interval_id=jnp.asarray([0], dtype=jnp.int32),
        corner_side_id=jnp.asarray([0], dtype=jnp.int32),
    )


def _read_both(
    *,
    query: NBEGMQuery,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData,
    point: ScalarFloat,
) -> tuple[_BlendedRow, ContinuationReadCounts]:
    """Compare ordinary and counted readers in eager and compiled execution."""
    arguments = {
        "grid": grid,
        "value": value,
        "marginal": marginal,
        "data": data,
        "row_queries": point,
        "row_gradients": jnp.asarray(1.0),
        "weights": (),
        "active": jnp.asarray([True]),
        "negative": jnp.asarray([False]),
        "paired_marginal_read": False,
    }
    ordinary = query._read_rows(**arguments)
    for call in (
        query._read_rows_with_counts,
        jax.jit(
            query._read_rows_with_counts, static_argnames=("paired_marginal_read",)
        ),
    ):
        result, counts = call(**arguments)
        for actual, expected in zip(result, ordinary, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert isinstance(counts, ContinuationReadCounts)
    return ordinary, counts


@pytest.mark.parametrize(
    ("point", "expected", "counts"),
    [
        (1.125, (11.40625, 2.25, 0.65), (0, 1)),
        (0.75, (10.5625, 1.5, 0.6), (0, 1)),
        (0.6, (10.225, 1.2, 0.58), (1, 1)),
        (4.0, (19.0, 6.0, 0.8), (0, 1)),
    ],
)
def test_scalar_nb_ordinary_value_is_linear_on_the_owned_piece(
    *,
    point: float,
    expected: tuple[float, float, float],
    counts: tuple[int, int],
) -> None:
    """Linear V retains the same marginal, saving, support and original-row counts."""
    query, _pool = _canonical_query()
    grid = jnp.asarray([0.75, 1.5, 3.0])
    data = _data(
        savings=jnp.asarray([0.6, 0.7, 0.8]),
        kinds=(QueryNodeKind.FINITE, QueryNodeKind.FINITE, QueryNodeKind.FINITE),
    )
    result, observed_counts = _read_both(
        query=query,
        grid=grid,
        value=10.0 + grid**2,
        marginal=2.0 * grid,
        data=data,
        point=jnp.asarray(point),
    )
    np.testing.assert_array_almost_equal(
        result[:3], expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result[3:], [True, False])
    np.testing.assert_array_equal(observed_counts, counts)


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_scalar_nb_duplicate_endpoint_keeps_its_owned_ordinate(*, side: int) -> None:
    """An exact duplicated coordinate keeps the rightmost tuple for every side code."""
    query, _pool = _canonical_query()
    data = _data(
        savings=jnp.asarray([0.6, 0.7, 0.8]),
        kinds=(QueryNodeKind.FINITE, QueryNodeKind.FINITE, QueryNodeKind.FINITE),
        sides=(0, -1, side),
        links=(True, False),
    )
    result, counts = _read_both(
        query=query,
        grid=jnp.asarray([0.75, 1.5, 1.5]),
        value=jnp.asarray([10.0, 11.0, 13.0]),
        marginal=jnp.asarray([2.0, 4.0, 6.0]),
        data=data,
        point=jnp.asarray(1.5),
    )
    np.testing.assert_array_almost_equal(
        result[:3], [13.0, 6.0, 0.8], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result[3:], [True, False])
    np.testing.assert_array_equal(counts, [0, 1])


@pytest.mark.parametrize("neighbor", ["pad", "economic"])
def test_scalar_nb_isolated_exact_node_ignores_zero_weight_neighbors(
    *,
    neighbor: Literal["pad", "economic"],
) -> None:
    """A finite exact ordinate remains readable beside absent or economic records."""
    query, _pool = _canonical_query()
    kind = QueryNodeKind.PAD if neighbor == "pad" else QueryNodeKind.ECONOMIC_MINUS_INF
    dead_value = jnp.nan if neighbor == "pad" else -jnp.inf
    dead_saving = jnp.nan if neighbor == "pad" else 0.8
    dead_marginal = jnp.nan if neighbor == "pad" else 0.0
    data = _data(
        savings=jnp.asarray([dead_saving, 0.7, dead_saving]),
        kinds=(kind, QueryNodeKind.FINITE, kind),
        links=(False, False),
    )
    result, counts = _read_both(
        query=query,
        grid=jnp.asarray([0.75, 1.5, 3.0]),
        value=jnp.asarray([dead_value, 11.0, dead_value]),
        marginal=jnp.asarray([dead_marginal, 4.0, dead_marginal]),
        data=data,
        point=jnp.asarray(1.5),
    )
    np.testing.assert_array_almost_equal(
        result[:3], [11.0, 4.0, 0.7], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result[3:], [True, False])
    np.testing.assert_array_equal(counts, [0, 1])


def test_scalar_nb_missing_link_does_not_gain_linear_value_support() -> None:
    """A missing owned link stays unresolved even when generic interpolation is
    finite.
    """
    query, _pool = _canonical_query()
    data = _data(
        savings=jnp.asarray([0.6, 0.7, 0.8]),
        kinds=(QueryNodeKind.FINITE, QueryNodeKind.FINITE, QueryNodeKind.FINITE),
        links=(False, True),
    )
    result, counts = _read_both(
        query=query,
        grid=jnp.asarray([0.75, 1.5, 3.0]),
        value=jnp.asarray([10.0, 11.0, 13.0]),
        marginal=jnp.asarray([2.0, 4.0, 6.0]),
        data=data,
        point=jnp.asarray(1.125),
    )
    np.testing.assert_array_equal(result.supported, desired=False)
    np.testing.assert_array_equal(result.invalid_numeric, desired=False)
    np.testing.assert_array_equal(counts, [0, 1])
