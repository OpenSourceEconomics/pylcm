"""Owned conditional rows retain one associated candidate through each read."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import carry
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from lcm.typing import Float1D, ScalarFloat
from tests.conftest import DECIMAL_PRECISION


def _data() -> EGMQueryData:
    return EGMQueryData(
        savings=jnp.array([0.25, 0.75, 1.25]),
        node_kind=jnp.array([1, 1, 1], dtype=jnp.int32),
        link_valid=jnp.array([True, True]),
        owner_id=jnp.array([0, 0, 0], dtype=jnp.int32),
        side_id=jnp.array([0, 0, 0], dtype=jnp.int32),
        floor_used=jnp.array(0.25),
        corner_continuation=jnp.array([0.0]),
        corner_kind=jnp.array([1], dtype=jnp.int32),
        corner_interval_id=jnp.array([0], dtype=jnp.int32),
        corner_side_id=jnp.array([0], dtype=jnp.int32),
    )


def _read(
    *,
    query: float | ScalarFloat,
    data: EGMQueryData,
    value: Float1D | None = None,
    marginal: Float1D | None = None,
) -> carry.OwnedRowRead:
    return carry.read_owned_row(
        query=jnp.array(query),
        grid=jnp.array([1.0, 2.0, 3.0]),
        value=jnp.array([2.0, 4.0, 6.0]) if value is None else value,
        marginal=jnp.array([2.0, 2.0, 2.0]) if marginal is None else marginal,
        data=data,
        owner_capacity=2,
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [(0.5, (1.0, 2.0, 0.0)), (1.5, (3.0, 2.0, 0.5)), (4.0, (6.0, 2.0, 1.25))],
)
def test_owned_read_preserves_associated_boundary_geometry(
    *, query: float, expected: tuple[float, float, float]
) -> None:
    """Value, marginal and original saving use one selected row bracket."""
    result = _read(query=query, data=_data())
    assert bool(result.supported)
    np.testing.assert_allclose(
        [result.value, result.marginal, result.savings],
        expected,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


@pytest.mark.parametrize(
    "defect",
    ["gap", "foreign_owner", "unknown_owner", "bad_side", "missing_node", "bad_saving"],
)
def test_owned_read_refuses_incomplete_active_tuple(*, defect: str) -> None:
    """An active missing or foreign candidate invalidates its complete tuple."""
    data = _data()
    if defect == "gap":
        data = replace(data, link_valid=jnp.array([False, True]))
    elif defect == "foreign_owner":
        data = replace(data, owner_id=jnp.array([0, 1, 1], dtype=jnp.int32))
    elif defect == "unknown_owner":
        data = replace(data, owner_id=jnp.array([2, 2, 2], dtype=jnp.int32))
    elif defect == "bad_side":
        data = replace(data, side_id=jnp.array([4, 4, 4], dtype=jnp.int32))
    elif defect == "missing_node":
        data = replace(data, node_kind=jnp.array([1, 0, 1], dtype=jnp.int32))
    else:
        data = replace(data, savings=jnp.array([0.25, jnp.nan, 1.25]))
    assert not bool(_read(query=1.5, data=data).supported)


def test_exact_owned_node_ignores_missing_adjacent_links() -> None:
    """An isolated represented node remains readable without its neighbours."""
    data = replace(
        _data(),
        link_valid=jnp.array([False, False]),
        node_kind=jnp.array([0, 1, 0], dtype=jnp.int32),
        savings=jnp.array([jnp.nan, 0.75, jnp.nan]),
    )
    result = _read(
        query=2.0,
        data=data,
        value=jnp.array([jnp.nan, 4.0, jnp.nan]),
        marginal=jnp.array([jnp.nan, 2.0, jnp.nan]),
    )
    assert bool(result.supported)
    np.testing.assert_array_equal(
        [result.value, result.marginal, result.savings], [4.0, 2.0, 0.75]
    )


@pytest.mark.parametrize(("query", "supported"), [(1.5, True), (0.5, False)])
def test_economic_minus_inf_obeys_signed_active_support(
    *, query: float, supported: bool
) -> None:
    """Negative economic weights are missing while positive weights stay economic."""
    data = replace(
        _data(),
        node_kind=jnp.array([1, QueryNodeKind.ECONOMIC_MINUS_INF, 1], dtype=jnp.int32),
    )
    result = _read(
        query=query,
        data=data,
        value=jnp.array([2.0, -jnp.inf, 6.0]),
        marginal=jnp.array([2.0, 0.0, 2.0]),
    )
    assert bool(result.supported) is supported
    if supported:
        assert jnp.isneginf(result.value)
        assert float(result.marginal) == 0.0
        assert float(result.savings) == 0.5


@pytest.mark.parametrize("channel", ["value", "marginal", "savings"])
def test_active_finite_numeric_corruption_remains_distinct_from_missing_support(
    *, channel: str
) -> None:
    """A finite record's invalid numbers remain fatal even beside other support."""
    data = _data()
    value = jnp.array([2.0, 4.0, 6.0])
    marginal = jnp.array([2.0, 2.0, 2.0])
    if channel == "savings":
        data = replace(data, savings=jnp.array([0.25, jnp.nan, 1.25]))
    elif channel == "value":
        value = value.at[1].set(jnp.nan)
    else:
        marginal = marginal.at[1].set(jnp.inf)
    result = _read(query=1.5, data=data, value=value, marginal=marginal)
    assert not bool(result.supported)
    assert bool(result.invalid_numeric)
    missing = _read(
        query=1.5, data=replace(_data(), link_valid=jnp.array([False, True]))
    )
    assert not bool(missing.supported)
    assert not bool(missing.invalid_numeric)


@pytest.mark.parametrize(
    ("query", "isolated", "expected"),
    [
        (0.5, False, 2.0),
        (1.0, False, 2.0),
        (2.0, False, 2.0),
        (3.0, False, 2.0),
        (4.0, False, 0.0),
        (2.0, True, 0.0),
    ],
)
def test_owned_value_keeps_native_query_derivative_at_exact_nodes(
    *, query: float, isolated: bool, expected: float
) -> None:
    """Value gradients use owning pieces; isolated exact reads remain constant."""
    data = _data()
    if isolated:
        data = replace(
            data,
            link_valid=jnp.array([False, False]),
            node_kind=jnp.array([0, 1, 0], dtype=jnp.int32),
        )
    derivative = jax.grad(lambda point: _read(query=point, data=data).value)(
        jnp.array(query)
    )
    np.testing.assert_allclose(
        derivative,
        expected,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


@pytest.mark.parametrize("metadata", ["kind", "owner", "side"])
def test_malformed_active_provenance_cannot_be_recovered_as_ordinary_missing(
    *, metadata: str
) -> None:
    """Malformed active record metadata remains fatal beside any finite corner."""
    data = _data()
    if metadata == "kind":
        data = replace(data, node_kind=jnp.array([1, 99, 1], dtype=jnp.int32))
    elif metadata == "owner":
        data = replace(data, owner_id=jnp.array([0, 2, 0], dtype=jnp.int32))
    else:
        data = replace(data, side_id=jnp.array([0, 99, 0], dtype=jnp.int32))
    result = _read(query=1.5, data=data)
    assert not bool(result.supported)
    assert bool(result.invalid_numeric)
    inactive = _read(query=1.0, data=data)
    assert bool(inactive.supported)
    assert not bool(inactive.invalid_numeric)
    padded = replace(
        data,
        node_kind=jnp.array([1, 0, 1], dtype=jnp.int32),
        owner_id=jnp.array([0, 99, 0], dtype=jnp.int32),
        side_id=jnp.array([0, 99, 0], dtype=jnp.int32),
    )
    missing = _read(query=1.5, data=padded)
    assert not bool(missing.supported)
    assert not bool(missing.invalid_numeric)


@pytest.mark.parametrize(
    ("query", "isolated", "expected"),
    [(2.0, False, 6.0), (3.0, False, 6.0), (2.0, True, 0.0)],
)
def test_exact_node_value_gradient_uses_supported_piece_slope(
    *, query: float, isolated: bool, expected: float
) -> None:
    """Connected Hermite slopes are limited without inventing an isolated piece."""
    data = _data()
    if isolated:
        data = replace(
            data,
            link_valid=jnp.array([False, False]),
            node_kind=jnp.array([0, 1, 0], dtype=jnp.int32),
        )
    derivative = jax.grad(
        lambda point: _read(query=point, data=data, marginal=jnp.full(3, 20.0)).value
    )(jnp.array(query))
    np.testing.assert_allclose(
        derivative,
        expected,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


def test_padding_only_row_has_zero_safe_value_gradient() -> None:
    """Missing padded geometry cannot poison a differentiable corner recovery."""
    data = replace(
        _data(),
        savings=jnp.full(3, jnp.nan),
        node_kind=jnp.zeros(3, dtype=jnp.int32),
        owner_id=jnp.full(3, 99, dtype=jnp.int32),
        side_id=jnp.full(3, 99, dtype=jnp.int32),
        link_valid=jnp.array([False, False]),
    )

    def read(*, query: ScalarFloat) -> carry.OwnedRowRead:
        return carry.read_owned_row(
            query=query,
            grid=jnp.full(3, jnp.nan),
            value=jnp.full(3, jnp.nan),
            marginal=jnp.full(3, jnp.nan),
            data=data,
            owner_capacity=2,
        )

    result = read(query=jnp.array(2.0))
    assert not bool(result.supported)
    assert not bool(result.invalid_numeric)
    assert float(result.value) == 0.0
    derivative = jax.grad(lambda point: read(query=point).value)(jnp.array(2.0))
    assert float(derivative) == 0.0


def test_true_singleton_keeps_marginal_distinct_from_value_gradient() -> None:
    """A constant singleton value has zero derivative beside its stored marginal."""
    data = replace(
        _data(),
        savings=jnp.array([0.25, jnp.nan, jnp.nan]),
        node_kind=jnp.array([1, 0, 0], dtype=jnp.int32),
        link_valid=jnp.array([False, False]),
    )

    def read(*, query: ScalarFloat) -> carry.OwnedRowRead:
        return carry.read_owned_row(
            query=query,
            grid=jnp.array([1.0, jnp.nan, jnp.nan]),
            value=jnp.array([2.0, jnp.nan, jnp.nan]),
            marginal=jnp.array([20.0, jnp.nan, jnp.nan]),
            data=data,
            owner_capacity=2,
        )

    result = read(query=jnp.array(1.0))
    assert bool(result.supported)
    np.testing.assert_array_equal([result.value, result.marginal], [2.0, 20.0])
    derivative = jax.grad(lambda point: read(query=point).value)(jnp.array(1.0))
    assert float(derivative) == 0.0


@pytest.mark.parametrize(
    ("query", "expected"),
    [(0.0, 3.0), (1.0, -3.0), (1e-6, 6.0 - 12e-6), (1.0 - 1e-6, -6.0 + 12e-6)],
)
def test_owned_value_preserves_native_hermite_query_curvature(
    *, query: float, expected: float
) -> None:
    """Value curvature follows the prepared-read endpoint and interior rules."""
    data = replace(
        _data(),
        savings=jnp.array([0.0, 0.5, jnp.nan]),
        node_kind=jnp.array([1, 1, 0], dtype=jnp.int32),
        link_valid=jnp.array([True, False]),
        side_id=jnp.array([1, -1, 0], dtype=jnp.int32),
    )

    def value_at(*, query: ScalarFloat) -> ScalarFloat:
        return carry.read_owned_row(
            query=query,
            grid=jnp.array([0.0, 1.0, jnp.nan]),
            value=jnp.array([0.0, 1.0, jnp.nan]),
            marginal=jnp.array([0.0, 0.0, jnp.nan]),
            data=data,
            owner_capacity=2,
        ).value

    curvature = jax.grad(jax.grad(lambda point: value_at(query=point)))(
        jnp.array(query)
    )
    np.testing.assert_allclose(
        curvature,
        expected,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


@pytest.mark.parametrize("incoming", [False, True])
def test_exact_boundary_uses_its_declared_owning_value_piece(*, incoming: bool) -> None:
    """Outgoing gap endpoints use their left piece; crossing right copies use right."""
    if incoming:
        grid = jnp.array([1.0, 2.0, 2.0, 3.0])
        value = jnp.array([1.0, 2.0, 5.0, 8.0])
        marginal = jnp.array([1.0, 1.0, 3.0, 3.0])
        data = replace(
            _data(),
            savings=jnp.array([0.25, 0.75, 1.25, 1.75]),
            node_kind=jnp.ones(4, dtype=jnp.int32),
            link_valid=jnp.array([True, False, True]),
            owner_id=jnp.array([0, 0, 1, 1], dtype=jnp.int32),
            side_id=jnp.array([1, -1, 1, -1], dtype=jnp.int32),
        )
        expected_value, expected_slope = 5.0, 3.0
    else:
        grid, value, marginal = (
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([2.0, 4.0, 6.0]),
            jnp.full(3, 2.0),
        )
        data = replace(
            _data(),
            link_valid=jnp.array([True, False]),
            owner_id=jnp.array([0, 0, 1], dtype=jnp.int32),
            side_id=jnp.array([1, -1, 1], dtype=jnp.int32),
        )
        expected_value, expected_slope = 4.0, 2.0

    def value_at(*, query: ScalarFloat) -> ScalarFloat:
        return carry.read_owned_row(
            query=query,
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            owner_capacity=2,
        ).value

    assert float(value_at(query=jnp.array(2.0))) == expected_value
    derivative = jax.grad(lambda point: value_at(query=point))(jnp.array(2.0))
    np.testing.assert_allclose(
        derivative,
        expected_slope,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


def test_whole_owned_row_record_survives_jitted_query_mapping() -> None:
    """Mapped query lanes preserve every associated channel and mandatory flag."""
    mapped = jax.jit(jax.vmap(lambda query: _read(query=query, data=_data())))(
        jnp.array([1.5, 2.0])
    )
    np.testing.assert_allclose(
        mapped.value,
        [3.0, 4.0],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    np.testing.assert_array_equal(mapped.marginal, [2.0, 2.0])
    np.testing.assert_array_equal(mapped.savings, [0.5, 0.75])
    np.testing.assert_array_equal(mapped.supported, [True, True])
    np.testing.assert_array_equal(mapped.invalid_numeric, [False, False])
    np.testing.assert_array_equal(mapped.owner_id, [0, 0])
    np.testing.assert_array_equal(mapped.side_id, [0, 0])
    role = object()
    roles = jax.tree.map(lambda _: role, mapped)
    assert all(leaf is role for leaf in roles)
