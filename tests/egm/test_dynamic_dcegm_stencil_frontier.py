"""Missing corner support and canonical passive-axis order remain explicit."""

from dataclasses import replace
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from tests.conftest import DECIMAL_PRECISION
from tests.egm.test_dynamic_dcegm_query import _adapter, _data, _query


@pytest.mark.parametrize("z", [0.0, 0.5, -0.5])
def test_missing_corner_requires_coverage_exactly_when_its_coefficient_is_active(
    *, z: float
) -> None:
    """A missing active corner stays unresolved under either coefficient sign."""
    data = replace(
        _data(),
        node_kind=jnp.full((2, 2), QueryNodeKind.PAD, dtype=jnp.int32),
        corner_continuation=jnp.array([[0.0], [jnp.nan]]),
        corner_kind=jnp.array(
            [[QueryNodeKind.FINITE], [QueryNodeKind.PAD]], dtype=jnp.int32
        ),
    )
    result = _query(data=data, z=z)
    if z == 0.0:
        np.testing.assert_allclose(
            [result.value, result.marginal, result.savings], [0.0, 10.0, 0.0]
        )
    else:
        assert bool(jnp.isneginf(result.value))
        np.testing.assert_allclose(
            [result.marginal, result.savings], [0.0, z * (1.0 - z)]
        )
    assert bool(result.coverage_missing) is (z != 0.0)
    assert not bool(result.bound_status.nonfinite_floor)
    assert not bool(result.bound_status.invalid_shifted_grid)


def test_economic_corner_does_not_certify_missing_interior_support() -> None:
    """An economic minus-infinity corner cannot prove every interior impossible."""
    data = replace(
        _data(),
        node_kind=jnp.full((2, 2), QueryNodeKind.PAD, dtype=jnp.int32),
        corner_continuation=jnp.full((2, 1), -jnp.inf),
        corner_kind=jnp.full((2, 1), QueryNodeKind.ECONOMIC_MINUS_INF, dtype=jnp.int32),
    )
    result = _query(data=data)
    assert bool(jnp.isneginf(result.value))
    np.testing.assert_allclose([result.marginal, result.savings], [0.0, 0.25])
    assert bool(result.coverage_missing)
    assert not bool(result.bound_status.nonfinite_floor)
    assert not bool(result.bound_status.invalid_shifted_grid)


@pytest.mark.parametrize("reverse_axes", [False, True])
def test_named_passive_axes_preserve_an_asymmetric_signed_corner_stencil(
    *, reverse_axes: bool
) -> None:
    """Permuting named grids and every bank together preserves the same query."""
    adapter = _adapter()
    names = ("w", "z") if reverse_axes else ("z", "w")
    adapter = replace(
        adapter,
        model=replace(
            adapter.model,
            authority=replace(adapter.model.authority, state_names=names),
        ),
    )
    bank = jnp.array([[2.0, 3.0, 7.0], [5.0, 11.0, 13.0]])
    if reverse_axes:
        bank = bank.T
    shape = bank.shape
    values = {"z": jnp.asarray(-0.5), "w": jnp.asarray(0.25)}
    grids = {"z": jnp.array([0.0, 1.0]), "w": jnp.array([0.0, 1.0, 2.0])}
    data = EGMQueryData(
        savings=jnp.zeros((*shape, 2)),
        node_kind=jnp.full((*shape, 2), QueryNodeKind.PAD, dtype=jnp.int32),
        link_valid=jnp.zeros((*shape, 1), dtype=jnp.bool_),
        owner_id=jnp.zeros((*shape, 2), dtype=jnp.int32),
        side_id=jnp.zeros((*shape, 2), dtype=jnp.int32),
        floor_used=jnp.zeros(shape),
        corner_continuation=bank[..., None],
        corner_kind=jnp.full((*shape, 1), QueryNodeKind.FINITE, dtype=jnp.int32),
        corner_interval_id=jnp.zeros((*shape, 1), dtype=jnp.int32),
        corner_side_id=jnp.zeros((*shape, 1), dtype=jnp.int32),
    )
    result = adapter(
        grid=jnp.broadcast_to(jnp.array([0.5, 1.5]), (*shape, 2)),
        value=jnp.zeros((*shape, 2)),
        marginal=jnp.zeros((*shape, 2)),
        data=data,
        row_queries=jnp.ones(shape),
        row_gradients=jnp.full(shape, 10.0),
        passive_values=tuple(values[name] for name in names),
        passive_grids=tuple(grids[name] for name in names),
        pool={
            "wealth": jnp.asarray(0.0),
            **values,
            "discount_factor": jnp.asarray(1.0),
            "eligibility": jnp.asarray(a=True),
        },
        resource_gradient=jnp.asarray(10.0),
    )
    reference = Fraction(3, 2) * (Fraction(3, 4) * 2 + Fraction(1, 4) * 3)
    reference -= Fraction(1, 2) * (Fraction(3, 4) * 5 + Fraction(1, 4) * 11)
    assert reference == Fraction(1, 8)
    np.testing.assert_allclose(
        [result.value, result.marginal, result.savings],
        [np.log(1.75) + float(reference), 40.0 / 7.0, -0.75],
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
    assert not bool(result.coverage_missing)
    assert not bool(result.bound_status.nonfinite_floor)
    assert not bool(result.bound_status.invalid_shifted_grid)
