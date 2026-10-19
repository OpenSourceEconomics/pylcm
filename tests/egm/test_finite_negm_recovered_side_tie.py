"""Legal recovered side profiles determine a right-indistinguishable tie."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.finite_negm_query import _read_finite_native_row
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind, select_query_data
from _lcm.egm.upper_envelope.segment_envelope import (
    OwnedEnvelope,
    refine_envelope_exact,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_germ_oracle import _emitted_row
from tests.egm._finite_germ_query_fixture import make_canonical_query
from tests.egm.test_finite_negm_query_integrity_and_ties import _read

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def test_recovered_left_value_decides_a_right_identical_ordinary_tie() -> None:
    """An illegal left owner cannot defeat a genuinely legal ordinary left side."""
    query, pool = make_canonical_query()
    topology = query.model.authority.row_topology
    assert topology is not None
    capacity = query.model.authority.row_capacity
    first = refine_envelope_exact(
        endog_grid=jnp.asarray([0.25, 1.25, 0.75, 1.75]),
        policy=jnp.asarray([0.125, 1.0, 0.25, 1.0]),
        value=jnp.asarray([0.0, 2.0, -0.5, 0.5]),
        savings=jnp.asarray([0.125, 0.25, 0.5, 0.75]),
        n_refined=capacity,
    )
    second = refine_envelope_exact(
        endog_grid=jnp.asarray([0.75, 1.25, 1.75]),
        policy=jnp.asarray([0.25, 0.625, 1.0]),
        value=jnp.asarray([-0.5, 0.0, 0.5]),
        savings=jnp.asarray([0.5, 0.625, 0.75]),
        n_refined=capacity,
    )
    empty = _emitted_row(capacity=capacity, empty=True)
    assert isinstance(first, OwnedEnvelope)
    assert isinstance(second, OwnedEnvelope)
    assert int(first.n_kept) == 5
    assert int(second.n_kept) == 3
    np.testing.assert_array_equal(first.grid[:5], [0.25, 0.75, 1.25, 1.25, 1.75])
    np.testing.assert_array_equal(first.value[:5], [0.0, 1.0, 2.0, 0.0, 0.5])
    np.testing.assert_array_equal(first.owner_id[:5], [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(first.savings[2:4], [0.25, 0.625])
    np.testing.assert_array_equal(second.grid[:3], [0.75, 1.25, 1.75])
    first_m = jnp.where(
        first.owner_id == 0, 0.5, jnp.where(first.owner_id == 1, 4.0, jnp.nan)
    )
    second_m = jnp.where(jnp.arange(capacity) < second.n_kept, 5.0, jnp.nan)

    def data_for(*, row: OwnedEnvelope) -> EGMQueryData:
        return EGMQueryData(
            savings=row.savings,
            node_kind=jnp.where(
                jnp.arange(capacity) < row.n_kept,
                QueryNodeKind.FINITE,
                QueryNodeKind.PAD,
            ).astype(jnp.int32),
            link_valid=row.link_valid,
            owner_id=row.owner_id,
            side_id=row.side_id,
            floor_used=jnp.asarray(0.0),
            corner_continuation=jnp.asarray([-100.0]),
            corner_kind=jnp.asarray([QueryNodeKind.FINITE], dtype=jnp.int32),
            corner_interval_id=jnp.asarray([0], dtype=jnp.int32),
            corner_side_id=jnp.asarray([0], dtype=jnp.int32),
        )

    per_outer = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *(data_for(row=row) for row in (first, second, empty)),
    )
    data = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), per_outer)
    data = replace(data, floor_used=jnp.asarray([0.0, 0.25]))
    grid = jnp.stack((first.grid, second.grid, empty.grid))
    value = jnp.stack((first.value, second.value, empty.value))
    marginal = jnp.stack((first_m, second_m, empty.value))
    raw = tuple(
        _read_finite_native_row(
            coordinate=jnp.asarray(1.25),
            grid=grid[outer],
            value=value[outer],
            marginal=marginal[outer],
            data=select_query_data(
                data=data, positions=(0, outer), floor_positions=(0,)
            ),
            owner_capacity=topology.owner_capacity,
            side_is_row_side=True,
        )
        for outer in (0, 1)
    )
    assert all(bool(row.ordinary.supported & row.left.supported) for row in raw)
    assert all(not bool(row.germ_invalid) for row in raw)
    np.testing.assert_array_equal([row.ordinary.value for row in raw], [0.0, 0.0])
    np.testing.assert_array_equal([row.ordinary.marginal for row in raw], [4.0, 5.0])
    np.testing.assert_array_equal(
        jnp.stack(raw[0].right_germ), jnp.stack(raw[1].right_germ)
    )
    np.testing.assert_array_equal([row.left.value for row in raw], [2.0, 0.0])
    np.testing.assert_array_equal([row.left.savings for row in raw], [0.25, 0.625])
    assert bool(raw[0].left_germ[1] < raw[1].left_germ[1])
    # The right cubic pieces coincide after slope limiting, but raw left owner0
    # is ineligible at actual b3/8. Its legal left recovery is far below owner1.
    pool = {**pool, "stock": jnp.asarray(1.5)}
    floor, _, status = query.evaluate_floor(pool=pool)
    np.testing.assert_array_equal(floor, 0.375)
    assert not bool(status.nonfinite_floor | status.invalid_shifted_grid)
    assert bool(raw[0].left.savings < floor)
    assert bool(raw[1].left.savings >= floor)
    result = _read(
        query=query,
        pool=pool,
        bank=(
            jnp.stack((grid, grid)),
            jnp.stack((value, value)),
            jnp.stack((marginal, marginal)),
            data,
        ),
        passive_coordinate=1.5,
    )
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.value, [0.0])
    np.testing.assert_array_equal(result.marginal, [5.0])
