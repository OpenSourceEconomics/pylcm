"""Mapped corner producers retain their associated channels."""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.finite_negm_query import (
    _FiniteCornerRead,
    _single_row_finite_corner_read,
)
from _lcm.utils.dispatchers import map_over_leading_axis
from lcm.typing import ScalarFloat


@pytest.mark.parametrize("batch_size", [0, 1], ids=["vmap", "batch-one"])
def test_mapped_corner_preserves_associated_channels(*, batch_size: int) -> None:
    """Both mapping paths preserve each node's complete associated corner tuple."""

    def at_node(offset: ScalarFloat) -> _FiniteCornerRead:
        return _single_row_finite_corner_read(
            value=3.0 + offset,
            marginal=5.0 + 2.0 * offset,
            savings=7.0 + 3.0 * offset,
            supported=offset == 0.0,
            finite=offset == 0.0,
            invalid=offset != 0.0,
        )

    mapped = map_over_leading_axis(
        func=at_node,
        xs=jnp.asarray([0.0, 10.0]),
        batch_size=batch_size,
    )
    assert isinstance(mapped, _FiniteCornerRead)
    assert len(mapped) == 6
    np.testing.assert_array_equal(mapped.value, [3.0, 13.0])
    np.testing.assert_array_equal(mapped.marginal, [5.0, 25.0])
    np.testing.assert_array_equal(mapped.savings, [7.0, 37.0])
    np.testing.assert_array_equal(mapped.supported, [True, False])
    np.testing.assert_array_equal(mapped.finite, [True, False])
    np.testing.assert_array_equal(mapped.invalid, [False, True])
