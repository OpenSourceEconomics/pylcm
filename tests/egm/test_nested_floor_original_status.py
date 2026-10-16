"""Original invalid current-floor inputs survive safe local numerical operands."""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nested_floor import nested_floor_at_cell, prepare_nested_floor
from lcm.typing import FloatND


@pytest.mark.parametrize("kind", ["nan", "positive_inf", "negative_inf", "collapse"])
def test_nested_floor_preserves_original_invalid_input_after_lookup(
    *, kind: Literal["nan", "positive_inf", "negative_inf", "collapse"]
) -> None:
    """Invalid arithmetic retains its own flags beside exact safe floor/node values."""
    bad = {
        "nan": jnp.nan,
        "positive_inf": jnp.inf,
        "negative_inf": -jnp.inf,
        "collapse": 2.0**100,
    }[kind]
    offsets = jnp.asarray([0.0, 0.25, 1.0])
    stock = jnp.asarray([0.0, 1.0])

    def lower(*, stock: FloatND) -> FloatND:
        return jnp.where(stock == 0.0, bad, 0.25)

    prepared = prepare_nested_floor(
        lower=lower, offsets=offsets, state_names=("stock",), stock=stock
    )
    nonfinite = kind != "collapse"
    np.testing.assert_array_equal(prepared.output, [0.0, 0.25])
    np.testing.assert_array_equal(
        prepared.bound_status[0].nonfinite_floor, [nonfinite, False]
    )
    np.testing.assert_array_equal(
        prepared.bound_status[0].invalid_shifted_grid, [True, False]
    )
    assert prepared.output.shape == (2,)
    for position in (0, 1):
        floor, nodes, status = nested_floor_at_cell(
            prepared=prepared,
            offsets=offsets,
            axes=(("stock", stock),),
            pool={"stock": stock[position]},
        )
        np.testing.assert_array_equal(floor, [0.0, 0.25][position])
        np.testing.assert_array_equal(
            nodes, [0.0, 1.0, 2.0] if position == 0 else [0.25, 0.5, 1.25]
        )
        assert bool(status.nonfinite_floor) is (nonfinite and position == 0)
        assert bool(status.invalid_shifted_grid) is (position == 0)
