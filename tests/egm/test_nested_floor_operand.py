"""A shared current floor retains only fixed-current scalars and input status."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nested_floor import nested_floor_at_cell, prepare_nested_floor
from _lcm.solution.feasibility import BoundedResult
from lcm.typing import FloatND, ScalarFloat


def _floor(*, category: FloatND, stock: FloatND) -> FloatND:
    return category / 8.0 + stock / 4.0


def _constant_floor() -> ScalarFloat:
    return jnp.asarray(0.25)


def _prepared() -> BoundedResult[FloatND]:
    return prepare_nested_floor(
        lower=_floor,
        offsets=jnp.array([0.0, 0.25, 1.0]),
        state_names=("category", "stock"),
        category=jnp.array([2, 0, 1], dtype=jnp.int32),
        stock=jnp.array([0.0, 1.0]),
    )


@pytest.mark.parametrize("kind", ["bool", "int", "complex", "shape"])
def test_nested_floor_requires_raw_real_floating_scalar(
    *, kind: Literal["bool", "int", "complex", "shape"]
) -> None:
    """A floor cannot become admissible by dtype conversion or dimension loss."""
    values = {
        "bool": jnp.asarray(a=True),
        "int": jnp.asarray(1, dtype=jnp.int32),
        "complex": jnp.asarray(0.25 + 0.5j),
        "shape": jnp.asarray([0.25]),
    }

    def lower() -> ScalarFloat:
        # The declared return type is intentionally violated to test the boundary.
        return values[kind]

    with pytest.raises(TypeError, match="one real floating scalar"):
        prepare_nested_floor(
            lower=lower,
            offsets=jnp.array([0.0, 0.25, 1.0]),
            state_names=(),
        )


def test_nested_floor_retains_only_current_cell_scalars() -> None:
    """Shared operands carry one floor and two status flags per current cell."""
    prepared = _prepared()
    assert type(prepared) is BoundedResult
    assert prepared.query_status is None
    assert prepared.continuation_read_counts is None
    assert len(prepared.bound_status) == 1
    leaves = jax.tree.leaves(prepared)
    assert len(leaves) == 3
    assert all(leaf.shape == (3, 2) for leaf in leaves)
    np.testing.assert_array_equal(
        prepared.output,
        [[0.25, 0.5], [0.0, 0.25], [0.125, 0.375]],
    )
    np.testing.assert_array_equal(
        prepared.bound_status[0].nonfinite_floor, np.zeros((3, 2), dtype=bool)
    )
    np.testing.assert_array_equal(
        prepared.bound_status[0].invalid_shifted_grid, np.zeros((3, 2), dtype=bool)
    )


@pytest.mark.parametrize("category", [2, 0, 1])
def test_nested_floor_selects_actual_unordered_codes_and_rebuilds_local_grid(
    *, category: int
) -> None:
    """The actual category identity selects b before one local offset translation."""
    floor, nodes, status = nested_floor_at_cell(
        prepared=_prepared(),
        offsets=jnp.array([0.0, 0.25, 1.0]),
        axes=(
            ("category", jnp.array([2, 0, 1], dtype=jnp.int32)),
            ("stock", jnp.array([0.0, 1.0])),
        ),
        pool={"category": jnp.int32(category), "stock": jnp.asarray(1.0)},
    )
    expected_floor = category / 8.0 + 0.25
    np.testing.assert_array_equal(floor, expected_floor)
    np.testing.assert_array_equal(nodes, np.array([0.0, 0.25, 1.0]) + expected_floor)
    assert not bool(status.nonfinite_floor)
    assert not bool(status.invalid_shifted_grid)


def test_nested_floor_off_axis_identity_preserves_invalid_input_status() -> None:
    """An absent current identity cannot acquire a neighbouring cell's authority."""
    _, _, status = nested_floor_at_cell(
        prepared=_prepared(),
        offsets=jnp.array([0.0, 0.25, 1.0]),
        axes=(
            ("category", jnp.array([2, 0, 1], dtype=jnp.int32)),
            ("stock", jnp.array([0.0, 1.0])),
        ),
        pool={"category": jnp.int32(7), "stock": jnp.asarray(1.0)},
    )
    assert not bool(status.nonfinite_floor)
    assert bool(status.invalid_shifted_grid)


def test_nested_floor_without_current_axes_has_one_scalar_operand() -> None:
    """A constant floor shares one scalar and reconstructs the supplied offsets."""
    prepared = prepare_nested_floor(
        lower=_constant_floor,
        offsets=jnp.array([0.0, 0.25, 1.0]),
        state_names=(),
    )
    assert [leaf.shape for leaf in jax.tree.leaves(prepared)] == [(), (), ()]
    floor, nodes, status = nested_floor_at_cell(
        prepared=prepared,
        offsets=jnp.array([0.0, 0.25, 1.0]),
        axes=(),
        pool={},
    )
    np.testing.assert_array_equal(floor, 0.25)
    np.testing.assert_array_equal(nodes, [0.25, 0.5, 1.25])
    assert not bool(status.nonfinite_floor)
    assert not bool(status.invalid_shifted_grid)
