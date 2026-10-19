"""Owned value, marginal, saving and side profiles agree under JAX transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.finite_negm_query import _FiniteRowRead, _read_finite_native_row
from _lcm.egm.query_data import QueryNodeKind
from _lcm.egm.upper_envelope.segment_envelope import DEFAULT_MAX_RUNS
from lcm.typing import ScalarFloat
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_finite_negm_owned_germs import _emit

pytestmark = [
    pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON),
    pytest.mark.coverage(backends=("cpu",), precisions="both"),
]


def test_owned_quadratic_profiles_match_under_scalar_and_batched_jit() -> None:
    """Native quadratic profiles retain their analytic side and economic channels."""
    row, marginal, data = _emit(
        coordinates=(1.0, 2.0, 3.0),
        values=(0.0, 1.0, 4.0),
        marginals=(0.0, 2.0, 4.0),
    )
    np.testing.assert_array_equal(row.grid[:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(row.value[:3], [0.0, 1.0, 4.0])
    np.testing.assert_array_equal(marginal[:3], [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(data.savings[:3], [0.125, 0.1875, 0.25])
    np.testing.assert_array_equal(data.node_kind[:3], [QueryNodeKind.FINITE] * 3)
    np.testing.assert_array_equal(data.owner_id, row.owner_id)
    np.testing.assert_array_equal(data.side_id, row.side_id)
    np.testing.assert_array_equal(data.link_valid, row.link_valid)
    np.testing.assert_array_equal(data.owner_id[:3], [0, 0, 0])
    np.testing.assert_array_equal(data.link_valid[:2], [True, True])
    assert bool(jnp.all(jnp.isnan(row.grid[3:])))
    np.testing.assert_array_equal(data.node_kind[3:], [QueryNodeKind.PAD] * 5)
    coordinates = jnp.asarray([1.0, 2.0, 3.0, 3.5])

    def read(coordinate: ScalarFloat) -> _FiniteRowRead:
        return _read_finite_native_row(
            coordinate=coordinate,
            grid=row.grid,
            value=row.value,
            marginal=marginal,
            data=data,
            owner_capacity=DEFAULT_MAX_RUNS,
            side_is_row_side=True,
        )

    def assert_profile(profile: _FiniteRowRead) -> None:
        np.testing.assert_array_equal(profile.ordinary.value, [0.0, 1.0, 4.0, 4.0])
        np.testing.assert_array_equal(profile.ordinary.marginal, [0.0, 2.0, 4.0, 4.0])
        np.testing.assert_array_equal(
            profile.ordinary.savings, [0.125, 0.1875, 0.25, 0.25]
        )
        np.testing.assert_array_equal(profile.ordinary.supported, [True] * 4)
        np.testing.assert_array_equal(profile.ordinary.invalid_numeric, [False] * 4)
        np.testing.assert_array_equal(profile.left.value, [0.0, 1.0, 4.0, 4.0])
        np.testing.assert_array_equal(profile.left.marginal, [0.0, 2.0, 4.0, 4.0])
        np.testing.assert_array_equal(profile.left.savings, [0.125, 0.1875, 0.25, 0.25])
        np.testing.assert_array_equal(profile.left.supported, [True] * 4)
        np.testing.assert_array_equal(profile.left.invalid_numeric, [False] * 4)
        np.testing.assert_array_equal(profile.right_germ[0], [True] * 4)
        np.testing.assert_array_equal(profile.left_germ[0], [False, True, True, True])
        np.testing.assert_array_equal(
            jnp.stack(profile.right_germ[1:], axis=-1),
            [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(
            jnp.stack(profile.left_germ[1:], axis=-1),
            [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [4.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(profile.right_derivative, [0.0, 2.0, 4.0, 0.0])
        np.testing.assert_array_equal(profile.left_derivative, [0.0, 2.0, 4.0, 0.0])
        np.testing.assert_array_equal(profile.first, [1.0] * 4)
        np.testing.assert_array_equal(profile.last, [3.0] * 4)
        np.testing.assert_array_equal(profile.has_first, [True] * 4)
        np.testing.assert_array_equal(profile.germ_invalid, [False] * 4)

    eager_profiles = tuple(read(coordinate) for coordinate in coordinates)
    eager = jax.tree.map(lambda *leaves: jnp.stack(leaves), *eager_profiles)
    assert_profile(eager)

    scalar_read = jax.jit(read)
    scalar_profiles = tuple(scalar_read(coordinate) for coordinate in coordinates)
    scalar = jax.tree.map(lambda *leaves: jnp.stack(leaves), *scalar_profiles)
    assert_profile(scalar)

    batched = jax.jit(jax.vmap(read))(coordinates)
    assert_profile(batched)
    for transformed in (scalar, batched):
        for expected, actual in zip(
            jax.tree.leaves(eager), jax.tree.leaves(transformed), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
