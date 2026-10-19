"""Savings support keeps native contracts through compiler tree descriptions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.egm.one_asset_egm_step import (
    _EGMSavingsSupport,
    _unflatten_egm_savings_support,
)


def test_savings_support_preserves_named_leaves_through_abstract_roundtrip() -> None:
    """Compiler descriptors retain field order and restore the original buffers."""
    floor = jnp.asarray(0.25)
    active = jnp.asarray([True, False, True])
    candidate_id = jnp.asarray([0, -1, 1], dtype=jnp.int32)
    support = _EGMSavingsSupport(floor=floor, active=active, candidate_id=candidate_id)
    original_leaves, definition = jax.tree.flatten(support)
    assert len(original_leaves) == 3
    assert all(
        observed is expected
        for observed, expected in zip(
            original_leaves, (floor, active, candidate_id), strict=True
        )
    )

    abstract = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape,
            leaf.dtype,
            weak_type=leaf.weak_type,
            sharding=leaf.sharding,
        ),
        support,
    )
    assert type(abstract) is _EGMSavingsSupport
    keyed, abstract_definition = jax.tree_util.tree_flatten_with_path(abstract)
    assert abstract_definition == definition
    assert tuple(tuple(key.name for key in path) for path, _ in keyed) == (
        ("floor",),
        ("active",),
        ("candidate_id",),
    )
    assert all(type(path[0]) is jax.tree_util.GetAttrKey for path, _ in keyed)
    for (_, described), original in zip(keyed, original_leaves, strict=True):
        assert type(described) is jax.ShapeDtypeStruct
        assert described.shape == original.shape
        assert described.dtype == original.dtype
        assert described.weak_type == original.weak_type
        assert described.sharding == original.sharding
    assert abstract.floor.shape == ()
    assert abstract.active.shape == abstract.candidate_id.shape == (3,)
    assert abstract.active.dtype == jnp.bool_
    assert abstract.candidate_id.dtype == jnp.int32

    restored = jax.tree.unflatten(abstract_definition, original_leaves)
    assert type(restored) is _EGMSavingsSupport
    assert restored.floor is floor
    assert restored.active is active
    assert restored.candidate_id is candidate_id
    np.testing.assert_array_equal(restored.floor, 0.25)
    np.testing.assert_array_equal(restored.active, [True, False, True])
    np.testing.assert_array_equal(restored.candidate_id, [0, -1, 1])


def test_savings_support_constructor_requires_a_scalar_floor() -> None:
    """A runtime floor cannot be a rank-one array."""
    with pytest.raises(BeartypeCallHintParamViolation, match="parameter floor"):
        _EGMSavingsSupport(
            floor=jnp.asarray([0.25]),
            active=jnp.asarray([True]),
            candidate_id=jnp.asarray([0], dtype=jnp.int32),
        )


@pytest.mark.parametrize("leaf_count", [2, 4], ids=["missing", "extra"])
def test_savings_support_reconstruction_requires_all_three_fields(
    *, leaf_count: int
) -> None:
    """Compiler reconstruction rejects missing or extra associated leaves."""
    with pytest.raises(
        TypeError, match="Savings support requires all its associated fields"
    ):
        _unflatten_egm_savings_support(_aux=None, children=(None,) * leaf_count)
