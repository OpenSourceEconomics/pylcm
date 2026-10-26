"""Original selected resources records satisfy the closed artifact boundary."""

import pickle
from dataclasses import fields, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_resource_query import OwnedResourcesRead
from lcm._solver_api.authority import _check_approved_artifact_containers
from lcm.solver_api import (
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactDescriptor,
    ArtifactKey,
    LeafAuthority,
    PersistencePolicy,
    TreePath,
)
from lcm.typing import ScalarFloat


class _CustomRead(NamedTuple):
    value: ScalarFloat


def test_owned_resources_record_preserves_keyed_leaves_and_closed_authority() -> None:
    selected = OwnedResourcesRead(
        value=jnp.asarray(7.0),
        marginal=jnp.asarray(2.0),
        policy=jnp.asarray(3.0),
        savings=jnp.asarray(4.0),
        native_owner=jnp.asarray(5, dtype=jnp.int32),
        side_id=jnp.asarray(-1, dtype=jnp.int32),
        corner=jnp.zeros((), dtype=jnp.bool_),
        feasible=jnp.ones((), dtype=jnp.bool_),
        valid=jnp.zeros((), dtype=jnp.bool_),
    )
    names = (
        "value",
        "marginal",
        "policy",
        "savings",
        "native_owner",
        "side_id",
        "corner",
        "feasible",
        "valid",
    )
    assert tuple(field.name for field in fields(selected)) == names
    with_paths, _ = jax.tree_util.tree_flatten_with_path(selected)
    assert tuple(path for path, _ in with_paths) == tuple(
        (jax.tree_util.GetAttrKey(name),) for name in names
    )
    abstract = jax.eval_shape(lambda: selected)
    restored = pickle.loads(pickle.dumps(selected))
    checked = jax.jit(lambda read: replace(read, valid=~read.valid))(selected)
    for name in names:
        original = getattr(selected, name)
        assert getattr(abstract, name) == jax.ShapeDtypeStruct(
            (), original.dtype, weak_type=jax.typeof(original).weak_type
        )
        np.testing.assert_array_equal(getattr(restored, name), original)
        np.testing.assert_array_equal(
            getattr(checked, name), ~original if name == "valid" else original
        )
    assert not selected.valid

    live = jnp.ones((), dtype=jnp.bool_)
    leaves: dict[TreePath, LeafAuthority] = {
        ("sequence:0", f"attribute:{name}"): LeafAuthority(
            path=("sequence:0", f"attribute:{name}"),
            runtime_type=jax.Array,
            shape=(),
            dtype=str(getattr(selected, name).dtype),
            axis_names=(),
        )
        for name in names
    }
    leaves[("sequence:1",)] = LeafAuthority(
        path=("sequence:1",),
        runtime_type=jax.Array,
        shape=(),
        dtype=str(live.dtype),
        axis_names=(),
    )
    authority = ArtifactAuthority(
        descriptor=ArtifactDescriptor(
            key=ArtifactKey(type_id="tests.owned_resources_read", schema_version=1),
            channel=ArtifactChannel.REPLAY,
            persistence=PersistencePolicy.MODEL_VERIFIABLE,
            payload_type_id="builtins.tuple",
            leaf_descriptors=tuple(leaf.descriptor for leaf in leaves.values()),
        ),
        payload_runtime_type=tuple,
        template=(selected, live),
        container_runtime_types={(): tuple, ("sequence:0",): OwnedResourcesRead},
        leaves=leaves,
    )
    snapshot = authority.template
    assert isinstance(snapshot, tuple)
    assert type(snapshot[0]) is OwnedResourcesRead
    for original, detached in zip(
        jax.tree.leaves((selected, live)), jax.tree.leaves(snapshot), strict=True
    ):
        assert detached is not original
        np.testing.assert_array_equal(detached, original)

    with pytest.raises(TypeError, match="unsupported _CustomRead"):
        _check_approved_artifact_containers(
            payload_runtime_type=tuple,
            container_runtime_types={(): tuple, ("sequence:0",): _CustomRead},
        )
