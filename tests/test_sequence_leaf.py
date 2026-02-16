from collections.abc import Sequence
from types import MappingProxyType

import jax
import jax.numpy as jnp

from lcm.params.sequence_leaf import SequenceLeaf
from lcm.utils import (
    ensure_containers_are_immutable,
    ensure_containers_are_mutable,
    flatten_regime_namespace,
)

# ======================================================================================
# Construction
# ======================================================================================


def test_construction_from_list():
    leaf = SequenceLeaf([1, 2, 3])
    assert isinstance(leaf.data, tuple)
    assert leaf.data == (1, 2, 3)


def test_construction_from_tuple():
    leaf = SequenceLeaf((1, 2, 3))
    assert isinstance(leaf.data, tuple)
    assert leaf.data == (1, 2, 3)


def test_construction_freezes_nested_dicts():
    leaf = SequenceLeaf([{"a": 1}])
    assert isinstance(leaf.data[0], MappingProxyType)


def test_construction_freezes_nested_lists():
    leaf = SequenceLeaf([[1, 2]])
    assert isinstance(leaf.data[0], tuple)


# ======================================================================================
# __repr__ and __eq__
# ======================================================================================


def test_repr():
    leaf = SequenceLeaf([1, 2])
    assert repr(leaf) == "SequenceLeaf([1, 2])"


def test_eq_equal():
    a = SequenceLeaf([1, 2])
    b = SequenceLeaf([1, 2])
    assert a == b


def test_eq_not_equal():
    a = SequenceLeaf([1, 2])
    b = SequenceLeaf([3, 4])
    assert a != b


def test_eq_not_sequence_leaf():
    leaf = SequenceLeaf([1, 2])
    assert leaf != [1, 2]


# ======================================================================================
# __hash__
# ======================================================================================


def test_hashable():
    leaf = SequenceLeaf([1, 2, 3])
    assert hash(leaf) == hash((1, 2, 3))


def test_equal_leaves_have_same_hash():
    a = SequenceLeaf([1, 2])
    b = SequenceLeaf([1, 2])
    assert hash(a) == hash(b)


def test_usable_as_dict_key():
    leaf = SequenceLeaf([1, 2])
    d = {leaf: "value"}
    assert d[SequenceLeaf([1, 2])] == "value"


# ======================================================================================
# JAX pytree
# ======================================================================================


def test_jax_tree_leaves():
    leaf = SequenceLeaf([jnp.array(1.0), jnp.array(2.0)])
    leaves = jax.tree.leaves(leaf)
    assert len(leaves) == 2
    assert float(leaves[0]) == 1.0
    assert float(leaves[1]) == 2.0


def test_jax_tree_map_roundtrip():
    leaf = SequenceLeaf([jnp.array(1.0), jnp.array(2.0)])
    doubled = jax.tree.map(lambda v: v * 2, leaf)
    assert isinstance(doubled, SequenceLeaf)
    assert float(doubled.data[0]) == 2.0
    assert float(doubled.data[1]) == 4.0


# ======================================================================================
# ensure_containers_are_immutable (no-op for leaf data)
# ======================================================================================


def test_immutable_sequence_leaf_already_frozen():
    leaf = SequenceLeaf([1, 2])
    result = ensure_containers_are_immutable({"leaf": leaf})
    inner = result["leaf"]
    assert isinstance(inner, SequenceLeaf)
    assert isinstance(inner.data, tuple)


def test_immutable_nested_dicts_inside_sequence_leaf():
    leaf = SequenceLeaf([{"inner": 1}])
    result = ensure_containers_are_immutable({"leaf": leaf})
    inner = result["leaf"]
    assert isinstance(inner, SequenceLeaf)
    assert isinstance(inner.data, tuple)
    assert isinstance(inner.data[0], MappingProxyType)


def test_immutable_sequence_leaf_nested_inside_larger_dict():
    data = {
        "regime": {
            "param": SequenceLeaf([1, 2]),
        },
    }
    result = ensure_containers_are_immutable(data)
    inner = result["regime"]["param"]
    assert isinstance(inner, SequenceLeaf)
    assert isinstance(inner.data, tuple)


# ======================================================================================
# ensure_containers_are_mutable (unwraps leaf to plain list)
# ======================================================================================


def test_mutable_unwraps_sequence_leaf_to_list():
    leaf = SequenceLeaf([1, 2])
    result = ensure_containers_are_mutable({"leaf": leaf})
    assert isinstance(result["leaf"], list)
    assert result["leaf"] == [1, 2]


# ======================================================================================
# Round-trips
# ======================================================================================


def test_roundtrip_immutable_to_mutable():
    original = {"leaf": SequenceLeaf([1, {"a": 2}])}
    immutable = ensure_containers_are_immutable(original)
    mutable = ensure_containers_are_mutable(immutable)
    inner = mutable["leaf"]
    assert isinstance(inner, list)
    assert inner[0] == 1
    assert isinstance(inner[1], dict)
    assert inner[1] == {"a": 2}


# ======================================================================================
# Not a Sequence
# ======================================================================================


def test_not_a_sequence():
    leaf = SequenceLeaf([1, 2])
    assert not isinstance(leaf, Sequence)


# ======================================================================================
# flatten_regime_namespace treats SequenceLeaf as a leaf
# ======================================================================================


def test_flatten_regime_namespace_treats_sequence_leaf_as_leaf():
    leaf = SequenceLeaf([1, 2])
    data = {"regime": {"param": leaf, "scalar": 3.0}}
    result = flatten_regime_namespace(data)
    assert result["regime__param"] is leaf
    assert result["regime__scalar"] == 3.0
