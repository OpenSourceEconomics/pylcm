from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp

from lcm.params import MappingLeaf
from lcm.utils import (
    ensure_containers_are_immutable,
    ensure_containers_are_mutable,
    flatten_regime_namespace,
)

# ======================================================================================
# Construction
# ======================================================================================


def test_construction_from_dict():
    leaf = MappingLeaf({"a": 1, "b": 2})
    assert leaf.data == {"a": 1, "b": 2}


def test_construction_from_mapping_proxy_type():
    leaf = MappingLeaf(MappingProxyType({"a": 1}))
    assert leaf.data == {"a": 1}


# ======================================================================================
# __repr__ and __eq__
# ======================================================================================


def test_repr():
    leaf = MappingLeaf({"x": 1})
    assert repr(leaf) == "MappingLeaf({'x': 1})"


def test_eq_equal():
    a = MappingLeaf({"x": 1})
    b = MappingLeaf({"x": 1})
    assert a == b


def test_eq_not_equal():
    a = MappingLeaf({"x": 1})
    b = MappingLeaf({"x": 2})
    assert a != b


def test_eq_not_mapping_leaf():
    leaf = MappingLeaf({"x": 1})
    assert leaf != {"x": 1}


# ======================================================================================
# JAX pytree
# ======================================================================================


def test_jax_tree_leaves():
    leaf = MappingLeaf({"b": jnp.array(2.0), "a": jnp.array(1.0)})
    leaves = jax.tree.leaves(leaf)
    # sorted keys: a, b
    assert len(leaves) == 2
    assert float(leaves[0]) == 1.0
    assert float(leaves[1]) == 2.0


def test_jax_tree_map_roundtrip():
    leaf = MappingLeaf({"x": jnp.array(1.0), "y": jnp.array(2.0)})
    doubled = jax.tree.map(lambda v: v * 2, leaf)
    assert isinstance(doubled, MappingLeaf)
    assert float(doubled.data["x"]) == 2.0
    assert float(doubled.data["y"]) == 4.0


# ======================================================================================
# ensure_containers_are_immutable
# ======================================================================================


def test_immutable_dict_inside_mapping_leaf():
    leaf = MappingLeaf({"a": 1})
    result = ensure_containers_are_immutable({"leaf": leaf})
    inner = result["leaf"]
    assert isinstance(inner, MappingLeaf)
    assert isinstance(inner.data, MappingProxyType)


def test_immutable_nested_dicts_inside_mapping_leaf():
    leaf = MappingLeaf({"outer": {"inner": 1}})
    result = ensure_containers_are_immutable({"leaf": leaf})
    inner = result["leaf"]
    assert isinstance(inner, MappingLeaf)
    assert isinstance(inner.data, MappingProxyType)
    assert isinstance(inner.data["outer"], MappingProxyType)


def test_immutable_already_immutable_mapping_leaf():
    leaf = MappingLeaf(MappingProxyType({"a": 1}))
    result = ensure_containers_are_immutable({"leaf": leaf})
    inner = result["leaf"]
    assert isinstance(inner, MappingLeaf)
    assert isinstance(inner.data, MappingProxyType)


def test_immutable_mapping_leaf_nested_inside_larger_dict():
    data = {
        "regime": {
            "param": MappingLeaf({"x": 1}),
        },
    }
    result = ensure_containers_are_immutable(data)
    assert isinstance(result["regime"], MappingProxyType)
    inner = result["regime"]["param"]
    assert isinstance(inner, MappingLeaf)
    assert isinstance(inner.data, MappingProxyType)


# ======================================================================================
# ensure_containers_are_mutable
# ======================================================================================


def test_mutable_mapping_proxy_inside_mapping_leaf():
    leaf = MappingLeaf(MappingProxyType({"a": 1}))
    result = ensure_containers_are_mutable({"leaf": leaf})
    inner = result["leaf"]
    assert isinstance(inner, MappingLeaf)
    assert isinstance(inner.data, dict)


def test_mutable_nested_mapping_proxy_inside_mapping_leaf():
    leaf = MappingLeaf(MappingProxyType({"outer": MappingProxyType({"inner": 1})}))
    result = ensure_containers_are_mutable({"leaf": leaf})
    inner = result["leaf"]
    assert isinstance(inner, MappingLeaf)
    assert isinstance(inner.data, dict)
    assert isinstance(inner.data["outer"], dict)


def test_mutable_already_mutable_mapping_leaf():
    leaf = MappingLeaf({"a": 1})
    result = ensure_containers_are_mutable({"leaf": leaf})
    inner = result["leaf"]
    assert isinstance(inner, MappingLeaf)
    assert isinstance(inner.data, dict)


def test_mutable_mapping_leaf_nested_inside_mapping_proxy():
    data = MappingProxyType(
        {
            "regime": MappingProxyType(
                {
                    "param": MappingLeaf(MappingProxyType({"x": 1})),
                }
            ),
        }
    )
    result = ensure_containers_are_mutable(data)
    assert isinstance(result["regime"], dict)
    inner = result["regime"]["param"]
    assert isinstance(inner, MappingLeaf)
    assert isinstance(inner.data, dict)


# ======================================================================================
# Round-trips
# ======================================================================================


def test_roundtrip_mutable_immutable_mutable():
    original = {"leaf": MappingLeaf({"a": 1, "b": {"c": 2}})}
    immutable = ensure_containers_are_immutable(original)
    mutable = ensure_containers_are_mutable(immutable)
    leaf = mutable["leaf"]
    assert isinstance(leaf, MappingLeaf)
    assert isinstance(leaf.data, dict)
    assert isinstance(leaf.data["b"], dict)
    assert leaf.data["a"] == 1
    assert leaf.data["b"]["c"] == 2


def test_roundtrip_immutable_mutable_immutable():
    original = MappingProxyType(
        {
            "leaf": MappingLeaf(MappingProxyType({"a": 1})),
        }
    )
    mutable = ensure_containers_are_mutable(original)
    immutable = ensure_containers_are_immutable(mutable)
    leaf = immutable["leaf"]
    assert isinstance(leaf, MappingLeaf)
    assert isinstance(leaf.data, MappingProxyType)


# ======================================================================================
# Not a Mapping
# ======================================================================================


def test_not_a_mapping():
    leaf = MappingLeaf({"a": 1})
    assert not isinstance(leaf, Mapping)


# ======================================================================================
# flatten_regime_namespace treats MappingLeaf as a leaf
# ======================================================================================


def test_flatten_regime_namespace_treats_mapping_leaf_as_leaf():
    leaf = MappingLeaf({"x": 1, "y": 2})
    data = {"regime": {"param": leaf, "scalar": 3.0}}
    result = flatten_regime_namespace(data)
    assert result["regime__param"] is leaf
    assert result["regime__scalar"] == 3.0
