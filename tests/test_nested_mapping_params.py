from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp

from lcm.nested_mapping_params import NestedMappingParams
from lcm.utils import (
    ensure_containers_are_immutable,
    ensure_containers_are_mutable,
    flatten_regime_namespace,
)

# ======================================================================================
# Construction
# ======================================================================================


def test_construction_from_dict():
    nmp = NestedMappingParams({"a": 1, "b": 2})
    assert nmp.data == {"a": 1, "b": 2}


def test_construction_from_mapping_proxy_type():
    nmp = NestedMappingParams(MappingProxyType({"a": 1}))
    assert nmp.data == {"a": 1}


# ======================================================================================
# __repr__ and __eq__
# ======================================================================================


def test_repr():
    nmp = NestedMappingParams({"x": 1})
    assert repr(nmp) == "NestedMappingParams({'x': 1})"


def test_eq_equal():
    a = NestedMappingParams({"x": 1})
    b = NestedMappingParams({"x": 1})
    assert a == b


def test_eq_not_equal():
    a = NestedMappingParams({"x": 1})
    b = NestedMappingParams({"x": 2})
    assert a != b


def test_eq_not_nested_mapping_params():
    nmp = NestedMappingParams({"x": 1})
    assert nmp != {"x": 1}


# ======================================================================================
# JAX pytree
# ======================================================================================


def test_jax_tree_leaves():
    nmp = NestedMappingParams({"b": jnp.array(2.0), "a": jnp.array(1.0)})
    leaves = jax.tree.leaves(nmp)
    # sorted keys: a, b
    assert len(leaves) == 2
    assert float(leaves[0]) == 1.0
    assert float(leaves[1]) == 2.0


def test_jax_tree_map_roundtrip():
    nmp = NestedMappingParams({"x": jnp.array(1.0), "y": jnp.array(2.0)})
    doubled = jax.tree.map(lambda v: v * 2, nmp)
    assert isinstance(doubled, NestedMappingParams)
    assert float(doubled.data["x"]) == 2.0
    assert float(doubled.data["y"]) == 4.0


# ======================================================================================
# ensure_containers_are_immutable
# ======================================================================================


def test_immutable_dict_inside_nmp():
    nmp = NestedMappingParams({"a": 1})
    result = ensure_containers_are_immutable({"nmp": nmp})
    inner = result["nmp"]
    assert isinstance(inner, NestedMappingParams)
    assert isinstance(inner.data, MappingProxyType)


def test_immutable_nested_dicts_inside_nmp():
    nmp = NestedMappingParams({"outer": {"inner": 1}})
    result = ensure_containers_are_immutable({"nmp": nmp})
    inner = result["nmp"]
    assert isinstance(inner, NestedMappingParams)
    assert isinstance(inner.data, MappingProxyType)
    assert isinstance(inner.data["outer"], MappingProxyType)


def test_immutable_already_immutable_nmp():
    nmp = NestedMappingParams(MappingProxyType({"a": 1}))
    result = ensure_containers_are_immutable({"nmp": nmp})
    inner = result["nmp"]
    assert isinstance(inner, NestedMappingParams)
    assert isinstance(inner.data, MappingProxyType)


def test_immutable_nmp_nested_inside_larger_dict():
    data = {
        "regime": {
            "param": NestedMappingParams({"x": 1}),
        },
    }
    result = ensure_containers_are_immutable(data)
    assert isinstance(result["regime"], MappingProxyType)
    inner = result["regime"]["param"]
    assert isinstance(inner, NestedMappingParams)
    assert isinstance(inner.data, MappingProxyType)


# ======================================================================================
# ensure_containers_are_mutable
# ======================================================================================


def test_mutable_mapping_proxy_inside_nmp():
    nmp = NestedMappingParams(MappingProxyType({"a": 1}))
    result = ensure_containers_are_mutable({"nmp": nmp})
    inner = result["nmp"]
    assert isinstance(inner, NestedMappingParams)
    assert isinstance(inner.data, dict)


def test_mutable_nested_mapping_proxy_inside_nmp():
    nmp = NestedMappingParams(
        MappingProxyType({"outer": MappingProxyType({"inner": 1})})
    )
    result = ensure_containers_are_mutable({"nmp": nmp})
    inner = result["nmp"]
    assert isinstance(inner, NestedMappingParams)
    assert isinstance(inner.data, dict)
    assert isinstance(inner.data["outer"], dict)


def test_mutable_already_mutable_nmp():
    nmp = NestedMappingParams({"a": 1})
    result = ensure_containers_are_mutable({"nmp": nmp})
    inner = result["nmp"]
    assert isinstance(inner, NestedMappingParams)
    assert isinstance(inner.data, dict)


def test_mutable_nmp_nested_inside_mapping_proxy():
    data = MappingProxyType(
        {
            "regime": MappingProxyType(
                {
                    "param": NestedMappingParams(MappingProxyType({"x": 1})),
                }
            ),
        }
    )
    result = ensure_containers_are_mutable(data)
    assert isinstance(result["regime"], dict)
    inner = result["regime"]["param"]
    assert isinstance(inner, NestedMappingParams)
    assert isinstance(inner.data, dict)


# ======================================================================================
# Round-trips
# ======================================================================================


def test_roundtrip_mutable_immutable_mutable():
    original = {"nmp": NestedMappingParams({"a": 1, "b": {"c": 2}})}
    immutable = ensure_containers_are_immutable(original)
    mutable = ensure_containers_are_mutable(immutable)
    nmp = mutable["nmp"]
    assert isinstance(nmp, NestedMappingParams)
    assert isinstance(nmp.data, dict)
    assert isinstance(nmp.data["b"], dict)
    assert nmp.data["a"] == 1
    assert nmp.data["b"]["c"] == 2


def test_roundtrip_immutable_mutable_immutable():
    original = MappingProxyType(
        {
            "nmp": NestedMappingParams(MappingProxyType({"a": 1})),
        }
    )
    mutable = ensure_containers_are_mutable(original)
    immutable = ensure_containers_are_immutable(mutable)
    nmp = immutable["nmp"]
    assert isinstance(nmp, NestedMappingParams)
    assert isinstance(nmp.data, MappingProxyType)


# ======================================================================================
# Not a Mapping
# ======================================================================================


def test_not_a_mapping():
    nmp = NestedMappingParams({"a": 1})
    assert not isinstance(nmp, Mapping)


# ======================================================================================
# flatten_regime_namespace treats NMP as a leaf
# ======================================================================================


def test_flatten_regime_namespace_treats_nmp_as_leaf():
    nmp = NestedMappingParams({"x": 1, "y": 2})
    data = {"regime": {"param": nmp, "scalar": 3.0}}
    result = flatten_regime_namespace(data)
    assert result["regime__param"] is nmp
    assert result["regime__scalar"] == 3.0
