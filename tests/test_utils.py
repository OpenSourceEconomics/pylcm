from types import MappingProxyType

from lcm.utils import (
    ensure_containers_are_immutable,
    ensure_containers_are_mutable,
    find_duplicates,
)

# ======================================================================================
# Tests for ensure_containers_are_immutable
# ======================================================================================


def test_ensure_containers_are_immutable_simple_dict():
    """Test conversion of a simple dict to MappingProxyType."""
    data = {"a": 1, "b": 2}
    result = ensure_containers_are_immutable(data)
    assert isinstance(result, MappingProxyType)
    assert result["a"] == 1
    assert result["b"] == 2


def test_ensure_containers_are_immutable_nested_dict():
    """Test conversion of nested dicts to MappingProxyType."""
    data = {"outer": {"inner": 1}}
    result = ensure_containers_are_immutable(data)
    assert isinstance(result, MappingProxyType)
    assert isinstance(result["outer"], MappingProxyType)
    assert result["outer"]["inner"] == 1


def test_ensure_containers_are_immutable_list_to_tuple():
    """Test conversion of lists to tuples."""
    data = {"items": [1, 2, 3]}
    result = ensure_containers_are_immutable(data)
    assert isinstance(result["items"], tuple)
    assert result["items"] == (1, 2, 3)


def test_ensure_containers_are_immutable_nested_list():
    """Test conversion of nested lists to tuples."""
    data = {"items": [[1, 2], [3, 4]]}
    result = ensure_containers_are_immutable(data)
    assert isinstance(result["items"], tuple)
    assert isinstance(result["items"][0], tuple)
    assert result["items"] == ((1, 2), (3, 4))


def test_ensure_containers_are_immutable_set_to_frozenset():
    """Test conversion of sets to frozensets."""
    data = {"items": {1, 2, 3}}
    result = ensure_containers_are_immutable(data)
    assert isinstance(result["items"], frozenset)
    assert result["items"] == frozenset({1, 2, 3})


def test_ensure_containers_are_immutable_already_immutable():
    """Test that already immutable containers are preserved."""
    data = MappingProxyType({"a": (1, 2), "b": frozenset({3, 4})})
    result = ensure_containers_are_immutable(data)
    assert isinstance(result, MappingProxyType)
    assert isinstance(result["a"], tuple)
    assert isinstance(result["b"], frozenset)


def test_ensure_containers_are_immutable_mixed():
    """Test conversion of mixed nested structures."""
    data = {
        "dict": {"nested": 1},
        "list": [1, 2],
        "set": {3, 4},
        "scalar": 5,
    }
    result = ensure_containers_are_immutable(data)
    assert isinstance(result, MappingProxyType)
    assert isinstance(result["dict"], MappingProxyType)
    assert isinstance(result["list"], tuple)
    assert isinstance(result["set"], frozenset)
    assert result["scalar"] == 5


# ======================================================================================
# Tests for ensure_containers_are_mutable
# ======================================================================================


def test_ensure_containers_are_mutable_simple_mapping_proxy():
    """Test conversion of a simple MappingProxyType to dict."""
    data = MappingProxyType({"a": 1, "b": 2})
    result = ensure_containers_are_mutable(data)
    assert isinstance(result, dict)
    assert result["a"] == 1
    assert result["b"] == 2


def test_ensure_containers_are_mutable_nested_mapping_proxy():
    """Test conversion of nested MappingProxyType to dict."""
    data = MappingProxyType({"outer": MappingProxyType({"inner": 1})})
    result = ensure_containers_are_mutable(data)
    assert isinstance(result, dict)
    assert isinstance(result["outer"], dict)
    assert result["outer"]["inner"] == 1


def test_ensure_containers_are_mutable_tuple_to_list():
    """Test conversion of tuples to lists."""
    data = MappingProxyType({"items": (1, 2, 3)})
    result = ensure_containers_are_mutable(data)
    assert isinstance(result["items"], list)
    assert result["items"] == [1, 2, 3]


def test_ensure_containers_are_mutable_nested_tuple():
    """Test conversion of nested tuples to lists."""
    data = MappingProxyType({"items": ((1, 2), (3, 4))})
    result = ensure_containers_are_mutable(data)
    assert isinstance(result["items"], list)
    assert isinstance(result["items"][0], list)
    assert result["items"] == [[1, 2], [3, 4]]


def test_ensure_containers_are_mutable_frozenset_to_set():
    """Test conversion of frozensets to sets."""
    data = MappingProxyType({"items": frozenset({1, 2, 3})})
    result = ensure_containers_are_mutable(data)
    assert isinstance(result["items"], set)
    assert result["items"] == {1, 2, 3}


def test_ensure_containers_are_mutable_already_mutable():
    """Test that already mutable containers are preserved."""
    data = {"a": [1, 2], "b": {3, 4}}
    result = ensure_containers_are_mutable(data)
    assert isinstance(result, dict)
    assert isinstance(result["a"], list)
    assert isinstance(result["b"], set)


def test_ensure_containers_are_mutable_mixed():
    """Test conversion of mixed nested structures."""
    data = MappingProxyType(
        {
            "mapping": MappingProxyType({"nested": 1}),
            "tuple": (1, 2),
            "frozenset": frozenset({3, 4}),
            "scalar": 5,
        }
    )
    result = ensure_containers_are_mutable(data)
    assert isinstance(result, dict)
    assert isinstance(result["mapping"], dict)
    assert isinstance(result["tuple"], list)
    assert isinstance(result["frozenset"], set)
    assert result["scalar"] == 5


# ======================================================================================
# Tests for round-trip conversion
# ======================================================================================


def test_immutable_then_mutable_roundtrip():
    """Test that immutable -> mutable produces equivalent structure."""
    original = {"dict": {"nested": 1}, "list": [1, 2], "set": {3, 4}, "scalar": 5}
    immutable = ensure_containers_are_immutable(original)
    mutable = ensure_containers_are_mutable(immutable)

    assert mutable["dict"]["nested"] == 1  # ty: ignore[invalid-argument-type,not-subscriptable]
    assert mutable["list"] == [1, 2]
    assert mutable["set"] == {3, 4}
    assert mutable["scalar"] == 5


def test_mutable_then_immutable_roundtrip():
    """Test that mutable -> immutable produces equivalent structure."""
    original = MappingProxyType(
        {
            "mapping": MappingProxyType({"nested": 1}),
            "tuple": (1, 2),
            "frozenset": frozenset({3, 4}),
            "scalar": 5,
        }
    )
    mutable = ensure_containers_are_mutable(original)
    immutable = ensure_containers_are_immutable(mutable)

    assert immutable["mapping"]["nested"] == 1
    assert immutable["tuple"] == (1, 2)
    assert immutable["frozenset"] == frozenset({3, 4})
    assert immutable["scalar"] == 5


# ======================================================================================
# Tests for find_duplicates
# ======================================================================================


def test_find_duplicates_singe_container_no_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5]) == set()


def test_find_duplicates_single_container_with_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5, 5]) == {5}


def test_find_duplicates_multiple_containers_no_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == set()


def test_find_duplicates_multiple_containers_with_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5, 5], [6, 7, 8, 9, 10, 5]) == {5}
