from types import MappingProxyType

from _lcm.utils.containers import (
    ensure_containers_are_immutable,
    find_duplicates,
)


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


def test_find_duplicates_singe_container_no_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5]) == set()


def test_find_duplicates_single_container_with_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5, 5]) == {5}


def test_find_duplicates_multiple_containers_no_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == set()


def test_find_duplicates_multiple_containers_with_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5, 5], [6, 7, 8, 9, 10, 5]) == {5}
