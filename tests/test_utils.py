from lcm.utils import find_duplicates


def test_find_duplicates_singe_container_no_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5]) == set()


def test_find_duplicates_single_container_with_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5, 5]) == {5}


def test_find_duplicates_multiple_containers_no_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == set()


def test_find_duplicates_multiple_containers_with_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5, 5], [6, 7, 8, 9, 10, 5]) == {5}
