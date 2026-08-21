"""The slow solution battery is partitioned exhaustively across CI jobs."""

from pathlib import Path

import pytest

from tests.ci.shard_test_files import assign_test_files


def test_solution_shards_cover_each_test_file_once() -> None:
    """Every slow solution module belongs to exactly one non-empty CI shard."""
    files = tuple(sorted(Path("tests/solution").glob("test_*.py")))

    groups = assign_test_files(files=files, n_shards=3)

    assert all(groups)
    assert sorted(path for group in groups for path in group) == list(files)


def test_solution_shards_are_independent_of_input_order() -> None:
    """Repository traversal order does not change a file's assigned shard."""
    files = tuple(Path(f"tests/solution/test_case_{i}.py") for i in range(12))

    forward = assign_test_files(files=files, n_shards=3)
    reverse = assign_test_files(files=tuple(reversed(files)), n_shards=3)

    assert forward == reverse


@pytest.mark.parametrize("n_shards", [0, -1])
def test_solution_shards_require_a_positive_shard_count(n_shards: int) -> None:
    """A non-positive shard count is rejected instead of dropping every file."""
    with pytest.raises(ValueError, match="positive"):
        assign_test_files(
            files=(Path("tests/solution/test_case.py"),),
            n_shards=n_shards,
        )
