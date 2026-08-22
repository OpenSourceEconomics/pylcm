"""NBEGM's published values do not depend on how the branch axis is partitioned.

`branch_batch_size` changes the fixed-window loop's commit stride. It changes no
operation, operand order, compiled window, or workspace, so every admitted stride
runs the same executable and publishes the same floating-point value.
"""

import numpy as np

from tests.solution.test_nbegm_branch_batch_size import _solve


def _published_pairs() -> tuple[np.ndarray, np.ndarray]:
    """Flatten the two solves into one aligned pair of floating arrays."""
    default = _solve(branch_batch_size=0)
    stride_one = _solve(branch_batch_size=1)
    assert default.keys() == stride_one.keys()
    left = []
    right = []
    for period in default:
        assert default[period].keys() == stride_one[period].keys()
        for name in default[period]:
            a = np.asarray(default[period][name])
            b = np.asarray(stride_one[period][name])
            if not np.issubdtype(a.dtype, np.floating):
                continue
            left.append(a.ravel())
            right.append(b.ravel())
    assert left, "no floating arrays were compared"
    return np.concatenate(left), np.concatenate(right)


def test_branch_partition_publishes_bit_identical_values() -> None:
    """Partitioning the branch axis leaves every published value bit for bit equal."""
    stride_one, default = _published_pairs()
    np.testing.assert_array_equal(stride_one, default)
