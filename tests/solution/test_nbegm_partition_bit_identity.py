"""NBEGM's published values do not depend on how the branch axis is partitioned.

`branch_batch_size` chooses how many discrete-action branches are held in
flight; it changes no operation and no operand order, so the solution it
publishes is the same real number either way. Bit identity is the stronger
claim that the two are also the same *floating* number — which holds only once
the partition stops being part of the compilation key, so that every partition
runs the same executable rather than a differently vectorized one.

`test_nbegm_branch_batch_size.py` states the weaker claim that currently holds,
bounded in ULP. This module states the target.
"""

import numpy as np
import pytest

from tests.solution.test_nbegm_branch_batch_size import _solve


def _published_pairs() -> tuple[np.ndarray, np.ndarray]:
    """Flatten the two solves into one aligned pair of floating arrays."""
    whole = _solve(branch_batch_size=0)
    streamed = _solve(branch_batch_size=1)
    assert whole.keys() == streamed.keys()
    left = []
    right = []
    for period in whole:
        assert whole[period].keys() == streamed[period].keys()
        for name in whole[period]:
            a = np.asarray(whole[period][name])
            b = np.asarray(streamed[period][name])
            if not np.issubdtype(a.dtype, np.floating):
                continue
            left.append(a.ravel())
            right.append(b.ravel())
    assert left, "no floating arrays were compared"
    return np.concatenate(left), np.concatenate(right)


@pytest.mark.xfail(
    strict=True,
    reason="The branch partition is still part of the compilation key, so XLA "
    "emits a differently vectorized kernel per block width. Remove this marker "
    "with the fixed-graph construction, which makes the partition a runtime "
    "operand of one executable.",
)
def test_branch_partition_publishes_bit_identical_values() -> None:
    """Partitioning the branch axis leaves every published value bit for bit equal."""
    streamed, whole = _published_pairs()
    np.testing.assert_array_equal(streamed, whole)
