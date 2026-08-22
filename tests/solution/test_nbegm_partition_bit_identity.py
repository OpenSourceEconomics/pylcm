"""Distinct admitted NBEGM branch strides publish the same floating values."""

from collections.abc import Mapping
from functools import lru_cache

import numpy as np

from tests.test_models import nbegm_multi_discrete_toy as toy


@lru_cache
def _solve(*, branch_batch_size: int) -> Mapping[int, Mapping]:
    """Solve a 20-branch model at one genuinely admitted branch stride."""
    model = toy.build_model(
        variant="nbegm",
        n_actions=3,
        n_periods=3,
        n_liquid=12,
        n_consumption=12,
        liquid_max=20.0,
        n_savings=16,
        savings_max=18.0,
        envelope_arithmetic="ordinary",
        branch_batch_size=branch_batch_size,
    )
    return model.solve(params=toy.build_params(n_actions=3), log_level="debug")


def _published_pairs() -> tuple[np.ndarray, np.ndarray]:
    """Flatten solves at admitted strides four and eight into aligned arrays."""
    stride_four = _solve(branch_batch_size=4)
    stride_eight = _solve(branch_batch_size=8)
    assert stride_four.keys() == stride_eight.keys()
    left = []
    right = []
    for period in stride_four:
        assert stride_four[period].keys() == stride_eight[period].keys()
        for name in stride_four[period]:
            a = np.asarray(stride_four[period][name])
            b = np.asarray(stride_eight[period][name])
            if not np.issubdtype(a.dtype, np.floating):
                continue
            left.append(a.ravel())
            right.append(b.ravel())
    assert left, "no floating arrays were compared"
    return np.concatenate(left), np.concatenate(right)


def test_distinct_admitted_branch_strides_publish_bit_identical_values() -> None:
    """Admitted strides four and eight leave every published value unchanged."""
    stride_four, stride_eight = _published_pairs()
    np.testing.assert_array_equal(stride_four, stride_eight)
