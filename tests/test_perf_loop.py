"""Benchmark fingerprints preserve values from legacy solution mappings."""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.perf_loop import _block, _values


@pytest.mark.parametrize("read_only", [False, True])
def test_legacy_solution_mapping_preserves_its_value_leaves(*, read_only: bool) -> None:
    """Each period/regime value reaches the fingerprint under its stable label."""
    values = {0: {"alive": jnp.asarray([2.0, 3.0])}}
    result = MappingProxyType(values) if read_only else values

    _block(result)
    got = _values(result)

    assert set(got) == {"0/alive"}
    np.testing.assert_array_equal(got["0/alive"], [2.0, 3.0])
