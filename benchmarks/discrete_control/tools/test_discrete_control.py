"""Existing distributed-versus-canonical value contract on four physical GPUs."""

import json
import os

# The original fixture helpers are deliberately reused unchanged.
# ruff: noqa: SLF001
from typing import Any

import discrete_fixture as fixture
import jax
import numpy as np
import pytest
from value_contract import assert_agrees_to_ulp


def test_four_device_discrete_values_and_coverage(record_property: Any) -> None:
    """Retain the existing nine-array roster and eight-ULP placement invariant."""
    jax.config.update("jax_default_matmul_precision", "highest")
    assert jax.default_backend() == "gpu"
    assert len(jax.devices()) == jax.local_device_count() == 4
    assert all("A40" in d.device_kind for d in jax.devices())
    precision = int(os.environ["PYLCM_B1_PRECISION"])
    assert jax.config.jax_enable_x64 == (precision == 64)
    expected_dtype = np.dtype(f"float{precision}")
    reference = np.array([1.0], dtype=expected_dtype)
    perturbed = reference.copy()
    for _ in range(9):
        perturbed = np.nextafter(perturbed, np.array([np.inf], dtype=expected_dtype))
    with pytest.raises(AssertionError, match="above the 8 allowed"):
        assert_agrees_to_ulp(got=perturbed, expected=reference, n_ulp=8)
    placed = fixture._make_three_type_model(
        distributed=True, devices=(0, 1, 2, 3), budget_bytes=134217728
    ).solve(params=fixture._PARAMS, log_level="off")
    canonical = fixture._make_three_type_model(
        distributed=False, devices=(0, 1, 2, 3), budget_bytes=134217728
    ).solve(params=fixture._PARAMS, log_level="off")
    roster = {t: ({"working", "retired"} if t < 4 else {"retired"}) for t in range(5)}
    assert {t: set(rs) for t, rs in placed.values.items()} == roster
    assert {t: set(rs) for t, rs in canonical.values.items()} == roster
    observations = []
    for t, regimes in placed.values.items():
        for regime, value in regimes.items():
            data = np.asarray(value)
            expected = np.asarray(canonical.values[t][regime])
            assert data.shape == expected.shape
            assert data.dtype == expected.dtype == expected_dtype
            assert np.isfinite(data).all()
            assert np.isfinite(expected).all()
            assert_agrees_to_ulp(got=data, expected=expected, n_ulp=8)
            devices = {0, 1, 2} if regime == "working" else {3}
            assert {d.id for d in value.sharding.device_set} == devices
            coverage = np.zeros(value.shape, dtype=np.int8)
            for shard in value.addressable_shards:
                assert shard.data.size > 0
                coverage[shard.index] += 1
            np.testing.assert_array_equal(coverage, np.ones(value.shape, dtype=np.int8))
            observations.append(
                {
                    "period": t,
                    "regime": regime,
                    "shape": value.shape,
                    "devices": sorted(devices),
                    "nonempty_shards": len(value.addressable_shards),
                }
            )
    record_property("arrays_checked", 9)
    record_property("coverage", json.dumps(observations))
