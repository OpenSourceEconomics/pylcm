"""Exercise the finite witness on eight actual A40 devices before measurement."""

import json
from typing import Any

import gpu_admission
import gpu_assets
import jax
import pytest


def test_gpu_topology(record_property: Any) -> None:
    """Require one process owning eight distinct physical A40 devices."""
    assert jax.default_backend() == "gpu"
    devices = jax.devices()
    assert len(devices) == jax.local_device_count() == 8
    assert len({device.id for device in devices}) == 8
    assert all(
        device.platform == "gpu" and "A40" in device.device_kind for device in devices
    )
    record_property("selected_device_count", gpu_assets.SCALING_DEVICES)
    record_property("devices", json.dumps([str(device) for device in devices]))


def test_gpu_finite_values_decisions_rng_and_shards(
    *, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    """Preserve exact decisions, eight-ULP values, global reads and 17 real rows."""
    gpu_assets.test_eight_assets_shards_use_full_reads_and_match_exact_bellman_reference(
        widths=(1, 1), monkeypatch=monkeypatch, record_property=record_property
    )


@pytest.mark.parametrize("shared", [False, True])
def test_gpu_pruned_threshold(
    *, shared: bool, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    """Refuse below native compiler plus retained, destination and scratch storage."""
    original = gpu_admission._case  # noqa: SLF001 - record sealed fixture accounting

    def record_case(*, shared: bool) -> Any:
        case = original(shared=shared)
        for name in ("compiler_bytes", "owner_bytes", "replica_bytes", "scratch_bytes"):
            record_property(name, getattr(case, name))
        return case

    monkeypatch.setattr(gpu_admission, "_case", record_case)
    gpu_admission.test_pruned_full_replicas_and_scratch_have_an_exact_admission_threshold(
        shared=shared, monkeypatch=monkeypatch
    )


def test_gpu_interrupted_copy_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain actual partial copies until the pending-work owner closes."""
    gpu_admission.test_interrupted_all_gather_materialization_keeps_copy_until_owner_close(
        monkeypatch=monkeypatch
    )


def test_gpu_shared_last_consumer(
    *, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    """Wait for both consumers before releasing the real shared GPU copy."""
    gpu_assets.test_shared_native_all_gather_releases_after_both_consumers_are_ready(
        monkeypatch=monkeypatch, record_property=record_property
    )
