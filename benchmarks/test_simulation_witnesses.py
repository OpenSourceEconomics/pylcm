"""Tests for the shared forward-simulation witnesses."""

import pytest

from benchmarks.asv._simulation_witnesses import WITNESSES
from lcm.execution import ExecutionConfig


@pytest.mark.parametrize("witness", sorted(WITNESSES))
def test_witness_builds_under_an_explicit_budget_on_an_unpreallocated_pool(
    *, monkeypatch: pytest.MonkeyPatch, witness: str
) -> None:
    """An explicit budget is admitted where the device default would refuse."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    monkeypatch.setattr(
        "lcm.model.visible_device_pool_limits",
        lambda: {0: 48_000_000_000},
    )

    model, _, _ = WITNESSES[witness](
        execution_config=ExecutionConfig(device_memory_bytes=1_000_000_000)
    )

    assert model._execution.budget_source == "explicit"  # noqa: SLF001
