"""Projected residency preserves byte accounting without revisiting other devices."""

from typing import cast
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import residency
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from lcm.exceptions import ExecutionPlanningError


def _no_op() -> None:
    """Module-level pure callable accepted by the profiled operation boundary."""


def _device(*, platform: str, index: int) -> jax.Device:
    return cast("jax.Device", Mock(spec=jax.Device, id=index, platform=platform))


def _bytes(spans: tuple[tuple[int, int], ...]) -> set[int]:
    """Enumerate individual byte addresses, independently of interval algorithms."""
    return {address for start, stop in spans for address in range(start, stop)}


def _scope(
    *, devices: tuple[jax.Device, ...], inputs: residency.DeviceBufferFootprint
) -> SimulationMemory:
    return SimulationMemory(
        budget_bytes=1_000_000,
        devices=devices,
        subject_devices=devices[:1],
        operations=ProfiledSimulationOperations(),
        inputs=inputs,
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_projected_union_matches_independent_byte_oracle(seed: int) -> None:
    """Overlaps, repeated aliases and equal pointers on different devices stay exact."""
    rng = np.random.default_rng(seed)
    devices = (
        _device(platform="cpu", index=0),
        _device(platform="gpu", index=0),
        _device(platform="gpu", index=1),
    )
    sources = []
    for _ in range(3):
        spans = {}
        for device in devices:
            bounds = rng.integers(0, 64, size=(12, 2))
            spans[device] = tuple((int(min(a, b)), int(max(a, b))) for a, b in bounds)
        sources.append(spans)
    footprints = tuple(residency.DeviceBufferFootprint(spans=s) for s in sources)
    arguments = residency.DeviceBufferFootprint(
        spans=dict.fromkeys(devices, ((10, 35), (30, 50)))
    )
    for selected in ((), devices[:1], devices[1:], devices):
        projected = residency.union_buffer_footprints(
            footprints=(*footprints, footprints[0]), devices=selected
        )
        got = residency.resident_bytes_by_device(
            live=projected, arguments=arguments, devices=selected
        )
        expected = {
            device: len(
                set().union(*(_bytes(s[device]) for s in sources)) - set(range(10, 50))
            )
            for device in selected
        }
        assert dict(got) == expected
        assert set(projected.spans) == set(selected)
    assert all(set(f.spans) == set(devices) for f in footprints)


def test_budget_projection_keeps_foreign_gpu_and_full_cpu_inventory() -> None:
    """The existing budget device set includes retained same-backend sources."""
    cpu = _device(platform="cpu", index=0)
    gpu = _device(platform="gpu", index=0)
    other_gpu = _device(platform="gpu", index=1)
    live = residency.DeviceBufferFootprint(
        spans={cpu: ((0, 200),), gpu: ((0, 30),), other_gpu: ((0, 50),)}
    )
    budget_devices = residency.resolve_budget_devices(
        execution_devices=(gpu,), live=live
    )
    scope = _scope(devices=budget_devices, inputs=live)
    assert set(scope.budget_snapshot().spans) == {gpu, other_gpu}
    assert scope.snapshot().spans == live.spans
    cpu_scope = _scope(devices=(cpu,), inputs=live)
    cpu_live = cpu_scope.budget_snapshot()
    assert cpu_live.spans == {cpu: ((0, 200),)}
    assert cpu_scope.snapshot().spans == live.spans


@pytest.mark.parametrize("history", [0, 1000, 50000])
def test_budget_snapshot_interval_work_excludes_unselected_history(
    *, history: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Count normalized intervals instead of asserting noisy wall-clock speed."""
    cpu = _device(platform="cpu", index=0)
    gpu = _device(platform="gpu", index=0)
    live = residency.DeviceBufferFootprint(spans={gpu: ((10, 20),)})
    scope = _scope(devices=(gpu,), inputs=live)
    scope.outputs = residency.DeviceBufferFootprint(
        spans={cpu: tuple((i * 4, i * 4 + 2) for i in range(history))}
    )
    merge = Mock(wraps=residency._merge_spans)
    monkeypatch.setattr(residency, "_merge_spans", merge)
    for _ in range(3):
        assert scope.budget_snapshot().spans == live.spans
    # Projection still excludes the unselected CPU history, and the owner ledger
    # now merges the one selected interval once for the whole unchanged epoch.
    assert sum(len(call.kwargs["spans"]) for call in merge.call_args_list) == 1
    assert len(scope.outputs.spans[cpu]) == history


def test_budget_snapshot_rereads_live_roots_after_previous_snapshot() -> None:
    """A previous metadata query neither owns arrays nor caches later live roots."""
    device = jax.devices()[0]
    scope = _scope(devices=(device,), inputs=residency.DeviceBufferFootprint(spans={}))
    first = jnp.arange(2, dtype=jnp.int32)
    scope.hold(tree=first)
    previous = scope.budget_snapshot()
    scope.close_unit()
    second = jnp.arange(5, dtype=jnp.int32)
    scope.hold(tree=second)
    current = scope.budget_snapshot()
    empty = residency.DeviceBufferFootprint(spans={})
    assert dict(
        residency.resident_bytes_by_device(
            live=previous, arguments=empty, devices=(device,)
        )
    ) == {device: 8}
    assert dict(
        residency.resident_bytes_by_device(
            live=current, arguments=empty, devices=(device,)
        )
    ) == {device: 20}


def test_run_passes_projected_live_callback_to_host_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The profiled operation sees current budget metadata, not foreign history."""
    cpu = _device(platform="cpu", index=0)
    gpu = _device(platform="gpu", index=0)
    live = residency.DeviceBufferFootprint(spans={cpu: ((0, 200),), gpu: ((0, 30),)})
    scope = _scope(devices=(gpu,), inputs=live)
    dispatch = Mock(return_value=None)
    monkeypatch.setattr(ProfiledSimulationOperations, "dispatch", dispatch)
    scope.run(function=_no_op, arguments={})
    callback = dispatch.call_args.kwargs["live_footprint"]
    assert callback().spans == {gpu: ((0, 30),)}
    assert scope.snapshot().spans == live.spans


def test_host_operation_refuses_execution_outside_projection_budget() -> None:
    """A CPU consumer cannot lose its own bytes through a GPU-only projection."""
    cpu = _device(platform="cpu", index=0)
    gpu = _device(platform="gpu", index=0)
    live = Mock(side_effect=AssertionError("admission must refuse before querying"))
    with pytest.raises(ExecutionPlanningError, match="omits executing devices"):
        ProfiledSimulationOperations().dispatch(
            function=_no_op,
            arguments={},
            subject_arg_names=(),
            devices=(cpu,),
            live_footprint=live,
            budget_devices=(gpu,),
            budget_bytes=1000,
        )
    live.assert_not_called()
