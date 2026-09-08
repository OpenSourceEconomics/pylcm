"""Concrete simulation buffers are charged once on their actual devices."""

import gc
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import cast
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import TiledOutputAxis
from _lcm.execution.workspace_planning import plan_workspace
from _lcm.simulation import residency
from lcm.exceptions import ExecutionPlanningError


def test_repeated_array_references_occupy_one_resident_buffer() -> None:
    """Repeated source/result aliases do not multiply the live device payload."""
    array = jnp.arange(4, dtype=jnp.int32)
    live = residency.measure_buffer_footprint(tree=(array, {"alias": array}, array))
    empty = residency.measure_buffer_footprint(tree=())
    assert dict(
        residency.resident_bytes_by_device(
            live=live, arguments=empty, devices=tuple(array.devices())
        )
    ) == {next(iter(array.devices())): 16}


@pytest.mark.parametrize(
    ("live_spans", "argument_spans", "expected"),
    [
        (((100, 200), (100, 200), (100, 200)), (), 100),
        (((100, 200),), ((100, 140),), 60),
        (((100, 180), (140, 220)), ((160, 200),), 80),
        (((100, 120), (100, 120), (200, 260)), ((100, 120),), 60),
        (((100, 200),), ((20, 70), (300, 400)), 100),
        (((100, 200),), ((50, 300),), 0),
        (((100, 200),), ((120, 150), (140, 170)), 50),
        (((100, 100),), (), 0),
    ],
)
def test_only_argument_covered_bytes_leave_residency(
    *,
    live_spans: tuple[tuple[int, int], ...],
    argument_spans: tuple[tuple[int, int], ...],
    expected: int,
) -> None:
    """Literal overlapping-interval examples distinguish aliases from fresh copies."""
    device = jax.devices()[0]
    live = residency.DeviceBufferFootprint(spans={device: live_spans})
    arguments = residency.DeviceBufferFootprint(spans={device: argument_spans})
    assert dict(
        residency.resident_bytes_by_device(
            live=live, arguments=arguments, devices=(device,)
        )
    ) == {device: expected}


def test_distinct_backend_devices_do_not_share_pointer_names() -> None:
    """Hardware device tokens with the same local id and pointer remain distinct."""
    # External hardware descriptors let CPU-only CI exercise mixed-backend identity.
    cpu = cast("jax.Device", Mock(spec=jax.Device, id=0, platform="cpu"))
    gpu = cast("jax.Device", Mock(spec=jax.Device, id=0, platform="gpu"))
    live = residency.DeviceBufferFootprint(
        spans={cpu: ((100, 200),), gpu: ((100, 200),)}
    )
    arguments = residency.DeviceBufferFootprint(spans={gpu: ((100, 200),)})
    assert dict(
        residency.resident_bytes_by_device(
            live=live, arguments=arguments, devices=(cpu, gpu)
        )
    ) == {cpu: 100, gpu: 0}


def test_incremental_publications_union_shared_params_once() -> None:
    """New output metadata adds storage while preserving aliases of earlier inputs."""
    device = jax.devices()[0]
    original = residency.DeviceBufferFootprint(spans={device: ((100, 120),)})
    output = residency.DeviceBufferFootprint(spans={device: ((100, 120), (200, 260))})
    live = residency.union_buffer_footprints(footprints=(original, output))
    assert dict(
        residency.resident_bytes_by_device(
            live=live,
            arguments=residency.DeviceBufferFootprint(spans={}),
            devices=(device,),
        )
    ) == {device: 80}


def test_typed_random_keys_have_a_measured_payload() -> None:
    """Typed simulation RNG keys retain their underlying two uint32 words."""
    key = jax.random.key(0, impl="threefry2x32")
    measured = residency.measure_buffer_footprint(tree=key)
    assert dict(
        residency.resident_bytes_by_device(
            live=measured,
            arguments=residency.DeviceBufferFootprint(spans={}),
            devices=tuple(key.devices()),
        )
    ) == {next(iter(key.devices())): 8}


def test_snapshot_owns_no_array_and_keeps_no_mutable_span_mapping() -> None:
    """Snapshots contain only device descriptors and integer address metadata."""
    array = jnp.arange(3, dtype=jnp.int32)
    reference = weakref.ref(array)
    measured = residency.measure_buffer_footprint(tree=array)
    del array
    gc.collect()
    assert reference() is None
    assert isinstance(measured.spans, MappingProxyType)


@pytest.mark.parametrize("budget", [349, 350])
def test_transfer_headroom_covers_source_destination_and_scratch(budget: int) -> None:
    """The 150-byte resident plus 100-byte copy and scratch needs 350 bytes."""
    device = jax.devices()[0]
    live = residency.DeviceBufferFootprint(spans={device: ((100, 250),)})
    copied = []

    def acquire() -> None:
        residency.require_transfer_headroom(
            live=live,
            destination_bytes={device: 100},
            scratch_bytes={device: 100},
            budget_bytes=budget,
            devices=(device,),
        )
        copied.append("allocated")

    if budget == 349:
        with pytest.raises(ExecutionPlanningError, match="350"):
            acquire()
        assert copied == []
    else:
        acquire()
        assert copied == ["allocated"]


def test_transfer_checks_retained_source_devices_without_a_core() -> None:
    """A source device's retained original can exceed budget before a remote copy."""
    source = cast("jax.Device", Mock(spec=jax.Device, id=0, platform="gpu"))
    destination = cast("jax.Device", Mock(spec=jax.Device, id=1, platform="gpu"))
    with pytest.raises(ExecutionPlanningError, match="300"):
        residency.require_transfer_headroom(
            live=residency.DeviceBufferFootprint(spans={source: ((100, 400),)}),
            destination_bytes={destination: 50},
            scratch_bytes={destination: 50},
            budget_bytes=200,
            devices=(source, destination),
        )


@pytest.mark.parametrize(
    ("budget", "destination", "scratch"), [(0, 0, 0), (100, -1, 0), (100, 0, -1)]
)
def test_transfer_costs_cannot_create_fictitious_headroom(
    *, budget: int, destination: int, scratch: int
) -> None:
    """Nonpositive budgets and negative allocation costs cannot license a transfer."""
    device = jax.devices()[0]
    with pytest.raises(ExecutionPlanningError):
        residency.require_transfer_headroom(
            live=residency.DeviceBufferFootprint(spans={}),
            destination_bytes={device: destination},
            scratch_bytes={device: scratch},
            budget_bytes=budget,
            devices=(device,),
        )


def test_generated_interval_counts_agree_with_individual_byte_ownership() -> None:
    """Exhaustive byte sets independently check overlapping metadata intervals."""
    device = jax.devices()[0]
    rng = np.random.default_rng(904)
    actual = []
    expected = []
    for _ in range(100):
        sources = tuple(
            (int(start), int(start + length))
            for start, length in rng.integers(0, 20, size=(5, 2))
        )
        arguments = tuple(
            (int(start), int(start + length))
            for start, length in rng.integers(0, 20, size=(3, 2))
        )
        source_bytes = {byte for start, stop in sources for byte in range(start, stop)}
        argument_bytes = {
            byte for start, stop in arguments for byte in range(start, stop)
        }
        expected.append(len(source_bytes - argument_bytes))
        actual.append(
            residency.resident_bytes_by_device(
                live=residency.DeviceBufferFootprint(spans={device: sources}),
                arguments=residency.DeviceBufferFootprint(spans={device: arguments}),
                devices=(device,),
            )[device]
        )
    assert actual == expected


def test_deleted_arrays_cannot_supply_a_live_inventory() -> None:
    """A deleted buffer cannot silently contribute zero to a budgeted call."""
    array = jnp.arange(3, dtype=jnp.int32)
    array.delete()
    with pytest.raises(ExecutionPlanningError, match="deleted"):
        residency.measure_buffer_footprint(tree=array)


def test_nonaddressable_arrays_cannot_supply_a_complete_inventory() -> None:
    """Unobserved remote shards refuse a complete local accounting claim."""
    array = Mock(
        spec=jax.Array,
        is_deleted=Mock(return_value=False),
        is_fully_addressable=False,
    )
    with pytest.raises(ExecutionPlanningError, match="addressable"):
        residency.measure_buffer_footprint(tree=array)


@dataclass(frozen=True, kw_only=True)
class _CompiledCandidate:
    """A compiler-boundary fixture with explicit output/workspace memory reports."""

    width: int
    peak_bytes: int

    def memory_analysis(self) -> SimpleNamespace:
        """Report known candidate peaks for the selection oracle."""
        return SimpleNamespace(peak_memory_in_bytes=self.peak_bytes)


def test_fresh_admission_uses_current_residency_with_cached_width_candidates() -> None:
    """Output growth narrows then refuses a plan without recompiling cached widths."""
    device = jax.devices()[0]
    axis = TiledOutputAxis(
        name="subject", state_names=("subject",), extent=4, width_keyword="width"
    )
    cached: dict[int, _CompiledCandidate] = {}
    compiled_widths = []

    def compile_candidate(widths: Mapping[str, int]) -> _CompiledCandidate:
        width = widths["subject"]
        if width not in cached:
            cached[width] = _CompiledCandidate(
                width=width, peak_bytes={4: 80, 2: 80, 1: 40}[width]
            )
            compiled_widths.append(width)
        return cached[width]

    def select(retained_bytes: int) -> _CompiledCandidate:
        per_device = residency.resident_bytes_by_device(
            live=residency.DeviceBufferFootprint(
                spans={device: ((100, 100 + retained_bytes),)}
            ),
            arguments=residency.DeviceBufferFootprint(spans={}),
            devices=(device,),
        )
        return plan_workspace(
            axes=(axis,),
            compile_candidate=compile_candidate,
            resident_bytes=per_device[device],
            budget_bytes=100,
        ).compiled

    wide = select(10)
    narrow = select(30)
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        select(70)
    assert wide is cached[4]
    assert narrow is cached[1]
    assert compiled_widths == [4, 2, 1]
