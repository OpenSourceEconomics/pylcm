"""Bounded selector controls and measured helper-profile owner admission."""

import gc
import weakref
from types import MappingProxyType, SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import TiledOutputAxis
from _lcm.simulation import chunk_admission as admission
from _lcm.simulation.chunk_planning import (
    SimulationChunkProfile,
    SimulationStageProfile,
    _required_bytes,
)
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from _lcm.simulation.simulate import _lookup_values_from_indices
from lcm.exceptions import ExecutionPlanningError


def _selector(
    *,
    monkeypatch: pytest.MonkeyPatch,
    costs: dict[tuple[int, int], int],
    pinned_action: int | None = None,
    population: int = 6,
) -> tuple[admission._ChunkProfiler, list[SimulationChunkProfile]]:
    """Supply typed profile shells with scalar costs; execute no numerical body."""
    device = jax.devices()[0]
    pins = {"subject": 2}
    if pinned_action is not None:
        pins["action_product"] = pinned_action
    # Only fields consumed by the selector are populated. No fake executable is
    # used to claim native compiler memory or ownership coverage.
    profiler = object.__new__(admission._ChunkProfiler)
    for name, value in {
        "runtime": SimpleNamespace(
            execution=SimpleNamespace(device_memory_bytes=100, axis_widths=pins)
        ),
        "population": population,
        "original_population": population,
        "regimes": MappingProxyType({}),
        "resident": {device: 5},
        "devices": (device,),
    }.items():
        object.__setattr__(profiler, name, value)
    axes = tuple(
        TiledOutputAxis(
            name=name, state_names=(name,), extent=extent, width_keyword=name
        )
        for name, extent in (("subject", 2), ("action_product", 8))
    )
    monkeypatch.setattr(admission, "_common_axes", lambda **_: axes)
    original_widths = admission.workspace_width_candidates

    def no_cartesian(**kwargs: Any) -> Any:
        assert kwargs["budget_bytes"] is None
        return original_widths(**kwargs)

    monkeypatch.setattr(admission, "workspace_width_candidates", no_cartesian)
    profiles = []

    # keyword-only-exempt: primary-argument=_self
    def profile_widths(
        _self: admission._ChunkProfiler, *, n_subjects: int, widths: Any
    ) -> SimulationChunkProfile:
        stage = object.__new__(SimulationStageProfile)
        for name, value in {
            "name": "synthetic",
            "devices": (device,),
            "reservation_bytes": 10,
            "peak_bytes": 10,
        }.items():
            object.__setattr__(stage, name, value)
        profile = SimulationChunkProfile(
            n_subjects=n_subjects,
            padded_population=-(-population // n_subjects) * n_subjects,
            stages=(stage,),
            fixed_reservation={device: costs[n_subjects, widths["action_product"]]},
            output_reservation={device: 5},
            axis_widths=widths,
        )
        profiles.append(profile)
        return profile

    monkeypatch.setattr(admission._ChunkProfiler, "profile_widths", profile_widths)
    return profiler, profiles


@pytest.mark.parametrize("larger_fits", [False, True])
def test_independent_freezes_full_map_and_reuses_anchor(
    *, monkeypatch: pytest.MonkeyPatch, larger_fits: bool
) -> None:
    profiler, profiles = _selector(
        monkeypatch=monkeypatch,
        costs={(2, 8): 30, (4, 8): 40 if larger_fits else 90, (6, 8): 45},
    )
    selected = admission._plan_independent_chunks(profiler=profiler, alignment=1)
    assert selected.profile is profiles[2 if larger_fits else 0]
    assert [dict(p.axis_widths) for p in profiles] == [
        {"subject": 2, "action_product": 8}
    ] * (3 if larger_fits else 2)
    receipt = selected.receipt
    assert receipt is not None
    assert receipt.profile_count == (3 if larger_fits else 2)
    assert receipt.anchor_map_reason == "bootstrap skipped after full anchor admitted"
    assert [attempt.admitted for attempt in receipt.attempts] == (
        [True, True, True] if larger_fits else [True, False]
    )
    assert receipt.frontier_version == 2
    assert receipt.unique_backend_compile_requests is None
    assert receipt.attempts[0].devices[0].required_bytes == 50


def test_independent_bootstrap_anchor_is_frozen_for_larger_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler, profiles = _selector(
        monkeypatch=monkeypatch, costs={(2, 8): 90, (2, 4): 30, (4, 4): 40, (6, 4): 90}
    )
    selected = admission._plan_independent_chunks(profiler=profiler, alignment=1)
    assert [(p.n_subjects, p.axis_widths["action_product"]) for p in profiles] == [
        (2, 8),
        (2, 4),
        (4, 4),
        (6, 4),
    ]
    assert selected.profile is profiles[2]
    assert selected.receipt is not None
    assert selected.receipt.profile_count == 4


@pytest.mark.parametrize(("population", "expected_profiles"), [(2, 1), (4, 2), (6, 3)])
def test_independent_pins_and_duplicate_frontiers(
    *, monkeypatch: pytest.MonkeyPatch, population: int, expected_profiles: int
) -> None:
    profiler, profiles = _selector(
        monkeypatch=monkeypatch,
        costs={(2, 3): 30, (4, 3): 40, (6, 3): 45},
        pinned_action=3,
        population=population,
    )
    selected = admission._plan_independent_chunks(profiler=profiler, alignment=1)
    assert len(profiles) == expected_profiles
    assert all(p.axis_widths == {"subject": 2, "action_product": 3} for p in profiles)
    assert selected.receipt is not None
    assert "duplicate suppressed" in selected.receipt.anchor_map_reason


def test_independent_bounded_refusal_does_not_probe_larger_or_intermediate_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler, profiles = _selector(
        monkeypatch=monkeypatch, costs={(2, 8): 90, (2, 4): 100}
    )
    with pytest.raises(
        ExecutionPlanningError,
        match="no candidate in the bounded independent frontier fits",
    ) as error:
        admission._plan_independent_chunks(profiler=profiler, alignment=1)
    assert len(profiles) == 2
    assert "rejected attempts" in str(error.value)
    assert "required_bytes=110" in str(error.value)


def test_independent_invalid_larger_metadata_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler, profiles = _selector(
        monkeypatch=monkeypatch, costs={(2, 8): 30, (4, 8): -1}
    )
    with pytest.raises(ExecutionPlanningError, match="nonnegative integer"):
        admission._plan_independent_chunks(profiler=profiler, alignment=1)
    assert len(profiles) == 1


def test_selected_plan_does_not_retain_transient_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler, _ = _selector(
        monkeypatch=monkeypatch, costs={(2, 8): 30, (4, 8): 40, (6, 8): 45}
    )
    transient = weakref.ref(profiler)
    selected = admission._plan_independent_chunks(profiler=profiler, alignment=1)
    del profiler
    gc.collect()
    assert transient() is None
    assert selected.receipt is not None
    assert selected.receipt.profile_count == 3


def test_singleton_anchor_retains_subject_pin_without_manufacturing_axis() -> None:
    assert admission._independent_outer_candidates(
        population=3, alignment=1, subject_width=1
    ) == (1, 2, 3)
    choices = admission._independent_anchor_widths(axes=(), configured={"subject": 1})
    assert choices == ({"subject": 1},)


def test_independent_actual_profiles_reject_larger_and_recheck_live_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise independent selection/require_chunk with real helper profiles.

    The represented bank below is explicit test storage, not a complete model's
    profiler inventory. This tests measured-cost admission and fresh owners;
    whole-model shape/lifetime coverage lives in the public cohort tests.
    """
    device = jax.devices()[0]
    layout = jax.sharding.SingleDeviceSharding(device)
    grid = jax.device_put(jnp.asarray([1.0, 3.0, 7.0]), layout)
    originals = jax.device_put(jnp.zeros(256, dtype=jnp.int32), layout)
    operations = ProfiledSimulationOperations()
    bank_bytes = 2 * originals.size * grid.dtype.itemsize
    profiles = {}
    for width in (64, 128):
        prepared = operations.prepare_abstract(
            function=_lookup_values_from_indices,
            arguments={
                "flat_indices": jax.ShapeDtypeStruct(
                    (width,), jnp.int32, sharding=layout
                ),
                "grids": MappingProxyType(
                    {
                        "consumption": jax.ShapeDtypeStruct(
                            grid.shape, grid.dtype, sharding=layout
                        )
                    }
                ),
            },
            subject_arg_names=("flat_indices",),
            devices=(device,),
        )
        profiles[width] = SimulationChunkProfile(
            n_subjects=width,
            padded_population=256,
            stages=(
                SimulationStageProfile(
                    name="decode_actions",
                    executable=prepared.executable,
                    devices=(device,),
                ),
            ),
            fixed_reservation={device: width * originals.dtype.itemsize},
            output_reservation={device: bank_bytes},
            axis_widths={"subject": 64},
        )
    live = measure_buffer_footprint(tree=(grid, originals, originals))
    resident = resident_bytes_by_device(
        live=live, arguments=DeviceBufferFootprint(spans={}), devices=(device,)
    )
    assert resident[device] == grid.nbytes + originals.nbytes
    totals = {
        width: _required_bytes(profile=profile, resident=resident, devices=(device,))[
            device
        ]
        for width, profile in profiles.items()
    }
    assert totals[128] > totals[64]
    profiler = object.__new__(admission._ChunkProfiler)
    for name, value in {
        "runtime": SimpleNamespace(
            execution=SimpleNamespace(
                device_memory_bytes=totals[64], axis_widths={"subject": 64}
            )
        ),
        "population": 256,
        "original_population": 256,
        "regimes": MappingProxyType({}),
        "resident": resident,
        "devices": (device,),
    }.items():
        object.__setattr__(profiler, name, value)
    requested = []

    # keyword-only-exempt: primary-argument=_self
    def profile_widths(
        _self: admission._ChunkProfiler, *, n_subjects: int, widths: Any
    ) -> SimulationChunkProfile:
        assert widths == {"subject": 64}
        requested.append(n_subjects)
        return profiles[n_subjects]

    monkeypatch.setattr(admission._ChunkProfiler, "profile_widths", profile_widths)
    plan = admission._plan_independent_chunks(profiler=profiler, alignment=1)
    assert requested == [64, 128]
    assert plan.profile is profiles[64]
    assert plan.required_bytes[device] == totals[64]
    memory = SimulationMemory(
        budget_bytes=totals[64],
        devices=(device,),
        subject_devices=(device,),
        operations=operations,
        inputs=live,
    )
    # require_chunk consumes no call-input methods, but owns the selected plan.
    chunks = object.__new__(admission.PreparedSimulationChunks)
    object.__setattr__(chunks, "plan", plan)
    object.__setattr__(chunks, "admitted_inputs", live)
    chunks.require_chunk(memory=memory)
    extra = jax.device_put(jnp.ones(1, dtype=jnp.uint8), layout)
    memory.derived = extra
    with pytest.raises(ExecutionPlanningError, match="no longer fits"):
        chunks.require_chunk(memory=memory)
    object.__setattr__(
        profiler,
        "resident",
        resident_bytes_by_device(
            live=memory.snapshot(),
            arguments=DeviceBufferFootprint(spans={}),
            devices=(device,),
        ),
    )
    with pytest.raises(ExecutionPlanningError, match="bounded independent frontier"):
        admission._plan_independent_chunks(profiler=profiler, alignment=1)
    memory.derived = ()
    del extra
    gc.collect()
    chunks.require_chunk(memory=memory)
    object.__setattr__(profiler, "resident", resident)
    recovered = admission._plan_independent_chunks(profiler=profiler, alignment=1)
    assert recovered.profile is profiles[64]
    transient = weakref.ref(profiler)
    del profiler
    gc.collect()
    assert transient() is None
    assert recovered.required_bytes[device] == totals[64]


def test_independent_frontier_doubles_the_inner_width_up_to_the_population() -> None:
    """Outer candidates double from the inner width until one covers everyone."""
    assert admission._independent_outer_candidates(
        population=226848, alignment=8, subject_width=2048
    ) == (2048, 4096, 8192, 16384, 32768, 65536, 131072, 226848)


def test_independent_frontier_aligns_every_candidate_to_the_device_count() -> None:
    assert admission._independent_outer_candidates(
        population=226848, alignment=3, subject_width=2048
    ) == (2049, 4098, 8193, 16386, 32769, 65538, 131073, 226848)


def test_independent_retains_the_last_admitted_extent_before_a_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planner walks the frontier in order and stops at the first refusal."""
    profiler, profiles = _selector(
        monkeypatch=monkeypatch, costs={(2, 8): 30, (4, 8): 40, (6, 8): 90}
    )
    selected = admission._plan_independent_chunks(profiler=profiler, alignment=1)
    assert selected.profile is profiles[1]
    assert selected.receipt is not None
    assert selected.receipt.profile_count == 3
    assert selected.receipt.selected_subjects == 4
    assert selected.receipt.stopping_reason == (
        "larger candidate rejected; last admitted extent retained"
    )


def test_independent_reports_frontier_exhaustion_when_everyone_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler, _ = _selector(
        monkeypatch=monkeypatch, costs={(2, 8): 30, (4, 8): 40, (6, 8): 45}
    )
    selected = admission._plan_independent_chunks(profiler=profiler, alignment=1)
    assert selected.receipt is not None
    assert selected.receipt.selected_subjects == 6
    assert selected.receipt.stopping_reason == (
        "bounded frontier exhausted; largest candidate admitted"
    )
