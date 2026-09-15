"""Top-first search contracts, with explicit synthetic compiler-memory profiles.

These are planner unit tests, not numerical or physical-device memory witnesses.
The real candidate validation, byte summation and scalar receipts remain in use.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Any

import jax
import pytest

import _lcm.simulation.chunk_admission as admission
from _lcm.execution.core_program import TiledOutputAxis
from _lcm.simulation.chunk_planning import (
    SimulationChunkProfile,
    SimulationStageProfile,
)
from lcm.exceptions import ExecutionPlanningError


class _Profiler:
    """Scalar-cost stand-in for the transient profiler, on actual devices.

    Profiles are real `SimulationChunkProfile` records whose stages are typed
    shells carrying fixed reservation bytes; no executable is compiled. Tests
    needing more devices than the session offers are skipped.
    """

    def __init__(
        self,
        *,
        population: int = 12,
        width: int = 2,
        devices: int = 1,
        budget: int | None = 100,
        cost: Callable[[int, int], int | Mapping[int, int]] | None = None,
        pinned_action: int | None = None,
        malformed: str | None = None,
    ) -> None:
        available = jax.devices()
        needs_foreign = malformed in {
            "stage_device",
            "reservation_device",
            "host_device",
        }
        if len(available) < devices + int(needs_foreign):
            pytest.skip(f"Requires {devices + int(needs_foreign)} devices")
        self.devices = tuple(available[:devices])
        self.foreign = available[devices] if needs_foreign else None
        self.population = population
        self.original_population = population
        pins = {"subject": width}
        if pinned_action is not None:
            pins["action_product"] = pinned_action
        self.runtime = SimpleNamespace(
            execution=SimpleNamespace(
                device_memory_bytes=budget, axis_widths=MappingProxyType(pins)
            )
        )
        self.regimes: MappingProxyType[str, object] = MappingProxyType({})
        self.resident = dict.fromkeys(self.devices, 5)
        self.cost = cost or (lambda _n, _a: 30)
        self.malformed = malformed
        self.calls: list[tuple[int, int]] = []

    def profile_widths(
        self, *, n_subjects: int, widths: Mapping[str, int]
    ) -> SimulationChunkProfile:
        self.calls.append((n_subjects, widths["action_product"]))
        fixed = self.cost(n_subjects, widths["action_product"])
        costs = (
            dict.fromkeys(self.devices, fixed)
            if isinstance(fixed, int)
            else {device: fixed[device.id] for device in self.devices}
        )
        actual_widths = dict(widths)
        if self.malformed == "widths":
            actual_widths["action_product"] += 1
        stage_devices = self.devices
        if self.malformed == "stage_device":
            stage_devices = (self.foreign,)
        if self.malformed == "reservation_device":
            costs[self.foreign] = 1
        host_stages: tuple[SimulationStageProfile, ...] = ()
        if self.malformed == "host_device":
            host_stages = (_stage(devices=(self.foreign,)),)
        return SimulationChunkProfile(
            n_subjects=n_subjects + int(self.malformed == "extent"),
            padded_population=-(-self.population // n_subjects) * n_subjects,
            axis_widths=MappingProxyType(actual_widths),
            fixed_reservation=costs,
            output_reservation=dict.fromkeys(self.devices, 5),
            stages=(_stage(devices=stage_devices),),
            host_stages=host_stages,
        )


def _stage(*, devices: tuple[jax.Device, ...]) -> SimulationStageProfile:
    """Build a typed stage shell with a fixed reservation and no executable."""
    stage = object.__new__(SimulationStageProfile)
    for name, value in {
        "name": "synthetic-stage",
        "devices": devices,
        "reservation_bytes": 10,
        "peak_bytes": 10,
    }.items():
        object.__setattr__(stage, name, value)
    return stage


def _axis(*, name: str, extent: int) -> TiledOutputAxis:
    return TiledOutputAxis(
        name=name, state_names=(name,), extent=extent, width_keyword=name
    )


def _axes(*, regimes: object, n_subjects: int) -> tuple[TiledOutputAxis, ...]:
    del regimes
    return (
        _axis(name="subject", extent=n_subjects),
        _axis(name="action_product", extent=8),
    )


def _width_choices(
    *, axes: tuple[Any, ...], fixed_widths: Mapping[str, int], budget_bytes: int | None
) -> tuple[dict[str, int], ...]:
    assert budget_bytes is None
    return (
        {axis.name: min(fixed_widths.get(axis.name, 4), axis.extent) for axis in axes},
    )


@pytest.fixture(autouse=True)
def _synthetic_axis_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "_common_axes", _axes)
    monkeypatch.setattr(admission, "workspace_width_candidates", _width_choices)


def _select(*, profiler: _Profiler) -> Any:
    """Run the planner on a typed profiler shell that delegates to the stand-in.

    The shell is rebuilt on every call so the stand-in's current residency and
    budget are what the planner reads.
    """
    shell = object.__new__(admission._ChunkProfiler)
    for name in (
        "runtime",
        "population",
        "original_population",
        "regimes",
        "resident",
        "devices",
    ):
        object.__setattr__(shell, name, getattr(profiler, name))
    object.__setattr__(shell, "profile_widths", profiler.profile_widths)
    return admission._plan_independent_chunks(
        profiler=shell, alignment=len(profiler.devices)
    )


def _check_receipt(*, plan: Any, profiler: _Profiler) -> None:
    receipt = plan.receipt
    assert receipt.frontier_version == 3
    assert receipt.profile_count == len(profiler.calls)
    assert len(profiler.calls) == len(set(profiler.calls))
    assert receipt.candidates == tuple(sorted(set(receipt.candidates)))
    assert receipt.selected_subjects == plan.profile.n_subjects
    assert dict(receipt.axis_widths) == dict(plan.profile.axis_widths)
    assert all(value <= 100 for value in plan.required_bytes.values())
    selected = [a for a in receipt.attempts if a.admitted][-1]
    assert selected.n_subjects == plan.profile.n_subjects
    assert receipt.planning_seconds >= 0
    assert receipt.unique_backend_compile_requests is None


@pytest.mark.parametrize(
    ("population", "width", "devices", "largest"),
    [(226848, 2048, 8, 226848), (12, 2, 1, 12), (7, 3, 4, 8), (1, 1, 1, 1)],
)
def test_full_population_needs_only_one_complete_profile(
    *, population: int, width: int, devices: int, largest: int
) -> None:
    profiler = _Profiler(population=population, width=width, devices=devices)
    plan = _select(profiler=profiler)
    assert profiler.calls == [(largest, 8)]
    assert plan.profile.padded_population // largest == 1
    assert dict(plan.profile.axis_widths)["subject"] == min(width, largest)
    _check_receipt(plan=plan, profiler=profiler)


@pytest.mark.parametrize(
    ("last_fit", "expected"),
    [
        (8, [(12, 8), (2, 8), (8, 8)]),
        (4, [(12, 8), (2, 8), (8, 8), (4, 8)]),
        (2, [(12, 8), (2, 8), (8, 8), (4, 8)]),
    ],
)
def test_refusal_descends_with_the_admitted_anchor_map(
    *, last_fit: int, expected: list[tuple[int, int]]
) -> None:
    profiler = _Profiler(cost=lambda n, _a: 30 if n <= last_fit else 90)
    plan = _select(profiler=profiler)
    assert profiler.calls == expected
    assert plan.profile.n_subjects == last_fit
    _check_receipt(plan=plan, profiler=profiler)


@pytest.mark.parametrize("last_fit", [2, 4, 8, 12])
def test_bootstrap_map_is_selected_once_then_frozen(*, last_fit: int) -> None:
    profiler = _Profiler(cost=lambda n, a: 90 if a == 8 or n > last_fit else 30)
    plan = _select(profiler=profiler)
    assert profiler.calls[:3] == [(12, 8), (2, 8), (2, 4)]
    assert profiler.calls[3] == (12, 4)  # Same extent, genuinely different map.
    assert all(a == 4 for _n, a in profiler.calls[2:])
    assert plan.profile.n_subjects == last_fit
    _check_receipt(plan=plan, profiler=profiler)


@pytest.mark.parametrize("top_fits", [False, True])
def test_duplicate_width_maps_are_not_profiled_twice(*, top_fits: bool) -> None:
    profiler = _Profiler(
        pinned_action=3, cost=lambda n, _a: 30 if top_fits or n == 2 else 90
    )
    plan = _select(profiler=profiler)
    assert all(a == 3 for _n, a in profiler.calls)
    assert plan.profile.n_subjects == (12 if top_fits else 2)
    _check_receipt(plan=plan, profiler=profiler)


@pytest.mark.parametrize("bootstrap_fits", [False, True])
def test_collapsed_frontier_profiles_each_map_at_most_once(
    *, bootstrap_fits: bool
) -> None:
    profiler = _Profiler(
        population=1,
        width=8,
        devices=4,
        cost=lambda _n, a: 30 if bootstrap_fits and a == 4 else 90,
    )
    if bootstrap_fits:
        plan = _select(profiler=profiler)
        assert plan.profile.n_subjects == 4
        _check_receipt(plan=plan, profiler=profiler)
    else:
        with pytest.raises(ExecutionPlanningError, match="no anchor map fits"):
            _select(profiler=profiler)
    assert profiler.calls == [(4, 8), (4, 4)]


def test_top_fit_does_not_depend_on_unmeasured_smaller_shapes() -> None:
    profiler = _Profiler(cost=lambda n, _a: 30 if n == 12 else 90)
    plan = _select(profiler=profiler)
    assert profiler.calls == [(12, 8)]
    assert plan.profile.n_subjects == 12
    _check_receipt(plan=plan, profiler=profiler)


def test_nonmonotone_refusals_do_not_skip_an_admissible_descending_candidate() -> None:
    profiler = _Profiler(cost=lambda n, _a: 30 if n in {2, 8} else 90)
    plan = _select(profiler=profiler)
    assert profiler.calls == [(12, 8), (2, 8), (8, 8)]
    assert plan.profile.n_subjects == 8
    _check_receipt(plan=plan, profiler=profiler)


def test_no_anchor_fit_refuses_without_a_width_by_outer_sweep() -> None:
    profiler = _Profiler(cost=lambda _n, _a: 90)
    with pytest.raises(ExecutionPlanningError, match="3 complete profiles"):
        _select(profiler=profiler)
    assert profiler.calls == [(12, 8), (2, 8), (2, 4)]


@pytest.mark.parametrize("fixed", [80, 81])
def test_exact_budget_boundary_is_checked_on_every_device(*, fixed: int) -> None:
    profiler = _Profiler(
        population=12,
        width=2,
        devices=3,
        cost=lambda n, _a: {0: 30, 1: fixed if n == 12 else 30, 2: 30},
    )
    plan = _select(profiler=profiler)
    assert plan.profile.n_subjects == (12 if fixed == 80 else 9)
    receipt = plan.receipt.attempts[0].devices[1]
    assert receipt.required_bytes == 5 + fixed + 5 + 10
    assert plan.receipt.attempts[0].admitted is (fixed == 80)
    _check_receipt(plan=plan, profiler=profiler)


@pytest.mark.parametrize(
    "malformed",
    ["extent", "widths", "stage_device", "reservation_device", "host_device"],
)
def test_invalid_profiles_are_not_treated_as_memory_refusals(*, malformed: str) -> None:
    profiler = _Profiler(malformed=malformed)
    with pytest.raises(ExecutionPlanningError):
        _select(profiler=profiler)
    assert profiler.calls == [(12, 8)]


@pytest.mark.parametrize(
    "exception", [RuntimeError, ExecutionPlanningError, MemoryError]
)
def test_profiler_errors_propagate_without_smaller_shape_retry(
    *, exception: type[Exception]
) -> None:
    def fail(*_shape: int) -> int:
        raise exception("compiler failure, not an admission refusal")

    profiler = _Profiler(cost=fail)
    with pytest.raises(exception, match="compiler failure"):
        _select(profiler=profiler)
    assert profiler.calls == [(12, 8)]


def test_missing_budget_refuses_before_any_profile() -> None:
    profiler = _Profiler(budget=None)
    with pytest.raises(ExecutionPlanningError, match="needs a budget"):
        _select(profiler=profiler)
    assert profiler.calls == []


def test_admission_is_rechecked_when_current_residency_changes() -> None:
    profiler = _Profiler()
    assert _select(profiler=profiler).profile.n_subjects == 12
    profiler.resident = dict.fromkeys(profiler.devices, 90)
    with pytest.raises(ExecutionPlanningError, match="no anchor map fits"):
        _select(profiler=profiler)
    assert profiler.calls == [(12, 8), (12, 8), (2, 8), (2, 4)]


def test_pin_is_retained_when_scalar_anchor_omits_the_subject_axis(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    def scalar_axes(*, regimes: object, n_subjects: int) -> tuple[TiledOutputAxis, ...]:
        del regimes
        assert n_subjects == 1
        return (_axis(name="action_product", extent=8),)

    monkeypatch.setattr(admission, "_common_axes", scalar_axes)
    profiler = _Profiler(population=5, width=1)
    plan = _select(profiler=profiler)
    assert dict(plan.profile.axis_widths) == {"subject": 1, "action_product": 8}
    assert profiler.calls == [(5, 8)]
