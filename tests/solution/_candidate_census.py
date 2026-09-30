"""Observe what a budgeted solve lowers, measures and admits, wave by wave.

The recorder wraps four engine seams without replacing their behaviour, so a
solve observed through it computes exactly what an unobserved one computes:

- `_lower_and_compile_wave` names the keys a wave newly lowered and, through the
  candidate each key was lowered for, which triple and widths the key belongs
  to.
- `compiler_memory_reservation` names each executable's reservation.
- `_candidate_resident_bytes` names each executable's residency against its
  triple's inventory.
- `_compile_all_functions` names the fallbacks the finished solve selected.

Every wave issues its primary call before its fallback call, so the two calls of
a wave alternate; the recorder labels them by that order and checks that a call
it labelled a fallback carries no donating candidate.
"""

import dataclasses
from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import Any

import pytest

from _lcm.execution.workspace_planning import CompilerMemoryReservation
from _lcm.solution import backward_induction
from lcm import ExecutionConfig
from lcm.solver_api import ResultRetention
from tests.test_models import nbegm_ride_along_toy

type WidthKey = tuple[tuple[str, int], ...]
type Triple = tuple[str, int, str]

# Role of the wave call that lowers each triple's donating variant.
PRIMARY = "primary"

# Role of the wave call that lowers a candidate's donation-free variant.
FALLBACK = "fallback"


@dataclasses.dataclass(frozen=True, kw_only=True)
class WaveCall:
    """One `_lower_and_compile_wave` call of one wave."""

    wave: int
    """Zero-based index of the wave the call belongs to."""
    role: str
    """`PRIMARY` or `FALLBACK`."""
    keys: tuple[Hashable, ...]
    """Lowering keys this call newly lowered, in call order."""
    candidates: tuple[tuple[Triple, WidthKey], ...]
    """The candidate each newly lowered key was lowered for."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class CandidateRow:
    """One measured variant of one triple's width candidate."""

    triple: Triple
    """Regime, period and core the candidate belongs to."""
    widths: WidthKey
    """Tile widths of the candidate, in axis declaration order."""
    role: str
    """`PRIMARY` or `FALLBACK`, read from the wave call that lowered the key."""
    lowering_key: Hashable
    """Key of this variant's own executable."""
    fallback_key: Hashable | None
    """Key of the candidate's donation-free variant, when a wave lowered it."""
    has_fallback: bool
    """Whether the candidate names a donation-free variant at all."""
    reservation: int
    """Compiler reservation of this variant's executable."""
    residency: int
    """Bytes this variant leaves resident against its triple's inventory."""
    admitted: bool
    """Whether the finished solve selected this candidate's widths."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class Census:
    """Everything one observed solve revealed about its candidate frontier."""

    rows: tuple[CandidateRow, ...]
    """One row per measured (triple, widths, role), in measurement order."""
    calls: tuple[WaveCall, ...]
    """Every wave call, in call order."""
    unique_keys: frozenset[Hashable]
    """Every lowering key the solve compiled."""
    selected_widths: Mapping[Triple, WidthKey]
    """Widths the finished solve dispatched for each triple."""
    donation_fallbacks: Mapping[Triple, Any]
    """The finished solve's selected donation-free fallbacks."""
    selected_cores: Mapping[Triple, Any]
    """The planned core the finished solve dispatches for each triple."""

    def rows_for(self, *, role: str) -> tuple[CandidateRow, ...]:
        """Return the measured rows carrying one role."""
        return tuple(row for row in self.rows if row.role == role)

    def keys_in(self, *, role: str) -> tuple[Hashable, ...]:
        """Return every key lowered by the calls carrying one role."""
        return tuple(
            key for call in self.calls if call.role == role for key in call.keys
        )

    def widths_on_axis(self, *, axis: str) -> tuple[int, ...]:
        """Return the distinct widths measured on one axis, widest first."""
        seen = {dict(row.widths)[axis] for row in self.rows if axis in dict(row.widths)}
        return tuple(sorted(seen, reverse=True))


def donor_pair_model(
    *,
    device_memory_bytes: int | None,
    axis_widths: Mapping[str, int] = MappingProxyType({}),
) -> Any:
    """Build the donor-pair fixture under one budget and one width constraint."""
    return nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_liquid=8,
        n_savings=10,
        n_consumption=12,
        execution_config=ExecutionConfig(
            device_memory_bytes=device_memory_bytes, axis_widths=dict(axis_widths)
        ),
    )


def solve_donor_pair(*, model: Any) -> Any:
    """Solve the donor-pair fixture, retaining values only."""
    return model.solve(
        params=nbegm_ride_along_toy.build_params(),
        retention=ResultRetention.VALUES,
        log_level="off",
    )


class CensusRecorder:
    """Record one solve's waves, measurements and selection as they happen."""

    def __init__(self) -> None:
        """Start an empty recording."""
        self._calls: list[WaveCall] = []
        self._wave = -1
        self._next_role = PRIMARY
        self._candidate_of_program: dict[int, tuple[Triple, WidthKey]] = {}
        self._role_of_compiled: dict[int, str] = {}
        self._key_of_compiled: dict[int, Hashable] = {}
        self._fallback_key_of_candidate: dict[tuple[Triple, WidthKey], Hashable] = {}
        self._fallback_eligible: set[tuple[Triple, WidthKey]] = set()
        self._reservations: dict[int, int] = {}
        self._measured: list[tuple[Triple, WidthKey, str, int, int]] = []
        self._selected_widths: dict[Triple, WidthKey] = {}
        self._donation_fallbacks: Mapping[Triple, Any] = {}
        self._selected_cores: dict[Triple, Any] = {}

    def role_of(self, *, compiled: object) -> str:
        """Name the wave-call role the executable's key was lowered by."""
        return self._role_of_compiled[id(compiled)]

    def install(self, *, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrap the four seams for the duration of the test."""
        wave = backward_induction._lower_and_compile_wave
        resident = backward_induction._candidate_resident_bytes
        compile_all = backward_induction._compile_all_functions
        reservation = backward_induction.compiler_memory_reservation

        def observe_wave(**kwargs: Any) -> None:
            role = self._open_call(**kwargs)
            wave(**kwargs)
            self._close_call(role=role, **kwargs)

        def observe_reservation(**kwargs: Any) -> CompilerMemoryReservation:
            value = reservation(**kwargs)
            self._reservations.setdefault(
                id(kwargs["compiled"]), value.reservation_bytes
            )
            return value

        def observe_residency(**kwargs: Any) -> int:
            actual = resident(**kwargs)
            self._record_measurement(residency=actual, **kwargs)
            return actual

        def observe_compilation(**kwargs: Any) -> Any:
            result = compile_all(**kwargs)
            self._record_result(result=result)
            return result

        monkeypatch.setattr(backward_induction, "_lower_and_compile_wave", observe_wave)
        monkeypatch.setattr(
            backward_induction, "compiler_memory_reservation", observe_reservation
        )
        monkeypatch.setattr(
            backward_induction, "_candidate_resident_bytes", observe_residency
        )
        monkeypatch.setattr(
            backward_induction, "_compile_all_functions", observe_compilation
        )

    def census(self) -> Census:
        """Freeze the recording into a census of the solve just observed."""
        rows = tuple(
            CandidateRow(
                triple=triple,
                widths=widths,
                role=role,
                lowering_key=self._key_of_compiled[compiled_id],
                fallback_key=self._fallback_key_of_candidate.get((triple, widths)),
                has_fallback=(triple, widths) in self._fallback_eligible,
                reservation=self._reservations[compiled_id],
                residency=residency,
                admitted=self._selected_widths.get(triple) == widths,
            )
            for triple, widths, role, compiled_id, residency in self._measured
        )
        return Census(
            rows=rows,
            calls=tuple(self._calls),
            unique_keys=frozenset(self._key_of_compiled.values()),
            selected_widths=dict(self._selected_widths),
            donation_fallbacks=self._donation_fallbacks,
            selected_cores=dict(self._selected_cores),
        )

    def _open_call(self, **kwargs: Any) -> str:
        """Label the call by wave order and learn every candidate's program."""
        role = self._next_role
        if role == PRIMARY:
            self._wave += 1
        self._next_role = FALLBACK if role == PRIMARY else PRIMARY
        for candidate, program in kwargs["resolved_programs"].items():
            self._candidate_of_program[id(program)] = candidate
        donations = kwargs["donations"]
        if role == FALLBACK:
            assert not any(
                donations.get(c) for c in kwargs["new_lowerings"].values()
            ), "A call labelled a fallback carried a donating candidate."
            self._fallback_eligible.update(donations)
            for key, candidate in kwargs["new_lowerings"].items():
                self._fallback_key_of_candidate[candidate] = key
        return role

    def _close_call(self, *, role: str, **kwargs: Any) -> None:
        """Read back the executables the finished call left behind."""
        keys = tuple(kwargs["new_lowerings"])
        candidates = tuple(kwargs["new_lowerings"].values())
        for key in keys:
            compiled = kwargs["compiled"][key]
            self._role_of_compiled.setdefault(id(compiled), role)
            self._key_of_compiled.setdefault(id(compiled), key)
        self._calls.append(
            WaveCall(wave=self._wave, role=role, keys=keys, candidates=candidates)
        )

    def _record_measurement(self, *, residency: int, **kwargs: Any) -> None:
        """Attach one residency reading to the candidate it was taken for."""
        triple, widths = self._candidate_of_program[id(kwargs["program"])]
        compiled = kwargs["compiled"]
        self._measured.append(
            (triple, widths, self.role_of(compiled=compiled), id(compiled), residency)
        )

    def _record_result(self, *, result: Any) -> None:
        """Read the finished solve's selection and its donation-free fallbacks."""
        self._donation_fallbacks = dict(result.donation_fallbacks)
        for (regime, period), cores in result.executables.items():
            for core_key, planned in cores.items():
                self._selected_widths[(regime, period, core_key)] = tuple(
                    planned.tile_widths.items()
                )
                self._selected_cores[(regime, period, core_key)] = planned
