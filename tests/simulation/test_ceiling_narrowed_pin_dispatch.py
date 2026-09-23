"""Admission and budgeted dispatch agree on a ceiling-normalised width pin.

Checked against an independent finite-set reference over a neighbourhood of
extents, alignments, minimum widths, pins and ceilings.
"""

import dataclasses
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import jax
import pytest

from _lcm.execution.workspace_planning import bootstrap_widths
from _lcm.simulation.runtime import SimulationDispatchContext, _dispatch_widths
from lcm.exceptions import ExecutionPlanningError
from tests.simulation._width_reference import (
    Axis,
    NoLegalWidthError,
    cases,
    effective_pin,
    legal_widths,
)
from tests.simulation.test_unbudgeted_subject_width import (
    _empty_footprint,
    _materialized,
)

_CASES = tuple(cases())
_SINGLETON = tuple(c for c in _CASES if c[0].extent == 1)
_NON_SINGLETON = tuple(c for c in _CASES if c[0].extent > 1)


def _kw(case: tuple[Axis, int, int]) -> dict[str, Any]:
    axis, pin, ceiling = case
    return {"axis": axis, "pin": pin, "ceiling": ceiling}


_PROGRAMS: dict[tuple[int, int, int], Any] = {}


def _program(axis: Axis) -> Any:
    key = (axis.extent, axis.alignment, axis.minimum)
    if key not in _PROGRAMS:
        template = _materialized(n_subjects=axis.extent)
        if axis.extent == 1:
            # Real materialization omits the trivial subject planner axis. Keep
            # that representation instead of indexing or inventing an axis.
            _PROGRAMS[key] = template
            return template
        tiled = dataclasses.replace(
            template.requirements.tiled_axes[0],
            alignment=axis.alignment,
            minimum_width=axis.minimum,
        )
        _PROGRAMS[key] = dataclasses.replace(
            template,
            requirements=dataclasses.replace(
                template.requirements, tiled_axes=(tiled,)
            ),
        )
    return _PROGRAMS[key]


def _expected(*, axis: Axis, pin: int, ceiling: int) -> int | None:
    try:
        return effective_pin(axis=axis, pin=pin, ceiling=ceiling)
    except NoLegalWidthError:
        return None


_LEGAL = tuple(c for c in _NON_SINGLETON if _expected(**_kw(c)) is not None)
_IMPOSSIBLE = tuple(c for c in _NON_SINGLETON if _expected(**_kw(c)) is None)
_CONTESTED = tuple(c for c in _LEGAL if len(legal_widths(axis=c[0], ceiling=c[2])) > 1)


def _ids(case: tuple[Axis, int, int]) -> str:
    axis, pin, ceiling = case
    return f"E{axis.extent}-a{axis.alignment}-m{axis.minimum}-p{pin}-c{ceiling}"


def _admit(*, axis: Axis, pin: int, ceiling: int) -> Mapping[str, int]:
    return bootstrap_widths(
        axes=_program(axis).requirements.axes,
        fixed_widths=MappingProxyType({"subject": pin}),
        width_ceilings=MappingProxyType({"subject": ceiling}),
    )


def _dispatch(
    *, axis: Axis, pin: int, ceiling: int, reserved: int | None
) -> Mapping[str, int]:
    residency = (
        None
        if reserved is None
        else SimulationDispatchContext(
            live_footprint=_empty_footprint,
            budget_devices=(jax.devices()[0],),
            axis_widths={"subject": reserved},
        )
    )
    return _dispatch_widths(
        program=_program(axis),
        configured=MappingProxyType({"subject": pin}),
        residency=residency,
        width_ceilings=MappingProxyType({"subject": ceiling}),
    )


def test_reference_neighbourhood_covers_every_branch() -> None:
    """The case set holds singleton, legal, impossible and contested cases."""
    assert min(len(_SINGLETON), len(_LEGAL), len(_IMPOSSIBLE), len(_CONTESTED)) > 0


def test_reference_partition_covers_every_case() -> None:
    """Singleton, legal and impossible cases partition the reference neighbourhood."""
    assert len(_SINGLETON) + len(_LEGAL) + len(_IMPOSSIBLE) == len(_CASES)


@pytest.mark.parametrize("case", _SINGLETON, ids=_ids)
def test_singleton_materialization_omits_the_subject_planner_axis(case: tuple) -> None:
    """One subject has implicit width one, not a declared tiled planner axis."""
    assert _program(case[0]).requirements.axes == ()


@pytest.mark.parametrize("case", _SINGLETON, ids=_ids)
def test_singleton_reference_width_is_one(case: tuple) -> None:
    """The reference assigns a single subject the physical width one."""
    assert _expected(**_kw(case)) == 1


@pytest.mark.parametrize("case", _SINGLETON, ids=_ids)
def test_singleton_admission_plans_no_width(case: tuple) -> None:
    """Admission has no subject axis to plan for a single subject."""
    assert dict(_admit(**_kw(case))) == {}


def _singleton_reservations(case: tuple) -> tuple[int | None, ...]:
    return (None, 1, case[1])


_SINGLETON_RESERVED = tuple(
    (case, reserved)
    for case in _SINGLETON
    for reserved in _singleton_reservations(case)
)


def _singleton_ids(item: tuple) -> str:
    case, reserved = item
    return f"{_ids(case)}-r{reserved}"


@pytest.mark.parametrize("item", _SINGLETON_RESERVED, ids=_singleton_ids)
def test_singleton_dispatch_forwards_the_pin_under_any_reservation(item: tuple) -> None:
    """An omitted subject axis cannot conflict with any reservation map."""
    case, reserved = item
    assert dict(_dispatch(**_kw(case), reserved=reserved)) == {"subject": case[1]}


@pytest.mark.parametrize("item", _SINGLETON_RESERVED, ids=_singleton_ids)
def test_singleton_forwarded_pin_plans_no_width(item: tuple) -> None:
    """Planning the forwarded pin against the singleton program selects nothing."""
    case, reserved = item
    forwarded = _dispatch(**_kw(case), reserved=reserved)
    planned = bootstrap_widths(
        axes=_program(case[0]).requirements.axes,
        fixed_widths=forwarded,
        width_ceilings=MappingProxyType({"subject": case[2]}),
    )
    assert dict(planned) == {}


@pytest.mark.parametrize("case", _LEGAL, ids=_ids)
def test_bootstrap_widths_admits_reference_effective_pin(case: tuple) -> None:
    """Admission selects the largest legal width under both pin and ceiling."""
    assert _admit(**_kw(case))["subject"] == _expected(**_kw(case))


@pytest.mark.parametrize("case", _LEGAL, ids=_ids)
def test_budgeted_dispatch_accepts_admitted_width(case: tuple) -> None:
    """A reservation at the admitted width dispatches at that width."""
    got = _dispatch(**_kw(case), reserved=_admit(**_kw(case))["subject"])
    assert got["subject"] == _expected(**_kw(case))


@pytest.mark.parametrize("case", _LEGAL, ids=_ids)
def test_unbudgeted_dispatch_plans_reference_effective_pin(case: tuple) -> None:
    """Without a budget, the forwarded pin plans to the same effective width."""
    axis, _, ceiling = case
    forwarded = _dispatch(**_kw(case), reserved=None)
    planned = bootstrap_widths(
        axes=_program(axis).requirements.axes,
        fixed_widths=forwarded,
        width_ceilings=MappingProxyType({"subject": ceiling}),
    )
    assert planned["subject"] == _expected(**_kw(case))


@pytest.mark.parametrize("case", _CONTESTED, ids=_ids)
def test_budgeted_dispatch_refuses_other_legal_width(case: tuple) -> None:
    """A reservation at any other legal width is an incompatible specialisation."""
    axis, _, ceiling = case
    other = next(
        w
        for w in legal_widths(axis=axis, ceiling=ceiling)
        if w != _expected(**_kw(case))
    )
    with pytest.raises(ExecutionPlanningError, match="conflicts"):
        _dispatch(**_kw(case), reserved=other)


@pytest.mark.parametrize("case", _IMPOSSIBLE, ids=_ids)
def test_bootstrap_widths_refuses_ceiling_below_every_legal_width(
    case: tuple,
) -> None:
    """A ceiling that excludes every legal width is refused, naming the ceiling."""
    _, _, ceiling = case
    with pytest.raises(ExecutionPlanningError) as error:
        _admit(**_kw(case))
    assert {"subject", str(ceiling)} <= set(re.findall(r"\w+", str(error.value)))
