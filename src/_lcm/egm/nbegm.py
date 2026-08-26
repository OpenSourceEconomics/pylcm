"""Collect and validate a regime's case-piece metadata for NBEGM.

A case-piece model declares, for one or more DAG outputs, a smooth formula per
side of a Boolean case boundary (see `lcm.case_piece`). NBEGM solves each case
separately so that within a case the Euler RHS is smooth. This module reads the
declarations off a regime's function pool — structured boundaries,
when/otherwise piece sets, piecewise-affine schedules — and validates coverage;
the solver resolves the collected registry into its per-case specification.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from _lcm.typing import FunctionName
from lcm.case_piece import CaseBoundary, PieceMeta, PiecewiseAffineMeta
from lcm.exceptions import NBEGMCaseError
from lcm.phased import Phased


@dataclass(frozen=True)
class PieceSet:
    """The pair of pieces covering both sides of one split output."""

    output: FunctionName
    """Name of the DAG output the pieces produce."""
    predicate_name: FunctionName
    """Name of the case-boundary predicate that splits the output."""
    when_func: FunctionName
    """Name of the piece producing the output where the predicate holds."""
    otherwise_func: FunctionName
    """Name of the piece producing the output where the predicate fails."""


@dataclass(frozen=True)
class NBEGMRegistry:
    """Collected case-piece metadata of one regime's function pool."""

    boundaries: MappingProxyType[FunctionName, CaseBoundary]
    """Immutable mapping of predicate name to its structured comparison."""
    piece_sets: tuple[PieceSet, ...]
    """Tuple of fully-covered split outputs, one per (output, predicate)."""
    piecewise_affine_schedules: tuple[PiecewiseAffineMeta, ...]
    """Tuple of declared piecewise-affine schedules, one per schedule output."""


def collect_nbegm_metadata(
    *,
    functions: Mapping[FunctionName, object],
) -> NBEGMRegistry:
    """Collect and validate the case-piece metadata of a function pool.

    Args:
        functions: Mapping of function name to a regime's DAG functions, some
            carrying `__lcm_case_boundary__` or `__lcm_piece__` metadata. A
            `Phased` entry is read through its solve-phase variant, which is
            where the declaration lives.

    Returns:
        The collected metadata as a `NBEGMRegistry`.

    Raises:
        NBEGMCaseError: If a split output is not covered by exactly one `when`
            and one `otherwise` piece, a piece references a predicate that is
            not a declared case boundary, a schedule declares no breakpoint or
            repeats another schedule's output, or an entry is neither callable
            nor inspectable.

    """
    resolved = resolve_declaration_pool(functions=functions)
    boundaries = _collect_boundaries(resolved)
    piece_sets = _collect_piece_sets(resolved, boundaries=boundaries)
    schedules = _collect_piecewise_affine_schedules(resolved)
    return NBEGMRegistry(
        boundaries=MappingProxyType(boundaries),
        piece_sets=piece_sets,
        piecewise_affine_schedules=schedules,
    )


def resolve_declaration_pool(
    *,
    functions: Mapping[FunctionName, object],
) -> dict[FunctionName, Callable[..., object]]:
    """Return the callable solve-phase pool the NB-EGM declarations live on.

    `Regime.functions` legitimately holds `Phased` entries (a solve/simulate
    pair) and `None` entries (model-level broadcast masks). Reading decorator
    attributes straight off those silently yields nothing, so a declaration on
    the solve variant would be invisible to every collector. Resolving here
    keeps the collectors' `getattr` fail-open behaviour honest: after this,
    a missing attribute really does mean "no declaration".

    Args:
        functions: Mapping of function name to a regime's DAG entries.

    Returns:
        Mapping of function name to the callable carrying its declarations.

    Raises:
        NBEGMCaseError: If an entry is neither `None`, a `Phased` pair, nor
            callable, so its declarations cannot be read.

    """
    resolved: dict[FunctionName, Callable[..., object]] = {}
    for name, func in functions.items():
        entry = func.solve if isinstance(func, Phased) else func
        if entry is None:
            continue
        if not callable(entry):
            msg = (
                f"Regime function {name!r} is a {type(entry).__name__}, which "
                "NBEGM cannot inspect for case-piece or piecewise-affine "
                "declarations. Provide a callable (or a `Phased` pair of them)."
            )
            raise NBEGMCaseError(msg)
        resolved[name] = cast("Callable[..., object]", entry)
    return resolved


def _collect_piecewise_affine_schedules(
    functions: Mapping[FunctionName, Callable[..., object]],
) -> tuple[PiecewiseAffineMeta, ...]:
    """Read every declared piecewise-affine schedule, one per schedule output."""
    schedules: list[PiecewiseAffineMeta] = []
    declaring_function: dict[FunctionName, FunctionName] = {}
    for name, func in functions.items():
        meta: PiecewiseAffineMeta | None = getattr(
            func, "__lcm_piecewise_affine__", None
        )
        if meta is None:
            continue
        if not meta.breakpoints:
            msg = (
                f"Piecewise-affine schedule {name!r} declares no breakpoint. Add "
                "at least one `lcm.affine_breakpoint(threshold=..., kind=...)`, or "
                "drop the decorator — an empty schedule still routes the regime "
                "through the breakpoint-aware kernels."
            )
            raise NBEGMCaseError(msg)
        if meta.output in declaring_function:
            msg = (
                f"Piecewise-affine schedules {declaring_function[meta.output]!r} "
                f"and {name!r} both declare the output {meta.output!r}. NBEGM "
                "reads one schedule per output; its threshold parameters are "
                "keyed by output name and would collide. Merge them into one "
                "schedule with all breakpoints."
            )
            raise NBEGMCaseError(msg)
        declaring_function[meta.output] = name
        schedules.append(meta)
    return tuple(schedules)


def _collect_boundaries(
    functions: Mapping[FunctionName, Callable[..., object]],
) -> dict[FunctionName, CaseBoundary]:
    """Read every structured case boundary in the function pool."""
    boundaries: dict[FunctionName, CaseBoundary] = {}
    for name, func in functions.items():
        boundary = (
            func
            if isinstance(func, CaseBoundary)
            else getattr(func, "__lcm_case_boundary__", None)
        )
        if isinstance(boundary, CaseBoundary):
            boundaries[name] = boundary
    return boundaries


def _collect_piece_sets(
    functions: Mapping[FunctionName, Callable[..., object]],
    *,
    boundaries: Mapping[FunctionName, CaseBoundary],
) -> tuple[PieceSet, ...]:
    """Group pieces by (output, predicate) and require both sides exactly once."""
    boundary_name_by_identity = {
        id(boundary): name for name, boundary in boundaries.items()
    }
    sides: dict[tuple[FunctionName, FunctionName], dict[str, FunctionName]] = {}
    for name, func in functions.items():
        meta: PieceMeta | None = getattr(func, "__lcm_piece__", None)
        if meta is None:
            continue
        predicate_name = boundary_name_by_identity.get(id(meta.predicate))
        if predicate_name is None:
            msg = (
                f"Piece {name!r} splits {meta.output!r} on a case boundary that "
                "is not present in the regime's function mapping. Add that exact "
                "`lcm.case_boundary(...)` object to `functions`."
            )
            raise NBEGMCaseError(msg)
        key = (meta.output, predicate_name)
        bucket = sides.setdefault(key, {})
        if meta.side in bucket:
            msg = (
                f"Output {meta.output!r} has two {meta.side!r} pieces for "
                f"{predicate_name!r} ({bucket[meta.side]!r} and {name!r}); "
                f"each side must be covered exactly once."
            )
            raise NBEGMCaseError(msg)
        bucket[meta.side] = name

    piece_sets: list[PieceSet] = []
    for (output, predicate_name), bucket in sides.items():
        missing = {"when", "otherwise"} - set(bucket)
        if missing:
            side = missing.pop()
            msg = (
                f"Output {output!r} split on {predicate_name!r} is missing its "
                f"{side!r} piece. Cover both sides with `@lcm.piece({output!r}, "
                f"{side}={predicate_name})`."
            )
            raise NBEGMCaseError(msg)
        piece_sets.append(
            PieceSet(
                output=output,
                predicate_name=predicate_name,
                when_func=bucket["when"],
                otherwise_func=bucket["otherwise"],
            )
        )
    return tuple(piece_sets)
