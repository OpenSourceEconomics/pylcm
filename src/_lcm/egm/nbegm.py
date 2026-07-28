"""Collect and validate a regime's case-piece metadata for NBEGM.

A case-piece model declares, for one or more DAG outputs, a smooth formula per
side of a Boolean case boundary (see `lcm.case_piece`). NBEGM solves each case
separately so that within a case the Euler RHS is smooth. This module reads the
decorator metadata off a regime's function pool — boundaries, when/otherwise
piece sets, piecewise-affine schedules — and validates coverage; the solver
resolves the collected registry into its per-case specification.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from _lcm.typing import FunctionName
from lcm.case_piece import CaseBoundaryMeta, PieceMeta, PiecewiseAffineMeta
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

    boundaries: MappingProxyType[FunctionName, CaseBoundaryMeta]
    """Immutable mapping of predicate name to its declared boundary surfaces."""
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
        NBEGMCaseError: If a case boundary declares no surface, a split output
            is not covered by exactly one `when` and one `otherwise` piece, a
            piece references a predicate that is not a declared case boundary, a
            schedule declares no breakpoint or repeats another schedule's
            output, or an entry is neither callable nor inspectable.

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


def jump_moving_state_names(
    *,
    functions: Mapping[FunctionName, object],
    state_names: frozenset[str],
    euler_state_name: str,
) -> frozenset[str]:
    """Get the non-Euler states that move any declared jump's liquid preimage.

    A jump's preimage on the Euler axis is `(threshold - offset) / slope`,
    where offset and slope come from the jump variable's DAG at the cell's
    non-Euler values and the threshold may be a state-indexed table. A state
    moves the preimage when it reaches the variable or the threshold through
    the DAG, or indexes the threshold table. States that reach neither leave
    every cell's published jump locations identical along their axis.

    Args:
        functions: Mapping of function name to the regime's DAG functions.
        state_names: The regime's state names.
        euler_state_name: The Euler-axis state (excluded from the result).

    Returns:
        Frozen set of the state names that move at least one jump preimage.

    """
    resolved = resolve_declaration_pool(functions=functions)
    registry = collect_nbegm_metadata(functions=resolved)
    moving: set[str] = set()
    for schedule in registry.piecewise_affine_schedules:
        moving |= _schedule_jump_state_args(
            schedule=schedule, functions=resolved, state_names=state_names
        )
    for boundary_meta in registry.boundaries.values():
        for surface in boundary_meta.boundaries:
            if surface.kind == "jump":
                moving |= _transitive_state_args(
                    surface.variable, functions=resolved, state_names=state_names
                )
                moving |= _transitive_state_args(
                    surface.threshold, functions=resolved, state_names=state_names
                )
    return frozenset(moving) - {euler_state_name}


def _schedule_jump_state_args(
    *,
    schedule: PiecewiseAffineMeta,
    functions: Mapping[FunctionName, object],
    state_names: frozenset[str],
) -> set[str]:
    """Get the states moving one schedule's jump preimages (empty if no jump)."""
    jump_breakpoints = tuple(
        breakpoint_meta
        for breakpoint_meta in schedule.breakpoints
        if breakpoint_meta.kind == "jump"
    )
    if not jump_breakpoints:
        return set()
    moving = _transitive_state_args(
        schedule.variable, functions=functions, state_names=state_names
    )
    for breakpoint_meta in jump_breakpoints:
        moving |= _transitive_state_args(
            breakpoint_meta.threshold, functions=functions, state_names=state_names
        )
        if breakpoint_meta.indexed_by is not None:
            moving.add(breakpoint_meta.indexed_by)
    return moving


def _transitive_state_args(
    target: str,
    *,
    functions: Mapping[FunctionName, object],
    state_names: frozenset[str],
) -> set[str]:
    """Get the state names reachable from `target` through the function DAG."""
    visited: set[str] = set()
    stack = [target]
    while stack:
        name = stack.pop()
        if name in visited:
            continue
        visited.add(name)
        func = functions.get(name)
        if callable(func):
            stack.extend(inspect.signature(func).parameters)
    return visited & state_names


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
                "at least one `lcm.affine_breakpoint(threshold, kind=...)`, or "
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
) -> dict[FunctionName, CaseBoundaryMeta]:
    """Read every declared case boundary and check it carries a surface."""
    boundaries: dict[FunctionName, CaseBoundaryMeta] = {}
    for name, func in functions.items():
        meta = getattr(func, "__lcm_case_boundary__", None)
        if meta is None:
            continue
        if not meta.boundaries:
            msg = (
                f"Case boundary {name!r} declares no boundary surface. Add at "
                f"least one `lcm.boundary(variable, threshold, equality=..., "
                f"kind=...)`."
            )
            raise NBEGMCaseError(msg)
        boundaries[name] = meta
    return boundaries


def _collect_piece_sets(
    functions: Mapping[FunctionName, Callable[..., object]],
    *,
    boundaries: Mapping[FunctionName, CaseBoundaryMeta],
) -> tuple[PieceSet, ...]:
    """Group pieces by (output, predicate) and require both sides exactly once."""
    sides: dict[tuple[FunctionName, FunctionName], dict[str, FunctionName]] = {}
    for name, func in functions.items():
        meta: PieceMeta | None = getattr(func, "__lcm_piece__", None)
        if meta is None:
            continue
        if meta.predicate_name not in boundaries:
            msg = (
                f"Piece {name!r} splits {meta.output!r} on {meta.predicate_name!r}, "
                f"which is not a declared `case_boundary`. Decorate the predicate "
                f"with `@lcm.case_boundary(...)`."
            )
            raise NBEGMCaseError(msg)
        key = (meta.output, meta.predicate_name)
        bucket = sides.setdefault(key, {})
        if meta.side in bucket:
            msg = (
                f"Output {meta.output!r} has two {meta.side!r} pieces for "
                f"{meta.predicate_name!r} ({bucket[meta.side]!r} and {name!r}); "
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
