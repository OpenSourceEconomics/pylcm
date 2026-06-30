"""Case-boundary and formula-piece decorators for the BQSEGM solver.

A *case* is a region of the state space carved out by a Boolean predicate (e.g.
Medicaid eligibility); a *piece* is the smooth formula a DAG output takes inside
one side of that predicate. The decorators here only attach metadata to a user's
existing DAG functions and return them unchanged — they never wrap or alter
runtime behavior, so a model still solves identically under BruteForce. BQSEGM
reads the metadata to lower each case to a smooth per-case DAG and to apply
open/closed endpoint eligibility at the exact boundary query.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from lcm.exceptions import BQSEGMCaseError

type BoundaryKind = Literal["continuous_kink", "jump", "hard_constraint"]
type EqualityOwner = Literal["when", "otherwise"]


@dataclass(frozen=True)
class BoundarySurface:
    """One equality surface `variable == threshold` of a case boundary."""

    variable: str
    """Name of the DAG variable compared against the threshold."""
    threshold: str
    """Name of the DAG variable (or parameter) holding the threshold value."""
    equality_owner: EqualityOwner
    """Predicate side that owns the exact-equality point (`when` or `otherwise`)."""
    kind: BoundaryKind
    """Discontinuity kind at the surface: kink, jump, or hard constraint."""


@dataclass(frozen=True)
class CaseBoundaryMeta:
    """Metadata attached to a predicate declaring its case boundaries."""

    boundaries: tuple[BoundarySurface, ...]
    """Tuple of equality surfaces that together define the case boundary."""


@dataclass(frozen=True)
class PieceMeta:
    """Metadata attached to a formula piece selecting one side of a predicate."""

    output: str
    """Name of the DAG output this piece produces."""
    predicate_name: str
    """Name of the case-boundary predicate that splits the output."""
    side: Literal["when", "otherwise"]
    """Predicate side this piece applies to."""


@dataclass(frozen=True)
class AffineBreakpoint:
    """One threshold of a piecewise-affine schedule on a monotone variable."""

    threshold: str
    """Name of the DAG variable or parameter holding the threshold value."""
    kind: BoundaryKind
    """Discontinuity kind at the threshold (a bracket edge is a continuous kink)."""
    indexed_by: str | None = None
    """Name of the ride-along state indexing the threshold table, or `None` for a
    scalar threshold. When set, the threshold parameter is a table read per
    ride-along cell as `threshold[cell_state, static_index]`."""
    static_index: int | None = None
    """Static column index into the threshold table (e.g. a bracket edge), applied
    after the ride-along-state row index. `None` leaves the indexed value as-is."""


@dataclass(frozen=True)
class PiecewiseAffineMeta:
    """Metadata attached to a single piecewise-affine schedule.

    A schedule (a tax with brackets, a cost-sharing step, a phase-out) is affine
    between thresholds of one monotone variable. Each threshold contributes a
    breakpoint to the same liquid-axis partition the case boundaries feed, so the
    solver treats a jump and a bracket edge uniformly.
    """

    output: str
    """Name of the DAG output this schedule produces."""
    variable: str
    """Name of the monotone variable the schedule's thresholds compare against."""
    breakpoints: tuple[AffineBreakpoint, ...]
    """Ordered thresholds splitting the schedule into affine segments."""


def boundary(
    variable: str,
    threshold: str,
    *,
    equality: EqualityOwner,
    kind: BoundaryKind,
) -> BoundarySurface:
    """Declare one equality surface of a case boundary.

    Args:
        variable: Name of the DAG variable compared against the threshold.
        threshold: Name of the DAG variable or parameter holding the threshold.
        equality: Predicate side that owns the exact-equality point.
        kind: Discontinuity kind at the surface.

    Returns:
        The equality surface as a `BoundarySurface`.

    """
    return BoundarySurface(
        variable=variable,
        threshold=threshold,
        equality_owner=equality,
        kind=kind,
    )


def case_boundary(
    *boundaries: BoundarySurface | tuple[str, str],
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Mark a Boolean DAG function as a case boundary with declared surfaces.

    The decorated function stays an ordinary DAG node returning a Boolean array;
    the decorator only records its equality surfaces in `__lcm_case_boundary__`.

    Args:
        *boundaries: One or more `boundary(...)` surfaces. A bare
            `(variable, threshold)` tuple is rejected because it cannot declare
            which side owns equality — use `lcm.boundary(...)` instead.

    Returns:
        A decorator that attaches the metadata and returns the function
        unchanged.

    """
    coerced = tuple(_coerce_boundary(boundary_spec) for boundary_spec in boundaries)

    def deco(func: Callable[..., object]) -> Callable[..., object]:
        func.__lcm_case_boundary__ = CaseBoundaryMeta(coerced)  # ty: ignore[unresolved-attribute]
        return func

    return deco


def piece(
    output: str,
    *,
    when: Callable[..., object] | None = None,
    otherwise: Callable[..., object] | None = None,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Mark a DAG function as the formula for one side of a case boundary.

    Args:
        output: Name of the DAG output this piece produces.
        when: Case-boundary predicate selecting the side where this piece applies.
        otherwise: Case-boundary predicate selecting the complementary side.

    Returns:
        A decorator that attaches the metadata in `__lcm_piece__` and returns the
        function unchanged.

    Raises:
        ValueError: If neither or both of `when`/`otherwise` are given.

    """
    if when is not None and otherwise is None:
        predicate: Callable[..., object] = when
        side: Literal["when", "otherwise"] = "when"
    elif otherwise is not None and when is None:
        predicate = otherwise
        side = "otherwise"
    else:
        msg = "Use exactly one of when= or otherwise=."
        raise ValueError(msg)

    def deco(func: Callable[..., object]) -> Callable[..., object]:
        func.__lcm_piece__ = PieceMeta(  # ty: ignore[unresolved-attribute]
            output=output,
            predicate_name=predicate.__name__,  # ty: ignore[unresolved-attribute]
            side=side,
        )
        return func

    return deco


def affine_breakpoint(
    threshold: str,
    *,
    kind: BoundaryKind = "continuous_kink",
    indexed_by: str | None = None,
    static_index: int | None = None,
) -> AffineBreakpoint:
    """Declare one threshold of a piecewise-affine schedule.

    Args:
        threshold: Name of the DAG variable or parameter holding the threshold.
        kind: Discontinuity kind at the threshold; a bracket edge is a continuous
            kink (the schedule is continuous, only its slope changes).
        indexed_by: Name of the ride-along state indexing the threshold table. When
            given, the threshold parameter is a table and BQSEGM reads each cell's
            threshold as `threshold[cell_state, static_index]`; the bare scalar
            form (`None`) is unchanged.
        static_index: Static column index into the threshold table (e.g. a bracket
            edge), applied after the ride-along-state row index.

    Returns:
        The threshold as an `AffineBreakpoint`.

    """
    return AffineBreakpoint(
        threshold=threshold,
        kind=kind,
        indexed_by=indexed_by,
        static_index=static_index,
    )


def piecewise_affine(
    output: str,
    *,
    variable: str,
    breakpoints: tuple[AffineBreakpoint, ...],
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Mark a DAG function as a piecewise-affine schedule on a monotone variable.

    The decorated function stays an ordinary DAG node; the decorator only records
    its schedule metadata in `__lcm_piecewise_affine__`, so the model still solves
    identically under `GridSearch`. BQSEGM reads the metadata to merge each
    threshold into the liquid-axis interval partition and to recover the active
    affine segment per interval.

    Args:
        output: Name of the DAG output this schedule produces.
        variable: Name of the monotone variable the thresholds compare against.
        breakpoints: Ordered thresholds splitting the schedule into segments.

    Returns:
        A decorator that attaches the metadata and returns the function unchanged.

    """

    def deco(func: Callable[..., object]) -> Callable[..., object]:
        func.__lcm_piecewise_affine__ = PiecewiseAffineMeta(  # ty: ignore[unresolved-attribute]
            output=output,
            variable=variable,
            breakpoints=breakpoints,
        )
        return func

    return deco


def smooth_helper(func: Callable[..., object]) -> Callable[..., object]:
    """Attest that a user node's `max`/`clip`/`abs` use is numerical, not economic.

    The smoothness gate rejects piecewise primitives in user economic nodes
    because they usually hide an undeclared case boundary. A reviewed helper whose
    `clip`/`maximum`/`abs` only guards a numerical edge (a positivity floor, an
    overflow clamp) is exempt: this decorator marks it `__lcm_smooth_helper__` and
    returns the same object, so it is skipped by the AST and JAXPR gate.

    Args:
        func: The user helper whose piecewise primitive is numerical, not a
            hidden economic case.

    Returns:
        The same function, marked as a trusted smooth helper.

    """
    func.__lcm_smooth_helper__ = True  # ty: ignore[unresolved-attribute]
    return func


def _coerce_boundary(spec: BoundarySurface | tuple[str, str]) -> BoundarySurface:
    """Coerce a boundary specification into a `BoundarySurface`.

    A `BoundarySurface` passes through. A bare `(variable, threshold)` tuple is
    rejected: equality ownership cannot be inferred without the validator, so the
    explicit `lcm.boundary(..., equality=..., kind=...)` form is required.
    """
    if isinstance(spec, BoundarySurface):
        return spec
    msg = (
        f"Cannot infer equality ownership from {spec!r}. Declare the boundary "
        f"explicitly with `lcm.boundary(variable, threshold, equality=..., "
        f"kind=...)`."
    )
    raise BQSEGMCaseError(msg)
