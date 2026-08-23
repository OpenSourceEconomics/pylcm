"""Structured case boundaries and formula-piece declarations for NBEGM.

A *case* is a region of the state space carved out by a Boolean predicate (e.g.
Medicaid eligibility); a *piece* is the smooth formula a DAG output takes inside
one side of that predicate. A case boundary is one executable, inspectable
`Condition`; the piece decorators attach that same object to each user formula.
Model finalization composes every complete pair into its declared output with
``where(predicate, when, otherwise)``, so all solvers read the same predicate;
NBEGM additionally lowers its comparison and applies the declared open/closed
ownership at the exact boundary query.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from _lcm.constraints.ir import Compare, Condition
from lcm.exceptions import NBEGMCaseError

type BoundaryKind = Literal["continuous_kink", "jump", "hard_constraint"]
type EqualityOwner = Literal["when", "otherwise"]
type CoordinateEqualityOwner = Literal["below", "above"]


@dataclass(frozen=True, eq=False)
class CaseBoundary(Condition):
    """A structured binary comparison that partitions a case-piece output."""

    kind: BoundaryKind
    """Discontinuity kind at the comparison surface."""


@dataclass(frozen=True)
class PieceMeta:
    """Metadata attached to a formula piece selecting one side of a predicate."""

    output: str
    """Name of the DAG output this piece produces."""
    predicate: CaseBoundary
    """Structured case boundary that splits the output."""
    side: Literal["when", "otherwise"]
    """Predicate side this piece applies to."""


@dataclass(frozen=True)
class AffineBreakpoint:
    """One threshold of a piecewise-affine schedule on a monotone variable."""

    threshold: str
    """Name of the DAG variable or parameter holding the threshold value. When the
    threshold lives inside a `MappingLeaf` param, this is the leaf's name and
    `threshold_subkey` selects the entry within it."""
    kind: BoundaryKind
    """Discontinuity kind at the threshold (a bracket edge is a continuous kink)."""
    equality_owner: CoordinateEqualityOwner = "above"
    """Side of the schedule coordinate containing the exact threshold."""
    indexed_by: str | None = None
    """Name of the ride-along state indexing the threshold table, or `None` for a
    scalar threshold. When set, the threshold parameter is a table read per
    ride-along cell as `threshold[cell_state, static_index]`."""
    static_index: int | None = None
    """Static column index into the threshold table (e.g. a bracket edge), applied
    after the ride-along-state row index. `None` leaves the indexed value as-is."""
    threshold_subkey: str | None = None
    """Entry to select inside a `MappingLeaf` threshold param (`leaf.data[subkey]`),
    or `None` when the threshold param is a bare array. Resolved before the
    ride-along-state row index and static column index."""


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


def case_boundary(
    condition: Condition,
    *,
    kind: BoundaryKind,
) -> CaseBoundary:
    """Declare an executable binary case split from one structured condition.

    Args:
        condition: One comparison built from `lcm.ref`.
        kind: Discontinuity kind at the comparison surface.

    Returns:
        The same expression as an executable, inspectable case boundary.

    Raises:
        NBEGMCaseError: If the expression is not one supported binary ordering.

    """
    expression = condition.expression
    if not isinstance(expression, Compare) or expression.op not in {
        "<",
        "<=",
        ">",
        ">=",
    }:
        msg = (
            "A case boundary is exactly one `<`, `<=`, `>`, or `>=` "
            "comparison built from `lcm.ref`; conjunctions, unions, "
            "complements, implications, equality tests, and opaque callables "
            "cannot define one binary split."
        )
        raise NBEGMCaseError(msg)
    return CaseBoundary(expression=expression, kind=kind)


def piece[F: Callable[..., object]](
    *,
    output: str,
    when: CaseBoundary | None = None,
    otherwise: CaseBoundary | None = None,
) -> Callable[[F], F]:
    """Mark a DAG function as the formula for one side of a case boundary.

    Args:
        output: Name of the DAG output this piece produces.
        when: Case-boundary predicate selecting the side where this piece applies.
        otherwise: Case-boundary predicate selecting the complementary side.

    Returns:
        A decorator that attaches the metadata in `__lcm_piece__` and returns the
        function unchanged.

    Raises:
        NBEGMCaseError: If neither or both of `when`/`otherwise` are given.

    """
    if when is not None and otherwise is None:
        predicate = when
        side: Literal["when", "otherwise"] = "when"
    elif otherwise is not None and when is None:
        predicate = otherwise
        side = "otherwise"
    else:
        msg = (
            f"Piece for output {output!r} must name exactly one of `when=` or "
            "`otherwise=`."
        )
        raise NBEGMCaseError(msg)

    def attach_piece(func: F) -> F:
        func.__lcm_piece__ = PieceMeta(  # ty: ignore[unresolved-attribute]
            output=output,
            predicate=predicate,
            side=side,
        )
        return func

    return attach_piece


def affine_breakpoint(
    *,
    threshold: str,
    kind: BoundaryKind = "continuous_kink",
    equality: CoordinateEqualityOwner = "above",
    indexed_by: str | None = None,
    static_index: int | None = None,
) -> AffineBreakpoint:
    """Declare one threshold of a piecewise-affine schedule.

    Args:
        threshold: Name of the DAG variable or parameter holding the threshold. A
            single-dotted name `leaf.subkey` reads the threshold from a
            `MappingLeaf` param: `leaf` is the parameter, `subkey` the entry
            within its `.data`.
        kind: Discontinuity kind at the threshold; a bracket edge is a continuous
            kink (the schedule is continuous, only its slope changes).
        equality: Side of the schedule coordinate that owns the exact threshold.
        indexed_by: Name of the ride-along state indexing the threshold table. When
            given, the threshold parameter is a table and NBEGM reads each cell's
            threshold as `threshold[cell_state, static_index]`; the bare scalar
            form (`None`) is unchanged.
        static_index: Static column index into the threshold table (e.g. a bracket
            edge), applied after the ride-along-state row index.

    Returns:
        The threshold as an `AffineBreakpoint`.

    Raises:
        NBEGMCaseError: If `threshold` carries more than one dot — a `MappingLeaf`
            holds a flat `.data`, so there is no nested entry to reach.

    """
    leaf, _, subkey = threshold.partition(".")
    if "." in subkey:
        msg = (
            f"Breakpoint threshold {threshold!r} carries more than one dot. A "
            "`MappingLeaf` param holds a flat `.data`, so a threshold names at "
            "most `leaf.subkey`."
        )
        raise NBEGMCaseError(msg)
    return AffineBreakpoint(
        threshold=leaf,
        kind=kind,
        equality_owner=equality,
        indexed_by=indexed_by,
        static_index=static_index,
        threshold_subkey=subkey or None,
    )


def piecewise_affine[F: Callable[..., object]](
    *,
    output: str,
    variable: str,
    breakpoints: tuple[AffineBreakpoint, ...],
) -> Callable[[F], F]:
    """Mark a DAG function as a piecewise-affine schedule on a monotone variable.

    The decorated function stays an ordinary DAG node; the decorator only records
    its schedule metadata in `__lcm_piecewise_affine__`, so the model still solves
    identically under `GridSearch`. NBEGM reads the metadata to merge each
    threshold into the liquid-axis interval partition and to recover the active
    affine segment per interval.

    Args:
        output: Name of the DAG output this schedule produces.
        variable: Name of the monotone variable the thresholds compare against.
        breakpoints: Ordered thresholds splitting the schedule into segments.

    Returns:
        A decorator that attaches the metadata and returns the function unchanged.

    """

    def attach_schedule(func: F) -> F:
        func.__lcm_piecewise_affine__ = PiecewiseAffineMeta(  # ty: ignore[unresolved-attribute]
            output=output,
            variable=variable,
            breakpoints=breakpoints,
        )
        return func

    return attach_schedule


def smooth_helper[F: Callable[..., object]](func: F) -> F:
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
