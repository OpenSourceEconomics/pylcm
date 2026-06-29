"""Mixed-kind breakpoint IR: the solver's uniform view of a case boundary.

BQSEGM merges every institutional boundary on the liquid axis into one sorted
interval partition. That requires a representation independent of which user form
declared the boundary: a Medicaid asset-test jump, a tax-bracket continuous kink,
and a consumption-floor hard constraint all become a `BreakpointSource` carrying
the monotone boundary variable, its threshold, the discontinuity kind, and the
side that owns the exact boundary point.

This module only lifts the declared metadata (`lcm.case_boundary` /
`lcm.boundary`) into that IR. Mapping each source to its asset preimage, merging
the breakpoints into intervals, specializing per-interval formulas, and emitting
per-kind candidates are downstream solver steps.
"""

from dataclasses import dataclass
from typing import Literal

from _lcm.egm.bqsegm import BQSEGMRegistry
from _lcm.typing import FunctionName
from lcm.case_piece import EqualityOwner

type BreakpointKind = Literal[
    "jump", "continuous_kink", "hard_constraint", "open_boundary"
]


@dataclass(frozen=True)
class BreakpointSource:
    """One breakpoint on the liquid axis, in mixed-kind solver IR form.

    A user-declared boundary surface (jump, continuous kink, hard constraint) and
    a solver-synthesized open one-sided limit (`open_boundary`) share this record,
    so the downstream interval partition treats them uniformly.
    """

    variable: FunctionName
    """Name of the monotone boundary variable compared against the threshold."""
    threshold: str
    """Name of the DAG variable or parameter holding the threshold value."""
    kind: BreakpointKind
    """Discontinuity kind: jump, continuous kink, hard constraint, or open limit."""
    equality_owner: EqualityOwner
    """Side that owns the exact boundary point (`when` or `otherwise`)."""


def breakpoint_sources_from_registry(
    registry: BQSEGMRegistry,
) -> tuple[BreakpointSource, ...]:
    """Lift every declared boundary surface to a breakpoint source.

    Args:
        registry: Collected case-piece metadata of one regime's function pool.

    Returns:
        Tuple of breakpoint sources, one per declared equality surface, in
        predicate-then-surface declaration order.

    """
    return tuple(
        BreakpointSource(
            variable=surface.variable,
            threshold=surface.threshold,
            kind=surface.kind,
            equality_owner=surface.equality_owner,
        )
        for meta in registry.boundaries.values()
        for surface in meta.boundaries
    )
