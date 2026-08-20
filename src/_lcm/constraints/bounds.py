"""Recognising a declared lower bound by what it says.

An endogenous-grid solve enforces its borrowing limit through the savings grid,
so a constraint of the form `post_decision >= number` is a claim about that
grid rather than work for the engine to repeat. Which constraints those are is
read off the condition itself, so the convenience constructor and a hand-written
comparison of the same shape are recognised alike — the sugar is a way to spell
a comparison, not a privileged kind of one.
"""

from collections.abc import Mapping
from types import MappingProxyType

from _lcm.constraints.ir import Const, Ref
from _lcm.constraints.processed import (
    ProcessedConstraint,
    ProcessedConstraintsMapping,
)
from _lcm.typing import FunctionName


def lower_bound_declaration(
    *, constraint: ProcessedConstraint
) -> tuple[str, float] | None:
    """Return the single lower bound a constraint declares, if that is all it says.

    Args:
        constraint: The normalized constraint to read.

    Returns:
        The bounded name and the number it is bounded below by, or `None` when
        the constraint says anything else — several comparisons, a bound in the
        other direction, a comparison against another reference, or an opaque
        predicate.

    """
    surfaces = constraint.boundary_surfaces
    if surfaces is None or len(surfaces) != 1:
        return None
    surface = surfaces[0]
    if surface.op != ">=":
        return None
    if not isinstance(surface.left, Ref) or not isinstance(surface.right, Const):
        return None
    return surface.left.name, float(surface.right.value)


def without_proved_lower_bounds(
    *,
    constraints: Mapping[FunctionName, ProcessedConstraint],
    proved_post_decision: FunctionName | None,
) -> ProcessedConstraintsMapping:
    """Drop declared lower bounds a savings grid already enforces.

    A declared bound is a claim about the savings grid, checked when the model
    is built. Once proved it carries no information the grid does not already
    carry: the solve enforces it by inverting on that grid, and the simulate
    phase enforces it through the mask synthesized from the same lowest node.
    Leaving it in the engine's constraint set would have it evaluated a second
    time — and the solve's feasibility predicate is built per discrete combo,
    which is not a place a continuous post-decision state can be read.

    A solver that does not invert on a savings grid — grid search is the case —
    enforces nothing implicitly, so for it the declaration is an ordinary
    constraint and must survive. The same declaration is therefore load-bearing
    in one regime and redundant in another, which is the point: one spelling
    that both arms of a model honour.

    Only a bound on the state that grid spans is dropped. A lower bound on any
    other name says nothing about the grid, so nothing has proved it and it
    stays an ordinary constraint; dropping it on the strength of its shape
    alone would silently discard a constraint the model relies on.

    Args:
        constraints: The regime's normalized constraints.
        proved_post_decision: Name of the post-decision state whose bound the
            solver enforces through its savings grid, or `None` when the solver
            enforces nothing implicitly.

    Returns:
        Immutable mapping of the constraints the engine evaluates.

    """
    if proved_post_decision is None:
        return MappingProxyType(dict(constraints))
    return MappingProxyType(
        {
            name: constraint
            for name, constraint in constraints.items()
            if not _bounds_the_proved_state(
                constraint=constraint, proved_post_decision=proved_post_decision
            )
        }
    )


def _bounds_the_proved_state(
    *, constraint: ProcessedConstraint, proved_post_decision: FunctionName
) -> bool:
    declaration = lower_bound_declaration(constraint=constraint)
    return declaration is not None and declaration[0] == proved_post_decision
