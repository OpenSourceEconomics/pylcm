"""Recognising a declared lower bound by what it says.

An endogenous-grid solve enforces its borrowing limit through the savings grid,
so a constraint of the form `post_decision >= number` is a claim about that
grid rather than work for the engine to repeat. Which constraints those are is
read off the condition itself, so the convenience constructor and a hand-written
comparison of the same shape are recognised alike — the sugar is a way to spell
a comparison, not a privileged kind of one.
"""

from _lcm.constraints.dispositions import (
    ConstraintContext,
    Proof,
    ProvedByConstruction,
)
from _lcm.constraints.ir import Const, Ref
from _lcm.constraints.processed import ProcessedConstraint
from _lcm.constraints.routes import BoundConstraint, StructuralProof
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


def proves_the_savings_grids_lower_bound(
    *, post_decision: FunctionName
) -> StructuralProof:
    """Build the proof that a savings grid already enforces a declared bound.

    An endogenous-grid solve inverts on its savings grid, so that grid's lowest
    node *is* the limit the solve enforces. A declared bound on the
    post-decision state the grid spans therefore carries no information the
    solve grid does not already carry — but it carries it as a *claim*, checked
    against the grid when the model is built, which is what makes proving it
    different from ignoring it. The declaration remains executable in
    simulation, where the phase-resolved post-decision function can differ.

    Only a bound on that state is proved. A lower bound on any other name says
    nothing about the grid, so nothing has proved it and it stays an ordinary
    constraint; discharging it on the strength of its shape alone would
    silently drop a constraint the model relies on.

    Args:
        post_decision: Name of the post-decision state the savings grid spans.

    Returns:
        The proof, for a site to consult.

    """

    def prove(
        *,
        bound: BoundConstraint,
        context: ConstraintContext,  # noqa: ARG001
    ) -> ProvedByConstruction | None:
        declaration = lower_bound_declaration(constraint=bound.constraint)
        if declaration is None or declaration[0] != post_decision:
            return None
        surfaces = bound.constraint.boundary_surfaces
        return ProvedByConstruction(
            constraint=bound.constraint,
            proof=Proof(
                reason=(
                    f"the savings grid the solve inverts on spans "
                    f"'{post_decision}', and its lowest node is the limit "
                    f"enforced — checked against this declaration when the "
                    f"model was built"
                ),
                surface=surfaces[0] if surfaces else None,
            ),
        )

    return prove
