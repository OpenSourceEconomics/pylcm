"""Recognising a declared lower bound by what it says.

An endogenous-grid solve enforces its borrowing limit through the savings grid,
so a constraint of the form `post_decision >= number` is a claim about that
grid rather than work for the engine to repeat. Which constraints those are is
read off the condition itself, so the convenience constructor and a hand-written
comparison of the same shape are recognised alike — the sugar is a way to spell
a comparison, not a privileged kind of one.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from _lcm.constraints.dispositions import (
    ConstraintContext,
    Proof,
    ProvedByConstruction,
)
from _lcm.constraints.ir import Const, Ref
from _lcm.constraints.processed import ProcessedConstraint
from _lcm.constraints.routes import BoundConstraint, StructuralProof
from _lcm.dtypes import canonical_float_dtype
from _lcm.post_decision_bound import PostDecisionBoundSpec
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
    *, post_decision: FunctionName, bound_spec: PostDecisionBoundSpec | None = None
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

    return _SavingsGridLowerBoundProof(
        post_decision=post_decision, bound_spec=bound_spec
    )


def matches_installed_literal_bound(
    *,
    constraint: ProcessedConstraint,
    bound_spec: PostDecisionBoundSpec | None,
) -> bool:
    """Require the declared name and canonical literal to match the installed floor."""
    declaration = lower_bound_declaration(constraint=constraint)
    if declaration is None or bound_spec is None or bound_spec.kind != "literal":
        return False
    if declaration[0] != bound_spec.post_decision or bound_spec.literal_value is None:
        return False
    dtype = canonical_float_dtype()
    return bool(
        jnp.array_equal(
            jnp.asarray(declaration[1], dtype=dtype),
            jnp.asarray(bound_spec.literal_value, dtype=dtype),
        )
    )


@dataclass(frozen=True, kw_only=True)
class _SavingsGridLowerBoundProof:
    """Proof that the savings grid enforces a declared bound on `post_decision`."""

    post_decision: FunctionName
    """Name of the post-decision state the savings grid spans."""
    bound_spec: PostDecisionBoundSpec | None
    """Exact floor capability installed by the bound grid configuration."""

    def __call__(
        self,
        *,
        bound: BoundConstraint,
        context: ConstraintContext,  # noqa: ARG002
    ) -> ProvedByConstruction | None:
        if self.bound_spec is not None and self.bound_spec.kind == "dag":
            surfaces = bound.constraint.boundary_surfaces
            if (
                surfaces is None
                or len(surfaces) != 1
                or surfaces[0].op != ">="
                or not isinstance(surfaces[0].left, Ref)
                or not isinstance(surfaces[0].right, Ref)
                or surfaces[0].left.name != self.bound_spec.post_decision
                or surfaces[0].right.name != self.bound_spec.lower_target
                or bound.constraint.name not in self.bound_spec.declaration_names
            ):
                return None
            return ProvedByConstruction(
                constraint=bound.constraint,
                proof=Proof(
                    reason="the installed canonical floor shifts every savings node",
                    surface=surfaces[0],
                ),
            )
        declaration = lower_bound_declaration(constraint=bound.constraint)
        if (
            declaration is None
            or declaration[0] != self.post_decision
            or not matches_installed_literal_bound(
                constraint=bound.constraint,
                bound_spec=self.bound_spec,
            )
        ):
            return None
        surfaces = bound.constraint.boundary_surfaces
        return ProvedByConstruction(
            constraint=bound.constraint,
            proof=Proof(
                reason=(
                    f"the savings grid the solve inverts on spans "
                    f"'{self.post_decision}', and its lowest node is the limit "
                    f"enforced — checked against this declaration when the "
                    f"model was built"
                ),
                surface=surfaces[0] if surfaces else None,
            ),
        )
