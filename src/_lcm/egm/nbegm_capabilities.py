"""What the case-piece solvers can do with a declared constraint.

The NBEGM kernels recover consumption by inverting the Euler equation at each
node of a savings grid: the action is produced first and the liquid state falls
out of the budget identity afterwards. There is no point in that step at which a
predicate over `(state, action)` is evaluable, and the candidates the kernels
publish are never masked by one. The declaration is therefore that no name is
readable where a constraint would be called, which sends every constraint to
`Reject` unless a proof claims it first.

One proof claims anything: the borrowing limit the savings grid already
enforces. It keys on the comparison a declaration stands for rather than on the
kind of object the user constructed, so a bound written out as
`ref("savings") >= 0.0` is discharged exactly as the convenience constructor's
is. Keying on the constructor instead would admit one spelling and refuse the
other while both describe the same feasible set.
"""

from dataclasses import dataclass

from _lcm.constraints.capabilities import BoundaryCompiler, StructuralProof
from _lcm.constraints.dispositions import (
    ConstraintContext,
    EvaluationStage,
    Proof,
    ProvedByConstruction,
)
from _lcm.constraints.ir import Compare, Const, Ref
from _lcm.constraints.processed import ProcessedConstraint
from _lcm.grids import ContinuousGrid
from _lcm.typing import FunctionName


def case_piece_capabilities(
    *, savings_grid: ContinuousGrid, post_decision_function: FunctionName | None
) -> _CasePieceCapabilities:
    """Declare what a case-piece kernel can do with a constraint.

    Args:
        savings_grid: The grid the kernel inverts on, whose lowest node is the
            borrowing limit it enforces.
        post_decision_function: Name of the post-decision state that grid
            spans, or `None` before the solver has been bound to a margin —
            nothing can be proved against a grid whose state is not yet
            named, but the empty allow-list holds either way.

    Returns:
        The solver's constraint capabilities.

    """
    return _CasePieceCapabilities(
        savings_grid=savings_grid, post_decision_function=post_decision_function
    )


@dataclass(frozen=True)
class _CasePieceCapabilities:
    """The case-piece kernels' declaration, as `ConstraintCapabilities`."""

    savings_grid: ContinuousGrid
    """Grid the kernel inverts on."""

    post_decision_function: FunctionName | None
    """Post-decision state that grid spans, or `None` before margin binding."""

    @property
    def pre_inner_available_names(self) -> frozenset[str] | None:
        """No name is readable where the kernel would call a constraint.

        An empty allow-list, not `None`: `None` would say the kernel imposes no
        restriction and reads everything, which is grid search's declaration and
        the opposite of this one.
        """
        return frozenset()

    @property
    def evaluation_stage(self) -> EvaluationStage:
        """Unreachable — the empty allow-list refuses before a stage is used."""
        return "state_action"

    @property
    def structural_proofs(self) -> tuple[StructuralProof, ...]:
        """The savings grid's own borrowing limit, and nothing else."""
        return (self._prove_the_grids_borrowing_limit,)

    @property
    def boundary_compilers(self) -> tuple[BoundaryCompiler, ...]:
        """None: a case boundary reaches the kernel as case metadata, not as a
        constraint, so there is no constraint boundary for it to compile."""
        return ()

    def _prove_the_grids_borrowing_limit(
        self,
        *,
        constraint: ProcessedConstraint,
        context: ConstraintContext,  # noqa: ARG002
    ) -> ProvedByConstruction | None:
        """Discharge a lower bound the savings grid's lowest node already imposes.

        Takes `context` because the protocol passes it, and reads none of it:
        the grid and the state it spans are the solver's own configuration,
        so the verdict does not vary with the regime or the phase.

        Declines rather than refuses when the shape does not match, so a
        constraint this proof has nothing to say about falls through to the
        allow-list and is refused there with a message about what it reads.
        """
        if self.post_decision_function is None:
            return None
        bound = _lower_bound(constraint=constraint)
        if bound is None:
            return None
        if bound.name != self.post_decision_function:
            return None
        if not _matches_grid_start(grid=self.savings_grid, value=bound.value):
            return None
        return ProvedByConstruction(
            constraint=constraint,
            proof=Proof(
                reason=(
                    f"The savings grid the kernel inverts on starts at "
                    f"{bound.value}, so it enforces this bound on "
                    f"'{self.post_decision_function}' at every node it "
                    f"publishes."
                ),
                surface=bound.surface,
            ),
        )


@dataclass(frozen=True)
class _LowerBound:
    """A constraint's single `<name> >= <number>` surface, read apart."""

    surface: Compare
    """The surface itself, as the discharged constraint's proof reports it."""

    name: FunctionName
    """Name the bound is imposed on."""

    value: float
    """Number the name is bounded below by."""


def _lower_bound(*, constraint: ProcessedConstraint) -> _LowerBound | None:
    """Read the constraint as `<name> >= <number>`, if that is what it is.

    Returns `None` for anything else — several surfaces, an operator other than
    `>=`, a bound that is not a plain number, or either side being an expression
    rather than a name and a constant.
    """
    surfaces = constraint.boundary_surfaces
    if surfaces is None or len(surfaces) != 1:
        return None
    surface = surfaces[0]
    if surface.op != ">=":
        return None
    if not isinstance(surface.left, Ref) or not isinstance(surface.right, Const):
        return None
    if not isinstance(surface.right.value, float | int):
        return None
    return _LowerBound(
        surface=surface,
        name=surface.left.name,
        value=float(surface.right.value),
    )


def _matches_grid_start(*, grid: ContinuousGrid, value: float) -> bool:
    """Whether a declared bound is the grid's own lowest node.

    Compared against the grid's *declared* start where it has one, keeping both
    sides in user space: the materialized node carries the grid's floating-point
    representation, which would reject a faithful declaration at reduced
    precision.
    """
    declared_start = getattr(grid, "start", None)
    grid_low = (
        float(declared_start) if declared_start is not None else float(grid.to_jax()[0])
    )
    return value == grid_low
