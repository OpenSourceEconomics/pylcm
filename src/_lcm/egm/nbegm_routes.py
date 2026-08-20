"""The routes the case-piece solvers walk, and what can happen along them.

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

from _lcm.constraints.dispositions import (
    ConstraintContext,
    Proof,
    ProvedByConstruction,
)
from _lcm.constraints.ir import Compare, Const, Ref
from _lcm.constraints.processed import ProcessedConstraint
from _lcm.constraints.routes import (
    BoundConstraint,
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.grids import ContinuousGrid
from _lcm.solution.contract import ConstraintRouteContext, simulation_route
from _lcm.typing import EconFunctionsMapping, FunctionName


def case_piece_routes(
    *,
    context: ConstraintRouteContext,
    savings_grid: ContinuousGrid,
    post_decision_function: FunctionName | None,
    solver_path: tuple[str, ...],
    function_pool: EconFunctionsMapping | None = None,
) -> tuple[ConstraintRoute, ...]:
    """Declare the one route a case-piece kernel walks in one phase.

    One site, at the savings stage. The kernels evaluate no user constraint
    anywhere, so the only thing that can happen to one along this pipeline is
    the savings grid's own proof, and a proof belongs to a site rather than
    needing one of its own. Declaring the partition and envelope stages as
    further sites would describe the program rather than where a constraint can
    be met, and neither has a stage that names it.

    One route, though NBEGM dispatches several mutually exclusive kernels — case
    pieces, a piecewise-affine schedule, a discrete envelope, their composition,
    and the ride-along co-state kernels. They differ in the program they compile
    and in nothing a route carries: none evaluates a user constraint, all invert
    on the same savings grid, none rewrites its pool. Emitting one route each
    would put five entries in the plan where there is one fact.

    Args:
        context: What the solver may read about the regime and the phase.
        savings_grid: The grid the kernel inverts on, whose lowest node is the
            borrowing limit it enforces.
        post_decision_function: Name of the post-decision state that grid
            spans, or `None` before the solver has been bound to a margin —
            nothing can be proved against a grid whose state is not yet named,
            and the empty allow-list refuses every constraint either way.
        solver_path: The nest of solvers producing the candidates.
        function_pool: The pool in scope at the site, for a nesting solver that
            rewrites it going into this branch. Defaults to the phase's own
            pool, which is what a solver that rewrites nothing hands over.

    Returns:
        The route, as a one-tuple.

    """
    if context.phase == "simulate":
        return (simulation_route(context=context, solver_path=solver_path),)
    proof = _BorrowingLimitProof(
        savings_grid=savings_grid, post_decision_function=post_decision_function
    )
    return (
        ConstraintRoute(
            key=ConstraintRouteKey(
                phase="solve", period_group=None, solver_path=solver_path
            ),
            sites=(
                ConstraintSite(
                    stage="savings_stage",
                    function_pool=(
                        context.functions if function_pool is None else function_pool
                    ),
                    available_names=frozenset(),
                    structural_proofs=(proof,),
                ),
            ),
        ),
    )


@dataclass(frozen=True)
class _BorrowingLimitProof:
    """Discharges the lower bound a savings grid's lowest node already imposes."""

    savings_grid: ContinuousGrid
    """Grid the kernel inverts on."""

    post_decision_function: FunctionName | None
    """Post-decision state that grid spans, or `None` before margin binding."""

    def __call__(
        self,
        *,
        bound: BoundConstraint,
        context: ConstraintContext,  # noqa: ARG002
    ) -> ProvedByConstruction | None:
        """Discharge a lower bound the savings grid's lowest node already imposes.

        Takes `context` because the protocol passes it, and reads none of it:
        the grid and the state it spans are the solver's own configuration, so
        the verdict does not vary with the regime or the phase.

        Declines rather than refuses when the shape does not match, so a
        constraint this proof has nothing to say about falls through to the
        site's empty allow-list and is refused there instead.

        Args:
            bound: The constraint resolved against the site.
            context: Unread, as above.

        Returns:
            The discharge, or `None` to decline.

        """
        if self.post_decision_function is None:
            return None
        lower = _lower_bound(constraint=bound.constraint)
        if lower is None:
            return None
        if lower.name != self.post_decision_function:
            return None
        if not _matches_grid_start(grid=self.savings_grid, value=lower.value):
            return None
        return ProvedByConstruction(
            constraint=bound.constraint,
            proof=Proof(
                reason=(
                    f"The savings grid the kernel inverts on starts at "
                    f"{lower.value}, so it enforces this bound on "
                    f"'{self.post_decision_function}' at every node it "
                    f"publishes."
                ),
                surface=lower.surface,
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
