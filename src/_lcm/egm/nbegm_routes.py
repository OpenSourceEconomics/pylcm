"""The routes the case-piece solvers walk, and what can happen along them.

The NBEGM kernels recover consumption by inverting the Euler equation at each
node of a savings grid: the action is produced first and the liquid state falls
out of the budget identity afterwards. There is no point in that step at which a
predicate over `(state, action)` is evaluable, and the candidates the kernels
publish are never masked by one. The declaration is therefore that no name is
readable where a constraint would be called, which sends every constraint to
`Reject` unless a proof claims it first.

On the solve route, one proof claims anything: the borrowing limit the savings
grid already enforces, which the whole endogenous-grid family shares. Whether
the declared number is the lowest grid node is a different question, asked once
when the model is built rather than again here — proving the bound and
checking the claim are separate jobs, and doing the second one twice is how two
answers to it come to disagree.
"""

from _lcm.constraints.bounds import proves_the_savings_grids_lower_bound
from _lcm.constraints.routes import (
    BoundaryCompiler,
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.solution.contract import ConstraintRouteContext, simulation_route
from _lcm.typing import EconFunctionsMapping, FunctionName


def case_piece_routes(
    *,
    context: ConstraintRouteContext,
    post_decision_function: FunctionName | None,
    solver_path: tuple[str, ...],
    function_pool: EconFunctionsMapping | None = None,
    boundary_compilers: tuple[BoundaryCompiler, ...] = (),
) -> tuple[ConstraintRoute, ...]:
    """Declare the one route a case-piece kernel walks in one phase.

    The solve route has one site, at the savings stage. The kernels evaluate no
    user constraint anywhere, so the only thing that can happen to one along
    that pipeline is the proof supplied by the savings grid, and a proof belongs
    to a site rather than needing one of its own. Simulation instead takes the
    shared whole-candidate route and evaluates the phase-resolved declaration.
    Declaring the partition and envelope stages as further sites would describe
    the program rather than where a constraint can be met, and neither has a
    stage that names it.

    One route, though NBEGM dispatches several mutually exclusive kernels — case
    pieces, a piecewise-affine schedule, a discrete envelope, their composition,
    and the ride-along co-state kernels. They differ in the program they compile
    and in nothing a route carries: none evaluates a user constraint, all invert
    on the same savings grid, none rewrites its pool. Emitting one route each
    would put five entries in the plan where there is one fact.

    Args:
        context: What the solver may read about the regime and the phase.
        post_decision_function: Name of the post-decision state the savings
            grid spans, or `None` before the solver has been bound to a margin —
            nothing can be proved against a grid whose state is not yet named,
            and the empty allow-list refuses every constraint either way.
        solver_path: The nest of solvers producing the candidates.
        function_pool: The pool in scope at the site, for a nesting solver that
            rewrites it going into this branch. Defaults to the phase's own
            pool, which is what a solver that rewrites nothing hands over.
        boundary_compilers: Boundary compilers offered at the solve savings
            stage. They are ignored on the shared simulation route.

    Returns:
        The route, as a one-tuple.

    """
    proofs = (
        ()
        if post_decision_function is None
        else (
            proves_the_savings_grids_lower_bound(post_decision=post_decision_function),
        )
    )
    if context.phase == "simulate":
        return (simulation_route(context=context, solver_path=solver_path),)
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
                    structural_proofs=proofs,
                    boundary_compilers=boundary_compilers,
                ),
            ),
        ),
    )
