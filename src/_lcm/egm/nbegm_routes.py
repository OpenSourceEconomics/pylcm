"""NB inversion routes with canonical optional current-node evaluation.

The savings stage proves the borrowing limit or compiles supported one-sided
boundaries. A separately qualified current-state/discrete predicate may be
evaluated at published nodes. Continuous actions are unavailable there.
Simulation uses the shared complete-candidate route.
"""

from _lcm.constraints.bounds import proves_the_savings_grids_lower_bound
from _lcm.constraints.routes import (
    BoundaryCompiler,
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.solution.contract import ConstraintRouteContext, simulation_route
from _lcm.typing import EconFunctionsMapping, FunctionName
from lcm.typing import ReferenceName


def case_piece_routes(
    *,
    context: ConstraintRouteContext,
    post_decision_function: FunctionName | None,
    bound_spec: PostDecisionBoundSpec | None = None,
    solver_path: tuple[str, ...],
    function_pool: EconFunctionsMapping | None = None,
    boundary_compilers: tuple[BoundaryCompiler, ...] = (),
    evaluated_predicate_inputs: frozenset[ReferenceName] = frozenset(),
) -> tuple[ConstraintRoute, ...]:
    """Declare the one route a case-piece kernel walks in one phase.

    The savings stage proves or compiles each supported declaration. Canonical
    evaluated predicates add a state-node site with an explicit leaf allow-list.
    Other declarations retain the original compiler verdict; simulation uses
    the shared whole-candidate route.

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
        evaluated_predicate_inputs: Canonically qualified current-node leaves,
            or empty when the solver evaluates no current-node predicate.

    Returns:
        The route, as a one-tuple.

    """
    proofs = (
        ()
        if post_decision_function is None
        else (
            proves_the_savings_grids_lower_bound(
                post_decision=post_decision_function, bound_spec=bound_spec
            ),
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
                *(
                    (
                        ConstraintSite(
                            stage="state_node",
                            function_pool=(
                                context.functions
                                if function_pool is None
                                else function_pool
                            ),
                            available_names=evaluated_predicate_inputs,
                        ),
                    )
                    if evaluated_predicate_inputs
                    else ()
                ),
            ),
        ),
    )
