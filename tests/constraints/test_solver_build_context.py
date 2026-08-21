"""Constraint routing reaches the solver builder as one complete contract."""

from dataclasses import dataclass, field

from _lcm.constraints.dispositions import (
    BoundaryProgram,
    CompileBoundary,
    ConstraintContext,
)
from _lcm.constraints.routes import (
    BoundConstraint,
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.solution.contract import (
    ConstraintRouteContext,
    SolutionKernels,
    SolverBuildContext,
)
from lcm.solvers import GridSearch
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid_from_savings,
    savings,
    utility,
)


def _resources(liquid: ContinuousState) -> FloatND:
    """Use liquid wealth directly as resources."""
    return liquid


@dataclass(frozen=True, kw_only=True)
class _BoundaryRecordingGridSearch(GridSearch):
    """Compile solve constraints and record the context handed to the builder."""

    contexts: list[SolverBuildContext] = field(default_factory=list)

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Compile every solve constraint and evaluate every simulation constraint."""
        if context.phase == "simulate":
            return super().build_constraint_routes(context=context)

        def compile_boundary(
            *,
            bound: BoundConstraint,
            context: ConstraintContext,  # noqa: ARG001
        ) -> CompileBoundary:
            return CompileBoundary(
                constraint=bound.constraint,
                program=BoundaryProgram(surfaces=(), payload="compiled"),
            )

        return (
            ConstraintRoute(
                key=ConstraintRouteKey(
                    phase="solve",
                    period_group=None,
                    solver_path=("recording",),
                ),
                sites=(
                    ConstraintSite(
                        stage="state_action",
                        function_pool=context.functions,
                        available_names=frozenset(),
                        boundary_compilers=(compile_boundary,),
                    ),
                ),
            ),
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Record the complete build context and delegate numerical construction."""
        self.contexts.append(context)
        return super().build_period_kernels(context=context)


def test_compiled_constraint_reaches_the_solver_builder() -> None:
    """A compiled constraint keeps its callable, declaration, and route verdict."""
    solver = _BoundaryRecordingGridSearch()

    make_alive_dead_model(
        n_periods=2,
        n_liquid=4,
        liquid_max=4.0,
        n_consumption=4,
        alive_functions={
            "utility": utility,
            "resources": _resources,
            "savings": savings,
        },
        liquid_law=next_liquid_from_savings,
        alive_solver=solver,
        constraints={"feasible": feasible},
    )

    context = solver.contexts[0]
    assert context.constraint_plan is not None
    assert (
        tuple(context.constraints),
        tuple(context.constraint_functions),
        tuple(context.processed_constraints),
        isinstance(context.constraint_plan.entries[0].disposition, CompileBoundary),
    ) == ((), ("feasible",), ("feasible",), True)
