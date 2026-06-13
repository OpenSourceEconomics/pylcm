"""Intrinsic budget constraint of a DC-EGM regime, for the simulate phase.

A DC-EGM regime's spec carries no borrowing constraint — the EGM solve
enforces `continuous_action <= resources - savings_grid lower bound`
intrinsically by inverting the Euler equation on the exogenous savings grid.
The forward simulation, however, recomputes the argmax over the gridded
action space, so the constraint must be made explicit there: without a mask,
consumption points above resources imply below-limit savings whose
continuation is edge-clamped to the lowest wealth node and can win the
argmax. The builder here synthesizes that mask as an ordinary constraint
function; the simulation-phase builder injects it into the regime's
constraint set, where it enters the feasibility array `F` exactly like a
user-declared constraint.
"""

from dags import get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.typing import (
    ConstraintFunction,
    EconFunctionsMapping,
    FunctionName,
    StateOrActionName,
)
from lcm.solvers import DCEGM
from lcm.typing import BoolND, FloatND

DCEGM_BUDGET_CONSTRAINT_NAME: FunctionName = "dcegm_budget_constraint"


def get_intrinsic_budget_constraint(
    *,
    solver: DCEGM,
    functions: EconFunctionsMapping,
) -> ConstraintFunction:
    """Build the budget-feasibility mask the EGM solve enforces intrinsically.

    The returned function reads the regime's continuous action and resources
    function from the DAG and marks an action feasible iff
    `continuous_action <= resources - borrowing_limit`, where the borrowing
    limit is the savings grid's lowest node.

    Args:
        solver: The regime's DC-EGM solver configuration.
        functions: The regime's processed functions, used to stamp argument
            annotations consistent with the rest of the DAG (the resources
            function's return annotation, and the continuous action's
            annotation as the other functions declare it).

    Returns:
        Constraint function over the continuous action and the resources
        function.

    """
    borrowing_limit = float(solver.savings_grid.to_jax()[0])
    action_name = solver.continuous_action
    resources_name = solver.resources

    @with_signature(
        args={
            action_name: _find_annotation_of_arg(
                functions=functions, arg_name=action_name
            ),
            resources_name: ensure_annotations_are_strings(
                get_annotations(functions[resources_name])
            )["return"],
        },
        return_annotation="BoolND",
        enforce=False,
    )
    def budget_constraint(**action_and_resources: FloatND) -> BoolND:
        return (
            action_and_resources[action_name]
            <= action_and_resources[resources_name] - borrowing_limit
        )

    return budget_constraint  # ty: ignore[invalid-return-type]


def _find_annotation_of_arg(
    *,
    functions: EconFunctionsMapping,
    arg_name: StateOrActionName,
) -> str:
    """Return the annotation the regime's functions use for one argument.

    The DAG's annotation-consistency check requires every consumer of a leaf
    to agree on its annotation, so the synthesized constraint copies it from
    the first regime function that declares the argument. Falls back to
    `"FloatND"` when no function annotates it.
    """
    for func in functions.values():
        annotations = ensure_annotations_are_strings(get_annotations(func))
        annotation = annotations.get(arg_name, "no_annotation_found")
        if annotation != "no_annotation_found":
            return annotation
    return "FloatND"
