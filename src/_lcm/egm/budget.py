"""Intrinsic budget constraint of an endogenous-grid regime, for simulation.

An endogenous-grid regime's spec carries no borrowing constraint — the solve
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

from typing import TYPE_CHECKING, cast

from dags import get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.typing import (
    ConstraintFunction,
    EconFunctionsMapping,
    FunctionName,
    StateOrActionName,
)
from lcm.solvers import DCEGM, NBEGM, NEGM, NNBEGM

if TYPE_CHECKING:
    from _lcm.solution.dcegm import _BoundDCEGM
    from _lcm.solution.nbegm import _BoundNBEGM
    from _lcm.solution.negm import _BoundNEGM
    from _lcm.solution.nnbegm import _BoundNNBEGM
else:
    _BoundDCEGM = DCEGM
    _BoundNBEGM = NBEGM
    _BoundNEGM = NEGM
    _BoundNNBEGM = NNBEGM
from lcm.typing import BoolND, FloatND

# Solvers whose 1-D liquid step enforces the borrowing limit intrinsically.
INTRINSIC_BUDGET_SOLVERS = (DCEGM, NEGM, NBEGM, NNBEGM)

DCEGM_BUDGET_CONSTRAINT_NAME: FunctionName = "dcegm_budget_constraint"


def get_intrinsic_budget_constraint(
    *,
    solver: DCEGM | NEGM | NBEGM | NNBEGM,
    functions: EconFunctionsMapping,
) -> ConstraintFunction:
    """Build the budget-feasibility mask the EGM solve enforces intrinsically.

    The returned function reads the regime's continuous action and resources
    function from the DAG and marks an action feasible iff
    `continuous_action <= resources - borrowing_limit`, where the borrowing
    limit is the savings grid's lowest node. Which config names that margin
    depends on the solver:

    - `DCEGM` and `NBEGM` govern the liquid margin themselves
    - `NEGM` and `NNBEGM` nest the same 1-D consumption-savings solve, so the
      mask comes from `solver.inner`, with the outer durable margin already
      folded into the inner resources

    Args:
        solver: The regime's endogenous-grid solver configuration; for a
            nested solver, its inner config supplies the liquid margin.
        functions: The regime's processed functions, used to stamp argument
            annotations consistent with the rest of the DAG (the resources
            function's return annotation, and the continuous action's
            annotation as the other functions declare it).

    Returns:
        Constraint function over the continuous action and the resources
        function.

    """
    if isinstance(solver, NNBEGM):
        nested_inner = cast("_BoundNNBEGM", solver).inner
        borrowing_limit = float(nested_inner.savings_grid.to_jax()[0])
        action_name = nested_inner.continuous_action
        resources_name = nested_inner.budget_target
    elif isinstance(solver, NBEGM):
        case_piece = cast("_BoundNBEGM", solver)
        borrowing_limit = float(case_piece.savings_grid.to_jax()[0])
        action_name = case_piece.continuous_action
        resources_name = case_piece.budget_target
    else:
        inner = (
            cast("_BoundNEGM", solver).inner
            if isinstance(solver, NEGM)
            else cast("_BoundDCEGM", solver)
        )
        borrowing_limit = float(inner.savings_grid.to_jax()[0])
        action_name = inner.continuous_action
        resources_name = inner.resources

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
