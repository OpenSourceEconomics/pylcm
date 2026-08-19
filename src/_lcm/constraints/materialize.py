"""Turning a normalized constraint into a function the DAG can compose.

The DAG's annotation-consistency check requires every consumer of a value to
annotate it the same way its producer returns it, so a constraint cannot
annotate itself: `savings` is a float in one regime and an integer-coded state
in another, and only the surrounding functions know which. The annotation is
therefore read off the regime's own function pool, which is also the only way a
condition over a discrete state can compose at all.
"""

from collections.abc import Mapping

from dags import get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.constraints.processed import ProcessedConstraint
from _lcm.typing import ConstraintFunction, FunctionName
from lcm.typing import BoolND, UserFunction, ValueND

# Used when no function in the pool says anything about a name.
FALLBACK_ANNOTATION = "FloatND"


def as_constraint_function(
    *,
    constraint: ProcessedConstraint,
    pool: Mapping[FunctionName, UserFunction],
) -> ConstraintFunction:
    """Build the DAG-composable function a constraint is evaluated through.

    Args:
        constraint: The normalized constraint.
        pool: The regime's functions, whose annotations the result adopts.

    Returns:
        A callable whose signature names the constraint's dependencies and
        annotates each as the regime's own functions do.

    """
    arg_names = constraint.condition.arg_names

    @with_signature(
        args={
            arg_name: annotation_of(pool=pool, name=arg_name) for arg_name in arg_names
        },
        return_annotation="BoolND",
        enforce=False,
    )
    def evaluate_constraint(**values: ValueND) -> BoolND:
        return constraint.evaluate(**values)

    evaluate_constraint.__name__ = constraint.name
    return evaluate_constraint  # ty: ignore[invalid-return-type]


def annotation_of(*, pool: Mapping[FunctionName, UserFunction], name: str) -> str:
    """Return the annotation the regime's functions use for one name.

    A name produced by a function in the pool takes that function's return
    annotation; otherwise the first function that annotates it as a parameter
    decides. Falls back to a continuous value when the pool says nothing,
    which is what a name supplied at runtime rather than computed looks like.

    Args:
        pool: The regime's functions, keyed by the name each produces.
        name: The name whose annotation is wanted.

    Returns:
        The annotation, as a string.

    """
    producer = pool.get(name)
    if producer is not None:
        annotations = ensure_annotations_are_strings(get_annotations(producer))
        produced = annotations.get("return", "no_annotation_found")
        if produced != "no_annotation_found":
            return produced
    for func in pool.values():
        annotations = ensure_annotations_are_strings(get_annotations(func))
        annotation = annotations.get(name, "no_annotation_found")
        if annotation != "no_annotation_found":
            return annotation
    return FALLBACK_ANNOTATION
