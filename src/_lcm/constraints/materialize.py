"""Turning a normalized constraint into a function the DAG can compose.

The DAG's annotation-consistency check requires every consumer of a value to
annotate it the same way its producer returns it, so a constraint cannot
annotate itself: `savings` is a float in one regime and an integer-coded state
in another, and only the surrounding functions know which. The annotation is
therefore read off the regime's own function pool, which is also the only way a
condition over a discrete state can compose at all.
"""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast, no_type_check

from dags import concatenate_functions, get_annotations
from dags.annotations import ensure_annotations_are_strings

from _lcm.constraints.processed import ProcessedConstraint
from _lcm.typing import ConstraintFunction, FunctionName
from _lcm.utils.functools import get_union_of_args
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
    if constraint.is_opaque:
        # An opaque constraint *is* a composable callable, carrying the
        # annotations its author wrote. Rebuilding it from the pool would
        # replace those with whatever the surrounding functions happen to say,
        # and the DAG requires every consumer of a name to annotate it as its
        # producer does — so a rewrite turns a composable predicate into an
        # annotation conflict. Only a condition built from references needs
        # annotations supplied, because it has none of its own.
        return cast("ConstraintFunction", constraint.declaration)

    return cast(
        "ConstraintFunction",
        _ConstraintEvaluator(
            constraint=constraint,
            argument_annotations={
                arg_name: annotation_of(pool=pool, name=arg_name)
                for arg_name in constraint.condition.arg_names
            },
        ),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _ConstraintEvaluator:
    """A condition evaluated through the DAG under the regime's own annotations."""

    constraint: ProcessedConstraint
    """The normalized constraint, evaluated on each call."""
    argument_annotations: dict[str, str]
    """Annotation of each dependency, in signature order."""

    def __post_init__(self) -> None:
        signature = inspect.Signature(
            [
                inspect.Parameter(
                    arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation
                )
                for arg, annotation in self.argument_annotations.items()
            ],
            return_annotation="BoolND",
        )
        object.__setattr__(self, "__signature__", signature)
        object.__setattr__(
            self, "__annotations__", {**self.argument_annotations, "return": "BoolND"}
        )
        object.__setattr__(self, "__name__", self.constraint.name)

    # The kernel is traced with whatever leaves its caller supplies -- tracers,
    # Python scalars, arrays of either integer width -- so its annotations
    # document the contract and are not enforced at call time.
    @no_type_check
    def __call__(self, **values: ValueND) -> BoolND:
        return self.constraint.evaluate(**values)


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


def transitive_arg_names(
    *,
    constraint: ProcessedConstraint,
    pool: Mapping[FunctionName, UserFunction],
) -> frozenset[str]:
    """Return the names a constraint needs once resolved through a pool.

    What a constraint requires is not what it spells. `spendable >= 0` names a
    helper; where that helper is in the pool the requirement is really the
    helper's own leaves, and where it is not the name itself is the leaf. Both
    spellings of one requirement therefore resolve alike, which is the point:
    classifying them differently would make an equivalent declaration solvable
    under one wording and refused under the other.

    Args:
        constraint: The normalized constraint.
        pool: The functions in scope where the constraint is evaluated. A name
            this pool produces is resolved through it rather than demanded.

    Returns:
        Frozenset of the leaf names that must be available.

    """
    composed = concatenate_functions(
        functions={
            **dict(pool),
            constraint.name: as_constraint_function(constraint=constraint, pool=pool),
        },
        targets=constraint.name,
        enforce_signature=False,
        set_annotations=True,
    )
    return frozenset(get_union_of_args([composed]))
