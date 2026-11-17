"""Declaring a condition once, for every solver to read.

A condition states a fact about the model — `savings >= 0`, `assets < limit` —
without saying what should happen at that boundary. What happens is decided by
whoever consumes it: a constraint makes the region infeasible, and a case
selects which formula applies on each side. The same declaration therefore
serves both, and cannot mean two different things in the two places.

Build one from `ref`:

```python
savings = lcm.ref("savings")
borrowing = savings >= 0.0
```

Combine with `&`, `|`, `~`, and `implies`. The comparison operator settles who
owns the boundary point: `<` leaves it outside the region, `<=` brings it in.

An arbitrary predicate is still legal — `Condition.from_callable` carries it
unchanged — but it exposes no structure, so a solver that must reason about a
condition refuses it instead of accepting it and ignoring what it says.

Both sides of a comparison are names or numbers. A limit that is itself
computed — one that varies with age, or comes out of a table — is written as a
reference to whatever computes it, `assets < ref("borrowing_limit")`, and the
condition stays readable to a solver even though the number is not in it.
"""

import inspect
from dataclasses import dataclass
from typing import Self

from _lcm.constraints.ir import (
    And,
    BoolExpr,
    Compare,
    ComparisonOperator,
    Const,
    Implies,
    Not,
    Opaque,
    Or,
    _Ref,
    annotations_of_signature,
    arg_names_of,
    dependencies_of,
    describe,
    evaluate_expression,
)
from lcm.typing import BoolND, FloatND, ReferenceName, UserFunction, ValueND

__all__ = ["Condition", "implies", "ref"]


class Ref(_Ref):
    """A named value the condition reads: a state, action, DAG output, or param.

    Comparison operators build conditions instead of returning booleans, which
    is what makes `assets < limit` a declaration rather than a computation.
    Equality therefore does not answer "are these the same reference" — compare
    `.name` for that. A number on the other side is stored as a float, and a flag
    as a bool.
    """

    def __lt__(self, other: Self | bool | float) -> Condition:
        return _compare(left=self, op="<", right=other)

    def __le__(self, other: Self | bool | float) -> Condition:
        return _compare(left=self, op="<=", right=other)

    def __gt__(self, other: Self | bool | float) -> Condition:
        return _compare(left=self, op=">", right=other)

    def __ge__(self, other: Self | bool | float) -> Condition:
        return _compare(left=self, op=">=", right=other)

    def __eq__(self, other: Self | bool | float) -> Condition:  # ty: ignore[invalid-method-override]
        return _compare(left=self, op="==", right=other)

    def __ne__(self, other: Self | bool | float) -> Condition:  # ty: ignore[invalid-method-override]
        return _compare(left=self, op="!=", right=other)

    def __hash__(self) -> int:
        return hash(("lcm.ref", self.name))

    def __repr__(self) -> str:
        return f"ref({self.name!r})"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, eq=False)
class Condition:
    """A declared fact about the model, readable by every solver.

    Carries the expression rather than a predicate, so a solver can inspect
    what the condition says instead of only being able to call it. The
    evaluator is generated from the same expression, which is what keeps the
    predicate a solver proves and the predicate simulation evaluates identical.

    Instances compare by identity. `==` on a reference builds a condition
    rather than answering a question, so field-by-field equality on an
    expression would return a truthy object for any two nodes and report
    unrelated declarations as the same one. Compare renderings or dependencies
    when structural sameness is what is wanted.
    """

    expression: BoolExpr
    """The expression tree this condition declares."""

    def __post_init__(self) -> None:
        # A condition has to pass for an ordinary constraint callable, because
        # every existing consumer -- the DAG builder, the params template, the
        # feasibility mask -- reaches a constraint through its signature.
        # Stamping one here is what lets a declared condition reach them all
        # with no rewiring.
        if isinstance(self.expression, Opaque):
            signature = inspect.signature(self.expression.func)
            name = getattr(self.expression.func, "__name__", "condition")
        else:
            parameters = [
                inspect.Parameter(
                    arg_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=FloatND,
                )
                for arg_name in arg_names_of(self.expression)
            ]
            signature = inspect.Signature(parameters, return_annotation=BoolND)
            name = "condition"
        object.__setattr__(self, "__signature__", signature)
        object.__setattr__(self, "__name__", name)
        object.__setattr__(self, "__annotations__", annotations_of_signature(signature))

    @property
    def arg_names(self) -> tuple[str, ...]:
        """Tuple of the argument names the condition is called with."""
        return arg_names_of(self.expression)

    def __call__(self, **values: ValueND) -> BoolND:
        return self.evaluate(**values)

    @classmethod
    def from_callable(cls, func: UserFunction) -> Condition:
        """Carry an arbitrary predicate as a condition without structure.

        Args:
            func: A predicate over model values.

        Returns:
            A condition that evaluates exactly as `func` does.

        """
        return cls(expression=Opaque(func=func))

    @property
    def dependencies(self) -> frozenset[str]:
        """Frozenset of names that must be available to evaluate the condition."""
        return dependencies_of(self.expression)

    @property
    def is_opaque(self) -> bool:
        """Whether the condition offers a solver no structure to reason about."""
        return isinstance(self.expression, Opaque)

    def evaluate(self, **values: ValueND) -> BoolND:
        """Evaluate the condition on the supplied values.

        Args:
            **values: One entry per dependency.

        Returns:
            The elementwise truth of the condition.

        """
        return evaluate_expression(expression=self.expression, values=values)

    def __and__(self, other: Condition) -> Condition:
        return Condition(expression=And(left=self.expression, right=other.expression))

    def __or__(self, other: Condition) -> Condition:
        return Condition(expression=Or(left=self.expression, right=other.expression))

    def __invert__(self) -> Condition:
        return Condition(expression=Not(operand=self.expression))

    def __str__(self) -> str:
        return describe(self.expression)


def ref(name: ReferenceName) -> Ref:
    """Refer to a named value by name, for use in a condition.

    Args:
        name: A state, action, DAG output, parameter, or declared margin role.

    Returns:
        A reference whose comparison operators build conditions.

    """
    return Ref(name)


def implies(*, premise: Condition, consequent: Condition) -> Condition:
    """Require the consequent wherever the premise holds.

    True wherever the premise fails, so it constrains only the region the
    premise selects.

    Args:
        premise: The condition selecting where the requirement applies.
        consequent: What must hold there.

    Returns:
        The implication as a single condition.

    """
    return Condition(
        expression=Implies(premise=premise.expression, consequent=consequent.expression)
    )


def _compare(
    *, left: Ref, op: ComparisonOperator, right: Ref | bool | float
) -> Condition:
    """Build a comparison as a condition, keeping the declared name on the left.

    A literal is stored as the engine reads it: a flag stays a `bool`, and any
    other number becomes a `float`, so `ref("x") <= 1` and `ref("x") <= 1.0`
    declare the same condition.
    """
    if isinstance(right, Ref):
        return Condition(expression=Compare(left=left, op=op, right=right))
    if isinstance(right, bool):
        return Condition(expression=Compare(left=left, op=op, right=Const(right)))
    if isinstance(right, int | float):
        return Condition(
            expression=Compare(left=left, op=op, right=Const(float(right)))
        )
    raise TypeError(
        f"Cannot compare {left!r} with {right!r}: a condition compares a "
        f"reference with another reference or with a number."
    )
