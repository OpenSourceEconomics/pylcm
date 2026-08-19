"""The expression tree behind a declared condition.

A condition is stored as a small immutable tree rather than as a predicate,
because two consumers need two different things from it. Grid search needs a
callable; an endogenous-grid solver needs to know *what* the condition says, so
it can decide whether its own construction already enforces it. Both come from
this one tree, so the predicate a solver proves and the predicate simulation
evaluates cannot drift apart.

An arbitrary callable remains legal and is carried as `Opaque`. It evaluates
like anything else and exposes no structure, so a solver that needs structure
refuses it rather than accepting it and quietly ignoring what it says.

Expression nodes subclass `BoolExpr` and operands subclass `Operand` rather
than forming union aliases. The tree is recursive, and a recursive alias is not
resolvable at runtime by the type checking this package runs under; a base
class is, and it also makes "there is no other kind of node" a property of the
class hierarchy instead of a convention.
"""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from lcm.typing import BoolND, UserFunction, ValueND

type ComparisonOperator = Literal["<", "<=", ">", ">=", "==", "!="]

# What an operand resolves to: a model value, or a literal exactly as written.
type OperandValue = ValueND | float | int | bool


class Operand:
    """Base class for the two things a comparison can compare."""


class BoolExpr:
    """Base class for every node of a condition's expression tree."""


class Ref(Operand):
    """A named value the condition reads: a state, action, DAG output, or param.

    Comparison operators build conditions instead of returning booleans, which
    is what makes `assets < limit` a declaration rather than a computation.
    Equality therefore does not answer "are these the same reference" — compare
    `.name` for that.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        """The name this leaf resolves against when the condition is evaluated."""

    def __lt__(self, other: object) -> Condition:
        return _compare(left=self, op="<", right=other)

    def __le__(self, other: object) -> Condition:
        return _compare(left=self, op="<=", right=other)

    def __gt__(self, other: object) -> Condition:
        return _compare(left=self, op=">", right=other)

    def __ge__(self, other: object) -> Condition:
        return _compare(left=self, op=">=", right=other)

    def __eq__(self, other: object) -> Condition:  # ty: ignore[invalid-method-override]
        return _compare(left=self, op="==", right=other)

    def __ne__(self, other: object) -> Condition:  # ty: ignore[invalid-method-override]
        return _compare(left=self, op="!=", right=other)

    def __hash__(self) -> int:
        return hash(("lcm.ref", self.name))

    def __repr__(self) -> str:
        return f"ref({self.name!r})"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, eq=False)
class Const(Operand):
    """A literal operand."""

    value: float | int | bool
    """The declared number, kept exactly as written."""


@dataclass(frozen=True, eq=False)
class Compare(BoolExpr):
    """A comparison of two operands.

    The operator decides who owns the boundary point: `<` leaves equality
    outside the admitted region and `<=` brings it in. That is the whole of
    equality ownership, so no separate convention is needed.
    """

    left: Operand
    """Left operand."""
    op: ComparisonOperator
    """The comparison, which also fixes equality ownership."""
    right: Operand
    """Right operand."""


@dataclass(frozen=True, eq=False)
class And(BoolExpr):
    """Both sides must hold."""

    left: BoolExpr
    """Left operand."""
    right: BoolExpr
    """Right operand."""


@dataclass(frozen=True, eq=False)
class Or(BoolExpr):
    """At least one side must hold."""

    left: BoolExpr
    """Left operand."""
    right: BoolExpr
    """Right operand."""


@dataclass(frozen=True, eq=False)
class Not(BoolExpr):
    """The operand must not hold."""

    operand: BoolExpr
    """The negated expression."""


@dataclass(frozen=True, eq=False)
class Implies(BoolExpr):
    """Constrains only where the premise holds, and is true elsewhere."""

    premise: BoolExpr
    """The condition under which the consequent is required."""
    consequent: BoolExpr
    """What must hold wherever the premise does."""


@dataclass(frozen=True, eq=False)
class Opaque(BoolExpr):
    """An arbitrary predicate, carried without structure.

    Legal wherever a condition is legal, and evaluated unchanged. It offers a
    solver nothing to reason about, so a solver needing structure refuses it
    rather than accepting it and ignoring what it says.
    """

    func: UserFunction
    """The user's predicate, called with the names in its own signature."""


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


def dependencies_of(expression: BoolExpr) -> frozenset[str]:
    """Return every name the expression reads.

    Args:
        expression: The expression tree to inspect.

    Returns:
        Frozenset of names that must be available to evaluate it.

    """
    match expression:
        case Compare(left=left, right=right):
            return frozenset(
                operand.name for operand in (left, right) if isinstance(operand, Ref)
            )
        case And(left=left, right=right) | Or(left=left, right=right):
            return dependencies_of(left) | dependencies_of(right)
        case Not(operand=operand):
            return dependencies_of(operand)
        case Implies(premise=premise, consequent=consequent):
            return dependencies_of(premise) | dependencies_of(consequent)
        case Opaque(func=func):
            return frozenset(signature_names(func))
    raise TypeError(f"Not a condition expression: {expression!r}")


def evaluate_expression(
    *,
    expression: BoolExpr,
    values: Mapping[str, ValueND],
) -> BoolND:
    """Evaluate the expression against named values.

    Args:
        expression: The expression tree to evaluate.
        values: Mapping of name to value, covering every dependency.

    Returns:
        The elementwise truth of the expression.

    """
    match expression:
        case Compare(left=left, op=op, right=right):
            return _apply_comparison(
                left=_operand_value(operand=left, values=values),
                op=op,
                right=_operand_value(operand=right, values=values),
            )
        case And(left=left, right=right):
            return jnp.logical_and(
                evaluate_expression(expression=left, values=values),
                evaluate_expression(expression=right, values=values),
            )
        case Or(left=left, right=right):
            return jnp.logical_or(
                evaluate_expression(expression=left, values=values),
                evaluate_expression(expression=right, values=values),
            )
        case Not(operand=operand):
            return jnp.logical_not(
                evaluate_expression(expression=operand, values=values)
            )
        case Implies(premise=premise, consequent=consequent):
            return jnp.logical_or(
                jnp.logical_not(evaluate_expression(expression=premise, values=values)),
                evaluate_expression(expression=consequent, values=values),
            )
        case Opaque(func=func):
            names = signature_names(func)
            _fail_if_names_are_missing(names=names, values=values)
            return func(**{name: values[name] for name in names})
    raise TypeError(f"Not a condition expression: {expression!r}")


def describe(expression: BoolExpr) -> str:
    """Render the expression as the declaration a user would recognise.

    Args:
        expression: The expression tree to render.

    Returns:
        A single-line rendering, e.g. `"savings >= 0.0"`.

    """
    match expression:
        case Compare(left=left, op=op, right=right):
            return f"{_describe_operand(left)} {op} {_describe_operand(right)}"
        case And(left=left, right=right):
            return f"({describe(left)}) and ({describe(right)})"
        case Or(left=left, right=right):
            return f"({describe(left)}) or ({describe(right)})"
        case Not(operand=operand):
            return f"not ({describe(operand)})"
        case Implies(premise=premise, consequent=consequent):
            return f"({describe(premise)}) implies ({describe(consequent)})"
        case Opaque(func=func):
            name = getattr(func, "__name__", repr(func))
            return f"{name}(...)"
    raise TypeError(f"Not a condition expression: {expression!r}")


def _describe_operand(operand: Operand) -> str:
    if isinstance(operand, Ref):
        return operand.name
    if isinstance(operand, Const):
        return repr(operand.value)
    raise TypeError(f"Not an operand: {operand!r}")


def signature_names(func: UserFunction) -> tuple[str, ...]:
    """Return the parameter names a predicate must be called with.

    Args:
        func: Any user-supplied callable.

    Returns:
        Tuple of its parameter names, in declaration order.

    """
    return tuple(inspect.signature(func).parameters)


def _compare(*, left: Ref, op: ComparisonOperator, right: object) -> Condition:
    """Build a comparison as a condition, keeping the declared name on the left."""
    if isinstance(right, Ref):
        return Condition(expression=Compare(left=left, op=op, right=right))
    if isinstance(right, bool | int | float):
        return Condition(expression=Compare(left=left, op=op, right=Const(right)))
    raise TypeError(
        f"Cannot compare {left!r} with {right!r}: a condition compares a "
        f"reference with another reference or with a number."
    )


def _apply_comparison(
    *,
    left: OperandValue,
    op: ComparisonOperator,
    right: OperandValue,
) -> BoolND:
    match op:
        case "<":
            result = left < right
        case "<=":
            result = left <= right
        case ">":
            result = left > right
        case ">=":
            result = left >= right
        case "==":
            result = left == right
        case "!=":
            result = left != right
        case _:
            raise ValueError(f"Not a comparison operator: {op!r}")
    # Comparing two literals yields a Python bool; every consumer wants an array.
    return jnp.asarray(result)


def _operand_value(
    *,
    operand: Operand,
    values: Mapping[str, ValueND],
) -> OperandValue:
    if isinstance(operand, Const):
        return operand.value
    if isinstance(operand, Ref):
        _fail_if_names_are_missing(names=(operand.name,), values=values)
        return values[operand.name]
    raise TypeError(f"Not an operand: {operand!r}")


def _fail_if_names_are_missing(
    *,
    names: tuple[str, ...],
    values: Mapping[str, ValueND],
) -> None:
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(
            f"Cannot evaluate the condition: no value supplied for "
            f"{', '.join(repr(name) for name in missing)}."
        )
