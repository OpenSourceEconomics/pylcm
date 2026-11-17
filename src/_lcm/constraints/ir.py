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

An operand is a name or a literal, and nothing else. A bound that has to be
looked up — one that varies by age, or is read out of a table — is named by a
`Ref` to whatever computes it, so the tree stays a statement about names
rather than growing an indexing sublanguage of its own. The cost is that a
solver cannot read such a bound's number off the tree: it has a surface and
the name of a value, and turning that pair into something it can prove against
its own grid is the boundary compiler's job, not the tree's.
"""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from _lcm.typing import AnnotationForm
from _lcm.utils.functools import array_result
from lcm.typing import BoolND, ReferenceName, UserFunction, ValueND

type ComparisonOperator = Literal["<", "<=", ">", ">=", "==", "!="]

# What an operand resolves to: a model value, or a literal as stored.
type OperandValue = ValueND | float | bool


class Operand:
    """Base class for the two things a comparison can compare."""


class BoolExpr:
    """Base class for every node of a condition's expression tree."""


class _Ref(Operand):
    """A named value the condition reads; `lcm.condition.Ref` adds the operators.

    The expression tree only needs the name, so the walks below recognise a
    reference by this base and the public class keeps the comparison operators that
    build conditions.
    """

    def __init__(self, name: ReferenceName) -> None:
        self.name = name
        """The name this leaf resolves against when the condition is evaluated."""


@dataclass(frozen=True, eq=False)
class Const(Operand):
    """A literal operand."""

    value: float | bool
    """The declared number as a float, or a flag; `lcm.condition.Ref` converts ints."""


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

    @property
    def admits_equality(self) -> bool:
        """Whether the boundary point itself lies inside the admitted region."""
        return self.op in {"<=", ">=", "=="}


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
                operand.name for operand in (left, right) if isinstance(operand, _Ref)
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


def conjuncts(expression: BoolExpr) -> tuple[Compare, ...] | None:
    """Split an expression into the comparisons that bound the admitted region.

    A solver can only split a candidate grid on a set of surfaces when every
    comparison must hold, which is what a conjunction says. Under an `or` the
    admitted region is a union of regions, and under a negation each
    comparison's admitted side flips, so neither yields surfaces a grid can be
    split on.

    Args:
        expression: The expression tree to decompose.

    Returns:
        Tuple of comparisons, in the order they were declared, or `None` when
        the expression is not a conjunction of comparisons. `None` says the
        expression does not decompose and must never be read as "there are no
        boundaries" — an expression always has at least one node, so an empty
        tuple cannot occur.

    """
    match expression:
        case Compare():
            return (expression,)
        case And(left=left, right=right):
            left_parts = conjuncts(left)
            right_parts = conjuncts(right)
            if left_parts is None or right_parts is None:
                return None
            return left_parts + right_parts
        case Or() | Not() | Implies() | Opaque():
            return None
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
            return array_result(func(**{name: values[name] for name in names}))
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
    if isinstance(operand, _Ref):
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
    if isinstance(operand, _Ref):
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


def annotations_of_signature(
    signature: inspect.Signature,
) -> dict[str, AnnotationForm]:
    """Return a signature's annotations as the plain dict a callable carries."""
    # It becomes the callable's `__annotations__`, which `annotationlib` reads only
    # as a plain dict.
    annotations: dict[str, AnnotationForm] = {
        arg_name: parameter.annotation
        for arg_name, parameter in signature.parameters.items()
        if parameter.annotation is not inspect.Parameter.empty
    }
    if signature.return_annotation is not inspect.Signature.empty:
        annotations["return"] = signature.return_annotation
    return annotations


def arg_names_of(expression: BoolExpr) -> tuple[str, ...]:
    """Return the names a condition is called with, in a stable order."""
    if isinstance(expression, Opaque):
        return signature_names(expression.func)
    return tuple(sorted(dependencies_of(expression)))
