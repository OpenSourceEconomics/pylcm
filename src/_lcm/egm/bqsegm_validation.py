"""Smoothness gate for the user-authored economic nodes of a BQSEGM case.

BQSEGM lowers each case to a smooth DAG and runs EGM on it, which is only valid if
every user formula in the case is genuinely smooth. Two complementary static
checks enforce that on the *user-authored* sub-DAG only:

- AST: reject Python branching (`if`, conditional expression, `match`) and, in a
  smooth piece, bare comparisons / boolean logic / piecewise calls that hide an
  undeclared case boundary.
- JAXPR: trace the function and reject piecewise JAX primitives (`select_n`,
  `cond`, `lt`, `max`, `searchsorted`, …), including those nested in a `cond` or
  `scan` helper that the AST cannot see.

Both gates run only on nodes reachable from the case's economic targets and not
marked trusted: continuation interpolation, grid location, and EGM-kernel
internals legitimately use these primitives and are excluded by scoping. A
reviewed numerical helper opts out with `lcm.smooth_helper`.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Literal

import jax
from dags import get_ancestors

from _lcm.typing import FunctionName

type CheckMode = Literal["smooth_user", "boundary"]

_PIECEWISE_CALL_NAMES = frozenset(
    {
        "where",
        "select",
        "cond",
        "switch",
        "clip",
        "maximum",
        "minimum",
        "max",
        "min",
        "abs",
        "sign",
        "heaviside",
        "searchsorted",
        "digitize",
        "interp",
        "piecewise",
    }
)

_SMOOTH_FORBIDDEN_PRIMS = frozenset(
    {
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
        "ne",
        "select_n",
        "cond",
        "switch",
        "max",
        "min",
        "clamp",
        "abs",
        "sign",
        "sort",
        "searchsorted",
    }
)


def find_ast_violations(func: Callable[..., object], *, mode: CheckMode) -> list[str]:
    """Find AST-level smoothness violations in a user function's source.

    Args:
        func: The user-authored economic function to inspect.
        mode: `"smooth_user"` for a smooth piece (bans comparisons / boolean
            logic / piecewise calls); `"boundary"` for a case-boundary predicate
            (permits comparisons, still bans Python branching).

    Returns:
        List of human-readable violation messages; empty when the source is
        smooth. A function whose source cannot be read yields a single
        loud-failure message rather than silently passing.

    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as exc:
        name = getattr(func, "__name__", "<unknown>")
        return [f"source unavailable for AST validation of {name!r}: {exc}"]
    tree = ast.parse(source)
    checker = _PiecewiseASTChecker(mode=mode)
    checker.visit(tree)
    return checker.violations


def find_jaxpr_violations(
    func: Callable[..., object],
    *,
    abstract_args: Iterable[object],
    mode: CheckMode,
) -> list[str]:
    """Find piecewise JAX primitives in a user function's traced jaxpr.

    Args:
        func: The user-authored economic function to trace.
        abstract_args: Positional sample arguments to trace `func` against.
        mode: `"smooth_user"` bans piecewise primitives; `"boundary"` permits
            them (a predicate is meant to compare).

    Returns:
        List of human-readable violation messages; empty when the jaxpr uses no
        forbidden primitive.

    """
    if mode != "smooth_user":
        return []
    jaxpr = jax.make_jaxpr(func)(*abstract_args)
    name = getattr(func, "__name__", "<unknown>")
    violations: list[str] = []
    for eqn in _iter_jaxpr_eqns(jaxpr):
        prim = eqn.primitive.name  # ty: ignore[unresolved-attribute]
        if prim in _SMOOTH_FORBIDDEN_PRIMS:
            violations.append(
                f"JAX primitive `{prim}` in {name!r} indicates hidden piecewise "
                f"logic inside a smooth BQSEGM formula. Expose it as a case "
                f"boundary or mark a reviewed helper with `lcm.smooth_helper`."
            )
    return violations


def user_economic_nodes(
    functions: Mapping[FunctionName, Callable[..., object]],
    *,
    targets: Iterable[FunctionName],
    trusted_names: Iterable[FunctionName],
) -> set[FunctionName]:
    """Find the user-authored producers to validate for one case variant.

    The validated set is every producer reachable from the case's economic
    targets, minus the trusted solver nodes (continuation interpolation, grid
    location, EGM-kernel internals). Leaf inputs without a producer are not
    returned — only nodes the user actually authored a formula for.

    Args:
        functions: Mapping of name to the case variant's DAG producers.
        targets: Economic target names whose ancestors define the scope.
        trusted_names: Names of solver-provided nodes to exclude from the gate.

    Returns:
        Set of producer names to run the smoothness gate on.

    """
    trusted = set(trusted_names)
    ancestors = get_ancestors(functions, targets=list(targets), include_targets=True)
    return {name for name in ancestors if name in functions and name not in trusted}


def is_smooth_helper(func: Callable[..., object]) -> bool:
    """Return whether a node is an `lcm.smooth_helper`-attested numerical helper."""
    return getattr(func, "__lcm_smooth_helper__", False) is True


class _PiecewiseASTChecker(ast.NodeVisitor):
    """Collect Python-level branching and undeclared-boundary AST violations."""

    def __init__(self, *, mode: CheckMode) -> None:
        self.mode = mode
        self.violations: list[str] = []

    def visit_If(self, node: ast.If) -> None:
        self.violations.append("Python `if` is not allowed in a BQSEGM piece.")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.violations.append("Conditional expression is not allowed in a piece.")
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.violations.append("`match`/`case` is not allowed in a BQSEGM piece.")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if self.mode == "smooth_user":
            self.violations.append(
                "Comparison in a smooth formula creates a hidden case boundary."
            )
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if self.mode == "smooth_user":
            self.violations.append(
                "Boolean logic in a smooth formula creates a hidden case boundary."
            )
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if self.mode == "smooth_user" and isinstance(node.op, ast.Not):
            self.violations.append(
                "`not` in a smooth formula creates a hidden case boundary."
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        leaf = name.split(".")[-1] if name else None
        if self.mode == "smooth_user" and leaf in _PIECEWISE_CALL_NAMES:
            self.violations.append(
                f"Call to piecewise function `{name}` is not allowed in a smooth "
                f"BQSEGM piece."
            )
        self.generic_visit(node)


def _call_name(node: ast.expr) -> str | None:
    """Return the dotted name of a call target, or `None` if it is not a name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _iter_jaxpr_eqns(jaxpr_like: object) -> Iterator[object]:
    """Yield every equation of a closed jaxpr and any jaxprs nested in its params."""
    jaxpr = getattr(jaxpr_like, "jaxpr", jaxpr_like)
    for eqn in jaxpr.eqns:  # ty: ignore[unresolved-attribute]
        yield eqn
        for value in eqn.params.values():
            if hasattr(value, "jaxpr"):
                yield from _iter_jaxpr_eqns(value)
            elif isinstance(value, tuple | list):
                for item in value:
                    if hasattr(item, "jaxpr"):
                        yield from _iter_jaxpr_eqns(item)
