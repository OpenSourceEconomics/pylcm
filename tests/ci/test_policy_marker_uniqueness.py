"""No test declares an execution-policy marker twice across its scopes.

`pytest_policy.apply` refuses such a test with a `UsageError`. Under xdist only the
workers collect, so the refusal ends every worker's session silently and the
controller aborts on an unrelated crash item. This scan keeps the defect visible.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

from tests.ci.pytest_policy import _POLICY_MARKERS

_TESTS = Path(__file__).resolve().parents[1]


def _policy_marker_names(node: ast.expr | None) -> list[str]:
    """Return the policy marker names an expression applies, with repeats."""
    if node is None:
        return []
    if isinstance(node, ast.List | ast.Tuple):
        return [name for element in node.elts for name in _policy_marker_names(element)]
    target = node.func if isinstance(node, ast.Call) else node
    if (
        isinstance(target, ast.Attribute)
        and target.attr in _POLICY_MARKERS
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
    ):
        return [target.attr]
    return []


def _pytestmark(body: list[ast.stmt]) -> list[str]:
    return [
        name
        for statement in body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in statement.targets
        )
        for name in _policy_marker_names(statement.value)
    ]


def _decorators(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    return [name for d in node.decorator_list for name in _policy_marker_names(d)]


def _repeats(*, body: list[ast.stmt], inherited: list[str]) -> Iterator[str]:
    for node in body:
        if isinstance(node, ast.ClassDef):
            scope = inherited + _decorators(node) + _pytestmark(node.body)
            yield from (
                f"{node.name}.{r}" for r in _repeats(body=node.body, inherited=scope)
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names = inherited + _decorators(node)
            if node.name.startswith("test") and len(names) != len(set(names)):
                yield node.name


def test_no_test_declares_a_policy_marker_twice() -> None:
    """Module, class and function scopes together carry each policy marker once."""
    offenders = [
        f"{path.relative_to(_TESTS.parent).as_posix()}::{name}"
        for path in sorted(_TESTS.rglob("*.py"))
        for module in [ast.parse(path.read_text(encoding="utf-8"))]
        for name in _repeats(body=module.body, inherited=_pytestmark(module.body))
    ]
    assert offenders == []
