"""Constructor keywords shown in the book exist on the classes they name.

The API index guards exported *names*, so a page can keep spelling a keyword
argument that a dataclass no longer has: the name still resolves, the fence
still renders, and a reader copying it gets `TypeError`. This module compares
every `ClassName(kwarg=...)` written in a documentation fence against the public
constructor signature (using dataclass fields where available).
"""

import ast
import dataclasses
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

import lcm
import lcm.consumption_savings_regime
import lcm.outer_search
import lcm.solvers

_DOCS = Path(__file__).parents[1] / "docs"

_FENCE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
_QUALIFIED_PUBLIC_MODULES = {
    "lcm",
    "lcm.consumption_savings_regime",
    "lcm.outer_search",
    "lcm.solvers",
}


def _public_constructors() -> dict[str, Any]:
    """Map each documented public class name to the class object."""
    modules = (lcm, lcm.solvers, lcm.outer_search, lcm.consumption_savings_regime)
    found: dict[str, Any] = {}
    for module in modules:
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if isinstance(obj, type):
                found.setdefault(name, obj)
    return found


def _accepted_keywords(cls: type) -> set[str] | None:
    """Return explicit constructor keywords, or `None` for open `**kwargs`."""
    if dataclasses.is_dataclass(cls):
        return {field.name for field in dataclasses.fields(cls)}
    parameters = inspect.signature(cls).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return None
    return {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def _keywords_in_block(*, block: str, names: set[str]) -> list[tuple[str, str]]:
    """List `(class name, keyword)` for each named class constructed in `block`."""
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return []
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted_name = _dotted_name(node.func)
        if dotted_name is None:
            continue
        prefix, separator, class_name = dotted_name.rpartition(".")
        if not separator:
            class_name = dotted_name
        elif prefix not in _QUALIFIED_PUBLIC_MODULES:
            continue
        if class_name not in names:
            continue
        calls.extend(
            (class_name, keyword.arg)
            for keyword in node.keywords
            if keyword.arg is not None
        )
    return calls


def _dotted_name(node: ast.expr) -> str | None:
    """Return a dotted spelling for a simple name/attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _documentation_pages() -> list[Path]:
    """List the tracked documentation pages, excluding build output."""
    return [p for p in sorted(_DOCS.rglob("*.md")) if "_build" not in p.parts]


def _documented_calls() -> list[tuple[Path, str, str]]:
    """List `(page, class name, keyword)` for every keyword written in a fence."""
    names = set(_public_constructors())
    return [
        (page, cls_name, kwarg)
        for page in _documentation_pages()
        for block in _FENCE.findall(page.read_text(encoding="utf-8"))
        for cls_name, kwarg in _keywords_in_block(block=block, names=names)
    ]


def test_documented_constructor_keywords_exist_on_their_class():
    """No documentation fence passes a keyword its class does not define."""
    classes = _public_constructors()
    orphans = [
        f"{page.relative_to(_DOCS)}: {cls_name}({kwarg}=...)"
        for page, cls_name, kwarg in _documented_calls()
        if (accepted := _accepted_keywords(classes[cls_name])) is not None
        if kwarg not in accepted
    ]

    assert orphans == []


def test_the_sweep_reaches_the_documentation():
    """The sweep is non-empty, so an empty orphan list means agreement."""
    calls = _documented_calls()

    assert len(calls) > 50


def test_the_sweep_collects_qualified_constructor_calls():
    """Qualified public constructors are covered alongside bare imports."""
    names = set(_public_constructors())

    assert _keywords_in_block(
        block="""
lcm.Model(bogus=1)
lcm.AgeGrid(bogus=1)
lcm.LiquidMargin(bogus=1)
lcm.OuterContinuousMargin(bogus=1)
""",
        names=names,
    ) == [
        ("Model", "bogus"),
        ("AgeGrid", "bogus"),
        ("LiquidMargin", "bogus"),
        ("OuterContinuousMargin", "bogus"),
    ]


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (lcm.solvers.NNBEGM, "outer_search"),
        (lcm.outer_search.AdaptiveOuterMesh, "golden_iterations"),
        (lcm.consumption_savings_regime.OuterContinuousMargin, "adjustment_cost"),
    ],
)
def test_the_field_probe_resolves_real_fields(cls, expected):
    """The probe reads real constructor fields, so a negative result is evidence."""
    accepted = _accepted_keywords(cls)

    assert accepted is not None
    assert expected in accepted
