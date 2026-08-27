"""Constructor keywords shown in the book exist on the classes they name.

The API index guards exported *names*, so a page can keep spelling a keyword
argument that a dataclass no longer has: the name still resolves, the fence
still renders, and a reader copying it gets `TypeError`. This module compares
every `ClassName(kwarg=...)` written in a documentation fence against
`dataclasses.fields` of the class that name resolves to.
"""

import ast
import dataclasses
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


def _public_dataclasses() -> dict[str, Any]:
    """Map each documented public class name to the class object."""
    modules = (lcm, lcm.solvers, lcm.outer_search, lcm.consumption_savings_regime)
    found: dict[str, Any] = {}
    for module in modules:
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if isinstance(obj, type) and dataclasses.is_dataclass(obj):
                found.setdefault(name, obj)
    return found


def _keywords_in_block(*, block: str, names: set[str]) -> list[tuple[str, str]]:
    """List `(class name, keyword)` for each named class constructed in `block`."""
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return []
    return [
        (node.func.id, keyword.arg)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        if node.func.id in names
        for keyword in node.keywords
        if keyword.arg is not None
    ]


def _documentation_pages() -> list[Path]:
    """List the tracked documentation pages, excluding build output."""
    return [p for p in sorted(_DOCS.rglob("*.md")) if "_build" not in p.parts]


def _documented_calls() -> list[tuple[Path, str, str]]:
    """List `(page, class name, keyword)` for every keyword written in a fence."""
    names = set(_public_dataclasses())
    return [
        (page, cls_name, kwarg)
        for page in _documentation_pages()
        for block in _FENCE.findall(page.read_text(encoding="utf-8"))
        for cls_name, kwarg in _keywords_in_block(block=block, names=names)
    ]


def test_documented_constructor_keywords_exist_on_their_class():
    """No documentation fence passes a keyword its class does not define."""
    classes = _public_dataclasses()
    orphans = [
        f"{page.relative_to(_DOCS)}: {cls_name}({kwarg}=...)"
        for page, cls_name, kwarg in _documented_calls()
        if kwarg not in {f.name for f in dataclasses.fields(classes[cls_name])}
    ]

    assert orphans == []


def test_the_sweep_reaches_the_documentation():
    """The sweep is non-empty, so an empty orphan list means agreement."""
    calls = _documented_calls()

    assert len(calls) > 50


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (lcm.solvers.NNBEGM, "outer_search"),
        (lcm.outer_search.AdaptiveOuterMesh, "golden_iterations"),
        (lcm.consumption_savings_regime.OuterContinuousMargin, "adjustment_cost"),
    ],
)
def test_the_field_probe_resolves_real_fields(cls, expected):
    """The probe reads real dataclass fields, so a negative result is evidence."""
    assert expected in {f.name for f in dataclasses.fields(cls)}
