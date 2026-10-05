"""Docs, examples, tests and error messages use the keyword-only declaration API.

Every call to a multi-argument declaration constructor (`Model`, `ByAge`,
`ByAge.until`, `AgeRange`, `DeterministicTransition`, `StochasticTransition`) names
its arguments,
and no prose states an endpoint convention contradicting `ByAge.until`: the age
`stop_age_exclusive` is excluded and the last source below it uses `then`.
"""

import ast
import json
import re
from pathlib import Path

import pytest

from tests.ci.keyword_only_convention import (
    _markdown_source_units,
    _notebook_source_units,
)

_ROOT = Path(__file__).resolve().parents[1]
_DECLARATIONS = frozenset(
    {
        "Model",
        "ByAge",
        "until",
        "AgeRange",
        "DeterministicTransition",
        "StochasticTransition",
    }
)
# A call a test makes positionally on purpose, to assert it is rejected, carries
# this suppression on its line.
_DELIBERATE_POSITIONAL = "too-many-positional-arguments"
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_CONTRADICTORY_ENDPOINT_TEXT = (
    re.compile(r"last source[^.]*\buses `law`", re.IGNORECASE),
    re.compile(
        r"`?stop_age_exclusive`?[^.]*\b(?:is|are) (?:included|inclusive)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"`?start_age_inclusive`?[^.]*\b(?:is|are) (?:excluded|exclusive)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\buntil\(\s*boundary\b"),
)


def test_scanner_reports_positional_declaration_calls() -> None:
    code = (
        "ByAge.until(62, law=a, then=b)\n"
        "AgeRange(51, 62)\n"
        "DeterministicTransition(func=f)\n"
        "StochasticTransition(f)\n"
        "Model(...)\n"
        "Model({}, ages)  # ty: ignore[too-many-positional-arguments]\n"
    )
    assert _positional_declaration_calls(code=code) == (
        "until",
        "AgeRange",
        "StochasticTransition",
    )


@pytest.mark.parametrize(
    "text",
    [
        "The last source below the boundary uses `law`.",
        "Here `stop_age_exclusive` is included in the source ages.",
        "The age `start_age_inclusive` is excluded.",
        "Call `ByAge.until(boundary=62, law=a, then=b)`.",
    ],
)
def test_scanner_reports_contradictory_endpoint_text(text: str) -> None:
    assert _contradictory_endpoint_text(text=text)


def test_scanner_accepts_the_documented_endpoint_convention() -> None:
    text = (
        "The last source below `stop_age_exclusive` uses `then`, not `law`. "
        "The age `start_age_inclusive` is the first source using `law`."
    )
    assert _contradictory_endpoint_text(text=text) == ()


def test_declaration_calls_in_repository_sources_name_their_arguments() -> None:
    offenders = [
        f"{path.relative_to(_ROOT)}: {name}"
        for path, code in _code_units()
        for name in _positional_declaration_calls(code=code)
    ]
    assert offenders == []


def test_repository_prose_states_no_contradictory_endpoint_convention() -> None:
    offenders = [
        f"{path.relative_to(_ROOT)}: {match}"
        for path, text in _prose_units()
        for match in _contradictory_endpoint_text(text=text)
    ]
    assert offenders == []


def _positional_declaration_calls(*, code: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    lines = code.splitlines()
    return tuple(
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (name := _called_name(func=node.func)) in _DECLARATIONS
        and any(not _is_elision(arg=arg) for arg in node.args)
        and _DELIBERATE_POSITIONAL not in lines[node.lineno - 1]
    )


def _is_elision(*, arg: ast.expr) -> bool:
    """Whether `arg` is the `...` placeholder of an elided fragment."""
    return isinstance(arg, ast.Constant) and arg.value is Ellipsis


def _called_name(*, func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _contradictory_endpoint_text(*, text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0)
        for pattern in _CONTRADICTORY_ENDPOINT_TEXT
        for match in pattern.finditer(text)
    )


def _code_units() -> list[tuple[Path, str]]:
    units: list[tuple[Path, str]] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        units.append((path, source))
        units.extend((path, span) for span in _string_code_spans(source=source))
    for path in sorted((_ROOT / "docs").rglob("*.md")):
        source = path.read_text(encoding="utf-8")
        units.extend(
            (path, unit.source) for unit in _markdown_source_units(source=source)
        )
        units.extend((path, span) for span in _INLINE_CODE.findall(source))
    for path in sorted((_ROOT / "docs").rglob("*.ipynb")):
        source = path.read_text(encoding="utf-8")
        units.extend(
            (path, unit.source) for unit in _notebook_source_units(source=source)
        )
        units.extend(
            (path, span)
            for text in _notebook_markdown(source=source)
            for span in _INLINE_CODE.findall(text)
        )
    return units


def _prose_units() -> list[tuple[Path, str]]:
    units: list[tuple[Path, str]] = [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted((_ROOT / "docs").rglob("*.md"))
    ]
    units.extend(
        (path, text)
        for path in sorted((_ROOT / "docs").rglob("*.ipynb"))
        for text in _notebook_markdown(source=path.read_text(encoding="utf-8"))
    )
    units.extend(
        (path, text)
        for path in sorted((_ROOT / "src").rglob("*.py"))
        for text in _string_constants(source=path.read_text(encoding="utf-8"))
    )
    return units


def _python_files() -> list[Path]:
    this_file = Path(__file__).resolve()
    return [
        path
        for directory in ("src", "tests", "docs", "benchmarks")
        for path in sorted((_ROOT / directory).rglob("*.py"))
        if path.resolve() != this_file
    ]


def _string_constants(*, source: str) -> list[str]:
    return [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _string_code_spans(*, source: str) -> list[str]:
    return [
        span
        for text in _string_constants(source=source)
        for span in _INLINE_CODE.findall(text)
    ]


def _notebook_markdown(*, source: str) -> list[str]:
    return [
        "".join(cell["source"])
        if isinstance(cell["source"], list)
        else str(cell["source"])
        for cell in json.loads(source)["cells"]
        if cell.get("cell_type") == "markdown"
    ]
