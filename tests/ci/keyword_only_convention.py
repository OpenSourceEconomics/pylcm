"""Check the repository's keyword-only function convention."""

import ast
import io
import json
import re
import sys
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

_EXEMPTION_PREFIX = "# keyword-only-exempt:"
_LIBRARY_CALLBACK_EXEMPTION = re.compile(
    rf"{_EXEMPTION_PREFIX} library-callback=[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
)
_PRIMARY_ARGUMENT_EXEMPTION = re.compile(
    rf"{_EXEMPTION_PREFIX} primary-argument=(?P<name>[A-Za-z_]\w*)"
)
_MARKDOWN_FENCE_START = re.compile(
    r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})[ \t]*(?:python|py)(?:[ \t]+.*)?$"
)


_ARITHMETIC_MODULE_SUFFIX = (
    "src",
    "_lcm",
    "egm",
    "upper_envelope",
    "double_double.py",
)
_ARITHMETIC_DECLARATION = "Keyword-only exemption: arithmetic-only module."
_ARITHMETIC_OPERATORS = frozenset(
    {
        "_shift_clear_of_the_split_floor",
        "_split",
        "dd_add",
        "dd_add_float",
        "dd_from_difference",
        "dd_mul",
        "dd_mul_float",
        "dd_negate",
        "dd_quotient",
        "dd_quotient_bounded",
        "is_stored_zero",
        "normalizing_exponent",
        "scale_by_power_of_two",
        "scale_tail_bound",
        "two_prod",
        "two_sum",
    }
)


@dataclass(frozen=True, kw_only=True)
class KeywordOnlyViolation:
    """A function definition that violates the keyword-only convention."""

    path: Path
    line: int
    qualified_name: str
    code: str
    positional_parameters: tuple[str, ...]
    cell: int | None = None


@dataclass(frozen=True, kw_only=True)
class _Definition:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    qualified_name: str
    is_method: bool


@dataclass(frozen=True, kw_only=True)
class _SourceUnit:
    source: str
    line_offset: int = 0
    cell: int | None = None


class _DefinitionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.definitions: list[_Definition] = []
        self._scope: list[tuple[str, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append((node.name, "class"))
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.definitions.append(
            _Definition(
                node=node,
                qualified_name=".".join(
                    (*[name for name, _kind in self._scope], node.name)
                ),
                is_method=bool(self._scope and self._scope[-1][1] == "class"),
            )
        )
        self._scope.append((node.name, "function"))
        self.generic_visit(node)
        self._scope.pop()


def _parameter_info(
    *, node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool
) -> tuple[tuple[str, ...], int]:
    positional_parameters = [
        argument.arg for argument in (*node.args.posonlyargs, *node.args.args)
    ]
    parameter_count = len(positional_parameters) + len(node.args.kwonlyargs)

    if (
        is_method
        and positional_parameters
        and positional_parameters[0] in {"self", "cls"}
    ):
        positional_parameters = positional_parameters[1:]
        parameter_count -= 1

    return tuple(positional_parameters), parameter_count


def _standalone_comments(*, source: str) -> dict[int, str]:
    source_lines = source.splitlines()
    comments: dict[int, str] = {}
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line_number, column = token.start
        if source_lines[line_number - 1][:column].strip():
            continue
        comments[line_number] = token.string.strip()
    return comments


def _exemption_for(
    *,
    definition: _Definition,
    comments: dict[int, str],
) -> tuple[str | None, int | None]:
    decorator_lines = [decorator.lineno for decorator in definition.node.decorator_list]
    definition_start = min([definition.node.lineno, *decorator_lines])
    marker_line = definition_start - 1
    if marker_line < 1:
        return None, None

    marker = comments.get(marker_line, "")
    if not marker.startswith(_EXEMPTION_PREFIX):
        return None, None
    if _LIBRARY_CALLBACK_EXEMPTION.fullmatch(marker):
        return "library-callback", marker_line
    primary_match = _PRIMARY_ARGUMENT_EXEMPTION.fullmatch(marker)
    if primary_match is not None:
        return f"primary-argument={primary_match.group('name')}", marker_line
    return "malformed", marker_line


def _is_declared_arithmetic_module(*, path: Path, tree: ast.Module) -> bool:
    suffix_length = len(_ARITHMETIC_MODULE_SUFFIX)
    has_expected_path = path.parts[-suffix_length:] == _ARITHMETIC_MODULE_SUFFIX
    docstring = ast.get_docstring(tree, clean=False) or ""
    return has_expected_path and _ARITHMETIC_DECLARATION in docstring


def _regular_violation(
    *,
    definition: _Definition,
    is_noncompliant: bool,
    path: Path,
    positional_parameters: tuple[str, ...],
    comments: dict[int, str],
) -> KeywordOnlyViolation | None:
    exemption, marker_line = _exemption_for(definition=definition, comments=comments)
    violation_line = marker_line if marker_line is not None else definition.node.lineno
    if exemption == "malformed":
        code = "KWO002"
    elif exemption == "library-callback":
        if is_noncompliant:
            return None
        code = "KWO003"
    elif exemption is not None and exemption.startswith("primary-argument="):
        primary_argument = exemption.removeprefix("primary-argument=")
        if not is_noncompliant:
            code = "KWO003"
        elif positional_parameters != (primary_argument,):
            code = "KWO002"
        else:
            return None
    elif not is_noncompliant:
        return None
    else:
        code = "KWO001"

    return KeywordOnlyViolation(
        path=path,
        line=violation_line,
        qualified_name=definition.qualified_name,
        code=code,
        positional_parameters=positional_parameters,
    )


def _arithmetic_violation(
    *,
    definition: _Definition,
    path: Path,
    positional_parameters: tuple[str, ...],
) -> KeywordOnlyViolation | None:
    if definition.node.name in _ARITHMETIC_OPERATORS:
        return None
    return KeywordOnlyViolation(
        path=path,
        line=definition.node.lineno,
        qualified_name=definition.qualified_name,
        code="KWO004",
        positional_parameters=positional_parameters,
    )


def _orphaned_exemption_violations(
    *,
    definitions: list[_Definition],
    path: Path,
    comments: dict[int, str],
) -> list[KeywordOnlyViolation]:
    attached_marker_lines = {
        marker_line
        for definition in definitions
        for _exemption, marker_line in [
            _exemption_for(definition=definition, comments=comments)
        ]
        if marker_line is not None
    }
    violations = []
    for line_number, marker in comments.items():
        if (
            not marker.startswith(_EXEMPTION_PREFIX)
            or line_number in attached_marker_lines
        ):
            continue
        is_recognized = bool(
            _LIBRARY_CALLBACK_EXEMPTION.fullmatch(marker)
            or _PRIMARY_ARGUMENT_EXEMPTION.fullmatch(marker)
        )
        code = "KWO003" if is_recognized else "KWO002"
        violations.append(
            KeywordOnlyViolation(
                path=path,
                line=line_number,
                qualified_name="<module>",
                code=code,
                positional_parameters=(),
            )
        )
    return violations


def _violations_for_source(
    *, path: Path, source_unit: _SourceUnit
) -> list[KeywordOnlyViolation]:
    tree = ast.parse(source_unit.source, filename=str(path))
    comments = _standalone_comments(source=source_unit.source)
    arithmetic_module = _is_declared_arithmetic_module(path=path, tree=tree)
    used_arithmetic_exemption = False
    violations: list[KeywordOnlyViolation] = []
    visitor = _DefinitionVisitor()
    visitor.visit(tree)
    violations.extend(
        _orphaned_exemption_violations(
            definitions=visitor.definitions,
            path=path,
            comments=comments,
        )
    )
    for definition in visitor.definitions:
        positional_parameters, parameter_count = _parameter_info(
            node=definition.node, is_method=definition.is_method
        )
        is_noncompliant = parameter_count >= 2 and bool(positional_parameters)
        if arithmetic_module:
            violation = _arithmetic_violation(
                definition=definition,
                path=path,
                positional_parameters=positional_parameters,
            )
            used_arithmetic_exemption |= (
                is_noncompliant and definition.node.name in _ARITHMETIC_OPERATORS
            )
        else:
            violation = _regular_violation(
                definition=definition,
                is_noncompliant=is_noncompliant,
                path=path,
                positional_parameters=positional_parameters,
                comments=comments,
            )
        if violation is not None:
            violations.append(violation)

    if arithmetic_module and not used_arithmetic_exemption:
        violations.append(
            KeywordOnlyViolation(
                path=path,
                line=1,
                qualified_name="<module>",
                code="KWO003",
                positional_parameters=(),
            )
        )
    violations.sort(key=lambda violation: violation.line)
    return [
        replace(
            violation,
            line=violation.line + source_unit.line_offset,
            cell=source_unit.cell,
        )
        for violation in violations
    ]


def _markdown_source_units(*, source: str) -> tuple[_SourceUnit, ...]:
    lines = source.splitlines()
    source_units: list[_SourceUnit] = []
    line_index = 0
    while line_index < len(lines):
        opening_match = _MARKDOWN_FENCE_START.fullmatch(lines[line_index])
        if opening_match is None:
            line_index += 1
            continue

        fence = opening_match.group("fence")
        indentation = len(opening_match.group("indent"))
        closing_fence = re.compile(
            rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$"
        )
        line_offset = line_index + 1
        line_index += 1
        code_lines: list[str] = []
        while line_index < len(lines) and not closing_fence.fullmatch(
            lines[line_index]
        ):
            line = lines[line_index]
            if indentation and line.startswith(" " * indentation):
                line = line[indentation:]
            code_lines.append(line)
            line_index += 1
        source_units.append(
            _SourceUnit(source="\n".join(code_lines), line_offset=line_offset)
        )
        line_index += 1

    return tuple(source_units)


def _notebook_source_units(*, source: str) -> tuple[_SourceUnit, ...]:
    notebook = json.loads(source)
    source_units: list[_SourceUnit] = []
    for cell_number, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        cell_source = cell.get("source", "")
        if isinstance(cell_source, list):
            code = "".join(cell_source)
        else:
            code = str(cell_source)
        source_units.append(_SourceUnit(source=code, cell=cell_number))
    return tuple(source_units)


def _source_units_for_path(*, path: Path) -> tuple[_SourceUnit, ...]:
    source = path.read_text()
    if path.suffix == ".py":
        return (_SourceUnit(source=source),)
    if path.suffix == ".md":
        return _markdown_source_units(source=source)
    if path.suffix == ".ipynb":
        return _notebook_source_units(source=source)
    return ()


def find_keyword_only_violations(
    *, paths: Iterable[Path]
) -> tuple[KeywordOnlyViolation, ...]:
    """Return convention violations in ``paths`` in stable source order."""
    return tuple(
        violation
        for path in paths
        for source_unit in _source_units_for_path(path=path)
        for violation in _violations_for_source(path=path, source_unit=source_unit)
    )


def _render_violation(violation: KeywordOnlyViolation) -> str:
    if violation.code == "KWO001":
        detail = "make positional parameters keyword-only: " + ", ".join(
            violation.positional_parameters
        )
    elif violation.code == "KWO002":
        detail = "malformed or invalid keyword-only exemption"
    elif violation.code == "KWO003":
        detail = "stale keyword-only exemption"
    else:
        detail = "non-operator definition in arithmetic-only module"
    if violation.cell is None:
        location = f"{violation.path}:{violation.line}"
    else:
        location = f"{violation.path}:cell {violation.cell}:line {violation.line}"
    return f"{location}: {violation.code} {violation.qualified_name}: {detail}"


def main(*, paths: Iterable[Path]) -> int:
    """Print violations for pre-commit and return its process exit status."""
    violations = find_keyword_only_violations(paths=paths)
    for violation in violations:
        sys.stdout.write(f"{_render_violation(violation)}\n")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main(paths=(Path(argument) for argument in sys.argv[1:])))
