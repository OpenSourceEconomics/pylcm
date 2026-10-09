"""Check that source annotations name precise types.

Three rules apply to every annotation position: parameters, returns, variables,
attributes, type aliases, type-parameter bounds and defaults, class bases and
`cast` targets, string annotations included:

- `PAN001`: `object` in an annotation;
- `PAN002`: `Any` in an annotation;
- `PAN003`: bare `str` on a regime, state, action or function name, which
  `lcm.typing` has an alias for.

Two placements of `object` pass without a marker:

- a parameter of a comparison or containment dunder, whose signature the data
  model fixes;
- a parameter that its own function narrows with `isinstance`, `issubclass` or
  `match`.

Any other finding is fixed, exempted by a marker on its own line directly above
it, or counted against the per-file baseline. The baseline only ever falls: a
count above it fails, and a count below it lowers it.
"""

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

BASELINE_PATH = Path(__file__).with_name("precise-annotations-baseline.json")

_EXEMPTION_PREFIX = "# annotation-exempt:"
_EXEMPTION = re.compile(
    rf"{_EXEMPTION_PREFIX} (?:heterogeneous=[a-z][a-z0-9-]*"
    r"|library-signature=[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+"
    r"|import-cycle=[A-Z]\w*)"
)
_BASELINED_CODES = ("PAN001", "PAN002", "PAN003")
_DETAILS = {
    "PAN001": "`object` in an annotation; name the precise type",
    "PAN002": "`Any` in an annotation; name the precise type",
    "PAN004": (
        "malformed exemption; write `# annotation-exempt: heterogeneous=<slug>`, "
        "`library-signature=<dotted.name>` or `import-cycle=<TypeName>` on its own "
        "line above the finding"
    ),
    "PAN005": "stale exemption; the line below it has no finding",
}
_COMPARISON_DUNDERS = frozenset(
    {"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__contains__"}
)
_ANY_MODULES = frozenset({"t", "typing", "typing_extensions"})
_OBJECT_MODULES = frozenset({"builtins"})
_FIRST_ARGUMENT_ONLY = frozenset(
    {
        "Annotated",
        "BFloat16",
        "Bool",
        "Complex",
        "Complex64",
        "Complex128",
        "Float",
        "Float16",
        "Float32",
        "Float64",
        "Inexact",
        "Int",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Integer",
        "Key",
        "Num",
        "PyTree",
        "Real",
        "Shaped",
        "UInt",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    }
)
_IMPLICIT_ALIAS_HEADS = frozenset(
    {
        "AbstractSet",
        "Callable",
        "Collection",
        "Iterable",
        "Iterator",
        "Mapping",
        "MappingProxyType",
        "MutableMapping",
        "Sequence",
        "dict",
        "frozenset",
        "list",
        "set",
        "tuple",
        "type",
    }
)
_COLLECTION_HEADS = frozenset(
    {
        "AbstractSet",
        "Collection",
        "Iterable",
        "Iterator",
        "Sequence",
        "frozenset",
        "list",
        "set",
        "tuple",
    }
)
_MAPPING_HEADS = frozenset({"Mapping", "MappingProxyType", "dict"})
_STATE = r"(?<!post_decision_)(?<!next_)state"
_LABEL_NAMES = (
    (re.compile(r"(?:\w+_)?regime(?:_name)?"), "RegimeName"),
    (re.compile(rf"(?:\w+_)?{_STATE}(?:_name)?"), "StateName"),
    (re.compile(r"(?:\w+_)?action(?:_name)?"), "ActionName"),
    (re.compile(r"(?:\w+_)?func(?:tion)?(?:_name)?"), "FunctionName"),
)
_LABEL_COLLECTIONS = (
    (re.compile(r"(?:\w+_)?regime_names"), "RegimeName"),
    (re.compile(rf"(?:\w+_)?{_STATE}_names"), "StateName"),
    (re.compile(r"(?:\w+_)?action_names"), "ActionName"),
    (re.compile(r"(?:\w+_)?func(?:tion)?_names"), "FunctionName"),
)
_LABEL_KEYED_MAPPINGS = (
    (
        re.compile(r"(?:\w+_)?(?:regime_to_\w+|(?:by|per)_regime(?:_name)?)"),
        "RegimeName",
    ),
)


@dataclass(frozen=True, kw_only=True)
class AnnotationViolation:
    """One imprecise annotation, or one problem with an exemption marker."""

    path: Path
    line: int
    code: str
    construct: str
    detail: str


@dataclass(frozen=True, kw_only=True)
class _Finding:
    line: int
    code: str
    construct: str
    detail: str


def find_annotation_violations(
    *, paths: Iterable[Path]
) -> tuple[AnnotationViolation, ...]:
    """Return the violations in the `.py` files among `paths` in source order."""
    return tuple(
        AnnotationViolation(
            path=path,
            line=finding.line,
            code=finding.code,
            construct=finding.construct,
            detail=finding.detail,
        )
        for path in paths
        if path.suffix == ".py"
        for finding in _findings_for_source(source=path.read_text())
    )


def main(
    *,
    paths: Iterable[Path],
    baseline_path: Path = BASELINE_PATH,
    write_baseline: bool = False,
) -> int:
    """Hold each file's counts to the baseline and return the exit status.

    Exemption problems always fail. A count above the baseline fails and lists
    that rule's findings in the file; a count below it, or an entry for a file
    that no longer exists, lowers the baseline and fails once so the rewrite is
    staged. `write_baseline` records the current counts of `paths` instead.
    """
    sources = tuple(path for path in paths if path.suffix == ".py")
    violations = find_annotation_violations(paths=sources)
    baseline = _read_baseline(path=baseline_path)
    if write_baseline:
        baseline.update(
            {source.as_posix(): {} for source in sources} | _counts(violations)
        )
        _write_baseline(path=baseline_path, baseline=baseline)
        return 0

    lines: list[str] = [
        _render(violation)
        for violation in violations
        if violation.code not in _BASELINED_CODES
    ]
    updated = _ratchet(
        sources=sources, violations=violations, baseline=baseline, lines=lines
    )
    for key in sorted(baseline):
        if not Path(key).exists():
            del updated[key]
            lines.append(f"{key}: removed from the baseline; the file no longer exists")
    if updated != baseline:
        _write_baseline(path=baseline_path, baseline=updated)
    for line in lines:
        sys.stdout.write(f"{line}\n")
    return int(bool(lines))


def _ratchet(
    *,
    sources: Sequence[Path],
    violations: Sequence[AnnotationViolation],
    baseline: dict[str, dict[str, int]],
    lines: list[str],
) -> dict[str, dict[str, int]]:
    counts = _counts(violations)
    updated = {key: dict(entry) for key, entry in baseline.items()}
    for source in sources:
        key = source.as_posix()
        for code in _BASELINED_CODES:
            current = counts.get(key, {}).get(code, 0)
            allowed = baseline.get(key, {}).get(code, 0)
            if current > allowed:
                lines.extend(
                    _render(violation)
                    for violation in violations
                    if violation.path == source and violation.code == code
                )
                lines.append(
                    f"{key}: {code} count {current} exceeds the baseline {allowed}"
                )
            elif current < allowed:
                updated[key][code] = current
                lines.append(
                    f"{key}: {code} baseline lowered from {allowed} to {current}"
                )
    return updated


def _counts(
    violations: Iterable[AnnotationViolation],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for violation in violations:
        if violation.code in _BASELINED_CODES:
            counts.setdefault(violation.path.as_posix(), Counter())[violation.code] += 1
    return {key: dict(counter) for key, counter in counts.items()}


def _read_baseline(*, path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        msg = f"{path}: the baseline must map file paths to rule counts."
        raise TypeError(msg)
    baseline: dict[str, dict[str, int]] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict) or not all(
            code in _BASELINED_CODES and isinstance(count, int)
            for code, count in entry.items()
        ):
            msg = f"{path}: {key!r} must map rule codes to integer counts."
            raise TypeError(msg)
        baseline[key] = dict(entry)
    return baseline


def _write_baseline(*, path: Path, baseline: dict[str, dict[str, int]]) -> None:
    kept = {
        key: {code: count for code, count in sorted(entry.items()) if count}
        for key, entry in sorted(baseline.items())
    }
    path.write_text(
        json.dumps({key: entry for key, entry in kept.items() if entry}, indent=2)
        + "\n"
    )


def _render(violation: AnnotationViolation) -> str:
    return (
        f"{violation.path}:{violation.line}: {violation.code} "
        f"{violation.construct}: {violation.detail}"
    )


def _findings_for_source(*, source: str) -> list[_Finding]:
    visitor = _AnnotationVisitor()
    visitor.visit(ast.parse(source))
    standalone, trailing = _exemption_comments(source=source)
    markers = {
        line: _EXEMPTION.fullmatch(text) is not None
        for line, text in standalone.items()
    }
    exempted_lines = {line + 1 for line, wellformed in markers.items() if wellformed}
    findings = [
        finding for finding in visitor.findings if finding.line not in exempted_lines
    ]
    lines_with_findings = {finding.line for finding in visitor.findings}
    for line, wellformed in markers.items():
        if not wellformed:
            findings.append(_problem(line=line, code="PAN004"))
        elif line + 1 not in lines_with_findings:
            findings.append(_problem(line=line, code="PAN005"))
    findings.extend(_problem(line=line, code="PAN004") for line in trailing)
    return sorted(findings, key=lambda finding: (finding.line, finding.code))


def _problem(*, line: int, code: str) -> _Finding:
    return _Finding(line=line, code=code, construct="exemption", detail=_DETAILS[code])


def _exemption_comments(*, source: str) -> tuple[dict[int, str], list[int]]:
    source_lines = source.splitlines()
    standalone: dict[int, str] = {}
    trailing: list[int] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        text = token.string.strip()
        if not text.startswith(_EXEMPTION_PREFIX.rstrip(":")):
            continue
        line, column = token.start
        if source_lines[line - 1][:column].strip():
            trailing.append(line)
        else:
            standalone[line] = text
    return standalone, trailing


class _AnnotationVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[_Finding] = []
        self._scopes: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_type_params(node.type_params)
        for base in node.bases:
            self._check_type(node=base, construct="base")
        for child in (*node.decorator_list, *node.keywords):
            self.visit(child)
        self._scopes.append("class")
        for statement in node.body:
            self.visit(statement)
        self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _head_name(node.annotation) == "TypeAlias" and node.value is not None:
            self._check_type(node=node.value, construct="type-alias")
            return
        if isinstance(node.target, ast.Attribute) or self._scopes[-1:] == ["class"]:
            construct = "attribute"
        else:
            construct = "variable"
        self._check_type(node=node.annotation, construct=construct)
        self._check_label(
            annotation=node.annotation,
            identifier=_target_name(node.target),
            construct=construct,
        )
        if node.value is not None:
            self.visit(node.value)

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        self._check_type_params(node.type_params)
        self._check_type(node=node.value, construct="type-alias")

    def visit_Assign(self, node: ast.Assign) -> None:
        target = node.targets[0] if len(node.targets) == 1 else None
        if (
            not self._scopes
            and isinstance(target, ast.Name)
            and target.id[:1].isupper()
            and _is_implicit_alias(node.value)
        ):
            self._check_type(node=node.value, construct="type-alias")
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _head_name(node.func)
        if callee == "cast" and node.args:
            self._check_type(node=node.args[0], construct="cast")
            arguments = node.args[1:]
        elif callee in {"NewType", "TypeAliasType"} and len(node.args) >= 2:
            self._check_type(node=node.args[1], construct="type-alias")
            arguments = node.args[2:]
        elif callee == "TypeVar":
            for expression in (
                *node.args[1:],
                *(
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"bound", "default"}
                ),
            ):
                self._check_type(node=expression, construct="type-param")
            return
        else:
            self.generic_visit(node)
            return
        for argument in (*arguments, *(keyword.value for keyword in node.keywords)):
            self.visit(argument)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._check_type_params(node.type_params)
        narrowed = _narrowed_names(node)
        arguments = node.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if argument.annotation is None:
                continue
            if _is_bare_object(argument.annotation) and (
                node.name in _COMPARISON_DUNDERS or argument.arg in narrowed
            ):
                continue
            self._check_type(node=argument.annotation, construct="param")
            self._check_label(
                annotation=argument.annotation,
                identifier=argument.arg,
                construct="param",
            )
        for argument, construct in (
            (arguments.vararg, "vararg"),
            (arguments.kwarg, "kwarg"),
        ):
            if argument is not None:
                self._check_type(node=argument.annotation, construct=construct)
        self._check_type(node=node.returns, construct="return")
        for child in (
            *node.decorator_list,
            *arguments.defaults,
            *(default for default in arguments.kw_defaults if default is not None),
        ):
            self.visit(child)
        self._scopes.append("function")
        for statement in node.body:
            self.visit(statement)
        self._scopes.pop()

    def _check_type_params(self, type_params: list[ast.type_param]) -> None:
        for type_param in type_params:
            if isinstance(type_param, ast.TypeVar):
                self._check_type(node=type_param.bound, construct="type-param")
            if isinstance(type_param, ast.TypeVar | ast.ParamSpec | ast.TypeVarTuple):
                self._check_type(node=type_param.default_value, construct="type-param")

    def _check_type(self, *, node: ast.expr | None, construct: str) -> None:
        self.findings.extend(
            _Finding(line=line, code=code, construct=construct, detail=_DETAILS[code])
            for line, code in _imprecise_tokens(node=node, line_offset=0)
        )

    def _check_label(
        self, *, annotation: ast.expr, identifier: str | None, construct: str
    ) -> None:
        if identifier is None:
            return
        detail = _label_detail(annotation=annotation, identifier=identifier)
        if detail is not None:
            self.findings.append(
                _Finding(
                    line=annotation.lineno,
                    code="PAN003",
                    construct=construct,
                    detail=detail,
                )
            )


def _imprecise_tokens(
    *, node: ast.expr | None, line_offset: int
) -> Iterator[tuple[int, str]]:
    if node is None:
        return
    if _is_spelled(node=node, name="object", modules=_OBJECT_MODULES):
        yield node.lineno + line_offset, "PAN001"
    elif _is_spelled(node=node, name="Any", modules=_ANY_MODULES):
        yield node.lineno + line_offset, "PAN002"
    else:
        for child, child_offset in _type_children(node=node, line_offset=line_offset):
            yield from _imprecise_tokens(node=child, line_offset=child_offset)


def _type_children(*, node: ast.expr, line_offset: int) -> list[tuple[ast.expr, int]]:
    """Return the sub-expressions of a type expression that are types themselves.

    A string is parsed as the type it spells. `Literal` contributes nothing; a
    jaxtyping dtype, `PyTree` and `Annotated` contribute only their first argument,
    so a shape, a structure name or metadata is never read as a type.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parsed = _parse_type_string(node.value)
        return [] if parsed is None else [(parsed, line_offset + node.lineno - 1)]
    if isinstance(node, ast.Subscript):
        head = _head_name(node.value)
        elements = (
            node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        )
        if head == "Literal":
            elements = []
        elif head in _FIRST_ARGUMENT_ONLY:
            elements = elements[:1]
        return [(element, line_offset) for element in elements]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return [(node.left, line_offset), (node.right, line_offset)]
    if isinstance(node, ast.Tuple | ast.List):
        return [(element, line_offset) for element in node.elts]
    if isinstance(node, ast.Starred):
        return [(node.value, line_offset)]
    return []


def _label_detail(*, annotation: ast.expr, identifier: str) -> str | None:
    name = identifier.lstrip("_")
    annotation = _without_none(_unquoted(annotation))
    if _is_str(annotation):
        alias = _matching_alias(rules=_LABEL_NAMES, name=name)
        return (
            None
            if alias is None
            else f"annotate `{identifier}` as `{alias}`, not `str`"
        )
    if not isinstance(annotation, ast.Subscript):
        return None
    head = _head_name(annotation.value)
    elements = (
        annotation.slice.elts
        if isinstance(annotation.slice, ast.Tuple)
        else [annotation.slice]
    )
    if not _is_str(elements[0]):
        return None
    if head in _COLLECTION_HEADS:
        alias = _matching_alias(rules=_LABEL_COLLECTIONS, name=name)
        role = "elements"
    elif head in _MAPPING_HEADS:
        alias = _matching_alias(rules=_LABEL_KEYED_MAPPINGS, name=name)
        role = "keys"
    else:
        return None
    if alias is None:
        return None
    return f"annotate the {role} of `{identifier}` as `{alias}`, not `str`"


def _matching_alias(
    *, rules: Iterable[tuple[re.Pattern[str], str]], name: str
) -> str | None:
    return next((alias for pattern, alias in rules if pattern.fullmatch(name)), None)


def _narrowed_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and _head_name(child.func) in {"isinstance", "issubclass"}
            and child.args
            and isinstance(child.args[0], ast.Name)
        ):
            names.add(child.args[0].id)
        elif isinstance(child, ast.Match) and isinstance(child.subject, ast.Name):
            names.add(child.subject.id)
    return frozenset(names)


def _is_bare_object(annotation: ast.expr) -> bool:
    return _is_spelled(
        node=_without_none(_unquoted(annotation)),
        name="object",
        modules=_OBJECT_MODULES,
    )


def _is_implicit_alias(value: ast.expr) -> bool:
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
        return True
    return isinstance(value, ast.Subscript) and (
        _head_name(value.value) in _IMPLICIT_ALIAS_HEADS
    )


def _is_spelled(*, node: ast.expr, name: str, modules: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id == name
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id in modules
    )


def _is_str(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "str"


def _unquoted(annotation: ast.expr) -> ast.expr:
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        parsed = _parse_type_string(annotation.value)
        if parsed is not None:
            return parsed
    return annotation


def _without_none(annotation: ast.expr) -> ast.expr:
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        if _is_none(annotation.right):
            return annotation.left
        if _is_none(annotation.left):
            return annotation.right
    return annotation


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _parse_type_string(text: str) -> ast.expr | None:
    try:
        return ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return None


def _head_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _target_name(target: ast.expr) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m tests.ci.precise_annotations")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current counts of the given files as their baseline",
    )
    parser.add_argument("paths", nargs="*", type=Path)
    return parser.parse_args(arguments)


if __name__ == "__main__":
    _arguments = _parse_arguments(sys.argv[1:])
    raise SystemExit(
        main(paths=_arguments.paths, write_baseline=_arguments.write_baseline)
    )
