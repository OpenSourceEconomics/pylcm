"""Check that annotations name precise types.

The rules apply to every annotation position: parameters, returns, variables,
attributes, type aliases, type-parameter bounds and defaults, class bases and
`cast` targets, string annotations included:

- `PAN001`: `object` in an annotation;
- `PAN002`: `Any` in an annotation;
- `PAN003`: bare `str` on a name that has a domain alias (a regime, state,
  action, function, qualified or parameter name), or on a target, source or
  argument name, whose alias is picked by hand;
- `PAN006`: a generic without its type arguments (`Callable`, `dict`, ...),
  which leaves them `Any`;
- `PAN007`: a type alias not written as a `type X = ...` statement;
- `PAN008`: a `# noqa` naming a `PAN` code that its line does not report.

Two placements are exempt by rule:

- `object` in a parameter whose type the data model fixes: every parameter of a
  comparison or containment dunder, `__setattr__`'s `value` and
  `__deepcopy__`'s `memo`;
- `object` or `Any` in an alias of the `else:` branch of `if TYPE_CHECKING:`,
  the runtime fallback that the beartype claw sees, when a comment directly
  above it, or above the run of statements it ends, gives the reason.

Any other deliberate finding takes `# noqa: PANxxx - <reason>` on its line. A
code without a reason suppresses nothing.
"""

import ast
import io
import json
import re
import sys
import tokenize
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

type SourceText = str


_DETAILS = {
    "PAN001": "`object` in an annotation; name the precise type",
    "PAN002": "`Any` in an annotation; name the precise type",
}
_FALLBACK_DETAILS = {
    "PAN001": "`object` in a runtime fallback; give its reason in a comment "
    "directly above it",
    "PAN002": "`Any` in a runtime fallback; give its reason in a comment "
    "directly above it",
}
_COMPARISON_DUNDERS = frozenset(
    {"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__contains__"}
)
_DATA_MODEL_PARAMETERS = frozenset({("__setattr__", "value"), ("__deepcopy__", "memo")})
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
_GENERICS = frozenset(
    {
        "AbstractSet",
        "AsyncGenerator",
        "AsyncIterable",
        "AsyncIterator",
        "Awaitable",
        "Callable",
        "ChainMap",
        "Collection",
        "Container",
        "Coroutine",
        "Counter",
        "Generator",
        "ItemsView",
        "Iterable",
        "Iterator",
        "KeysView",
        "Mapping",
        "MappingProxyType",
        "MutableMapping",
        "MutableSequence",
        "MutableSet",
        "OrderedDict",
        "Reversible",
        "Sequence",
        "Set",
        "ValuesView",
        "defaultdict",
        "deque",
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
_DOMAIN = r"(?:target|source|arg)(?:_name)?"


@dataclass(frozen=True, kw_only=True)
class _NameRule:
    alias: str | None
    """The alias to name, or `None` when the author picks a domain alias."""

    scalar: re.Pattern[str]
    elements: re.Pattern[str]
    keys: re.Pattern[str] | None = None


_NAME_RULES = (
    _NameRule(
        alias="RegimeName",
        scalar=re.compile(r"(?:\w+_)?regime(?:_name)?"),
        elements=re.compile(r"(?:\w+_)?regime_names"),
        keys=re.compile(r"(?:\w+_)?(?:regime_to_\w+|(?:by|per)_regime(?:_name)?)"),
    ),
    _NameRule(
        alias="StateName",
        scalar=re.compile(rf"(?:\w+_)?{_STATE}(?:_name)?"),
        elements=re.compile(rf"(?:\w+_)?{_STATE}_names"),
    ),
    _NameRule(
        alias="ActionName",
        scalar=re.compile(r"(?:\w+_)?action(?:_name)?"),
        elements=re.compile(r"(?:\w+_)?action_names"),
    ),
    _NameRule(
        alias="FunctionName",
        scalar=re.compile(r"(?:\w+_)?func(?:tion)?(?:_name)?"),
        elements=re.compile(r"(?:\w+_)?func(?:tion)?_names"),
    ),
    _NameRule(
        alias="QualifiedName",
        scalar=re.compile(r"(?:\w+_)?(?:flat_param(?:eter)?_name|q(?:ualified_)?name)"),
        elements=re.compile(
            r"(?:\w+_)?(?:flat_param(?:eter)?_names|q(?:ualified_)?names)"
        ),
    ),
    _NameRule(
        alias="ParameterName",
        scalar=re.compile(r"(?:\w+_)?(?<!signature_)param(?:eter)?_name"),
        elements=re.compile(r"(?:\w+_)?(?<!signature_)param(?:eter)?_names"),
    ),
    _NameRule(
        alias=None,
        scalar=re.compile(rf"(?:\w+_)?{_DOMAIN}"),
        elements=re.compile(rf"(?:\w+_)?{_DOMAIN}s"),
        keys=re.compile(
            rf"(?:\w+_)?(?:{_DOMAIN}s|{_DOMAIN}_to_\w+|(?:by|per)_{_DOMAIN})"
        ),
    ),
)
_NOQA = re.compile(r"#\s*noqa:\s*(?P<codes>[A-Z]+\d+(?:[\s,]+[A-Z]+\d+)*)(?P<rest>.*)")
_NOQA_REASON = re.compile(r"\s+(?:--?|—)\s+\S.*")


@dataclass(frozen=True, kw_only=True)
class AnnotationViolation:
    """One annotation that does not name a precise type."""

    path: Path
    line: int
    code: str
    construct: str
    detail: str
    cell: int | None = None
    """The one-based cell number when `path` is a notebook."""


@dataclass(frozen=True, kw_only=True)
class _Finding:
    line: int
    code: str
    construct: str
    detail: str


@dataclass(frozen=True, kw_only=True)
class _SourceUnit:
    source: SourceText
    cell: int | None = None


def find_annotation_violations(
    *, paths: Iterable[Path]
) -> tuple[AnnotationViolation, ...]:
    """Return the violations in the `.py` files and notebooks among `paths`.

    Findings keep the order of `paths`, and within a file or cell are sorted by
    line and code.
    """
    return tuple(
        AnnotationViolation(
            path=path,
            line=finding.line,
            code=finding.code,
            construct=finding.construct,
            detail=finding.detail,
            cell=unit.cell,
        )
        for path in paths
        for unit in _source_units(path=path)
        for finding in _findings_for_source(source=unit.source)
    )


def main(*, paths: Iterable[Path]) -> int:
    """Print every violation and a count per rule; return the exit status."""
    violations = find_annotation_violations(paths=paths)
    if not violations:
        return 0
    for violation in violations:
        sys.stdout.write(f"{_render(violation)}\n")
    counts = Counter(violation.code for violation in violations)
    per_code = ", ".join(f"{code} {counts[code]}" for code in sorted(counts))
    files = len({violation.path for violation in violations})
    sys.stdout.write(f"{len(violations)} findings in {files} files: {per_code}\n")
    return 1


def _source_units(*, path: Path) -> tuple[_SourceUnit, ...]:
    if path.suffix == ".py":
        return (_SourceUnit(source=path.read_text()),)
    if path.suffix != ".ipynb":
        return ()
    cells = json.loads(path.read_text())["cells"]
    return tuple(
        _SourceUnit(source="".join(cell["source"]), cell=number)
        for number, cell in enumerate(cells, start=1)
        if cell["cell_type"] == "code"
    )


def _render(violation: AnnotationViolation) -> str:
    location = (
        f"{violation.path}:{violation.line}"
        if violation.cell is None
        else f"{violation.path}:cell {violation.cell}:line {violation.line}"
    )
    return f"{location}: {violation.code} {violation.construct}: {violation.detail}"


def _findings_for_source(*, source: SourceText) -> list[_Finding]:
    comments = _comments(source=source)
    visitor = _AnnotationVisitor(
        standalone_comment_lines=frozenset(
            line for line, (text, standalone) in comments.items() if standalone
        )
    )
    visitor.visit(ast.parse(source))
    findings: list[_Finding] = []
    reported = {(finding.line, finding.code) for finding in visitor.findings}
    for line, (comment, _) in comments.items():
        findings.extend(
            _Finding(
                line=line,
                code="PAN008",
                construct="noqa",
                detail=f"`# noqa: {code}` matches no finding on this line; remove it",
            )
            for code in _noqa_reason(comment=comment)
            if code.startswith("PAN") and (line, code) not in reported
        )
    for finding in visitor.findings:
        noqa = _noqa_reason(comment=comments.get(finding.line, ("", False))[0])
        reason = noqa.get(finding.code)
        if reason is None:
            findings.append(finding)
        elif not reason:
            findings.append(
                replace(
                    finding,
                    detail=f"{finding.detail}; `# noqa: {finding.code}` needs a "
                    f"reason: `# noqa: {finding.code} - <why>`",
                )
            )
    return sorted(findings, key=lambda finding: (finding.line, finding.code))


def _comments(*, source: SourceText) -> dict[int, tuple[str, bool]]:
    """Map each line with a comment to its text and whether it stands alone."""
    lines = source.splitlines()
    comments: dict[int, tuple[str, bool]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            line, column = token.start
            comments[line] = (token.string, not lines[line - 1][:column].strip())
    return comments


def _noqa_reason(*, comment: str) -> dict[str, bool]:
    """Map each code a `# noqa:` comment names to whether it gives a reason."""
    match = _NOQA.search(comment)
    if match is None:
        return {}
    has_reason = _NOQA_REASON.fullmatch(match["rest"]) is not None
    return dict.fromkeys(re.split(r"[\s,]+", match["codes"]), has_reason)


class _AnnotationVisitor(ast.NodeVisitor):
    def __init__(self, *, standalone_comment_lines: frozenset[int]) -> None:
        self.findings: list[_Finding] = []
        self._standalone_comment_lines = standalone_comment_lines
        self._scopes: list[str] = []
        self._fallback_has_reason: bool | None = None

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

    def visit_If(self, node: ast.If) -> None:
        if _head_name(node.test) != "TYPE_CHECKING":
            self.generic_visit(node)
            return
        for statement in node.body:
            self.visit(statement)
        commented_until = 0
        for statement in node.orelse:
            line_above = statement.lineno - 1
            has_reason = (
                line_above in self._standalone_comment_lines
                or line_above == commented_until
            )
            self._fallback_has_reason = has_reason
            self.visit(statement)
            self._fallback_has_reason = None
            commented_until = statement.end_lineno if has_reason else 0

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _head_name(node.annotation) == "TypeAlias" and node.value is not None:
            self._report_alias(node=node, name=_target_name(node.target))
            self._check_alias_value(node=node.value)
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
        self._check_alias_value(node=node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        target = node.targets[0] if len(node.targets) == 1 else None
        if (
            not self._scopes
            and isinstance(target, ast.Name)
            and _is_type_name(target.id)
            and _is_implicit_alias(node.value)
        ):
            self._report_alias(node=node, name=target.id)
            self._check_alias_value(node=node.value)
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _head_name(node.func)
        if callee == "cast" and node.args:
            self._check_type(node=node.args[0], construct="cast")
            arguments = node.args[1:]
        elif callee == "TypeAliasType" and len(node.args) >= 2:
            name = node.args[0]
            self._report_alias(
                node=node,
                name=name.value
                if isinstance(name, ast.Constant) and isinstance(name.value, str)
                else None,
            )
            self._check_alias_value(node=node.args[1])
            arguments = node.args[2:]
        elif callee == "NewType" and len(node.args) >= 2:
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
        arguments = node.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            self._check_type(
                node=argument.annotation,
                construct="param",
                allow_object=node.name in _COMPARISON_DUNDERS
                or (node.name, argument.arg) in _DATA_MODEL_PARAMETERS,
            )
            if argument.annotation is not None:
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

    def _check_alias_value(self, *, node: ast.expr) -> None:
        if self._fallback_has_reason is None:
            self._check_type(node=node, construct="type-alias")
            return
        for line, code, name in _imprecise_tokens(node=node, line_offset=0):
            if code == "PAN006":
                self._add(line=line, code=code, construct="type-alias", name=name)
            elif not self._fallback_has_reason:
                self.findings.append(
                    _Finding(
                        line=line,
                        code=code,
                        construct="type-alias",
                        detail=_FALLBACK_DETAILS[code],
                    )
                )

    def _check_type(
        self, *, node: ast.expr | None, construct: str, allow_object: bool = False
    ) -> None:
        for line, code, name in _imprecise_tokens(node=node, line_offset=0):
            if not (allow_object and code == "PAN001"):
                self._add(line=line, code=code, construct=construct, name=name)

    def _add(self, *, line: int, code: str, construct: str, name: str) -> None:
        detail = (
            f"`{name}` without type arguments leaves them `Any`; spell them out"
            if code == "PAN006"
            else _DETAILS[code]
        )
        self.findings.append(
            _Finding(line=line, code=code, construct=construct, detail=detail)
        )

    def _report_alias(self, *, node: ast.stmt | ast.expr, name: str | None) -> None:
        self.findings.append(
            _Finding(
                line=node.lineno,
                code="PAN007",
                construct="type-alias",
                detail=f"write the alias as a `type {name or 'X'} = ...` statement",
            )
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
) -> Iterator[tuple[int, str, str]]:
    """Yield the line, code and spelled name of every imprecise type in `node`."""
    if node is None:
        return
    if _is_spelled(node=node, name="object", modules=_OBJECT_MODULES):
        yield node.lineno + line_offset, "PAN001", "object"
    elif _is_spelled(node=node, name="Any", modules=_ANY_MODULES):
        yield node.lineno + line_offset, "PAN002", "Any"
    elif isinstance(node, ast.Name | ast.Attribute) and _head_name(node) in _GENERICS:
        yield node.lineno + line_offset, "PAN006", str(_head_name(node))
    else:
        for child, child_offset in _type_children(node=node, line_offset=line_offset):
            yield from _imprecise_tokens(node=child, line_offset=child_offset)


def _type_children(*, node: ast.expr, line_offset: int) -> list[tuple[ast.expr, int]]:
    """Return the sub-expressions of a type expression that are types themselves.

    A string is parsed as the type it spells. A subscript contributes its
    arguments, never its head. `Literal` contributes nothing; a jaxtyping dtype,
    `PyTree` and `Annotated` contribute only their first argument, so a shape, a
    structure name or metadata is never read as a type.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parsed = _parse_type_string(node.value)
        return [] if parsed is None else [(parsed, line_offset + node.lineno - 1)]
    if isinstance(node, ast.Subscript):
        head = _head_name(node.value)
        elements = _subscript_elements(node)
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
        role, rule = "", _matching_rule(name=name, shape="scalar")
    elif isinstance(annotation, ast.Subscript) and _is_str(
        next(iter(_subscript_elements(annotation)), None)
    ):
        head = _head_name(annotation.value)
        if head in _COLLECTION_HEADS:
            role, rule = "the elements of ", _matching_rule(name=name, shape="elements")
        elif head in _MAPPING_HEADS:
            role, rule = "the keys of ", _matching_rule(name=name, shape="keys")
        else:
            return None
    else:
        return None
    if rule is None:
        return None
    target = "a domain alias" if rule.alias is None else f"`{rule.alias}`"
    verb = "with" if rule.alias is None else "as"
    return f"annotate {role}`{identifier}` {verb} {target}, not `str`"


def _matching_rule(*, name: str, shape: str) -> _NameRule | None:
    for rule in _NAME_RULES:
        pattern = getattr(rule, shape)
        if pattern is not None and pattern.fullmatch(name):
            return rule
    return None


def _subscript_elements(node: ast.Subscript) -> list[ast.expr]:
    return node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]


def _is_type_name(name: str) -> bool:
    """Return whether `name` is CapWords, as opposed to a constant or a variable."""
    bare = name.lstrip("_")
    return bare[:1].isupper() and any(character.islower() for character in bare)


def _is_implicit_alias(value: ast.expr) -> bool:
    """Return whether a module-level assignment to a type name defines an alias."""
    return (
        isinstance(value, ast.Subscript)
        or (isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr))
        or _is_spelled(node=value, name="Any", modules=_ANY_MODULES)
        or _is_spelled(node=value, name="object", modules=_OBJECT_MODULES)
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


def _is_str(node: ast.expr | None) -> bool:
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


if __name__ == "__main__":
    raise SystemExit(main(paths=[Path(argument) for argument in sys.argv[1:]]))
