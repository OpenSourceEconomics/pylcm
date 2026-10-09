import json
from pathlib import Path

import pytest

from tests.ci.precise_annotations import (
    AnnotationViolation,
    find_annotation_violations,
    main,
)

SPELLINGS = (
    pytest.param("object", ("PAN001",), id="object"),
    pytest.param("Any", ("PAN002",), id="Any"),
    pytest.param("int", (), id="precise"),
)

CONSTRUCTS = (
    pytest.param(
        "def f(*, value: {T}) -> None: ...\n", 1, "param", id="keyword-parameter"
    ),
    pytest.param(
        "def f(value: {T}, /) -> None: ...\n",
        1,
        "param",
        id="positional-only-parameter",
    ),
    pytest.param("def f(*values: {T}) -> None: ...\n", 1, "vararg", id="star-args"),
    pytest.param(
        "def f(**options: {T}) -> None: ...\n", 1, "kwarg", id="star-star-kwargs"
    ),
    pytest.param("def f() -> {T}: ...\n", 1, "return", id="return"),
    pytest.param("async def f() -> {T}: ...\n", 1, "return", id="async-return"),
    pytest.param("value: {T} = 1\n", 1, "variable", id="module-variable"),
    pytest.param(
        "def f() -> None:\n    value: {T} = 1\n", 2, "variable", id="local-variable"
    ),
    pytest.param(
        "from dataclasses import dataclass\n\n@dataclass\nclass C:\n    value: {T}\n",
        5,
        "attribute",
        id="dataclass-field",
    ),
    pytest.param("class C:\n    value: {T}\n", 2, "attribute", id="class-attribute"),
    pytest.param(
        "class C:\n    def __init__(self) -> None:\n        self.value: {T} = 1\n",
        3,
        "attribute",
        id="instance-attribute",
    ),
    pytest.param("type Alias = {T}\n", 1, "type-alias", id="type-statement"),
    pytest.param(
        "from typing import TypeAlias\n\nAlias: TypeAlias = {T}\n",
        3,
        "type-alias",
        id="typealias-annotation",
    ),
    pytest.param(
        "from typing import TypeAliasType\n\nAlias = TypeAliasType('Alias', {T})\n",
        3,
        "type-alias",
        id="typealiastype-call",
    ),
    pytest.param(
        "from typing import NewType\n\nAlias = NewType('Alias', {T})\n",
        3,
        "type-alias",
        id="newtype-call",
    ),
    pytest.param("Alias = dict[str, {T}]\n", 1, "type-alias", id="implicit-alias"),
    pytest.param(
        "_Alias = dict[str, {T}]\n", 1, "type-alias", id="private-implicit-alias"
    ),
    pytest.param(
        "from typing import Annotated\n\nAlias = Annotated[{T}, 'unit']\n",
        3,
        "type-alias",
        id="implicit-alias-of-any-generic",
    ),
    pytest.param(
        "from typing import TypeVar\n\nT = TypeVar('T', bound={T})\n",
        3,
        "type-param",
        id="typevar-bound",
    ),
    pytest.param(
        "def f[S: {T}](*, value: S) -> S: ...\n", 1, "type-param", id="pep695-bound"
    ),
    pytest.param("class C[S = {T}]:\n    pass\n", 1, "type-param", id="pep695-default"),
    pytest.param(
        "from collections.abc import Mapping\n\nclass C(Mapping[str, {T}]): ...\n",
        3,
        "base",
        id="class-base",
    ),
    pytest.param(
        "from typing import cast\n\nvalue = cast({T}, 1)\n", 3, "cast", id="cast-target"
    ),
    pytest.param(
        "from typing import cast\n\nvalue = cast('dict[str, {T}]', {})\n",
        3,
        "cast",
        id="string-cast-target",
    ),
    pytest.param(
        "def f(*, value: 'dict[str, {T}]') -> None: ...\n",
        1,
        "param",
        id="string-annotation",
    ),
    pytest.param(
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    value: {T}\n",
        4,
        "variable",
        id="type-checking-block",
    ),
    pytest.param(
        "def f(*, value: dict[str, list[{T}]]) -> None: ...\n",
        1,
        "param",
        id="nested-container",
    ),
    pytest.param(
        "def f(*, value: int | {T} | None) -> None: ...\n", 1, "param", id="union-arm"
    ),
    pytest.param(
        "from collections.abc import Callable\n\n"
        "def f(*, func: Callable[..., {T}]) -> None: ...\n",
        3,
        "param",
        id="callable-return",
    ),
    pytest.param(
        "from collections.abc import Callable\n\n"
        "def f(*, func: Callable[[{T}], int]) -> None: ...\n",
        3,
        "param",
        id="callable-argument",
    ),
    pytest.param(
        "from typing import Annotated\n\n"
        "def f(*, value: Annotated[{T}, 'unit']) -> None: ...\n",
        3,
        "param",
        id="annotated-type",
    ),
    pytest.param(
        "from jaxtyping import Float\n\n"
        "def f(*, value: Float[{T}, 'n']) -> None: ...\n",
        3,
        "param",
        id="jaxtyping-array-type",
    ),
    pytest.param("def f(*, value: type[{T}]) -> None: ...\n", 1, "param", id="type-of"),
    pytest.param(
        "def f(\n    *,\n    value: dict[\n"
        "        str,\n        {T},\n    ],\n) -> None: ...\n",
        5,
        "param",
        id="multi-line-annotation",
    ),
)

QUALIFIED_SPELLINGS = (
    pytest.param(
        "import typing\n\ndef f(*, value: typing.Any) -> None: ...\n",
        [(3, "PAN002", "param")],
        id="typing-any",
    ),
    pytest.param(
        "import typing_extensions\n\nvalue: typing_extensions.Any\n",
        [(3, "PAN002", "variable")],
        id="typing-extensions-any",
    ),
    pytest.param(
        "import builtins\n\ndef f(*, value: builtins.object) -> None: ...\n",
        [(3, "PAN001", "param")],
        id="builtins-object",
    ),
)

IGNORED_POSITIONS = (
    pytest.param("def f(*, value: int = object()) -> None: ...\n", id="default-value"),
    pytest.param("flag = isinstance(1, object)\n", id="isinstance-argument"),
    pytest.param(
        "from typing import Literal\n\n"
        "def f(*, mode: Literal['object', 'Any']) -> None: ...\n",
        id="literal-values",
    ),
    pytest.param(
        "from typing import Annotated\n\n"
        "def f(*, value: Annotated[int, object]) -> None: ...\n",
        id="annotated-metadata",
    ),
    pytest.param(
        "from jax import Array\nfrom jaxtyping import Float\n\n"
        "def f(*, value: Float[Array, 'object Any']) -> None: ...\n",
        id="jaxtyping-shape",
    ),
    pytest.param(
        "from jaxtyping import PyTree\n\n"
        "def f(*, tree: PyTree[int, 'object']) -> None: ...\n",
        id="pytree-structure-name",
    ),
    pytest.param('def f() -> None:\n    """value: object"""\n', id="docstring"),
    pytest.param("# value: object\nvalue = 1\n", id="comment"),
    pytest.param("from typing import Any\n", id="import"),
    pytest.param(
        "table = {'object': 1}\nentry = table['object']\n", id="runtime-subscript"
    ),
    pytest.param("registry = {}\nregistry[object] = 1\n", id="runtime-key"),
    pytest.param(
        "def f(*, value: int) -> None:\n    print(object, value)\n",
        id="runtime-reference",
    ),
    pytest.param(
        '"""Show the marker.\n\n# annotation-exempt: heterogeneous=leaf\n"""\n'
        "value: int\n",
        id="marker-text-in-docstring",
    ),
)

COMPARISON_DUNDERS = (
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__contains__",
)

DUNDER_NEAR_MISSES = (
    pytest.param(
        "class C:\n    def __init__(self, *, other: object) -> None: ...\n",
        [(2, "PAN001", "param")],
        id="init-parameter",
    ),
    pytest.param(
        "class C:\n    def __eq__(self, other: object) -> object: ...\n",
        [(2, "PAN001", "return")],
        id="dunder-return",
    ),
    pytest.param(
        "from typing import Any\n\n"
        "class C:\n    def __eq__(self, other: Any) -> bool: ...\n",
        [(4, "PAN002", "param")],
        id="dunder-any",
    ),
)

NARROWED_PARAMETERS = (
    pytest.param(
        "def f(*, value: object) -> int:\n"
        "    if isinstance(value, int):\n"
        "        return value\n"
        "    raise TypeError\n",
        id="isinstance",
    ),
    pytest.param(
        "def f(*, value: object) -> bool:\n    return issubclass(value, int)\n",
        id="issubclass",
    ),
    pytest.param(
        "def f(*, value: object) -> int:\n"
        "    match value:\n"
        "        case int():\n"
        "            return value\n"
        "    return 0\n",
        id="match",
    ),
    pytest.param(
        "def f(*, value: object | None) -> int:\n"
        "    if isinstance(value, int):\n"
        "        return value\n"
        "    return 0\n",
        id="optional",
    ),
)

UNNARROWED_PARAMETERS = (
    pytest.param(
        "def f(*, value: object, other: object) -> bool:\n"
        "    return isinstance(other, int)\n",
        [(1, "PAN001", "param")],
        id="narrows-another-parameter",
    ),
    pytest.param(
        "def f(*, values: list[object]) -> bool:\n"
        "    return isinstance(values, list)\n",
        [(1, "PAN001", "param")],
        id="container-of-object",
    ),
    pytest.param(
        "from typing import Any\n\n"
        "def f(*, value: Any) -> bool:\n"
        "    return isinstance(value, int)\n",
        [(3, "PAN002", "param")],
        id="narrowed-any",
    ),
    pytest.param(
        "def f(*, value: object) -> None:\n    check(value=value)\n",
        [(1, "PAN001", "param")],
        id="delegated-to-a-helper",
    ),
)

LABEL_NAMES = (
    pytest.param(
        "regime_name",
        "str",
        "annotate `regime_name` as `RegimeName`, not `str`",
        id="regime-name",
    ),
    pytest.param(
        "source_regime",
        "str | None",
        "annotate `source_regime` as `RegimeName`, not `str`",
        id="source-regime-optional",
    ),
    pytest.param(
        "regime_name",
        "'str'",
        "annotate `regime_name` as `RegimeName`, not `str`",
        id="quoted-str",
    ),
    pytest.param(
        "state_name",
        "str",
        "annotate `state_name` as `StateName`, not `str`",
        id="state-name",
    ),
    pytest.param(
        "liquid_state",
        "str",
        "annotate `liquid_state` as `StateName`, not `str`",
        id="qualified-state",
    ),
    pytest.param(
        "action_name",
        "str",
        "annotate `action_name` as `ActionName`, not `str`",
        id="action-name",
    ),
    pytest.param(
        "outer_action",
        "str",
        "annotate `outer_action` as `ActionName`, not `str`",
        id="qualified-action",
    ),
    pytest.param(
        "func_name",
        "str",
        "annotate `func_name` as `FunctionName`, not `str`",
        id="func-name",
    ),
    pytest.param(
        "scale_function",
        "str",
        "annotate `scale_function` as `FunctionName`, not `str`",
        id="qualified-function",
    ),
    pytest.param(
        "regime_names",
        "tuple[str, ...]",
        "annotate the elements of `regime_names` as `RegimeName`, not `str`",
        id="regime-names",
    ),
    pytest.param(
        "state_names",
        "frozenset[str]",
        "annotate the elements of `state_names` as `StateName`, not `str`",
        id="state-names",
    ),
    pytest.param(
        "discrete_action_names",
        "list[str]",
        "annotate the elements of `discrete_action_names` as `ActionName`, not `str`",
        id="action-names",
    ),
    pytest.param(
        "function_names",
        "set[str]",
        "annotate the elements of `function_names` as `FunctionName`, not `str`",
        id="function-names",
    ),
    pytest.param(
        "state_names",
        "Sequence[str] | None",
        "annotate the elements of `state_names` as `StateName`, not `str`",
        id="optional-sequence",
    ),
    pytest.param(
        "regime_to_codes",
        "dict[str, int]",
        "annotate the keys of `regime_to_codes` as `RegimeName`, not `str`",
        id="regime-to-mapping",
    ),
    pytest.param(
        "V_by_regime",
        "Mapping[str, float]",
        "annotate the keys of `V_by_regime` as `RegimeName`, not `str`",
        id="by-regime-mapping",
    ),
)

LABEL_NAME_NEAR_MISSES = (
    pytest.param(
        "def f(*, post_decision_state: str) -> None: ...\n", id="post-decision-state"
    ),
    pytest.param(
        "def f(*, outer_post_decision_state: str) -> None: ...\n",
        id="qualified-post-decision-state",
    ),
    pytest.param("def f(*, next_state_name: str) -> None: ...\n", id="next-state"),
    pytest.param("def f(*, transaction: str) -> None: ...\n", id="embedded-word"),
    pytest.param(
        "def f(*, regime_name_suffix: str) -> None: ...\n", id="trailing-word"
    ),
    pytest.param("def f(*, regimes: str) -> None: ...\n", id="plural-without-names"),
    pytest.param("def f(*, label: str) -> None: ...\n", id="label"),
    pytest.param("def f(*, name: str) -> None: ...\n", id="bare-name"),
    pytest.param("def f(*, target: str) -> None: ...\n", id="bare-target"),
    pytest.param("def f(*, source: str) -> None: ...\n", id="bare-source"),
    pytest.param("def f(*, param_name: str) -> None: ...\n", id="parameter-name"),
    pytest.param(
        "def f(*, regime_name: RegimeName) -> None: ...\n", id="already-aliased"
    ),
    pytest.param("def f(*, regime_name: int) -> None: ...\n", id="not-a-string"),
    pytest.param(
        "def f(*, state_names: tuple[StateName, ...]) -> None: ...\n",
        id="aliased-elements",
    ),
    pytest.param("def f(*, regime_names: str) -> None: ...\n", id="plural-scalar"),
    pytest.param(
        "def f(*, regime_to_codes: dict[RegimeName, int]) -> None: ...\n",
        id="aliased-keys",
    ),
    pytest.param(
        "def f(*, regime_to_codes: dict[int, str]) -> None: ...\n", id="str-values"
    ),
    pytest.param("def regime_name() -> str: ...\n", id="return-annotation"),
)

LABEL_NAME_POSITIONS = (
    pytest.param(
        "def f(*, regime_name: str) -> None: ...\n", 1, "param", id="parameter"
    ),
    pytest.param("class C:\n    regime_name: str\n", 2, "attribute", id="attribute"),
    pytest.param(
        "class C:\n    def __init__(self) -> None:\n"
        "        self.regime_name: str = 'a'\n",
        3,
        "attribute",
        id="instance-attribute",
    ),
    pytest.param(
        "def f() -> None:\n    regime_name: str = 'a'\n",
        2,
        "variable",
        id="local-variable",
    ),
    pytest.param("regime_name: str = 'a'\n", 1, "variable", id="module-variable"),
)

WELLFORMED_EXEMPTIONS = (
    pytest.param(
        "# annotation-exempt: heterogeneous=pytree-leaf\nvalue: object\n",
        id="heterogeneous",
    ),
    pytest.param(
        "# annotation-exempt: library-signature=jax.tree_util.register_pytree_node\n"
        "value: object\n",
        id="library-signature",
    ),
    pytest.param(
        "# annotation-exempt: import-cycle=ContinuationPlan\nvalue: object\n",
        id="import-cycle",
    ),
    pytest.param(
        "from typing import Any\n\n"
        "# annotation-exempt: heterogeneous=pair\nvalue: tuple[object, Any]\n",
        id="every-finding-on-the-next-line",
    ),
    pytest.param(
        "class C:\n    # annotation-exempt: heterogeneous=leaf\n    value: object\n",
        id="indented",
    ),
)

MALFORMED_EXEMPTIONS = (
    pytest.param(
        "# annotation-exempt:\nvalue: object\n",
        [(1, "PAN004", "exemption"), (2, "PAN001", "variable")],
        id="no-reason",
    ),
    pytest.param(
        "# annotation-exempt: because it is needed\nvalue: object\n",
        [(1, "PAN004", "exemption"), (2, "PAN001", "variable")],
        id="free-text",
    ),
    pytest.param(
        "# annotation-exempt: heterogeneous=\nvalue: object\n",
        [(1, "PAN004", "exemption"), (2, "PAN001", "variable")],
        id="empty-slug",
    ),
    pytest.param(
        "# annotation-exempt: import-cycle=continuation_plan\nvalue: object\n",
        [(1, "PAN004", "exemption"), (2, "PAN001", "variable")],
        id="lowercase-type-name",
    ),
    pytest.param(
        "value: object  # annotation-exempt: heterogeneous=leaf\n",
        [(1, "PAN001", "variable"), (1, "PAN004", "exemption")],
        id="trailing-marker",
    ),
)

STALE_EXEMPTIONS = (
    pytest.param(
        "# annotation-exempt: heterogeneous=leaf\nvalue: int\n",
        [(1, "PAN005", "exemption")],
        id="precise-line-below",
    ),
    pytest.param(
        "# annotation-exempt: heterogeneous=leaf\n\nvalue: object\n",
        [(1, "PAN005", "exemption"), (3, "PAN001", "variable")],
        id="two-lines-above",
    ),
    pytest.param(
        "value: int\n# annotation-exempt: heterogeneous=leaf\n",
        [(2, "PAN005", "exemption")],
        id="end-of-file",
    ),
    pytest.param(
        "class C:\n"
        "    # annotation-exempt: heterogeneous=leaf\n"
        "    def __eq__(self, other: object) -> bool: ...\n",
        [(2, "PAN005", "exemption")],
        id="above-an-allowed-placement",
    ),
)


def test_find_annotation_violations_fires_beside_ignored_positions(
    tmp_path: Path,
) -> None:
    """One seeded `object` annotation is the only finding among ignored positions."""
    ignored = "".join(str(case.values[0]) for case in IGNORED_POSITIONS)
    seeded_line = ignored.count("\n") + 1
    source = _write(tmp_path=tmp_path, text=f"{ignored}seeded: object\n")

    assert _summarize(find_annotation_violations(paths=[source])) == [
        (seeded_line, "PAN001", "variable")
    ]


@pytest.mark.parametrize(("template", "line", "construct"), CONSTRUCTS)
@pytest.mark.parametrize(("spelling", "codes"), SPELLINGS)
def test_find_annotation_violations_reports_object_and_any_in_every_construct(
    *,
    tmp_path: Path,
    template: str,
    line: int,
    construct: str,
    spelling: str,
    codes: tuple[str, ...],
) -> None:
    """`object` and `Any` are reported in each annotation construct; `int` is not."""
    source = _write(tmp_path=tmp_path, text=template.replace("{T}", spelling))

    assert _summarize(find_annotation_violations(paths=[source])) == [
        (line, code, construct) for code in codes
    ]


@pytest.mark.parametrize(("text", "expected"), QUALIFIED_SPELLINGS)
def test_find_annotation_violations_reports_qualified_spellings(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """Module-qualified `typing.Any` and `builtins.object` count like the bare names."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


@pytest.mark.parametrize("text", IGNORED_POSITIONS)
def test_find_annotation_violations_ignores_non_annotation_positions(
    *, tmp_path: Path, text: str
) -> None:
    """Defaults, literals, metadata, shapes, strings and runtime code pass."""
    source = _write(tmp_path=tmp_path, text=text)

    assert find_annotation_violations(paths=[source]) == ()


@pytest.mark.parametrize("dunder", COMPARISON_DUNDERS)
def test_find_annotation_violations_allows_object_in_comparison_dunders(
    *, tmp_path: Path, dunder: str
) -> None:
    """A comparison or containment dunder may take `other: object`."""
    source = _write(
        tmp_path=tmp_path,
        text=f"class C:\n    def {dunder}(self, other: object) -> bool: ...\n",
    )

    assert find_annotation_violations(paths=[source]) == ()


@pytest.mark.parametrize(("text", "expected"), DUNDER_NEAR_MISSES)
def test_find_annotation_violations_reports_object_outside_dunder_parameters(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """The dunder allowance covers `object` parameters of comparison dunders only."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


@pytest.mark.parametrize("text", NARROWED_PARAMETERS)
def test_find_annotation_violations_allows_object_parameters_narrowed_in_the_body(
    *, tmp_path: Path, text: str
) -> None:
    """An `object` parameter that its own function narrows needs no marker."""
    source = _write(tmp_path=tmp_path, text=text)

    assert find_annotation_violations(paths=[source]) == ()


@pytest.mark.parametrize(("text", "expected"), UNNARROWED_PARAMETERS)
def test_find_annotation_violations_reports_object_parameters_not_narrowed(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """The narrowing allowance needs a bare `object` narrowed in the same body."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


@pytest.mark.parametrize(("identifier", "annotation", "detail"), LABEL_NAMES)
def test_find_annotation_violations_flags_bare_str_label_names(
    *, tmp_path: Path, identifier: str, annotation: str, detail: str
) -> None:
    """A `str` on a regime, state, action or function name points to its alias."""
    source = _write(
        tmp_path=tmp_path, text=f"def f(*, {identifier}: {annotation}) -> None: ...\n"
    )

    assert [
        (violation.line, violation.code, violation.detail)
        for violation in find_annotation_violations(paths=[source])
    ] == [(1, "PAN003", detail)]


@pytest.mark.parametrize("text", LABEL_NAME_NEAR_MISSES)
def test_find_annotation_violations_ignores_str_on_other_names(
    *, tmp_path: Path, text: str
) -> None:
    """Names outside the four label patterns, aliased names and returns pass."""
    source = _write(tmp_path=tmp_path, text=text)

    assert find_annotation_violations(paths=[source]) == ()


@pytest.mark.parametrize(("text", "line", "construct"), LABEL_NAME_POSITIONS)
def test_find_annotation_violations_flags_bare_str_in_every_named_position(
    *, tmp_path: Path, text: str, line: int, construct: str
) -> None:
    """The label rule covers parameters, attributes and variables alike."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == [
        (line, "PAN003", construct)
    ]


@pytest.mark.parametrize("text", WELLFORMED_EXEMPTIONS)
def test_find_annotation_violations_honours_wellformed_exemptions(
    *, tmp_path: Path, text: str
) -> None:
    """A marker on its own line exempts every finding on the line below it."""
    source = _write(tmp_path=tmp_path, text=text)

    assert find_annotation_violations(paths=[source]) == ()


@pytest.mark.parametrize(("text", "expected"), MALFORMED_EXEMPTIONS)
def test_find_annotation_violations_rejects_malformed_exemptions(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """A marker without a recognized reason, or beside the code, exempts nothing."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


@pytest.mark.parametrize(("text", "expected"), STALE_EXEMPTIONS)
def test_find_annotation_violations_reports_stale_exemptions(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """A marker whose next line has no finding is reported."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


def test_find_annotation_violations_ignores_files_that_are_not_python_sources(
    tmp_path: Path,
) -> None:
    """Only `.py` files are read."""
    source = tmp_path / "page.md"
    source.write_text("```python\nvalue: object\n```\n")

    assert find_annotation_violations(paths=[source]) == ()


def test_main_passes_when_each_count_equals_the_baseline(tmp_path: Path) -> None:
    """Counts equal to the baseline pass and leave the baseline unchanged."""
    source = _write(tmp_path=tmp_path, text="value: object\n")
    baseline = _write_baseline(
        tmp_path=tmp_path, counts={source.as_posix(): {"PAN001": 1}}
    )

    exit_code = main(paths=[source], baseline_path=baseline)

    assert (exit_code, _read_json(baseline)) == (
        0,
        {source.as_posix(): {"PAN001": 1}},
    )


def test_main_fails_when_a_count_rises_above_the_baseline(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A count above the baseline fails and lists that rule's findings in the file."""
    source = _write(tmp_path=tmp_path, text="first: object\nsecond: object\n")
    baseline = _write_baseline(
        tmp_path=tmp_path, counts={source.as_posix(): {"PAN001": 1}}
    )

    exit_code = main(paths=[source], baseline_path=baseline)

    detail = "`object` in an annotation; name the precise type"
    assert (exit_code, capsys.readouterr().out) == (
        1,
        (
            f"{source}:1: PAN001 variable: {detail}\n"
            f"{source}:2: PAN001 variable: {detail}\n"
            f"{source.as_posix()}: PAN001 count 2 exceeds the baseline 1\n"
        ),
    )


def test_main_lowers_the_baseline_when_a_count_falls(tmp_path: Path) -> None:
    """A count below the baseline rewrites the baseline down and fails once."""
    source = _write(tmp_path=tmp_path, text="value: int\n")
    baseline = _write_baseline(
        tmp_path=tmp_path, counts={source.as_posix(): {"PAN001": 1}}
    )

    exit_code = main(paths=[source], baseline_path=baseline)

    assert (exit_code, _read_json(baseline)) == (1, {})


def test_main_fails_on_findings_in_a_file_the_baseline_does_not_list(
    tmp_path: Path,
) -> None:
    """A file missing from the baseline is held to zero findings."""
    source = _write(tmp_path=tmp_path, text="value: object\n")
    baseline = _write_baseline(tmp_path=tmp_path, counts={})

    assert main(paths=[source], baseline_path=baseline) == 1


def test_main_drops_baseline_entries_for_files_that_no_longer_exist(
    tmp_path: Path,
) -> None:
    """An entry for a deleted file is removed, and the run fails once."""
    source = _write(tmp_path=tmp_path, text="value: int\n")
    deleted = (tmp_path / "deleted.py").as_posix()
    baseline = _write_baseline(tmp_path=tmp_path, counts={deleted: {"PAN001": 2}})

    exit_code = main(paths=[source], baseline_path=baseline)

    assert (exit_code, _read_json(baseline)) == (1, {})


def test_main_never_baselines_exemption_problems(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed marker fails even when the file's counts match the baseline."""
    source = _write(tmp_path=tmp_path, text="# annotation-exempt:\nvalue: object\n")
    baseline = _write_baseline(
        tmp_path=tmp_path, counts={source.as_posix(): {"PAN001": 1}}
    )

    exit_code = main(paths=[source], baseline_path=baseline)

    assert (exit_code, capsys.readouterr().out.splitlines()[0]) == (
        1,
        (
            f"{source}:1: PAN004 exemption: malformed exemption; write "
            "`# annotation-exempt: heterogeneous=<slug>`, "
            "`library-signature=<dotted.name>` or `import-cycle=<TypeName>` "
            "on its own line above the finding"
        ),
    )


def test_main_writes_the_baseline_from_current_counts(tmp_path: Path) -> None:
    """`write_baseline` records each rule's count per file, two per line if two."""
    source = _write(
        tmp_path=tmp_path,
        text="pair: tuple[object, object]\nvalue: Any\nregime_name: str\n",
    )
    baseline = _write_baseline(tmp_path=tmp_path, counts={})

    exit_code = main(paths=[source], baseline_path=baseline, write_baseline=True)

    assert (exit_code, _read_json(baseline)) == (
        0,
        {source.as_posix(): {"PAN001": 2, "PAN002": 1, "PAN003": 1}},
    )


def _write(*, tmp_path: Path, text: str) -> Path:
    source = tmp_path / "module.py"
    source.write_text(text)
    return source


def _write_baseline(*, tmp_path: Path, counts: dict[str, dict[str, int]]) -> Path:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(counts))
    return baseline


def _read_json(path: Path) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text())


def _summarize(
    violations: tuple[AnnotationViolation, ...],
) -> list[tuple[int, str, str]]:
    return [
        (violation.line, violation.code, violation.construct)
        for violation in violations
    ]
