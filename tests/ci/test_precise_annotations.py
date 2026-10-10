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
        "from typing import NewType\n\nAlias = NewType('Alias', {T})\n",
        3,
        "type-alias",
        id="newtype-call",
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

ALIAS_CONSTRUCTS = (
    pytest.param(
        "from typing import TypeAlias\n\nAlias: TypeAlias = {T}\n",
        3,
        id="typealias-annotation",
    ),
    pytest.param(
        "from typing import TypeAliasType\n\nAlias = TypeAliasType('Alias', {T})\n",
        3,
        id="typealiastype-call",
    ),
    pytest.param("Alias = dict[str, {T}]\n", 1, id="implicit-alias"),
    pytest.param("_Alias = dict[str, {T}]\n", 1, id="private-implicit-alias"),
    pytest.param("Alias = int | {T}\n", 1, id="implicit-union-alias"),
    pytest.param(
        "from typing import Annotated\n\nAlias = Annotated[{T}, 'unit']\n",
        3,
        id="implicit-alias-of-any-generic",
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
    pytest.param("def f(*, value: int = dict()) -> None: ...\n", id="generic-default"),
    pytest.param("flag = isinstance(1, object)\n", id="isinstance-argument"),
    pytest.param("flag = isinstance({}, dict)\n", id="isinstance-generic"),
    pytest.param(
        "from typing import Literal\n\n"
        "def f(*, mode: Literal['object', 'Any', 'dict']) -> None: ...\n",
        id="literal-values",
    ),
    pytest.param(
        "from typing import Annotated\n\n"
        "def f(*, value: Annotated[int, object, dict]) -> None: ...\n",
        id="annotated-metadata",
    ),
    pytest.param(
        "from jax import Array\nfrom jaxtyping import Float\n\n"
        "def f(*, value: Float[Array, 'object Any dict']) -> None: ...\n",
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

DATA_MODEL_PARAMETERS = (
    pytest.param(
        "class C:\n    def __setattr__(self, name: str, value: object) -> None: ...\n",
        id="setattr-value",
    ),
    pytest.param(
        "class C:\n    def __deepcopy__(self, memo: dict[int, object]) -> 'C': ...\n",
        id="deepcopy-memo",
    ),
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
    pytest.param(
        "class C:\n    def __setattr__(self, name: object, value: int) -> None: ...\n",
        [(2, "PAN001", "param")],
        id="setattr-name",
    ),
    pytest.param(
        "from typing import Any\n\n"
        "class C:\n    def __deepcopy__(self, memo: dict[int, Any]) -> 'C': ...\n",
        [(4, "PAN002", "param")],
        id="deepcopy-any-memo",
    ),
)

NARROWED_PARAMETERS = (
    pytest.param(
        "def f(*, value: object) -> int:\n"
        "    if isinstance(value, int):\n"
        "        return value\n"
        "    raise TypeError\n",
        [(1, "PAN001", "param")],
        id="isinstance",
    ),
    pytest.param(
        "def f(*, value: object) -> bool:\n    return issubclass(value, int)\n",
        [(1, "PAN001", "param")],
        id="issubclass",
    ),
    pytest.param(
        "def f(*, value: object) -> int:\n"
        "    match value:\n"
        "        case int():\n"
        "            return value\n"
        "    return 0\n",
        [(1, "PAN001", "param")],
        id="match",
    ),
    pytest.param(
        "def f(*, value: object | None) -> int:\n"
        "    if isinstance(value, int):\n"
        "        return value\n"
        "    return 0\n",
        [(1, "PAN001", "param")],
        id="optional",
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
        "qname",
        "str",
        "annotate `qname` as `QualifiedName`, not `str`",
        id="qname",
    ),
    pytest.param(
        "utility_qualified_name",
        "str",
        "annotate `utility_qualified_name` as `QualifiedName`, not `str`",
        id="qualified-name",
    ),
    pytest.param(
        "ce_flat_param_name",
        "str",
        "annotate `ce_flat_param_name` as `QualifiedName`, not `str`",
        id="flat-param-name",
    ),
    pytest.param(
        "param_name",
        "str",
        "annotate `param_name` as `ParameterName`, not `str`",
        id="param-name",
    ),
    pytest.param(
        "threshold_parameter_name",
        "str",
        "annotate `threshold_parameter_name` as `ParameterName`, not `str`",
        id="parameter-name",
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
        "flat_param_names",
        "tuple[str, ...]",
        "annotate the elements of `flat_param_names` as `QualifiedName`, not `str`",
        id="flat-param-names",
    ),
    pytest.param(
        "qnames",
        "list[str]",
        "annotate the elements of `qnames` as `QualifiedName`, not `str`",
        id="qnames",
    ),
    pytest.param(
        "discount_param_names",
        "tuple[str, ...]",
        "annotate the elements of `discount_param_names` as `ParameterName`, not `str`",
        id="param-names",
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

DOMAIN_NAMES = (
    pytest.param(
        "target",
        "str",
        "annotate `target` with a domain alias, not `str`",
        id="target",
    ),
    pytest.param(
        "source",
        "str | None",
        "annotate `source` with a domain alias, not `str`",
        id="source-optional",
    ),
    pytest.param(
        "budget_target",
        "str",
        "annotate `budget_target` with a domain alias, not `str`",
        id="qualified-target",
    ),
    pytest.param(
        "arg_name",
        "str",
        "annotate `arg_name` with a domain alias, not `str`",
        id="arg-name",
    ),
    pytest.param(
        "arg",
        "str",
        "annotate `arg` with a domain alias, not `str`",
        id="arg",
    ),
    pytest.param(
        "targets",
        "frozenset[str]",
        "annotate the elements of `targets` with a domain alias, not `str`",
        id="targets",
    ),
    pytest.param(
        "subject_arg_names",
        "tuple[str, ...]",
        "annotate the elements of `subject_arg_names` with a domain alias, not `str`",
        id="arg-names",
    ),
    pytest.param(
        "other_args",
        "list[str]",
        "annotate the elements of `other_args` with a domain alias, not `str`",
        id="args",
    ),
    pytest.param(
        "sources",
        "Mapping[str, int]",
        "annotate the keys of `sources` with a domain alias, not `str`",
        id="sources-mapping",
    ),
    pytest.param(
        "cells_by_target",
        "dict[str, int]",
        "annotate the keys of `cells_by_target` with a domain alias, not `str`",
        id="by-target-mapping",
    ),
    pytest.param(
        "arg_to_index",
        "dict[str, int]",
        "annotate the keys of `arg_to_index` with a domain alias, not `str`",
        id="arg-to-mapping",
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
    pytest.param("def f(*, resource: str) -> None: ...\n", id="word-ending-in-source"),
    pytest.param("def f(*, argument: str) -> None: ...\n", id="word-starting-with-arg"),
    pytest.param("def f(*, kwargs: dict[str, int]) -> None: ...\n", id="kwargs"),
    pytest.param(
        "def f(*, signature_parameter_names: tuple[str, ...]) -> None: ...\n",
        id="signature-parameter-names",
    ),
    pytest.param(
        "def f(*, regime_name: RegimeName) -> None: ...\n", id="already-aliased"
    ),
    pytest.param("def f(*, target: RegimeName) -> None: ...\n", id="aliased-target"),
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

BARE_GENERICS = (
    pytest.param(
        "from collections.abc import Callable\n\n"
        "def f(*, func: Callable) -> None: ...\n",
        [(3, "PAN006", "param")],
        id="callable",
    ),
    pytest.param("value: dict = {}\n", [(1, "PAN006", "variable")], id="dict"),
    pytest.param("def f() -> tuple: ...\n", [(1, "PAN006", "return")], id="tuple"),
    pytest.param(
        "def f(*, cls: type) -> None: ...\n", [(1, "PAN006", "param")], id="type"
    ),
    pytest.param(
        "from collections.abc import Callable\n\n"
        "def f(*, funcs: dict[str, Callable]) -> None: ...\n",
        [(3, "PAN006", "param")],
        id="nested",
    ),
    pytest.param(
        "import collections.abc\n\n"
        "def f(*, func: collections.abc.Callable) -> None: ...\n",
        [(3, "PAN006", "param")],
        id="qualified",
    ),
    pytest.param(
        "from types import MappingProxyType\n\n"
        "def f(*, table: 'MappingProxyType | None') -> None: ...\n",
        [(3, "PAN006", "param")],
        id="string-union-arm",
    ),
    pytest.param(
        "from collections.abc import Set as AbstractSet\n\n"
        "def f(*, names: AbstractSet) -> None: ...\n",
        [(3, "PAN006", "param")],
        id="renamed-abstract-set",
    ),
    pytest.param(
        "from typing import Annotated\n\n"
        "def f(*, value: Annotated[list, 'unit']) -> None: ...\n",
        [(3, "PAN006", "param")],
        id="annotated-type",
    ),
    pytest.param(
        "from typing import cast\n\nvalue = cast(dict, {})\n",
        [(3, "PAN006", "cast")],
        id="cast-target",
    ),
    pytest.param(
        "from collections.abc import Mapping\n\ntype Table = Mapping\n",
        [(3, "PAN006", "type-alias")],
        id="type-statement",
    ),
)

PARAMETERIZED_GENERICS = (
    pytest.param("def f(*, value: dict[str, int]) -> None: ...\n", id="dict"),
    pytest.param("def f(*, cls: type[int]) -> None: ...\n", id="type"),
    pytest.param("def f(*, value: tuple[()]) -> None: ...\n", id="empty-tuple"),
    pytest.param(
        "from collections.abc import Callable\n\n"
        "def f(*, func: Callable[..., None]) -> None: ...\n",
        id="callable",
    ),
    pytest.param("def f(*, path: Path) -> None: ...\n", id="not-generic"),
)

TYPE_STATEMENT_NEAR_MISSES = (
    pytest.param("type Alias = dict[str, int]\n", id="type-statement"),
    pytest.param("TABLE = {}\nENTRY = TABLE['x']\n", id="constant-subscript"),
    pytest.param(
        "LEFT = frozenset()\nRIGHT = frozenset()\nBOTH = LEFT | RIGHT\n",
        id="constant-union",
    ),
    pytest.param("table = {}\nentry = table['x']\n", id="lowercase-subscript"),
    pytest.param("Alias = Model\n", id="class-rebinding"),
    pytest.param(
        "from typing import NewType\n\nAlias = NewType('Alias', int)\n", id="newtype"
    ),
    pytest.param("from typing import TypeVar\n\nT = TypeVar('T')\n", id="typevar"),
    pytest.param("class C:\n    Alias = dict[str, int]\n", id="class-scope"),
    pytest.param("def f() -> None:\n    Alias = dict[str, int]\n", id="function-scope"),
)

RUNTIME_FALLBACKS = (
    pytest.param(
        "from typing import TYPE_CHECKING, Any\n\n"
        "if TYPE_CHECKING:\n"
        "    from plans import Plan\n"
        "else:\n"
        "    # The planner imports this module.\n"
        "    type Plan = Any\n",
        [],
        id="commented-any",
    ),
    pytest.param(
        "import typing\n\n"
        "if typing.TYPE_CHECKING:\n"
        "    from stores import Store\n"
        "else:\n"
        "    # Stores validate their own contents.\n"
        "    type Store = object\n",
        [],
        id="commented-object-qualified-guard",
    ),
    pytest.param(
        "from typing import TYPE_CHECKING, Any\n\n"
        "if TYPE_CHECKING:\n"
        "    from plans import Plan, Step\n"
        "else:\n"
        "    # The planner imports this module, and both types live there.\n"
        "    type Plan = Any\n"
        "    type Step = Any\n",
        [],
        id="comment-above-a-run-of-bindings",
    ),
    pytest.param(
        "from typing import TYPE_CHECKING, Any\n\n"
        "if TYPE_CHECKING:\n"
        "    from plans import Plan\n"
        "else:\n"
        "    # The planner imports this module.\n"
        "    Plan = Any\n",
        [(7, "PAN007", "type-alias")],
        id="commented-plain-assignment",
    ),
    pytest.param(
        "from typing import TYPE_CHECKING, Any\n\n"
        "if TYPE_CHECKING:\n"
        "    from plans import Plan\n"
        "else:\n"
        "    type Plan = Any\n",
        [(6, "PAN002", "type-alias")],
        id="uncommented",
    ),
    pytest.param(
        "from typing import TYPE_CHECKING, Any\n\n"
        "if TYPE_CHECKING:\n"
        "    from plans import Plan\n"
        "else:\n"
        "    # The planner imports this module.\n"
        "\n"
        "    type Plan = Any\n",
        [(8, "PAN002", "type-alias")],
        id="blank-line-after-comment",
    ),
    pytest.param(
        "from typing import TYPE_CHECKING, Any\n\n"
        "if TYPE_CHECKING:\n"
        "    from plans import Plan, Step\n"
        "else:\n"
        "    # The planner imports this module.\n"
        "    type Plan = Any\n"
        "\n"
        "    type Step = Any\n",
        [(9, "PAN002", "type-alias")],
        id="blank-line-ends-the-run",
    ),
    pytest.param(
        "from typing import TYPE_CHECKING, Any\n\n"
        "if TYPE_CHECKING:\n"
        "    # Static checking only.\n"
        "    type Plan = Any\n",
        [(5, "PAN002", "type-alias")],
        id="wide-in-the-static-branch",
    ),
    pytest.param(
        "from typing import Any\n\n"
        "if flag:\n"
        "    pass\n"
        "else:\n"
        "    # Some reason.\n"
        "    type Plan = Any\n",
        [(7, "PAN002", "type-alias")],
        id="else-of-another-condition",
    ),
)

NOQA_SUPPRESSIONS = (
    pytest.param(
        "value: object  # noqa: PAN001 - hashes arbitrary objects\n", id="one-code"
    ),
    pytest.param(
        "from typing import Any\n\n"
        "value: tuple[object, Any]  # noqa: PAN001, PAN002 - a pair of anything\n",
        id="two-codes",
    ),
    pytest.param(
        "def f(*, value: object) -> None:  # noqa: ARG001, PAN001 - unused hook\n"
        "    pass\n",
        id="beside-another-linter-code",
    ),
    pytest.param("value: object  # noqa: PAN001 -- anything\n", id="double-dash"),
    pytest.param("value: object  # noqa: PAN001 \u2014 anything\n", id="em-dash"),
    pytest.param(
        "def f(\n    *,\n    value: dict[\n"
        "        str,\n        object,  # noqa: PAN001 - payloads vary\n"
        "    ],\n) -> None: ...\n",
        id="on-the-finding-line-of-a-multi-line-annotation",
    ),
)

NOQA_NON_SUPPRESSIONS = (
    pytest.param(
        "value: object  # noqa: PAN002 - another code\n",
        [(1, "PAN001", "variable"), (1, "PAN008", "noqa")],
        id="another-code",
    ),
    pytest.param(
        "value: object  # noqa\n", [(1, "PAN001", "variable")], id="blanket-noqa"
    ),
    pytest.param(
        "value: object  # noqa: PAN001 -\n",
        [(1, "PAN001", "variable")],
        id="empty-reason",
    ),
    pytest.param(
        "value: object = '# noqa: PAN001 - inside a string'\n",
        [(1, "PAN001", "variable")],
        id="text-in-a-string",
    ),
    pytest.param(
        "# annotation-exempt: heterogeneous=leaf\nvalue: object\n",
        [(2, "PAN001", "variable")],
        id="annotation-exempt-comment",
    ),
)

STALE_NOQA = (
    pytest.param(
        "value: int  # noqa: PAN001 - nothing to suppress\n",
        [(1, "PAN008", "noqa")],
        id="precise-line",
    ),
    pytest.param(
        "from typing import Any\n\n"
        "value: object  # noqa: PAN001, PAN002 - only one applies\n",
        [(3, "PAN008", "noqa")],
        id="one-of-two-codes",
    ),
    pytest.param(
        "value: int  # noqa: PAN001\n", [(1, "PAN008", "noqa")], id="without-reason"
    ),
    pytest.param(
        "# noqa: PAN001 - on the line above\nvalue: object\n",
        [(1, "PAN008", "noqa"), (2, "PAN001", "variable")],
        id="line-above-its-finding",
    ),
    pytest.param(
        "class C:\n"
        "    def __eq__(self, other: object) -> bool: ...  # noqa: PAN001 - dunder\n",
        [(2, "PAN008", "noqa")],
        id="exempt-by-rule",
    ),
    pytest.param(
        "value: int  # noqa: PAN004 - a retired code\n",
        [(1, "PAN008", "noqa")],
        id="unknown-code",
    ),
)

USED_NOQA = (
    pytest.param("value: int  # noqa: ARG001\n", id="another-linter"),
    pytest.param("value: int  # noqa\n", id="blanket"),
    pytest.param(
        "value: object  # noqa: E501, PAN001 - beside another linter's code\n",
        id="used-beside-another-code",
    ),
)

OBJECT_DETAIL = "`object` in an annotation; name the precise type"
FALLBACK_DETAIL = (
    "`object` in a runtime fallback; give its reason in a comment directly above it"
)
BARE_NOQA_DETAIL = (
    f"{OBJECT_DETAIL}; `# noqa: PAN001` needs a reason: `# noqa: PAN001 - <why>`"
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


@pytest.mark.parametrize(("template", "line"), ALIAS_CONSTRUCTS)
@pytest.mark.parametrize(("spelling", "codes"), SPELLINGS)
def test_find_annotation_violations_reports_aliases_not_written_as_type_statements(
    *, tmp_path: Path, template: str, line: int, spelling: str, codes: tuple[str, ...]
) -> None:
    """An alias outside a `type` statement is reported, with its imprecise value."""
    source = _write(tmp_path=tmp_path, text=template.replace("{T}", spelling))

    assert _summarize(find_annotation_violations(paths=[source])) == [
        *((line, code, "type-alias") for code in codes),
        (line, "PAN007", "type-alias"),
    ]


def test_find_annotation_violations_names_the_type_statement(tmp_path: Path) -> None:
    """The alias finding tells the author to write a `type` statement."""
    source = _write(tmp_path=tmp_path, text="Alias = dict[str, int]\n")

    assert [
        violation.detail for violation in find_annotation_violations(paths=[source])
    ] == ["write the alias as a `type Alias = ...` statement"]


@pytest.mark.parametrize("text", TYPE_STATEMENT_NEAR_MISSES)
def test_find_annotation_violations_ignores_assignments_that_are_not_aliases(
    *, tmp_path: Path, text: str
) -> None:
    """`type` statements, constants, rebindings and non-module scopes pass."""
    source = _write(tmp_path=tmp_path, text=text)

    assert find_annotation_violations(paths=[source]) == ()


def test_find_annotation_violations_reports_aliases_bound_to_any(
    tmp_path: Path,
) -> None:
    """A module-level `Alias = Any` is an alias that hides `Any`."""
    source = _write(tmp_path=tmp_path, text="from typing import Any\n\nAlias = Any\n")

    assert _summarize(find_annotation_violations(paths=[source])) == [
        (3, "PAN002", "type-alias"),
        (3, "PAN007", "type-alias"),
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


@pytest.mark.parametrize("text", DATA_MODEL_PARAMETERS)
def test_find_annotation_violations_allows_object_where_the_data_model_fixes_it(
    *, tmp_path: Path, text: str
) -> None:
    """`__setattr__`'s `value` and `__deepcopy__`'s `memo` may hold `object`."""
    source = _write(tmp_path=tmp_path, text=text)

    assert find_annotation_violations(paths=[source]) == ()


@pytest.mark.parametrize(("text", "expected"), DUNDER_NEAR_MISSES)
def test_find_annotation_violations_reports_object_outside_data_model_parameters(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """The dunder allowance covers `object` in the data-model parameters only."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


@pytest.mark.parametrize(("text", "expected"), NARROWED_PARAMETERS)
def test_find_annotation_violations_reports_object_parameters_narrowed_in_the_body(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """Narrowing an `object` parameter in its body does not excuse the annotation."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


@pytest.mark.parametrize(("identifier", "annotation", "detail"), LABEL_NAMES)
def test_find_annotation_violations_flags_bare_str_label_names(
    *, tmp_path: Path, identifier: str, annotation: str, detail: str
) -> None:
    """A `str` on a name with a domain alias points to that alias."""
    source = _write(
        tmp_path=tmp_path, text=f"def f(*, {identifier}: {annotation}) -> None: ...\n"
    )

    assert _details(find_annotation_violations(paths=[source])) == [
        (1, "PAN003", detail)
    ]


@pytest.mark.parametrize(("identifier", "annotation", "detail"), DOMAIN_NAMES)
def test_find_annotation_violations_flags_bare_str_on_targets_sources_and_args(
    *, tmp_path: Path, identifier: str, annotation: str, detail: str
) -> None:
    """A `str` on a target, source or argument name asks for a domain alias."""
    source = _write(
        tmp_path=tmp_path, text=f"def f(*, {identifier}: {annotation}) -> None: ...\n"
    )

    assert _details(find_annotation_violations(paths=[source])) == [
        (1, "PAN003", detail)
    ]


@pytest.mark.parametrize("text", LABEL_NAME_NEAR_MISSES)
def test_find_annotation_violations_ignores_str_on_other_names(
    *, tmp_path: Path, text: str
) -> None:
    """Names outside the label patterns, aliased names and returns pass."""
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


@pytest.mark.parametrize(("text", "expected"), BARE_GENERICS)
def test_find_annotation_violations_reports_generics_without_type_arguments(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """A generic written without its type arguments is reported."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


def test_find_annotation_violations_names_the_bare_generic(tmp_path: Path) -> None:
    """The bare-generic finding names the generic."""
    source = _write(tmp_path=tmp_path, text="value: dict = {}\n")

    assert [
        violation.detail for violation in find_annotation_violations(paths=[source])
    ] == ["`dict` without type arguments leaves them `Any`; spell them out"]


@pytest.mark.parametrize("text", PARAMETERIZED_GENERICS)
def test_find_annotation_violations_accepts_generics_with_type_arguments(
    *, tmp_path: Path, text: str
) -> None:
    """A generic with its type arguments, or a non-generic class, passes."""
    source = _write(tmp_path=tmp_path, text=text)

    assert find_annotation_violations(paths=[source]) == ()


@pytest.mark.parametrize(("text", "expected"), RUNTIME_FALLBACKS)
def test_find_annotation_violations_allows_commented_runtime_fallbacks(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """A wide alias passes only in the runtime branch of `TYPE_CHECKING`, commented."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


def test_find_annotation_violations_asks_for_the_reason_of_a_runtime_fallback(
    tmp_path: Path,
) -> None:
    """An uncommented runtime fallback asks for a comment giving its reason."""
    source = _write(
        tmp_path=tmp_path,
        text="from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from stores import Store\n"
        "else:\n"
        "    type Store = object\n",
    )

    assert [
        violation.detail for violation in find_annotation_violations(paths=[source])
    ] == [FALLBACK_DETAIL]


@pytest.mark.parametrize("text", NOQA_SUPPRESSIONS)
def test_find_annotation_violations_honours_noqa_with_a_reason(
    *, tmp_path: Path, text: str
) -> None:
    """`# noqa: PANxxx - <reason>` on the finding's line suppresses that code."""
    source = _write(tmp_path=tmp_path, text=text)

    assert find_annotation_violations(paths=[source]) == ()


@pytest.mark.parametrize(("text", "expected"), NOQA_NON_SUPPRESSIONS)
def test_find_annotation_violations_keeps_findings_without_a_matching_noqa(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """A noqa elsewhere, for another code, or without a reason suppresses nothing."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


def test_find_annotation_violations_asks_for_the_reason_of_a_bare_noqa(
    tmp_path: Path,
) -> None:
    """A `# noqa: PANxxx` without a reason keeps the finding and asks for one."""
    source = _write(tmp_path=tmp_path, text="value: object  # noqa: PAN001\n")

    assert _details(find_annotation_violations(paths=[source])) == [
        (1, "PAN001", BARE_NOQA_DETAIL)
    ]


@pytest.mark.parametrize(("text", "expected"), STALE_NOQA)
def test_find_annotation_violations_reports_noqa_codes_without_a_finding(
    *, tmp_path: Path, text: str, expected: list[tuple[int, str, str]]
) -> None:
    """A `# noqa` that names a `PAN` code its line does not report is a finding."""
    source = _write(tmp_path=tmp_path, text=text)

    assert _summarize(find_annotation_violations(paths=[source])) == expected


@pytest.mark.parametrize("text", USED_NOQA)
def test_find_annotation_violations_leaves_other_noqa_codes_alone(
    *, tmp_path: Path, text: str
) -> None:
    """Codes of other linters and blanket `noqa` comments are not this check's."""
    source = _write(tmp_path=tmp_path, text=text)

    assert find_annotation_violations(paths=[source]) == ()


def test_find_annotation_violations_names_the_stale_noqa_code(tmp_path: Path) -> None:
    """The stale-noqa finding names the code to remove."""
    source = _write(tmp_path=tmp_path, text="value: int  # noqa: PAN002 - none\n")

    assert _details(find_annotation_violations(paths=[source])) == [
        (1, "PAN008", "`# noqa: PAN002` matches no finding on this line; remove it")
    ]


def test_find_annotation_violations_reads_notebook_code_cells(tmp_path: Path) -> None:
    """Notebook findings carry the one-based cell number and the line in the cell."""
    source = _write_notebook(
        tmp_path=tmp_path,
        cells=(
            ("markdown", "value: object\n"),
            ("code", "import math\n"),
            ("code", "x = 1\nvalue: object = math.pi  # noqa: PAN001\n"),
        ),
    )

    assert [
        (violation.cell, violation.line, violation.code)
        for violation in find_annotation_violations(paths=[source])
    ] == [(3, 2, "PAN001")]


def test_find_annotation_violations_honours_noqa_in_notebook_cells(
    tmp_path: Path,
) -> None:
    """A `# noqa` with a reason works inside a notebook code cell."""
    source = _write_notebook(
        tmp_path=tmp_path,
        cells=(("code", "value: object  # noqa: PAN001 - any payload\n"),),
    )

    assert find_annotation_violations(paths=[source]) == ()


def test_find_annotation_violations_ignores_files_that_are_not_sources(
    tmp_path: Path,
) -> None:
    """Only `.py` files and notebooks are read."""
    source = tmp_path / "page.md"
    source.write_text("```python\nvalue: object\n```\n")

    assert find_annotation_violations(paths=[source]) == ()


def test_main_passes_silently_without_findings(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clean file exits 0 and prints nothing."""
    source = _write(tmp_path=tmp_path, text="value: int\n")

    exit_code = main(paths=[source])

    assert (exit_code, capsys.readouterr().out) == (0, "")


def test_main_prints_every_finding_and_a_summary(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each finding is printed, then a count per rule, and the run fails."""
    source = _write(tmp_path=tmp_path, text="first: object\nsecond: object\n")
    notebook = _write_notebook(
        tmp_path=tmp_path, cells=(("code", "regime_name: str\n"),)
    )

    exit_code = main(paths=[source, notebook])

    assert (exit_code, capsys.readouterr().out) == (
        1,
        (
            f"{source}:1: PAN001 variable: {OBJECT_DETAIL}\n"
            f"{source}:2: PAN001 variable: {OBJECT_DETAIL}\n"
            f"{notebook}:cell 1:line 1: PAN003 variable: "
            "annotate `regime_name` as `RegimeName`, not `str`\n"
            "3 findings in 2 files: PAN001 2, PAN003 1\n"
        ),
    )


def _write(*, tmp_path: Path, text: str) -> Path:
    source = tmp_path / "module.py"
    source.write_text(text)
    return source


def _write_notebook(*, tmp_path: Path, cells: tuple[tuple[str, str], ...]) -> Path:
    source = tmp_path / "page.ipynb"
    source.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": cell_type,
                        "metadata": {},
                        "source": text.splitlines(keepends=True),
                    }
                    for cell_type, text in cells
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    return source


def _summarize(
    violations: tuple[AnnotationViolation, ...],
) -> list[tuple[int, str, str]]:
    return [
        (violation.line, violation.code, violation.construct)
        for violation in violations
    ]


def _details(
    violations: tuple[AnnotationViolation, ...],
) -> list[tuple[int, str, str]]:
    return [
        (violation.line, violation.code, violation.detail) for violation in violations
    ]
