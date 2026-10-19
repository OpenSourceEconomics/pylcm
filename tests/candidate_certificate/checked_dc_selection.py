"""Select checked DC bodies beside their exact ordered typing declarations."""

import ast
from types import ModuleType

_DECLARATIONS = {
    "_read_one_row_dc_child": """@overload
def _read_one_row_dc_child(
    *,
    query: DCEGMQuery | NBEGMQuery,
    liquid: Float1D,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData | None,
    pool: EconFunctionKwargs | None,
    floor_state: _DCFloorState | None = None,
    active: BoolND | None = None,
    collect_continuation_read_counts: Literal[False] = False,
) -> _DCChildRead: ...
@overload
def _read_one_row_dc_child(
    *,
    query: DCEGMQuery | NBEGMQuery,
    liquid: Float1D,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData | None,
    pool: EconFunctionKwargs | None,
    floor_state: _DCFloorState | None = None,
    active: BoolND | None = None,
    collect_continuation_read_counts: Literal[True],
) -> _CountedDCChildRead: ...
@overload
def _read_one_row_dc_child(
    *,
    query: DCEGMQuery | NBEGMQuery,
    liquid: Float1D,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData | None,
    pool: EconFunctionKwargs | None,
    floor_state: _DCFloorState | None = None,
    active: BoolND | None = None,
    collect_continuation_read_counts: bool,
) -> _DCChildRead | _CountedDCChildRead: ...""",
    "DCEGMQuery.read_choices": """@overload
def read_choices(
    self,
    *,
    grid: FloatND,
    value: FloatND,
    marginal: FloatND,
    data: EGMQueryData,
    row_queries: FloatND,
    row_gradients: FloatND,
    passive_values: tuple[ScalarFloat, ...],
    passive_grids: tuple[Float1D, ...],
    action_names: tuple[ActionName, ...],
    action_values: tuple[FloatND | IntND, ...],
    pool: EconFunctionKwargs,
    euler_gradient: ScalarFloat,
    paired_marginal_read: bool = False,
    collect_continuation_read_counts: Literal[False] = False,
) -> DCEGMChoiceResult: ...
@overload
def read_choices(
    self,
    *,
    grid: FloatND,
    value: FloatND,
    marginal: FloatND,
    data: EGMQueryData,
    row_queries: FloatND,
    row_gradients: FloatND,
    passive_values: tuple[ScalarFloat, ...],
    passive_grids: tuple[Float1D, ...],
    action_names: tuple[ActionName, ...],
    action_values: tuple[FloatND | IntND, ...],
    pool: EconFunctionKwargs,
    euler_gradient: ScalarFloat,
    paired_marginal_read: bool = False,
    collect_continuation_read_counts: Literal[True],
) -> tuple[DCEGMChoiceResult, ContinuationReadCounts]: ...
@overload
def read_choices(
    self,
    *,
    grid: FloatND,
    value: FloatND,
    marginal: FloatND,
    data: EGMQueryData,
    row_queries: FloatND,
    row_gradients: FloatND,
    passive_values: tuple[ScalarFloat, ...],
    passive_grids: tuple[Float1D, ...],
    action_names: tuple[ActionName, ...],
    action_values: tuple[FloatND | IntND, ...],
    pool: EconFunctionKwargs,
    euler_gradient: ScalarFloat,
    paired_marginal_read: bool = False,
    collect_continuation_read_counts: bool,
) -> DCEGMChoiceResult | tuple[DCEGMChoiceResult, ContinuationReadCounts]: ...""",
}
_TYPING_IMPORTS = {
    "_read_one_row_dc_child": "from typing import Literal, NamedTuple, cast, overload",
    "DCEGMQuery.read_choices": "from typing import Literal, NamedTuple, overload",
}


def select_checked_dc_runtime(
    *, tree: ast.Module, qualname: str, checker: ModuleType
) -> ast.FunctionDef:
    """Require three literal stubs and one runtime binding in only two named scopes."""
    if qualname not in _DECLARATIONS:
        raise ValueError(f"No checked DC overload authority for {qualname!r}")
    bindings = checker._scope_binding_counts(tree.body)
    if qualname == "DCEGMQuery.read_choices":
        cls = checker._class_definition(tree=tree, name="DCEGMQuery")
        if bindings.get("DCEGMQuery") != 1:
            raise ValueError("DCEGMQuery.read_choices overload declarations changed")
        scope = cls.body
        name = "read_choices"
    else:
        scope = tree.body
        name = "_read_one_row_dc_child"
    expected = [
        node
        for node in ast.parse(_DECLARATIONS[qualname]).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    functions = [
        node
        for node in scope
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if (
        len(expected) != 3
        or len(functions) != 4
        or checker._scope_binding_counts(scope).get(name) != 4
        or bindings.get("overload") != 1
        or checker._relevant_imports(tree=tree, names={"overload"})
        != [_TYPING_IMPORTS[qualname]]
        or [checker._ast_key(node) for node in functions[:3]]
        != [checker._ast_key(node) for node in expected]
        or functions[-1].decorator_list
    ):
        raise ValueError(f"{qualname} overload declarations changed")
    return functions[-1]
