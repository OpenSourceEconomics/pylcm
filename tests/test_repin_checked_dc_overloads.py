"""Maintained repinning uses precise checked DC runtime implementations."""

import ast
from pathlib import Path

import pytest

from tests.candidate_certificate import direct_flow, repin_corridors

_ROOT = Path(__file__).parents[1]
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


def _select_runtime(
    *, tree: ast.Module, qualname: str, expected_import: str
) -> ast.FunctionDef:
    """Select only beside all three literal stubs and exact unique scope bindings."""
    module_bindings = direct_flow._scope_binding_counts(tree.body)
    if qualname == "DCEGMQuery.read_choices":
        cls = direct_flow._class_definition(tree=tree, name="DCEGMQuery")
        scope = cls.body
        name = "read_choices"
        assert module_bindings.get("DCEGMQuery") == 1
    else:
        assert qualname == "_read_one_row_dc_child"
        scope = tree.body
        name = qualname
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
    assert len(expected) == 3
    assert len(functions) == 4
    assert direct_flow._scope_binding_counts(scope).get(name) == 4
    assert module_bindings.get("overload") == 1
    assert direct_flow._relevant_imports(tree=tree, names={"overload"}) == [
        expected_import
    ]
    assert [direct_flow._ast_key(node) for node in functions[:3]] == [
        direct_flow._ast_key(node) for node in expected
    ]
    assert functions[-1].decorator_list == []
    return functions[-1]


@pytest.mark.parametrize(
    ("source", "qualname", "expected_import"),
    [
        (
            "src/_lcm/solution/egm.py",
            "_read_one_row_dc_child",
            "from typing import Literal, NamedTuple, cast, overload",
        ),
        (
            "src/_lcm/egm/dcegm_query.py",
            "DCEGMQuery.read_choices",
            "from typing import Literal, NamedTuple, overload",
        ),
    ],
)
def test_repin_checked_dc_overloads_uses_the_exact_runtime_body(
    *, source: direct_flow.SourcePath, qualname: str, expected_import: str
) -> None:
    """Exact checked DC declarations identify the pinned runtime implementation."""
    tree = ast.parse((_ROOT / source).read_text(encoding="utf-8"))
    implementation = _select_runtime(
        tree=tree, qualname=qualname, expected_import=expected_import
    )
    expected = direct_flow._callable_ast_sha256(implementation)
    pin = repin_corridors.CorridorPin(
        source=source,
        kind="callable",
        name=qualname,
        pinned=expected,
        line=1,
        column=0,
    )
    assert (
        repin_corridors._recompute(pin=pin, tree=tree, module=direct_flow) == expected
    )
