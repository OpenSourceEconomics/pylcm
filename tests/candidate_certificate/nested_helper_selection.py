"""Select the NB query method beside its exact ordered overload declarations."""

import ast
from types import ModuleType

_DECLARATIONS = """@overload
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
) -> _NBChoiceResult: ...

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
) -> tuple[_NBChoiceResult, ContinuationReadCounts]: ...

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
) -> _NBChoiceResult | tuple[_NBChoiceResult, ContinuationReadCounts]: ...


"""


def select_nbegm_query_read_choices(
    *, tree: ast.Module, checker: ModuleType
) -> ast.FunctionDef:
    """Require the reviewed typing stubs and one undecorated runtime method."""
    cls = checker._class_definition(tree=tree, name="NBEGMQuery")
    expected = [
        node
        for node in ast.parse(_DECLARATIONS).body
        if isinstance(node, ast.FunctionDef) and node.name == "read_choices"
    ]
    functions = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "read_choices"
    ]
    module_bindings = checker._scope_binding_counts(tree.body)
    class_bindings = checker._scope_binding_counts(cls.body)
    if (
        len(expected) != 3
        or len(functions) != 4
        or module_bindings.get("NBEGMQuery") != 1
        or class_bindings.get("read_choices") != 4
        or module_bindings.get("overload") != 1
        or checker._relevant_imports(tree=tree, names={"overload"})
        != ["from typing import Literal, NamedTuple, overload"]
        or [checker._ast_key(node) for node in functions[:3]]
        != [checker._ast_key(node) for node in expected]
        or functions[-1].decorator_list
    ):
        raise ValueError("NBEGMQuery.read_choices overload declarations changed")
    return functions[-1]


_INTERVAL_WINNER_DECLARATIONS = """@overload
def __call__(
    self,
    held: EnvelopeWinner,
    block_inputs: tuple[IntND, BoolND],
) -> tuple[EnvelopeWinner, None]: ...

@overload
def __call__(
    self,
    held: _IntervalQueryWinner,
    block_inputs: tuple[IntND, BoolND],
) -> tuple[_IntervalQueryWinner, None]: ...

@overload
def __call__(
    self,
    held: _IntervalIntegrityWinner,
    block_inputs: tuple[IntND, BoolND],
) -> tuple[_IntervalIntegrityWinner, None]: ...
"""


def select_interval_winner_call(
    *, tree: ast.Module, checker: ModuleType
) -> ast.FunctionDef:
    """Require three exact ordered scan declarations and their runtime method."""
    cls = checker._class_definition(tree=tree, name="_IntervalWinnerStep")
    expected = [
        node
        for node in ast.parse(_INTERVAL_WINNER_DECLARATIONS).body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    ]
    functions = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    ]
    module_bindings = checker._scope_binding_counts(tree.body)
    class_bindings = checker._scope_binding_counts(cls.body)
    if (
        len(expected) != 3
        or len(functions) != 4
        or module_bindings.get("_IntervalWinnerStep") != 1
        or class_bindings.get("__call__") != 4
        or module_bindings.get("overload") != 1
        or checker._relevant_imports(tree=tree, names={"overload"})
        != ["from typing import Literal, NamedTuple, cast, overload"]
        or [checker._ast_key(node) for node in functions[:3]]
        != [checker._ast_key(node) for node in expected]
        or functions[-1].decorator_list
    ):
        raise ValueError("_IntervalWinnerStep.__call__ overload declarations changed")
    return functions[-1]
