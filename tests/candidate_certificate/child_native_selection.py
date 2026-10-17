"""Select exact native-binding composition beside its ordered declarations."""

import ast
from types import ModuleType

_DECLARATIONS = """@overload
def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    savings: Float1D,
    _binding_source_count: int,
    _binding_queries: FloatND,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_width: int = 1,
) -> _OwnedNegmBindingEnvelope: ...
@overload
def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    _binding_source_count: int,
    _binding_queries: FloatND,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_width: int = 1,
    savings: None = None,
) -> _NegmBindingEnvelope: ...
@overload
def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_width: int = 1,
    savings: None = None,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]: ...
@overload
def refine_envelope_exact(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
    max_runs: int = DEFAULT_MAX_RUNS,
    cell_width: int = 1,
    savings: Float1D,
) -> OwnedEnvelope: ..."""


def select_refine_envelope_exact(
    *, tree: ast.Module, checker: ModuleType
) -> ast.FunctionDef:
    """Require four exact ordered stubs and one undecorated runtime definition."""
    expected = [
        node
        for node in ast.parse(_DECLARATIONS).body
        if isinstance(node, ast.FunctionDef) and node.name == "refine_envelope_exact"
    ]
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "refine_envelope_exact"
    ]
    bindings = checker._scope_binding_counts(tree.body)
    if (
        len(expected) != 4
        or len(functions) != 5
        or bindings.get("refine_envelope_exact") != 5
        or bindings.get("overload") != 1
        or checker._relevant_imports(tree=tree, names={"overload"})
        != ["from typing import NamedTuple, overload"]
        or [checker._ast_key(node) for node in functions[:4]]
        != [checker._ast_key(node) for node in expected]
        or functions[-1].decorator_list
    ):
        raise ValueError("refine_envelope_exact overload declarations changed")
    return functions[-1]
