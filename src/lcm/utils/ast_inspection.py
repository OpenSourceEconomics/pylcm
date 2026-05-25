"""AST helpers for inspecting array-subscript patterns in user functions."""

import ast
import inspect
import textwrap
from collections.abc import Callable


def _get_func_indexing_params(
    *,
    func: Callable,
    array_param_name: str,
) -> list[str]:
    """Return indexing parameter names by inspecting array subscripts.

    Inspect `array_param_name`'s subscripts in the function source for
    `param[x, y, ...]` patterns where all index elements are bare names
    that are also function parameters.

    Args:
        func: The function to inspect.
        array_param_name: The array parameter whose subscripts to inspect.

    Returns:
        List of indexing parameter names, or empty list if no array
        subscripts are found (scalar function).

    Raises:
        TypeError: If the function source cannot be inspected (e.g., lambda).
        ValueError: If computed indices are used instead of bare names.

    """
    func_name = getattr(func, "__name__", "<unknown>")

    if func_name == "<lambda>":
        msg = "Cannot inspect lambda functions. Define a named function instead."
        raise TypeError(msg)

    try:
        source = textwrap.dedent(inspect.getsource(func))
    except OSError, TypeError:
        msg = (
            f"Cannot inspect source of '{func_name}'. "
            f"Define a named function instead of a lambda."
        )
        raise TypeError(msg) from None

    tree = ast.parse(source)
    sig = inspect.signature(func)
    param_names = set(sig.parameters)

    subscripts = _collect_subscripts(tree=tree, param_name=array_param_name)
    if not subscripts:
        return []

    if len(subscripts) > 1:
        msg = (
            f"Function '{func_name}' has multiple `{array_param_name}[...]` "
            f"subscripts. Use exactly one subscript so the indexing order "
            f"can be determined unambiguously."
        )
        raise ValueError(msg)

    names = _extract_bare_names(subscripts[0])

    if names is not None and all(n in param_names for n in names):
        return names

    if names is not None:
        non_params = [n for n in names if n not in param_names]
        msg = (
            f"Function '{func_name}' indexes `{array_param_name}` with names "
            f"{non_params} that are not function parameters. All subscript "
            f"indices must be function parameters (not aliased variables)."
        )
        raise ValueError(msg)

    if _slice_references_params(slice_node=subscripts[0], param_names=param_names):
        msg = (
            f"Function '{func_name}' uses computed indices in "
            f"`{array_param_name}[...]`. Use bare parameter names as indices. "
            f"If you need a computed index, extract it into a separate "
            f"function in the regime (e.g., "
            f"`adjusted_period(period): return period - 1`) "
            f"and use the function output as the index."
        )
        raise ValueError(msg)

    return []


def _slice_references_params(
    *,
    slice_node: ast.expr,
    param_names: set[str],
) -> bool:
    """Check if any `ast.Name` in the slice is a function parameter.

    Args:
        slice_node: AST node for the subscript slice.
        param_names: Set of function parameter names.

    Returns:
        `True` if any bare name in the slice matches a parameter.

    """
    return any(
        isinstance(node, ast.Name) and node.id in param_names
        for node in ast.walk(slice_node)
    )


def _collect_subscripts(
    *,
    tree: ast.Module,
    param_name: str,
) -> list[ast.expr]:
    """Find all `param_name[...]` subscript slice nodes in an AST.

    Args:
        tree: Parsed AST module.
        param_name: Name of the parameter to search for subscripts.

    Returns:
        List of AST slice nodes from matching subscripts.

    """
    return [
        node.slice
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == param_name
    ]


def _extract_bare_names(slice_node: ast.expr) -> list[str] | None:
    """Extract bare variable names from a subscript slice.

    Return `None` if any index element is not a bare `ast.Name` (e.g. a
    `BinOp` or `Call`).
    """
    if isinstance(slice_node, ast.Name):
        return [slice_node.id]

    if isinstance(slice_node, ast.Tuple):
        names: list[str] = []
        for elt in slice_node.elts:
            if not isinstance(elt, ast.Name):
                return None
            names.append(elt.id)
        return names

    return None
