"""Checked DC declarations keep their exact runtime and typing authorities."""

import ast
import copy
from pathlib import Path

import pytest

from tests.candidate_certificate import direct_flow, repin_corridors
from tests.candidate_certificate.checked_dc_selection import select_checked_dc_runtime

_ROOT = Path(__file__).parents[1]
_OWNERS = [
    ("src/_lcm/solution/egm.py", "_read_one_row_dc_child"),
    ("src/_lcm/egm/dcegm_query.py", "DCEGMQuery.read_choices"),
]
_ATTACKS = [
    "missing_stub",
    "extra_stub",
    "reordered_stubs",
    "changed_stub",
    "runtime_decorator",
    "extra_runtime_binding",
    "overload_alias",
    "overload_rebinding",
    "overload_import_removed",
]


def _alter_declaration(*, tree: ast.Module, qualname: str, attack: str) -> None:
    """Change one explicit declaration channel after genuine strict selection."""
    runtime = select_checked_dc_runtime(
        tree=tree, qualname=qualname, checker=direct_flow
    )
    if "." in qualname:
        class_name, name = qualname.split(".", maxsplit=1)
        scope = direct_flow._class_definition(tree=tree, name=class_name).body
    else:
        name = qualname
        scope = tree.body
    functions = [
        node
        for node in scope
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if attack == "missing_stub":
        scope.remove(functions[0])
    elif attack == "extra_stub":
        scope.insert(scope.index(functions[0]), copy.deepcopy(functions[0]))
    elif attack == "reordered_stubs":
        first, second = scope.index(functions[0]), scope.index(functions[1])
        scope[first], scope[second] = scope[second], scope[first]
    elif attack == "changed_stub":
        functions[0].returns = ast.Name(id="changed_annotation", ctx=ast.Load())
    elif attack == "runtime_decorator":
        runtime.decorator_list.append(ast.Name(id="property", ctx=ast.Load()))
    elif attack == "extra_runtime_binding":
        scope.extend(ast.parse(f"{name} = None").body)
    elif attack == "overload_rebinding":
        tree.body.extend(ast.parse("overload = None").body)
    else:
        _alter_import(tree=tree, attack=attack)


def _alter_import(*, tree: ast.Module, attack: str) -> None:
    """Modify only the exact existing typing overload import."""
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "typing"
    ]
    aliases = [
        (node, alias)
        for node in imports
        for alias in node.names
        if alias.name == "overload"
    ]
    assert len(aliases) == 1
    node, alias = aliases[0]
    if attack == "overload_alias":
        alias.asname = "changed_overload"
    elif attack == "overload_import_removed":
        node.names.remove(alias)
        if not node.names:
            tree.body.remove(node)
    else:
        raise ValueError(attack)


def _assert_shared_rejection(
    *, tree: ast.Module, source: direct_flow.SourcePath, qualname: str, pinned: str
) -> None:
    """The same changed declarations fail selector, verifier and maintained repin."""
    diagnostic = f"{qualname} overload declarations changed"
    with pytest.raises(ValueError, match="overload declarations changed"):
        select_checked_dc_runtime(tree=tree, qualname=qualname, checker=direct_flow)
    pin = repin_corridors.CorridorPin(
        source=source,
        kind="callable",
        name=qualname,
        pinned=pinned,
        line=1,
        column=0,
    )
    with pytest.raises(ValueError, match="overload declarations changed"):
        repin_corridors._recompute(pin=pin, tree=tree, module=direct_flow)
    assert direct_flow._exact_callable_errors(
        tree=tree, label="checked DC", contracts={qualname: pinned}
    ) == [f"checked DC: {diagnostic}"]


@pytest.mark.parametrize(("source", "qualname"), _OWNERS)
@pytest.mark.parametrize("attack", _ATTACKS)
def test_checked_dc_declarations_refuse_each_changed_channel(
    *, source: direct_flow.SourcePath, qualname: str, attack: str
) -> None:
    """Each stub, binding and import attack fails the same exact runtime authority."""
    tree = ast.parse((_ROOT / source).read_text(encoding="utf-8"))
    runtime = select_checked_dc_runtime(
        tree=tree, qualname=qualname, checker=direct_flow
    )
    pinned = direct_flow._callable_ast_sha256(runtime)
    _alter_declaration(tree=tree, qualname=qualname, attack=attack)
    _assert_shared_rejection(tree=tree, source=source, qualname=qualname, pinned=pinned)


def test_checked_dc_method_refuses_an_extra_class_binding() -> None:
    """One class definition cannot hide a second stored binding of its class name."""
    source, qualname = _OWNERS[1]
    tree = ast.parse((_ROOT / source).read_text(encoding="utf-8"))
    runtime = select_checked_dc_runtime(
        tree=tree, qualname=qualname, checker=direct_flow
    )
    pinned = direct_flow._callable_ast_sha256(runtime)
    tree.body.extend(ast.parse("DCEGMQuery = None").body)
    _assert_shared_rejection(tree=tree, source=source, qualname=qualname, pinned=pinned)
