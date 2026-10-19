"""Require exact scan overload declarations beside their runtime body."""

import ast
import copy
from pathlib import Path

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.nested_helper_selection import (
    select_interval_winner_call,
)

_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_stub",
        "extra_stub",
        "reordered_stubs",
        "changed_stub",
        "runtime_decorator",
        "extra_method_binding",
        "extra_class_binding",
        "overload_alias",
        "overload_rebinding",
    ],
)
def test_interval_winner_method_refuses_changed_overload_declarations(
    *, mutation: str
) -> None:
    """The runtime method is selected only beside its exact ordered declarations."""
    tree = ast.parse((_ROOT / "src/_lcm/egm/nbegm_step.py").read_text(encoding="utf-8"))
    runtime = select_interval_winner_call(tree=tree, checker=direct_flow)
    cls = direct_flow._class_definition(tree=tree, name="_IntervalWinnerStep")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    ]
    assert len(methods) == 4
    assert runtime is methods[-1]
    if mutation == "missing_stub":
        cls.body.remove(methods[0])
    elif mutation == "extra_stub":
        cls.body.insert(cls.body.index(methods[0]), copy.deepcopy(methods[0]))
    elif mutation == "reordered_stubs":
        first, second = cls.body.index(methods[0]), cls.body.index(methods[1])
        cls.body[first], cls.body[second] = cls.body[second], cls.body[first]
    elif mutation == "changed_stub":
        methods[0].returns = ast.Name(id="changed_annotation", ctx=ast.Load())
    elif mutation == "runtime_decorator":
        runtime.decorator_list.append(ast.Name(id="property", ctx=ast.Load()))
    elif mutation == "extra_method_binding":
        cls.body.extend(ast.parse("__call__ = None").body)
    elif mutation == "extra_class_binding":
        tree.body.extend(ast.parse("_IntervalWinnerStep = None").body)
    elif mutation == "overload_alias":
        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "typing"
        ]
        aliases = [
            alias
            for node in imports
            for alias in node.names
            if alias.name == "overload"
        ]
        assert len(aliases) == 1
        aliases[0].asname = "changed_overload"
    elif mutation == "overload_rebinding":
        tree.body.extend(ast.parse("overload = None").body)
    else:
        raise AssertionError(mutation)
    with pytest.raises(
        ValueError, match=r"_IntervalWinnerStep.__call__ overload declarations changed"
    ):
        select_interval_winner_call(tree=tree, checker=direct_flow)
