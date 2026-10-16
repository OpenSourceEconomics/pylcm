"""Maintained repinning selects the exact native composition implementation."""

import ast
from pathlib import Path

from tests.candidate_certificate import direct_flow, repin_corridors
from tests.candidate_certificate.child_native_selection import (
    select_refine_envelope_exact,
)

_ROOT = Path(__file__).parents[1]
_SOURCE = "src/_lcm/egm/upper_envelope/segment_envelope.py"
_EXPECTED_RUNTIME_SHA256 = (
    "f38c9eec39f60b607f5c98bfee1ac196d952b84baa85ef2b35ec6d31d6c4cbc3"
)


def test_repin_native_composition_uses_the_strict_runtime_selector() -> None:
    """Four exact ordered overloads identify the genuine runtime callable."""
    tree = ast.parse((_ROOT / _SOURCE).read_text(encoding="utf-8"), filename=_SOURCE)
    runtime = select_refine_envelope_exact(tree=tree, checker=direct_flow)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "refine_envelope_exact"
    ]
    assert len(functions) == 5
    assert runtime is functions[4]
    assert runtime.decorator_list == []
    assert direct_flow._callable_ast_sha256(runtime) == _EXPECTED_RUNTIME_SHA256
    pin = repin_corridors.CorridorPin(
        source=_SOURCE,
        kind="callable",
        name="refine_envelope_exact",
        pinned=_EXPECTED_RUNTIME_SHA256,
        line=1,
        column=0,
    )
    assert repin_corridors._recompute(pin=pin, tree=tree, module=direct_flow) == (
        _EXPECTED_RUNTIME_SHA256
    )
