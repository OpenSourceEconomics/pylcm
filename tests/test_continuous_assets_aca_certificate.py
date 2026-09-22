"""Keep the continuous-sharding eligibility helpers in every exact AST corridor."""

import ast
from pathlib import Path

import pytest

from tests.candidate_certificate import direct_flow


@pytest.mark.parametrize(
    "helper",
    [
        "_supports_continuous_sharding_vocabulary",
        "_supports_unsharded_continuous_process",
    ],
)
@pytest.mark.parametrize(
    "checker_name",
    [
        "_simulation_dispatch_corridor_errors",
        "_finite_budget_errors",
        "_combined_input_errors",
        "_uniform_process_errors",
    ],
)
def test_permissive_eligibility_helper_is_rejected_without_byte_seals(
    *,
    helper: str,
    checker_name: str,
) -> None:
    """Each corridor rejects unconditional eligibility even with unchanged bindings."""
    source = Path(__file__).resolve().parents[1] / direct_flow.MODEL_SOURCE
    tree = ast.parse(source.read_text())
    checker = getattr(direct_flow, checker_name)
    assert checker(tree=tree, source=direct_flow.MODEL_SOURCE) == []
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == helper
    )
    node.body = [ast.Return(value=ast.Constant(value=True))]
    errors = checker(tree=tree, source=direct_flow.MODEL_SOURCE)
    assert len(errors) == 1
    assert f"exact callable corridor {helper!r} changed" in errors[0]
