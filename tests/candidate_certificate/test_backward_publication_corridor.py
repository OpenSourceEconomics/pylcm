"""Solve publication preserves feature authority and diagnostic transport."""

import ast
import copy
from pathlib import Path
from typing import Literal

import pytest

from tests.candidate_certificate import direct_flow

_ROOT = Path(__file__).parents[2]
_DIAGNOSTIC = "backward output-layout transport: solve publication corridor changed"


def _reviewed_tree() -> ast.Module:
    """Locate the actual solve and its three explicit feature transports."""
    tree = ast.parse(
        (_ROOT / direct_flow.BACKWARD_INDUCTION_SOURCE).read_text(encoding="utf-8")
    )
    _feature_transports(tree=tree)
    return tree


def _feature_transports(*, tree: ast.Module) -> tuple[ast.Call, ast.If, ast.Call]:
    """Require the actual descriptor, generated authority and paired read counts."""
    solves = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    ]
    assert len(solves) == 1
    solve = solves[0]
    consumers = [
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "consume_kernel_output"
    ]
    folds = [
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_fold_period_diagnostics"
    ]
    assert len(consumers) == len(folds) == 1
    consumer, fold = consumers[0], folds[0]
    query = [
        item.value
        for item in consumer.keywords
        if item.arg == "continuation_query_spec"
    ]
    counts = [
        item.value for item in fold.keywords if item.arg == "continuation_read_counts"
    ]
    assert len(query) == len(counts) == 1
    expected_query = ast.parse(
        "continuation_spec.layout.query_spec "
        "if isinstance(continuation_spec, EGMContinuationSpec) else None",
        mode="eval",
    ).body
    expected_counts = ast.parse("result.continuation_read_counts", mode="eval").body
    assert ast.dump(query[0]) == ast.dump(expected_query)
    assert ast.dump(counts[0]) == ast.dump(expected_counts)
    authorities = [
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "result.generated_continuation_authority is not None"
    ]
    assert len(authorities) == 1
    authority = authorities[0]
    expected_authority = ast.parse(
        "if result.generated_continuation_authority is not None:\n"
        "    period_generated_continuation_authorities[regime_name] = "
        "result.generated_continuation_authority\n"
    ).body[0]
    assert ast.dump(authority) == ast.dump(expected_authority)
    return consumer, authority, fold


def test_reviewed_solve_publication_preserves_exact_feature_transport() -> None:
    """Accept the solve's actual descriptor, authority and diagnostic publication."""
    tree = _reviewed_tree()
    errors = direct_flow._backward_output_layout_errors(tree)
    assert all(error == _DIAGNOSTIC for error in errors)
    assert errors == []


@pytest.mark.parametrize(
    "damage",
    ["query_descriptor", "generated_authority", "read_counts"],
)
def test_solve_publication_rejects_missing_feature_transport(
    *, damage: Literal["query_descriptor", "generated_authority", "read_counts"]
) -> None:
    """Dropping one actual transport cannot pass solve publication validation."""
    tree = copy.deepcopy(_reviewed_tree())
    consumer, authority, fold = _feature_transports(tree=tree)
    if damage == "generated_authority":
        authority.test = ast.Constant(value=False)
    else:
        call = consumer if damage == "query_descriptor" else fold
        name = (
            "continuation_query_spec"
            if damage == "query_descriptor"
            else "continuation_read_counts"
        )
        keywords = [item for item in call.keywords if item.arg == name]
        assert len(keywords) == 1
        keywords[0].value = ast.Constant(value=None)
    assert direct_flow._backward_output_layout_errors(tree) == [_DIAGNOSTIC]
