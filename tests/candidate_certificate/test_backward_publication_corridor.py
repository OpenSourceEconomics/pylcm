"""Solve publication preserves diagnostic transport."""

import ast
import copy
from pathlib import Path

from tests.candidate_certificate import direct_flow

_ROOT = Path(__file__).parents[2]
_DIAGNOSTIC = "backward output-layout transport: solve publication corridor changed"


def _reviewed_tree() -> ast.Module:
    """Locate the actual solve and its paired read-count transport."""
    tree = ast.parse(
        (_ROOT / direct_flow.BACKWARD_INDUCTION_SOURCE).read_text(encoding="utf-8")
    )
    _read_count_transport(tree=tree)
    return tree


def _read_count_transport(*, tree: ast.Module) -> ast.Call:
    """Require the solve's single diagnostic fold to carry the paired read counts."""
    solves = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    ]
    assert len(solves) == 1
    folds = [
        node
        for node in ast.walk(solves[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_fold_period_diagnostics"
    ]
    assert len(folds) == 1
    counts = [
        item.value
        for item in folds[0].keywords
        if item.arg == "continuation_read_counts"
    ]
    assert len(counts) == 1
    expected = ast.parse("result.continuation_read_counts", mode="eval").body
    assert ast.dump(counts[0]) == ast.dump(expected)
    return folds[0]


def test_reviewed_solve_publication_preserves_exact_feature_transport() -> None:
    """Accept the solve's actual diagnostic publication."""
    errors = direct_flow._backward_output_layout_errors(_reviewed_tree())
    assert errors == []


def test_solve_publication_rejects_missing_read_counts() -> None:
    """Dropping the read-count transport cannot pass solve publication validation."""
    tree = copy.deepcopy(_reviewed_tree())
    fold = _read_count_transport(tree=tree)
    keywords = [
        item for item in fold.keywords if item.arg == "continuation_read_counts"
    ]
    keywords[0].value = ast.Constant(value=None)
    assert direct_flow._backward_output_layout_errors(tree) == [_DIAGNOSTIC]
