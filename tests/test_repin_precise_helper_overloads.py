"""Recompute precise overload pins with their exact reviewed runtime selectors."""

import ast
from pathlib import Path

import pytest

from tests.candidate_certificate import direct_flow, repin_corridors

_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("source", "name", "selector"),
    [
        ("src/_lcm/egm/upper_envelope/query.py", "envelope_at_query", "associated"),
        ("src/_lcm/egm/upper_envelope/query.py", "empty_envelope_winner", "associated"),
        ("src/_lcm/egm/upper_envelope/query.py", "merge_envelope_winner", "associated"),
        (
            "src/_lcm/egm/upper_envelope/query.py",
            "finish_envelope_winner",
            "associated",
        ),
        ("src/_lcm/egm/nbegm_step.py", "nbegm_multi_interval_step_savings", "producer"),
        (
            "src/_lcm/egm/nbegm_step.py",
            "nbegm_per_interval_continuation_step_savings",
            "producer",
        ),
        ("src/_lcm/egm/nbegm_step.py", "_IntervalWinnerStep.__call__", "interval"),
        ("src/_lcm/egm/nbegm_query.py", "NBEGMQuery.read_choices", "query"),
    ],
)
def test_repin_uses_the_canonical_precise_runtime_selector(
    *, source: direct_flow.SourcePath, name: str, selector: str
) -> None:
    """Each maintained overload pin hashes the same validated implementation body."""
    tree = ast.parse((_ROOT / source).read_text(encoding="utf-8"))
    if selector == "associated":
        implementation = direct_flow._associated_saving_implementation(
            tree=tree, name=name
        )
    elif selector == "producer":
        implementation = direct_flow._nb_producer_implementation(tree=tree, name=name)
    elif selector == "interval":
        implementation = direct_flow._select_interval_winner_call(
            tree=tree, checker=direct_flow
        )
    elif selector == "query":
        implementation = direct_flow._select_nbegm_query_read_choices(
            tree=tree, checker=direct_flow
        )
    else:
        raise AssertionError(selector)
    expected = direct_flow._callable_ast_sha256(implementation)
    pin = repin_corridors.CorridorPin(
        source=source,
        kind="callable",
        name=name,
        pinned=expected,
        line=1,
        column=0,
    )
    assert (
        repin_corridors._recompute(pin=pin, tree=tree, module=direct_flow) == expected
    )
