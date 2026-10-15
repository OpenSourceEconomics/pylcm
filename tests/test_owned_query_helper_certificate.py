"""Require explicit body obligations for owned query integrity and authority."""

import ast
import copy
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow

type MutationSpec = tuple[str, str, str, str, str, int]

_FEASIBILITY = "src/_lcm/solution/feasibility.py"
_BACKWARD = "src/_lcm/solution/backward_induction.py"
_CONTINUATION = "src/_lcm/continuation.py"
_PROCESSING = "src/_lcm/regime_building/processing.py"
_PRIORITY_MUTATION = "owned_query_status:validator_coverage_before_arithmetic"
_OWNED_QUERY_HELPER_MUTATIONS: dict[str, MutationSpec] = {
    "owned_query_status:pickle_wrong_record": (
        _FEASIBILITY,
        "QueryInputStatus.__reduce__",
        "expression",
        "_unflatten_query_input_status",
        "_unflatten_bound_status",
        1,
    ),
    "owned_query_status:flatten_floor_grid_swapped": (
        _FEASIBILITY,
        "_flatten_query_input_status",
        "expression",
        "tuple(status)",
        (
            "(status.invalid_shifted_grid, status.nonfinite_floor, "
            "status.coverage_missing)"
        ),
        1,
    ),
    "owned_query_status:keyed_flatten_floor_grid_swapped": (
        _FEASIBILITY,
        "_flatten_query_input_status_with_keys",
        "expression",
        "zip(QueryInputStatus._fields, status, strict=True)",
        (
            "zip(QueryInputStatus._fields, (status.invalid_shifted_grid, "
            "status.nonfinite_floor, status.coverage_missing), strict=True)"
        ),
        1,
    ),
    "owned_query_status:unflatten_floor_grid_swapped": (
        _FEASIBILITY,
        "_unflatten_query_input_status",
        "expression",
        "tuple.__new__(QueryInputStatus, leaves)",
        "tuple.__new__(QueryInputStatus, (leaves[1], leaves[0], leaves[2]))",
        1,
    ),
    "owned_query_status:reducer_inactive_lanes": (
        _FEASIBILITY,
        "reduce_query_input_status",
        "expression",
        "active & flag",
        "~active & flag",
        1,
    ),
    "owned_query_status:merge_requires_both_windows": (
        _FEASIBILITY,
        "merge_query_input_status",
        "expression",
        "a | b",
        "a & b",
        1,
    ),
    "owned_query_status:validator_arithmetic_gate_removed": (
        _FEASIBILITY,
        "validate_query_input_status",
        "expression",
        "nonfinite or invalid_grid",
        "False",
        1,
    ),
    "owned_query_status:validator_coverage_gate_removed": (
        _FEASIBILITY,
        "validate_query_input_status",
        "expression",
        "missing",
        "False",
        1,
    ),
    "owned_query_status:block_slot_count_ignored": (
        _BACKWARD,
        "_query_block_status",
        "expression",
        "len(statuses or ()) != n_query_slots",
        "False",
        1,
    ),
    "owned_query_authority:topology_schema_not_checked": (
        _CONTINUATION,
        "EGMRowTopologySpec.__post_init__",
        "expression",
        "self.schema_version != 1",
        "False",
        1,
    ),
    "owned_query_authority:topology_exact_type_not_checked": (
        _CONTINUATION,
        "_validate_row_topology",
        "expression",
        "type(topology) is not EGMRowTopologySpec",
        "False",
        1,
    ),
    "owned_query_authority:topology_authority_not_compared": (
        _CONTINUATION,
        "_validate_row_topology_authority",
        "expression",
        "not same",
        "False",
        1,
    ),
    "owned_query_authority:dc_later_periods_omitted": (
        _PROCESSING,
        "_prepare_dcegm_query_models",
        "keyword",
        "target_periods",
        "(periods[0],)",
        1,
    ),
}
_MUTATION_NAMES = (*_OWNED_QUERY_HELPER_MUTATIONS, _PRIORITY_MUTATION)


@pytest.mark.parametrize("mutation", _MUTATION_NAMES)
def test_owned_query_body_change_adds_specific_diagnostic(
    *, mutation: str, tmp_path: Path
) -> None:
    """Reject each owned helper corruption with its own additional diagnostic."""
    root = Path(__file__).parents[1]
    sources = direct_flow.verify_direct_candidate_flow(repo_root=root)[
        "certified_corridor_sources"
    ]
    relative, helper, mutated_source = _create_owned_helper_mutant(
        repo_root=root, mutation=mutation
    )
    _assert_single_body_change(
        original=(root / relative).read_text(),
        mutated=mutated_source,
        helper=helper,
    )
    assert relative in sources
    for source in sources:
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / source, destination)
    baseline = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    diagnostic = (
        f"S02 query/status dependency: exact callable corridor {helper!r} changed"
    )
    assert diagnostic not in baseline["errors"], {
        "unmodified_helper_must_lack_same_diagnostic": diagnostic,
        "baseline_errors": baseline["errors"],
    }
    (tmp_path / relative).write_text(mutated_source, encoding="utf-8")
    mutated = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert diagnostic in mutated["errors"], {
        "required_additional_diagnostic": diagnostic,
        "baseline_errors": baseline["errors"],
        "mutated_errors": mutated["errors"],
    }


def _create_owned_helper_mutant(
    *, repo_root: Path, mutation: str
) -> tuple[str, str, str]:
    """Construct one uniquely anchored body change in a separate control family."""
    if mutation != _PRIORITY_MUTATION:
        spec = _OWNED_QUERY_HELPER_MUTATIONS[mutation]
        changed = direct_flow._callable_mutation_specs(
            repo_root=repo_root, mutations={mutation: spec}
        )[mutation]
        return spec[0], spec[1], changed["source"]
    tree = ast.parse((repo_root / _FEASIBILITY).read_text())
    function = direct_flow._definition(tree=tree, name="validate_query_input_status")
    loop_indices = [
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.For)
    ]
    assert len(loop_indices) == 2
    first, second = loop_indices
    function.body[first], function.body[second] = (
        function.body[second],
        function.body[first],
    )
    return _FEASIBILITY, "validate_query_input_status", ast.unparse(tree) + "\n"


def _get_owned_helper(*, tree: ast.Module, helper: str) -> ast.FunctionDef:
    """Resolve one exact top-level function or class method by its owner name."""
    if "." in helper:
        class_name, method_name = helper.split(".", maxsplit=1)
        return direct_flow._method_definition(
            tree=tree, class_name=class_name, method_name=method_name
        )[1]
    return direct_flow._definition(tree=tree, name=helper)


def _assert_single_body_change(*, original: str, mutated: str, helper: str) -> None:
    """Require a live body mutation that changes no other executable AST node."""
    original_tree = ast.parse(original)
    mutated_tree = ast.parse(mutated)
    original_function = _get_owned_helper(tree=original_tree, helper=helper)
    mutated_function = _get_owned_helper(tree=mutated_tree, helper=helper)
    original_body = ast.dump(ast.Module(body=original_function.body, type_ignores=[]))
    mutated_body = ast.dump(ast.Module(body=mutated_function.body, type_ignores=[]))
    assert original_body != mutated_body
    mutated_function.body = copy.deepcopy(original_function.body)
    assert ast.dump(original_tree) == ast.dump(mutated_tree)
