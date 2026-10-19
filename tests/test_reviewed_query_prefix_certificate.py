"""Authenticate owned conditional row arithmetic and checked output transport."""

import ast
import copy
import hashlib
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow

type SourceText = str

type MutationSpec = tuple[str, str, str, str, str, int]

_INDEPENDENT_CONTROLS: dict[str, MutationSpec] = {
    "reviewed_prefix:row_poison_not_legal": (
        "src/_lcm/egm/dcegm_query.py",
        "DCEGMQuery._read_at_floor_with_summary",
        "expression",
        "row.supported & ~row.invalid_numeric",
        "row.supported",
        1,
    ),
    "reviewed_prefix:inactive_poison_not_reduced": (
        "src/_lcm/egm/dcegm_query.py",
        "_blend_owned_rows",
        "expression",
        "jnp.any(active & result.invalid_numeric)",
        "jnp.any(result.invalid_numeric)",
        1,
    ),
    "reviewed_prefix:compact_corner_owner_retained": (
        "src/_lcm/egm/dcegm_query.py",
        "_read_compact_corner",
        "expression",
        "known & owned",
        "known",
        1,
    ),
    "reviewed_prefix:zero_passive_factor_dead": (
        "src/_lcm/egm/dcegm_query.py",
        "_passive_weights",
        "expression",
        "~is_represented_zero(axis_weights).reshape(shape)",
        "jnp.ones_like(axis_weights, dtype=jnp.bool_).reshape(shape)",
        1,
    ),
    "reviewed_prefix:product_jvp_keeps_other_factors": (
        "src/_lcm/egm/dcegm_query.py",
        "_factor_product_jvp",
        "expression",
        "_factor_product(others) * factor_tangents[index]",
        "factor_tangents[index]",
        1,
    ),
    "reviewed_prefix:prepared_floor_not_recomputed": (
        "src/_lcm/egm/dcegm_query.py",
        "DCEGMQuery.read_prepared",
        "keyword",
        "floor_state",
        "self.evaluate_floor(pool=pool)",
        1,
    ),
    "reviewed_prefix:row_link_same_owner": (
        "src/_lcm/egm/carry.py",
        "_read_owned_row_with_support",
        "expression",
        "pair_link & (data.owner_id[lower] == data.owner_id[upper])",
        "pair_link",
        1,
    ),
    "reviewed_prefix:economic_rows_stay_live": (
        "src/_lcm/egm/carry.py",
        "_read_owned_row_with_support",
        "expression",
        "finite | economic",
        "finite",
        1,
    ),
    "reviewed_prefix:associated_saving_not_value": (
        "src/_lcm/egm/carry.py",
        "_read_owned_channels",
        "expression",
        "read(fp=savings[indices])",
        "read(fp=value[indices])",
        1,
    ),
    "reviewed_prefix:floor_axes_not_row_axes": (
        "src/_lcm/egm/query_data.py",
        "select_query_data",
        "expression",
        "positions if floor_positions is None else floor_positions",
        "positions",
        1,
    ),
    "reviewed_prefix:literal_target_period_not_source": (
        "src/_lcm/solution/egm.py",
        "_LiteralDCResourceTransport._at_state",
        "expression",
        "jnp.int32(self.target_period)",
        "jnp.int32(0)",
        1,
    ),
    "reviewed_prefix:literal_resource_chain_once": (
        "src/_lcm/solution/egm.py",
        "_LiteralDCResourceTransport.__call__",
        "expression",
        "gradients * marginal_return",
        "marginal_return",
        1,
    ),
    "reviewed_prefix:literal_valuation_in_resources": (
        "src/_lcm/solution/egm.py",
        "_EGMCore._value_reconstructed_action",
        "expression",
        "resource_landing",
        "valuation_landing",
        1,
    ),
    "reviewed_prefix:inactive_flow_operand_safe": (
        "src/_lcm/solution/egm.py",
        "_EGMCore._value_reconstructed_action",
        "expression",
        "jnp.where(rescore, step.consumption, 1.0)",
        "step.consumption",
        1,
    ),
    "reviewed_prefix:same_child_slot_activity": (
        "src/_lcm/solution/egm.py",
        "_read_one_row_dc_child",
        "expression",
        "reduce_query_input_status(statuses=(status,), active=activity)",
        (
            "reduce_query_input_status(statuses=(status,), "
            "active=jnp.ones(liquid.shape, dtype=jnp.bool_))"
        ),
        1,
    ),
    "reviewed_prefix:literal_child_status_preserved": (
        "src/_lcm/egm/step_core.py",
        "_with_literal_child_status",
        "keyword",
        "query_status",
        "None",
        1,
    ),
    "reviewed_prefix:asset_child_status_preserved": (
        "src/_lcm/egm/asset_row.py",
        "_with_asset_child_status",
        "keyword",
        "query_status",
        "None",
        1,
    ),
    "reviewed_prefix:stochastic_dead_node_not_counted": (
        "src/_lcm/egm/continuation.py",
        "_WeightedNodeBlockFold.__call__",
        "keyword",
        "active",
        "jnp.ones_like(block_weights, dtype=jnp.bool_)",
        1,
    ),
    "reviewed_prefix:count_pair_order": (
        "src/_lcm/solution/feasibility.py",
        "_flatten_continuation_read_counts",
        "expression",
        "(status.below_first_live, status.eligible)",
        "(status.eligible, status.below_first_live)",
        1,
    ),
    "reviewed_prefix:optional_count_aux_inert": (
        "src/_lcm/solution/feasibility.py",
        "_flatten_bounded_result",
        "expression",
        "('continuation_read_counts', result.query_status is not None)",
        "(type(result), result.query_status is not None)",
        1,
    ),
    "reviewed_prefix:checked_exact_bool": (
        "src/_lcm/execution/core_program.py",
        "select_programs",
        "expression",
        "type(collect_continuation_read_counts) is not bool",
        "False",
        1,
    ),
    "reviewed_prefix:checked_same_labels": (
        "src/_lcm/execution/core_program.py",
        "_validate_checked_variant",
        "expression",
        "set(checked) != set(ordinary)",
        "False",
        1,
    ),
    "reviewed_prefix:checked_fixed_params_both": (
        "src/_lcm/execution/core_program.py",
        "bind_core_program_fixed_params",
        "expression",
        "functools.partial(variant.function, **params)",
        "variant.function",
        1,
    ),
    "reviewed_prefix:locator_exact_known_container": (
        "src/_lcm/execution/internal_outputs.py",
        "_select_path",
        "expression",
        "type(node) is BoundedResult and step == 'output'",
        "step == 'output'",
        1,
    ),
    "reviewed_prefix:dc_resources_not_identity": (
        "src/_lcm/egm/regime_introspection.py",
        "_get_child_resources",
        "expression",
        "_as_dcegm(user_regime) is not None",
        "False",
        1,
    ),
    "reviewed_prefix:canonical_combo_permutation": (
        "src/_lcm/egm/step.py",
        "_reorder_combo_axes",
        "expression",
        "jnp.transpose(arr, (*perm, *trailing))",
        "arr",
        1,
    ),
    "reviewed_prefix:dc_fixed_params_reach_pair": (
        "src/_lcm/solution/dcegm.py",
        "_DCEGMPeriodKernel.with_fixed_params",
        "expression",
        "bind_core_program_fixed_params(program=program, params=egm_fixed)",
        "program",
        1,
    ),
}

_EXPECTED_DIAGNOSTICS = {
    "reviewed_prefix:row_poison_not_legal": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor 'DCEGMQuery._read_at_floor_with_summary' changed"
    ),
    "reviewed_prefix:inactive_poison_not_reduced": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_blend_owned_rows' changed"
    ),
    "reviewed_prefix:compact_corner_owner_retained": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_read_compact_corner' changed"
    ),
    "reviewed_prefix:zero_passive_factor_dead": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_passive_weights' changed"
    ),
    "reviewed_prefix:product_jvp_keeps_other_factors": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_factor_product_jvp' changed"
    ),
    "reviewed_prefix:prepared_floor_not_recomputed": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor 'DCEGMQuery.read_prepared' changed"
    ),
    "reviewed_prefix:row_link_same_owner": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_read_owned_row_with_support' changed"
    ),
    "reviewed_prefix:economic_rows_stay_live": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_read_owned_row_with_support' changed"
    ),
    "reviewed_prefix:associated_saving_not_value": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_read_owned_channels' changed"
    ),
    "reviewed_prefix:floor_axes_not_row_axes": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor 'select_query_data' changed"
    ),
    "reviewed_prefix:literal_target_period_not_source": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_LiteralDCResourceTransport._at_state' changed"
    ),
    "reviewed_prefix:literal_resource_chain_once": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_LiteralDCResourceTransport.__call__' changed"
    ),
    "reviewed_prefix:literal_valuation_in_resources": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_EGMCore._value_reconstructed_action' changed"
    ),
    "reviewed_prefix:inactive_flow_operand_safe": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_EGMCore._value_reconstructed_action' changed"
    ),
    "reviewed_prefix:same_child_slot_activity": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_read_one_row_dc_child' changed"
    ),
    "reviewed_prefix:literal_child_status_preserved": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_with_literal_child_status' changed"
    ),
    "reviewed_prefix:asset_child_status_preserved": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_with_asset_child_status' changed"
    ),
    "reviewed_prefix:stochastic_dead_node_not_counted": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_WeightedNodeBlockFold.__call__' changed"
    ),
    "reviewed_prefix:count_pair_order": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_flatten_continuation_read_counts' changed"
    ),
    "reviewed_prefix:optional_count_aux_inert": (
        "S02 query/status dependency: "
        "exact callable corridor '_flatten_bounded_result' changed"
    ),
    "reviewed_prefix:checked_exact_bool": (
        "core-program provider-to-resolver transport: "
        "exact callable corridor 'select_programs' changed"
    ),
    "reviewed_prefix:checked_same_labels": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_validate_checked_variant' changed"
    ),
    "reviewed_prefix:checked_fixed_params_both": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor 'bind_core_program_fixed_params' changed"
    ),
    "reviewed_prefix:locator_exact_known_container": (
        "reviewed S04/S05 dependency: exact callable corridor '_select_path' changed"
    ),
    "reviewed_prefix:dc_resources_not_identity": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_get_child_resources' changed"
    ),
    "reviewed_prefix:canonical_combo_permutation": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_reorder_combo_axes' changed"
    ),
    "reviewed_prefix:dc_fixed_params_reach_pair": (
        "reviewed S04/S05 dependency: "
        "exact callable corridor '_DCEGMPeriodKernel.with_fixed_params' changed"
    ),
    "reviewed_prefix:model_authority_immutable": (
        "reviewed S04/S05 dependency: "
        "src/_lcm/egm/dcegm_query_model.py declaration surface changed"
    ),
}


def test_reviewed_prefix_controls_preserve_exact_named_specimens() -> None:
    """The canonical population matches independently declared semantic defects."""
    root = Path(__file__).parents[1]
    expected = direct_flow._callable_mutation_specs(
        repo_root=root, mutations=_INDEPENDENT_CONTROLS
    )
    relative = "src/_lcm/egm/dcegm_query_model.py"
    source = (root / relative).read_text(encoding="utf-8")
    marker = "@dataclass(frozen=True, kw_only=True)"
    assert source.count(marker) == 1
    expected["reviewed_prefix:model_authority_immutable"] = {
        "path": relative,
        "source": source.replace(marker, "@dataclass(frozen=False, kw_only=True)", 1),
    }
    actual = direct_flow.reviewed_prefix_mutation_specs(repo_root=root)
    assert actual == expected
    assert len(actual) == direct_flow.EXPECTED_REVIEWED_PREFIX_MUTATION_COUNT == 28
    names = "\n".join(sorted(actual)) + "\n"
    expected_digest = "71a70a218ef33bcd2ac4506902d63d3a4fb1f5a51a021a56cb4ccc9f691726cf"
    assert hashlib.sha256(names.encode()).hexdigest() == expected_digest
    assert expected_digest == direct_flow.EXPECTED_REVIEWED_PREFIX_MUTATION_NAMES_SHA256
    assert direct_flow._REVIEWED_PREFIX_DIAGNOSTICS == _EXPECTED_DIAGNOSTICS


@pytest.mark.parametrize("mutation", tuple(_INDEPENDENT_CONTROLS))
def test_reviewed_prefix_mutation_adds_owned_diagnostic(
    *, mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owned semantic defect fails its own obligation after byte resealing."""
    root = Path(__file__).parents[1]
    relative, helper, *_ = _INDEPENDENT_CONTROLS[mutation]
    source = (root / relative).read_text(encoding="utf-8")
    mutated = direct_flow._callable_mutation_specs(
        repo_root=root, mutations={mutation: _INDEPENDENT_CONTROLS[mutation]}
    )[mutation]["source"]
    _assert_one_owned_body_changed(source=source, mutated=mutated, helper=helper)
    sources = direct_flow.verify_direct_candidate_flow(repo_root=root)[
        "certified_corridor_sources"
    ]
    assert relative in sources
    for path in sources:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / path, destination)
    baseline = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert baseline["ok"], baseline["errors"]
    diagnostic = _EXPECTED_DIAGNOSTICS[mutation]
    assert diagnostic not in baseline["errors"]
    target = tmp_path / relative
    target.write_text(mutated, encoding="utf-8")
    monkeypatch.setitem(
        direct_flow._SOURCE_SEALS, relative, direct_flow.sha256_file(target)
    )

    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)

    assert not result["ok"]
    assert diagnostic in result["errors"], result["errors"]
    assert not any("source seal mismatch" in error for error in result["errors"])


def _assert_one_owned_body_changed(
    *, source: SourceText, mutated: SourceText, helper: str
) -> None:
    """The specimen changes its declared helper body and no other source AST."""
    original_tree = ast.parse(source)
    mutated_tree = ast.parse(mutated)

    def locate(tree: ast.Module) -> ast.FunctionDef:
        scope = tree.body
        if "." in helper:
            class_name, name = helper.split(".", maxsplit=1)
            classes = [
                node
                for node in scope
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ]
            assert len(classes) == 1
            scope = classes[0].body
        else:
            name = helper
        functions = [
            node
            for node in scope
            if isinstance(node, ast.FunctionDef)
            and node.name == name
            and not any(
                isinstance(decorator, ast.Name) and decorator.id == "overload"
                for decorator in node.decorator_list
            )
        ]
        assert len(functions) == 1
        return functions[0]

    original_function = locate(original_tree)
    mutated_function = locate(mutated_tree)
    assert ast.dump(original_function) != ast.dump(mutated_function)
    mutated_function.body = copy.deepcopy(original_function.body)
    assert ast.dump(original_tree) == ast.dump(mutated_tree)


def test_reviewed_model_authority_declaration_is_immutable(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model authority and offsets remain bound to one immutable evaluator pool."""
    root = Path(__file__).parents[1]
    relative = "src/_lcm/egm/dcegm_query_model.py"
    source = (root / relative).read_text(encoding="utf-8")
    marker = "@dataclass(frozen=True, kw_only=True)"
    assert source.count(marker) == 1
    mutated = source.replace(marker, "@dataclass(frozen=False, kw_only=True)", 1)
    sources = direct_flow.verify_direct_candidate_flow(repo_root=root)[
        "certified_corridor_sources"
    ]
    assert relative in sources
    for path in sources:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / path, destination)
    baseline = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert baseline["ok"], baseline["errors"]
    target = tmp_path / relative
    target.write_text(mutated, encoding="utf-8")
    monkeypatch.setitem(
        direct_flow._SOURCE_SEALS, relative, direct_flow.sha256_file(target)
    )

    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)

    diagnostic = f"reviewed S04/S05 dependency: {relative} declaration surface changed"
    assert not result["ok"]
    assert diagnostic in result["errors"]
    assert not any("source seal mismatch" in error for error in result["errors"])
