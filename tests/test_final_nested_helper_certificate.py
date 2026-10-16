"""Keep nested helper bodies, bindings and overload selection explicit."""

import ast
import copy
import hashlib
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.nested_helper_selection import (
    select_nbegm_query_read_choices,
)

_ROOT = Path(__file__).parents[1]

_BODY_OWNERS = {
    "src/_lcm/egm/nested_floor.py": (
        "prepare_nested_floor",
        "nested_floor_at_cell",
    ),
    "src/_lcm/egm/finite_negm_query.py": (
        "read_finite_outer_choices",
        "_single_row_finite_corner_read",
        "_read_finite_resource_gradient",
        "_read_finite_corner_row",
        "_read_finite_native_row",
        "_read_finite_owned_side",
        "_validate_finite_outer_rows",
        "_filter_finite_survivors",
        "_select_finite_outer_profiles",
        "_collapse_finite_selection",
    ),
    "src/_lcm/egm/adaptive_mesh_authority.py": (
        "admit_continuation_mesh",
        "validate_continuation_mesh",
        "validate_bound_continuation_mesh",
    ),
    "src/_lcm/egm/adaptive_nbegm_carry.py": (
        "adaptive_nbegm_template",
        "assemble_adaptive_nbegm_carry",
    ),
    "src/_lcm/egm/adaptive_nbegm_producer.py": (
        "_tuple_roles",
        "_floor_template",
        "PreparedFloorArguments.__call__",
        "with_keeper_floor",
        "prepare_nested_inner_kernels",
        "fold_executed_summaries",
    ),
    "src/_lcm/egm/adaptive_nbegm_query.py": (
        "_numeric_operand",
        "_Profile.read",
        "_Profile.interpolate",
        "_Profile.status_at",
        "_take",
        "read_adaptive_choices",
        "_read_one_choice",
        "_validate_tuple",
    ),
    "src/_lcm/egm/nbegm_query_model.py": ("derive_adaptive_inner_models",),
    "src/_lcm/egm/nbegm_query.py": (
        "build_nbegm_query",
        "NBEGMQuery.capabilities",
        "NBEGMQuery.input_names",
        "NBEGMQuery.evaluate_floor",
        "NBEGMQuery.read_prepared",
        "NBEGMQuery.read_prepared_with_counts",
        "NBEGMQuery._read_at_floor_with_summary",
        "NBEGMQuery._read_rows_without_counts",
        "NBEGMQuery._read_rows",
        "NBEGMQuery._read_rows_with_counts",
        "_read_nb_compact_corner_rows",
        "_read_nb_compact_corner",
        "NBEGMQuery.read_choices",
    ),
    "src/_lcm/egm/carry.py": ("_get_owned_local_piece",),
    "src/_lcm/egm/nbegm_step.py": ("_single_row_nbegm_query_step",),
    "src/_lcm/solution/negm.py": (
        "NEGM.build_period_kernels",
        "_derive_negm_inner_query_model",
        "_build_negm_floor_preparation",
        "_durable_values_at",
        "_NEGMPeriodKernel.__post_init__",
        "_NEGMPeriodKernel.__call__",
        "_nested_feasibility_programs",
        "_nested_tuple_roles",
        "_stacked_nested_carry_roles",
        "_stacked_nested_carry_role",
        "_keeper_with_prepared_floor",
        "_run_nested_feasibility_kernel",
        "_nested_outer_sweep_program",
        "_count_nested_current_feasibility",
        "_NestedNodeSolver.__call__",
        "_NEGMSweepArgumentBuilder.__call__",
        "_periodized_function_groups",
        "_with_declared_keeper_reads",
    ),
    "src/_lcm/solution/nnbegm.py": (
        "NNBEGM.build_period_kernels",
        "_conditional_nnbegm_banks",
        "_NestedArgumentBuilder.__call__",
        "_NNBEGMPeriodKernel.__post_init__",
        "_NNBEGMPeriodKernel.__call__",
        "_NNBEGMPeriodKernel._solve_keeper",
        "_NNBEGMPeriodKernel._solve_adjuster_node",
        "_FiniteNNBEGMPeriodKernel._solve_outer",
        "_AdaptiveNNBEGMPeriodKernel._solve_outer",
        "_role_carry_reads",
        "_AdaptiveNodeSolver.__call__",
    ),
    "src/_lcm/solution/dcegm.py": (
        "_counted_dcegm_output_roles",
        "_with_nested_feasibility_role",
        "_DCEGMPeriodKernel.retains_current_feasibility",
        "_strip_nested_feasibility",
    ),
    "src/_lcm/solution/nbegm.py": (
        "_ride_along_output_roles",
        "_nbegm_bounded_output_roles",
        "_with_nbegm_scalar_output_roles",
        "_with_nbegm_scalar_carry_reads",
        "_NBEGMOwnedSingleAxisCore.__call__",
    ),
    "src/_lcm/solution/feasibility.py": (
        "reduce_continuation_read_counts",
        "merge_continuation_read_counts",
    ),
    "src/_lcm/egm/query_data.py": ("select_adaptive_query_data",),
    "src/_lcm/regime_building/processing.py": (
        "_prepare_nbegm_query_models",
        "_canonical_constraint_routing_inputs",
    ),
    "src/_lcm/egm/outer_candidates.py": (
        "build_outer_candidate_bank",
        "_stack_leaves",
        "OuterCandidateBank.n_candidates",
        "OuterCandidateBank.candidate_v_arr",
        "OuterCandidateBank.candidate_carry",
    ),
}


def _implementation(
    *, tree: ast.Module, source: direct_flow.SourcePath, qualname: str
) -> ast.FunctionDef:
    """Select the declared runtime function without choosing an overload stub."""
    if (
        source == "src/_lcm/egm/nbegm_query.py"
        and qualname == "NBEGMQuery.read_choices"
    ):
        return select_nbegm_query_read_choices(tree=tree, checker=direct_flow)
    if "." in qualname:
        class_name, method_name = qualname.split(".", maxsplit=1)
        return direct_flow._method_definition(
            tree=tree, class_name=class_name, method_name=method_name
        )[1]
    return direct_flow._definition(tree=tree, name=qualname)


def test_final_nested_selection_names_each_body_and_surface() -> None:
    """Each literal dependency has exactly one canonical stored selection."""
    actual = {
        source: tuple(callables)
        for source, (_surface, callables) in direct_flow._FINAL_NESTED_CONTRACTS.items()
    }
    assert actual == _BODY_OWNERS
    assert sum(map(len, actual.values())) == 95
    assert all("_three_jets" not in names for names in actual.values())
    assert direct_flow._unselected_pins() == ()
    assert all(source in direct_flow._CERTIFIED_CORRIDOR_SOURCES for source in actual)


@pytest.mark.parametrize(
    ("source", "qualname"),
    [(source, name) for source, names in _BODY_OWNERS.items() for name in names],
    ids=[
        f"{source}::{name}" for source, names in _BODY_OWNERS.items() for name in names
    ],
)
def test_final_nested_body_requires_its_owned_error_after_reseal(
    *,
    source: direct_flow.SourcePath,
    qualname: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed helper body is rejected by its owned corridor after byte reseal."""
    clean = direct_flow.verify_direct_candidate_flow(repo_root=_ROOT)
    assert clean["ok"]
    assert clean["errors"] == []
    assert clean["offending_paths"] == []
    originals = {}
    for relative in clean["certified_corridor_sources"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(_ROOT / relative, destination)
        originals[relative] = destination.read_bytes()
    assert direct_flow.verify_direct_candidate_flow(repo_root=tmp_path) == clean
    text = originals[source].decode("utf-8")
    tree = ast.parse(text)
    function = _implementation(tree=tree, source=source, qualname=qualname)
    statements = function.body
    if (
        isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]
    assert statements
    lines = text.splitlines(keepends=True)
    index = statements[0].lineno - 1
    indentation = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    lines.insert(index, f"{indentation}return None\n")
    mutated = "".join(lines)
    assert ast.dump(ast.parse(mutated)) != ast.dump(tree)
    target = tmp_path / source
    target.write_text(mutated, encoding="utf-8")
    monkeypatch.setitem(
        direct_flow._SOURCE_SEALS,
        source,
        hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert not result["ok"]
    assert result["errors"] == [
        f"final nested dependency: exact callable corridor {qualname!r} changed"
    ]
    assert result["offending_paths"] == [source]
    assert not any("source seal mismatch" in error for error in result["errors"])
    assert all(
        (tmp_path / relative).read_bytes() == original
        for relative, original in originals.items()
        if relative != source
    )
    target.write_bytes(originals[source])
    monkeypatch.setitem(
        direct_flow._SOURCE_SEALS, source, hashlib.sha256(originals[source]).hexdigest()
    )
    assert direct_flow.verify_direct_candidate_flow(repo_root=tmp_path) == clean


@pytest.mark.parametrize("source", tuple(_BODY_OWNERS))
def test_final_nested_import_binding_requires_its_owned_surface_error(
    *, source: direct_flow.SourcePath
) -> None:
    """Redirecting a dependency binding changes the complete module surface."""
    tree = ast.parse((_ROOT / source).read_text(encoding="utf-8"))
    assert direct_flow._final_nested_errors(tree=tree, source=source) == []
    altered = copy.deepcopy(tree)
    imports = [
        node for node in altered.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert imports
    imports[0].names[0].asname = "_final_nested_redirected_binding"
    errors = direct_flow._final_nested_errors(tree=altered, source=source)
    assert f"final nested dependency: {source} declaration surface changed" in errors
    assert direct_flow._transport_module_surface(tree) != (
        direct_flow._transport_module_surface(altered)
    )


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
def test_nbegm_query_method_refuses_changed_overload_declarations(
    *, mutation: str
) -> None:
    """The runtime method is selected only beside its exact ordered declarations."""
    tree = ast.parse(
        (_ROOT / "src/_lcm/egm/nbegm_query.py").read_text(encoding="utf-8")
    )
    runtime = select_nbegm_query_read_choices(tree=tree, checker=direct_flow)
    cls = direct_flow._class_definition(tree=tree, name="NBEGMQuery")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "read_choices"
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
        cls.body.extend(ast.parse("read_choices = None").body)
    elif mutation == "extra_class_binding":
        tree.body.extend(ast.parse("NBEGMQuery = None").body)
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
        ValueError, match=r"NBEGMQuery.read_choices overload declarations changed"
    ):
        select_nbegm_query_read_choices(tree=tree, checker=direct_flow)
