"""Construct distinct nested body, declaration and exact-overload controls."""

import ast
import copy
from pathlib import Path
from types import ModuleType

try:
    from generate_sources import SourcePath
except ModuleNotFoundError:
    from tests.candidate_certificate.generate_sources import SourcePath


try:
    from nested_helper_selection import select_nbegm_query_read_choices
except ModuleNotFoundError:
    from tests.candidate_certificate.nested_helper_selection import (
        select_nbegm_query_read_choices,
    )

BODY_OWNERS = {
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
    "src/_lcm/egm/adaptive_nbegm_producer.py": (
        "_tuple_roles",
        "_floor_template",
        "PreparedFloorArguments.__call__",
        "with_keeper_floor",
        "prepare_nested_inner_kernels",
        "fold_executed_summaries",
    ),
    "src/_lcm/egm/nbegm_query_model.py": (),
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
    "src/_lcm/egm/query_data.py": (),
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

OVERLOAD_MUTATIONS = (
    "missing_stub",
    "extra_stub",
    "reordered_stubs",
    "changed_stub",
    "runtime_decorator",
    "extra_method_binding",
    "extra_class_binding",
    "overload_alias",
    "overload_rebinding",
)

_SURFACE_COOWNERS = {
    "src/_lcm/egm/carry.py": (
        "reviewed S04/S05 dependency",
        "S02 query/status dependency",
    ),
    "src/_lcm/egm/query_data.py": (
        "reviewed S04/S05 dependency",
        "S02 query/status dependency",
    ),
    "src/_lcm/solution/dcegm.py": ("reviewed S04/S05 dependency",),
    "src/_lcm/solution/feasibility.py": (
        "S03 scalar dependency",
        "reviewed S04/S05 dependency",
        "S02 query/status dependency",
    ),
    "src/_lcm/solution/nbegm.py": ("reviewed S04/S05 dependency",),
    "src/_lcm/regime_building/processing.py": (
        "S03 scalar dependency",
        "S02 query/status dependency",
    ),
    "src/_lcm/egm/nbegm_step.py": ("retained DC/NB dependency",),
}


def primary_diagnostic(*, name: str) -> str:
    """Return the required owned diagnostic of one literal control."""
    if name.startswith("final_nested:body:"):
        qualname = name.split("::", maxsplit=1)[1]
        return f"final nested dependency: exact callable corridor {qualname!r} changed"
    if name.startswith("final_nested:surface:"):
        source = name.removeprefix("final_nested:surface:")
        return f"final nested dependency: {source} declaration surface changed"
    if name.startswith("final_nested:overload:"):
        return (
            "final nested dependency: "
            "NBEGMQuery.read_choices overload declarations changed"
        )
    raise ValueError(f"Unknown nested control {name!r}")


def expected_errors(*, source: SourcePath, diagnostic: str) -> tuple[str, ...]:
    """Keep exact co-owned surface errors visible beside the required new owner."""
    surface_error = f"final nested dependency: {source} declaration surface changed"
    errors = [diagnostic]
    if diagnostic == surface_error:
        errors.extend(
            f"{owner}: {source} declaration surface changed"
            for owner in _SURFACE_COOWNERS.get(source, ())
        )
        if source in ("src/_lcm/egm/nbegm_query.py", "src/_lcm/solution/nbegm.py"):
            errors.append(
                f"additional helper dependency: {source} declaration surface changed"
            )
        if source == "src/_lcm/solution/nbegm.py":
            errors.append("NB-EGM donation declaration: module bindings changed")
    elif diagnostic == (
        "final nested dependency: NBEGMQuery.read_choices overload declarations changed"
    ):
        errors.append(surface_error)
    return tuple(sorted(errors))


def _implementation(
    *, tree: ast.Module, source: SourcePath, qualname: str, checker: ModuleType
) -> ast.FunctionDef:
    """Use the same strict runtime selector as the canonical verifier."""
    if (
        source == "src/_lcm/egm/nbegm_query.py"
        and qualname == "NBEGMQuery.read_choices"
    ):
        return select_nbegm_query_read_choices(tree=tree, checker=checker)
    if "." in qualname:
        class_name, method_name = qualname.split(".", maxsplit=1)
        return checker._method_definition(
            tree=tree, class_name=class_name, method_name=method_name
        )[1]
    return checker._definition(tree=tree, name=qualname)


def _alter_overload(*, tree: ast.Module, mutation: str, checker: ModuleType) -> None:
    """Change exactly the reviewed stub, binding or runtime decoration."""
    runtime = select_nbegm_query_read_choices(tree=tree, checker=checker)
    cls = checker._class_definition(tree=tree, name="NBEGMQuery")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "read_choices"
    ]
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
        aliases = [
            alias
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "typing"
            for alias in node.names
            if alias.name == "overload"
        ]
        if len(aliases) != 1:
            raise ValueError("The canonical typing overload import is not unique")
        aliases[0].asname = "changed_overload"
    elif mutation == "overload_rebinding":
        tree.body.extend(ast.parse("overload = None").body)
    else:
        raise ValueError(f"Unknown exact overload mutation {mutation!r}")


def final_nested_mutation_specs(
    *, repo_root: Path, checker: ModuleType
) -> dict[str, dict[str, str]]:
    """Construct all 80 body, 15 module-surface and 9 exact-overload controls."""
    specifications = {}
    for source, names in BODY_OWNERS.items():
        original = (repo_root / source).read_text(encoding="utf-8")
        for qualname in names:
            tree = ast.parse(original, filename=source)
            function = _implementation(
                tree=tree, source=source, qualname=qualname, checker=checker
            )
            original_body = checker._callable_ast_sha256(function)
            function.body = ast.parse("return None").body
            if checker._callable_ast_sha256(function) == original_body:
                raise ValueError(f"Nested body mutation did not alter {qualname!r}")
            name = f"final_nested:body:{source}::{qualname}"
            specifications[name] = {"path": source, "source": ast.unparse(tree) + "\n"}
        tree = ast.parse(original, filename=source)
        surface = checker._transport_module_surface(tree)
        tree.body.extend(ast.parse("import math as _final_nested_surface_probe").body)
        if checker._transport_module_surface(tree) == surface:
            raise ValueError(f"Nested surface mutation did not alter {source!r}")
        specifications[f"final_nested:surface:{source}"] = {
            "path": source,
            "source": ast.unparse(tree) + "\n",
        }
    source = "src/_lcm/egm/nbegm_query.py"
    original = (repo_root / source).read_text(encoding="utf-8")
    for mutation in OVERLOAD_MUTATIONS:
        tree = ast.parse(original, filename=source)
        _alter_overload(tree=tree, mutation=mutation, checker=checker)
        try:
            select_nbegm_query_read_choices(tree=tree, checker=checker)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"Nested overload mutation was not constructed: {mutation}"
            )
        specifications[f"final_nested:overload:{mutation}"] = {
            "path": source,
            "source": ast.unparse(tree) + "\n",
        }
    if len(specifications) != 104:
        raise ValueError("The declared final nested control population changed")
    return specifications
