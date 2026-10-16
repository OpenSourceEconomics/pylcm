"""Construct the declared helper body, surface and source sensitivity controls."""

import ast
from pathlib import Path
from types import ModuleType

try:
    from generate_sources import SourcePath
except ModuleNotFoundError:
    from tests.candidate_certificate.generate_sources import SourcePath

BODY_OWNERS: dict[SourcePath, tuple[str, ...]] = {
    "src/_lcm/egm/adaptive_nbegm_query.py": ("_fold_adaptive_choice",),
    "src/_lcm/egm/continuation.py": (
        "_select_feature_child_rows",
        "_build_feature_child_query",
    ),
    "src/_lcm/egm/finite_negm_query.py": ("_finite_outer_publication_inputs",),
    "src/_lcm/egm/nbegm_query.py": (
        "_read_scalar_nbegm_owned_row",
        "_read_scalar_nbegm_owned_row_with_support",
        "_with_scalar_nbegm_linear_value",
        "_validate_nbegm_query_recipe",
    ),
    "src/_lcm/egm/nbegm_step.py": ("_prepare_nbegm_interval_block",),
    "src/_lcm/egm/outer_refinement.py": ("_fold_exact_outer_nodes",),
    "src/_lcm/egm/step_core.py": (
        "_assemble_combo_candidates",
        "_mask_dead_combo_rows",
    ),
    "src/_lcm/execution/internal_outputs.py": (
        "assert_internal_inputs",
        "_published_signature",
        "_tree_signature",
        "_leaf_signature",
    ),
    "src/_lcm/model_processing.py": ("_validate_callable_borrowing_solver",),
    "src/_lcm/regime_building/processing.py": (
        "_nbegm_query_function_pools",
        "_nbegm_query_evaluator_targets",
        "_nbegm_evaluated_query_predicates",
        "_nbegm_adaptive_query_spec",
        "_solver_with_query_authority",
        "_nbegm_query_row_state_names",
    ),
    "src/_lcm/solution/dcegm.py": ("_dcegm_query_slots_and_capabilities",),
    "src/_lcm/solution/egm.py": ("_unpack_scalar_child_read",),
    "src/_lcm/solution/kernel_output.py": (
        "_validate_generated_continuation_authority",
    ),
    "src/_lcm/solution/model_authority.py": (
        "_validate_query_policy_budget_authority",
        "_bind_generated_continuation_authority",
        "_bind_declared_continuation_authority",
    ),
    "src/_lcm/solution/nbegm.py": (
        "_NBEGMScalarChildReads.read_rows",
        "_NBEGMScalarChildReads.__call__",
        "_prepare_nbegm_scalar_child_reads",
        "_NBEGMActionEligibility.__call__",
        "_NBEGMTabulatedEulerLaw.__call__",
        "_scalar_conditional_nbegm_carry",
        "_with_scalar_nbegm_sampled_run",
        "_record_nbegm_scalar_child_slots",
        "_nbegm_cell_continuation_columns",
        "_fold_nbegm_scalar_read_records",
        "_RideAlongNBEGMPeriodKernel._consume_output",
        "_validate_nbegm_integrity_slot_counts",
        "_bounded_nbegm_kernel_output",
    ),
    "src/_lcm/solution/nnbegm.py": (
        "_nnbegm_query_period_groups",
        "_prepare_nnbegm_query_inner_pair",
        "_publish_adaptive_nnbegm_query",
        "_nnbegm_outer_replay_function",
    ),
}
SURFACE_OWNERS = (
    "src/_lcm/egm/nbegm_query.py",
    "src/_lcm/execution/internal_outputs.py",
    "src/_lcm/solution/nbegm.py",
)
SEMANTIC_OWNERS = {
    "schedule_guard_removed": (
        "src/_lcm/solution/nbegm.py",
        "_with_scalar_nbegm_sampled_run",
    ),
    "pad_bridge": ("src/_lcm/solution/nbegm.py", "_with_scalar_nbegm_sampled_run"),
    "same_owner_false_link_bridge": (
        "src/_lcm/solution/nbegm.py",
        "_with_scalar_nbegm_sampled_run",
    ),
    "paired_marginal_changed": (
        "src/_lcm/egm/nbegm_query.py",
        "_with_scalar_nbegm_linear_value",
    ),
    "invalid_numeric_laundered": (
        "src/_lcm/solution/nbegm.py",
        "_NBEGMScalarChildReads.read_rows",
    ),
}


def primary_diagnostic(*, name: str) -> str:
    """Name the exact helper owner whose declaration or body was changed."""
    if name.startswith("additional_helper:body:"):
        qualname = name.split("::", maxsplit=1)[1]
    elif name.startswith("additional_helper:semantic:"):
        _, qualname = SEMANTIC_OWNERS[name.removeprefix("additional_helper:semantic:")]
    elif name.startswith("additional_helper:surface:"):
        source = name.removeprefix("additional_helper:surface:")
        return f"additional helper dependency: {source} declaration surface changed"
    else:
        raise ValueError(f"Unknown additional helper control {name!r}")
    return f"additional helper dependency: exact callable corridor {qualname!r} changed"


def expected_errors(*, source: SourcePath, diagnostic: str) -> tuple[str, ...]:
    """Require the new owner and every independently co-owned refusal."""
    errors = [diagnostic]
    if (
        diagnostic
        == f"additional helper dependency: {source} declaration surface changed"
    ):
        if source in ("src/_lcm/egm/nbegm_query.py", "src/_lcm/solution/nbegm.py"):
            errors.append(
                f"final nested dependency: {source} declaration surface changed"
            )
        if source in (
            "src/_lcm/execution/internal_outputs.py",
            "src/_lcm/solution/nbegm.py",
        ):
            errors.append(
                f"reviewed S04/S05 dependency: {source} declaration surface changed"
            )
        if source == "src/_lcm/solution/nbegm.py":
            errors.append("NB-EGM donation declaration: module bindings changed")
    return tuple(sorted(errors))


def _function(*, tree: ast.Module, name: str, checker: ModuleType) -> ast.FunctionDef:
    """Select one exact ordinary helper or one uniquely declared method."""
    if "." in name:
        class_name, method_name = name.split(".", maxsplit=1)
        _, function = checker._method_definition(
            tree=tree, class_name=class_name, method_name=method_name
        )
        return function
    return checker._definition(tree=tree, name=name)


def _sampled_mutation(*, function: ast.FunctionDef, mutation: str) -> None:
    """Alter one named guard, represented-row classification or adjacency check."""
    if mutation == "schedule_guard_removed":
        matches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "any((kind != 'continuous_kink' for kind in kinds))"
        ]
        if len(matches) != 1:
            raise ValueError("The scalar schedule guard is not uniquely represented")
        matches[0].test = ast.Constant(value=False)
    elif mutation == "pad_bridge":
        matches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "represented"
        ]
        if len(matches) != 1:
            raise ValueError("The scalar represented-row classification is not unique")
        matches[0].value = ast.parse(
            "jnp.ones_like(data.node_kind, dtype=jnp.bool_)", mode="eval"
        ).body
    elif mutation == "same_owner_false_link_bridge":
        matches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotEq)
            and ast.unparse(node.left) == "data.owner_id[..., :-1]"
            and ast.unparse(node.comparators[0]) == "data.owner_id[..., 1:]"
        ]
        if len(matches) != 1:
            raise ValueError("The scalar owner-boundary adjacency check is not unique")
        matches[0].ops = [ast.Eq()]
    else:
        raise ValueError(f"Unknown scalar sampled-row mutation {mutation!r}")


def _semantic_mutation(*, tree: ast.Module, mutation: str, checker: ModuleType) -> None:
    """Build a genuine source attack without claiming numerical qualification."""
    _, name = SEMANTIC_OWNERS[mutation]
    function = _function(tree=tree, name=name, checker=checker)
    before = checker._callable_ast_sha256(function)
    if mutation == "paired_marginal_changed":
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func) == "result._replace"
        ]
        if len(calls) != 1 or any(item.arg == "marginal" for item in calls[0].keywords):
            raise ValueError("The scalar paired-record replacement is not unique")
        calls[0].keywords.append(
            ast.keyword(
                arg="marginal", value=ast.parse("result.value", mode="eval").body
            )
        )
    elif mutation == "invalid_numeric_laundered":
        keywords = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.keyword)
            and node.arg == "query_status"
            and isinstance(node.value, ast.Name)
            and node.value.id == "statuses"
        ]
        if len(keywords) != 1:
            raise ValueError("The original query-status publication is not unique")
        keywords[0].value = ast.parse(
            "tuple(status._replace(nonfinite_floor=jnp.asarray(a=False), "
            "invalid_shifted_grid=jnp.asarray(a=False)) for status in statuses)",
            mode="eval",
        ).body
    else:
        _sampled_mutation(function=function, mutation=mutation)
    if checker._callable_ast_sha256(function) == before:
        raise ValueError(f"The declared source mutation did not alter {name!r}")


def additional_helper_mutation_specs(
    *, repo_root: Path, checker: ModuleType
) -> dict[str, dict[str, str]]:
    """Construct forty-six body, three surface and five source attack controls."""
    specifications: dict[str, dict[str, str]] = {}
    for source, names in BODY_OWNERS.items():
        original = (repo_root / source).read_text(encoding="utf-8")
        for name in names:
            tree = ast.parse(original, filename=source)
            function = _function(tree=tree, name=name, checker=checker)
            before = checker._callable_ast_sha256(function)
            function.body = ast.parse("return None").body
            if checker._callable_ast_sha256(function) == before:
                raise ValueError(f"The helper body mutation did not alter {name!r}")
            specifications[f"additional_helper:body:{source}::{name}"] = {
                "path": source,
                "source": ast.unparse(tree) + "\n",
            }
    for source in SURFACE_OWNERS:
        tree = ast.parse(
            (repo_root / source).read_text(encoding="utf-8"), filename=source
        )
        before = checker._transport_module_surface(tree)
        tree.body.extend(
            ast.parse("import math as _additional_helper_surface_probe").body
        )
        if checker._transport_module_surface(tree) == before:
            raise ValueError(f"The helper declaration surface did not change: {source}")
        specifications[f"additional_helper:surface:{source}"] = {
            "path": source,
            "source": ast.unparse(tree) + "\n",
        }
    for mutation, (source, _) in SEMANTIC_OWNERS.items():
        tree = ast.parse(
            (repo_root / source).read_text(encoding="utf-8"), filename=source
        )
        _semantic_mutation(tree=tree, mutation=mutation, checker=checker)
        specifications[f"additional_helper:semantic:{mutation}"] = {
            "path": source,
            "source": ast.unparse(tree) + "\n",
        }
    if len(specifications) != 54:
        raise ValueError("The declared additional helper population changed")
    return specifications
