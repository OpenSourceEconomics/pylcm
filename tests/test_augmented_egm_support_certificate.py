"""Authenticate augmented savings metadata and represented support semantics."""

import ast
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.test_reviewed_query_prefix_certificate import _assert_one_owned_body_changed

_STEP = "src/_lcm/egm/one_asset_egm_step.py"
_BUILDER = "src/_lcm/solution/egm.py"
type MutationSpec = tuple[str, str, str, str, str, int]

_CONTROLS: dict[str, MutationSpec] = {
    "inactive_not_canonical": (
        _STEP,
        "_prepare_augmented_savings_support",
        "expression",
        "active & ~jnp.concatenate((jnp.asarray([False]), same_read))",
        "~jnp.concatenate((jnp.asarray([False]), same_read))",
        1,
    ),
    "rank_before_economic_filter": (
        _STEP,
        "_prepare_augmented_savings_support",
        "expression",
        "jnp.cumsum(keep, dtype=jnp.int32) - 1",
        "jnp.cumsum(keep & jnp.isfinite(value_next), dtype=jnp.int32) - 1",
        1,
    ),
    "flatten_leaf_order": (
        _STEP,
        "_flatten_egm_savings_support",
        "expression",
        "tuple(support)",
        "(support.floor, support.candidate_id, support.active)",
        1,
    ),
    "keyed_leaf_order": (
        _STEP,
        "_flatten_egm_savings_support_with_keys",
        "expression",
        "_EGMSavingsSupport._fields",
        "tuple(reversed(_EGMSavingsSupport._fields))",
        1,
    ),
    "abstract_reconstruction": (
        _STEP,
        "_unflatten_egm_savings_support",
        "expression",
        "tuple.__new__(_EGMSavingsSupport, leaves)",
        "_EGMSavingsSupport(*leaves)",
        1,
    ),
    "reconstruction_arity": (
        _STEP,
        "_unflatten_egm_savings_support",
        "expression",
        "len(leaves) != len(_EGMSavingsSupport._fields)",
        "False",
        1,
    ),
    "activity_before_numeric_poison": (
        _STEP,
        "_solve_viable_feature_row",
        "expression",
        "savings_support.active",
        "jnp.ones_like(savings_support.active)",
        1,
    ),
    "original_floor_identity": (
        _STEP,
        "_solve_viable_feature_row",
        "expression",
        "jnp.argmax(savings_support.candidate_id == 0)",
        "jnp.argmax(savings_support.candidate_id == 1)",
        1,
    ),
    "none_keeps_ordinary_route": (
        _STEP,
        "egm_one_asset_step",
        "expression",
        "feature_floor or savings_support is not None",
        "True",
        1,
    ),
    "empty_bank_keeps_metadata_absent": (
        _BUILDER,
        "_EGMArgumentBuilder._law_readings",
        "expression",
        "self.augment_plain_savings and next_breakpoints.size > 0",
        "self.augment_plain_savings",
        1,
    ),
    "translated_original_nodes": (
        _BUILDER,
        "_EGMArgumentBuilder._law_readings",
        "expression",
        "saving_nodes if augmented else self.savings_grid",
        "self.savings_grid",
        1,
    ),
    "floor_not_sorted_tail": (
        _BUILDER,
        "_EGMArgumentBuilder._law_readings",
        "keyword",
        "floor",
        "saving_nodes[-1]",
        1,
    ),
    "none_omits_builder_keyword": (
        _BUILDER,
        "_EGMArgumentBuilder.__call__",
        "expression",
        "{'savings_support': savings_support} if savings_support is not None else {}",
        "{'savings_support': savings_support}",
        1,
    ),
    "activity_before_child_coverage": (
        _BUILDER,
        "_EGMCore._prepare_child_read",
        "expression",
        "jnp.any(savings_support.active & result.coverage_missing)",
        "jnp.any(result.coverage_missing)",
        1,
    ),
}

# Every represented channel participates in exact duplicate identity.
for _channel in (
    "savings_grid",
    "next_liquid",
    "marginal_return",
    "value_next",
    "marginal_next",
):
    _CONTROLS[f"duplicate_requires_{_channel}"] = (
        _STEP,
        "_prepare_augmented_savings_support",
        "expression",
        f"{_channel}[1:] == {_channel}[:-1]",
        "jnp.asarray(True)",
        1,
    )

_SURFACE_CONTROLS = {
    "strict_runtime_floor_type": ("    floor: ScalarFloat", "    floor: Float1D"),
    "nominal_tuple_base": (
        "class _EGMSavingsSupport(NamedTuple):",
        "class _EGMSavingsSupport(tuple):",
    ),
    "keyed_registration": (
        "    _flatten_egm_savings_support_with_keys,",
        "    _flatten_egm_savings_support,",
    ),
}


def test_augmented_support_has_exact_nominal_fields_and_keyed_registration() -> None:
    """Floor, activity and construction identity retain their associated keys."""
    source = (Path(__file__).parents[1] / _STEP).read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_EGMSavingsSupport"
    ]
    assert len(classes) == 1
    support = classes[0]
    assert [ast.unparse(base) for base in support.bases] == ["NamedTuple"]
    assert not support.decorator_list
    assert [
        (ast.unparse(node.target), ast.unparse(node.annotation))
        for node in support.body
        if isinstance(node, ast.AnnAssign)
    ] == [("floor", "ScalarFloat"), ("active", "Bool1D"), ("candidate_id", "Int1D")]
    registrations = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "jax.tree_util.register_pytree_with_keys"
        and node.value.args
        and ast.unparse(node.value.args[0]) == "_EGMSavingsSupport"
    ]
    assert len(registrations) == 1
    assert [ast.unparse(arg) for arg in registrations[0].args] == [
        "_EGMSavingsSupport",
        "_flatten_egm_savings_support_with_keys",
        "_unflatten_egm_savings_support",
        "_flatten_egm_savings_support",
    ]
    assert not registrations[0].keywords


@pytest.mark.parametrize("mutation", tuple(_CONTROLS))
def test_augmented_support_body_mutation_fails_after_byte_resealing(
    *, mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed support obligation fails its exact semantic callable corridor."""
    root = Path(__file__).parents[1]
    relative, helper, *_ = _CONTROLS[mutation]
    source = (root / relative).read_text(encoding="utf-8")
    mutated = direct_flow._callable_mutation_specs(
        repo_root=root, mutations={mutation: _CONTROLS[mutation]}
    )[mutation]["source"]
    _assert_one_owned_body_changed(source=source, mutated=mutated, helper=helper)
    diagnostic = f"S03 scalar dependency: exact callable corridor {helper!r} changed"
    _assert_resealed_mutant_fails(
        root=root,
        relative=relative,
        mutated=mutated,
        diagnostic=diagnostic,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize("mutation", tuple(_SURFACE_CONTROLS))
def test_augmented_support_surface_mutation_fails_after_byte_resealing(
    *, mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nominal fields and keyed registration remain semantic obligations."""
    root = Path(__file__).parents[1]
    source = (root / _STEP).read_text(encoding="utf-8")
    old, new = _SURFACE_CONTROLS[mutation]
    assert source.count(old) == 1
    mutated = source.replace(old, new, 1)
    _assert_resealed_mutant_fails(
        root=root,
        relative=_STEP,
        mutated=mutated,
        diagnostic=f"S03 scalar dependency: {_STEP} declaration surface changed",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


def _assert_resealed_mutant_fails(
    *,
    root: Path,
    relative: str,
    mutated: str,
    diagnostic: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean certified source tree gains its owned diagnostic after mutation."""
    original = direct_flow.verify_direct_candidate_flow(repo_root=root)
    assert original["ok"], original["errors"]
    sources = original["certified_corridor_sources"]
    assert relative in sources
    for path in sources:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / path, destination)
    baseline = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert baseline["ok"], baseline["errors"]
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
