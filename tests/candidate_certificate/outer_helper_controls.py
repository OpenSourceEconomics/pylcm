"""Explicit source mutations for the outer interpolation/search dependency closure."""

import ast
import copy
from pathlib import Path
from typing import Literal, NamedTuple

try:
    from generate_sources import SourcePath
except ModuleNotFoundError:
    from tests.candidate_certificate.generate_sources import SourcePath


type SourceText = str


INTERPOLATION = "src/_lcm/egm/outer_interpolation.py"
REFINEMENT = "src/_lcm/egm/outer_refinement.py"
GOLDEN = "src/_lcm/optimization/golden_section.py"

BODY_OWNERS = {
    INTERPOLATION: (
        "LocalCubicOuterInterpolant.evaluate",
        "LocalCubicOuterInterpolant.evaluate_with_derivative",
        "LocalCubicOuterInterpolant.evaluate_validity",
        "LocalCubicOuterInterpolant._evaluate",
        "_pad_state_axes",
        "_validate_nodes",
        "_node_slopes",
        "_four_point_slopes",
        "_three_point_slopes",
        "_one_sided_parabolic_slope",
        "_prefix_node_slopes",
        "_prefix_three_point_slopes",
        "_prefix_four_point_slopes",
    ),
    REFINEMENT: (
        "safeguarded_continuous_argmax",
        "_node_local_max_mask",
        "_consider",
        "_tie_band",
    ),
    GOLDEN: ("maximize_golden_section", "_probe", "_shrink_bracket"),
}

_BODY_CASES = (
    ("interpolation:evaluate", INTERPOLATION, "LocalCubicOuterInterpolant.evaluate"),
    (
        "interpolation:evaluate_derivative",
        INTERPOLATION,
        "LocalCubicOuterInterpolant.evaluate_with_derivative",
    ),
    (
        "interpolation:evaluate_validity",
        INTERPOLATION,
        "LocalCubicOuterInterpolant.evaluate_validity",
    ),
    (
        "interpolation:evaluate_core",
        INTERPOLATION,
        "LocalCubicOuterInterpolant._evaluate",
    ),
    ("interpolation:pad_axes", INTERPOLATION, "_pad_state_axes"),
    ("interpolation:validate_nodes", INTERPOLATION, "_validate_nodes"),
    ("interpolation:node_slopes", INTERPOLATION, "_node_slopes"),
    ("interpolation:four_point", INTERPOLATION, "_four_point_slopes"),
    ("interpolation:three_point", INTERPOLATION, "_three_point_slopes"),
    ("interpolation:one_sided", INTERPOLATION, "_one_sided_parabolic_slope"),
    ("interpolation:prefix_node", INTERPOLATION, "_prefix_node_slopes"),
    ("interpolation:prefix_three", INTERPOLATION, "_prefix_three_point_slopes"),
    ("interpolation:prefix_four", INTERPOLATION, "_prefix_four_point_slopes"),
    ("search:safeguarded", REFINEMENT, "safeguarded_continuous_argmax"),
    ("search:local_max", REFINEMENT, "_node_local_max_mask"),
    ("search:consider", REFINEMENT, "_consider"),
    ("search:tie_band", REFINEMENT, "_tie_band"),
    ("golden:maximize", GOLDEN, "maximize_golden_section"),
    ("golden:probe", GOLDEN, "_probe"),
    ("golden:shrink", GOLDEN, "_shrink_bracket"),
)


class OuterHelperControl(NamedTuple):
    """One finite owner-local mutation and its required semantic diagnostic."""

    source: SourcePath
    owner: str
    mode: Literal["body", "expression", "lexical", "import", "constant", "schema"]
    before: str
    after: str

    @property
    def diagnostic(self) -> str:
        """Name the actual body owner or declaration surface that must reject."""
        if self.owner == "MODULE_SURFACE":
            return f"outer helper dependency: {self.source} declaration surface changed"
        return (
            f"outer helper dependency: exact callable corridor {self.owner!r} changed"
        )


CONTROLS = {
    f"outer_helpers:body:{name}": OuterHelperControl(source, owner, "body", "", "")
    for name, source, owner in _BODY_CASES
}
CONTROLS.update(
    {
        "outer_helpers:prefix:capacity_domain": OuterHelperControl(
            INTERPOLATION,
            "LocalCubicOuterInterpolant._evaluate",
            "expression",
            "nodes[valid_count - 1]",
            "nodes[-1]",
        ),
        "outer_helpers:prefix:ordinary_slopes": OuterHelperControl(
            INTERPOLATION,
            "LocalCubicOuterInterpolant._evaluate",
            "expression",
            (
                "_prefix_node_slopes(nodes=nodes, values=values_f, "
                "valid_count=valid_count)"
            ),
            "_node_slopes(nodes=nodes, values=values_f)",
        ),
        "outer_helpers:prefix:cubic_early": OuterHelperControl(
            INTERPOLATION,
            "_prefix_node_slopes",
            "expression",
            "valid_count >= _CUBIC_STENCIL_SIZE",
            "True",
        ),
        "outer_helpers:prefix:nested_without_cubic": OuterHelperControl(
            INTERPOLATION,
            "_prefix_node_slopes",
            "lexical",
            "without_cubic",
            "return None",
        ),
        "outer_helpers:prefix:secant_reads_padding": OuterHelperControl(
            INTERPOLATION,
            "_prefix_three_point_slopes",
            "expression",
            "jnp.minimum(jnp.arange(nodes.shape[0] - 1), valid_count - 2)",
            "jnp.arange(nodes.shape[0] - 1)",
        ),
        "outer_helpers:prefix:endpoint_capacity": OuterHelperControl(
            INTERPOLATION,
            "_prefix_three_point_slopes",
            "expression",
            "jnp.arange(nodes.shape[0]) == valid_count - 1",
            "jnp.arange(nodes.shape[0]) == nodes.shape[0] - 1",
        ),
        "outer_helpers:prefix:cubic_unclamped": OuterHelperControl(
            INTERPOLATION,
            "_prefix_four_point_slopes",
            "expression",
            "jnp.clip(indices - 1, 0, valid_count - _CUBIC_STENCIL_SIZE)",
            "jnp.clip(indices - 1, 0, nodes.shape[0] - _CUBIC_STENCIL_SIZE)",
        ),
        "outer_helpers:search:pads_compete": OuterHelperControl(
            REFINEMENT,
            "safeguarded_continuous_argmax",
            "expression",
            "jnp.where(active_values, node_values, -jnp.inf)",
            "node_values",
        ),
        "outer_helpers:search:bracket_capacity": OuterHelperControl(
            REFINEMENT,
            "safeguarded_continuous_argmax",
            "expression",
            "jnp.clip(top_idx + 1, 0, valid_count - 1)",
            "jnp.clip(top_idx + 1, 0, n_nodes - 1)",
        ),
        "outer_helpers:search:objective_not_pinned": OuterHelperControl(
            REFINEMENT,
            "safeguarded_continuous_argmax",
            "expression",
            "active_objective(jnp.where(bracket_valid, query, nodes[0]))",
            "active_objective(query)",
        ),
        "outer_helpers:search:exact_nodes_omitted": OuterHelperControl(
            REFINEMENT,
            "safeguarded_continuous_argmax",
            "expression",
            (
                "_fold_exact_outer_nodes(nodes=nodes, finite_values=finite_va"
                "lues, state_shape=state_shape, n_nodes=n_nodes, node_compani"
                "ons=node_companions)"
            ),
            (
                "_fold_exact_outer_nodes(nodes=nodes, finite_values=finite_va"
                "lues, state_shape=state_shape, n_nodes=1, node_companions=no"
                "de_companions)"
            ),
        ),
        "outer_helpers:search:duplicate_runner_up": OuterHelperControl(
            REFINEMENT, "_consider", "expression", "cand_x != best_x", "True"
        ),
        "outer_helpers:golden:nan_not_sanitized": OuterHelperControl(
            GOLDEN,
            "_probe",
            "expression",
            "jnp.where(jnp.isnan(raw), -jnp.inf, raw)",
            "raw",
        ),
        "outer_helpers:golden:interior_tie_right": OuterHelperControl(
            GOLDEN, "_shrink_bracket", "expression", "fa >= fb", "fa > fb"
        ),
        "outer_helpers:golden:cached_value_reprobed": OuterHelperControl(
            GOLDEN,
            "_shrink_bracket",
            "expression",
            "jnp.where(take_left, fa, fb)",
            "_probe(objective=objective, x=x_kept)",
        ),
        "outer_helpers:golden:upper_endpoint_omitted": OuterHelperControl(
            GOLDEN,
            "maximize_golden_section",
            "expression",
            ("((xa, fa), (xb, fb), (safe_upper, upper_f))"),
            "((xa, fa), (xb, fb))",
        ),
        "outer_helpers:golden:loop_callback_replaced": OuterHelperControl(
            GOLDEN,
            "maximize_golden_section",
            "expression",
            "functools.partial(_shrink_bracket, objective=objective)",
            "lambda _index, carry: carry",
        ),
        "outer_helpers:surface:golden_import_redirected": OuterHelperControl(
            REFINEMENT,
            "MODULE_SURFACE",
            "import",
            "from _lcm.optimization.golden_section import maximize_golden_section",
            "def maximize_golden_section(**kwargs):\n    return None",
        ),
        "outer_helpers:surface:prefix_minimum_changed": OuterHelperControl(
            INTERPOLATION,
            "MODULE_SURFACE",
            "constant",
            "_MIN_PREFIX_SIZE = 2",
            "_MIN_PREFIX_SIZE = 3",
        ),
        "outer_helpers:surface:golden_result_reordered": OuterHelperControl(
            GOLDEN, "MODULE_SURFACE", "schema", "GoldenSectionResult", "lower,upper"
        ),
    }
)


def _key(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _owner(*, tree: ast.Module, name: str) -> ast.FunctionDef:
    """Resolve only an explicit top-level function or one class method."""
    if "." in name:
        class_name, name = name.split(".", maxsplit=1)
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise ValueError(f"Expected one class owner {class_name!r}")
        scope = classes[0].body
    else:
        scope = tree.body
    functions = [
        node
        for node in scope
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise ValueError(f"Expected one callable owner {name!r}")
    return functions[0]


def _replace_expression(
    *, owner: ast.FunctionDef, before: str, after: str
) -> tuple[ast.AST, str, int | None, ast.AST, ast.AST]:
    """Return the unique exact expression replacement and its reversible slot."""
    expected = _key(ast.parse(before, mode="eval").body)
    slots = []
    for parent in ast.walk(owner):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, ast.expr) and _key(value) == expected:
                slots.append((parent, field, None, value))
            elif isinstance(value, list):
                slots.extend(
                    (parent, field, index, child)
                    for index, child in enumerate(value)
                    if isinstance(child, ast.expr) and _key(child) == expected
                )
    if len(slots) != 1:
        raise ValueError(
            f"Expected one owner-local expression {before!r}, found {len(slots)}"
        )
    parent, field, index, original = slots[0]
    return parent, field, index, original, ast.parse(after, mode="eval").body


def _set_slot(
    *, parent: ast.AST, field: str, index: int | None, node: ast.AST | list[ast.stmt]
) -> None:
    if index is None:
        setattr(parent, field, node)
    else:
        values = getattr(parent, field)
        values[index] = node


def mutate_outer_helper(
    *, source: SourceText, control: OuterHelperControl
) -> SourceText:
    """Apply exactly one reviewed delta and prove its whole-module inverse."""
    tree = ast.parse(source)
    original_key = _key(tree)
    if control.mode in {"body", "expression", "lexical"}:
        owner = _owner(tree=tree, name=control.owner)
        if control.mode == "expression":
            parent, field, index, original, replacement = _replace_expression(
                owner=owner, before=control.before, after=control.after
            )
        else:
            if control.mode == "lexical":
                matches = [
                    node
                    for node in owner.body
                    if isinstance(node, ast.FunctionDef) and node.name == control.before
                ]
                if len(matches) != 1 or _key(
                    ast.Module(body=matches[0].body, type_ignores=[])
                ) != _key(ast.parse("return three_point")):
                    raise ValueError("The exact lexical without_cubic body is required")
                owner = matches[0]
            parent, field, index = owner, "body", None
            original = copy.deepcopy(owner.body)
            replacement = ast.parse("return None").body
    elif control.mode in {"import", "constant"}:
        expected = _key(ast.parse(control.before).body[0])
        matches = [
            (index, node)
            for index, node in enumerate(tree.body)
            if _key(node) == expected
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one module declaration {control.before!r}")
        index, original = matches[0]
        parent, field = tree, "body"
        replacement = ast.parse(control.after).body[0]
    else:
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == control.before
        ]
        if len(classes) != 1:
            raise ValueError("The exact GoldenSectionResult schema is required")
        parent, field, index = classes[0], "body", None
        original = copy.deepcopy(classes[0].body)
        replacement = copy.deepcopy(original)
        positions = {
            node.target.id: position
            for position, node in enumerate(replacement)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in {"lower", "upper"}
        }
        if set(positions) != {"lower", "upper"}:
            raise ValueError("The result schema must declare lower and upper once")
        lower, upper = positions["lower"], positions["upper"]
        replacement[lower], replacement[upper] = replacement[upper], replacement[lower]
    _set_slot(parent=parent, field=field, index=index, node=replacement)
    changed_key = _key(tree)
    if changed_key == original_key:
        raise ValueError("The selected control did not change its owner")
    mutated = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    if _key(ast.parse(mutated)) != changed_key:
        raise ValueError("Mutation serialization changed the reviewed AST")
    _set_slot(parent=parent, field=field, index=index, node=original)
    if _key(tree) != original_key:
        raise ValueError("The exact mutation inverse changed another module node")
    return mutated


def outer_helper_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build the explicit forty controls without discovering additional owners."""
    return {
        name: {
            "path": control.source,
            "source": mutate_outer_helper(
                source=(repo_root / control.source).read_text(encoding="utf-8"),
                control=control,
            ),
        }
        for name, control in CONTROLS.items()
    }
