#!/usr/bin/env python3
"""Prove route-local candidate-array flow from ``Q_and_F`` to full reducers.

The executable certificate is intentionally finite. Its universal half has two
explicitly separated claims. First, every coordinate produced by an already-constructed, finalized concrete
built-in action-grid object or supplied through the public runtime-points seam reaches
the pointwise ``Q_and_F`` call as an action argument. Second, on each GridSearch
route, the exact arrays bound by ``Q_arr, F_arr = Q_and_F(...)`` must reach the
full reducer without an intervening candidate-changing expression. The economic
construction of Q/F values and feasibility—including user DAGs, constraints,
transitions, continuation values, interpolation, and fold weights—is an explicit
semantic boundary and is not re-proved here. The proof is strict by design: a new
statement in either certified transport corridor is not assumed harmless; it has
to enter the explicit, independently checked representation allowlist.

The six corridors are:

* singleton solve -> ``Q_arr.max(where=F_arr, ...)``;
* singleton simulate -> ``argmax_and_max(Q_arr, where=F_arr, ...)``;
* collective solve -> ``collective_readout(..., feasibility=F_arr, ...)``;
* collective simulate -> ``collective_argmax_and_readout(...,
  feasibility=F_arr, ...)``;
* taste-shock solve -> exact mask, continuous maximum, then full discrete logsum;
* taste-shock simulate -> exact mask, row-major continuous maximum, one
  mean-zero Gumbel draw per discrete cell, and exact flat-index reconstruction.

The only allowed representation change is the collective split of the trailing
stakeholder axis, exactly ``Q_arr[..., index]`` for every enumerated stakeholder.
It cannot select, reorder, or mask an action axis. The common feasibility array
is passed by identity. The shared axis-move, flatten, scalarization, full argmax,
collective delegation, and value-gather bodies are pinned as exact AST shapes,
so moving a filter into a helper does not evade the route proof. The shared
logsum and taste-noise helpers are pinned too.
"""

# Exact production and mutation snippets intentionally preserve long source lines.
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from generate_sources import sha256_file
except ModuleNotFoundError:  # Imported as tests.candidate_certificate.direct_flow.
    from tests.candidate_certificate.generate_sources import sha256_file

MAX_Q_SOURCE = "src/_lcm/regime_building/max_Q_over_a.py"
ARGMAX_SOURCE = "src/_lcm/regime_building/argmax.py"
COLLECTIVE_SOURCE = "src/_lcm/regime_building/collective.py"
LOGSUM_SOURCE = "src/_lcm/logsum.py"
GRID_SEARCH_SOURCE = "src/_lcm/solution/grid_search.py"
PROCESSING_SOURCE = "src/_lcm/regime_building/processing.py"
DISPATCHERS_SOURCE = "src/_lcm/utils/dispatchers.py"
FUNCTOOLS_SOURCE = "src/_lcm/utils/functools.py"
CONTAINERS_SOURCE = "src/_lcm/utils/containers.py"
ZERO_SAFE_SOURCE = "src/_lcm/zero_safe.py"
PROBABILITY_SOURCE = "src/_lcm/probability.py"
ENGINE_SOURCE = "src/_lcm/engine.py"
STATE_ACTION_SPACE_SOURCE = "src/_lcm/state_action_space.py"
SIMULATION_SOURCE = "src/_lcm/simulation/simulate.py"
SIMULATION_TRANSITIONS_SOURCE = "src/_lcm/simulation/transitions.py"
SIMULATION_COMPILE_SOURCE = "src/_lcm/simulation/compile.py"
MODEL_SOURCE = "src/lcm/model.py"
BACKWARD_INDUCTION_SOURCE = "src/_lcm/solution/backward_induction.py"
INITIAL_CONDITIONS_SOURCE = "src/_lcm/simulation/initial_conditions.py"
RESULT_SOURCE = "src/lcm/result.py"
RESULT_DATAFRAME_SOURCE = "src/_lcm/simulation/result_dataframe.py"
RESULT_METADATA_SOURCE = "src/_lcm/simulation/result_metadata.py"
ADDITIONAL_TARGETS_SOURCE = "src/_lcm/simulation/additional_targets.py"
SIMULATION_RANDOM_SOURCE = "src/_lcm/simulation/random.py"
FOLD_ZERO_SAFE_SOURCE = "src/_lcm/regime_building/zero_safe.py"
SOLUTION_CONTRACT_SOURCE = "src/_lcm/solution/contract.py"
GRIDS_INIT_SOURCE = "src/_lcm/grids/__init__.py"
GRID_BASE_SOURCE = "src/_lcm/grids/base.py"
GRID_COORDINATES_SOURCE = "src/_lcm/grids/coordinates.py"
DISCRETE_GRID_SOURCE = "src/_lcm/grids/discrete.py"
CONTINUOUS_GRID_SOURCE = "src/_lcm/grids/continuous.py"
PIECEWISE_GRID_SOURCE = "src/_lcm/grids/piecewise.py"
PROCESSES_INIT_SOURCE = "src/_lcm/processes/__init__.py"
PROCESS_BASE_SOURCE = "src/_lcm/processes/base.py"
PROCESS_IID_SOURCE = "src/_lcm/processes/iid.py"
PROCESS_AR1_SOURCE = "src/_lcm/processes/ar1.py"
VARIABLES_SOURCE = "src/_lcm/variables.py"
PARAMS_REGIME_TEMPLATE_SOURCE = "src/_lcm/params/regime_template.py"
PARAMS_PROCESSING_SOURCE = "src/_lcm/params/processing.py"
DTYPES_SOURCE = "src/_lcm/dtypes.py"
NAMESPACE_SOURCE = "src/_lcm/utils/namespace.py"
PANDAS_UTILS_SOURCE = "src/_lcm/pandas_utils.py"
MODEL_PROCESSING_SOURCE = "src/_lcm/model_processing.py"

_CERTIFIED_CORRIDOR_SOURCES = (
    MAX_Q_SOURCE,
    ARGMAX_SOURCE,
    COLLECTIVE_SOURCE,
    LOGSUM_SOURCE,
    GRID_SEARCH_SOURCE,
    PROCESSING_SOURCE,
    DISPATCHERS_SOURCE,
    FUNCTOOLS_SOURCE,
    CONTAINERS_SOURCE,
    ZERO_SAFE_SOURCE,
    PROBABILITY_SOURCE,
    ENGINE_SOURCE,
    STATE_ACTION_SPACE_SOURCE,
    SIMULATION_SOURCE,
    SIMULATION_TRANSITIONS_SOURCE,
    SIMULATION_COMPILE_SOURCE,
    MODEL_SOURCE,
    BACKWARD_INDUCTION_SOURCE,
    INITIAL_CONDITIONS_SOURCE,
    RESULT_SOURCE,
    RESULT_DATAFRAME_SOURCE,
    RESULT_METADATA_SOURCE,
    ADDITIONAL_TARGETS_SOURCE,
    SIMULATION_RANDOM_SOURCE,
    FOLD_ZERO_SAFE_SOURCE,
    SOLUTION_CONTRACT_SOURCE,
    GRIDS_INIT_SOURCE,
    GRID_BASE_SOURCE,
    GRID_COORDINATES_SOURCE,
    DISCRETE_GRID_SOURCE,
    CONTINUOUS_GRID_SOURCE,
    PIECEWISE_GRID_SOURCE,
    PROCESSES_INIT_SOURCE,
    PROCESS_BASE_SOURCE,
    PROCESS_IID_SOURCE,
    PROCESS_AR1_SOURCE,
    VARIABLES_SOURCE,
    PARAMS_REGIME_TEMPLATE_SOURCE,
    PARAMS_PROCESSING_SOURCE,
    DTYPES_SOURCE,
    NAMESPACE_SOURCE,
    PANDAS_UTILS_SOURCE,
    MODEL_PROCESSING_SOURCE,
)

# These seals are deliberately internal to the semantic verifier. The generated
# inventory, contract, policy, and manifest can all be synchronized after a source
# mutation; this independent allowlist cannot. It closes relocation into every
# repository-local helper on which the six certified routes depend.
_SOURCE_SEALS = {
    LOGSUM_SOURCE: "e12061dd4f0f0176324182a2eb875cb6ebe4b97174091c597d46a622df93ff1b",
    ARGMAX_SOURCE: "c0b411682d277907c65e67765e18e8b2c4d8c35d5c0509b8857f9225844bd9f9",
    COLLECTIVE_SOURCE: "5ca9663523a257eeef45305c571c18689157a0997d64be0d3f90861f11e830e9",
    MAX_Q_SOURCE: "40689cb24c7646b2caf7e6b2f4f50319a6249615957c2618c3995b4680d194cf",
    PROCESSING_SOURCE: "46f5ba1374bd8d71ab32d9f3f924be4d0a61965365ecd76ae44d87c02e397611",
    GRID_SEARCH_SOURCE: "a035b1f5bd50e5ab831d9731341c0b48f1fb8112be8bb6411bec09a7159df58d",
    DISPATCHERS_SOURCE: "2965c95e06f92c9108eaec983d7c44df08476d4e025274fbcc8068e3edb0ced0",
    FUNCTOOLS_SOURCE: "678abc78139862535fb1a0440f31c9b46584d6cfa3ef04a824d5f1b2ce14c922",
    CONTAINERS_SOURCE: "0838079e35ba498009d8af7e6ed717f870a96a2fdc628d25e80310cd630174a9",
    ZERO_SAFE_SOURCE: "64a6582fd25fa016fce431981cce701f0b0249609e68a31a1f7fbf4577e26fd6",
    PROBABILITY_SOURCE: "e672f6a73e746213806db8ee29503cd6bbf883d00d096f4e80b325edd0d08196",
    ENGINE_SOURCE: "245b4dc2ec1f8b582864e6717ccf72ec91cd2a867ccbda04617dc6dda020ca1d",
    STATE_ACTION_SPACE_SOURCE: "c7af3ea4c3912efa3d5d7daa0d420168a7545e327f6e4c581b3baf54efc79f11",
    SIMULATION_SOURCE: "63e978935ff1eb9f65ae81d70ed3808696387e13da534cfc214827c619dbccac",
    SIMULATION_TRANSITIONS_SOURCE: "ee59545d27352be39458de6ea160f15ed1503ca343825069cbf84e80ea297aba",
    SIMULATION_COMPILE_SOURCE: "75a2da153e3d30cd0c900271a0685f481df2f43d0e0e1c5f0e1b48dbc1808719",
    MODEL_SOURCE: "5524e38f9812477eb0371ce83dc670ea3bb06f65d7c521bf48a74ef7c04a47e0",
    BACKWARD_INDUCTION_SOURCE: "feef18b2376de49bf6133afc65ee339a92fc5d45eed5172bfecfeac340c928cd",
    INITIAL_CONDITIONS_SOURCE: "82444fb4a38422911e702be68cf18971c187ead21f642353f7e51c280c617503",
    RESULT_SOURCE: "23d046876abc003cca8ee3929304a8fbd667a8f6b497c940bf830bd6fe0b0523",
    RESULT_DATAFRAME_SOURCE: "025e273c4d3bb9d8f9787189a551b113708c86b1e868d16178aa39555abf49a4",
    RESULT_METADATA_SOURCE: "34bf3383c75ebcf4eb8ba9c090fe86228a7ae0c6aa16c87f273d431d51fdf94a",
    ADDITIONAL_TARGETS_SOURCE: "1e3b0a520922e2c0253ee20445db9ed42969aaae53bdfd9130b0a1081dccb36b",
    SIMULATION_RANDOM_SOURCE: "2133b4cfd84cfeb62c8a8c43a6c714f9772993dacaa3a151157ca36a722a089b",
    FOLD_ZERO_SAFE_SOURCE: "7260208332df791eec5c423835a2bf0662819053c4bf488809798572cf5bc094",
    SOLUTION_CONTRACT_SOURCE: "db6bb256635a2f66e4ebab1e661ad2d6fe8a86bce79ee3cb1467ef959bc98887",
    GRIDS_INIT_SOURCE: "c66aed5ef6cdb56cfa38eebb7f870f12475f7a5f62ca1962c17230f66fd3268a",
    GRID_BASE_SOURCE: "fd1064986abdbe1755383fb08758f74d40cad419c0da312f38b521c7d78ce59c",
    GRID_COORDINATES_SOURCE: "e0f3cffc38e2a854426309b3eacab5783a0a5725cc4e763a06969e03914619e8",
    DISCRETE_GRID_SOURCE: "6abd3cdff06b2a85778bd0f42db523c0b809cb2d304bdc49db53756dc203f1a3",
    CONTINUOUS_GRID_SOURCE: "21eb6c7dbee165b3e78fbca43c3cc07bb99c449c8caf4abedcb30910396eeed0",
    PIECEWISE_GRID_SOURCE: "fff3f42ba655c03135d7c443588c615223f8c56f425ee82333fe0b234332eb24",
    PROCESSES_INIT_SOURCE: "db7892762ef1b5635b61b4e57ef97ae0becdd1fc7507ef8efd202323ced7671f",
    PROCESS_BASE_SOURCE: "a058be1a20858d4208305114e86b8a2e0bb04c75182a92b610f32c1e08cb5e03",
    PROCESS_IID_SOURCE: "a00ef015e1a5629f66059ddb5e0b81e3999d44181ab4294c1cc86201cf322c63",
    PROCESS_AR1_SOURCE: "1895a70a5abbe4375fb450cd5c066ee6138bfb4d145846a59c4c2933d01a183d",
    VARIABLES_SOURCE: "1a6aac8a24d98edc858f1ff0f3f978498cb240bead66bba4de61ce8aaeea87b7",
    PARAMS_REGIME_TEMPLATE_SOURCE: "685dbc5098695fa21e3dc37ee314e9c883c1aa6e3ab13d5b3ac764ffea703c61",
    PARAMS_PROCESSING_SOURCE: "eebcd5daca0b9e5711fd58d84d65462c8074dbc8b7e266dd8357618b8f8b5c94",
    DTYPES_SOURCE: "6f6a6707dba0ef72e7ce9689dc671f0f1cdbdee4bc5d2c591bd5c096f6670b19",
    NAMESPACE_SOURCE: "254509e538c6a2264a71e04cdd5abdb60ad92f04899a37f710004222ae855bea",
    PANDAS_UTILS_SOURCE: "714eb0104a8392dac83aac6a2b3c23ccafe16d9d8b2871145f86e71b1d8880a3",
    MODEL_PROCESSING_SOURCE: "1d459d796afdd493bbaf18dfdf2be282e7bb38d581e74cf91675a5b80a1e97cd",
}

EXPECTED_DIRECT_FLOW_MUTATION_COUNT = 176
EXPECTED_DIRECT_FLOW_MUTATION_NAMES_SHA256 = (
    "fea9bb27d93337e16d2505c643660b3816cb37d5e87d3cb41a0a2e4baad4739f"
)


def canonical_json(payload: dict[str, Any]) -> str:
    """Render deterministic JSON for command-line controls."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _mutation_name_digest(names: Sequence[str]) -> str:
    """Hash the exact sorted mutation-name family, including its cardinality."""
    payload = ("\n".join(sorted(names)) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _name(node: ast.AST | None, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _call_name(call: ast.Call) -> str | None:
    """Return only an unqualified direct callee; attributes are not lookalikes."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def _unparse(node: ast.AST | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def _target_names(target: ast.expr) -> tuple[str, ...] | None:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Tuple) and all(
        isinstance(item, ast.Name) for item in target.elts
    ):
        return tuple(item.id for item in target.elts if isinstance(item, ast.Name))
    return None


def _assigned_names(statement: ast.stmt) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        targets.extend(statement.targets)
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign):
        targets.append(statement.target)
    elif isinstance(statement, ast.Delete):
        targets.extend(statement.targets)
    found: set[str] = set()
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                found.add(child.id)
    return found


def _stored_name_count(node: ast.AST, name: str) -> int:
    """Count every assignment-form store of ``name`` below one AST node."""
    return sum(
        isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Store)
        and child.id == name
        for child in ast.walk(node)
    )


def _definition(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one top-level function {name!r}, found {len(matches)}"
        )
    return matches[0]


def _guarded_nested(
    tree: ast.Module,
    *,
    outer_name: str,
    nested_name: str,
    taste_shocks: bool,
) -> ast.FunctionDef:
    """Resolve one reducer from the direct ``has_taste_shocks`` guard branch."""
    outer = _definition(tree, outer_name)
    guards = [
        statement
        for statement in outer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "has_taste_shocks"
    ]
    if len(guards) != 1:
        raise ValueError(
            f"expected one direct has_taste_shocks guard in {outer_name!r}, "
            f"found {len(guards)}"
        )
    guard = guards[0]
    branch = guard.body if taste_shocks else guard.orelse
    matches = [
        node
        for node in branch
        if isinstance(node, ast.FunctionDef) and node.name == nested_name
    ]
    all_matches = [
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.FunctionDef)
        and node is not outer
        and node.name == nested_name
    ]
    nested_scopes = [
        node
        for node in ast.walk(outer)
        if node is not outer
        and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    route = "taste-shock" if taste_shocks else "ordinary"
    if (
        len(matches) != 1
        or len(all_matches) != 2
        or len(nested_scopes) != 2
        or any(
            not isinstance(node, ast.FunctionDef) or node.name != nested_name
            for node in nested_scopes
        )
        or len(guard.body) != 1
        or len(guard.orelse) != 1
    ):
        raise ValueError(
            f"expected one direct {route} nested {nested_name!r}, exactly two "
            f"route definitions, no other nested scope, and no guard-branch "
            f"side statements in {outer_name!r}; found {len(matches)} direct, "
            f"{len(all_matches)} named, {len(nested_scopes)} total nested, and "
            f"branch lengths {len(guard.body)}/{len(guard.orelse)}"
        )
    return matches[0]


def _ordinary_nested(
    tree: ast.Module, *, outer_name: str, nested_name: str
) -> ast.FunctionDef:
    """Return the reducer from the false arm of ``has_taste_shocks``."""
    return _guarded_nested(
        tree,
        outer_name=outer_name,
        nested_name=nested_name,
        taste_shocks=False,
    )


def _taste_nested(
    tree: ast.Module, *, outer_name: str, nested_name: str
) -> ast.FunctionDef:
    """Return the reducer from the true arm of ``has_taste_shocks``."""
    return _guarded_nested(
        tree,
        outer_name=outer_name,
        nested_name=nested_name,
        taste_shocks=True,
    )


def _body_without_docstring(node: ast.FunctionDef) -> list[ast.stmt]:
    """Return executable statements, excluding the function docstring."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def _ast_key(node: ast.AST) -> str:
    """Return a location-independent structural representation."""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _expected_statements(source: str) -> list[ast.stmt]:
    """Parse an allowlisted statement sequence under the running Python AST."""
    return ast.parse(source).body


def _body_matches(node: ast.FunctionDef, expected_source: str) -> bool:
    observed = [_ast_key(item) for item in _body_without_docstring(node)]
    expected = [_ast_key(item) for item in _expected_statements(expected_source)]
    return observed == expected


def _expression_matches(node: ast.AST | None, source: str) -> bool:
    """Compare one expression with a hard-coded, location-free AST."""
    return node is not None and _ast_key(node) == _ast_key(
        ast.parse(source, mode="eval").body
    )


def _exact_reducer_decorator(
    node: ast.FunctionDef, *, simulate: bool, taste_shocks: bool
) -> bool:
    """Pin the signature wrapper that exposes the exact action/state inputs."""
    if len(node.decorator_list) != 1 or not isinstance(
        node.decorator_list[0], ast.Call
    ):
        return False
    call = node.decorator_list[0]
    if _call_name(call) != "with_signature" or call.args:
        return False
    args_source = (
        '["next_regime_to_V_arr", "taste_shock_key", '
        "*action_names, *state_names, *extra_param_names]"
        if simulate and taste_shocks
        else '["next_regime_to_V_arr", *action_names, *state_names, *extra_param_names]'
    )
    if simulate:
        annotation_source = '"tuple[IntND, FloatND]"'
    elif taste_shocks:
        annotation_source = '"FloatND"'
    else:
        annotation_source = (
            '"tuple[FloatND, BoolND]" if stakeholders is not None else "FloatND"'
        )
    return (
        _expression_matches(_keyword(call, "args"), args_source)
        and _expression_matches(_keyword(call, "return_annotation"), annotation_source)
        and _expression_matches(_keyword(call, "enforce"), "False")
        and {item.arg for item in call.keywords}
        == {"args", "return_annotation", "enforce"}
    )


def _nested_reducer_signature(node: ast.FunctionDef) -> bool:
    args = node.args
    return (
        not args.posonlyargs
        and tuple(item.arg for item in args.args) == ("next_regime_to_V_arr",)
        and args.vararg is None
        and not args.kwonlyargs
        and not args.kw_defaults
        and args.kwarg is not None
        and args.kwarg.arg == "states_actions_params"
        and not args.defaults
    )


def _positional_signature(
    node: ast.FunctionDef,
    *,
    names: tuple[str, ...],
    defaults: tuple[str, ...] = (),
) -> bool:
    args = node.args
    return (
        not args.posonlyargs
        and tuple(item.arg for item in args.args) == names
        and args.vararg is None
        and not args.kwonlyargs
        and not args.kw_defaults
        and args.kwarg is None
        and tuple(ast.unparse(item) for item in args.defaults) == defaults
    )


def _keyword_only_signature(
    node: ast.FunctionDef,
    *,
    names: tuple[str, ...],
    defaults: tuple[str | None, ...] | None = None,
) -> bool:
    args = node.args
    expected_defaults = (None,) * len(names) if defaults is None else defaults
    return (
        not args.posonlyargs
        and not args.args
        and args.vararg is None
        and tuple(item.arg for item in args.kwonlyargs) == names
        and tuple(
            ast.unparse(item) if item is not None else None for item in args.kw_defaults
        )
        == expected_defaults
        and args.kwarg is None
        and not args.defaults
    )


def _bound_import_names(statement: ast.Import | ast.ImportFrom) -> set[str]:
    names: set[str] = set()
    for alias in statement.names:
        names.add(alias.asname or alias.name.split(".")[0])
    return names


def _relevant_imports(tree: ast.Module, *, names: set[str]) -> list[str]:
    return sorted(
        ast.unparse(statement)
        for statement in tree.body
        if isinstance(statement, ast.Import | ast.ImportFrom)
        and names & _bound_import_names(statement)
    )


class _ScopeBindingVisitor(ast.NodeVisitor):
    """Collect names bound in one lexical scope, excluding nested scopes."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def _bind(self, name: str | None) -> None:
        if name is not None:
            self.counts[name] = self.counts.get(name, 0) + 1

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store | ast.Del):
            self._bind(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bind(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._bind(node.name)
        if node.type is not None:
            self.visit(node.type)
        for statement in node.body:
            self.visit(statement)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self._bind(node.name)
        if node.pattern is not None:
            self.visit(node.pattern)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self._bind(node.name)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._bind(name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._bind(name)


def _scope_binding_counts(statements: list[ast.stmt]) -> dict[str, int]:
    visitor = _ScopeBindingVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.counts


def _statements_match(observed: Sequence[ast.stmt], expected_source: str) -> bool:
    """Compare one statement sequence with a hard-coded AST allowlist."""
    return [_ast_key(item) for item in observed] == [
        _ast_key(item) for item in _expected_statements(expected_source)
    ]


def _module_contract_errors(
    tree: ast.Module,
    *,
    label: str,
    relevant_import_names: set[str],
    expected_imports: list[str],
    expected_binding_counts: dict[str, int],
) -> list[str]:
    """Pin critical imports and reject every same-scope shadowing form."""
    errors: list[str] = []
    observed_imports = _relevant_imports(tree, names=relevant_import_names)
    if observed_imports != sorted(expected_imports):
        errors.append(f"{label}: critical import bindings changed")
    counts = _scope_binding_counts(tree.body)
    mismatches = {
        name: (counts.get(name, 0), expected)
        for name, expected in expected_binding_counts.items()
        if counts.get(name, 0) != expected
    }
    if mismatches:
        errors.append(f"{label}: critical module bindings changed: {mismatches}")
    return errors


def _class_definition(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one top-level class {name!r}, found {len(matches)}")
    return matches[0]


def _method_definition(
    tree: ast.Module, *, class_name: str, method_name: str
) -> tuple[ast.ClassDef, ast.FunctionDef]:
    cls = _class_definition(tree, class_name)
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise ValueError(
            f"expected one {class_name}.{method_name}, found {len(methods)}"
        )
    return cls, methods[0]


def _function_definition(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one top-level function {name!r}, found {len(matches)}"
        )
    return matches[0]


def _grid_base_errors(tree: ast.Module) -> list[str]:
    """Forbid inherited interception of concrete grid coordinate materializers."""
    errors: list[str] = []
    try:
        cls = _class_definition(tree, "Grid")
    except ValueError as error:
        return [f"grid base: {error}"]
    body = list(cls.body)
    has_docstring = bool(
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )
    if has_docstring:
        body.pop(0)
    expected_names = ("batch_size", "distributed", "to_jax")
    if (
        [ast.unparse(base) for base in cls.bases] != ["ABC"]
        or cls.keywords
        or cls.decorator_list
        or len(body) != len(expected_names)
        or not all(isinstance(node, ast.FunctionDef) for node in body)
        or tuple(node.name for node in body if isinstance(node, ast.FunctionDef))
        != expected_names
    ):
        errors.append(
            "grid base: Grid may contain only its docstring and the three abstract "
            "coordinate-interface methods"
        )
        return errors
    expected = {
        "batch_size": (("property", "abstractmethod"), "int"),
        "distributed": (("property", "abstractmethod"), "bool"),
        "to_jax": (("abstractmethod",), "Int1D | Float1D"),
    }
    for node in body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators, returns = expected[node.name]
        if (
            tuple(ast.unparse(item) for item in node.decorator_list) != decorators
            or not _positional_signature(node, names=("self",))
            or node.returns is None
            or ast.unparse(node.returns) != returns
            or _body_without_docstring(node)
        ):
            errors.append(f"grid base: Grid.{node.name} contract changed")
    return errors


def _engine_state_action_space_errors(tree: ast.Module) -> list[str]:
    """Pin action publication and replacement to full, order-preserving mappings."""
    errors: list[str] = []
    try:
        _, action_names = _method_definition(
            tree, class_name="StateActionSpace", method_name="action_names"
        )
        _, actions = _method_definition(
            tree, class_name="StateActionSpace", method_name="actions"
        )
        _, shapes = _method_definition(
            tree, class_name="StateActionSpace", method_name="actions_grid_shapes"
        )
        _, replace = _method_definition(
            tree, class_name="StateActionSpace", method_name="replace"
        )
    except ValueError as error:
        return [f"state-action space: {error}"]
    property_methods = (
        (
            action_names,
            "return tuple(self.discrete_actions) + tuple(self.continuous_actions)",
        ),
        (
            actions,
            (
                "return MappingProxyType(\n"
                "    dict(self.discrete_actions) | dict(self.continuous_actions)\n"
                ")"
            ),
        ),
        (shapes, "return tuple(len(grid) for grid in self.actions.values())"),
    )
    for node, expected_body in property_methods:
        if (
            tuple(ast.unparse(item) for item in node.decorator_list) != ("property",)
            or not _positional_signature(node, names=("self",))
            or not _body_matches(node, expected_body)
        ):
            errors.append(
                f"state-action space: StateActionSpace.{node.name} no longer "
                "publishes the full ordered candidate mapping"
            )
    if (
        replace.decorator_list
        or not _positional_signature(
            replace,
            names=("self", "states", "discrete_actions", "continuous_actions"),
            defaults=("None", "None", "None"),
        )
        or not _body_matches(
            replace,
            "states = first_non_none(states, self.states)\n"
            "discrete_actions = first_non_none(discrete_actions, self.discrete_actions)\n"
            "continuous_actions = first_non_none(\n"
            "    continuous_actions, self.continuous_actions\n"
            ")\n"
            "return dataclasses.replace(\n"
            "    self,\n"
            "    states=states,\n"
            "    discrete_actions=discrete_actions,\n"
            "    continuous_actions=continuous_actions,\n"
            ")",
        )
    ):
        errors.append(
            "state-action space: replace no longer preserves every inherited action "
            "mapping unless that mapping is explicitly replaced"
        )
    return errors


def _simulation_state_action_space_errors(tree: ast.Module) -> list[str]:
    """Pin the simulation adapter to a state-only replacement of the completed base."""
    try:
        node = _function_definition(tree, "create_regime_state_action_space")
    except ValueError as error:
        return [f"simulation state-action adapter: {error}"]
    expected_body = (
        "states_for_state_action_space = {\n"
        "    sn: regime_states[sn] for sn in regime.solution.state_names\n"
        "}\n"
        "_validate_all_states_present(\n"
        "    provided_states=states_for_state_action_space,\n"
        "    required_state_names=set(regime.solution.state_names),\n"
        ")\n"
        "return base.replace(states=MappingProxyType(states_for_state_action_space))"
    )
    if (
        node.decorator_list
        or not _keyword_only_signature(node, names=("regime", "regime_states", "base"))
        or not _body_matches(node, expected_body)
    ):
        return [
            (
                "simulation state-action adapter: completed base actions are "
                "not preserved exactly while current states are installed"
            )
        ]
    return []


def _simulation_state_action_space_caller_errors(tree: ast.Module) -> list[str]:
    """Pin the live simulation caller to the params-completed base without wrapping."""
    try:
        node = _function_definition(tree, "_simulate_regime_in_period")
    except ValueError as error:
        return [f"simulation state-action caller: {error}"]
    body = _body_without_docstring(node)
    bindings = _scope_binding_counts(body)
    matches = [
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and _target_names(statement.targets[0]) == ("state_action_space",)
        and isinstance(statement.value, ast.Call)
    ]
    if len(matches) != 1 or bindings.get("state_action_space") != 1:
        return [
            (
                "simulation state-action caller: expected exactly one live "
                "state_action_space binding"
            )
        ]
    call = matches[0].value
    if not isinstance(call, ast.Call):
        return ["simulation state-action caller: adapter call changed"]
    if not (
        _call_name(call) == "create_regime_state_action_space"
        and not call.args
        and {item.arg for item in call.keywords} == {"regime", "regime_states", "base"}
        and _name(_keyword(call, "regime"), "regime")
        and _expression_matches(_keyword(call, "regime_states"), "states[regime_name]")
        and _name(_keyword(call, "base"), "base_state_action_space")
    ):
        return [
            (
                "simulation state-action caller: adapter does not receive the "
                "exact params-completed base"
            )
        ]
    return []


def _q_and_f_origin(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    if _target_names(statement.targets[0]) != ("Q_arr", "F_arr"):
        return False
    if not (
        isinstance(statement.value, ast.Call)
        and _call_name(statement.value) == "Q_and_F"
    ):
        return False
    if statement.value.args:
        return False
    if _unparse(_keyword(statement.value, "next_regime_to_V_arr")) != (
        "next_regime_to_V_arr"
    ):
        return False
    splats = [item.value for item in statement.value.keywords if item.arg is None]
    return len(splats) == 1 and _name(splats[0], "states_actions_params")


def _negative_infinity(node: ast.expr | None) -> bool:
    return node is not None and ast.unparse(node) == "-jnp.inf"


def _exact_singleton_solve_return(statement: ast.stmt) -> bool:
    if not (
        isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call)
    ):
        return False
    call = statement.value
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "max"
        and _name(call.func.value, "Q_arr")
        and not call.args
    ):
        return False
    return (
        _name(_keyword(call, "where"), "F_arr")
        and _negative_infinity(_keyword(call, "initial"))
        and _keyword(call, "axis") is None
        and {item.arg for item in call.keywords} == {"where", "initial"}
    )


def _exact_singleton_simulate_return(statement: ast.stmt) -> bool:
    if not (
        isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call)
    ):
        return False
    call = statement.value
    return (
        _call_name(call) == "argmax_and_max"
        and len(call.args) == 1
        and _name(call.args[0], "Q_arr")
        and _name(_keyword(call, "where"), "F_arr")
        and _negative_infinity(_keyword(call, "initial"))
        and _keyword(call, "axis") is None
        and {item.arg for item in call.keywords} == {"where", "initial"}
    )


def _exact_action_axes(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and _target_names(statement.targets[0]) == ("action_axes",)
        and ast.unparse(statement.value) == "tuple(range(F_arr.ndim))"
    )


def _exact_stakeholder_split(statement: ast.stmt) -> bool:
    """Allow only a split of the trailing stakeholder axis, never an action axis."""
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and _target_names(statement.targets[0]) == ("stakeholder_Q",)
        and isinstance(statement.value, ast.DictComp)
    ):
        return False
    comp = statement.value
    if not (_name(comp.key, "name") and isinstance(comp.value, ast.Subscript)):
        return False
    if not _name(comp.value.value, "Q_arr"):
        return False
    slice_node = comp.value.slice
    if not (
        isinstance(slice_node, ast.Tuple)
        and len(slice_node.elts) == 2
        and isinstance(slice_node.elts[0], ast.Constant)
        and slice_node.elts[0].value is Ellipsis
        and _name(slice_node.elts[1], "index")
    ):
        return False
    if len(comp.generators) != 1:
        return False
    generator = comp.generators[0]
    return (
        _target_names(generator.target) == ("index", "name")
        and isinstance(generator.iter, ast.Call)
        and _call_name(generator.iter) == "enumerate"
        and len(generator.iter.args) == 1
        and _name(generator.iter.args[0], "stakeholders")
        and not generator.iter.keywords
        and not generator.ifs
        and generator.is_async == 0
    )


def _exact_weights_call(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and _call_name(node) == "_evaluate_pareto_weights"
        and not node.args
        and _name(_keyword(node, "pareto_weights"), "pareto_weights")
        and _name(_keyword(node, "states_actions_params"), "states_actions_params")
        and {item.arg for item in node.keywords}
        == {"pareto_weights", "states_actions_params"}
    )


def _exact_collective_reducer_assignment(
    statement: ast.stmt, *, simulate: bool
) -> bool:
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.value, ast.Call)
    ):
        return False
    expected_targets = (
        ("argmax_flat", "values", "_dissolution")
        if simulate
        else ("values", "dissolution")
    )
    expected_call = (
        "collective_argmax_and_readout" if simulate else "collective_readout"
    )
    call = statement.value
    return (
        _target_names(statement.targets[0]) == expected_targets
        and _call_name(call) == expected_call
        and not call.args
        and _name(_keyword(call, "stakeholder_Q"), "stakeholder_Q")
        and _name(_keyword(call, "feasibility"), "F_arr")
        and _exact_weights_call(_keyword(call, "weights"))
        and _name(_keyword(call, "action_axes"), "action_axes")
        and {item.arg for item in call.keywords}
        == {"stakeholder_Q", "feasibility", "weights", "action_axes"}
    )


def _exact_values_stack(node: ast.expr | None) -> bool:
    if not isinstance(node, ast.Call) or ast.unparse(node.func) != "jnp.stack":
        return False
    if len(node.args) != 1 or not isinstance(node.args[0], ast.ListComp):
        return False
    comp = node.args[0]
    if not (
        isinstance(comp.elt, ast.Subscript)
        and _name(comp.elt.value, "values")
        and _name(comp.elt.slice, "name")
        and len(comp.generators) == 1
    ):
        return False
    generator = comp.generators[0]
    return (
        _name(generator.target, "name")
        and _name(generator.iter, "stakeholders")
        and not generator.ifs
        and generator.is_async == 0
        and _unparse(_keyword(node, "axis")) == "-1"
        and {item.arg for item in node.keywords} == {"axis"}
    )


def _exact_collective_return(statement: ast.stmt, *, simulate: bool) -> bool:
    if not (
        isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple)
    ):
        return False
    if simulate:
        return len(statement.value.elts) == 2 and all(
            _name(node, expected)
            for node, expected in zip(
                statement.value.elts, ("argmax_flat", "V_stacked"), strict=True
            )
        )
    return (
        len(statement.value.elts) == 2
        and _exact_values_stack(statement.value.elts[0])
        and _name(statement.value.elts[1], "dissolution")
    )


def _exact_collective_body(node: ast.If, *, simulate: bool) -> bool:  # noqa: PLR0911
    if ast.unparse(node.test) != "stakeholders is not None" or node.orelse:
        return False
    expected_length = 5 if simulate else 4
    if len(node.body) != expected_length:
        return False
    if not _exact_action_axes(node.body[0]):
        return False
    if not _exact_stakeholder_split(node.body[1]):
        return False
    if not _exact_collective_reducer_assignment(node.body[2], simulate=simulate):
        return False
    if simulate:
        stack = node.body[3]
        if not (
            isinstance(stack, ast.Assign)
            and len(stack.targets) == 1
            and _target_names(stack.targets[0]) == ("V_stacked",)
            and _exact_values_stack(stack.value)
        ):
            return False
        return _exact_collective_return(node.body[4], simulate=True)
    return _exact_collective_return(node.body[3], simulate=False)


def _productmap_binding_errors(
    tree: ast.Module, *, outer_name: str, nested_name: str
) -> list[str]:
    """Require one unwrapped, unbatched action product and no later rebinding."""
    try:
        outer = _definition(tree, outer_name)
    except ValueError as error:
        return [f"{outer_name}: {error}"]
    assignments = [
        statement
        for statement in ast.walk(outer)
        if isinstance(statement, ast.Assign)
        and any(_target_names(target) == ("Q_and_F",) for target in statement.targets)
    ]
    store_count = _stored_name_count(outer, "Q_and_F")
    errors: list[str] = []
    if len(assignments) != 1 or store_count != 1:
        errors.append(
            f"{outer_name}: Q_and_F must be bound exactly once to productmap and "
            f"never shadowed; found {len(assignments)} plain assignments and "
            f"{store_count} assignment-form stores"
        )
    else:
        value = assignments[0].value
        if not (
            isinstance(value, ast.Call)
            and _call_name(value) == "productmap"
            and not value.args
            and _name(_keyword(value, "func"), "Q_and_F")
            and _name(_keyword(value, "variables"), "action_names")
            and _unparse(_keyword(value, "batch_sizes"))
            == "dict.fromkeys(action_names, 0)"
            and {item.arg for item in value.keywords}
            == {"func", "variables", "batch_sizes"}
        ):
            errors.append(
                f"{outer_name}: action productmap is wrapped, filtered, batched, "
                "or does not consume the original Q_and_F"
            )
    if _stored_name_count(outer, nested_name):
        errors.append(
            f"{outer_name}: returned reducer {nested_name} is rebound after definition"
        )
    captured_rebindings = any(
        _stored_name_count(outer, name)
        for name in ("has_taste_shocks", "n_discrete_action_axes")
    )
    if captured_rebindings:
        errors.append(
            f"{outer_name}: taste-route guard or discrete-axis count is rebound"
        )
    return errors


def _max_builder_wiring_errors(tree: ast.Module) -> list[str]:
    """Pin both reducers from inputs, through the guard, to the live return."""
    solve_prefix = r"""_fail_if_co_map_states_not_leading(
    state_names=state_names, co_map_state_names=co_map_state_names
)
extra_param_names = _get_extra_param_names(
    Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
)
if pareto_weights is not None:
    extra_param_names = list(
        dict.fromkeys((*extra_param_names, *pareto_weights.param_names))
    )
Q_and_F = productmap(
    func=Q_and_F,
    variables=action_names,
    batch_sizes=dict.fromkeys(action_names, 0),
)
"""
    solve_suffix = r"""inner_state_names = tuple(
    name for name in state_names if name not in co_map_state_names
)
mapped = productmap(
    func=max_Q_over_a,
    variables=inner_state_names,
    batch_sizes={name: batch_sizes[name] for name in inner_state_names},
)
if fold_state_names:
    _fail_if_collective(
        fold_state_names=fold_state_names, stakeholders=stakeholders
    )
    mapped = _wrap_with_fold_reduction(
        cast("Callable[..., FloatND]", mapped),
        fold_state_names=fold_state_names,
        fold_weights=fold_weights,
        fold_conditioning=fold_conditioning,
        inner_state_names=inner_state_names,
        action_names=action_names,
        state_names=state_names,
        extra_param_names=extra_param_names,
    )
if not co_map_state_names:
    return mapped
mapped = allow_args(mapped)
for state_name, v_arr_in_axes in zip(
    reversed(co_map_state_names), reversed(co_map_v_arr_in_axes), strict=True
):
    mapped = vmap_1d(
        func=mapped,
        variables=(state_name,),
        co_mapped_in_axes=MappingProxyType(
            {"next_regime_to_V_arr": v_arr_in_axes}
        ),
        callable_with="only_args",
    )
return cast("MaxQOverAFunction", allow_only_kwargs(mapped, enforce=False))
"""
    simulate_prefix = r"""extra_param_names = _get_extra_param_names(
    Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
)
if pareto_weights is not None:
    extra_param_names = list(
        dict.fromkeys((*extra_param_names, *pareto_weights.param_names))
    )
Q_and_F = productmap(
    func=Q_and_F,
    variables=action_names,
    batch_sizes=dict.fromkeys(action_names, 0),
)
"""
    contracts = (
        ("get_max_Q_over_a", solve_prefix, solve_suffix),
        (
            "get_argmax_and_max_Q_over_a",
            simulate_prefix,
            "return argmax_and_max_Q_over_a\n",
        ),
    )
    signatures: dict[str, tuple[tuple[str, ...], tuple[str | None, ...]]] = {
        "get_max_Q_over_a": (
            (
                "Q_and_F",
                "batch_sizes",
                "action_names",
                "state_names",
                "n_discrete_action_axes",
                "has_taste_shocks",
                "co_map_state_names",
                "co_map_v_arr_in_axes",
                "stakeholders",
                "pareto_weights",
                "fold_state_names",
                "fold_weights",
                "fold_conditioning",
            ),
            (
                None,
                None,
                None,
                None,
                "0",
                "False",
                "()",
                "()",
                "None",
                "None",
                "()",
                "MappingProxyType({})",
                "MappingProxyType({})",
            ),
        ),
        "get_argmax_and_max_Q_over_a": (
            (
                "Q_and_F",
                "action_names",
                "state_names",
                "n_discrete_action_axes",
                "has_taste_shocks",
                "stakeholders",
                "pareto_weights",
            ),
            (None, None, None, "0", "False", "None", "None"),
        ),
    }
    errors: list[str] = []
    for outer_name, expected_prefix, expected_suffix in contracts:
        try:
            outer = _definition(tree, outer_name)
        except ValueError as error:
            errors.append(f"{outer_name}: {error}")
            continue
        if outer.decorator_list:
            errors.append(f"{outer_name}: builder decorators are not allowlisted")
        names, defaults = signatures[outer_name]
        if not _keyword_only_signature(outer, names=names, defaults=defaults):
            errors.append(f"{outer_name}: builder signature or defaults changed")
        body = _body_without_docstring(outer)
        guards = [
            (index, statement)
            for index, statement in enumerate(body)
            if isinstance(statement, ast.If)
            and ast.unparse(statement.test) == "has_taste_shocks"
        ]
        if len(guards) != 1:
            errors.append(
                f"{outer_name}: live taste guard count changed: {len(guards)}"
            )
            continue
        index, _guard = guards[0]
        if not _statements_match(body[:index], expected_prefix):
            errors.append(
                f"{outer_name}: pre-guard Q/action wiring differs from the allowlist"
            )
        if not _statements_match(body[index + 1 :], expected_suffix):
            errors.append(
                f"{outer_name}: certified reducer is not the exact mapped/returned route"
            )
    errors.extend(
        _module_contract_errors(
            tree,
            label="max-Q builders",
            relevant_import_names={
                "MappingProxyType",
                "ParetoWeights",
                "allow_args",
                "allow_only_kwargs",
                "argmax_and_max",
                "cast",
                "collective_argmax_and_readout",
                "collective_readout",
                "EULER_GAMMA",
                "jax",
                "jnp",
                "logsum_and_softmax",
                "math",
                "productmap",
                "vmap_1d",
                "with_signature",
            },
            expected_imports=[
                "import math",
                "from types import MappingProxyType",
                "from typing import cast",
                "import jax",
                "import jax.numpy as jnp",
                "from dags import with_signature",
                "from _lcm.logsum import EULER_GAMMA, logsum_and_softmax",
                "from _lcm.regime_building.argmax import argmax_and_max",
                "from _lcm.regime_building.collective import ParetoWeights, collective_argmax_and_readout, collective_readout",
                "from _lcm.utils.dispatchers import productmap, vmap_1d",
                "from _lcm.utils.functools import allow_args, allow_only_kwargs",
            ],
            expected_binding_counts={
                "MappingProxyType": 1,
                "ParetoWeights": 1,
                "allow_args": 1,
                "allow_only_kwargs": 1,
                "argmax_and_max": 1,
                "cast": 1,
                "collective_argmax_and_readout": 1,
                "collective_readout": 1,
                "dict": 0,
                "draw_taste_shock_noise": 1,
                "enumerate": 0,
                "EULER_GAMMA": 1,
                "get_argmax_and_max_Q_over_a": 1,
                "get_max_Q_over_a": 1,
                "jax": 1,
                "jnp": 1,
                "list": 0,
                "logsum_and_softmax": 1,
                "math": 1,
                "productmap": 1,
                "range": 0,
                "reversed": 0,
                "tuple": 0,
                "vmap_1d": 1,
                "with_signature": 1,
                "zip": 0,
            },
        )
    )
    return errors


def _grid_search_caller_errors(tree: ast.Module) -> list[str]:
    """Pin solve-side action metadata and the exact live core publication."""
    errors: list[str] = []
    try:
        cls, method = _method_definition(
            tree, class_name="GridSearch", method_name="build_period_kernels"
        )
    except ValueError as error:
        return [f"solve caller: {error}"]
    if (
        [ast.unparse(item) for item in cls.decorator_list]
        != ["beartype(conf=REGIME_CONF)", "dataclass(frozen=True, kw_only=True)"]
        or [ast.unparse(item) for item in cls.bases] != ["Solver"]
        or cls.keywords
    ):
        errors.append("solve caller: GridSearch class binding/decorators changed")
    args = method.args
    if not (
        not args.posonlyargs
        and tuple(item.arg for item in args.args) == ("self",)
        and args.vararg is None
        and tuple(item.arg for item in args.kwonlyargs) == ("context",)
        and args.kw_defaults == [None]
        and args.kwarg is None
        and not args.defaults
        and not method.decorator_list
    ):
        errors.append("solve caller: build_period_kernels signature changed")
    expected_body = r"""from _lcm.regime_building.max_Q_over_a import get_max_Q_over_a
from _lcm.regime_building.processing import get_conditioned_fold_weights_by_code
built: dict[int, MaxQOverAFunction] = {}
result: dict[int, PeriodKernel] = {}
fold_weights: dict[StateName, FloatND] = {}
fold_conditioning: dict[StateName, StateName] = {}
for name in context.fold_state_names:
    process = cast("_ContinuousStochasticProcess", context.grids[name])
    if process.state_conditioned is None:
        fold_weights[name] = process.get_transition_probs()[0]
    else:
        fold_weights[name] = get_conditioned_fold_weights_by_code(
            name=name, grid=process, grids=context.grids
        )
        fold_conditioning[name] = process.state_conditioned.on
for period, Q_and_F in context.Q_and_F_functions.items():
    q_id = id(Q_and_F)
    if q_id not in built:
        func = get_max_Q_over_a(
            Q_and_F=Q_and_F,
            batch_sizes={
                name: grid.batch_size
                for name, grid in context.grids.items()
                if name in context.state_action_space.state_names
            },
            action_names=context.state_action_space.action_names,
            state_names=context.state_action_space.state_names,
            n_discrete_action_axes=len(
                context.state_action_space.discrete_actions
            ),
            has_taste_shocks=context.has_taste_shocks,
            co_map_state_names=context.co_map_state_names,
            co_map_v_arr_in_axes=context.co_map_v_arr_in_axes,
            stakeholders=context.stakeholders,
            pareto_weights=context.pareto_weights,
            fold_state_names=context.fold_state_names,
            fold_weights=MappingProxyType(fold_weights),
            fold_conditioning=MappingProxyType(fold_conditioning),
        )
        built[q_id] = jax.jit(func) if context.enable_jit else func
    result[period] = _GridSearchPeriodKernel(
        core=built[q_id],
        regime_name=context.regime_name,
        collective=context.stakeholders is not None,
        same_period_ref_regimes=context.same_period_ref_regimes,
        edge_reference_regimes=context.edge_reference_regimes,
        edge_target_regimes=context.edge_target_regimes,
    )
return SolutionKernels(period_kernels=MappingProxyType(result))
"""
    if not _body_matches(method, expected_body):
        errors.append(
            "solve caller: action metadata, core wiring, or published result changed"
        )
    errors.extend(
        _module_contract_errors(
            tree,
            label="solve caller",
            relevant_import_names={
                "MappingProxyType",
                "REGIME_CONF",
                "Solver",
                "beartype",
                "cast",
                "dataclass",
                "jax",
            },
            expected_imports=[
                "from dataclasses import dataclass, replace",
                "from types import MappingProxyType",
                "from typing import cast",
                "import jax",
                "from beartype import beartype",
                "from _lcm.beartype_conf import REGIME_CONF",
                "from _lcm.solution.contract import ConstraintRouteContext, ContinuationPayload, KernelResult, PeriodKernel, SolutionKernels, Solver, SolverBuildContext, simulation_route",
            ],
            expected_binding_counts={
                "GridSearch": 1,
                "MappingProxyType": 1,
                "REGIME_CONF": 1,
                "Solver": 1,
                "beartype": 1,
                "cast": 1,
                "dataclass": 1,
                "id": 0,
                "jax": 1,
                "len": 0,
            },
        )
    )
    return errors


def _processing_caller_errors(tree: ast.Module) -> list[str]:
    """Pin simulation metadata, spacemapping, and live phase publication."""
    errors: list[str] = []
    try:
        builder = _definition(tree, "_build_argmax_and_max_Q_over_a_per_period")
        live = _definition(tree, "_build_simulation_phase")
    except ValueError as error:
        return [f"simulate caller: {error}"]
    args = builder.args
    if not (
        not args.posonlyargs
        and not args.args
        and args.vararg is None
        and tuple(item.arg for item in args.kwonlyargs)
        == (
            "state_action_space",
            "Q_and_F_functions",
            "enable_jit",
            "has_taste_shocks",
            "stakeholders",
            "pareto_weights",
        )
        and tuple(
            ast.unparse(item) if item is not None else None for item in args.kw_defaults
        )
        == (None, None, None, "False", "None", "None")
        and args.kwarg is None
        and not args.defaults
        and not builder.decorator_list
    ):
        errors.append("simulate caller: period-builder signature changed")
    expected_builder = r"""spacemapped_names = tuple(state_action_space.states)
if has_taste_shocks:
    spacemapped_names = (*spacemapped_names, "taste_shock_key")
built: dict[int, ArgmaxQOverAFunction] = {}
result: dict[int, ArgmaxQOverAFunction] = {}
for period, Q_and_F in Q_and_F_functions.items():
    q_id = id(Q_and_F)
    if q_id not in built:
        func = get_argmax_and_max_Q_over_a(
            Q_and_F=Q_and_F,
            action_names=state_action_space.action_names,
            state_names=state_action_space.state_names,
            n_discrete_action_axes=len(state_action_space.discrete_actions),
            has_taste_shocks=has_taste_shocks,
            stakeholders=stakeholders,
            pareto_weights=pareto_weights,
        )
        if enable_jit:
            func = jax.jit(func)
        built[q_id] = simulation_spacemap(
            func=func,
            action_names=(),
            state_names=spacemapped_names,
        )
    result[period] = built[q_id]
return MappingProxyType(result)
"""
    if not _body_matches(builder, expected_builder):
        errors.append(
            "simulate caller: action metadata, reducer wiring, or spacemap changed"
        )

    live_body = _body_without_docstring(live)
    live_assignments = [
        statement
        for statement in live_body
        if isinstance(statement, ast.Assign)
        and any(
            _target_names(target) == ("argmax_and_max_Q_over_a",)
            for target in statement.targets
        )
    ]
    expected_live_assignment = r"""argmax_and_max_Q_over_a = (
    _build_argmax_and_max_Q_over_a_per_period(
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        enable_jit=enable_jit,
        has_taste_shocks=has_taste_shocks,
        stakeholders=stakeholders,
        pareto_weights=pareto_weights,
    )
)
"""
    if len(live_assignments) != 1 or not _statements_match(
        live_assignments, expected_live_assignment
    ):
        errors.append("simulate caller: live phase does not call the certified builder")
    returns = [node for node in ast.walk(live) if isinstance(node, ast.Return)]
    if len(returns) != 1 or live_body[-1] is not returns[0]:
        errors.append("simulate caller: live phase gained a bypass return")
    elif not (
        isinstance(returns[0].value, ast.Call)
        and _call_name(returns[0].value) == "SimulationPhase"
        and _name(
            _keyword(returns[0].value, "argmax_and_max_Q_over_a"),
            "argmax_and_max_Q_over_a",
        )
    ):
        errors.append("simulate caller: certified reducer mapping is not published")
    live_bindings = _scope_binding_counts(live_body)
    expected_live_bindings = {
        "_build_argmax_and_max_Q_over_a_per_period": 0,
        "argmax_and_max_Q_over_a": 1,
        "has_taste_shocks": 0,
    }
    if any(
        live_bindings.get(name, 0) != count
        for name, count in expected_live_bindings.items()
    ):
        errors.append("simulate caller: live taste/reducer bindings changed")
    errors.extend(
        _module_contract_errors(
            tree,
            label="simulate caller",
            relevant_import_names={
                "MappingProxyType",
                "get_argmax_and_max_Q_over_a",
                "jax",
                "simulation_spacemap",
            },
            expected_imports=[
                "from types import MappingProxyType",
                "import jax",
                "from _lcm.regime_building.max_Q_over_a import get_argmax_and_max_Q_over_a",
                "from _lcm.utils.dispatchers import simulation_spacemap, vmap_1d",
            ],
            expected_binding_counts={
                "MappingProxyType": 1,
                "_build_argmax_and_max_Q_over_a_per_period": 1,
                "_build_simulation_phase": 1,
                "get_argmax_and_max_Q_over_a": 1,
                "id": 0,
                "jax": 1,
                "len": 0,
                "simulation_spacemap": 1,
                "tuple": 0,
            },
        )
    )
    return errors


def _corridor_errors(
    tree: ast.Module,
    *,
    outer_name: str,
    nested_name: str,
    simulate: bool,
) -> list[str]:
    label = "simulate" if simulate else "solve"
    errors: list[str] = []
    try:
        nested = _ordinary_nested(tree, outer_name=outer_name, nested_name=nested_name)
    except ValueError as error:
        return [f"{label}: {error}"]
    if not _exact_reducer_decorator(nested, simulate=simulate, taste_shocks=False):
        errors.append(f"{label}: ordinary reducer signature wrapper changed")
    if len(nested.body) != 3:
        errors.append(
            f"{label}: ordinary reducer corridor must contain exactly origin, "
            f"collective branch, singleton return; found {len(nested.body)} statements"
        )
        return errors
    origin, collective, singleton = nested.body
    if not _q_and_f_origin(origin):
        errors.append(
            f"{label}: first corridor statement is not exact Q_arr/F_arr origin"
        )
    if not isinstance(collective, ast.If) or not _exact_collective_body(
        collective, simulate=simulate
    ):
        errors.append(
            f"{label}: collective corridor contains a non-allowlisted transformation"
        )
    singleton_ok = (
        _exact_singleton_simulate_return(singleton)
        if simulate
        else _exact_singleton_solve_return(singleton)
    )
    if not singleton_ok:
        errors.append(
            f"{label}: singleton corridor does not pass raw Q_arr/F_arr directly "
            "to the full reducer"
        )
    for statement in ast.walk(nested):
        if isinstance(statement, ast.stmt) and statement is not origin:
            assigned = _assigned_names(statement)
            if {"Q_arr", "F_arr"} & assigned:
                errors.append(
                    f"{label}: candidate arrays are reassigned or deleted at line "
                    f"{getattr(statement, 'lineno', '?')}"
                )
    return errors


def _taste_corridor_errors(
    tree: ast.Module,
    *,
    outer_name: str,
    nested_name: str,
    simulate: bool,
) -> list[str]:
    """Pin one taste-shock route from exact Q/F origin through its full reducer."""
    label = "taste-shock simulate" if simulate else "taste-shock solve"
    try:
        nested = _taste_nested(tree, outer_name=outer_name, nested_name=nested_name)
    except ValueError as error:
        return [f"{label}: {error}"]
    errors: list[str] = []
    if not _exact_reducer_decorator(nested, simulate=simulate, taste_shocks=True):
        errors.append(f"{label}: taste reducer signature wrapper changed")
    if not _nested_reducer_signature(nested):
        errors.append(f"{label}: nested reducer signature changed")

    expected_solve = r"""Q_arr, F_arr = Q_and_F(
    next_regime_to_V_arr=next_regime_to_V_arr,
    **states_actions_params,
)
Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)
continuous_axes = tuple(range(n_discrete_action_axes, Q_arr.ndim))
Qc = Q_masked.max(axis=continuous_axes) if continuous_axes else Q_masked
smoothed, _ = logsum_and_softmax(
    values=Qc,
    scale=cast(
        "ScalarFloat", states_actions_params[TASTE_SHOCK_SCALE_PARAM]
    ),
    axes=tuple(range(Qc.ndim)),
)
return smoothed
"""
    expected_simulate = r"""taste_shock_key = cast(
    "Array", states_actions_params.pop("taste_shock_key")
)
Q_arr, F_arr = Q_and_F(
    next_regime_to_V_arr=next_regime_to_V_arr,
    **states_actions_params,
)
Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)
n_discrete_cells = math.prod(Q_arr.shape[:n_discrete_action_axes])
n_continuous_cells = math.prod(Q_arr.shape[n_discrete_action_axes:])
Q_flat = Q_masked.reshape(n_discrete_cells, n_continuous_cells)
continuous_argmax = jnp.argmax(Q_flat, axis=1)
Qc = Q_flat.max(axis=1)
scale = cast("FloatND", states_actions_params[TASTE_SHOCK_SCALE_PARAM])
noise = draw_taste_shock_noise(
    key=taste_shock_key, shape=Qc.shape, scale=scale
)
noisy_Qc = Qc + noise
discrete_argmax = jnp.argmax(noisy_Qc)
flat_index = (
    discrete_argmax * n_continuous_cells
    + continuous_argmax[discrete_argmax]
)
return flat_index.astype(jnp.int32), Qc[discrete_argmax]
"""
    expected = expected_simulate if simulate else expected_solve
    if not _body_matches(nested, expected):
        errors.append(
            f"{label}: executable body differs from the exact raw-Q/F full reduction"
        )
    return errors


def _taste_noise_errors(tree: ast.Module) -> list[str]:
    """Pin the per-discrete-cell mean-zero Gumbel helper and its imports."""
    errors: list[str] = []
    try:
        node = _definition(tree, "draw_taste_shock_noise")
    except ValueError as error:
        return [f"taste noise: {error}"]
    if node.decorator_list:
        errors.append("taste noise: decorators are not allowlisted")
    if not _keyword_only_signature(node, names=("key", "shape", "scale")):
        errors.append("taste noise: signature changed")
    expected_body = r"""return scale * (
    jax.random.gumbel(key, shape) - EULER_GAMMA
)
"""
    if not _body_matches(node, expected_body):
        errors.append(
            "taste noise: body is not one scaled mean-zero Gumbel draw per cell"
        )

    imports = _relevant_imports(
        tree,
        names={
            "cast",
            "jax",
            "jnp",
            "math",
            "range",
            "tuple",
            "EULER_GAMMA",
            "logsum_and_softmax",
        },
    )
    expected_imports = sorted(
        [
            "import jax",
            "import jax.numpy as jnp",
            "import math",
            "from typing import cast",
            "from _lcm.logsum import EULER_GAMMA, logsum_and_softmax",
        ]
    )
    if imports != expected_imports:
        errors.append("taste noise: JAX/logsum import bindings changed")

    constants = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            _target_names(target) == ("TASTE_SHOCK_SCALE_PARAM",)
            for target in statement.targets
        )
    ]
    if not (
        len(constants) == 1
        and isinstance(constants[0].value, ast.Constant)
        and constants[0].value.value == "taste_shocks__scale"
    ):
        errors.append("taste noise: taste-shock scale parameter binding changed")

    if any(
        isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)
        and {
            "cast",
            "draw_taste_shock_noise",
            "jax",
            "jnp",
            "math",
            "range",
            "tuple",
            "EULER_GAMMA",
            "logsum_and_softmax",
        }
        & _assigned_names(statement)
        for statement in tree.body
    ):
        errors.append(
            "taste noise: module-level helper/import rebinding is not allowlisted"
        )
    return errors


def _logsum_reducer_errors(tree: ast.Module) -> list[str]:
    """Pin the stable full-axis logsum and softmax helper exactly."""
    errors: list[str] = []
    try:
        node = _definition(tree, "logsum_and_softmax")
    except ValueError as error:
        return [f"logsum reducer: {error}"]
    if node.decorator_list:
        errors.append("logsum reducer: decorators are not allowlisted")
    if not _keyword_only_signature(node, names=("values", "scale", "axes")):
        errors.append("logsum reducer: signature changed")
    expected_body = r"""v_max = jnp.max(values, axis=axes, keepdims=True)
finite_max = jnp.where(jnp.isneginf(v_max), 0.0, v_max)
shifted = (values - finite_max) / scale
smoothed = jnp.squeeze(finite_max, axis=axes) + scale * logsumexp(
    shifted, axis=axes
)
all_masked = jnp.all(jnp.isneginf(values), axis=axes, keepdims=True)
probs = jnp.where(all_masked, 0.0, jax.nn.softmax(shifted, axis=axes))
return smoothed, probs
"""
    if not _body_matches(node, expected_body):
        errors.append(
            "logsum reducer: executable body differs from the exact full-value flow"
        )
    imports = _relevant_imports(tree, names={"jax", "jnp", "logsumexp"})
    expected_imports = sorted(
        [
            "import jax",
            "import jax.numpy as jnp",
            "from jax.scipy.special import logsumexp",
        ]
    )
    if imports != expected_imports:
        errors.append("logsum reducer: JAX/logsumexp import bindings changed")
    gamma = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            _target_names(target) == ("EULER_GAMMA",) for target in statement.targets
        )
    ]
    if not (
        len(gamma) == 1
        and isinstance(gamma[0].value, ast.Constant)
        and gamma[0].value.value == 0.5772156649015329
    ):
        errors.append("logsum reducer: Euler-Gamma centering constant changed")
    if any(
        isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)
        and {"logsum_and_softmax", "jax", "jnp", "logsumexp"}
        & _assigned_names(statement)
        for statement in tree.body
    ):
        errors.append("logsum reducer: module-level rebinding is not allowlisted")
    return errors


def _argmax_reducer_errors(tree: ast.Module) -> list[str]:
    """Certify the shared argmax and its representation helpers exactly."""
    errors: list[str] = []
    try:
        node = _definition(tree, "argmax_and_max")
        move = _definition(tree, "_move_axes_to_back")
        flatten = _definition(tree, "_flatten_last_n_axes")
    except ValueError as error:
        return [f"argmax reducer: {error}"]

    if node.decorator_list or move.decorator_list or flatten.decorator_list:
        errors.append("argmax reducer: decorators are not allowlisted")
    if not _positional_signature(
        node,
        names=("a", "axis", "initial", "where"),
        defaults=("None", "None", "None"),
    ):
        errors.append("argmax reducer: signature/defaults changed")
    if not _positional_signature(move, names=("a", "axes")):
        errors.append("argmax reducer: axis-move helper signature changed")
    if not _positional_signature(flatten, names=("a", "n")):
        errors.append("argmax reducer: flatten helper signature changed")

    expected_argmax = r"""if axis is None:
    axis = tuple(range(a.ndim))
elif isinstance(axis, int):
    axis = (axis,)
if a.ndim == 0 or len(axis) == 0:
    return jnp.array(0, dtype=jnp.int32), a
if a.ndim != 0:
    a = _move_axes_to_back(a, axes=axis)
    a = _flatten_last_n_axes(a, n=len(axis))
if where is not None and where.ndim != 0:
    where = _move_axes_to_back(where, axes=axis)
    where = _flatten_last_n_axes(where, n=len(axis))
_max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)
max_value_mask = a == _max
if where is not None:
    max_value_mask = jnp.logical_and(max_value_mask, where)
_argmax = jnp.argmax(max_value_mask, axis=-1).astype(jnp.int32)
return _argmax, _max.reshape(_argmax.shape)
"""
    expected_move = r"""front_axes = sorted(set(range(a.ndim)) - set(axes))
return a.transpose((*front_axes, *axes))
"""
    expected_flatten = r"""return a.reshape(*a.shape[:-n], -1)
"""
    if not _body_matches(node, expected_argmax):
        errors.append(
            "argmax reducer: executable body differs from the full paired "
            "value/feasibility reduction"
        )
    if not _body_matches(move, expected_move):
        errors.append(
            "argmax reducer: action-axis move is not the exact order-preserving "
            "representation change"
        )
    if not _body_matches(flatten, expected_flatten):
        errors.append(
            "argmax reducer: action-axis flatten is not the exact shape-only "
            "representation change"
        )

    if any(
        isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)
        and {
            "argmax_and_max",
            "_move_axes_to_back",
            "_flatten_last_n_axes",
        }
        & _assigned_names(statement)
        for statement in tree.body
    ):
        errors.append("argmax reducer: module-level rebinding is not allowlisted")
    return errors


def _collective_reducer_errors(tree: ast.Module) -> list[str]:
    """Certify collective scalarization, argmax, delegation, and gather exactly."""
    errors: list[str] = []
    try:
        argmax_node = _definition(tree, "collective_argmax_and_readout")
        readout_node = _definition(tree, "collective_readout")
        weighted = _definition(tree, "_weighted_sum")
        gather = _definition(tree, "_gather_along_actions")
    except ValueError as error:
        return [f"collective reducer: {error}"]

    if any(
        node.decorator_list for node in (argmax_node, readout_node, weighted, gather)
    ):
        errors.append("collective reducer: decorators are not allowlisted")
    signatures = (
        (
            argmax_node,
            ("stakeholder_Q", "feasibility", "weights", "action_axes"),
            "argmax/readout",
        ),
        (
            readout_node,
            ("stakeholder_Q", "feasibility", "weights", "action_axes"),
            "readout delegation",
        ),
        (weighted, ("stakeholder_Q", "weights"), "weighted scalarization"),
        (gather, ("q", "argmax_flat", "action_axes"), "value gather"),
    )
    for node, names, label in signatures:
        if not _keyword_only_signature(node, names=names):
            errors.append(f"collective reducer: {label} signature changed")

    expected_argmax = r"""if not stakeholder_Q:
    msg = "collective_argmax_and_readout requires at least one stakeholder."
    raise ValueError(msg)
if set(stakeholder_Q) != set(weights):
    msg = (
        "stakeholder_Q and weights must have identical keys; got "
        f"{sorted(stakeholder_Q)} vs {sorted(weights)}."
    )
    raise ValueError(msg)
objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)
argmax_flat, _ = argmax_and_max(
    objective, axis=action_axes, initial=-jnp.inf, where=feasibility
)
dissolution = (
    ~jnp.any(feasibility, axis=action_axes) if action_axes else ~feasibility
)
values = {
    name: jnp.where(
        dissolution,
        -jnp.inf,
        _gather_along_actions(
            q=q, argmax_flat=argmax_flat, action_axes=action_axes
        ),
    )
    for name, q in stakeholder_Q.items()
}
return argmax_flat, values, dissolution
"""
    expected_readout = r"""_argmax_flat, values, dissolution = collective_argmax_and_readout(
    stakeholder_Q=stakeholder_Q,
    feasibility=feasibility,
    weights=weights,
    action_axes=action_axes,
)
return values, dissolution
"""
    expected_weighted = r"""names = list(stakeholder_Q)
def _term(name: str) -> FloatND:
    return zero_safe_weighted_term(
        weight=jnp.asarray(weights[name]),
        value=stakeholder_Q[name],
        subnormal_is_accounted_for=False,
    )
terms = [_term(name) for name in names]
if len(terms) <= _LARGEST_ORDER_FREE_HOUSEHOLD:
    objective = terms[0]
    for term in terms[1:]:
        objective = objective + term
    return objective
return sum_in_value_order(jnp.stack(terms, axis=0), axis=0)
"""
    expected_gather = r"""if not action_axes:
    return q
q_moved = _move_axes_to_back(q, axes=action_axes)
q_flat = _flatten_last_n_axes(q_moved, n=len(action_axes))
gathered = jnp.take_along_axis(q_flat, argmax_flat[..., None], axis=-1)
return gathered[..., 0]
"""
    expected = (
        (argmax_node, expected_argmax, "household argmax/readout"),
        (readout_node, expected_readout, "solve-side delegation"),
        (weighted, expected_weighted, "pointwise weighted scalarization"),
        (gather, expected_gather, "shared-index stakeholder gather"),
    )
    for node, body, label in expected:
        if not _body_matches(node, body):
            errors.append(
                f"collective reducer: {label} differs from the exact allowlisted flow"
            )

    if any(
        isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)
        and {
            "collective_argmax_and_readout",
            "collective_readout",
            "_weighted_sum",
            "_gather_along_actions",
        }
        & _assigned_names(statement)
        for statement in tree.body
    ):
        errors.append("collective reducer: module-level rebinding is not allowlisted")
    return errors


def verify_direct_candidate_flow(*, repo_root: Path) -> dict[str, Any]:
    """Verify the complete direct-flow architecture against one repository tree."""
    root = repo_root.resolve()
    errors: list[str] = []
    offending: set[str] = set()
    parsed: dict[str, ast.Module] = {}
    if len(set(_CERTIFIED_CORRIDOR_SOURCES)) != len(_CERTIFIED_CORRIDOR_SOURCES):
        errors.append("certificate: the corridor source tuple contains duplicates")
        offending.add("tests/candidate_certificate/direct_flow.py")
    if set(_SOURCE_SEALS) != set(_CERTIFIED_CORRIDOR_SOURCES):
        errors.append("certificate: the source-seal set differs from the corridor set")
        offending.add("tests/candidate_certificate/direct_flow.py")
    for relative in _CERTIFIED_CORRIDOR_SOURCES:
        path = root / relative
        try:
            actual_sha256 = sha256_file(path)
            expected_sha256 = _SOURCE_SEALS[relative]
            if actual_sha256 != expected_sha256:
                errors.append(
                    f"{relative}: source seal mismatch: expected {expected_sha256}, "
                    f"got {actual_sha256}"
                )
                offending.add(relative)
            parsed[relative] = ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            )
        except (KeyError, OSError, SyntaxError, UnicodeError) as error:
            errors.append(f"{relative}: {error}")
            offending.add(relative)
    max_tree = parsed.get(MAX_Q_SOURCE)
    if max_tree is not None:
        binding_errors = _productmap_binding_errors(
            max_tree, outer_name="get_max_Q_over_a", nested_name="max_Q_over_a"
        )
        binding_errors += _productmap_binding_errors(
            max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
        )
        solve_errors = _corridor_errors(
            max_tree,
            outer_name="get_max_Q_over_a",
            nested_name="max_Q_over_a",
            simulate=False,
        )
        simulate_errors = _corridor_errors(
            max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
            simulate=True,
        )
        taste_solve_errors = _taste_corridor_errors(
            max_tree,
            outer_name="get_max_Q_over_a",
            nested_name="max_Q_over_a",
            simulate=False,
        )
        taste_simulate_errors = _taste_corridor_errors(
            max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
            simulate=True,
        )
        taste_noise_errors = _taste_noise_errors(max_tree)
        wiring_errors = _max_builder_wiring_errors(max_tree)
        max_errors = (
            binding_errors
            + solve_errors
            + simulate_errors
            + taste_solve_errors
            + taste_simulate_errors
            + taste_noise_errors
            + wiring_errors
        )
        errors.extend(max_errors)
        if max_errors:
            offending.add(MAX_Q_SOURCE)
    argmax_tree = parsed.get(ARGMAX_SOURCE)
    if argmax_tree is not None:
        new_errors = _argmax_reducer_errors(argmax_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(ARGMAX_SOURCE)
    collective_tree = parsed.get(COLLECTIVE_SOURCE)
    if collective_tree is not None:
        new_errors = _collective_reducer_errors(collective_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(COLLECTIVE_SOURCE)
    logsum_tree = parsed.get(LOGSUM_SOURCE)
    if logsum_tree is not None:
        new_errors = _logsum_reducer_errors(logsum_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(LOGSUM_SOURCE)
    grid_tree = parsed.get(GRID_SEARCH_SOURCE)
    if grid_tree is not None:
        new_errors = _grid_search_caller_errors(grid_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(GRID_SEARCH_SOURCE)
    processing_tree = parsed.get(PROCESSING_SOURCE)
    if processing_tree is not None:
        new_errors = _processing_caller_errors(processing_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(PROCESSING_SOURCE)
    grid_base_tree = parsed.get(GRID_BASE_SOURCE)
    if grid_base_tree is not None:
        new_errors = _grid_base_errors(grid_base_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(GRID_BASE_SOURCE)
    engine_tree = parsed.get(ENGINE_SOURCE)
    if engine_tree is not None:
        new_errors = _engine_state_action_space_errors(engine_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(ENGINE_SOURCE)
    transitions_tree = parsed.get(SIMULATION_TRANSITIONS_SOURCE)
    if transitions_tree is not None:
        new_errors = _simulation_state_action_space_errors(transitions_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(SIMULATION_TRANSITIONS_SOURCE)
    simulation_tree = parsed.get(SIMULATION_SOURCE)
    if simulation_tree is not None:
        new_errors = _simulation_state_action_space_caller_errors(simulation_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(SIMULATION_SOURCE)
    return {
        "ok": not errors,
        "result": "pass" if not errors else "fail",
        "errors": errors,
        "offending_paths": sorted(offending),
        "routes": {
            "singleton_solve": "Q_and_F -> Q_arr.max(where=F_arr)",
            "singleton_simulate": "Q_and_F -> argmax_and_max(Q_arr, where=F_arr)",
            "collective_solve": (
                "Q_and_F -> trailing stakeholder split -> "
                "collective_readout(feasibility=F_arr)"
            ),
            "collective_simulate": (
                "Q_and_F -> trailing stakeholder split -> "
                "collective_argmax_and_readout(feasibility=F_arr)"
            ),
            "taste_shock_solve": (
                "Q_and_F -> exact feasibility mask -> continuous max -> "
                "full discrete logsum"
            ),
            "taste_shock_simulate": (
                "Q_and_F -> exact feasibility mask -> row-major continuous max -> "
                "per-cell Gumbel-max -> exact flat index/value"
            ),
        },
        "certified_corridor_sources": list(_CERTIFIED_CORRIDOR_SOURCES),
        "source_seals": dict(_SOURCE_SEALS),
    }


def _insert_before_nth(
    text: str, marker: str, insertion: str, *, occurrence: int
) -> str:
    start = -1
    for _ in range(occurrence):
        start = text.find(marker, start + 1)
        if start < 0:
            raise ValueError(
                f"marker not found for occurrence {occurrence}: {marker!r}"
            )
    return text[:start] + insertion + text[start:]


def _replace_nth(text: str, marker: str, replacement: str, *, occurrence: int) -> str:
    """Replace exactly the requested occurrence of one production marker."""
    start = -1
    for _ in range(occurrence):
        start = text.find(marker, start + 1)
        if start < 0:
            raise ValueError(
                f"marker not found for occurrence {occurrence}: {marker!r}"
            )
    return text[:start] + replacement + text[start + len(marker) :]


def direct_flow_mutations(source: str) -> dict[str, str]:
    """Generate the required route/value/support/shape/index perturbation family."""
    mutations: dict[str, str] = {}
    solve_singleton = "            return Q_arr.max(where=F_arr, initial=-jnp.inf)"
    simulate_singleton = (
        "            return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)"
    )
    collective_marker = "                action_axes = tuple(range(F_arr.ndim))"

    mutations["singleton_solve:q_order"] = _insert_before_nth(
        source,
        solve_singleton,
        "            Q_flat = Q_arr.reshape(-1)\n"
        "            order_filter = Q_flat[0] > Q_flat[1]\n"
        "            F_arr = jnp.where(\n"
        "                order_filter,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=1,
    )
    mutations["singleton_simulate:mt9_rank_permutation"] = _insert_before_nth(
        source,
        simulate_singleton,
        "            Q_flat = Q_arr.reshape(-1)\n"
        "            mt9_order = (\n"
        "                (Q_flat[0] > Q_flat[2])\n"
        "                & (Q_flat[2] > Q_flat[1])\n"
        "                & (Q_flat[1] > Q_flat[3])\n"
        "                & (Q_flat[3] > Q_flat[4])\n"
        "                & (Q_flat[4] > Q_flat[5])\n"
        "                & jnp.all(F_arr)\n"
        "            )\n"
        "            F_arr = jnp.where(\n"
        "                mt9_order,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=1,
    )
    mutations["singleton_simulate:q_gap"] = _insert_before_nth(
        source,
        simulate_singleton,
        "            gap_filter = (\n"
        "                Q_arr.reshape(-1)[0] - Q_arr.reshape(-1)[1] > 0.5\n"
        "            )\n"
        "            F_arr = jnp.where(\n"
        "                gap_filter,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=1,
    )
    mutations["collective_solve:support_size"] = _insert_before_nth(
        source,
        collective_marker,
        "                support_filter = jnp.sum(F_arr) > 1\n"
        "                F_arr = jnp.where(\n"
        "                    support_filter,\n"
        "                    F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                    F_arr,\n"
        "                )\n",
        occurrence=1,
    )
    mutations["collective_simulate:shape_axis"] = _insert_before_nth(
        source,
        collective_marker,
        "                shape_filter = (F_arr.ndim == 2) & (F_arr.shape[-1] > 1)\n"
        "                F_arr = jnp.where(\n"
        "                    shape_filter,\n"
        "                    F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                    F_arr,\n"
        "                )\n",
        occurrence=2,
    )

    mutations["singleton_solve:inline_where_transform"] = source.replace(
        "return Q_arr.max(where=F_arr, initial=-jnp.inf)",
        "return Q_arr.max(\n"
        "                where=F_arr.reshape(-1).at[0].set(False)\n"
        "                .reshape(F_arr.shape),\n"
        "                initial=-jnp.inf,\n"
        "            )",
        1,
    )
    mutations["singleton_simulate:inline_q_transform"] = source.replace(
        "return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)",
        "return argmax_and_max(\n"
        "                Q_arr.reshape(-1)[::-1], where=F_arr, initial=-jnp.inf\n"
        "            )",
        1,
    )
    mutations["collective_solve:inline_feasibility_transform"] = source.replace(
        "feasibility=F_arr,",
        "feasibility=F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),",
        1,
    )
    mutations["collective_simulate:action_axis_slice"] = source.replace(
        "name: Q_arr[..., index] for index, name in enumerate(stakeholders)",
        "name: Q_arr[1:, ..., index] for index, name in enumerate(stakeholders)",
        2,
    )
    mutations["solve:wrapped_productmap_input"] = source.replace(
        "func=Q_and_F,\n        variables=action_names,",
        "func=lambda **kwargs: Q_and_F(**kwargs),\n        variables=action_names,",
        1,
    )
    # Replace the second productmap binding independently for simulate.
    first = source.find("func=Q_and_F,\n        variables=action_names,")
    second = source.find("func=Q_and_F,\n        variables=action_names,", first + 1)
    if second < 0:
        raise ValueError("second Q_and_F productmap binding not found")
    mutations["simulate:wrapped_productmap_input"] = source[:second] + source[
        second:
    ].replace(
        "func=Q_and_F,\n        variables=action_names,",
        "func=lambda **kwargs: Q_and_F(**kwargs),\n        variables=action_names,",
        1,
    )

    route_specs = {
        "singleton_solve": (solve_singleton, 1, "            "),
        "singleton_simulate": (simulate_singleton, 1, "            "),
        "collective_solve": (collective_marker, 1, "                "),
        "collective_simulate": (collective_marker, 2, "                "),
    }
    for route, (marker, occurrence, indent) in route_specs.items():
        for index in range(6):
            insertion = (
                f"{indent}F_arr = F_arr.reshape(-1).at[{index}]"
                ".set(False).reshape(F_arr.shape)\n"
            )
            mutations[f"{route}:candidate_index_{index}"] = _insert_before_nth(
                source, marker, insertion, occurrence=occurrence
            )
    taste_mask = "            Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)"
    taste_routes = {
        "taste_shock_solve": 1,
        "taste_shock_simulate": 2,
    }
    semantic_insertions = {
        "q_order": (
            "            Q_flat_attack = Q_arr.reshape(-1)\n"
            "            order_filter = Q_flat_attack[0] > Q_flat_attack[1]\n"
            "            F_arr = jnp.where(\n"
            "                order_filter,\n"
            "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "                F_arr,\n"
            "            )\n"
        ),
        "q_gap": (
            "            gap_filter = (\n"
            "                Q_arr.reshape(-1)[0] - Q_arr.reshape(-1)[1] > 0.5\n"
            "            )\n"
            "            F_arr = jnp.where(\n"
            "                gap_filter,\n"
            "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "                F_arr,\n"
            "            )\n"
        ),
        "support_size": (
            "            support_filter = jnp.sum(F_arr) > 1\n"
            "            F_arr = jnp.where(\n"
            "                support_filter,\n"
            "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "                F_arr,\n"
            "            )\n"
        ),
        "all_feasible": (
            "            all_filter = jnp.all(F_arr)\n"
            "            F_arr = jnp.where(\n"
            "                all_filter,\n"
            "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "                F_arr,\n"
            "            )\n"
        ),
        "intermediate_support": (
            "            intermediate_filter = (jnp.sum(F_arr) > 1) & (~jnp.all(F_arr))\n"
            "            F_arr = jnp.where(\n"
            "                intermediate_filter,\n"
            "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "                F_arr,\n"
            "            )\n"
        ),
        "shape_axis": (
            "            shape_filter = (F_arr.ndim == 2) & (F_arr.shape[-1] > 1)\n"
            "            F_arr = jnp.where(\n"
            "                shape_filter,\n"
            "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "                F_arr,\n"
            "            )\n"
        ),
    }
    for route, occurrence in taste_routes.items():
        for family, insertion in semantic_insertions.items():
            mutations[f"{route}:{family}"] = _insert_before_nth(
                source, taste_mask, insertion, occurrence=occurrence
            )
        for index in range(6):
            mutations[f"{route}:candidate_index_{index}"] = _insert_before_nth(
                source,
                taste_mask,
                f"            F_arr = F_arr.reshape(-1).at[{index}]"
                ".set(False).reshape(F_arr.shape)\n",
                occurrence=occurrence,
            )

    mutations["taste_shock_simulate:mt10_rank_permutation"] = _insert_before_nth(
        source,
        taste_mask,
        "            Q_flat_attack = Q_arr.reshape(-1)\n"
        "            mt10_order = (\n"
        "                jnp.all(F_arr)\n"
        "                & (Q_flat_attack[0] > Q_flat_attack[2])\n"
        "                & (Q_flat_attack[2] > Q_flat_attack[1])\n"
        "            )\n"
        "            F_arr = jnp.where(\n"
        "                mt10_order,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=2,
    )

    mutations["taste_shock_solve:inline_q_transform"] = _replace_nth(
        source,
        taste_mask,
        "            Q_masked = jnp.where(\n"
        "                F_arr, Q_arr.reshape(-1)[::-1].reshape(Q_arr.shape), -jnp.inf\n"
        "            )",
        occurrence=1,
    )
    mutations["taste_shock_solve:inline_f_transform"] = _replace_nth(
        source,
        taste_mask,
        "            Q_masked = jnp.where(\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                Q_arr,\n"
        "                -jnp.inf,\n"
        "            )",
        occurrence=1,
    )
    mutations["taste_shock_simulate:inline_q_transform"] = _replace_nth(
        source,
        taste_mask,
        "            Q_masked = jnp.where(\n"
        "                F_arr, Q_arr.reshape(-1)[::-1].reshape(Q_arr.shape), -jnp.inf\n"
        "            )",
        occurrence=2,
    )
    mutations["taste_shock_simulate:inline_f_transform"] = _replace_nth(
        source,
        taste_mask,
        "            Q_masked = jnp.where(\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                Q_arr,\n"
        "                -jnp.inf,\n"
        "            )",
        occurrence=2,
    )

    mutations["taste_shock_solve:continuous_axis_prefix"] = source.replace(
        "continuous_axes = tuple(range(n_discrete_action_axes, Q_arr.ndim))",
        "continuous_axes = tuple(range(n_discrete_action_axes, Q_arr.ndim - 1))",
        1,
    )
    mutations["taste_shock_solve:continuous_max_slice"] = source.replace(
        "Qc = Q_masked.max(axis=continuous_axes) if continuous_axes else Q_masked",
        "Qc = (\n"
        "                Q_masked[..., 1:].max(axis=continuous_axes)\n"
        "                if continuous_axes\n"
        "                else Q_masked\n"
        "            )",
        1,
    )
    mutations["taste_shock_solve:logsum_axis_prefix"] = source.replace(
        "axes=tuple(range(Qc.ndim)),",
        "axes=tuple(range(Qc.ndim - 1)),",
        1,
    )
    mutations["taste_shock_solve:logsum_value_slice"] = source.replace(
        "                values=Qc,",
        "                values=Qc.reshape(-1)[1:],",
        1,
    )
    mutations["taste_shock_solve:scale_transform"] = source.replace(
        '"ScalarFloat", states_actions_params[TASTE_SHOCK_SCALE_PARAM]',
        '"ScalarFloat", states_actions_params[TASTE_SHOCK_SCALE_PARAM] * 2',
        1,
    )
    mutations["taste_shock_solve:wrong_return"] = source.replace(
        "            return smoothed",
        "            return Qc.reshape(-1)[0]",
        1,
    )

    mutations["taste_shock_simulate:reshape_drop"] = source.replace(
        "Q_flat = Q_masked.reshape(n_discrete_cells, n_continuous_cells)",
        "Q_flat = Q_masked.reshape(-1)[:-1].reshape(\n"
        "                n_discrete_cells, n_continuous_cells\n"
        "            )",
        1,
    )
    mutations["taste_shock_simulate:wrong_discrete_count"] = source.replace(
        "n_discrete_cells = math.prod(Q_arr.shape[:n_discrete_action_axes])",
        "n_discrete_cells = math.prod(Q_arr.shape[: n_discrete_action_axes - 1])",
        1,
    )
    mutations["taste_shock_simulate:wrong_continuous_count"] = source.replace(
        "n_continuous_cells = math.prod(Q_arr.shape[n_discrete_action_axes:])",
        "n_continuous_cells = math.prod(Q_arr.shape[n_discrete_action_axes + 1 :])",
        1,
    )
    mutations["taste_shock_simulate:continuous_axis_mismatch"] = source.replace(
        "continuous_argmax = jnp.argmax(Q_flat, axis=1)",
        "continuous_argmax = jnp.argmax(Q_flat, axis=0)",
        1,
    )
    mutations["taste_shock_simulate:noise_shape"] = source.replace(
        "key=taste_shock_key, shape=Qc.shape, scale=scale",
        "key=taste_shock_key, shape=(1,), scale=scale",
        1,
    )
    mutations["taste_shock_simulate:discrete_slice"] = source.replace(
        "discrete_argmax = jnp.argmax(noisy_Qc)",
        "discrete_argmax = jnp.argmax(noisy_Qc[1:]) + 1",
        1,
    )
    mutations["taste_shock_simulate:wrong_stride"] = source.replace(
        "discrete_argmax * n_continuous_cells",
        "discrete_argmax * (n_continuous_cells - 1)",
        1,
    )
    mutations["taste_shock_simulate:wrong_continuous_index"] = source.replace(
        "+ continuous_argmax[discrete_argmax]",
        "+ continuous_argmax[0]",
        1,
    )
    mutations["taste_shock_simulate:noisy_value_return"] = source.replace(
        "return flat_index.astype(jnp.int32), Qc[discrete_argmax]",
        "return flat_index.astype(jnp.int32), noisy_Qc[discrete_argmax]",
        1,
    )

    mutations["shared_taste_noise:shared_draw"] = source.replace(
        "return scale * (jax.random.gumbel(key, shape) - EULER_GAMMA)",
        "return scale * ("
        "jnp.broadcast_to(jax.random.gumbel(key, (1,)), shape) - EULER_GAMMA)",
        1,
    )
    mutations["shared_taste_noise:permuted_draw"] = source.replace(
        "return scale * (jax.random.gumbel(key, shape) - EULER_GAMMA)",
        "return scale * (jax.random.gumbel(key, shape).reshape(-1)[::-1]"
        ".reshape(shape) - EULER_GAMMA)",
        1,
    )
    mutations["shared_taste_noise:first_candidate_zeroed"] = source.replace(
        "return scale * (jax.random.gumbel(key, shape) - EULER_GAMMA)",
        "return scale * (jax.random.gumbel(key, shape).reshape(-1).at[0]"
        ".set(0).reshape(shape) - EULER_GAMMA)",
        1,
    )
    mutations["shared_taste_noise:wrong_scale"] = source.replace(
        "return scale * (jax.random.gumbel(key, shape) - EULER_GAMMA)",
        "return scale**2 * (jax.random.gumbel(key, shape) - EULER_GAMMA)",
        1,
    )
    mutations["shared_taste_noise:cast_import_replaced"] = source.replace(
        "from typing import cast",
        "from candidate_filter import cast",
        1,
    )
    mutations["taste_shock_solve:captured_axis_rebinding"] = _insert_before_nth(
        source,
        "    if has_taste_shocks:",
        "    n_discrete_action_axes = n_discrete_action_axes - 1\n",
        occurrence=1,
    )
    mutations["taste_shock_simulate:captured_axis_rebinding"] = _insert_before_nth(
        source,
        "    if has_taste_shocks:",
        "    n_discrete_action_axes = n_discrete_action_axes - 1\n",
        occurrence=2,
    )
    mutations["solve:action_names_rebinding"] = _insert_before_nth(
        source,
        "    Q_and_F = productmap(",
        "    action_names = action_names[:-1]\n",
        occurrence=1,
    )
    mutations["simulate:action_names_rebinding"] = _insert_before_nth(
        source,
        "    Q_and_F = productmap(",
        "    action_names = action_names[:-1]\n",
        occurrence=2,
    )
    mutations["solve:dormant_certified_reducer"] = source.replace(
        "        func=max_Q_over_a,\n        variables=inner_state_names,",
        "        func=Q_and_F,\n        variables=inner_state_names,",
        1,
    )
    mutations["simulate:return_bypasses_certified_reducer"] = source.replace(
        "    return argmax_and_max_Q_over_a",
        "    return Q_and_F",
        1,
    )
    mutations["shared_max:productmap_module_shadow"] = source.replace(
        "from _lcm.utils.dispatchers import productmap, vmap_1d",
        "from _lcm.utils.dispatchers import productmap, vmap_1d\n"
        "productmap = candidate_filter",
        1,
    )
    mutations["solve:builder_decorator_wrapper"] = source.replace(
        "def get_max_Q_over_a(", "@candidate_filter\ndef get_max_Q_over_a(", 1
    )
    mutations["simulate:builder_decorator_wrapper"] = source.replace(
        "def get_argmax_and_max_Q_over_a(",
        "@candidate_filter\ndef get_argmax_and_max_Q_over_a(",
        1,
    )
    mutations["solve:builder_default_changed"] = _replace_nth(
        source,
        "    n_discrete_action_axes: int = 0,",
        "    n_discrete_action_axes: int = 1,",
        occurrence=1,
    )
    mutations["simulate:builder_default_changed"] = _replace_nth(
        source,
        "    n_discrete_action_axes: int = 0,",
        "    n_discrete_action_axes: int = 1,",
        occurrence=2,
    )
    mutations["taste_shock_solve:attribute_with_signature"] = _replace_nth(
        source,
        "        @with_signature(",
        "        @candidate_filter.with_signature(",
        occurrence=1,
    )
    mutations["taste_shock_simulate:attribute_with_signature"] = _replace_nth(
        source,
        "        @with_signature(",
        "        @candidate_filter.with_signature(",
        occurrence=4,
    )
    mutations["taste_shock_solve:attribute_q_and_f"] = _replace_nth(
        source,
        "            Q_arr, F_arr = Q_and_F(",
        "            Q_arr, F_arr = candidate_filter.Q_and_F(",
        occurrence=1,
    )
    mutations["taste_shock_simulate:attribute_q_and_f"] = _replace_nth(
        source,
        "            Q_arr, F_arr = Q_and_F(",
        "            Q_arr, F_arr = candidate_filter.Q_and_F(",
        occurrence=3,
    )
    mutations["singleton_simulate:attribute_argmax_and_max"] = source.replace(
        "return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)",
        "return candidate_filter.argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)",
        1,
    )
    mutations["collective_solve:attribute_collective_readout"] = _replace_nth(
        source,
        "collective_readout(",
        "candidate_filter.collective_readout(",
        occurrence=1,
    )
    mutations["collective_simulate:attribute_collective_argmax"] = _replace_nth(
        source,
        "collective_argmax_and_readout(",
        "candidate_filter.collective_argmax_and_readout(",
        occurrence=1,
    )
    mutations["solve:attribute_productmap"] = _replace_nth(
        source,
        "    Q_and_F = productmap(",
        "    Q_and_F = candidate_filter.productmap(",
        occurrence=1,
    )
    mutations["simulate:attribute_productmap"] = _replace_nth(
        source,
        "    Q_and_F = productmap(",
        "    Q_and_F = candidate_filter.productmap(",
        occurrence=2,
    )
    return mutations


def direct_flow_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Return semantic mutations across every certified corridor source."""
    root = repo_root.resolve()
    max_source = (root / MAX_Q_SOURCE).read_text(encoding="utf-8")
    argmax_source = (root / ARGMAX_SOURCE).read_text(encoding="utf-8")
    collective_source = (root / COLLECTIVE_SOURCE).read_text(encoding="utf-8")
    logsum_source = (root / LOGSUM_SOURCE).read_text(encoding="utf-8")
    grid_source = (root / GRID_SEARCH_SOURCE).read_text(encoding="utf-8")
    processing_source = (root / PROCESSING_SOURCE).read_text(encoding="utf-8")
    dispatchers_source = (root / DISPATCHERS_SOURCE).read_text(encoding="utf-8")
    functools_source = (root / FUNCTOOLS_SOURCE).read_text(encoding="utf-8")
    containers_source = (root / CONTAINERS_SOURCE).read_text(encoding="utf-8")
    zero_safe_source = (root / ZERO_SAFE_SOURCE).read_text(encoding="utf-8")
    probability_source = (root / PROBABILITY_SOURCE).read_text(encoding="utf-8")
    engine_source = (root / ENGINE_SOURCE).read_text(encoding="utf-8")
    state_action_space_source = (root / STATE_ACTION_SPACE_SOURCE).read_text(
        encoding="utf-8"
    )
    simulation_source = (root / SIMULATION_SOURCE).read_text(encoding="utf-8")
    simulation_transitions_source = (root / SIMULATION_TRANSITIONS_SOURCE).read_text(
        encoding="utf-8"
    )
    simulation_compile_source = (root / SIMULATION_COMPILE_SOURCE).read_text(
        encoding="utf-8"
    )
    model_source = (root / MODEL_SOURCE).read_text(encoding="utf-8")
    backward_induction_source = (root / BACKWARD_INDUCTION_SOURCE).read_text(
        encoding="utf-8"
    )
    initial_conditions_source = (root / INITIAL_CONDITIONS_SOURCE).read_text(
        encoding="utf-8"
    )
    result_source = (root / RESULT_SOURCE).read_text(encoding="utf-8")
    result_dataframe_source = (root / RESULT_DATAFRAME_SOURCE).read_text(
        encoding="utf-8"
    )
    result_metadata_source = (root / RESULT_METADATA_SOURCE).read_text(encoding="utf-8")
    additional_targets_source = (root / ADDITIONAL_TARGETS_SOURCE).read_text(
        encoding="utf-8"
    )
    simulation_random_source = (root / SIMULATION_RANDOM_SOURCE).read_text(
        encoding="utf-8"
    )
    fold_zero_safe_source = (root / FOLD_ZERO_SAFE_SOURCE).read_text(encoding="utf-8")
    solution_contract_source = (root / SOLUTION_CONTRACT_SOURCE).read_text(
        encoding="utf-8"
    )
    grids_init_source = (root / GRIDS_INIT_SOURCE).read_text(encoding="utf-8")
    grid_base_source = (root / GRID_BASE_SOURCE).read_text(encoding="utf-8")
    grid_coordinates_source = (root / GRID_COORDINATES_SOURCE).read_text(
        encoding="utf-8"
    )
    discrete_grid_source = (root / DISCRETE_GRID_SOURCE).read_text(encoding="utf-8")
    continuous_grid_source = (root / CONTINUOUS_GRID_SOURCE).read_text(encoding="utf-8")
    piecewise_grid_source = (root / PIECEWISE_GRID_SOURCE).read_text(encoding="utf-8")
    processes_init_source = (root / PROCESSES_INIT_SOURCE).read_text(encoding="utf-8")
    process_base_source = (root / PROCESS_BASE_SOURCE).read_text(encoding="utf-8")
    process_iid_source = (root / PROCESS_IID_SOURCE).read_text(encoding="utf-8")
    process_ar1_source = (root / PROCESS_AR1_SOURCE).read_text(encoding="utf-8")
    variables_source = (root / VARIABLES_SOURCE).read_text(encoding="utf-8")
    params_regime_template_source = (root / PARAMS_REGIME_TEMPLATE_SOURCE).read_text(
        encoding="utf-8"
    )
    params_processing_source = (root / PARAMS_PROCESSING_SOURCE).read_text(
        encoding="utf-8"
    )
    dtypes_source = (root / DTYPES_SOURCE).read_text(encoding="utf-8")
    namespace_source = (root / NAMESPACE_SOURCE).read_text(encoding="utf-8")
    pandas_utils_source = (root / PANDAS_UTILS_SOURCE).read_text(encoding="utf-8")
    model_processing_source = (root / MODEL_PROCESSING_SOURCE).read_text(
        encoding="utf-8"
    )
    specs: dict[str, dict[str, str]] = {
        name: {"path": MAX_Q_SOURCE, "source": mutated}
        for name, mutated in direct_flow_mutations(max_source).items()
    }

    def replace_once(source: str, old: str, new: str, *, label: str) -> str:
        if source.count(old) != 1:
            raise ValueError(
                f"{label}: expected one mutation marker, found {source.count(old)}"
            )
        return source.replace(old, new, 1)

    grid_cases = {
        "caller_solve:action_names_slice": replace_once(
            grid_source,
            "                    action_names=context.state_action_space.action_names,",
            "                    action_names=context.state_action_space.action_names[:-1],",
            label="solve caller action names",
        ),
        "caller_solve:wrong_discrete_axis_count": replace_once(
            grid_source,
            "                    n_discrete_action_axes=len(\n"
            "                        context.state_action_space.discrete_actions\n"
            "                    ),",
            "                    n_discrete_action_axes=max(\n"
            "                        0, len(context.state_action_space.discrete_actions) - 1\n"
            "                    ),",
            label="solve caller axis count",
        ),
        "caller_solve:taste_flag_disabled": replace_once(
            grid_source,
            "                    has_taste_shocks=context.has_taste_shocks,",
            "                    has_taste_shocks=False,",
            label="solve caller taste flag",
        ),
        "caller_solve:published_empty_mapping": replace_once(
            grid_source,
            "        return SolutionKernels(period_kernels=MappingProxyType(result))",
            "        return SolutionKernels(period_kernels=MappingProxyType({}))",
            label="solve caller publication",
        ),
    }
    specs.update(
        {
            name: {"path": GRID_SEARCH_SOURCE, "source": mutated}
            for name, mutated in grid_cases.items()
        }
    )

    processing_cases = {
        "caller_simulate:action_names_slice": replace_once(
            processing_source,
            "                action_names=state_action_space.action_names,",
            "                action_names=state_action_space.action_names[:-1],",
            label="simulate caller action names",
        ),
        "caller_simulate:wrong_discrete_axis_count": replace_once(
            processing_source,
            "                n_discrete_action_axes=len(state_action_space.discrete_actions),",
            "                n_discrete_action_axes=max(\n"
            "                    0, len(state_action_space.discrete_actions) - 1\n"
            "                ),",
            label="simulate caller axis count",
        ),
        "caller_simulate:taste_flag_disabled": replace_once(
            processing_source,
            "                n_discrete_action_axes=len(state_action_space.discrete_actions),\n"
            "                has_taste_shocks=has_taste_shocks,",
            "                n_discrete_action_axes=len(state_action_space.discrete_actions),\n"
            "                has_taste_shocks=False,",
            label="simulate caller taste flag",
        ),
        "caller_simulate:live_taste_flag_rebinding": replace_once(
            processing_source,
            "    argmax_and_max_Q_over_a = _build_argmax_and_max_Q_over_a_per_period(",
            "    has_taste_shocks = False\n\n"
            "    argmax_and_max_Q_over_a = _build_argmax_and_max_Q_over_a_per_period(",
            label="simulate live taste rebinding",
        ),
        "caller_simulate:published_empty_mapping": replace_once(
            processing_source,
            "        argmax_and_max_Q_over_a=argmax_and_max_Q_over_a,",
            "        argmax_and_max_Q_over_a=MappingProxyType({}),",
            label="simulate caller publication",
        ),
        "caller_simulate:attribute_simulation_phase": replace_once(
            processing_source,
            "    return SimulationPhase(",
            "    return candidate_filter.SimulationPhase(",
            label="simulate caller publication callee",
        ),
    }
    specs.update(
        {
            name: {"path": PROCESSING_SOURCE, "source": mutated}
            for name, mutated in processing_cases.items()
        }
    )

    argmax_cases = {
        "shared_argmax:q_order_early_return": replace_once(
            argmax_source,
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            "    if a.reshape(-1)[0] > a.reshape(-1)[1]:\n"
            "        return jnp.array(1, dtype=jnp.int32), a.reshape(-1)[1]\n"
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            label="argmax q-order",
        ),
        "shared_argmax:support_filter": replace_once(
            argmax_source,
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            "    where = jnp.where(\n"
            "        jnp.sum(where) > 1,\n"
            "        where.reshape(-1).at[0].set(False).reshape(where.shape),\n"
            "        where,\n"
            "    )\n"
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            label="argmax support",
        ),
        "shared_argmax:axis_prefix": replace_once(
            argmax_source,
            "        axis = tuple(range(a.ndim))",
            "        axis = tuple(range(a.ndim - 1))",
            label="argmax axis prefix",
        ),
        "shared_argmax:axis_reorder": replace_once(
            argmax_source,
            "    return a.transpose((*front_axes, *axes))",
            "    return a.transpose((*front_axes, *reversed(axes)))",
            label="argmax axis reorder",
        ),
        "shared_argmax:flatten_drop_last": replace_once(
            argmax_source,
            "    return a.reshape(*a.shape[:-n], -1)",
            "    return a[..., :-1].reshape(*a.shape[:-n], -1)",
            label="argmax flatten drop",
        ),
        "shared_argmax:range_module_shadow": replace_once(
            argmax_source,
            "from lcm.typing import BoolND, FloatND, IntND\n",
            "from lcm.typing import BoolND, FloatND, IntND\n\n"
            "range = candidate_filter\n",
            label="argmax range shadow",
        ),
    }
    specs.update(
        {
            name: {"path": ARGMAX_SOURCE, "source": mutated}
            for name, mutated in argmax_cases.items()
        }
    )

    collective_cases = {
        "shared_collective:q_gap_filter": replace_once(
            collective_source,
            "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)",
            "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)\n"
            "    objective = jnp.where(\n"
            "        objective.reshape(-1)[0] - objective.reshape(-1)[1] > 0.5,\n"
            "        objective.reshape(-1).at[0].set(-jnp.inf).reshape(\n"
            "            objective.shape\n"
            "        ),\n"
            "        objective,\n"
            "    )",
            label="collective q-gap",
        ),
        "shared_collective:feasibility_inline_filter": replace_once(
            collective_source,
            "        objective, axis=action_axes, initial=-jnp.inf, where=feasibility",
            "        objective, axis=action_axes, initial=-jnp.inf,\n"
            "        where=feasibility.reshape(-1).at[0].set(False).reshape(\n"
            "            feasibility.shape\n"
            "        )",
            label="collective feasibility inline",
        ),
        "shared_collective:action_axis_prefix": replace_once(
            collective_source,
            "        objective, axis=action_axes, initial=-jnp.inf, where=feasibility",
            "        objective, axis=action_axes[:-1], initial=-jnp.inf, where=feasibility",
            label="collective action axis",
        ),
        "shared_collective:gather_next_candidate": replace_once(
            collective_source,
            "    gathered = jnp.take_along_axis(q_flat, argmax_flat[..., None], axis=-1)",
            "    gathered = jnp.take_along_axis(q_flat, (argmax_flat + 1)[..., None], axis=-1)",
            label="collective gather",
        ),
        "shared_collective:early_candidate_return": replace_once(
            collective_source,
            "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)",
            "    if jnp.all(feasibility):\n"
            "        return (\n"
            "            jnp.array(1, dtype=jnp.int32),\n"
            "            {name: q.reshape(-1)[1] for name, q in stakeholder_Q.items()},\n"
            "            jnp.array(False),\n"
            "        )\n"
            "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)",
            label="collective early return",
        ),
        "shared_collective:argmax_module_shadow": replace_once(
            collective_source,
            "    argmax_and_max,\n)\n",
            "    argmax_and_max,\n)\n\nargmax_and_max = candidate_filter\n",
            label="collective argmax shadow",
        ),
    }
    specs.update(
        {
            name: {"path": COLLECTIVE_SOURCE, "source": mutated}
            for name, mutated in collective_cases.items()
        }
    )

    logsum_cases = {
        "shared_logsum:q_gap_filter": replace_once(
            logsum_source,
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            "    gap_filter = values.reshape(-1)[0] - values.reshape(-1)[1] > 0.5\n"
            "    values = jnp.where(\n"
            "        gap_filter,\n"
            "        values.reshape(-1).at[0].set(-jnp.inf).reshape(values.shape),\n"
            "        values,\n"
            "    )\n"
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            label="logsum q-gap",
        ),
        "shared_logsum:support_filter": replace_once(
            logsum_source,
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            "    support_filter = jnp.sum(~jnp.isneginf(values)) > 1\n"
            "    values = jnp.where(\n"
            "        support_filter,\n"
            "        values.reshape(-1).at[0].set(-jnp.inf).reshape(values.shape),\n"
            "        values,\n"
            "    )\n"
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            label="logsum support",
        ),
        "shared_logsum:axis_prefix": replace_once(
            logsum_source,
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            "    v_max = jnp.max(values, axis=axes[:-1], keepdims=True)",
            label="logsum axis prefix",
        ),
        "shared_logsum:value_slice": replace_once(
            logsum_source,
            "        shifted, axis=axes",
            "        shifted[..., 1:], axis=axes",
            label="logsum value slice",
        ),
        "shared_logsum:rank_early_return": replace_once(
            logsum_source,
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            "    if values.reshape(-1)[0] > values.reshape(-1)[1]:\n"
            "        return values.reshape(-1)[1], jnp.zeros_like(values)\n"
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            label="logsum early return",
        ),
        "shared_logsum:softmax_slice": replace_once(
            logsum_source,
            "jax.nn.softmax(shifted, axis=axes)",
            "jax.nn.softmax(shifted[..., 1:], axis=axes)",
            label="logsum softmax slice",
        ),
        "shared_logsum:import_rebinding": replace_once(
            logsum_source,
            "from jax.scipy.special import logsumexp",
            "from jax.scipy.special import logsumexp\n\nlogsumexp = jnp.max",
            label="logsum import rebinding",
        ),
        "shared_logsum:wrong_euler_gamma": replace_once(
            logsum_source,
            "EULER_GAMMA = 0.5772156649015329",
            "EULER_GAMMA = 0.0",
            label="logsum Euler-Gamma",
        ),
    }
    specs.update(
        {
            name: {"path": LOGSUM_SOURCE, "source": mutated}
            for name, mutated in logsum_cases.items()
        }
    )

    dependency_cases = {
        "shared_productmap:drop_last_axis": {
            "path": DISPATCHERS_SOURCE,
            "source": replace_once(
                dispatchers_source,
                "        product_axes=variables,",
                "        product_axes=variables[:-1],",
                label="productmap action-axis drop",
            ),
        },
        "shared_functools:drop_last_argument": {
            "path": FUNCTOOLS_SOURCE,
            "source": replace_once(
                functools_source,
                "        kwargs_names = list(parameters)[n_positional_only_parameters:]",
                "        kwargs_names = list(parameters)[n_positional_only_parameters:-1]",
                label="allow-args argument drop",
            ),
        },
        "shared_containers:duplicate_threshold": {
            "path": CONTAINERS_SOURCE,
            "source": replace_once(
                containers_source,
                "return {v for v, count in counts.items() if count > 1}",
                "return {v for v, count in counts.items() if count > 2}",
                label="duplicate threshold",
            ),
        },
        "shared_zero_safe:ordered_sum_slice": {
            "path": ZERO_SAFE_SOURCE,
            "source": replace_once(
                zero_safe_source,
                "return jnp.sum(jnp.sort(arr, axis=axis), axis=axis)",
                "return jnp.sum(jnp.sort(arr, axis=axis)[1:], axis=axis)",
                label="ordered scalarization slice",
            ),
        },
        "shared_probability:unbalanced_product": {
            "path": PROBABILITY_SOURCE,
            "source": replace_once(
                probability_source,
                "return _balanced_with_tangent(jnp.asarray(weight), jnp.asarray(value))",
                "return jnp.asarray(weight) * jnp.asarray(value)",
                label="zero-safe balanced product",
            ),
        },
        "candidate_materialization:grid_base_intercepts_to_jax": {
            "path": GRID_BASE_SOURCE,
            "source": replace_once(
                grid_base_source,
                'class Grid(ABC):\n    """LCM Grid base class."""',
                'class Grid(ABC):\n    """LCM Grid base class."""\n\n    def __getattribute__(self, name):\n        value = super().__getattribute__(name)\n        if name == "to_jax":\n            return lambda: value()[:-1]\n        return value',
                label="inherited grid coordinate interception",
            ),
        },
        "simulation_state_action_space:drops_inherited_candidates": {
            "path": SIMULATION_TRANSITIONS_SOURCE,
            "source": replace_once(
                simulation_transitions_source,
                "    return base.replace(states=MappingProxyType(states_for_state_action_space))",
                "    return base.replace(\n        states=MappingProxyType(states_for_state_action_space),\n        discrete_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base.discrete_actions.items()}),\n        continuous_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base.continuous_actions.items()}),\n    )",
                label="simulation base-action preservation",
            ),
        },
        "simulation_state_action_space:caller_drops_inherited_candidates": {
            "path": SIMULATION_SOURCE,
            "source": replace_once(
                simulation_source,
                "        base=base_state_action_space,",
                "        base=base_state_action_space.replace(continuous_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base_state_action_space.continuous_actions.items()})),",
                label="simulation adapter caller base wrapping",
            ),
        },
        "shared_engine:action_order_reversed": {
            "path": ENGINE_SOURCE,
            "source": replace_once(
                engine_source,
                "return tuple(self.discrete_actions) + tuple(self.continuous_actions)",
                "return tuple(self.continuous_actions) + tuple(self.discrete_actions)",
                label="state-action metadata order",
            ),
        },
        "shared_engine:actions_drop_last_candidate": {
            "path": ENGINE_SOURCE,
            "source": replace_once(
                engine_source,
                "            dict(self.discrete_actions) | dict(self.continuous_actions)",
                "            {name: values.at[-1].set(values[0]) for name, values in self.discrete_actions.items()} | {name: values.at[-1].set(values[0]) for name, values in self.continuous_actions.items()}",
                label="combined action mapping candidate omission",
            ),
        },
        "shared_engine:replace_drops_inherited_candidates": {
            "path": ENGINE_SOURCE,
            "source": replace_once(
                engine_source,
                "        discrete_actions = first_non_none(discrete_actions, self.discrete_actions)",
                "        discrete_actions = first_non_none(discrete_actions, MappingProxyType({name: values.at[-1].set(values[0]) for name, values in self.discrete_actions.items()}))",
                label="StateActionSpace.replace inherited candidate omission",
            ),
        },
        "shared_state_action_space:continuous_order_reversed": {
            "path": STATE_ACTION_SPACE_SOURCE,
            "source": replace_once(
                state_action_space_source,
                "        for name in variables.continuous_action_names",
                "        for name in reversed(variables.continuous_action_names)",
                label="continuous candidate order",
            ),
        },
        "shared_state_action_space:drop_last_continuous_candidate": {
            "path": STATE_ACTION_SPACE_SOURCE,
            "source": replace_once(
                state_action_space_source,
                "        name: _grid_to_jax_or_placeholder(grids[name])\n        for name in variables.continuous_action_names",
                "        name: _grid_to_jax_or_placeholder(grids[name]).at[-1].set(_grid_to_jax_or_placeholder(grids[name])[0])\n        for name in variables.continuous_action_names",
                label="state-action continuous candidate omission",
            ),
        },
        "simulation_index_consumer:next_candidate": {
            "path": SIMULATION_SOURCE,
            "source": replace_once(
                simulation_source,
                "            flat_indices=indices_optimal_actions,",
                "            flat_indices=indices_optimal_actions + 1,",
                label="published simulation index consumer",
            ),
        },
        "aot_compile:argmax_index_shift": {
            "path": SIMULATION_COMPILE_SOURCE,
            "source": replace_once(
                simulation_compile_source,
                "            argmax_func = sf.argmax_and_max_Q_over_a[period]",
                "            argmax_func = sf.argmax_and_max_Q_over_a[period]\n"
                "            original_argmax_func = argmax_func\n"
                "            def argmax_func(**kwargs):\n"
                "                index, value = original_argmax_func(**kwargs)\n"
                "                return index + 1, value",
                label="AOT argmax index shift",
            ),
        },
        "aot_model:compiled_regime_filter": {
            "path": MODEL_SOURCE,
            "source": replace_once(
                model_source,
                "            return self._simulate_compile_cache[compile_batch_size]",
                "            return candidate_filter(\n"
                "                self._simulate_compile_cache[compile_batch_size]\n"
                "            )",
                label="public Model AOT regime selection",
            ),
        },
        "shared_dedup_key:collapse_plain_callables": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                backward_induction_source,
                "    return id(func)",
                "    return 0",
                label="plain-callable dedup identity",
            ),
        },
        "simulation_publication:shift_padded_actions": {
            "path": INITIAL_CONDITIONS_SOURCE,
            "source": replace_once(
                initial_conditions_source,
                "                        {k: v[:original_n_subjects] for k, v in value.items()}",
                "                        {k: (v[1 : original_n_subjects + 1] if name == 'actions' else v[:original_n_subjects]) for k, v in value.items()}",
                label="padded action row shift",
            ),
        },
        "simulation_result:shift_raw_actions": {
            "path": RESULT_SOURCE,
            "source": replace_once(
                result_source,
                "        self._raw_results = raw_results",
                "        self._raw_results = MappingProxyType({regime: MappingProxyType({period: __import__('dataclasses').replace(data, actions=MappingProxyType({name: jnp.roll(values, 1) for name, values in data.actions.items()})) for period, data in periods.items()}) for regime, periods in raw_results.items()})",
                label="SimulationResult raw action shift",
            ),
        },
        "simulation_dataframe:shift_action_column": {
            "path": RESULT_DATAFRAME_SOURCE,
            "source": replace_once(
                result_dataframe_source,
                "            data[name] = result.actions[name]",
                "            data[name] = jnp.roll(result.actions[name], 1)",
                label="DataFrame action-column shift",
            ),
        },
        "simulation_metadata:drop_regime_actions": {
            "path": RESULT_METADATA_SOURCE,
            "source": replace_once(
                result_metadata_source,
                "        regime_to_actions[regime_name] = regime.simulation.action_names",
                "        regime_to_actions[regime_name] = ()",
                label="result metadata action omission",
            ),
        },
        "additional_targets:overwrite_actions_single_pass": {
            "path": ADDITIONAL_TARGETS_SOURCE,
            "source": replace_once(
                additional_targets_source,
                "        return {k: _one_value_per_row(v, n_rows=n_rows) for k, v in result.items()}",
                "        return {**{k: _one_value_per_row(v, n_rows=n_rows) for k, v in result.items()}, **{name: jnp.roll(jnp.asarray(data[name]), 1) for name in regime.simulation.action_names if name in data}}",
                label="single-pass additional-target action overwrite",
            ),
        },
        "additional_targets:overwrite_actions_chunked": {
            "path": ADDITIONAL_TARGETS_SOURCE,
            "source": replace_once(
                additional_targets_source,
                "    return {\n        name: np.concatenate([out[name] for out in chunk_outputs])\n        for name in chunk_outputs[0]\n    }",
                "    return {\n        **{name: np.concatenate([out[name] for out in chunk_outputs]) for name in chunk_outputs[0]},\n        **{name: jnp.roll(jnp.asarray(data[name]), 1) for name in regime.simulation.action_names if name in data},\n    }",
                label="chunked additional-target action overwrite",
            ),
        },
        "simulation_random:reassign_taste_keys": {
            "path": SIMULATION_RANDOM_SOURCE,
            "source": replace_once(
                simulation_random_source,
                '        simulation_keys[f"key_{name}"] = per_subject_keys',
                '        simulation_keys[f"key_{name}"] = jnp.roll(per_subject_keys, 1, axis=0)',
                label="subject taste-key reassignment",
            ),
        },
        "shared_fold_average:negated_value": {
            "path": FOLD_ZERO_SAFE_SOURCE,
            "source": replace_once(
                fold_zero_safe_source,
                "    return numerator / total_weight",
                "    return -numerator / total_weight",
                label="folded zero-safe average negation",
            ),
        },
        "solution_contract:negate_kernel_result": {
            "path": SOLUTION_CONTRACT_SOURCE,
            "source": replace_once(
                solution_contract_source,
                "    diagnostics: SolverDiagnostics | None = None",
                '    diagnostics: SolverDiagnostics | None = None\n\n    def __post_init__(self) -> None:\n        object.__setattr__(self, "V_arr", -self.V_arr)',
                label="KernelResult value transport negation",
            ),
        },
        "candidate_materialization:rebind_continuous_grid": {
            "path": GRIDS_INIT_SOURCE,
            "source": replace_once(
                grids_init_source,
                "from _lcm.grids.discrete import DiscreteGrid",
                "from _lcm.grids.discrete import DiscreteGrid\n\nContinuousGrid = DiscreteGrid",
                label="continuous-grid classification rebinding",
            ),
        },
        "candidate_materialization:rebind_process_class": {
            "path": PROCESSES_INIT_SOURCE,
            "source": replace_once(
                processes_init_source,
                "from _lcm.processes.iid import _IIDProcess",
                "from _lcm.processes.iid import _IIDProcess\n\n_ContinuousStochasticProcess = _AR1Process",
                label="process-action classification rebinding",
            ),
        },
        "candidate_materialization:drop_last_discrete_code": {
            "path": DISCRETE_GRID_SOURCE,
            "source": replace_once(
                discrete_grid_source,
                "        return jnp.array(self.codes, dtype=jnp.int32)",
                "        return jnp.array(self.codes[:-1], dtype=jnp.int32)",
                label="discrete action code omission",
            ),
        },
        "candidate_materialization:drop_last_linear_point": {
            "path": CONTINUOUS_GRID_SOURCE,
            "source": replace_once(
                continuous_grid_source,
                "        return grid_coordinates.linspace(\n            start=self.start, stop=self.stop, n_points=self.n_points\n        )",
                "        return grid_coordinates.linspace(\n            start=self.start, stop=self.stop, n_points=self.n_points\n        )[:-1]",
                label="linear action point omission",
            ),
        },
        "candidate_materialization:drop_last_coordinate_point": {
            "path": GRID_COORDINATES_SOURCE,
            "source": replace_once(
                grid_coordinates_source,
                "    return jnp.linspace(start, stop, n_points)  # ty: ignore[no-matching-overload]",
                "    return jnp.linspace(start, stop, n_points)[:-1]  # ty: ignore[no-matching-overload]",
                label="shared linear coordinate omission",
            ),
        },
        "candidate_materialization:drop_last_piecewise_point": {
            "path": PIECEWISE_GRID_SOURCE,
            "source": replace_once(
                piecewise_grid_source,
                "        return jnp.concatenate(segments)",
                "        return jnp.concatenate(segments)[:-1]",
                label="piecewise action point omission",
            ),
        },
        "candidate_materialization:drop_last_process_node": {
            "path": PROCESS_BASE_SOURCE,
            "source": replace_once(
                process_base_source,
                "        return self.compute_gridpoints(**self.params)",
                "        return self.compute_gridpoints(**self.params)[:-1]",
                label="process action node omission",
            ),
        },
        "candidate_materialization:drop_last_iid_node": {
            "path": PROCESS_IID_SOURCE,
            "source": replace_once(
                process_iid_source,
                '        return jnp.linspace(\n            start=kwargs["start"], stop=kwargs["stop"], num=self.n_points\n        )',
                '        return jnp.linspace(\n            start=kwargs["start"], stop=kwargs["stop"], num=self.n_points\n        )[:-1]',
                label="IID action node omission",
            ),
        },
        "candidate_materialization:drop_last_ar1_node": {
            "path": PROCESS_AR1_SOURCE,
            "source": replace_once(
                process_ar1_source,
                "        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)",
                "        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)[:-1]",
                label="AR1 action node omission",
            ),
        },
        "candidate_materialization:drop_last_action_name": {
            "path": VARIABLES_SOURCE,
            "source": replace_once(
                variables_source,
                '    actions = [name for name, var_info in info.items() if var_info.kind == "action"]',
                '    actions = [name for name, var_info in info.items() if var_info.kind == "action"][:-1]',
                label="finalized action-name omission",
            ),
        },
        "candidate_materialization:skip_first_runtime_action_template": {
            "path": PARAMS_REGIME_TEMPLATE_SOURCE,
            "source": replace_once(
                params_regime_template_source,
                "    for action_name, grid in user_regime.actions.items():",
                "    for action_name, grid in tuple(user_regime.actions.items())[1:]:",
                label="runtime action template omission",
            ),
        },
        "candidate_materialization:negate_broadcast_runtime_points": {
            "path": PARAMS_PROCESSING_SOURCE,
            "source": replace_once(
                params_processing_source,
                "            result[regime][remainder] = params_flat[chosen]",
                '            result[regime][remainder] = (-params_flat[chosen] if remainder.endswith("__points") else params_flat[chosen])',
                label="runtime action points changed during broadcast",
            ),
        },
        "candidate_materialization:negate_flattened_runtime_points": {
            "path": NAMESPACE_SOURCE,
            "source": replace_once(
                namespace_source,
                "    return MappingProxyType(flatten_to_qnames(d))",
                '    flat = flatten_to_qnames(d)\n    return MappingProxyType({key: -value if key.endswith("__points") and hasattr(value, "dtype") else value for key, value in flat.items()})',
                label="runtime action points changed during namespace flattening",
            ),
        },
        "candidate_materialization:negate_cast_runtime_points": {
            "path": DTYPES_SOURCE,
            "source": replace_once(
                dtypes_source,
                "    return jnp.asarray(np_value, dtype=target_dtype)",
                '    out = jnp.asarray(np_value, dtype=target_dtype)\n    return -out if name.endswith("__points") else out',
                label="runtime action points changed during canonical cast",
            ),
        },
        "candidate_materialization:negate_series_runtime_points": {
            "path": PANDAS_UTILS_SOURCE,
            "source": replace_once(
                pandas_utils_source,
                "    if func is None:\n        return jnp.array(sr.to_numpy(), dtype=canonical_float_dtype())",
                "    if func is None:\n        return -jnp.array(sr.to_numpy(), dtype=canonical_float_dtype())",
                label="Series runtime action points changed during conversion",
            ),
        },
        "candidate_materialization:negate_fixed_runtime_points": {
            "path": MODEL_PROCESSING_SOURCE,
            "source": replace_once(
                model_processing_source,
                "        regime_fixed = dict(fixed_flat_params.get(regime_name, MappingProxyType({})))",
                '        regime_fixed = dict(fixed_flat_params.get(regime_name, MappingProxyType({})))\n        regime_fixed = {key: -value if key.endswith("__points") else value for key, value in regime_fixed.items()}',
                label="fixed runtime action points changed before state-space completion",
            ),
        },
    }
    specs.update(dependency_cases)

    originals = {
        MAX_Q_SOURCE: max_source,
        ARGMAX_SOURCE: argmax_source,
        COLLECTIVE_SOURCE: collective_source,
        LOGSUM_SOURCE: logsum_source,
        GRID_SEARCH_SOURCE: grid_source,
        PROCESSING_SOURCE: processing_source,
        DISPATCHERS_SOURCE: dispatchers_source,
        FUNCTOOLS_SOURCE: functools_source,
        CONTAINERS_SOURCE: containers_source,
        ZERO_SAFE_SOURCE: zero_safe_source,
        PROBABILITY_SOURCE: probability_source,
        ENGINE_SOURCE: engine_source,
        STATE_ACTION_SPACE_SOURCE: state_action_space_source,
        SIMULATION_SOURCE: simulation_source,
        SIMULATION_TRANSITIONS_SOURCE: simulation_transitions_source,
        SIMULATION_COMPILE_SOURCE: simulation_compile_source,
        MODEL_SOURCE: model_source,
        BACKWARD_INDUCTION_SOURCE: backward_induction_source,
        INITIAL_CONDITIONS_SOURCE: initial_conditions_source,
        RESULT_SOURCE: result_source,
        RESULT_DATAFRAME_SOURCE: result_dataframe_source,
        RESULT_METADATA_SOURCE: result_metadata_source,
        ADDITIONAL_TARGETS_SOURCE: additional_targets_source,
        SIMULATION_RANDOM_SOURCE: simulation_random_source,
        FOLD_ZERO_SAFE_SOURCE: fold_zero_safe_source,
        SOLUTION_CONTRACT_SOURCE: solution_contract_source,
        GRIDS_INIT_SOURCE: grids_init_source,
        GRID_BASE_SOURCE: grid_base_source,
        GRID_COORDINATES_SOURCE: grid_coordinates_source,
        DISCRETE_GRID_SOURCE: discrete_grid_source,
        CONTINUOUS_GRID_SOURCE: continuous_grid_source,
        PIECEWISE_GRID_SOURCE: piecewise_grid_source,
        PROCESSES_INIT_SOURCE: processes_init_source,
        PROCESS_BASE_SOURCE: process_base_source,
        PROCESS_IID_SOURCE: process_iid_source,
        PROCESS_AR1_SOURCE: process_ar1_source,
        VARIABLES_SOURCE: variables_source,
        PARAMS_REGIME_TEMPLATE_SOURCE: params_regime_template_source,
        PARAMS_PROCESSING_SOURCE: params_processing_source,
        DTYPES_SOURCE: dtypes_source,
        NAMESPACE_SOURCE: namespace_source,
        PANDAS_UTILS_SOURCE: pandas_utils_source,
        MODEL_PROCESSING_SOURCE: model_processing_source,
    }
    mutated_paths = {spec["path"] for spec in specs.values()}
    certified_paths = set(_CERTIFIED_CORRIDOR_SOURCES)
    if mutated_paths != certified_paths:
        raise ValueError(
            "mutation-source coverage differs from the certified corridor: "
            f"missing={sorted(certified_paths - mutated_paths)}, "
            f"extra={sorted(mutated_paths - certified_paths)}"
        )
    for name, spec in specs.items():
        if spec["source"] == originals[spec["path"]]:
            raise ValueError(f"{name}: mutation did not change its certified source")
        try:
            ast.parse(spec["source"], filename=spec["path"])
        except SyntaxError as error:
            raise ValueError(
                f"{name}: mutation is not valid Python: {error}"
            ) from error
    return specs


def run_direct_flow_mutation_controls(*, repo_root: Path) -> dict[str, Any]:
    """Show every semantic mutation is rejected by the route-local AST proof.

    The repository control wraps these exact mutations in full source inventory,
    contract, compiled-policy, and manifest re-anchoring. This in-process half
    isolates the semantic proof and changes only temporary copies.
    """
    root = repo_root.resolve()
    clean = verify_direct_candidate_flow(repo_root=root)
    originals = {
        relative: (root / relative).read_text(encoding="utf-8")
        for relative in _CERTIFIED_CORRIDOR_SOURCES
    }
    cases: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as raw:
        temp_root = Path(raw) / "repo"
        for relative, source in originals.items():
            target = temp_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        for name, spec in direct_flow_mutation_specs(repo_root=root).items():
            relative = spec["path"]
            target = temp_root / relative
            target.write_text(spec["source"], encoding="utf-8")
            result = verify_direct_candidate_flow(repo_root=temp_root)
            cases[name] = {
                "path": relative,
                "rejected": not result["ok"],
                "errors": result["errors"],
                "offending_paths": result["offending_paths"],
            }
            target.write_text(originals[relative], encoding="utf-8")
    admitted = sorted(name for name, result in cases.items() if not result["rejected"])
    count_matches_expected = len(cases) == EXPECTED_DIRECT_FLOW_MUTATION_COUNT
    mutation_names_sha256 = _mutation_name_digest(tuple(cases))
    names_match_expected = (
        mutation_names_sha256 == EXPECTED_DIRECT_FLOW_MUTATION_NAMES_SHA256
    )
    return {
        "clean": clean,
        "mutations": cases,
        "mutation_count": len(cases),
        "expected_mutation_count": EXPECTED_DIRECT_FLOW_MUTATION_COUNT,
        "mutation_count_matches_expected": count_matches_expected,
        "mutation_names_sha256": mutation_names_sha256,
        "expected_mutation_names_sha256": (EXPECTED_DIRECT_FLOW_MUTATION_NAMES_SHA256),
        "mutation_names_match_expected": names_match_expected,
        "admitted_mutations": admitted,
        "all_rejected": (
            clean["ok"]
            and not admitted
            and count_matches_expected
            and names_match_expected
        ),
    }
