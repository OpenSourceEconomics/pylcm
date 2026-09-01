#!/usr/bin/env python3
"""Prove route-local candidate-array flow from ``Q_and_F`` to full reducers.

The executable certificate is intentionally finite. Its universal half has two
explicitly separated claims. First, every coordinate produced by an already-constructed, finalized concrete
built-in action-grid object or supplied through the public runtime-points seam reaches
the pointwise ``Q_and_F`` call as an action argument. Second, on each GridSearch
route, the exact arrays bound by ``Q_arr, F_arr = Q_and_F(...)`` must reach the
full reducer without an intervening candidate-changing expression. The streamed
solve routes make the equivalent pointwise claim: every canonical global action
identity is decoded in C order, its exact Q/feasibility pair enters the
route-appropriate mergeable hard-max reduction, and the resolved streamed program is
the one lowered, compiled, and dispatched. A collective block is scalarized without
changing action support; every stakeholder value is gathered at the one shared
household winner and published with the empty-feasible-set dissolution flag. An EV1
stream preserves the discrete-prefix branch order, hard-maxes each branch, then adds
exactly one branch value to a log-sum-exp reduction bound to the runtime scale. The economic
construction of Q/F values and feasibility—including user DAGs, constraints,
transitions, continuation values, interpolation, and fold weights—is an explicit
semantic boundary and is not re-proved here. The proof is strict by design: a new
statement in either certified transport corridor is not assumed harmless; it has
to enter the explicit, independently checked representation allowlist.

The nine corridors are:

* singleton solve -> ``Q_arr.max(where=F_arr, ...)``;
* singleton streamed solve -> complete C-order blocks -> mergeable hard max ->
  compiled VALUE core;
* singleton simulate -> ``argmax_and_max(Q_arr, where=F_arr, ...)``;
* collective solve -> ``collective_readout(..., feasibility=F_arr, ...)``;
* collective streamed solve -> complete C-order stakeholder blocks -> shared
  household hard max -> compiled ``(VALUE, DISSOLUTION_FLAG)`` core;
* collective simulate -> ``collective_argmax_and_readout(...,
  feasibility=F_arr, ...)``;
* taste-shock dense solve fallback -> exact mask, continuous maximum, then full
  discrete logsum;
* taste-shock streamed solve -> ordered discrete-prefix branch hard max -> one
  dynamically bound log-sum-exp -> compiled VALUE core;
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
CORE_PROGRAM_SOURCE = "src/_lcm/execution/core_program.py"
OUTPUT_LAYOUT_SOURCE = "src/_lcm/execution/output_layout.py"
ACTION_STREAMING_SOURCE = "src/_lcm/solution/action_streaming.py"
ACTION_REDUCTION_SOURCE = "src/_lcm/solution/action_reduction.py"
COLLECTIVE_ACTION_REDUCTION_SOURCE = "src/_lcm/solution/collective_action_reduction.py"
LOGSUMEXP_ACTION_REDUCTION_SOURCE = "src/_lcm/solution/logsumexp_action_reduction.py"
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
    CORE_PROGRAM_SOURCE,
    OUTPUT_LAYOUT_SOURCE,
    ACTION_STREAMING_SOURCE,
    ACTION_REDUCTION_SOURCE,
    COLLECTIVE_ACTION_REDUCTION_SOURCE,
    PROCESSING_SOURCE,
    LOGSUMEXP_ACTION_REDUCTION_SOURCE,
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
# repository-local helper on which the nine certified routes depend.
_SOURCE_SEALS = {
    LOGSUM_SOURCE: "e12061dd4f0f0176324182a2eb875cb6ebe4b97174091c597d46a622df93ff1b",
    ARGMAX_SOURCE: "0d179a5aa65a6f310f598bdad8f75a9318a24832e31bd529184c2ea90356a72d",
    COLLECTIVE_SOURCE: "c30b746e574f1462a152c62b72c788730bdcdceabd2d71e525bf49a6a2c2e8c0",
    MAX_Q_SOURCE: "511b3af312da2e81f8c2b5b7098b48124410b141d50e8f9666ba02a55c04543c",
    PROCESSING_SOURCE: "e9fc3dc1b8b703f12867f336ce4439356edc2fbafc919ae2014cd939234a1b56",
    GRID_SEARCH_SOURCE: "20d36ae31f1c026b3d7eb120d120c8da7dd91c7d050d24777a1c1066e141946a",
    CORE_PROGRAM_SOURCE: "661af8a35e2eea2e29ebfbe4d4ba9ba15e00972e910d93a98e06a5fa1286d84f",
    OUTPUT_LAYOUT_SOURCE: "d08aab63241c85d14ef8a59b105a9e45effc5cc2e4e3d0beb6d8fb36431fa43e",
    ACTION_STREAMING_SOURCE: "5aeafaa39498845cd7104c57aa811a108594bfc9755a3b2cd93ea592ebbc245e",
    ACTION_REDUCTION_SOURCE: "6cee6ea2dbef0ba710fa4a318a2113377d6513cee508e73004861597f9c220f9",
    COLLECTIVE_ACTION_REDUCTION_SOURCE: "7a80418764fdf9754062a23707c5d39fa7abfcaac9c8e8f7803c4d8f1b461347",
    DISPATCHERS_SOURCE: "ef1d85c4fa7dbfbedc2c4afc36f307b915975c8a4a379aa40380fe8c89ecb663",
    FUNCTOOLS_SOURCE: "e6707e1f76493a28023a1f1f536414ba6792df063e6cfc161fca2690b8f5bc1c",
    CONTAINERS_SOURCE: "0838079e35ba498009d8af7e6ed717f870a96a2fdc628d25e80310cd630174a9",
    ZERO_SAFE_SOURCE: "6b85bacd7c01fec283fcd309a731ab73d6639975ff34edbcce1a8450fbac5f33",
    LOGSUMEXP_ACTION_REDUCTION_SOURCE: "732fda3ed4058445dedd160e58e3899ba286929413fd533e57113b5f772e7b79",
    PROBABILITY_SOURCE: "b59d16c16147af2518daaed643c10be43c506c6e3ac751cd52f04fa8fdab20d2",
    ENGINE_SOURCE: "6d3962297eab6d6053619533211fa0e5a79fecf925cad671dc3c4d9d5bcc80e7",
    STATE_ACTION_SPACE_SOURCE: "c7af3ea4c3912efa3d5d7daa0d420168a7545e327f6e4c581b3baf54efc79f11",
    SIMULATION_SOURCE: "b55ee1a42dea02e46cbf0036ee6ee46cfdbb81ed29b9b45d8cd2d474d93b35bb",
    SIMULATION_TRANSITIONS_SOURCE: "1c503777887af52d1d5de36cf86acb4d8431fbe3d71203d7da881b4d0742c928",
    SIMULATION_COMPILE_SOURCE: "926feb249828f03cf722f8e517706e6651b0359f93fd29c6f87650773bfcaf04",
    MODEL_SOURCE: "5045a42535a33d4337700c7161bf31ade717b43a60c3f79ef71cbe3c537347cf",
    BACKWARD_INDUCTION_SOURCE: "a0ae203d1388130d5ab017a40d19eb86d0653edd5c6bbd11b5507487afc7d83b",
    INITIAL_CONDITIONS_SOURCE: "cb3663f59d10fa288d3da322b5f154545bb1ac4b9262073a87c405b5e950507f",
    RESULT_SOURCE: "992f8e14d2f47f505f6883e340e89d68dfc41311d4c36f7849a2f79331e4ba01",
    RESULT_DATAFRAME_SOURCE: "025e273c4d3bb9d8f9787189a551b113708c86b1e868d16178aa39555abf49a4",
    RESULT_METADATA_SOURCE: "5745acf8a75655a4da87c1d305d79db31582d1e4df419c059059d515770ed563",
    ADDITIONAL_TARGETS_SOURCE: "d1c8787e7968b868b4b09a90544050c5da65d2ca6203f2bc52fe6b7b7dd351e4",
    SIMULATION_RANDOM_SOURCE: "0f7d81ab5c36343ab24363dca159451c0a44a399ab150abd2e52fea6715ff20e",
    FOLD_ZERO_SAFE_SOURCE: "0f6c6c3ad1a69ea2ef241f8f0ce924e18c00e6515c7509577c761a8151d57feb",
    SOLUTION_CONTRACT_SOURCE: "e0ea892e2a146395a9eaa4b084f1a29b5064ce80093ac3ae674445e29475df9a",
    GRIDS_INIT_SOURCE: "c66aed5ef6cdb56cfa38eebb7f870f12475f7a5f62ca1962c17230f66fd3268a",
    GRID_BASE_SOURCE: "fd1064986abdbe1755383fb08758f74d40cad419c0da312f38b521c7d78ce59c",
    GRID_COORDINATES_SOURCE: "e0f3cffc38e2a854426309b3eacab5783a0a5725cc4e763a06969e03914619e8",
    DISCRETE_GRID_SOURCE: "7356e8ef8fffaadcdbb01db2862065dd146d1c538cb71bf5292a89e0b98d0409",
    CONTINUOUS_GRID_SOURCE: "41243fac783c95e2a6a9a7f431e8b7fec31918ab6e2a41c23669fdfca0646e60",
    PIECEWISE_GRID_SOURCE: "75077bbce519ebcfd00497102666e9470850019cb735101a626b9555114544fe",
    PROCESSES_INIT_SOURCE: "db7892762ef1b5635b61b4e57ef97ae0becdd1fc7507ef8efd202323ced7671f",
    PROCESS_BASE_SOURCE: "a058be1a20858d4208305114e86b8a2e0bb04c75182a92b610f32c1e08cb5e03",
    PROCESS_IID_SOURCE: "4696d7356181bc07e2d661d9010c95f014a99a5110935ead56ff05ae1725d3bf",
    PROCESS_AR1_SOURCE: "05c03c7ac6f9b600a160be81851b8c868c7919daeeeadb817b201c87d7d9213e",
    VARIABLES_SOURCE: "f416e9442bc6140ed9c590a589e3552ccbf5be42af126d69ef5b439038a89ddf",
    PARAMS_REGIME_TEMPLATE_SOURCE: "5fb2c126565ebc2e2c9cc2079abeee7015d3da4deee3bf19fa13105a66bc225e",
    PARAMS_PROCESSING_SOURCE: "6a67a1d9c67a0b95427041eedd80d49def9e2bd42e11f2711aa4a52a1faff2c0",
    DTYPES_SOURCE: "0df3aa83d3d7d2f55438d91b9d4af2f25a17ea0ce836e923ab06a7458c59e73a",
    NAMESPACE_SOURCE: "254509e538c6a2264a71e04cdd5abdb60ad92f04899a37f710004222ae855bea",
    PANDAS_UTILS_SOURCE: "714eb0104a8392dac83aac6a2b3c23ccafe16d9d8b2871145f86e71b1d8880a3",
    MODEL_PROCESSING_SOURCE: "8a159831e37a9582852908c7e213106aa1070500f4e64fa12f1a7ed628854740",
}

EXPECTED_DIRECT_FLOW_MUTATION_COUNT = 273
EXPECTED_DIRECT_FLOW_MUTATION_NAMES_SHA256 = (
    "0c479d1be6b5d707c18f3f95ed4c79a94d42bf2780a4fcf6d598980ac71de1eb"
)


def canonical_json(payload: dict[str, Any]) -> str:
    """Render deterministic JSON for command-line controls."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _mutation_name_digest(names: Sequence[str]) -> str:
    """Hash the exact sorted mutation-name family, including its cardinality."""
    payload = ("\n".join(sorted(names)) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _name(*, node: ast.AST | None, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _call_name(call: ast.Call) -> str | None:
    """Return only an unqualified direct callee; attributes are not lookalikes."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _keyword(*, call: ast.Call, name: str) -> ast.expr | None:
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


def _stored_name_count(*, node: ast.AST, name: str) -> int:
    """Count every assignment-form store of ``name`` below one AST node."""
    return sum(
        isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Store)
        and child.id == name
        for child in ast.walk(node)
    )


def _definition(*, tree: ast.Module, name: str) -> ast.FunctionDef:
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
    *,
    tree: ast.Module,
    outer_name: str,
    nested_name: str,
    taste_shocks: bool,
) -> ast.FunctionDef:
    """Resolve one reducer from the direct ``has_taste_shocks`` guard branch."""
    outer = _definition(tree=tree, name=outer_name)
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
    *, tree: ast.Module, outer_name: str, nested_name: str
) -> ast.FunctionDef:
    """Return the reducer from the false arm of ``has_taste_shocks``."""
    return _guarded_nested(
        tree=tree,
        outer_name=outer_name,
        nested_name=nested_name,
        taste_shocks=False,
    )


def _taste_nested(
    *, tree: ast.Module, outer_name: str, nested_name: str
) -> ast.FunctionDef:
    """Return the reducer from the true arm of ``has_taste_shocks``."""
    return _guarded_nested(
        tree=tree,
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


def _callable_ast_sha256(node: ast.FunctionDef) -> str:
    """Hash one callable's exact AST, ignoring only its docstring and locations.

    This is the compact form of the fail-closed allowlist for long transport
    adapters. It includes the signature, annotations, decorators, type
    parameters, and every executable statement; comments, formatting, and a
    documentation-only edit do not force a semantic re-anchor.
    """
    parts: list[str | None] = [
        _ast_key(node.args),
        *(_ast_key(item) for item in node.decorator_list),
        _ast_key(node.returns) if node.returns is not None else None,
        node.type_comment,
        *(_ast_key(item) for item in getattr(node, "type_params", ())),
        *(_ast_key(item) for item in _body_without_docstring(node)),
    ]
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _statements_ast_sha256(statements: Sequence[ast.stmt]) -> str:
    """Hash an exact ordered statement corridor, independent of locations."""
    payload = json.dumps(
        [_ast_key(item) for item in statements],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _exact_callable_errors(
    *,
    tree: ast.Module,
    label: str,
    contracts: dict[str, str],
) -> list[str]:
    """Check exact callable ASTs selected by top-level or ``Class.method`` name."""
    errors: list[str] = []
    for qualname, expected in contracts.items():
        try:
            if "." in qualname:
                class_name, method_name = qualname.split(".", maxsplit=1)
                _, node = _method_definition(
                    tree=tree, class_name=class_name, method_name=method_name
                )
            else:
                node = _definition(tree=tree, name=qualname)
        except ValueError as error:
            errors.append(f"{label}: {error}")
            continue
        actual = _callable_ast_sha256(node)
        if actual != expected:
            errors.append(f"{label}: exact callable corridor {qualname!r} changed")
    return errors


def _class_surface_errors(
    *,
    tree: ast.Module,
    label: str,
    class_name: str,
    fields: tuple[str, ...],
    methods: tuple[str, ...],
    decorators: tuple[str, ...] = ("dataclass(frozen=True, kw_only=True)",),
) -> list[str]:
    """Forbid descriptors or magic methods from bypassing an exact corridor."""
    try:
        cls = _class_definition(tree=tree, name=class_name)
    except ValueError as error:
        return [f"{label}: {error}"]
    observed_fields = tuple(
        ast.unparse(item)
        for item in cls.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    )
    observed_methods = tuple(
        item.name for item in cls.body if isinstance(item, ast.FunctionDef)
    )
    errors: list[str] = []
    if (
        tuple(ast.unparse(item) for item in cls.decorator_list) != decorators
        or cls.bases
        or cls.keywords
        or observed_fields != fields
        or observed_methods != methods
    ):
        errors.append(f"{label}: {class_name} class surface changed")
    return errors


def _expected_statements(source: str) -> list[ast.stmt]:
    """Parse an allowlisted statement sequence under the running Python AST."""
    return ast.parse(source).body


def _body_matches(*, node: ast.FunctionDef, expected_source: str) -> bool:
    observed = [_ast_key(item) for item in _body_without_docstring(node)]
    expected = [_ast_key(item) for item in _expected_statements(expected_source)]
    return observed == expected


def _expression_matches(*, node: ast.AST | None, source: str) -> bool:
    """Compare one expression with a hard-coded, location-free AST."""
    return node is not None and _ast_key(node) == _ast_key(
        ast.parse(source, mode="eval").body
    )


def _exact_reducer_decorator(
    *, node: ast.FunctionDef, simulate: bool, taste_shocks: bool
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
        _expression_matches(node=_keyword(call=call, name="args"), source=args_source)
        and _expression_matches(
            node=_keyword(call=call, name="return_annotation"), source=annotation_source
        )
        and _expression_matches(
            node=_keyword(call=call, name="enforce"), source="False"
        )
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
    *,
    node: ast.FunctionDef,
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
    *,
    node: ast.FunctionDef,
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


def _method_keyword_only_signature(
    *,
    node: ast.FunctionDef,
    self_name: str,
    names: tuple[str, ...],
    defaults: tuple[str | None, ...] | None = None,
) -> bool:
    args = node.args
    expected_defaults = (None,) * len(names) if defaults is None else defaults
    return (
        not args.posonlyargs
        and tuple(item.arg for item in args.args) == (self_name,)
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


def _relevant_imports(*, tree: ast.Module, names: set[str]) -> list[str]:
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


def _statements_match(*, observed: Sequence[ast.stmt], expected_source: str) -> bool:
    """Compare one statement sequence with a hard-coded AST allowlist."""
    return [_ast_key(item) for item in observed] == [
        _ast_key(item) for item in _expected_statements(expected_source)
    ]


def _module_contract_errors(
    *,
    tree: ast.Module,
    label: str,
    relevant_import_names: set[str],
    expected_imports: list[str],
    expected_binding_counts: dict[str, int],
) -> list[str]:
    """Pin critical imports and reject every same-scope shadowing form."""
    errors: list[str] = []
    observed_imports = _relevant_imports(tree=tree, names=relevant_import_names)
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


def _class_definition(*, tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one top-level class {name!r}, found {len(matches)}")
    return matches[0]


def _method_definition(
    *, tree: ast.Module, class_name: str, method_name: str
) -> tuple[ast.ClassDef, ast.FunctionDef]:
    cls = _class_definition(tree=tree, name=class_name)
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


def _function_definition(*, tree: ast.Module, name: str) -> ast.FunctionDef:
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
        cls = _class_definition(tree=tree, name="Grid")
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
            or not _positional_signature(node=node, names=("self",))
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
            tree=tree, class_name="StateActionSpace", method_name="action_names"
        )
        _, actions = _method_definition(
            tree=tree, class_name="StateActionSpace", method_name="actions"
        )
        _, shapes = _method_definition(
            tree=tree, class_name="StateActionSpace", method_name="actions_grid_shapes"
        )
        _, replace = _method_definition(
            tree=tree, class_name="StateActionSpace", method_name="replace"
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
            or not _positional_signature(node=node, names=("self",))
            or not _body_matches(node=node, expected_source=expected_body)
        ):
            errors.append(
                f"state-action space: StateActionSpace.{node.name} no longer "
                "publishes the full ordered candidate mapping"
            )
    if (
        replace.decorator_list
        or not _method_keyword_only_signature(
            node=replace,
            self_name="self",
            names=("states", "discrete_actions", "continuous_actions"),
            defaults=("None", "None", "None"),
        )
        or not _body_matches(
            node=replace,
            expected_source="states = first_non_none(states, self.states)\n"
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
        node = _function_definition(tree=tree, name="create_regime_state_action_space")
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
        or not _keyword_only_signature(
            node=node, names=("regime", "regime_states", "base")
        )
        or not _body_matches(node=node, expected_source=expected_body)
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
        node = _function_definition(tree=tree, name="_simulate_regime_in_period")
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
        and _name(node=_keyword(call=call, name="regime"), expected="regime")
        and _expression_matches(
            node=_keyword(call=call, name="regime_states"), source="states[regime_name]"
        )
        and _name(
            node=_keyword(call=call, name="base"), expected="base_state_action_space"
        )
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
    if _unparse(_keyword(call=statement.value, name="next_regime_to_V_arr")) != (
        "next_regime_to_V_arr"
    ):
        return False
    splats = [item.value for item in statement.value.keywords if item.arg is None]
    return len(splats) == 1 and _name(node=splats[0], expected="states_actions_params")


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
        and _name(node=call.func.value, expected="Q_arr")
        and not call.args
    ):
        return False
    return (
        _name(node=_keyword(call=call, name="where"), expected="F_arr")
        and _negative_infinity(_keyword(call=call, name="initial"))
        and _keyword(call=call, name="axis") is None
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
        and not call.args
        and _name(node=_keyword(call=call, name="a"), expected="Q_arr")
        and _name(node=_keyword(call=call, name="where"), expected="F_arr")
        and _negative_infinity(_keyword(call=call, name="initial"))
        and _keyword(call=call, name="axis") is None
        and {item.arg for item in call.keywords} == {"a", "where", "initial"}
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
    if not (
        _name(node=comp.key, expected="name") and isinstance(comp.value, ast.Subscript)
    ):
        return False
    if not _name(node=comp.value.value, expected="Q_arr"):
        return False
    slice_node = comp.value.slice
    if not (
        isinstance(slice_node, ast.Tuple)
        and len(slice_node.elts) == 2
        and isinstance(slice_node.elts[0], ast.Constant)
        and slice_node.elts[0].value is Ellipsis
        and _name(node=slice_node.elts[1], expected="index")
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
        and _name(node=generator.iter.args[0], expected="stakeholders")
        and not generator.iter.keywords
        and not generator.ifs
        and generator.is_async == 0
    )


def _exact_weights_call(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and _call_name(node) == "_evaluate_pareto_weights"
        and not node.args
        and _name(
            node=_keyword(call=node, name="pareto_weights"), expected="pareto_weights"
        )
        and _name(
            node=_keyword(call=node, name="states_actions_params"),
            expected="states_actions_params",
        )
        and {item.arg for item in node.keywords}
        == {"pareto_weights", "states_actions_params"}
    )


def _exact_collective_reducer_assignment(
    *, statement: ast.stmt, simulate: bool
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
        and _name(
            node=_keyword(call=call, name="stakeholder_Q"), expected="stakeholder_Q"
        )
        and _name(node=_keyword(call=call, name="feasibility"), expected="F_arr")
        and _exact_weights_call(_keyword(call=call, name="weights"))
        and _name(node=_keyword(call=call, name="action_axes"), expected="action_axes")
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
        and _name(node=comp.elt.value, expected="values")
        and _name(node=comp.elt.slice, expected="name")
        and len(comp.generators) == 1
    ):
        return False
    generator = comp.generators[0]
    return (
        _name(node=generator.target, expected="name")
        and _name(node=generator.iter, expected="stakeholders")
        and not generator.ifs
        and generator.is_async == 0
        and _unparse(_keyword(call=node, name="axis")) == "-1"
        and {item.arg for item in node.keywords} == {"axis"}
    )


def _exact_collective_return(*, statement: ast.stmt, simulate: bool) -> bool:
    if not (
        isinstance(statement, ast.Return) and isinstance(statement.value, ast.Tuple)
    ):
        return False
    if simulate:
        return len(statement.value.elts) == 2 and all(
            _name(node=node, expected=expected)
            for node, expected in zip(
                statement.value.elts, ("argmax_flat", "V_stacked"), strict=True
            )
        )
    return (
        len(statement.value.elts) == 2
        and _exact_values_stack(statement.value.elts[0])
        and _name(node=statement.value.elts[1], expected="dissolution")
    )


def _exact_collective_body(*, node: ast.If, simulate: bool) -> bool:  # noqa: PLR0911
    if ast.unparse(node.test) != "stakeholders is not None" or node.orelse:
        return False
    expected_length = 5 if simulate else 4
    if len(node.body) != expected_length:
        return False
    if not _exact_action_axes(node.body[0]):
        return False
    if not _exact_stakeholder_split(node.body[1]):
        return False
    if not _exact_collective_reducer_assignment(
        statement=node.body[2], simulate=simulate
    ):
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
        return _exact_collective_return(statement=node.body[4], simulate=True)
    return _exact_collective_return(statement=node.body[3], simulate=False)


def _productmap_binding_errors(
    *, tree: ast.Module, outer_name: str, nested_name: str
) -> list[str]:
    """Require one unwrapped, unbatched action product and no later rebinding."""
    try:
        outer = _definition(tree=tree, name=outer_name)
    except ValueError as error:
        return [f"{outer_name}: {error}"]
    assignments = [
        statement
        for statement in ast.walk(outer)
        if isinstance(statement, ast.Assign)
        and any(_target_names(target) == ("Q_and_F",) for target in statement.targets)
    ]
    store_count = _stored_name_count(node=outer, name="Q_and_F")
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
            and _name(node=_keyword(call=value, name="func"), expected="Q_and_F")
            and _name(
                node=_keyword(call=value, name="variables"), expected="action_names"
            )
            and _unparse(_keyword(call=value, name="batch_sizes"))
            == "dict.fromkeys(action_names, 0)"
            and {item.arg for item in value.keywords}
            == {"func", "variables", "batch_sizes"}
        ):
            errors.append(
                f"{outer_name}: action productmap is wrapped, filtered, batched, "
                "or does not consume the original Q_and_F"
            )
    if _stored_name_count(node=outer, name=nested_name):
        errors.append(
            f"{outer_name}: returned reducer {nested_name} is rebound after definition"
        )
    captured_rebindings = any(
        _stored_name_count(node=outer, name=name)
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
        mapped=cast("Callable[..., FloatND]", mapped),
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
return cast("MaxQOverAFunction", allow_only_kwargs(func=mapped, enforce=False))
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
            outer = _definition(tree=tree, name=outer_name)
        except ValueError as error:
            errors.append(f"{outer_name}: {error}")
            continue
        if outer.decorator_list:
            errors.append(f"{outer_name}: builder decorators are not allowlisted")
        names, defaults = signatures[outer_name]
        if not _keyword_only_signature(node=outer, names=names, defaults=defaults):
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
        if not _statements_match(
            observed=body[:index], expected_source=expected_prefix
        ):
            errors.append(
                f"{outer_name}: pre-guard Q/action wiring differs from the allowlist"
            )
        if not _statements_match(
            observed=body[index + 1 :], expected_source=expected_suffix
        ):
            errors.append(
                f"{outer_name}: certified reducer is not the exact mapped/returned route"
            )
    errors.extend(
        _module_contract_errors(
            tree=tree,
            label="max-Q builders",
            relevant_import_names={
                "MappingProxyType",
                "ParetoWeights",
                "allow_args",
                "allow_only_kwargs",
                "argmax_and_max",
                "build_streaming_collective_max_Q_over_a",
                "build_streaming_ev1_max_Q_over_a",
                "build_streaming_max_Q_over_a",
                "cast",
                "collective_argmax_and_readout",
                "collective_readout",
                "EULER_GAMMA",
                "inspect",
                "jax",
                "jnp",
                "logsum_and_softmax",
                "math",
                "productmap",
                "vmap_1d",
                "with_signature",
                "ScalarFloat",
            },
            expected_imports=[
                "import inspect",
                "import math",
                "from types import MappingProxyType",
                "from typing import cast",
                "import jax",
                "import jax.numpy as jnp",
                "from dags import with_signature",
                "from _lcm.logsum import EULER_GAMMA, logsum_and_softmax",
                "from _lcm.regime_building.argmax import argmax_and_max",
                "from _lcm.regime_building.collective import ParetoWeights, collective_argmax_and_readout, collective_readout",
                "from _lcm.solution.action_streaming import build_streaming_collective_max_Q_over_a, build_streaming_ev1_max_Q_over_a, build_streaming_max_Q_over_a",
                "from _lcm.utils.dispatchers import productmap, vmap_1d",
                "from _lcm.utils.functools import allow_args, allow_only_kwargs",
                "from lcm.typing import BoolND, FloatND, IntND, ScalarFloat",
            ],
            expected_binding_counts={
                "MappingProxyType": 1,
                "ParetoWeights": 1,
                "allow_args": 1,
                "allow_only_kwargs": 1,
                "argmax_and_max": 1,
                "build_streaming_collective_max_Q_over_a": 1,
                "build_streaming_ev1_max_Q_over_a": 1,
                "build_streaming_max_Q_over_a": 1,
                "cast": 1,
                "collective_argmax_and_readout": 1,
                "collective_readout": 1,
                "dict": 0,
                "draw_taste_shock_noise": 1,
                "enumerate": 0,
                "EULER_GAMMA": 1,
                "get_argmax_and_max_Q_over_a": 1,
                "get_max_Q_over_a": 1,
                "get_streaming_max_Q_over_a": 1,
                "inspect": 1,
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
                "ScalarFloat": 1,
                "zip": 0,
            },
        )
    )
    return errors


def _streamed_max_builder_errors(tree: ast.Module) -> list[str]:
    """Pin the VALUE-producing streamed builder and its fail-closed boundary."""
    return _exact_callable_errors(
        tree=tree,
        label="streamed max-Q builder",
        contracts={
            "get_streaming_max_Q_over_a": "c900b6ce4111752f253aa395a68fb4ed686937769d1fc4b74543785223e2fa4b",
            "_fail_if_full_V_streaming_route_is_unsupported": (
                "f0b883e839ecb385873df428718b60ad53745623fe135726b69d640737a61987"
            ),
        },
    )


def _core_program_transport_errors(tree: ast.Module) -> list[str]:
    """Pin provider declarations through static width materialization."""
    errors = _class_surface_errors(
        tree=tree,
        label="core-program action axis",
        class_name="StreamableProductAxis",
        fields=(
            "name: str",
            "coordinate_names: tuple[ActionName, ...]",
            "coordinate_extents: tuple[int, ...]",
            "canonical_order: str",
            "reduction: ReductionSemantics",
            "width_keyword: str",
        ),
        methods=("__post_init__", "extent"),
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="core-program execution requirements",
            class_name="CoreExecutionRequirements",
            fields=("streamable_axes: tuple[StreamableProductAxis, ...] = ()",),
            methods=("__post_init__",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="core-program declaration",
            class_name="CoreProgram",
            fields=(
                "function: Callable[..., object]",
                "arguments: Mapping[str, object]",
                "requirements: CoreExecutionRequirements",
                "output_roles: object",
            ),
            methods=("__post_init__",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="resolved core-program declaration",
            class_name="ResolvedCoreProgram",
            fields=(
                "function: Callable[..., object]",
                "arguments: Mapping[str, object]",
                "static_kwargs: Mapping[str, int]",
                "output_roles: object",
                "tile_widths: Mapping[str, int]",
                "specialization_key: Hashable",
            ),
            methods=("__post_init__",),
        )
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="core-program provider-to-resolver transport",
            contracts={
                "StreamableProductAxis.__post_init__": "a83a07c71ec8a26d8ad9b8cc71ea65e3df8d000e7efdcc44802bc240a5db95c6",
                "StreamableProductAxis.extent": "aebdd54708094461d473977783f0588d2491c212629e1f5a40f8cf24c789802a",
                "CoreExecutionRequirements.__post_init__": "af2a5d55e0053f39dc21f2487f2ed4fc227b3e9f963d5df6f0395b112163b203",
                "CoreProgram.__post_init__": "909fc04f725fc31d2ea54e87b9fe093ebe642b6b3576bd47b8777da77d80d99a",
                "ResolvedCoreProgram.__post_init__": "b380cc2a415554b9a546f6b80c7608d4b6636271ba53ba01dc6ecfb7fbe443ee",
                "resolve_core_program": "d729e28465dcc3bd8718d3c7207e9f744de5eca41e53307b0c3b782b61e5e70c",
                "_validate_core_program": "5878fa720a522b99c150093d5342bdbf3e321217278be102ac0d606df3b7d07e",
                "_validate_streamable_axis": "49d750f0fa436a7971480f75485b11dfa08643b1d709cadefbeedea2e0b9dd45",
                "_validate_tile_width": "15f543450ad5f6b739cfa8aafbf0a4fb34a0c5b677da4866a2afedd1effde831",
                "_validate_coordinate_argument": "401392c0069102b2983559a9b7c53bba6bd66ccf0b0126f8e6688bff49564b27",
                "_validate_width_keyword": "57c76f0f7ffbb02e0ab500552522e0d1ec8652173d7b9ba81c842ec558beeaff",
            },
        )
    )
    return errors


def _action_streaming_errors(tree: ast.Module) -> list[str]:
    """Pin complete C-order block evaluation and exact reducer delegation."""
    errors = _class_surface_errors(
        tree=tree,
        label="streamed action evaluator",
        class_name="_StreamingHardMax",
        fields=(
            "Q_and_F: Callable[..., tuple[Any, Any]]",
            "action_names: tuple[str, ...]",
            "block_width: int",
        ),
        methods=("__call__",),
        decorators=("dataclass(frozen=True)",),
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="streamed collective action evaluator",
            class_name="_StreamingCollectiveHardMax",
            fields=(
                "Q_and_F: Callable[..., tuple[Any, Any]]",
                "action_names: tuple[str, ...]",
                "block_width: int",
                "stakeholders: tuple[str, ...]",
                "weights: Mapping[str, Any]",
            ),
            methods=("__call__",),
            decorators=("dataclass(frozen=True)",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="streamed EV1 reduction identity",
            class_name="GridSearchEV1ActionReduction",
            fields=("n_discrete_action_axes: int",),
            methods=("semantic_key",),
            decorators=("dataclass(frozen=True)",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="streamed EV1 action evaluator",
            class_name="_StreamingEV1ExpectedMax",
            fields=(
                "Q_and_F: Callable[..., tuple[Any, Any]]",
                "action_names: tuple[str, ...]",
                "n_discrete_action_axes: int",
                "block_width: int",
                "scale: Any",
            ),
            methods=("__call__",),
            decorators=("dataclass(frozen=True)",),
        )
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="streamed action evaluator",
            contracts={
                "build_streaming_max_Q_over_a": "e5ae9742ba7baf789ff4659fd14c544ae5e5034a14117a1e704b27babd1087c6",
                "build_streaming_collective_max_Q_over_a": "9b855229c633e1a03ccbd56aa626be0d6f5ab668b1881d9f54f9010083842d70",
                "build_streaming_ev1_max_Q_over_a": "58c1d90efaa42c39bf4e8fb919a74cc1462bdec5420f016cc509af1151491236",
                "GridSearchEV1ActionReduction.semantic_key": "94630b954057476990e5872b65ba67a691852e9a1555267a2bd5dc3041915454",
                "_StreamingHardMax.__call__": "98e334a19ebeaa5e3aab9f87205166256a7cad545eba2bc2e7c1bc758d334b5b",
                "_StreamingCollectiveHardMax.__call__": "9674dcfc6fd7026209d97cb30018f96088595a132bd88124b35e65c97bd7b3af",
                "_StreamingEV1ExpectedMax.__call__": "ab1943585e65116b905bee383216c8dc7f9fe62840f550943a46fc9d85677d2e",
                "_prepare_action_call": "290fd810472aeb3336fdd437ccba50159e9d28c6822d6b8122fa4bdec8c71159",
                "_evaluate_block": "ee4a59644e10e1f5ebb8aaa917967a74eab852bd1bf1d5e824310bc2a26ab63f",
                "_evaluate_collective_block": "8030607cabf36d76d766c5e8eea3ec1e8d9bbd8f41214c3490691f8dbe718d89",
                "_start_reduction": "034b3966dd04c0e0d66e085e8e2c4e16b127e1d8a9869ecfa039c3b5e7928b04",
                "_scan_remaining_blocks": "ccfb2af321657915e360867fd57bfec3e26eae86a031380fba94cd73a7384bc7",
                "_add_block": "a5047bea80275b77727b69b06d563bcdfea7e80c0f99dda34f6570948ccd1a72",
                "_reduce_no_action": "47067ad94112c815eccf14395b03ad372376e53cfd5b4f337ec70404d3a32da6",
                "_start_collective_reduction": "019872b31d89c2a2c8ba68ee6128a9feb2b90b52c6ee3b00506708db6195adce",
                "_scan_remaining_collective_blocks": "15c53ba8cc665ca03333103fe94a5f101e92fb216341e521fd1360097eb7987b",
                "_add_collective_block": "5ed017db741c30cae43dd1c4a526c703c8601857c2fbca23e9900ca560e78198",
                "_reduce_collective_no_action": "dbd215d04c091bdf6942fa6edb4a561468a5bc234f13ae24a67362dc5d48a9ff",
                "_decode_action": "5ac47e5a2d400754255cd938bf9e27ec91007741c8bbf5ad18e04bb3a9a24cbe",
                "_validate_scalar_Q_and_F": "a2b49855d9fe1572f7440248db7edfce283a2179394648ca55b6b8ce550f5364",
                "_validate_block_Q_and_F": "7f00abbccfe23768df403596eb22c715eb3542b4e43dd67593f7bd3e487fdeef",
                "_validate_collective_scalar_Q_and_F": "600332c08d2ed5aa6a4ffaeee07a878c8a713a674f54456c8a9d05dbe9eddd35",
                "_validate_collective_block_Q_and_F": "1e12615fa5fa04137cebb137387bc1c2ad98b31f1978dd47f209d00207036bef",
                "_initialize_ev1_reduction": "8f832560dbfc9f1106332f4add12870ce8b27d9e42294697927e685002a39773",
                "_add_ev1_block": "5b95419379b0bc2ae1b8cfe9e8d8dd825350d79b0a816ddb7ecf22c0e50295f4",
                "_add_valid_ev1_candidate": "a629858dcdb86fc359a7364b152543827de929b5b72d5c5b32e0a6341588dd24",
                "_finalize_open_ev1_branch": "7e2a8e044b042f1feb5b529c5a4dce226271aa3d7c95f0d2c306285f11fdcd09",
                "_scan_remaining_ev1_blocks": "1ffbb0552e114c84c38f3879fe8bf5152fc7091c0576c101935209089ad3d3cd",
                "_flush_ev1_branch": "06b30206a5a2d0d265dd0153e1d8f695b162ddf6af0174d2449c5b21a87d7c86",
            },
        )
    )
    return errors


def _hard_max_streaming_reduction_errors(tree: ast.Module) -> list[str]:
    """Pin the complete hard-max accumulator and global-identity merge law."""
    return _exact_callable_errors(
        tree=tree,
        label="streamed singleton hard-max reduction",
        contracts={
            "HardMaxReduction.semantic_key": "f024d59aadbce68d4647522cd802f542ed3a39c7cbc05664b03c5a362c6468bd",
            "HardMaxReduction.initialize": "b29e84926276a74848f11826cb36ca2442e00cbc3ab3819bd197bfad624bc671",
            "HardMaxReduction.add": "5264b88c3ba353f158b394889295be544309038425796dc8f68859ff977c3880",
            "HardMaxReduction.merge": "de104bfa46bf5dff388f43bd1c4c696a4f1527613a2efcb359a762b513f28e2b",
            "HardMaxReduction.finalize": "40a21bb4b44366d00ec79a56e7aa7594a7b7b5427e3c29d9910cbc9a1e69bed3",
            "_reduce_block": "177143b0222c6386a30827b154bc0f618b7cebf9991d978c4afcc7575dc0dcd7",
        },
    )


def _collective_hard_max_streaming_reduction_errors(
    tree: ast.Module,
) -> list[str]:
    """Pin the shared-household winner and stakeholder-value gather law."""
    return _exact_callable_errors(
        tree=tree,
        label="streamed collective hard-max reduction",
        contracts={
            "CollectiveHardMaxReduction.semantic_key": "1a2875f2b718377e51a76e0d37f1614f40f3e08b4cc730032add7a2373c5734b",
            "CollectiveHardMaxReduction.initialize": "78e55824232f334491375fa641fb20e28bdc7be3dc2cd59cceedbe1b1e74db03",
            "CollectiveHardMaxReduction.add": "8d69ebff49fe1f231e7941129ea582137430dfeb5c2719061676691e80aee747",
            "CollectiveHardMaxReduction.merge": "4e288cd957f4840ebc2f5c185051a208c8e82d7df59d8d64dda3f9e5a42f530c",
            "CollectiveHardMaxReduction.finalize": "2c2128c3095d373853e0bfc2bf9f8519d8782c58c9170fd79a5cc96358d6ee47",
            "_validate_block_shapes": "ef3ba0ed14e345bd21da5ab0ac1e79824b04317f8817fce58f8ecd07a8a1b8a5",
            "_reduce_block": "75ee08bb4dc9fc5bc9ec1ef3d700dba200b3e3cea5fd8def060f12d70403bd31",
            "_take_stakeholder_values": "b84709a267bb886bef97f01076e40d5670e30caa1c4ffeede8d402008848072d",
        },
    )


def _logsumexp_streaming_reduction_errors(tree: ast.Module) -> list[str]:
    """Pin one dynamically bound log-sum-exp session across every branch."""
    errors = _class_surface_errors(
        tree=tree,
        label="streamed bound log-sum-exp reduction",
        class_name="BoundLogSumExpReduction",
        fields=("scale: FloatND",),
        methods=("initialize", "add", "merge", "finalize"),
        decorators=("dataclass(frozen=True)",),
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="streamed log-sum-exp reduction",
            contracts={
                "BoundLogSumExpReduction.initialize": "a85a8161da058019e24d2b33ce72d9881e4d8df843bcb57e2ea1143c7ad37d36",
                "BoundLogSumExpReduction.add": "bc07d13e5fa3101df3216f75942966086537395d81bb2c6abc4276158ccf9a4d",
                "BoundLogSumExpReduction.merge": "e1b12504e631c9659f1de5f55a48de26e3d4a1d5c089e6bc75738a52313fffc7",
                "BoundLogSumExpReduction.finalize": "a5d920589ee7f6b7454e241c9ef5b0be41c11b75c806e578932d1244884ce5cb",
                "LogSumExpReduction.semantic_key": "13bc88fd7862f49c2ef01b2de88e9c695276a163bc896d9c40dae5d4b104c671",
                "LogSumExpReduction.bind": "1e3f07eb92d208799636c7958d2d2fdd8c955863630606ffa1a1fbba7c8de06a",
            },
        )
    )
    return errors


def _grid_search_caller_errors(tree: ast.Module) -> list[str]:
    """Pin solve-side action metadata and the exact live core publication."""
    errors: list[str] = []
    try:
        cls, method = _method_definition(
            tree=tree, class_name="GridSearch", method_name="build_period_kernels"
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
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="solve caller live streamed provider",
            contracts={
                "GridSearch.build_period_kernels": "9f9021e00e2d709988d00015de23b0ee421bfce81158eb282eb5bd2909ede366",
                "_supports_action_streaming": "51e15216b7ff2b1e0051a0104bbfbabf5f7a896d1050f7e8fbb67432de77280d",
            },
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="solve caller output-layout adapter",
            class_name="_GridSearchPeriodKernel",
            fields=(
                "core: Callable",
                "unwrapped_core: Callable | None = None",
                "streamed_core: Callable | None = None",
                "action_names: tuple[ActionName, ...] = ()",
                "action_extents: tuple[int, ...] = ()",
                "regime_name: RegimeName",
                "collective: bool = False",
                "has_taste_shocks: bool = False",
                "n_discrete_action_axes: int = 0",
                "edge_reference_regimes: tuple[RegimeName, ...] = ()",
                "same_period_ref_regimes: tuple[RegimeName, ...] = ()",
                "edge_target_regimes: tuple[RegimeName, ...] = ()",
            ),
            methods=(
                "_with_edge_substitution",
                "cores",
                "build_core_program",
                "output_roles",
                "core_for_output_layout",
                "with_fixed_params",
                "build_lower_args",
                "_edge_reference_args",
                "_same_period_params",
                "__call__",
            ),
        )
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="solve caller output-layout adapter",
            contracts={
                "_GridSearchPeriodKernel.cores": "9ee3a0a03870ede4b4361004c34a9b7a50d2074a9e96d7c0c7aa44111efe5c13",
                "_GridSearchPeriodKernel.build_core_program": "cd7e1aa10a0b1bd80a70550b057764428b489ba734fa4d9da6466c38d70c016f",
                "_GridSearchPeriodKernel.output_roles": "e48524eea7cd3dce438a9826ca0efd7cc45ddf681da8d7d9b14ad7a03c3a55de",
                "_GridSearchPeriodKernel.core_for_output_layout": "c06f276c3430fe788227503d0c24602558259a1a076fd76061e613db191b87aa",
                "_GridSearchPeriodKernel.with_fixed_params": "6432aa32d09899a4b82c17568336be04fac9933c14e81f4e8460d252663b04ac",
                "_GridSearchPeriodKernel.__call__": "2b5e10a6f2282f493d49e912175cf782c7e8ddd42b1bdcca0142f896329df225",
            },
        )
    )
    errors.extend(
        _module_contract_errors(
            tree=tree,
            label="solve caller",
            relevant_import_names={
                "CoreExecutionRequirements",
                "CoreProgram",
                "COLLECTIVE_HARD_MAX_REDUCTION",
                "HARD_MAX_REDUCTION",
                "MappingProxyType",
                "REGIME_CONF",
                "Solver",
                "GridSearchEV1ActionReduction",
                "StreamableProductAxis",
                "beartype",
                "cast",
                "dataclass",
                "inspect",
                "jax",
                "math",
            },
            expected_imports=[
                "from dataclasses import dataclass, replace",
                "from types import MappingProxyType",
                "from typing import cast",
                "import jax",
                "import inspect",
                "import math",
                "from beartype import beartype",
                "from _lcm.beartype_conf import REGIME_CONF",
                "from _lcm.execution.core_program import CoreExecutionRequirements, CoreProgram, StreamableProductAxis",
                "from _lcm.solution.action_reduction import COLLECTIVE_HARD_MAX_REDUCTION, HARD_MAX_REDUCTION",
                "from _lcm.solution.action_streaming import GridSearchEV1ActionReduction",
                "from _lcm.solution.contract import ConstraintRouteContext, ContinuationPayload, KernelResult, PeriodKernel, SolutionKernels, Solver, SolverBuildContext, simulation_route",
            ],
            expected_binding_counts={
                "CoreExecutionRequirements": 1,
                "CoreProgram": 1,
                "COLLECTIVE_HARD_MAX_REDUCTION": 1,
                "GridSearch": 1,
                "HARD_MAX_REDUCTION": 1,
                "MappingProxyType": 1,
                "REGIME_CONF": 1,
                "Solver": 1,
                "StreamableProductAxis": 1,
                "beartype": 1,
                "cast": 1,
                "dataclass": 1,
                "id": 0,
                "inspect": 1,
                "jax": 1,
                "len": 0,
                "math": 1,
                "GridSearchEV1ActionReduction": 1,
            },
        )
    )
    return errors


def _output_layout_errors(tree: ast.Module) -> list[str]:
    """Pin planned lowering, validation, and identity-return publication."""
    errors: list[str] = []
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="output layout",
            class_name="ResolvedOutputLayout",
            fields=(
                "out_shardings: object",
                "compilation_key: Hashable",
                "expected_value_shape: tuple[int, ...]",
                "expected_value_dtype: object",
                "expected_dissolution_shape: tuple[int, ...] | None",
                "expected_dissolution_dtype: object | None",
            ),
            methods=(),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="output layout",
            class_name="PlannedCore",
            fields=("compiled: Callable", "layout: ResolvedOutputLayout"),
            methods=("__call__",),
        )
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="output layout",
            contracts={
                "resolve_output_layout": "17fd47d504f55333c24318d33b756838cac847be84dc737dfdafba85c53ee26e",
                "_validate_output_roles": "8bf237a4b6a84fddd8bb2d425c25561498531ff01121c205fcd3be28fca480c7",
                "assert_output_layout": "78b4040ad158cd8201ce739cfc08ee3e0ddc1d91b0019bbff5689e5b2bebd0f3",
                "_assert_output_metadata": "8612094c85183f9c7cfa568bb0aad875ce729c79c60b31126efd8d04785644ca",
                "PlannedCore.__call__": "05bb94fd00a52947741192747b970a1ef45b8a01c8dcb814c4099ab50b9c4707",
                "planned_output_layout": "0518c706a7de02411a990d4cfa9c40f426cc83f825791c735aeb19c4603e5e1e",
            },
        )
    )
    assignments = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            _target_names(target) in {("VALUE",), ("DISSOLUTION_FLAG",), ("UNPLANNED",)}
            for target in statement.targets
        )
    ]
    if not _statements_match(
        observed=assignments,
        expected_source="""VALUE = OutputRole.VALUE
DISSOLUTION_FLAG = OutputRole.DISSOLUTION_FLAG
UNPLANNED = _Unplanned.TOKEN
""",
    ):
        errors.append("output layout: logical output-role bindings changed")
    errors.extend(
        _module_contract_errors(
            tree=tree,
            label="output layout",
            relevant_import_names={
                "Callable",
                "Hashable",
                "Protocol",
                "cast",
                "dataclass",
                "jax",
                "runtime_checkable",
            },
            expected_imports=[
                "from collections.abc import Callable, Hashable",
                "from dataclasses import dataclass",
                "from typing import Protocol, cast, runtime_checkable",
                "import jax",
            ],
            expected_binding_counts={
                "Callable": 1,
                "DISSOLUTION_FLAG": 1,
                "Hashable": 1,
                "OutputRole": 1,
                "PlannedCore": 1,
                "Protocol": 1,
                "ResolvedOutputLayout": 1,
                "UNPLANNED": 1,
                "VALUE": 1,
                "_assert_output_metadata": 1,
                "_validate_output_roles": 1,
                "assert_output_layout": 1,
                "cast": 1,
                "dataclass": 1,
                "jax": 1,
                "planned_output_layout": 1,
                "resolve_output_layout": 1,
                "runtime_checkable": 1,
            },
        )
    )
    return errors


def _processing_caller_errors(tree: ast.Module) -> list[str]:
    """Pin simulation metadata, spacemapping, and live phase publication."""
    errors: list[str] = []
    try:
        builder = _definition(
            tree=tree, name="_build_argmax_and_max_Q_over_a_per_period"
        )
        live = _definition(tree=tree, name="_build_simulation_phase")
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
    if not _body_matches(node=builder, expected_source=expected_builder):
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
        observed=live_assignments, expected_source=expected_live_assignment
    ):
        errors.append("simulate caller: live phase does not call the certified builder")
    returns = [node for node in ast.walk(live) if isinstance(node, ast.Return)]
    if len(returns) != 1 or live_body[-1] is not returns[0]:
        errors.append("simulate caller: live phase gained a bypass return")
    elif not (
        isinstance(returns[0].value, ast.Call)
        and _call_name(returns[0].value) == "SimulationPhase"
        and _name(
            node=_keyword(call=returns[0].value, name="argmax_and_max_Q_over_a"),
            expected="argmax_and_max_Q_over_a",
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
            tree=tree,
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


def _initial_tile_width_validation_errors(tree: ast.Module) -> list[str]:
    """Require full program validation to dominate every width computation."""
    label = "streamed validation before initial width selection"
    try:
        node = _definition(tree=tree, name="_initial_tile_widths")
    except ValueError as error:
        return [f"{label}: {error}"]

    body = _body_without_docstring(node)
    validation_calls = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_name(child) == "_validate_core_program"
    ]
    sensitive: list[tuple[str, int]] = []
    for index, statement in enumerate(body):
        for child in ast.walk(statement):
            descendants = tuple(ast.walk(child))
            reads_extent = any(
                isinstance(item, ast.Attribute) and item.attr == "extent"
                for item in descendants
            )
            if isinstance(child, ast.Compare) and reads_extent:
                sensitive.append(("extent comparison", index))
            elif isinstance(child, ast.BinOp) and reads_extent:
                sensitive.append(("extent arithmetic", index))
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "bit_length"
            ):
                sensitive.append(("bit_length", index))

    expected_first = _expected_statements("_validate_core_program(program=program)")[0]
    required = {"extent comparison", "extent arithmetic", "bit_length"}
    observed = {kind for kind, _index in sensitive}
    if (
        not body
        or _ast_key(body[0]) != _ast_key(expected_first)
        or len(validation_calls) != 1
        or not required <= observed
        or any(index <= 0 for _kind, index in sensitive)
    ):
        return [
            (
                f"{label}: _validate_core_program must dominate every extent "
                "comparison, arithmetic operation, and bit_length call"
            )
        ]
    return []


def _backward_output_layout_errors(tree: ast.Module) -> list[str]:
    """Pin planned lowering and V/D publication through backward induction."""
    errors = _exact_callable_errors(
        tree=tree,
        label="backward output-layout transport",
        contracts={
            "_compile_all_functions": "ca5da41341db893a8b01283cc98c03402dfa4138980b5ba1f092c2acee766838",
            "_resolve_output_layouts_and_lowering_keys": "8b33f8a0bdc7071cea8cea8efe07355f46f0032b4683147bf729dfa71ff4c4ae",
            "_initial_tile_widths": "f6a820999afacbccc0b4c2544cc6501dc50d856b372a2630efa2602292144a40",
            "_assert_core_program_arguments": "0ac8d8bacbaeff579daba66defdbd5258beae61d806ec8eddb4faff4575bb32d",
            "_lowering_key": "68e82c0372400e4577a6e38d4f5c64bdff88f5f732b86b15383a5e3157973b56",
            "_abstract_arguments_key": "becd5c3e94366bc4e3e0afa31ea20886002228f7064c9a9ea0e7d0e681630dfa",
            "_abstract_value_key": "b79bdd528ed264be0093eb5d04a0341e7376606d478ba0770aa5cb14f813638b",
            "_abstract_leaf_key": "94afc53d035e8ad9812025f72adcbf76d8966430ef259d339977e9cf90262b66",
            "_output_roles_key": "c73436e6abaec7f0388386d353675200d7bf3d5d3544654f8d47f37ad0e5e8da",
            "_assert_lowered_output_roles": "46823cef80ae0bd1c5294a63472a23c4f8ade153fe8eae8327403fe637f59df8",
            "_attach_resolved_output_layout": "c85a1d4825deca8b1af859536fad9b68f25ffb434072346b3bce826de74c9342",
            "_publish_kernel_value": "5547b40f26e1fc86303ada9503f65c4210cbb4bb0b4fe6e715b9d99a58f78901",
        },
    )
    errors.extend(_initial_tile_width_validation_errors(tree))
    try:
        solve = _definition(tree=tree, name="solve")
    except ValueError as error:
        errors.append(f"backward output-layout transport: {error}")
    else:

        def _is_kernel_result(statement: ast.stmt) -> bool:
            return (
                isinstance(statement, ast.Assign)
                and any(
                    _target_names(target) == ("result",) for target in statement.targets
                )
                and isinstance(statement.value, ast.Call)
                and _call_name(statement.value) == "_run_period_kernel"
            )

        loops = [
            node
            for node in ast.walk(solve)
            if isinstance(node, ast.For)
            and any(_is_kernel_result(statement) for statement in node.body)
        ]
        corridors: list[list[ast.stmt]] = []
        for loop in loops:
            starts = [
                index
                for index, statement in enumerate(loop.body)
                if _is_kernel_result(statement)
            ]
            ends = [
                index
                for index, statement in enumerate(loop.body)
                if isinstance(statement, ast.Assign)
                and any(
                    ast.unparse(target) == "period_solution[regime_name]"
                    for target in statement.targets
                )
            ]
            if len(starts) == len(ends) == 1 and starts[0] <= ends[0]:
                corridors.append(loop.body[starts[0] : ends[0] + 1])
        expected = "bbbc89a518b5e5f8e334ceb7c38ae9472f52c8e52215a55d89a1eafab16b6bda"
        if len(corridors) != 1 or _statements_ast_sha256(corridors[0]) != expected:
            errors.append(
                "backward output-layout transport: solve publication corridor changed"
            )
    errors.extend(
        _module_contract_errors(
            tree=tree,
            label="backward output-layout transport",
            relevant_import_names={
                "CoreProgram",
                "CoreProgramAware",
                "DISSOLUTION_FLAG",
                "OutputLayoutAware",
                "PlannedCore",
                "ResolvedOutputLayout",
                "UNPLANNED",
                "VALUE",
                "assert_output_layout",
                "cast",
                "jax",
                "planned_output_layout",
                "_validate_core_program",
                "resolve_core_program",
                "resolve_output_layout",
            },
            expected_imports=[
                "from typing import cast",
                "import jax",
                "from _lcm.execution.core_program import CoreProgram, CoreProgramAware, _validate_core_program, resolve_core_program",
                "from _lcm.execution.output_layout import DISSOLUTION_FLAG, UNPLANNED, VALUE, OutputLayoutAware, PlannedCore, ResolvedOutputLayout, assert_output_layout, planned_output_layout, resolve_output_layout",
            ],
            expected_binding_counts={
                "CoreProgram": 1,
                "CoreProgramAware": 1,
                "DISSOLUTION_FLAG": 1,
                "OutputLayoutAware": 1,
                "PlannedCore": 1,
                "ResolvedOutputLayout": 1,
                "UNPLANNED": 1,
                "VALUE": 1,
                "_attach_resolved_output_layout": 1,
                "_abstract_arguments_key": 1,
                "_abstract_leaf_key": 1,
                "_abstract_value_key": 1,
                "_assert_core_program_arguments": 1,
                "_assert_lowered_output_roles": 1,
                "_compile_all_functions": 1,
                "_initial_tile_widths": 1,
                "_lowering_key": 1,
                "_output_roles_key": 1,
                "_publish_kernel_value": 1,
                "_resolve_output_layouts_and_lowering_keys": 1,
                "_validate_core_program": 1,
                "assert_output_layout": 1,
                "cast": 1,
                "jax": 1,
                "planned_output_layout": 1,
                "resolve_core_program": 1,
                "resolve_output_layout": 1,
            },
        )
    )
    return errors


def _terminal_output_wrapper_errors(tree: ast.Module) -> list[str]:
    """Pin terminal decoration as identity on the certified V/D channels."""
    errors = _class_surface_errors(
        tree=tree,
        label="terminal output-layout wrapper",
        class_name="_TerminalCarryPeriodKernel",
        fields=(
            "base: PeriodKernel",
            "carry_producer: EGMCarryProducer",
            "regime_name: RegimeName",
        ),
        methods=(
            "core",
            "cores",
            "build_core_program",
            "output_roles",
            "core_for_output_layout",
            "with_fixed_params",
            "build_lower_args",
            "__call__",
        ),
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="terminal output-layout wrapper",
            contracts={
                "_TerminalCarryPeriodKernel.cores": "690929f62b4052a0aaeab9620cf05e98bfbf7ada392d666324cadaf74e1e3d6c",
                "_TerminalCarryPeriodKernel.build_core_program": "7fb2bd594d69687b1ed37b0584f6c115a5fd0c3f2469449d73dd4d5b32d14df2",
                "_TerminalCarryPeriodKernel.output_roles": "45247a9d3be2b2696dc75097a8c3bfd97c9b40db39b87c505300903dcd4cdc55",
                "_TerminalCarryPeriodKernel.core_for_output_layout": "83978827554d2070cf6934f43151bc9af8e2b0da7ec542838d7038c49b9bce5a",
                "_TerminalCarryPeriodKernel.with_fixed_params": "c9069b5fcc41d4b7b42a2ec21be12004fa57379216fbd7e4fc624473b4d89cc6",
                "_TerminalCarryPeriodKernel.__call__": "77f27cca4b3d5da43fc3911cb736cf124829edeb7644ccabd14f0fa5607c2edc",
            },
        )
    )
    errors.extend(
        _module_contract_errors(
            tree=tree,
            label="terminal output-layout wrapper",
            relevant_import_names={
                "CoreProgram",
                "CoreProgramAware",
                "KernelResult",
                "OutputLayoutAware",
                "PeriodKernel",
                "dataclass",
                "dataclass_replace",
                "functools",
                "require_legacy_kernel_result",
            },
            expected_imports=[
                "import functools",
                "from dataclasses import dataclass",
                "from dataclasses import replace as dataclass_replace",
                "from _lcm.execution.core_program import CoreProgram, CoreProgramAware",
                "from _lcm.execution.output_layout import OutputLayoutAware",
                "from _lcm.solution.contract import ConstraintRouteContext, ContinuationPayload, KernelResult, PeriodKernel, SolverBuildContext, SolverModelContext",
                "from _lcm.solution.kernel_output import require_legacy_kernel_result",
            ],
            expected_binding_counts={
                "CoreProgram": 1,
                "CoreProgramAware": 1,
                "KernelResult": 1,
                "OutputLayoutAware": 1,
                "PeriodKernel": 1,
                "_TerminalCarryPeriodKernel": 1,
                "dataclass": 1,
                "dataclass_replace": 1,
                "functools": 1,
                "require_legacy_kernel_result": 1,
            },
        )
    )
    return errors


def _corridor_errors(
    *,
    tree: ast.Module,
    outer_name: str,
    nested_name: str,
    simulate: bool,
) -> list[str]:
    label = "simulate" if simulate else "solve"
    errors: list[str] = []
    try:
        nested = _ordinary_nested(
            tree=tree, outer_name=outer_name, nested_name=nested_name
        )
    except ValueError as error:
        return [f"{label}: {error}"]
    if not _exact_reducer_decorator(node=nested, simulate=simulate, taste_shocks=False):
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
        node=collective, simulate=simulate
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
    *,
    tree: ast.Module,
    outer_name: str,
    nested_name: str,
    simulate: bool,
) -> list[str]:
    """Pin one taste-shock route from exact Q/F origin through its full reducer."""
    label = "taste-shock simulate" if simulate else "taste-shock solve"
    try:
        nested = _taste_nested(
            tree=tree, outer_name=outer_name, nested_name=nested_name
        )
    except ValueError as error:
        return [f"{label}: {error}"]
    errors: list[str] = []
    if not _exact_reducer_decorator(node=nested, simulate=simulate, taste_shocks=True):
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
    if not _body_matches(node=nested, expected_source=expected):
        errors.append(
            f"{label}: executable body differs from the exact raw-Q/F full reduction"
        )
    return errors


def _taste_noise_errors(tree: ast.Module) -> list[str]:
    """Pin the per-discrete-cell mean-zero Gumbel helper and its imports."""
    errors: list[str] = []
    try:
        node = _definition(tree=tree, name="draw_taste_shock_noise")
    except ValueError as error:
        return [f"taste noise: {error}"]
    if node.decorator_list:
        errors.append("taste noise: decorators are not allowlisted")
    if not _keyword_only_signature(node=node, names=("key", "shape", "scale")):
        errors.append("taste noise: signature changed")
    expected_body = r"""return scale * (
    jax.random.gumbel(key, shape) - EULER_GAMMA
)
"""
    if not _body_matches(node=node, expected_source=expected_body):
        errors.append(
            "taste noise: body is not one scaled mean-zero Gumbel draw per cell"
        )

    imports = _relevant_imports(
        tree=tree,
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
        node = _definition(tree=tree, name="logsum_and_softmax")
    except ValueError as error:
        return [f"logsum reducer: {error}"]
    if node.decorator_list:
        errors.append("logsum reducer: decorators are not allowlisted")
    if not _keyword_only_signature(node=node, names=("values", "scale", "axes")):
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
    if not _body_matches(node=node, expected_source=expected_body):
        errors.append(
            "logsum reducer: executable body differs from the exact full-value flow"
        )
    imports = _relevant_imports(tree=tree, names={"jax", "jnp", "logsumexp"})
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
        node = _definition(tree=tree, name="argmax_and_max")
        move = _definition(tree=tree, name="_move_axes_to_back")
        flatten = _definition(tree=tree, name="_flatten_last_n_axes")
    except ValueError as error:
        return [f"argmax reducer: {error}"]

    if node.decorator_list or move.decorator_list or flatten.decorator_list:
        errors.append("argmax reducer: decorators are not allowlisted")
    if not _keyword_only_signature(
        node=node,
        names=("a", "axis", "initial", "where"),
        defaults=(None, "None", "None", "None"),
    ):
        errors.append("argmax reducer: signature/defaults changed")
    if not _keyword_only_signature(node=move, names=("a", "axes")):
        errors.append("argmax reducer: axis-move helper signature changed")
    if not _keyword_only_signature(node=flatten, names=("a", "n")):
        errors.append("argmax reducer: flatten helper signature changed")

    expected_argmax = r"""if axis is None:
    axis = tuple(range(a.ndim))
elif isinstance(axis, int):
    axis = (axis,)
if a.ndim == 0 or len(axis) == 0:
    return jnp.array(0, dtype=jnp.int32), a
if a.ndim != 0:
    a = _move_axes_to_back(a=a, axes=axis)
    a = _flatten_last_n_axes(a=a, n=len(axis))
if where is not None and where.ndim != 0:
    where = _move_axes_to_back(a=where, axes=axis)
    where = _flatten_last_n_axes(a=where, n=len(axis))
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
    if not _body_matches(node=node, expected_source=expected_argmax):
        errors.append(
            "argmax reducer: executable body differs from the full paired "
            "value/feasibility reduction"
        )
    if not _body_matches(node=move, expected_source=expected_move):
        errors.append(
            "argmax reducer: action-axis move is not the exact order-preserving "
            "representation change"
        )
    if not _body_matches(node=flatten, expected_source=expected_flatten):
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
        argmax_node = _definition(tree=tree, name="collective_argmax_and_readout")
        readout_node = _definition(tree=tree, name="collective_readout")
        weighted = _definition(tree=tree, name="_weighted_sum")
        gather = _definition(tree=tree, name="_gather_along_actions")
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
        if not _keyword_only_signature(node=node, names=names):
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
    a=objective, axis=action_axes, initial=-jnp.inf, where=feasibility
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
return sum_in_value_order(values=jnp.stack(terms, axis=0), axis=0)
"""
    expected_gather = r"""if not action_axes:
    return q
q_moved = _move_axes_to_back(a=q, axes=action_axes)
q_flat = _flatten_last_n_axes(a=q_moved, n=len(action_axes))
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
        if not _body_matches(node=node, expected_source=body):
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
            tree=max_tree, outer_name="get_max_Q_over_a", nested_name="max_Q_over_a"
        )
        binding_errors += _productmap_binding_errors(
            tree=max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
        )
        solve_errors = _corridor_errors(
            tree=max_tree,
            outer_name="get_max_Q_over_a",
            nested_name="max_Q_over_a",
            simulate=False,
        )
        simulate_errors = _corridor_errors(
            tree=max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
            simulate=True,
        )
        taste_solve_errors = _taste_corridor_errors(
            tree=max_tree,
            outer_name="get_max_Q_over_a",
            nested_name="max_Q_over_a",
            simulate=False,
        )
        taste_simulate_errors = _taste_corridor_errors(
            tree=max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
            simulate=True,
        )
        taste_noise_errors = _taste_noise_errors(max_tree)
        wiring_errors = _max_builder_wiring_errors(max_tree)
        streamed_builder_errors = _streamed_max_builder_errors(max_tree)
        max_errors = (
            binding_errors
            + solve_errors
            + simulate_errors
            + taste_solve_errors
            + taste_simulate_errors
            + taste_noise_errors
            + wiring_errors
            + streamed_builder_errors
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
    core_program_tree = parsed.get(CORE_PROGRAM_SOURCE)
    if core_program_tree is not None:
        new_errors = _core_program_transport_errors(core_program_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(CORE_PROGRAM_SOURCE)
    action_streaming_tree = parsed.get(ACTION_STREAMING_SOURCE)
    if action_streaming_tree is not None:
        new_errors = _action_streaming_errors(action_streaming_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(ACTION_STREAMING_SOURCE)
    action_reduction_tree = parsed.get(ACTION_REDUCTION_SOURCE)
    if action_reduction_tree is not None:
        new_errors = _hard_max_streaming_reduction_errors(action_reduction_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(ACTION_REDUCTION_SOURCE)
    collective_action_reduction_tree = parsed.get(COLLECTIVE_ACTION_REDUCTION_SOURCE)
    if collective_action_reduction_tree is not None:
        new_errors = _collective_hard_max_streaming_reduction_errors(
            collective_action_reduction_tree
        )
        errors.extend(new_errors)
        if new_errors:
            offending.add(COLLECTIVE_ACTION_REDUCTION_SOURCE)
    logsumexp_action_reduction_tree = parsed.get(LOGSUMEXP_ACTION_REDUCTION_SOURCE)
    if logsumexp_action_reduction_tree is not None:
        new_errors = _logsumexp_streaming_reduction_errors(
            logsumexp_action_reduction_tree
        )
        errors.extend(new_errors)
        if new_errors:
            offending.add(LOGSUMEXP_ACTION_REDUCTION_SOURCE)
    output_layout_tree = parsed.get(OUTPUT_LAYOUT_SOURCE)
    if output_layout_tree is not None:
        new_errors = _output_layout_errors(output_layout_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(OUTPUT_LAYOUT_SOURCE)
    processing_tree = parsed.get(PROCESSING_SOURCE)
    if processing_tree is not None:
        new_errors = _processing_caller_errors(processing_tree)
        new_errors += _terminal_output_wrapper_errors(processing_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(PROCESSING_SOURCE)
    backward_tree = parsed.get(BACKWARD_INDUCTION_SOURCE)
    if backward_tree is not None:
        new_errors = _backward_output_layout_errors(backward_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(BACKWARD_INDUCTION_SOURCE)
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
            "singleton_streamed_solve": (
                "Q_and_F -> canonical C-order action blocks -> exact mergeable "
                "hard max -> VALUE-only compiled core"
            ),
            "singleton_simulate": "Q_and_F -> argmax_and_max(a=Q_arr, where=F_arr)",
            "collective_solve": (
                "Q_and_F -> trailing stakeholder split -> "
                "collective_readout(feasibility=F_arr)"
            ),
            "collective_streamed_solve": (
                "Q_and_F -> weighted C-order stakeholder blocks -> one shared "
                "household hard max -> exact stakeholder gather -> compiled "
                "(VALUE, DISSOLUTION_FLAG) core"
            ),
            "collective_simulate": (
                "Q_and_F -> trailing stakeholder split -> "
                "collective_argmax_and_readout(feasibility=F_arr)"
            ),
            "taste_shock_solve": (
                "Q_and_F -> exact feasibility mask -> continuous max -> "
                "full discrete logsum"
            ),
            "taste_shock_streamed_solve": (
                "Q_and_F -> ordered discrete-prefix hard maxima -> dynamically "
                "bound log-sum-exp -> VALUE-only compiled core"
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
    *, text: str, marker: str, insertion: str, occurrence: int
) -> str:
    start = -1
    for _ in range(occurrence):
        start = text.find(marker, start + 1)
        if start < 0:
            raise ValueError(
                f"marker not found for occurrence {occurrence}: {marker!r}"
            )
    return text[:start] + insertion + text[start:]


def _replace_nth(*, text: str, marker: str, replacement: str, occurrence: int) -> str:
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
        "            return argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)"
    )
    collective_marker = "                action_axes = tuple(range(F_arr.ndim))"

    mutations["singleton_solve:q_order"] = _insert_before_nth(
        text=source,
        marker=solve_singleton,
        insertion="            Q_flat = Q_arr.reshape(-1)\n"
        "            order_filter = Q_flat[0] > Q_flat[1]\n"
        "            F_arr = jnp.where(\n"
        "                order_filter,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=1,
    )
    mutations["singleton_simulate:mt9_rank_permutation"] = _insert_before_nth(
        text=source,
        marker=simulate_singleton,
        insertion="            Q_flat = Q_arr.reshape(-1)\n"
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
        text=source,
        marker=simulate_singleton,
        insertion="            gap_filter = (\n"
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
        text=source,
        marker=collective_marker,
        insertion="                support_filter = jnp.sum(F_arr) > 1\n"
        "                F_arr = jnp.where(\n"
        "                    support_filter,\n"
        "                    F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                    F_arr,\n"
        "                )\n",
        occurrence=1,
    )
    mutations["collective_simulate:shape_axis"] = _insert_before_nth(
        text=source,
        marker=collective_marker,
        insertion="                shape_filter = (F_arr.ndim == 2) & (F_arr.shape[-1] > 1)\n"
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
        "return argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)",
        "return argmax_and_max(\n"
        "                a=Q_arr.reshape(-1)[::-1], where=F_arr, initial=-jnp.inf\n"
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
                text=source, marker=marker, insertion=insertion, occurrence=occurrence
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
                text=source,
                marker=taste_mask,
                insertion=insertion,
                occurrence=occurrence,
            )
        for index in range(6):
            mutations[f"{route}:candidate_index_{index}"] = _insert_before_nth(
                text=source,
                marker=taste_mask,
                insertion=f"            F_arr = F_arr.reshape(-1).at[{index}]"
                ".set(False).reshape(F_arr.shape)\n",
                occurrence=occurrence,
            )

    mutations["taste_shock_simulate:mt10_rank_permutation"] = _insert_before_nth(
        text=source,
        marker=taste_mask,
        insertion="            Q_flat_attack = Q_arr.reshape(-1)\n"
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
        text=source,
        marker=taste_mask,
        replacement="            Q_masked = jnp.where(\n"
        "                F_arr, Q_arr.reshape(-1)[::-1].reshape(Q_arr.shape), -jnp.inf\n"
        "            )",
        occurrence=1,
    )
    mutations["taste_shock_solve:inline_f_transform"] = _replace_nth(
        text=source,
        marker=taste_mask,
        replacement="            Q_masked = jnp.where(\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                Q_arr,\n"
        "                -jnp.inf,\n"
        "            )",
        occurrence=1,
    )
    mutations["taste_shock_simulate:inline_q_transform"] = _replace_nth(
        text=source,
        marker=taste_mask,
        replacement="            Q_masked = jnp.where(\n"
        "                F_arr, Q_arr.reshape(-1)[::-1].reshape(Q_arr.shape), -jnp.inf\n"
        "            )",
        occurrence=2,
    )
    mutations["taste_shock_simulate:inline_f_transform"] = _replace_nth(
        text=source,
        marker=taste_mask,
        replacement="            Q_masked = jnp.where(\n"
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
        text=source,
        marker="    if has_taste_shocks:",
        insertion="    n_discrete_action_axes = n_discrete_action_axes - 1\n",
        occurrence=1,
    )
    mutations["taste_shock_simulate:captured_axis_rebinding"] = _insert_before_nth(
        text=source,
        marker="    if has_taste_shocks:",
        insertion="    n_discrete_action_axes = n_discrete_action_axes - 1\n",
        occurrence=3,
    )
    mutations["solve:action_names_rebinding"] = _insert_before_nth(
        text=source,
        marker="    Q_and_F = productmap(",
        insertion="    action_names = action_names[:-1]\n",
        occurrence=1,
    )
    mutations["simulate:action_names_rebinding"] = _insert_before_nth(
        text=source,
        marker="    Q_and_F = productmap(",
        insertion="    action_names = action_names[:-1]\n",
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
        text=source,
        marker="    n_discrete_action_axes: int = 0,",
        replacement="    n_discrete_action_axes: int = 1,",
        occurrence=1,
    )
    mutations["simulate:builder_default_changed"] = _replace_nth(
        text=source,
        marker="    n_discrete_action_axes: int = 0,",
        replacement="    n_discrete_action_axes: int = 1,",
        occurrence=2,
    )
    mutations["taste_shock_solve:attribute_with_signature"] = _replace_nth(
        text=source,
        marker="        @with_signature(",
        replacement="        @candidate_filter.with_signature(",
        occurrence=1,
    )
    mutations["taste_shock_simulate:attribute_with_signature"] = _replace_nth(
        text=source,
        marker="        @with_signature(",
        replacement="        @candidate_filter.with_signature(",
        occurrence=4,
    )
    mutations["taste_shock_solve:attribute_q_and_f"] = _replace_nth(
        text=source,
        marker="            Q_arr, F_arr = Q_and_F(",
        replacement="            Q_arr, F_arr = candidate_filter.Q_and_F(",
        occurrence=1,
    )
    mutations["taste_shock_simulate:attribute_q_and_f"] = _replace_nth(
        text=source,
        marker="            Q_arr, F_arr = Q_and_F(",
        replacement="            Q_arr, F_arr = candidate_filter.Q_and_F(",
        occurrence=3,
    )
    mutations["singleton_simulate:attribute_argmax_and_max"] = source.replace(
        "return argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)",
        "return candidate_filter.argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)",
        1,
    )
    mutations["collective_solve:attribute_collective_readout"] = _replace_nth(
        text=source,
        marker="collective_readout(",
        replacement="candidate_filter.collective_readout(",
        occurrence=1,
    )
    mutations["collective_simulate:attribute_collective_argmax"] = _replace_nth(
        text=source,
        marker="collective_argmax_and_readout(",
        replacement="candidate_filter.collective_argmax_and_readout(",
        occurrence=1,
    )
    mutations["solve:attribute_productmap"] = _replace_nth(
        text=source,
        marker="    Q_and_F = productmap(",
        replacement="    Q_and_F = candidate_filter.productmap(",
        occurrence=1,
    )
    mutations["simulate:attribute_productmap"] = _replace_nth(
        text=source,
        marker="    Q_and_F = productmap(",
        replacement="    Q_and_F = candidate_filter.productmap(",
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
    core_program_source = (root / CORE_PROGRAM_SOURCE).read_text(encoding="utf-8")
    output_layout_source = (root / OUTPUT_LAYOUT_SOURCE).read_text(encoding="utf-8")
    action_streaming_source = (root / ACTION_STREAMING_SOURCE).read_text(
        encoding="utf-8"
    )
    action_reduction_source = (root / ACTION_REDUCTION_SOURCE).read_text(
        encoding="utf-8"
    )
    collective_action_reduction_source = (
        root / COLLECTIVE_ACTION_REDUCTION_SOURCE
    ).read_text(encoding="utf-8")
    logsumexp_action_reduction_source = (
        root / LOGSUMEXP_ACTION_REDUCTION_SOURCE
    ).read_text(encoding="utf-8")
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

    def replace_once(*, source: str, old: str, new: str, label: str) -> str:
        if source.count(old) != 1:
            raise ValueError(
                f"{label}: expected one mutation marker, found {source.count(old)}"
            )
        return source.replace(old, new, 1)

    grid_cases = {
        "caller_solve:action_names_slice": replace_once(
            source=grid_source,
            old="                    action_names=context.state_action_space.action_names,",
            new="                    action_names=context.state_action_space.action_names[:-1],",
            label="solve caller action names",
        ),
        "caller_solve:wrong_discrete_axis_count": replace_once(
            source=grid_source,
            old="                    n_discrete_action_axes=len(\n"
            "                        context.state_action_space.discrete_actions\n"
            "                    ),",
            new="                    n_discrete_action_axes=max(\n"
            "                        0, len(context.state_action_space.discrete_actions) - 1\n"
            "                    ),",
            label="solve caller axis count",
        ),
        "caller_solve:taste_flag_disabled": _replace_nth(
            text=grid_source,
            marker="                    has_taste_shocks=context.has_taste_shocks,",
            replacement="                    has_taste_shocks=False,",
            occurrence=1,
        ),
        "caller_solve:published_empty_mapping": replace_once(
            source=grid_source,
            old="        return SolutionKernels(period_kernels=MappingProxyType(result))",
            new="        return SolutionKernels(period_kernels=MappingProxyType({}))",
            label="solve caller publication",
        ),
        "caller_solve:raw_core_uses_wrapped_core": replace_once(
            source=grid_source,
            old="                unwrapped[q_id] = func",
            new="                unwrapped[q_id] = built[q_id]",
            label="solve caller raw core identity",
        ),
        "caller_solve:layout_delegates_wrapped_core": replace_once(
            source=grid_source,
            old="        return self.unwrapped_core",
            new="        return self.core",
            label="solve caller layout core delegation",
        ),
        "caller_solve:collective_output_role_dropped": replace_once(
            source=grid_source,
            old="        return (VALUE, DISSOLUTION_FLAG) if self.collective else VALUE",
            new="        return VALUE",
            label="solve caller output roles",
        ),
        "caller_solve:fixed_raw_core_filtered": replace_once(
            source=grid_source,
            old="                else functools.partial(self.unwrapped_core, **regime_fixed)",
            new="                else candidate_filter(\n"
            "                    functools.partial(self.unwrapped_core, **regime_fixed)\n"
            "                )",
            label="solve caller fixed raw core",
        ),
        "caller_solve:published_value_filtered": replace_once(
            source=grid_source,
            old="        return KernelResult(V_arr=out)",
            new="        return KernelResult(V_arr=candidate_filter(out))",
            label="solve caller value publication",
        ),
        "streaming_provider:dense_function": replace_once(
            source=grid_source,
            old="            function=self.streamed_core,",
            new="            function=self.unwrapped_core,",
            label="streamed provider function",
        ),
        "streaming_provider:arguments_filtered": replace_once(
            source=grid_source,
            old="            arguments=arguments,",
            new="            arguments=dict(tuple(arguments.items())[:-1]),",
            label="streamed provider arguments",
        ),
        "streaming_provider:action_names_slice": replace_once(
            source=grid_source,
            old="                        coordinate_names=self.action_names,",
            new="                        coordinate_names=self.action_names[:-1],",
            label="streamed provider action names",
        ),
        "streaming_provider:action_extents_slice": replace_once(
            source=grid_source,
            old="                        coordinate_extents=self.action_extents,",
            new="                        coordinate_extents=self.action_extents[:-1],",
            label="streamed provider action extents",
        ),
        "streaming_provider:width_keyword_collision_bypassed": replace_once(
            source=grid_source,
            old=(
                "        and all(\n"
                "            _ACTION_WIDTH_KEYWORD not in inspect.signature(Q_and_F).parameters\n"
                "            for Q_and_F in context.Q_and_F_functions.values()\n"
                "        )"
            ),
            new="        and True",
            label="streamed reserved width-keyword collision guard",
        ),
        "streaming_dispatch:bypass_compiled_core": replace_once(
            source=grid_source,
            old='        out = compiled_cores["main"](',
            new="        out = self.core(",
            label="streamed compiled dispatch",
        ),
        "streaming_provider:collective_reducer_dropped": replace_once(
            source=grid_source,
            old=(
                "                            COLLECTIVE_HARD_MAX_REDUCTION\n"
                "                            if self.collective"
            ),
            new=(
                "                            HARD_MAX_REDUCTION\n"
                "                            if self.collective"
            ),
            label="streamed collective reduction contract",
        ),
        "streaming_ev1_provider:reducer_dropped": replace_once(
            source=grid_source,
            old=(
                "                            else GridSearchEV1ActionReduction(\n"
                "                                n_discrete_action_axes=self.n_discrete_action_axes\n"
                "                            )\n"
                "                            if self.has_taste_shocks"
            ),
            new=(
                "                            else HARD_MAX_REDUCTION\n"
                "                            if self.has_taste_shocks"
            ),
            label="streamed EV1 composite reduction contract",
        ),
        "streaming_ev1_provider:discrete_axis_count_changed": replace_once(
            source=grid_source,
            old="                                n_discrete_action_axes=self.n_discrete_action_axes",
            new=(
                "                                n_discrete_action_axes=max(\n"
                "                                    1, self.n_discrete_action_axes - 1\n"
                "                                )"
            ),
            label="streamed EV1 discrete-prefix width",
        ),
        "streaming_provider:collective_output_role_dropped": replace_once(
            source=grid_source,
            old="            output_roles=((VALUE, DISSOLUTION_FLAG) if self.collective else VALUE),",
            new="            output_roles=VALUE,",
            label="streamed collective program output roles",
        ),
        "streaming_collective:published_dissolution_inverted": replace_once(
            source=grid_source,
            old="            return KernelResult(V_arr=V_arr, dissolution=dissolution)",
            new="            return KernelResult(V_arr=V_arr, dissolution=~dissolution)",
            label="streamed collective result publication",
        ),
    }
    specs.update(
        {
            name: {"path": GRID_SEARCH_SOURCE, "source": mutated}
            for name, mutated in grid_cases.items()
        }
    )

    specs["streaming_collective:builder_dissolution_inverted"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old="            ~collective_result.any_feasible,",
            new="            collective_result.any_feasible,",
            label="streamed collective dissolution output",
        ),
    }
    specs["streaming_collective:builder_stakeholder_axis_sliced"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old="            collective_result.best_stakeholder_values,",
            new="            collective_result.best_stakeholder_values[..., :-1],",
            label="streamed collective stakeholder output",
        ),
    }
    specs["streaming_ev1:runtime_scale_constant"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old=(
                "                    states_actions_params[TASTE_SHOCK_SCALE_PARAM],\n"
                "                ),\n"
                "            )\n"
                "            ev1_result = ev1_cell("
            ),
            new=(
                "                    1.0,\n"
                "                ),\n"
                "            )\n"
                "            ev1_result = ev1_cell("
            ),
            label="streamed EV1 runtime scale",
        ),
    }
    specs["streaming_ev1:scale_signature_dropped"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old="        extra_param_names.append(TASTE_SHOCK_SCALE_PARAM)",
            new="        pass",
            label="streamed EV1 scale signature",
        ),
    }
    specs["streaming_ev1:published_value_negated"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old="            return ev1_result.smoothed_value",
            new="            return -ev1_result.smoothed_value",
            label="streamed EV1 value publication",
        ),
    }

    core_program_cases = {
        "streaming_resolver:bypass_static_width_binding": replace_once(
            source=core_program_source,
            old="        static_kwargs=width_bindings,",
            new="        static_kwargs={},",
            label="streamed resolver static kwargs",
        ),
        "streaming_resolver:arguments_filtered": replace_once(
            source=core_program_source,
            old="        arguments=program.arguments,",
            new="        arguments=dict(tuple(program.arguments.items())[:-1]),",
            label="streamed resolver arguments",
        ),
        "streaming_resolver:specialization_drops_axes": replace_once(
            source=core_program_source,
            old="            tuple(compilation_axes),",
            new="            (),",
            label="streamed resolver specialization",
        ),
        "streaming_resolver:output_roles_dropped": replace_once(
            source=core_program_source,
            old="        output_roles=program.output_roles,",
            new="        output_roles=None,",
            label="streamed resolver output roles",
        ),
    }
    specs.update(
        {
            name: {"path": CORE_PROGRAM_SOURCE, "source": mutated}
            for name, mutated in core_program_cases.items()
        }
    )

    action_streaming_cases = {
        "streaming_blocks:skip_last_block": _replace_nth(
            text=action_streaming_source,
            marker="            n_remaining=n_blocks - 1,",
            replacement="            n_remaining=n_blocks - 2,",
            occurrence=1,
        ),
        "streaming_blocks:admit_padded_tail": replace_once(
            source=action_streaming_source,
            old="    return values, feasible & valid, global_ids",
            new="    return values, feasible, global_ids",
            label="streamed padded-tail mask",
        ),
        "streaming_blocks:block_local_action_ids": _replace_nth(
            text=action_streaming_source,
            marker="    global_ids = block_start + safe_offsets",
            replacement="    global_ids = safe_offsets",
            occurrence=1,
        ),
        "streaming_blocks:reverse_coordinate_decode": replace_once(
            source=action_streaming_source,
            old="    for name, grid, size in zip(action_names, action_grids, action_sizes, strict=True):",
            new=(
                "    for name, grid, size in zip(reversed(action_names), "
                "reversed(action_grids), reversed(action_sizes), strict=True):"
            ),
            label="streamed C-order coordinate decode",
        ),
        "streaming_ev1:skip_last_block": _replace_nth(
            text=action_streaming_source,
            marker="            n_remaining=n_blocks - 1,",
            replacement="            n_remaining=n_blocks - 2,",
            occurrence=2,
        ),
        "streaming_collective_blocks:skip_last_block": _replace_nth(
            text=action_streaming_source,
            marker="            n_remaining=n_blocks - 1,",
            replacement="            n_remaining=n_blocks - 2,",
            occurrence=3,
        ),
        "streaming_collective_blocks:admit_padded_tail": replace_once(
            source=action_streaming_source,
            old=(
                "    return objectives, stakeholder_values, feasible & valid, "
                "global_ids"
            ),
            new="    return objectives, stakeholder_values, feasible, global_ids",
            label="streamed collective padded-tail mask",
        ),
        "streaming_collective_blocks:block_local_action_ids": _replace_nth(
            text=action_streaming_source,
            marker="    global_ids = block_start + safe_offsets",
            replacement="    global_ids = safe_offsets",
            occurrence=2,
        ),
        "streaming_ev1:composite_version_changed": replace_once(
            source=action_streaming_source,
            old='            "grid-search-ev1-action-reduction",\n            1,',
            new='            "grid-search-ev1-action-reduction",\n            2,',
            label="streamed EV1 composite semantic identity",
        ),
        "streaming_ev1:hard_max_semantic_key_dropped": replace_once(
            source=action_streaming_source,
            old="            HARD_MAX_REDUCTION.semantic_key,",
            new='            ("hard-max", 0),',
            label="streamed EV1 hard-max semantic identity",
        ),
        "streaming_ev1:logsum_semantic_key_dropped": replace_once(
            source=action_streaming_source,
            old="            LOGSUMEXP_REDUCTION.semantic_key,",
            new='            ("logsumexp", 0),',
            label="streamed EV1 log-sum-exp semantic identity",
        ),
        "streaming_ev1:scale_rebound": replace_once(
            source=action_streaming_source,
            old="        reduction = LOGSUMEXP_REDUCTION.bind(scale=jnp.asarray(self.scale))",
            new="        reduction = LOGSUMEXP_REDUCTION.bind(scale=jnp.asarray(1.0))",
            label="streamed EV1 one-session scale binding",
        ),
        "streaming_ev1:continuous_extent_changed": replace_once(
            source=action_streaming_source,
            old="        continuous_extent = math.prod(action_sizes[self.n_discrete_action_axes :])",
            new="        continuous_extent = 1",
            label="streamed EV1 discrete-prefix branch extent",
        ),
        "streaming_ev1:admit_padded_tail": replace_once(
            source=action_streaming_source,
            old="    valid = jnp.arange(block_width, dtype=jnp.int32) < remaining",
            new="    valid = jnp.ones(block_width, dtype=bool)",
            label="streamed EV1 padded tail",
        ),
        "streaming_ev1:branch_identity_shifted": replace_once(
            source=action_streaming_source,
            old="    branch_id = global_id // continuous_extent",
            new="    branch_id = (global_id + 1) // continuous_extent",
            label="streamed EV1 ordered branch identity",
        ),
        "streaming_ev1:branch_transition_ignored": replace_once(
            source=action_streaming_source,
            old="    branch_changed = (accumulator.active_branch_id >= 0) & (",
            new="    branch_changed = jnp.asarray(False) & (",
            label="streamed EV1 branch transition",
        ),
        "streaming_ev1:branch_value_negated": replace_once(
            source=action_streaming_source,
            old="        values=branch.best_value[jnp.newaxis],",
            new="        values=(-branch.best_value)[jnp.newaxis],",
            label="streamed EV1 finalized branch value",
        ),
        "streaming_ev1:last_branch_not_flushed": replace_once(
            source=action_streaming_source,
            old=(
                "        accumulator = _flush_ev1_branch(\n"
                "            accumulator=accumulator,\n"
                "            reduction=reduction,\n"
                "        )"
            ),
            new="        accumulator = accumulator",
            label="streamed EV1 final branch flush",
        ),
        "streaming_ev1:reverse_block_order": replace_once(
            source=action_streaming_source,
            old="        (values, feasible, global_ids, valid),",
            new="        (values[::-1], feasible[::-1], global_ids[::-1], valid[::-1]),",
            label="streamed EV1 within-block order",
        ),
        "streaming_collective_blocks:objective_uses_first_stakeholder": replace_once(
            source=action_streaming_source,
            old=(
                "    objectives = _weighted_sum(\n"
                "        stakeholder_Q={\n"
                "            name: stakeholder_values[..., index]\n"
                "            for index, name in enumerate(stakeholders)\n"
                "        },\n"
                "        weights=weights,\n"
                "    )"
            ),
            new="    objectives = stakeholder_values[..., 0]",
            label="streamed collective household objective",
        ),
    }
    for index in range(6):
        action_streaming_cases[f"streaming_blocks:candidate_index_{index}"] = (
            replace_once(
                source=action_streaming_source,
                old="    return values, feasible & valid, global_ids",
                new=(
                    "    feasible = feasible & (global_ids != "
                    f"{index})\n    return values, feasible & valid, global_ids"
                ),
                label=f"streamed candidate identity {index}",
            )
        )
        action_streaming_cases[f"streaming_ev1:candidate_index_{index}"] = replace_once(
            source=action_streaming_source,
            old="            is_valid,",
            new=f"            is_valid & (global_id != {index}),",
            label=f"streamed EV1 candidate identity {index}",
        )
        action_streaming_cases[
            f"streaming_collective_blocks:candidate_index_{index}"
        ] = replace_once(
            source=action_streaming_source,
            old=(
                "    return objectives, stakeholder_values, feasible & valid, "
                "global_ids"
            ),
            new=(
                "    feasible = feasible & (global_ids != "
                f"{index})\n"
                "    return objectives, stakeholder_values, feasible & valid, "
                "global_ids"
            ),
            label=f"streamed collective candidate identity {index}",
        )
    specs.update(
        {
            name: {"path": ACTION_STREAMING_SOURCE, "source": mutated}
            for name, mutated in action_streaming_cases.items()
        }
    )

    action_reduction_cases = {
        "streaming_hard_max:filter_last_identity": replace_once(
            source=action_reduction_source,
            old="            feasible=jnp.broadcast_to(feasible, values.shape),",
            new=(
                "            feasible=jnp.broadcast_to(feasible, values.shape)\n"
                "            & (jnp.broadcast_to(action_ids, values.shape) != 5),"
            ),
            label="streamed hard-max feasibility",
        ),
        "streaming_hard_max:ignore_right_partial": replace_once(
            source=action_reduction_source,
            old="        choose_right = right.any_feasible & (",
            new="        choose_right = jnp.zeros_like(right.any_feasible) & (",
            label="streamed hard-max merge",
        ),
        "streaming_hard_max:signed_zero_normalization_bypassed": replace_once(
            source=action_reduction_source,
            old=(
                "            best_value=jnp.where(\n"
                "                both_feasible_zero, signed_zero_max, selected_best_value\n"
                "            ),"
            ),
            new="            best_value=selected_best_value,",
            label="streamed hard-max signed-zero numeric normalization",
        ),
        "streaming_hard_max:semantic_key_changed": replace_once(
            source=action_reduction_source,
            old='        return ("hard-max", 1)',
            new='        return ("hard-max", 2)',
            label="streamed hard-max semantic identity",
        ),
    }
    specs.update(
        {
            name: {"path": ACTION_REDUCTION_SOURCE, "source": mutated}
            for name, mutated in action_reduction_cases.items()
        }
    )

    collective_action_reduction_cases = {
        "streaming_collective_hard_max:filter_last_identity": replace_once(
            source=collective_action_reduction_source,
            old="            feasible=jnp.broadcast_to(feasible, objectives.shape),",
            new=(
                "            feasible=jnp.broadcast_to(feasible, objectives.shape)\n"
                "            & (jnp.broadcast_to(action_ids, objectives.shape) != 5),"
            ),
            label="streamed collective hard-max feasibility",
        ),
        "streaming_collective_hard_max:ignore_right_partial": replace_once(
            source=collective_action_reduction_source,
            old="        choose_right = right.any_feasible & (",
            new="        choose_right = jnp.zeros_like(right.any_feasible) & (",
            label="streamed collective hard-max merge",
        ),
        "streaming_collective_hard_max:signed_zero_normalization_bypassed": replace_once(
            source=collective_action_reduction_source,
            old=(
                "            best_objective=jnp.where(\n"
                "                both_feasible_zero,\n"
                "                signed_zero_max,\n"
                "                selected_best_objective,\n"
                "            ),"
            ),
            new="            best_objective=selected_best_objective,",
            label="streamed collective hard-max signed-zero numeric normalization",
        ),
        "streaming_collective_hard_max:semantic_key_changed": replace_once(
            source=collective_action_reduction_source,
            old='        return ("collective-hard-max", 1)',
            new='        return ("collective-hard-max", 2)',
            label="streamed collective hard-max semantic identity",
        ),
        "streaming_collective_hard_max:stakeholder_gather_decoupled": replace_once(
            source=collective_action_reduction_source,
            old="        positions=winner_position,",
            new="        positions=jnp.zeros_like(winner_position),",
            label="streamed collective shared-winner gather",
        ),
        "streaming_collective_hard_max:winner_identity_shifted": replace_once(
            source=collective_action_reduction_source,
            old=(
                "    best_global_action_id = jnp.where("
                "any_feasible_nan, 0, best_global_action_id)"
            ),
            new=(
                "    best_global_action_id = jnp.where("
                "any_feasible_nan, 0, best_global_action_id + 1)"
            ),
            label="streamed collective winner identity",
        ),
    }
    specs.update(
        {
            name: {
                "path": COLLECTIVE_ACTION_REDUCTION_SOURCE,
                "source": mutated,
            }
            for name, mutated in collective_action_reduction_cases.items()
        }
    )

    logsumexp_action_reduction_cases = {
        "streaming_logsumexp:filter_first_branch": replace_once(
            source=logsumexp_action_reduction_source,
            old="        finite = jnp.isfinite(values)",
            new="        finite = jnp.isfinite(values).at[..., 0].set(False)",
            label="streamed log-sum-exp branch admission",
        ),
        "streaming_logsumexp:ignore_right_partial": replace_once(
            source=logsumexp_action_reduction_source,
            old=(
                "                left.rescaled_sum * left_factor"
                " + right.rescaled_sum * right_factor"
            ),
            new="                left.rescaled_sum * left_factor + 0.0 * right_factor",
            label="streamed log-sum-exp merge",
        ),
        "streaming_logsumexp:scale_dropped_at_finalize": replace_once(
            source=logsumexp_action_reduction_source,
            old="        finite_result = accumulator.running_max + self.scale * jnp.log(",
            new="        finite_result = accumulator.running_max + jnp.log(",
            label="streamed log-sum-exp final scale",
        ),
        "streaming_logsumexp:semantic_key_changed": replace_once(
            source=logsumexp_action_reduction_source,
            old='        return ("logsumexp", 1)',
            new='        return ("logsumexp", 2)',
            label="streamed log-sum-exp semantic identity",
        ),
        "streaming_logsumexp:bind_negates_scale": replace_once(
            source=logsumexp_action_reduction_source,
            old="        return BoundLogSumExpReduction(scale=scale)",
            new="        return BoundLogSumExpReduction(scale=-scale)",
            label="streamed log-sum-exp dynamic binding",
        ),
    }
    specs.update(
        {
            name: {
                "path": LOGSUMEXP_ACTION_REDUCTION_SOURCE,
                "source": mutated,
            }
            for name, mutated in logsumexp_action_reduction_cases.items()
        }
    )

    output_layout_cases = {
        "output_layout:assert_then_filter": replace_once(
            source=output_layout_source,
            old="        assert_output_layout(output=output, layout=self.layout)\n"
            "        return output",
            new="        assert_output_layout(output=output, layout=self.layout)\n"
            "        return candidate_filter(output)",
            label="planned core post-assert identity",
        ),
        "output_layout:filter_before_assert": replace_once(
            source=output_layout_source,
            old="        output = self.compiled(*args, **kwargs)",
            new="        output = candidate_filter(self.compiled(*args, **kwargs))",
            label="planned core pre-assert identity",
        ),
        "output_layout:sharding_check_disabled": replace_once(
            source=output_layout_source,
            old="        if actual != expected:",
            new="        if False and actual != expected:",
            label="planned output sharding assertion",
        ),
        "output_layout:expected_value_shape_sliced": replace_once(
            source=output_layout_source,
            old="    expected_value_shape = tuple(int(size) for size in value_shape)",
            new="    expected_value_shape = tuple(int(size) for size in value_shape)[1:]",
            label="planned absolute value shape",
        ),
    }
    specs.update(
        {
            name: {"path": OUTPUT_LAYOUT_SOURCE, "source": mutated}
            for name, mutated in output_layout_cases.items()
        }
    )

    processing_cases = {
        "caller_simulate:action_names_slice": replace_once(
            source=processing_source,
            old="                action_names=state_action_space.action_names,",
            new="                action_names=state_action_space.action_names[:-1],",
            label="simulate caller action names",
        ),
        "caller_simulate:wrong_discrete_axis_count": replace_once(
            source=processing_source,
            old="                n_discrete_action_axes=len(state_action_space.discrete_actions),",
            new="                n_discrete_action_axes=max(\n"
            "                    0, len(state_action_space.discrete_actions) - 1\n"
            "                ),",
            label="simulate caller axis count",
        ),
        "caller_simulate:taste_flag_disabled": replace_once(
            source=processing_source,
            old="                n_discrete_action_axes=len(state_action_space.discrete_actions),\n"
            "                has_taste_shocks=has_taste_shocks,",
            new="                n_discrete_action_axes=len(state_action_space.discrete_actions),\n"
            "                has_taste_shocks=False,",
            label="simulate caller taste flag",
        ),
        "caller_simulate:live_taste_flag_rebinding": replace_once(
            source=processing_source,
            old="    argmax_and_max_Q_over_a = _build_argmax_and_max_Q_over_a_per_period(",
            new="    has_taste_shocks = False\n\n"
            "    argmax_and_max_Q_over_a = _build_argmax_and_max_Q_over_a_per_period(",
            label="simulate live taste rebinding",
        ),
        "caller_simulate:published_empty_mapping": replace_once(
            source=processing_source,
            old="        argmax_and_max_Q_over_a=argmax_and_max_Q_over_a,",
            new="        argmax_and_max_Q_over_a=MappingProxyType({}),",
            label="simulate caller publication",
        ),
        "caller_simulate:attribute_simulation_phase": replace_once(
            source=processing_source,
            old="    return SimulationPhase(",
            new="    return candidate_filter.SimulationPhase(",
            label="simulate caller publication callee",
        ),
        "terminal_wrapper:core_program_dropped": replace_once(
            source=processing_source,
            old=(
                "        return self.base.build_core_program(\n"
                "            core_key=core_key,\n"
                "            arguments=arguments,\n"
                "        )"
            ),
            new="        return None",
            label="terminal wrapper core-program delegation",
        ),
        "terminal_wrapper:output_roles_dropped": replace_once(
            source=processing_source,
            old="        return self.base.output_roles(core_key=core_key)",
            new="        return None",
            label="terminal wrapper output roles",
        ),
        "terminal_wrapper:layout_uses_wrapped_core": replace_once(
            source=processing_source,
            old="        return self.base.core_for_output_layout(core_key=core_key)",
            new='        return self.base.cores()["main"]',
            label="terminal wrapper layout delegation",
        ),
        "terminal_wrapper:published_value_filtered": replace_once(
            source=processing_source,
            old="            V_arr=result.V_arr,\n            continuation=carry,",
            new="            V_arr=candidate_filter(result.V_arr),\n            continuation=carry,",
            label="terminal wrapper value publication",
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
            source=argmax_source,
            old="    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            new="    if a.reshape(-1)[0] > a.reshape(-1)[1]:\n"
            "        return jnp.array(1, dtype=jnp.int32), a.reshape(-1)[1]\n"
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            label="argmax q-order",
        ),
        "shared_argmax:support_filter": replace_once(
            source=argmax_source,
            old="    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            new="    where = jnp.where(\n"
            "        jnp.sum(where) > 1,\n"
            "        where.reshape(-1).at[0].set(False).reshape(where.shape),\n"
            "        where,\n"
            "    )\n"
            "    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)",
            label="argmax support",
        ),
        "shared_argmax:axis_prefix": replace_once(
            source=argmax_source,
            old="        axis = tuple(range(a.ndim))",
            new="        axis = tuple(range(a.ndim - 1))",
            label="argmax axis prefix",
        ),
        "shared_argmax:axis_reorder": replace_once(
            source=argmax_source,
            old="    return a.transpose((*front_axes, *axes))",
            new="    return a.transpose((*front_axes, *reversed(axes)))",
            label="argmax axis reorder",
        ),
        "shared_argmax:flatten_drop_last": replace_once(
            source=argmax_source,
            old="    return a.reshape(*a.shape[:-n], -1)",
            new="    return a[..., :-1].reshape(*a.shape[:-n], -1)",
            label="argmax flatten drop",
        ),
        "shared_argmax:range_module_shadow": replace_once(
            source=argmax_source,
            old="from lcm.typing import BoolND, FloatND, IntND\n",
            new="from lcm.typing import BoolND, FloatND, IntND\n\n"
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
            source=collective_source,
            old="    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)",
            new="    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)\n"
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
            source=collective_source,
            old="        a=objective, axis=action_axes, initial=-jnp.inf, where=feasibility",
            new="        a=objective, axis=action_axes, initial=-jnp.inf,\n"
            "        where=feasibility.reshape(-1).at[0].set(False).reshape(\n"
            "            feasibility.shape\n"
            "        )",
            label="collective feasibility inline",
        ),
        "shared_collective:action_axis_prefix": replace_once(
            source=collective_source,
            old="        a=objective, axis=action_axes, initial=-jnp.inf, where=feasibility",
            new="        a=objective, axis=action_axes[:-1], initial=-jnp.inf, where=feasibility",
            label="collective action axis",
        ),
        "shared_collective:gather_next_candidate": replace_once(
            source=collective_source,
            old="    gathered = jnp.take_along_axis(q_flat, argmax_flat[..., None], axis=-1)",
            new="    gathered = jnp.take_along_axis(q_flat, (argmax_flat + 1)[..., None], axis=-1)",
            label="collective gather",
        ),
        "shared_collective:early_candidate_return": replace_once(
            source=collective_source,
            old="    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)",
            new="    if jnp.all(feasibility):\n"
            "        return (\n"
            "            jnp.array(1, dtype=jnp.int32),\n"
            "            {name: q.reshape(-1)[1] for name, q in stakeholder_Q.items()},\n"
            "            jnp.array(False),\n"
            "        )\n"
            "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)",
            label="collective early return",
        ),
        "shared_collective:argmax_module_shadow": replace_once(
            source=collective_source,
            old="    argmax_and_max,\n)\n",
            new="    argmax_and_max,\n)\n\nargmax_and_max = candidate_filter\n",
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
            source=logsum_source,
            old="    v_max = jnp.max(values, axis=axes, keepdims=True)",
            new="    gap_filter = values.reshape(-1)[0] - values.reshape(-1)[1] > 0.5\n"
            "    values = jnp.where(\n"
            "        gap_filter,\n"
            "        values.reshape(-1).at[0].set(-jnp.inf).reshape(values.shape),\n"
            "        values,\n"
            "    )\n"
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            label="logsum q-gap",
        ),
        "shared_logsum:support_filter": replace_once(
            source=logsum_source,
            old="    v_max = jnp.max(values, axis=axes, keepdims=True)",
            new="    support_filter = jnp.sum(~jnp.isneginf(values)) > 1\n"
            "    values = jnp.where(\n"
            "        support_filter,\n"
            "        values.reshape(-1).at[0].set(-jnp.inf).reshape(values.shape),\n"
            "        values,\n"
            "    )\n"
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            label="logsum support",
        ),
        "shared_logsum:axis_prefix": replace_once(
            source=logsum_source,
            old="    v_max = jnp.max(values, axis=axes, keepdims=True)",
            new="    v_max = jnp.max(values, axis=axes[:-1], keepdims=True)",
            label="logsum axis prefix",
        ),
        "shared_logsum:value_slice": replace_once(
            source=logsum_source,
            old="        shifted, axis=axes",
            new="        shifted[..., 1:], axis=axes",
            label="logsum value slice",
        ),
        "shared_logsum:rank_early_return": replace_once(
            source=logsum_source,
            old="    v_max = jnp.max(values, axis=axes, keepdims=True)",
            new="    if values.reshape(-1)[0] > values.reshape(-1)[1]:\n"
            "        return values.reshape(-1)[1], jnp.zeros_like(values)\n"
            "    v_max = jnp.max(values, axis=axes, keepdims=True)",
            label="logsum early return",
        ),
        "shared_logsum:softmax_slice": replace_once(
            source=logsum_source,
            old="jax.nn.softmax(shifted, axis=axes)",
            new="jax.nn.softmax(shifted[..., 1:], axis=axes)",
            label="logsum softmax slice",
        ),
        "shared_logsum:import_rebinding": replace_once(
            source=logsum_source,
            old="from jax.scipy.special import logsumexp",
            new="from jax.scipy.special import logsumexp\n\nlogsumexp = jnp.max",
            label="logsum import rebinding",
        ),
        "shared_logsum:wrong_euler_gamma": replace_once(
            source=logsum_source,
            old="EULER_GAMMA = 0.5772156649015329",
            new="EULER_GAMMA = 0.0",
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
                source=dispatchers_source,
                old="        product_axes=variables,",
                new="        product_axes=variables[:-1],",
                label="productmap action-axis drop",
            ),
        },
        "shared_functools:drop_last_argument": {
            "path": FUNCTOOLS_SOURCE,
            "source": replace_once(
                source=functools_source,
                old="        for name, value in bound.arguments.items():",
                new="        for name, value in list(bound.arguments.items())[:-1]:",
                label="allow-args argument drop",
            ),
        },
        "shared_containers:duplicate_threshold": {
            "path": CONTAINERS_SOURCE,
            "source": replace_once(
                source=containers_source,
                old="return {v for v, count in counts.items() if count > 1}",
                new="return {v for v, count in counts.items() if count > 2}",
                label="duplicate threshold",
            ),
        },
        "shared_zero_safe:ordered_sum_slice": {
            "path": ZERO_SAFE_SOURCE,
            "source": replace_once(
                source=zero_safe_source,
                old="return jnp.sum(jnp.sort(arr, axis=axis), axis=axis)",
                new="return jnp.sum(jnp.sort(arr, axis=axis)[1:], axis=axis)",
                label="ordered scalarization slice",
            ),
        },
        "shared_probability:unbalanced_product": {
            "path": PROBABILITY_SOURCE,
            "source": replace_once(
                source=probability_source,
                old="return _balanced_with_tangent(jnp.asarray(weight), jnp.asarray(value))",
                new="return jnp.asarray(weight) * jnp.asarray(value)",
                label="zero-safe balanced product",
            ),
        },
        "candidate_materialization:grid_base_intercepts_to_jax": {
            "path": GRID_BASE_SOURCE,
            "source": replace_once(
                source=grid_base_source,
                old='class Grid(ABC):\n    """LCM Grid base class."""',
                new='class Grid(ABC):\n    """LCM Grid base class."""\n\n    def __getattribute__(self, name):\n        value = super().__getattribute__(name)\n        if name == "to_jax":\n            return lambda: value()[:-1]\n        return value',
                label="inherited grid coordinate interception",
            ),
        },
        "simulation_state_action_space:drops_inherited_candidates": {
            "path": SIMULATION_TRANSITIONS_SOURCE,
            "source": replace_once(
                source=simulation_transitions_source,
                old="    return base.replace(states=MappingProxyType(states_for_state_action_space))",
                new="    return base.replace(\n        states=MappingProxyType(states_for_state_action_space),\n        discrete_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base.discrete_actions.items()}),\n        continuous_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base.continuous_actions.items()}),\n    )",
                label="simulation base-action preservation",
            ),
        },
        "simulation_state_action_space:caller_drops_inherited_candidates": {
            "path": SIMULATION_SOURCE,
            "source": replace_once(
                source=simulation_source,
                old="        base=base_state_action_space,",
                new="        base=base_state_action_space.replace(continuous_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base_state_action_space.continuous_actions.items()})),",
                label="simulation adapter caller base wrapping",
            ),
        },
        "shared_engine:action_order_reversed": {
            "path": ENGINE_SOURCE,
            "source": replace_once(
                source=engine_source,
                old="return tuple(self.discrete_actions) + tuple(self.continuous_actions)",
                new="return tuple(self.continuous_actions) + tuple(self.discrete_actions)",
                label="state-action metadata order",
            ),
        },
        "shared_engine:actions_drop_last_candidate": {
            "path": ENGINE_SOURCE,
            "source": replace_once(
                source=engine_source,
                old="            dict(self.discrete_actions) | dict(self.continuous_actions)",
                new="            {name: values.at[-1].set(values[0]) for name, values in self.discrete_actions.items()} | {name: values.at[-1].set(values[0]) for name, values in self.continuous_actions.items()}",
                label="combined action mapping candidate omission",
            ),
        },
        "shared_engine:replace_drops_inherited_candidates": {
            "path": ENGINE_SOURCE,
            "source": replace_once(
                source=engine_source,
                old="        discrete_actions = first_non_none(discrete_actions, self.discrete_actions)",
                new="        discrete_actions = first_non_none(discrete_actions, MappingProxyType({name: values.at[-1].set(values[0]) for name, values in self.discrete_actions.items()}))",
                label="StateActionSpace.replace inherited candidate omission",
            ),
        },
        "shared_state_action_space:continuous_order_reversed": {
            "path": STATE_ACTION_SPACE_SOURCE,
            "source": replace_once(
                source=state_action_space_source,
                old="        for name in variables.continuous_action_names",
                new="        for name in reversed(variables.continuous_action_names)",
                label="continuous candidate order",
            ),
        },
        "shared_state_action_space:drop_last_continuous_candidate": {
            "path": STATE_ACTION_SPACE_SOURCE,
            "source": replace_once(
                source=state_action_space_source,
                old="        name: _grid_to_jax_or_placeholder(grids[name])\n        for name in variables.continuous_action_names",
                new="        name: _grid_to_jax_or_placeholder(grids[name]).at[-1].set(_grid_to_jax_or_placeholder(grids[name])[0])\n        for name in variables.continuous_action_names",
                label="state-action continuous candidate omission",
            ),
        },
        "simulation_index_consumer:next_candidate": {
            "path": SIMULATION_SOURCE,
            "source": replace_once(
                source=simulation_source,
                old="            flat_indices=indices_optimal_actions,",
                new="            flat_indices=indices_optimal_actions + 1,",
                label="published simulation index consumer",
            ),
        },
        "aot_compile:argmax_index_shift": {
            "path": SIMULATION_COMPILE_SOURCE,
            "source": replace_once(
                source=simulation_compile_source,
                old="            argmax_func = sf.argmax_and_max_Q_over_a[period]",
                new="            argmax_func = sf.argmax_and_max_Q_over_a[period]\n"
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
                source=model_source,
                old="            return self._simulate_compile_cache[compile_batch_size]",
                new="            return candidate_filter(\n"
                "                self._simulate_compile_cache[compile_batch_size]\n"
                "            )",
                label="public Model AOT regime selection",
            ),
        },
        "backward_layout:planned_uses_wrapped_core": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="        if program is None and layout is not UNPLANNED:",
                new="        if layout is not UNPLANNED:",
                label="planned raw lowering callable",
            ),
        },
        "streaming_aot:provider_ignored": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            if isinstance(kernel, CoreProgramAware)",
                new="            if False and isinstance(kernel, CoreProgramAware)",
                label="streamed program provider dispatch",
            ),
        },
        "streaming_aot:validation_before_width_selection_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="    _validate_core_program(program=program)",
                new="    program = program",
                label="streamed validation before initial width selection",
            ),
        },
        "streaming_aot:resolved_function_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            lowering_func = resolved.function",
                new="            lowering_func = func",
                label="streamed resolved lowering function",
            ),
        },
        "streaming_aot:resolved_arguments_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            lowering_args = resolved.arguments",
                new="            lowering_args = arguments",
                label="streamed resolved lowering arguments",
            ),
        },
        "streaming_aot:specialization_dropped": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            specialization_key = resolved.specialization_key",
                new="            specialization_key = None",
                label="streamed lowering specialization",
            ),
        },
        "streaming_aot:lower_dense_function": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="    for triple, func in lowering_functions.items():",
                new="    for triple, func in all_functions.items():",
                label="streamed function selected for lowering",
            ),
        },
        "backward_layout:out_shardings_disabled": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old='                out_shardings=cast("ResolvedOutputLayout", layout).out_shardings,',
                new="                out_shardings=None,",
                label="planned JIT output sharding",
            ),
        },
        "backward_layout:planned_tag_dropped": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="    return PlannedCore(\n"
                "        compiled=compiled,\n"
                '        layout=cast("ResolvedOutputLayout", layout),\n'
                "    )",
                new="    return compiled",
                label="planned core attachment",
            ),
        },
        "backward_layout:publish_after_assert_filtered": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="        assert_output_layout(\n"
                "            output=(value, dissolution) if dissolution is not None else value,\n"
                "            layout=layout,\n"
                "        )\n"
                "        return value",
                new="        assert_output_layout(\n"
                "            output=(value, dissolution) if dissolution is not None else value,\n"
                "            layout=layout,\n"
                "        )\n"
                "        return candidate_filter(value)",
                label="planned publication identity",
            ),
        },
        "backward_layout:solve_publication_filtered": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            period_solution[regime_name] = V_arr",
                new="            period_solution[regime_name] = candidate_filter(V_arr)",
                label="solve-loop value publication",
            ),
        },
        "shared_dedup_key:collapse_plain_callables": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="    return id(func)",
                new="    return 0",
                label="plain-callable dedup identity",
            ),
        },
        "simulation_publication:shift_padded_actions": {
            "path": INITIAL_CONDITIONS_SOURCE,
            "source": replace_once(
                source=initial_conditions_source,
                old="                        {k: v[:original_n_subjects] for k, v in value.items()}",
                new="                        {k: (v[1 : original_n_subjects + 1] if name == 'actions' else v[:original_n_subjects]) for k, v in value.items()}",
                label="padded action row shift",
            ),
        },
        "simulation_result:shift_raw_actions": {
            "path": RESULT_SOURCE,
            "source": replace_once(
                source=result_source,
                old="        self._raw_results = raw_results",
                new="        self._raw_results = MappingProxyType({regime: MappingProxyType({period: __import__('dataclasses').replace(data, actions=MappingProxyType({name: jnp.roll(values, 1) for name, values in data.actions.items()})) for period, data in periods.items()}) for regime, periods in raw_results.items()})",
                label="SimulationResult raw action shift",
            ),
        },
        "simulation_dataframe:shift_action_column": {
            "path": RESULT_DATAFRAME_SOURCE,
            "source": replace_once(
                source=result_dataframe_source,
                old="            data[name] = result.actions[name]",
                new="            data[name] = jnp.roll(result.actions[name], 1)",
                label="DataFrame action-column shift",
            ),
        },
        "simulation_metadata:drop_regime_actions": {
            "path": RESULT_METADATA_SOURCE,
            "source": replace_once(
                source=result_metadata_source,
                old="        regime_to_actions[regime_name] = regime.simulation.action_names",
                new="        regime_to_actions[regime_name] = ()",
                label="result metadata action omission",
            ),
        },
        "additional_targets:overwrite_actions_single_pass": {
            "path": ADDITIONAL_TARGETS_SOURCE,
            "source": replace_once(
                source=additional_targets_source,
                old="        return {\n            k: _one_value_per_row(values=v, n_rows=n_rows) for k, v in result.items()\n        }",
                new="        return {\n            k: _one_value_per_row(values=v, n_rows=n_rows) for k, v in result.items()\n        } | {name: jnp.roll(jnp.asarray(data[name]), 1) for name in regime.simulation.action_names if name in data}",
                label="single-pass additional-target action overwrite",
            ),
        },
        "additional_targets:overwrite_actions_chunked": {
            "path": ADDITIONAL_TARGETS_SOURCE,
            "source": replace_once(
                source=additional_targets_source,
                old="    return {\n        name: np.concatenate([out[name] for out in chunk_outputs])\n        for name in chunk_outputs[0]\n    }",
                new="    return {\n        **{name: np.concatenate([out[name] for out in chunk_outputs]) for name in chunk_outputs[0]},\n        **{name: jnp.roll(jnp.asarray(data[name]), 1) for name in regime.simulation.action_names if name in data},\n    }",
                label="chunked additional-target action overwrite",
            ),
        },
        "simulation_random:reassign_taste_keys": {
            "path": SIMULATION_RANDOM_SOURCE,
            "source": replace_once(
                source=simulation_random_source,
                old='        simulation_keys[f"key_{name}"] = per_subject_keys',
                new='        simulation_keys[f"key_{name}"] = jnp.roll(per_subject_keys, 1, axis=0)',
                label="subject taste-key reassignment",
            ),
        },
        "shared_fold_average:negated_value": {
            "path": FOLD_ZERO_SAFE_SOURCE,
            "source": replace_once(
                source=fold_zero_safe_source,
                old="    return numerator / total_weight",
                new="    return -numerator / total_weight",
                label="folded zero-safe average negation",
            ),
        },
        "solution_contract:negate_kernel_result": {
            "path": SOLUTION_CONTRACT_SOURCE,
            "source": replace_once(
                source=solution_contract_source,
                old="    diagnostics: SolverDiagnostics | None = None",
                new='    diagnostics: SolverDiagnostics | None = None\n\n    def __post_init__(self) -> None:\n        object.__setattr__(self, "V_arr", -self.V_arr)',
                label="KernelResult value transport negation",
            ),
        },
        "candidate_materialization:rebind_continuous_grid": {
            "path": GRIDS_INIT_SOURCE,
            "source": replace_once(
                source=grids_init_source,
                old="from _lcm.grids.discrete import DiscreteGrid",
                new="from _lcm.grids.discrete import DiscreteGrid\n\nContinuousGrid = DiscreteGrid",
                label="continuous-grid classification rebinding",
            ),
        },
        "candidate_materialization:rebind_process_class": {
            "path": PROCESSES_INIT_SOURCE,
            "source": replace_once(
                source=processes_init_source,
                old="from _lcm.processes.iid import _IIDProcess",
                new="from _lcm.processes.iid import _IIDProcess\n\n_ContinuousStochasticProcess = _AR1Process",
                label="process-action classification rebinding",
            ),
        },
        "candidate_materialization:drop_last_discrete_code": {
            "path": DISCRETE_GRID_SOURCE,
            "source": replace_once(
                source=discrete_grid_source,
                old="        return jnp.array(self.codes, dtype=jnp.int32)",
                new="        return jnp.array(self.codes[:-1], dtype=jnp.int32)",
                label="discrete action code omission",
            ),
        },
        "candidate_materialization:drop_last_linear_point": {
            "path": CONTINUOUS_GRID_SOURCE,
            "source": replace_once(
                source=continuous_grid_source,
                old="        return grid_coordinates.linspace(\n            start=self.start, stop=self.stop, n_points=self.n_points\n        )",
                new="        return grid_coordinates.linspace(\n            start=self.start, stop=self.stop, n_points=self.n_points\n        )[:-1]",
                label="linear action point omission",
            ),
        },
        "candidate_materialization:drop_last_coordinate_point": {
            "path": GRID_COORDINATES_SOURCE,
            "source": replace_once(
                source=grid_coordinates_source,
                old="    return jnp.linspace(start, stop, n_points)  # ty: ignore[no-matching-overload]",
                new="    return jnp.linspace(start, stop, n_points)[:-1]  # ty: ignore[no-matching-overload]",
                label="shared linear coordinate omission",
            ),
        },
        "candidate_materialization:drop_last_piecewise_point": {
            "path": PIECEWISE_GRID_SOURCE,
            "source": replace_once(
                source=piecewise_grid_source,
                old="        return jnp.concatenate(segments)",
                new="        return jnp.concatenate(segments)[:-1]",
                label="piecewise action point omission",
            ),
        },
        "candidate_materialization:drop_last_process_node": {
            "path": PROCESS_BASE_SOURCE,
            "source": replace_once(
                source=process_base_source,
                old="        return self.compute_gridpoints(**self.params)",
                new="        return self.compute_gridpoints(**self.params)[:-1]",
                label="process action node omission",
            ),
        },
        "candidate_materialization:drop_last_iid_node": {
            "path": PROCESS_IID_SOURCE,
            "source": replace_once(
                source=process_iid_source,
                old='        return jnp.linspace(\n            start=kwargs["start"], stop=kwargs["stop"], num=self.n_points\n        )',
                new='        return jnp.linspace(\n            start=kwargs["start"], stop=kwargs["stop"], num=self.n_points\n        )[:-1]',
                label="IID action node omission",
            ),
        },
        "candidate_materialization:drop_last_ar1_node": {
            "path": PROCESS_AR1_SOURCE,
            "source": replace_once(
                source=process_ar1_source,
                old="        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)",
                new="        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)[:-1]",
                label="AR1 action node omission",
            ),
        },
        "candidate_materialization:drop_last_action_name": {
            "path": VARIABLES_SOURCE,
            "source": replace_once(
                source=variables_source,
                old='    actions = [name for name, var_info in info.items() if var_info.kind == "action"]',
                new='    actions = [name for name, var_info in info.items() if var_info.kind == "action"][:-1]',
                label="finalized action-name omission",
            ),
        },
        "candidate_materialization:skip_first_runtime_action_template": {
            "path": PARAMS_REGIME_TEMPLATE_SOURCE,
            "source": replace_once(
                source=params_regime_template_source,
                old="    for action_name, grid in user_regime.actions.items():",
                new="    for action_name, grid in tuple(user_regime.actions.items())[1:]:",
                label="runtime action template omission",
            ),
        },
        "candidate_materialization:negate_broadcast_runtime_points": {
            "path": PARAMS_PROCESSING_SOURCE,
            "source": replace_once(
                source=params_processing_source,
                old="            result[regime][remainder] = params_flat[chosen]",
                new='            result[regime][remainder] = (-params_flat[chosen] if remainder.endswith("__points") else params_flat[chosen])',
                label="runtime action points changed during broadcast",
            ),
        },
        "candidate_materialization:negate_flattened_runtime_points": {
            "path": NAMESPACE_SOURCE,
            "source": replace_once(
                source=namespace_source,
                old="    return MappingProxyType(flatten_to_qnames(d))",
                new='    flat = flatten_to_qnames(d)\n    return MappingProxyType({key: -value if key.endswith("__points") and hasattr(value, "dtype") else value for key, value in flat.items()})',
                label="runtime action points changed during namespace flattening",
            ),
        },
        "candidate_materialization:negate_cast_runtime_points": {
            "path": DTYPES_SOURCE,
            "source": replace_once(
                source=dtypes_source,
                old="    return jnp.asarray(np_value, dtype=target_dtype)",
                new='    out = jnp.asarray(np_value, dtype=target_dtype)\n    return -out if name.endswith("__points") else out',
                label="runtime action points changed during canonical cast",
            ),
        },
        "candidate_materialization:negate_series_runtime_points": {
            "path": PANDAS_UTILS_SOURCE,
            "source": replace_once(
                source=pandas_utils_source,
                old="    if func is None:\n        return jnp.array(sr.to_numpy(), dtype=canonical_float_dtype())",
                new="    if func is None:\n        return -jnp.array(sr.to_numpy(), dtype=canonical_float_dtype())",
                label="Series runtime action points changed during conversion",
            ),
        },
        "candidate_materialization:negate_fixed_runtime_points": {
            "path": MODEL_PROCESSING_SOURCE,
            "source": replace_once(
                source=model_processing_source,
                old="        regime_fixed = dict(fixed_flat_params.get(regime_name, MappingProxyType({})))",
                new='        regime_fixed = dict(fixed_flat_params.get(regime_name, MappingProxyType({})))\n        regime_fixed = {key: -value if key.endswith("__points") else value for key, value in regime_fixed.items()}',
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
        CORE_PROGRAM_SOURCE: core_program_source,
        OUTPUT_LAYOUT_SOURCE: output_layout_source,
        ACTION_STREAMING_SOURCE: action_streaming_source,
        ACTION_REDUCTION_SOURCE: action_reduction_source,
        COLLECTIVE_ACTION_REDUCTION_SOURCE: collective_action_reduction_source,
        LOGSUMEXP_ACTION_REDUCTION_SOURCE: logsumexp_action_reduction_source,
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
