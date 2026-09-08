#!/usr/bin/env python3
"""Prove route-local candidate-array flow from ``Q_and_F`` to full reducers.

The executable certificate is intentionally finite. Its universal half has two
explicitly separated claims. First, every coordinate produced by an already-constructed, finalized concrete
built-in action-grid object or supplied through the public runtime-points seam reaches
the pointwise ``Q_and_F`` call as an action argument. Second, on each GridSearch
route, the exact arrays bound by ``Q_arr, F_arr = Q_and_F(...)`` must reach the
full reducer without an intervening candidate-changing expression. The streamed
hard-max production route makes the equivalent pointwise claim: every canonical global
action identity is decoded in C order, its exact Q/feasibility pair enters the
mergeable hard-max reduction, and the resolved streamed program is the one lowered,
compiled, and dispatched. The collective and EV1 streaming implementations remain
sealed reference algorithms, but the native program graph deliberately selects their
dense canonical reducers instead. A collective reference block is scalarized without
changing action support; every stakeholder value is gathered at the one shared
household winner and published with the empty-feasible-set dissolution flag. The EV1
reference stream preserves the discrete-prefix branch order, hard-maxes each branch,
then adds exactly one branch value to a log-sum-exp reduction bound to the runtime scale.
The economic
construction of Q/F values and feasibility—including user DAGs, constraints,
transitions, continuation values, interpolation, and fold weights—is an explicit
semantic boundary and is not re-proved here. The proof is strict by design: a new
statement in either certified transport corridor is not assumed harmless; it has
to enter the explicit, independently checked representation allowlist.

The nine corridors are:

* singleton solve -> ``Q_arr.max(where=F_arr, ...)``;
* singleton streamed solve -> complete C-order blocks -> mergeable hard max ->
  optional unchanged fold quadrature -> compiled VALUE core;
* singleton simulate -> published dense argmax or streamed C-order hard max ->
  subject tiles -> materialized, resolved and dispatched decision program;
* collective solve -> ``collective_readout(..., feasibility=F_arr, ...)``;
* collective streamed reference -> complete C-order stakeholder blocks -> shared
  household hard max -> compiled ``(VALUE, DISSOLUTION_FLAG)`` core;
* collective simulate -> published dense ``collective_argmax_and_readout(...,
  feasibility=F_arr, ...)`` -> subject tiles -> selected decision executable;
* taste-shock dense solve fallback -> exact mask, continuous maximum, then full
  discrete logsum;
* taste-shock streamed reference -> ordered discrete-prefix branch hard max -> one
  dynamically bound log-sum-exp -> compiled VALUE core;
* taste-shock simulate -> published dense exact mask, row-major continuous maximum,
  one mean-zero Gumbel draw per discrete cell, and exact flat-index reconstruction
  -> subject tiles -> selected decision executable.

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

import ast
import copy
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
VALUE_TRANSFER_SOURCE = "src/_lcm/execution/value_transfer.py"
FOOTPRINT_SOURCE = "src/_lcm/execution/footprint.py"
INTERNAL_OUTPUTS_SOURCE = "src/_lcm/execution/internal_outputs.py"
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
SIMULATION_PROGRAMS_SOURCE = "src/_lcm/simulation/programs.py"
SIMULATION_PROGRAM_TYPES_SOURCE = "src/_lcm/simulation/program_types.py"
SIMULATION_RUNTIME_SOURCE = "src/_lcm/simulation/runtime.py"
MODEL_SOURCE = "src/lcm/model.py"
SOLVER_API_SOURCE = "src/lcm/_solver_api/replay.py"
BACKWARD_INDUCTION_SOURCE = "src/_lcm/solution/backward_induction.py"
PERIOD_REPLAY_SOURCE = "src/_lcm/solution/period_replay.py"
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

SIMULATION_OPERANDS_SOURCE = "src/_lcm/simulation/operand_placement.py"
SIMULATION_UNIT_SOURCE = "src/_lcm/simulation/unit_executor.py"
SIMULATION_HOST_SOURCE = "src/_lcm/simulation/host_operations.py"
SIMULATION_MEMORY_SOURCE = "src/_lcm/simulation/memory.py"
SIMULATION_PERIOD_INPUTS_SOURCE = "src/_lcm/simulation/period_inputs.py"
SIMULATION_REPLAY_INPUTS_SOURCE = "src/_lcm/simulation/replay_inputs.py"
SIMULATION_VALUE_READS_SOURCE = "src/_lcm/simulation/value_reads.py"
SIMULATION_VALUE_PLACEMENT_SOURCE = "src/_lcm/simulation/value_placement.py"
SIMULATION_CHUNK_INPUTS_SOURCE = "src/_lcm/simulation/chunk_inputs.py"
SIMULATION_ENTRY_INPUTS_SOURCE = "src/_lcm/simulation/entry_inputs.py"
SIMULATION_RESIDENCY_SOURCE = "src/_lcm/simulation/residency.py"
SIMULATION_GATED_ROUTING_SOURCE = "src/_lcm/simulation/gated_routing.py"
VALUE_TOPOLOGY_SOURCE = "src/_lcm/solution/v_topology.py"
RETAINED_BUFFERS_SOURCE = "src/_lcm/solution/retained_buffers.py"
SCHEDULER_SOURCE = "src/_lcm/execution/scheduler.py"
LIVENESS_SOURCE = "src/_lcm/execution/liveness.py"
CONTINUATION_READS_SOURCE = "src/_lcm/solution/continuation_reads.py"
WORKSPACE_PLANNING_SOURCE = "src/_lcm/execution/workspace_planning.py"

COMPILER_INPUTS_SOURCE = "src/_lcm/execution/compiler_inputs.py"

SIMULATION_MEMBERSHIP_SOURCE = "src/_lcm/simulation/membership.py"

SIMULATION_TASTE_STREAM_SOURCE = "src/_lcm/simulation/taste_stream.py"

_CERTIFIED_CORRIDOR_SOURCES = (
    SIMULATION_TASTE_STREAM_SOURCE,
    SIMULATION_MEMBERSHIP_SOURCE,
    COMPILER_INPUTS_SOURCE,
    MAX_Q_SOURCE,
    ARGMAX_SOURCE,
    COLLECTIVE_SOURCE,
    LOGSUM_SOURCE,
    GRID_SEARCH_SOURCE,
    CORE_PROGRAM_SOURCE,
    OUTPUT_LAYOUT_SOURCE,
    VALUE_TRANSFER_SOURCE,
    FOOTPRINT_SOURCE,
    INTERNAL_OUTPUTS_SOURCE,
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
    SIMULATION_PROGRAMS_SOURCE,
    SIMULATION_PROGRAM_TYPES_SOURCE,
    SIMULATION_RUNTIME_SOURCE,
    SIMULATION_OPERANDS_SOURCE,
    SIMULATION_UNIT_SOURCE,
    SIMULATION_HOST_SOURCE,
    SIMULATION_MEMORY_SOURCE,
    SIMULATION_PERIOD_INPUTS_SOURCE,
    SIMULATION_REPLAY_INPUTS_SOURCE,
    SIMULATION_VALUE_READS_SOURCE,
    SIMULATION_VALUE_PLACEMENT_SOURCE,
    SIMULATION_CHUNK_INPUTS_SOURCE,
    SIMULATION_ENTRY_INPUTS_SOURCE,
    SIMULATION_RESIDENCY_SOURCE,
    SIMULATION_GATED_ROUTING_SOURCE,
    VALUE_TOPOLOGY_SOURCE,
    RETAINED_BUFFERS_SOURCE,
    SCHEDULER_SOURCE,
    LIVENESS_SOURCE,
    CONTINUATION_READS_SOURCE,
    WORKSPACE_PLANNING_SOURCE,
    MODEL_SOURCE,
    SOLVER_API_SOURCE,
    BACKWARD_INDUCTION_SOURCE,
    PERIOD_REPLAY_SOURCE,
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

# Byte seals bind reviewed source files to the generated inventory. check_seals
# refreshes these hashes only; the independent callable and module contracts below
# still reject altered transport after byte resealing and require semantic review.
_SOURCE_SEALS = {
    SIMULATION_TASTE_STREAM_SOURCE: "022512bc75e5a30d22ee5e7ace2ab6a422658e7c192ad6b8a092a17049eddfc7",
    SIMULATION_MEMBERSHIP_SOURCE: "c0c92de4e55be3e7b814a67761caa75e8affe835e1887359d432341f1d85a18d",
    COMPILER_INPUTS_SOURCE: "c28ac1ab4acab2166120867158dec5eb866c082cdd05f7cb7877f2dc3c52ef8a",
    SIMULATION_OPERANDS_SOURCE: "4c0333036fd861ace225a0a38c925fa2a694433e3eddcc47d9fe1c0eea2ee9f8",
    SIMULATION_UNIT_SOURCE: "c2c8b34310339ced638e67503251a398047b9c673146bd7d49f69670717c4d3b",
    SIMULATION_HOST_SOURCE: "74bd0ccccd0b25968e502a42ab0a6c7ae19341eb946dcd1b1c21907a469786d3",
    SIMULATION_MEMORY_SOURCE: "917677a73a623cc2c8d836957d7a86832403bb1f319ec5eeeb967abad9ec31a1",
    SIMULATION_PERIOD_INPUTS_SOURCE: "70a1dc14b09edf1101dea7dac7a82e18693f1a299c413b34c8cd955b0892bac0",
    SIMULATION_REPLAY_INPUTS_SOURCE: "6a8f9e1eafbebb264b11d7b6be5535c1e9cdf90e2ec1e17f2343eddc222b01b9",
    SIMULATION_VALUE_READS_SOURCE: "b00b335634970802bc240962c6c7eeba99598924a00b4c0059b87663d4e928e4",
    SIMULATION_VALUE_PLACEMENT_SOURCE: "afd58b343d0385934c3add5552d7926dc3f9ba2700e5a40b26a65a62a75b1e68",
    SIMULATION_CHUNK_INPUTS_SOURCE: "cf82d14c75cd4826c27dba1f7ccd365f5c0d514b2c9dd4f5535ccf74335a637d",
    SIMULATION_ENTRY_INPUTS_SOURCE: "546ff0670f64c41fcfd3c3f629e783e30697b9de6164411150d8d69f40ec6e24",
    SIMULATION_RESIDENCY_SOURCE: "78486d94eecdc7b7fcaacc4c24e3a7db6610f7d6bc83c1a18f835a10eb989594",
    SIMULATION_GATED_ROUTING_SOURCE: "9e16638f296169a876c37f0c18399a8ed894dc2f4e1cb54a3466f909ddffb85e",
    VALUE_TOPOLOGY_SOURCE: "80bf43a740c77b1ddb1675a2510859b521efc6fc0f60a6a8ab7319e0bc98a4f2",
    RETAINED_BUFFERS_SOURCE: "8556ec2e2ba4265349acb6517523f70bf65528abd3e1d631c55072e847e5f870",
    SCHEDULER_SOURCE: "8071b900fa30fe5a9e38ee2133e320bf68cbe2f2e733fa3b4114adcd3bd9ad65",
    LIVENESS_SOURCE: "50c1f0d0658baf5802dcbd459ac5a2b5e0a866ef5c87289d2f448b0b1f1e4ada",
    CONTINUATION_READS_SOURCE: "43eb385d2b2795e81a17ae01b98a795c8296ce396a49ca4d3c3dc68e33cc12cf",
    WORKSPACE_PLANNING_SOURCE: "05972cd1ecf48b86c6a9ad4828e7b750ca62c130c1945118735ff002ac586f48",
    SIMULATION_PROGRAMS_SOURCE: "dc05c99c5b3e083e20e5b909ffe3a7910b94697cfb3d0fd9b9426d9cd80d5f1e",
    SIMULATION_PROGRAM_TYPES_SOURCE: "44b0bf3b343b6a0ad3659e78666c602bf4c64547ddc558c45337db1c2cafd19f",
    SIMULATION_RUNTIME_SOURCE: "e629b4fad00474a3f638481b1c7d1591a041df7d44af8bfa7b9fc23595a01656",
    LOGSUM_SOURCE: "e12061dd4f0f0176324182a2eb875cb6ebe4b97174091c597d46a622df93ff1b",
    ARGMAX_SOURCE: "0d179a5aa65a6f310f598bdad8f75a9318a24832e31bd529184c2ea90356a72d",
    COLLECTIVE_SOURCE: "c30b746e574f1462a152c62b72c788730bdcdceabd2d71e525bf49a6a2c2e8c0",
    MAX_Q_SOURCE: "e01b3de5cdad12804794a3ce83ff3a74424a9539caa70330e8792ca7878f1bc9",
    PROCESSING_SOURCE: "3aa08741042207d1dfd26e4bd8e6b65b1c20d80d5eb2b0825f3c5682d76a4ad9",
    GRID_SEARCH_SOURCE: "ed33d78e3cd17921cbde158534f386014f3a2529af5cf686b7e10c8e3fa43d7e",
    CORE_PROGRAM_SOURCE: "078df8c1d82cd31e9cfa1bfb7acd263154676ecbd13117ff7c446bab5130fdba",
    OUTPUT_LAYOUT_SOURCE: "b5e9c667a15dcb6a1393f54f4ba7d8fcea85d3e78104f6e9e4cbd16c60fd1bb5",
    VALUE_TRANSFER_SOURCE: "d1b2c3282b8504346738257983266b298d46dced2d3f49212f23b729d5877f12",
    FOOTPRINT_SOURCE: "0dbaa3c673f053a51fa7bcab4e06d2080252dd84d827ed28123d085c4000fd0f",
    INTERNAL_OUTPUTS_SOURCE: "ce6677ef989669033ad8b24ab5321e0596657b1befea6988689f96eb8b365f25",
    ACTION_STREAMING_SOURCE: "b13962dbc446a0962bf397ea3f4ecca3be3eea158bc270547251b7f92b160dc8",
    ACTION_REDUCTION_SOURCE: "c83a1147bd432a793b60706ea50f9735de418e2c7cf42090ed426672d2027135",
    COLLECTIVE_ACTION_REDUCTION_SOURCE: "5a7b0d0e530a483604018dc0bd9ee34f5ff65d3a53d507cb0c0962cf4ee732be",
    DISPATCHERS_SOURCE: "2b7efd1df0a3b8fdf0d90d6ea38b95456ca74b180cc617ebe63266a9f5dca03d",
    FUNCTOOLS_SOURCE: "578df5a2b97727d5b993d4e828bc80910a80f9781c8819935b76549ab5c17b88",
    CONTAINERS_SOURCE: "0838079e35ba498009d8af7e6ed717f870a96a2fdc628d25e80310cd630174a9",
    ZERO_SAFE_SOURCE: "6b85bacd7c01fec283fcd309a731ab73d6639975ff34edbcce1a8450fbac5f33",
    LOGSUMEXP_ACTION_REDUCTION_SOURCE: "4799ad9bfbc02ae1e5d5270a18ed81fe682f1004d63ae6cf796ff48ac5699445",
    PROBABILITY_SOURCE: "b59d16c16147af2518daaed643c10be43c506c6e3ac751cd52f04fa8fdab20d2",
    ENGINE_SOURCE: "400326d4fe75f7ee62daf88ca334c095f30689b37ef5cf16f0aacd61b9cb2b75",
    STATE_ACTION_SPACE_SOURCE: "c7af3ea4c3912efa3d5d7daa0d420168a7545e327f6e4c581b3baf54efc79f11",
    SIMULATION_SOURCE: "6215536d5dfe2c2dc32639280c591abf54df561c303dc85e4231138ed8212818",
    SIMULATION_TRANSITIONS_SOURCE: "18f53b35a8a11536f9e3df03cfc48cec357bbc2fa11308d04091620d02e179a6",
    SIMULATION_COMPILE_SOURCE: "3ca1959d612dbfbf323e523fee496a56390ddaa1f509f607803a111d728899d2",
    MODEL_SOURCE: "8ef3cdf76aa2ba621c0d556c9bbd51ef336f645b0d6a94759f73d01cbb927cab",
    SOLVER_API_SOURCE: "fbf4085b2275c96b2fa4ef85c36bfe92a015dea19a103e1e05f9ee8377d30428",
    BACKWARD_INDUCTION_SOURCE: "320d88ecb4959c19274da453a5bacd7995b5e7d54a6da76d47d0fe24f583809d",
    PERIOD_REPLAY_SOURCE: "6e08c2c390cc0cca9633236f3b7cfffdf6745526cef891f87748aca9c974803b",
    INITIAL_CONDITIONS_SOURCE: "7017df18560acd35178c29f24c87977bcf2eda20eb16f1dccc28dfe417f4757f",
    RESULT_SOURCE: "bfc4b1f19bf1852c86c4f5b87d40abed2521d0519af99ff613975fb8d3ab09f0",
    RESULT_DATAFRAME_SOURCE: "025e273c4d3bb9d8f9787189a551b113708c86b1e868d16178aa39555abf49a4",
    RESULT_METADATA_SOURCE: "5745acf8a75655a4da87c1d305d79db31582d1e4df419c059059d515770ed563",
    ADDITIONAL_TARGETS_SOURCE: "d1c8787e7968b868b4b09a90544050c5da65d2ca6203f2bc52fe6b7b7dd351e4",
    SIMULATION_RANDOM_SOURCE: "2b7fb689c8bbfdc8689916bc25063c3c001aed4103d0381e71f61b4c889a865e",
    FOLD_ZERO_SAFE_SOURCE: "0f6c6c3ad1a69ea2ef241f8f0ce924e18c00e6515c7509577c761a8151d57feb",
    SOLUTION_CONTRACT_SOURCE: "000a633cf6544fa24a057e5c2c668b94ffda232faca10bcc06e21474773d5678",
    GRIDS_INIT_SOURCE: "c66aed5ef6cdb56cfa38eebb7f870f12475f7a5f62ca1962c17230f66fd3268a",
    GRID_BASE_SOURCE: "7b3fd1b01bad73e8fb08cff1155d6a556cc8a19301107c4e60d67c113740653c",
    GRID_COORDINATES_SOURCE: "e0f3cffc38e2a854426309b3eacab5783a0a5725cc4e763a06969e03914619e8",
    DISCRETE_GRID_SOURCE: "ffea24a5a96c762a8aaad4a159c57951470ba9cee99c96dfb6e32c5b61dc6297",
    CONTINUOUS_GRID_SOURCE: "c3dbec16a9a2ba3c1f24b768b4fbf8165f38f98b222f04e7d9f7a3e613f6f431",
    PIECEWISE_GRID_SOURCE: "dd4f5444c3186c40973e4b2d9a283f240cf54ff2cda3ed14172b96d653dce5b5",
    PROCESSES_INIT_SOURCE: "db7892762ef1b5635b61b4e57ef97ae0becdd1fc7507ef8efd202323ced7671f",
    PROCESS_BASE_SOURCE: "8d4add80b7c95f89a0671506dfaeb276cf9cc0afb1ad76305686b346f0e1dddb",
    PROCESS_IID_SOURCE: "4696d7356181bc07e2d661d9010c95f014a99a5110935ead56ff05ae1725d3bf",
    PROCESS_AR1_SOURCE: "05c03c7ac6f9b600a160be81851b8c868c7919daeeeadb817b201c87d7d9213e",
    VARIABLES_SOURCE: "b22e58d7bbf84bb6235a6296c3a4f90f6d216de1f087bc6f31547b3892ee8cb6",
    PARAMS_REGIME_TEMPLATE_SOURCE: "f8d856ab22f7308e27454e9f045ac0258e74c87f42b513840aa81a48d9cb5ae2",
    PARAMS_PROCESSING_SOURCE: "c9056369922b5f088355f7f404b6fa6377c6a2b587bbf5c804e9686f0dcab8e9",
    DTYPES_SOURCE: "0df3aa83d3d7d2f55438d91b9d4af2f25a17ea0ce836e923ab06a7458c59e73a",
    NAMESPACE_SOURCE: "254509e538c6a2264a71e04cdd5abdb60ad92f04899a37f710004222ae855bea",
    PANDAS_UTILS_SOURCE: "a0e3f5efb79cf0b252690ee1177bb0c6a3d52882dea15cc70854e71da4ea57a8",
    MODEL_PROCESSING_SOURCE: "7bf036615ed4014bb2a7e9c90348dd0932d0c377a420ede40068b0af53071fc5",
}

EXPECTED_DIRECT_FLOW_MUTATION_COUNT = 404
EXPECTED_DIRECT_FLOW_MUTATION_NAMES_SHA256 = (
    "7fb69a87daf4bd1f927d76d0c6507138d8b0d7eda25d478c3f5f551b1bdd1b61"
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


def _guarded_binding(
    *,
    tree: ast.Module,
    outer_name: str,
    nested_name: str,
    taste_shocks: bool,
) -> ast.Call:
    """Resolve the signature-wrapping call one ``has_taste_shocks`` arm binds.

    Each arm binds the returned reducer to exactly one
    ``with_signature(<kernel instance>, ...)`` call, and the builder itself
    defines no callable of its own, so nothing the reducer reads outlives the
    build that produced it.
    """
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
    nested_scopes = [
        node
        for node in ast.walk(outer)
        if node is not outer
        and isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        )
    ]
    route = "taste-shock" if taste_shocks else "ordinary"
    if nested_scopes or len(guard.body) != 1 or len(guard.orelse) != 1:
        raise ValueError(
            f"expected the {route} arm of {outer_name!r} to bind {nested_name!r} "
            f"in one statement and the builder to define no callable of its own; "
            f"found {len(nested_scopes)} nested scopes and branch lengths "
            f"{len(guard.body)}/{len(guard.orelse)}"
        )
    statement = (guard.body if taste_shocks else guard.orelse)[0]
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and _target_names(statement.targets[0]) == (nested_name,)
        and isinstance(statement.value, ast.Call)
        and _call_name(statement.value) == "with_signature"
    ):
        raise ValueError(
            f"the {route} arm of {outer_name!r} does not bind {nested_name!r} to "
            "one signature-wrapped kernel"
        )
    return statement.value


# Construction keywords of a taste-shock max-Q kernel, in declaration order.
_TASTE_KERNEL_FIELDS = ("Q_and_F", "n_discrete_action_axes")

# Construction keywords of a hard-max max-Q kernel, in declaration order.
_HARD_MAX_KERNEL_FIELDS = ("Q_and_F", "stakeholders", "pareto_weights")


def _binding_kernel(
    *, tree: ast.Module, call: ast.Call, class_name: str, fields: tuple[str, ...]
) -> ast.FunctionDef:
    """Return the ``__call__`` of the kernel class one binding constructs.

    The construction names the route's own kernel class and passes exactly
    ``fields``, each the builder local of the same name, so an arm cannot hand
    the kernel a narrowed axis count, a disabled collective route, or any value
    the builder did not compute for it.
    """
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Call):
        raise ValueError("the signature wrapper does not take one kernel instance")
    construction = call.args[0]
    if _call_name(construction) != class_name:
        raise ValueError(
            f"the signature wrapper's kernel is not a plain {class_name} construction"
        )
    named = [
        (keyword.arg, keyword.value)
        for keyword in construction.keywords
        if keyword.arg is not None
    ]
    if construction.args or len(named) != len(construction.keywords):
        raise ValueError(f"{class_name} is not constructed from named builder locals")
    observed = tuple(argument for argument, _ in named)
    if observed != fields:
        raise ValueError(
            f"{class_name} is not constructed from exactly its declared fields; "
            f"expected {fields}, found {observed}"
        )
    rewired = tuple(
        argument
        for argument, value in named
        if not _name(node=value, expected=argument)
    )
    if rewired:
        raise ValueError(
            f"{class_name} fields {rewired} are not the builder locals of the same name"
        )
    return _method_definition(tree=tree, class_name=class_name, method_name="__call__")[
        1
    ]


def _ordinary_kernel(
    *,
    tree: ast.Module,
    outer_name: str,
    nested_name: str,
    class_name: str,
    fields: tuple[str, ...],
) -> tuple[ast.Call, ast.FunctionDef]:
    """Return the binding and kernel from the false arm of ``has_taste_shocks``."""
    call = _guarded_binding(
        tree=tree,
        outer_name=outer_name,
        nested_name=nested_name,
        taste_shocks=False,
    )
    return call, _binding_kernel(
        tree=tree, call=call, class_name=class_name, fields=fields
    )


def _taste_kernel(
    *,
    tree: ast.Module,
    outer_name: str,
    nested_name: str,
    class_name: str,
    fields: tuple[str, ...],
) -> tuple[ast.Call, ast.FunctionDef]:
    """Return the binding and kernel from the true arm of ``has_taste_shocks``."""
    call = _guarded_binding(
        tree=tree,
        outer_name=outer_name,
        nested_name=nested_name,
        taste_shocks=True,
    )
    return call, _binding_kernel(
        tree=tree, call=call, class_name=class_name, fields=fields
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


def _kernel_field(*, node: ast.AST | None, expected: str) -> bool:
    """Match one read of a frozen kernel's field, spelled ``self.<expected>``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == expected
        and _name(node=node.value, expected="self")
    )


def _exact_reducer_signature_call(
    *, call: ast.Call, simulate: bool, taste_shocks: bool
) -> bool:
    """Pin the signature wrapper that exposes the exact action/state inputs."""
    if _call_name(call) != "with_signature" or len(call.args) != 1:
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
        and tuple(item.arg for item in args.args) == ("self", "next_regime_to_V_arr")
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
    expected_names = ("to_jax",)
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
            "grid base: Grid may contain only its docstring and the abstract "
            "coordinate materializer"
        )
        return errors
    expected = {
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
        and _kernel_field(node=statement.value.func, expected="Q_and_F")
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
        and _kernel_field(node=generator.iter.args[0], expected="stakeholders")
        and not generator.iter.keywords
        and not generator.ifs
        and generator.is_async == 0
    )


def _exact_weights_call(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and _call_name(node) == "_evaluate_pareto_weights"
        and not node.args
        and _kernel_field(
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
        and _kernel_field(node=generator.iter, expected="stakeholders")
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
    if ast.unparse(node.test) != "self.stakeholders is not None" or node.orelse:
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
    # One binding per taste-guard arm, plus the annotation that declares the
    # reducer's type before either arm binds it.
    expected_stores = 3
    if _stored_name_count(node=outer, name=nested_name) != expected_stores:
        errors.append(
            f"{outer_name}: returned reducer {nested_name} is not bound exactly "
            "once per taste-guard arm under one annotation"
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
mapped = (
    tiled_productmap(
        func=max_Q_over_a,
        variables=inner_state_names,
        width_keyword=cell_width_keyword,
        untiled_variables=untiled_state_names,
    )
    if cell_width_keyword is not None
    else productmap(
        func=max_Q_over_a,
        variables=inner_state_names,
        batch_sizes={name: batch_sizes[name] for name in inner_state_names},
    )
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
        extra_param_names=[
            *extra_param_names,
            *((cell_width_keyword,) if cell_width_keyword is not None else ()),
        ],
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
    solve_prefix += "max_Q_over_a: Callable[..., FloatND | tuple[FloatND, BoolND]]\n"
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
    simulate_prefix += "argmax_and_max_Q_over_a: ArgmaxQOverAFunction\n"
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
                "cell_width_keyword",
                "untiled_state_names",
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
                "None",
                "()",
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
                "Any",
                "cast",
                "ClassVar",
                "collective_argmax_and_readout",
                "collective_readout",
                "dataclass",
                "EULER_GAMMA",
                "inspect",
                "jax",
                "jnp",
                "logsum_and_softmax",
                "math",
                "productmap",
                "tiled_productmap",
                "vmap_1d",
                "with_signature",
                "ScalarFloat",
            },
            expected_imports=[
                "import inspect",
                "import math",
                "from dataclasses import dataclass",
                "from types import MappingProxyType",
                "from typing import Any, ClassVar, cast",
                "import jax",
                "import jax.numpy as jnp",
                "from dags import with_signature",
                "from _lcm.logsum import EULER_GAMMA, logsum_and_softmax",
                "from _lcm.regime_building.argmax import argmax_and_max",
                "from _lcm.regime_building.collective import ParetoWeights, collective_argmax_and_readout, collective_readout",
                "from _lcm.solution.action_streaming import build_streaming_collective_max_Q_over_a, build_streaming_ev1_max_Q_over_a, build_streaming_max_Q_over_a",
                "from _lcm.utils.dispatchers import productmap, tiled_productmap, vmap_1d",
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
                "Any": 1,
                "cast": 1,
                "ClassVar": 1,
                "collective_argmax_and_readout": 1,
                "collective_readout": 1,
                "dataclass": 1,
                "dict": 0,
                "draw_taste_shock_noise": 1,
                "_HardMaxArgmaxQOverA": 1,
                "_HardMaxQOverA": 1,
                "_SmoothedMaxQOverA": 1,
                "_StreamedMaxQOverA": 1,
                "_TasteShockArgmaxQOverA": 1,
                "enumerate": 0,
                "EULER_GAMMA": 1,
                "get_argmax_and_max_Q_over_a": 1,
                "get_max_Q_over_a": 1,
                "get_streaming_max_Q_over_a": 1,
                "_fail_if_action_width_keyword_collides": 1,
                "inspect": 1,
                "jax": 1,
                "jnp": 1,
                "list": 0,
                "logsum_and_softmax": 1,
                "math": 1,
                "productmap": 1,
                "tiled_productmap": 1,
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
    """Pin streamed VALUE production, optional folding, and fail-closed boundaries."""
    return _exact_callable_errors(
        tree=tree,
        label="streamed max-Q builder",
        contracts={
            "get_streaming_max_Q_over_a": "cfdef947f803d14dd8299471295492ba69f2f91f96a79bd56df24e47d4646ce6",
            "_fail_if_action_width_keyword_collides": (
                "20d3a1998c95f4decc9c5b5c8971ddc98fd1140c1954f427863409de33d2b2c4"
            ),
            "_fail_if_full_V_streaming_route_is_unsupported": (
                "cd4c96d572ec7df9dc269f5fa2bfc1ec5c16fe0a78de3be56adc28c15f065d2c"
            ),
            "_fail_if_streaming_co_map_layout_is_invalid": (
                "59c06aedafc8bcbe31d7f2f7f7b7d94e1d8044bf529c6f11a05882c5bf1d7979"
            ),
            "_wrap_with_fold_reduction": (
                "a586674124f90ff862f64458b40d6a6f8bbf6586e9772d9bdb4d31bf68c9c34c"
            ),
            "_StreamedMaxQOverA.__call__": (
                "b4865fc0cc966b6f49e58923347328ada708be3cacb4ab5897a87e2d8bc13fda"
            ),
        },
    )


def _max_kernel_surface_errors(tree: ast.Module) -> list[str]:
    """Pin the five max-Q kernel classes to fields, one ``__call__``, and no more.

    Every corridor above reads the reduction out of a kernel's ``__call__`` and
    its operands out of ``self.<field>``. A property, a descriptor, or a
    ``__post_init__`` on one of these classes could answer a field read with
    something the builder never passed, so the class surface is allowlisted
    exactly: the frozen dataclass decorator, no base and no class keyword, the
    declared fields in order, and ``__call__`` as the only method.
    """
    surfaces = (
        (
            "smoothed max-Q kernel",
            "_SmoothedMaxQOverA",
            (
                "__name__: ClassVar[str] = 'max_Q_over_a'",
                "Q_and_F: Callable[..., tuple[FloatND, BoolND]]",
                "n_discrete_action_axes: int",
            ),
        ),
        (
            "hard-max max-Q kernel",
            "_HardMaxQOverA",
            (
                "__name__: ClassVar[str] = 'max_Q_over_a'",
                "Q_and_F: Callable[..., tuple[FloatND, BoolND]]",
                "stakeholders: tuple[str, ...] | None",
                "pareto_weights: ParetoWeights | None",
            ),
        ),
        (
            "streamed max-Q kernel",
            "_StreamedMaxQOverA",
            (
                "__name__: ClassVar[str] = 'streamed_max_Q_over_a'",
                "Q_and_F: Callable[..., tuple[FloatND, BoolND]]",
                "action_names: tuple[ActionName, ...]",
                "n_discrete_action_axes: int",
                "has_taste_shocks: bool",
                "stakeholders: tuple[str, ...] | None",
                "pareto_weights: ParetoWeights | None",
                "q_and_f_arg_names: frozenset[str]",
                "action_width_keyword: str",
            ),
        ),
        (
            "taste-shock argmax kernel",
            "_TasteShockArgmaxQOverA",
            (
                "__name__: ClassVar[str] = 'argmax_and_max_Q_over_a'",
                "Q_and_F: Callable[..., tuple[FloatND, BoolND]]",
                "n_discrete_action_axes: int",
            ),
        ),
        (
            "hard-max argmax kernel",
            "_HardMaxArgmaxQOverA",
            (
                "__name__: ClassVar[str] = 'argmax_and_max_Q_over_a'",
                "Q_and_F: Callable[..., tuple[FloatND, BoolND]]",
                "stakeholders: tuple[str, ...] | None",
                "pareto_weights: ParetoWeights | None",
            ),
        ),
    )
    errors: list[str] = []
    for label, class_name, fields in surfaces:
        errors.extend(
            _class_surface_errors(
                tree=tree,
                label=label,
                class_name=class_name,
                fields=fields,
                methods=("__call__",),
                decorators=("dataclass(frozen=True, kw_only=True, eq=False)",),
            )
        )
    return errors


def _functools_adapter_errors(tree: ast.Module) -> list[str]:
    """Pin positional-origin preservation through nested co-map adapters."""
    return _exact_callable_errors(
        tree=tree,
        label="allow-args positional transport",
        contracts={
            "_split_bound_arguments": (
                "a7350bd29eb79dcb783acfc520638b8e959684ba5be19f3c4e4cac6f895534d8"
            ),
            "allow_args": (
                "aafbd21439f91d8e29c52097e4b0711469942a9ca6bafce071278ef69b3d78fe"
            ),
        },
    )


def _core_program_transport_errors(tree: ast.Module) -> list[str]:
    """Pin the sole native program graph through materialization and resolution."""
    errors = _class_surface_errors(
        tree=tree,
        label="core-program value read",
        class_name="ValueRead",
        fields=(
            "target: ValueArtifactAddress",
            "source: ValueConsumerAddress",
        ),
        methods=("__post_init__",),
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="core-program reduced axis",
            class_name="ReducedAxis",
            fields=(
                "name: str",
                "coordinate_names: tuple[ActionName, ...]",
                "coordinate_extents: tuple[int, ...]",
                "canonical_order: Literal['c']",
                "reduction: ReductionDeclaration",
                "width_keyword: str",
                "minimum_width: int = 1",
                "alignment: int = 1",
            ),
            methods=("__post_init__", "extent"),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="core-program tiled axis",
            class_name="TiledOutputAxis",
            fields=(
                "name: str",
                "state_names: tuple[StateName, ...]",
                "extent: int",
                "width_keyword: str",
                "minimum_width: int = 1",
                "alignment: int = 1",
            ),
            methods=("__post_init__",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="core-program execution requirements",
            class_name="CoreExecutionRequirements",
            fields=(
                "reduced_axes: tuple[ReducedAxis, ...] = ()",
                "tiled_axes: tuple[TiledOutputAxis, ...] = ()",
                "value_reads: tuple[ValueRead, ...] = ()",
                "internal_inputs: Mapping[str, InternalInputRef] = MappingProxyType({})",
                "host_axis_names: tuple[str, ...] = ()",
            ),
            methods=("__post_init__", "axes", "axis_names"),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="core-program build context",
            class_name="CoreBuildContext",
            fields=(
                "state_action_space: object",
                "next_regime_to_V_arr: Mapping[str, object]",
                "next_regime_to_continuation: Mapping[str, object]",
                "flat_params: Mapping[str, object]",
                "period: int",
                "ages: object",
                "edge_regime_to_V_arr: Mapping[str, object] | None = None",
                "same_period_regime_to_V_arr: Mapping[str, object] | None = None",
            ),
            methods=("__post_init__",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="core-program declaration",
            class_name="CoreProgram",
            fields=(
                "name: str",
                "function: Callable[..., object]",
                "argument_builder: CoreArgumentBuilder",
                "requirements: CoreExecutionRequirements",
                "output_roles: object",
                "disposition: CoreExecutionDisposition",
                "disposition_reason: str | None = None",
                "donation_candidates: tuple[str, ...] = ()",
                "scope: ProgramScope = ProgramScope.ANY",
                "retained_artifact_keys: _RetainedArtifactKeys = ()",
                "retained_artifact_payload_types: _RetainedArtifactPayloadTypes = MappingProxyType({})",
                "replaces_program: str | None = None",
                "internal_outputs: tuple[InternalOutputSpec, ...] = ()",
                "compiler_options: tuple[tuple[str, int], ...] = ()",
            ),
            methods=("__post_init__",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="materialized core-program declaration",
            class_name="MaterializedCoreProgram",
            fields=(
                "name: str",
                "function: Callable[..., object]",
                "arguments: Mapping[str, object]",
                "requirements: CoreExecutionRequirements",
                "output_roles: object",
                "disposition: CoreExecutionDisposition",
                "donation_candidates: tuple[str, ...]",
                "disposition_reason: str | None = None",
                "scope: ProgramScope = ProgramScope.ANY",
                "retained_artifact_keys: tuple[ArtifactKey, ...] = ()",
                "replaces_program: str | None = None",
                "internal_outputs: tuple[InternalOutputSpec, ...] = ()",
                "compiler_options: tuple[tuple[str, int], ...] = ()",
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
                "name: str",
                "function: Callable[..., object]",
                "arguments: Mapping[str, object]",
                "static_kwargs: Mapping[str, int]",
                "requirements: CoreExecutionRequirements",
                "output_roles: object",
                "disposition: CoreExecutionDisposition",
                "donation_candidates: tuple[str, ...]",
                "tile_widths: Mapping[str, int]",
                "specialization_key: Hashable",
                "input_transfer_plan: tuple[ResolvedValueTransfer, ...]",
                "disposition_reason: str | None = None",
                "scope: ProgramScope = ProgramScope.ANY",
                "retained_artifact_keys: tuple[ArtifactKey, ...] = ()",
                "replaces_program: str | None = None",
                "internal_outputs: tuple[InternalOutputSpec, ...] = ()",
                "compiler_options: tuple[tuple[str, int], ...] = ()",
            ),
            methods=("__post_init__",),
        )
    )
    try:
        disposition = _class_definition(tree=tree, name="CoreExecutionDisposition")
    except ValueError as error:
        errors.append(f"core-program disposition: {error}")
    else:
        body = list(disposition.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
        if (
            [ast.unparse(item) for item in disposition.bases] != ["StrEnum"]
            or disposition.decorator_list
            or disposition.keywords
            or not _statements_match(
                observed=body,
                expected_source="""PLANNED = "planned"
DENSE = "dense"
HOST_DRIVEN = "host_driven"
""",
            )
        ):
            errors.append("core-program disposition contract changed")
    try:
        scope = _class_definition(tree=tree, name="ProgramScope")
    except ValueError as error:
        errors.append(f"core-program scope: {error}")
    else:
        body = list(scope.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
        if (
            [ast.unparse(item) for item in scope.bases] != ["StrEnum"]
            or scope.decorator_list
            or scope.keywords
            or not _statements_match(
                observed=body,
                expected_source="""ANY = "any"
VALUES_ONLY = "values-only"
REPLAY = "replay"
ARTIFACT = "artifact"
""",
            )
        ):
            errors.append("core-program scope contract changed")
    version_assignments = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            _target_names(target) == ("_CORE_PROGRAM_VERSION",)
            for target in statement.targets
        )
    ]
    if not _statements_match(
        observed=version_assignments,
        expected_source="_CORE_PROGRAM_VERSION = 7",
    ):
        errors.append("core-program specialization version binding changed")
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="core-program provider-to-resolver transport",
            contracts={
                "ValueRead.__post_init__": "e5281a9a44553519fe08e3a5248cb54b2a107e02de63727145651525fd033843",
                "ReducedAxis.__post_init__": "030c5cd6aadd91db7c7362241817b7b1734cf27d853824376ba1206bdfae05ef",
                "ReducedAxis.extent": "aebdd54708094461d473977783f0588d2491c212629e1f5a40f8cf24c789802a",
                "TiledOutputAxis.__post_init__": "01f1caecfb2879cb72f69852f893e7c1b72f4a1b94baf02222d7bc8fe6b5e140",
                "CoreExecutionRequirements.__post_init__": "b8c76151ae6388cf29f8b580d1b6604c5969829bfcc46838a5adcd7d04e73c7b",
                "CoreBuildContext.__post_init__": "c001bdfea799659c6e0f1d0ee09180940d5176d8a59675dcabef190a12716d7b",
                "CoreProgram.__post_init__": "cc2b0daa4751a08b0be23ba75f086bc0c1d8ffcb6a45a9f7b1e3d914375bbf47",
                "MaterializedCoreProgram.__post_init__": "91815679a6b9b6ac14dfc35f3a2f40b3fecf31200cb03105692a963712570437",
                "ResolvedCoreProgram.__post_init__": "468796cd81e23cb793ba8fcc49edcac4167064550a54c16fcfdbd4abe48674c0",
                "core_program_graph": "0fb65d36c8408bfcec4b084cb304e3085037c266e932524d411654efac5c47f7",
                "_reject_native_duplicate_authorities": "02f48b342f94e73d6c57732bb28b31d9c64f33df9f5f4b7ea63930450831d4f9",
                "_snapshot_and_validate_graph": "391b89b66355ecdba2d0c869b4a748f2a0ef065610255e4819e398cf53faa7ca",
                "_validate_replay_replacements": "2aed7e1993700410979ff136c2dc13104a1f210b7d1eabf3bfd7dae22b75eb69",
                "_validate_program_declaration": "6158dc42296c246bbb8ab23d270dcc46e8175e89ed190af503bdebe788f11c6b",
                "_validate_retention_declaration": "62f75acabee1eb14ea74360f8563970ba1f5639dd06a0573a49d80402e437ba0",
                "_validate_retained_artifact_payload_types": "8ceae90284dbdc25b64e0b66d79990e8f9104f9a6410d90d8eb94c426d24ecbd",
                "_validate_disposition_reason": "5df6651c2d9db4ebeb5542394c738c5238144ff19f48f0639b3bb6b43c63bc8d",
                "materialize_core_program": "3fca95ce9b59331c45e956c4b5a86d468cfc10ff726875ec36f607a83f0cfb31",
                "resolve_core_program": "5bbda604fd5e7ca64673ae837d8797db3fccd54bef3d5118faf20170ebb56ea3",
                "select_programs": "545d2aaa5fd158f5cbbe4c8a2bf69cfffdaff588eb5066fcde54de59fb37c91b",
                "_validate_core_program": "26c1ac9dd671db992b08622950966686a4a99a98e29931ef8a6af8b4bb27e50d",
                "_validate_materialized_declaration": "d4a7b72b1943b877e689ed05bc6e83e8194fa20169615d7a0a8bd970bce96534",
                "_resolve_input_transfer_plan": "82f85e723e106098c2fa06f08d3e13bab356c75d313b73d10ac0faec54d70c93",
                "_validate_value_reads": "7195e459ee532bf8ba4761bbf4570079282122e85a588f4f77064b0bb3a07ca0",
                "_value_read_argument_leaf": "5578c82ba01f569fd8ab63c4d2f09bb94529a08f07a9ee203f4c22be737eede2",
                "_validate_transfer_argument_metadata": "982c86620af578c44c9a3881eca2917d84773b724a7072e54a134fdc5523820e",
                "_validate_reduced_axis": "ddbd1f45950e27f98e26c4ec6e966cb81720a58fc76b509304a1584e66b39c55",
                "_validate_tile_width": "12b8130b0c8fd36ca93fb5c8d5fc47b7b02b802e5c0c84a47858c7999d61e98e",
                "_validate_coordinate_argument": "b37c132508f8a2ca8803c1db613fff68b4506e417b0832638cbd1ff432a74cd0",
                "_validate_width_keyword": "b30b4f2a38c8a23089c9443ce0df2079589b6d5c881035bfbb3d50da9eadb659",
            },
        )
    )
    return errors


def _internal_outputs_transport_errors(tree: ast.Module) -> list[str]:
    """Pin the internal-output resolver a consumer is lowered against.

    A consumer names a producer's declared subtree and is traced against the
    template that subtree publishes; the width invariance check is what lets
    the planner select a producer's width independently of every consumer
    already lowered against it. Each of these four callables is pinned by
    exact AST, so relocating the collision check, the invariance refusal, or
    the traced-argument set into a helper does not evade the proof.
    """
    return _exact_callable_errors(
        tree=tree,
        label="internal-output producer-to-consumer transport",
        contracts={
            "resolve_producer": "7b9c19ee788a2bd30a18756f4d0fb9395023d1f6f5bf6f02bb0f49ddb3d51496",
            "assert_width_invariant_internal_outputs": "cf54d009a3abce9124ea579209b720ab06f7fe9bff641753f60753cc2ab1e975",
            "consumed_producer_names": "6c7019c97744a8bd73f34344a6dc45f195972e393dfeff1d1a206f0dce286ee2",
            "internal_input_templates": "4854454958ec8301926d97ddce7e09d5557babb4d79fab1aa5c475bd9477769b",
        },
    )


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
            methods=("semantic_key", "exactness"),
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
                "_StreamingEV1ExpectedMax.__call__": "9ad46e41de04ab0efb45f873459340f6933d984939f51d120afa6e53e2ad9510",
                "_prepare_action_call": "290fd810472aeb3336fdd437ccba50159e9d28c6822d6b8122fa4bdec8c71159",
                "_evaluate_block": "ef561620b6da26cd26cd4dda89f95cf480bc1c31a3ecd5d27eee12f8c7f3fc17",
                "_evaluate_ev1_branch_block": "5f88bff6dc8900c8d179df8c4be4763ff7c9656ee66bc78fe44f10f3939517fe",
                "_evaluate_collective_block": "33db932726923fa9e5cde59e3d1974b6500ef8cfebc166cc65f67a39d6dd8616",
                "_start_reduction": "034b3966dd04c0e0d66e085e8e2c4e16b127e1d8a9869ecfa039c3b5e7928b04",
                "_scan_remaining_blocks": "c22ed2d6f392d72483a2d8cfbf5e741d8c248a8aef1baf9b914938ef255f03d2",
                "_add_block": "a5047bea80275b77727b69b06d563bcdfea7e80c0f99dda34f6570948ccd1a72",
                "_reduce_no_action": "47067ad94112c815eccf14395b03ad372376e53cfd5b4f337ec70404d3a32da6",
                "_start_collective_reduction": "019872b31d89c2a2c8ba68ee6128a9feb2b90b52c6ee3b00506708db6195adce",
                "_scan_remaining_collective_blocks": "e40fba3b89b43e03a36d8b5fd6ebc32560bbe7fb8b5817c47bd8735fa27b641b",
                "_add_collective_block": "5ed017db741c30cae43dd1c4a526c703c8601857c2fbca23e9900ca560e78198",
                "_reduce_collective_no_action": "dbd215d04c091bdf6942fa6edb4a561468a5bc234f13ae24a67362dc5d48a9ff",
                "_decode_action": "5ac47e5a2d400754255cd938bf9e27ec91007741c8bbf5ad18e04bb3a9a24cbe",
                "_validate_scalar_Q_and_F": "a2b49855d9fe1572f7440248db7edfce283a2179394648ca55b6b8ce550f5364",
                "_validate_block_Q_and_F": "7f00abbccfe23768df403596eb22c715eb3542b4e43dd67593f7bd3e487fdeef",
                "_validate_collective_scalar_Q_and_F": "600332c08d2ed5aa6a4ffaeee07a878c8a713a674f54456c8a9d05dbe9eddd35",
                "_validate_collective_block_Q_and_F": "1e12615fa5fa04137cebb137387bc1c2ad98b31f1978dd47f209d00207036bef",
                "_initialize_ev1_reduction": "41728c9433880bdb06c0ab5d3c0821a7f100f238fa1faec90dc5ca967c627650",
                "_add_ev1_block": "5c63a9ca306c889a05581aaab90a9da4a43082772a67a7098603818fc5dbb283",
                "_finalize_open_ev1_branch_group": "f1eea36aa27ec48e7e79557c8928ed62d5e6cf01d0cc100a8e558ebdb31196fd",
                "_scan_remaining_ev1_blocks": "219ff26f338b96e2fce757277ef6495afcda3bf134c7a36fe1d8822aaa0151a9",
                "_flush_ev1_branch_group": "cb87cd4217df5fb3e45806593d2013a2ba452f17cbb08097a9d0cacd1e6f9270",
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
    """Pin GridSearch's sole native program declaration and shared builder."""
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
    try:
        disposition_cls = _class_definition(
            tree=tree, name="_ActionStreamingDisposition"
        )
    except ValueError as error:
        errors.append(f"solve caller: {error}")
    else:
        disposition_body = list(disposition_cls.body)
        if (
            disposition_body
            and isinstance(disposition_body[0], ast.Expr)
            and isinstance(disposition_body[0].value, ast.Constant)
            and isinstance(disposition_body[0].value.value, str)
        ):
            disposition_body.pop(0)
        if (
            [ast.unparse(item) for item in disposition_cls.bases] != ["StrEnum"]
            or disposition_cls.decorator_list
            or disposition_cls.keywords
            or not _statements_match(
                observed=disposition_body,
                expected_source='''STREAMED = "streamed"
DENSE_EV1_NONCANONICAL = "deliberately_dense:ev1_canonical_reduction_order"
DENSE_COLLECTIVE_RESOURCES = "deliberately_dense:collective_resource_regression"
DENSE_TRIVIAL_ACTION_PRODUCT = "deliberately_dense:trivial_action_product"
DENSE_CO_MAP_REFERENCE_CHANNEL = "deliberately_dense:co_map_with_separate_reference_channel"
UNSUPPORTED_COLLECTIVE_EV1 = "unsupported:collective_ev1"
UNSUPPORTED_EV1_FOLD = "unsupported:ev1_fold"
UNSUPPORTED_COLLECTIVE_FOLD = "unsupported:collective_fold"
UNSUPPORTED_EV1_WITHOUT_DISCRETE_ACTION = "unsupported:ev1_without_discrete_action"

@property
def category(self) -> str:
    """Return the stable streamed/deliberately-dense/unsupported category."""
    return self.value.partition(":")[0]
''',
            )
        ):
            errors.append("solve caller: action-streaming disposition contract changed")
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="solve caller live streamed provider",
            contracts={
                "_select_action_width_keyword": "b45663df866d5a48c05b8955b6cdc68515697e8fa925566ae72afd06b3850104",
                "_select_cell_width_keyword": "f686d6cc7ae0d93dd1e3c301600872996943c7e3d6788c9d5098d39449793727",
                "_select_width_keyword": "00cd19cec6e137d7d9e044bc1625793b1d6f78bbdfc93d6858bb6f8e9d3c022f",
                "GridSearch.build_period_kernels": "4be9b67b9b257b966ff9585d02a744f307a0f2b9a37dadfee2732b29874aad42",
                "_edge_reference_regimes_for_targets": "fae893f62c5a3eb6e8d4df88dae39fd283a5d86cd1c87a173da15287ea945af0",
                "_classify_action_streaming": "09d190475ffaf8c269880b7062a4be39e149f27d801e5fb640fa171753337ebf",
                "_supports_action_streaming": "d93f977fad68ad528beb9d4b9e6d45e5eb95b53c9a0398ff6f6a62ec548bad11",
                "_value_reads": "9712ae402debbd0c37a12999b224e365c0ca1e1a8bcbb6e7bbb4043cbcfcacfc",
                "_value_read": "082a372c7e48bf7a32d390079e2dd60b9ed5aa56868cbf9fbccf0b8f924b3bb6",
            },
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="solve caller shared argument builder",
            class_name="_GridSearchArgumentBuilder",
            fields=(
                "regime_name: RegimeName",
                "same_period_ref_regimes: tuple[RegimeName, ...] = ()",
                "edge_reference_regimes: tuple[RegimeName, ...] = ()",
                "edge_target_regimes: tuple[RegimeName, ...] = ()",
            ),
            methods=(
                "__call__",
                "_with_edge_substitution",
                "_edge_reference_args",
                "_same_period_params",
            ),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="solve caller native program graph",
            class_name="_GridSearchPeriodKernel",
            fields=("_core_programs: Mapping[str, CoreProgram]",),
            methods=(
                "__post_init__",
                "core_programs",
                "with_fixed_params",
                "__call__",
            ),
        )
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="solve caller shared builder and native program graph",
            contracts={
                "_GridSearchArgumentBuilder.__call__": "44f395121e063a099adc1c8fea5b7fd377ce5ac8019ea2ec59ffd82a9fb0915e",
                "_GridSearchArgumentBuilder._with_edge_substitution": "8d253b526755274685f3dbf2e28efe4f35062993bedd017cffddf2d17fe657cc",
                "_GridSearchArgumentBuilder._edge_reference_args": "dbc3490fe06a35e27546fd6c4a29d185c4e05c3ebc500aa2a1f220615f55c0b9",
                "_GridSearchArgumentBuilder._same_period_params": "8a9e646c6abededa222865ead1ad7e07c5355626b32a9fb1e631497a4889a58e",
                "_GridSearchPeriodKernel.__post_init__": "056fa9a213d05cd91edced63b00b0968a0cb34bd4aae6262d429fc46107d6b57",
                "_GridSearchPeriodKernel.core_programs": "0d96f7bea814e419ef1dbdebbc3257d63c3c0d36d9e6e52e619f8f44ae5a8a56",
                "_GridSearchPeriodKernel.with_fixed_params": "3ac015a29f87cd287c740773c837852d814c646a3131a9a2aa72ed99954bff1c",
                "_GridSearchPeriodKernel.__call__": "060901593efed23b0252a251f1a720e96769a7fa116ef3dfe4f2a41749304252",
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
                "expected_leaves: tuple[ExpectedOutputLeaf, ...]",
            ),
            methods=(),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="output layout",
            class_name="PlannedCore",
            fields=(
                "compiled: Callable",
                "layout: ResolvedOutputLayout",
                "tile_widths: Mapping[str, int]",
                "input_transfer_plan: tuple[ResolvedValueTransfer, ...] = ()",
                "internal_input_templates: Mapping[str, object] = MappingProxyType({})",
                "transfer_cache: TransferCache | None = None",
                "donated_arguments: tuple[str, ...] = ()",
                "name: str",
            ),
            methods=("__post_init__", "__call__"),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="output layout",
            class_name="StateAxesLeading",
            fields=(
                "state_names: tuple[StateName, ...]",
                "n_free_leading_axes: int = 0",
                "dtype: object | None = None",
                "shape: tuple[int, ...] | None = None",
            ),
            methods=("__post_init__",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="output layout",
            class_name="ExpectedOutputLeaf",
            fields=(
                "label: str",
                "shape: tuple[int, ...] | None",
                "dtype: object | None",
                "sharding: jax.sharding.Sharding",
            ),
            methods=(),
        )
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="output layout",
            contracts={
                "resolve_output_layout": "b22b19e7669e657b6b0e37190f52c4b8a4f8e39cb4101a3d7b2fc5c522c25fee",
                "_validate_output_roles": "7cbf07ffbcb4becaabdb34d86ea44caca2bb8561b14f6768970ebccea79e4b08",
                "assert_output_layout": "daad2a8d5013f547b1004e7f51d7c71d6051b57badf5d3933c866774f862c2a8",
                "_assert_output_metadata": "4fdb1d8439ec600990e801f0da9197759f71dc01661a1d0cf14d332be05b1209",
                "PlannedCore.__post_init__": "92f25766a12bbd9fe2e9be8a5b8a4034c537633499adcd7acc832b460becb73c",
                "PlannedCore.__call__": "5ee61337ebbab597c5e7fee8e8a90c19bc50331a747ba57a93e7540e065c8703",
                "assert_value_leaf_layout": "9362aaf98344976ae13de7ac67e05d23fe983f12e34eac9f2ad439762a5d159d",
                "_assert_output_leaf": "8a75af878a85e6cc997e5d823b55ba0545ff3f147a30a7ddbfd2666db07d70fd",
                "_resolve_output_leaf": "b321849f4ff64dbab550be131a31a74157e9749a396958917b8e587fdffe5fa4",
                "_state_axes_leading_sharding": "275366296965e4160350df8a1ca9482d4985be24f394afd4a441bbf0fbd8404e",
                "_state_axis_spec": "49e002ddc32f703b089ecc147a6bb5296157f2c4f6313335c308e82f06ddd18d",
                "StateAxesLeading.__post_init__": "0bd0d18d5a0caff060ce3a860622a537a26aa061a164e9bdab4423b2ea4d2698",
            },
        )
    )
    assignments = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            _target_names(target) in {("VALUE",), ("DISSOLUTION_FLAG",)}
            for target in statement.targets
        )
    ]
    if not _statements_match(
        observed=assignments,
        expected_source="""VALUE = OutputRole.VALUE
DISSOLUTION_FLAG = OutputRole.DISSOLUTION_FLAG
""",
    ):
        errors.append("output layout: logical output-role bindings changed")
    return errors


def _value_transfer_errors(tree: ast.Module) -> list[str]:
    """Pin exact target artifacts through immutable source-core input adapters."""
    errors = _class_surface_errors(
        tree=tree,
        label="value transfer artifact address",
        class_name="ValueArtifactAddress",
        fields=(
            "kind: ValueArtifactKind",
            "period: int",
            "regime: RegimeName",
            "target_regime: RegimeName | None = None",
            "artifact_key: ArtifactKey | None = None",
            "leaf_path: tuple[str, ...] = ()",
        ),
        methods=("__post_init__",),
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="value transfer consumer address",
            class_name="ValueConsumerAddress",
            fields=(
                "source_period: int",
                "source_regime: RegimeName",
                "core_key: str",
                "channel: ValueInputChannel",
                "path: tuple[str | int, ...]",
                "argument: str | None = None",
            ),
            methods=("__post_init__",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="resolved value transfer",
            class_name="ResolvedValueTransfer",
            fields=(
                "target: ValueArtifactAddress",
                "source: ValueConsumerAddress",
                "kind: ValueTransferKind",
                "stored_sharding: jax.sharding.Sharding",
                "source_sharding: jax.sharding.Sharding",
                "expected_shape: tuple[int, ...]",
                "expected_dtype: object",
                "reused_by_several_consumers: bool = False",
                "specialization_key: Hashable = field(init=False)",
            ),
            methods=("__post_init__", "cost"),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="value transfer cost",
            class_name="TransferCost",
            fields=(
                "operation_class: TransferOperationClass",
                "logical_bytes: int",
                "per_device_bytes: int",
                "temporary_bytes: int",
                "devices: tuple[int, ...]",
                "reused_by_several_consumers: bool",
            ),
            methods=(),
        )
    )
    enum_contracts = {
        "ValueArtifactKind": """REGIME_VALUE = "regime_value"
GATED_CONTINUATION = "gated_continuation"
CONTINUATION_LEAF = "continuation_leaf"
REPLAY_ARTIFACT_LEAF = "replay_artifact_leaf"
""",
        "ValueInputChannel": """NEXT_REGIME_VALUE = "next_regime_to_V_arr"
SAME_PERIOD_VALUE = "same_period_regime_to_V_arr"
EDGE_REFERENCE_VALUE = "edge_reference_regime_to_V_arr"
CONTINUATION_LEAF = "next_regime_to_continuation"
CURRENT_REPLAY_ARTIFACT = "current_replay_artifact"
NEXT_REPLAY_ARTIFACT = "next_replay_artifact"
""",
        "ValueTransferKind": """ALIGNED_LOCAL = "aligned_local"
COPY_TO_SOURCE_LAYOUT = "copy_to_source_layout"
ALL_GATHER = "all_gather"
LOCAL_SLICE = "local_slice"
RESHARD = "reshard"
CROSS_MESH_COPY = "cross_mesh_copy"
""",
        "TransferOperationClass": """LOCAL = "local"
DEVICE_COPY = "device_copy"
COLLECTIVE = "collective"
""",
    }
    for class_name, expected_source in enum_contracts.items():
        try:
            cls = _class_definition(tree=tree, name=class_name)
        except ValueError as error:
            errors.append(f"value transfer: {error}")
            continue
        body = list(cls.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
        if (
            [ast.unparse(item) for item in cls.bases] != ["StrEnum"]
            or cls.decorator_list
            or cls.keywords
            or not _statements_match(
                observed=body,
                expected_source=expected_source,
            )
        ):
            errors.append(f"value transfer: {class_name} enum contract changed")

    version_assignments = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            _target_names(target) == ("_VALUE_TRANSFER_VERSION",)
            for target in statement.targets
        )
    ]
    if not _statements_match(
        observed=version_assignments,
        expected_source="_VALUE_TRANSFER_VERSION = 2",
    ):
        errors.append("value transfer: specialization version binding changed")

    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="value transfer",
            contracts={
                "ValueArtifactAddress.__post_init__": "cafe101592a7a7d2019ac1f2bc57e2e64295da6093a399dbb948bf8c38765e0e",
                "ValueConsumerAddress.__post_init__": "e2a6c26a492e21aef01bc5ae519d02426a0f6e067201925b2797eed13fadcc41",
                "ResolvedValueTransfer.__post_init__": "6fc7223a75c3dc5fbb6072004915bbf08a05c9d1d42711b12ba6aae4306192c8",
                "resolve_value_transfer": "232ec967f8b1c8c9055ae513e5afce130d808e8afe7f90271fc19f111db0cc1e",
                "apply_value_transfer": "76b11059204dc6786aab1701a6c2523ddedd4f6fff16d80df49b1428784db311",
                "apply_value_transfer_plan": "d17240965105a83382d4a0a71809dece1aa4770143729968bbc370177ea3e7e6",
                "classify_value_transfer": "c55b04af4530ac76c2dce1b47bb3ccf0d4056efa7c4dd68417257f59e03d506e",
                "ResolvedValueTransfer.cost": "b62e0d903fb8fb628e869b4fa90e089f8e4cc118bbbad34d8e654dfe0386f8e4",
                "_named_axes": "46fa78227ccbe7e1dd881030b05d16794d3f45003c37b638ba0badc434d977ed",
                "_replace_transfer_leaf": "f075cfa66700c236bae59a1d26278a61ac916e793e6b3cd78f6b7abfb4f64732",
                "_validate_edge_identity": "b9dc316bc5c544a59041fd3c7c67a48766829f66db5db31a6d1bb48eb96afc05",
                "_validate_replay_leaf_identity": "0da3b8851138ef9c733fa1b64977f7ceb243e6185b5e44fa49236619ccb2e779",
                "_validate_continuation_leaf_identity": "88493fd0b4b5ef9670c2958f580de3d53e855b187f3548e335193364a3cb61f8",
                "_assert_value_metadata": "7ebe6eb927b678ae88ed2a12d8f7571bdd8a86d5a4bf3dc55cd24f92da6e389d",
                "_normalize_shape": "346ecb8fac04e0e000005dbc613e197e374b60cd9b08d1a53bba790e04c3621a",
                "_require_period": "31f344bbdf23177b3a5bd9d2561bcbe4c9398814c1e7cbd6920afdbc0c013d68",
                "_require_name": "af49e5c3217919d353aa787908d9d0764ff17792b1d64aaeb08afa1f0a983628",
                "_require_enum": "4294ec9593c985870368a5d21f476900ddfffe66f42d275dbc02bec9708aba0d",
                "_validate_path_segment": "1c985002819fb5feb6148fede65bb43334852e7ac653e75322aa79fb15209787",
                "_require_sharding": "f3f13469e2413f320cc998e10097b00fca815669d417189e7689366b9baf67df",
                "_check_sharding_shape": "dba09ca24339aa961d7d8bc46a685c3f04c0c8b4669cc27482e0df08217c26aa",
            },
        )
    )
    errors.extend(
        _module_contract_errors(
            tree=tree,
            label="value transfer",
            relevant_import_names={
                "ArtifactKey",
                "ExecutionPlanningError",
                "Hashable",
                "Iterable",
                "Mapping",
                "MappingProxyType",
                "Protocol",
                "RegimeName",
                "StrEnum",
                "dataclass",
                "field",
                "fields",
                "is_dataclass",
                "jax",
                "jnp",
                "layout_footprint",
                "math",
                "runtime_checkable",
                "sharding_device_ids",
            },
            expected_imports=[
                "import math",
                "from collections.abc import Hashable, Iterable, Mapping",
                "from dataclasses import dataclass, field, fields, is_dataclass, replace",
                "from enum import StrEnum",
                "from types import MappingProxyType",
                "from typing import Protocol, runtime_checkable",
                "import jax",
                "import jax.numpy as jnp",
                "from _lcm.execution.footprint import layout_footprint, sharding_device_ids",
                "from _lcm.typing import RegimeName",
                "from lcm.exceptions import ExecutionPlanningError",
                "from lcm.solver_api import ArtifactKey",
            ],
            expected_binding_counts={
                "ArtifactKey": 1,
                "ExecutionPlanningError": 1,
                "Hashable": 1,
                "Iterable": 1,
                "Mapping": 1,
                "MappingProxyType": 1,
                "Protocol": 1,
                "RegimeName": 1,
                "ResolvedValueTransfer": 1,
                "StrEnum": 1,
                "TransferCache": 1,
                "TransferCost": 1,
                "TransferOperationClass": 1,
                "ValueArtifactAddress": 1,
                "ValueArtifactKind": 1,
                "ValueConsumerAddress": 1,
                "ValueInputChannel": 1,
                "ValueTransferKind": 1,
                "_OPERATION_CLASS_BY_KIND": 1,
                "_VALUE_TRANSFER_VERSION": 1,
                "_assert_value_metadata": 1,
                "_check_sharding_shape": 1,
                "_named_axes": 1,
                "_normalize_shape": 1,
                "_replace_transfer_leaf": 1,
                "_require_enum": 1,
                "_require_name": 1,
                "_require_period": 1,
                "_require_sharding": 1,
                "_validate_replay_leaf_identity": 1,
                "_validate_continuation_leaf_identity": 1,
                "_validate_edge_identity": 1,
                "_validate_path_segment": 1,
                "all": 0,
                "any": 0,
                "apply_value_transfer": 1,
                "apply_value_transfer_plan": 1,
                "callable": 0,
                "classify_value_transfer": 1,
                "dataclass": 1,
                "dict": 0,
                "field": 1,
                "frozenset": 0,
                "getattr": 0,
                "isinstance": 0,
                "jax": 1,
                "jnp": 1,
                "layout_footprint": 1,
                "len": 0,
                "list": 0,
                "math": 1,
                "object": 0,
                "resolve_value_transfer": 1,
                "runtime_checkable": 1,
                "set": 0,
                "sharding_device_ids": 1,
                "sorted": 0,
                "tuple": 0,
                "type": 0,
            },
        )
    )
    return errors


def _processing_caller_errors(tree: ast.Module) -> list[str]:
    """Pin canonical dense reducers and live publication of the program bundle."""
    errors = _exact_callable_errors(
        tree=tree,
        label="simulate caller",
        contracts={
            "_build_per_subject_decisions_per_period": "762d57a4dca8e9ce3cafc9725c81032b1a87dff7a2297593395cb5040fab1e2d",
            "_argmax_reducer": "d703712f2beec0f93e9cacbb67754faffe20bf4628e71979dba4dd8f4dd6842c",
        },
    )
    try:
        live = _definition(tree=tree, name="_build_simulation_phase")
    except ValueError as error:
        return [*errors, f"simulate caller: {error}"]
    live_body = _body_without_docstring(live)
    assignments = {
        "per_subject_decisions": """per_subject_decisions = _build_per_subject_decisions_per_period(
    state_action_space=state_action_space,
    Q_and_F_functions=Q_and_F_functions,
    has_taste_shocks=has_taste_shocks,
    stakeholders=stakeholders,
    pareto_weights=pareto_weights,
)""",
        "programs": """programs = build_simulation_programs(
    context=solver_context,
    Q_and_F_functions=Q_and_F_functions,
    per_subject_decisions=per_subject_decisions,
    per_subject_transitions=next_state_build.per_subject_by_period,
    per_subject_route=per_subject_route,
    simulation_state_names=simulation_variables.state_names,
    active_periods=tuple(regimes_to_active_periods[regime_name]),
    has_gated_edges=bool(user_regime.gated_edges),
)""",
    }
    bindings = _scope_binding_counts(live_body)
    for name, expected in assignments.items():
        observed = [
            statement
            for statement in live_body
            if isinstance(statement, ast.Assign)
            and any(_target_names(target) == (name,) for target in statement.targets)
        ]
        if bindings.get(name) != 1 or not _statements_match(
            observed=observed, expected_source=expected
        ):
            errors.append(f"simulate caller: live {name} transport changed")
    returns = [node for node in ast.walk(live) if isinstance(node, ast.Return)]
    if len(returns) != 1 or live_body[-1] is not returns[0]:
        errors.append("simulate caller: live phase gained a bypass return")
    elif not (
        isinstance(returns[0].value, ast.Call)
        and _call_name(returns[0].value) == "SimulationPhase"
        and _name(
            node=_keyword(call=returns[0].value, name="programs"), expected="programs"
        )
    ):
        errors.append("simulate caller: certified program bundle is not published")
    if any(
        bindings.get(name, 0)
        for name in (
            "has_taste_shocks",
            "_build_per_subject_decisions_per_period",
            "build_simulation_programs",
            "SimulationPhase",
        )
    ):
        errors.append("simulate caller: live taste/program bindings changed")
    errors.extend(
        _module_contract_errors(
            tree=tree,
            label="simulate caller",
            relevant_import_names={
                "MappingProxyType",
                "get_argmax_and_max_Q_over_a",
                "build_simulation_programs",
                "SimulationPhase",
            },
            expected_imports=[
                "from types import MappingProxyType",
                "from _lcm.engine import EGMPolicyRead, NNBEGMPolicyRead, Regime, SimulationPhase, SolutionPhase, StateActionSpace, Variables, _fail_if_template_is_misplaced, placed_devices_for_ids",
                "from _lcm.regime_building.max_Q_over_a import get_argmax_and_max_Q_over_a",
                "from _lcm.simulation.programs import build_simulation_programs",
            ],
            expected_binding_counts={
                "MappingProxyType": 1,
                "get_argmax_and_max_Q_over_a": 1,
                "build_simulation_programs": 1,
                "SimulationPhase": 1,
                "_build_per_subject_decisions_per_period": 1,
                "_argmax_reducer": 1,
                "id": 0,
                "len": 0,
            },
        )
    )
    return errors


def _transport_surface_statement(statement: ast.stmt) -> ast.stmt:
    """Keep bindings and class schemas while callable bodies are checked separately."""
    result = copy.copy(statement)
    if isinstance(result, ast.FunctionDef | ast.AsyncFunctionDef):
        result.body = [ast.Pass()]
    elif isinstance(result, ast.ClassDef):
        result.body = [
            _transport_surface_statement(item)
            for item in result.body
            if not (
                isinstance(item, ast.Expr)
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            )
        ]
    return result


def _transport_module_surface(tree: ast.Module) -> str:
    """Pin imports, constants, decorators, schemas and every callable's binding."""
    return _statements_ast_sha256(
        [
            _transport_surface_statement(item)
            for item in tree.body
            if not (
                isinstance(item, ast.Expr)
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            )
        ]
    )


def _simulation_program_corridor_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Pin the reviewed declaration → materialization → dispatch corridor.

    The declaration binds the same Q/F function into either the canonical dense
    argmax or the sealed C-order hard-max fold. Subject tiling maps that body
    without altering action support. Complete per-call arguments materialize
    exactly once; the planner compiles the resolved function, and dispatch calls
    its selected executable with those same placed arguments. Unbudgeted AOT
    prepares the same cache with bounded transient templates; budgeted preparation
    is deferred until live residency is available at the first dispatch.

    Callable ASTs pin these executable bodies independently of refreshable byte
    seals. Separate module surfaces forbid import rebinding, altered constants,
    descriptors or replacement classes from bypassing those body checks.
    """
    contracts = {
        SIMULATION_PROGRAMS_SOURCE: (
            "efd2ee1f5f1d5c2787d30e58c809f6164f78a56793894d83a56536a2dce66e2f",
            {
                "build_simulation_programs": "9232abe5d1727745bc3715e2b13c355abc057beea8bdd037eae1757a95f61e29",
                "_fail_if_the_streamed_reduction_is_wrong": "4393d8f3dc1dded01122dc6bf97e4f51c88e52f3804f79b08d600bc75496dd0a",
                "_decision_subject_arg_names": "3bf80bc49a6326e6be6369de80863482db1615e713b95ce476966e9562d6fde5",
                "_decision_value_reads": "7480309f78099994ba417dab87ff44c8ccc545f662855a44bc142bd3509cc1a9",
                "_decision_body": "42f84d94a9520d5462d81c7f4e1ae322f832b14a98945346f57469c1799d978e",
                "_StreamedArgmaxQOverA.__call__": "a32599a767896a0389a70cbac3e193c656abaebae1b90c237deb9bd7cc80c721",
                "_StreamedArgmaxQOverA._fold": "0396d37b300cace88652b5ad0c5040a82f3b45df2ef8892cea61e37d3e2ce793",
                "_SubjectTiled.__call__": "18e7c8d24d06f4d10915e6a770a61485625ceda42d158d9d24258c6ff6593ef5",
                "_evaluate_subject_tile": "9094e457bed0e17f6bfc4d6fc00ed22ccaa39bfa5a74a5b6371da7f6e34d72c2",
                "_ArgumentsBoundAtDispatch.__call__": "fd35ff6f3ed6f5a272f4aa291869b2972b07533cb051bdf86e00ca82c6edd34d",
            },
        ),
        SIMULATION_PROGRAM_TYPES_SOURCE: (
            "bf2290be45903028142ff3cf2fe6239595eba5d4b7208d9e06700a4f984d1e32",
            {
                "SimulationBuildContext.__post_init__": "00641d48094283340c57d2137f90f4568bd7dfa8ff6372e0348ccf6abad54a31",
                "SimulationProgramExecutor.dispatch": "7dc2a7b53175a8eb36dba1ccaf927391aaaffc12a3a913d695f16346ff38883b",
                "SimulationPrograms.__post_init__": "8a101d3c4179b03c7c0a0759a20e57c9c96535fa5e062cfac4736ddb97e443f3",
                "SimulationPrograms.declared_axis_names": "4de2577510af20c9e9c469173eee78444dbec2fb5d9e6463c0d475865e0a3ac5",
                "transition_output_roles": "ed772c2beff03f47113b71f5d3f0405469b6ef4bf67d4c3ad4df1a3f064e3fab",
                "route_output_roles": "81ccd2cdf1a29d1dcb3021d775c4abc8cb70364819a027303968afc69af19a2e",
                "subject_axis": "2d1dda5c95debf5b8c7d0a72c8fa71db6c42b2fb22763cc943ec494dfd3d942f",
            },
        ),
        SIMULATION_RUNTIME_SOURCE: (
            "ae82802787969da21e89856de64453944733967642a5696109e3c91403a135a8",
            {
                "CompiledSimulationProgram.__call__": "4329a4109ff7b367918e5570f8ef892aed246f2192a95825bbd48e01d60d6c96",
                "SimulationRuntime.dispatch": "b936af2fdda53d89102dc39ee6846e1c53072222a216aeeca3e69f34df201165",
                "SimulationRuntime.prepare": "86e18a7900a901875d89609c94cf96f75866f404a7d1f2c3834966b467019a48",
                "SimulationRuntime.is_prepared": "9495309ce3a74126c48f3fc04b517c738081fc7a1a8e20a8de3f0b2ddd28f6b5",
                "SimulationRuntime._prepare_materialized": "e1660f9fe46061be1e96201b8e2b4f12bcbd3352805428925a9f1d0ae97c36ba",
                "execute_simulation_program": "6248e86d6a967be348ab00114e7d036d6f40e233c80a71ad07dd4dab67d4667e",
                "_SimulationCandidateCompiler.__call__": "34118b52721ac3374cc3b9744b5196428fbb603551f6d297e1b6ab7deea9e907",
                "_with_subject_extent": "c48502d31e8f9fb29221122b0de6d10f38f3d6f740854a82512c2b964079e463",
                "_build_context": "d2de03c66739cd53f5c9e1f95f9fd0ab4c57334ca853e7a5e8f0422cf733852d",
                "SimulationRuntime._materialize": "ad4bea86c85d6fb3ac6e72f164a852c42f677fc3fab5e321ec34805f8c351aaf",
                "SimulationRuntime._require_budget_context": "6818040f38e55d623ae48f58b61ab3f1858d7ecd79df5b3f68321c364bfa6910",
                "SimulationRuntime.compile_candidate": "367181627553e0272f4fcc771adb900296d4c3409fda324ea45ac799e2803e0c",
                "_CachedSimulationCandidateCompiler.__call__": "28dcd8f5efac82c89e21143112cf45402a5dde3585423bd912f4c7b455e62107",
                "_simulation_peak_bytes": "d78167b86cd2b5e0c77fe41aa86ae64b66f4e372bb187f68f4893d323cdce78d",
                "_SimulationResidentBytes.__call__": "1de7cce7b963299d2803cc325d95389fe99fad8282a3b9a707bdddbb3dccd450",
            },
        ),
        SIMULATION_COMPILE_SOURCE: (
            "5ab22656078a47b54126cd1c9bd87566a09ede55ca25c9f8e1633bf9d6d8c8d7",
            {
                "bind_simulation_runtime": "18a84d95cd797683166f366e455a5c1210b039b11c0ecc6f8a040b16332f57ab",
                "lower_simulation_programs": "1f62ea281a355ab2f849207188f3d5fc58e504674ee299ae08d4b2b21be7a843",
                "_prepare_and_log": "6380580f956921448a5106ee53accfa712c9bf62bded1b224c18b7561bf63f67",
                "_compile_and_install_gate": "a0551505388660688c0465fdd862906f608840bc0189c31243f5cb83a8c90b5f",
                "_drain_compilations": "0d226343d92bb2cc8f0e9d661135cf92e332fe7499dc9c4e910e75fb723fac64",
                "_compile_and_log": "6d216963f57fe9fcc5182a4ca62df0779376d14613ab2755adca60860d14fbb3",
                "_edge_fold_periods": "7ac4cf626ea06a1a26c7744cec2647ec202afe46a39ce42bef6bf594013da64c",
                "_build_gate_evaluator_args": "cb02ee3a863169e02d119826d7ad4b721a2a00e210155ffa24a1bcfc5079be83",
                "_subject_state_carrier_template": "36bef4048f2f1a4cbac3f120b14380b348751bd4e862a6bbfa9aadbfba2c03d4",
                "_with_edge_substitution": "0085398d5da99b120b93658a96b488e0185c2453e8c98614d78dc09428487119",
                "_build_argmax_args": "812d30ffdd0073ea9353fd1b25db27027a1a54e94c001c114ddb043d0557d7fb",
                "_build_next_state_args": "917d68840e18c2750b508761601bbae2ef4c049e6323d63ff56097c475b71e80",
                "_build_crtp_args": "59fc3bc51d17a7d5ac6ded8492ee1dad4fc0eb76eeb757809618177c28fabb56",
                "_simulate_only_subject_states": "5bf91104667ed6f854fdf9fbebfab497ff37ddce57caf644344e042ec8785edb",
                "_subject_shape_arrays": "e1a0a530003382d3defe2810eb3c39e1684f580ab6a3878d492825efd25307b3",
                "_subject_devices": "7252a005c9a532853fbd2c821bc593a0e52213288271000c4c2820a0079ed7f6",
                "_prepare_edge_gate_evaluators": "4e9c8410228c8d3a65a4aa018d7198d063b6ac6019b81b8b0a8e106833345d51",
            },
        ),
    }
    surface, callables = contracts[source]
    errors = _exact_callable_errors(
        tree=tree, label="simulation program corridor", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(
            "simulation program corridor: module bindings or class surface changed"
        )
    return errors


def _simulation_dispatch_corridor_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Pin phase publication and consumption of the selected decision's exact pair.

    The live caller passes complete action grids, states, values and addressed
    random keys to the published period program. It decodes that program's flat
    index through the same completed action grids. Model AOT publication and lazy
    dispatch share the same executor; phase schemas cannot substitute a property.
    """
    contracts = {
        SIMULATION_SOURCE: (
            "e6ec229f947fcfd8826271670ca17a532fdcc4d5897c9d78ca3456f7c2ea4750",
            {
                "_simulate_regime_in_period": "2d0ede012becb7860aef63976765e1f9965793b449338399f7ea3ad238208448",
                "simulate": "de24d5855fc9ca1cd45fbec89bafa783759bc66ecc0f5dce5fcb6d8e0a117c13",
                "_simulate_subject_chunk": "040cee82b493bbc8630962270a8c692c08186085f8c43bc8108ea673d53b2af9",
                "_bind_unit_executor": "8a2a8bc6a28f5d7c6a4a2c8d3bea6db86e79c211a8a9b2a409336126a7572a66",
                "_lookup_values_from_indices": "ebc4a036a447857f061c117b2eb0c9b9e61d5f17a40e90ea14a6e6205233ea9f",
                "_read_external_replay": "7972d2214dbda0c4173fedfabe12b9ae3b03ed099db9f950cd176ee082131c90",
                "_initialize_chunk_state": "82b413fe657adbce560df50fae814cf80a268eabc0839cbb99a2242c4b2ab482",
            },
        ),
        SIMULATION_TRANSITIONS_SOURCE: (
            "8eda19c2d7ca972338f756ace688e6a037fc2e469c3b15614727e74bf8db46ab",
            {
                "calculate_next_states": "1d896f682a473fe16b75c9546929ba657dae9e292ab99f99fb5b8f0407232b99",
                "calculate_next_regime_membership": "061071105fad806a8aa8b452e853f2fb2048476580eb712732c8fd3ece993998",
                "_update_regime_ids": "93e8f8c1aba6e42f14269aac5d911c722e5e132c207cae6fa3e07bf138a58163",
                "_draw_random_regime_ids": "87fa7035d1095f0dc1a17e51e517ef559808039ac8a107864995ce94f2bf7430",
                "_advance_states_for_subjects": "a7ea5a0c113ec1aca76552c7d54502133076bee2ebe96052299ef46bf9518547",
            },
        ),
        MODEL_SOURCE: (
            "6c848dbfdd6217204bb3b50b1e45b0de9328aabb4fdb596ac32ddf70af6ffc1a",
            {
                "Model._resolve_simulate_regimes": "5a11f26e52496d8210ae0ab2b9846d5ca234569764f82d47e871f26e33751ee2",
                "Model._runtime_regimes_for_shape": "b85ceab93d6b925942a9d577c69afcb4df55bae3beb6aaf2220e2249d24697f8",
                "Model._ensure_simulate_compiled": "2e59224333ce9bd7453608d05359e933217971dbf8867e3f9851b9f503a2df86",
                "Model.simulate": "03aae0b664add4294cb63aac980e9446afc5dfa5d16a64492039312191b8cd0e",
                "Model._build_external_replay_readers": "33c41f2c3f7bd300f3e0b4ac042f7371b93efbfe1762e5fe639fc854b19f1f5f",
                "_fail_if_invalid_taste_shock_seed": "5a8c4643d73c99c83160024bac7ede90deb9da24141e750103ec48eb76ae0486",
            },
        ),
        SIMULATION_RANDOM_SOURCE: (
            "b52d2661652f8b3924beb34a3e04a3a207ed5584c70a532c83a2d5312cfecf17",
            {
                "create_simulation_key": "76fca4fbc2e18885d2ee67512e8207616fe8d17443954d9b36765a4bc52b3dd4",
                "_create_simulation_key": "1ab7921d3e42ea7255fb8bbc60d1ed58e408e8818d0bac91ec132c276c62a05d",
                "split_simulation_key": "9968b4530b74638b082b63c0160391573b8ae4c20a9dc3b733a9b424de259921",
                "_split_simulation_key": "666d1e23f386815a8fd331cd543813fa2fcdb501f7672a82be9f48fe31c293b2",
                "generate_simulation_keys": "a0ada0cd4a1e738dcf51b6966b081504b19f86d9ca5a344111ccc25ee7daaf89",
                "_generate_simulation_keys": "87d86ba01310209a8ac5b222c2ea6aebc8ff4488b2ba373b1b28d5b682c737b8",
                "draw_random_seed": "42fb402a3282c789664d45fb988b3ed07b8cec706924d3e6e11e26b44c393502",
            },
        ),
        ENGINE_SOURCE: (
            "f70c863c395af0876b56a9c201980bfb8421e7fc56322428e6ca235376449009",
            {},
        ),
    }
    surface, callables = contracts[source]
    errors = _exact_callable_errors(
        tree=tree, label="simulation dispatch corridor", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(
            "simulation dispatch corridor: publication or module surface changed"
        )
    return errors


_SIMULATION_ADAPTER_CONTRACTS = {
    # The entry coordinator preserves both validation families and passes the
    # real retained inventory to each newly profiled summary allocation.
    INITIAL_CONDITIONS_SOURCE: (
        "eb55929b8e8ee3b0ef22e204e7110331435f523ecbab40af05d93292d933126f",
        {
            "validate_simulation_inputs": "db221b7a09a394ea9d6eb5e1c0329d4ff61a78737de85ea604b5467ba9179f86",
            "_preflight_memory": "24af3b888fd7f85f2f580b19b08c0ed565ed2bc3da34ed883c4e7982f5bc5caf",
            "_discrete_initial_specs": "6e837939569bad3b27093ab4a14ce38ad16756f86910f37127432d07291195a6",
            "_pack_initial_summary": "504e89911be3016190066e13c8acebdb890328a4a7d40c4f9e8afd6a6f0b748a",
            "_read_initial_cohorts": "c2de600ad5640aabae10671db6123ec1dcd7dac80b9bf50a8a21e022d30838ad",
            "validate_initial_conditions": "41eb24fe5620d7482a6e5d35e9b02d23b85b9a4a253e28bd14f170340ea64d70",
            "_collect_feasibility_errors": "4b3f79dc5f3eabc418e9a22cfbd8bb19d215c2bb7a27364ce36e20e35e022307",
            "_age_specialized_feasibility_message": "1eefe701cada0b668ab2eab2e404120b47d2c65ff390afeb1e726ae420fde174",
            "_check_regime_feasibility": "75a10480a8c9f5d9eb249295ad232dd2d42e1b8d91985b86e3f16a9a8b27c1be",
            "_gather_feasibility_inputs": "042f8474176d738b9219516fd96c3d09c5ccc3e4ce18d0641bede57a8ce7d40e",
            "_subject_feasibility_flag": "58bfc237bc5b8a51897de10ffee1eb33f7a2cb6504c094ed48544946185bd404",
            "_constant_feasibility_flag": "f7b994fa04576c7157f8bab30ef0e17126c181ba874cbe493c1e19e6be056f17",
        },
    ),
    SIMULATION_TASTE_STREAM_SOURCE: (
        "947474b7e88fd4664df411678ada61eaf7c026cf7bed0abb6fb348057809bc93",
        {
            "create_taste_shock_key": "d0b1e91beb7a9d482b7cc520dc00543112c082ff285f37e17c8ea1a283685d32",
            "prepare_decision_taste_keys": "ef085a5cfac60dce08367fc50b280500313659e270771b1e94367b73e3ddf717",
            "generate_taste_shock_keys": "66f0dddec645c123e66957e33ea5da7f932c83c23822dd9a49c308034339a038",
            "advance_simulation_taste_key": "49eecd1a5724049acc7108921c86640236140477924fbf3b561156c7a449381b",
            "build_taste_stream_addresses": "b880a2eb4e2eed50de19654b6385ab4a46e68530fbd017643cf1fd599a47947d",
            "_encode_subject_row": "9206dbef80b1bf9607d6cfc42db24af9d7a01ee34ee04079d13c61582eeba82a",
            "_advance_simulation_taste_key": "d8854c351a6aac51cc858691332541a21413b90baccf6dd2a52fd2ae601a3752",
            "_taste_address_words": "4490c80c002e777c58e57d242f5a722e17f31d10e7d8120e32b31123ea284478",
            "draw_taste_shock_keys": "bdc7eb50b64e31949af64b051935650b7f744467074adadc2dea930012d9d577",
            "_row_offset_words": "5b7e8a0ca6032c00fd2db2d24513c6642a270e20d98facc362b45bc784d80778",
            "_fold_subject_key": "6c3faf14f43234d25195bd3525659c095512857f4b9d4422e05e43e33d936493",
        },
    ),
    COMPILER_INPUTS_SOURCE: (
        "ab4df3a33f7b46329e2eb3a4ac7c6a96c75560172f7528e2874e06bed37dae4d",
        {
            "compiler_input_paths": "2e9389aca0e3ee57a2a79fe4ef17ba9f09f1fe8eca8711ccfb3e4f5f837d2413",
            "_is_none": "70af7d997029f74d8784c981b5512e4e6e622af65c8d1ba75d0f55eebafc1a4e",
        },
    ),
    SIMULATION_MEMBERSHIP_SOURCE: (
        "a57a98eed95f797a3fe7a1a4337db52ecdf78785af4985be4118ae5a4b76a556",
        {
            "initialize_subject_membership": "3d9e4107e20e56414489ba06572036dc00f3f950581dcc239e9f1f43b4d3be6b",
            "activate_subject_membership": "277a6e91b924ca50b5e4bdfc3b5e1d58a2131ddeab82c80838702f7183c2a960",
            "_empty_subject_membership": "9176841676cd38a179ffecbfdd0b78aa8df87a385b4503dc199cfd7776389a1d",
            "_activate_subject_membership": "f75db18f16aa2f67a76a5e88cc0b9ab683638e364e97f80685b3ef154f54cc71",
        },
    ),
    FOOTPRINT_SOURCE: (
        "3435d0f9b6f7f6f223374d5678f79343e509e480238bce3df2829f4e748c7726",
        {
            "ArtifactFootprint.__post_init__": "508bd37bbb30d5e0511ddb2de9d98f6c1c2318ab928635b73b413c942d72a4a7",
            "ScheduledUnit.__post_init__": "2c144891456c3626aaf1b62591a35e6c004e5d182b1ceb12dab4ce10d0b4d7d5",
            "ResidentInventory.__post_init__": "c3c42a4df472d8fab0ff8c1132f6e17cd7a1dbb578c7fc9f9898d8a4de553804",
            "ResidentInventory.resident_bytes": "46aa63f56074ead0c13cebc130e4667dcb73744a1c0a2edfd478fb835a7b9b92",
            "concrete_device_bytes": "14a88daf386078b838eacdf48a33e3c9894b3e958894e6973be7f823b6ffaab0",
            "plan_resident_bytes": "e9c46841e6663fe1e7b33a952b2bd7c08335e87e70ca813cd0e3314068ab985f",
            "plan_resident_inventory": "ad8c2a316cade31c67c6ee4ccfcb13d93334971bcac34a9bddb65b39473ece8a",
            "per_device_footprint": "35c8fc0f90d8a9342d58a87f2286cf08426043a03b3dc5577f4e13adab75cfcd",
            "layout_footprint": "6d4954af608d5c781e8ae3093a435d745e989333cfa9f3d62570ca366c6a33e6",
            "sharding_device_ids": "c66e89e2760e52cbdda063cccb1e45e26f78a66be5f9bebb97b28a076261e049",
            "_walk_wave": "00277db02d7a09bb58776c3b9da6cbab067c49e0079c97c6b563299f1fcea3d3",
            "_walk_period_folds": "b24a1e035b26b6aada8169115aa7d2319f769e5e7ae7a4530c333b56eaf9c8b1",
            "_register_outputs": "53c8b7518505e12ff7e2c670a90db42e7d61cf7dcda3591d0a9fb33ab3f2093e",
            "_resident_inventory": "8f2d1cb122164804dfaeaa79821253732f556dcc65d1b0b82ad4b13a60556f9d",
            "_device_bytes": "9a13b9a28cbd04171b79b314b94c4ca52c82b4cea75a508477aa59f2b1651a1e",
            "_group_bytes": "d3921a94db53838b762ad28d47dbd821ccdd061c9b8a3111a531edd6dc5ace3c",
            "_group_is_present": "0ca9e11967bb6f3677c2ae4f18b306a39d35ff58a15d96ff75d8bcb48d507ab0",
            "_group_is_consumed": "be4f8158fb0972735e2a5609c453744a62bb5bbb8084de5ea2032f0723cc6539",
            "_release_after_dispatch": "7c60f35772266db8d9a97c0793b5fb8f15aafa41570f8c24a9c7d12bb4180005",
            "_fail_if_footprint_is_unplanned": "7e40a8e62d5394d451086773fd4bd9ac08ecde2f5b3ed244c2ad5a8c8826e511",
            "_fail_if_period_disagrees": "238a0dde28594bfb1a48ac4eccba31453144c123bebe3c61d733ed1528a11c81",
            "_fail_if_negative": "aee2b71bd25be15a27a70981c5c9f27b78f67a3b53622169be9512db47b300c1",
            "_fail_if_not_a_device_set": "9d500f39fd8e6e8486888d403cc0c4a80c9c292f35a33644268656270b93d9b1",
        },
    ),
    SIMULATION_OPERANDS_SOURCE: (
        "413d8faddf9b69c2ee63be62f4da376c5b6d32b555c8940ee22b89fca2cddcd1",
        {
            "SubjectArgumentNames.subject_arg_names": "46cc9cd2b5af3c1a622819cf136f77ca33152cb4dbe6a00b2a111d50848d1f4e",
            "place_simulation_arguments": "fd38dfdc116cc2afeb8bfb9f1e1db13951b4ef4038694a5b9548f1f8e2826b84",
            "_require_operand_headroom": "b5c9e594752891c3fe1c8a5356cf9232dbc3cc8611cd8435985c5e4813099c31",
            "_required_operand_bytes": "59fd695f16874fd63291f8e13dff55411252c5fa6ca72b90260737ac3abf78e6",
            "_operand_leaves": "16b5429e78fa8a8bd71b2f7b232fe2e276c567c40415c40623301b02c4137d06",
            "_paths_below": "b12d8a06f78558f8c618e6c5e3209bb42e0011190230e7a68e660589ad961d52",
            "_place_operand_tree": "4e5754d22e67058a3acf38d11f9e24a54dd7bd2e4a77b51692ca9dac9ca84fa4",
            "_place_operand_leaf": "2e245da8cd56498ca70f0f58c7f21d10964c820335842509cc426c65adc05dc2",
            "subject_operand_sharding": "7f408b68a50569861eac3e980e9426df3140e27fa65791e98a1cd4e8385de7bb",
        },
    ),
    SIMULATION_UNIT_SOURCE: (
        "cb5d5f101cbc8f99d94a434f0a22720a6c8f5f386ec465b70940b686bad47233",
        {
            "SimulationUnitExecutor._live": "68d7baf210f0c7b7c027d9651e180bdba46d7059a7426577ebc27b9f28ec1ff6",
            "SimulationUnitExecutor.dispatch": "b723f14d6d52fcc4fd650ac2faebf76515f2afdc03113a5986b7882e1fb17aa2",
            "SimulationUnitExecutor.close": "5778ba5f7d0fd5d53898d15913ae8534b4ba43b8de7ffb25b5e79223174f6d80",
        },
    ),
    SIMULATION_HOST_SOURCE: (
        "08056a9f79f36a104d8d2d922541650a1a682e0fbb646dca702e8b4ef66d7de1",
        {
            "ProfiledSimulationOperations.dispatch": "f9902fe6a00e363bb238abe3be592cd56e05037db500bcdd5d5c88e1ab5e4a11",
            "ProfiledSimulationOperations.compile_candidate": "beec6504b97db1a6dd3456ea28534ad9b8dda2d05c7c1f32535a6e82b3534c45",
            "_OperationCompiler.__call__": "4f2d9bccfcd9524fef520a64eef658fe2f88484ff7443700e0b987f62f0e9913",
            "_abstract_operand": "72b7f4217b9ec8773711a592ba298526bb7ac32e691393d6aee78d2b8300beba",
            "_static_identity": "23489d8dc1669d265b3206adce1b77cf1cd2d2162444a2431ec141f3b0ce317a",
            "_operation_peak": "e103a5109d610acb69e4b52428086d1f39e7be16cb72c818eb468aea9fb79b2e",
        },
    ),
    SIMULATION_MEMORY_SOURCE: (
        "cbcc9d23b3d9bd9a2204eae258a039e79c2c27932e223430c94d5ef8990476f0",
        {
            "SimulationMemory.snapshot": "6c4a5f9c11f7540cfc6bdf495fc1a3b3dd1bbc1b2ddc14ef4a058733bf2c344f",
            "SimulationMemory.set_chunk_inputs": "1132e41412c470c63b70847dbaac4a22bb081fcbda2be1b7406b1857703270f7",
            "SimulationMemory.publish": "d8b7ef7cca100a9635d7f9b601a7c22c00a17fe5ed7e8033c2a358582a88423d",
            "SimulationMemory.replace_outputs": "4c7488bc6cc184501c1009a48c0f3b4d73b836f0ad56ae5af1d311712f8ec4d8",
            "SimulationMemory.set_derived": "10e766ae18a8838bf4d202f2ecb594c76377ee9594759b5f945604815e1479bb",
            "SimulationMemory.hold": "bd646a4bf1856382dc29c6487c7f11b7a0df8475e13eaff559fa349ee33632b3",
            "SimulationMemory.before_transfer": "7dc46a79cb4f6d561edc90798900314266a3aebb09bc219896611b76d53f0912",
            "SimulationMemory.check_resident": "e0f2aae1b218f4c67ee1b3ead1bd7771ab7740e0bc83ba26fca648e7299e372d",
            "SimulationMemory.run": "d6d9cd5147fecc8eb6c31629826bfe562f5413b510590af04836b978bcbd07bc",
            "SimulationMemory.close_unit": "ad93e305391d4eea94af93d4561ff2ff0a699c691f24084f5f1bcb63ae589838",
            "run_simulation_operation": "2c0982ac9b27ad92642cfc81da98c9cb4678c20cad7e130d281a0e43f2fdc1d3",
        },
    ),
    SIMULATION_PERIOD_INPUTS_SOURCE: (
        "3baeebebf1d4693045a911202220c4effdb83c630b7f2ec0ed0c6938dd7e7994",
        {
            "decision_reads": "f0bb30423ce4d7bee526d21e86cb0ca10851979b130f9340dd350274732f5573",
            "unit_value_reads": "f493f8fca6a88fbd3e69a9820ddc53457cc13ddb25692dc0974f203686222971",
            "gate_reads": "9e57467e4a7147bb1de88c0010c73960fd3b966ac76b3e5e575bde8791edd261",
            "acquire_gate_inputs": "a49be6fadbf76e8f8b2fe9b289049711ec3a6fe48851caa0fe31467ecc249abc",
            "acquire_decision_inputs": "752a3aa7e4de32488ec11d44725744c2eb31fdde30ba4ba87c71a217bcfc4ec3",
        },
    ),
    SIMULATION_REPLAY_INPUTS_SOURCE: (
        "b941a23abc378d6908b6f270e429ba226c569a65697b252e410867bd9d8b5053",
        {
            "replay_payload_reads": "e1b545e6577d7b1056e5d0d3884bab82dcd804af33135b3e78dbf3244f96eff4",
            "place_replay_payload": "acccbca62b49189e11f8e84426f8c37b39e8bc9e5648bb8eaa9ecbd61e2e7ee1",
            "_payload_read": "e21661556db28a965ccb171eb9ee1f109992bd81a7c25fdc82066c550022e4a3",
            "_consumer_step": "2d11c2e3653d69682cc29d7f29d95525f81749f3911869baf8ae68d178c23ef1",
            "PreparedReplayReader.reads": "3a9de61be4467a66b9cd87633adaa6be14787697bcb60816796972ad066d0dc5",
            "PreparedReplayReader.build": "81f1567656c679ca538b5fca9230befbd96bd189db9f2a07d4cf12823353ea6b",
        },
    ),
    SIMULATION_VALUE_READS_SOURCE: (
        "b0ee224746567aef12dc8acac13e80a2f02804bcfa1279e00919de272e3f7734",
        {
            "BeforeValueTransfer.__call__": "82a3f70c8e1a601fc8a5e52ec2b029488fee92b0f4f6a103c21513943f3688aa",
            "PeriodSimulationReads.__init__": "144f549e3f616445cf8faeaac7bf0ba6eed79560df2422b0db0ca7142afd867c",
            "PeriodSimulationReads.read": "70d61f5d54d7d047b0942f3694ba15968aa7160d945f6bc0dc231cfe5301ece8",
            "PeriodSimulationReads.commit": "160a8585ecc850b713e70422c7124725e3ad3045812ec167eea8ea2271209bca",
            "PeriodSimulationReads.live_values": "14324293d4a4ed4e3853ee801aa47bdba89b38fbb91559daf7aeaac2779a5c32",
            "PeriodSimulationReads.finish": "8097897dff0922356d53736d4c84c2f814bdd64aeca9791d12d4e10ebfc43c95",
            "PeriodSimulationReads._read_host": "414fef3fd580133bf176b4a260185084322127be9f4e6517b9c714b5fd1f67a1",
            "PeriodSimulationReads._check_open_unit": "057505de76b8bc662e1d69a1ba83e5021cca56afb01ab67423c659cff4a3f640",
            "_host_replay_sharding": "aa665562686325f36892f458794d1efa178492c4611e72e11817e52e2a60c739",
        },
    ),
    SIMULATION_VALUE_PLACEMENT_SOURCE: (
        "9dfbadd65ed8404b9d058ca7e2541f179c1fdccd126c1fac1f40d69a0abb1eeb",
        {
            "simulation_value_sharding": "529cb86cb0e1cfe3c72cfd2e5e8cbb3e235d717649b971085aff5a5da893eca2",
        },
    ),
    SIMULATION_CHUNK_INPUTS_SOURCE: (
        "930c7917f4e71d0c91d4eb52e9207bffdb2b433eeb1e2ae60a2f1f61224dc032",
        {
            "prepare_simulation_chunk_inputs": "994f93b22d316bf7709059c323192cc32df53f3b4637c9eba44e7bd1abab7da1",
        },
    ),
    SIMULATION_ENTRY_INPUTS_SOURCE: (
        "39ce39b1b492106b807d61567b78df58eb9091776d2421a97dc6c21269cdf263",
        {
            "SimulationEntryInputs.footprint": "54cdab53cc743f08e647b64c58fb17500e927aabddedaed36a2cb38d8be543b9",
            "capture_simulation_entry_inputs": "1f9530b0127480aac7b8257fb03510dad8682c81c8bec6f86dfa280f41f158bf",
            "_caller_arrays": "b5816c3bf9689817cf2cff29df7bba818657ddf0baaa182036946fb3f9d1324d",
        },
    ),
    SIMULATION_RESIDENCY_SOURCE: (
        "04bdef375e7bd307bde0a9f9e96e9c01e0e436050894a3a927b404de95f4ee60",
        {
            "DeviceBufferFootprint.__post_init__": "6b91013dac5e169e65e44c92701ce26bbc790d4e51e49b73b96725b8c02b9cef",
            "measure_buffer_footprint": "99ba8ec6f6b1496b580ecf82d4d979f76441aabedcb1872af22f200678262c61",
            "union_buffer_footprints": "9a01fcbfb0768c582f9e8529416ce190c26756d26052e0f5f8cfffef7d954a3b",
            "resident_bytes_by_device": "7f98f5a753acace9389784c2200b2b94375c26cb7523092c10f09e890dee2ffa",
            "require_transfer_headroom": "f7a7f1d9690273661ae12293e289ae36424438ce27ec8e247f6b05a769ee72e2",
            "_merge_spans": "3cdde57f56837f60963871f3ae1f04ca2f92f45a7ddc4d1c35bd3b52f2ead33f",
            "_uncovered_bytes": "ec6738bad93d29c3c742d696b5cabce48748755ac79ce71fcde583a94442df52",
        },
    ),
    SIMULATION_GATED_ROUTING_SOURCE: (
        "76e63ab700d062764944cb55dec6a50947bb67a9abf0f6bd1cba8740c76b4c32",
        {
            "simulation_gate_fold": "8cf8f48b2dffaf19843784eeea95850b7da45923eb54446bd2233c5804b02b14",
            "simulation_gate_route": "9c8806f1da92742c9368b257c39895e08f304f42068aae51260439b28bd63f93",
            "substitute_gated_edge_continuations": "8bab41cbc9c3b79c121b041b041f5363d21d1be4fc9f5ef50e9ea4cbdf301860",
            "route_gated_edges": "fb145cef7cc7b3f5dac024db7292697ed9598f7b6a45f5d6fcffe57a18943628",
            "_per_row_leg_outcomes": "adfd58415412e5f4ec1e734d52514531a9f21e758fd65ef58a0f4f40ca7dd668",
            "bind_provenance_params": "dba3f4248af81f4151fcdb79b5053b20b4a866da1903a0203d774af32cf31465",
            "_call_vmapped_with_accepted_kwargs": "730413601802a7a1c48cb48eaa6aa6115c9aebaeb04ea8d59463cc5449cbdad3",
            "split_population_call_args": "b779611293e15e0e7ddb15ce213f7e158ebe7edadcc9c62a84524d0bd66e9dfc",
            "install_population_call": "3effa9005aac7f577374c9b2c6278299da30c362020b9ce43e95489dad639519",
            "_accepted_arg_names": "5edc187eeed7499d363dbbe554d978d04cbf1c63a951bf07d1b49e58146c23f9",
            "_role_code": "439c955f56094cfce8195a6516abbed95c0a5990b6ac8cc8e8b85ca177b7e6bd",
            "population_call": "a0f1f85f21a4d9aba2b4c74275008ae64f254a031674b8562c3c42a1e9a3926a",
            "_call_one_subject": "31a86a138a27481b19a238ea0bc680382ccddcd02d5cce3ee3b4e2e7db850ece",
        },
    ),
    VALUE_TOPOLOGY_SOURCE: (
        "a821930e2607c4081a1621a66ac46188a335a8f23581894af9b4870de4de88ee",
        {
            "expected_V_rank": "3fb0dac3bed9027261436cff2316bd122cff31e99dd1b825977272d09e6212f7",
            "placed_V_sharding": "2232aa6bbc1e01337e19bd39637a254e34d14e9e6296407c2aceb6b2467333c2",
            "_get_regime_V_shapes_and_shardings": "accb519b857b3ff5743d9c076dbecbd0ce15538028d9217f57e6a33cf4662359",
            "_build_zero_V_arr": "83c861c7149e4ee341e409c56e8c08fe2a8a01d859bb8d6e80b7ff62e04a632f",
        },
    ),
    RETAINED_BUFFERS_SOURCE: (
        "394e2c6abaa4e8b59852f53358fb5005572d79fa46638faf4461dfd74a42fefb",
        {
            "retained_solution_buffers": "75b31cd7df576899d18b203e4963dd06acd47b900bf1b6c13c678c62e9d12d58",
            "_RetainedBuffers.collect": "3388eed966975167c9162aa814f2870da5d61cb6dcd2a6d217ecd7215065481e",
            "_RetainedBuffers.collect_lazy": "ddff92c86acac4cfec6093996a8f2ebec04cdd0a3ae2618e8d254b63d20426f9",
            "_RetainedBuffers.collect_authority": "402406b70f1437e1d6427b96af3bea127d3ec5906a9a97eb9b19de07c8c0b081",
            "_RetainedBuffers.collect_reader": "ab65de29bcfc68b9a694f47a926115124c4ee19dfd455b1c2f4a9b58317b7d99",
            "_unsupported": "5eb49f472abdea16b95d9f0c9a4f6e56297f0cf65466d18cd87cb4ff530dd21b",
        },
    ),
    SCHEDULER_SOURCE: (
        "1ee33c5b8472f2e50cb6c0ca944865e0f65d488b85d84ad4a78c3ccb61b46488",
        {
            "buffer_identity": "773d4a78840d9f58d52a23420af0b9942d22ce1e7d84af0d6e84fb3f0207716d",
            "shard_identities": "5dcf1a2c1f363f904cd238cebc7607d4a23350cd2ea7e7673c93a4df12334559",
            "shares_a_buffer": "d08dc057ffb25ffe0583d85004270348f45209039cddbd99762d9eac9166dc31",
            "BufferRegistry.__init__": "929753ce0b5f18af3d68aa592eb75102496f6a9c5307d27e068f9779b548d913",
            "BufferRegistry.declare_not_produced": "84f91c8e3740a64ca028ccd0ea11be158e6355972a07be943a6aa74a55962896",
            "BufferRegistry.declare_passed_through": "4e02f1db9ca9d2fd707600a6ab91d9d7239b9b651705a17dc4da6ee184136f7c",
            "BufferRegistry.declared_shards": "b1b775cd19d0837f801d0cb95e97d49aaeda1df9d0a291c831a3d0f65f82a33a",
            "BufferRegistry.is_not_produced": "cea14b09a1cf43447b882ceff465849c21c3bbec5d9a7432d21b39a53e427c4b",
            "BufferRegistry.register": "09edd35bee4350c467c8e224d6fa9a17cc62a8c55c9d870961c03a2e777275e8",
            "BufferRegistry.artifacts_sharing": "477157e8be682555c1c2c9010c0d530a2303b3b49f2486144e4863ba4add7078",
            "BufferRegistry.forget": "9ef1e2b80e6713c961ae890a09e95af5c1ff6b1ee4b4a55747279bac9497a8b5",
            "BufferRegistry.forget_identity": "de10a1a351d0d573c380eb0db602cc2ea3f297be95df5bc18c5dbf5e1c63c9b8",
            "BufferRegistry._prune_dead_declarations": "bc21c180ad19b29c98a38972521639dd454636f572602ffd68529e7689b2e4ab",
            "release_closed_artifacts": "bca1a611edf7a6e9fd4af7d634e35d820438a8a6b77d5b042c383cab847bab88",
            "_one_delete_per_shared_buffer": "c6824683620084bbb6acbd667b2657a3ba1afdcf425268d7f1d384ee4b308f71",
            "plan_period_waves": "0f2995c211c6472a1f378a69861a15181f6f2e0856db85191883176788017999",
            "replace_leaf_by_identity": "12231a2d73d86bda8fd3611d877f8b2d23bb2006a4abd0134d480c88e7426d50",
            "PeriodTransferCache.__init__": "72346f56ffc5316a0871756fd988ec73c9274daefd52c1e72370114d2163e569",
            "PeriodTransferCache.get": "f16cad2377317e84a219306a4cc2669dd74afc472906974c751452976ae63041",
            "PeriodTransferCache.put": "28648ac358887b0434396849a8827980e1f7be1249a0b8a0bf43416a0f1ea8d8",
            "PeriodTransferCache.commit_consumer": "1e50f0f76ca23c95019747cecc025601d103b6297ccd8df906ed85ce3c59b8df",
            "PeriodTransferCache.__len__": "9fb60f4b369c44a05b578b7d992aab8b3f956ffa5c5b465715beceba5fe2728d",
            "_add_declaring_array": "8cc69ee60ad06411506f95155fbef57acea86a11ca7246182e25c62336ade59c",
            "_keep_live_declaring_arrays": "120348a883a1615abc6106151a8da9b6212cc82b0c5574806f4e08ae609f3a1e",
        },
    ),
    LIVENESS_SOURCE: (
        "ef675175e370550bf000b240702edb45f755588b88ff9308634756a7898f97f3",
        {
            "PlannedInputLiveness.__init__": "77663425c9c46d7ceeb6befac1ab122018b8a05f12680906322c7e048823577f",
            "PlannedInputLiveness.pending_dispatches": "a24e02872f9504560a6fc23550115d0f1b45c67ad2436659fe400612ef328849",
            "PlannedInputLiveness.remaining_counts": "989f7890f975e8ec088e58629a075db4223a4ab2c838ee8af82e52f919a93b09",
            "PlannedInputLiveness.retained_artifacts": "3bf4021d7ae1455fcfdac1c60f13ec0997a34f69dd451c646e13761616e6f494",
            "PlannedInputLiveness.aliases": "0655f5d6cdbc718cd5e57d930cc6c4b3bdb2a745ff19dc95437f68a6422003ff",
            "PlannedInputLiveness.accesses_of": "d5e07768777e51024a01c99334a0ceab92e753793731a60414993a6d65ba3244",
            "PlannedInputLiveness.is_known": "4c69d8e584489542b5218b9529f2d3174ae437cc0ef98c99f9c6bd752b69a4fd",
            "PlannedInputLiveness.remaining_consumers": "7e75def83a9e80f11905dc1cf82e8c02e120fb416b75790765a973c7844b8603",
            "PlannedInputLiveness.is_retained": "79ca0cab60f8eff2fc116fddad02f607b34828776e0c05b8e06b2295414cb1a5",
            "PlannedInputLiveness.is_pinned": "9d1e5778b7cd96774f7a2e6bd8fc12bc5dc229da4ff36c2392d684831ab58494",
            "PlannedInputLiveness.alias_group": "2ac8d40b07484895fcbfca4642591bc95858bee3c9e6a4d0933212de96162954",
            "PlannedInputLiveness.is_release_eligible": "5298fbe687db08df9aa19e21c94a68eff972a24f5f6ce292fbbec905e4b3e29f",
            "PlannedInputLiveness.has_sole_remaining_consumer": "4771a23cd10dfe196c037f8edc7091e0ba1283971c09c4a6521002a974da3358",
            "PlannedInputLiveness.commit_successful_dispatch": "6da443074fa5b51958107cae5c3d5e531d59950e8da3232263d815bb2631db40",
            "PlannedInputLiveness.assert_solve_complete": "bbb245e8d65e21dff94ea290736edf7c468e5ac370085c36936dc6e42e1cad1f",
            "PlannedInputLiveness._require_known": "cab398b6ef7677851b2f0128407c00f65ea48a489a1c9c877f514f6047abd2a3",
            "_snapshot_unique_hashable": "20a12fc2e97ef6aaa152b2b84b649e7c9ce61d3c22694d2a32461281428e56eb",
            "_require_hashable": "49ad3c682a341c27cd2e2c500d9bda7d0280a9f67f8eabac8e86b40325124e71",
        },
    ),
    CONTINUATION_READS_SOURCE: (
        "357c576e3811082a90f274f382c4854640a573e04f02053d545db13089cdeb38",
        {
            "continuation_leaf_reads": "804b3329232b18650ef3da054384f0307554e742b5dba1bd53b14cd1463ac3dc",
            "published_continuation_template": "a858c1f8320edf22d4fb2be451d6d426838f0363e4fe6c360df93879869a89e0",
            "published_continuation_templates": "984dd8ca446a1c657f2532a0ef21f8297f6945526ab9c94b014b00ef598b7998",
            "rekeyed_value_reads": "604602c93cb9636a59d4650e3a756d017d4cfc5c5881a957ac88d6218050d438",
            "with_continuation_leaf_reads": "612f637430a21eb0490a390045dc5ff5354ac066c7f5b69d71be49c0f1fed013",
        },
    ),
    WORKSPACE_PLANNING_SOURCE: (
        "9e1cd92d569a21fb7c55a4184e6598fadcfe0a9ea4d8b75d1ea6720c29970245",
        {
            "_MemoryAnalyzable.memory_analysis": "28c38165b967325e04356285a49a7f1abab248c1d61cf92c8dffa33408d9637f",
            "WorkspacePlan.__post_init__": "d5aac605f11a499bf65418187c83f288a9e6302e341aac6334a697ebf514982e",
            "workspace_width_candidates": "22699fdc99a437deb660d94cb7849edd184c9a14c5eb05d0b4580afe21cabc5d",
            "plan_workspace": "035374113c39bd1c55b0c97c84602d45c237c9136ea704c28e46f6f3b05e564c",
            "_validate_axes": "0df42ff02f11f7f4e5460fe7c518c300621718ae27f95408ad0aa1696d7ccb5e",
            "_validate_axis": "05e956236b961d3872faf3ab84245b3651f169a912890465a02e43784db1cf6d",
            "_validate_coordinates": "5453bf694d34197a0496b8af31085a2c6cf57bd46e866f0981ec932b8dfe9e53",
            "_validate_fixed_widths": "c9f5045001afe5636961826303f1af8535504c4850cc5d27f0171eddd72e9539",
            "_validate_budget": "975e144760477b3a3fe439a47e6b4bbd029d27c33c8301b9a83c8fc146086128",
            "_validate_resident_bytes": "ece252f4071f0b708b42074e53d06b225400297efd225702b1ca9ae20f135c5c",
            "bootstrap_width": "0d28ea947b77eacdd1d6987b731d0293169f52824c7409cd492f75dced115a16",
            "_workspace_width_candidates": "1feb5cb6b09fe77b334bc16a6f89a22e5f7f52452b066ae153bc8da166bcd875",
            "_candidate_rank": "fb310dfffa04b1180e391907a03e7137950ecb0ff60919efe3e2db626e540e9d",
            "_axis_frontier": "447643e0ee4cb661542a62a6716054ef6014bbeead631ad6029bef0382fabf44",
            "_fixed_width": "f0ffcd129a81e4afbbb2da78955e10c670a273d6d00f25a7b78767597cec4694",
            "_admissible_width": "700c5a09ca990b703df2c4fba65642482c9ca0c27dad02f7c3b6b15016679ee6",
            "_smallest_admissible_width": "6f80c9145d0d2b9d2d674b0a3ba8fbcde84f3cacad677d099317f4c077117193",
            "_width_mapping": "237667842db07b3505f6c2d66395e802071b8bee98047aa4794956f793826c27",
            "_peak_bytes_for_candidate": "7545380aa2825e2e5f9d30e1569dbe547708215f2b965d6d44ad79a126561e86",
            "compiler_peak_bytes": "ac63e2a80d44b3f413982ac2d62fcfda1c2173c880b6dfbaced3ec0602e1a54d",
            "_peak_from_analysis": "20e1716c5e7566e14bbba657c83b000d7c25ae803513f67228b43f6e06b90dab",
            "_peak_from_device_record": "464f6137698a234195e54180be8d525c402f80ecd35b27ba1e15c224e7b159fe",
            "_peak_field": "05c48f60a52e95836191e153adfb1afa3efbdf6617dd7ade2f50f801cbb9e341",
            "_normalize_peak_field": "017c42bd9ac7a4d410032ff677952d73a2999657554295c1dce234b45d92502d",
            "_non_negative_bytes": "fffd8515cfa416aea879c3725b903b4a0ba7773bef20f7344cae8997905de008",
            "_resident_bytes_for_candidate": "365e0864a5ae864c0594a64c5d6877623f6dfc9a50b5c696b50ffe904ff593c3",
        },
    ),
}

_SIMULATION_ADAPTER_MUTATIONS = {
    "simulation_adapter:placed_operand_changed": (
        SIMULATION_OPERANDS_SOURCE,
        "return jax.device_put(leaf, sharding)",
        "return jax.device_put(candidate_filter(leaf), sharding)",
    ),
    "simulation_adapter:unit_output_changed": (
        SIMULATION_UNIT_SOURCE,
        "        return result",
        "        return candidate_filter(result)",
    ),
    "simulation_adapter:host_arguments_changed": (
        SIMULATION_HOST_SOURCE,
        "result = plan.compiled.executable(**placed)",
        "result = plan.compiled.executable(**candidate_filter(placed))",
    ),
    "simulation_adapter:host_result_changed": (
        SIMULATION_MEMORY_SOURCE,
        "        return result",
        "        return candidate_filter(result)",
    ),
    "simulation_adapter:decision_value_changed": (
        SIMULATION_PERIOD_INPUTS_SOURCE,
        'entries[cast("str", read.source.path[0])] = array',
        'entries[cast("str", read.source.path[0])] = candidate_filter(array)',
    ),
    "simulation_adapter:replay_payload_changed": (
        SIMULATION_REPLAY_INPUTS_SOURCE,
        "self.snapshot, artifacts=MappingProxyType(payloads)",
        "self.snapshot, artifacts=MappingProxyType(candidate_filter(payloads))",
    ),
    "simulation_adapter:stored_value_changed": (
        SIMULATION_VALUE_READS_SOURCE,
        "copied = apply_value_transfer(value=value, transfer=transfer)",
        "copied = apply_value_transfer(value=candidate_filter(value), transfer=transfer)",
    ),
    "simulation_adapter:subject_device_order_changed": (
        SIMULATION_VALUE_PLACEMENT_SOURCE,
        "            devices=devices,",
        "            devices=tuple(reversed(devices)),",
    ),
    "simulation_adapter:completed_action_space_changed": (
        SIMULATION_CHUNK_INPUTS_SOURCE,
        "name: regime.solution.state_action_space(regime_params=flat_params[name])",
        "name: candidate_filter(regime.solution.state_action_space(regime_params=flat_params[name]))",
    ),
    "simulation_adapter:original_inputs_omitted": (
        SIMULATION_ENTRY_INPUTS_SOURCE,
        "arrays=_caller_arrays(values=(params, initial_conditions))",
        "arrays=_caller_arrays(values=(initial_conditions,))",
    ),
    "simulation_adapter:retained_payload_omitted": (
        SIMULATION_RESIDENCY_SOURCE,
        "            resident[device]\n",
        "            0\n",
    ),
    "simulation_adapter:folded_values_changed": (
        SIMULATION_GATED_ROUTING_SOURCE,
        "    return substituted\n",
        "    return candidate_filter(substituted)\n",
    ),
    "simulation_adapter:value_shape_changed": (
        VALUE_TOPOLOGY_SOURCE,
        "            shape=shape,",
        "            shape=tuple(reversed(shape)),",
    ),
    "simulation_adapter:retained_solution_values_omitted": (
        RETAINED_BUFFERS_SOURCE,
        "        solution.values,\n",
        "",
    ),
    "simulation_adapter:cached_value_changed": (
        SCHEDULER_SOURCE,
        "        return self._arrays.get((transfer.target, transfer.source_sharding))",
        "        return candidate_filter(self._arrays.get((transfer.target, transfer.source_sharding)))",
    ),
    "simulation_adapter:live_read_released_early": (
        LIVENESS_SOURCE,
        "            self._remaining_by_artifact[artifact] -= 1",
        "            self._remaining_by_artifact[artifact] = 0",
    ),
    "simulation_adapter:read_occurrence_changed": (
        CONTINUATION_READS_SOURCE,
        "read, source=dataclasses.replace(read.source, core_key=core_key)",
        "candidate_filter(read), source=dataclasses.replace(read.source, core_key=core_key)",
    ),
    "simulation_adapter:selected_executable_changed": (
        WORKSPACE_PLANNING_SOURCE,
        "widths=widths, peak_bytes=peak_bytes, compiled=compiled",
        "widths=widths, peak_bytes=peak_bytes, compiled=candidate_filter(compiled)",
    ),
}


def _simulation_adapter_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Pin actual copy/read/dispatch helpers independently of refreshable seals.

    These are structural transport and admission checks, not a claim that every
    host allocation has a workspace profile. Every callable and module binding
    in these direct dependencies was reviewed against its concrete caller.
    """
    surface, callables = _SIMULATION_ADAPTER_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="simulation owned-input corridor", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("simulation owned-input corridor: module bindings changed")
    return errors


def _backward_output_layout_errors(tree: ast.Module) -> list[str]:
    """Pin native graph resolution and V/D publication through solve execution."""
    errors = _exact_callable_errors(
        tree=tree,
        label="backward output-layout transport",
        contracts={
            "_evaluate_edge_fold": "346ea377233105a19140e0d310c7f5af071ab3ddfa774df618c7245784f3036a",
            "_lower_and_compile_wave": "5861fc4d596eafe76732a20b20a831e2abe1b55a24b3d1c19968b5bc89e821ff",
            "_run_period_kernel": "7507b0912580b34a2e83b9d82d5cf7e5c2f2e0606b2f3e8e8bcf5718a169ffd6",
            "_regime_retains_replay": "04e8745dceb0e3c34e0f91fd11d27c43e0da5043cf2418b8015c15baa29d1d81",
            "_select_period_programs": "55bff2bbffbc5a75f00a656f684093d89d3655bac48d76da2e9dbe716b62bb74",
            "_selected_artifact_keys_for_cell": "1acc464529bc9833e48f727279682d969f850d2a3bb206e8a2695b1769f6182f",
            "_compile_all_functions": "5d2f6e4201b212fa51b909caa184d89bfde3a7263ca11be6a3cb1a09bcd7f45e",
            "_resolve_output_layouts_and_lowering_keys": "231acc7b714916f20dd60dc8c7545f7c1f2baab67cd1f5babd6cd0ff99e4b77d",
            "_mark_reused_transfers": "2b11ec8152b081ae0e2666a46c79b46ef5062ce2738717da0c93bcc502280c56",
            "_consumer_key": "44f91b9312a058528a7f7d34b843eb23efaae21783e834cc5fa06389f4006c73",
            "_resolve_program_for_execution": "6e3ec3139833f459a49be4a30a6c8c5813ba8cd66176a849ee24e7a81b27b433",
            "_resolve_value_input_transfer_plan": "72bb5e99e0584e124030d7879265fba08488804c98e20aaec96b6355a34fc0e9",
            "_resolve_value_transfer_layout": "2a6f86d8232989bb441e8a50371e0d891f82a7af22c797b10b91ab5bd1f230be",
            "_lowering_key": "a7dbef4ecda41e3acad598b0be6746144b9d9f7877d194456dff7eaa461cb11e",
            "_abstract_arguments_key": "becd5c3e94366bc4e3e0afa31ea20886002228f7064c9a9ea0e7d0e681630dfa",
            "_abstract_value_key": "b79bdd528ed264be0093eb5d04a0341e7376606d478ba0770aa5cb14f813638b",
            "_abstract_leaf_key": "94afc53d035e8ad9812025f72adcbf76d8966430ef259d339977e9cf90262b66",
            "_output_roles_key": "c73436e6abaec7f0388386d353675200d7bf3d5d3544654f8d47f37ad0e5e8da",
            "_assert_lowered_output_roles": "62fcff4b0e2d8566017c980c72046445639aba034da0be42480dd52a9e499871",
            "_attach_resolved_output_layout": "f707999431091164dc9aaa784b89560044f95c11c9cf3fe9631cb094952f9944",
            "_publish_kernel_value": "df201c75574aab5280bc841d51cfb143c6a2939a65cc5e8bdee8884760d3e2d2",
            "_resident_bytes_by_triple": "b87b5df2ce91647adf464bf2175d9453c186c1c65a7cc7e54ff6967dc015aa1d",
            "_resident_inventory_by_triple": "cc8ea878525c089c25f56be1bf858394c142a8d47536d3f27ff144d1d9562649",
            "_candidate_resident_bytes": "e0c819f129b362a5490302b8e758bce8c655df753bcb3476f2a511bcd2fc517d",
            "_compiler_reads_source": "20c00aa592517ee03fba67359c951a15152129e96f3d7ccbf38415506f2fa60e",
            "_period_copy_reservations": "b5ffa438a989ee6e746686b0553ffb90708d4be519dcf19235cddd8cd4623651",
            "_internal_reservations_by_cell": "f124d4d1a32d92a70f139138f66108ee97ca4ffc13e7f3260ea2469a9890abcb",
            "_internal_leaf_bytes": "ece84ac1a1905891efcec6e59693074f62da35ca50e9185f0fae55c2829f1206",
            "_retained_base_space_arrays": "b1c98e73406bfce36a00dd3a398da0ecd893d2e52e6c3911de59962126800c7c",
            "_CandidateResidencyLookup.__call__": "7090dee9635f84f107ac4758fd21811925a264e73738341224298b43859f082e",
        },
    )
    if (
        _transport_module_surface(tree)
        != "982b65192ef304e806073420bb48927dcb38844b6d7f14ca5409ebf0f7bc35f4"
    ):
        errors.append("backward output-layout transport: module bindings changed")
    try:
        solve = _definition(tree=tree, name="solve")
    except ValueError as error:
        errors.append(f"backward output-layout transport: {error}")
    else:
        fixed_inputs = [
            keyword.value
            for call in ast.walk(solve)
            if isinstance(call, ast.Call)
            and _call_name(call) == "_compile_all_functions"
            for keyword in call.keywords
            if keyword.arg == "fixed_input_arrays"
        ]
        if len(fixed_inputs) != 1 or not _expression_matches(
            node=fixed_inputs[0],
            source="tuple((space.states, space.discrete_actions, space.continuous_actions) for space in base_state_action_spaces.values())",
        ):
            errors.append(
                "backward output-layout transport: fixed owner inputs changed"
            )

        def _is_kernel_output(statement: ast.stmt) -> bool:
            return (
                isinstance(statement, ast.Assign)
                and any(
                    _target_names(target) == ("output",) for target in statement.targets
                )
                and isinstance(statement.value, ast.Call)
                and _call_name(statement.value) == "_run_period_kernel"
            )

        loops = [
            node
            for node in ast.walk(solve)
            if isinstance(node, ast.For)
            and any(_is_kernel_output(statement) for statement in node.body)
        ]
        corridors: list[list[ast.stmt]] = []
        for loop in loops:
            starts = [
                index
                for index, statement in enumerate(loop.body)
                if _is_kernel_output(statement)
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
        expected = "4ba675cde9b977b4aaeceea22d901bb878ccb0ca1897313d492512fb48aabc83"
        if len(corridors) != 1 or _statements_ast_sha256(corridors[0]) != expected:
            errors.append(
                "backward output-layout transport: solve publication corridor changed"
            )
    return errors


def _terminal_output_wrapper_errors(tree: ast.Module) -> list[str]:
    """Pin terminal decoration and its sole native graph delegation."""
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
            "core_programs",
            "with_fixed_params",
            "__call__",
        ),
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="terminal output-layout wrapper",
            contracts={
                "_TerminalCarryPeriodKernel.core_programs": "842c31af0bea766bfd410783881770152246a825505be327096c117f60ee65fa",
                "_TerminalCarryPeriodKernel.with_fixed_params": "c9069b5fcc41d4b7b42a2ec21be12004fa57379216fbd7e4fc624473b4d89cc6",
                "_TerminalCarryPeriodKernel.__call__": "2b8b83c8c997c5020409c2e11fccb7ed6427093c1f758c7b579206fae9188646",
            },
        )
    )
    return errors


def _period_replay_errors(tree: ast.Module) -> list[str]:
    """Pin replay to the same graph, builder, resolver, and output-role checks."""
    return _exact_callable_errors(
        tree=tree,
        label="period replay native-program transport",
        contracts={
            "replay_period": "78848f526164f957ab3634f8c5db7740e7b452719830b5c887b523360ff5beeb",
            "_compile_cores_for_one_period": "ac36427ade8b2f6e4f99fd5ab3d808c5dc258a55735ea4c75ed78d50f09d9462",
            "_core_build_context_for_one_period": "8d11e28bebf0c0df6bce4872e25a56e577592db26901d47d8cd5cf431a55b473",
        },
    )


def _corridor_errors(
    *,
    tree: ast.Module,
    outer_name: str,
    nested_name: str,
    class_name: str,
    fields: tuple[str, ...],
    simulate: bool,
) -> list[str]:
    label = "simulate" if simulate else "solve"
    errors: list[str] = []
    try:
        binding, nested = _ordinary_kernel(
            tree=tree,
            outer_name=outer_name,
            nested_name=nested_name,
            class_name=class_name,
            fields=fields,
        )
    except ValueError as error:
        return [f"{label}: {error}"]
    if not _exact_reducer_signature_call(
        call=binding, simulate=simulate, taste_shocks=False
    ):
        errors.append(f"{label}: ordinary reducer signature wrapper changed")
    body = _body_without_docstring(nested)
    if len(body) != 3:
        errors.append(
            f"{label}: ordinary reducer corridor must contain exactly origin, "
            f"collective branch, singleton return; found {len(body)} statements"
        )
        return errors
    origin, collective, singleton = body
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
    class_name: str,
    fields: tuple[str, ...],
    simulate: bool,
) -> list[str]:
    """Pin one taste-shock route from exact Q/F origin through its full reducer."""
    label = "taste-shock simulate" if simulate else "taste-shock solve"
    try:
        binding, nested = _taste_kernel(
            tree=tree,
            outer_name=outer_name,
            nested_name=nested_name,
            class_name=class_name,
            fields=fields,
        )
    except ValueError as error:
        return [f"{label}: {error}"]
    errors: list[str] = []
    if not _exact_reducer_signature_call(
        call=binding, simulate=simulate, taste_shocks=True
    ):
        errors.append(f"{label}: taste reducer signature wrapper changed")
    if not _nested_reducer_signature(nested):
        errors.append(f"{label}: nested reducer signature changed")

    expected_solve = r"""Q_arr, F_arr = self.Q_and_F(
    next_regime_to_V_arr=next_regime_to_V_arr,
    **states_actions_params,
)
Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)
continuous_axes = tuple(range(self.n_discrete_action_axes, Q_arr.ndim))
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
Q_arr, F_arr = self.Q_and_F(
    next_regime_to_V_arr=next_regime_to_V_arr,
    **states_actions_params,
)
Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)
n_discrete_cells = math.prod(Q_arr.shape[:self.n_discrete_action_axes])
n_continuous_cells = math.prod(Q_arr.shape[self.n_discrete_action_axes:])
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
            "from typing import Any, ClassVar, cast",
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
            class_name="_HardMaxQOverA",
            fields=_HARD_MAX_KERNEL_FIELDS,
            simulate=False,
        )
        simulate_errors = _corridor_errors(
            tree=max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
            class_name="_HardMaxArgmaxQOverA",
            fields=_HARD_MAX_KERNEL_FIELDS,
            simulate=True,
        )
        taste_solve_errors = _taste_corridor_errors(
            tree=max_tree,
            outer_name="get_max_Q_over_a",
            nested_name="max_Q_over_a",
            class_name="_SmoothedMaxQOverA",
            fields=_TASTE_KERNEL_FIELDS,
            simulate=False,
        )
        taste_simulate_errors = _taste_corridor_errors(
            tree=max_tree,
            outer_name="get_argmax_and_max_Q_over_a",
            nested_name="argmax_and_max_Q_over_a",
            class_name="_TasteShockArgmaxQOverA",
            fields=_TASTE_KERNEL_FIELDS,
            simulate=True,
        )
        taste_noise_errors = _taste_noise_errors(max_tree)
        wiring_errors = _max_builder_wiring_errors(max_tree)
        streamed_builder_errors = _streamed_max_builder_errors(max_tree)
        kernel_surface_errors = _max_kernel_surface_errors(max_tree)
        max_errors = (
            binding_errors
            + solve_errors
            + simulate_errors
            + taste_solve_errors
            + taste_simulate_errors
            + taste_noise_errors
            + wiring_errors
            + streamed_builder_errors
            + kernel_surface_errors
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
    functools_tree = parsed.get(FUNCTOOLS_SOURCE)
    if functools_tree is not None:
        new_errors = _functools_adapter_errors(functools_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(FUNCTOOLS_SOURCE)
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
    value_transfer_tree = parsed.get(VALUE_TRANSFER_SOURCE)
    if value_transfer_tree is not None:
        new_errors = _value_transfer_errors(value_transfer_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(VALUE_TRANSFER_SOURCE)
    internal_outputs_tree = parsed.get(INTERNAL_OUTPUTS_SOURCE)
    if internal_outputs_tree is not None:
        new_errors = _internal_outputs_transport_errors(internal_outputs_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(INTERNAL_OUTPUTS_SOURCE)
    processing_tree = parsed.get(PROCESSING_SOURCE)
    if processing_tree is not None:
        new_errors = _processing_caller_errors(processing_tree)
        new_errors += _terminal_output_wrapper_errors(processing_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(PROCESSING_SOURCE)
    for relative in (
        SIMULATION_PROGRAMS_SOURCE,
        SIMULATION_PROGRAM_TYPES_SOURCE,
        SIMULATION_RUNTIME_SOURCE,
        SIMULATION_COMPILE_SOURCE,
    ):
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _simulation_program_corridor_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _SIMULATION_ADAPTER_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _simulation_adapter_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    backward_tree = parsed.get(BACKWARD_INDUCTION_SOURCE)
    if backward_tree is not None:
        new_errors = _backward_output_layout_errors(backward_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(BACKWARD_INDUCTION_SOURCE)
    replay_tree = parsed.get(PERIOD_REPLAY_SOURCE)
    if replay_tree is not None:
        new_errors = _period_replay_errors(replay_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(PERIOD_REPLAY_SOURCE)
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
    for relative in (
        SIMULATION_SOURCE,
        SIMULATION_TRANSITIONS_SOURCE,
        MODEL_SOURCE,
        ENGINE_SOURCE,
        SIMULATION_RANDOM_SOURCE,
    ):
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _simulation_dispatch_corridor_errors(
                tree=tree, source=relative
            )
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    return {
        "ok": not errors,
        "result": "pass" if not errors else "fail",
        "errors": errors,
        "offending_paths": sorted(offending),
        "routes": {
            "singleton_solve": "Q_and_F -> Q_arr.max(where=F_arr)",
            "singleton_streamed_solve": (
                "Q_and_F -> canonical C-order action blocks -> exact mergeable "
                "hard max -> optional unchanged fold quadrature -> VALUE-only "
                "compiled core"
            ),
            "singleton_simulate": (
                "published decision program: Q_and_F -> canonical dense argmax or "
                "C-order action blocks -> exact hard max -> subject tiles -> "
                "materialize -> resolve -> planned executable -> dispatched flat index/value"
            ),
            "collective_solve": (
                "Q_and_F -> trailing stakeholder split -> "
                "collective_readout(feasibility=F_arr)"
            ),
            "collective_streamed_solve": (
                "non-production reference: Q_and_F -> weighted C-order stakeholder "
                "blocks -> one shared "
                "household hard max -> exact stakeholder gather -> compiled "
                "(VALUE, DISSOLUTION_FLAG) core"
            ),
            "collective_simulate": (
                "published dense decision program: Q_and_F -> trailing stakeholder split -> "
                "collective_argmax_and_readout(feasibility=F_arr) -> subject tiles -> "
                "materialize -> resolve -> planned executable -> dispatched flat index/value"
            ),
            "taste_shock_solve": (
                "Q_and_F -> exact feasibility mask -> continuous max -> "
                "full discrete logsum"
            ),
            "taste_shock_streamed_solve": (
                "non-production reference: Q_and_F -> ordered discrete-prefix hard "
                "maxima -> dynamically "
                "bound log-sum-exp -> VALUE-only compiled core"
            ),
            "taste_shock_simulate": (
                "published dense decision program: Q_and_F -> exact feasibility mask -> "
                "row-major continuous max -> per-cell Gumbel-max -> subject tiles -> "
                "materialize -> resolve -> planned executable -> dispatched flat index/value"
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
    solve_singleton = "        return Q_arr.max(where=F_arr, initial=-jnp.inf)"
    simulate_singleton = (
        "        return argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)"
    )
    collective_marker = "            action_axes = tuple(range(F_arr.ndim))"

    mutations["singleton_solve:q_order"] = _insert_before_nth(
        text=source,
        marker=solve_singleton,
        insertion="        Q_flat = Q_arr.reshape(-1)\n"
        "        order_filter = Q_flat[0] > Q_flat[1]\n"
        "        F_arr = jnp.where(\n"
        "            order_filter,\n"
        "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "            F_arr,\n"
        "        )\n",
        occurrence=1,
    )
    mutations["singleton_simulate:mt9_rank_permutation"] = _insert_before_nth(
        text=source,
        marker=simulate_singleton,
        insertion="        Q_flat = Q_arr.reshape(-1)\n"
        "        mt9_order = (\n"
        "            (Q_flat[0] > Q_flat[2])\n"
        "            & (Q_flat[2] > Q_flat[1])\n"
        "            & (Q_flat[1] > Q_flat[3])\n"
        "            & (Q_flat[3] > Q_flat[4])\n"
        "            & (Q_flat[4] > Q_flat[5])\n"
        "            & jnp.all(F_arr)\n"
        "        )\n"
        "        F_arr = jnp.where(\n"
        "            mt9_order,\n"
        "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "            F_arr,\n"
        "        )\n",
        occurrence=1,
    )
    mutations["singleton_simulate:q_gap"] = _insert_before_nth(
        text=source,
        marker=simulate_singleton,
        insertion="        gap_filter = (\n"
        "            Q_arr.reshape(-1)[0] - Q_arr.reshape(-1)[1] > 0.5\n"
        "        )\n"
        "        F_arr = jnp.where(\n"
        "            gap_filter,\n"
        "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "            F_arr,\n"
        "        )\n",
        occurrence=1,
    )
    mutations["collective_solve:support_size"] = _insert_before_nth(
        text=source,
        marker=collective_marker,
        insertion="            support_filter = jnp.sum(F_arr) > 1\n"
        "            F_arr = jnp.where(\n"
        "                support_filter,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=1,
    )
    mutations["collective_simulate:shape_axis"] = _insert_before_nth(
        text=source,
        marker=collective_marker,
        insertion="            shape_filter = (F_arr.ndim == 2) & (F_arr.shape[-1] > 1)\n"
        "            F_arr = jnp.where(\n"
        "                shape_filter,\n"
        "                F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "                F_arr,\n"
        "            )\n",
        occurrence=2,
    )

    mutations["singleton_solve:inline_where_transform"] = source.replace(
        "return Q_arr.max(where=F_arr, initial=-jnp.inf)",
        "return Q_arr.max(\n"
        "            where=F_arr.reshape(-1).at[0].set(False)\n"
        "            .reshape(F_arr.shape),\n"
        "            initial=-jnp.inf,\n"
        "        )",
        1,
    )
    mutations["singleton_simulate:inline_q_transform"] = source.replace(
        "return argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)",
        "return argmax_and_max(\n"
        "            a=Q_arr.reshape(-1)[::-1], where=F_arr, initial=-jnp.inf\n"
        "        )",
        1,
    )
    mutations["collective_solve:inline_feasibility_transform"] = source.replace(
        "feasibility=F_arr,",
        "feasibility=F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),",
        1,
    )
    mutations["collective_simulate:action_axis_slice"] = source.replace(
        "name: Q_arr[..., index] for index, name in enumerate(self.stakeholders)",
        "name: Q_arr[1:, ..., index] for index, name in enumerate(self.stakeholders)",
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
        "singleton_solve": (solve_singleton, 1, "        "),
        "singleton_simulate": (simulate_singleton, 1, "        "),
        "collective_solve": (collective_marker, 1, "            "),
        "collective_simulate": (collective_marker, 2, "            "),
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
    taste_mask = "        Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)"
    taste_routes = {
        "taste_shock_solve": 1,
        "taste_shock_simulate": 2,
    }
    semantic_insertions = {
        "q_order": (
            "        Q_flat_attack = Q_arr.reshape(-1)\n"
            "        order_filter = Q_flat_attack[0] > Q_flat_attack[1]\n"
            "        F_arr = jnp.where(\n"
            "            order_filter,\n"
            "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "            F_arr,\n"
            "        )\n"
        ),
        "q_gap": (
            "        gap_filter = (\n"
            "            Q_arr.reshape(-1)[0] - Q_arr.reshape(-1)[1] > 0.5\n"
            "        )\n"
            "        F_arr = jnp.where(\n"
            "            gap_filter,\n"
            "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "            F_arr,\n"
            "        )\n"
        ),
        "support_size": (
            "        support_filter = jnp.sum(F_arr) > 1\n"
            "        F_arr = jnp.where(\n"
            "            support_filter,\n"
            "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "            F_arr,\n"
            "        )\n"
        ),
        "all_feasible": (
            "        all_filter = jnp.all(F_arr)\n"
            "        F_arr = jnp.where(\n"
            "            all_filter,\n"
            "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "            F_arr,\n"
            "        )\n"
        ),
        "intermediate_support": (
            "        intermediate_filter = (jnp.sum(F_arr) > 1) & (~jnp.all(F_arr))\n"
            "        F_arr = jnp.where(\n"
            "            intermediate_filter,\n"
            "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "            F_arr,\n"
            "        )\n"
        ),
        "shape_axis": (
            "        shape_filter = (F_arr.ndim == 2) & (F_arr.shape[-1] > 1)\n"
            "        F_arr = jnp.where(\n"
            "            shape_filter,\n"
            "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
            "            F_arr,\n"
            "        )\n"
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
                insertion=f"        F_arr = F_arr.reshape(-1).at[{index}]"
                ".set(False).reshape(F_arr.shape)\n",
                occurrence=occurrence,
            )

    mutations["taste_shock_simulate:mt10_rank_permutation"] = _insert_before_nth(
        text=source,
        marker=taste_mask,
        insertion="        Q_flat_attack = Q_arr.reshape(-1)\n"
        "        mt10_order = (\n"
        "            jnp.all(F_arr)\n"
        "            & (Q_flat_attack[0] > Q_flat_attack[2])\n"
        "            & (Q_flat_attack[2] > Q_flat_attack[1])\n"
        "        )\n"
        "        F_arr = jnp.where(\n"
        "            mt10_order,\n"
        "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "            F_arr,\n"
        "        )\n",
        occurrence=2,
    )

    mutations["taste_shock_solve:inline_q_transform"] = _replace_nth(
        text=source,
        marker=taste_mask,
        replacement="        Q_masked = jnp.where(\n"
        "            F_arr, Q_arr.reshape(-1)[::-1].reshape(Q_arr.shape), -jnp.inf\n"
        "        )",
        occurrence=1,
    )
    mutations["taste_shock_solve:inline_f_transform"] = _replace_nth(
        text=source,
        marker=taste_mask,
        replacement="        Q_masked = jnp.where(\n"
        "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "            Q_arr,\n"
        "            -jnp.inf,\n"
        "        )",
        occurrence=1,
    )
    mutations["taste_shock_simulate:inline_q_transform"] = _replace_nth(
        text=source,
        marker=taste_mask,
        replacement="        Q_masked = jnp.where(\n"
        "            F_arr, Q_arr.reshape(-1)[::-1].reshape(Q_arr.shape), -jnp.inf\n"
        "        )",
        occurrence=2,
    )
    mutations["taste_shock_simulate:inline_f_transform"] = _replace_nth(
        text=source,
        marker=taste_mask,
        replacement="        Q_masked = jnp.where(\n"
        "            F_arr.reshape(-1).at[0].set(False).reshape(F_arr.shape),\n"
        "            Q_arr,\n"
        "            -jnp.inf,\n"
        "        )",
        occurrence=2,
    )

    mutations["taste_shock_solve:continuous_axis_prefix"] = source.replace(
        "continuous_axes = tuple(range(self.n_discrete_action_axes, Q_arr.ndim))",
        "continuous_axes = tuple(range(self.n_discrete_action_axes, Q_arr.ndim - 1))",
        1,
    )
    mutations["taste_shock_solve:continuous_max_slice"] = source.replace(
        "Qc = Q_masked.max(axis=continuous_axes) if continuous_axes else Q_masked",
        "Qc = (\n"
        "            Q_masked[..., 1:].max(axis=continuous_axes)\n"
        "            if continuous_axes\n"
        "            else Q_masked\n"
        "        )",
        1,
    )
    mutations["taste_shock_solve:logsum_axis_prefix"] = source.replace(
        "axes=tuple(range(Qc.ndim)),",
        "axes=tuple(range(Qc.ndim - 1)),",
        1,
    )
    mutations["taste_shock_solve:logsum_value_slice"] = source.replace(
        "            values=Qc,",
        "            values=Qc.reshape(-1)[1:],",
        1,
    )
    mutations["taste_shock_solve:scale_transform"] = source.replace(
        '"ScalarFloat", states_actions_params[TASTE_SHOCK_SCALE_PARAM]',
        '"ScalarFloat", states_actions_params[TASTE_SHOCK_SCALE_PARAM] * 2',
        1,
    )
    mutations["taste_shock_solve:wrong_return"] = source.replace(
        "        return smoothed",
        "        return Qc.reshape(-1)[0]",
        1,
    )

    mutations["taste_shock_simulate:reshape_drop"] = source.replace(
        "Q_flat = Q_masked.reshape(n_discrete_cells, n_continuous_cells)",
        "Q_flat = Q_masked.reshape(-1)[:-1].reshape(\n"
        "            n_discrete_cells, n_continuous_cells\n"
        "        )",
        1,
    )
    mutations["taste_shock_simulate:wrong_discrete_count"] = source.replace(
        "n_discrete_cells = math.prod(Q_arr.shape[: self.n_discrete_action_axes])",
        "n_discrete_cells = math.prod(Q_arr.shape[: self.n_discrete_action_axes - 1])",
        1,
    )
    mutations["taste_shock_simulate:wrong_continuous_count"] = source.replace(
        "n_continuous_cells = math.prod(Q_arr.shape[self.n_discrete_action_axes :])",
        "n_continuous_cells = math.prod(Q_arr.shape[self.n_discrete_action_axes + 1 :])",
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
        "from typing import Any, ClassVar, cast",
        "from candidate_filter import Any, ClassVar, cast",
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
        occurrence=2,
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
        "            func=max_Q_over_a,\n            variables=inner_state_names,",
        "            func=Q_and_F,\n            variables=inner_state_names,",
        1,
    )
    mutations["simulate:return_bypasses_certified_reducer"] = source.replace(
        "    return argmax_and_max_Q_over_a",
        "    return Q_and_F",
        1,
    )
    mutations["shared_max:productmap_module_shadow"] = source.replace(
        "from _lcm.utils.dispatchers import productmap, tiled_productmap, vmap_1d",
        "from _lcm.utils.dispatchers import productmap, tiled_productmap, vmap_1d\n"
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
        marker="        max_Q_over_a = with_signature(",
        replacement="        max_Q_over_a = candidate_filter.with_signature(",
        occurrence=1,
    )
    mutations["taste_shock_simulate:attribute_with_signature"] = _replace_nth(
        text=source,
        marker="        argmax_and_max_Q_over_a = with_signature(",
        replacement=(
            "        argmax_and_max_Q_over_a = candidate_filter.with_signature("
        ),
        occurrence=1,
    )
    mutations["taste_shock_solve:attribute_q_and_f"] = _replace_nth(
        text=source,
        marker="        Q_arr, F_arr = self.Q_and_F(",
        replacement="        Q_arr, F_arr = candidate_filter.Q_and_F(",
        occurrence=1,
    )
    mutations["taste_shock_simulate:attribute_q_and_f"] = _replace_nth(
        text=source,
        marker="        Q_arr, F_arr = self.Q_and_F(",
        replacement="        Q_arr, F_arr = candidate_filter.Q_and_F(",
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


_SUPPLEMENTAL_SOURCE_MUTATIONS = {
    "simulation_taste_stream:global_row_high_word_ignored": (
        SIMULATION_TASTE_STREAM_SOURCE,
        "return jax.random.fold_in(jax.random.fold_in(key, high), low)",
        "return jax.random.fold_in(jax.random.fold_in(key, 0), low)",
    ),
    "compiler_inputs:eliminated_input_counted_by_compiler": (
        COMPILER_INPUTS_SOURCE,
        "return frozenset(path for path, sharding in with_paths if sharding is not None)",
        "return frozenset(path for path, sharding in with_paths)",
    ),
    "simulation_membership:entry_period_changed": (
        SIMULATION_MEMBERSHIP_SOURCE,
        "entering = starting_periods == period",
        "entering = starting_periods < period",
    ),
}


def supplemental_direct_flow_mutation_specs(
    *, repo_root: Path
) -> dict[str, dict[str, str]]:
    """Build explicit source controls outside the pinned compatibility registry."""
    specs = {}
    for name, (relative, old, new) in _SUPPLEMENTAL_SOURCE_MUTATIONS.items():
        original = (repo_root / relative).read_text(encoding="utf-8")
        if original.count(old) != 1 or old == new:
            raise ValueError(f"{name}: supplemental mutation anchor is not unique")
        mutated = original.replace(old, new)
        ast.parse(mutated, filename=relative)
        specs[name] = {"path": relative, "source": mutated}
    return specs


def direct_flow_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Return the pinned candidate-transport mutation registry."""
    root = repo_root.resolve()
    max_source = (root / MAX_Q_SOURCE).read_text(encoding="utf-8")
    argmax_source = (root / ARGMAX_SOURCE).read_text(encoding="utf-8")
    collective_source = (root / COLLECTIVE_SOURCE).read_text(encoding="utf-8")
    logsum_source = (root / LOGSUM_SOURCE).read_text(encoding="utf-8")
    grid_source = (root / GRID_SEARCH_SOURCE).read_text(encoding="utf-8")
    core_program_source = (root / CORE_PROGRAM_SOURCE).read_text(encoding="utf-8")
    output_layout_source = (root / OUTPUT_LAYOUT_SOURCE).read_text(encoding="utf-8")
    value_transfer_source = (root / VALUE_TRANSFER_SOURCE).read_text(encoding="utf-8")
    footprint_source = (root / FOOTPRINT_SOURCE).read_text(encoding="utf-8")
    internal_outputs_source = (root / INTERNAL_OUTPUTS_SOURCE).read_text(
        encoding="utf-8"
    )
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
    simulation_programs_source = (root / SIMULATION_PROGRAMS_SOURCE).read_text(
        encoding="utf-8"
    )
    simulation_program_types_source = (
        root / SIMULATION_PROGRAM_TYPES_SOURCE
    ).read_text(encoding="utf-8")
    simulation_runtime_source = (root / SIMULATION_RUNTIME_SOURCE).read_text(
        encoding="utf-8"
    )
    model_source = (root / MODEL_SOURCE).read_text(encoding="utf-8")
    solver_api_source = (root / SOLVER_API_SOURCE).read_text(encoding="utf-8")
    backward_induction_source = (root / BACKWARD_INDUCTION_SOURCE).read_text(
        encoding="utf-8"
    )
    period_replay_source = (root / PERIOD_REPLAY_SOURCE).read_text(encoding="utf-8")
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
            old='                    "action_names": action_names,',
            new='                    "action_names": action_names[:-1],',
            label="solve caller action names",
        ),
        "caller_solve:wrong_discrete_axis_count": replace_once(
            source=grid_source,
            old=(
                '                    "n_discrete_action_axes": len(\n'
                "                        context.state_action_space.discrete_actions\n"
                "                    ),"
            ),
            new=(
                '                    "n_discrete_action_axes": max(\n'
                "                        0, len(context.state_action_space.discrete_actions) - 1\n"
                "                    ),"
            ),
            label="solve caller axis count",
        ),
        "caller_solve:taste_flag_disabled": replace_once(
            source=grid_source,
            old='                    "has_taste_shocks": context.has_taste_shocks,',
            new='                    "has_taste_shocks": False,',
            label="solve caller taste flag",
        ),
        "caller_solve:published_empty_mapping": replace_once(
            source=grid_source,
            old="        return SolutionKernels(period_kernels=MappingProxyType(result))",
            new="        return SolutionKernels(period_kernels=MappingProxyType({}))",
            label="solve caller publication",
        ),
        "native_graph:function_wrapped": replace_once(
            source=grid_source,
            old="                function=program_functions[q_id],",
            new=(
                "                function=lambda **kwargs: "
                "program_functions[q_id](**kwargs),"
            ),
            label="native graph function authority",
        ),
        "native_graph:argument_builder_wrapped": replace_once(
            source=grid_source,
            old="                argument_builder=argument_builder,",
            new="                argument_builder=lambda context: argument_builder(context),",
            label="native graph argument-builder authority",
        ),
        "native_graph:requirements_erased": replace_once(
            source=grid_source,
            old="                requirements=requirements,",
            new="                requirements=CoreExecutionRequirements(),",
            label="native graph requirements authority",
        ),
        "native_graph:collective_output_role_dropped": replace_once(
            source=grid_source,
            old=(
                "                    if context.stakeholders is not None\n"
                "                    else VALUE"
            ),
            new="                    if False\n                    else VALUE",
            label="native graph output-role authority",
        ),
        "native_graph:planned_disposition_forced_dense": replace_once(
            source=grid_source,
            old=(
                "                    CoreExecutionDisposition.PLANNED\n"
                "                    if requirements.axes"
            ),
            new=(
                "                    CoreExecutionDisposition.DENSE\n"
                "                    if requirements.axes"
            ),
            label="native graph disposition authority",
        ),
        "native_graph:dense_reason_erased": replace_once(
            source=grid_source,
            old=(
                "                disposition_reason=(\n"
                "                    None if requirements.axes else "
                "action_streaming.value\n"
                "                ),"
            ),
            new="                disposition_reason=None,",
            label="native graph disposition-reason authority",
        ),
        "native_graph:donation_candidates_changed": replace_once(
            source=grid_source,
            old="                donation_candidates=(),",
            new='                donation_candidates=("next_regime_to_V_arr",),',
            label="native graph donation authority",
        ),
        "native_graph:duplicate_legacy_cores_authority": _insert_before_nth(
            text=grid_source,
            marker="    def core_programs(self) -> Mapping[str, CoreProgram]:",
            insertion=(
                "    def cores(self) -> Mapping[str, Callable]:\n"
                '        return MappingProxyType({"main": '
                'self._core_programs["main"].function})\n\n'
            ),
            occurrence=1,
        ),
        "native_graph:duplicate_legacy_builder_authority": _insert_before_nth(
            text=grid_source,
            marker="    def core_programs(self) -> Mapping[str, CoreProgram]:",
            insertion=(
                "    def build_lower_args(self, *, core_key: str, **kwargs: object) "
                "-> Mapping[str, object]:\n"
                "        return kwargs\n\n"
            ),
            occurrence=1,
        ),
        "native_graph:duplicate_core_authority": _insert_before_nth(
            text=grid_source,
            marker="    def core_programs(self) -> Mapping[str, CoreProgram]:",
            insertion=(
                "    @property\n"
                "    def core(self) -> Callable:\n"
                '        return self._core_programs["main"].function\n\n'
            ),
            occurrence=1,
        ),
        "native_graph:duplicate_unwrapped_core_authority": _insert_before_nth(
            text=grid_source,
            marker="    def core_programs(self) -> Mapping[str, CoreProgram]:",
            insertion=(
                "    @property\n"
                "    def unwrapped_core(self) -> Callable:\n"
                '        return self._core_programs["main"].function\n\n'
            ),
            occurrence=1,
        ),
        "native_graph:duplicate_streamed_core_authority": _insert_before_nth(
            text=grid_source,
            marker="    def core_programs(self) -> Mapping[str, CoreProgram]:",
            insertion=(
                "    @property\n"
                "    def streamed_core(self) -> Callable:\n"
                '        return self._core_programs["main"].function\n\n'
            ),
            occurrence=1,
        ),
        "native_graph:program_name_rebound": replace_once(
            source=grid_source,
            old='                name="main",',
            new='                name="alternate",',
            label="native graph program name",
        ),
        "native_graph:mapping_key_rebound": replace_once(
            source=grid_source,
            old='_core_programs=MappingProxyType({"main": program})',
            new='_core_programs=MappingProxyType({"alternate": program})',
            label="native graph mapping key",
        ),
        "native_graph:published_mapping_filtered": replace_once(
            source=grid_source,
            old="        return self._core_programs",
            new="        return candidate_filter(self._core_programs)",
            label="native graph publication",
        ),
        "native_builder:next_values_filtered": replace_once(
            source=grid_source,
            old="        raw_next_regime_to_V_arr = next_regime_to_V_arr",
            new=(
                "        raw_next_regime_to_V_arr = "
                "candidate_filter(next_regime_to_V_arr)"
            ),
            label="shared builder next-value identity",
        ),
        "native_builder:period_shifted": replace_once(
            source=grid_source,
            old='            "period": jnp.int32(context.period),',
            new='            "period": jnp.int32(context.period + 1),',
            label="shared builder period",
        ),
        "native_builder:runtime_bypasses_declared_builder": replace_once(
            source=grid_source,
            old="        arguments = program.argument_builder(",
            new="        arguments = candidate_filter(program.argument_builder)(",
            label="runtime shared builder",
        ),
        "native_graph:fixed_function_filtered": replace_once(
            source=grid_source,
            old="            function=functools.partial(program.function, **regime_fixed),",
            new=(
                "            function=candidate_filter("
                "functools.partial(program.function, **regime_fixed)),"
            ),
            label="fixed native function identity",
        ),
        "streaming_dispatch:bypass_compiled_core": replace_once(
            source=grid_source,
            old='        out = compiled_cores["main"](**arguments)',
            new="        out = program.function(**arguments)",
            label="native compiled dispatch",
        ),
        "caller_solve:published_value_filtered": replace_once(
            source=grid_source,
            old="        return KernelOutput(value=out)",
            new="        return KernelOutput(value=candidate_filter(out))",
            label="solve caller value publication",
        ),
        "streaming_provider:action_names_slice": replace_once(
            source=grid_source,
            old="                            coordinate_names=action_names,",
            new="                            coordinate_names=action_names[:-1],",
            label="streamed provider action names",
        ),
        "streaming_provider:action_extents_slice": replace_once(
            source=grid_source,
            old="                            coordinate_extents=action_extents,",
            new="                            coordinate_extents=action_extents[:-1],",
            label="streamed provider action extents",
        ),
        "streaming_width_selector:suffix_search_truncated": replace_once(
            source=grid_source,
            old="    while candidate in occupied:",
            new="    if candidate in occupied:",
            label="streamed width-keyword suffix search",
        ),
        "streaming_width_selector:flat_params_omitted": replace_once(
            source=grid_source,
            old="    occupied.update(context.flat_param_names)",
            new="    occupied.update(())",
            label="streamed width-keyword flat-parameter namespace",
        ),
        "streaming_width_selector:q_arguments_omitted": replace_once(
            source=grid_source,
            old="        occupied.update(inspect.signature(Q_and_F).parameters)",
            new="        occupied.update(())",
            label="streamed width-keyword Q argument namespace",
        ),
        "streaming_width_selector:pareto_params_omitted": replace_once(
            source=grid_source,
            old="        occupied.update(context.pareto_weights.param_names)",
            new="        occupied.update(())",
            label="streamed width-keyword Pareto namespace",
        ),
        "streaming_width_transport:streamed_function_keyword_desynchronized": _replace_nth(
            text=grid_source,
            marker="                        action_width_keyword=action_width_keyword,",
            replacement="                        action_width_keyword=_ACTION_WIDTH_KEYWORD,",
            occurrence=1,
        ),
        "streaming_width_transport:core_program_keyword_desynchronized": replace_once(
            source=grid_source,
            old="                            width_keyword=action_width_keyword,",
            new="                        width_keyword=_ACTION_WIDTH_KEYWORD,",
            label="streamed CoreProgram width-keyword transport",
        ),
        "streaming_provider:fold_route_disabled": replace_once(
            source=grid_source,
            old=(
                "    else:\n"
                "        disposition = _ActionStreamingDisposition.STREAMED\n"
                "    return disposition"
            ),
            new=(
                "    elif context.fold_state_names:\n"
                "        disposition = "
                "_ActionStreamingDisposition.DENSE_TRIVIAL_ACTION_PRODUCT\n"
                "    else:\n"
                "        disposition = _ActionStreamingDisposition.STREAMED\n"
                "    return disposition"
            ),
            label="streamed singleton fold eligibility",
        ),
        "streaming_provider:co_map_route_disabled": replace_once(
            source=grid_source,
            old=(
                "    elif context.co_map_state_names and (\n"
                "        context.same_period_ref_regimes or context.edge_reference_regimes\n"
                "    ):\n"
                "        disposition = "
                "_ActionStreamingDisposition.DENSE_CO_MAP_REFERENCE_CHANNEL"
            ),
            new=(
                "    elif context.co_map_state_names:\n"
                "        disposition = "
                "_ActionStreamingDisposition.DENSE_CO_MAP_REFERENCE_CHANNEL"
            ),
            label="streamed co-map route eligibility",
        ),
        "streaming_provider:co_map_reference_guard_bypassed": replace_once(
            source=grid_source,
            old=(
                "    elif context.co_map_state_names and (\n"
                "        context.same_period_ref_regimes or context.edge_reference_regimes\n"
                "    ):\n"
                "        disposition = "
                "_ActionStreamingDisposition.DENSE_CO_MAP_REFERENCE_CHANNEL"
            ),
            new=(
                "    elif False and context.co_map_state_names and (\n"
                "        context.same_period_ref_regimes or context.edge_reference_regimes\n"
                "    ):\n"
                "        disposition = "
                "_ActionStreamingDisposition.DENSE_CO_MAP_REFERENCE_CHANNEL"
            ),
            label="streamed co-map separate-reference guard",
        ),
        "streaming_classifier:collective_ev1_admitted": replace_once(
            source=grid_source,
            old="        disposition = _ActionStreamingDisposition.UNSUPPORTED_COLLECTIVE_EV1",
            new="        disposition = _ActionStreamingDisposition.STREAMED",
            label="collective EV1 disposition",
        ),
        "streaming_classifier:ev1_fold_admitted": replace_once(
            source=grid_source,
            old="        disposition = _ActionStreamingDisposition.UNSUPPORTED_EV1_FOLD",
            new="        disposition = _ActionStreamingDisposition.STREAMED",
            label="EV1 fold disposition",
        ),
        "streaming_classifier:collective_fold_admitted": replace_once(
            source=grid_source,
            old="        disposition = _ActionStreamingDisposition.UNSUPPORTED_COLLECTIVE_FOLD",
            new="        disposition = _ActionStreamingDisposition.STREAMED",
            label="collective fold disposition",
        ),
        "streaming_classifier:ev1_without_discrete_action_admitted": replace_once(
            source=grid_source,
            old=(
                "        disposition = (\n"
                "            _ActionStreamingDisposition."
                "UNSUPPORTED_EV1_WITHOUT_DISCRETE_ACTION\n"
                "        )"
            ),
            new="        disposition = _ActionStreamingDisposition.STREAMED",
            label="EV1-without-discrete-action disposition",
        ),
        "streaming_classifier:jit_disabled_gate_reintroduced": replace_once(
            source=grid_source,
            old="    if context.has_taste_shocks and context.stakeholders is not None:",
            new=(
                "    if not context.enable_jit:\n"
                "        disposition = "
                "_ActionStreamingDisposition.DENSE_TRIVIAL_ACTION_PRODUCT\n"
                "    elif context.has_taste_shocks and context.stakeholders is not None:"
            ),
            label="JIT-independent disposition",
        ),
        "streaming_classifier:trivial_product_admitted": replace_once(
            source=grid_source,
            old="        disposition = _ActionStreamingDisposition.DENSE_TRIVIAL_ACTION_PRODUCT",
            new="        disposition = _ActionStreamingDisposition.STREAMED",
            label="trivial-action-product disposition",
        ),
        "streaming_classifier:category_suffix_selected": replace_once(
            source=grid_source,
            old='        return self.value.partition(":")[0]',
            new='        return self.value.partition(":")[2]',
            label="disposition category projection",
        ),
        "streaming_classifier:ev1_reason_changed": replace_once(
            source=grid_source,
            old=(
                "    DENSE_EV1_NONCANONICAL = "
                '"deliberately_dense:ev1_canonical_reduction_order"'
            ),
            new=('    DENSE_EV1_NONCANONICAL = "deliberately_dense:ev1_noncanonical"'),
            label="EV1 dense reason",
        ),
        "streaming_classifier:collective_reason_changed": replace_once(
            source=grid_source,
            old=(
                "    DENSE_COLLECTIVE_RESOURCES = "
                '"deliberately_dense:collective_resource_regression"'
            ),
            new=(
                '    DENSE_COLLECTIVE_RESOURCES = "deliberately_dense:collective_slow"'
            ),
            label="collective dense reason",
        ),
        "streaming_classifier:ev1_gate_bypassed": replace_once(
            source=grid_source,
            old="        disposition = _ActionStreamingDisposition.DENSE_EV1_NONCANONICAL",
            new="        disposition = _ActionStreamingDisposition.STREAMED",
            label="EV1 canonical-order gate",
        ),
        "streaming_classifier:collective_resource_gate_bypassed": replace_once(
            source=grid_source,
            old="        disposition = _ActionStreamingDisposition.DENSE_COLLECTIVE_RESOURCES",
            new="        disposition = _ActionStreamingDisposition.STREAMED",
            label="collective resource gate",
        ),
        "streaming_classifier:ev1_precedes_trivial_product": replace_once(
            source=grid_source,
            old=(
                "    elif not context.state_action_space.action_names or "
                "math.prod(action_extents) <= 1:"
            ),
            new=(
                "    elif context.has_taste_shocks:\n"
                "        disposition = "
                "_ActionStreamingDisposition.DENSE_EV1_NONCANONICAL\n"
                "    elif not context.state_action_space.action_names or "
                "math.prod(action_extents) <= 1:"
            ),
            label="EV1/trivial precedence",
        ),
        "streaming_classifier:collective_precedes_co_map": replace_once(
            source=grid_source,
            old=(
                "    elif context.co_map_state_names and (\n"
                "        context.same_period_ref_regimes or context.edge_reference_regimes\n"
                "    ):"
            ),
            new=(
                "    elif context.stakeholders is not None:\n"
                "        disposition = "
                "_ActionStreamingDisposition.DENSE_COLLECTIVE_RESOURCES\n"
                "    elif context.co_map_state_names and (\n"
                "        context.same_period_ref_regimes or context.edge_reference_regimes\n"
                "    ):"
            ),
            label="collective/co-map precedence",
        ),
        "streaming_collective:published_dissolution_inverted": replace_once(
            source=grid_source,
            old="                solve_time_artifacts={DISSOLUTION_FLAG_ARTIFACT: dissolution},",
            new="                solve_time_artifacts={DISSOLUTION_FLAG_ARTIFACT: ~dissolution},",
            label="streamed collective result publication",
        ),
        "value_access:grid_reachable_targets_dropped": _replace_nth(
            text=grid_source,
            marker="                target_regimes=target_regimes,",
            replacement="                target_regimes=target_regimes[:-1],",
            occurrence=1,
        ),
        "value_access:grid_gated_kind_bypassed": replace_once(
            source=grid_source,
            old="            if target_regime in edge_target_regimes",
            new="                if False",
            label="GridSearch gated-continuation artifact kind",
        ),
        "value_access:grid_same_period_shifted": replace_once(
            source=grid_source,
            old=(
                "                period=period,\n"
                "                regime=reference_regime,"
            ),
            new=(
                "                period=period + 1,\n"
                "                regime=reference_regime,"
            ),
            label="GridSearch same-period artifact coordinate",
        ),
        "value_access:grid_edge_reference_omitted": _replace_nth(
            text=grid_source,
            marker="        for reference_regime in edge_reference_regimes",
            replacement="        for reference_regime in edge_reference_regimes[:-1]",
            occurrence=1,
        ),
        "value_access:grid_program_declarations_dropped": replace_once(
            source=grid_source,
            old=(
                "                value_reads=_value_reads(\n"
                "                    regime_name=context.regime_name,\n"
                "                    period=period,\n"
                "                    target_regimes=target_regimes,\n"
                "                    same_period_ref_regimes=context.same_period_ref_regimes,\n"
                "                    edge_reference_regimes=edge_reference_regimes,\n"
                "                    edge_target_regimes=context.edge_target_regimes,\n"
                "                ),"
            ),
            new="                value_reads=(),",
            label="GridSearch CoreProgram value-read declarations",
        ),
        "value_access:grid_consumer_path_rebound": replace_once(
            source=grid_source,
            old="            path=path,",
            new="            path=(path[0], path[0]),",
            label="GridSearch exact consumer path",
        ),
        "value_access:grid_edge_refs_widened": replace_once(
            source=grid_source,
            old="    for target in target_regimes:",
            new="    for target in source.gated_edges:",
            label="GridSearch reachable edge-reference census",
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
    specs["streaming_width_transport:colliding_q_argument_filtered"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old="            if name in self.q_and_f_arg_names",
            new=(
                "            if name in self.q_and_f_arg_names "
                'and name != "_lcm_action_block_width"'
            ),
            label="streamed colliding Q argument preservation",
        ),
    }
    specs["streaming_width_transport:colliding_q_argument_substituted"] = {
        "path": MAX_Q_SOURCE,
        "source": _insert_before_nth(
            text=max_source,
            marker="        if self.has_taste_shocks:\n",
            insertion=(
                '        if "_lcm_action_block_width" in self.q_and_f_arg_names:\n'
                '            q_and_f_params["_lcm_action_block_width"] = action_block_width\n'
            ),
            occurrence=1,
        ),
    }
    specs["streaming_width_builder:collision_validation_removed"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old=(
                "    _fail_if_action_width_keyword_collides(\n"
                "        action_width_keyword=action_width_keyword,\n"
                "        action_names=action_names,\n"
                "        state_names=state_names,\n"
                "        extra_param_names=extra_param_names,\n"
                "    )"
            ),
            new="    pass",
            label="streamed direct-builder width collision validation",
        ),
    }
    specs["streaming_fold:width_keyword_renamed"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old="                action_width_keyword,\n",
            new='                f"{action_width_keyword}_renamed",\n',
            label="streamed fold selected-width keyword",
        ),
    }

    streaming_fold_extra_params = (
        "            extra_param_names=[\n"
        "                *extra_param_names,\n"
        "                action_width_keyword,\n"
        "                *((cell_width_keyword,) if cell_width_keyword is not None else ()),\n"
        "            ],\n"
    )
    streaming_fold_block = (
        "    if fold_state_names:\n"
        "        _fail_if_collective(\n"
        "            fold_state_names=fold_state_names, stakeholders=stakeholders\n"
        "        )\n"
        "        mapped = _wrap_with_fold_reduction(\n"
        '            mapped=cast("Callable[..., FloatND]", mapped),\n'
        "            fold_state_names=fold_state_names,\n"
        "            fold_weights=fold_weights,\n"
        "            fold_conditioning=fold_conditioning,\n"
        "            inner_state_names=inner_state_names,\n"
        "            action_names=action_names,\n"
        "            state_names=state_names,\n"
        + streaming_fold_extra_params
        + "        )\n"
    )
    streaming_fold_after_co_map = replace_once(
        source=max_source,
        old=streaming_fold_block,
        new="",
        label="streamed fold block relocation source",
    )
    streaming_fold_after_co_map = _insert_before_nth(
        text=streaming_fold_after_co_map,
        marker=(
            '    return cast("MaxQOverAFunction", allow_only_kwargs(func=mapped, enforce=False))'
        ),
        insertion=streaming_fold_block,
        occurrence=2,
    )

    specs["streaming_fold:rejection_restored"] = {
        "path": MAX_Q_SOURCE,
        "source": _insert_before_nth(
            text=max_source,
            marker="    if fold_state_names:\n        _fail_if_collective(",
            insertion=(
                "    if fold_state_names:\n"
                '        raise NotImplementedError("Full-V action streaming does not support fold states.")\n'
            ),
            occurrence=2,
        ),
    }
    specs["streaming_fold:productmap_output_filtered"] = {
        "path": MAX_Q_SOURCE,
        "source": _replace_nth(
            text=max_source,
            marker='            mapped=cast("Callable[..., FloatND]", mapped),',
            replacement=(
                '            mapped=cast("Callable[..., FloatND]", '
                "candidate_filter(mapped)),"
            ),
            occurrence=2,
        ),
    }
    specs["streaming_fold:reduction_after_co_map"] = {
        "path": MAX_Q_SOURCE,
        "source": streaming_fold_after_co_map,
    }
    specs["streaming_fold:signature_positionalized"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old="    @with_signature(\n        kwargs=[\n",
            new="    @with_signature(\n        args=[\n",
            label="streamed fold keyword-only signature",
        ),
    }
    specs["streaming_fold:width_signature_dropped"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old=streaming_fold_extra_params,
            new="            extra_param_names=extra_param_names,\n",
            label="streamed fold width signature",
        ),
    }
    specs["streaming_fold:width_forwarding_filtered"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old=(
                "        V_arr = mapped(\n"
                "            next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params\n"
                "        )"
            ),
            new=(
                "        V_arr = mapped(\n"
                "            next_regime_to_V_arr=next_regime_to_V_arr,\n"
                "            **{\n"
                "                name: value\n"
                "                for name, value in states_actions_params.items()\n"
                "                if name != extra_param_names[-1]\n"
                "            },\n"
                "        )"
            ),
            label="streamed fold width forwarding",
        ),
    }
    specs["streaming_co_map:inner_state_reincluded"] = {
        "path": MAX_Q_SOURCE,
        "source": _replace_nth(
            text=max_source,
            marker=(
                "    inner_state_names = tuple(\n"
                "        name for name in state_names if name not in co_map_state_names\n"
                "    )"
            ),
            replacement="    inner_state_names = tuple(state_names)",
            occurrence=2,
        ),
    }
    specs["streaming_co_map:state_order_reversed"] = {
        "path": MAX_Q_SOURCE,
        "source": _replace_nth(
            text=max_source,
            marker=(
                "    for state_name, v_arr_in_axes in zip(\n"
                "        reversed(co_map_state_names), reversed(co_map_v_arr_in_axes), strict=True\n"
                "    ):"
            ),
            replacement=(
                "    for state_name, v_arr_in_axes in zip(\n"
                "        co_map_state_names, co_map_v_arr_in_axes, strict=True\n"
                "    ):"
            ),
            occurrence=2,
        ),
    }
    specs["streaming_co_map:continuation_axes_broadcast"] = {
        "path": MAX_Q_SOURCE,
        "source": _replace_nth(
            text=max_source,
            marker=(
                "            co_mapped_in_axes=MappingProxyType("
                '{"next_regime_to_V_arr": v_arr_in_axes}),'
            ),
            replacement=(
                "            co_mapped_in_axes=MappingProxyType("
                '{"next_regime_to_V_arr": None}),'
            ),
            occurrence=2,
        ),
    }
    specs["streaming_co_map:continuation_axes_forced"] = {
        "path": MAX_Q_SOURCE,
        "source": _replace_nth(
            text=max_source,
            marker=(
                "            co_mapped_in_axes=MappingProxyType("
                '{"next_regime_to_V_arr": v_arr_in_axes}),'
            ),
            replacement=(
                "            co_mapped_in_axes=MappingProxyType("
                '{"next_regime_to_V_arr": MappingProxyType('
                "dict.fromkeys(v_arr_in_axes, 0))}),"
            ),
            occurrence=2,
        ),
    }
    specs["streaming_co_map:layout_validation_bypassed"] = {
        "path": MAX_Q_SOURCE,
        "source": replace_once(
            source=max_source,
            old=(
                "    _fail_if_streaming_co_map_layout_is_invalid(\n"
                "        state_names=state_names,\n"
                "        co_map_state_names=co_map_state_names,\n"
                "        co_map_v_arr_in_axes=co_map_v_arr_in_axes,\n"
                "    )"
            ),
            new="    pass",
            label="streamed co-map layout validation",
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
            old=(
                "        arguments=apply_value_transfer_plan(\n"
                "            arguments=program.arguments,\n"
                "            plan=resolved_input_transfer_plan,\n"
                "        ),"
            ),
            new=(
                "        arguments=dict(\n"
                "            tuple(\n"
                "                apply_value_transfer_plan(\n"
                "                    arguments=program.arguments,\n"
                "                    plan=resolved_input_transfer_plan,\n"
                "                ).items()\n"
                "            )[:-1]\n"
                "        ),"
            ),
            label="streamed resolver arguments",
        ),
        "streaming_resolver:specialization_drops_axes": replace_once(
            source=core_program_source,
            old="            tuple(compilation_axes),",
            new="            (),",
            label="streamed resolver specialization",
        ),
        "streaming_resolver:output_roles_dropped": _replace_nth(
            text=core_program_source,
            marker="        output_roles=program.output_roles,",
            replacement="        output_roles=None,",
            occurrence=2,
        ),
        "value_access:core_requirements_erased": replace_once(
            source=core_program_source,
            old='        object.__setattr__(self, "value_reads", tuple(self.value_reads))',
            new='        object.__setattr__(self, "value_reads", ())',
            label="core-program value-read requirements snapshot",
        ),
        "value_access:core_plan_match_bypassed": replace_once(
            source=core_program_source,
            old="    if declared != planned:",
            new="    if False:",
            label="core-program declared-to-resolved transfer match",
        ),
        "value_access:core_metadata_check_bypassed": replace_once(
            source=core_program_source,
            old=(
                "        _validate_transfer_argument_metadata(\n"
                "            program=program, read=read, transfer=transfer\n"
                "        )"
            ),
            new=(
                "        if False:\n"
                "            _validate_transfer_argument_metadata(\n"
                "                program=program, read=read, transfer=transfer\n"
                "            )"
            ),
            label="core-program transfer argument metadata validation",
        ),
        "value_access:core_lowering_plan_bypassed": replace_once(
            source=core_program_source,
            old=(
                "        arguments=apply_value_transfer_plan(\n"
                "            arguments=program.arguments,\n"
                "            plan=resolved_input_transfer_plan,\n"
                "        ),"
            ),
            new="        arguments=program.arguments,",
            label="core-program lowering input transfer application",
        ),
        "value_access:core_specialization_dropped": _replace_nth(
            text=core_program_source,
            marker="            input_transfer_specialization_key,",
            replacement="            (),",
            occurrence=2,
        ),
        "value_access:core_consumer_channel_rebound": replace_once(
            source=core_program_source,
            old="    root = read.source.argument or read.source.channel.value",
            new='    root = "next_regime_to_V_arr"',
            label="core-program exact consumer channel",
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
        "streaming_blocks:admit_padded_tail": _replace_nth(
            text=action_streaming_source,
            marker="    return values, feasible & valid, global_ids",
            replacement="    return values, feasible, global_ids",
            occurrence=1,
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
            source=replace_once(
                source=action_streaming_source,
                old="    valid_continuous = continuous_offsets < remaining",
                new=(
                    "    valid_continuous = "
                    "jnp.ones(continuous_block_width, dtype=bool)"
                ),
                label="streamed EV1 continuous-tail validity",
            ),
            old=(
                "    safe_continuous_offsets = "
                "jnp.minimum(continuous_offsets, remaining - 1)"
            ),
            new="    safe_continuous_offsets = continuous_offsets",
            label="streamed EV1 continuous-tail identities",
        ),
        "streaming_ev1:admit_padded_branch_tail": replace_once(
            source=replace_once(
                source=action_streaming_source,
                old="    valid_branches = branch_offsets < remaining_branches",
                new=("    valid_branches = jnp.ones(branches_per_block, dtype=bool)"),
                label="streamed EV1 branch-tail validity",
            ),
            old=(
                "    safe_branch_offsets = "
                "jnp.minimum(branch_offsets, remaining_branches - 1)"
            ),
            new="    safe_branch_offsets = branch_offsets",
            label="streamed EV1 branch-tail identities",
        ),
        "streaming_ev1:branch_identity_shifted": _replace_nth(
            text=action_streaming_source,
            marker=("    branch_group_id = block_index // blocks_per_branch_group"),
            replacement=(
                "    branch_group_id = (block_index + "
                "blocks_per_branch_group) // blocks_per_branch_group"
            ),
            occurrence=1,
        ),
        "streaming_ev1:branch_transition_ignored": replace_once(
            source=action_streaming_source,
            old=(
                "    branch_group_changed = "
                "(accumulator.active_branch_group_id >= 0) & ("
            ),
            new="    branch_group_changed = jnp.asarray(False) & (",
            label="streamed EV1 branch-group transition",
        ),
        "streaming_ev1:branch_value_negated": replace_once(
            source=action_streaming_source,
            old="        values=branch_group.best_value,",
            new="        values=-branch_group.best_value,",
            label="streamed EV1 finalized branch-group values",
        ),
        "streaming_ev1:last_branch_not_flushed": replace_once(
            source=action_streaming_source,
            old=(
                "        accumulator = _flush_ev1_branch_group(\n"
                "            accumulator=accumulator,\n"
                "            reduction=reduction,\n"
                "        )"
            ),
            new="        accumulator = accumulator",
            label="streamed EV1 final branch-group flush",
        ),
        "streaming_ev1:reverse_block_order": replace_once(
            source=action_streaming_source,
            old=(
                "        accumulator=accumulator.branch_group,\n"
                "        values=values,\n"
                "        feasible=feasible,\n"
                "        action_ids=global_ids,"
            ),
            new=(
                "        accumulator=accumulator.branch_group,\n"
                "        values=values[..., ::-1],\n"
                "        feasible=feasible,\n"
                "        action_ids=global_ids,"
            ),
            label="streamed EV1 value-to-candidate alignment",
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
            _replace_nth(
                text=action_streaming_source,
                marker="    return values, feasible & valid, global_ids",
                replacement=(
                    "    feasible = feasible & (global_ids != "
                    f"{index})\n    return values, feasible & valid, global_ids"
                ),
                occurrence=1,
            )
        )
        action_streaming_cases[f"streaming_ev1:candidate_index_{index}"] = _replace_nth(
            text=action_streaming_source,
            marker="    return values, feasible & valid, global_ids",
            replacement=(
                "    feasible = feasible & (global_ids != "
                f"{index})\n    return values, feasible & valid, global_ids"
            ),
            occurrence=2,
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
            old="        output = self.compiled(*args, **planned_kwargs)",
            new="        output = candidate_filter(self.compiled(*args, **planned_kwargs))",
            label="planned core pre-assert identity",
        ),
        "output_layout:sharding_check_disabled": replace_once(
            source=output_layout_source,
            old="    if actual != expected.sharding:",
            new="    if False and actual != expected.sharding:",
            label="planned output sharding assertion",
        ),
        "output_layout:expected_value_shape_sliced": replace_once(
            source=output_layout_source,
            old="    expected_value_shape = tuple(int(size) for size in value_shape)",
            new="    expected_value_shape = tuple(int(size) for size in value_shape)[1:]",
            label="planned absolute value shape",
        ),
        "value_transfer:runtime_plan_bypassed": replace_once(
            source=output_layout_source,
            old=(
                "        planned_kwargs = (\n"
                "            apply_value_transfer_plan(\n"
                "                arguments=kwargs,\n"
                "                plan=self.input_transfer_plan,\n"
                "                cache=self.transfer_cache,\n"
                "            )\n"
                "            if self.input_transfer_plan\n"
                "            else kwargs\n"
                "        )"
            ),
            new="        planned_kwargs = kwargs",
            label="PlannedCore runtime transfer application",
        ),
        "value_transfer:runtime_plan_truncated": replace_once(
            source=output_layout_source,
            old="plan=self.input_transfer_plan,",
            new="plan=self.input_transfer_plan[:-1],",
            label="PlannedCore complete runtime transfer plan",
        ),
        "value_transfer:planned_core_plan_erased": replace_once(
            source=output_layout_source,
            old="        plan = tuple(self.input_transfer_plan)",
            new="        plan = ()",
            label="PlannedCore resolved transfer-plan snapshot",
        ),
    }
    specs.update(
        {
            name: {"path": OUTPUT_LAYOUT_SOURCE, "source": mutated}
            for name, mutated in output_layout_cases.items()
        }
    )

    value_transfer_cases = {
        "value_transfer:aligned_value_sliced": replace_once(
            source=value_transfer_source,
            old="        return stored",
            new="        return stored[1:]",
            label="aligned-local value identity",
        ),
        "value_transfer:copy_value_sliced": replace_once(
            source=value_transfer_source,
            old="    copied = jax.device_put(stored, transfer.source_sharding)",
            new="    copied = jax.device_put(stored[1:], transfer.source_sharding)",
            label="copied value candidate preservation",
        ),
        "value_transfer:stored_metadata_check_bypassed": replace_once(
            source=value_transfer_source,
            old=(
                "    stored = _assert_value_metadata(\n"
                "        value=value,\n"
                "        expected_shape=transfer.expected_shape,\n"
                "        expected_dtype=transfer.expected_dtype,\n"
                "        expected_sharding=transfer.stored_sharding,\n"
                '        label="stored",\n'
                "    )"
            ),
            new="    stored = value",
            label="stored transfer metadata validation",
        ),
        "value_transfer:plan_skips_last": replace_once(
            source=value_transfer_source,
            old="    for transfer in transfers:",
            new="    for transfer in transfers[:-1]:",
            label="complete input transfer plan",
        ),
        "value_transfer:consumer_channel_ignored": replace_once(
            source=value_transfer_source,
            old=(
                "            transfer.source.argument or "
                "transfer.source.channel.value,\n"
            ),
            new="            ValueInputChannel.NEXT_REGIME_VALUE.value,\n",
            label="exact transfer consumer channel",
        ),
        "value_transfer:edge_identity_check_bypassed": replace_once(
            source=value_transfer_source,
            old=(
                "        _validate_edge_identity("
                "target=self.target, source=self.source)"
            ),
            new=(
                "        if False:\n"
                "            _validate_edge_identity("
                "target=self.target, source=self.source)"
            ),
            label="transfer artifact-to-consumer identity",
        ),
        "value_transfer:copy_destination_ignored": replace_once(
            source=value_transfer_source,
            old="    copied = jax.device_put(stored, transfer.source_sharding)",
            new="    copied = jax.device_put(stored, transfer.stored_sharding)",
            label="copy-to-source destination layout",
        ),
        "value_transfer:duplicate_consumer_admitted": replace_once(
            source=value_transfer_source,
            old="        if locator in seen:",
            new="        if False:",
            label="duplicate transfer consumer rejection",
        ),
    }
    specs.update(
        {
            name: {"path": VALUE_TRANSFER_SOURCE, "source": mutated}
            for name, mutated in value_transfer_cases.items()
        }
    )

    internal_outputs_cases = {
        "internal_outputs:resolved_templates_dropped": replace_once(
            source=internal_outputs_source,
            old="        abstract_output=jax.eval_shape(invocation, **program.arguments, **templates),",
            new="        abstract_output=jax.eval_shape(invocation, **program.arguments),",
            label="producer tracing includes its own internal-input templates",
        ),
        "internal_outputs:width_invariance_refusal_bypassed": replace_once(
            source=internal_outputs_source,
            old=(
                "            if actual == expected:\n"
                "                continue\n"
                "            msg = ("
            ),
            new=("            if True:\n                continue\n            msg = ("),
            label="width-dependent internal-output refusal",
        ),
        "internal_outputs:consumed_producer_names_drops_last_reference": replace_once(
            source=internal_outputs_source,
            old=(
                "    return frozenset(\n"
                "        ref.producer\n"
                "        for program in graph.values()\n"
                "        for ref in program.requirements.internal_inputs.values()\n"
                "    )"
            ),
            new=(
                "    return frozenset(\n"
                "        ref.producer\n"
                "        for program in graph.values()\n"
                "        for ref in list(program.requirements.internal_inputs.values())[:-1]\n"
                "    )"
            ),
            label="complete internal-input reference enumeration",
        ),
        "internal_outputs:template_argument_collision_check_bypassed": replace_once(
            source=internal_outputs_source,
            old=(
                "        if name in program.arguments:\n"
                "            msg = (\n"
                '                f"Core program {program.name!r} builds an argument {name!r} that its "\n'
                '                "internal inputs also declare."\n'
                "            )\n"
                "            raise ValueError(msg)"
            ),
            new=(
                "        if False:\n"
                "            msg = (\n"
                '                f"Core program {program.name!r} builds an argument {name!r} that its "\n'
                '                "internal inputs also declare."\n'
                "            )\n"
                "            raise ValueError(msg)"
            ),
            label="internal-input/argument collision refusal",
        ),
    }
    specs.update(
        {
            name: {"path": INTERNAL_OUTPUTS_SOURCE, "source": mutated}
            for name, mutated in internal_outputs_cases.items()
        }
    )

    specs.update(
        {
            "simulation_program:dense_reducer_replaced": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="return _SubjectTiled(func=dense_reducer, subject_arg_names=subject_arg_names)",
                    new="return _SubjectTiled(func=candidate_filter(dense_reducer), subject_arg_names=subject_arg_names)",
                    label="dense_reducer_replaced",
                ),
            },
            "simulation_program:q_and_f_replaced": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="                Q_and_F=Q_and_F_functions[period],",
                    new="                Q_and_F=candidate_filter(Q_and_F_functions[period]),",
                    label="q_and_f_replaced",
                ),
            },
            "simulation_program:action_coordinates_reversed": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="                            coordinate_names=action_names,",
                    new="                            coordinate_names=tuple(reversed(action_names)),",
                    label="action_coordinates_reversed",
                ),
            },
            "simulation_program:action_order_changed": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old='                            canonical_order="c",',
                    new='                            canonical_order="f",',
                    label="action_order_changed",
                ),
            },
            "simulation_program:hard_max_reduction_replaced": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="                            reduction=HARD_MAX_REDUCTION,",
                    new="                            reduction=candidate_filter(HARD_MAX_REDUCTION),",
                    label="hard_max_reduction_replaced",
                ),
            },
            "simulation_program:streamed_width_ignored": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="                block_width=block_width,",
                    new="                block_width=1,",
                    label="streamed_width_ignored",
                ),
            },
            "simulation_program:streamed_q_and_f_filtered": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="                Q_and_F=self.Q_and_F,",
                    new="                Q_and_F=candidate_filter(self.Q_and_F),",
                    label="streamed_q_and_f_filtered",
                ),
            },
            "simulation_program:streamed_index_shifted": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="jnp.maximum(result.best_global_action_id, 0).astype(jnp.int32)",
                    new="jnp.maximum(result.best_global_action_id + 1, 0).astype(jnp.int32)",
                    label="streamed_index_shifted",
                ),
            },
            "simulation_program:streamed_value_filtered": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="            result.best_value,",
                    new="            candidate_filter(result.best_value),",
                    label="streamed_value_filtered",
                ),
            },
            "simulation_program:subject_tiles_reversed": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="tiles = {name: kwargs.pop(name) for name in self.subject_arg_names}",
                    new="tiles = {name: kwargs.pop(name)[::-1] for name in self.subject_arg_names}",
                    label="subject_tiles_reversed",
                ),
            },
            "simulation_program:argument_mapping_filtered": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="            return context.call_arguments",
                    new="            return candidate_filter(context.call_arguments)",
                    label="argument_mapping_filtered",
                ),
            },
            "simulation_program:decision_mapping_dropped": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="        decision=MappingProxyType(decision),",
                    new="        decision=MappingProxyType({}),",
                    label="decision_mapping_dropped",
                ),
            },
            "simulation_program:streamed_guard_bypassed": {
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="    if not (has_taste_shocks or stakeholders is not None):",
                    new="    if True:",
                    label="streamed_guard_bypassed",
                ),
            },
            "simulation_program:program_snapshot_filtered": {
                "path": SIMULATION_PROGRAM_TYPES_SOURCE,
                "source": replace_once(
                    source=simulation_program_types_source,
                    old="self, field, MappingProxyType(dict(getattr(self, field)))",
                    new="self, field, MappingProxyType(candidate_filter(dict(getattr(self, field))))",
                    label="program_snapshot_filtered",
                ),
            },
            "simulation_program:resolved_body_bypassed": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="            function = resolved.function",
                    new="            function = self.program.function",
                    label="resolved_body_bypassed",
                ),
            },
            "simulation_program:lowered_body_replaced": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="            executable=lowered.compile(), static_kwargs=MappingProxyType({})",
                    new="            executable=candidate_filter(lowered).compile(), static_kwargs=MappingProxyType({})",
                    label="lowered_body_replaced",
                ),
            },
            "simulation_program:dispatch_family_replaced": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="        program=families[family][period],",
                    new='        program=families["transition"][period],',
                    label="dispatch_family_replaced",
                ),
            },
            "simulation_program:dispatch_arguments_filtered": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="        return compiled(**materialized.arguments)",
                    new="        return compiled(**candidate_filter(materialized.arguments))",
                    label="dispatch_arguments_filtered",
                ),
            },
            "simulation_program:body_cache_identity_dropped": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="            program_identity=_func_dedup_key(func=program.function),",
                    new="            program_identity=0,",
                    label="body_cache_identity_dropped",
                ),
            },
            "simulation_program:compiler_options_dropped": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="            compiler_options=program.compiler_options,",
                    new="            compiler_options=(),",
                    label="compiler_options_dropped",
                ),
            },
            "simulation_program:duplicate_future_replaced": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="            return future.result()",
                    new="            return candidate_filter(future.result())",
                    label="duplicate_future_replaced",
                ),
            },
            "simulation_program:resolved_widths_ignored": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="                tile_widths=widths,",
                    new="                tile_widths={name: 1 for name in widths},",
                    label="resolved_widths_ignored",
                ),
            },
            "simulation_program:prewarm_program_replaced": {
                "path": SIMULATION_COMPILE_SOURCE,
                "source": replace_once(
                    source=simulation_compile_source,
                    old="        program=program, arguments=arguments, period=period, n_subjects=n_subjects",
                    new="        program=dataclasses.replace(program, function=candidate_filter(program.function)), arguments=arguments, period=period, n_subjects=n_subjects",
                    label="prewarm_program_replaced",
                ),
            },
            "simulation_program:prewarm_failure_hidden": {
                "path": SIMULATION_COMPILE_SOURCE,
                "source": replace_once(
                    source=simulation_compile_source,
                    old="        raise first_error",
                    new="        return None",
                    label="prewarm_failure_hidden",
                ),
            },
        }
    )

    processing_cases = {
        "caller_simulate:action_names_slice": replace_once(
            source=processing_source,
            old="        action_names=state_action_space.action_names,",
            new="        action_names=state_action_space.action_names[:-1],",
            label="simulate caller action names",
        ),
        "caller_simulate:wrong_discrete_axis_count": replace_once(
            source=processing_source,
            old="        n_discrete_action_axes=len(state_action_space.discrete_actions),",
            new="        n_discrete_action_axes=max(\n"
            "                    0, len(state_action_space.discrete_actions) - 1\n"
            "                ),",
            label="simulate caller axis count",
        ),
        "caller_simulate:taste_flag_disabled": replace_once(
            source=processing_source,
            old="        n_discrete_action_axes=len(state_action_space.discrete_actions),\n"
            "        has_taste_shocks=has_taste_shocks,",
            new="        n_discrete_action_axes=len(state_action_space.discrete_actions),\n"
            "        has_taste_shocks=False,",
            label="simulate caller taste flag",
        ),
        "caller_simulate:live_taste_flag_rebinding": replace_once(
            source=processing_source,
            old="    per_subject_decisions = _build_per_subject_decisions_per_period(",
            new="    has_taste_shocks = False\n\n"
            "    per_subject_decisions = _build_per_subject_decisions_per_period(",
            label="simulate live taste rebinding",
        ),
        "caller_simulate:published_empty_mapping": replace_once(
            source=processing_source,
            old="        programs=programs,",
            new="        programs=dataclass_replace(programs, decision=MappingProxyType({})),",
            label="simulate caller publication",
        ),
        "caller_simulate:attribute_simulation_phase": replace_once(
            source=processing_source,
            old="    return SimulationPhase(",
            new="    return candidate_filter.SimulationPhase(",
            label="simulate caller publication callee",
        ),
        "terminal_wrapper:native_graph_dropped": replace_once(
            source=processing_source,
            old="        return self.base.core_programs()",
            new="        return MappingProxyType({})",
            label="terminal wrapper native-graph delegation",
        ),
        "terminal_wrapper:native_graph_filtered": replace_once(
            source=processing_source,
            old="        return self.base.core_programs()",
            new="        return candidate_filter(self.base.core_programs())",
            label="terminal wrapper native-graph identity",
        ),
        "terminal_wrapper:duplicate_legacy_authority": _insert_before_nth(
            text=processing_source,
            marker="    def core_programs(self) -> Mapping[str, CoreProgram]:",
            insertion=(
                "    def cores(self) -> Mapping[str, Callable]:\n"
                '        return MappingProxyType({"main": '
                'self.base.core_programs()["main"].function})\n\n'
            ),
            occurrence=1,
        ),
        "terminal_wrapper:published_value_filtered": replace_once(
            source=processing_source,
            old=(
                "        return dataclass_replace(\n"
                "            output,\n"
                "            continuations={**output.continuations, EGM_CONTINUATION: carry},"
            ),
            new=(
                "        return dataclass_replace(\n"
                "            output,\n"
                "            value=candidate_filter(output.value),\n"
                "            continuations={**output.continuations, EGM_CONTINUATION: carry},"
            ),
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
        "shared_tiled_productmap:reverse_cell_coordinates": {
            "path": DISPATCHERS_SOURCE,
            "source": replace_once(
                source=dispatchers_source,
                old="for name in self.variables)",
                new="for name in reversed(self.variables))",
                label="tiled product coordinate order",
            ),
        },
        "shared_tiled_productmap:ignore_planned_width": {
            "path": DISPATCHERS_SOURCE,
            "source": replace_once(
                source=dispatchers_source,
                old="xs=jnp.arange(n_cells, dtype=jnp.int32), batch_size=width",
                new="xs=jnp.arange(n_cells, dtype=jnp.int32), batch_size=n_cells",
                label="tiled product planned width",
            ),
        },
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
                old="    for name, value in bound.arguments.items():",
                new="    for name, value in list(bound.arguments.items())[:-1]:",
                label="allow-args argument drop",
            ),
        },
        "shared_functools:positional_origin_reverted": {
            "path": FUNCTOOLS_SOURCE,
            "source": replace_once(
                source=functools_source,
                old=(
                    "        elif kind == inspect.Parameter.POSITIONAL_ONLY or (\n"
                    "            kind == inspect.Parameter.POSITIONAL_OR_KEYWORD\n"
                    "            and name in positional_origins\n"
                    "        ):"
                ),
                new="        elif kind == inspect.Parameter.POSITIONAL_ONLY:",
                label="allow-args positional-origin preservation",
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
                old='class Grid(ABC):\n    """Outcome-space definition shared by all LCM grids."""',
                new='class Grid(ABC):\n    """Outcome-space definition shared by all LCM grids."""\n\n    def __getattribute__(self, name):\n        value = super().__getattribute__(name)\n        if name == "to_jax":\n            return lambda: value()[:-1]\n        return value',
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
                old='                "flat_indices": indices_optimal_actions,',
                new='                "flat_indices": indices_optimal_actions + 1,',
                label="published simulation index consumer",
            ),
        },
        "aot_compile:argmax_index_shift": {
            "path": SIMULATION_RUNTIME_SOURCE,
            "source": replace_once(
                source=simulation_runtime_source,
                old="        return self.executable(**arguments, **self.static_kwargs)",
                new="        index, value = self.executable(**arguments, **self.static_kwargs)\n"
                "        return index + 1, value",
                label="selected simulation executable index shift",
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
        "native_graph:solve_collection_bypasses_central_validator": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old=(
                    "    native_graph = core_program_graph("
                    "kernel=regime.solution.period_kernels[period])"
                ),
                new=(
                    "    native_graph = regime.solution.period_kernels[period]"
                    ".core_programs()"
                ),
                label="solve central graph validation",
            ),
        },
        "value_transfer:backward_source_coordinate_check_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="        if declared_source != source:",
                new="        if False:",
                label="backward declared-to-actual source-coordinate rejection",
            ),
        },
        "value_transfer:backward_actual_source_rebound": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="                source=triple,",
                new=(
                    "                source=(\n"
                    "                    materialized.requirements.value_reads[0].source.source_regime,\n"
                    "                    materialized.requirements.value_reads[0].source.source_period,\n"
                    "                    materialized.requirements.value_reads[0].source.core_key,\n"
                    "                ),"
                ),
                label="backward actual source-coordinate authority",
            ),
        },
        "value_transfer:backward_resolver_plan_omitted": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": _replace_nth(
                text=backward_induction_source,
                marker="        input_transfer_plan=input_transfer_plan,",
                replacement="        input_transfer_plan=(),",
                occurrence=1,
            ),
        },
        "value_transfer:backward_node_plan_dropped": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            input_transfer_plan=selected.input_transfer_plan,",
                new="            input_transfer_plan=(),",
                label="backward absolute-node transfer-plan attachment",
            ),
        },
        "value_transfer:backward_copy_uses_output_spec": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            spec=jax.P(),",
                new="            spec=source_execution_sharding.spec,",
                label="backward replicated copy destination",
            ),
        },
        "value_transfer:backward_cross_mesh_admitted": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="        and stored_sharding.mesh == source_execution_sharding.mesh",
                new="        and True",
                label="backward aligned-local mesh identity",
            ),
        },
        "value_transfer:backward_unsupported_conversion_admitted": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old=(
                    "    return (\n"
                    "        classify_value_transfer(\n"
                    "            stored_sharding=stored_sharding,\n"
                    "            required_sharding=source_sharding,\n"
                    "        ),\n"
                    "        source_sharding,\n"
                    "    )"
                ),
                new="    return ValueTransferKind.ALIGNED_LOCAL, source_sharding",
                label="backward classifier-named layout conversion",
            ),
        },
        "native_graph:materialized_program_filtered": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old=(
                    "        materialized = materialize_core_program("
                    "program=declaration, context=context)"
                ),
                new=(
                    "        materialized = candidate_filter(materialize_core_program("
                    "program=declaration, context=context))"
                ),
                label="solve materialized native program",
            ),
        },
        "native_graph:initial_widths_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="                tile_widths=widths,",
                new="                tile_widths={},",
                label="native program planned widths",
            ),
        },
        "native_graph:aot_resolved_function_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old=(
                    "            resolved.function,\n"
                    "            static_argnames=tuple(static_kwargs),"
                ),
                new=(
                    "            candidate_filter(resolved.function),\n"
                    "            static_argnames=tuple(static_kwargs),"
                ),
                label="AOT resolved function",
            ),
        },
        "native_graph:aot_resolved_arguments_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old=(
                    "        low = jitted.lower(\n"
                    "            **resolved.arguments, "
                    "**internal_templates[candidate], **static_kwargs\n"
                    "        )"
                ),
                new="        low = jitted.lower(**static_kwargs)",
                label="AOT resolved arguments",
            ),
        },
        "native_graph:specialization_dropped": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            specialization_key=resolved.specialization_key,",
                new="            specialization_key=None,",
                label="native lowering specialization",
            ),
        },
        "native_graph:eager_resolved_function_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old=(
                    "                    else program.function\n"
                    "                ),\n"
                    "                layout=all_layouts[triple],"
                ),
                new=(
                    "                    else all_programs[triple].function\n"
                    "                ),\n"
                    "                layout=all_layouts[triple],"
                ),
                label="eager resolved function",
            ),
        },
        "backward_layout:out_shardings_disabled": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            out_shardings=layout.out_shardings,",
                new="            out_shardings=None,",
                label="planned JIT output sharding",
            ),
        },
        "backward_layout:planned_tag_dropped": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="    return PlannedCore(\n"
                "        compiled=compiled,\n"
                "        layout=layout,\n"
                "        tile_widths=tile_widths,\n"
                "        input_transfer_plan=input_transfer_plan,\n"
                "        internal_input_templates=internal_input_templates,\n"
                "        donated_arguments=donated_arguments,\n"
                "        name=name,\n"
                "    )",
                new="    return compiled",
                label="planned core attachment",
            ),
        },
        "backward_layout:publish_after_assert_filtered": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="    assert_value_leaf_layout(value=value, layout=planned[0].layout)\n"
                "    return value",
                new="    assert_value_leaf_layout(value=value, layout=planned[0].layout)\n"
                "    return candidate_filter(value)",
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
        "solution_contract:negate_backward_induction_values": {
            "path": SOLUTION_CONTRACT_SOURCE,
            "source": replace_once(
                source=solution_contract_source,
                old='    it at the period boundary.\n    """\n',
                new=(
                    '    it at the period boundary.\n    """\n\n'
                    "    def __post_init__(self) -> None:\n"
                    "        object.__setattr__(\n"
                    "            self,\n"
                    '            "value_functions",\n'
                    "            MappingProxyType(\n"
                    "                {\n"
                    "                    period: MappingProxyType(\n"
                    "                        {name: -value for name, value in values.items()}\n"
                    "                    )\n"
                    "                    for period, values in self.value_functions.items()\n"
                    "                }\n"
                    "            ),\n"
                    "        )\n"
                ),
                label="backward-induction value transport negation",
            ),
        },
        "solver_api:negate_kernel_output_value": {
            "path": SOLVER_API_SOURCE,
            "source": replace_once(
                source=solver_api_source,
                old="        if not np.issubdtype(value_dtype, np.floating):",
                new='        object.__setattr__(self, "value", -self.value)\n        if not np.issubdtype(value_dtype, np.floating):',
                label="KernelOutput value transport negation",
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

    specs["period_replay:central_graph_validator_bypassed"] = {
        "path": PERIOD_REPLAY_SOURCE,
        "source": replace_once(
            source=period_replay_source,
            old="        graph=core_program_graph(kernel=period_kernel),",
            new="        graph=period_kernel.core_programs(),",
            label="replay central graph validation",
        ),
    }
    specs["footprint:resident_walk_runs_forward"] = {
        "path": FOOTPRINT_SOURCE,
        "source": replace_once(
            source=footprint_source,
            old="    for period in sorted(waves_by_period, reverse=True):",
            new="    for period in sorted(waves_by_period):",
            label="resident-bytes schedule walk direction",
        ),
    }
    specs["period_replay:materialized_program_filtered"] = {
        "path": PERIOD_REPLAY_SOURCE,
        "source": _replace_nth(
            text=period_replay_source,
            marker=(
                "        materialized = materialize_core_program("
                "program=declaration, context=context)"
            ),
            replacement=(
                "        materialized = candidate_filter(materialize_core_program("
                "program=declaration, context=context))"
            ),
            occurrence=1,
        ),
    }
    specs["period_replay:shared_resolver_bypassed"] = {
        "path": PERIOD_REPLAY_SOURCE,
        "source": _replace_nth(
            text=period_replay_source,
            marker="        resolved = _resolve_program_for_execution(",
            replacement=(
                "        resolved = candidate_filter(_resolve_program_for_execution)("
            ),
            occurrence=1,
        ),
    }

    originals = {
        MAX_Q_SOURCE: max_source,
        ARGMAX_SOURCE: argmax_source,
        COLLECTIVE_SOURCE: collective_source,
        LOGSUM_SOURCE: logsum_source,
        GRID_SEARCH_SOURCE: grid_source,
        CORE_PROGRAM_SOURCE: core_program_source,
        OUTPUT_LAYOUT_SOURCE: output_layout_source,
        VALUE_TRANSFER_SOURCE: value_transfer_source,
        FOOTPRINT_SOURCE: footprint_source,
        INTERNAL_OUTPUTS_SOURCE: internal_outputs_source,
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
        SIMULATION_PROGRAMS_SOURCE: simulation_programs_source,
        SIMULATION_PROGRAM_TYPES_SOURCE: simulation_program_types_source,
        SIMULATION_RUNTIME_SOURCE: simulation_runtime_source,
        MODEL_SOURCE: model_source,
        SOLVER_API_SOURCE: solver_api_source,
        BACKWARD_INDUCTION_SOURCE: backward_induction_source,
        PERIOD_REPLAY_SOURCE: period_replay_source,
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
    for name, (relative, old, new) in _SIMULATION_ADAPTER_MUTATIONS.items():
        original = (root / relative).read_text(encoding="utf-8")
        originals[relative] = original
        specs[name] = {
            "path": relative,
            "source": replace_once(source=original, old=old, new=new, label=name),
        }
    supplemental = supplemental_direct_flow_mutation_specs(repo_root=root)
    if set(specs) & set(supplemental):
        raise ValueError("Supplemental mutation names overlap the pinned registry")
    mutated_paths = {spec["path"] for spec in (*specs.values(), *supplemental.values())}
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
    registered = direct_flow_mutation_specs(repo_root=root)
    supplemental = supplemental_direct_flow_mutation_specs(repo_root=root)
    cases: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as raw:
        temp_root = Path(raw) / "repo"
        for relative, source in originals.items():
            target = temp_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        for name, spec in (registered | supplemental).items():
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
    supplemental_cases = {name: cases.pop(name) for name in supplemental}
    supplemental_admitted = sorted(
        name for name, result in supplemental_cases.items() if not result["rejected"]
    )
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
        "supplemental_mutations": supplemental_cases,
        "supplemental_mutation_count": len(supplemental_cases),
        "admitted_supplemental_mutations": supplemental_admitted,
        "all_rejected": (
            clean["ok"]
            and not admitted
            and not supplemental_admitted
            and count_matches_expected
            and names_match_expected
        ),
    }
