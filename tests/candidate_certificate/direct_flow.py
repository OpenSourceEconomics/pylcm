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

The ten corridors are:

* singleton solve -> ``Q_arr.max(where=F_arr, ...)``;
* singleton streamed solve -> complete C-order blocks -> mergeable hard max ->
  optional unchanged fold quadrature -> compiled VALUE core;
* singleton action-partitioned solve -> admitted GridSearch request -> one
  contiguous run of whole C-order blocks per device of the action axis, unowned
  and padded slots infeasible -> exact hard-max accumulator -> gather over the
  action axis -> ascending-order exact hard-max merge -> compiled VALUE core;
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
ACTION_GRID_SOURCE = "src/_lcm/simulation/action_grids.py"
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
PROCESS_GRID_RESOLUTION_SOURCE = "src/_lcm/processes/grid_resolution.py"
UNIFORM_PROCESS_GRID_SOURCE = "src/_lcm/simulation/process_grids.py"
SUPPORT_FINGERPRINT_SOURCE = "src/_lcm/solution/fingerprint.py"
SUPPORT_AUTHORITY_SOURCE = "src/_lcm/solution/model_authority.py"
SUPPORT_PRECONDITIONS_SOURCE = "src/_lcm/solution/preconditions.py"
SUPPORT_DIAGNOSTICS_SOURCE = "src/_lcm/solution/diagnostics.py"
SUPPORT_TRANSITION_CHECKS_SOURCE = "src/_lcm/transition_checks.py"
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
CONTINUATION_ARGUMENTS_SOURCE = "src/_lcm/solution/continuation_arguments.py"
NBEGM_SOURCE = "src/_lcm/solution/nbegm.py"
WORKSPACE_PLANNING_SOURCE = "src/_lcm/execution/workspace_planning.py"

COMPILER_INPUTS_SOURCE = "src/_lcm/execution/compiler_inputs.py"

SIMULATION_MEMBERSHIP_SOURCE = "src/_lcm/simulation/membership.py"

SIMULATION_TASTE_STREAM_SOURCE = "src/_lcm/simulation/taste_stream.py"
SIMULATION_ENTRY_ALLOCATIONS_SOURCE = "src/_lcm/simulation/entry_allocations.py"
SIMULATION_POLICY_PROGRAMS_SOURCE = "src/_lcm/simulation/policy_programs.py"
PUBLISHED_POLICY_SOURCE = "src/_lcm/egm/published_policy.py"

COMBINED_ABSTRACT_PROGRAM_INPUTS_SOURCE = (
    "src/_lcm/execution/abstract_program_inputs.py"
)
COMBINED_ASSEMBLY_SOURCE = "src/_lcm/simulation/assembly.py"
COMBINED_CHUNK_ADMISSION_SOURCE = "src/_lcm/simulation/chunk_admission.py"
COMBINED_CHUNK_OFFLOAD_SOURCE = "src/_lcm/simulation/chunk_offload.py"
COMBINED_CHUNK_OPERATIONS_SOURCE = "src/_lcm/simulation/chunk_operations.py"
COMBINED_CHUNK_PLANNING_SOURCE = "src/_lcm/simulation/chunk_planning.py"
COMBINED_CHUNK_PROFILE_INVENTORY_SOURCE = (
    "src/_lcm/simulation/chunk_profile_inventory.py"
)
COMBINED_CHUNK_PROFILES_SOURCE = "src/_lcm/simulation/chunk_profiles.py"
COMBINED_DIAGNOSTIC_OPERATIONS_SOURCE = "src/_lcm/simulation/diagnostic_operations.py"
COMBINED_FORWARD_PROGRAM_PROFILES_SOURCE = (
    "src/_lcm/simulation/forward_program_profiles.py"
)
COMBINED_POPULATION_OPERATIONS_SOURCE = "src/_lcm/simulation/population_operations.py"
COMBINED_PROGRAM_ARGUMENTS_SOURCE = "src/_lcm/simulation/program_arguments.py"
COMBINED_SOLUTION_COPIES_SOURCE = "src/_lcm/simulation/solution_copies.py"
COMBINED_RESULT_SNAPSHOT_SOURCE = "src/_lcm/solution/result_snapshot.py"
COMBINED_VALIDATE_V_SOURCE = "src/_lcm/solution/validate_V.py"
COMBINED_LOGGING_SOURCE = "src/_lcm/utils/logging.py"
COMBINED_AUTHORITY_SOURCE = "src/lcm/_solver_api/authority.py"
COMBINED_ENTRIES_SOURCE = "src/lcm/_solver_api/entries.py"
COMBINED_STORES_SOURCE = "src/lcm/_solver_api/stores.py"

EAGER_CORE_SOURCE = "src/_lcm/execution/eager_core.py"
RUNTIME_SHARDING_SOURCE = "src/_lcm/execution/runtime_sharding.py"

POLICY_DIAGNOSTICS_SOURCE = "src/_lcm/simulation/policy_diagnostics.py"

SOLVE_PENDING_WORK_SOURCE = "src/_lcm/execution/pending_work.py"

NATIVE_VALUES_SOURCE = "src/_lcm/solution/native_values.py"
NATIVE_ARCHIVE_SOURCE = "src/_lcm/persistence/solution.py"
STRUCTURAL_BLUEPRINTS_SOURCE = "src/_lcm/solution/structural_blueprints.py"

_ACTION_GRID_SOURCES = (ACTION_GRID_SOURCE,)

_UNIFORM_PROCESS_SOURCES = (
    PROCESS_GRID_RESOLUTION_SOURCE,
    UNIFORM_PROCESS_GRID_SOURCE,
    SUPPORT_FINGERPRINT_SOURCE,
    SUPPORT_AUTHORITY_SOURCE,
    SUPPORT_PRECONDITIONS_SOURCE,
    SUPPORT_DIAGNOSTICS_SOURCE,
    SUPPORT_TRANSITION_CHECKS_SOURCE,
)


_CERTIFIED_CORRIDOR_SOURCES = (
    *_ACTION_GRID_SOURCES,
    *_UNIFORM_PROCESS_SOURCES,
    NATIVE_VALUES_SOURCE,
    NATIVE_ARCHIVE_SOURCE,
    STRUCTURAL_BLUEPRINTS_SOURCE,
    SOLVE_PENDING_WORK_SOURCE,
    POLICY_DIAGNOSTICS_SOURCE,
    EAGER_CORE_SOURCE,
    RUNTIME_SHARDING_SOURCE,
    COMBINED_ABSTRACT_PROGRAM_INPUTS_SOURCE,
    COMBINED_ASSEMBLY_SOURCE,
    COMBINED_CHUNK_ADMISSION_SOURCE,
    COMBINED_CHUNK_OFFLOAD_SOURCE,
    COMBINED_CHUNK_OPERATIONS_SOURCE,
    COMBINED_CHUNK_PLANNING_SOURCE,
    COMBINED_CHUNK_PROFILE_INVENTORY_SOURCE,
    COMBINED_CHUNK_PROFILES_SOURCE,
    COMBINED_DIAGNOSTIC_OPERATIONS_SOURCE,
    COMBINED_FORWARD_PROGRAM_PROFILES_SOURCE,
    COMBINED_POPULATION_OPERATIONS_SOURCE,
    COMBINED_PROGRAM_ARGUMENTS_SOURCE,
    COMBINED_SOLUTION_COPIES_SOURCE,
    COMBINED_RESULT_SNAPSHOT_SOURCE,
    COMBINED_VALIDATE_V_SOURCE,
    COMBINED_LOGGING_SOURCE,
    COMBINED_AUTHORITY_SOURCE,
    COMBINED_ENTRIES_SOURCE,
    COMBINED_STORES_SOURCE,
    SIMULATION_POLICY_PROGRAMS_SOURCE,
    PUBLISHED_POLICY_SOURCE,
    SIMULATION_ENTRY_ALLOCATIONS_SOURCE,
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
    CONTINUATION_ARGUMENTS_SOURCE,
    NBEGM_SOURCE,
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
    ACTION_GRID_SOURCE: "76a059f7cc7d7e7d2a99e0ffd626ddbd76b89724d5e24f3bb10ad86501268309",
    SUPPORT_TRANSITION_CHECKS_SOURCE: "fa11ec32880581afec3b48659b47ecd4f967f4a4a198507027d70437d9cefa44",
    SUPPORT_DIAGNOSTICS_SOURCE: "ffecc796fbca00bba02da803b0570494338c28fbb2687bcb3d1558d85afee4b4",
    SUPPORT_PRECONDITIONS_SOURCE: "52c97f5f2f6e9e3b6a7ac7215181874cef52948f613f59c270ef9638a51493ac",
    SUPPORT_AUTHORITY_SOURCE: "8b7506a1059feec48228ba3f31689ff325061db852e69b8a3ee5ed5003498137",
    SUPPORT_FINGERPRINT_SOURCE: "18cc7768552b2c8094a1a64f03e3437edb3f16d053cef1fee8a7e46033f7ba40",
    UNIFORM_PROCESS_GRID_SOURCE: "598d9631aa70d90927a03377a19e5e4c5e665d6e47243fc9c966a310814928ae",
    PROCESS_GRID_RESOLUTION_SOURCE: "c9eb81f9442d7628793d6ad905b2e96e4655e9eb48bf3de32f636541b985269f",
    NATIVE_VALUES_SOURCE: "f3f04d2c90ff3a2506b0fc41b89753433c31eb412f8b8572584ca263e4151f45",
    NATIVE_ARCHIVE_SOURCE: "722756066074796c202d72f7820b29a6f01b3552ba0cf64a9d7d3e649682ce29",
    STRUCTURAL_BLUEPRINTS_SOURCE: "e5385c90d816b17f15b0b02309ddff0f44f5280f09aeebb1f1198ea4bfd67875",
    SOLVE_PENDING_WORK_SOURCE: "5629a30a65f402a6c4a542dc3287b045e6b823c7f005f4b92cf0ffaac0930ad6",
    POLICY_DIAGNOSTICS_SOURCE: "ed41f7f7e0378b0d86e153c53b399bd01350ea24a58a9b80cc88224158a0c0d3",
    EAGER_CORE_SOURCE: "d0049c2a38d3b77abc1c4cff1300f20ca034dff37b77edf7183537471d1cdb90",
    RUNTIME_SHARDING_SOURCE: "13f6b68846532c6631f27db1c84b47f6c866a992e037dfd8e40c791c16f50f90",
    COMBINED_ABSTRACT_PROGRAM_INPUTS_SOURCE: "1ade513e829c0bd8e053b90ab312e53f7b51a5a4da6ae1000079c5684a8017e4",
    COMBINED_ASSEMBLY_SOURCE: "7e1a7967d013827a9fe51862d304e686105ae7886b3539122289ef3866f5f94a",
    COMBINED_CHUNK_ADMISSION_SOURCE: "88c28b335d895add1615132512a56b7e627e435a86619e82854248920a095706",
    COMBINED_CHUNK_OFFLOAD_SOURCE: "3b8f5efa8710f7ea8796ee656c1d13ad20099e141b31ba70b839d079c809371d",
    COMBINED_CHUNK_OPERATIONS_SOURCE: "481b768fc97b134c671c386336a25e9e3e78b631a0a3bc4643d9e74d0ba47e54",
    COMBINED_CHUNK_PLANNING_SOURCE: "7b8ff3abda0b6f57bc6bf3901287bb8cbc6fe6ba741cdbec902d45fb4fabb047",
    COMBINED_CHUNK_PROFILE_INVENTORY_SOURCE: "85070e5f57210e201be248c1f2deae8068dce674112f13efdb0eb94c1dd7f7b4",
    COMBINED_CHUNK_PROFILES_SOURCE: "1c1c2537717e35a7de14c6d26a359490156fc3c433aae74cbc8eebc0e16432ca",
    COMBINED_DIAGNOSTIC_OPERATIONS_SOURCE: "e27d3441be8b4e8bbe62fe31b948dddca47a1462ccd668d60f9479e0746890d4",
    COMBINED_FORWARD_PROGRAM_PROFILES_SOURCE: "d1baa80249776f88a63a257ec48940bb7280a44fa02ad07858955df4a95c4a3a",
    COMBINED_POPULATION_OPERATIONS_SOURCE: "a9fdd59887462abc4e2f4321730bcfb397f69ba5f44f3db08001288773082ab8",
    COMBINED_PROGRAM_ARGUMENTS_SOURCE: "da3ee40547e773ef94c9b0fe2e1c396d443e6894c752baf2d7eb6682744fce78",
    COMBINED_SOLUTION_COPIES_SOURCE: "fbe188d3954a523522624823c769fda4c5831e138d2cb8421fc0f1453946886d",
    COMBINED_RESULT_SNAPSHOT_SOURCE: "87cf247efc21c724f39b327e70abfbb0464a830037ffba5fa996c4572558cbf9",
    COMBINED_VALIDATE_V_SOURCE: "aabd25bb55ea88a370811134cde26677157c1cbe79441dfa7cb3557f72d59a32",
    COMBINED_LOGGING_SOURCE: "45973ef4d846e5202a2b7d370cce323bf56613824d97762cce1db01d004b5302",
    COMBINED_AUTHORITY_SOURCE: "9bac2b806c1a9440d3ef4ca7701ca820fa2c39091cde1ed607802b7c3a1e1f04",
    COMBINED_ENTRIES_SOURCE: "298ba9790d6fab60fd88729b476c75206319dba64f246a0979c7a58444f99bfb",
    COMBINED_STORES_SOURCE: "f40c0fdb20ebf37ff924fb970525084961ebf25e63ebfa142fff5082f2648a0d",
    SIMULATION_POLICY_PROGRAMS_SOURCE: "40eb5f58dcbe5430cbd65ccaa447e8b5f8d4d5e5f0f7d9ad1cc76f2c8eb79746",
    PUBLISHED_POLICY_SOURCE: "a4a723b0428567c32d715f32a0ae85b15ec0eb8107c098579f7c7a5db98be245",
    SIMULATION_ENTRY_ALLOCATIONS_SOURCE: "4e50d79fe971c9d5f0e59fc7b6942868cb09a1223bcde15754019d72efa2d979",
    NBEGM_SOURCE: "50ce2b08e2738f60feb0920f9d30f9efc8f6682729389de5434a927b60e009a0",
    CONTINUATION_ARGUMENTS_SOURCE: "48495f36be398f0d5eb1d1f9e161fb3a71d877dc164da416ada09aaeceda5f51",
    SIMULATION_TASTE_STREAM_SOURCE: "e1c4b55877a5bbc3a3bc74447c7fcdae6c6f8b2a88ce5f7d949bf082032226a9",
    SIMULATION_MEMBERSHIP_SOURCE: "c0c92de4e55be3e7b814a67761caa75e8affe835e1887359d432341f1d85a18d",
    COMPILER_INPUTS_SOURCE: "c28ac1ab4acab2166120867158dec5eb866c082cdd05f7cb7877f2dc3c52ef8a",
    SIMULATION_OPERANDS_SOURCE: "bc3329286e6f22b50a7ecbc4b19656d552b6b92818d3a707b2abc296e4f9839c",
    SIMULATION_UNIT_SOURCE: "bcd41f0970bb0bf3f54507a9b198e26373399da27155a3f4cd39d09ab6395587",
    SIMULATION_HOST_SOURCE: "9ea4554a83f9e324109227730cd39a5f292179032484dbdc5b50aa8ce7b31732",
    SIMULATION_MEMORY_SOURCE: "bddb622e9babeec27da2b363d37ead82c9ad186a7e093c0899840522059a2a5c",
    SIMULATION_PERIOD_INPUTS_SOURCE: "dcdc57a2d7374e2931f71add777e511360aebf44187f2c7becf26be44d1923fe",
    SIMULATION_REPLAY_INPUTS_SOURCE: "53aab5938b31d4c11bb735bf50d0c9b08857564505b607b845f2c8db542c3ccc",
    SIMULATION_VALUE_READS_SOURCE: "4ed63570347beab876c3ff87854dae6d84e9e9f2aa8755a261d65897ea0eba32",
    SIMULATION_VALUE_PLACEMENT_SOURCE: "afd58b343d0385934c3add5552d7926dc3f9ba2700e5a40b26a65a62a75b1e68",
    SIMULATION_CHUNK_INPUTS_SOURCE: "33449e9833e30dbe50f455603916489145b5ec9484639acf0609e8927bca42ed",
    SIMULATION_ENTRY_INPUTS_SOURCE: "de9b448a8e56e196d515e9e9ce958eb688173020137173a4f5c5ac97311a0c4e",
    SIMULATION_RESIDENCY_SOURCE: "922eb2b78189f0c9714b24dd865cb1b2c7eeff3ec5b861bb257efb9f079f78dc",
    SIMULATION_GATED_ROUTING_SOURCE: "3a1685c78c9d1dd345e4242bc008f2858578725232b85610d12fa08535f82274",
    VALUE_TOPOLOGY_SOURCE: "5809eef9ae77ef90075d57674db13a7ea89fb9c55bf4005daffee3b83991019d",
    RETAINED_BUFFERS_SOURCE: "139389ee4e5160c0610b087001abeed9868dda7a6f076fe57a19c13af182df89",
    SCHEDULER_SOURCE: "b5eb17b3df8fcaf5ab06ee7abf8703ce2b9ff589d596c0a0ed1d462d7c4f3a52",
    LIVENESS_SOURCE: "056bda4d48f26768ed8615925ce8ff89665bb8eed94b59d69f02701a95943dff",
    CONTINUATION_READS_SOURCE: "43eb385d2b2795e81a17ae01b98a795c8296ce396a49ca4d3c3dc68e33cc12cf",
    WORKSPACE_PLANNING_SOURCE: "0213e2fb88838275ff1a06e8e91b1a3f44f74edbf938032133af251f18d9d4f5",
    SIMULATION_PROGRAMS_SOURCE: "849b87c55de2a6a03cf65f8db659bb828b8b7bd82d11b1b387058a29aaa43fcc",
    SIMULATION_PROGRAM_TYPES_SOURCE: "cae8d10ac68b6762b4cd75e5cce862b06d96e2b5556b165fdd53f2bd2b03fa7a",
    SIMULATION_RUNTIME_SOURCE: "57b896b0cf23a5c24743fc5819dc3b60e571e0182d74915833450c96e296df35",
    LOGSUM_SOURCE: "e12061dd4f0f0176324182a2eb875cb6ebe4b97174091c597d46a622df93ff1b",
    ARGMAX_SOURCE: "681a7fe6d5a31497945ade5190da4134abd344d92bf5905345f5547d296c693a",
    COLLECTIVE_SOURCE: "3b7f2efb763c8334e9a70eaab00f7037c2dd990d0b310929d959d8919059c810",
    MAX_Q_SOURCE: "d3a6aa921d6c7f18763804c328f374b036787c4eda4b6acacbd974aa66789c3c",
    PROCESSING_SOURCE: "ee08122a0baa0167601b29fa38ae1cbf1d919f879c86611ad71d756000b0b1c8",
    GRID_SEARCH_SOURCE: "a656fc90438850ea4c61fe02062a118454de966a81dd7ff7be92b8de34958aea",
    CORE_PROGRAM_SOURCE: "5a4fa83b1710399974251168451b4cea637bd76f013e03c26b24d07de5361435",
    OUTPUT_LAYOUT_SOURCE: "06c64fdba2a77e10f1ae8ac6a2338edd090fa7727bcf9fb6647469ef59fcea6e",
    VALUE_TRANSFER_SOURCE: "a1cc299560699825518bef7ee3d904543853b916a450e49b075e23d2367bd967",
    FOOTPRINT_SOURCE: "db2e6d1c0f36f37d1aa73274a2903234b202410c2912f63bf765132d9c425c60",
    INTERNAL_OUTPUTS_SOURCE: "295ef18576013685f0dd70fe6f6ceb42acd777ca30d1daa397ab7a43191bae32",
    ACTION_STREAMING_SOURCE: "d7508b58e975394a7487732c1b6c0453d0edbb06ec1234a4e58bf9159cf8b473",
    ACTION_REDUCTION_SOURCE: "26355ac3edf5da80a44b4f9754b71c29e8edffc68b1a7b7ac253b9bfbf749a0f",
    COLLECTIVE_ACTION_REDUCTION_SOURCE: "40a7abd9b738055f00e347e6ac056fe69bf2501ba41f7ec82b40a18bf627c7e0",
    DISPATCHERS_SOURCE: "3d5731223dc0a93f94c3ebdff9b89be3c2b6db2de82e80e5413de8bebab240d8",
    FUNCTOOLS_SOURCE: "7eb73560d9644cb10a95176e06acae3007d7149ad47a5cc83f9786be82ff9dfa",
    CONTAINERS_SOURCE: "80f08f16c256512daa43b38bf9a6366a01413325607ecb2297aca6f1343f08e6",
    ZERO_SAFE_SOURCE: "99fc6b5647425c11e2c15abc9ff49c84d1b1ff3bace3f8e51e593ff8f98ec4ad",
    LOGSUMEXP_ACTION_REDUCTION_SOURCE: "4799ad9bfbc02ae1e5d5270a18ed81fe682f1004d63ae6cf796ff48ac5699445",
    PROBABILITY_SOURCE: "60ccd04b143c714360388fcce2730804a4ca729ea4db82e4a0100732e155e53f",
    ENGINE_SOURCE: "1a31f2797a8c8f519a5c7ebe0ba5a2fdf8f1dd52b24cd1f6ce33ae1c5fb3ab11",
    STATE_ACTION_SPACE_SOURCE: "bc1d4b798ae1beeef6ce6f655ba61cd3bbf064dbd1910b14aa5cf8ce28954e95",
    SIMULATION_SOURCE: "638b10ada11b4e0be0a8aaf01a26993bbb937bab377761785733159d8178128e",
    SIMULATION_TRANSITIONS_SOURCE: "b5936ecbe353fb7d147ee68d83db45a894a1ab2e951dc10d63ec99e91c677a1b",
    SIMULATION_COMPILE_SOURCE: "2c54bd385d0205897bebd42c6b63d55eb0896a04e208786f4e0028b20e81074d",
    MODEL_SOURCE: "46499ba0610b40a1d60533b00abe639afacb6d94c9dfe1ae60585f2c376e2f19",
    SOLVER_API_SOURCE: "3d581f3b0f6e07a024adc1bc223cf072ec0e49d3d5349d0c01994236bc9a4925",
    BACKWARD_INDUCTION_SOURCE: "cc7867c7bd944df07164aa873539f1e036150785f4e6e92c47a5bf1cd325e7e3",
    PERIOD_REPLAY_SOURCE: "3530017ce52b751a08e90015c74262528670cda663dd7308fa91b40fe5688133",
    INITIAL_CONDITIONS_SOURCE: "b9c69990cf6dd93505b22f4d572f1c6a4f6ce3a96530f4a28ffc742b77c7fc11",
    RESULT_SOURCE: "651b88ad561c48d52b294005686978b132fcb89113af48cb18d3163510e3299f",
    RESULT_DATAFRAME_SOURCE: "987eac0747f4ef7587ad646c52f776236eb0bf36f404608e10324da7ace52a76",
    RESULT_METADATA_SOURCE: "27c7d59bddf81c001ea13d76fc9ee09f1cd1f6045c8ed621b565837c651635c1",
    ADDITIONAL_TARGETS_SOURCE: "f0d197864621cef3c532c1061ac8fd769967bcea446b79b303b584d99a764420",
    SIMULATION_RANDOM_SOURCE: "f755c4a35a8ef395d2990d08e725f999a2deb4aff803df5940edbf3e1c4ccd2e",
    FOLD_ZERO_SAFE_SOURCE: "301fcd3bec1211b60872159fe585e55e9742be62a975dd76b33f6f5cb45971e5",
    SOLUTION_CONTRACT_SOURCE: "7c2919f67281b8227772b54dc01fe32633fd617b875141126c571d2520d48b9e",
    GRIDS_INIT_SOURCE: "3c720bc2240dd1cfd45855ce501f03f8a05abda5de1142b9d00208e517b9ddce",
    GRID_BASE_SOURCE: "045d3d462aa80c6f3444030003547f76864bb1f038b8ccb1bd15a00e29e0c2a7",
    GRID_COORDINATES_SOURCE: "9dd0cf9e3c702023cd3cfa40a3164bf399ea295c57349223b8ab617c7f3533d5",
    DISCRETE_GRID_SOURCE: "c27b095d373fef8e0c486c04ea625e3b11d3e26e950ba68877270aad8c69e722",
    CONTINUOUS_GRID_SOURCE: "a2b2c4f6a956e473193b46d7ba451a9e93f5c40fbb57130ecc2aa067e444c16a",
    PIECEWISE_GRID_SOURCE: "65d571f651695af2b925cda751c9c5ecdedb8dda55cba6dd0d1714f26840ca33",
    PROCESSES_INIT_SOURCE: "fbce6ed4c889cf4c30f5242c142b4dcf2fa9e186b75cd17be1557f827abff00a",
    PROCESS_BASE_SOURCE: "1c7ecfed16c3a696f0a3b337dbc1832c72fed513a4cf85ef21b91c956a4548be",
    PROCESS_IID_SOURCE: "2023a7010fc720877c176bffa70f846e8eae260c649af8bb994c484a52b4e8c1",
    PROCESS_AR1_SOURCE: "ea9235cfde4494f962f015fcbf443328b1802f6b97e9afab807c1bf160f59171",
    VARIABLES_SOURCE: "e7f2ceb78b82df85ee860e8e417138fa4ddfbd30d4125c58f7d2fd39d8d3c0f0",
    PARAMS_REGIME_TEMPLATE_SOURCE: "9914a582da13932976cb1e3922c553de1a550252901a7a2fbda4209e3c5ca892",
    PARAMS_PROCESSING_SOURCE: "ccdc017b9522c83641b31bd9b1595b621789f60a64616925748cbd2aa7a73b19",
    DTYPES_SOURCE: "1d38a54667166599fd692f8c58b37741f355753138d234b650c54384321bcbb2",
    NAMESPACE_SOURCE: "8d24bf94013b056001d150ced0c66c24e8534c1573972beefe63eeeb4ba9333b",
    PANDAS_UTILS_SOURCE: "b37c3df986b52f36dcf580b69235446cc827396f1bd43dfc49a4fb2ed348a93e",
    MODEL_PROCESSING_SOURCE: "93918a8b05575bbffbedde60d12e613a931d694cac5049324a2c196bc677869e",
}

EXPECTED_DIRECT_FLOW_MUTATION_COUNT = 406
EXPECTED_DIRECT_FLOW_MUTATION_NAMES_SHA256 = (
    "5c619c972a01ce46fe1b596952b1264750ae35895e0a4a6798388f173a8e6377"
)
EXPECTED_SUPPLEMENTAL_MUTATION_COUNT = 54
EXPECTED_SUPPLEMENTAL_MUTATION_NAMES_SHA256 = (
    "1d95a163810d9ffecb7bc2f2324065c0a3d6168d9dd01002cd00ff30c5802da5"
)
EXPECTED_UNIFORM_PROCESS_MUTATION_COUNT = 37
EXPECTED_UNIFORM_PROCESS_MUTATION_NAMES_SHA256 = (
    "63206e56f35b0b5d4581354c33f127416231b365384a78f2adee314ca93566b5"
)


# Every reviewed corridor pin stands here once, keyed by the certified source it
# describes: the module transport surface (`None` where no family checks one) and
# the docstring-free AST digest of each pinned callable. Each certificate family
# below selects its subset by name through `_contracts`, `_callable_pins` or
# `_surface_pin`, so one reviewed fact is never written twice, and a stored pin
# that no family selects is a verifier error rather than a silent no-op.
_CORRIDOR_PINS: dict[str, tuple[str | None, dict[str, str]]] = {
    PANDAS_UTILS_SOURCE: (
        "e14743ed97d459df9b1f70cdd4a22f302126566c223ef6241121643a898e1c6b",
        {
            "initial_conditions_from_dataframe": "996cc674b61f66fd3c1517429659eea8a6201779570bba88e2002cd2663b32e8",
            "_role_codes_from_labels": "7427bf2fa16abc7494e981ec61e9b4c5fdd71328cd880f707736cb7c443ea3bd",
            "_write_pandas_array": "fc6fdb8c0b16669a7672c6eac52c951bb92f6180ebf607af7bd15907260c0a4d",
            "convert_series_in_params": "72aacb8559c848685e694368b89578f91b82441daa0e84d22a50708cb4c7b001",
            "_convert_param_value": "3ae9d823e90ad2fc2d2034654811af04dc30f87fdeb6cf66f4bd17731f460b37",
            "array_from_series": "41a78d087318846b7bb9e980a9202346a2bdfcd9ed5eadf54c0ee0f1433a6ea0",
            "_scatter_series": "68981f1c512410fc68eccb42c39ecb3a4b0d4be3a3c4f414329f69b5287f79cf",
        },
    ),
    DTYPES_SOURCE: (
        "7bd5f48b5f03c10ce966764c1a47c51f07ec287894bf7058f5bfdfde7a134382",
        {
            "CanonicalArrayWriter.__call__": "909bf3b8d82f3612f5246d4c2a152acd55505894f856567f0b1bcbaa343a390b",
            "canonical_float_dtype": "ff7daf524547e5f62b1d854c3a2a606c3b5903f3a436eb4e8a087a0fda2ca3fe",
            "safe_to_int_dtype": "49f5760ac936afd767c31f97f9641d04143c81d7064c0fde6012849f22d0f023",
            "safe_to_float_dtype": "da9d8dfc4260639fcd6308babe91146a0b174ddbae228821d2332c8db193f32a",
        },
    ),
    PARAMS_PROCESSING_SOURCE: (
        "cbb1526c988aa3f42e055167aab68330a42c5891477ca02df6efc23b673ee6bb",
        {
            "cast_params_to_canonical_dtypes": "c9a5717bb221f1fd3f44a7584912784aa7237fd08b32a1f4b6404ae6fa0b2494",
            "_cast_shared": "4501d56de0c550130adbfdbe0541e69fcaf397b0f1c8a0631caf55c204c4cc0d",
            "_cast_leaves_to_canonical_dtype": "3c3edf1ef9bf2a7826f2f2550f7c654166e9a13def60491bcd634fead8524adf",
        },
    ),
    SIMULATION_ENTRY_ALLOCATIONS_SOURCE: (
        "d3cb23e30be394eb860a6f6136ba29233dfb04b6865626c298b36f080b0d2b44",
        {
            "SimulationEntryAllocations.snapshot": "3d6f0df5cbf4a49bbcb305a4db03698b5d6190b148c2f9eade3944db0f9d3367",
            "SimulationEntryAllocations.solve_input_roots": "8fa5804e7edd400ecd0d507503a11db3b1d1155a1a58375f598ceba780518953",
            "SimulationEntryAllocations.__call__": "d16c9d254801ebbff8ef1ce18125a3a9fe68073d83672bf8855eabf923da6624",
            "SimulationEntryAllocations.publish": "dfc3551541ba25f75eadcc85908e3c6e4f046ec233327c08b8e29a0598093005",
            "SimulationEntryAllocations.pad": "cfe078114a6c07cfa9ad0ecb0bf0411b8985742aee92a5359d8bae8315d4f0c5",
            "SimulationEntryAllocations.update_solution": "3f36dab7d76a35ac58c318b9e6b61ae0f8b78683f79c731bf7cd0cc3620863e6",
            "SimulationEntryAllocations.close": "b662848316cdc45090e3e37b26aa4f1f6a3bbd7afbc3fddde351f8bb8c20ef08",
            "_pad_initial_leaf": "b5ce0e61085c360e97280d593948a3e8e775e10424620a905c038f34142dd7a0",
            "SimulationEntryAllocations.place_solve_parameters": "6b713a8387f4113a0a187ac268a15d25f48e54a625b70ebe68b0cb6167f75a65",
            "SimulationEntryAllocations.copy_solution_leaf": "e4a19117088da19af2513976ccb2673f23fb21799ef5893dd6559ae091628de0",
            "SimulationEntryAllocations.release_foreign_copies": "096914c39bdc6f0f1668cbbad08d7d4b389d79448df16888ef471573b827e06f",
            "SimulationEntryAllocations.__post_init__": "ffa19e8bc72ec7fc9d0382c3cc4f6c7a501059dec278e6e3b7a1c5cc1fb27e18",
            "_EntryFootprint.__call__": "6a64901028e6ed4c92615e2b97beec064da8ad89c81003a9b66e46c2358c0a4e",
        },
    ),
    INITIAL_CONDITIONS_SOURCE: (
        "2752fa6133441734d31fad594b4904ff2f51e98bfde1c6a0a8aa92d3902966f5",
        {
            "validate_simulation_inputs": "1fdf79e49b90fb318a322e8cba358de1b71d6a612e661e089f2f6a26de66c525",
            "_preflight_memory": "55dcaad31fc350c161a1bfde013381ebe2e9948df1a55cee02044cf51a45c972",
            "_discrete_initial_specs": "6e837939569bad3b27093ab4a14ce38ad16756f86910f37127432d07291195a6",
            "_pack_initial_summary": "e674412100af558aae66243e25060f3ce8538ffd4ad138c7e0effd58e1e3357e",
            "_read_initial_cohorts": "a720c185a1ac905d76fb08dcdba34b7a964b9bf8cf461ad31f7f4e29c86ab8e2",
            "validate_initial_conditions": "91ad22d98c8777318b68d05eb4858f5f606a61bd8bcc61e639d2048e9533b147",
            "_collect_feasibility_errors": "bdc760357ec435abd0a5bacadc9890087789aedf351e1948f81080f66e7fc602",
            "_age_specialized_feasibility_message": "893004374a072675df58e2a22ef9905c784127e82283bb4a0ebac2d18aca0ff0",
            "_check_regime_feasibility": "dd5f6449c0e80fee60a9f1701c1d2908c1f88fb7a257efb9e035c921b3dcc0cd",
            "_regime_feasibility_mask": "376939e5868277767e4bee361d06c8fac5324b1a0a4552281042e24e40d410dc",
            "_run_profiled_feasibility": "4efae4cf1e6ba63418c8347c61843943079eadef62bb3b2f4cf49c9ad5961d98",
            "_batched_feasibility_check": "4f2637ac22488e5202f6bbc655f093753c5f296bd13629148c0ff937ab7d1177",
            "_evaluate_constant_feasibility": "3de32aeab79d2a95b7831fcafc75bb8da9478a8090f0859d73c2e43375f3f9c0",
            "_admits_any_action": "7b79e5bde0a32eb36456597caf5b61e1e04736099b78e8c9d2a89382747c4678",
            "_per_constraint_feasibility": "9a40c5071201df771c58b17e05c7eb088a22e94a8e5a06fef835a854b62b4b4f",
            "_format_infeasibility_message": "f8bbf735e2c797e53e1be248998b0349d82495354d4370412d6b98f991e032b2",
            "_gather_feasibility_inputs": "557c76957b3e216d9a95166daee80fefdd0af163633920b30b1d7215596d8515",
            "_subject_feasibility_flag": "58bfc237bc5b8a51897de10ffee1eb33f7a2cb6504c094ed48544946185bd404",
            "_constant_feasibility_flag": "f7b994fa04576c7157f8bab30ef0e17126c181ba874cbe493c1e19e6be056f17",
            "canonicalize_initial_conditions": "fa443022291d93758ac6127e99430d07f96cd4b91b6cbd0fea797467b702629e",
            "_CarrierWriter.__call__": "697bae0eaa97d3806c3079466e8c3de66ce08eec331d66402da3fa02958a573e",
            "_build_admitted_initial_states": "ca3e5c754d9c86385c7302441e2199c338239d89edde06c5620874793afc68ec",
            "_cast_carrier": "5a08758007268730f195c44800937e9274ee44167d6ca2ae6ea00dd75ea4800b",
            "_fill_carrier": "0045c72197d9e16fd510a8530d9c0866327182129a2383e902a9619030be4844",
            "_initial_own_stakeholder": "ed1ce36c6c2b26aa9f4787bd8133a1646138d2ad8e01a85f4809140a9e729278",
            "build_initial_states": "6bc9d8948f600be8b4099ad779be639d6a30c8a66e4612562cc700d9050144c2",
            "trim_pad_from_raw_results": "16dc20f6d29f59b624dfcc9e7575238dca7d2fe7ba5dbdde75046189856a1778",
            "_build_flat_action_grid": "65b4591dd17c07ed9e30c5ddf8f89ead9a2899c2af663b9a1d1286fb9e990e0a",
        },
    ),
    SIMULATION_TASTE_STREAM_SOURCE: (
        "baaefd0a6d556fb0cb5f8e6ccbd7fbc84e8adc8fdd138f6ff03e4578ff9d7791",
        {
            "create_taste_shock_key": "12900cfb3cc0a28f69b58dfcf4c590448d037a0ce5ff96344aaf46e42cabe3e6",
            "prepare_decision_taste_keys": "ef085a5cfac60dce08367fc50b280500313659e270771b1e94367b73e3ddf717",
            "generate_taste_shock_keys": "66f0dddec645c123e66957e33ea5da7f932c83c23822dd9a49c308034339a038",
            "advance_simulation_taste_key": "49eecd1a5724049acc7108921c86640236140477924fbf3b561156c7a449381b",
            "build_taste_stream_addresses": "ded7f3a707af20d492dcf9c396d52c9a8f2c9e9341814145b00b7bb829eeb249",
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
        "ada3a0a771debf6c7d40f4032b7d2e2a8fc9c6f3d99367544064b7917acada09",
        {
            "ArtifactFootprint.__post_init__": "508bd37bbb30d5e0511ddb2de9d98f6c1c2318ab928635b73b413c942d72a4a7",
            "ScheduledUnit.__post_init__": "2c144891456c3626aaf1b62591a35e6c004e5d182b1ceb12dab4ce10d0b4d7d5",
            "ResidentInventory.__post_init__": "75124f4adb54bfbfc0347d29218b4a83835ac800fe30a53f4008ab55f13df35d",
            "ResidentInventory.resident_bytes": "f3aeaf93e4e584e9cbe96b3cce8f6e0b49daf95e202bbba0dce89e7ae0a98aa7",
            "concrete_device_bytes": "19443725ff9af44c1a93207c48918ddc0e276bf42256741e11440dce3f5e2393",
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
        "dd7e8bf5359bb2e2371a538587dc7392bcde9390bfa0618917abb5f6f3f9b94d",
        {
            "SubjectArgumentNames.subject_arg_names": "46cc9cd2b5af3c1a622819cf136f77ca33152cb4dbe6a00b2a111d50848d1f4e",
            "place_simulation_arguments": "b1c02e4c30fb1a7694895ef3f8fc80170341f0924f65ba2c3f42d6dc3bb10057",
            "_require_operand_headroom": "a54b9b75270e80050fbb89998fc9325d19475fcc858d1b3796e7796027dc7853",
            "_required_operand_bytes": "267a7c776406555350af13b2df2baaa1a0aa9c0394ffd893edc35f610e822bfe",
            "_operand_leaves": "062b54c5cd607048d20bf14e589f655a75330b10f5403d98eedeee27b28ea489",
            "_paths_below": "b12d8a06f78558f8c618e6c5e3209bb42e0011190230e7a68e660589ad961d52",
            "_place_operand_tree": "66f30518d9f26e1f76b0ed2f896c83f59300a06c639ed8c46815cc16ac507b7b",
            "_place_operand_leaf": "f10e3ec218dc74b869f3783e72fe338be6bfbdb12940f1a832294ab3dc237c16",
            "subject_operand_sharding": "7f408b68a50569861eac3e980e9426df3140e27fa65791e98a1cd4e8385de7bb",
        },
    ),
    SIMULATION_UNIT_SOURCE: (
        "9e1a76b8a063be9f4908564de0790a1d96030563465b5542cad2d16bbdc9bf1d",
        {
            "SimulationUnitExecutor._live": "efff4a14869f95ae6371a201bf42c1bfb3ee5b4b682af149d4979a02655714f4",
            "SimulationUnitExecutor.dispatch": "a9054123390d9b4d6a7b1f33238a3d4e28fac66480005bbfd0c991a07ac43fb8",
            "SimulationUnitExecutor.close": "2baec2cefe68783c238e3f553ed3471b6520bb66f40aa7648a9961c3f760faca",
        },
    ),
    SIMULATION_HOST_SOURCE: (
        "7c03e9fd236ddb12b280c510c15956c96ca516d9b8d9f7953129c1227c9ace49",
        {
            "ProfiledSimulationOperations.dispatch": "20d68817ed760c316ad119ea002808c35581fe7b27553abf6f3efc37c469e178",
            "ProfiledSimulationOperations.compile_candidate": "fa16641b0058003789569be59cf753f15129b9ed0e3f906896b7fe593f64e4f6",
            "ProfiledSimulationOperations.lower_abstract": "81d21584d0d9278530d57827ab32a5c93c5f20f27632145ccf943802e950fd0d",
            "ProfiledSimulationOperations._publish": "51d44b3974028c72140f3d2ca7ebab375c93c51707e296d1bb648a7a817d97bb",
            "_abstract_operation": "081173d551705f14a7d3153138bc79663512bf8d69d8cd7d90524f50951e5b92",
            "_lower_operation": "9193768da59d7ae5904e4d46379de264c09b971f80d20dde7fd3939139117bd0",
            "_OperationCompiler.__call__": "4f2d9bccfcd9524fef520a64eef658fe2f88484ff7443700e0b987f62f0e9913",
            "_abstract_operand": "da4ed98dd978aa74b9e1e8886ab15e60f875198ad048d68ff23d143d370af33c",
            "_static_identity": "23489d8dc1669d265b3206adce1b77cf1cd2d2162444a2431ec141f3b0ce317a",
            "_operation_memory": "0754b599ab28c5933dc5e22c38d09eec65b32d22dee37c1e8b8b314027abdce5",
            "_ProfiledOperation.peak_bytes": "39cfc33459a34e6157f26e7de46841b13d60fb4a866854dbf9c7c89bf94f31d7",
            "_ProfiledOperation.reservation_bytes": "ad8fe8e77b995ab9398031a06d9492d3d9080b408f68124ea048f81449bb73e5",
            "ProfiledSimulationOperations.prepare_abstract": "ead5f0fef95021a23323298cdd070a6cc34849ed40a5e74e9a1b6694c3b5f49d",
            "_abstract_operation_tree": "c62b99152be54440a70dd2930d5758fc81a2b30b3fed9e65b90ab9bc6b0dd193",
            "_operation_key": "40fcb9b79828e12ba9f99c85012630da3d666f4ecf0ac53ac71881bd981aeb7b",
            "_validated_static_arguments": "378ee6a26654f8d03d3541b1b870a975b17d4650d5cede0b8fed6be145a4ae62",
            "_validated_operation_function": "c46194846f437f82ce69f5134881077482f056bc1c0a9cf98db61d74abf809e1",
        },
    ),
    SIMULATION_MEMORY_SOURCE: (
        "3749ded8a7805e0dd68a4c20a261330fc0c8397f3227e62301c78eb7b48e8a49",
        {
            "SimulationMemory.__setattr__": "f1f1f1e5c28e4d0099b7ed4b0c8fad853cd447efb65ea98af0268f8a5e16c5ff",
            "SimulationMemory._period_snapshot": "5c2d15995d981f85cea04e2d0c6b608dbd33916c604115b122a17efd69a7b774",
            "SimulationMemory.snapshot": "708ef98b6e55c80d81e4c7c8181325eda8b1d0cc8bc15ef5d804e7660e6a22c4",
            "SimulationMemory.budget_snapshot": "166c1642b6b2392cb0c6d4ae8c5fd8d5ce7d2d67116527193aacc89a469b8bd6",
            "SimulationMemory.set_chunk_inputs": "fe3e0c3a0903270b5b5414fd2b50585480a9bde3c4a09e1f0a14e90af29c6e28",
            "SimulationMemory.publish": "a1da3681994370b876c3f4386f2ae60d8c8886648281a7216b6941b0351029ad",
            "SimulationMemory.replace_outputs": "aa11c3311c7f52fdc4251a3631ff6721f77579300162b1b2385113d83c3e7399",
            "SimulationMemory.set_derived": "37d12b85b0715cc5e304cd4c1c883368232fff1b233c0c978c0153a8d9d5e37b",
            "SimulationMemory.hold": "a11353f3ca26fa3c13594e4a1e0c7e530b3c377dccb67746d708bf2f6fa3de53",
            "SimulationMemory.before_transfer": "8f90ea91ccd52885cef10e97c47c0789e4f7be21b0c69b846f8c82ba3c97b764",
            "SimulationMemory.check_resident": "f5eafe87346b680fc08d2617bb10f395585346ec569bb887a8eac725485daea5",
            "SimulationMemory.run": "5f032e3774d15890471a774391ffb79b18bda53f8e5d7e95b7076a3f3a69db60",
            "SimulationMemory.close_unit": "ad93e305391d4eea94af93d4561ff2ff0a699c691f24084f5f1bcb63ae589838",
            "run_simulation_operation": "e7113ea2849245cd18779eb8f15e3a83a6d59389b6baec8be5254cd5d26e53d8",
            "SimulationMemory.__post_init__": "7379e2d055a0060e503abf016cbd18fe13466a411701a471a263ec96ddf0d435",
        },
    ),
    SIMULATION_PERIOD_INPUTS_SOURCE: (
        "7319c6ab3c35153d1dabf2c0fb5be6a92bb6c855008a2432350c5ef165c40a14",
        {
            "decision_reads": "9d312519f855a144a449e827178f59ce62d92d5481f0e0c46f6c10d204d0c013",
            "unit_value_reads": "4b5e6bb1abe89deded6a27a55bd613d2999f0aee04b9e5816e293894b538abed",
            "gate_reads": "9e57467e4a7147bb1de88c0010c73960fd3b966ac76b3e5e575bde8791edd261",
            "acquire_gate_inputs": "a49be6fadbf76e8f8b2fe9b289049711ec3a6fe48851caa0fe31467ecc249abc",
            "acquire_decision_inputs": "e670be0f58abde8d579192c237babab554f418acdfd62d1123b1b5478366b2a8",
        },
    ),
    SIMULATION_REPLAY_INPUTS_SOURCE: (
        "9ff77521134b95708400b689c4fa5de7d8e8961ccb0c40d189e38f19773624f9",
        {
            "replay_payload_reads": "b4ae2cf7345b78f67d296fb837f59d0cd64a0b3251ff9717737d6c0f850db255",
            "place_replay_payload": "9817b31b7ae7601eb39a184179d4705a9547f4e0d4c64938dd9bc9f48185fcfa",
            "_payload_read": "fbd97a0f64f94d805ecaa71bd439035a57159aa476947b77a6d903ec6bce38ad",
            "_consumer_step": "74571a0fa030155534ad7f9407049bd2589a5d67dac306d7fc870456de1dfb13",
            "PreparedReplayReader.reads": "3a9de61be4467a66b9cd87633adaa6be14787697bcb60816796972ad066d0dc5",
            "PreparedReplayReader.build": "81f1567656c679ca538b5fca9230befbd96bd189db9f2a07d4cf12823353ea6b",
        },
    ),
    SIMULATION_VALUE_READS_SOURCE: (
        "d3536779348279cc4ec16c7ba70f66a4e64ab53c9778833874f904997d86f198",
        {
            "BeforeValueTransfer.__call__": "82a3f70c8e1a601fc8a5e52ec2b029488fee92b0f4f6a103c21513943f3688aa",
            "PeriodSimulationReads.__init__": "6e5fcd55510c97460912d23c87929113019c4c6555b845f22572a160535821ba",
            "PeriodSimulationReads.read": "8dca34e2aab9b6064d8db129c16ee36abf7f730b14868a7e62e11dd017bc8088",
            "PeriodSimulationReads.commit": "4676b7c7ec1e4e01f1495b48dbe495ce17a15fadb6ea209eb8a955408633d39a",
            "PeriodSimulationReads.live_values": "14324293d4a4ed4e3853ee801aa47bdba89b38fbb91559daf7aeaac2779a5c32",
            "PeriodSimulationReads.finish": "77a88f1d96ad536556eeb486540dbf17ebef640ab772d26683f74712a7a9a6b9",
            "PeriodSimulationReads._read_host": "a352dd98f9cdfabbac6852b32932e6ea9a98531c278d126ae09fa23035680717",
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
        "ba949a66e18f92beb936712bcd2485a3f6fb279689e5a08e40832a4d1a3c5a34",
        {
            "prepare_simulation_chunk_inputs": "2c0b380fd0b6e7dfd8f2e09902272b691f317b7b983f3af64ea0a899cff464e2",
            "SimulationCallInputs.array_roots": "359b12d50d6ef1371468e3544f44d2205c85d47248bb0802b2a91695d2c5395f",
            "prepare_simulation_call_inputs": "e474358cb7cd275d3a2de86d4d04f98bb550699bef7a23b450fa8a48cefa3d14",
        },
    ),
    SIMULATION_ENTRY_INPUTS_SOURCE: (
        "19298fcadc9496eb0783d317d0906a5395a72c1c44ed7b97d118c394543f2317",
        {
            "SimulationEntryInputs.footprint": "1953575a6f4762e5d24f20332868a5dfbffd884d41ac4334b8455317dfb55355",
            "capture_simulation_entry_inputs": "c31f897a2c610499ad3cbd7a860d06ed930d88c0f59b6589c17609459448eead",
            "_caller_arrays": "ee4e16b5f1f25036f1c23004d847c17eca19ca6d88e0b353cfebd736dc883b9e",
        },
    ),
    SIMULATION_RESIDENCY_SOURCE: (
        "7ccd2ba0f7c047da4de89e36306c46eee350ddee9684ba49694e88d0d62b6bfe",
        {
            "OwnerLedger.bind": "afdc24c3218a517f12d74f84fccfa7e5c7055bbb0b3f0ee659f375bab0119949",
            "OwnerLedger.measure": "689177e5451cc06fe4503e63e29951ed95df9ac37305830f92e9467a518c9f9f",
            "OwnerLedger.release": "77cebd7c193e2c14e76d9aaa981186b831626477111f14c27a999477296ac280",
            "OwnerLedger.release_prefix": "fe266006cae3b79cb1031bdabf42ed650d01ecb15163217c319a9c8dc3e9d8bf",
            "OwnerLedger.clear": "50605dc06e8805dc6e41bee2cc6db6e1959af9255cc79d63a88d4d2853f0e481",
            "OwnerLedger.bump": "8997d8a8f18dcd560f6ffd669ee0c2611ae836262925d4644aa2836e48617323",
            "OwnerLedger.union": "d77839f108e8fdcd189ead2af075daaae002dfb5ab61136484cde3c2f9bc9e28",
            "OwnerLedger._extend": "da833202cf9271e36a3bb69e39ffa5b2bd284854b9ad586d3916e830cf01cc0d",
            "OwnerLedger._invalidate": "bccac5f4bf4af8cca6bb493816b1c6bbb126121848f9895f8dc4a3ede25aa087",
            "DeviceBufferFootprint.__post_init__": "6b91013dac5e169e65e44c92701ce26bbc790d4e51e49b73b96725b8c02b9cef",
            "measure_buffer_footprint": "99ba8ec6f6b1496b580ecf82d4d979f76441aabedcb1872af22f200678262c61",
            "union_buffer_footprints": "5928dea9a80ee77082d80a749b8ef86fdf1e1d8fddfb1441ac9112f1352a1296",
            "resolve_budget_devices": "cb3ce34525277ff3e667132fdf3f0a0504fdff44dfd554cf2ef21c2f71a24d30",
            "resident_bytes_by_device": "7f98f5a753acace9389784c2200b2b94375c26cb7523092c10f09e890dee2ffa",
            "require_transfer_headroom": "74b69061e32088c8ab67c18fe47aa512d4aca140f27bfb8d16e5085216d46cef",
            "_merge_spans": "3cdde57f56837f60963871f3ae1f04ca2f92f45a7ddc4d1c35bd3b52f2ead33f",
            "_uncovered_bytes": "ec6738bad93d29c3c742d696b5cabce48748755ac79ce71fcde583a94442df52",
        },
    ),
    SIMULATION_GATED_ROUTING_SOURCE: (
        "c6864502602165957398d440d67aade4c99327364a9e8ad613f89d3079663246",
        {
            "simulation_gate_fold": "d9b1ba36710d5b9a2d6daa6ba87652cf2c32c9178c998285d645cca055a323d6",
            "simulation_gate_route": "6a261820a1fa4e08c2ef5bb2504b3697722c8f339323a151f6b58498e00f7337",
            "gated_route_candidates": "c5078c81064c73e2e57e3fa6cb1c6d9a72250493448e72977559ef61185c7724",
            "simulation_gate_route_delta": "fb0c0db6f7cc5514c7404b7e81f4b06ed1f0af60cfa4ad1169f2f0cda4ccc397",
            "commit_gated_route_delta": "efa0f6567db40ee808d0c36bf74250772935b1cccbf85290ed0c2b9a64730d0d",
            "substitute_gated_edge_continuations": "aecefe81ca82491c9be87871ed60803fe2b0b253215570d208845b8b6db9bf4b",
            "route_gated_edges": "d4840f36532453c356e1d9e4c2fda085944e9f9400a11dd6a0ac8eb0e0b44430",
            "_per_row_leg_outcomes": "adfd58415412e5f4ec1e734d52514531a9f21e758fd65ef58a0f4f40ca7dd668",
            "bind_provenance_params": "4a2c501c3502e8cf64098c0018abbfac2075af8e47683593d2bbf2cabd2f8a5c",
            "_call_vmapped_with_accepted_kwargs": "d5792812cbc1b03b3311af6d079c3378c5a52b83ff55704abe349664eab625b8",
            "split_population_call_args": "ae5ddfdf72bb5435930ee267d3e3fc446cf2f6bfb5d23e69b8707f8c0fec457c",
            "install_population_call": "90706c6aa460ada4aec4fa145a878ace15a15c6a6cec2a0c20e7bef75e2f50fc",
            "_accepted_arg_names": "ec042dd18f4ad40c9c0a3cdc1621aff72ada09674aaf9c61e1055787279dc2f9",
            "_role_code": "439c955f56094cfce8195a6516abbed95c0a5990b6ac8cc8e8b85ca177b7e6bd",
            "population_call": "6d1e070df8f8b68ad71178c9fd22fbee9997083c760a69e63edabf0f3f72e0c4",
            "_map_subject_tiles": "0ae12ef38d34badd5aef9bdd789074c84c678f3856ae321e49111372df7dc2c4",
            "_call_one_subject_with_shared": "6750dfef1ae75365da912f3ab94cb3fd8106ed54f9cf029376e5c2c1520b1bfa",
            "_call_one_subject": "8935deaede7a2422f4cf34b3f23545600591fb59772254c99936b905b00d9b3d",
        },
    ),
    VALUE_TOPOLOGY_SOURCE: (
        "f52f525cc74235088527229c0db000aefcc467cbb88cdba3743a4d40d57807e1",
        {
            "expected_V_rank": "3fb0dac3bed9027261436cff2316bd122cff31e99dd1b825977272d09e6212f7",
            "placed_V_sharding": "2232aa6bbc1e01337e19bd39637a254e34d14e9e6296407c2aceb6b2467333c2",
            "_get_regime_V_shapes_and_shardings": "c712d242ed26f7d444d84b9ae61add8b59037222ebe6c533b387a86fad4c5b4b",
            "_build_zero_V_arr": "83c861c7149e4ee341e409c56e8c08fe2a8a01d859bb8d6e80b7ff62e04a632f",
        },
    ),
    RETAINED_BUFFERS_SOURCE: (
        "ce79b7df172a647f8aba2b355f7b030464a1c2f36242f72983b745b46130ee95",
        {
            "retained_solution_buffers": "75b31cd7df576899d18b203e4963dd06acd47b900bf1b6c13c678c62e9d12d58",
            "_RetainedBuffers.collect": "fed7a0cc8a4221d66b9951e8f315bea7416b134af0eeed197e076facace39cb3",
            "_RetainedBuffers.collect_lazy": "e0390ecdd855167c8261898538e6d28ae62b33ce7170c9ec1c96d2d824edb634",
            "_RetainedBuffers.collect_authority": "402406b70f1437e1d6427b96af3bea127d3ec5906a9a97eb9b19de07c8c0b081",
            "_RetainedBuffers.collect_reader": "ab65de29bcfc68b9a694f47a926115124c4ee19dfd455b1c2f4a9b58317b7d99",
            "_unsupported": "5eb49f472abdea16b95d9f0c9a4f6e56297f0cf65466d18cd87cb4ff530dd21b",
        },
    ),
    SCHEDULER_SOURCE: (
        "e172fb1d453ab8fdaa64d3925534269a005e5707e62d2628e9bffe87effd152c",
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
            "release_closed_artifacts": "76bd3cb14ddc9fc90add485422a671db7096d06061fd34c2b48bc2158260995f",
            "_one_delete_per_shared_buffer": "c6824683620084bbb6acbd667b2657a3ba1afdcf425268d7f1d384ee4b308f71",
            "plan_period_waves": "0f2995c211c6472a1f378a69861a15181f6f2e0856db85191883176788017999",
            "replace_leaf_by_identity": "134c02bb276cd5f85e76bbe297a51af5926453810f64751ee68a30e5bc08b73e",
            "PeriodTransferCache.__init__": "5d29ffb98c2dff953aa020f6562a7116259bb53372cfbeeba00c25c7ca71c46d",
            "PeriodTransferCache.get": "2b12aac3bfbcef4e7d2f0da22f5b8ecd429656fa07f72ceceb51d76cc1fb9956",
            "PeriodTransferCache.put": "292575c03a3509d986086e6428e908b6303e852eb4ada1a3f095cceb226a83d8",
            "PeriodTransferCache.commit_consumer": "83ffdec9aea1efd675cad1d45d317a31e72775d52169df9e11bcb157b5d9b0ef",
            "PeriodTransferCache.__len__": "9fb60f4b369c44a05b578b7d992aab8b3f956ffa5c5b465715beceba5fe2728d",
            "_add_declaring_array": "8cc69ee60ad06411506f95155fbef57acea86a11ca7246182e25c62336ade59c",
            "_keep_live_declaring_arrays": "120348a883a1615abc6106151a8da9b6212cc82b0c5574806f4e08ae609f3a1e",
        },
    ),
    LIVENESS_SOURCE: (
        "05348749f2f308e9bb4e2b839a6d8515340aecb5f8f0132f844a4429088c4e52",
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
            "_snapshot_unique_hashable": "9d52fa6aba020683c311fe8949f8c5cdf145d7fd590c2c3fced372315f9a2ef1",
            "_require_hashable": "764db7a4ef81df40fba0cb520cdacf07e1076711394283b218435dc9cecc0463",
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
        "36ec6809e557b1bbaa12aef1a168d1bb4ee80110bdc220c1cd4ecccffc110736",
        {
            "WorkspacePlan.__post_init__": "d5aac605f11a499bf65418187c83f288a9e6302e341aac6334a697ebf514982e",
            "workspace_width_candidates": "6aa7f1ab2054fabf4f9232447e62b0ee672ef45210639a2be5465873e9326603",
            "plan_workspace": "510107572ee1896d4d68a8861e9812ed048f25e8727bd5ddf7cc5c61e72bbb19",
            "plan_axis_free_workspace": "b5cca5b8ba998f4e4db88e282856e4a360f8f1728e830c24ea3791d8da49d39c",
            "_resident_exhausts_budget_message": "c4d70c808081a036a2b167eada3441348817e49ea0561a1100305d6a44a015a1",
            "_no_candidate_fits_message": "ee996eb12950a8af3c1b54b384e567077d8c33e1906ce85fcbabc1391dba0d25",
            "_validate_axes": "0df42ff02f11f7f4e5460fe7c518c300621718ae27f95408ad0aa1696d7ccb5e",
            "_validate_axis": "05e956236b961d3872faf3ab84245b3651f169a912890465a02e43784db1cf6d",
            "_validate_coordinates": "5453bf694d34197a0496b8af31085a2c6cf57bd46e866f0981ec932b8dfe9e53",
            "_validate_fixed_widths": "c9f5045001afe5636961826303f1af8535504c4850cc5d27f0171eddd72e9539",
            "_validate_budget": "975e144760477b3a3fe439a47e6b4bbd029d27c33c8301b9a83c8fc146086128",
            "_validate_resident_bytes": "ece252f4071f0b708b42074e53d06b225400297efd225702b1ca9ae20f135c5c",
            "bootstrap_width": "7b74ba3484d72f9525e5e3fb978ad813ca6b684e34e5078a6424342e5977983f",
            "_tiled_bootstrap_cap": "d7e16de916ee80f4fb298ec36e65111eca2e971060759897e5e7d401a5cd3907",
            "bootstrap_widths": "b29de7d4573e8f085e3fe83015c2d3ae7f4a111d7cd7bba588fdaf889be5c05a",
            "_workspace_width_candidates": "fe684819ead8427ad60004be741a697acf76a54b8395ee5c71d28a367262ec53",
            "_candidate_rank": "fb310dfffa04b1180e391907a03e7137950ecb0ff60919efe3e2db626e540e9d",
            "_axis_frontier": "a49f8021fc84bcc1572acba3b8c95276decd31e3dd09eeb99915f7e0add84318",
            "_fixed_width": "ca34e5f5a2c461f15be91a1c7f41ee91de833b37b174ebd3bbdff3b21d4ec8cd",
            "_admissible_width": "e38d2036f99bc8edc1633bd783a6f8a7226fd11634826f853976d9af21f5623c",
            "_smallest_admissible_width": "6f80c9145d0d2b9d2d674b0a3ba8fbcde84f3cacad677d099317f4c077117193",
            "_width_mapping": "237667842db07b3505f6c2d66395e802071b8bee98047aa4794956f793826c27",
            "_memory_for_candidate": "95e7e8b701c9098afb3320d6024503baf0d74b30a9ff6acea9a8459274677734",
            "compiler_peak_bytes": "febd3e7fc76778a4f820fd4e84666459b76ed363df76fbce42395cb911a3161f",
            "_non_negative_bytes": "664e2f4e254d38d8e75fe8e23c01b4602c8f45ee0deccf2b637d1dcd2f49bdc3",
            "_resident_bytes_for_candidate": "365e0864a5ae864c0594a64c5d6877623f6dfc9a50b5c696b50ffe904ff593c3",
            "CompilerMemoryRecord.__post_init__": "ce6b673f2e4078e9fbd5cb66c194074fbce0b9f1b3555eb403d7d15ec0e77ec8",
            "CompilerMemoryRecord.allocation_bytes": "5555d63d219382decf78853463017ba5df55d7e567714110f64226f7a691ee7e",
            "CompilerMemoryRecord.reservation_bytes": "3bab6af91a467633cbcad7d4853eff1a27741d1b7a888c24eff51c89ffb8a0e5",
            "CompilerMemoryReservation.__post_init__": "a9f6a5d3a7afa2a595904bb54fb7f0978eb8d53364294f494ef22d5c58c7ed1e",
            "CompilerMemoryReservation.peak_bytes": "86e8e6c586154b1c40ef2f9938a1a8eb27c6f4911d016eab73d46fedda418bb7",
            "CompilerMemoryReservation.reservation_bytes": "dcaa109c5f2be124f904fbf9f9311650d9443d234f3442b22bcd0347ada332a4",
            "compiler_memory_reservation": "eb5b356c338812bc5f6d5c768c224a1b6ac55ded1f431c5da491c75e18db12d9",
            "_compiler_memory_analysis": "008f829f89cd0492af0af3fc9acbae65fe31ebcbba9a121856bd5ef3e2809bee",
            "_allocation_record": "0a51e3f104316e4440931bdae51fcf39673297812cca1ac157d852625b2121c5",
            "_fail_if_host_allocations": "83cbc5f652a502ae8cdea81ae129fd8a89aedb4955e3ca4f251636b5931363a6",
        },
    ),
    COMBINED_CHUNK_ADMISSION_SOURCE: (
        "22cec0e15611164ae2ceaecd9cc501f6769a10902cdfa38f4e1a8e0e1c32a595",
        {
            "_ChunkProfiler.profile_widths": "f20bd368bc6d1b96869937448eeccad1e6329b1685862aae0ca150573bef0edb",
            "_simulation_chunk_profile_key": "84cc9389ea8e24c4b76d1825f06b4b4ad1271b2ca1810dfe0b3b8875699cd454",
            "_independent_outer_candidates": "823294bb69d6649f9e4b3b9da1df9c070c39fc7e2945db56cd9bcbabac98da73",
            "_independent_anchor_widths": "1c9375085f39b01649366f937b1eb59c5fc120ba611efbe8870a4e0c9d40d078",
            "_plan_independent_chunks": "030a717fb2310c2c85cf9daa0a75ec918dc8e33dab682ba642fcbba27f723015",
            "_profile_independent_candidate": "ce068ef18ed9af1fe995d9d31853fce46414f4f8508fa94c8ada778556213491",
            "prepare_simulation_chunks": "6c7a057b92e35e5bc1f92434e71d93f5a2cf513806ba1af5bdf061cfc7fcecc4",
            "_ChunkProfiler.__call__": "dfdaeb448cdff047ed8bad87b46cbce7649782bd9ae525af5b240a0d836aa2f1",
            "_common_axes": "277ef7d313404ccdd8610cf7f5a4896b38710693206a213f6728e4377cf637ef",
            "PreparedSimulationChunks.require_chunk": "ab595a25771d41f7f5a98f4d95927d79813f70d306d856c2d53483a8da5bd88b",
        },
    ),
    COMBINED_CHUNK_PROFILES_SOURCE: (
        "bf88cc526a58312c40854afaadeb7fb8a57abd62536bd1040cbc7201dfc28f1f",
        {
            "profile_simulation_chunk": "2b42c353d0684dabb038166d436233c506c39c1a50cf7b2562850b4e3ccfd2e4",
            "_period_copy_reservation": "bf995f9f6b2ee949a83d862cac85fb1e0148fd80ac225c0e0c469f6f758b05f2",
            "_policy_read_sources": "a83d15eb8de8a505e80cdce7a9448b11d7a9ab3001e92beb7676b491fb63245e",
            "_retained_read_source": "e0233031b8fb4dcb494575a397f5c2e9ccbfd45a69d0e6f951bb3612323cf82f",
            "_profile_next_subjects": "28e46a009ac3b5002ef895d5f41a2f4caca207f177aa3c98110927cae3f0abca",
            "_profile_population_roles": "99510aeb32383e09d2b2972ef0ac6354dd0259aa494d77cefd9aa43d9fecad8e",
            "_profile_outer_storage": "f140b97925fd77d4b73acef4d6cea9c9cbc2db8cf67fb70d692be6130eed11c3",
            "_record_core": "1ca4c13d2facd739b83c8ee4c9cf265ddefcf982b4d937a6daedf39e2393684f",
            "_profile_initial_carrier": "1ef1aa933990fbe5e52659aed6d1f949d76b44ae0d840c0307c2d8764efce7cb",
            "_profile_keys": "5d56469de097f6ec27b7ac72ebf82561cff40c280c6f6737ea2148d440e4b5ab",
            "_profile_taste": "ce976aa42f4ef30bed63fc9fedcf72652b611f2c1a5262b8085a29da83b7e622",
            "_profile_entry_key": "306ad82a059557d575d50c24ed8c51a0d172ed97b6de476a99945bc8ca10599a",
        },
    ),
    COMBINED_FORWARD_PROGRAM_PROFILES_SOURCE: (
        "412054923d113173a6edb949e8409c8d18abac1d768355e18aca03839c81f9c7",
        {
            "profile_forward_programs": "effdfca96bc6322a037b7f0b4b9d93f97a776a55db9d0de0fcf60d32dddf7a3d",
            "profile_forward_unit": "a469dd2f7c0f55803d23c4328bf59c64d42710c04ac6cfc8fe0651ce563ec32f",
            "_profile_finite_decision": "5ca35d7e729d991a97f4d22e3f1a65b997c0403769ae6b76e5a35e512e5e66b0",
            "_abstract_policy_leaf": "96e133ac576cd9b40afbf10907084378253b43bf9376d39d0faa3743d58b0cd2",
            "AbstractSimulationProfile.__post_init__": "363e5274bc77382335a7ac5d5200ed644324ff361f578889d29dfac63bf3aa31",
            "_profile_program": "a5c79f57eb3820505ff54334e12007bf8cdd85b5343a20aca1cdaf8c0aa07f2e",
            "_prepare_program": "03310a0204cd76f335ab1e2241ee669684b83a4ba3d739acaef515b130e2f58d",
            "_concrete_widths": "d8085e4c2d9d3f24d6c5b34f26239fea9eec9fff79d8e52e23a2d539c7192eab",
            "_stochastic_keys": "dee659c1811c43e460e7e3d77191d61166c7df35b322128400244b3cbe07aa36",
            "_shared_tree": "ed4cecfe3eedeac6b79cfd078e28b839e9e2d3d943980ce932a82516378e9303",
            "_shared_leaf": "bf682e2f0efa2cb4f6fcd8356560d0e45538b59fd741f094e7245ca79b5aff14",
            "_placed_abstract": "7e7f6ef3a1b6d02436a1eebde1869d131f4f11680fff287a44763acc2040723c",
        },
    ),
    COMBINED_PROGRAM_ARGUMENTS_SOURCE: (
        "f10be88662d7a82e66428a5d3af8da024e0464cac414bbff0b1dac4e2368d1b8",
        {
            "policy_prepare_arguments": "8e82a8646d7b7553c1ce0748d6b321a72ad2a6542caff4ec92f5820733bbb8dd",
            "policy_rank_arguments": "451148291aca1118a504f213973ee246b11226789800fbcce38c47f866003730",
            "gate_fold_arguments": "14264886b5ffff2d7ecaab619eb270bfd895acf096cf32cfec5b259d0a74ea91",
            "gate_route_arguments": "f5d91292d00f3a029a7ba682d7482af2704733fd7bf7947885b4473a72f3f195",
            "decision_arguments": "46071286a5f53d60000d90ce098c2932699186c284db7717786a8a5f16d93566",
            "transition_arguments": "eab1b8193f71cd84e911506b827792b73ee6baf7144fead32d0ecdb3f846e8c3",
        },
    ),
    SIMULATION_SOURCE: (
        "3487535ecd1a1fbdfbe4a2501267bdbb34c6368ef13f35c04db894434caf692a",
        {
            "simulate": "63de7af7950c94b856783111458e25a8f334bbe1848654003578fdc73ae22891",
            "_simulate_regime_in_period": "b243490c4ffda012d55ec9021f34d35f725c04e1bc88caa465ca85fc68d3c7e2",
            "_execute_finite_replay": "8fbb9dc117de1e2154984bbc79668966690ed60572129e2d46a376e02a60d1d9",
            "_announce_dropped_outer_candidates": "977a3fc8627f295ad6837b46fcf94672449b532263c674be2209e509d6c088cd",
            "_report_dropped_outer_candidates": "0825c06c02ecfe2ad8270e6d606c67f3063f945befeccf6debb4880194d54dc1",
            "_compute_starting_periods": "f50f6f3c87b21bdf082f3f3d71793c03574dfb93f1aad875d8689b8f773b65a4",
            "_concatenate_chunk_results": "e1217d0e707ef6a8b3d00dcef6abce948d065e125fc2886336bf537d276c1a41",
            "_validate_period_values": "30fe242e786e7e38637d380ecad0aa4d1689bc71c1fa9cbf2ae3c781e011a3c3",
            "_validate_simulated_value": "7df6ad30de49b3e95662d631be0aca54ab9a0e65ba2a97f7f505013548600b5a",
            "_simulate_subject_chunk": "73992a1fb91a25fa59a07d2a6176e6ff4cf8b4730b7fb411ab292c547041a099",
            "_bind_unit_executor": "a787e30bafb80053703e1f4ce0c1d84feb5ac522cd41349665c600dcf9f1eda5",
            "_lookup_values_from_indices": "ebc4a036a447857f061c117b2eb0c9b9e61d5f17a40e90ea14a6e6205233ea9f",
            "_read_external_replay": "392ae18dc5916b0dd6b9539f395002fb97b1d1d985a8b985dc547785cd98ab53",
            "_replay_nnbegm_candidates": "8a0f5086bdeb56bb6622302db85cdc1b03d06ede559d45a95bf6346f31764de8",
            "_prepare_nnbegm_candidate_bank": "d5406bb8fb04f481aeb91ecbce93394cf531239a137df46d28ace9993797b62f",
            "_rank_nnbegm_candidate_bank": "fa90813a7edb9f5aa16be3f809c132f3d2e4f9ac87ce6729993e799a3ac26db5",
            "_initialize_chunk_state": "0597964759cc394aea8126642ecf7a6d0902c080cf9ceff9cd704e805a4c149d",
        },
    ),
    POLICY_DIAGNOSTICS_SOURCE: (
        "6ab53d88f11165b91371920ebecef50bf288c3ab3e4cec258e4d8787d8f84238",
        {
            "dropped_candidate_counts": "b82dee7807b85e869b9962f3b3b4f5fd0ee65a3ad51eeb54c074c814030b2afa",
        },
    ),
    MODEL_SOURCE: (
        "426b9ce35b2dd92125278c123d5eddf93af5c378d9b45ee6028b5014689b3e54",
        {
            "_validate_sharded_state_capability": "76a10f69e8f6acb1150617100119e1f61ad7ed4b759f4849c8ee7536f44f3f4c",
            "_supports_continuous_sharding_vocabulary": "1ce9646ee043fb623720d6367b531c860ab0020969a58ec62682784814843387",
            "_supports_unsharded_continuous_process": "f9458a12d933ec96b852ee69337c296a73cfe3e22dc9065847bc53f9352770c5",
            "Model.__init__": "5d9cd8d0ba154e15d2af7c19865f07464d2cbc4904c5590afd859b8bce231d42",
            "Model.simulate": "9be61cb6d18d858125aaedb91b3b610dc3811608d99ffe4bd1f970da50a3b650",
            "Model._open_entry_allocations": "eef364fb50a07f87d74caf5d7f84b2a3f6ba7e66c3d84b4a25e9bf8d73c4b543",
            "Model._check_solution_result_structure": "3e2f19b7fae40cede786a1debce00175907dd9edf1f59c9ff17ebcf834637736",
            "Model._consume_foreign_solution": "15a780da8bc0f5d8767481fc4f9442fd846d29e6105f72d9b795266256e328ff",
            "Model._resolve_compile_batch_size": "27791b63c37282ce72ab9e537fda65cd20796d2a494a804392862219401dba7e",
            "Model._resolve_solution_result": "35015256cc5d0d8a3fd9cfe66e059b993d9aa8cc5a2c33848dbb7bdca8149dda",
            "Model._snapshot_solution_envelope": "9a4c08dc99912920df4b670212a15313e4e4e79446b8ae6a19a11fe6ba66910a",
            "Model._declared_solution_authority": "606f4c123f2b6593e1e2f212f572a035cdb6b304b185055a08af490bd097a6a9",
            "Model._model_fingerprint": "3b5ede2a24fdb3a9ee1993d3b0ad6a9092e75b82089d341dd7bde349ff458816",
            "Model.solve": "9e634356fb4acee67f4ada3ff2412a9fedc322c5189c40dffca1a809fbbced5c",
            "Model._solve_from_flat_params": "ebdf26e694572b8d196a0ab6dcbe0b7298588ea20d42d6fb3b20ad099489018c",
            "Model._solve_compiled": "a6563c99059a09014fccf14794dc4c5390b757736a392a128c4d840f6bb37cd6",
            "Model._consume_owned_solution": "f646cbb48cea6c33227711df6ec93e2d51fe8fca435803069575c0cb0f01da9d",
            "Model._build_external_replay_readers": "62d5d2c5bd473f779ddb81644162279e0a3ca329105b42636220a862391dc771",
            "Model._runtime_regimes_for_shape": "b85ceab93d6b925942a9d577c69afcb4df55bae3beb6aaf2220e2249d24697f8",
            "_fail_if_invalid_taste_shock_seed": "5a8c4643d73c99c83160024bac7ede90deb9da24141e750103ec48eb76ae0486",
            "Model._process_params": "afd52e3f226b99d33a93068e91276c71827a7f8a1dc6d5f393ee7206d7b358c5",
            "_simulation_programs": "02d69b94d7005c2f01cb72585af823fcc62801fb5b45fbbac975fdfc1ceaae70",
        },
    ),
    OUTPUT_LAYOUT_SOURCE: (
        "6dc348c27ba6f7c5a932c9ee0604671fb09d4ab23e23a4cedbc9cbf4f1ff09c4",
        {
            "_assert_output_leaf": "845951305b74c0dbb2a93f7598fc9d9163f85279eb047595677fca6ba8c38f48",
            "PlannedCore.__call__": "82afebbf7faeab894944b641a83549ab412a2dccf2d635f26fde3f7efbc0b607",
            "resolve_output_layout": "d1ad41b210b6c1313e6a58fd403332bca8e7b15dadc4f784905b56f387e5fce8",
            "_validate_output_roles": "bdae035c0ba6d6b07d7bb536b689f407808ef9435d3a8146a924cfa2278abdd5",
            "assert_output_layout": "ef03e0e23694014524ff2b57c2f86bd38cff03cb3fe3e7f24007460e1d22d2b7",
            "_assert_output_metadata": "c455ca94a5f5c58a5d4441e8efb374e74702d1d236390e2899790eb17613e648",
            "PlannedCore.__post_init__": "92f25766a12bbd9fe2e9be8a5b8a4034c537633499adcd7acc832b460becb73c",
            "assert_value_leaf_layout": "1abeb94a2353ff938207399038fc8e9cb978ab47f07957728b70d7eb70fd4eff",
            "_resolve_output_leaf": "dd0f95072a0f88a5bc5e7e8652f7837e9d77656ef4797eadfad74d318dbcbf6f",
            "_state_axes_leading_sharding": "20d39af17152908d29ee6ab50a3334ccb7636547cf4558ae8e9b0a6392c61441",
            "_state_axis_spec": "1772f28f0ed2c23f764c7494c6a6925c3b8fcee59180122a31c75bfcdeea79d4",
            "StateAxesLeading.__post_init__": "0bd0d18d5a0caff060ce3a860622a537a26aa061a164e9bdab4423b2ea4d2698",
        },
    ),
    VALUE_TRANSFER_SOURCE: (
        "2a19176cd1b23c556651028daca2cf1e7c1b62a5fa3304fdf1b58ea473e9c2c0",
        {
            "_assert_value_metadata": "c87ee67a7e3c8121c7ca3beaa64cbb05d31fbaa9d4cb98039adf6e845d989b12",
            "MaterializedTransferObserver.__call__": "3e06bf2091d6a3291e5674060c34313e03f0e7419af22b033de98b2fb2845ca4",
            "apply_value_transfer": "3e78b67ca2fdbb26ad1b0c793905c5e0daa6e4496ccb0a57d4be403d925548a4",
            "apply_value_transfer_plan": "5935a6ddc11376327dbec4ea66035063acb212093c52f93d8f38f4cced622291",
            "_replace_transfer_leaf": "7d6561650cf00a181a636eff3c320653916aa95ed57b25b32f6699582bd3084b",
            "_transferred_leaf": "aba8642256f3a23c7606359cfdae47541f75df80c90aa9d1299927ccdbf63112",
            "_replace_dataclass_field": "cd3f27158ba16a5902ce704907c3f97c92846255b91238e3ff7a5c51c481bf69",
            "ValueArtifactAddress.__post_init__": "cafe101592a7a7d2019ac1f2bc57e2e64295da6093a399dbb948bf8c38765e0e",
            "ValueConsumerAddress.__post_init__": "e2a6c26a492e21aef01bc5ae519d02426a0f6e067201925b2797eed13fadcc41",
            "ResolvedValueTransfer.__post_init__": "60d77078407d8b253b0654dd03dca371c32a06cf1efb05470ff453c815a7092b",
            "resolve_value_transfer": "1fb20659b43515507a727b46ad274d5bdc1c67a55ece59d48ed7e48e1e435a95",
            "classify_value_transfer": "c55b04af4530ac76c2dce1b47bb3ccf0d4056efa7c4dd68417257f59e03d506e",
            "ResolvedValueTransfer.cost": "07d257016bddbec9e6eacefb5e26e26c8c612fd6a0884ea81b40ac667a0c9011",
            "_named_axes": "46fa78227ccbe7e1dd881030b05d16794d3f45003c37b638ba0badc434d977ed",
            "_validate_edge_identity": "b9dc316bc5c544a59041fd3c7c67a48766829f66db5db31a6d1bb48eb96afc05",
            "_validate_replay_leaf_identity": "0da3b8851138ef9c733fa1b64977f7ceb243e6185b5e44fa49236619ccb2e779",
            "_validate_continuation_leaf_identity": "88493fd0b4b5ef9670c2958f580de3d53e855b187f3548e335193364a3cb61f8",
            "_normalize_shape": "fcc4376911a98d6c3ffa498464eefec1e62c2d0b574132de419c675d9cc915be",
            "_require_period": "897a27459ce6fd12d18cc8e5715ed2ba5e83d9db67bdae5d3f1714085f1bc9f6",
            "_require_name": "ebec383a90145607260f8505d8c40a31809cb05f8f30bbc6765dfa8a035f167a",
            "_require_enum": "9ed7eece67c49a94ed2e2d435c33b185a10eb76be59a92835bf4901e91d36996",
            "_validate_path_segment": "7b061ae59bb4a26715dd9428079a9ddab4a307e2c8d093bef2cae886f39da045",
            "_require_sharding": "2ad1e9f14326ae0e9db9baf151cf38b7993b0167a8c631dd0e442ce7ed59e4d4",
            "_check_sharding_shape": "dba09ca24339aa961d7d8bc46a685c3f04c0c8b4669cc27482e0df08217c26aa",
            "ValueViewDescriptor.__post_init__": "e5e888614d92e690a128d3812b217f8ddfcf6637809a896fe14c2401b6f0ef98",
            "ValueViewDescriptor.selected_axes": "606104f0b662cf88a37ba1496a05ed142f93858c1646c2794c1aaf6cb34f1476",
            "ValueViewDescriptor.structure_key": "39bf29f8d65be0bc19a5ee9a4b4fa6dc73a21c641f72f5d499c5367b58e0177a",
            "ValueViewDescriptor.identity_key": "45ab6d675d4ef8c8350e34a7361dc6b307186a4558c3a420c5347bbec6326ee1",
            "CoordinateSelection.__post_init__": "45f22f279739118a21c04a97d8fc4ad7108de34ee7fdea60b276afd1f84ddc01",
            "TransferStage.allocates": "b17555d91ba1e9a31a9ed57148afa064982799c4953d95998b9a6fff344c472c",
            "TransferStage.output_footprint": "41b4f1136bd24e82268fee82b7064efd7761f69d803ffa224ef434f02153ea99",
            "ResolvedValueTransfer.consumer_shape": "2f7b722fc9a7cefd9bcc0c675fb33111ce3caff65a317e5d97014c7091b0fb95",
            "ResolvedValueTransfer.selects": "a70c4976882f86afbd8c12c20334d58d5545540cb75bad2ee46b5fdc5c2f76db",
            "ResolvedValueTransfer.delivers_stored_buffer": "8134faf2e24194ad2240f72eaf5ba7bdd6e63c81e1c4a84dce8d6ea7294194cf",
            "ResolvedValueTransfer.stages": "045d5012182682b1ea9d41d56ada26ef6f92f3680eedcd706bc1986a9ddd4078",
            "transfer_result_key": "9bfd1783f1feb8ea7549037e0505435b47fbdfd0e4b6f5dd37cb4ba762713f6a",
            "_select_view_blocks": "e576be8253cc6f346825def3852a3def6da2dfd09c83b319c5b39a3031177b6e",
            "_selection_operands": "c01c5e777fda08c71f686d971eca03f6935e7ac1ea908e008ca54da2974c6983",
            "_select_stored_block": "b6f1647938496460dca1cf12ca5b26ba285287a01bb00c6cae0f367d391cd454",
            "_fail_if_view_mismatches_transfer": "c7d588a05f07d1f2ecb8873539cb2ce62a3d28516bbdded5734172f8ba6cefc9",
            "_fail_if_selections_invalid": "1ccb459a7f6b511bf7a631da7c31b5614a3b2b3a3b06a983d07333b44d6481cb",
            "_selection_sharding": "ba7789147c8b08c57243e70d3906adc90aa8c9fe57021dc4fd7fcc27432a0e98",
        },
    ),
    BACKWARD_INDUCTION_SOURCE: (
        "31a32011f99e05187afa496000d3d6ed6f4a7c5272e813cebc79ef52afbddd9e",
        {
            "_period_transfer_scratch_reservations": "fdf69334cf139ae765e43ce566d5467cf9c55684a6ce8710ca3792f0b87951f5",
            "_continuous_value_replica_required": "4f22692255b3746899ee7b2849776ea8967090af80d68d46272cd0b9a5bc771b",
            "_compile_all_functions": "72158d96f697164fe8c50255376c6d99be7d59912eb84783d803d4bc1fe1daaf",
            "_prepare_solve_programs": "a27b9fd4211fae36c9c83b34fd628e062b8bfdc6f970e9944e5356937d7aa3f6",
            "solve": "8079ba9405e485eb5e7b77462e03f5cd22afa6f375171173fc3ae92aec73577e",
            "_cores_with_transfer_cache": "fba35f0f74a7a496f2302ea160d4ce6b832d56abc6d0fee14bc07843b47a0fd0",
            "_release_closed_period_inputs": "b6fbfbdb2200f128c8c60f096e2eeaf173dffcf770b7c4a177d69c7e3e270951",
            "_retire_donated_inputs": "d5c3862195f2733fa97a5489d8db20d7c55e53431f0b2313b1b4f9a49b34e20c",
            "_prepare_abstract_program": "6a4dfcaa4277f0a5e58965b73c5390129848bd243d1dbfc91c32806ee982867e",
            "_build_continuation_templates": "17db7479d707cffef247c5e24ee08e253ef0ab4efd4590bff5f9a9a0a258ad3f",
            "_iter_edge_topologies": "ba1a385a7779fb841a865a76bbb185c30653f27a7f66c2d888e147dcddcd4725",
            "_build_base_state_action_spaces": "7a0648369e6ca73456ba099ed2cde0bffd294728dfaf482a7ea07b477eea792d",
            "_resolve_output_layouts_and_lowering_keys": "ab88e41e9f93ed97e7dd0523b3ea57481a96063202fda872801632499536611c",
            "_build_structural_blueprint": "199d8dd3e96417010afd7180ecdfe5357849cc978bab0fe6ece06e1ac2df8463",
            "_bind_structural_blueprint": "98162c6c2c8bc8c35f0e792bd576f8fe6bc4d5fffccfaf7c483e5b637f709fc2",
            "_structural_key": "397031bfea5522cc07247b00863e8c76bd0a9b43150e6a48c89e5925c18957d4",
            "_evaluate_edge_fold": "6ea390648d0a06fb74959ac36694fad5eaab135c8c1e258d4eb48458148a60ee",
            "_lower_and_compile_wave": "23f0af3cacead6ad18ac584dcfef62f2f31b8d654bcaa3d9bfd4f1288e58384d",
            "_lower_resolved_candidate": "29768b2c44fc16f6ac66a95ca2d64d1780e84fcd096655105dce1ba78b63e693",
            "CompilationWave.lower": "68a15417ac990410be8075b1d5c549254a99bed19547c7aee4bb6f9ba37bcb91",
            "CompilationWave._submit": "4f3ae9b8f534ecedc4beee4f681a0c1d32cf8a3f3da4f87b6fd5b2ad34d80015",
            "CompilationWave.__exit__": "ae5af742eb203270826619ea9680584de04bf39f29de0e2d1329aa94e1c508f2",
            "CompilationWave._raise_first_compile_error": "787b7683440a2b26449c59bab1bd7986ff983e5210d3bff3d07be5135f5bc527",
            "_compile_and_log": "6c2e773eaf54f547353e2734fefa2429d266a05279d94e99c8f8e596b916f74d",
            "_run_period_kernel": "a19324443322421cfa31fed0902e6538eeba3e80040b2e709e1d237e8394906e",
            "_regime_retains_replay": "04e8745dceb0e3c34e0f91fd11d27c43e0da5043cf2418b8015c15baa29d1d81",
            "_select_period_programs": "55bff2bbffbc5a75f00a656f684093d89d3655bac48d76da2e9dbe716b62bb74",
            "_selected_artifact_keys_for_cell": "1acc464529bc9833e48f727279682d969f850d2a3bb206e8a2695b1769f6182f",
            "_CompilerMemoryLookup.__call__": "77803efda3bb7e83aa81966820c20b8c3ab51bf6b7a0b5f9d738cf587eea73ba",
            "_select_runtime_donation_cores": "2f79409a1373d240cb3366fb45ae33937aab26ac8707cd14a087209e888a3d19",
            "_donation_ownership_refusal": "64cc4f02e17b0d295aea9a7bf30c5fa13ab93578f6c226475461d4e45bb3a248",
            "_mark_reused_transfers": "55e6e53f9d98a5e16b8b0548de876ed9e73a208d92156cacb837f594784d17fd",
            "_consumer_key": "84e636efe031d30a19d76638e93a95b8a0debc28b2e2814f36361b40d6f3cb13",
            "_resolve_program_for_execution": "733461e0e7be3e2593e5668754c205c67388e541e42a8aa0ed2e819140a9bf78",
            "_resolve_value_input_transfer_plan": "deebbe66f82de9f1480f5fc6012bd36db2badca5181381df85737ef814b7b223",
            "_resolve_value_transfer_layout": "0d912a492ca62d2881126ef4e8dc6f161b73c9ca9ab5430b4affd10b5199f4ea",
            "_lowering_key": "fbd31dd18b07fa72901cd9f1b69bb78a78667b2eba8ca55393ad0d1c9a83b993",
            "_abstract_arguments_key": "becd5c3e94366bc4e3e0afa31ea20886002228f7064c9a9ea0e7d0e681630dfa",
            "_abstract_value_key": "b79bdd528ed264be0093eb5d04a0341e7376606d478ba0770aa5cb14f813638b",
            "_abstract_leaf_key": "b2ff6f4e4276a9f591af0453cb86129f6ae9a473dd268478082567aee3ea4e8a",
            "_output_roles_key": "e439233cdd3158ee8b32c8c7edacc2f1c944fdb4146f45886aa27c6b5ffe8f6d",
            "_assert_lowered_output_roles": "59a66e241c3515c1cefeb48af9681ab3598ad5c86f7d59764b11341c6a1bc68e",
            "_attach_resolved_output_layout": "9110992b8988012a43aefdfca4095937368fd1bc1c1803c789b52b7f350d5327",
            "_publish_kernel_value": "50d720543d37adaae9c5e8799d72306ab1456d98ce590b3083b0d501d97e3516",
            "_resident_bytes_by_triple": "b87b5df2ce91647adf464bf2175d9453c186c1c65a7cc7e54ff6967dc015aa1d",
            "_resident_inventory_by_triple": "62b94dba3601033214595d6964060fbe5c0562de7029e60b500664afa73333c0",
            "_candidate_resident_bytes": "ac8b5a9aad74d28aa50407882824f567314e9a582dc47b60766eef33686689df",
            "_compiler_reads_source": "3a1099c48958b2a5ec59affe0d9cf0dc179d773c8e96e9fcf821526572f23439",
            "_period_copy_reservations": "75082956fda5cd0e642e988ffbef655df444a391b815c5e50cdd6e7d8ec3e265",
            "_internal_reservations_by_cell": "a6549059fa6fe1afcebe7554a3a46e85de692eece772f48466a5fa984a1c9554",
            "_internal_leaf_bytes": "ece84ac1a1905891efcec6e59693074f62da35ca50e9185f0fae55c2829f1206",
            "_retained_base_space_arrays": "286541cb0ed42234d8f8e2344d4b7ff67e6334e03b58fa195ad88a4f3ab9b13b",
            "_CandidateResidencyLookup.__call__": "7090dee9635f84f107ac4758fd21811925a264e73738341224298b43859f082e",
        },
    ),
    EAGER_CORE_SOURCE: (
        "3a35129cae105f740e1d232efce19e1d081d3b281342f160dde3bfd1a58552e2",
        {
            "make_eager_core": "90e5b45cc48bd2b3497224148fa1465679f73109d33cd52bdb3ddf949033d861",
            "_EagerCore.__call__": "8df1657559e276593fdcb171537cf94fed0726bc334fd02d84205d9873583f50",
            "_EagerCore.place_operand": "70ee76424e4ccc5bc14854ccbac4ccf10b5e6062357c6c50ae64f7737b4da9b5",
            "_EagerCore._typed_sharding": "274e9d40da69831ec01f8b88626389d092d13b6e9218d5ce1f06d90bb08f1147",
            "_EagerPlacement.internal": "b2e782cde710372edaaa3879f03f81f1187a728c542e47baf962ad1ed18400c1",
            "_EagerPlacement.__call__": "d35e3780ea4a5e1902d8ce06d415653de2db898f86b18abe528afb09c4263cda",
        },
    ),
    RUNTIME_SHARDING_SOURCE: (
        "d43ee5cb155622b3ad3d2b24b65d82c24663f5f8ec81d33915f3423a4d2fd5a6",
        {
            "runtime_shardings_match": "653b901544ca2c4d0a072b15663c718f7a758a02c9774587f31ff2b33e876a32",
        },
    ),
    SOLVE_PENDING_WORK_SOURCE: (
        "e8a12909af9f32f59ec7e5d8053305894051b83d96e0876b3a5e662c0c4bf00d",
        {
            "BeforeArrayDelete.__call__": "1c24de0e7bdcf0cb0bc04e791baba1c42b13d9956ddf52f8317356a5266ad74b",
            "PendingSolveWork.__init__": "906f9616790021cbfb647a5be9e3e540f03684510fc0b077971a70102c825fa9",
            "PendingSolveWork.before": "ff4a1626ee01b3b7bd92e381028061cafaaf5e7783600221967ae733eb9833ae",
            "PendingSolveWork.record": "b619d051e6b34944a0075d01904e4d5e3907e00c225da283c8948534aa46f412",
            "PendingSolveWork.before_delete": "94a1cf03edab4724c3c89b5178739980fcbb921d8d5bdc7eaf398064917e1f23",
            "PendingSolveWork.close": "fe6414b23cac14171f27d9b47d1be4782ec87455a8a49757b9d492aec57539fd",
            "_MaterializedCopies.__call__": "2c4a4c66c43562237d26b463d44a44ec9fa04df33758ded7fd344aa7a86022c6",
            "_MaterializedCopies.close": "6fd7c6ce17c5b58c6c2f8e556eb199e073c2301f532d81a7b9acb618346e2c31",
            "execute_with_pending_work": "7d76a2465425003d683ec21d1cc2d611f9c4978e13ba8ae09312320eca554ee7",
            "_drain": "e29703e5803aebcfd54107150edd5bdafd73368d5d736bedece1e3e4f4573322",
            "_complete_array": "55a0795c2c3ab63912524fcce52fabd7fbd5c9b9aff1d2e6bce9d80e49d4f6e6",
        },
    ),
    NATIVE_VALUES_SOURCE: (
        "21d7f5a0c94a9061a0ad5407073037c71f443b175555955b717268791b4c0a31",
        {
            "NativeValueMaterializer.require_entry": "1e16de95d530e58382d789be1a135a8f9e759b2c6d00b03113205f665ad24c94",
            "NativeValueMaterializer.__call__": "7b89455bdca484a6e66a976a49bdca23dae9301397400711c28b8a711c74f9c2",
        },
    ),
    NATIVE_ARCHIVE_SOURCE: (
        "046ed38802f340d0178d00f441a7190c6eed8fefdf6bfa13589e81fa8da268e7",
        {
            "_LazyHdf5Entry._materialize": "458c0ecf2399a095bcb0f828d62d821097a5b66d9153730891a3147d880af82f",
            "_read_and_verify_leaves": "526b4f4645a43ad6dae6169ecf719c0992b314a0c2852d46b5764d52828d0243",
            "_require_local_group": "9e893f06d8382be11b4d21c0e755314ccf46baecdd01df838e90bbf0a59d2a5d",
            "_require_local_dataset": "29c000b8a6132d9ff764a8b5eda7e62108d85e82bec5dbbe99b0af26eb3ba8f1",
            "_array_checksum": "352a1711c1983dec499ae45a18630a645a35c38f53a9f1e39a3fba2bd0461f5b",
            "_array_checksum_from_leaf_metadata": "d1d5aaa96e9b10c0d453c148b9f2ee4d6aa26e64d14ba6a58959df32d281fa50",
            "_to_jax_without_narrowing": "1b25088759f37cb0e69618da38903e96e763aba1e0de238f5660190596a82071",
        },
    ),
    COMBINED_ABSTRACT_PROGRAM_INPUTS_SOURCE: (
        "ecd6cf3f145e1daee743baf3fbe6461e7fef1e01b0e7454a3db052a36eb21e09",
        {
            "abstract_program_inputs": "f21d37135c8ed74465da0d58bac771db60a9d337672d3c7a986cc9074c5b07bd",
            "_OperandDescriptor.__call__": "8096f112e7df91de2b547295183174efbc897fbf2bfd8b42bdf07e19e9ffdd77",
            "_identity": "8c6016b4b6184a6bac24f68cfad3f0af6924309b52f488a43443874ae7b7107d",
        },
    ),
    COMBINED_ASSEMBLY_SOURCE: (
        "0c1192fbad7c21aeadc8978d8b152fd22b205778b4d611353cd8098364a62902",
        {
            "concatenate_arrays": "166524acde202897680156949a7154946eada67c9d334c0c7049d44946378ca3",
            "slice_array": "cfc7f9f403e29db130f989ca681841baebb2a301672cd4acd7b21bdb55fbf37a",
            "_run_assembly": "0812d399de134340e25c4c6364f2255e6c9389dce2da8fccece730df1b0cf86b",
            "_concatenate_arrays": "3788d18fa58eacc8369dbe797ba53d55cb4176b41a5af5fd78eee982947c5716",
            "_slice_array": "7fcb5248d6614a00a301a43c2e5a7bc438d81f3cb7593894d6c44486bc84deb6",
        },
    ),
    COMBINED_CHUNK_OFFLOAD_SOURCE: (
        "0f0ca55a9885f55e30d261f515f62c64bb2a93f14edd7d4a5b81b250f9870fae",
        {
            "chunk_host_device": "78b1dab1f595c20eceb674b65fcd24f29280a050134327b9d297ecf541a2eb40",
            "offload_chunk": "c1879b72e6f1726a75b3766c8ecaf1e7da2039024cb77f4955383e4808899670",
            "_copy_reservation": "f35ab1db646eae5404bf84aed0a4a5a9a4ddd29ab0299717e5b10cb831fbf841",
        },
    ),
    COMBINED_CHUNK_OPERATIONS_SOURCE: (
        "f88474824ab712b580ee5ee0d909499e63eb9a8bbf308b810502dc6c5470f1c5",
        {
            "slice_population": "9cbd202511ad883dda5edf9265fbb254b071a241c46bc01bd9bc57b776a896cb",
            "_slice_population": "d792ed23b4f16950eec4a3db4ee0c8ff95aede40469d0f220bd697a47a2d661e",
            "period_age": "3f7f8c712cde6b3a6cd90ebfb6e66a1680010ca05670ccae90eaecb42353e333",
            "_period_age": "e514ab831281075c33ac82eaf68aeb57779770d04051aefe64e02a5762c9b820",
            "regime_mask": "8fbc0153274cb05704a54704fcbb370211e0b635bb2224ee2630a5fa06a5fc4f",
            "_regime_mask": "e520b5e61daa2fd190ab3b79c0a66e332787e977a4fb0b173e59884e7d8a840d",
            "broadcast_collective": "55c4bbcc3e163aa8fdb221c2ed3e0ebfdbefae7e86dfa76366f565f00075241d",
            "_broadcast_collective": "e4e18b73e6c89877dae70e21a6707472219d44ed6fb5eef672577c276e34620b",
            "broadcast_value": "d19aca835a4b98e2775d9e5a0a4c1099deb65f6c4af5d8887f492b440086d6f9",
            "_broadcast_value": "52ec120e28113306a52e9a9a2b868485148ebbc2b0cceccff3dd2864db949b4d",
            "empty_fallback": "6364471157867280847d71bfd4660faee1f5137d08b896f2d952dec8a86a9b50",
            "_empty_fallback": "974cd130dd3ba7977b514a21225c5936bb4545f20c9d92412b85021424af9627",
        },
    ),
    COMBINED_CHUNK_PLANNING_SOURCE: (
        "5bb1570fc049b40cc2c1856c3237d7f340cc9e75bf6b84e62043cea656362cec",
        {
            "IndependentChunkReceipt.profile_count": "246417809388fdd5e03c2e38c3b81863e76bb7078bd5f2d04338da0ac7caebde",
            "SimulationStageProfile.__post_init__": "2a0a901ce96cdd863b8b93d237c6ad88b40bb9bddd6f0b90290374c1074c7f4b",
            "SimulationChunkProfile.__post_init__": "dc35ef9a8a3b963fa62d6f3fb2f54eb645e6d7d2f024a50bf4174d0a32bba602",
            "SimulationChunkPlan.__post_init__": "f6c3be86ea1d1f22e94c94743dc6fea1275027095004ccfdb30684bca8428cca",
            "ChunkProfiler.__call__": "0d8cfb4c8530d329393a4f74150d216e68bb03a061b10b6fe93e401a33c6dc51",
            "plan_simulation_chunks": "56f7031e6be4dfbd264314cc8c6203b969e10c61f7a740c4aebff16e66a1d122",
            "_required_bytes": "41c7b9e4758796454b1eb87ec5d943331e76849976f567dec0e0628abeaeeeca",
            "_validate_devices": "c7387fae49d5a96a9a3b0454ffd74e0d5c7a798a8b4c837bd2c66d6751cd7c3e",
        },
    ),
    COMBINED_CHUNK_PROFILE_INVENTORY_SOURCE: (
        "1983b111d96d5623698445fc56805855c217ff122352cc2dfcdc78029c04e295",
        {
            "abstract_tree": "bb108082f7a50043295ea98d3b3df01d55c07c072d2310119d7fa20689a22ab2",
            "_abstract_leaf": "223bf0a519fc20dd9ce5d8515832eedf7f17ae5b819900915b1e5b1dfe1f0418",
            "payload_bytes": "f3f974511758fd0afd2467c4e657dcd9a4b62c4416e25dd1163bb3eec1827e39",
            "add_bytes": "dc8405be6264d34c70ef2522ab4792b55f21865953e2aecc0cd6a53a3bf8ef78",
            "maximum_bytes": "783dddf0ef2b38ae119990b8c4815f4469a7396b94a993a7ddfd3b3b36bc020c",
            "ChunkProfileInventory.operation": "f7e21b0c513d2c5e3e2e52aca45c0fe0ffe888ca841b3f97cca821172710d7be",
            "ChunkProfileInventory.compiled": "04ad615a043a0206948333e57fce3d685eeda452a8fb3c01565e37bcac9e5b9c",
            "ChunkProfileInventory.close_unit": "f65d7fedec0c9f24adf4d8be034aa5be7dd72ae9bb87d312c7dcdefeb885831c",
        },
    ),
    COMBINED_DIAGNOSTIC_OPERATIONS_SOURCE: (
        "3c71a47709094e14221c930d05cdc4f63539f519b98db50a9884d9424eef8fe1",
        {
            "period_value_flags": "1ca9e9e2920a0333915071a229cc907c2fdea761f313ebed1aa6e7d35235908c",
            "owned_value_nan_count": "e6168d82082f8c26ffd75a75386d46764a59d1ec3030b11e9688d019a9e668d7",
            "transition_counts": "9ab0013eeb414dcc3bde5bf91cf144079b27ea4b3538d4b7858687cfbe84dfbe",
            "profiled_transition_counts": "586661d6d039c19a88a0bc9c450342a8c31f36eb0453d0d82147ee6d15cf6def",
            "DiagnosticBinding.__post_init__": "e94a4dd9f381772b875a4dc7715ab199f7f0c7b5d49d3bcff67399f6473d5e72",
            "diagnostic_bindings": "0266571f8d59c095f9797b9e05bd43e5e5eb9463740b845896b51f555fcea090",
        },
    ),
    COMBINED_POPULATION_OPERATIONS_SOURCE: (
        "0a848b0b82869b622d9c3ef464c898935cebfb2f0b2f4c2e193d1b1f44581472",
        {
            "default_roles": "88afa3c0d2bc242098f72d9ab7232308a7a26ba97fdcf02620b70c0cb1929a13",
            "regime_is_occupied": "8e2e9ccdf0821653598ae983149767955a446b506bb532065f681e785addc62e",
            "canonical_roles": "4542061502177865523b5c1d0c5346c2aa2ff46286c5b81d3df65a0e7e29fb12",
            "role_mismatch": "faba46dd0931feb531692c4b87d8cce2991dcf2d7a6f6c14de8f9b24d6a4feb2",
            "starting_periods": "211a8bcd2a68fa7bf617cb176c7e1321af79a5b4fa4858d5dbab5283b0546672",
            "match_starting_periods": "0d27c548f2a58283b23dc3ae92cfe86fc12cde81286479eeb6dea43ba2dbcf12",
        },
    ),
    COMBINED_SOLUTION_COPIES_SOURCE: (
        "5894fd162877de03428bf8bc4b8c2ba446943541a5f78f4394f13e9194878dea",
        {
            "copy_solution_leaf": "7967a1766bfeeec72757ded50d7a17a32089f9395c95710da8afb2acf40c50df",
            "_copy_value_leaf": "70a9b70985afb5ad1252e4e430e7884366a8e19a09c363ae2efbcaa07d0970fb",
        },
    ),
    COMBINED_RESULT_SNAPSHOT_SOURCE: (
        "9ff9da5ea8ae404ecbdfeb4f35225cc44e34b10d1f09c181284b58f524191ad8",
        {
            "snapshot_value_store": "fd84054f9efd1fc5d1b1de1fd636d2eee0bcb3b73b12a9317c23dcbfd9c977f8",
            "_snapshot_value_coordinate": "3b3af3ca3c9ddeb4226368bbe6c8a652afeb307862d926e104f3e61663d6b4d2",
            "_keep_payload": "be488773d5eeb9976214ed9d50ad30c26c54fe938916958d58fdc9602d8db03f",
        },
    ),
    COMBINED_VALIDATE_V_SOURCE: (
        "2e26a8a63fe03169be2498e7085568e61a02e912089835aac5302f3b48e290a4",
        {
            "value_function_nan_error": "597cffa965bcef3bdb4ee5f09aa02688eea5ec7fe3e6148d4b2ce8b3082d9f65",
            "_entry_support_cause": "eaf8001c1762ca0d88cc374068a06ad22d43868b03ff8fa462c574dc788771b2",
        },
    ),
    COMBINED_LOGGING_SOURCE: (
        "cbf8ee6abe7f5856b9f86f20d3f1f760aea118908f67acfc99b1f302c5f25d79",
        {
            "_owned_values": "9295ced0a7415a8738e6e9dd1b6a2ee6dc933bd3fdf37193e591752239c411ed",
            "non_finite_by_regime": "aa597487e43e16a2a001d5c05c0e00977c494349b9af3e65f8afd47aa6d63de4",
            "log_non_finite_values": "c2cc7e2acc85cae7fface00d8b38c23b7530b4e2453ff699182a4d40aca1b964",
            "log_regime_transition_counts": "11c5f45c415bfac54b36a55d52cded2911152f3e411e61e2d90a47ab19b11725",
            "validation_enabled": "14475c5e923e2ddab2f7128215e0ebaecfb9ff756a6ad786d2139f1a490e4b35",
            "validation_raises": "1cce1da3fb0520f1118923b0d5873d6d38b2e7ca8ae72ee9968627f8b43d2206",
        },
    ),
    COMBINED_AUTHORITY_SOURCE: (
        "5e998d72c053088f93e60ebc0c899fb7acef04fac58ec2faf5395504cd196d67",
        {
            "_ArrayCopier.__call__": "d708cfc14d20e8e157d30abe2336b3faa117b225f4ed3b10840d4dfbd4fa4c36",
            "_copy_artifact_array_leaf": "fe5b9e45c06ac59059a010fd680ac193e87e12aa3c5e38e980d84063d139bc07",
        },
    ),
    COMBINED_ENTRIES_SOURCE: (
        "aa29f413e12ac9d21e18068a93d5d246522a6536c8b217fe38aa7a384e771e67",
        {
            "_ValueMaterializer.__call__": "5a9d22497c8d3cf499823257fa231a4fa17dbfc70eee3b00f24b5ae1689f1215",
            "_copy_solution_value": "2c10ae621eb19cb4fab9fc9dd5d96160f4c4dd91919de2b2e46bf784a3e7689a",
            "_CanonicalValueEntry.materialize": "775b1abcc8dd36db4238efd34ad5532553450e23d1fab7e1cd4bf405948789ef",
            "_CanonicalValueEntry._fresh": "f24838e27eb2fa8147650f7fcf63c29ea04faa4f23d99a6e47352cd24b99632c",
            "_canonical_value_entry": "c3c8a9a342aa0b8297e43623355bfb7449c1e8451f517a3f2604efe3b14a5b38",
        },
    ),
    COMBINED_STORES_SOURCE: (
        "0db2c6521e92c1d07c05e8bde6db3549a86b43947ef1fe948813b3da048a2dae",
        {
            "_admit_value_entry": "0b273e3cafc4337d7404e7c434158b643e5ffec6fddfddf24cf900a52584b04d",
            "ValueStore.__post_init__": "4f81b32757d589ced2e605619de79013a2d9450f319e7adf77094e898b8fe7bb",
            "ValueStore._initialize": "edb362c78b82381e5f48801ed31490f8e3c641f055fc4c15ec79a5a12673cf3c",
            "ValueStore._from_entries_with_copy": "3290446dc5ddebc395b60c198954da42c3b074bb80a12e43b9c9cecfdfb235e5",
            "ValueStore._load": "a85990275d471b14732175123f5cfd1ca8e83432b9ffddaacff748e13dc19788",
            "ValueStore.materialize": "3e32297e221c4bb51cd25687ff756f6c02180bb204978d4ac169c9d0a3b08802",
            "ValueStore._materialize_with_copy": "8577e39aaca402c4b60d400482cd7354fd7c89c6b3b9714adc054edd62973de1",
        },
    ),
    CORE_PROGRAM_SOURCE: (
        "88f8f3556e044399dc81434ff4eb534c2930a88e4b06d9c999b0c485bc8be31a",
        {
            "_validate_abstract_inputs": "70f96b7582b3a085fdac809c48c6cbe5788f28d5b5d94dd0bb19eec5a3bdc973",
            "ValueRead.__post_init__": "9eb25ccdd2f056a241092952e6b92acef943f1babd2f2063af7b32da935a7190",
            "ReducedAxis.__post_init__": "030c5cd6aadd91db7c7362241817b7b1734cf27d853824376ba1206bdfae05ef",
            "ReducedAxis.extent": "aebdd54708094461d473977783f0588d2491c212629e1f5a40f8cf24c789802a",
            "TiledOutputAxis.__post_init__": "dc42959d05c19bee720f1423b60179a0d729b74546bfd278e0928cd2a6b2d3e2",
            "CoreExecutionRequirements.__post_init__": "b8c76151ae6388cf29f8b580d1b6604c5969829bfcc46838a5adcd7d04e73c7b",
            "CoreBuildContext.__post_init__": "c001bdfea799659c6e0f1d0ee09180940d5176d8a59675dcabef190a12716d7b",
            "CoreProgram.__post_init__": "cc2b0daa4751a08b0be23ba75f086bc0c1d8ffcb6a45a9f7b1e3d914375bbf47",
            "MaterializedCoreProgram.__post_init__": "91815679a6b9b6ac14dfc35f3a2f40b3fecf31200cb03105692a963712570437",
            "ResolvedCoreProgram.__post_init__": "468796cd81e23cb793ba8fcc49edcac4167064550a54c16fcfdbd4abe48674c0",
            "core_program_graph": "ef88f028d82cd99b9d3768ad89df5f917e1b11142899b3ad6ac113f35a676226",
            "_reject_native_duplicate_authorities": "e07ed1c6ff7894b2b6f090c2850c45a342a75924271b2466d23ab4b007b1a13f",
            "_snapshot_and_validate_graph": "391b89b66355ecdba2d0c869b4a748f2a0ef065610255e4819e398cf53faa7ca",
            "_validate_replay_replacements": "2aed7e1993700410979ff136c2dc13104a1f210b7d1eabf3bfd7dae22b75eb69",
            "_validate_program_declaration": "6158dc42296c246bbb8ab23d270dcc46e8175e89ed190af503bdebe788f11c6b",
            "_validate_retention_declaration": "62f75acabee1eb14ea74360f8563970ba1f5639dd06a0573a49d80402e437ba0",
            "_validate_retained_artifact_payload_types": "8ceae90284dbdc25b64e0b66d79990e8f9104f9a6410d90d8eb94c426d24ecbd",
            "_validate_disposition_reason": "5df6651c2d9db4ebeb5542394c738c5238144ff19f48f0639b3bb6b43c63bc8d",
            "materialize_core_program": "7bde944455cb6ba4b0e8dfa5919970d0f2ad52b0af657dff49b597feb5cfbd6e",
            "resolve_core_program": "fc430e878ce5668159eaa6a26d7255ab7e90a61bf2bb570fe0cf89443416dfae",
            "resolve_core_program_candidates": "04b27f16722ddc627560a62ed9f436e8a4b0bb2646cc7edf8a7d7e99abe1a50d",
            "_resolve_core_program": "a23b23097feb561258fc5fb9e15e79b6286e1e566b155ce0ecd37c44b4bc6378",
            "select_programs": "545d2aaa5fd158f5cbbe4c8a2bf69cfffdaff588eb5066fcde54de59fb37c91b",
            "_validate_core_program": "b7558f7fe479363723c2b4d4925ab954e262883c4e07cbc956b72bbbe4291511",
            "_validate_materialized_declaration": "d4a7b72b1943b877e689ed05bc6e83e8194fa20169615d7a0a8bd970bce96534",
            "_resolve_input_transfer_plan": "f62a9fc86af542f15a99a0dd93978b681e1dd0c45d14772b711a6d80d3eceb2e",
            "_validate_value_reads": "cfa03931a7c89ebf7f94583e795f646dfc261653519e630070a5aa34e8954d9b",
            "_value_read_argument_leaf": "6cc2a71eca6c01578375919ccffcc90584eec8c39acceb69e7531aba15277aa2",
            "_validate_transfer_argument_metadata": "2da767cf572598810ce123e021513c3a2e0cae1813e07857a011b9262590c051",
            "_validate_reduced_axis": "d84a3086d5ad81400fe18832049b68f1b3ca81768f8ea3839e351b55d790c1a1",
            "_validate_tile_width": "e986f3d40ff2a7ebb398a28a1cc086740ac861788247716363079ea7ed438f93",
            "_validate_coordinate_argument": "eef5d339f81e5f24dfffa4215ff9eb27ee286915a6d5ec2aab0f2507f4cca546",
            "_validate_width_keyword": "d35cbcfe40e9888d2d9bdde001c488fd82f01b6a8eabbd7dbf6121bfe709674e",
        },
    ),
    SIMULATION_TRANSITIONS_SOURCE: (
        "634b07c5ad8460b7c8c46d3740af8a31d8e8285c84a9c8a929a2837ed6d1967c",
        {
            "_draw_random_regime_ids_from_scalars": "548fcaf12654d05a584fabee3fbd81fe42ad256307d5ea4df826cb3ae3355a5a",
            "draw_key_from_dict": "9dfed8dafa357fa87cde342dc7f174028e4adf8454b5906a4e0d24bee164cdae",
            "calculate_next_states": "fa123ef47cb0aff0c4e6f93753244292d2c25b331825fd9de573424770831922",
            "calculate_next_regime_membership": "c9083b3e4d83c8c30cb36a963093890e2388aba3bd203a6895cf1bb80eb828ec",
            "_update_regime_ids": "93e8f8c1aba6e42f14269aac5d911c722e5e132c207cae6fa3e07bf138a58163",
            "_draw_random_regime_ids": "87fa7035d1095f0dc1a17e51e517ef559808039ac8a107864995ce94f2bf7430",
            "_advance_states_for_subjects": "a7ea5a0c113ec1aca76552c7d54502133076bee2ebe96052299ef46bf9518547",
        },
    ),
    SIMULATION_RANDOM_SOURCE: (
        "c4d8e3dd9c67fb0ee5ae84d00d28d7962bb88dbe8409e3222c0b4959eafdb236",
        {
            "_generate_windowed_simulation_keys": "7f78949425a2334609fdcafc40af7004735f24079634d434a8ce4a0610092dd6",
            "_validated_chunk_window": "b3ea4d659dd8c777e88a08d8c3e3524236d5cecff309bc5cfdf3217e5766876c",
            "create_simulation_key": "76fca4fbc2e18885d2ee67512e8207616fe8d17443954d9b36765a4bc52b3dd4",
            "_create_simulation_key": "1ab7921d3e42ea7255fb8bbc60d1ed58e408e8818d0bac91ec132c276c62a05d",
            "split_simulation_key": "9968b4530b74638b082b63c0160391573b8ae4c20a9dc3b733a9b424de259921",
            "_split_simulation_key": "666d1e23f386815a8fd331cd543813fa2fcdb501f7672a82be9f48fe31c293b2",
            "generate_simulation_keys": "d56b65542b750105b09b3e7e08975cb0cf70bc24e95006e502c7fce88855c194",
            "_generate_simulation_keys": "87d86ba01310209a8ac5b222c2ea6aebc8ff4488b2ba373b1b28d5b682c737b8",
            "draw_random_seed": "42fb402a3282c789664d45fb988b3ed07b8cec706924d3e6e11e26b44c393502",
        },
    ),
    SIMULATION_RUNTIME_SOURCE: (
        "b3c4fed366a4d5d0b5e562b3782c75509722bd66d91d53969315b5c1d1f85863",
        {
            "SimulationDispatchContext.__post_init__": "6f9a709d7cf21cee4d48b57095eeff9d6ce553e1cea780afbb9f851e6079bba3",
            "SimulationRuntime.prepare_abstract": "9b5e9af148ef5076d1c76825fc30cdb9d160bca79d74f621044c162b14b4af9b",
            "_materialize_abstract": "1540e5a2279c62ebfa90dfa809d1d6e20da228fbfd418db6f7143a1d08760be5",
            "_dispatch_widths": "d68b9429ae1eed54ed99a49ce6793880f52bb5c09fa6248e1ffdf85f81806406",
            "_unbudgeted_subject_width": "2abdeb21c827cce18a98efda59515ac14649d7f20b797fa433b0765368c5457d",
            "_subject_slice_bytes": "d0a84a733b3d17d0575b5b319183c321f5798513ead149794b3ce75dfd95362c",
            "_require_abstract_arguments": "007bcbf724a27fc37a09a96ca30b4d2c201e6d73a19ce3c205f3efe0106c254a",
            "CompiledSimulationProgram.__call__": "b6d83e57181e390feab02f0cfde8d1d6eb88cebfb6e2638b88b86a7b1d4ecb06",
            "SimulationRuntime.dispatch": "4fb4940596e2fe53588413bb0053d8b483ab0920db37a8291034be442f140a35",
            "SimulationRuntime._bind_prepared_route": "3e1bb7ae085c3ffd03f46ee569630c9464639fc28196947eeaa8519353b34725",
            "SimulationRuntime._publish_prepared_route": "01e0eac9f1d73abe967ffcb570f7e6ec1e98eb139205ee0c50709e8028b637c1",
            "_operand_signature": "829c0a89fd44b607755806f6fcea34cb6d027c088a8db19f2eb700cd29567901",
            "_prepared_route_key": "25bd643ea3613ba2bb9531ce758cedcf4a908e3c1dbaa541b6eada3ab6a92ca6",
            "SimulationRuntime.prepare": "a5900dbf8be0f33ceff22c1babfcb6b706c05cc84fc6176446e024a8f4f40f0e",
            "SimulationRuntime.is_prepared": "9495309ce3a74126c48f3fc04b517c738081fc7a1a8e20a8de3f0b2ddd28f6b5",
            "SimulationRuntime._prepare_materialized": "3bb08504989dddee6748dec912a78f12b013ccc232a74070e05dabbb80727c95",
            "execute_simulation_program": "c945513d448bb584d7b14be0890f85edc929708343b9a25fa4ece8a82bb6c981",
            "_SimulationCandidateCompiler.__call__": "d2292d22403be650b888d30b7649b0a089ed81115e1056180712cb3f650d5ad5",
            "_SimulationCandidateCompiler.lower": "ccfeddebd99fe733f918fa07e8a056a9e8ec4390b691b5c6173d0e95a6e06a1a",
            "_SimulationCandidateCompiler._bound": "041de40ac82f2767010e4460f4f04617d4cbe643ed34c77d8ddf5b901f452ac3",
            "SimulationRuntime.lower_abstract": "f823fb439bae7e6077de35e53e0561e6e634e9b8fe2266fef9c401aa2dd1d7d8",
            "SimulationRuntime._publish": "840f5393ba14b7d8285808f30da1bf67ea8362ebf950f9728eb6faa0b10a1d01",
            "_with_compiler_memory": "70e07953cf8915500cc587dc425dc46f1cc81b1313517bbe675f527062b79bcf",
            "_with_subject_extent": "c48502d31e8f9fb29221122b0de6d10f38f3d6f740854a82512c2b964079e463",
            "_build_context": "2331196d1ea5a81fc01a55847b5a4559959de3076cf088ad521889eba414de59",
            "SimulationRuntime._materialize": "88aebad552787c6af950786e8e6d01340526dd719722ca007a2a5fb42133454a",
            "SimulationRuntime._require_budget_context": "6818040f38e55d623ae48f58b61ab3f1858d7ecd79df5b3f68321c364bfa6910",
            "SimulationRuntime.compile_candidate": "3dacb8cad7d3a1677b1e6e3a1e29613e641df3aa55b02c0c7e5c0c0da18b94f4",
            "_CachedSimulationCandidateCompiler.__call__": "28dcd8f5efac82c89e21143112cf45402a5dde3585423bd912f4c7b455e62107",
            "_simulation_memory": "f27db66a13a6dd01dbe8acea8f3fc1348257b55eebb8e4f45ed8c7f8ca3607ca",
            "_SimulationResidentBytes.__call__": "1de7cce7b963299d2803cc325d95389fe99fad8282a3b9a707bdddbb3dccd450",
            "_simulation_lowering_key": "707adbce53fb0fbb1c5d30dbed7d058357cd007ccd5a507952d64e17b8c0d63a",
        },
    ),
    STRUCTURAL_BLUEPRINTS_SOURCE: (
        "9c3b5702c28225f81c1d102c4bd1f78a57705202a6ecce43d60c01e1a8982658",
        {
            "StructuralBlueprintCache.__init__": "a8a8b85317f46c32cfcec6f9af8d52506bab96b00106b219c64e7afe0097b4b9",
            "StructuralBlueprintCache.get": "bbe960e78093b2927cf39b80aa9e33086af960f8d0f10cdaa649666b8d8b6826",
            "StructuralBlueprintCache.put": "94c09d180ba9771cc63017bb4701fd15df81fe100cac46584a0650d8b838ee87",
            "StructuralBlueprintCache.values": "50b589fd1b12dd52d036aab2fa8e32f04a3a943c30f0491e4693a6cdc6cc9cdb",
            "StructuralBlueprintCache.__len__": "1e7e25d95aac75c48b88dbb12749874be08aee275bd168bab4da5e03ad745403",
            "abstract_schema": "6a2b40cbc027312373c18991e12e71a24e48c818faa92078d98d83823bae9fd1",
            "_leaf_schema": "6a891176a75a2b7704e444d1afc5bb588e786b4e11dbdf40f9001fc67c5a7625",
            "frozen_policy": "3e715f11ee87aa2afb2f5e0f7721d75b80d8a1570f59a7deaaa5d99533ce22e9",
        },
    ),
    SIMULATION_POLICY_PROGRAMS_SOURCE: (
        "e0718083e409d7315942bfc4d1a6536fb955fe293cb56a002f32c1476773bfc7",
        {
            "ReplayPayload.from_policy": "e29e2d20b51b07a6412aff4392720c1818e0b8662ceac0eee986f31e786dbc98",
            "ReplayPayload.restore": "e1599af0b523cc29f9c85190c51f29c6276cbb995d9dab8f085543780e511170",
            "_flatten_payload": "76b9b73e16baac373476d7717962fd4970064ea18e249fa28ddb43546913e042",
            "_unflatten_payload": "3a6918f57cfbe806f6fdeecb68342fce7a8a3c8db200603402b87bb1d1afb222",
            "declare_finite_replay_programs": "ead89e6e9501a866d391bdd1b4b67d9770a5e95b33fab1b47bb2c9a7c1de9ea2",
            "_program": "c13ef5ee27095f9952bcefc30a80b58fc7384d52643430b056a2b4a6e9ec426f",
            "_policy_reads": "10f8afa6126d8f8fa340c021147fa8ea69ad8ca126830968a3e1c43ee8b1d396",
            "_Prepare.__call__": "3263959a9fb7fa81ba52eb9b0afe9231865c73049a8367a05b391f700a20dd9b",
            "_Rank.__signature__": "50aa82bb239a45eb831ec1de9cc04766c44b88f9e2a8898243273711d1ca5bd7",
            "_Rank.__call__": "5a234537dffa207a1b27a13009a48158129b3eabb63600ffe7010f69f3a340c0",
        },
    ),
    PUBLISHED_POLICY_SOURCE: (
        "cd4609c4504a7d66c3ad4050081530e01f76a3a77110245f11846fab6f472252",
        {
            "_flatten_nnbegm_policy": "b06597768369f79dc636dc72db2018a808f88e5e6557bfb0f9bf9098011269c0",
            "_unflatten_nnbegm_policy": "a4041cf4e29f681e9b61cc547d33d58da85f94b61195b88efc607e77275efce3",
        },
    ),
    MODEL_PROCESSING_SOURCE: (
        "d1c57a78539506754ed16bdb4f55967c61fe39a5aa0af9160d83dd9d8e664968",
        {
            "build_regimes_and_template": "da8e2ab92d9248698d20e43b0033ae1abdbdbff43f89f70cd2884304990bf1e1",
        },
    ),
    ENGINE_SOURCE: (
        "9425be330690fda85804474a11b04243f8a541b7902e9b0aac0efe6315957342",
        {
            "SolutionPhase.resolve_process_grids": "700a7c24fe948dd51bb3bf1057504a5d05ff7ea1499fe51f8341623ffe41e34e",
            "SolutionPhase.state_action_space": "582368bba80ec725dffb46af088874dae5dee05c8601290e9fe3e8ab34a68cab",
        },
    ),
    SIMULATION_COMPILE_SOURCE: (
        "d172ba65d290efe4cec76f6c4efe9488388b5377bc9f2610e8568a83faef474a",
        {
            "bind_simulation_runtime": "658c0e2dae1f0cbac50698962f522512dd10751aa1352174387ec15c7aace158",
            "_subject_devices": "712ffa064623ab4c54ccd34c6c6421e48cea41ed19a0c28038c01bf08bf9d46d",
        },
    ),
    SUPPORT_DIAGNOSTICS_SOURCE: (
        "f8bd7934c57ef789e893dc86b7a9d03a0016c298f3d3c78d298dc7e4e4f71e33",
        {
            "_emit_post_loop_diagnostics": "5b7023fdae689cc738497a394532a608c7500a92cb92959813be11ee8f37f650",
            "_raise_first_nan_row": "216ab701be0575c51f6d17ae8a27f6b831015758b6ad535cdb6508b0fc3ab1ad",
            "_raise_at": "2ea2db521ed4796c34b732a33262ca174162c1f4ea5282789631e2a2a1d67a50",
            "_reconstruct_next_regime_to_V_arr": "4bd93914f56d0a9e25592c7a50d64e303f314e591d31c55c5d19bf22bb67315f",
        },
    ),
    SUPPORT_FINGERPRINT_SOURCE: (
        "3f522e9778b89a760d96ab99e93b48f4d4fa846ffb084813c2dc7982d017f674",
        {
            "fingerprint_solution_support": "db6206fa4209d1d6892566377c6c62c6bed1aa28f23ce49d85fb3455f978d19f",
            "fingerprint_model": "04c8f2960e52d4144353b1e3b962edbc17b00eb05fc68a3207cf4826acbbda51",
            "_grid_support": "9709180334fc4732e2193f6b736667d6a96648f1346401a886bd7da0f41b65bf",
        },
    ),
    SUPPORT_AUTHORITY_SOURCE: (
        "2e587a86964b26421bd6306e39f11f6da5d99fa8f0865d528cab7cbd7f1474a4",
        {
            "build_solution_authority": "2ed2fe3b8445983cb07efbcd56e7fb1e0938dc0313a9b7a13c1265a5cb1fdc35",
        },
    ),
    SUPPORT_PRECONDITIONS_SOURCE: (
        "c63bfa104e38aa88a691619881ba6f7ec22bb9c1bfe7c1daa28704f42769ef39",
        {
            "check_pareto_weights": "c59b54d3349a086083b2904e0f49d2396da35debce8b339ac1923eadf263be11",
            "_check_one_regimes_weights": "1eb8d8f9eefe3f49f32294f69c2b619404aa892fc477497704ce290facbb01ff",
        },
    ),
    SUPPORT_TRANSITION_CHECKS_SOURCE: (
        "f38373edcbfca8693046b9a171a8ecf511abcb5c105191dbac2c8dd1c1284442",
        {
            "_ValidationSummary.state_action_space": "130e9abc335a298a37205deb98e1a3583e7632fd371ad0feae8413d91732cc4f",
            "validate_transitions": "6dad8f387ecda88d9984797cdd8da57d2d4609201fe8b455c7fb6bc59a27e2d4",
            "_validate_transition_sequence": "f57042cc6f993861cd755032f35e0edd80916cf9f2cc157ad09ddffa7c63ed72",
            "validate_regime_transition_probs_all_periods": "bd4b49bcb10108dd06f79e01d59ee449cd7db25bfb9e15449561d23655392c02",
            "_validate_regime_transition_single": "689a067b3c00cd24ae46e69c12e1a2a0f8f6f530607b105de2403aee82bd13dd",
            "_evaluate_regime_probability_law": "c26148c3519d19ff749d06a65bb484f50fbf0a7ce0df36d09e1a1d98fbbf2e7f",
            "_regime_probability_law": "45faf8cf57fd2fb8a754589d2edcda7a1eb82910c7dd4f3c9fc8115caf6728a1",
            "_check_and_release_regime_probability": "f54a64b0239559ff0527404fe7dcaaf0783361ee24d5bf628d971f8037873b4b",
            "_validate_regime_transition_probs": "403ad994ea48ce5ee2982ffe904f3d57adbd1039d8fe9056e989019d0af19749",
            "validate_state_transitions_all_periods": "6309dd9cf4fce55766e6c21c70964ef0f66d62c580ccddd2d9f884f0644d362b",
            "validate_joint_transitions_all_periods": "61e267de88fb2595c99f5cb25463ae1abdb12f673cae85f72ee7aa6a9b7f715c",
            "_own_transition_outputs": "3fbee994063b5cb63b318f51f35e7a88ada8a61ab8c9155867a54e9fd8b22a1e",
            "_set_transition_outputs": "fc03c7289678ba54ab855efb933f80ef877a071c2a2fb1566fc129053288a143",
            "_transition_owner_tree": "b977c18385d2630136d5e463d47786f1320862b7f58be015007eabbb19934bfb",
            "_validate_joint_laws": "41025cf308ccf0b7bf790c6e6a7dc2318135788f73a18827c4d5a984b8111b42",
            "_check_joint_support_schema": "4a31efe636b7d62f6b50ba39abc8aea16c7d46f1f1449983ef1e2c0512126da7",
            "_evaluate_joint_support": "79fa2c4e900159df49878f67069af9ba69eaa1f621502442b4bb8fe6ff37ec73",
            "_validate_joint_support": "d2d75d2d4cad7d0ed8508c00941c7e83875385b151046a7700c07ce728528ac2",
            "_validate_joint_probabilities": "3b124fe5d4fba178edd52773f48eb1361a1de5b437281bfb4c9434b001775442",
            "_evaluate_joint_weights": "ad98e50978a61c64fc8e90ac84db7cb7cac6551491aceb3446dc352e38f8622d",
            "_joint_weight_law": "3ee7cd29818e915de77c8ed40d905b741912c49706cf76a3fd09a900d3c38fd9",
            "_validate_state_transition_single": "441a9e31f06432159c8bf612c6b0085f016765dfb95efc0cc856efd16722d1d3",
            "_check_and_release_state_probability": "bbe5c53744d66283f4268a996341767fb18455c29017ca5bf6476e1f6ae1df6a",
            "_evaluate_state_probability_law": "91814f1439275d2c0af44fcff5cd2b6de9ce8f746a7b3945cf803120449e8e79",
            "_evaluate_admitted_transition_producer": "310d5e1a8e90866764775a44b77bed9103deac3a71b0ab01a75603f8206b7de0",
            "_state_probability_law": "fddff0c03e9d046b20c4694816898d5d0e3fbf00007c8f618811b0f43f9cd0c8",
            "_abstract_transition_operand": "da4ed98dd978aa74b9e1e8886ab15e60f875198ad048d68ff23d143d370af33c",
            "_check_state_probs": "4e4e56e44907490ddbf993e53fd1f4e49611a505884128c66124d455143f23c7",
        },
    ),
    PROCESS_GRID_RESOLUTION_SOURCE: (
        "f6e56d413ae3206bd18f352853b127a3166e26ea19efab1ceee2fd63f7b74aba",
        {
            "ProcessGridResolver.supports": "63f102efa777a1baf563b1f63f970cb699ab991e2e0e019973b02aec9e76a46a",
            "ProcessGridResolver.__call__": "9c9c287a6df809a803e3ece8c3c542d6af50c224bb90318b86ab1c5664d5b8b8",
        },
    ),
    UNIFORM_PROCESS_GRID_SOURCE: (
        "a10d3e7c62a846aa346003dd1a8531d491366047d6e2f959d31bed3e1c761fdd",
        {
            "_shared_uniform_operations": "77e4fa81b76a13e05edecb8cbc14571226bb6e34a9a80842289ac345cef3ad6e",
            "SimulationProcessGrids.supports": "47c6617af89cdcd6ba3a99aae345a8c32f8aec740f838bd0b37817236306ab79",
            "SimulationProcessGrids.array_roots": "ddc99b73e50d34ce46e2c362900c62af7a6954d15f96dec516f0776be32f6101",
            "SimulationProcessGrids.__call__": "65127cfecf2a7f45f02fa50d72a40f7e1977f1728f88c543689c5b43519bc921",
            "SimulationProcessGrids.seal": "71dde0f63544df96a4c57b99a75f9dd844dd904be7c599ad30a3125262756e9d",
            "SimulationProcessGrids.close": "1fbde590b371243da9503b714cc55b822e333e3e7a64f100ed57051209076a5a",
            "SimulationProcessGrids._produce": "a724e63f2d73bf06e0e0762cf47268e77868775f518e986c1ebc0d8c79b19eee",
            "SimulationProcessGrids.snapshot": "65d4d6df476276c3136bb02ba811eb5b8e1cfaddc248b6893375238f4800ca0b",
            "_uniform_parameters": "fc9a07f36c6a8882ac26609164df71175c8916a33484a464a4d5626e680b8851",
            "_parameter_bytes": "93e72d2f1aed3bfd4d0809cc8a5d47d0d1e7dc0f2664e926f53c1ece88cfaac5",
            "_abstract_grid_parameter": "8c1bd7f93b2aa50776102a6b2b2b733b80ddbd518f9d0b5eee47477bf3b8c8bf",
            "_compute_uniform_grid": "546e82063b47b05523ad23c64e681adaaea44dfda66071aa6b6f2ec94cde9d89",
            "SimulationProcessGrids._produce_normal": "5776e4f59b1eef6dbbcbacdb3d873321595222c3ab0d3939560c28d0eb1d6aed",
            "_normal_parameters": "893f4e4e80b34a1529672043f66cfbcef9989fd9730265858791d4d12321da1b",
            "_normal_fixed_identity": "467654dc343d02c419bd202358aad1e0ebad76ca7a754b8f82c708ee9ba4be1f",
            "_compute_normal_stage": "e49e8d6ed2760a94e39877a1717a6567220467d098de8f1670033411016d003f",
            "SimulationProcessGrids._produce_staged": "24d2c9c23ae74153fa9a63895f871104c8395c78c279a5354b1a70cb72ddce16",
            "_complete_process_parameters": "b11dea89c5a1b5ea44b482574f74ae6cb50eae5072293fb5116ddc1cab88bead",
            "_process_fixed_identity": "e2e8cd965379829662477a34f9fdc0eea48e7b83b8ffd69e24a30b64791c4e98",
            "_staged_parameter_is_weak": "408c2ee847275e1c74b09648ec40d930051387d6c0d8d341fde63c359086f567",
            "_trace_process_jaxpr": "af42b98b18382ee4e88a3031fcb39e6111e11ced6d0581d695ab28c0bd0360a7",
            "_process_grid_call": "d3c500914631c641753e0ee5cffc3d16a8ce2e8bbe720e76592d2067d2ac2000",
            "_validated_process_recipe": "de1e36e3f7b21ad614bc983eb3d2363988ca6f3ddae33fbe8306b64a71d89403",
            "_validate_attached_process_value": "316daad5ff10b7c79bc2ca6f314081e34c436b0cd9e3bdd8d91f8ea9ada9e3bb",
            "_validated_process_operand": "e12ce14e567bff2463294a163bac0e9510cf893e9142d76c34f641d61563ffdf",
            "_validate_process_equation": "6632117d9b6371442596b320bb2f0d72783044ca54b21fd736a1151462d90ac0",
            "_validate_linspace_jaxpr": "5e510b9af81fe99259e89f8371166fe7bd15837d5ffdc0c955068e09f36f8707",
            "_jaxpr_schema": "410b7fea2cb4cd88259c95f5a7952a6c8b6d4d51671ba8bc92c1579b87e52f3a",
            "_root_equation_schema": "8826967c6bb1df44ce8b9676285c2c79f05627f874a1e836f373b67e27db8420",
            "_equation_schema": "a0a7e7d1756b078e0f0a3b79fd0170b640042cf49ac5b356f4a4e6c01be0f2a5",
            "_graph_atom_schema": "6a156b062068ba3cb8aa26ce7145c7ac9596b3de15d4ad7cc7ce72387a904b23",
            "_aval_schema": "574ff0f8f86542bb23432f704bcb8f17b9aaea0e607780dc9830a2a9c4918690",
            "_aval_shape": "6007e8e38f5755ca0fa209d70d891cf3d0f73eccc7a1c72ad00d481175836b51",
            "_aval_dtype": "e360ab7aeae392e4af78f0ac779bad7d74e60a09c1549a1eb2d501022500e97e",
            "_read_process_operand": "f896cff2fbcd2a6b63ecb7174bfe806a1707882c2b4bb66e1e1f38d0efb419f9",
            "_compute_process_stage": "0cc67600ad383da92e58943450ba05e420d772d661d094451f24391fcb6f714e",
        },
    ),
    DISPATCHERS_SOURCE: (
        "9fc7b45ea5a1571398a79a7a455728b6742f2482787ca2160b0b5283c724bc85",
        {
            "tiled_productmap": "f9abfc940c4ade124483a286bdb5ab6eec4aac4c928680b2f685b859e673d104",
            "_CountBroadcastExtentInWidth.__call__": "9e1fbe5f50bdce7450a8b195401b036222290ff7b5c25bb10b7282e0e387c442",
            "_TiledProductMap.__call__": "4c2a22f6e7e41b0a85dfb680f9de845201e2d6fe5817741879ab409e458ea703",
            "_map_grouped_product": "e100fb75e4b1b818e22412a05fb44507621708e7edce49a9fba8a022ed68a775",
            "_MapOverFinalCoordinate.__call__": "667a360f1998f57268721fadb1febabb6b0d3501eb346df615eff2310fc86624",
            "_map_whole_product": "5226cfb3d079656599eb391c367423d867aa05350d6eebdcd1cc0da6cf734c79",
            "_final_mapper": "a7115ac1eda02cc0ef40fdd176ab1c7e4906e0e6cbbbcb01540510e2723d838d",
            "_MapWholeCoordinate.__call__": "68497277498dca5d11c4a3f4a86ad56d1e0d27a1c9bba1f9ebf5a872a7a3b7b2",
            "_EvaluateTiledCell.__call__": "2ea2996deffd1ca2d1747f2de0ab896be4f87ca79dad4b32017f4228d5ced8c5",
            "_restore_product_axes": "aefd91d6e6d5c1d49f1451435c80b04c616fc010784907adcf8e377508f087c2",
            "map_over_leading_axis": "b1d033acd692898271f38d8f2954c27d00b1b1b2198ea2dca48248466387c098",
            "_RestoreProductAxisOrder.__call__": "d82cdfbdca975232156d9d1e9b80d3892f811dcfb0b793a55257b50e3848c43b",
            "_transpose_product_axes": "54961f0159833bf5de6053053200dda0e3745caae17447c2aafd2e6f4faadae3",
        },
    ),
    ACTION_GRID_SOURCE: (
        "af26fe7c38efa57a95e4a879a2173a185f23edf23160684221ca636437f9d333",
        {
            "PreflightActionGrids.resolve": "a69e747af476459cef7fb72dfe542a80ad43545081a3e4a6a700dae8638a53f6",
            "PreflightActionGrids.close": "9551fdaca558706aeb32a670e27daf66e9e1ae0fddce3242be2a02b67c6b7ee9",
        },
    ),
    GRID_SEARCH_SOURCE: (
        "23dadb5101649abb1251ceca79d58b217161a2d5611cfd0d4f1df9804e146c79",
        {
            "GridSearch.build_period_kernels": "19bec5035c11fc9d3657e1a780169754c2e3cb1164f2d12804ce13e598dc986b",
            "_action_partition_mesh": "cab2381dbe9650e10620db20b6614ccb015eb147ff3356ab66e90cec58bf06fe",
            "_classify_action_streaming": "09d190475ffaf8c269880b7062a4be39e149f27d801e5fb640fa171753337ebf",
            "_select_action_width_keyword": "b45663df866d5a48c05b8955b6cdc68515697e8fa925566ae72afd06b3850104",
            "_select_cell_width_keyword": "f686d6cc7ae0d93dd1e3c301600872996943c7e3d6788c9d5098d39449793727",
            "_select_width_keyword": "00cd19cec6e137d7d9e044bc1625793b1d6f78bbdfc93d6858bb6f8e9d3c022f",
            "_edge_reference_regimes_for_targets": "9bd8c8de92411abd731020f62cb3c44ef814d4f3b64ab496b6299fcb925473e0",
            "_supports_action_streaming": "d93f977fad68ad528beb9d4b9e6d45e5eb95b53c9a0398ff6f6a62ec548bad11",
            "_value_reads": "9712ae402debbd0c37a12999b224e365c0ca1e1a8bcbb6e7bbb4043cbcfcacfc",
            "_value_read": "082a372c7e48bf7a32d390079e2dd60b9ed5aa56868cbf9fbccf0b8f924b3bb6",
            "_GridSearchArgumentBuilder.__call__": "cf6bf11deb8568dfbe00ae52b5431e85e842334218f05868cb597159b0b65539",
            "_GridSearchArgumentBuilder._with_edge_substitution": "8d253b526755274685f3dbf2e28efe4f35062993bedd017cffddf2d17fe657cc",
            "_GridSearchArgumentBuilder._edge_reference_args": "a8106a2808be98ae6e35f716b16fd7054c8da9180454bdd1fa8abfcbe6629d76",
            "_GridSearchArgumentBuilder._same_period_params": "52279dcdea37c1ea9ad066794696d3f77a368539faaf12e3417fcfb439be7303",
            "_GridSearchPeriodKernel.__post_init__": "1e8c230f0845d9b8335c667f38593221a957c34955efe44921763bce6cd24ac7",
            "_GridSearchPeriodKernel.core_programs": "0d96f7bea814e419ef1dbdebbc3257d63c3c0d36d9e6e52e619f8f44ae5a8a56",
            "_GridSearchPeriodKernel.with_fixed_params": "5f5a26e02b136c760bd242cd91c22b76153adc9828a13d20de7a6a98f5877e14",
            "_GridSearchPeriodKernel.__call__": "d5371441759416db418198b6b0164a0718d825e64c378696123eb09c12553a2a",
        },
    ),
    MAX_Q_SOURCE: (
        "585ece85c1cd50212e38344fd35d7aec7ec10a03bac7bfb87ca3ba69aea7e928",
        {
            "get_action_partitioned_max_Q_over_a": "9443177495a4aa17b12aed6bda675dace04c1422b7d37a96e6fc76df8e46c30e",
            "_ActionPartitionedMaxQOverA.__call__": "908cb48cd85a7987cbbf3c4c92a6929b5da1e2dd7f1c70900e4934c8b3ad3c10",
            "_arguments_named": "95c0088a6f36474a87a6dd5d4f0a72745787638106bcbb911a0f5aad35101365",
            "_OnActionPartitionAxis.__call__": "00f65bb9b858b815e7d5fc96fb308317a567e789ccb70cdbeea28a6e067ceab2",
            "_call_with_operands": "65f621e941c69e4f0023b040374993174c013fe747fc1e8fca726bd0b82205f8",
            "_get_extra_param_names": "ccb1bc531a850fb9475d70e0a06b5d0bf53888e4b7b447e9c06f9e8b8333e958",
            "_fail_if_action_width_keyword_collides": "d72946426a1e4a1d300fa4812f07eea8139236693ef8f79d0e8384b9d59a9f0d",
            "get_streaming_max_Q_over_a": "f6049aafdb1361ab3dc07245c3e201f35719fe199c7278fb748c7658ac5f8a63",
            "_fail_if_full_V_streaming_route_is_unsupported": "cd4c96d572ec7df9dc269f5fa2bfc1ec5c16fe0a78de3be56adc28c15f065d2c",
            "_fail_if_streaming_co_map_layout_is_invalid": "59c06aedafc8bcbe31d7f2f7f7b7d94e1d8044bf529c6f11a05882c5bf1d7979",
            "_wrap_with_fold_reduction": "20dee195475a290e229948d79aa1b0b65a0c20a59f0412468637039809b32f07",
            "_StreamedMaxQOverA.__call__": "825b72e2a3efd69ea6ae04ebb9ca712ba11709fae41162cab5d8a084d88b2053",
        },
    ),
    ACTION_STREAMING_SOURCE: (
        "af44e774aa5f951ce49affc0f20631c48da6fcaaeeee1763529638d2317dc15c",
        {
            "build_partitioned_streaming_max_Q_over_a": "2b1ab2a3075916b37d825af02061dd985063823e47aa8f57589a47c9dba86548",
            "merge_partition_accumulators": "490ff581c56b3c85bfbfc7d3824baed55adc1aa035970e5d15c6b433c6380098",
            "_fail_if_not_positive_int": "23eecceb2c4c24b75e70863f2a5f531c5a52e569400c090c382d5ba667cf6f9f",
            "ActionPartitionLayout.__post_init__": "9957631cd979a1ec48136310cb5e3cfb246b1c7367331544c68447b0b092539b",
            "ActionPartitionLayout.n_blocks": "269793628a54b8cd15b276fa360f884a97483ffbeb27bff2b68d4ba94ae055fe",
            "ActionPartitionLayout.blocks_per_partition": "b1109e10050ec6f186486a275a0467089c2c5101528b675c928ba0fd85d8b9b6",
            "ActionPartitionLayout.block_range": "c306308ded5a8642519248ce4bd7f400c8ea13b25b06a3fe2c01c24c90f03845",
            "ActionPartitionLayout.action_interval": "dbaa877fae3466c4ea5af7c2657fe18be813c4487ed3a26e6cbb06b1fac17aba",
            "_PartitionedStreamingHardMax.__call__": "382aeb28248e0392b3cb6455d434251a43bdc911b818b60c5defacce354abb20",
            "_PartitionedStreamingHardMax.local": "d22f025b6500a51c5337c53d870b88c80f82c9a6ed500e89b6650c1339dc3d5b",
            "_evaluate_owned_block": "f95eb5df6591891562f8d2779dd24a879e5f6fd931f146888b46393b7f4e6366",
            "_validate_streaming_configuration": "4f9f662fa73508293542ddd9e6ea428ac22f31f45376b80290f0e46b64a3f9b4",
            "_prepare_action_call": "27a8f85f8f16c7214846609213fdf9135bfb7118395aecdcf68ace17bea4e39c",
            "_evaluate_block": "5e6e90502c17da3b358b0c897dfa908f926836e89d2c26cf4ae074ca83c5e3af",
            "_evaluate_one_action": "0375e2b11924435daf1c8076ad5bc9369504f3b92665380d2bb92f74664be4c5",
            "_decode_action": "cd4d0f5f01fe2a5648faab36deda62d09452116e3e7cf1e691fe6214b8a692c5",
            "_validate_block_Q_and_F": "7f00abbccfe23768df403596eb22c715eb3542b4e43dd67593f7bd3e487fdeef",
            "_trace_block": "f7bfdbd047a5c51e80e99b7b936e063e772239e6156cde30989224841babb35e",
            "_empty_reduction": "8135c3d0fb8b4ad0068e59b705e587ad88f330228d00e020f579dd4e36d9c884",
            "_typed_like": "fc7809b3ba41f45007a2464926696f92b1cf2f937c2933223c49ce4984d56b22",
            "_start_reduction": "034b3966dd04c0e0d66e085e8e2c4e16b127e1d8a9869ecfa039c3b5e7928b04",
            "_scan_one_block": "1901bdf24caccc5087081f15fc69e9138db76545ae7bb1794539d05adf5af7c9",
            "_add_block": "a5047bea80275b77727b69b06d563bcdfea7e80c0f99dda34f6570948ccd1a72",
            "build_streaming_max_Q_over_a": "7e56c978cd164f847bf654ca9a63d0d901f25b79e92ece51f2cab3a452364b8d",
            "build_streaming_collective_max_Q_over_a": "8e4657ee33f24a6b271c35318b4dc16d04f46295f1882d81cdb4bc34ae1e2dfe",
            "build_streaming_ev1_max_Q_over_a": "22fbf958cab6b62ce9f8b9ad0014b18efc6664e978188366de187482e8bea73b",
            "GridSearchEV1ActionReduction.semantic_key": "d1760804377fac087e3151a6dcff3c5293e984eb79b4f3f76cfc46caafdda923",
            "_StreamingHardMax.__call__": "d0f0635deab0c3ee0bfde7d23d0a9a1062d896e116bc0250dd1bbda1f3eaea82",
            "_StreamingCollectiveHardMax.__call__": "bbeed1f3780a938705cc0d076e53e0774063e6c3705885cd4ecff99750b8c6d4",
            "_StreamingEV1ExpectedMax.__call__": "875de4ae1657fae190018c985512475fd6447677231086bdec2b1c0cbed7a164",
            "_evaluate_ev1_branch_block": "301c694809919e21aa8afcfcafa879383208870fbb45ca35970b3095a34391eb",
            "_evaluate_collective_block": "749560fecb8e6255290a47b07e1cdb84f8c13db506caa11ea178b8ef3c4b3b57",
            "_scan_blocks": "90d220fa12e133d9921aaad62053012fc47ab1917b16a623db2abea8a2ecc62f",
            "_reduce_no_action": "27a09e4e36247d95db9c565aaefa311f3c1ded3c88889f0ed305cb825bae243e",
            "_start_collective_reduction": "019872b31d89c2a2c8ba68ee6128a9feb2b90b52c6ee3b00506708db6195adce",
            "_scan_collective_blocks": "2af8554d14cb82cab0ccd29f86fe325860add8700fd726b33bffe92a95b4aa7a",
            "_add_collective_block": "5ed017db741c30cae43dd1c4a526c703c8601857c2fbca23e9900ca560e78198",
            "_reduce_collective_no_action": "3e2b5db13d002cec08e1e26cc6cd2aedd373fb3119e14e91a57fd8b5c6cfa02d",
            "_validate_scalar_Q_and_F": "a2b49855d9fe1572f7440248db7edfce283a2179394648ca55b6b8ce550f5364",
            "_validate_collective_scalar_Q_and_F": "600332c08d2ed5aa6a4ffaeee07a878c8a713a674f54456c8a9d05dbe9eddd35",
            "_validate_collective_block_Q_and_F": "1e12615fa5fa04137cebb137387bc1c2ad98b31f1978dd47f209d00207036bef",
            "_initialize_ev1_reduction": "41728c9433880bdb06c0ab5d3c0821a7f100f238fa1faec90dc5ca967c627650",
            "_add_ev1_block": "5c63a9ca306c889a05581aaab90a9da4a43082772a67a7098603818fc5dbb283",
            "_finalize_open_ev1_branch_group": "f1eea36aa27ec48e7e79557c8928ed62d5e6cf01d0cc100a8e558ebdb31196fd",
            "_scan_ev1_blocks": "e7e3248f6d4c9e175e76f3099655e7a1fb830c83803d49b89a912d8f53ea6f3f",
            "_flush_ev1_branch_group": "cb87cd4217df5fb3e45806593d2013a2ba452f17cbb08097a9d0cacd1e6f9270",
        },
    ),
    ACTION_REDUCTION_SOURCE: (
        "bfb8dc25b634d82451f43454df2465d9c523e69fa4cda2db3fa863802c6c7068",
        {
            "HardMaxReduction.initialize": "b29e84926276a74848f11826cb36ca2442e00cbc3ab3819bd197bfad624bc671",
            "HardMaxReduction.add": "5264b88c3ba353f158b394889295be544309038425796dc8f68859ff977c3880",
            "HardMaxReduction.merge": "de104bfa46bf5dff388f43bd1c4c696a4f1527613a2efcb359a762b513f28e2b",
            "HardMaxReduction.finalize": "40a21bb4b44366d00ec79a56e7aa7594a7b7b5427e3c29d9910cbc9a1e69bed3",
            "_reduce_block": "523679254fefd2b0e2b80c1dfc86a47841fd6d0eca64676ddcb5b267c5f8f165",
            "HardMaxReduction.semantic_key": "f024d59aadbce68d4647522cd802f542ed3a39c7cbc05664b03c5a362c6468bd",
        },
    ),
    FUNCTOOLS_SOURCE: (
        None,
        {
            "_split_bound_arguments": "9e970a7c5931df8b6b08bbc36ddf5e36715517ef273cc443b7e7c888e270ce1e",
            "allow_args": "aafbd21439f91d8e29c52097e4b0711469942a9ca6bafce071278ef69b3d78fe",
        },
    ),
    INTERNAL_OUTPUTS_SOURCE: (
        None,
        {
            "resolve_producer": "13dc702f647a9b6f5e51a3ba6d55c8d852b68342c16487132fefee19a729ea65",
            "assert_width_invariant_internal_outputs": "cf54d009a3abce9124ea579209b720ab06f7fe9bff641753f60753cc2ab1e975",
            "consumed_producer_names": "6c7019c97744a8bd73f34344a6dc45f195972e393dfeff1d1a206f0dce286ee2",
            "internal_input_templates": "6919cb41a458780ef41bc5bf08f5cad5f8097137461b73936dced1f0487699ce",
        },
    ),
    COLLECTIVE_ACTION_REDUCTION_SOURCE: (
        None,
        {
            "CollectiveHardMaxReduction.semantic_key": "1a2875f2b718377e51a76e0d37f1614f40f3e08b4cc730032add7a2373c5734b",
            "CollectiveHardMaxReduction.initialize": "78e55824232f334491375fa641fb20e28bdc7be3dc2cd59cceedbe1b1e74db03",
            "CollectiveHardMaxReduction.add": "8d69ebff49fe1f231e7941129ea582137430dfeb5c2719061676691e80aee747",
            "CollectiveHardMaxReduction.merge": "4e288cd957f4840ebc2f5c185051a208c8e82d7df59d8d64dda3f9e5a42f530c",
            "CollectiveHardMaxReduction.finalize": "2c2128c3095d373853e0bfc2bf9f8519d8782c58c9170fd79a5cc96358d6ee47",
            "_validate_block_shapes": "ef3ba0ed14e345bd21da5ab0ac1e79824b04317f8817fce58f8ecd07a8a1b8a5",
            "_reduce_block": "e84eb817b412c88fadd97a9f36db026c0fb89781bbde1d4d75033a7463d60f3e",
            "_take_stakeholder_values": "b84709a267bb886bef97f01076e40d5670e30caa1c4ffeede8d402008848072d",
        },
    ),
    LOGSUMEXP_ACTION_REDUCTION_SOURCE: (
        None,
        {
            "BoundLogSumExpReduction.initialize": "a85a8161da058019e24d2b33ce72d9881e4d8df843bcb57e2ea1143c7ad37d36",
            "BoundLogSumExpReduction.add": "bc07d13e5fa3101df3216f75942966086537395d81bb2c6abc4276158ccf9a4d",
            "BoundLogSumExpReduction.merge": "e1b12504e631c9659f1de5f55a48de26e3d4a1d5c089e6bc75738a52313fffc7",
            "BoundLogSumExpReduction.finalize": "a5d920589ee7f6b7454e241c9ef5b0be41c11b75c806e578932d1244884ce5cb",
            "LogSumExpReduction.semantic_key": "13bc88fd7862f49c2ef01b2de88e9c695276a163bc896d9c40dae5d4b104c671",
            "LogSumExpReduction.bind": "1e3f07eb92d208799636c7958d2d2fdd8c955863630606ffa1a1fbba7c8de06a",
        },
    ),
    PROCESSING_SOURCE: (
        None,
        {
            "_build_per_subject_decisions_per_period": "3b2dfd37e32f72e41f17264ec806fcb9309bf9f6d2faed91fafd1477871cea83",
            "process_regimes": "52cb3ab325c856b7b737d1aa359709943502b2af6cd62d89d977d4ad7883ac6b",
            "_TerminalCarryPeriodKernel.core_programs": "842c31af0bea766bfd410783881770152246a825505be327096c117f60ee65fa",
            "_TerminalCarryPeriodKernel.with_fixed_params": "4ee13dc7cbc4ebaa68102cc6eea4272791590fa1005080dbbbcf898e5f92a8f6",
            "_TerminalCarryPeriodKernel.__call__": "cdb11e9a08f14bc6d08289d277ca4d33d897553e16fa60eb7ff6ef36260a75a1",
        },
    ),
    SIMULATION_PROGRAMS_SOURCE: (
        "b0a12526d3de2c1a6313649cc1e0406a461811637d3cd5652ddab4d3b7efe180",
        {
            "build_simulation_programs": "5abc9f5eaf0211642d45521ec3232e21fffa5a79bde6fac36b4d25bfbca06ca0",
            "_decision_programs": "bd9e88a1511edb65636d9d154c56a64e78a33b094321803cef6355b4c6bec629",
            "attach_gated_simulation_programs": "e92be1be5b6628706b2a2c7cdfb749b3936ade622d352161fb72d9d949a59cfa",
            "gated_simulation_programs_ready": "fe5419d6b84ba1929acf26cde8dc2feedf9589a2345925a1ae6de25fde674def",
            "budgeted_simulation_programs_ready": "0cb813fd106dee1a74a7a04138ce35ed40fafbefc5731cdc618535313fc3f87f",
            "_GateFoldBody.__call__": "b24892f48c809e6aa25853831c72319e979078ae0dc0aa08a389755b2e2412e1",
            "_GateRouteBody.__call__": "bed23e58107f4a0b676644c86e50fc7fd545f1cb3ed8387126e735189740f73b",
            "_fail_if_the_streamed_reduction_is_wrong": "4393d8f3dc1dded01122dc6bf97e4f51c88e52f3804f79b08d600bc75496dd0a",
            "_decision_subject_arg_names": "3bf80bc49a6326e6be6369de80863482db1615e713b95ce476966e9562d6fde5",
            "_decision_value_reads": "7480309f78099994ba417dab87ff44c8ccc545f662855a44bc142bd3509cc1a9",
            "_decision_body": "b888d12d325f80f67e2e7681ed2938b53a0625b3925441e791e8ea5a32450ce4",
            "_StreamedArgmaxQOverA.__call__": "8ec8f0298dc6af4e9807195b1011dd9b1f99945e9c16990e167e1b7fcf86af45",
            "_StreamedArgmaxQOverA._fold": "a817cf3a8b385691bc4b98405197fa89fb5737d47cc61157687b847a83737ea1",
            "_SubjectTiled.__call__": "f62934dc58bb872bb7f39136966c083d5a53a56aea86ac606ea5eaa972ce2a04",
            "_evaluate_subject_tile": "da5ea37bc7faea1405b83b856ca8701e178e0daf0a26223eb70146589d6d26de",
            "_ArgumentsBoundAtDispatch.__call__": "46ee403fdfc0849f77b0727c94064168ad21c00f5072d34d15247e939594babd",
        },
    ),
    SIMULATION_PROGRAM_TYPES_SOURCE: (
        "1d82874db89f5b62480f55a9c141200b3ec2c6b43570f24e815a07491914e483",
        {
            "SimulationBuildContext.__post_init__": "00641d48094283340c57d2137f90f4568bd7dfa8ff6372e0348ccf6abad54a31",
            "SimulationProgramExecutor.dispatch": "cc42079ac6d204e3b4e22b058c24d7db4ac41c8269fd276a5ad224cb5bd2a7c7",
            "SimulationPrograms.__post_init__": "a7157a954fed1194c80116b6bb3dc746cf3da0cd6c0d1c0ee3284f51281f7f49",
            "SimulationPrograms.declared_axis_names": "1451b9d993c5de21035c9a25f9aa19ba953d2e8dfebb5f952bda617d2fa69b81",
            "SimulationPrograms.forward_decision": "369b61edb30600fcabc910d0d6a333a939519fcda229f315d91101b96cb4de3f",
            "transition_output_roles": "ed772c2beff03f47113b71f5d3f0405469b6ef4bf67d4c3ad4df1a3f064e3fab",
            "route_output_roles": "81ccd2cdf1a29d1dcb3021d775c4abc8cb70364819a027303968afc69af19a2e",
            "subject_axis": "2d1dda5c95debf5b8c7d0a72c8fa71db6c42b2fb22763cc943ec494dfd3d942f",
        },
    ),
    NBEGM_SOURCE: (
        "4424fb7c2fde34d86bd09c18b1f70e065cda2e080723ba20c965b9c721191784",
        {
            "NBEGM.declare_continuation_reads": "80df8b31258d82ba39fc2429030f30f1377f8875cf6e0704ce99e255dbb7adc2",
            "_with_ride_marginal_reads": "3a8b8d0642f60b4af9db3784e2b38394dca5785c9a69358605cf928721442747",
        },
    ),
    CONTINUATION_ARGUMENTS_SOURCE: (
        "1f622b85eec9efc12e42de55288c75ed71481cd8c5339903fe56b9e6252f15d1",
        {
            "MarginalLeafArguments.__call__": "e9887274c66d0f3ae9491e7f10524aae4c0cb3a0536dbeff5dc17ab8abbaef27",
            "MarginalLeafCore.__call__": "a27503e19218f862da83e64aff785cab8aa99e8da020d6e708d32edc31bb7741",
            "marginal_leaf_reads": "02ce2895a2ee099f1df276087ebc98d4e88c4bebc0fe841e4af2bab72a0af9e1",
        },
    ),
    PERIOD_REPLAY_SOURCE: (
        None,
        {
            "replay_period": "b7de416a252dccf06fc5390f7739a17298dcbf3ec372f3c61fee4087afe6f23d",
            "_compile_cores_for_one_period": "b4e2f901108d74b0e9070889f288a8b84652e56cad458463ce3d2859a14dfb18",
            "_core_build_context_for_one_period": "99623b13b7115d3e2bae90addfce95ea9fa10ef237529287a2c6cd3bab1bd8c0",
        },
    ),
}

_SELECTED_PINS: set[tuple[str, str]] = set()


def _surface_pin(source: str) -> str:
    """Select one certified source's pinned module transport surface."""
    surface = _CORRIDOR_PINS[source][0]
    if surface is None:
        raise ValueError(f"no module transport surface is pinned for {source}")
    _SELECTED_PINS.add((source, source))
    return surface


def _callable_pins(*, source: str, names: tuple[str, ...]) -> dict[str, str]:
    """Select pinned callable digests of one source, in the order named."""
    pinned = _CORRIDOR_PINS[source][1]
    missing = [name for name in names if name not in pinned]
    if missing:
        raise ValueError(f"no callable pin is stored for {source}: {missing}")
    _SELECTED_PINS.update((source, name) for name in names)
    return {name: pinned[name] for name in names}


def _contracts(
    selection: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, dict[str, str]]]:
    """Select one family's module surfaces and callable pins from the store."""
    return {
        source: (_surface_pin(source), _callable_pins(source=source, names=names))
        for source, names in selection.items()
    }


def _unselected_pins() -> tuple[tuple[str, str], ...]:
    """Return each stored `(source, name)` pin that no certificate family selects.

    A module surface is named by its source path.
    """
    stored = {
        (source, source)
        for source, (surface, _) in _CORRIDOR_PINS.items()
        if surface is not None
    } | {
        (source, name)
        for source, (_, callables) in _CORRIDOR_PINS.items()
        for name in callables
    }
    return tuple(sorted(stored - _SELECTED_PINS))


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
            f"expected one direct has_taste_shocks guard in {outer_name!r}, found {len(guards)}"
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
        node = _definition(tree=tree, name="create_regime_state_action_space")
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
        node = _definition(tree=tree, name="_simulate_regime_in_period")
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
        broadcast_variables=broadcast_state_names,
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
                "broadcast_state_names",
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
                "build_partitioned_streaming_max_Q_over_a",
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
                "from typing import ClassVar, cast",
                "import jax",
                "import jax.numpy as jnp",
                "from dags import with_signature",
                "from _lcm.logsum import EULER_GAMMA, logsum_and_softmax",
                "from _lcm.regime_building.argmax import argmax_and_max",
                (
                    "from _lcm.regime_building.collective import ParetoWeights, collective_argmax_and_readout, collective_readout"
                ),
                (
                    "from _lcm.solution.action_streaming import build_partitioned_streaming_max_Q_over_a, build_streaming_collective_max_Q_over_a, build_streaming_ev1_max_Q_over_a, build_streaming_max_Q_over_a"
                ),
                "from _lcm.utils.dispatchers import productmap, tiled_productmap, vmap_1d",
                "from _lcm.utils.functools import allow_args, allow_only_kwargs",
                "from lcm.typing import BoolND, FloatND, IntND, ReferenceName, ScalarFloat",
            ],
            expected_binding_counts={
                "MappingProxyType": 1,
                "ParetoWeights": 1,
                "allow_args": 1,
                "allow_only_kwargs": 1,
                "argmax_and_max": 1,
                "build_partitioned_streaming_max_Q_over_a": 1,
                "build_streaming_collective_max_Q_over_a": 1,
                "build_streaming_ev1_max_Q_over_a": 1,
                "build_streaming_max_Q_over_a": 1,
                "Any": 0,
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


_STREAMED_MAX_BUILDER_PINS = _callable_pins(
    source=MAX_Q_SOURCE,
    names=(
        "get_streaming_max_Q_over_a",
        "_fail_if_action_width_keyword_collides",
        "_fail_if_full_V_streaming_route_is_unsupported",
        "_fail_if_streaming_co_map_layout_is_invalid",
        "_wrap_with_fold_reduction",
        "_StreamedMaxQOverA.__call__",
    ),
)


def _streamed_max_builder_errors(tree: ast.Module) -> list[str]:
    """Pin streamed VALUE production, optional folding, and fail-closed boundaries."""
    return _exact_callable_errors(
        tree=tree,
        label="streamed max-Q builder",
        contracts=_STREAMED_MAX_BUILDER_PINS,
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
                "q_and_f_arg_names: frozenset[ReferenceName]",
                "action_width_keyword: str",
                "whole_product_Q_and_F: Callable[..., tuple[FloatND, BoolND]]",
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


_FUNCTOOLS_ADAPTER_PINS = _callable_pins(
    source=FUNCTOOLS_SOURCE,
    names=("_split_bound_arguments", "allow_args"),
)


def _functools_adapter_errors(tree: ast.Module) -> list[str]:
    """Pin positional-origin preservation through nested co-map adapters."""
    return _exact_callable_errors(
        tree=tree,
        label="allow-args positional transport",
        contracts=_FUNCTOOLS_ADAPTER_PINS,
    )


_CORE_PROGRAM_TRANSPORT_PINS = _callable_pins(
    source=CORE_PROGRAM_SOURCE,
    names=(
        "ValueRead.__post_init__",
        "ReducedAxis.__post_init__",
        "ReducedAxis.extent",
        "TiledOutputAxis.__post_init__",
        "CoreExecutionRequirements.__post_init__",
        "CoreBuildContext.__post_init__",
        "CoreProgram.__post_init__",
        "MaterializedCoreProgram.__post_init__",
        "ResolvedCoreProgram.__post_init__",
        "core_program_graph",
        "_reject_native_duplicate_authorities",
        "_snapshot_and_validate_graph",
        "_validate_replay_replacements",
        "_validate_program_declaration",
        "_validate_retention_declaration",
        "_validate_retained_artifact_payload_types",
        "_validate_disposition_reason",
        "materialize_core_program",
        "resolve_core_program",
        "resolve_core_program_candidates",
        "_resolve_core_program",
        "select_programs",
        "_validate_core_program",
        "_validate_materialized_declaration",
        "_resolve_input_transfer_plan",
        "_validate_value_reads",
        "_value_read_argument_leaf",
        "_validate_transfer_argument_metadata",
        "_validate_reduced_axis",
        "_validate_tile_width",
        "_validate_coordinate_argument",
        "_validate_width_keyword",
    ),
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
            "view: ValueViewDescriptor | None = None",
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
                "preferred_alignment: int = 1",
                "halve_on_materialised_gather: bool = False",
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
                "next_regime_to_V_arr: Mapping[RegimeName, FloatND | jax.ShapeDtypeStruct]",
                "next_regime_to_continuation: Mapping[RegimeName, object]",
                "flat_params: Mapping[str, object]",
                "period: int",
                "ages: object",
                "edge_regime_to_V_arr: Mapping[RegimeName, FloatND | jax.ShapeDtypeStruct] | None = None",
                "same_period_regime_to_V_arr: Mapping[RegimeName, FloatND | jax.ShapeDtypeStruct] | None = None",
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
                "function: Callable[..., PytreeValue]",
                "argument_builder: CoreArgumentBuilder",
                "requirements: CoreExecutionRequirements",
                "output_roles: OutputRoleTree",
                "disposition: CoreExecutionDisposition",
                "disposition_reason: str | None = None",
                "donation_candidates: tuple[str, ...] = ()",
                "scope: ProgramScope = ProgramScope.ANY",
                "retained_artifact_keys: _RetainedArtifactKeys = ()",
                "retained_artifact_payload_types: _RetainedArtifactPayloadTypes = MappingProxyType({})",
                "replaces_program: str | None = None",
                "internal_outputs: tuple[InternalOutputSpec, ...] = ()",
                "compiler_options: tuple[tuple[str, int], ...] = ()",
                "invariant_binding: InvariantBinding | None = None",
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
                "function: Callable[..., PytreeValue]",
                "arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]",
                "requirements: CoreExecutionRequirements",
                "output_roles: OutputRoleTree",
                "disposition: CoreExecutionDisposition",
                "donation_candidates: tuple[str, ...]",
                "disposition_reason: str | None = None",
                "scope: ProgramScope = ProgramScope.ANY",
                "retained_artifact_keys: tuple[ArtifactKey, ...] = ()",
                "replaces_program: str | None = None",
                "internal_outputs: tuple[InternalOutputSpec, ...] = ()",
                "compiler_options: tuple[tuple[str, int], ...] = ()",
                "invariant_binding: InvariantBinding | None = None",
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
                "function: Callable[..., PytreeValue]",
                "arguments: Mapping[str, object]",
                "static_kwargs: Mapping[str, int]",
                "requirements: CoreExecutionRequirements",
                "output_roles: OutputRoleTree",
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
                "invariant_binding: InvariantBinding | None = None",
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
            contracts=_CORE_PROGRAM_TRANSPORT_PINS,
        )
    )
    return errors


_INTERNAL_OUTPUTS_TRANSPORT_PINS = _callable_pins(
    source=INTERNAL_OUTPUTS_SOURCE,
    names=(
        "resolve_producer",
        "assert_width_invariant_internal_outputs",
        "consumed_producer_names",
        "internal_input_templates",
    ),
)


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
        contracts=_INTERNAL_OUTPUTS_TRANSPORT_PINS,
    )


_ACTION_STREAMING_PINS = _callable_pins(
    source=ACTION_STREAMING_SOURCE,
    names=(
        "build_streaming_max_Q_over_a",
        "build_streaming_collective_max_Q_over_a",
        "build_streaming_ev1_max_Q_over_a",
        "GridSearchEV1ActionReduction.semantic_key",
        "_StreamingHardMax.__call__",
        "_StreamingCollectiveHardMax.__call__",
        "_StreamingEV1ExpectedMax.__call__",
        "_prepare_action_call",
        "_evaluate_block",
        "_evaluate_ev1_branch_block",
        "_evaluate_collective_block",
        "_trace_block",
        "_empty_reduction",
        "_typed_like",
        "_start_reduction",
        "_scan_blocks",
        "_add_block",
        "_reduce_no_action",
        "_start_collective_reduction",
        "_scan_collective_blocks",
        "_add_collective_block",
        "_reduce_collective_no_action",
        "_decode_action",
        "_validate_scalar_Q_and_F",
        "_validate_block_Q_and_F",
        "_validate_collective_scalar_Q_and_F",
        "_validate_collective_block_Q_and_F",
        "_initialize_ev1_reduction",
        "_add_ev1_block",
        "_finalize_open_ev1_branch_group",
        "_scan_ev1_blocks",
        "_flush_ev1_branch_group",
    ),
)


def _action_streaming_errors(tree: ast.Module) -> list[str]:
    """Pin complete C-order block evaluation and exact reducer delegation."""
    errors = _class_surface_errors(
        tree=tree,
        label="streamed action evaluator",
        class_name="_StreamingHardMax",
        fields=(
            "Q_and_F: Callable[..., tuple[FloatND, BoolND]]",
            "action_names: tuple[ActionName, ...]",
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
                "Q_and_F: Callable[..., tuple[FloatND, BoolND]]",
                "action_names: tuple[ActionName, ...]",
                "block_width: int",
                "stakeholders: tuple[str, ...]",
                "weights: Mapping[str, FloatND | float]",
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
                "Q_and_F: Callable[..., tuple[FloatND, BoolND]]",
                "action_names: tuple[ActionName, ...]",
                "n_discrete_action_axes: int",
                "block_width: int",
                "scale: FloatND | float",
            ),
            methods=("__call__",),
            decorators=("dataclass(frozen=True)",),
        )
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="streamed action evaluator",
            contracts=_ACTION_STREAMING_PINS,
        )
    )
    return errors


_HARD_MAX_STREAMING_REDUCTION_PINS = _callable_pins(
    source=ACTION_REDUCTION_SOURCE,
    names=(
        "HardMaxReduction.semantic_key",
        "HardMaxReduction.initialize",
        "HardMaxReduction.add",
        "HardMaxReduction.merge",
        "HardMaxReduction.finalize",
        "_reduce_block",
    ),
)


def _hard_max_streaming_reduction_errors(tree: ast.Module) -> list[str]:
    """Pin the complete hard-max accumulator and global-identity merge law."""
    return _exact_callable_errors(
        tree=tree,
        label="streamed singleton hard-max reduction",
        contracts=_HARD_MAX_STREAMING_REDUCTION_PINS,
    )


_COLLECTIVE_HARD_MAX_STREAMING_REDUCTION_PINS = _callable_pins(
    source=COLLECTIVE_ACTION_REDUCTION_SOURCE,
    names=(
        "CollectiveHardMaxReduction.semantic_key",
        "CollectiveHardMaxReduction.initialize",
        "CollectiveHardMaxReduction.add",
        "CollectiveHardMaxReduction.merge",
        "CollectiveHardMaxReduction.finalize",
        "_validate_block_shapes",
        "_reduce_block",
        "_take_stakeholder_values",
    ),
)


def _collective_hard_max_streaming_reduction_errors(
    tree: ast.Module,
) -> list[str]:
    """Pin the shared-household winner and stakeholder-value gather law."""
    return _exact_callable_errors(
        tree=tree,
        label="streamed collective hard-max reduction",
        contracts=_COLLECTIVE_HARD_MAX_STREAMING_REDUCTION_PINS,
    )


_LOGSUMEXP_STREAMING_REDUCTION_PINS = _callable_pins(
    source=LOGSUMEXP_ACTION_REDUCTION_SOURCE,
    names=(
        "BoundLogSumExpReduction.initialize",
        "BoundLogSumExpReduction.add",
        "BoundLogSumExpReduction.merge",
        "BoundLogSumExpReduction.finalize",
        "LogSumExpReduction.semantic_key",
        "LogSumExpReduction.bind",
    ),
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
            contracts=_LOGSUMEXP_STREAMING_REDUCTION_PINS,
        )
    )
    return errors


_GRID_SEARCH_STREAMED_PROVIDER_PINS = _callable_pins(
    source=GRID_SEARCH_SOURCE,
    names=(
        "_select_action_width_keyword",
        "_select_cell_width_keyword",
        "_select_width_keyword",
        "GridSearch.build_period_kernels",
        "_edge_reference_regimes_for_targets",
        "_classify_action_streaming",
        "_supports_action_streaming",
        "_value_reads",
        "_value_read",
    ),
)


_GRID_SEARCH_NATIVE_GRAPH_PINS = _callable_pins(
    source=GRID_SEARCH_SOURCE,
    names=(
        "_GridSearchArgumentBuilder.__call__",
        "_GridSearchArgumentBuilder._with_edge_substitution",
        "_GridSearchArgumentBuilder._edge_reference_args",
        "_GridSearchArgumentBuilder._same_period_params",
        "_GridSearchPeriodKernel.__post_init__",
        "_GridSearchPeriodKernel.core_programs",
        "_GridSearchPeriodKernel.with_fixed_params",
        "_GridSearchPeriodKernel.__call__",
    ),
)


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
            contracts=_GRID_SEARCH_STREAMED_PROVIDER_PINS,
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
            contracts=_GRID_SEARCH_NATIVE_GRAPH_PINS,
        )
    )
    return errors


_OUTPUT_LAYOUT_PINS = _callable_pins(
    source=OUTPUT_LAYOUT_SOURCE,
    names=(
        "resolve_output_layout",
        "_validate_output_roles",
        "assert_output_layout",
        "_assert_output_metadata",
        "PlannedCore.__post_init__",
        "PlannedCore.__call__",
        "assert_value_leaf_layout",
        "_assert_output_leaf",
        "_resolve_output_leaf",
        "_state_axes_leading_sharding",
        "_state_axis_spec",
        "StateAxesLeading.__post_init__",
    ),
)


def _output_layout_errors(tree: ast.Module) -> list[str]:
    """Pin planned lowering, validation, and identity-return publication."""
    errors: list[str] = []
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="output layout",
            class_name="ResolvedOutputLayout",
            fields=(
                "out_shardings: OutputShardingTree",
                "compilation_key: Hashable",
                "expected_value_shape: tuple[int, ...]",
                "expected_value_dtype: DTypeLike",
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
                "compiled: Callable[..., PytreeValue]",
                "layout: ResolvedOutputLayout",
                "tile_widths: Mapping[str, int]",
                "input_transfer_plan: tuple[ResolvedValueTransfer, ...] = ()",
                "internal_input_templates: Mapping[ReferenceName, ShapeDtypePytree] = MappingProxyType({})",
                "transfer_cache: TransferCache | None = None",
                "pending_work: PendingSolveWork | None = field(default=None, compare=False, repr=False)",
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
                "dtype: DTypeLike | None = None",
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
                "dtype: DTypeLike | None",
                "sharding: jax.sharding.Sharding",
            ),
            methods=(),
        )
    )
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="output layout",
            contracts=_OUTPUT_LAYOUT_PINS,
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


_VALUE_TRANSFER_PINS = _callable_pins(
    source=VALUE_TRANSFER_SOURCE,
    names=(
        "ValueArtifactAddress.__post_init__",
        "ValueConsumerAddress.__post_init__",
        "ResolvedValueTransfer.__post_init__",
        "resolve_value_transfer",
        "apply_value_transfer",
        "apply_value_transfer_plan",
        "classify_value_transfer",
        "ResolvedValueTransfer.cost",
        "_named_axes",
        "_replace_transfer_leaf",
        "_validate_edge_identity",
        "_validate_replay_leaf_identity",
        "_validate_continuation_leaf_identity",
        "_assert_value_metadata",
        "_normalize_shape",
        "_require_period",
        "_require_name",
        "_require_enum",
        "_validate_path_segment",
        "_require_sharding",
        "_check_sharding_shape",
        "ValueViewDescriptor.__post_init__",
        "ValueViewDescriptor.selected_axes",
        "ValueViewDescriptor.structure_key",
        "ValueViewDescriptor.identity_key",
        "CoordinateSelection.__post_init__",
        "TransferStage.allocates",
        "TransferStage.output_footprint",
        "ResolvedValueTransfer.consumer_shape",
        "ResolvedValueTransfer.selects",
        "ResolvedValueTransfer.delivers_stored_buffer",
        "ResolvedValueTransfer.stages",
        "transfer_result_key",
        "_select_view_blocks",
        "_selection_operands",
        "_select_stored_block",
        "_fail_if_view_mismatches_transfer",
        "_fail_if_selections_invalid",
        "_selection_sharding",
    ),
)


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
                "expected_dtype: DTypeLike",
                "reused_by_several_consumers: bool = False",
                "view: ValueViewDescriptor | None = None",
                "specialization_key: Hashable = field(init=False)",
            ),
            methods=(
                "__post_init__",
                "consumer_shape",
                "selects",
                "delivers_stored_buffer",
                "stages",
                "cost",
            ),
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
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="value view coordinate selection",
            class_name="CoordinateSelection",
            fields=(
                "state_name: StateName",
                "start: int",
                "width: int",
                "codes: tuple[int, ...]",
                "keep_axis: bool = False",
            ),
            methods=("__post_init__",),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="value view descriptor",
            class_name="ValueViewDescriptor",
            fields=(
                "artifact: ValueArtifactAddress",
                "leaf: ValueViewLeaf",
                "stored_axis_names: tuple[StateName, ...]",
                "stored_shape: tuple[int, ...]",
                "dtype: DTypeLike",
                "weak_type: bool",
                "consumer_shape: tuple[int, ...]",
                "required_sharding: jax.sharding.Sharding",
                "selections: tuple[CoordinateSelection, ...] = ()",
            ),
            methods=(
                "__post_init__",
                "selected_axes",
                "structure_key",
                "identity_key",
            ),
        )
    )
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="value transfer stage",
            class_name="TransferStage",
            fields=(
                "kind: TransferStageKind",
                "input_shape: tuple[int, ...]",
                "output_shape: tuple[int, ...]",
                "input_sharding: jax.sharding.Sharding",
                "output_sharding: jax.sharding.Sharding",
                "operator: ValueTransferKind | None",
                "item_bytes: int",
            ),
            methods=("allocates", "output_footprint"),
        )
    )
    enum_contracts = {
        "ValueViewLeaf": """SHARED = "shared"
SELECTED = "selected"
""",
        "TransferStageKind": """SELECT = "select"
COMMUNICATE = "communicate"
""",
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
            contracts=_VALUE_TRANSFER_PINS,
        )
    )
    errors.extend(
        _module_contract_errors(
            tree=tree,
            label="value transfer",
            relevant_import_names={
                "ArtifactKey",
                "ArtifactFootprint",
                "ExecutionPlanningError",
                "Hashable",
                "Iterable",
                "Mapping",
                "MappingProxyType",
                "Protocol",
                "RegimeName",
                "StateName",
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
                "from typing import Protocol, TypedDict, runtime_checkable",
                "import jax",
                "import jax.numpy as jnp",
                "from _lcm.execution.footprint import ArtifactFootprint, layout_footprint, sharding_device_ids",
                "from _lcm.typing import DataclassInstance, HostArray, RegimeName, StateName",
                "from lcm.exceptions import ExecutionPlanningError",
                "from lcm.solver_api import ArtifactKey",
            ],
            expected_binding_counts={
                "ArtifactKey": 1,
                "ArtifactFootprint": 1,
                "CoordinateSelection": 1,
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
                "StateName": 1,
                "TransferStage": 1,
                "TransferStageKind": 1,
                "ValueViewDescriptor": 1,
                "ValueViewLeaf": 1,
                "_fail_if_selections_invalid": 1,
                "_fail_if_view_mismatches_transfer": 1,
                "_select_stored_block": 1,
                "_select_value_view": 1,
                "_select_view_blocks": 1,
                "_selection_operands": 1,
                "_selection_sharding": 1,
                "enumerate": 0,
                "max": 0,
                "transfer_result_key": 1,
                "zip": 0,
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


_PROCESSING_CALLER_PINS = _callable_pins(
    source=PROCESSING_SOURCE,
    names=("_build_per_subject_decisions_per_period", "process_regimes"),
)


def _processing_caller_errors(tree: ast.Module) -> list[str]:
    """Pin canonical dense reducers and live publication of the program bundle."""
    errors = _exact_callable_errors(
        tree=tree,
        label="simulate caller",
        contracts=_PROCESSING_CALLER_PINS,
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
        # The grouped route's dense reducer must reduce the type-local Q/F it is
        # declared beside, never the ordinary one.
        "type_local_per_subject_decisions": """type_local_per_subject_decisions = _build_per_subject_decisions_per_period(
    state_action_space=state_action_space,
    Q_and_F_functions=type_local_Q_and_F_functions,
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
    active_periods=tuple(simulated_periods[regime_name]),
    has_gated_edges=bool(law.gated_edges),
    type_local_Q_and_F_functions=type_local_Q_and_F_functions,
    type_local_per_subject_decisions=type_local_per_subject_decisions,
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
                "attach_gated_simulation_programs",
                "SimulationPhase",
            },
            expected_imports=[
                "from types import MappingProxyType",
                (
                    "from _lcm.engine import EGMPolicyRead, FeasibilityPoolsByPeriod, NNBEGMPolicyRead, Regime, SimulationPhase, SolutionPhase, StateActionSpace, Variables, _fail_if_template_is_misplaced, placed_devices_for_ids"
                ),
                "from _lcm.regime_building.max_Q_over_a import get_argmax_and_max_Q_over_a",
                "from _lcm.simulation.programs import attach_gated_simulation_programs, build_simulation_programs",
            ],
            expected_binding_counts={
                "MappingProxyType": 1,
                "get_argmax_and_max_Q_over_a": 1,
                "build_simulation_programs": 1,
                "attach_gated_simulation_programs": 1,
                "SimulationPhase": 1,
                "_build_per_subject_decisions_per_period": 1,
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


_SIMULATION_PROGRAM_CORRIDOR_CONTRACTS = _contracts(
    {
        SIMULATION_PROGRAMS_SOURCE: (
            "build_simulation_programs",
            "_decision_programs",
            "attach_gated_simulation_programs",
            "gated_simulation_programs_ready",
            "budgeted_simulation_programs_ready",
            "_GateFoldBody.__call__",
            "_GateRouteBody.__call__",
            "_fail_if_the_streamed_reduction_is_wrong",
            "_decision_subject_arg_names",
            "_decision_value_reads",
            "_decision_body",
            "_StreamedArgmaxQOverA.__call__",
            "_StreamedArgmaxQOverA._fold",
            "_SubjectTiled.__call__",
            "_evaluate_subject_tile",
            "_ArgumentsBoundAtDispatch.__call__",
        ),
        SIMULATION_PROGRAM_TYPES_SOURCE: (
            "SimulationBuildContext.__post_init__",
            "SimulationProgramExecutor.dispatch",
            "SimulationPrograms.__post_init__",
            "SimulationPrograms.declared_axis_names",
            "SimulationPrograms.forward_decision",
            "transition_output_roles",
            "route_output_roles",
            "subject_axis",
        ),
        SIMULATION_RUNTIME_SOURCE: (
            "CompiledSimulationProgram.__call__",
            "SimulationRuntime.dispatch",
            "SimulationRuntime._bind_prepared_route",
            "SimulationRuntime._publish_prepared_route",
            "_operand_signature",
            "_prepared_route_key",
            "SimulationRuntime.prepare",
            "SimulationRuntime.is_prepared",
            "SimulationRuntime._prepare_materialized",
            "execute_simulation_program",
            "_SimulationCandidateCompiler.__call__",
            "_SimulationCandidateCompiler.lower",
            "_SimulationCandidateCompiler._bound",
            "SimulationRuntime.lower_abstract",
            "SimulationRuntime._publish",
            "_materialize_abstract",
            "_with_compiler_memory",
            "_with_subject_extent",
            "_build_context",
            "SimulationRuntime._materialize",
            "SimulationRuntime._require_budget_context",
            "SimulationRuntime.compile_candidate",
            "_CachedSimulationCandidateCompiler.__call__",
            "_simulation_memory",
            "_SimulationResidentBytes.__call__",
            "_simulation_lowering_key",
        ),
        SIMULATION_COMPILE_SOURCE: ("bind_simulation_runtime", "_subject_devices"),
    }
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

    The ordinary and the type-local decision families are declared by the same
    pinned per-period builder, and the forward selector that substitutes the
    type-local family on the grouped route is pinned beside them, so both
    families stay inside this corridor.

    Callable ASTs pin these executable bodies independently of refreshable byte
    seals. Separate module surfaces forbid import rebinding, altered constants,
    descriptors or replacement classes from bypassing those body checks.
    """
    surface, callables = _SIMULATION_PROGRAM_CORRIDOR_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="simulation program corridor", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(
            "simulation program corridor: module bindings or class surface changed"
        )
    return errors


_SIMULATION_DISPATCH_CORRIDOR_CONTRACTS = _contracts(
    {
        SIMULATION_SOURCE: (
            "_simulate_regime_in_period",
            "_execute_finite_replay",
            "simulate",
            "_simulate_subject_chunk",
            "_bind_unit_executor",
            "_lookup_values_from_indices",
            "_read_external_replay",
            "_replay_nnbegm_candidates",
            "_prepare_nnbegm_candidate_bank",
            "_rank_nnbegm_candidate_bank",
            "_initialize_chunk_state",
        ),
        SIMULATION_TRANSITIONS_SOURCE: (
            "calculate_next_states",
            "calculate_next_regime_membership",
            "_update_regime_ids",
            "_draw_random_regime_ids",
            "_advance_states_for_subjects",
        ),
        MODEL_SOURCE: (
            "_validate_sharded_state_capability",
            "_supports_continuous_sharding_vocabulary",
            "_supports_unsharded_continuous_process",
            "Model.__init__",
            "Model._runtime_regimes_for_shape",
            "Model.simulate",
            "Model._open_entry_allocations",
            # Fixed caller owners flow through both private automatic-solve
            # boundaries without becoming numerical operands or cache keys.
            "Model._solve_from_flat_params",
            "Model._solve_compiled",
            "Model._build_external_replay_readers",
            "_fail_if_invalid_taste_shock_seed",
            "Model._process_params",
            "_simulation_programs",
        ),
        SIMULATION_RANDOM_SOURCE: (
            "create_simulation_key",
            "_create_simulation_key",
            "split_simulation_key",
            "_split_simulation_key",
            "generate_simulation_keys",
            "_generate_simulation_keys",
            "draw_random_seed",
        ),
        ENGINE_SOURCE: (),
    }
)


def _simulation_dispatch_corridor_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Pin phase publication and consumption of the selected decision's exact pair.

    The live caller passes complete action grids, states, values and addressed
    random keys to the published period program. It decodes that program's flat
    index through the same completed action grids. Model call-shape publication and
    dispatch share the same executor; phase schemas cannot substitute a property.
    """
    surface, callables = _SIMULATION_DISPATCH_CORRIDOR_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="simulation dispatch corridor", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(
            "simulation dispatch corridor: publication or module surface changed"
        )
    return errors


_SIMULATION_ADAPTER_CONTRACTS = _contracts(
    {
        # Pandas validates labels and assembles numeric host arrays before the same
        # admitted writer used by ordinary inputs. Recursion keeps that writer and
        # completed leaves stay owned until the complete mapping is published.
        PANDAS_UTILS_SOURCE: (
            "initial_conditions_from_dataframe",
            "_role_codes_from_labels",
            "_write_pandas_array",
            "convert_series_in_params",
            "_convert_param_value",
            "array_from_series",
            "_scatter_series",
        ),
        DTYPES_SOURCE: (
            "CanonicalArrayWriter.__call__",
            "canonical_float_dtype",
            "safe_to_int_dtype",
            "safe_to_float_dtype",
        ),
        PARAMS_PROCESSING_SOURCE: (
            "cast_params_to_canonical_dtypes",
            "_cast_shared",
            "_cast_leaves_to_canonical_dtype",
        ),
        SIMULATION_ENTRY_ALLOCATIONS_SOURCE: (
            "SimulationEntryAllocations.snapshot",
            "SimulationEntryAllocations.solve_input_roots",
            "SimulationEntryAllocations.__call__",
            "SimulationEntryAllocations.publish",
            "SimulationEntryAllocations.pad",
            "SimulationEntryAllocations.update_solution",
            "SimulationEntryAllocations.close",
            "_pad_initial_leaf",
            "SimulationEntryAllocations.place_solve_parameters",
        ),
        # The entry coordinator preserves both validation families and passes the
        # real retained inventory to each newly profiled summary allocation.
        INITIAL_CONDITIONS_SOURCE: (
            "validate_simulation_inputs",
            "_preflight_memory",
            "_discrete_initial_specs",
            "_pack_initial_summary",
            "_read_initial_cohorts",
            "validate_initial_conditions",
            "_collect_feasibility_errors",
            "_age_specialized_feasibility_message",
            "_check_regime_feasibility",
            "_regime_feasibility_mask",
            "_run_profiled_feasibility",
            "_batched_feasibility_check",
            "_evaluate_constant_feasibility",
            "_admits_any_action",
            "_per_constraint_feasibility",
            "_format_infeasibility_message",
            "_gather_feasibility_inputs",
            "_subject_feasibility_flag",
            "_constant_feasibility_flag",
            "canonicalize_initial_conditions",
        ),
        SIMULATION_TASTE_STREAM_SOURCE: (
            "create_taste_shock_key",
            "prepare_decision_taste_keys",
            "generate_taste_shock_keys",
            "advance_simulation_taste_key",
            "build_taste_stream_addresses",
            "_encode_subject_row",
            "_advance_simulation_taste_key",
            "_taste_address_words",
            "draw_taste_shock_keys",
            "_row_offset_words",
            "_fold_subject_key",
        ),
        COMPILER_INPUTS_SOURCE: ("compiler_input_paths", "_is_none"),
        SIMULATION_MEMBERSHIP_SOURCE: (
            "initialize_subject_membership",
            "activate_subject_membership",
            "_empty_subject_membership",
            "_activate_subject_membership",
        ),
        FOOTPRINT_SOURCE: (
            "ArtifactFootprint.__post_init__",
            "ScheduledUnit.__post_init__",
            "ResidentInventory.__post_init__",
            "ResidentInventory.resident_bytes",
            "concrete_device_bytes",
            "plan_resident_bytes",
            "plan_resident_inventory",
            "per_device_footprint",
            "layout_footprint",
            "sharding_device_ids",
            "_walk_wave",
            "_walk_period_folds",
            "_register_outputs",
            "_resident_inventory",
            "_device_bytes",
            "_group_bytes",
            "_group_is_present",
            "_group_is_consumed",
            "_release_after_dispatch",
            "_fail_if_footprint_is_unplanned",
            "_fail_if_period_disagrees",
            "_fail_if_negative",
            "_fail_if_not_a_device_set",
        ),
        SIMULATION_OPERANDS_SOURCE: (
            "SubjectArgumentNames.subject_arg_names",
            "place_simulation_arguments",
            "_require_operand_headroom",
            "_required_operand_bytes",
            "_operand_leaves",
            "_paths_below",
            "_place_operand_tree",
            "_place_operand_leaf",
            "subject_operand_sharding",
        ),
        SIMULATION_UNIT_SOURCE: (
            "SimulationUnitExecutor._live",
            "SimulationUnitExecutor.dispatch",
            "SimulationUnitExecutor.close",
        ),
        SIMULATION_HOST_SOURCE: (
            "ProfiledSimulationOperations.dispatch",
            "ProfiledSimulationOperations.compile_candidate",
            "ProfiledSimulationOperations.lower_abstract",
            "ProfiledSimulationOperations._publish",
            "_abstract_operation",
            "_lower_operation",
            "_OperationCompiler.__call__",
            "_abstract_operand",
            "_static_identity",
            "_operation_memory",
            "_ProfiledOperation.peak_bytes",
            "_ProfiledOperation.reservation_bytes",
        ),
        SIMULATION_MEMORY_SOURCE: (
            "SimulationMemory.__setattr__",
            "SimulationMemory._period_snapshot",
            "SimulationMemory.snapshot",
            "SimulationMemory.budget_snapshot",
            "SimulationMemory.set_chunk_inputs",
            "SimulationMemory.publish",
            "SimulationMemory.replace_outputs",
            "SimulationMemory.set_derived",
            "SimulationMemory.hold",
            "SimulationMemory.before_transfer",
            "SimulationMemory.check_resident",
            "SimulationMemory.run",
            "SimulationMemory.close_unit",
            "run_simulation_operation",
        ),
        SIMULATION_PERIOD_INPUTS_SOURCE: (
            "decision_reads",
            "unit_value_reads",
            "gate_reads",
            "acquire_gate_inputs",
            "acquire_decision_inputs",
        ),
        SIMULATION_REPLAY_INPUTS_SOURCE: (
            "replay_payload_reads",
            "place_replay_payload",
            "_payload_read",
            "_consumer_step",
            "PreparedReplayReader.reads",
            "PreparedReplayReader.build",
        ),
        SIMULATION_VALUE_READS_SOURCE: (
            "BeforeValueTransfer.__call__",
            "PeriodSimulationReads.__init__",
            "PeriodSimulationReads.read",
            "PeriodSimulationReads.commit",
            "PeriodSimulationReads.live_values",
            "PeriodSimulationReads.finish",
            "PeriodSimulationReads._read_host",
            "PeriodSimulationReads._check_open_unit",
            "_host_replay_sharding",
        ),
        SIMULATION_VALUE_PLACEMENT_SOURCE: ("simulation_value_sharding",),
        SIMULATION_CHUNK_INPUTS_SOURCE: ("prepare_simulation_chunk_inputs",),
        SIMULATION_ENTRY_INPUTS_SOURCE: (
            "SimulationEntryInputs.footprint",
            "capture_simulation_entry_inputs",
            "_caller_arrays",
        ),
        SIMULATION_RESIDENCY_SOURCE: (
            "OwnerLedger.bind",
            "OwnerLedger.measure",
            "OwnerLedger.release",
            "OwnerLedger.release_prefix",
            "OwnerLedger.clear",
            "OwnerLedger.bump",
            "OwnerLedger.union",
            "OwnerLedger._extend",
            "OwnerLedger._invalidate",
            "DeviceBufferFootprint.__post_init__",
            "measure_buffer_footprint",
            "union_buffer_footprints",
            "resolve_budget_devices",
            "resident_bytes_by_device",
            "require_transfer_headroom",
            "_merge_spans",
            "_uncovered_bytes",
        ),
        SIMULATION_GATED_ROUTING_SOURCE: (
            "simulation_gate_fold",
            "simulation_gate_route",
            "gated_route_candidates",
            "simulation_gate_route_delta",
            "commit_gated_route_delta",
            "substitute_gated_edge_continuations",
            "route_gated_edges",
            "_per_row_leg_outcomes",
            "bind_provenance_params",
            "_call_vmapped_with_accepted_kwargs",
            "split_population_call_args",
            "install_population_call",
            "_accepted_arg_names",
            "_role_code",
            "population_call",
            "_map_subject_tiles",
            "_call_one_subject_with_shared",
            "_call_one_subject",
        ),
        VALUE_TOPOLOGY_SOURCE: (
            "expected_V_rank",
            "placed_V_sharding",
            "_get_regime_V_shapes_and_shardings",
            "_build_zero_V_arr",
        ),
        RETAINED_BUFFERS_SOURCE: (
            "retained_solution_buffers",
            "_RetainedBuffers.collect",
            "_RetainedBuffers.collect_lazy",
            "_RetainedBuffers.collect_authority",
            "_RetainedBuffers.collect_reader",
            "_unsupported",
        ),
        SCHEDULER_SOURCE: (
            "buffer_identity",
            "shard_identities",
            "shares_a_buffer",
            "BufferRegistry.__init__",
            "BufferRegistry.declare_not_produced",
            "BufferRegistry.declare_passed_through",
            "BufferRegistry.declared_shards",
            "BufferRegistry.is_not_produced",
            "BufferRegistry.register",
            "BufferRegistry.artifacts_sharing",
            "BufferRegistry.forget",
            "BufferRegistry.forget_identity",
            "BufferRegistry._prune_dead_declarations",
            "release_closed_artifacts",
            "_one_delete_per_shared_buffer",
            "plan_period_waves",
            "replace_leaf_by_identity",
            "PeriodTransferCache.__init__",
            "PeriodTransferCache.get",
            "PeriodTransferCache.put",
            "PeriodTransferCache.commit_consumer",
            "PeriodTransferCache.__len__",
            "_add_declaring_array",
            "_keep_live_declaring_arrays",
        ),
        LIVENESS_SOURCE: (
            "PlannedInputLiveness.__init__",
            "PlannedInputLiveness.pending_dispatches",
            "PlannedInputLiveness.remaining_counts",
            "PlannedInputLiveness.retained_artifacts",
            "PlannedInputLiveness.aliases",
            "PlannedInputLiveness.accesses_of",
            "PlannedInputLiveness.is_known",
            "PlannedInputLiveness.remaining_consumers",
            "PlannedInputLiveness.is_retained",
            "PlannedInputLiveness.is_pinned",
            "PlannedInputLiveness.alias_group",
            "PlannedInputLiveness.is_release_eligible",
            "PlannedInputLiveness.has_sole_remaining_consumer",
            "PlannedInputLiveness.commit_successful_dispatch",
            "PlannedInputLiveness.assert_solve_complete",
            "PlannedInputLiveness._require_known",
            "_snapshot_unique_hashable",
            "_require_hashable",
        ),
        CONTINUATION_READS_SOURCE: (
            "continuation_leaf_reads",
            "published_continuation_template",
            "published_continuation_templates",
            "rekeyed_value_reads",
            "with_continuation_leaf_reads",
        ),
        WORKSPACE_PLANNING_SOURCE: (
            "WorkspacePlan.__post_init__",
            "workspace_width_candidates",
            "plan_workspace",
            "plan_axis_free_workspace",
            "_resident_exhausts_budget_message",
            "_no_candidate_fits_message",
            "_validate_axes",
            "_validate_axis",
            "_validate_coordinates",
            "_validate_fixed_widths",
            "_validate_budget",
            "_validate_resident_bytes",
            "bootstrap_width",
            "_tiled_bootstrap_cap",
            "bootstrap_widths",
            "_workspace_width_candidates",
            "_candidate_rank",
            "_axis_frontier",
            "_fixed_width",
            "_admissible_width",
            "_smallest_admissible_width",
            "_width_mapping",
            "_memory_for_candidate",
            "compiler_peak_bytes",
            "_non_negative_bytes",
            "_resident_bytes_for_candidate",
            "CompilerMemoryRecord.__post_init__",
            "CompilerMemoryRecord.allocation_bytes",
            "CompilerMemoryRecord.reservation_bytes",
            "CompilerMemoryReservation.__post_init__",
            "CompilerMemoryReservation.peak_bytes",
            "CompilerMemoryReservation.reservation_bytes",
            "compiler_memory_reservation",
            "_compiler_memory_analysis",
            "_allocation_record",
            "_fail_if_host_allocations",
        ),
    }
)

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
        (
            "copied = apply_value_transfer(value=candidate_filter(value), transfer=transfer)"
        ),
    ),
    "simulation_adapter:subject_device_order_changed": (
        SIMULATION_VALUE_PLACEMENT_SOURCE,
        "            devices=devices,",
        "            devices=tuple(reversed(devices)),",
    ),
    "simulation_adapter:completed_action_space_changed": (
        SIMULATION_CHUNK_INPUTS_SOURCE,
        "spaces[name] = space",
        "spaces[name] = candidate_filter(space)",
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
        "        return self._arrays.get(key)",
        "        return candidate_filter(self._arrays.get(key))",
    ),
    "simulation_adapter:live_read_released_early": (
        LIVENESS_SOURCE,
        "            self._remaining_by_artifact[artifact] -= 1",
        "            self._remaining_by_artifact[artifact] = 0",
    ),
    "simulation_adapter:read_occurrence_changed": (
        CONTINUATION_READS_SOURCE,
        "read, source=dataclasses.replace(read.source, core_key=core_key)",
        (
            "candidate_filter(read), source=dataclasses.replace(read.source, core_key=core_key)"
        ),
    ),
    "simulation_adapter:selected_executable_changed": (
        WORKSPACE_PLANNING_SOURCE,
        "                compiled=compiled,\n            )",
        "                compiled=candidate_filter(compiled),\n            )",
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


# These interfaces connect admitted private copies, descriptor-only preparation,
# exact chunk profiles and the actual dispatch. Bodies below are structural
# obligations; compiler/numerical evidence is established in the scoped tests.
# Eager execution preserves the resolved numerical body and exact outputs.
# These AST guards authenticate pre-body placement, transient ownership, and the
# physical-layout predicate; scoped real-device tests establish executable behavior.
# The finite-policy profile uses the actual retained payload and compiler bank
# schema. These structural guards pin admission, transfer addresses, diagnostics,
# and dispatch bindings; numerical ranking bodies retain their existing contract.
_FINITE_BUDGET_CONTRACTS = _contracts(
    {
        "src/_lcm/simulation/chunk_admission.py": (
            "_ChunkProfiler.profile_widths",
            "_simulation_chunk_profile_key",
            "_independent_outer_candidates",
            "_independent_anchor_widths",
            "_plan_independent_chunks",
            "_profile_independent_candidate",
            "prepare_simulation_chunks",
            "_ChunkProfiler.__call__",
            "_common_axes",
        ),
        "src/_lcm/simulation/chunk_profiles.py": (
            "profile_simulation_chunk",
            "_period_copy_reservation",
            "_policy_read_sources",
            "_retained_read_source",
        ),
        "src/_lcm/simulation/forward_program_profiles.py": (
            "profile_forward_programs",
            "profile_forward_unit",
            "_profile_finite_decision",
            "_abstract_policy_leaf",
        ),
        "src/_lcm/simulation/program_arguments.py": (
            "policy_prepare_arguments",
            "policy_rank_arguments",
            "gate_fold_arguments",
            "gate_route_arguments",
        ),
        "src/_lcm/simulation/simulate.py": (
            "simulate",
            "_simulate_regime_in_period",
            "_execute_finite_replay",
            "_announce_dropped_outer_candidates",
            "_report_dropped_outer_candidates",
        ),
        "src/_lcm/simulation/policy_diagnostics.py": ("dropped_candidate_counts",),
        "src/lcm/model.py": (
            "_validate_sharded_state_capability",
            "_supports_continuous_sharding_vocabulary",
            "_supports_unsharded_continuous_process",
            "Model.__init__",
            "Model.simulate",
            "Model._open_entry_allocations",
        ),
    }
)


def _finite_budget_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Reject changed finite budget transport independently of byte seals."""
    surface, callables = _FINITE_BUDGET_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="finite policy budget transport", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("finite policy budget transport: module bindings changed")
    return errors


_EAGER_INPUT_CONTRACTS = _contracts(
    {
        "src/_lcm/execution/output_layout.py": ("_assert_output_leaf",),
        "src/_lcm/execution/value_transfer.py": ("_assert_value_metadata",),
        "src/_lcm/solution/backward_induction.py": (
            "_period_transfer_scratch_reservations",
            "_continuous_value_replica_required",
            "_compile_all_functions",
            "_prepare_solve_programs",
        ),
        EAGER_CORE_SOURCE: (
            "make_eager_core",
            "_EagerCore.__call__",
            "_EagerCore.place_operand",
            "_EagerCore._typed_sharding",
            "_EagerPlacement.internal",
            "_EagerPlacement.__call__",
        ),
        RUNTIME_SHARDING_SOURCE: ("runtime_shardings_match",),
    }
)


def _eager_input_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Reject changed eager transport even after an independent byte reseal."""
    surface, callables = _EAGER_INPUT_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="eager input transport", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("eager input transport: module bindings changed")
    return errors


# The call-local owner authenticates completion, not numerical arithmetic or
# physical allocator peaks. Real pending-array and transfer regressions establish
# execution behavior; these guards reject changed lifetime/dispatch corridors.
_SOLVE_READINESS_CONTRACTS = _contracts(
    {
        SOLVE_PENDING_WORK_SOURCE: (
            "BeforeArrayDelete.__call__",
            "PendingSolveWork.__init__",
            "PendingSolveWork.before",
            "PendingSolveWork.record",
            "PendingSolveWork.before_delete",
            "PendingSolveWork.close",
            "_MaterializedCopies.__call__",
            "_MaterializedCopies.close",
            "execute_with_pending_work",
            "_drain",
            "_complete_array",
        ),
        "src/_lcm/execution/output_layout.py": ("PlannedCore.__call__",),
        "src/_lcm/execution/value_transfer.py": (
            "MaterializedTransferObserver.__call__",
            "apply_value_transfer",
            "apply_value_transfer_plan",
            "_replace_transfer_leaf",
            "_transferred_leaf",
            "_replace_dataclass_field",
        ),
        "src/_lcm/execution/scheduler.py": (
            "release_closed_artifacts",
            "PeriodTransferCache.__init__",
            "PeriodTransferCache.commit_consumer",
        ),
        "src/_lcm/solution/backward_induction.py": (
            "_period_transfer_scratch_reservations",
            "_continuous_value_replica_required",
            "solve",
            "_cores_with_transfer_cache",
            "_release_closed_period_inputs",
            "_retire_donated_inputs",
        ),
    }
)


def _solve_readiness_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Reject changed solve completion ownership after independent byte resealing."""
    surface, callables = _SOLVE_READINESS_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="solve completion ownership", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("solve completion ownership: module bindings changed")
    return errors


_COMBINED_INPUT_CONTRACTS = _contracts(
    {
        # Trusted single-array native values only. Artifact codec reconstruction
        # remains outside this claim; module surfaces pin imports and cache schemas.
        NATIVE_VALUES_SOURCE: (
            "NativeValueMaterializer.require_entry",
            "NativeValueMaterializer.__call__",
        ),
        NATIVE_ARCHIVE_SOURCE: (
            "_LazyHdf5Entry._materialize",
            "_read_and_verify_leaves",
            "_require_local_group",
            "_require_local_dataset",
            "_array_checksum",
            "_array_checksum_from_leaf_metadata",
            "_to_jax_without_narrowing",
        ),
        COMBINED_ABSTRACT_PROGRAM_INPUTS_SOURCE: (
            "abstract_program_inputs",
            "_OperandDescriptor.__call__",
            "_identity",
        ),
        COMBINED_ASSEMBLY_SOURCE: (
            "concatenate_arrays",
            "slice_array",
            "_run_assembly",
            "_concatenate_arrays",
            "_slice_array",
        ),
        COMBINED_CHUNK_ADMISSION_SOURCE: (
            "_ChunkProfiler.profile_widths",
            "_simulation_chunk_profile_key",
            "_independent_outer_candidates",
            "_independent_anchor_widths",
            "_plan_independent_chunks",
            "_profile_independent_candidate",
            "PreparedSimulationChunks.require_chunk",
            "prepare_simulation_chunks",
            "_ChunkProfiler.__call__",
            "_common_axes",
        ),
        COMBINED_CHUNK_OFFLOAD_SOURCE: (
            "chunk_host_device",
            "offload_chunk",
            "_copy_reservation",
        ),
        COMBINED_CHUNK_OPERATIONS_SOURCE: (
            "slice_population",
            "_slice_population",
            "period_age",
            "_period_age",
            "regime_mask",
            "_regime_mask",
            "broadcast_collective",
            "_broadcast_collective",
            "broadcast_value",
            "_broadcast_value",
            "empty_fallback",
            "_empty_fallback",
        ),
        COMBINED_CHUNK_PLANNING_SOURCE: (
            "IndependentChunkReceipt.profile_count",
            "SimulationStageProfile.__post_init__",
            "SimulationChunkProfile.__post_init__",
            "SimulationChunkPlan.__post_init__",
            "ChunkProfiler.__call__",
            "plan_simulation_chunks",
            "_required_bytes",
            "_validate_devices",
        ),
        COMBINED_CHUNK_PROFILE_INVENTORY_SOURCE: (
            "abstract_tree",
            "_abstract_leaf",
            "payload_bytes",
            "add_bytes",
            "maximum_bytes",
            "ChunkProfileInventory.operation",
            "ChunkProfileInventory.compiled",
            "ChunkProfileInventory.close_unit",
        ),
        COMBINED_CHUNK_PROFILES_SOURCE: (
            "profile_simulation_chunk",
            "_profile_next_subjects",
            "_profile_population_roles",
            "_profile_outer_storage",
            "_record_core",
            "_profile_initial_carrier",
            "_profile_keys",
            "_profile_taste",
            "_period_copy_reservation",
            "_profile_entry_key",
        ),
        COMBINED_DIAGNOSTIC_OPERATIONS_SOURCE: (
            "period_value_flags",
            "owned_value_nan_count",
            "transition_counts",
            "profiled_transition_counts",
            "DiagnosticBinding.__post_init__",
            "diagnostic_bindings",
        ),
        COMBINED_FORWARD_PROGRAM_PROFILES_SOURCE: (
            "AbstractSimulationProfile.__post_init__",
            "profile_forward_programs",
            "profile_forward_unit",
            "_profile_program",
            "_prepare_program",
            "_concrete_widths",
            "_stochastic_keys",
            "_shared_tree",
            "_shared_leaf",
            "_placed_abstract",
        ),
        COMBINED_POPULATION_OPERATIONS_SOURCE: (
            "default_roles",
            "regime_is_occupied",
            "canonical_roles",
            "role_mismatch",
            "starting_periods",
            "match_starting_periods",
        ),
        COMBINED_PROGRAM_ARGUMENTS_SOURCE: (
            "decision_arguments",
            "transition_arguments",
        ),
        COMBINED_SOLUTION_COPIES_SOURCE: ("copy_solution_leaf", "_copy_value_leaf"),
        COMBINED_RESULT_SNAPSHOT_SOURCE: (
            "snapshot_value_store",
            "_snapshot_value_coordinate",
            "_keep_payload",
        ),
        COMBINED_VALIDATE_V_SOURCE: (
            "value_function_nan_error",
            "_entry_support_cause",
        ),
        COMBINED_LOGGING_SOURCE: (
            "_owned_values",
            "non_finite_by_regime",
            "log_non_finite_values",
            "log_regime_transition_counts",
            "validation_enabled",
            "validation_raises",
        ),
        COMBINED_AUTHORITY_SOURCE: (
            "_ArrayCopier.__call__",
            "_copy_artifact_array_leaf",
        ),
        COMBINED_ENTRIES_SOURCE: (
            "_ValueMaterializer.__call__",
            "_copy_solution_value",
            "_CanonicalValueEntry.materialize",
            "_CanonicalValueEntry._fresh",
            "_canonical_value_entry",
        ),
        COMBINED_STORES_SOURCE: (
            "_admit_value_entry",
            "ValueStore.__post_init__",
            "ValueStore._initialize",
            "ValueStore._from_entries_with_copy",
            "ValueStore._load",
            "ValueStore.materialize",
            "ValueStore._materialize_with_copy",
        ),
        "src/lcm/model.py": (
            "_validate_sharded_state_capability",
            "_supports_continuous_sharding_vocabulary",
            "_supports_unsharded_continuous_process",
            "Model.__init__",
            "Model._check_solution_result_structure",
            "Model._consume_foreign_solution",
            "Model._resolve_compile_batch_size",
            "Model._resolve_solution_result",
            "Model._snapshot_solution_envelope",
        ),
        "src/_lcm/execution/core_program.py": ("_validate_abstract_inputs",),
        "src/_lcm/solution/backward_induction.py": (
            "_period_transfer_scratch_reservations",
            "_continuous_value_replica_required",
            "_prepare_abstract_program",
        ),
        "src/_lcm/simulation/simulate.py": (
            "_compute_starting_periods",
            "_concatenate_chunk_results",
            "_validate_period_values",
            "_validate_simulated_value",
        ),
        "src/_lcm/simulation/transitions.py": (
            "_draw_random_regime_ids_from_scalars",
            "draw_key_from_dict",
        ),
        "src/_lcm/simulation/chunk_inputs.py": (
            "SimulationCallInputs.array_roots",
            "prepare_simulation_call_inputs",
        ),
        "src/_lcm/simulation/entry_allocations.py": (
            "SimulationEntryAllocations.copy_solution_leaf",
            "SimulationEntryAllocations.release_foreign_copies",
        ),
        "src/_lcm/simulation/host_operations.py": (
            "ProfiledSimulationOperations.prepare_abstract",
            "_abstract_operation",
            "_lower_operation",
            "_abstract_operation_tree",
            "_operation_key",
            "_validated_static_arguments",
            "_validated_operation_function",
            "_ProfiledOperation.peak_bytes",
            "_ProfiledOperation.reservation_bytes",
        ),
        "src/_lcm/simulation/initial_conditions.py": (
            "_CarrierWriter.__call__",
            "_build_admitted_initial_states",
            "_cast_carrier",
            "_fill_carrier",
            "_initial_own_stakeholder",
            "build_initial_states",
            "trim_pad_from_raw_results",
        ),
        "src/_lcm/simulation/memory.py": (
            "SimulationMemory.__setattr__",
            "SimulationMemory._period_snapshot",
            "SimulationMemory.__post_init__",
        ),
        "src/_lcm/simulation/random.py": (
            "_generate_windowed_simulation_keys",
            "_validated_chunk_window",
        ),
        "src/_lcm/simulation/runtime.py": (
            "SimulationDispatchContext.__post_init__",
            "SimulationRuntime.prepare_abstract",
            "_materialize_abstract",
            "_dispatch_widths",
            "_unbudgeted_subject_width",
            "_subject_slice_bytes",
            "_require_abstract_arguments",
        ),
    }
)


def _combined_input_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Authenticate reviewed copy/profile boundaries after independent byte reseals."""
    surface, callables = _COMBINED_INPUT_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="simulation copy and chunk profile", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("simulation copy and chunk profile: module bindings changed")
    return errors


_STRUCTURAL_BLUEPRINT_CONTRACTS = _contracts(
    {
        # A warm solve binds a stored structural blueprint on a key hit. The key
        # is the inputs' abstract schema plus frozen policy values, so changed
        # parameter values with an unchanged schema hit; a changed shape, dtype,
        # weak type or placement misses. The store, its lookup and both key
        # derivations are one corridor.
        STRUCTURAL_BLUEPRINTS_SOURCE: (
            "StructuralBlueprintCache.__init__",
            "StructuralBlueprintCache.get",
            "StructuralBlueprintCache.put",
            "StructuralBlueprintCache.values",
            "StructuralBlueprintCache.__len__",
            "abstract_schema",
            "_leaf_schema",
            "frozen_policy",
        ),
    }
)


def _structural_blueprint_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Authenticate the structural-blueprint store and its key derivations."""
    surface, callables = _STRUCTURAL_BLUEPRINT_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="structural blueprint cache", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("structural blueprint cache: module bindings changed")
    return errors


_FINITE_REPLAY_CONTRACTS = _contracts(
    {
        SIMULATION_POLICY_PROGRAMS_SOURCE: (
            "ReplayPayload.from_policy",
            "ReplayPayload.restore",
            "_flatten_payload",
            "_unflatten_payload",
            "declare_finite_replay_programs",
            "_program",
            "_policy_reads",
            "_Prepare.__call__",
            "_Rank.__signature__",
            "_Rank.__call__",
        ),
        PUBLISHED_POLICY_SOURCE: ("_flatten_nnbegm_policy", "_unflatten_nnbegm_policy"),
        MODEL_PROCESSING_SOURCE: ("build_regimes_and_template",),
    }
)


def _finite_replay_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Pin the finite producer schema, dynamic payload and declared stage transport.

    Producer numerical kernels are outside this structural obligation. The
    registered leaf order and immutable reconstruction schema authenticate the
    four/five array locators, while the stage bodies retain the complete bank
    and unchanged canonical ranking. Executable tests separately compare real
    producer payloads, dynamic replacement, masks, candidate owners and levels.
    """
    surface, callables = _FINITE_REPLAY_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="finite replay transport", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("finite replay transport: module or producer schema changed")
    return errors


_NBEGM_DONATION_PINS = _callable_pins(
    source=NBEGM_SOURCE,
    names=("NBEGM.declare_continuation_reads", "_with_ride_marginal_reads"),
)


_NBEGM_DONATION_SURFACE = _surface_pin(NBEGM_SOURCE)


def _nbegm_donation_errors(tree: ast.Module) -> list[str]:
    """Pin donation installation and eligibility, excluding NB-EGM method arithmetic."""
    errors = _exact_callable_errors(
        tree=tree,
        label="NB-EGM donation declaration",
        contracts=_NBEGM_DONATION_PINS,
    )
    if _transport_module_surface(tree) != _NBEGM_DONATION_SURFACE:
        errors.append("NB-EGM donation declaration: module bindings changed")
    return errors


_CONTINUATION_ARGUMENT_PINS = _callable_pins(
    source=CONTINUATION_ARGUMENTS_SOURCE,
    names=(
        "MarginalLeafArguments.__call__",
        "MarginalLeafCore.__call__",
        "marginal_leaf_reads",
    ),
)


_CONTINUATION_ARGUMENT_SURFACE = _surface_pin(CONTINUATION_ARGUMENTS_SOURCE)


def _continuation_argument_errors(tree: ast.Module) -> list[str]:
    """Pin the sole marginal operand, residual tree and exact carry reconstruction."""
    errors = _exact_callable_errors(
        tree=tree,
        label="donation argument transport",
        contracts=_CONTINUATION_ARGUMENT_PINS,
    )
    if _transport_module_surface(tree) != _CONTINUATION_ARGUMENT_SURFACE:
        errors.append("donation argument transport: module bindings changed")
    return errors


_BACKWARD_OUTPUT_LAYOUT_PINS = _callable_pins(
    source=BACKWARD_INDUCTION_SOURCE,
    names=(
        "_period_transfer_scratch_reservations",
        "_continuous_value_replica_required",
        "_evaluate_edge_fold",
        "_lower_and_compile_wave",
        "_lower_resolved_candidate",
        "CompilationWave.lower",
        "CompilationWave._submit",
        "CompilationWave.__exit__",
        "CompilationWave._raise_first_compile_error",
        "_compile_and_log",
        "_run_period_kernel",
        "_regime_retains_replay",
        "_select_period_programs",
        "_selected_artifact_keys_for_cell",
        "_compile_all_functions",
        "_prepare_solve_programs",
        "_CompilerMemoryLookup.__call__",
        "_resolve_output_layouts_and_lowering_keys",
        # The structural blueprint is where every program is materialized
        # and its top-ranked candidate resolved against abstract inputs; a
        # warm solve binds the stored blueprint, so the recipe, its key and
        # its per-call binding are one corridor with the resolver above.
        "_build_structural_blueprint",
        "_bind_structural_blueprint",
        "_structural_key",
        "_select_runtime_donation_cores",
        "_donation_ownership_refusal",
        "_mark_reused_transfers",
        "_consumer_key",
        "_resolve_program_for_execution",
        "_resolve_value_input_transfer_plan",
        "_resolve_value_transfer_layout",
        "_lowering_key",
        "_abstract_arguments_key",
        "_abstract_value_key",
        "_abstract_leaf_key",
        "_output_roles_key",
        "_assert_lowered_output_roles",
        "_attach_resolved_output_layout",
        "_publish_kernel_value",
        "_resident_bytes_by_triple",
        "_resident_inventory_by_triple",
        "_candidate_resident_bytes",
        "_compiler_reads_source",
        "_period_copy_reservations",
        "_internal_reservations_by_cell",
        "_internal_leaf_bytes",
        "_retained_base_space_arrays",
        "_CandidateResidencyLookup.__call__",
    ),
)


_BACKWARD_OUTPUT_LAYOUT_SURFACE = _surface_pin(BACKWARD_INDUCTION_SOURCE)


def _backward_output_layout_errors(tree: ast.Module) -> list[str]:
    """Pin native graph resolution and V/D publication through solve execution."""
    errors = _exact_callable_errors(
        tree=tree,
        label="backward output-layout transport",
        contracts=_BACKWARD_OUTPUT_LAYOUT_PINS,
    )
    # A stored blueprint outlives its solve, so it may hold only abstract,
    # immutable recipe facts: no ledger, donation, lowering key or cursor.
    errors.extend(
        _class_surface_errors(
            tree=tree,
            label="backward output-layout transport",
            class_name="_StructuralBlueprint",
            fields=(
                "programs: tuple[CoreProgram, ...]",
                "layouts: MappingProxyType[_CoreTriple, ResolvedOutputLayout]",
                "resolved_programs: MappingProxyType[_CoreCandidate, ResolvedCoreProgram]",
                "internal_templates: MappingProxyType[_CoreCandidate, Mapping[ReferenceName, ShapeDtypePytree]]",
                "frontiers: MappingProxyType[_CoreTriple, _CoreFrontier]",
                "frontier_lengths: MappingProxyType[_CoreTriple, int]",
                "transfer_consumers: MappingProxyType[_ConsumerKey, frozenset[_CoreTriple]]",
                "representative_metadata: MappingProxyType[_CoreTriple, _ProgramExecutionMetadata]",
            ),
            methods=(),
            decorators=("dataclasses.dataclass(frozen=True, kw_only=True)",),
        )
    )
    if _transport_module_surface(tree) != _BACKWARD_OUTPUT_LAYOUT_SURFACE:
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
            source=(
                "(retained_input_arrays, tuple((space.states, space.discrete_actions, space.continuous_actions) for space in base_state_action_spaces.values()))"
            ),
        ):
            errors.append(
                "backward output-layout transport: fixed owner inputs changed"
            )
        if _scope_binding_counts(solve.body).get("retained_input_arrays", 0):
            errors.append(
                "backward output-layout transport: retained caller inputs rebound"
            )

        def _is_kernel_output(statement: ast.stmt) -> bool:
            return (
                isinstance(statement, ast.Assign)
                and any(
                    _target_names(target) == ("output", "blocked")
                    for target in statement.targets
                )
                and isinstance(statement.value, ast.Call)
                and _call_name(statement.value) == "_run_dispatch_unit"
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
        expected = "a4cee1b42742def4634b2c087305aaaf927d12e839f9a72b2795cbb79a6836aa"
        if len(corridors) != 1 or _statements_ast_sha256(corridors[0]) != expected:
            errors.append(
                "backward output-layout transport: solve publication corridor changed"
            )
    return errors


_TERMINAL_OUTPUT_WRAPPER_PINS = _callable_pins(
    source=PROCESSING_SOURCE,
    names=(
        "_TerminalCarryPeriodKernel.core_programs",
        "_TerminalCarryPeriodKernel.with_fixed_params",
        "_TerminalCarryPeriodKernel.__call__",
    ),
)


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
            contracts=_TERMINAL_OUTPUT_WRAPPER_PINS,
        )
    )
    return errors


_PERIOD_REPLAY_PINS = _callable_pins(
    source=PERIOD_REPLAY_SOURCE,
    names=(
        "replay_period",
        "_compile_cores_for_one_period",
        "_core_build_context_for_one_period",
    ),
)


def _period_replay_errors(tree: ast.Module) -> list[str]:
    """Pin replay to the same graph, builder, resolver, and output-role checks."""
    return _exact_callable_errors(
        tree=tree,
        label="period replay native-program transport",
        contracts=_PERIOD_REPLAY_PINS,
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
            "from typing import ClassVar, cast",
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
        pair_max = _definition(tree=tree, name="max_and_smallest_id")
        pair_order = _definition(tree=tree, name="_larger_value_then_smaller_id")
        pair_reduce = _definition(tree=tree, name="_paired_max")
        pair_jvp = _definition(tree=tree, name="_paired_max_jvp")
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
where = jnp.ones(a.shape, dtype=bool) if where is None else jnp.broadcast_to(
    where, a.shape
)
is_nan = jnp.isnan(a)
comparable = where & ~is_nan
lowest = -jnp.inf if jnp.issubdtype(a.dtype, jnp.floating) else jnp.iinfo(a.dtype).min
positions = jnp.broadcast_to(jnp.arange(a.shape[-1], dtype=jnp.int32), a.shape)
_max, _argmax = max_and_smallest_id(
    values=jnp.where(comparable, a, lowest),
    ids=jnp.where(comparable, positions, NO_ID),
    initial=lowest if initial is None else initial,
)
any_nan = jnp.any(where & is_nan, axis=-1)
_max = jnp.where(any_nan, jnp.full_like(_max, jnp.nan), _max)
_argmax = jnp.where(any_nan | (_argmax == NO_ID), 0, _argmax)
return _argmax, _max
"""
    expected_pair_max = r"""initial_arr = jnp.asarray(initial, dtype=values.dtype)
if jnp.issubdtype(values.dtype, jnp.floating):
    return _paired_max_with_tangent(values, ids, initial_arr)
return _paired_max(values, ids, initial_arr)
"""
    expected_pair_reduce = r"""return jax.lax.reduce(
    (values, ids),
    (initial, jnp.asarray(NO_ID, dtype=jnp.int32)),
    _larger_value_then_smaller_id,
    (values.ndim - 1,),
)
"""
    expected_pair_jvp = r"""values, ids, initial = primals
values = jax.lax.optimization_barrier(values)
values_dot, _, initial_dot = tangents
best, best_id = _paired_max(values, ids, initial)
attains = (values == best[..., jnp.newaxis]).astype(values.dtype)
count = jnp.sum(attains, axis=-1)
elements_dot = jnp.sum(values_dot * attains, axis=-1) / jnp.maximum(count, 1)
initial_dot = jnp.broadcast_to(initial_dot, best.shape)
best_dot = jnp.where(
    count > 0,
    jnp.where(initial == best, (elements_dot + initial_dot) / 2, elements_dot),
    initial_dot,
)
return (best, best_id), (best_dot, np.zeros(best_id.shape, dtype=float0))
"""
    expected_pair_order = r"""left_value, left_id = left
right_value, right_id = right
tie = right_value == left_value
take_right = (right_value > left_value) | (tie & (right_id < left_id))
value = jnp.where(take_right, right_value, left_value)
value = jnp.where(tie & (left_value == 0), left_value + right_value, value)
return value, jnp.where(take_right, right_id, left_id)
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
    if (
        pair_max.decorator_list
        or pair_order.decorator_list
        or pair_reduce.decorator_list
        or pair_jvp.decorator_list
    ):
        errors.append("argmax reducer: decorators are not allowlisted")
    if not _positional_signature(
        node=pair_reduce, names=("values", "ids", "initial")
    ) or not _body_matches(node=pair_reduce, expected_source=expected_pair_reduce):
        errors.append("argmax reducer: the paired primal is not one exact `lax.reduce`")
    if not _positional_signature(
        node=pair_jvp, names=("primals", "tangents")
    ) or not _body_matches(node=pair_jvp, expected_source=expected_pair_jvp):
        errors.append(
            "argmax reducer: the paired max's tangent is not `jnp.max`'s "
            "tie-averaged tangent"
        )
    tangent_wiring = [
        statement
        for statement in tree.body
        if (
            isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign)
            and "_paired_max_with_tangent" in _assigned_names(statement)
        )
        or (
            isinstance(statement, ast.Expr)
            and _expression_matches(
                node=statement.value,
                source="_paired_max_with_tangent.defjvp(_paired_max_jvp)",
            )
        )
    ]
    if not (
        len(tangent_wiring) == 2
        and isinstance(tangent_wiring[0], ast.Assign)
        and _expression_matches(
            node=tangent_wiring[0].value, source="jax.custom_jvp(_paired_max)"
        )
        and isinstance(tangent_wiring[1], ast.Expr)
    ):
        errors.append(
            "argmax reducer: the paired primal is not wrapped with its tangent rule"
        )
    if not _keyword_only_signature(
        node=pair_max, names=("values", "ids", "initial")
    ) or not _body_matches(node=pair_max, expected_source=expected_pair_max):
        errors.append(
            "argmax reducer: the (value, id) maximum is not one exact paired reduction"
        )
    if not _body_matches(node=pair_order, expected_source=expected_pair_order):
        errors.append(
            "argmax reducer: the (value, id) order is not larger value, then smaller id"
        )
    no_id = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(_target_names(target) == ("NO_ID",) for target in statement.targets)
    ]
    if not (
        len(no_id) == 1
        and _expression_matches(node=no_id[0].value, source="jnp.iinfo(jnp.int32).max")
    ):
        errors.append("argmax reducer: the no-identity sentinel is not int32 max")
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
            "max_and_smallest_id",
            "_larger_value_then_smaller_id",
            "_paired_max",
            "_paired_max_jvp",
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
    for source, name in _unselected_pins():
        errors.append(
            f"certificate: stored corridor pin {source}::{name} is selected by no family"
        )
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
    for relative in _GROUPED_MAPPER_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _grouped_mapper_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _ACTION_GRID_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _action_grid_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _ACTION_PARTITION_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _action_partition_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _UNIFORM_PROCESS_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _uniform_process_errors(tree=tree, source=relative)
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
    for relative in _FINITE_BUDGET_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _finite_budget_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _EAGER_INPUT_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _eager_input_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _SOLVE_READINESS_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _solve_readiness_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _COMBINED_INPUT_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _combined_input_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _STRUCTURAL_BLUEPRINT_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _structural_blueprint_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _FINITE_REPLAY_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _finite_replay_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    nbegm_tree = parsed.get(NBEGM_SOURCE)
    if nbegm_tree is not None:
        new_errors = _nbegm_donation_errors(nbegm_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(NBEGM_SOURCE)
    continuation_arguments_tree = parsed.get(CONTINUATION_ARGUMENTS_SOURCE)
    if continuation_arguments_tree is not None:
        new_errors = _continuation_argument_errors(continuation_arguments_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(CONTINUATION_ARGUMENTS_SOURCE)
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
            "singleton_action_partitioned_solve": (
                "admitted GridSearch request -> device-mapped kernel over the "
                "action axis -> per-device contiguous run of whole C-order blocks, "
                "every block inside the scan, unowned and padded slots infeasible "
                "-> exact hard-max accumulator -> gather over the action axis -> "
                "ascending-order exact hard-max merge -> VALUE-only compiled core"
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
            "finite_policy_simulate": (
                "published NNBEGM payload leaves -> addressed dynamic replay tree -> "
                "planned complete-bank preparation -> ready host drop diagnostic -> "
                "planned unchanged canonical Q/F ranking -> selected action/value"
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
        (
            "n_continuous_cells = math.prod(Q_arr.shape[self.n_discrete_action_axes + 1 :])"
        ),
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
        "from typing import ClassVar, cast",
        "from candidate_filter import ClassVar, cast",
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
        (
            "return candidate_filter.argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)"
        ),
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
    "solve_completion:conflict_wait_bypassed": (
        "src/_lcm/execution/pending_work.py",
        "    owner.before(devices=devices)",
        "    owner.before(devices=frozenset())",
    ),
    "policy_diagnostics:represented_mask_ignored": (
        POLICY_DIAGNOSTICS_SOURCE,
        "jnp.sum(live & ~represented)",
        "jnp.sum(live)",
    ),
    "eager_core:planned_operand_placement_bypassed": (
        "src/_lcm/execution/eager_core.py",
        (
            "            placed = jax.tree.map(\n                placement,"
            "\n                {\n                    name: value\n        "
            "            for name, value in arguments.items()\n            "
            "        if name not in self.internal_input_templates\n        "
            "        },\n                dict(self.arguments),\n           "
            " )\n            placed.update(\n                jax.tree.map("
            "\n                    placement.internal,\n                   "
            " {name: arguments[name] for name in self."
            "internal_input_templates},\n                    dict(self."
            "internal_input_templates),\n                )\n            )"
        ),
        "            placed = dict(arguments)",
    ),
    "runtime_sharding:physical_partition_ignored": (
        "src/_lcm/execution/runtime_sharding.py",
        "and actual.is_equivalent_to(expected, ndim)",
        "and True",
    ),
    "solve_descriptors:required_layout_replaced": (
        "src/_lcm/execution/abstract_program_inputs.py",
        "sharding=transfer.source_sharding,",
        "sharding=transfer.stored_sharding,",
    ),
    "chunk_assembly:cpu_budget_exclusion_broadened": (
        "src/_lcm/simulation/assembly.py",
        'and all(device.platform == "gpu" for device in memory.devices)',
        'and all(device.platform in ("gpu", "cpu") for device in memory.devices)',
    ),
    "chunk_admission:setup_fulfilled_by_unrelated_inputs": (
        "src/_lcm/simulation/chunk_admission.py",
        "else completed_setup,",
        "else memory.snapshot(),",
    ),
    "chunk_offload:source_scratch_omitted": (
        "src/_lcm/simulation/chunk_offload.py",
        "scratch_bytes=scratch,",
        "scratch_bytes={},",
    ),
    "chunk_operations:population_window_shifted": (
        "src/_lcm/simulation/chunk_operations.py",
        "jax.lax.dynamic_slice_in_dim(array, start, width, axis=0)",
        "jax.lax.dynamic_slice_in_dim(array, start + 1, width, axis=0)",
    ),
    "chunk_planning:published_output_reservation_omitted": (
        "src/_lcm/simulation/chunk_planning.py",
        "+ profile.output_reservation.get(device, 0)",
        "+ 0",
    ),
    "chunk_inventory:compiler_output_owners_omitted": (
        "src/_lcm/simulation/chunk_profile_inventory.py",
        "add_bytes(target=self.unit, source=payload_bytes(tree=executable.out_info))",
        "add_bytes(target=self.unit, source={})",
    ),
    "chunk_profiles:action_decoder_profile_omitted": (
        "src/_lcm/simulation/chunk_profiles.py",
        "executable=decoder.executable,",
        "executable=decision.executable,",
    ),
    "chunk_diagnostics:ownership_mask_ignored": (
        "src/_lcm/simulation/diagnostic_operations.py",
        "owned = _owned_values(value=value, in_regime=in_regime)",
        "owned = value",
    ),
    "forward_profiles:current_carrier_ignored": (
        "src/_lcm/simulation/forward_program_profiles.py",
        "states = {state: current[state] for state in base.states}",
        "states = base.states",
    ),
    "population_operations:entry_period_shifted": (
        "src/_lcm/simulation/population_operations.py",
        "return periods, valid, xp.asarray(xp.all(valid))",
        "return periods + 1, valid, xp.asarray(xp.all(valid))",
    ),
    "program_arguments:continuous_actions_omitted": (
        "src/_lcm/simulation/program_arguments.py",
        "        **continuous_actions,\n",
        "",
    ),
    "foreign_copy:source_layout_ignored": (
        "src/_lcm/simulation/solution_copies.py",
        "output_sharding=leaf.sharding,",
        "output_sharding=None,",
    ),
    "foreign_snapshot:lazy_materializer_admitted": (
        "src/_lcm/solution/result_snapshot.py",
        "if type(entry) is not _CanonicalValueEntry:",
        "if False and type(entry) is not _CanonicalValueEntry:",
    ),
    "diagnostic_error:partial_solution_owner_omitted": (
        "src/_lcm/solution/validate_V.py",
        "exc.partial_solution = partial_solution",
        "exc.partial_solution = None",
    ),
    "diagnostic_logging:profiled_callback_bypassed": (
        "src/_lcm/utils/logging.py",
        "counts_host = counts_factory()",
        "counts_host = []",
    ),
    "foreign_authority:copy_admission_bypassed": (
        "src/lcm/_solver_api/authority.py",
        "else array_copier(leaf=leaf, label=label)",
        "else jax.numpy.array(leaf, copy=True)",
    ),
    "foreign_entry:canonical_copy_dependency_omitted": (
        "src/lcm/_solver_api/entries.py",
        'value=source, label="Solution value", array_copier=array_copier',
        'value=source, label="Solution value", array_copier=None',
    ),
    "foreign_store:copy_constructor_dependency_omitted": (
        "src/lcm/_solver_api/stores.py",
        "store._initialize(array_copier=array_copier)",
        "store._initialize(array_copier=None)",
    ),
    "simulation_finite_policy:consumer_locator_shifted": (
        SIMULATION_POLICY_PROGRAMS_SOURCE,
        'path=("arrays", i),',
        'path=("arrays", i + 1),',
    ),
    "simulation_finite_policy:producer_leaf_order_changed": (
        PUBLISHED_POLICY_SOURCE,
        (
            "        policy.candidate_inner_action,\n        policy.candidate_outer_target,"
        ),
        (
            "        policy.candidate_outer_target,\n        policy.candidate_inner_action,"
        ),
    ),
    "simulation_entry:upload_budget_omitted": (
        SIMULATION_ENTRY_ALLOCATIONS_SOURCE,
        "budget_bytes=self.budget_bytes,\n            live_footprint=live,",
        "budget_bytes=None,\n            live_footprint=live,",
    ),
    "donation:unsupported_scope_admitted": (
        NBEGM_SOURCE,
        (
            "context.sharded_state_names\n"
            "        or kernel.statics.co_map_state_names\n"
            "        or kernel.stateful_targets != frozenset({kernel.regime_name})"
        ),
        "False",
    ),
    "donation:replay_nomination_admitted": (
        NBEGM_SOURCE,
        'donation_candidates=(MARGINAL_ARGUMENT,) if name == "main" else (),',
        "donation_candidates=(MARGINAL_ARGUMENT,),",
    ),
    "donation:residual_duplicates_marginal_operand": (
        CONTINUATION_ARGUMENTS_SOURCE,
        'if field.name != "marginal_utility"',
        "if True",
    ),
    "donation:ordinary_fallback_filtered": (
        BACKWARD_INDUCTION_SOURCE,
        "cores[name] = compiled_programs.donation_fallbacks[triple]",
        "cores[name] = candidate_filter(compiled_programs.donation_fallbacks[triple])",
    ),
    "donation:physical_alias_protection_bypassed": (
        BACKWARD_INDUCTION_SOURCE,
        (
            "partners = registry.artifacts_sharing(array=array) - {artifact}\n"
            "    if partners:\n"
            "        return"
        ),
        "partners = set()\n    if partners:\n        return",
    ),
    "donation:paired_residency_omitted": (
        BACKWARD_INDUCTION_SOURCE,
        (
            "memory_by_lowering_key[key].reservation_bytes\n"
            "                        + variant_residency[key]"
        ),
        "memory_by_lowering_key[key].reservation_bytes",
    ),
    "simulation_taste_stream:global_row_high_word_ignored": (
        SIMULATION_TASTE_STREAM_SOURCE,
        "return jax.random.fold_in(jax.random.fold_in(key, high), low)",
        "return jax.random.fold_in(jax.random.fold_in(key, 0), low)",
    ),
    "compiler_inputs:eliminated_input_counted_by_compiler": (
        COMPILER_INPUTS_SOURCE,
        (
            "return frozenset(path for path, sharding in with_paths if sharding is not None)"
        ),
        "return frozenset(path for path, sharding in with_paths)",
    ),
    "simulation_membership:entry_period_changed": (
        SIMULATION_MEMBERSHIP_SOURCE,
        "entering = starting_periods == period",
        "entering = starting_periods < period",
    ),
    # Native single-array admission and source inventory; these controls are
    # semantic AST rejections, separate from executable topology evidence.
    "native_values:entry_scope_guard_bypassed": (
        "src/_lcm/solution/native_values.py",
        "            type(entry) is not _LazyHdf5Entry",
        "            False and type(entry) is not _LazyHdf5Entry",
    ),
    "native_values:upload_writer_omitted": (
        "src/_lcm/solution/native_values.py",
        "            array_writer=self.array_writer,",
        "            array_writer=None,",
    ),
    "native_values:first_detached_copy_borrowed": (
        "src/_lcm/persistence/solution.py",
        "            return _copy_artifact_array_leaf(\n                leaf=cached.leaves[0],\n                label=self.label,\n                array_copier=array_copier,\n            )",
        "            return cached.leaves[0]",
    ),
    "native_values:second_detached_copy_borrowed": (
        "src/lcm/_solver_api/stores.py",
        "                else _copy_solution_value(\n                    value=value, label=label, array_copier=array_copier\n                )",
        "                else value",
    ),
    "native_values:store_loader_forwarding_omitted": (
        "src/lcm/_solver_api/stores.py",
        "                            value_materializer=value_materializer,",
        "                            value_materializer=None,",
    ),
    "native_values:materialization_precedes_metadata_validation": (
        "src/lcm/model.py",
        "            solution=solution, array_copier=array_copier, native_values=native_values\n        )\n        metadata = solution.metadata",
        "            solution=solution, array_copier=array_copier, native_values=native_values\n        )\n        solution.values._materialize_with_copy(\n            array_copier=array_copier, value_materializer=native_values\n        )\n        metadata = solution.metadata",
    ),
    "native_values:first_load_serialization_omitted": (
        "src/_lcm/persistence/solution.py",
        "        with self._cache.materialization_lock:",
        "        with threading.Lock():",
    ),
    "native_values:observation_lock_spans_upload": (
        "src/_lcm/persistence/solution.py",
        (
            "                private_leaves = tuple(\n"
            "                    _to_jax_without_narrowing(\n"
            "                        array=array,\n"
            '                        label=f"{self.label} leaf {index}",\n'
            "                        array_writer=array_writer,\n"
            "                    )\n"
            "                    for index, array in enumerate(arrays)\n"
            "                )"
        ),
        (
            "                with self._cache.lock:\n"
            "                    private_leaves = tuple(\n"
            "                        _to_jax_without_narrowing(\n"
            "                            array=array,\n"
            '                            label=f"{self.label} leaf {index}",\n'
            "                            array_writer=array_writer,\n"
            "                        )\n"
            "                        for index, array in enumerate(arrays)\n"
            "                    )"
        ),
    ),
    "native_values:upload_readiness_omitted": (
        "src/_lcm/persistence/solution.py",
        "        result.block_until_ready()",
        "        pass  # publish without upload completion",
    ),
    "native_values:checksum_ignored": (
        "src/_lcm/persistence/solution.py",
        '            if actual != str(leaf["sha256"]):',
        '            if False and actual != str(leaf["sha256"]):',
    ),
    "native_values:preupload_dtype_check_omitted": (
        "src/_lcm/persistence/solution.py",
        "        if target_dtype != array.dtype:",
        "        if False and target_dtype != array.dtype:",
    ),
    "native_values:preparation_source_budget_omitted": (
        "src/_lcm/simulation/chunk_admission.py",
        "        live=inputs,\n    )\n    memory = SimulationMemory(",
        "        live=DeviceBufferFootprint(spans={}),\n    )\n    memory = SimulationMemory(",
    ),
    "native_values:live_source_budget_omitted": (
        "src/_lcm/simulation/simulate.py",
        "                    live=inputs,\n                ),\n                inputs=inputs,",
        (
            "                    live=DeviceBufferFootprint(spans={}),\n"
            "                ),\n"
            "                inputs=inputs,"
        ),
    ),
    "native_values:host_receives_accelerator_ceiling": (
        "src/_lcm/simulation/residency.py",
        "if spans and device.platform in platforms",
        "if spans",
    ),
    "constant_program:lower_compile_mesh_context_omitted": (
        "src/_lcm/simulation/runtime.py",
        "        with jax.set_mesh(mesh):",
        "        with jax.set_mesh(None):",
    ),
    # The grouped route dispatches the type-local decision family in place of
    # the ordinary one; both its selection and its dense reducer's Q/F pairing
    # are live candidate transport.
    "type_local_decision:ordinary_decision_shadows_type_local": (
        SIMULATION_PROGRAM_TYPES_SOURCE,
        "MappingProxyType({**self.decision, **self.type_local_decision})",
        "MappingProxyType({**self.type_local_decision, **self.decision})",
    ),
    "type_local_decision:reducer_reads_ordinary_q_and_f": (
        PROCESSING_SOURCE,
        "        Q_and_F_functions=type_local_Q_and_F_functions,\n        has_taste_shocks",
        "        Q_and_F_functions=Q_and_F_functions,\n        has_taste_shocks",
    ),
    "structural_blueprint:cache_hit_ignores_key": (
        STRUCTURAL_BLUEPRINTS_SOURCE,
        "            blueprint = self._entries.get(key)\n",
        "            blueprint = next(reversed(self._entries.values()), None)\n",
    ),
    "structural_blueprint:schema_drops_dtype": (
        STRUCTURAL_BLUEPRINTS_SOURCE,
        "            leaf.shape,\n            leaf.dtype,\n",
        "            leaf.shape,\n",
    ),
}


# These contracts cover admitted Uniform support and its explicit consumer transport.
# Runtime tests establish numerical support, budget refusal and ownership lifetimes.
_UNIFORM_PROCESS_CONTRACTS: dict[str, tuple[str, dict[str, str]]] = _contracts(
    {
        "src/_lcm/engine.py": (
            "SolutionPhase.resolve_process_grids",
            "SolutionPhase.state_action_space",
        ),
        "src/_lcm/simulation/chunk_admission.py": (
            "_ChunkProfiler.profile_widths",
            "_simulation_chunk_profile_key",
            "_independent_outer_candidates",
            "_independent_anchor_widths",
            "_plan_independent_chunks",
            "_profile_independent_candidate",
            "prepare_simulation_chunks",
        ),
        "src/_lcm/simulation/chunk_inputs.py": (
            "prepare_simulation_call_inputs",
            "prepare_simulation_chunk_inputs",
        ),
        "src/_lcm/simulation/compile.py": ("bind_simulation_runtime",),
        "src/_lcm/simulation/entry_allocations.py": (
            "SimulationEntryAllocations.__post_init__",
            "SimulationEntryAllocations.snapshot",
            "SimulationEntryAllocations.solve_input_roots",
            "SimulationEntryAllocations.close",
            "_EntryFootprint.__call__",
        ),
        "src/_lcm/simulation/initial_conditions.py": (
            "validate_simulation_inputs",
            "validate_initial_conditions",
            "_collect_feasibility_errors",
            "_check_regime_feasibility",
            "_regime_feasibility_mask",
        ),
        "src/_lcm/simulation/simulate.py": ("simulate", "_simulate_subject_chunk"),
        "src/_lcm/solution/backward_induction.py": (
            "_period_transfer_scratch_reservations",
            "_continuous_value_replica_required",
            "solve",
            "_build_continuation_templates",
            "_iter_edge_topologies",
            "_build_base_state_action_spaces",
            "_compile_all_functions",
            "_prepare_solve_programs",
            "_resolve_output_layouts_and_lowering_keys",
            "_build_structural_blueprint",
            "_bind_structural_blueprint",
            "_structural_key",
        ),
        "src/_lcm/solution/diagnostics.py": (
            "_emit_post_loop_diagnostics",
            "_raise_first_nan_row",
            "_raise_at",
            "_reconstruct_next_regime_to_V_arr",
        ),
        "src/_lcm/solution/fingerprint.py": (
            "fingerprint_solution_support",
            "fingerprint_model",
            "_grid_support",
        ),
        "src/_lcm/solution/model_authority.py": ("build_solution_authority",),
        "src/_lcm/solution/preconditions.py": (
            "check_pareto_weights",
            "_check_one_regimes_weights",
        ),
        "src/_lcm/solution/v_topology.py": ("_get_regime_V_shapes_and_shardings",),
        "src/_lcm/transition_checks.py": (
            "_ValidationSummary.state_action_space",
            "validate_transitions",
            "_validate_transition_sequence",
            "validate_regime_transition_probs_all_periods",
            "_validate_regime_transition_single",
            "_evaluate_regime_probability_law",
            "_regime_probability_law",
            "_check_and_release_regime_probability",
            "_validate_regime_transition_probs",
            "validate_state_transitions_all_periods",
            "validate_joint_transitions_all_periods",
            "_own_transition_outputs",
            "_set_transition_outputs",
            "_transition_owner_tree",
            "_validate_joint_laws",
            "_check_joint_support_schema",
            "_evaluate_joint_support",
            "_validate_joint_support",
            "_validate_joint_probabilities",
            "_evaluate_joint_weights",
            "_joint_weight_law",
            "_validate_state_transition_single",
            "_check_and_release_state_probability",
            "_evaluate_state_probability_law",
            "_evaluate_admitted_transition_producer",
            "_state_probability_law",
            "_abstract_transition_operand",
            "_check_state_probs",
        ),
        "src/lcm/model.py": (
            "_validate_sharded_state_capability",
            "_supports_continuous_sharding_vocabulary",
            "_supports_unsharded_continuous_process",
            "Model.__init__",
            "Model._declared_solution_authority",
            "Model._model_fingerprint",
            "Model.solve",
            "Model._solve_from_flat_params",
            "Model._solve_compiled",
            "Model._resolve_solution_result",
            "Model._consume_owned_solution",
            "Model._consume_foreign_solution",
            "Model._check_solution_result_structure",
            "Model._build_external_replay_readers",
            "Model.simulate",
            "Model._open_entry_allocations",
            "Model._resolve_compile_batch_size",
        ),
        "src/_lcm/processes/grid_resolution.py": (
            "ProcessGridResolver.supports",
            "ProcessGridResolver.__call__",
        ),
        "src/_lcm/simulation/process_grids.py": (
            "_shared_uniform_operations",
            "SimulationProcessGrids.supports",
            "SimulationProcessGrids.array_roots",
            "SimulationProcessGrids.__call__",
            "SimulationProcessGrids.seal",
            "SimulationProcessGrids.close",
            "SimulationProcessGrids._produce",
            "SimulationProcessGrids.snapshot",
            "_uniform_parameters",
            "_parameter_bytes",
            "_abstract_grid_parameter",
            "_compute_uniform_grid",
            "SimulationProcessGrids._produce_normal",
            "_normal_parameters",
            "_normal_fixed_identity",
            "_compute_normal_stage",
            "SimulationProcessGrids._produce_staged",
            "_complete_process_parameters",
            "_process_fixed_identity",
            "_staged_parameter_is_weak",
            "_trace_process_jaxpr",
            "_process_grid_call",
            "_validated_process_recipe",
            "_validate_attached_process_value",
            "_validated_process_operand",
            "_validate_process_equation",
            "_validate_linspace_jaxpr",
            "_jaxpr_schema",
            "_root_equation_schema",
            "_equation_schema",
            "_graph_atom_schema",
            "_aval_schema",
            "_aval_shape",
            "_aval_dtype",
            "_read_process_operand",
            "_compute_process_stage",
        ),
    }
)


_GROUPED_MAPPER_CONTRACTS = _contracts(
    {
        "src/_lcm/utils/dispatchers.py": (
            "tiled_productmap",
            "_CountBroadcastExtentInWidth.__call__",
            "_TiledProductMap.__call__",
            "_map_grouped_product",
            "_MapOverFinalCoordinate.__call__",
            "_map_whole_product",
            "_final_mapper",
            "_MapWholeCoordinate.__call__",
            "_EvaluateTiledCell.__call__",
            "_restore_product_axes",
            "map_over_leading_axis",
            "_RestoreProductAxisOrder.__call__",
            "_transpose_product_axes",
        )
    }
)


def _grouped_mapper_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Pin C-order cells, bounded two-axis windows, and unchanged output roles."""
    surface, callables = _GROUPED_MAPPER_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="grouped mapper", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("grouped mapper: module bindings or mapper schema changed")
    return errors


_GROUPED_MAPPER_MUTATIONS = {
    "grouped_mapper:int32_limit_inflated": ("<module>", "2**31 - 1", "2**63 - 1"),
    "grouped_mapper:extent_guard_omitted": (
        "_TiledProductMap.__call__",
        "n_cells < 1 or n_cells > _MAX_FLAT_CELL_INDEX",
        "False",
    ),
    "grouped_mapper:grouped_reuse_disabled": (
        "_TiledProductMap.__call__",
        "len(self.variables) > 1",
        "False",
    ),
    "grouped_mapper:nonmapped_arguments_omitted": (
        "_TiledProductMap.__call__",
        "MappingProxyType({name: value for name, value in kwargs.items() if name not in self.variables})",
        "MappingProxyType({})",
    ),
    "grouped_mapper:inner_window_unbounded": (
        "_map_grouped_product",
        "min(width, shape[-1])",
        "shape[-1]",
    ),
    "grouped_mapper:outer_window_unbounded": (
        "_map_grouped_product",
        "max(1, width // inner_width)",
        "math.prod(shape[:-1])",
    ),
    "grouped_mapper:outer_window_rounds_up": (
        "_map_grouped_product",
        "width // inner_width",
        "(width + inner_width - 1) // inner_width",
    ),
    "grouped_mapper:prefix_strides_include_final_axis": (
        "_map_grouped_product",
        "shape[index + 1:-1]",
        "shape[index + 1:]",
    ),
    "grouped_mapper:prefix_variable_order_reversed": (
        "_map_grouped_product",
        "variables[:-1]",
        "tuple(reversed(variables[:-1]))",
    ),
    "grouped_mapper:prefix_grid_omitted": (
        "_map_grouped_product",
        "coordinates[:-1]",
        "coordinates[:-2]",
    ),
    "grouped_mapper:final_variable_changed": (
        "_map_grouped_product",
        "variables[-1]",
        "variables[0]",
    ),
    "grouped_mapper:final_grid_changed": (
        "_map_grouped_product",
        "coordinates[-1]",
        "coordinates[0]",
    ),
    "grouped_mapper:prefix_extent_truncated": (
        "_map_grouped_product",
        "math.prod(shape[:-1])",
        "math.prod(shape[:-1]) - 1",
    ),
    "grouped_mapper:trailing_roles_shifted": (
        "_restore_product_axes",
        "value.shape[n_flat_axes:]",
        "value.shape[1:]",
    ),
    "grouped_mapper:grouped_restore_consumes_one_axis": (
        "_map_grouped_product",
        "2",
        "1",
    ),
    "grouped_mapper:final_window_unbounded": (
        "_MapOverFinalCoordinate.__call__",
        "self.width",
        "self.coordinate.shape[0]",
    ),
    "grouped_mapper:decoded_coordinate_replaced": (
        "_EvaluateTiledCell.__call__",
        "coordinate[(index // stride) % coordinate.shape[0]]",
        "coordinate[0]",
    ),
    "grouped_mapper:cell_arguments_dropped": (
        "_EvaluateTiledCell.__call__",
        "self.func(**self.arguments, **cell)",
        "self.func(**cell)",
    ),
}

EXPECTED_GROUPED_MAPPER_MUTATION_COUNT = 18
EXPECTED_GROUPED_MAPPER_MUTATION_NAMES_SHA256 = (
    "7a71b3f1917dc1ac14cefc08e6b36c8d3a09522bc20d7ef775686a2878db8189"
)


def grouped_mapper_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build separate controls without changing historical mutation populations."""
    result = {}
    relative = DISPATCHERS_SOURCE
    original = (repo_root / relative).read_text(encoding="utf-8")
    for name, (qualname, old, new) in _GROUPED_MAPPER_MUTATIONS.items():
        tree = ast.parse(original, filename=relative)
        if qualname == "<module>":
            function = tree
        elif "." in qualname:
            class_name, method_name = qualname.split(".", maxsplit=1)
            _, function = _method_definition(
                tree=tree, class_name=class_name, method_name=method_name
            )
        else:
            function = _definition(tree=tree, name=qualname)
        expected = ast.dump(ast.parse(old, mode="eval").body, include_attributes=False)
        matches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.expr)
            and ast.dump(node, include_attributes=False) == expected
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{name}: expected one expression anchor, found {len(matches)}"
            )
        replacement = ast.parse(new, mode="eval").body
        target = matches[0]
        for parent in ast.walk(function):
            for attribute, value in ast.iter_fields(parent):
                if value is target:
                    setattr(parent, attribute, copy.deepcopy(replacement))
                elif isinstance(value, list):
                    setattr(
                        parent,
                        attribute,
                        [
                            copy.deepcopy(replacement) if item is target else item
                            for item in value
                        ],
                    )
        mutated = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        ast.parse(mutated, filename=relative)
        result[name] = {"path": relative, "source": mutated}
    return result


_GROUPED_GUARD_MUTATIONS = {
    "grouped_guard:eligibility_omitted": (
        "src/_lcm/utils/dispatchers.py",
        "_TiledProductMap.__call__",
        "expression",
        "width // min(width, shape[-1]) > 1",
        "True",
        1,
    ),
    "grouped_guard:eligibility_inverted": (
        "src/_lcm/utils/dispatchers.py",
        "_TiledProductMap.__call__",
        "expression",
        "width // min(width, shape[-1]) > 1",
        "width // min(width, shape[-1]) <= 1",
        1,
    ),
    "grouped_guard:threshold_lowered": (
        "src/_lcm/utils/dispatchers.py",
        "_TiledProductMap.__call__",
        "expression",
        "width // min(width, shape[-1]) > 1",
        "width // min(width, shape[-1]) > 0",
        1,
    ),
    "grouped_guard:threshold_raised": (
        "src/_lcm/utils/dispatchers.py",
        "_TiledProductMap.__call__",
        "expression",
        "width // min(width, shape[-1]) > 1",
        "width // min(width, shape[-1]) > 2",
        1,
    ),
    "grouped_guard:fallback_width_unbounded": (
        "src/_lcm/utils/dispatchers.py",
        "_TiledProductMap.__call__",
        "keyword",
        "batch_size",
        "n_cells",
        1,
    ),
    "grouped_guard:fallback_strides_truncated": (
        "src/_lcm/utils/dispatchers.py",
        "_TiledProductMap.__call__",
        "expression",
        "shape[index + 1:]",
        "shape[index + 1:-1]",
        1,
    ),
}

EXPECTED_GROUPED_GUARD_MUTATION_COUNT = 6
EXPECTED_GROUPED_GUARD_MUTATION_NAMES_SHA256 = (
    "e866f1cbeda39679ae30891885e1101904d48a5a571add28552fdacc7dd2f5ca"
)


def grouped_guard_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Keep route-eligibility controls separate from the accepted mapper population."""
    return _callable_mutation_specs(
        repo_root=repo_root, mutations=_GROUPED_GUARD_MUTATIONS
    )


_ACTION_GRID_CONTRACTS = _contracts(
    {
        "src/_lcm/simulation/action_grids.py": (
            "PreflightActionGrids.resolve",
            "PreflightActionGrids.close",
        ),
        "src/_lcm/simulation/initial_conditions.py": (
            "validate_simulation_inputs",
            "validate_initial_conditions",
            "_collect_feasibility_errors",
            "_check_regime_feasibility",
            "_regime_feasibility_mask",
            "_build_flat_action_grid",
        ),
    }
)


def _action_grid_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Pin preflight producer admission, reuse and cleanup independently of byte seals."""
    surface, callables = _ACTION_GRID_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="action grid admission", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("action grid admission: module bindings or owner schema changed")
    return errors


_ACTION_GRID_MUTATIONS = {
    "action_grid:budget_guard_bypassed": (
        "src/_lcm/simulation/action_grids.py",
        "PreflightActionGrids.resolve",
        "expression",
        "self.memory is None",
        "True",
        1,
    ),
    "action_grid:input_identity_layout_omitted": (
        "src/_lcm/simulation/action_grids.py",
        "PreflightActionGrids.resolve",
        "expression",
        (
            "(action_names, tuple((id(grids[name]), "
            "grids[name].sharding) for name in action_names))"
        ),
        "(action_names,)",
        1,
    ),
    "action_grid:input_owners_omitted": (
        "src/_lcm/simulation/action_grids.py",
        "PreflightActionGrids.resolve",
        "expression",
        "(grids, flat)",
        "({}, flat)",
        2,
    ),
    "action_grid:current_roots_omitted": (
        "src/_lcm/simulation/action_grids.py",
        "PreflightActionGrids.resolve",
        "expression",
        "self.memory.set_derived(retained_arrays)",
        "None",
        1,
    ),
    "action_grid:cumulative_publication_omitted": (
        "src/_lcm/simulation/action_grids.py",
        "PreflightActionGrids.resolve",
        "expression",
        "self.memory.publish(tree=(grids, flat))",
        "None",
        1,
    ),
    "action_grid:entry_and_retry_resolver_omitted": (
        "src/_lcm/simulation/initial_conditions.py",
        "validate_simulation_inputs",
        "keyword",
        "action_grid_resolver",
        "None",
        2,
    ),
    "action_grid:exit_cleanup_omitted": (
        "src/_lcm/simulation/initial_conditions.py",
        "validate_simulation_inputs",
        "expression",
        "contextlib.closing",
        "contextlib.nullcontext",
        1,
    ),
    "action_grid:binding_release_omitted": (
        "src/_lcm/simulation/action_grids.py",
        "PreflightActionGrids.close",
        "expression",
        "self.bindings.clear()",
        "None",
        1,
    ),
    "action_grid:publication_release_omitted": (
        "src/_lcm/simulation/action_grids.py",
        "PreflightActionGrids.close",
        "expression",
        "self.memory.replace_outputs(tree=())",
        "None",
        1,
    ),
    "action_grid:cartesian_order_changed": (
        "src/_lcm/simulation/initial_conditions.py",
        "_build_flat_action_grid",
        "keyword",
        "indexing",
        "'xy'",
        1,
    ),
}

EXPECTED_ACTION_GRID_MUTATION_COUNT = 10
EXPECTED_ACTION_GRID_MUTATION_NAMES_SHA256 = (
    "7e488a937dac3679ca20bb050837c968cd491e53f431ddea0ffab388a2d19d50"
)


def _uniform_process_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Protect reviewed support admission independently of source-byte resealing."""
    surface, callables = _UNIFORM_PROCESS_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="uniform process admission", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(
            "uniform process admission: module bindings or owner schema changed"
        )
    return errors


# This group preserves the independently pinned candidate and supplemental families.
_UNIFORM_PROCESS_MUTATIONS = {
    "uniform_grid:replicated_layout_omitted": (
        ENGINE_SOURCE,
        "SolutionPhase.resolve_process_grids",
        "expression",
        "jax.sharding.NamedSharding(plan.mesh, jax.sharding.PartitionSpec())",
        "jax.sharding.SingleDeviceSharding(devices[0])",
        1,
    ),
    "uniform_grid:producer_output_layout_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce",
        "keyword",
        "output_sharding",
        "None",
        1,
    ),
    "uniform_grid:selected_device_ignored": (
        ENGINE_SOURCE,
        "SolutionPhase.resolve_process_grids",
        "expression",
        "jax.sharding.SingleDeviceSharding(devices[0])",
        "jax.sharding.SingleDeviceSharding(jax.devices()[0])",
        1,
    ),
    "uniform_grid:layout_identity_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.__call__",
        "expression",
        "(spec.n_points, required, tuple((name, _parameter_bytes(value)) for name, value in sorted(complete.items())))",
        "(spec.n_points, tuple((name, _parameter_bytes(value)) for name, value in sorted(complete.items())))",
        1,
    ),
    "uniform_grid:support_extent_identity_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.__call__",
        "expression",
        "(spec.n_points, required, tuple((name, _parameter_bytes(value)) for name, value in sorted(complete.items())))",
        "(required, tuple((name, _parameter_bytes(value)) for name, value in sorted(complete.items())))",
        1,
    ),
    "uniform_grid:foreign_owner_budget_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce",
        "expression",
        "tuple(dict.fromkeys((*self.devices, *current.spans)))",
        "self.devices",
        1,
    ),
    "uniform_grid:resolver_result_rank_changed": (
        PROCESS_GRID_RESOLUTION_SOURCE,
        "ProcessGridResolver.__call__",
        "expression",
        "Float1D",
        "ScalarFloat",
        1,
    ),
    "uniform_grid:exact_family_broadened": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.supports",
        "expression",
        "type(spec) is UniformIIDProcess",
        "True",
        1,
    ),
    "uniform_grid:retained_support_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.array_roots",
        "expression",
        "tuple(self.grids.values())",
        "()",
        1,
    ),
    "uniform_grid:binding_owners_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.array_roots",
        "expression",
        "tuple(binding.parameters for binding in self.bindings.values())",
        "()",
        1,
    ),
    "uniform_grid:endpoint_content_ignored": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_parameter_bytes",
        "expression",
        "array.tobytes()",
        "b''",
        1,
    ),
    "uniform_grid:endpoint_dtype_ignored": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_parameter_bytes",
        "expression",
        "array.dtype.str",
        "'float32'",
        1,
    ),
    "uniform_grid:endpoint_shape_ignored": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_parameter_bytes",
        "expression",
        "array.shape",
        "()",
        1,
    ),
    "uniform_grid:sealed_support_changed": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.__call__",
        "expression",
        "self.sealed",
        "False",
        1,
    ),
    "uniform_grid:binding_owner_not_retained": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.__call__",
        "expression",
        "MappingProxyType(dict(parameters))",
        "MappingProxyType({})",
        1,
    ),
    "uniform_grid:dispatch_before_admission": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce",
        "before_assignment",
        "plan",
        "_compute_uniform_grid(parameters=placed['parameters'], n_points=n_points)",
        1,
    ),
    "uniform_grid:external_residency_ignored": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce",
        "expression",
        "max(external.values())",
        "0",
        1,
    ),
    "uniform_grid:peak_admission_ignored": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce",
        "keyword",
        "budget_bytes",
        "None",
        2,
    ),
    "uniform_grid:shared_cache_keeps_operands": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce",
        "expression",
        "abstract",
        "placed",
        2,
    ),
    "uniform_grid:readiness_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce",
        "expression",
        "plan.compiled.executable(**placed).block_until_ready()",
        "plan.compiled.executable(**placed)",
        1,
    ),
    "uniform_grid:caller_residency_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.snapshot",
        "expression",
        "self.live_footprint()",
        "DeviceBufferFootprint(spans={})",
        1,
    ),
    "uniform_grid:grid_cleanup_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.close",
        "expression",
        "self.grids.clear()",
        "None",
        1,
    ),
    "uniform_grid:binding_cleanup_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.close",
        "expression",
        "self.bindings.clear()",
        "None",
        1,
    ),
    "uniform_grid:fixed_endpoint_dtype_changed": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_uniform_parameters",
        "expression",
        "canonical_float_dtype()",
        "np.float16",
        1,
    ),
    "uniform_grid:fixed_endpoint_shifted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_uniform_parameters",
        "expression",
        "np.asarray(value, dtype=dtype)",
        "np.asarray(value + 1, dtype=dtype)",
        1,
    ),
    "uniform_grid:support_order_reversed": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_compute_uniform_grid",
        "expression",
        "UniformIIDProcess(n_points=n_points).compute_gridpoints(**parameters)",
        "UniformIIDProcess(n_points=n_points).compute_gridpoints(**parameters)[::-1]",
        1,
    ),
    "uniform_grid:failure_lifetime_cycle": (
        SIMULATION_ENTRY_ALLOCATIONS_SOURCE,
        "SimulationEntryAllocations.__post_init__",
        "expression",
        "_EntryFootprint(owner=weakref.ref(self))",
        "self.snapshot",
        1,
    ),
    "uniform_grid:entry_roots_omitted": (
        SIMULATION_ENTRY_ALLOCATIONS_SOURCE,
        "SimulationEntryAllocations.snapshot",
        "expression",
        "self.process_grid_resolver.array_roots",
        "()",
        1,
    ),
    "uniform_grid:automatic_solve_roots_omitted": (
        SIMULATION_ENTRY_ALLOCATIONS_SOURCE,
        "SimulationEntryAllocations.solve_input_roots",
        "expression",
        "self.process_grid_resolver.array_roots",
        "()",
        1,
    ),
    "uniform_grid:entry_seal_omitted": (
        MODEL_SOURCE,
        "Model.simulate",
        "expression",
        "process_grid_resolver.seal()",
        "None",
        1,
    ),
    "uniform_grid:engine_scope_broadened": (
        ENGINE_SOURCE,
        "SolutionPhase.resolve_process_grids",
        "expression",
        "process_grid_resolver.supports(spec)",
        "True",
        1,
    ),
    "uniform_grid:engine_recomputes_admitted_support": (
        ENGINE_SOURCE,
        "SolutionPhase.state_action_space",
        "expression",
        "name not in state_replacements",
        "True",
        1,
    ),
    "uniform_grid:fingerprint_transport_omitted": (
        SUPPORT_FINGERPRINT_SOURCE,
        "_grid_support",
        "expression",
        "process_grid_resolver",
        "None",
        1,
    ),
    "uniform_grid:authority_transport_omitted": (
        SUPPORT_AUTHORITY_SOURCE,
        "build_solution_authority",
        "expression",
        "process_grid_resolver",
        "None",
        2,
    ),
    "uniform_grid:pareto_transport_omitted": (
        SUPPORT_PRECONDITIONS_SOURCE,
        "_check_one_regimes_weights",
        "expression",
        "process_grid_resolver",
        "None",
        1,
    ),
    "uniform_grid:diagnostic_transport_omitted": (
        SUPPORT_DIAGNOSTICS_SOURCE,
        "_reconstruct_next_regime_to_V_arr",
        "expression",
        "process_grid_resolver",
        "None",
        1,
    ),
    "uniform_grid:validation_transport_omitted": (
        SUPPORT_TRANSITION_CHECKS_SOURCE,
        "_ValidationSummary.state_action_space",
        "expression",
        "self.process_grid_resolver",
        "None",
        1,
    ),
}


_NORMAL_PROCESS_MUTATIONS = {
    "normal_grid:profile_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.supports",
        "expression",
        "type(spec) is NormalIIDProcess and not spec.gauss_hermite",
        "False",
        1,
    ),
    "normal_grid:quadrature_scope_broadened": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.supports",
        "expression",
        "not spec.gauss_hermite",
        "True",
        1,
    ),
    "normal_grid:fixed_binding_identity_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.__call__",
        "expression",
        "_normal_fixed_identity(spec)",
        "()",
        1,
    ),
    "normal_grid:weak_array_identity_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.__call__",
        "expression",
        "getattr(value, 'weak_type', False)",
        "False",
        1,
    ),
    "normal_grid:weak_host_identity_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.__call__",
        "expression",
        "type(value) in (bool, int, float)",
        "False",
        1,
    ),
    "normal_grid:temporary_roots_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.array_roots",
        "expression",
        "tuple(self.temporary_roots)",
        "()",
        1,
    ),
    "normal_grid:fixed_parameter_promotion_changed": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_normal_parameters",
        "expression",
        "np.asarray(value, dtype=np.int32) if isinstance(value, bool | int) else value",
        "np.asarray(value, dtype=canonical_float_dtype())",
        1,
    ),
    "normal_grid:fixed_bytes_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_normal_fixed_identity",
        "expression",
        "_parameter_bytes(value)",
        "None",
        1,
    ),
    "normal_grid:fixed_type_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_normal_fixed_identity",
        "expression",
        "type(value)",
        "None",
        1,
    ),
    "normal_grid:placed_parameter_owner_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce_normal",
        "expression",
        "self.temporary_roots.append(placed)",
        "None",
        1,
    ),
    "normal_grid:offset_owner_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce_normal",
        "expression",
        "self.temporary_roots.append(offset)",
        "None",
        1,
    ),
    "normal_grid:endpoint_owner_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce_normal",
        "expression",
        "self.temporary_roots.append(value)",
        "None",
        1,
    ),
    "normal_grid:failure_cleanup_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce_normal",
        "expression",
        "self.temporary_roots.clear()",
        "None",
        1,
    ),
    "normal_grid:lower_sign_changed": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_compute_normal_stage",
        "expression",
        "parameters['left'] - parameters['right']",
        "parameters['left'] + parameters['right']",
        1,
    ),
    "normal_grid:upper_sign_changed": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "_compute_normal_stage",
        "expression",
        "parameters['left'] + parameters['right']",
        "parameters['left'] - parameters['right']",
        1,
    ),
    "normal_grid:stage_identity_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids._produce",
        "expression",
        "MappingProxyType({'n_points': n_points, 'stage': stage})",
        "MappingProxyType({'n_points': n_points})",
        1,
    ),
    "normal_grid:stage_admission_bypassed": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.__call__",
        "expression",
        "self._produce_normal(parameters=complete, n_points=spec.n_points, required=required)",
        "spec.compute_gridpoints(**complete)",
        1,
    ),
}

EXPECTED_NORMAL_PROCESS_MUTATION_COUNT = 17
EXPECTED_NORMAL_PROCESS_MUTATION_NAMES_SHA256 = (
    "4cccec12c8b31fcfd18df9c8ec2459c96c8300e4b0f76118c1f8fcc40a79858a"
)


def normal_process_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build normal stage, identity and temporary-ownership semantic controls."""
    return _callable_mutation_specs(
        repo_root=repo_root, mutations=_NORMAL_PROCESS_MUTATIONS
    )


def uniform_process_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build named producer controls without altering either historical population."""
    return _callable_mutation_specs(
        repo_root=repo_root, mutations=_UNIFORM_PROCESS_MUTATIONS
    )


def action_grid_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build the independent Cartesian preflight admission mutation population."""
    return _callable_mutation_specs(
        repo_root=repo_root, mutations=_ACTION_GRID_MUTATIONS
    )


_FEASIBILITY_MUTATIONS = {
    "feasibility:summary_and_serial_owner_omitted": (
        INITIAL_CONDITIONS_SOURCE,
        "validate_simulation_inputs",
        "keyword",
        "memory",
        "None",
        5,
    ),
    "feasibility:joint_admission_bypassed": (
        INITIAL_CONDITIONS_SOURCE,
        "_batched_feasibility_check",
        "expression",
        "memory is not None",
        "False",
        1,
    ),
    "feasibility:constant_admission_bypassed": (
        INITIAL_CONDITIONS_SOURCE,
        "_evaluate_constant_feasibility",
        "expression",
        "memory is not None",
        "False",
        1,
    ),
    "feasibility:diagnostic_gather_and_predicates_unowned": (
        INITIAL_CONDITIONS_SOURCE,
        "_per_constraint_feasibility",
        "keyword",
        "memory",
        "None",
        3,
    ),
    "feasibility:placement_budget_omitted": (
        INITIAL_CONDITIONS_SOURCE,
        "_run_profiled_feasibility",
        "keyword",
        "budget_bytes",
        "None",
        2,
    ),
    "feasibility:live_owners_omitted": (
        INITIAL_CONDITIONS_SOURCE,
        "_run_profiled_feasibility",
        "expression",
        "max(external.values())",
        "0",
        1,
    ),
    "feasibility:concrete_inputs_closed_over": (
        INITIAL_CONDITIONS_SOURCE,
        "_run_profiled_feasibility",
        "expression",
        "dict(placed)",
        "{}",
        1,
    ),
    "feasibility:unprofiled_dispatch": (
        INITIAL_CONDITIONS_SOURCE,
        "_run_profiled_feasibility",
        "expression",
        "executable(**placed)",
        "function(**placed)",
        1,
    ),
    "feasibility:readiness_omitted": (
        INITIAL_CONDITIONS_SOURCE,
        "_run_profiled_feasibility",
        "expression",
        "jax.block_until_ready(result)",
        "None",
        1,
    ),
    "feasibility:result_lifetime_omitted": (
        INITIAL_CONDITIONS_SOURCE,
        "_run_profiled_feasibility",
        "expression",
        "memory.hold(tree=result)",
        "None",
        1,
    ),
    "feasibility:compiler_input_ownership_dropped": (
        SIMULATION_HOST_SOURCE,
        "_lower_operation",
        "keyword",
        "keep_unused",
        "False",
        2,
    ),
}


EXPECTED_FEASIBILITY_MUTATION_COUNT = 11
EXPECTED_FEASIBILITY_MUTATION_NAMES_SHA256 = (
    "b2e6c3e575bb8eacb5f88a78f3315cf5b144918af506e228aba95a4e0e258ea7"
)


def feasibility_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Protect feasibility producers without changing historical control sets."""
    return _callable_mutation_specs(
        repo_root=repo_root, mutations=_FEASIBILITY_MUTATIONS
    )


_ALLOCATION_RESERVATION_MUTATIONS = {
    "allocation_reservation:temporary_omitted": (
        WORKSPACE_PLANNING_SOURCE,
        "CompilerMemoryRecord.allocation_bytes",
        "expression",
        "self.temporary_bytes",
        "0",
        1,
    ),
    "allocation_reservation:alias_added": (
        WORKSPACE_PLANNING_SOURCE,
        "CompilerMemoryRecord.allocation_bytes",
        "expression",
        "self.argument_bytes + self.output_bytes - self.alias_bytes",
        "self.argument_bytes + self.output_bytes + self.alias_bytes",
        1,
    ),
    "allocation_reservation:alias_bound_relaxed": (
        WORKSPACE_PLANNING_SOURCE,
        "CompilerMemoryRecord.__post_init__",
        "expression",
        "min(self.argument_bytes, self.output_bytes)",
        "max(self.argument_bytes, self.output_bytes)",
        1,
    ),
    "allocation_reservation:non_record_report_accepted": (
        WORKSPACE_PLANNING_SOURCE,
        "_compiler_memory_analysis",
        "expression",
        "not isinstance(report, CompilerMemoryReport)",
        "False",
        1,
    ),
    "allocation_reservation:missing_attribute_counter_zeroed": (
        WORKSPACE_PLANNING_SOURCE,
        "_allocation_record",
        "expression",
        "getattr(record, name)",
        "getattr(record, name, 0)",
        1,
    ),
    "allocation_reservation:host_space_ignored": (
        WORKSPACE_PLANNING_SOURCE,
        "_fail_if_host_allocations",
        "expression",
        'any(value for name, value in values.items() if name.startswith("host_"))',
        "False",
        1,
    ),
    "allocation_reservation:devices_summed": (
        WORKSPACE_PLANNING_SOURCE,
        "CompilerMemoryReservation.reservation_bytes",
        "expression",
        "max(record.reservation_bytes for record in self.records)",
        "sum(record.reservation_bytes for record in self.records)",
        1,
    ),
    "allocation_reservation:raw_peak_relabeled": (
        WORKSPACE_PLANNING_SOURCE,
        "CompilerMemoryReservation.peak_bytes",
        "expression",
        "record.peak_bytes",
        "record.reservation_bytes",
        1,
    ),
    "allocation_reservation:planner_peak_only": (
        WORKSPACE_PLANNING_SOURCE,
        "plan_workspace",
        "expression",
        "memory.reservation_bytes + candidate_resident",
        "memory.peak_bytes + candidate_resident",
        1,
    ),
    "allocation_reservation:chunk_peak_only": (
        "src/_lcm/simulation/chunk_planning.py",
        "_required_bytes",
        "expression",
        "stage.reservation_bytes",
        "stage.peak_bytes",
        1,
    ),
    "allocation_reservation:solve_wave_peak_only": (
        BACKWARD_INDUCTION_SOURCE,
        "_compile_all_functions",
        "expression",
        "memory_by_lowering_key[lowering_key].reservation_bytes + resident",
        "memory_by_lowering_key[lowering_key].peak_bytes + resident",
        1,
    ),
    "allocation_reservation:donation_variant_peak_only": (
        BACKWARD_INDUCTION_SOURCE,
        "_compile_all_functions",
        "expression",
        "memory_by_lowering_key[key].reservation_bytes + variant_residency[key]",
        "memory_by_lowering_key[key].peak_bytes + variant_residency[key]",
        1,
    ),
    "allocation_reservation:operation_cache_peak_only": (
        SIMULATION_HOST_SOURCE,
        "_operation_memory",
        "expression",
        "compiled.memory",
        "compiled.memory.peak_bytes",
        1,
    ),
    "allocation_reservation:runtime_peak_only": (
        SIMULATION_RUNTIME_SOURCE,
        "_with_compiler_memory",
        "expression",
        (
            "compiler_memory_reservation("
            "compiled=cast('jax.stages.Compiled', compiled.executable), "
            "widths=compiled.widths)"
        ),
        (
            "compiler_peak_bytes("
            "compiled=cast('jax.stages.Compiled', compiled.executable), "
            "widths=compiled.widths)"
        ),
        1,
    ),
}

EXPECTED_ALLOCATION_RESERVATION_MUTATION_COUNT = 14
EXPECTED_ALLOCATION_RESERVATION_MUTATION_NAMES_SHA256 = (
    "e2c9b6ce3ec5e2abd7e258aa9fcbe24deff0f3dfa1c46ce940f2c7f25f0ded03"
)


def allocation_reservation_mutation_specs(
    *, repo_root: Path
) -> dict[str, dict[str, str]]:
    """Keep represented-allocation controls separately identifiable."""
    return _callable_mutation_specs(
        repo_root=repo_root, mutations=_ALLOCATION_RESERVATION_MUTATIONS
    )


# The action-partitioned singleton solve is its own corridor. GridSearch selects
# it only for an admitted request; the device-mapped kernel hands Q_and_F
# exactly its declared arguments and the planner-bound width; every device
# streams the contiguous run of whole blocks the layout assigns it, every block
# inside the scan as on the unpartitioned stream, with blocks it does not own
# and padded slots infeasible; the per-device hard-max accumulators are
# gathered over the action axis and merged in ascending partition order with the
# exact hard-max law. Shared stream helpers are pinned here too, at the digests
# the unpartitioned corridor pins them to, so the route is proved end to end.
_ACTION_PARTITION_CONTRACTS = _contracts(
    {
        GRID_SEARCH_SOURCE: (
            "GridSearch.build_period_kernels",
            "_action_partition_mesh",
            "_classify_action_streaming",
            "_select_action_width_keyword",
            "_select_cell_width_keyword",
        ),
        MAX_Q_SOURCE: (
            "get_action_partitioned_max_Q_over_a",
            "_ActionPartitionedMaxQOverA.__call__",
            "_arguments_named",
            "_OnActionPartitionAxis.__call__",
            "_call_with_operands",
            "_get_extra_param_names",
            "_fail_if_action_width_keyword_collides",
        ),
        ACTION_STREAMING_SOURCE: (
            "build_partitioned_streaming_max_Q_over_a",
            "merge_partition_accumulators",
            "_fail_if_not_positive_int",
            "ActionPartitionLayout.__post_init__",
            "ActionPartitionLayout.n_blocks",
            "ActionPartitionLayout.blocks_per_partition",
            "ActionPartitionLayout.block_range",
            "ActionPartitionLayout.action_interval",
            "_PartitionedStreamingHardMax.__call__",
            "_PartitionedStreamingHardMax.local",
            "_evaluate_owned_block",
            "_validate_streaming_configuration",
            "_prepare_action_call",
            "_evaluate_block",
            "_evaluate_one_action",
            "_decode_action",
            "_validate_block_Q_and_F",
            "_trace_block",
            "_empty_reduction",
            "_typed_like",
            "_scan_one_block",
            "_add_block",
        ),
        ACTION_REDUCTION_SOURCE: (
            "HardMaxReduction.initialize",
            "HardMaxReduction.add",
            "HardMaxReduction.merge",
            "HardMaxReduction.finalize",
            "_reduce_block",
        ),
    }
)


def _action_partition_errors(*, tree: ast.Module, source: str) -> list[str]:
    """Pin the action-partitioned hard-max route independently of byte seals."""
    surface, callables = _ACTION_PARTITION_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="action-partitioned route", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(
            "action-partitioned route: module bindings or kernel schemas changed"
        )
    return errors


# Each control seeds one defect into the action-partitioned hard-max route: the
# GridSearch selection, the device-mapped kernel, the contiguous block layout,
# the per-device stream, the accumulator exchange, and the ordered merge.
_ACTION_PARTITION_MUTATIONS = {
    "action_partition:merge_order_reversed": (
        ACTION_STREAMING_SOURCE,
        "_PartitionedStreamingHardMax.__call__",
        "expression",
        "tuple(range(self.n_partitions))",
        "tuple(reversed(range(self.n_partitions)))",
        1,
    ),
    "action_partition:merge_starts_at_last_partition": (
        ACTION_STREAMING_SOURCE,
        "merge_partition_accumulators",
        "expression",
        "order[0]",
        "order[-1]",
        1,
    ),
    "action_partition:merge_skips_last_partition": (
        ACTION_STREAMING_SOURCE,
        "merge_partition_accumulators",
        "expression",
        "order[1:]",
        "order[1:-1]",
        1,
    ),
    "action_partition:elementwise_max_replaces_hard_max_merge": (
        ACTION_STREAMING_SOURCE,
        "merge_partition_accumulators",
        "expression",
        (
            "HARD_MAX_REDUCTION.merge(left=merged, right=jax.tree.map("
            "lambda leaf, index=partition: leaf[index], accumulators))"
        ),
        (
            "jax.tree.map(jnp.maximum, merged, jax.tree.map("
            "lambda leaf, index=partition: leaf[index], accumulators))"
        ),
        1,
    ),
    "action_partition:merged_accumulator_discarded": (
        ACTION_STREAMING_SOURCE,
        "merge_partition_accumulators",
        "expression",
        "HARD_MAX_REDUCTION.finalize(accumulator=merged)",
        (
            "HARD_MAX_REDUCTION.finalize(accumulator=jax.tree.map("
            "lambda leaf: leaf[order[0]], accumulators))"
        ),
        1,
    ),
    "action_partition:local_only_reduce": (
        ACTION_STREAMING_SOURCE,
        "_PartitionedStreamingHardMax.__call__",
        "expression",
        (
            "merge_partition_accumulators(accumulators=gathered, "
            "order=tuple(range(self.n_partitions)))"
        ),
        "HARD_MAX_REDUCTION.finalize(accumulator=local)",
        1,
    ),
    "action_partition:gather_replaced_by_local_copies": (
        ACTION_STREAMING_SOURCE,
        "_PartitionedStreamingHardMax.__call__",
        "expression",
        "jax.lax.all_gather(local, self.axis_name)",
        "jax.tree.map(lambda leaf: jnp.stack([leaf] * self.n_partitions), local)",
        1,
    ),
    "action_partition:partition_index_constant": (
        ACTION_STREAMING_SOURCE,
        "_PartitionedStreamingHardMax.__call__",
        "expression",
        "jax.lax.axis_index(self.axis_name).astype(jnp.int32)",
        "jnp.int32(0)",
        1,
    ),
    "action_partition:scan_starts_at_block_zero": (
        ACTION_STREAMING_SOURCE,
        "_PartitionedStreamingHardMax.local",
        "expression",
        "(_empty_reduction(evaluate_block=evaluate_block), first_block_index)",
        "(_empty_reduction(evaluate_block=evaluate_block), jnp.int32(0))",
        1,
    ),
    "action_partition:scan_skips_first_owned_block": (
        ACTION_STREAMING_SOURCE,
        "_PartitionedStreamingHardMax.local",
        "expression",
        "(_empty_reduction(evaluate_block=evaluate_block), first_block_index)",
        "(_empty_reduction(evaluate_block=evaluate_block), first_block_index + 1)",
        1,
    ),
    "action_partition:start_bound_off_by_one": (
        ACTION_STREAMING_SOURCE,
        "_PartitionedStreamingHardMax.local",
        "expression",
        "partition * n_blocks // self.n_partitions",
        "partition * n_blocks // self.n_partitions + 1",
        1,
    ),
    "action_partition:stop_bound_off_by_one": (
        ACTION_STREAMING_SOURCE,
        "_PartitionedStreamingHardMax.local",
        "expression",
        "(partition + 1) * n_blocks // self.n_partitions",
        "(partition + 1) * n_blocks // self.n_partitions - 1",
        1,
    ),
    "action_partition:padded_tail_admitted": (
        ACTION_STREAMING_SOURCE,
        "_PartitionedStreamingHardMax.local",
        "keyword",
        "n_actions",
        "n_actions + self.block_width",
        2,
    ),
    "action_partition:layout_block_count_floored": (
        ACTION_STREAMING_SOURCE,
        "ActionPartitionLayout.n_blocks",
        "expression",
        "-(-self.n_actions // self.block_width)",
        "self.n_actions // self.block_width",
        1,
    ),
    "action_partition:layout_run_length_floored": (
        ACTION_STREAMING_SOURCE,
        "ActionPartitionLayout.blocks_per_partition",
        "expression",
        "-(-self.n_blocks // self.n_partitions)",
        "self.n_blocks // self.n_partitions",
        1,
    ),
    "action_partition:layout_block_range_overlaps": (
        ACTION_STREAMING_SOURCE,
        "ActionPartitionLayout.block_range",
        "expression",
        "(partition + 1) * self.n_blocks // self.n_partitions",
        "(partition + 1) * self.n_blocks // self.n_partitions + 1",
        1,
    ),
    "action_partition:unowned_blocks_admitted": (
        ACTION_STREAMING_SOURCE,
        "_evaluate_owned_block",
        "expression",
        "feasible & owned",
        "feasible",
        1,
    ),
    "action_partition:owned_upper_bound_inclusive": (
        ACTION_STREAMING_SOURCE,
        "_evaluate_owned_block",
        "expression",
        "block_index < stop_block_index",
        "block_index <= stop_block_index",
        1,
    ),
    "action_partition:partition_local_action_ids": (
        ACTION_STREAMING_SOURCE,
        "_evaluate_owned_block",
        "expression",
        "global_ids",
        "global_ids - global_ids[0]",
        1,
    ),
    "action_partition:kernel_ignores_planned_width": (
        MAX_Q_SOURCE,
        "_ActionPartitionedMaxQOverA.__call__",
        "keyword",
        "block_width",
        "1",
        1,
    ),
    "action_partition:q_arguments_unfiltered": (
        MAX_Q_SOURCE,
        "_ActionPartitionedMaxQOverA.__call__",
        "expression",
        (
            "_arguments_named(arguments=states_actions_params, "
            "names=self.q_and_f_arg_names)"
        ),
        "states_actions_params",
        1,
    ),
    "action_partition:kernel_partition_count_dropped": (
        MAX_Q_SOURCE,
        "get_action_partitioned_max_Q_over_a",
        "keyword",
        "n_partitions",
        "1",
        1,
    ),
    "action_partition:state_product_order_reversed": (
        MAX_Q_SOURCE,
        "get_action_partitioned_max_Q_over_a",
        "keyword",
        "variables",
        "tuple(reversed(state_names))",
        2,
    ),
    "action_partition:manual_axes_widened": (
        MAX_Q_SOURCE,
        "_OnActionPartitionAxis.__call__",
        "expression",
        "frozenset({ACTION_PARTITION_AXIS})",
        "frozenset(self.mesh.axis_names)",
        1,
    ),
    "action_partition:route_selection_bypassed": (
        GRID_SEARCH_SOURCE,
        "GridSearch.build_period_kernels",
        "expression",
        "action_partition_mesh is not None",
        "False",
        1,
    ),
    "action_partition:request_silently_ignored": (
        GRID_SEARCH_SOURCE,
        "_action_partition_mesh",
        "expression",
        "context.action_partitions == 1",
        "True",
        1,
    ),
    "action_partition:unstreamed_route_admitted": (
        GRID_SEARCH_SOURCE,
        "_action_partition_mesh",
        "expression",
        "action_streaming is not _ActionStreamingDisposition.STREAMED",
        "False",
        1,
    ),
    "action_partition:fold_route_admitted": (
        GRID_SEARCH_SOURCE,
        "_action_partition_mesh",
        "expression",
        "bool(context.fold_state_names)",
        "False",
        1,
    ),
    "action_partition:co_map_route_admitted": (
        GRID_SEARCH_SOURCE,
        "_action_partition_mesh",
        "expression",
        "bool(context.co_map_state_names)",
        "False",
        1,
    ),
}

EXPECTED_ACTION_PARTITION_MUTATION_COUNT = 29
EXPECTED_ACTION_PARTITION_MUTATION_NAMES_SHA256 = (
    "cf858b9b0b747da0137c39fab58d7005555f2eda7aeecf471d43b4d3ad5c2ba3"
)


def action_partition_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build the action-partitioned hard-max route's mutation population."""
    return _callable_mutation_specs(
        repo_root=repo_root, mutations=_ACTION_PARTITION_MUTATIONS
    )


def _callable_mutation_specs(
    *, repo_root: Path, mutations: dict[str, tuple[str, str, str, str, str, int]]
) -> dict[str, dict[str, str]]:
    """Apply exact, named source mutations within one reviewed callable."""
    result: dict[str, dict[str, str]] = {}
    for name, (
        relative,
        qualname,
        kind,
        old,
        new,
        count,
    ) in mutations.items():
        source = (repo_root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        if "." in qualname:
            class_name, method_name = qualname.split(".", maxsplit=1)
            _, function = _method_definition(
                tree=tree, class_name=class_name, method_name=method_name
            )
        else:
            function = _definition(tree=tree, name=qualname)
        replacement = ast.parse(new, mode="eval").body
        if kind == "keyword":
            matches = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.keyword) and node.arg == old
            ]
            if len(matches) != count:
                raise ValueError(
                    f"{name}: expected {count} keyword anchors, found {len(matches)}"
                )
            for keyword in matches:
                keyword.value = copy.deepcopy(replacement)
        elif kind == "before_assignment":
            matches = [
                index
                for index, statement in enumerate(function.body)
                if isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == old
                    for target in statement.targets
                )
            ]
            if len(matches) != count:
                raise ValueError(
                    f"{name}: expected {count} assignment anchors, found {len(matches)}"
                )
            function.body.insert(matches[0], ast.Expr(value=replacement))
        else:
            expected = ast.dump(
                ast.parse(old, mode="eval").body, include_attributes=False
            )
            matches = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.expr)
                and ast.dump(node, include_attributes=False) == expected
            ]
            if len(matches) != count:
                raise ValueError(
                    f"{name}: expected {count} expression anchors, found {len(matches)}"
                )
            identities = {id(node) for node in matches}
            for parent in ast.walk(function):
                for attribute, value in ast.iter_fields(parent):
                    if isinstance(value, ast.AST) and id(value) in identities:
                        setattr(parent, attribute, copy.deepcopy(replacement))
                    elif isinstance(value, list):
                        setattr(
                            parent,
                            attribute,
                            [
                                copy.deepcopy(replacement)
                                if isinstance(item, ast.AST) and id(item) in identities
                                else item
                                for item in value
                            ],
                        )
        mutated = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        ast.parse(mutated, filename=relative)
        result[name] = {"path": relative, "source": mutated}
    return result


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
            new=(
                "                argument_builder=lambda context: argument_builder(context),"
            ),
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
                "                    if requires_plan"
            ),
            new=(
                "                    CoreExecutionDisposition.DENSE\n"
                "                    if requires_plan"
            ),
            label="native graph disposition authority",
        ),
        "native_graph:dense_reason_erased": replace_once(
            source=grid_source,
            old=(
                "                disposition_reason=(None if requires_plan else "
                "action_streaming.value),"
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
            old="        out = compiled_cores[core_key](**arguments)",
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
            old=(
                "        disposition = _ActionStreamingDisposition.UNSUPPORTED_COLLECTIVE_EV1"
            ),
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
            old=(
                "        disposition = _ActionStreamingDisposition.UNSUPPORTED_COLLECTIVE_FOLD"
            ),
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
            old=(
                "        disposition = _ActionStreamingDisposition.DENSE_TRIVIAL_ACTION_PRODUCT"
            ),
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
            old=(
                "        disposition = _ActionStreamingDisposition.DENSE_COLLECTIVE_RESOURCES"
            ),
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
            old=(
                "                solve_time_artifacts={DISSOLUTION_FLAG_ARTIFACT: dissolution},"
            ),
            new=(
                "                solve_time_artifacts={DISSOLUTION_FLAG_ARTIFACT: ~dissolution},"
            ),
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
        "streaming_resolver:candidate_validation_bypassed": replace_once(
            source=core_program_source,
            old="    _validate_core_program(program=program)",
            new="    pass",
            label="candidate declaration validation",
        ),
        "streaming_resolver:later_candidates_dropped": replace_once(
            source=core_program_source,
            old="        for widths in tile_widths",
            new="        for widths in tile_widths[:1]",
            label="complete width candidate enumeration",
        ),
        "streaming_resolver:bypass_static_width_binding": replace_once(
            source=core_program_source,
            old="        static_kwargs=width_bindings,",
            new="        static_kwargs={},",
            label="streamed resolver static kwargs",
        ),
        "streaming_resolver:arguments_filtered": replace_once(
            source=core_program_source,
            old=(
                "            else apply_value_transfer_plan(\n"
                "                arguments=program.arguments,\n"
                "                plan=resolved_input_transfer_plan,\n"
                "            )"
            ),
            new=(
                "            else dict(tuple(apply_value_transfer_plan(\n"
                "                arguments=program.arguments,\n"
                "                plan=resolved_input_transfer_plan,\n"
                "            ).items())[:-1])"
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
                "            program=program,\n"
                "            read=read,\n"
                "            transfer=transfer,\n"
                "            abstract_inputs=abstract_inputs,\n"
                "        )"
            ),
            new="        pass",
            label="core-program transfer argument metadata validation",
        ),
        "value_access:core_lowering_plan_bypassed": replace_once(
            source=core_program_source,
            old=(
                "            else apply_value_transfer_plan(\n"
                "                arguments=program.arguments,\n"
                "                plan=resolved_input_transfer_plan,\n"
                "            )"
            ),
            new="            else program.arguments",
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
            marker="            n_blocks=n_blocks,",
            replacement="            n_blocks=n_blocks - 1,",
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
            old=(
                "    for name, grid, size in zip(action_names, action_grids, action_sizes, strict=True):"
            ),
            new=(
                "    for name, grid, size in zip(reversed(action_names), "
                "reversed(action_grids), reversed(action_sizes), strict=True):"
            ),
            label="streamed C-order coordinate decode",
        ),
        "streaming_ev1:skip_last_block": _replace_nth(
            text=action_streaming_source,
            marker="            n_blocks=n_blocks,",
            replacement="            n_blocks=n_blocks - 1,",
            occurrence=2,
        ),
        "streaming_collective_blocks:skip_last_block": _replace_nth(
            text=action_streaming_source,
            marker="            n_blocks=n_blocks,",
            replacement="            n_blocks=n_blocks - 1,",
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
            old=(
                "        reduction = LOGSUMEXP_REDUCTION.bind(scale=jnp.asarray(self.scale))"
            ),
            new="        reduction = LOGSUMEXP_REDUCTION.bind(scale=jnp.asarray(1.0))",
            label="streamed EV1 one-session scale binding",
        ),
        "streaming_ev1:continuous_extent_changed": replace_once(
            source=action_streaming_source,
            old=(
                "        continuous_extent = math.prod(action_sizes[self.n_discrete_action_axes :])"
            ),
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
            old="    if not runtime_shardings_match(\n",
            new="    if False and not runtime_shardings_match(\n",
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
            old=(
                "        abstract_output=jax.eval_shape(invocation, **program.arguments, **templates),"
            ),
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
                    old=(
                        "return _SubjectTiled(func=dense_reducer, subject_arg_names=subject_arg_names)"
                    ),
                    new=(
                        "return _SubjectTiled(func=candidate_filter(dense_reducer), subject_arg_names=subject_arg_names)"
                    ),
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
                    new=(
                        "                            coordinate_names=tuple(reversed(action_names)),"
                    ),
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
                    new=(
                        "                            reduction=candidate_filter(HARD_MAX_REDUCTION),"
                    ),
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
                    new=(
                        "self, field, MappingProxyType(candidate_filter(dict(getattr(self, field))))"
                    ),
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
                    old="        return lowered\n",
                    new="        return candidate_filter(lowered)\n",
                    label="lowered_body_replaced",
                ),
            },
            "simulation_program:dispatch_family_replaced": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="        program=families[family][period],",
                    new='        program=families["route"][period],',
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
                    old="        program_identity=_func_dedup_key(func=program.function),",
                    new="        program_identity=0,",
                    label="body_cache_identity_dropped",
                ),
            },
            "simulation_program:compiler_options_dropped": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="        compiler_options=program.compiler_options,",
                    new="        compiler_options=(),",
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
            "simulation_program:runtime_binding_replaced": {
                "path": SIMULATION_COMPILE_SOURCE,
                "source": replace_once(
                    source=simulation_compile_source,
                    old="                        regime.simulation.programs, executor=executor",
                    new="                        candidate_filter(regime.simulation.programs), executor=executor",
                    label="runtime_binding_replaced",
                ),
            },
            "simulation_program:runtime_failure_hidden": {
                "path": SIMULATION_RUNTIME_SOURCE,
                "source": replace_once(
                    source=simulation_runtime_source,
                    old="                future.set_exception(error)",
                    new="                future.set_result(None)",
                    label="runtime_failure_hidden",
                ),
            },
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
            old="    per_subject_decisions = _build_per_subject_decisions_per_period(",
            new="    has_taste_shocks = False\n\n"
            "    per_subject_decisions = _build_per_subject_decisions_per_period(",
            label="simulate live taste rebinding",
        ),
        "caller_simulate:published_empty_mapping": replace_once(
            source=processing_source,
            old="        programs=programs,",
            new=(
                "        programs=dataclass_replace(programs, decision=MappingProxyType({})),"
            ),
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
            old="    is_nan = jnp.isnan(a)\n",
            new="    if a.reshape(-1)[0] > a.reshape(-1)[1]:\n"
            "        return jnp.array(1, dtype=jnp.int32), a.reshape(-1)[1]\n"
            "    is_nan = jnp.isnan(a)\n",
            label="argmax q-order",
        ),
        "shared_argmax:support_filter": replace_once(
            source=argmax_source,
            old="    is_nan = jnp.isnan(a)\n",
            new="    where = jnp.where(\n"
            "        jnp.sum(where) > 1,\n"
            "        where.reshape(-1).at[0].set(False).reshape(where.shape),\n"
            "        where,\n"
            "    )\n"
            "    is_nan = jnp.isnan(a)\n",
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
            old=(
                "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)"
            ),
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
            new=(
                "        a=objective, axis=action_axes[:-1], initial=-jnp.inf, where=feasibility"
            ),
            label="collective action axis",
        ),
        "shared_collective:gather_next_candidate": replace_once(
            source=collective_source,
            old=(
                "    gathered = jnp.take_along_axis(q_flat, argmax_flat[..., None], axis=-1)"
            ),
            new=(
                "    gathered = jnp.take_along_axis(q_flat, (argmax_flat + 1)[..., None], axis=-1)"
            ),
            label="collective gather",
        ),
        "shared_collective:early_candidate_return": replace_once(
            source=collective_source,
            old=(
                "    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)"
            ),
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
                old=(
                    'class Grid(ABC):\n    """Outcome-space definition shared by all LCM grids."""'
                ),
                new=(
                    'class Grid(ABC):\n    """Outcome-space definition shared by all LCM grids."""\n\n    def __getattribute__(self, name):\n        value = super().__getattribute__(name)\n        if name == "to_jax":\n            return lambda: value()[:-1]\n        return value'
                ),
                label="inherited grid coordinate interception",
            ),
        },
        "simulation_state_action_space:drops_inherited_candidates": {
            "path": SIMULATION_TRANSITIONS_SOURCE,
            "source": replace_once(
                source=simulation_transitions_source,
                old=(
                    "    return base.replace(states=MappingProxyType(states_for_state_action_space))"
                ),
                new=(
                    "    return base.replace(\n        states=MappingProxyType(states_for_state_action_space),\n        discrete_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base.discrete_actions.items()}),\n        continuous_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base.continuous_actions.items()}),\n    )"
                ),
                label="simulation base-action preservation",
            ),
        },
        "simulation_state_action_space:caller_drops_inherited_candidates": {
            "path": SIMULATION_SOURCE,
            "source": replace_once(
                source=simulation_source,
                old="        base=base_state_action_space,",
                new=(
                    "        base=base_state_action_space.replace(continuous_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base_state_action_space.continuous_actions.items()})),"
                ),
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
                new=(
                    "            {name: values.at[-1].set(values[0]) for name, values in self.discrete_actions.items()} | {name: values.at[-1].set(values[0]) for name, values in self.continuous_actions.items()}"
                ),
                label="combined action mapping candidate omission",
            ),
        },
        "shared_engine:replace_drops_inherited_candidates": {
            "path": ENGINE_SOURCE,
            "source": replace_once(
                source=engine_source,
                old=(
                    "        discrete_actions = first_non_none(discrete_actions, self.discrete_actions)"
                ),
                new=(
                    "        discrete_actions = first_non_none(discrete_actions, MappingProxyType({name: values.at[-1].set(values[0]) for name, values in self.discrete_actions.items()}))"
                ),
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
                old=(
                    "        name: _grid_to_jax_or_placeholder(grids[name])\n        for name in variables.continuous_action_names"
                ),
                new=(
                    "        name: _grid_to_jax_or_placeholder(grids[name]).at[-1].set(_grid_to_jax_or_placeholder(grids[name])[0])\n        for name in variables.continuous_action_names"
                ),
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
        "runtime_model:compiled_regime_filter": {
            "path": MODEL_SOURCE,
            "source": replace_once(
                source=model_source,
                old="            return self._simulate_runtime_regimes[compile_batch_size]",
                new="            return candidate_filter(\n"
                "                self._simulate_runtime_regimes[compile_batch_size]\n"
                "            )",
                label="public Model runtime regime selection",
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
                old="            source=triple,",
                new=(
                    "            source=(\n"
                    "                materialized.requirements.value_reads[0].source.source_regime,\n"
                    "                materialized.requirements.value_reads[0].source.source_period,\n"
                    "                materialized.requirements.value_reads[0].source.core_key,\n"
                    "            ),"
                ),
                label="backward actual source-coordinate authority",
            ),
        },
        "value_transfer:backward_resolver_plan_omitted": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": _replace_nth(
                text=backward_induction_source,
                marker="            input_transfer_plan=transfer_plan,",
                replacement="            input_transfer_plan=(),",
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
                old=(
                    "        source_sharding = jax.NamedSharding(\n"
                    "            mesh=source_execution_sharding.mesh,\n"
                    "            spec=jax.P(),"
                ),
                new=(
                    "        source_sharding = jax.NamedSharding(\n"
                    "            mesh=source_execution_sharding.mesh,\n"
                    "            spec=source_execution_sharding.spec,"
                ),
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
                    "    try:\n"
                    "        kind = classify_value_transfer(\n"
                    "            stored_sharding=stored_sharding,\n"
                    "            required_sharding=source_sharding,\n"
                    "        )"
                ),
                new="    try:\n        kind = ValueTransferKind.ALIGNED_LOCAL",
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
                old="            tile_widths=width_candidates[:1],",
                new="            tile_widths=({},),",
                label="native program planned widths",
            ),
        },
        "native_graph:aot_resolved_function_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old=(
                    "        resolved.function,\n"
                    "        static_argnames=tuple(static_kwargs),"
                ),
                new=(
                    "        candidate_filter(resolved.function),\n"
                    "        static_argnames=tuple(static_kwargs),"
                ),
                label="AOT resolved function",
            ),
        },
        "native_graph:aot_resolved_arguments_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old=(
                    "    low = jitted.lower(**resolved.arguments, "
                    "**internal_templates, **static_kwargs)"
                ),
                new="    low = jitted.lower(**static_kwargs)",
                label="AOT resolved arguments",
            ),
        },
        "native_graph:specialization_dropped": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="            resolved.specialization_key,",
                new="            None,",
                label="native lowering specialization",
            ),
        },
        "native_graph:eager_resolved_function_bypassed": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old=(
                    "                compiled=make_eager_core(\n                   "
                    " program=program,\n                    execution_sharding="
                    "all_layouts[triple].expected_leaves[0].sharding,\n            "
                    "        internal_input_templates=internal_templates[\n        "
                    "                selected_candidates[triple]\n                 "
                    "   ],\n                ),"
                ),
                new="                compiled=all_programs[triple].function,",
                label="eager resolved function",
            ),
        },
        "backward_layout:out_shardings_disabled": {
            "path": BACKWARD_INDUCTION_SOURCE,
            "source": replace_once(
                source=backward_induction_source,
                old="        out_shardings=layout.out_shardings,",
                new="        out_shardings=None,",
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
                old=(
                    "                                start=0,\n                                stop=original_n_subjects,"
                ),
                new=(
                    "                                start=1 if name == 'actions' else 0,\n                                stop=original_n_subjects + (name == 'actions'),"
                ),
                label="padded action row shift",
            ),
        },
        "simulation_result:shift_raw_actions": {
            "path": RESULT_SOURCE,
            "source": replace_once(
                source=result_source,
                old="        self._raw_results = raw_results",
                new=(
                    "        self._raw_results = MappingProxyType({regime: MappingProxyType({period: __import__('dataclasses').replace(data, actions=MappingProxyType({name: jnp.roll(values, 1) for name, values in data.actions.items()})) for period, data in periods.items()}) for regime, periods in raw_results.items()})"
                ),
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
                old=(
                    "        return {\n            k: _one_value_per_row(values=v, n_rows=n_rows) for k, v in result.items()\n        }"
                ),
                new=(
                    "        return {\n            k: _one_value_per_row(values=v, n_rows=n_rows) for k, v in result.items()\n        } | {name: jnp.roll(jnp.asarray(data[name]), 1) for name in regime.simulation.action_names if name in data}"
                ),
                label="single-pass additional-target action overwrite",
            ),
        },
        "additional_targets:overwrite_actions_chunked": {
            "path": ADDITIONAL_TARGETS_SOURCE,
            "source": replace_once(
                source=additional_targets_source,
                old=(
                    "    return {\n        name: np.concatenate([out[name] for out in chunk_outputs])\n        for name in chunk_outputs[0]\n    }"
                ),
                new=(
                    "    return {\n        **{name: np.concatenate([out[name] for out in chunk_outputs]) for name in chunk_outputs[0]},\n        **{name: jnp.roll(jnp.asarray(data[name]), 1) for name in regime.simulation.action_names if name in data},\n    }"
                ),
                label="chunked additional-target action overwrite",
            ),
        },
        "simulation_random:reassign_taste_keys": {
            "path": SIMULATION_RANDOM_SOURCE,
            "source": replace_once(
                source=simulation_random_source,
                old='        simulation_keys[f"key_{name}"] = per_subject_keys',
                new=(
                    '        simulation_keys[f"key_{name}"] = jnp.roll(per_subject_keys, 1, axis=0)'
                ),
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
                new=(
                    '        object.__setattr__(self, "value", -self.value)\n        if not np.issubdtype(value_dtype, np.floating):'
                ),
                label="KernelOutput value transport negation",
            ),
        },
        "candidate_materialization:rebind_continuous_grid": {
            "path": GRIDS_INIT_SOURCE,
            "source": replace_once(
                source=grids_init_source,
                old="from _lcm.grids.discrete import DiscreteGrid",
                new=(
                    "from _lcm.grids.discrete import DiscreteGrid\n\nContinuousGrid = DiscreteGrid"
                ),
                label="continuous-grid classification rebinding",
            ),
        },
        "candidate_materialization:rebind_process_class": {
            "path": PROCESSES_INIT_SOURCE,
            "source": replace_once(
                source=processes_init_source,
                old="from _lcm.processes.iid import _IIDProcess",
                new=(
                    "from _lcm.processes.iid import _IIDProcess\n\n_ContinuousStochasticProcess = _AR1Process"
                ),
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
                old=(
                    "        return grid_coordinates.linspace(\n            start=self.start, stop=self.stop, n_points=self.n_points\n        )"
                ),
                new=(
                    "        return grid_coordinates.linspace(\n            start=self.start, stop=self.stop, n_points=self.n_points\n        )[:-1]"
                ),
                label="linear action point omission",
            ),
        },
        "candidate_materialization:drop_last_coordinate_point": {
            "path": GRID_COORDINATES_SOURCE,
            "source": replace_once(
                source=grid_coordinates_source,
                old=(
                    "    return jnp.linspace(start, stop, n_points)  # ty: ignore[no-matching-overload]"
                ),
                new=(
                    "    return jnp.linspace(start, stop, n_points)[:-1]  # ty: ignore[no-matching-overload]"
                ),
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
                old=(
                    '        return jnp.linspace(\n            start=kwargs["start"], stop=kwargs["stop"], num=self.n_points\n        )'
                ),
                new=(
                    '        return jnp.linspace(\n            start=kwargs["start"], stop=kwargs["stop"], num=self.n_points\n        )[:-1]'
                ),
                label="IID action node omission",
            ),
        },
        "candidate_materialization:drop_last_ar1_node": {
            "path": PROCESS_AR1_SOURCE,
            "source": replace_once(
                source=process_ar1_source,
                old=(
                    "        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)"
                ),
                new=(
                    "        return jnp.linspace(long_run_mean - nu, long_run_mean + nu, n_points)[:-1]"
                ),
                label="AR1 action node omission",
            ),
        },
        "candidate_materialization:drop_last_action_name": {
            "path": VARIABLES_SOURCE,
            "source": replace_once(
                source=variables_source,
                old=(
                    '    actions = [name for name, var_info in info.items() if var_info.kind == "action"]'
                ),
                new=(
                    '    actions = [name for name, var_info in info.items() if var_info.kind == "action"][:-1]'
                ),
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
                new=(
                    '            result[regime][remainder] = (-params_flat[chosen] if remainder.endswith("__points") else params_flat[chosen])'
                ),
                label="runtime action points changed during broadcast",
            ),
        },
        "candidate_materialization:negate_flattened_runtime_points": {
            "path": NAMESPACE_SOURCE,
            "source": replace_once(
                source=namespace_source,
                old="    return MappingProxyType(flatten_to_qnames(d))",
                new=(
                    '    flat = flatten_to_qnames(d)\n    return MappingProxyType({key: -value if key.endswith("__points") and hasattr(value, "dtype") else value for key, value in flat.items()})'
                ),
                label="runtime action points changed during namespace flattening",
            ),
        },
        "candidate_materialization:negate_cast_runtime_points": {
            "path": DTYPES_SOURCE,
            "source": replace_once(
                source=dtypes_source,
                old="    return jnp.asarray(np_value, dtype=target_dtype)",
                new=(
                    '    out = jnp.asarray(np_value, dtype=target_dtype)\n    return -out if name.endswith("__points") else out'
                ),
                label="runtime action points changed during canonical cast",
            ),
        },
        "candidate_materialization:negate_series_runtime_points": {
            "path": PANDAS_UTILS_SOURCE,
            "source": replace_once(
                source=pandas_utils_source,
                old=(
                    "    if not indexing_params:\n"
                    "        return _write_pandas_array(\n"
                    "            value=sr.to_numpy(),"
                ),
                new=(
                    "    if not indexing_params:\n"
                    "        return _write_pandas_array(\n"
                    "            value=-sr.to_numpy(),"
                ),
                label="Series runtime action points changed during conversion",
            ),
        },
        "candidate_materialization:negate_fixed_runtime_points": {
            "path": MODEL_PROCESSING_SOURCE,
            "source": replace_once(
                source=model_processing_source,
                old=(
                    "        regime_fixed = dict(\n"
                    "            regime_kernel_params(fixed_flat_params, regime_name=regime_name)\n"
                    "        )"
                ),
                new=(
                    "        regime_fixed = dict(\n"
                    "            regime_kernel_params(fixed_flat_params, regime_name=regime_name)\n"
                    "        )\n"
                    '        regime_fixed = {key: -value if key.endswith("__points") else value for key, value in regime_fixed.items()}'
                ),
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
    certified_paths = (
        set(_CERTIFIED_CORRIDOR_SOURCES)
        - set(_UNIFORM_PROCESS_SOURCES)
        - set(_ACTION_GRID_SOURCES)
    )
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
    uniform = uniform_process_mutation_specs(repo_root=root)
    action_grid = action_grid_mutation_specs(repo_root=root)
    feasibility = feasibility_mutation_specs(repo_root=root)
    grouped = grouped_mapper_mutation_specs(repo_root=root)
    guard = grouped_guard_mutation_specs(repo_root=root)
    allocation = allocation_reservation_mutation_specs(repo_root=root)
    normal = normal_process_mutation_specs(repo_root=root)
    partition = action_partition_mutation_specs(repo_root=root)
    cases: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as raw:
        temp_root = Path(raw) / "repo"
        for relative, source in originals.items():
            target = temp_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        for name, spec in (
            registered
            | supplemental
            | uniform
            | action_grid
            | feasibility
            | grouped
            | guard
            | allocation
            | normal
            | partition
        ).items():
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
    partition_cases = {name: cases.pop(name) for name in partition}
    partition_admitted = sorted(
        name for name, result in partition_cases.items() if not result["rejected"]
    )
    partition_names_match = (
        len(partition_cases) == EXPECTED_ACTION_PARTITION_MUTATION_COUNT
        and _mutation_name_digest(tuple(partition_cases))
        == EXPECTED_ACTION_PARTITION_MUTATION_NAMES_SHA256
    )
    normal_cases = {name: cases.pop(name) for name in normal}
    normal_admitted = sorted(
        name for name, result in normal_cases.items() if not result["rejected"]
    )
    normal_names_match = (
        len(normal_cases) == EXPECTED_NORMAL_PROCESS_MUTATION_COUNT
        and _mutation_name_digest(tuple(normal_cases))
        == EXPECTED_NORMAL_PROCESS_MUTATION_NAMES_SHA256
    )
    allocation_cases = {name: cases.pop(name) for name in allocation}
    allocation_admitted = sorted(
        name for name, result in allocation_cases.items() if not result["rejected"]
    )
    allocation_names_match = (
        len(allocation_cases) == EXPECTED_ALLOCATION_RESERVATION_MUTATION_COUNT
        and _mutation_name_digest(tuple(allocation_cases))
        == EXPECTED_ALLOCATION_RESERVATION_MUTATION_NAMES_SHA256
    )
    feasibility_cases = {name: cases.pop(name) for name in feasibility}
    feasibility_admitted = sorted(
        name for name, result in feasibility_cases.items() if not result["rejected"]
    )
    feasibility_names_match = (
        len(feasibility_cases) == EXPECTED_FEASIBILITY_MUTATION_COUNT
        and _mutation_name_digest(tuple(feasibility_cases))
        == EXPECTED_FEASIBILITY_MUTATION_NAMES_SHA256
    )
    guard_cases = {name: cases.pop(name) for name in guard}
    guard_admitted = sorted(
        name for name, result in guard_cases.items() if not result["rejected"]
    )
    guard_names_match = (
        len(guard_cases) == EXPECTED_GROUPED_GUARD_MUTATION_COUNT
        and _mutation_name_digest(tuple(guard_cases))
        == EXPECTED_GROUPED_GUARD_MUTATION_NAMES_SHA256
    )
    grouped_cases = {name: cases.pop(name) for name in grouped}
    grouped_admitted = sorted(
        name for name, result in grouped_cases.items() if not result["rejected"]
    )
    grouped_names_match = (
        len(grouped_cases) == EXPECTED_GROUPED_MAPPER_MUTATION_COUNT
        and _mutation_name_digest(tuple(grouped_cases))
        == EXPECTED_GROUPED_MAPPER_MUTATION_NAMES_SHA256
    )
    action_grid_cases = {name: cases.pop(name) for name in action_grid}
    action_grid_admitted = sorted(
        name for name, result in action_grid_cases.items() if not result["rejected"]
    )
    action_grid_names_match = (
        len(action_grid_cases) == EXPECTED_ACTION_GRID_MUTATION_COUNT
        and _mutation_name_digest(tuple(action_grid_cases))
        == EXPECTED_ACTION_GRID_MUTATION_NAMES_SHA256
    )
    uniform_cases = {name: cases.pop(name) for name in uniform}
    uniform_admitted = sorted(
        name for name, result in uniform_cases.items() if not result["rejected"]
    )
    supplemental_cases = {name: cases.pop(name) for name in supplemental}
    supplemental_admitted = sorted(
        name for name, result in supplemental_cases.items() if not result["rejected"]
    )
    admitted = sorted(name for name, result in cases.items() if not result["rejected"])
    supplemental_names_match = (
        len(supplemental_cases) == EXPECTED_SUPPLEMENTAL_MUTATION_COUNT
        and _mutation_name_digest(tuple(supplemental_cases))
        == EXPECTED_SUPPLEMENTAL_MUTATION_NAMES_SHA256
    )
    uniform_names_match = (
        len(uniform_cases) == EXPECTED_UNIFORM_PROCESS_MUTATION_COUNT
        and _mutation_name_digest(tuple(uniform_cases))
        == EXPECTED_UNIFORM_PROCESS_MUTATION_NAMES_SHA256
    )
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
        "action_partition_mutations": partition_cases,
        "action_partition_mutation_count": len(partition_cases),
        "action_partition_names_match_expected": partition_names_match,
        "admitted_action_partition_mutations": partition_admitted,
        "normal_process_mutations": normal_cases,
        "normal_process_mutation_count": len(normal_cases),
        "normal_process_names_match_expected": normal_names_match,
        "admitted_normal_process_mutations": normal_admitted,
        "allocation_reservation_mutations": allocation_cases,
        "allocation_reservation_mutation_count": len(allocation_cases),
        "allocation_reservation_names_match_expected": allocation_names_match,
        "admitted_allocation_reservation_mutations": allocation_admitted,
        "feasibility_mutations": feasibility_cases,
        "feasibility_mutation_count": len(feasibility_cases),
        "feasibility_names_match_expected": feasibility_names_match,
        "admitted_feasibility_mutations": feasibility_admitted,
        "grouped_guard_mutations": guard_cases,
        "grouped_guard_mutation_count": len(guard_cases),
        "grouped_guard_names_match_expected": guard_names_match,
        "admitted_grouped_guard_mutations": guard_admitted,
        "grouped_mapper_mutations": grouped_cases,
        "grouped_mapper_mutation_count": len(grouped_cases),
        "grouped_mapper_names_match_expected": grouped_names_match,
        "admitted_grouped_mapper_mutations": grouped_admitted,
        "action_grid_mutations": action_grid_cases,
        "action_grid_mutation_count": len(action_grid_cases),
        "action_grid_names_match_expected": action_grid_names_match,
        "admitted_action_grid_mutations": action_grid_admitted,
        "uniform_process_mutations": uniform_cases,
        "uniform_process_mutation_count": len(uniform_cases),
        "uniform_process_names_match_expected": uniform_names_match,
        "admitted_uniform_process_mutations": uniform_admitted,
        "supplemental_mutations": supplemental_cases,
        "supplemental_mutation_count": len(supplemental_cases),
        "supplemental_names_match_expected": supplemental_names_match,
        "admitted_supplemental_mutations": supplemental_admitted,
        "all_rejected": (
            clean["ok"]
            and not partition_admitted
            and partition_names_match
            and not normal_admitted
            and normal_names_match
            and not allocation_admitted
            and allocation_names_match
            and not admitted
            and not supplemental_admitted
            and not uniform_admitted
            and not action_grid_admitted
            and not feasibility_admitted
            and feasibility_names_match
            and not grouped_admitted
            and not guard_admitted
            and guard_names_match
            and grouped_names_match
            and action_grid_names_match
            and supplemental_names_match
            and uniform_names_match
            and count_matches_expected
            and names_match_expected
        ),
    }
