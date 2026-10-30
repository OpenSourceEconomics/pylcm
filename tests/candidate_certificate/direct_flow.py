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
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

try:
    from outer_helper_controls import (
        BODY_OWNERS as _OUTER_HELPER_OWNER_SELECTION,
    )
    from outer_helper_controls import (
        CONTROLS as _OUTER_HELPER_CONTROLS,
    )
    from outer_helper_controls import (
        outer_helper_mutation_specs as _outer_helper_control_specs,
    )
except ModuleNotFoundError:  # Imported as tests.candidate_certificate.direct_flow.
    from tests.candidate_certificate.outer_helper_controls import (
        BODY_OWNERS as _OUTER_HELPER_OWNER_SELECTION,
    )
    from tests.candidate_certificate.outer_helper_controls import (
        CONTROLS as _OUTER_HELPER_CONTROLS,
    )
    from tests.candidate_certificate.outer_helper_controls import (
        outer_helper_mutation_specs as _outer_helper_control_specs,
    )

try:
    from nested_helper_selection import (
        select_interval_winner_call as _select_interval_winner_call,
    )
    from nested_helper_selection import (
        select_nbegm_query_read_choices as _select_nbegm_query_read_choices,
    )
except ModuleNotFoundError:  # Imported as tests.candidate_certificate.direct_flow.
    from tests.candidate_certificate.nested_helper_selection import (
        select_interval_winner_call as _select_interval_winner_call,
    )
    from tests.candidate_certificate.nested_helper_selection import (
        select_nbegm_query_read_choices as _select_nbegm_query_read_choices,
    )

try:
    from final_nested_controls import (
        expected_errors as _final_nested_expected_errors,
    )
    from final_nested_controls import (
        final_nested_mutation_specs as _final_nested_control_specs,
    )
    from final_nested_controls import (
        primary_diagnostic as _final_nested_primary_diagnostic,
    )
except ModuleNotFoundError:
    from tests.candidate_certificate.final_nested_controls import (
        expected_errors as _final_nested_expected_errors,
    )
    from tests.candidate_certificate.final_nested_controls import (
        final_nested_mutation_specs as _final_nested_control_specs,
    )
    from tests.candidate_certificate.final_nested_controls import (
        primary_diagnostic as _final_nested_primary_diagnostic,
    )


try:
    from child_native_controls import (
        BODY_OWNERS as _CHILD_NATIVE_OWNER_SELECTION,
    )
    from child_native_controls import (
        child_native_mutation_specs as _child_native_control_specs,
    )
    from child_native_controls import (
        expected_errors as _child_native_expected_errors,
    )
    from child_native_controls import (
        primary_diagnostic as _child_native_primary_diagnostic,
    )
    from child_native_selection import (
        select_refine_envelope_exact as _select_refine_envelope_exact,
    )
except ModuleNotFoundError:
    from tests.candidate_certificate.child_native_controls import (
        BODY_OWNERS as _CHILD_NATIVE_OWNER_SELECTION,
    )
    from tests.candidate_certificate.child_native_controls import (
        child_native_mutation_specs as _child_native_control_specs,
    )
    from tests.candidate_certificate.child_native_controls import (
        expected_errors as _child_native_expected_errors,
    )
    from tests.candidate_certificate.child_native_controls import (
        primary_diagnostic as _child_native_primary_diagnostic,
    )
    from tests.candidate_certificate.child_native_selection import (
        select_refine_envelope_exact as _select_refine_envelope_exact,
    )


try:
    from checked_dc_selection import select_checked_dc_runtime
except ModuleNotFoundError:
    from tests.candidate_certificate.checked_dc_selection import (
        select_checked_dc_runtime,
    )


try:
    from additional_helper_controls import (
        BODY_OWNERS as _ADDITIONAL_HELPER_OWNER_SELECTION,
    )
    from additional_helper_controls import (
        SURFACE_OWNERS as _ADDITIONAL_HELPER_SURFACE_SELECTION,
    )
    from additional_helper_controls import (
        additional_helper_mutation_specs as _additional_helper_control_specs,
    )
    from additional_helper_controls import (
        expected_errors as _additional_helper_expected_errors,
    )
    from additional_helper_controls import (
        primary_diagnostic as _additional_helper_primary_diagnostic,
    )
except ModuleNotFoundError:
    from tests.candidate_certificate.additional_helper_controls import (
        BODY_OWNERS as _ADDITIONAL_HELPER_OWNER_SELECTION,
    )
    from tests.candidate_certificate.additional_helper_controls import (
        SURFACE_OWNERS as _ADDITIONAL_HELPER_SURFACE_SELECTION,
    )
    from tests.candidate_certificate.additional_helper_controls import (
        additional_helper_mutation_specs as _additional_helper_control_specs,
    )
    from tests.candidate_certificate.additional_helper_controls import (
        expected_errors as _additional_helper_expected_errors,
    )
    from tests.candidate_certificate.additional_helper_controls import (
        primary_diagnostic as _additional_helper_primary_diagnostic,
    )


try:
    from generate_sources import sha256_file
except ModuleNotFoundError:  # Imported as tests.candidate_certificate.direct_flow.
    from tests.candidate_certificate.generate_sources import sha256_file

MAX_Q_SOURCE = "src/_lcm/regime_building/max_Q_over_a.py"
ARGMAX_SOURCE = "src/_lcm/regime_building/argmax.py"
COLLECTIVE_SOURCE = "src/_lcm/regime_building/collective.py"
LOGSUM_SOURCE = "src/_lcm/logsum.py"
GRID_SEARCH_SOURCE = "src/_lcm/solution/grid_search.py"
FEASIBILITY_SOURCE = "src/_lcm/solution/feasibility.py"
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
SIMULATION_RANDOM_ADDRESS_SOURCE = "src/_lcm/simulation/random_address.py"
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
EAGER_STAGE_SOURCE = "src/_lcm/simulation/eager_stages.py"
SUPPORT_FINGERPRINT_SOURCE = "src/_lcm/solution/fingerprint.py"
SUPPORT_AUTHORITY_SOURCE = "src/_lcm/solution/model_authority.py"
SUPPORT_PRECONDITIONS_SOURCE = "src/_lcm/solution/preconditions.py"
SUPPORT_DIAGNOSTICS_SOURCE = "src/_lcm/solution/diagnostics.py"
SUPPORT_TRANSITION_CHECKS_SOURCE = "src/_lcm/transition_checks.py"
VARIABLES_SOURCE = "src/_lcm/variables.py"
PARAMS_REGIME_TEMPLATE_SOURCE = "src/_lcm/params/regime_template.py"
PARAMS_PROCESSING_SOURCE = "src/_lcm/params/processing.py"
DTYPES_SOURCE = "src/_lcm/dtypes.py"
LCM_TYPING_SOURCE = "src/lcm/typing.py"
NAMESPACE_SOURCE = "src/_lcm/utils/namespace.py"
PANDAS_UTILS_SOURCE = "src/_lcm/pandas_utils.py"
MODEL_PROCESSING_SOURCE = "src/_lcm/model_processing.py"

DYNAMIC_BOUND_SOURCE = "src/_lcm/constraints/dynamic_bound.py"
POST_DECISION_BOUND_SOURCE = "src/_lcm/post_decision_bound.py"
BOUND_PROOF_SOURCE = "src/_lcm/constraints/bounds.py"
BOUND_NORMALIZATION_SOURCE = "src/_lcm/constraints/processed.py"
USER_REGIME_BOUND_SOURCE = "src/lcm/regime.py"
LIQUID_BOUND_DECLARATION_SOURCE = "src/lcm/consumption_savings_regime.py"

AGE_BOUND_NORMALIZATION_SOURCE = "src/_lcm/regime_building/age_normalization.py"

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


SOLVE_PENDING_WORK_SOURCE = "src/_lcm/execution/pending_work.py"

NATIVE_VALUES_SOURCE = "src/_lcm/solution/native_values.py"
NATIVE_ARCHIVE_SOURCE = "src/_lcm/persistence/solution.py"
STRUCTURAL_BLUEPRINTS_SOURCE = "src/_lcm/solution/structural_blueprints.py"

_ACTION_GRID_SOURCES = (ACTION_GRID_SOURCE,)

_UNIFORM_PROCESS_SOURCES = (
    PROCESS_GRID_RESOLUTION_SOURCE,
    UNIFORM_PROCESS_GRID_SOURCE,
    EAGER_STAGE_SOURCE,
    SUPPORT_FINGERPRINT_SOURCE,
    SUPPORT_AUTHORITY_SOURCE,
    SUPPORT_PRECONDITIONS_SOURCE,
    SUPPORT_DIAGNOSTICS_SOURCE,
    SUPPORT_TRANSITION_CHECKS_SOURCE,
)


S02__LCM_CONTINUATION_SOURCE = "src/_lcm/continuation.py"
S02__LCM_EGM_BUDGET_SOURCE = "src/_lcm/egm/budget.py"
S02__LCM_EGM_CARRY_SOURCE = "src/_lcm/egm/carry.py"
S02__LCM_EGM_QUERY_DATA_SOURCE = "src/_lcm/egm/query_data.py"
S02__LCM_REGIME_BUILDING_Q_AND_F_SOURCE = "src/_lcm/regime_building/Q_and_F.py"
S02__LCM_SIMULATION_SUBJECT_PARALLEL_SOURCE = "src/_lcm/simulation/subject_parallel.py"
S02__LCM_SOLUTION_EGM_SOURCE = "src/_lcm/solution/egm.py"
S02__LCM_SOLUTION_KERNEL_OUTPUT_SOURCE = "src/_lcm/solution/kernel_output.py"

S02_LANDING_COUNTS_SOURCE = "src/_lcm/solution/landing_counts.py"

S02__LCM_EGM_OUTER_ENVELOPE_SOURCE = "src/_lcm/egm/outer_envelope.py"

S03_EULER_SOURCE = "src/_lcm/egm/euler.py"
S03_DECLARED_LAW_SOURCE = "src/_lcm/egm/declared_law.py"
S03_INTERP_SOURCE = "src/_lcm/egm/interp.py"
S03_NUMERIC_INVERSE_SOURCE = "src/_lcm/egm/numeric_inverse.py"
S03_ONE_ASSET_STEP_SOURCE = "src/_lcm/egm/one_asset_egm_step.py"
S03_PREFERENCES_SOURCE = "src/_lcm/egm/preferences.py"
S03_SCALAR_QUERY_SOURCE = "src/_lcm/egm/scalar_query.py"
DECOUPLED_BOUND_SOURCE = "src/_lcm/regime_building/decoupled.py"

REVIEWED_PREFIX_ASSET_ROW_SOURCE = "src/_lcm/egm/asset_row.py"
REVIEWED_PREFIX_CONTINUATION_SOURCE = "src/_lcm/egm/continuation.py"
REVIEWED_PREFIX_DCEGM_QUERY_SOURCE = "src/_lcm/egm/dcegm_query.py"
REVIEWED_PREFIX_DCEGM_QUERY_MODEL_SOURCE = "src/_lcm/egm/dcegm_query_model.py"
REVIEWED_PREFIX_REGIME_INTROSPECTION_SOURCE = "src/_lcm/egm/regime_introspection.py"
REVIEWED_PREFIX_STEP_SOURCE = "src/_lcm/egm/step.py"
REVIEWED_PREFIX_STEP_CORE_SOURCE = "src/_lcm/egm/step_core.py"
REVIEWED_PREFIX_DCEGM_SOURCE = "src/_lcm/solution/dcegm.py"

ASSOCIATED_SAVING_SOURCE = "src/_lcm/egm/upper_envelope/query.py"

OUTER_INTERPOLATION_SOURCE = "src/_lcm/egm/outer_interpolation.py"
OUTER_REFINEMENT_SOURCE = "src/_lcm/egm/outer_refinement.py"
OUTER_GOLDEN_SECTION_SOURCE = "src/_lcm/optimization/golden_section.py"

_CURRENT_IID_SOURCE = "src/_lcm/current_iid.py"
_CURRENT_IID_KEYS_SOURCE = "src/_lcm/simulation/current_iid.py"
_CURRENT_IID_OVERRIDES_SOURCE = "src/_lcm/simulation/current_iid_overrides.py"

_CERTIFIED_CORRIDOR_SOURCES = (
    _CURRENT_IID_SOURCE,
    _CURRENT_IID_KEYS_SOURCE,
    _CURRENT_IID_OVERRIDES_SOURCE,
    "src/lcm/shocks.py",
    "src/_lcm/shock_overrides.py",
    "src/_lcm/egm/upper_envelope/__init__.py",
    "src/_lcm/egm/upper_envelope/segment_envelope.py",
    "src/_lcm/egm/nested_floor.py",
    "src/_lcm/egm/finite_negm_query.py",
    "src/_lcm/egm/adaptive_nbegm_producer.py",
    "src/_lcm/egm/nbegm_query_model.py",
    "src/_lcm/egm/nbegm_query.py",
    "src/_lcm/solution/negm.py",
    "src/_lcm/solution/nnbegm.py",
    "src/_lcm/egm/outer_candidates.py",
    OUTER_INTERPOLATION_SOURCE,
    OUTER_REFINEMENT_SOURCE,
    OUTER_GOLDEN_SECTION_SOURCE,
    ASSOCIATED_SAVING_SOURCE,
    REVIEWED_PREFIX_ASSET_ROW_SOURCE,
    REVIEWED_PREFIX_CONTINUATION_SOURCE,
    REVIEWED_PREFIX_DCEGM_QUERY_SOURCE,
    REVIEWED_PREFIX_DCEGM_QUERY_MODEL_SOURCE,
    REVIEWED_PREFIX_REGIME_INTROSPECTION_SOURCE,
    REVIEWED_PREFIX_STEP_SOURCE,
    REVIEWED_PREFIX_STEP_CORE_SOURCE,
    REVIEWED_PREFIX_DCEGM_SOURCE,
    DECOUPLED_BOUND_SOURCE,
    S03_EULER_SOURCE,
    S03_DECLARED_LAW_SOURCE,
    S03_INTERP_SOURCE,
    S03_NUMERIC_INVERSE_SOURCE,
    S03_ONE_ASSET_STEP_SOURCE,
    S03_PREFERENCES_SOURCE,
    S03_SCALAR_QUERY_SOURCE,
    S02__LCM_EGM_OUTER_ENVELOPE_SOURCE,
    S02_LANDING_COUNTS_SOURCE,
    S02__LCM_CONTINUATION_SOURCE,
    S02__LCM_EGM_BUDGET_SOURCE,
    S02__LCM_EGM_CARRY_SOURCE,
    S02__LCM_EGM_QUERY_DATA_SOURCE,
    S02__LCM_REGIME_BUILDING_Q_AND_F_SOURCE,
    S02__LCM_SIMULATION_SUBJECT_PARALLEL_SOURCE,
    S02__LCM_SOLUTION_EGM_SOURCE,
    S02__LCM_SOLUTION_KERNEL_OUTPUT_SOURCE,
    FEASIBILITY_SOURCE,
    *_ACTION_GRID_SOURCES,
    *_UNIFORM_PROCESS_SOURCES,
    NATIVE_VALUES_SOURCE,
    NATIVE_ARCHIVE_SOURCE,
    STRUCTURAL_BLUEPRINTS_SOURCE,
    SOLVE_PENDING_WORK_SOURCE,
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
    SIMULATION_RANDOM_ADDRESS_SOURCE,
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
    LCM_TYPING_SOURCE,
    NAMESPACE_SOURCE,
    PANDAS_UTILS_SOURCE,
    MODEL_PROCESSING_SOURCE,
    DYNAMIC_BOUND_SOURCE,
    POST_DECISION_BOUND_SOURCE,
    BOUND_PROOF_SOURCE,
    BOUND_NORMALIZATION_SOURCE,
    USER_REGIME_BOUND_SOURCE,
    LIQUID_BOUND_DECLARATION_SOURCE,
    AGE_BOUND_NORMALIZATION_SOURCE,
    "src/_lcm/utils/counts.py",
    "src/_lcm/egm/nbegm_step.py",
    "src/_lcm/egm/nbegm_segments.py",
    "src/_lcm/egm/framed_arithmetic.py",
    "src/_lcm/egm/ez_kernel.py",
    "src/_lcm/power_mean.py",
)

# Byte seals bind reviewed source files to the generated inventory. check_seals
# refreshes these hashes only; the independent callable and module contracts below
# still reject altered transport after byte resealing and require semantic review.
# Source seals and corridor anchors require semantic verification of the combined
# inventories before refresh; these inherited pins deliberately retain that gate.
_SOURCE_SEALS = {
    "src/_lcm/current_iid.py": "5ba6b89d75a0fe53195cd7377fde08d7e91b3187be2b32bedf5185b449e1ce8b",
    "src/_lcm/simulation/current_iid.py": "c675e9dffaa205b4049bf4c26438a70005766c368bc69a3dbb1d6e35cb66db99",
    "src/_lcm/egm/asset_row.py": "7a130991643a90a86349f7fd6b221950fb046489bf205677f3869a8607c260da",
    "src/_lcm/egm/continuation.py": "92cfa8d92681ba603ac878f1aba255512430667c149a866bc2ba3c778cbfa69b",
    "src/_lcm/egm/dcegm_query.py": "8d21916983c632df46ad96c57726cac6137322db985b2e0b9a9f7cb41b8ba9bb",
    "src/_lcm/egm/dcegm_query_model.py": "92788d75814f82e8acea2a9174a210997e04f68ec8253957836c8b327a03542a",
    "src/_lcm/egm/regime_introspection.py": "e1268a84eb34d1377c1fdf4672ccca73353190e0a77ebec405b7846fea11bbd0",
    "src/_lcm/egm/step.py": "220e68c78ab20c3f16d6f3f1b72b5429a5ac1694b3e3c1a374ba979cf87c126e",
    "src/_lcm/egm/step_core.py": "4e231dd2ba0ae20f62ce0ffc24fd36f487f9538da8c51eb0285422a22fbfa57a",
    "src/_lcm/solution/dcegm.py": "3ca93843fc7f58ccc015203247f90a89ccb7b1102d590af569e5942310033439",
    DECOUPLED_BOUND_SOURCE: "f86bcae9805cf3c72cb3be638d20070dc80b5cd3b94fcfdd2dfe85a0da4757cf",
    S03_EULER_SOURCE: "a5e95cde939adcefdf3d8310f6617df4f203ae3479bc318cd92ac51c020d0f55",
    S03_DECLARED_LAW_SOURCE: "635329b1a16ebe7b80ae57acfcc3fa3fd2c1b65a7b6778d1f72dc985f47e0e69",
    S03_INTERP_SOURCE: "b1898c4abced14f72e14e7aa7e5d2c20e53771122a0e964ea6d1d0dfbb858ab1",
    S03_NUMERIC_INVERSE_SOURCE: "57b4d9983131c5b4ce102c96372c21517b7d8932e0e5d576a4e95e585a7597ea",
    S03_ONE_ASSET_STEP_SOURCE: "c63df7e9a681def81e2fcf6b578f3f817af7d402b819e674f441989abce403ff",
    S03_PREFERENCES_SOURCE: "12d41ccb73192a7742267ffaa688fff3fe337ae40ef2373f6fe93541b8effcba",
    S03_SCALAR_QUERY_SOURCE: "709d16f6377da621eee7292c7eb33c7b2c40f65cf8916d7b80d149f6f87c94d8",
    S02__LCM_EGM_OUTER_ENVELOPE_SOURCE: "0c17015f9a980411b047387354f23886cc7aead0e461eec1b87ceca07911b7fe",
    S02_LANDING_COUNTS_SOURCE: "ac2f58e204659a210bc0f4dcb851cb8487aa538ac1b03d6d5a532be7b95aa70d",
    S02__LCM_CONTINUATION_SOURCE: "1c631bf45d0935cf5c9dec4624ca3724cec10bb47154529add810948c7cf4760",
    S02__LCM_EGM_BUDGET_SOURCE: "1c8fb0cbc8f9156950bea34c71c9ccce2c58ac17ec6b9f1221f8c8966f1031d2",
    S02__LCM_EGM_CARRY_SOURCE: "17c85a9146e68a8b57487754bcb89b879e62bcd391699d82ac4116ec7e4c4691",
    S02__LCM_EGM_QUERY_DATA_SOURCE: "5ca53117248f9009d4d9b696fdb46a361fa47c5b082757ccc9beb6ccd967bde3",
    S02__LCM_REGIME_BUILDING_Q_AND_F_SOURCE: "0addee8dc9d16c95cd6ec0f82d30bc457d1a0481d5bbea57f5711ffd28e25825",
    S02__LCM_SIMULATION_SUBJECT_PARALLEL_SOURCE: "66d70a81a7ec535d92933bc8841c6bbe7472e227509f88fc6d3bdd00b6c95e76",
    S02__LCM_SOLUTION_EGM_SOURCE: "c554d6e993da4b3b253ae8235a7fcac56c74a7bdc9c2537e84515d6c39af52bf",
    S02__LCM_SOLUTION_KERNEL_OUTPUT_SOURCE: "89ec60c1d59ace639e8df466f06c3a2167e9376633bc7e54c7ba050a84362ab3",
    AGE_BOUND_NORMALIZATION_SOURCE: "65601fa2da76f018fb695a76500e2917f518e98407fa6d2dd259a3f3a5bd1df9",
    DYNAMIC_BOUND_SOURCE: "11fdcbff51d64a5a8d7dd914102457469820270cf8ddf718e27e1074fcabd41c",
    POST_DECISION_BOUND_SOURCE: "4ff8b8c7b64be2395c821acb5f328765c7a9ac2c5198fd0b2be3960227837f6c",
    BOUND_PROOF_SOURCE: "e2d2d2ae850a3c4c91f7b094a6f603dbd77aa9f6e202d17ec4ab0af512fd1c35",
    BOUND_NORMALIZATION_SOURCE: "55f0e24bed63303ed9338798a876b6ad4a4b287b174d9513c2065fede183d495",
    USER_REGIME_BOUND_SOURCE: "a970b46791d8e8037b7d9a34d047d4b9582c3ed9cc3a12394f112aea28ab4ccb",
    LIQUID_BOUND_DECLARATION_SOURCE: "16118c29929f206f05925d88688edbc447f31c3a7aae1e73f022570e981ab886",
    FEASIBILITY_SOURCE: "f129bedefd4b78af9cc9e132deaf7a6014b2388e600fd2a8a64ff0a81b277be6",
    ACTION_GRID_SOURCE: "76a059f7cc7d7e7d2a99e0ffd626ddbd76b89724d5e24f3bb10ad86501268309",
    SUPPORT_TRANSITION_CHECKS_SOURCE: "ace7cc6b2d5a820ad1401912c0a744a67a849cf8c6bd050480412a9cbca75601",
    SUPPORT_DIAGNOSTICS_SOURCE: "477761e6556a83fd2e271e4ab6c258cf4799d62d10d3a2d46dbf31f69b7161d8",
    SUPPORT_PRECONDITIONS_SOURCE: "5df8b182b8cb3c37f6ab76719386cc809052bcc00cda63772d698c75071f4092",
    SUPPORT_AUTHORITY_SOURCE: "d04c1c11d281a24d4c8eabeab0a1376f6a8d7d3cbf337c5abe182ffcdef34567",
    SUPPORT_FINGERPRINT_SOURCE: "6ec013715e6faac2d13f7a27cd34c9c4dd10199379cfd4db4030aade3df39b75",
    UNIFORM_PROCESS_GRID_SOURCE: "f03298f080dfe4bace2e3affc82e0ca8e5d7adf20fe58978433c19305667b299",
    EAGER_STAGE_SOURCE: "d0a52778e4966ecf9bbac4ecaee87f85f49df0aeb4bf7c390f8599bf7d97efa8",
    PROCESS_GRID_RESOLUTION_SOURCE: "c9eb81f9442d7628793d6ad905b2e96e4655e9eb48bf3de32f636541b985269f",
    NATIVE_VALUES_SOURCE: "4e7302d6cacb53a20f6a0a08e7f57d6e074dedddd8f6a95944a9ed1ed4df7bda",
    NATIVE_ARCHIVE_SOURCE: "fe7947028f564d83ca513f92e398d2fe7c32e13d3fcd071a1dd9a1a24a722890",
    STRUCTURAL_BLUEPRINTS_SOURCE: "28bbe1b8f6301458e86ad7ee8bcf3bfb02184b163c3f3b038d88b0a261cf8af9",
    SOLVE_PENDING_WORK_SOURCE: "5629a30a65f402a6c4a542dc3287b045e6b823c7f005f4b92cf0ffaac0930ad6",
    EAGER_CORE_SOURCE: "d0049c2a38d3b77abc1c4cff1300f20ca034dff37b77edf7183537471d1cdb90",
    RUNTIME_SHARDING_SOURCE: "13f6b68846532c6631f27db1c84b47f6c866a992e037dfd8e40c791c16f50f90",
    COMBINED_ABSTRACT_PROGRAM_INPUTS_SOURCE: "1ade513e829c0bd8e053b90ab312e53f7b51a5a4da6ae1000079c5684a8017e4",
    COMBINED_ASSEMBLY_SOURCE: "7e1a7967d013827a9fe51862d304e686105ae7886b3539122289ef3866f5f94a",
    COMBINED_CHUNK_ADMISSION_SOURCE: "3e03fb9371067a96b58e310cc75da24224a5dfb6e1b0ba187f0710f73c0d025a",
    COMBINED_CHUNK_OFFLOAD_SOURCE: "4afae1dea7452ecada529d2ffd60026bae24c945fcf9ac8c021445c4ec7e073c",
    COMBINED_CHUNK_OPERATIONS_SOURCE: "481b768fc97b134c671c386336a25e9e3e78b631a0a3bc4643d9e74d0ba47e54",
    COMBINED_CHUNK_PLANNING_SOURCE: "807378c7ac745eb7fc3d42c27020f6b40d10d40da3041dcdb647e7e2d3e0d90a",
    COMBINED_CHUNK_PROFILE_INVENTORY_SOURCE: "85070e5f57210e201be248c1f2deae8068dce674112f13efdb0eb94c1dd7f7b4",
    COMBINED_CHUNK_PROFILES_SOURCE: "1b0dd0e0b3aef291b0b01373bd77a890ffbd2b7eca369ebb2bc9d2c50a284908",
    COMBINED_DIAGNOSTIC_OPERATIONS_SOURCE: "477f39609fff5371d88d941cc5c4a0c9ab9fe3692ad2845a3fc3a6befb633f42",
    COMBINED_FORWARD_PROGRAM_PROFILES_SOURCE: "44fcf687143efefbda1c29327f19433527ad63f92ef582920a75aeb94f93a0e6",
    COMBINED_POPULATION_OPERATIONS_SOURCE: "a9fdd59887462abc4e2f4321730bcfb397f69ba5f44f3db08001288773082ab8",
    COMBINED_PROGRAM_ARGUMENTS_SOURCE: "e15d79c337b5c5556d021c337564d58f629f87c0013fea3940de89168d65040e",
    COMBINED_SOLUTION_COPIES_SOURCE: "fbe188d3954a523522624823c769fda4c5831e138d2cb8421fc0f1453946886d",
    COMBINED_RESULT_SNAPSHOT_SOURCE: "7f80572d1135587912fccef482ce9aae26770bffa881b2bb63c99accbeefd0d9",
    COMBINED_VALIDATE_V_SOURCE: "70609bbd14dba6eeccf204a8f112c78cbd85799efbd007ecb880157259ecd2f3",
    COMBINED_LOGGING_SOURCE: "95f189e5713c7e97516b69e8298a79eea24f2baca9f7b55b62b56b81a1163023",
    COMBINED_AUTHORITY_SOURCE: "a98061fe4f30d97362d9668dc7461e8023052a46fad7513d3751637d7981d120",
    COMBINED_ENTRIES_SOURCE: "0c27d42858dde4a34b5e82a3622dee9c7b240c0c9c3284a2ba4a898bbf9a6510",
    COMBINED_STORES_SOURCE: "032482548248851cc53c7105a107bd9bde3aa110e1ca4e44f1d32460e7d47630",
    SIMULATION_POLICY_PROGRAMS_SOURCE: "46cbbf762052c1fe42b4c589c9ea0f486cea9edae1cb92043733a1a1c4d1e27e",
    PUBLISHED_POLICY_SOURCE: "c536ee71203f8eae66e77e9f4120e7f4dff9553ce528506d1a06b521f38456b2",
    SIMULATION_ENTRY_ALLOCATIONS_SOURCE: "41eb7238435a1c9780164197170c8de26650e1930dcc09c2e8d6454ad455fa90",
    NBEGM_SOURCE: "eea43dd08c6b9242e75b323e6eb3e7f2826298f49d2c2fdfd74f7852b78fa4fa",
    CONTINUATION_ARGUMENTS_SOURCE: "2bfffbd49e5a2e267fcfdefadd21cf4249dcc6979a6ff4889098c37a06dcc0e1",
    SIMULATION_TASTE_STREAM_SOURCE: "5b875618ea92b0470d0b7cd160349450e6df124ffec390bc7c6b133e7d484da3",
    SIMULATION_MEMBERSHIP_SOURCE: "c0c92de4e55be3e7b814a67761caa75e8affe835e1887359d432341f1d85a18d",
    COMPILER_INPUTS_SOURCE: "c2df405f63a61f7d5201d754c95bcca9fdc213c97c1533025080da80814cda70",
    SIMULATION_OPERANDS_SOURCE: "7645b400641b5d995e1a2f651dc5d6ccb54df3e16732db64a75765c5a5f31645",
    SIMULATION_UNIT_SOURCE: "74859e03e21e9861cc7847f49a8cde2301700b656b2556ad0fb47334d6e809c6",
    SIMULATION_HOST_SOURCE: "8e1198104cdb99c2e785cac9f7193d022b96029b58b90d35e9f5983f2e3f5074",
    SIMULATION_MEMORY_SOURCE: "26856b23317124b6221b00a04f8d7b52c1071644f4a4e026d42ca7da09ccbed5",
    SIMULATION_PERIOD_INPUTS_SOURCE: "cabfaf99af98887f1f142c8ddcd67cd84e9f59edfb9defffefff8d5aef08a9d4",
    SIMULATION_REPLAY_INPUTS_SOURCE: "6d3eda0b93377f146015bd233736bf983d4cd34c253a96791ed191259001d34e",
    SIMULATION_VALUE_READS_SOURCE: "4ed63570347beab876c3ff87854dae6d84e9e9f2aa8755a261d65897ea0eba32",
    SIMULATION_VALUE_PLACEMENT_SOURCE: "afd58b343d0385934c3add5552d7926dc3f9ba2700e5a40b26a65a62a75b1e68",
    SIMULATION_CHUNK_INPUTS_SOURCE: "33449e9833e30dbe50f455603916489145b5ec9484639acf0609e8927bca42ed",
    SIMULATION_ENTRY_INPUTS_SOURCE: "4cc1968398b169fe94d7ebde349b58192dabfca4bee93f1d5e98a0d3da0839e7",
    SIMULATION_RESIDENCY_SOURCE: "269a66f4a0abd46e3bb6b9bd8b9df6c3c73bd5507c2c236edea7d881efe78274",
    SIMULATION_GATED_ROUTING_SOURCE: "3a1685c78c9d1dd345e4242bc008f2858578725232b85610d12fa08535f82274",
    VALUE_TOPOLOGY_SOURCE: "9220e9bfd60e2f0c9221075ab62fb72ac5f7e9b25b3fc31fb629e7c3e50d3ca7",
    RETAINED_BUFFERS_SOURCE: "83631a3b82143b44a695d35436f10282a6b77ab688f47fbcbf69a1ac01a8c653",
    SCHEDULER_SOURCE: "772152206f9e9b8361f276e995123ba1b6109e9a476a455aa322e9c6503197ec",
    LIVENESS_SOURCE: "056bda4d48f26768ed8615925ce8ff89665bb8eed94b59d69f02701a95943dff",
    CONTINUATION_READS_SOURCE: "e512fe5403d78d23a330e3ebf5142457333ffcae94c5ad32f6973bc6e3993078",
    WORKSPACE_PLANNING_SOURCE: "31ca5578da208e786d82a99bd167acd76267380f3fcd169766f080a0a042d990",
    SIMULATION_PROGRAMS_SOURCE: "a7940b560b305a810286e1b06f8dff1b291efa93f78bf2b25bca0013ffb21f40",
    SIMULATION_PROGRAM_TYPES_SOURCE: "43c381884a3475516bd25d1463e83a6560e0d2ba3c697a290fad1d005d8dbf30",
    SIMULATION_RUNTIME_SOURCE: "88c9757287b5c16607becef498c1cc52cbaa809bbfd0269d337811ed31d858e7",
    LOGSUM_SOURCE: "e12061dd4f0f0176324182a2eb875cb6ebe4b97174091c597d46a622df93ff1b",
    ARGMAX_SOURCE: "b508d4e10350de8b54707db238d01c7de6c73b4ca6b0176e0c6c12ec762952dc",
    COLLECTIVE_SOURCE: "4cc0683d493f597684f5801909d297145d8bade0aa752c3e35cf8af2cb55edd9",
    MAX_Q_SOURCE: "274859ca8f33be8e6d8204cfc5bd968977c5648d6f9e0b673467c6f7e145aaf8",
    PROCESSING_SOURCE: "64bbf31068eeb656a3d007187775baad354f4fd676783d87c6ceba227babf1f8",
    GRID_SEARCH_SOURCE: "4821bad4a079ef3575ab1474f620ce369aec4d0ee84c88e2ddfa47116a4a6be9",
    CORE_PROGRAM_SOURCE: "7d8197a3d2607a3380be63408ae4cf2b8d8ebbe68b043cc3aff76bf1f5c6b78a",
    OUTPUT_LAYOUT_SOURCE: "515b2ba9c622053fb51f4711201b489efe12e1739114135f91038f1bedd0067a",
    VALUE_TRANSFER_SOURCE: "73e10e24135c76e550ddadd5c3982469e164983d3b852d8c77f77b7b9f81a769",
    FOOTPRINT_SOURCE: "229c23faa12723aeb8725c2a1681ee56011b317f31ba75896762876a0a143de0",
    INTERNAL_OUTPUTS_SOURCE: "e41e5581295b427e22ecb3f533281c030fec77c01cf224d1a844ae6f54fe28e1",
    ACTION_STREAMING_SOURCE: "bd9f6fd2b95a01cb8fed06664be6c0d99f94fe73f74e1cbf65d4be3d36ba4c8f",
    ACTION_REDUCTION_SOURCE: "26355ac3edf5da80a44b4f9754b71c29e8edffc68b1a7b7ac253b9bfbf749a0f",
    COLLECTIVE_ACTION_REDUCTION_SOURCE: "40a7abd9b738055f00e347e6ac056fe69bf2501ba41f7ec82b40a18bf627c7e0",
    DISPATCHERS_SOURCE: "a78d5271fcfc7437b68bd755ce7c2b8b0b78d5f26ed00a3fb4424133725da8b2",
    FUNCTOOLS_SOURCE: "8dbb11ab2f7bac3cefe98c732a0ddb429d9e4189ea79517cff3f54d90d405cd5",
    CONTAINERS_SOURCE: "1b58c79878a182dd61929a15d4e3a2f3c249411d19d5df8853476bc57c2a87e1",
    ZERO_SAFE_SOURCE: "8d54492b73e33461f750420eef45ba8c8281c467acd0c77ba2eb90281793c8d3",
    LOGSUMEXP_ACTION_REDUCTION_SOURCE: "4799ad9bfbc02ae1e5d5270a18ed81fe682f1004d63ae6cf796ff48ac5699445",
    PROBABILITY_SOURCE: "60ccd04b143c714360388fcce2730804a4ca729ea4db82e4a0100732e155e53f",
    ENGINE_SOURCE: "5077cdfc6d99fec6ffef74236640434c1855c602ff8bf1e341c31dffc64edb31",
    STATE_ACTION_SPACE_SOURCE: "3c6a3a1111abd0889d9af97e0e88c0cf7dc27f07b5381746a5011cf4d8b423e7",
    SIMULATION_SOURCE: "df7cef0d01c1d75ad916ffb34e804dce4306bd5060d2580becc539ab5c2d0143",
    SIMULATION_TRANSITIONS_SOURCE: "5cef83cc6d594d6d7004dee5cfc82dec5be33d669ac7779d3f9a393058e548da",
    SIMULATION_COMPILE_SOURCE: "2c54bd385d0205897bebd42c6b63d55eb0896a04e208786f4e0028b20e81074d",
    MODEL_SOURCE: "f9f444eefe0154a6fe7f8181214c5534d52bf523bb306f07899a1a05ff9834a8",
    SOLVER_API_SOURCE: "dee76933a640b4e9086abf89244e21b6e63a72244209f93fbaa39bb74dc08450",
    BACKWARD_INDUCTION_SOURCE: "9e0a603437e7b06ff277dd7f0ddec038547f15b8fb6e3f951033ccad8be23460",
    PERIOD_REPLAY_SOURCE: "ce332a3cf77a3d6c2344298b65234263006025de9d21b120d00941f7549f9965",
    INITIAL_CONDITIONS_SOURCE: "bb8e072ad4981e03f4a6980a1745d4b6ea78578529e041789576d5c6e3f156bb",
    RESULT_SOURCE: "a84d216d5063b60fb5cd14548266986da36d2f32caee7a4f806a2b5261872814",
    RESULT_DATAFRAME_SOURCE: "25a69be5f649e3b52dbcebd63656dd62311164fd4aa2b9d31051e14fe5b9864b",
    RESULT_METADATA_SOURCE: "ba9ec4f48e7df67b4a57a35cd3e781906d7ef64a35f3bc37238033c1c168df3e",
    ADDITIONAL_TARGETS_SOURCE: "3a4d95d5c88c801bb8bb0881a52cd8985e1cc17efdbfbf5474332e46b8d46614",
    SIMULATION_RANDOM_SOURCE: "54092637d7965da26670b1f1dca8b04cf780815a349a8a71dbb243d81f5a7c25",
    FOLD_ZERO_SAFE_SOURCE: "301fcd3bec1211b60872159fe585e55e9742be62a975dd76b33f6f5cb45971e5",
    SOLUTION_CONTRACT_SOURCE: "a991aeedf1955ef38a290c69d4746a451b79e870ba71dc4496b02d38fe45fbed",
    GRIDS_INIT_SOURCE: "3c720bc2240dd1cfd45855ce501f03f8a05abda5de1142b9d00208e517b9ddce",
    GRID_BASE_SOURCE: "2d9c7f1a0827f3ca172417a2ee2bf60b15c9ccb3dcedf3a94fb502e2d57538de",
    GRID_COORDINATES_SOURCE: "9dd0cf9e3c702023cd3cfa40a3164bf399ea295c57349223b8ab617c7f3533d5",
    DISCRETE_GRID_SOURCE: "0d729b4544bc578cb0a95dca01b01e8c202fa3dea2e84984b5c7b497c500f144",
    CONTINUOUS_GRID_SOURCE: "37eeadb95ca49855248da90a99fd5b5424f2acc28c9bf1535d228bfcf0b112f5",
    PIECEWISE_GRID_SOURCE: "88dea9f6573e8b88db1a4f7a28f77b80eeb7a06bc29e70d4314be76237a1bc9f",
    PROCESSES_INIT_SOURCE: "fbce6ed4c889cf4c30f5242c142b4dcf2fa9e186b75cd17be1557f827abff00a",
    PROCESS_BASE_SOURCE: "d58cde6ce85e550fa957eedcce87a65185a8b46015b7ad275fe604fd9383c5ac",
    PROCESS_IID_SOURCE: "6a622d0886aa903d9c5f98ae3e26e0a94eb4d16209a2481074a89fda99b1e308",
    PROCESS_AR1_SOURCE: "00720117e85027b487947dc8f28c028a56cb0cc7a3dc4f011f4d7932646af313",
    VARIABLES_SOURCE: "8445fd5d92144d013955027460ee204debfdcedf8bb5ef5d10092538cd18de8c",
    PARAMS_REGIME_TEMPLATE_SOURCE: "4b459d9eac328fd274ec8a7c6f21bae36233acceecd5a6fe32cc3e92f5e21b67",
    PARAMS_PROCESSING_SOURCE: "cd703789b7defbf24801e2450efc536600351b905d5b2a211c3a634bbc3cb4c9",
    DTYPES_SOURCE: "fc6694d64abf7d7a18b565d85986c71f71f376b020291fb9b82ce250fc264d26",
    LCM_TYPING_SOURCE: "23ea630cff08b58db30b95fdef72d4b961dcddb431ee5b2459826307fd655737",
    NAMESPACE_SOURCE: "9a33080663f01a6d65501663e8436b132176be51c8223c28b2e749d8dbf8b570",
    PANDAS_UTILS_SOURCE: "5410dc75ce909a4a1899bf87793addc1eda4bb46b5370d8d43a87c50f2043e62",
    MODEL_PROCESSING_SOURCE: "9a7ddc527fb531fb8840305f0f922ae05f1e294e37e91f70968b91c7b71cda64",
    "src/_lcm/egm/upper_envelope/__init__.py": "2615b9a364b7ca1c927ca68b752cff17b318268adddbb2c234c145ed59c196d8",
    "src/_lcm/egm/upper_envelope/segment_envelope.py": "f718075c3f0849107ba6289e96d8c539cb4392b0a5f6bf0c1dd490ed33adff0c",
    "src/_lcm/egm/nested_floor.py": "8983b037774946af7a90aa106268659a69963e2d8f5dd4b55c6f5532dd2dbb72",
    "src/_lcm/egm/finite_negm_query.py": "7ce90cf75e272da35fbe28608c88b2cbd1a93201e5c1b785b480b75f1e311289",
    "src/_lcm/egm/adaptive_nbegm_producer.py": "6e981492ef9f803816a1b6e579818e02c9754395b0dac2a19355c0361b115574",
    "src/_lcm/egm/nbegm_query_model.py": "d57f368e399523c823c3caa2082ae06cdec74a5c5b0da20d8a6a88abdf555ddd",
    "src/_lcm/egm/nbegm_query.py": "594f8e44f7a3ba541dfa35fab59a6e9629e2dee31dd63dff57b2ee07afc32702",
    "src/_lcm/solution/negm.py": "a9a92d7fdfca5093921082ea0c00a5757a004cfb9603091aa1d9904f99aaae9c",
    "src/_lcm/solution/nnbegm.py": "77e113ce8e966fc875623cd4f76008722fb4178318d0a20fa133e5996c1c5b24",
    "src/_lcm/egm/outer_candidates.py": "7a30d25c51c0254d2fb20a86d30b1865854f8134954bce5bb7a8c72eac218f2f",
    "src/_lcm/egm/outer_interpolation.py": "fd68eb3baa62055bf4a92a572fbec5e055a08809be044a647a31ff7982fc92a8",
    "src/_lcm/egm/outer_refinement.py": "7335546f8df6f1981544cca692d29823f4dc7b7af52d737b7df5c62ac62b8368",
    "src/_lcm/optimization/golden_section.py": "9e76572097e0281e76e59c0b7b430929cc9147754354780a6321f500bcd53f85",
    "src/_lcm/egm/upper_envelope/query.py": "6754d07b78d075c9b407635deea53bcac4f4f1702fb8bc366b3d9e937c6f8dec",
    "src/_lcm/utils/counts.py": "a9087a3c42d32ff0ca82cf24fda2b59ed57cb77efef94714a37a6ca7cf7a3f82",
    "src/_lcm/egm/nbegm_step.py": "c3e05112a95b3def7d47a31b5f0dc7cdcd652ef30b3cee7bddd1083e60f06539",
    "src/_lcm/egm/nbegm_segments.py": "658a74ca40e3da1c025c3d03e0f8bf695feb45953113521a4700f2c1d1890505",
    "src/_lcm/egm/framed_arithmetic.py": "f4b1fa68cc765fc1450222f7370256306d22b92a6488b40c3575e98f44e43877",
    "src/_lcm/egm/ez_kernel.py": "aef102b6dbc73efae7e103e34597948e8e354f3195a33383973bfbd0da112a72",
    "src/_lcm/power_mean.py": "cc82199d42488b7619917fcfb3d8112c7c76d99253ca8812cb41b2a79b5e3e59",
    "src/_lcm/simulation/random_address.py": "97264a0a575e5c1a7e3bb978cae5c0a2f28f21b7f5c393bad10930cfcab70af4",
    "src/lcm/shocks.py": "9a8c5a9d198cff657958d4c516626dd9d44612b7b808182e7498085d27247176",
    "src/_lcm/shock_overrides.py": "89d13b523a6d0270e144ea60077c5e5f7e306a1b9da7878a9a9b565f920f6808",
    "src/_lcm/simulation/current_iid_overrides.py": "c5b8d450cdad4980a694df285502b70cf1238762440f1302d664b2e84d964ced",
}

EXPECTED_DIRECT_FLOW_MUTATION_COUNT = 422
EXPECTED_DIRECT_FLOW_MUTATION_NAMES_SHA256 = (
    "e679f793253493d8b2353546edd92c9a927dff19ea2a5814bb5a68b3733ff346"
)
EXPECTED_SUPPLEMENTAL_MUTATION_COUNT = 97
EXPECTED_SUPPLEMENTAL_MUTATION_NAMES_SHA256 = (
    "3feb53024d4903cc245632895f4237a694517ed9f38f48d0e5b5feb7f5611204"
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
    "src/lcm/shocks.py": (
        "2aac63541b42c3a501333b8e952102dea716513a882ff79cea48172b9c317c81",
        {
            "BaseInnovations.__post_init__": "f881b9a60ffff3c463ce4634482537616c91472d0d6c0738852bd1abe0887525",
            "PhysicalValues.__post_init__": "a7ae8f2100a70874e481e2bab3ed06ba2099ba794e6467a542b0dce42c2e7d22",
            "ShockGroup._correlation_factor": "63131c3fc79d6ea15d665202027320733069aa201528b47a83a3290642c9ab41",
            "ShockGroup._has_identity_correlation": "64d074fcb1627d6131ef7e6bf2cbe83680517393bedb055cb111d9b0915f01ca",
            "ShockGroup._realise": "679cc71ccda5b7e17cc632c9332e840808a35dcadba561a7f2be551c444fdd82",
            "ShockGroup.validate_runtime_correlation": "43808302a68465c883813581ac0c7f8e22724e1e04d546536814358e30204d26",
            "_continuous_value": "2cfd7501a31336a6508ac7c448029e2bf07d1da0ce2f95604fe6ae553871f8f6",
            "_correlation_pivots": "00aa6d7b708eed0e08b2586c83a7f8b50bbb57bdfa208e7cc0dd569d93b93268",
            "_law_validity": "cb654b7e88ff187f30340f9c01e4e83d616aefb0ffff409a7b35e042feb59cde",
            "_resolve_parameter": "b9c6ee41d3b6ed31ab4773e27606374b3f3dc7061934367a97ed105acf7b1e75",
            "_runtime_support_validity": "e01d0604977cbf07322b20099429dda05d4b3b3d184ea7744b0bc7b9f68754c8",
            "_validate_correlation": "3382c36108771bbd9eb9d7276871ddf6279b167fbb2d7f7b470c3a2ccea15b57",
            "_validate_override_array": "37956f38f5bc7c2ffa72c0c54f08ee8d21ddeaa5195bb8b7d345503acfdfe744",
        },
    ),
    "src/_lcm/shock_overrides.py": (
        "819ad9eb2fd82474356ff86ddaa7945044bb20f1af414a0aa0eb4a957f23413e",
        {
            "_flatten_override": "edb14b6f253fe08300be4c6a0fb384d53654bf5a4460590cc9312c05eb82639a",
            "_flatten_override_with_keys": "b59c5584a47f2d873fbb65f9a90d02690c3c9917ead9ef29fccba4624f570db4",
            "_unflatten_override": "1948a8607582b4202a2211d79232cead2e7ed51f86edbcb86d84f9641b8bfa79",
            "_inverse": "62519a8fa4dac189ece08dadf6520d6d8ebd7f76d596f2239d2e16b030c22e61",
            "_recover_correlated": "324f06fdc6901cf43da77f8d6c262eb51021adbc80a057e9ee4195a1972aac34",
            "apply_overrides": "8829911a548faec9548d02c9f4a0f137a8b141b3b4d0327bb35a396ac3e5840c",
        },
    ),
    _CURRENT_IID_OVERRIDES_SOURCE: (
        "45debd7415d001aa21108dd9f7cb9c074d9d9e0ca24229bef50e78d91c3b638c",
        {
            "CurrentIIDOverrides.schema": "85e28ec3820472a9ca5093f22676172b6d5f1065775a44bf3b6a913c093c5fa8",
            "split_current_iid_inputs": "2fba8863d5a348f2840d1b9461d90a83f2abd83756ee5e6389d2e940ab719f3d",
            "resolve_current_iid_overrides": "4eafa97e34f7d61f901797a3ae4158d237adceefc4e9b05ff9d4de7723eaaea0",
            "gather_current_iid_overrides": "e26858641ea06a09dae949814aa29d6472bd37922a888674e9eec707fc232b5a",
            "current_iid_override_prototype": "12e8ccde0d5f3ddabd21a92cc85f093128928333923f3e215ffdce898388263b",
        },
    ),
    "src/_lcm/current_iid.py": (
        "87e10aed8c30598cd6f6d37285d713364168ac9519940f3c5ae2259aad3eed14",
        {
            "CurrentIIDPlan.wrap": "415ff5f7a33f87b3a4b144f1c8e40fa396647c10eba3f8bd0a4d8f55ac6f9899",
            "CurrentIIDPlan.realise_nodes": "10f8f048fa4869011b54643a8b587a935d5cb357c4fec4d81fbe3b052b91cd49",
            "CurrentIIDPlan.realise_information_nodes": "a5a534312b1dfc8c295533986ad5be93cc248a3e4c779c5e6000f9772ad4f3c2",
            "law_parameters": "505b635e5e1b780512a6bff80897ec809f2f6134c30842985ed95bbc3fcac209",
            "law_informativeness": "ddf05d5b6c372f3eb607c047e708b1d16f6022934a5d9488e6656513ad5f4e3a",
            "CurrentIIDDrawPlan.wrap": "481df62f00544beb8a3df789a4be5d684713e4218aea5ef7bb8409339a3eab00",
            "_CurrentIIDExpectation.__call__": "b9c862c37800e73695ff6d3cd8f312ac30c5b8e2072061a5ed1ec873d35e10f7",
            "_CurrentIIDDraw.__call__": "6ca53c447d665625b4dee743fabc7d51ee1f2e87bb3de7cd86881a1891c423a5",
            "_evaluate_node": "3e61f7799d0388e6211fca9d7495a09bcfae32ca1573e6e6314bad050e0f1e78",
            "CurrentIIDDrawPlan.realise": "39b31890b7521b6fd2888b0d584a7674446fb853dcca60f392e8dc1f16e34b91",
            "_draw_base_value": "4554e685f57a99188070a0050d53d1caa2b36c79d9c023ed629ea25aece79746",
            "_law_inputs": "b29f6f0fc9a9a5e0e206bad5d4888e74fdcef311ca2fe8aac47160398bde06a0",
        },
    ),
    "src/_lcm/simulation/current_iid.py": (
        "19c78f588220c34ac39e5f2e15fa953053be9041c0370087c8934fd3cab7eb5e",
        {
            "prepare_current_iid_keys": "8e85e5643cdd4c24c9e1d782028ebc20ac090776b48d514ccda415ee78a11d70"
        },
    ),
    S03_EULER_SOURCE: (
        "944d6c6dbb26155884fc73c821f47f602361ae70cced81f2175ebce029824ad5",
        {
            "invert_euler": "4cf2bc5c6acf096274efe52bd4f82fc57afde7abdc86e9a2c7043e7a4a251f8e",
        },
    ),
    S03_DECLARED_LAW_SOURCE: (
        "8ad22099349717932ce8fd500fd77db9b11eadbbe3953c1caed31dd8c0198a17",
        {
            "build_declared_liquid_law": "ac9193e826a0e5e660137587809c9acfb5a7e0291c90b141b2ae38f7e0098bc3",
            "_DeclaredLiquidLaw.__call__": "6cfbcac48738a6a76668bb3daec63d71c915d6b52a742b248c2d5aedbf93623f",
            "_landing_point": "a19cff0149007de4c87559d2ca46c8aac04b1a35a920d273a02782020b608e0e",
            "_fail_if_law_reaches_past_the_post_decision": "adb005aa0ff6c06981963994baf59dc9c30ee0f5776347354138de1d47858749",
            "fail_if_declared_law_is_not_increasing": "68452a37b5c0062fd509841dc67026f3896ef5e64c3a830d9650bea5a8cac3ca",
        },
    ),
    S03_INTERP_SOURCE: (
        "8c8658fbbe19f1c6c43a7c07ccc6c7f9b20972815d43ff9a220d80e7fb3cf6f3",
        {
            "locate_on_grid": "d2efde2d8211f14cce8e9bdafb34e01b96886533e7984a9e3d1f581d1a3701a1",
            "_derivative_between_nodes": "fbdfb439dcb74798763a1765aab150286981237d4e84f3194ea8cd4229102707",
            "_hermite_bracket_derivatives": "e80d9f40cf6bcc04be13b0dab6fe05d284807dd1af2d4fbe5cf3c01dcec0fc3c",
            "_hermite_coefficients": "e24b38f878a2602ec6b19b17884329b8ebe2001a91c33f471e904724d71ce2e6",
            "_hermite_correction": "c46878cd615a109cd479878f126522bda30f348c6e1297c47e31dd431bad4c10",
            "_hermite_prepared_read_jvp": "9660c7ba1f3fe66802f87edcddf62a352647ed13aa93a80d23b79bb0a4848754",
            "_hermite_prepared_read_primal": "0bcdf81a2810aada544cb28216179606fabf49ecf37b18ccb25bb6f673f08d13",
            "_hermite_query_derivative": "416d4b35f85258068bb7b49f58fabf28486970764189e7899aa7f65a8d2dcf13",
            "_interp_between_nodes": "4518e5255350a91f0236cf7d880c351259945e19b18cbce12f2d458a06e3dc09",
            "_limit_slope": "52eb93d842fe5e7e54d6c4f8e293e43b25f63f6e9cc15269e1e4f7af2612b5f5",
            "_linear_prepared_read_jvp": "e70cc0af44acacc256a6174cf613210d1bc984582b980d3b089a2383850acaa8",
            "_linear_prepared_read_primal": "53703a429fce004320e58e3ad9965585fa80927d72593a023ee25f291b8bd6d7",
            "_linear_query_derivative": "9d787289a36cda29e41870c2b2f3cb09e5827bd0bd0c193610c32ee3f941872c",
            "_read_prepared_row": "1006696c73bf21d67ff773527c39d338e97d1597a4181ac24640d19b9b4f8df2",
            "interp_and_derivative_on_prepared_grid": "5b392da953a3d6ec02d45ffd0a832647abc18aff4cf6667adb2c6f64bb4323cc",
            "interp_on_padded_grid": "7bb309d76796ace15cac5dac4fa8364d63a503d45a80a88dff24454c75675ec2",
            "interp_on_prepared_grid": "597ec1e1cc32af27f82969334a6050723504e77de778cfa0e07a9d7a4caba0c7",
            "prepare_padded_grid": "110b9fd67d18b2a4b3e6234b48a3c3aca4cc1bc8407afc50631558fd27230bd7",
            "_hermite_read_at_fixed_query": "85544cecd9e5737937ce8cc514a5b19354eed1d3d171779f3f9b800e994a5292",
            "_linear_read_at_fixed_query": "e5ee72aa389f42657ac6a97621d9d93e75d9a330e84581f408df4fc953c49f1c",
            "_sanitize_gathered": "72dbdaa706e9540c1eb5833ae4ba933310565f91a3615f5a7969553fa80c5a86",
            "interp_left_record_on_prepared_grid": "d88caae609ae928a65c968280a9c718ba5ea3580664f745b94482891b310ba66",
            "interp_on_regular_grid": "71992b703d659849d9558d353a8314a42650a9d8d6fb41c452076b6189137681",
        },
    ),
    S03_NUMERIC_INVERSE_SOURCE: (
        "fe2ba4922ba677dad9b962345d3a43b1dc7b8e7ccce8028bfb8e3b2c3d4a2bf4",
        {
            "numeric_inverse_marginal_utility": "1685c0ae1c2d958e69677046134f6d2d8600834295dafc198d35a1183b62fc01",
            "_expand_upper_bracket": "e3fca2527d416e6563a46185dad34d7cfa01def58ee526e3b7c3bd94a78c0125",
            "_LogMarginalUtility.__call__": "241b4f538a34a7775f56e413e725541d3dca36b6209c0b24ef49489cf21c456b",
        },
    ),
    S03_ONE_ASSET_STEP_SOURCE: (
        "b33bf81bdc79b71a9210787bae58e801ddb00944ec73580b16f21c2b172f69e4",
        {
            "egm_one_asset_step": "9c71ed5510a2abe16d6c6a0cc190a757a497994966a5186def23d167ebf2f938",
            "_solve_viable_feature_row": "1fb71647ca75791e9c8b774cac31140226f24d4953e006ecb2e36548d071ec8a",
            "_flatten_egm_savings_support": "e2c2420f75b4b36f32694ff5f000c54d6ff0d41d288defc148f0164e211c429e",
            "_flatten_egm_savings_support_with_keys": "643e1e72a5f632e8429d8829a37d12183472c9702ea14514f17306d864064a42",
            "_prepare_augmented_savings_support": "f01d756bc7b1c07debec9f404e6fadd1bff50b76f134d8666b53a334d3299388",
            "_unflatten_egm_savings_support": "866053fa72fd8c81ebf56a6f2809abe0ca0f904dfbad9a9fbdd82cfbc8dacd31",
        },
    ),
    S03_PREFERENCES_SOURCE: (
        "74f2df9886eb4e6628c346357b44c4cf998887e103728418d6e53849aa082c0b",
        {
            "newton_action_ceiling": "e34e9f8c2814162ef9ecda6a7d1d773defae93ef425c1e7bb5e2188e4c266b4e",
            "get_numeric_inverse_marginal_utility": "8f3757b16a2bba65acdfbf62393a530692abc2b3156cf5280a6668c1de38e0b8",
            "preferences_from_utility": "3e50df18f34f5a3f88c7e375eca2f7fadd2ca812b537c95b1338bdeb4ff51bd1",
            "get_preferences_builder": "3cf2c8f0360d36184a8e912ff90c94903e771d52870f2d2dbabd985c36f72246",
            "get_discount_factor_reader": "7516c622ccbccab48e778af98a445b67b1b4aa621ffd88dc6704e990312a0c6f",
            "concatenate_regime_function": "8b1ccf7173e4a730762ba94f6abc4b0064dbb98154bbd64b77774e746ed18705",
            "_elementwise": "f4bb75fe03e8f400583a0f866fe143c613295f73346b16f65a5fd2152a67319e",
            "_Elementwise.__call__": "3e91d64bbd8b1c61fd53caa400c041e39d1d0f2dcb90dc54a98f971571a4129b",
            "BoundUtilityOfAction.__call__": "936b1b53bd9486a01075c20b60046e37b20cb92dafa5ccbc27ee7b769c22a1a8",
            "AnalyticInverseMarginalUtility.__call__": "fd7a15208b88452c2d03d9220769c29e5f3e9ba6af498e0cf45c5923cc2f3eb1",
            "_NumericInverseMarginalUtility.__call__": "d82b34b60da6901207fa057d710698733de1ad7d357c613980846989b8e11ab2",
            "_PreferencesBuilder.__call__": "903d50608a73a37a86ee5f06171e78b6d08c2198561a207f1df833a2a858ae77",
            "_DiscountFactorReader.__call__": "590c8f0a27732210299102338efb8c70ea3d013d5a66db30ce19adf789bb73f1",
        },
    ),
    S03_SCALAR_QUERY_SOURCE: (
        "160e2ce889d4a859206444dd3393ec552b97e5c51becfea0854f0bd16acd1657",
        {
            "build_scalar_egm_query": "16c64c19727c802d0892c9529abf4f49703891289a5da067723a7017346d519c",
            "ScalarEGMQuery.capabilities": "97c1acf69c7dd500d9f263c861ecf832fe217552c85ae2923716fe48d32cdce7",
            "ScalarEGMQuery.evaluate_floor": "25e9e4a37f6624d0566990e6b3230b1e9bb708c7229ee62f7a9d8ba5eacd4012",
            "ScalarEGMQuery.__call__": "053a48530815e7614062c5082d4ddd6f19ab0f3bf6a40853daff64c3d590ed9d",
            "_weighted_pair": "a63a34b27ad1e06e13da6bd4937dff959eb5b08ec3923be71212e1565808552a",
            "ScalarEGMQuery.prepare": "0542998764bcb78ed7d8c594b5d6a20440fc89a99c3b20b4994c051c27d710e1",
            "ScalarEGMQuery.read_prepared": "3958b50bd2b6942457a927518894f7d10f40540dbc81ac0ca069d2b796f5ee18",
        },
    ),
    "src/_lcm/solution/contract.py": (
        "42b733d3c85f26c0480cfa8b234ee4c70fd79750d6b9d77cff040bc898fed8f7",
        {
            "SolutionKernels.__post_init__": "1af79c468efac956445e2c9a1e816b260851b589681fe69754c72d395153f795",
        },
    ),
    "src/_lcm/zero_safe.py": (
        "32bf37e7db6997c36d0faa5a152ea883f4588c650174d5805c3f4c805682bffd",
        {
            "zero_safe_weighted_term": "1f9682a923d63877e5ec668bb404bf92911278ee806e99b5ef60c098a9c82dd1",
        },
    ),
    AGE_BOUND_NORMALIZATION_SOURCE: (
        "82d419b21e138b4af9182ecb1eb89a5b167d804fd0f64eca2ea0f20224cce0f0",
        {
            "_representative_regime": "4f6137dc37333d66e0cbc14b6e68c007a0f402a30189a302318758cd0b66052d",
            "_regime_has_markers": "28456000043a2cabeb4b417ee9853405d405bfe8b6e12b44ed9635fce2f649b8",
            "_collect_function_markers": "dd469b1c230a57d7ba2368697a924b94266208d6fe3900f3367e3e002aa5c242",
            "_restore_representative_floor_source": "4ffb71f1966b89d46ce5baf7964e11e4b17a46bf3095ed1a3f55e1518081641f",
        },
    ),
    DYNAMIC_BOUND_SOURCE: (
        "98d6f6c9f3b5580d64c4526eab4f00f3eeb02430e1401209dc25499b4b3b3446",
        {
            "ScalarFloor.__post_init__": "0c6e83daf4c061cf5cd846e49bfe3ea06bf12e1e3636962b3742d36e408b9a7e",
            "ScalarFloor.__call__": "335229fda495386a981dfb9cbad601a2734945842ffb084aa7d0dcbd6967f351",
            "_ScalarFloorFactory.__call__": "021a814f63a55e5a75a06548bebbc242db92d74415cd64868ea9b45148aacda0",
            "get_floor_node": "7de2956f7e461605802f1fa1ec0ba6a1e6e797e64e55021e5236d2c4d1171741",
            "get_floor_source": "d7ab7aceedff9f318bebe82eee38fccdd496bb636d9054e26c499a1dfcb6e403",
            "_bound_surface": "8e47db2328d338759919fc77e2f4a99ac9c81b8e5baabdb3750033d9e29c9ea2",
            "validate_declared_bounds": "3c90c906035e0ad89975437352e10f3e29156c86d9b7d4061a2c41182a775d75",
            "_resolved_rhs_contract": "31cda272fef9bdcfd345768ee3f696aff78eb9e0477186d0b56b40362a44af64",
            "_walk_rhs": "477512c34af977b4b73b0057c3e56b7caffcc08aa54cbd04b47a2c3bbbf469b2",
            "_validate_scalar_trace": "d23b27b4f7d060d7f1093e6176e56407a3b5a780079adcde6dc4847dec276f6b",
        },
    ),
    POST_DECISION_BOUND_SOURCE: (
        "bf35e83f02ac09406eac2dd7722edd6154187ad6a59008119e439626ed90d090",
        {
            "grid_origin_bound_spec": "2ad5ca872ca937ee9455f6606c30d4f81f5995156c74c9c4304e90043433755b",
            "_PostDecisionLowerBound.__post_init__": "741a1302a031d8823455500f2471faecd9ee4474d747b671353d9c53109ab506",
            "_PostDecisionLowerBound.__call__": "36cb8edd373144c40b2eb742e2ad274e81f37c0d34ed42da57e852afd99e78ca",
        },
    ),
    BOUND_PROOF_SOURCE: (
        "4be3d14ca613b21ea8cd3bb0ac6195e784c4991db9e8f8b57050d419dd16f161",
        {
            "lower_bound_declaration": "544d2c467a8976cc790d529c1371d1b3e10ee32e4ef2644b70ef486a3cd1c79b",
            "proves_the_savings_grids_lower_bound": "28a158196cf7594964967030d3ba824bf1ea57725ae29f3ecea5f89ad5b8ea23",
            "matches_installed_literal_bound": "6a7ea707ac99b2ede4e63e26dcb8f2798ef250752ab7734c5ae82856b50953cd",
            "_SavingsGridLowerBoundProof.__call__": "cff1c48d67ed9841c4be9a235c6d7a55ecbe8b74cc845f88e456afaaa4aa753a",
        },
    ),
    BOUND_NORMALIZATION_SOURCE: (
        "6aca1fbac79d79c91750075c02eab1dcf72a17ec3d80209c87d580f2855a53af",
        {
            "_as_condition": "dc68d4edb55c754abfba0a38fe3100b635e670dc00eb3c1ee515ebba412c6881",
        },
    ),
    USER_REGIME_BOUND_SOURCE: (
        "9cc05c78defa22042af634f0840e63e689875252f2e2126872d6d958735a6f82",
        {
            "Regime.decomposed_functions": "14e9da26e8f480f3d41b5b1307a899bbf328aaa3bfc7241fe29177add000e4d5",
            "Regime.decomposed_constraints": "8f3528801f6043bb4c70eebc6f88e42a64b6e7c514721abc04350e0e45a79c97",
            "Regime.get_all_functions": "4721a0482130332d4d96277c97020e1925b8dc59c1ee0ca815f77016f194fe61",
            "Regime._restore_floor_sources": "1cda33ceebab85e6c63d15e62e5b2a1a69c9c60b8e37e23fb0d3bbb65646c9e6",
            "decompose_constraints": "f2a315f499cec4019b3a116d1e7695e4bf75c4b69812ea42c5a61073a0018701",
            "Regime.with_engine_functions": "d18c34ca58cba8b91fbf59a0a8bf03f057e719f4a6c25a788a623864e9046ad7",
        },
    ),
    LIQUID_BOUND_DECLARATION_SOURCE: (
        "d7fa3ac8ed4747703df866eb5e6810396333b9a8329c92ce10efba461136d99d",
        {
            "post_decision_lower_bound": "3bed1940e3a65135f139facb93dbeaed4de5be3b95b5cb77156baffa2dd7adc2",
        },
    ),
    FEASIBILITY_SOURCE: (
        "2ba638375db093b4f81d885b944f4c6ada988c41e1641f8adb0a5d0d95236bbf",
        {
            "CountedGridSearch.__call__": "aa2fb5423c29121bafae479f6e7d0fe5c658a8e0785b19bb0ecf9f713c832d3b",
            "FeasibilityKernelOutput.__post_init__": "86b70647353aab0cba371e2414e90c5e20ceca94c7dfc903f033928d5d1b6279",
            "_flatten_bound_status": "927d19863379fd04d664d9257c6c8d0694571e234e0e287202971bc264922f73",
            "_flatten_bound_status_with_keys": "0b34cc1ec4c4ec1600e53141fff93d963d97bf4f993b4f59a73bf3d2b510467c",
            "_unflatten_bound_status": "94e7365621f266cf09054cb632cfe0d3b123498528029069cd90138cb4f98233",
            "reduce_bound_input_status": "abfb1b30702d44dd348d5d176079858ed03f6e136547c6a181168a711e4203ac",
            "summarize_bound_inputs": "e9683d831c3c6b7d6a93fd33e94e5d2bfdfb735a8311e3772631f926834b0ed3",
            "shift_savings_grid": "9ff7e6ecc5eafaed0f44786ec29b857141245b1c734170c6de122b183feaaa90",
            "merge_bound_input_status": "863e849be315918ab0295ec97387db6e18d666f746ce4a4f4a13f9e071a4928a",
            "validate_runtime_bound_status": "69c3f042b58a0c671d7a8aeacb5bf3c8a3dc665122459ccb6597a97ba8866b45",
            "consume_runtime_bound_outputs": "bf756b1adde834a23d7f8b9ac0c8782e14c4e63ba3594b47d88401609d63fabf",
            "_flatten_bounded_result": "5c22614c143da6a9a3b0e5a077680042b1ee7f33c5d0511d978e51de8c2bb0d6",
            "_flatten_bounded_result_with_keys": "c0451fe26707b51155fb64afbb55c35314cb010279ede871c9389bc1c7e2da2a",
            "_unflatten_bounded_result": "acc1718c80562ae91ac36f21d64fd916caee1caa63ddef4f1897af6a6e2c75af",
            "BoundInputStatus.__reduce__": "33d2023a9031d6df24452121ed7d9e6c21cb97f6d589f16aaefe6a6cca07f584",
            "QueryInputStatus.__reduce__": "bfc887d0254213a99e6aff139a18fc1087dbe21a07c9e6d1da05d90516039892",
            "_flatten_query_input_status": "33c2f1a195bd106b2bdcd6975c6a874e3fc14c81e9385d25c7f8c7855de0cda9",
            "_flatten_query_input_status_with_keys": "e7e87d2d4b6745cfcc836ac775d37c7ad00aae26049d5b8bd4bd9274a4fd2cbc",
            "_unflatten_query_input_status": "142486aca425d21e531e79440cf1e71859dec6a015718864073a72cdfb47479c",
            "reduce_query_input_status": "3abe4ed7820131b3e4dfab3b6c3ef3c456f471435f1b8a6f62ec15ec779ac754",
            "merge_query_input_status": "43d6e3c63706c3238912cbc043ca6c16f0fcf03c9790ebd5f8454f9e6f2977a5",
            "validate_query_input_status": "f9c2c2e0327fe510b3ae03d82f339565503a3e610515466172b3ce10e0033e66",
            "ContinuationReadCounts.__reduce__": "5b13f55ffd9b93e2955b58c8d8561ffb83aa98a9bc17b07b671d1a03a850ceb9",
            "_flatten_continuation_read_counts": "2e4e457a7d297d7ce25951289d70d65870e1f7c9cce15c3062a84434d547052a",
            "_flatten_continuation_read_counts_with_keys": "80883939f39137ae6eab82eaabd667d6b0e5603af135b97c3d2ab975b96c6252",
            "_unflatten_continuation_read_counts": "af3a2fa1b310179a880d138c68d95a7f5d7843676a5fc872a616251c0bf591a9",
            "count_current_feasibility": "a9c7ed1619ce2e010ede9a9924233d0266eeddc064599697fc69c4a02b799730",
            "merge_continuation_read_counts": "8a5a05cd0ee21d8ac4e2c57dd2290da33b5a4243ae7eaafbc328f948f74b3222",
            "reduce_continuation_read_counts": "15687533f32a66607570a5c5d2c3af81a4dca3f9d54eaa6e3b425bae7f941116",
        },
    ),
    PANDAS_UTILS_SOURCE: (
        "284a4b1f9d476d08bf0dcccd3d48bd49115d4fbbcac737c8fd65536c602d7062",
        {
            "initial_conditions_from_dataframe": "d40f121eb30a57420a0e8b9cde4830871d5745e2832a1378a1245ead322787e5",
            "_role_codes_from_labels": "7427bf2fa16abc7494e981ec61e9b4c5fdd71328cd880f707736cb7c443ea3bd",
            "_write_pandas_array": "fc6fdb8c0b16669a7672c6eac52c951bb92f6180ebf607af7bd15907260c0a4d",
            "convert_series_in_params": "e10df3d0045050c92df6636935501d439955ae1895182aa8fed315fd5067c020",
            "_convert_param_value": "78704aa755b151ed88d3c8799d6729a3d538c5acf4fca91c6ee911cc39077403",
            "array_from_series": "ff66ccbcbb61993ba38099f5e1c6fd321f6e2400d6a12029da8e4f60dfa63a34",
            "_scatter_series": "ccb844566097dc7bb6412b0f81b2e28d60b84b960bb73ee8be3f9acdb6681bcc",
        },
    ),
    LCM_TYPING_SOURCE: (
        "40fbc826679d8f83a911d8101edb51927d4eff3f2d8ddbb07682d5b42802c8e0",
        {
            "CanonicalArrayWriter.__call__": "640b0dac3e722ee0e7b2a9c62216bf746d2f5b3ad3d83ea95e4a7982313241d5",
        },
    ),
    DTYPES_SOURCE: (
        "17d71edcd836eead7b3b72f717137062a52ba8dac5e4ebda6bf3b1343d8c7c1f",
        {
            "canonical_float_dtype": "ff7daf524547e5f62b1d854c3a2a606c3b5903f3a436eb4e8a087a0fda2ca3fe",
            "safe_to_int_dtype": "49f5760ac936afd767c31f97f9641d04143c81d7064c0fde6012849f22d0f023",
            "safe_to_float_dtype": "da9d8dfc4260639fcd6308babe91146a0b174ddbae228821d2332c8db193f32a",
        },
    ),
    PARAMS_PROCESSING_SOURCE: (
        "b212291a1158aad65846a2f43e4c2bee9189b30d6f0516270ca7857f337d6d34",
        {
            "cast_params_to_canonical_dtypes": "c9a5717bb221f1fd3f44a7584912784aa7237fd08b32a1f4b6404ae6fa0b2494",
            "_cast_shared": "4501d56de0c550130adbfdbe0541e69fcaf397b0f1c8a0631caf55c204c4cc0d",
            "_cast_leaves_to_canonical_dtype": "3c3edf1ef9bf2a7826f2f2550f7c654166e9a13def60491bcd634fead8524adf",
        },
    ),
    SIMULATION_ENTRY_ALLOCATIONS_SOURCE: (
        "84c53c0199868bbb197e7072da4b134a3e4c4d03fe13c99b8eac4c082f42c9ca",
        {
            "SimulationEntryAllocations.snapshot": "a2b219011564236e6b8f863d59086d802a43ea49a2f0f73a01b99358c7faf49f",
            "SimulationEntryAllocations.solve_input_roots": "abbd1f8e3a3956b718cefcc7de938bbaea7cc909e6a898c1e95607cb60c345f3",
            "SimulationEntryAllocations.__call__": "d16c9d254801ebbff8ef1ce18125a3a9fe68073d83672bf8855eabf923da6624",
            "SimulationEntryAllocations.publish": "dfc3551541ba25f75eadcc85908e3c6e4f046ec233327c08b8e29a0598093005",
            "SimulationEntryAllocations.pad": "cfe078114a6c07cfa9ad0ecb0bf0411b8985742aee92a5359d8bae8315d4f0c5",
            "SimulationEntryAllocations.update_solution": "3f36dab7d76a35ac58c318b9e6b61ae0f8b78683f79c731bf7cd0cc3620863e6",
            "SimulationEntryAllocations.close": "3badf65d0ee4eaa37cb23c3816d74d1de15c12b1381bc1e77528ba968ec50543",
            "_pad_initial_leaf": "b5ce0e61085c360e97280d593948a3e8e775e10424620a905c038f34142dd7a0",
            "SimulationEntryAllocations.place_solve_parameters": "0fa3d8f5c5845c7fe8a7fbd7c5ef4cfaf0c5c8fb00efee1d63cec2dfa750b6b3",
            "SimulationEntryAllocations.copy_solution_leaf": "e4a19117088da19af2513976ccb2673f23fb21799ef5893dd6559ae091628de0",
            "SimulationEntryAllocations.release_foreign_copies": "096914c39bdc6f0f1668cbbad08d7d4b389d79448df16888ef471573b827e06f",
            "SimulationEntryAllocations.__post_init__": "ffa19e8bc72ec7fc9d0382c3cc4f6c7a501059dec278e6e3b7a1c5cc1fb27e18",
            "_EntryFootprint.__call__": "6a64901028e6ed4c92615e2b97beec064da8ad89c81003a9b66e46c2358c0a4e",
        },
    ),
    INITIAL_CONDITIONS_SOURCE: (
        "438ded9ef5d826c9736cffba905f8198e04825b2e3a3b5508b1e601a3945e660",
        {
            "validate_simulation_inputs": "9ec7f8f50c4629f2de4eaef2865470018a8ce2e1dea616b1c453ce2b4af1779a",
            "_preflight_memory": "ca5810a8d31174676bdf127ae9811cb04adcf10762d8efd568b404334de01807",
            "_discrete_initial_specs": "6e837939569bad3b27093ab4a14ce38ad16756f86910f37127432d07291195a6",
            "_pack_initial_summary": "e674412100af558aae66243e25060f3ce8538ffd4ad138c7e0effd58e1e3357e",
            "_read_initial_cohorts": "a720c185a1ac905d76fb08dcdba34b7a964b9bf8cf461ad31f7f4e29c86ab8e2",
            "validate_initial_conditions": "0ae9a5ae11d6f1354f47ab38a51e815ecc1669afb5c8564f3816197b6938b004",
            "_collect_feasibility_errors": "25a87a11f22ee779312be43af4bcb0d9a02b951cd5471443ed86ff86bd6d90d9",
            "_regime_views_by_start_pool": "c88cc1b948f62f55650b7be5d63e58096b570fad5c434328addc50de805c0da6",
            "_constraints_read_age_specialized_functions": "fe0bd564dc6a3bfbff42a2109c9cccc2eeecafd71d27036f48431af4c19a4b83",
            "_check_regime_feasibility": "ea0c4a0511cb581f1dd1a71921f2a6e194ba66ac35b7ad94e5f42dc9bebcdd09",
            "_regime_feasibility_mask": "5fa671841a7963ee31b606b8ab32ee742e1459305224332c25dbb27a2abc20be",
            "_run_profiled_feasibility": "e5b5204b49c6c9063036b61adc76522b7ee614ad0a0d358b916862b03295e646",
            "_batched_feasibility_check": "9a4fdd6ce7fed08a0ac43507ba4f4167d9683bd84bb5b044abfc00f089cafcbe",
            "_evaluate_constant_feasibility": "598395298e67b40806256a1626043ffde0d3df09293e86a0c221cb266315bbd3",
            "_admits_any_action": "fbde6d75f84cd48f2234779915cd957f305e6d8833daea30c3cc58c151d38e01",
            "_per_constraint_feasibility": "8b95dc48aade9ef6e575c1c56b31999eeaae4e2bcdccc2b30e78da6013607ffe",
            "_initial_physical_joint_states": "86f8eedbe28cf8cfe0f69cafb8968bd619de673540f0325c495de0e2085df68b",
            "_replace_joint_coordinates": "926e669f6474495d0b914bd8dbc6268f3182efbd7cad92b95ae6d232ab397bf8",
            "_format_infeasibility_message": "b6683eabf3a727ebf1f61b8aa98c2818fa0fd00a7c9f295692fefbad40c75b03",
            "_gather_feasibility_inputs": "557c76957b3e216d9a95166daee80fefdd0af163633920b30b1d7215596d8515",
            "_subject_feasibility_flag": "58bfc237bc5b8a51897de10ffee1eb33f7a2cb6504c094ed48544946185bd404",
            "_constant_feasibility_flag": "f7b994fa04576c7157f8bab30ef0e17126c181ba874cbe493c1e19e6be056f17",
            "canonicalize_initial_conditions": "d3638b17606badee06c1e8c337a8b0f8a66d0342696116269b7700ad2a10bcc0",
            "_CarrierWriter.__call__": "697bae0eaa97d3806c3079466e8c3de66ce08eec331d66402da3fa02958a573e",
            "_build_admitted_initial_states": "ca3e5c754d9c86385c7302441e2199c338239d89edde06c5620874793afc68ec",
            "_cast_carrier": "5a08758007268730f195c44800937e9274ee44167d6ca2ae6ea00dd75ea4800b",
            "_fill_carrier": "0045c72197d9e16fd510a8530d9c0866327182129a2383e902a9619030be4844",
            "_initial_own_stakeholder": "4cb016c4b23d90d4b0c7aadfd99e693addc4fd283e97a9cb7da90ee516504adf",
            "build_initial_states": "6bc9d8948f600be8b4099ad779be639d6a30c8a66e4612562cc700d9050144c2",
            "trim_pad_from_raw_results": "16dc20f6d29f59b624dfcc9e7575238dca7d2fe7ba5dbdde75046189856a1778",
            "_build_flat_action_grid": "65b4591dd17c07ed9e30c5ddf8f89ead9a2899c2af663b9a1d1286fb9e990e0a",
            "_is_any_action_feasible": "ee8f21a2af6b34da408acdfed7b1f684b536d4356510a99a5735a80acf5af249",
            "_is_feasible_without_actions": "97f6b2168cbc1416ac271dc3fd43accc1bbfd312859420a65e8db59a49de7f73",
            "_validate_initial_bound_status": "0d1720adcca73c904ea9d361eace61cf934ca1036dc5b367411f9e4d8932ac88",
        },
    ),
    SIMULATION_TASTE_STREAM_SOURCE: (
        "81ee7ba23f63b0e6ef6454e93a13a6c402cc541b75bf618b9f6c5b55d55593bd",
        {
            "create_taste_shock_key": "76cdca8e4c1fec745ba4041395bb23454129fc5a751b89d0b5dfddeccc689fa0",
            "prepare_decision_taste_keys": "c7cb5f5016fcfa051c34fe1341f88fe6fc92d04e134501387ab1bd03133277ba",
            "generate_taste_shock_keys": "a728f3d9793b2994e7de7ffd46216ac1f082b1e4ea60511c954961adefabd2fe",
            "advance_simulation_taste_key": "49eecd1a5724049acc7108921c86640236140477924fbf3b561156c7a449381b",
            "build_taste_stream_addresses": "97cde050565e0adf61ac0246010ed764501437aa65e0761a91319d98ef578a7a",
            "_encode_subject_row": "9206dbef80b1bf9607d6cfc42db24af9d7a01ee34ee04079d13c61582eeba82a",
            "_advance_simulation_taste_key": "acb5787c60e5ed254b21ec0d7bd6ca0da50c19b8b4107af884676689c940c3a4",
            "draw_taste_shock_keys": "792bd7c17be4be6f2590b11aefc9d6d7a41545852cd787d3438f4d5bd991f7ba",
        },
    ),
    COMPILER_INPUTS_SOURCE: (
        "7b9a7fc3be0978782bc6162ee72cd245f1a3052812a69521cb88e93bb8e49ba5",
        {
            "compiler_input_paths": "4ddea6614dbc34f8fa7a95647d17a20f2cb2f328fd268a136b80806920e99a57",
            "_is_none": "51dadee63190e851305dec4e7c999b489eabca55e4af33c3680c1d6be96ae475",
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
        "bebd2d1ef6990c756d686115f8312b03452203881fd9a749abab63df335cb516",
        {
            "ArtifactFootprint.__post_init__": "508bd37bbb30d5e0511ddb2de9d98f6c1c2318ab928635b73b413c942d72a4a7",
            "ScheduledUnit.__post_init__": "2c144891456c3626aaf1b62591a35e6c004e5d182b1ceb12dab4ce10d0b4d7d5",
            "ResidentInventory.resident_bytes": "f3aeaf93e4e584e9cbe96b3cce8f6e0b49daf95e202bbba0dce89e7ae0a98aa7",
            "concrete_device_bytes": "63f9887ce54803c381bfc61eeb542463e06437ee04ae27b35a4af80f40db9d9f",
            "plan_resident_bytes": "e9c46841e6663fe1e7b33a952b2bd7c08335e87e70ca813cd0e3314068ab985f",
            "plan_resident_inventory": "ad8c2a316cade31c67c6ee4ccfcb13d93334971bcac34a9bddb65b39473ece8a",
            "per_device_footprint": "9c9e558685bfb57d4358b1f5bb7d56ee27a8e159a4a061d423cabfd6c069c5cc",
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
        "9c7b05dd63551790635c1abddd7837f18346d677c88c46b85d05c416b74c4724",
        {
            "SubjectArgumentNames.subject_arg_names": "46cc9cd2b5af3c1a622819cf136f77ca33152cb4dbe6a00b2a111d50848d1f4e",
            "place_simulation_arguments": "e0bf104ab444c69c12cfcf3783bdca7df6b89e82ebc4da68e5168a1ef8ff6f7c",
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
        "9cdbb9b4b6e67022a6d40fa706324a82533ffa23a24edf8efdbe550c055c81d3",
        {
            "SimulationUnitExecutor._live": "efff4a14869f95ae6371a201bf42c1bfb3ee5b4b682af149d4979a02655714f4",
            "SimulationUnitExecutor.dispatch": "a9054123390d9b4d6a7b1f33238a3d4e28fac66480005bbfd0c991a07ac43fb8",
            "SimulationUnitExecutor.close": "2baec2cefe68783c238e3f553ed3471b6520bb66f40aa7648a9961c3f760faca",
        },
    ),
    SIMULATION_HOST_SOURCE: (
        "e0a9230e3d41e3f808fb60a95a9790e817e657f1715147cf4b9c6506a6ccde67",
        {
            "ProfiledSimulationOperations.dispatch": "20d68817ed760c316ad119ea002808c35581fe7b27553abf6f3efc37c469e178",
            "ProfiledSimulationOperations.compile_candidate": "fa16641b0058003789569be59cf753f15129b9ed0e3f906896b7fe593f64e4f6",
            "ProfiledSimulationOperations.lower_abstract": "81d21584d0d9278530d57827ab32a5c93c5f20f27632145ccf943802e950fd0d",
            "ProfiledSimulationOperations._publish": "51d44b3974028c72140f3d2ca7ebab375c93c51707e296d1bb648a7a817d97bb",
            "_abstract_operation": "b66a5ab8009173a1fc8ed6904037c69865803a8d9911920558ffcfdb37bac59c",
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
            "_validated_static_arguments": "019f4c7280741a96a875804ba30c9747807566d8983346e27f6a3023c15636eb",
            "_validated_operation_function": "c46194846f437f82ce69f5134881077482f056bc1c0a9cf98db61d74abf809e1",
            "ProfiledSimulationOperations.built": "adfa6a6e02536c8d81b4ce4da1981ae935cde7b4569a4154a948bcfd8ccddc11",
            "_bound_identity": "2d62fc7d2cd30115b45cc3cb608004212f2403bd3d012496523b53ad5e7c7e94",
        },
    ),
    SIMULATION_MEMORY_SOURCE: (
        "450708e0973082ecf8daab7526b8dbd9dbd62a9cb7e824fe5d5407c0d9475d2b",
        {
            "SimulationMemory.__setattr__": "f1f1f1e5c28e4d0099b7ed4b0c8fad853cd447efb65ea98af0268f8a5e16c5ff",
            "SimulationMemory._period_snapshot": "5c2d15995d981f85cea04e2d0c6b608dbd33916c604115b122a17efd69a7b774",
            "SimulationMemory.snapshot": "708ef98b6e55c80d81e4c7c8181325eda8b1d0cc8bc15ef5d804e7660e6a22c4",
            "SimulationMemory.budget_snapshot": "166c1642b6b2392cb0c6d4ae8c5fd8d5ce7d2d67116527193aacc89a469b8bd6",
            "SimulationMemory.set_chunk_inputs": "fe3e0c3a0903270b5b5414fd2b50585480a9bde3c4a09e1f0a14e90af29c6e28",
            "SimulationMemory.publish": "a1da3681994370b876c3f4386f2ae60d8c8886648281a7216b6941b0351029ad",
            "SimulationMemory.replace_outputs": "e01242175d3a6c8261b8f94b8e8361147019e0d8e667d176d8c314fd0e044e90",
            "SimulationMemory.set_derived": "37d12b85b0715cc5e304cd4c1c883368232fff1b233c0c978c0153a8d9d5e37b",
            "SimulationMemory.hold": "3492055e2604c528002da2df4b0dde23e40962e193dad2dfff1a22a7bfea255e",
            "SimulationMemory.before_transfer": "16a2151cf001c60c14e40f6a7511d39e3a687b079fce838f97258ebefe03048b",
            "SimulationMemory.check_resident": "2dc1e19c92cffb46032324e68a41c655a31cf77c702f060caa1b328516988c92",
            "SimulationMemory.run": "5f032e3774d15890471a774391ffb79b18bda53f8e5d7e95b7076a3f3a69db60",
            "SimulationMemory.close_unit": "ad93e305391d4eea94af93d4561ff2ff0a699c691f24084f5f1bcb63ae589838",
            "run_simulation_operation": "e7113ea2849245cd18779eb8f15e3a83a6d59389b6baec8be5254cd5d26e53d8",
            "run_simulation_producer": "a04ae6add579d765738f5a57b20e79a9f20b991ee0dfe63ab9eb6646d9fd10af",
            "SimulationMemory.__post_init__": "1df0e2ec9057b88a7547c16aefb671617055808c1aefe7db8ce6a2849445cabb",
        },
    ),
    SIMULATION_PERIOD_INPUTS_SOURCE: (
        "a2a62f06fffb741e38e1b07976444739084d53690c902e72ba55328a56617e05",
        {
            "decision_reads": "a10afb4f89a9bd1033327b41f58c57f5f0d6607d3e7fd88a735baee1c3e6eb8c",
            "unit_value_reads": "ae8d2579ea2f3427357e26f2295327dfe2c13e2c5b9d2c0f082dbb5985008c28",
            "gate_reads": "9e57467e4a7147bb1de88c0010c73960fd3b966ac76b3e5e575bde8791edd261",
            "acquire_gate_inputs": "a49be6fadbf76e8f8b2fe9b289049711ec3a6fe48851caa0fe31467ecc249abc",
            "acquire_decision_inputs": "e4934ed4a98d252ef97eecbab40da1cd94ec141a0f48a86f6a857b4d1506e9a8",
        },
    ),
    SIMULATION_REPLAY_INPUTS_SOURCE: (
        "a7a25f330bba3f78426f57a582f234615a75ca3dfe276d74be45683a05f6808a",
        {
            "replay_payload_reads": "a84c80ddfa8290879fd983970aa59d046ca604cb5d2b2eefc464829c8c23499e",
            "place_replay_payload": "968b4ebcf3a8f631c756bb31f830be2973189279515c4cddd48de3e54874cd9a",
            "_payload_read": "7564ed18eb615d0d35dc24b9586dbc660dce2450e1674e505858518840345153",
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
        "ab92ff72cdf347d2aadbe49dc2d4dddd3b9ebc56fd5e9d4e5468628ace04dc2a",
        {
            "SimulationEntryInputs.footprint": "1953575a6f4762e5d24f20332868a5dfbffd884d41ac4334b8455317dfb55355",
            "capture_simulation_entry_inputs": "08797a7cb0a3caa760f9eb24fd99d21b3f3fe239192a61bafc9301dc706ffd14",
            "_caller_arrays": "59f486ba1f3c9b4c1214fe02612bcc95effed5d636f01e942c129fc073ac256e",
        },
    ),
    SIMULATION_RESIDENCY_SOURCE: (
        "3255465cff9bb0a025f0409f54438692ada6d2358dbccdee7bc5fe8ceeb28cb2",
        {
            "OwnerLedger.bind": "afdc24c3218a517f12d74f84fccfa7e5c7055bbb0b3f0ee659f375bab0119949",
            "OwnerLedger.measure": "c2f195a9582262aba740819a0e87359822609835e86c1eb34493d3a090cfc99f",
            "OwnerLedger.release": "77cebd7c193e2c14e76d9aaa981186b831626477111f14c27a999477296ac280",
            "OwnerLedger.release_prefix": "fe266006cae3b79cb1031bdabf42ed650d01ecb15163217c319a9c8dc3e9d8bf",
            "OwnerLedger.clear": "50605dc06e8805dc6e41bee2cc6db6e1959af9255cc79d63a88d4d2853f0e481",
            "OwnerLedger.bump": "8997d8a8f18dcd560f6ffd669ee0c2611ae836262925d4644aa2836e48617323",
            "OwnerLedger.union": "d77839f108e8fdcd189ead2af075daaae002dfb5ab61136484cde3c2f9bc9e28",
            "OwnerLedger._extend": "da833202cf9271e36a3bb69e39ffa5b2bd284854b9ad586d3916e830cf01cc0d",
            "OwnerLedger._invalidate": "bccac5f4bf4af8cca6bb493816b1c6bbb126121848f9895f8dc4a3ede25aa087",
            "DeviceBufferFootprint.__post_init__": "6b91013dac5e169e65e44c92701ce26bbc790d4e51e49b73b96725b8c02b9cef",
            "measure_buffer_footprint": "e60555f9cfa4e000990a64339f66c9916b8153aaa5d2f73794f1a073b3d59d8e",
            "union_buffer_footprints": "22c658a719f154e8b942a9d275e2fce472b8e106ff724029ccc9c1db3d930ad5",
            "resolve_budget_devices": "cb3ce34525277ff3e667132fdf3f0a0504fdff44dfd554cf2ef21c2f71a24d30",
            "resident_bytes_by_device": "b9deb15cd1ce6a33cfdd717609212652b34dbfb7a55d92203d37ebab6e55ee40",
            "require_transfer_headroom": "eca32b846445bb763dacf2b2f63ec0f95b87cbdea388a7dac97cead0fd65e525",
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
        "459380b89036a1887a51d2de505b82ba24e928214799557da5dbff55f035df03",
        {
            "expected_V_rank": "3fb0dac3bed9027261436cff2316bd122cff31e99dd1b825977272d09e6212f7",
            "placed_V_sharding": "2232aa6bbc1e01337e19bd39637a254e34d14e9e6296407c2aceb6b2467333c2",
            "_get_regime_V_shapes_and_shardings": "c712d242ed26f7d444d84b9ae61add8b59037222ebe6c533b387a86fad4c5b4b",
            "_build_V_arr_template": "84aa944560553fc96f52d39bfca74c8707d376dd5e9493235a907dba9c26f396",
        },
    ),
    RETAINED_BUFFERS_SOURCE: (
        "3379b5cdaf022dc2dd5f0ae9ef53f490cdcf695874043fa82e85448138b08042",
        {
            "retained_solution_buffers": "1bde0842e5804f402c80ec2fe90a4ed38176a8378ae0b037cfc346c9a36cfe5a",
            "_RetainedBuffers.collect": "17eed3c80658835d2cdcf0883971175befbc8409f5ff1166149cef32c2aaf89f",
            "_RetainedBuffers.collect_lazy": "e0390ecdd855167c8261898538e6d28ae62b33ce7170c9ec1c96d2d824edb634",
            "_RetainedBuffers.collect_authority": "402406b70f1437e1d6427b96af3bea127d3ec5906a9a97eb9b19de07c8c0b081",
            "_RetainedBuffers.collect_reader": "fbfa12ee965f9d9bb3af77c76ebd2756b94dbdc4050d40f46a6573ae263f5f79",
            "_unsupported": "5eb49f472abdea16b95d9f0c9a4f6e56297f0cf65466d18cd87cb4ff530dd21b",
        },
    ),
    SCHEDULER_SOURCE: (
        "8283a044f90a05234d5452c67ed9f424b66aa2179d559c9a572515dc9e42b7f9",
        {
            "buffer_identity": "773d4a78840d9f58d52a23420af0b9942d22ce1e7d84af0d6e84fb3f0207716d",
            "shard_identities": "5dcf1a2c1f363f904cd238cebc7607d4a23350cd2ea7e7673c93a4df12334559",
            "shares_a_buffer": "d08dc057ffb25ffe0583d85004270348f45209039cddbd99762d9eac9166dc31",
            "BufferRegistry.__init__": "929753ce0b5f18af3d68aa592eb75102496f6a9c5307d27e068f9779b548d913",
            "BufferRegistry.declare_not_produced": "8704b1abcad0ef81ac0f64f94765e97887b39d1ce20630b04d72f7da6e445d91",
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
            "replace_leaf_by_identity": "b1361487a5260d4fe741c8638829962831a45ab77af969a1c0f0bda8ca4f07de",
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
        "52b9e56b51072ef34e7c44d521a530c8128942b81c1a6f03972ba5ff28bf4672",
        {
            "continuation_leaf_reads": "38923496bb171a9ddb4a1a0112c19c0a8169574a24f4d15399319be81ecc1204",
            "published_continuation_template": "2e18c3dbae9e6155fb4c5a0889381bd1b2a696d6c311061d29ef19861b9a0483",
            "published_continuation_templates": "984dd8ca446a1c657f2532a0ef21f8297f6945526ab9c94b014b00ef598b7998",
            "rekeyed_value_reads": "604602c93cb9636a59d4650e3a756d017d4cfc5c5881a957ac88d6218050d438",
            "with_continuation_leaf_reads": "612f637430a21eb0490a390045dc5ff5354ac066c7f5b69d71be49c0f1fed013",
        },
    ),
    WORKSPACE_PLANNING_SOURCE: (
        "c04f016cb268496a273a79896c388bd1e3378c3e9f6f43bcaaef55dd149ff6bc",
        {
            "workspace_width_candidates": "6aa7f1ab2054fabf4f9232447e62b0ee672ef45210639a2be5465873e9326603",
            "plan_workspace": "5d28ec9a75ef889f43f7981036a9fb5d2e4d6edf05357fc1cdff2c5bacdaa239",
            "plan_axis_free_workspace": "300710be5b1c0444f87f41c2013c9b47768d16773a135fa4f6fa5bed5a87443c",
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
        "4ff1a25936b1b69c3e09114a6cdcd49b491e174e74e3fd1f4e04b44db5a5190c",
        {
            "_ChunkProfiler.profile_widths": "0dc63e394a75fcffdc24e7cb113e39913ec081ef0066dd17315d4470526c3d14",
            "_simulation_chunk_profile_key": "977fc3e83a2ff80979796f7fd8f42bc684552f40cb79e2e5856e96db7d916fba",
            "_independent_outer_candidates": "823294bb69d6649f9e4b3b9da1df9c070c39fc7e2945db56cd9bcbabac98da73",
            "_independent_anchor_widths": "1c9375085f39b01649366f937b1eb59c5fc120ba611efbe8870a4e0c9d40d078",
            "_plan_independent_chunks": "030a717fb2310c2c85cf9daa0a75ec918dc8e33dab682ba642fcbba27f723015",
            "_profile_independent_candidate": "ce068ef18ed9af1fe995d9d31853fce46414f4f8508fa94c8ada778556213491",
            "prepare_simulation_chunks": "80ec54d9f8d9fab4b75480e95eb0a61b0e33220d3fba5e60260dde23c01d63c4",
            "_ChunkProfiler.__call__": "dfdaeb448cdff047ed8bad87b46cbce7649782bd9ae525af5b240a0d836aa2f1",
            "_common_axes": "277ef7d313404ccdd8610cf7f5a4896b38710693206a213f6728e4377cf637ef",
            "PreparedSimulationChunks.require_chunk": "4c567102baec45a7386ba33c9159ea2ce4a29fa292795c29b41faa18021e94a6",
        },
    ),
    COMBINED_CHUNK_PROFILES_SOURCE: (
        "43497d6181a3cd190c101158b16b6c7c70da45e81ca266e18d989a09a1e428b8",
        {
            "profile_simulation_chunk": "cb8521bdeb105d6c0c800c2efbb3d51c4485bb728d971a6fc5e3a98df9f4353e",
            "_period_copy_reservation": "0410fc6db2147bafa3c0db2e5976d7356831c35a8c3639bd333b476af9e9ace4",
            "_policy_read_sources": "c6a1175eb04bf64833a8a8ac696863b939d03dd33c6a77c446bf6d052249f5d2",
            "_retained_read_source": "ed76584900138b8fb392cb2a789f1a4c848698e4308662c02b4fe4bd7b6ad78e",
            "_profile_next_subjects": "59bad3c0d44aae40422e4b1295fbe5ede5a65bae38558f9c227af6f61982df0b",
            "_profile_population_roles": "99510aeb32383e09d2b2972ef0ac6354dd0259aa494d77cefd9aa43d9fecad8e",
            "_profile_outer_storage": "f140b97925fd77d4b73acef4d6cea9c9cbc2db8cf67fb70d692be6130eed11c3",
            "_record_core": "1ca4c13d2facd739b83c8ee4c9cf265ddefcf982b4d937a6daedf39e2393684f",
            "_profile_initial_carrier": "1ef1aa933990fbe5e52659aed6d1f949d76b44ae0d840c0307c2d8764efce7cb",
            "_profile_keys": "21f1136c5ae608fb1cefee9b6299348722e95c9a67467f7a76779a002ebc62c6",
            "_profile_taste": "b095824f8ad7289d68073bebcf3e115d2b1cad84e7a329f127f339f360449f34",
            "_profile_entry_key": "306ad82a059557d575d50c24ed8c51a0d172ed97b6de476a99945bc8ca10599a",
        },
    ),
    COMBINED_FORWARD_PROGRAM_PROFILES_SOURCE: (
        "ab734e0d0a7ea5d3d26cef8651e0a59c9b480144addf923ae61940e93fea5eec",
        {
            "profile_forward_programs": "e1e32d001801d76967782eecb8572c0d911dc899975c0021b7b9ec57ab93b9c9",
            "profile_forward_unit": "905f7c01ffe4442a77383403c24aeb7008d6d9598faa2356e4db450e9a03fc86",
            "_profile_finite_decision": "5ca35d7e729d991a97f4d22e3f1a65b997c0403769ae6b76e5a35e512e5e66b0",
            "_abstract_policy_leaf": "96e133ac576cd9b40afbf10907084378253b43bf9376d39d0faa3743d58b0cd2",
            "AbstractSimulationProfile.__post_init__": "cbd87055b30ad84f7904480cff150f5d376033767ac3e759cfb7a086d7d19123",
            "_profile_program": "1cbd2f4a0f8b81da9b783f3dcb70dd38801bf372e48063e5484b3a3cb3bc77f2",
            "_prepare_program": "03310a0204cd76f335ab1e2241ee669684b83a4ba3d739acaef515b130e2f58d",
            "_concrete_widths": "d8085e4c2d9d3f24d6c5b34f26239fea9eec9fff79d8e52e23a2d539c7192eab",
            "_stochastic_keys": "dee659c1811c43e460e7e3d77191d61166c7df35b322128400244b3cbe07aa36",
            "_shared_tree": "ed4cecfe3eedeac6b79cfd078e28b839e9e2d3d943980ce932a82516378e9303",
            "_shared_leaf": "bf682e2f0efa2cb4f6fcd8356560d0e45538b59fd741f094e7245ca79b5aff14",
            "_placed_abstract": "7e7f6ef3a1b6d02436a1eebde1869d131f4f11680fff287a44763acc2040723c",
        },
    ),
    COMBINED_PROGRAM_ARGUMENTS_SOURCE: (
        "89a5b03accff7eaffe2d16f5675c7db7a02a06e77c27da95dde71f8fee38e759",
        {
            "policy_prepare_arguments": "8e82a8646d7b7553c1ce0748d6b321a72ad2a6542caff4ec92f5820733bbb8dd",
            "policy_rank_arguments": "451148291aca1118a504f213973ee246b11226789800fbcce38c47f866003730",
            "gate_fold_arguments": "14264886b5ffff2d7ecaab619eb270bfd895acf096cf32cfec5b259d0a74ea91",
            "gate_route_arguments": "f5d91292d00f3a029a7ba682d7482af2704733fd7bf7947885b4473a72f3f195",
            "decision_arguments": "e7ea6625e6c98c9ea34279a71b22cdffa88c68e4053baef49ada8d2505297a0f",
            "transition_arguments": "e684d0566f3f0a0c1c4e1bfe8edd4c96bad445c62236cc4244e95b68409a5975",
        },
    ),
    SIMULATION_SOURCE: (
        "acda34d63c2d76164a8f6e570fa3da5395f1353999428a7c7e266b9f329199dd",
        {
            "simulate": "1cd13b9d16d2e7b786e14424450f1dbb8f326de768eae3ec0e23defd43b2bcf8",
            "_simulate_regime_in_period": "308be89c19c2b43dcb9fee3751d93ca815755e19f2926909c0f23434363e4812",
            "_execute_finite_replay": "6ded3ff0186eeba07f2880caf5c21a9d08c6396d9806d49afd7985d8e10cab21",
            "_compute_starting_periods": "f50f6f3c87b21bdf082f3f3d71793c03574dfb93f1aad875d8689b8f773b65a4",
            "_concatenate_chunk_results": "e1217d0e707ef6a8b3d00dcef6abce948d065e125fc2886336bf537d276c1a41",
            "_validate_period_values": "30fe242e786e7e38637d380ecad0aa4d1689bc71c1fa9cbf2ae3c781e011a3c3",
            "_validate_simulated_value": "5938880b9d43f485c98f21b559d9b6d9326adc0dfde9dbca9d6a831e128cfd00",
            "_simulate_subject_chunk": "ca524c5572e5a19bdab3a68abbd4b112e02392efed8a6e2389bfb63f31d9e1ef",
            "_bind_unit_executor": "a787e30bafb80053703e1f4ce0c1d84feb5ac522cd41349665c600dcf9f1eda5",
            "_lookup_values_from_indices": "ebc4a036a447857f061c117b2eb0c9b9e61d5f17a40e90ea14a6e6205233ea9f",
            "_read_external_replay": "93258aae4ce02f4d6852f784b35412ba2bbf1d865d14c31cc107911f4119021b",
            "_replay_nnbegm_candidates": "80189ce07b3cb1f06f04a4a2daa1cb098eb45649c6ab8620ff1383b8d0b48fe8",
            "_prepare_nnbegm_candidate_bank": "88fd8e8d35a769e86a683d4f0a3041927b9c7daf0e6327a61ea8b143123bfae6",
            "_rank_nnbegm_candidate_bank": "e58788ebde739cee544df7a8b5cdddc0522ca5f08836ef6c2492bb69b01991c9",
            "_initialize_chunk_state": "ce97dba339426c926a50f326577ea8a09cdc8f18f9193b3ea3cdcc5403956092",
            "_canonical_Q_at_actions": "cef0bc3a11bbae4ae74b98a55dea15e8cf6007e6bdb35b5c4863d3fc65b9f14e",
            "_current_iid_invalid": "03f8f1abc5e8c93c93b5e606af988dbc139a874fe75a93c7aca63dad8900c877",
        },
    ),
    MODEL_SOURCE: (
        "d53013f64805ef33de28ca2c1700d8d8e32eb31489df326bc9789120cb95f197",
        {
            "_validate_sharded_state_capability": "ee8b7e8a97fd413d2376a655558ea7b2a2ab61589a0ddd624ed44b69dfe5fa8d",
            "_supports_continuous_sharding_vocabulary": "1ce9646ee043fb623720d6367b531c860ab0020969a58ec62682784814843387",
            "_supports_unsharded_continuous_process": "d837bd4f6c97be377ae02dca1f2044e1eb9974c6e7e08bc48866bed0f6886b18",
            "Model.__init__": "09bb295447634b0d6a39600a67707990a1071f07d7f4c7184a4ce36d36c61470",
            "Model.simulate": "9edbe1d363a0ee6dcb5d2e4746f13bdd61444f2e72e4e138af112dfb78b4faa8",
            "Model._open_entry_allocations": "f04741303c8c423e4cf4a738c7385390001be4fa113ad691be00e412fe15f07e",
            "Model._check_solution_result_structure": "4c8dc7a19d5e3ba4ecb885a34ed0bf7632ac79fea02be289f4e76ef53496556c",
            "Model._consume_foreign_solution": "45d8b6331ed24445eee7f89f0cc49cf437e6ecad1a3c4b1b1b6aecdf5754e1f2",
            "Model._resolve_compile_batch_size": "27791b63c37282ce72ab9e537fda65cd20796d2a494a804392862219401dba7e",
            "Model._resolve_solution_result": "92f777d40e2bcdd96e0e597f61529481ec859268de344cc40312232e550942c6",
            "Model._snapshot_solution_envelope": "9a4c08dc99912920df4b670212a15313e4e4e79446b8ae6a19a11fe6ba66910a",
            "Model._declared_solution_authority": "c1dc4eb2b584204f4b99cbe83721b2eac44d16021d5d4711f3ddbf7cb4c11dbf",
            "Model._model_fingerprint": "3b5ede2a24fdb3a9ee1993d3b0ad6a9092e75b82089d341dd7bde349ff458816",
            "Model.solve": "8b4a53a6c70dfa2ad61c915bfe96bffa6e74d3afe46ea15c009778992b47d9aa",
            "Model._solve_from_flat_params": "595c790306817d45e5e0b7684409e8b4f086a857303143ee0bd910815a2184cc",
            "Model._solve_compiled": "111649145e833e065853d4d3edb2ca659759c5eb05ac4b9e9a7fa471fa32a1fa",
            "Model._consume_owned_solution": "045498125d95e899a4aaf261ed15b6b30d0ba19ce2b61807bcf6ba0fac72c45d",
            "Model._build_external_replay_readers": "62d5d2c5bd473f779ddb81644162279e0a3ca329105b42636220a862391dc771",
            "Model._runtime_regimes_for_shape": "f524f330e24be0042a19fdd3e877959c5361d2932003854910c23365f90d4902",
            "Model._process_params": "74a3f61d98286019425166a134c5c9e15cbff36c0528197b57d21950ecb5eb13",
            "Model._canonicalize_params": "85e45144d2b8188b46b095e46d5c229e08f6c98f5919b18a9cd2265723ecdb72",
            "Model._validate_canonical_params": "2dc4199f507ab9aee966ee503409f81da63ff1a0411d428ec3593bd9fb105f56",
            "_simulation_programs": "d90b290c2d54c4e305a413267ff02bfc62c018b7fd8675aecd6fa403de914c85",
            "Model._lookup_policy": "b0ac7d08a0aa120265b64033bdce362d59fdae020c88948c514619ebf9fab2d5",
        },
    ),
    OUTPUT_LAYOUT_SOURCE: (
        "fd5aaa06d7edcae90067a183eb30a3f0f553020b02227285c0d3b76d963a60a3",
        {
            "_assert_output_leaf": "845951305b74c0dbb2a93f7598fc9d9163f85279eb047595677fca6ba8c38f48",
            "PlannedCore.__call__": "82afebbf7faeab894944b641a83549ab412a2dccf2d635f26fde3f7efbc0b607",
            "resolve_output_layout": "fd4b0bf4b456d80b0ae34fb7758020d60a6b8066b69e9c1a33eecf3cb8dd7b2a",
            "_validate_output_roles": "69cd6ed164c72556999cb6eb25879fb8872c85833e3457253815683d6fe68d2d",
            "assert_output_layout": "ef03e0e23694014524ff2b57c2f86bd38cff03cb3fe3e7f24007460e1d22d2b7",
            "_assert_output_metadata": "c455ca94a5f5c58a5d4441e8efb374e74702d1d236390e2899790eb17613e648",
            "PlannedCore.__post_init__": "4286c0114b8d1e6303ac63c7d4c91b3a05671b19d671bc62981e050301cefc3e",
            "assert_value_leaf_layout": "1abeb94a2353ff938207399038fc8e9cb978ab47f07957728b70d7eb70fd4eff",
            "_resolve_output_leaf": "dd0f95072a0f88a5bc5e7e8652f7837e9d77656ef4797eadfad74d318dbcbf6f",
            "_state_axes_leading_sharding": "20d39af17152908d29ee6ab50a3334ccb7636547cf4558ae8e9b0a6392c61441",
            "_state_axis_spec": "1772f28f0ed2c23f764c7494c6a6925c3b8fcee59180122a31c75bfcdeea79d4",
            "StateAxesLeading.__post_init__": "0bd0d18d5a0caff060ce3a860622a537a26aa061a164e9bdab4423b2ea4d2698",
        },
    ),
    VALUE_TRANSFER_SOURCE: (
        "d739b525209f802de8b5f1de63f4b2f15da300b4329df9ebc2c38175eefa137e",
        {
            "_assert_value_metadata": "44a9f8a3185c53e3fbc8a35745b009d25b2ef91e572cf950fbe0c9ba46cfd670",
            "MaterializedTransferObserver.__call__": "3e06bf2091d6a3291e5674060c34313e03f0e7419af22b033de98b2fb2845ca4",
            "apply_value_transfer": "09e4496fb06a55c5fd2eea3994f8832f7b3c096b91b258855a6f934759a1deae",
            "apply_value_transfer_plan": "f2abf0339ef4f70308e49a686b31200bfa6faed352873cc73cd27dbe8a13f19f",
            "_replace_transfer_leaf": "a3f754c5b7d39e87629bd4cc717658738d0d8ce529081c6a943c029c490a6448",
            "_transferred_leaf": "a4b3169ae883347725eabaae5cc4cac8e650b034a02841aa06c09fc92539ff94",
            "_replace_dataclass_field": "cd3f27158ba16a5902ce704907c3f97c92846255b91238e3ff7a5c51c481bf69",
            "ValueArtifactAddress.__post_init__": "cafe101592a7a7d2019ac1f2bc57e2e64295da6093a399dbb948bf8c38765e0e",
            "ValueConsumerAddress.__post_init__": "0a5d4ed3df5fb0be1089efbdd538b2e13473f6b0631aef1b699d17030039ede1",
            "ResolvedValueTransfer.__post_init__": "60d77078407d8b253b0654dd03dca371c32a06cf1efb05470ff453c815a7092b",
            "resolve_value_transfer": "f36134bd6c6cd164231b6968958975693a7222a2c6b7ad3c9f41c057832bb219",
            "classify_value_transfer": "c55b04af4530ac76c2dce1b47bb3ccf0d4056efa7c4dd68417257f59e03d506e",
            "ResolvedValueTransfer.cost": "07d257016bddbec9e6eacefb5e26e26c8c612fd6a0884ea81b40ac667a0c9011",
            "_named_axes": "46fa78227ccbe7e1dd881030b05d16794d3f45003c37b638ba0badc434d977ed",
            "_validate_edge_identity": "7f3d452cc22b9a4e39c49ff1187870c2fcefa80152b41d2fcc22a59f4f28c98b",
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
        "f3861348800ad791b6339675c7a22cfa2bbc639056606476a1186557253c29a8",
        {
            "_period_transfer_scratch_reservations": "e614b686048f4131f4b612386ca89ed8cec5f786fc64ac82f7ba87866e54fc76",
            "_continuous_value_replica_required": "4f22692255b3746899ee7b2849776ea8967090af80d68d46272cd0b9a5bc771b",
            "_compile_all_functions": "ed9819ba6df4bdcb67db2a020993608812dcba72e620523cbab3df2d9f15f99a",
            "_prepare_solve_programs": "e33a8807e20366842eb782fa8b06423d699bf8ecd3c33fc1c622bcd6f1512f0d",
            "solve": "e94f70fdad5510a6238150fda37c45956af50807b200257d84a819aaf0b67596",
            "_cores_with_transfer_cache": "fba35f0f74a7a496f2302ea160d4ce6b832d56abc6d0fee14bc07843b47a0fd0",
            "_release_closed_period_inputs": "9fef8a1766bde0e8904fa6588651beea415b6ddcc1d3fec126bf68e100039863",
            "_retire_donated_inputs": "c760bf519c3eab446a2cf7164c4158e5f318de35d9787a62ba24ee8ce5c90a12",
            "_prepare_abstract_program": "e5ce732766b8e62aeddd6e2047957c918876fb5d895bdc0190601e9566e6e34b",
            "_build_continuation_templates": "458ac8482471a3d565a61dba63830e30a1fe2271a8e40128fbb1e7da2b4cb46a",
            "_iter_edge_topologies": "ba1a385a7779fb841a865a76bbb185c30653f27a7f66c2d888e147dcddcd4725",
            "_build_base_state_action_spaces": "7a0648369e6ca73456ba099ed2cde0bffd294728dfaf482a7ea07b477eea792d",
            "_resolve_output_layouts_and_lowering_keys": "9a7ca112027ab83e686d6bdb7c36305f3268b676dad01d5fc4cdc0caa365cd0e",
            "_build_structural_blueprint": "1cc25b2c279aaf71fc3d449cac9d78d2f05b7285a06983bbb1aac04f4a4bfd56",
            "_bind_structural_blueprint": "71990a1c2417e81464e44045b10cad6d6e61b5da2db5775aedcfcbbfec3cbee0",
            "_structural_key": "c45c76077325375e6721a821bac2f0406779f1e9cfdd4422414757c783c631c7",
            "_evaluate_edge_fold": "6ea390648d0a06fb74959ac36694fad5eaab135c8c1e258d4eb48458148a60ee",
            "_lower_and_compile_wave": "23f0af3cacead6ad18ac584dcfef62f2f31b8d654bcaa3d9bfd4f1288e58384d",
            "_lower_resolved_candidate": "29768b2c44fc16f6ac66a95ca2d64d1780e84fcd096655105dce1ba78b63e693",
            "CompilationWave.lower": "68a15417ac990410be8075b1d5c549254a99bed19547c7aee4bb6f9ba37bcb91",
            "CompilationWave._submit": "4f3ae9b8f534ecedc4beee4f681a0c1d32cf8a3f3da4f87b6fd5b2ad34d80015",
            "CompilationWave.__exit__": "ae5af742eb203270826619ea9680584de04bf39f29de0e2d1329aa94e1c508f2",
            "CompilationWave._raise_first_compile_error": "787b7683440a2b26449c59bab1bd7986ff983e5210d3bff3d07be5135f5bc527",
            "_compile_and_log": "6c2e773eaf54f547353e2734fefa2429d266a05279d94e99c8f8e596b916f74d",
            "_run_period_kernel": "acb2452a5b028d0057807e7e21d424965a74ebc37a4bde19e59c569000462bd9",
            "_regime_retains_replay": "3363e2d2108ed0140b859e8851d3e6e8b0ff4d159d40644868626c17199e7da3",
            "_select_period_programs": "b370f6dda6dd4d8e0d91c8bf9bb6fc9d44345f67c191efd86e761befb69357fc",
            "_selected_artifact_keys_for_cell": "1acc464529bc9833e48f727279682d969f850d2a3bb206e8a2695b1769f6182f",
            "_CompilerMemoryLookup.__call__": "77803efda3bb7e83aa81966820c20b8c3ab51bf6b7a0b5f9d738cf587eea73ba",
            "_select_runtime_donation_cores": "cd681c45f6e9c16c6de136403f427acb33074b5f1013c1547c69fb4f2b030e47",
            "_donation_ownership_refusal": "b529d9757013d2f0fb6f44a08d9d56f78713670775a14b80df05e3115c55c239",
            "_mark_reused_transfers": "55e6e53f9d98a5e16b8b0548de876ed9e73a208d92156cacb837f594784d17fd",
            "_consumer_key": "84e636efe031d30a19d76638e93a95b8a0debc28b2e2814f36361b40d6f3cb13",
            "_resolve_program_for_execution": "c70022e481b5cfad0aa34d7b26a427396f33cfb6a08d9e63ca9227d2d4437d8f",
            "_resolve_value_input_transfer_plan": "1e77e1762def03b1ae2dba257dc16a43474bbdfa4b4bf73b315039c00d347934",
            "_resolve_value_transfer_layout": "0d912a492ca62d2881126ef4e8dc6f161b73c9ca9ab5430b4affd10b5199f4ea",
            "_lowering_key": "be1057a978ea37e2f180672a33b3ed72f6e851483834c224f428dc9633a30412",
            "_abstract_arguments_key": "b9c3fa7ccf8652de781f0e083dcfd4cad6edb1e8c10762d94e402d0e5a112469",
            "_abstract_value_key": "1d0f76837ec388633e2e290f2e577245107c2d4a968faae4de5d0e80fda890ee",
            "_abstract_leaf_key": "b2ff6f4e4276a9f591af0453cb86129f6ae9a473dd268478082567aee3ea4e8a",
            "_output_roles_key": "e439233cdd3158ee8b32c8c7edacc2f1c944fdb4146f45886aa27c6b5ffe8f6d",
            "_assert_lowered_output_roles": "59a66e241c3515c1cefeb48af9681ab3598ad5c86f7d59764b11341c6a1bc68e",
            "_attach_resolved_output_layout": "86b17ee66949cdc2a0751de70bf01eae7498333b18d9252d0c969e272498069c",
            "_publish_kernel_value": "d722838b82cde45902fe2090cfa2c11097856413ae848bb2df85ae9159b2a01a",
            "_resident_bytes_by_triple": "b9d5cf6a983082a3a66a3e2fb5a13c29104e2907a864fad8f5f4fd787e98fbf0",
            "_resident_inventory_by_triple": "05625ca0f3f24ff842e9f12913bfc1cb4481470339942d4eeef8e3fd251e4a9a",
            "_period_feasibility_counter_reservations": "4c8041a3e811d0581689eb4877a6a8d7355f4e87514c9a6b49f968e08911fdf1",
            "_candidate_resident_bytes": "ac8b5a9aad74d28aa50407882824f567314e9a582dc47b60766eef33686689df",
            "_compiler_reads_source": "3a1099c48958b2a5ec59affe0d9cf0dc179d773c8e96e9fcf821526572f23439",
            "_period_copy_reservations": "4bf41c2f458b194cef80ab182a8b9e3db392353ae606828c3699b07c0c63a831",
            "_internal_reservations_by_cell": "e3bba8d91a629f155b1c3e10733f83239e8f32bcf0f50d184ee504eb22441359",
            "_internal_leaf_bytes": "ece84ac1a1905891efcec6e59693074f62da35ca50e9185f0fae55c2829f1206",
            "_retained_base_space_arrays": "286541cb0ed42234d8f8e2344d4b7ff67e6334e03b58fa195ad88a4f3ab9b13b",
            "_CandidateResidencyLookup.__call__": "7090dee9635f84f107ac4758fd21811925a264e73738341224298b43859f082e",
            "_run_dispatch_unit": "11118e9698670a7840d459195a7cd0e3cae5f315d2d2c5baaad574d77a27a154",
            "_accumulate_landing_totals": "79d2778f19979ae4fd30b8701e50f03126de733b918539025a1f78a4e1a98ccf",
            "_match_leaf_pair_sharding": "81ce99279371a26877d29acaa2760f8c78502c91406fe11b7154f372b3c682b7",
            "_match_leaf_template_sharding": "a0277b79e39918c189e294812f915c330e6a4c61a6ae96d567acff48ad26679a",
            "_query_block_status": "4ac85b3fb90eb040d57669178385641b56292f253a1e36642a9396bbd7662130",
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
        "b32a0edd159cd8ff72af2b103807cdb8339f2c045df4ef355c53ed7593a288a0",
        {
            "NativeValueMaterializer.require_entry": "8c031f35097f8a08ab93f935734d7d0c85441950197be3b8f0382974017f6ba6",
            "NativeValueMaterializer.__call__": "7b89455bdca484a6e66a976a49bdca23dae9301397400711c28b8a711c74f9c2",
        },
    ),
    NATIVE_ARCHIVE_SOURCE: (
        "2b0fe2dacaf1491dc523bd4b5d2f1b213cb3439aaf0b3dce707674301d240274",
        {
            "_LazyHdf5Entry._materialize": "d2c6729aaa7a495a7bb5e658a7909f5c07dc2e18beface9639ed9a2de6a20459",
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
        "edcabb3cc084e19f6a0a066796ef93160c5db7d32f4438c5680851b7e9cf55dc",
        {
            "chunk_host_device": "78b1dab1f595c20eceb674b65fcd24f29280a050134327b9d297ecf541a2eb40",
            "offload_chunk": "e564c76f6d399b839b4ce7eb0eb2f4b46c30de67c3928ae8bd75e6a2b0bc83de",
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
        "9e59d5d2c98a2a3402c629f4a6b7a499fb17b616c64920596f739326e238dc4a",
        {
            "IndependentChunkReceipt.profile_count": "246417809388fdd5e03c2e38c3b81863e76bb7078bd5f2d04338da0ac7caebde",
            "SimulationStageProfile.__post_init__": "2a0a901ce96cdd863b8b93d237c6ad88b40bb9bddd6f0b90290374c1074c7f4b",
            "SimulationChunkProfile.__post_init__": "fd99aa93cd12c99270edaa7d419bbeeaec7fcb53139038dc2858f1a1d8b489b6",
            "ChunkProfiler.__call__": "0d8cfb4c8530d329393a4f74150d216e68bb03a061b10b6fe93e401a33c6dc51",
            "plan_simulation_chunks": "f5882ea0b3b7569f7eb7b823c9dd47e7dcee550acc7975a561aaf2c2210a5ed6",
            "_required_bytes": "342a4c7068895974b53365b8e71ed0f58b0b5f86f44ddd011ab224f576ec4740",
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
        "b55fc761493b0cc6f6fc69b90a45e657ad223a3b3f604938116049ffee697b50",
        {
            "period_value_flags": "1ca9e9e2920a0333915071a229cc907c2fdea761f313ebed1aa6e7d35235908c",
            "owned_value_nan_count": "e6168d82082f8c26ffd75a75386d46764a59d1ec3030b11e9688d019a9e668d7",
            "transition_counts": "9ab0013eeb414dcc3bde5bf91cf144079b27ea4b3538d4b7858687cfbe84dfbe",
            "profiled_transition_counts": "586661d6d039c19a88a0bc9c450342a8c31f36eb0453d0d82147ee6d15cf6def",
            "DiagnosticBinding.__post_init__": "a068c4933e67e24736709e489c7fa1bed9db434a47ea747525ec017c1b416fb1",
            "diagnostic_bindings": "1f5b45cbcf4dc87b68f6ef37d0c684335049e65c3af516d8679d16acb8eeb7ad",
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
        "b26a4193d4dba3866bf74126b427128f7acbf71680328d260951fda6311eb5d0",
        {
            "snapshot_value_store": "fd84054f9efd1fc5d1b1de1fd636d2eee0bcb3b73b12a9317c23dcbfd9c977f8",
            "_snapshot_value_coordinate": "bc16aa89beafddf4d4c5adeb3deef22c5241f0cac473bf3d3a0c5549fefd15fc",
            "_keep_payload": "be488773d5eeb9976214ed9d50ad30c26c54fe938916958d58fdc9602d8db03f",
        },
    ),
    COMBINED_VALIDATE_V_SOURCE: (
        "2c9a7294e78697f505d80b244b56ee1b5fce1e1dd44cabd87cf2fbcc705137f5",
        {
            "value_function_nan_error": "597cffa965bcef3bdb4ee5f09aa02688eea5ec7fe3e6148d4b2ce8b3082d9f65",
            "_entry_support_cause": "eaf8001c1762ca0d88cc374068a06ad22d43868b03ff8fa462c574dc788771b2",
        },
    ),
    COMBINED_LOGGING_SOURCE: (
        "d9eaf3a024725d295041180cc1592db8a2a62e64f4ea19869610ad150e6bedea",
        {
            "_owned_values": "9295ced0a7415a8738e6e9dd1b6a2ee6dc933bd3fdf37193e591752239c411ed",
            "non_finite_by_regime": "aa597487e43e16a2a001d5c05c0e00977c494349b9af3e65f8afd47aa6d63de4",
            "log_non_finite_values": "c2cc7e2acc85cae7fface00d8b38c23b7530b4e2453ff699182a4d40aca1b964",
            "log_regime_transition_counts": "11c5f45c415bfac54b36a55d52cded2911152f3e411e61e2d90a47ab19b11725",
            "validation_enabled": "d7a747744573883a7b690bb4092122d41e75be2f540eff8dcc256de6ab84fbbf",
            "validation_raises": "0554f2a49d983df79e67c167b5f16195459754c4a2d80357f894bbef28f8799a",
        },
    ),
    COMBINED_AUTHORITY_SOURCE: (
        "3407db3573ba51aa758a6f713b144381d92ead96eb569238be8665a13b0fc895",
        {
            "_ArrayCopier.__call__": "d708cfc14d20e8e157d30abe2336b3faa117b225f4ed3b10840d4dfbd4fa4c36",
            "_copy_artifact_array_leaf": "fe5b9e45c06ac59059a010fd680ac193e87e12aa3c5e38e980d84063d139bc07",
        },
    ),
    COMBINED_ENTRIES_SOURCE: (
        "9acebfbfa938048b8e2bd7401ede89bb7c0f165882a86d268d0ca8a1118e045a",
        {
            "_ValueMaterializer.__call__": "5a9d22497c8d3cf499823257fa231a4fa17dbfc70eee3b00f24b5ae1689f1215",
            "_copy_solution_value": "2c10ae621eb19cb4fab9fc9dd5d96160f4c4dd91919de2b2e46bf784a3e7689a",
            "_CanonicalValueEntry.materialize": "775b1abcc8dd36db4238efd34ad5532553450e23d1fab7e1cd4bf405948789ef",
            "_CanonicalValueEntry._fresh": "f24838e27eb2fa8147650f7fcf63c29ea04faa4f23d99a6e47352cd24b99632c",
            "_canonical_value_entry": "c3c8a9a342aa0b8297e43623355bfb7449c1e8451f517a3f2604efe3b14a5b38",
        },
    ),
    COMBINED_STORES_SOURCE: (
        "689f6378c196b769452c4aca5a92f4cac9a63fb56bbb6a23092a1335d569be88",
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
        "7cd45f6904f59958ac42339485ba9762d9ab6c8dea9ad7dad461632586ae79bb",
        {
            "_validate_abstract_inputs": "70f96b7582b3a085fdac809c48c6cbe5788f28d5b5d94dd0bb19eec5a3bdc973",
            "ValueRead.__post_init__": "9eb25ccdd2f056a241092952e6b92acef943f1babd2f2063af7b32da935a7190",
            "ReducedAxis.__post_init__": "030c5cd6aadd91db7c7362241817b7b1734cf27d853824376ba1206bdfae05ef",
            "ReducedAxis.extent": "aebdd54708094461d473977783f0588d2491c212629e1f5a40f8cf24c789802a",
            "TiledOutputAxis.__post_init__": "dc42959d05c19bee720f1423b60179a0d729b74546bfd278e0928cd2a6b2d3e2",
            "CoreExecutionRequirements.__post_init__": "34931cebee77f38a07e4b7fbbdadeb4571656f5c86d167e56a5199420735c067",
            "CoreBuildContext.__post_init__": "c001bdfea799659c6e0f1d0ee09180940d5176d8a59675dcabef190a12716d7b",
            "CoreProgram.__post_init__": "343fe026561f51d8d0fa7b60286b339c46546236ac0871bb119dd3064eef46cd",
            "MaterializedCoreProgram.__post_init__": "53205c8a2ef9ec2c5d62c759c66d151a5a2da91a1e880dc7ea46dcd700676b80",
            "ResolvedCoreProgram.__post_init__": "37bcf7ec896bdd4d78125aeb116b7f22835cf142b07075d3ae2586cfe1afee91",
            "core_program_graph": "ef88f028d82cd99b9d3768ad89df5f917e1b11142899b3ad6ac113f35a676226",
            "_reject_native_duplicate_authorities": "e07ed1c6ff7894b2b6f090c2850c45a342a75924271b2466d23ab4b007b1a13f",
            "_snapshot_and_validate_graph": "391b89b66355ecdba2d0c869b4a748f2a0ef065610255e4819e398cf53faa7ca",
            "_validate_replay_replacements": "2aed7e1993700410979ff136c2dc13104a1f210b7d1eabf3bfd7dae22b75eb69",
            "_validate_program_declaration": "6158dc42296c246bbb8ab23d270dcc46e8175e89ed190af503bdebe788f11c6b",
            "_validate_retention_declaration": "62f75acabee1eb14ea74360f8563970ba1f5639dd06a0573a49d80402e437ba0",
            "_validate_retained_artifact_payload_types": "8ceae90284dbdc25b64e0b66d79990e8f9104f9a6410d90d8eb94c426d24ecbd",
            "_validate_disposition_reason": "5df6651c2d9db4ebeb5542394c738c5238144ff19f48f0639b3bb6b43c63bc8d",
            "materialize_core_program": "091a23d1346e0acba9f06d9c54d3ac2bc57bd001b8ab0fe5d9228bb8dbbf3dfd",
            "resolve_core_program": "fc430e878ce5668159eaa6a26d7255ab7e90a61bf2bb570fe0cf89443416dfae",
            "resolve_core_program_candidates": "04b27f16722ddc627560a62ed9f436e8a4b0bb2646cc7edf8a7d7e99abe1a50d",
            "_resolve_core_program": "a80dbc848211a46e0cfcbde76d530805a417feea09c2e6724e171fcf96218606",
            "select_programs": "a8364953a60797905b376ed2a53cd94d8671448707983d595135a4d0fcd52eb4",
            "_validate_core_program": "b7558f7fe479363723c2b4d4925ab954e262883c4e07cbc956b72bbbe4291511",
            "_validate_materialized_declaration": "d4a7b72b1943b877e689ed05bc6e83e8194fa20169615d7a0a8bd970bce96534",
            "_resolve_input_transfer_plan": "f62a9fc86af542f15a99a0dd93978b681e1dd0c45d14772b711a6d80d3eceb2e",
            "_validate_value_reads": "cfa03931a7c89ebf7f94583e795f646dfc261653519e630070a5aa34e8954d9b",
            "_value_read_argument_leaf": "9902d410df54de4513cdbb01d2346ab7e9388212bf6434fa9e525e2038c6aa1f",
            "_validate_transfer_argument_metadata": "2da767cf572598810ce123e021513c3a2e0cae1813e07857a011b9262590c051",
            "_validate_reduced_axis": "d84a3086d5ad81400fe18832049b68f1b3ca81768f8ea3839e351b55d790c1a1",
            "_validate_tile_width": "e986f3d40ff2a7ebb398a28a1cc086740ac861788247716363079ea7ed438f93",
            "_validate_coordinate_argument": "eef5d339f81e5f24dfffa4215ff9eb27ee286915a6d5ec2aab0f2507f4cca546",
            "_validate_width_keyword": "d35cbcfe40e9888d2d9bdde001c488fd82f01b6a8eabbd7dbf6121bfe709674e",
            "_CheckedCoreVariant.__post_init__": "49ac00a088f14200eceeabd7f8f81b5f404b638a25d9aa1ee1bb28d4f6139514",
            "_validate_checked_variant": "31eb594494a78753522a888b94ab7c60ccce5473cec16c87a6744c4644c83cde",
            "bind_core_program_fixed_params": "092c9e956fd8fa8673e2d5d7917022cdd042d2b674575c95772a56397d0e9d2d",
        },
    ),
    SIMULATION_TRANSITIONS_SOURCE: (
        "5be6fad4eef28fb43ada3bc36a1a8f4be017e089dc6875331866137982c400a4",
        {
            "_draw_random_regime_ids_from_scalars": "548fcaf12654d05a584fabee3fbd81fe42ad256307d5ea4df826cb3ae3355a5a",
            "draw_key_from_dict": "9dfed8dafa357fa87cde342dc7f174028e4adf8454b5906a4e0d24bee164cdae",
            "calculate_next_states": "1e6aa76d5fad1e952ee75e6da524462db2059511d0eca9ec3100895e86466cb6",
            "transition_random_addresses": "7543b7d506f9d6229450524ba46a332100ea7508d7cb5611181dcb95819b2cd2",
            "calculate_next_regime_membership": "6ea64c3e5190942b9b514daf0ce474201b985d13d14038668279062632a2e807",
            "_update_regime_ids": "93e8f8c1aba6e42f14269aac5d911c722e5e132c207cae6fa3e07bf138a58163",
            "_draw_random_regime_ids": "87fa7035d1095f0dc1a17e51e517ef559808039ac8a107864995ce94f2bf7430",
            "_advance_states_for_subjects": "a7ea5a0c113ec1aca76552c7d54502133076bee2ebe96052299ef46bf9518547",
        },
    ),
    SIMULATION_RANDOM_SOURCE: (
        "63f8001bb36219b6768b93f6ea09eaf8500e79b10212545698daf9963dbe0ba8",
        {
            "canonicalize_simulation_seed": "fe5056a999fa8e071b29dbba6833b7aeef20cce18f38379b313720c6955342f4",
            "_generate_windowed_simulation_keys": "dad6a1e750b25555822110f10680b25b84ddd37b4a133aeef10cce02ebb0d43a",
            "_validated_chunk_window": "aa32a232c1fcdb895b2da64e612b73992db690a406150a244216559e3d96f3b3",
            "create_simulation_key": "0fcbb9bb357166e3812b53292c1490fdde56c237f15a502009d4f85d44e4ff47",
            "_create_simulation_key": "1ab7921d3e42ea7255fb8bbc60d1ed58e408e8818d0bac91ec132c276c62a05d",
            "split_simulation_key": "9968b4530b74638b082b63c0160391573b8ae4c20a9dc3b733a9b424de259921",
            "_split_simulation_key": "666d1e23f386815a8fd331cd543813fa2fcdb501f7672a82be9f48fe31c293b2",
            "generate_simulation_keys": "56bfd46c5aea8ca4cb6ceebdf478580667e3065823f3f4949cd9c2c0f0173cc8",
            "_generate_rowed_simulation_keys": "8a3f72dac10c12885e880b51cffed268f9d60341529b3bb2cff80948e23652a5",
            "draw_random_seed": "42fb402a3282c789664d45fb988b3ed07b8cec706924d3e6e11e26b44c393502",
        },
    ),
    SIMULATION_RANDOM_ADDRESS_SOURCE: (
        "7708edf35a968842011117681ce03123419eb56def125b1880bb485339334506",
        {
            "RandomAddress.__post_init__": "492900c559c9021831e4fb34df52dc8a3b913771efc601a9699de2cff8eb42c0",
            "RandomAddress.words": "14d65e58045eff086d86843f04fbcc95d5864bd9acd6b8bfb04579878ffc1467",
            "encode_subject_row": "446201dfa9a26b336900410b452a5124d7b35e462f13a683a910906ed8c961fd",
            "draw_window_keys": "1f141d2939a22547cd1c9bd4a0958e419436507664dc6178f582a0d1988a8d52",
            "draw_row_keys": "ee8f51828ebab5f2c057f324b6f0d0a550f68c4d9e6a610fecbb1b44f89b318a",
            "scalar_draw_axis": "19ad514b2eb20eb3b3062428e2fd9e0d274e9fdf757b69222efd3a756256c736",
            "inverse_cdf_index": "6455e3025a3b58dc54c8eee7c3f01640a40dfaf574f078e4bf16c021551b33bc",
            "draw_categorical_index": "c937f4cc9e2b81874a6d5b9b517d0d5f8019612c32a7a129e4f21846feed080e",
            "_threefry_key_dtype": "fbd2d0d9f527a817b48a8f94e1a9c59f8ffef434658e392ca1c2a171d94a9643",
            "_row_offset_words": "c6169dce1a0bd8e4fee1de76cafd2c7ee4939b7b7f0648fd589000628690e0a3",
            "_fold_subject_key": "ec2de14aee66046985b59e498d6a3f8a58b02558ec395c8edb69df1822ee1bca",
        },
    ),
    SIMULATION_RUNTIME_SOURCE: (
        "718d533934c4d8f243fcd0cc4cb8c88ec87e24fd067f8cd4298667e5b08f4a97",
        {
            "SimulationDispatchContext.__post_init__": "3ac9b845383980d1dd53680eee8dd927277a471733111d8387f7bb819f4ee2e3",
            "SimulationRuntime.prepare_abstract": "9b5e9af148ef5076d1c76825fc30cdb9d160bca79d74f621044c162b14b4af9b",
            "_materialize_abstract": "1540e5a2279c62ebfa90dfa809d1d6e20da228fbfd418db6f7143a1d08760be5",
            "_dispatch_widths": "0b3d0082d7d09bce7260cdcb8df4b13968ba85ab0f32eabe51e2c821381b87b5",
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
            "execute_simulation_program": "3b5ca2876c6e7cffa9048848e6761a6539717a8f81dcde4650f8aa66739f166c",
            "_SimulationCandidateCompiler.__call__": "d2292d22403be650b888d30b7649b0a089ed81115e1056180712cb3f650d5ad5",
            "_SimulationCandidateCompiler.lower": "ccfeddebd99fe733f918fa07e8a056a9e8ec4390b691b5c6173d0e95a6e06a1a",
            "_SimulationCandidateCompiler._bound": "4ffa018af94bd045ebcbb2ad9a1169092c3e6452d182aa2861ffcbfff6916069",
            "SimulationRuntime.lower_abstract": "f823fb439bae7e6077de35e53e0561e6e634e9b8fe2266fef9c401aa2dd1d7d8",
            "SimulationRuntime._publish": "840f5393ba14b7d8285808f30da1bf67ea8362ebf950f9728eb6faa0b10a1d01",
            "_with_compiler_memory": "70e07953cf8915500cc587dc425dc46f1cc81b1313517bbe675f527062b79bcf",
            "_with_subject_extent": "c48502d31e8f9fb29221122b0de6d10f38f3d6f740854a82512c2b964079e463",
            "_build_context": "1066056833ff638d8f54438d142ff429eecc1d4adf57c4827943e1d180e0c024",
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
        "959611aec9ab1771cd96939206b1cd89f01a94eb4cfcb2e7252b585dd1489750",
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
        "369f8ea3cd3798698a37af6b77bbe78d85b9d447362d754f12cf4f0ebedb6a02",
        {
            "ReplayPayload.from_policy": "e29e2d20b51b07a6412aff4392720c1818e0b8662ceac0eee986f31e786dbc98",
            "ReplayPayload.restore": "e1599af0b523cc29f9c85190c51f29c6276cbb995d9dab8f085543780e511170",
            "_flatten_payload": "76b9b73e16baac373476d7717962fd4970064ea18e249fa28ddb43546913e042",
            "_unflatten_payload": "3a6918f57cfbe806f6fdeecb68342fce7a8a3c8db200603402b87bb1d1afb222",
            "declare_finite_replay_programs": "266fa720a7f101a29ee69956b46487aa2ca9896dd2fb5f45a4f73f3f65aa4fc9",
            "_program": "28db8919d1fd3fedc4488a07cff0d779569cd723bad25cb1d0fc7c2373e47e32",
            "_policy_reads": "10f8afa6126d8f8fa340c021147fa8ea69ad8ca126830968a3e1c43ee8b1d396",
            "_Prepare.__call__": "3263959a9fb7fa81ba52eb9b0afe9231865c73049a8367a05b391f700a20dd9b",
            "_Rank.__signature__": "50aa82bb239a45eb831ec1de9cc04766c44b88f9e2a8898243273711d1ca5bd7",
            "_Rank.__call__": "d8a0d391afeab8c0d447395e7d4fcec9147389fcbe3a040501370dc68de1fca0",
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
        "19ca82a498bba8f05a7ecea4cb5ef9b9d8b34ffc7fdc4688a97ad69e4902acdb",
        {
            "validate_model_inputs": "5bdecc0ed3c750043cbf71b187300ed1c2c92016575b22eef992e970b1b54e2b",
            "_validate_borrowing_bound_routes": "e45c81a294f1af461e8c06201bb12ef68a302a6bc70b6bb7aa04dae67ce61c56",
            "_law_phase_varies": "9525a791fb3294f12a8f5e71906bbc2c1ea35cedfc4b885ebc27a7bdd47242f4",
            "_is_solve_proved_post_decision_lower_bound": "20eff9706986b1e34261ad846c3b8db55ef631514e5824563168b3bef97adc99",
            "build_regimes_and_template": "5b8674686f01725ffc7a932332df522123e9a7fad547813ec1268b7174761651",
            "_validate_callable_borrowing_solver": "b3bd802d5fa0734d179f33569f4af8503726570e80c7b3c965fa4e104aec6044",
        },
    ),
    ENGINE_SOURCE: (
        "0e08b01fcf618042c8b116f2236073cc44ac7e0b6b0ecaac31db8d6022a4aac2",
        {
            "SolutionPhase.resolve_process_grids": "433e871004a339428068e27d0e414e035eb8d1789fd73dc8b9cd85328362a045",
            "SolutionPhase.state_action_space": "c54e576769c9c3addaf2dc961cf604747d2fb3d6b29b5b3ec9a80c3b4f3a692e",
            "resolve_phase_process_grids": "ef605a0ec86dc094d2613c8e3e6e27c3dd8ee91d21618f37386a24e7acc3339e",
            "phase_state_action_space": "f27fbdd014df4497ca6531fa57d55fcfb77c995069815d63f47613833c70f126",
        },
    ),
    EAGER_STAGE_SOURCE: (
        "d4ab23671c44c85c1a367e02c547259599f7b8a946183bc6169ca1d86f80778f",
        {
            "_abstract_stage_operand": "25a8cdec68bfaa93e8b9260c5f4d5a63671069f06726ada770eb0274978446bd",
            "run_eager_stage": "5c76a3887978ea959365beafdb8d8e78dde07bdc364826765e0677df3c18b548",
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
        "f10909bcc6d8469234c4efef432a4da02f880b5ea8afcae38e4f5e1541adb32f",
        {
            "_emit_post_loop_diagnostics": "c43efbb3c9bb0f6e304c2aa6ac2a9513e2cede49c1eae983fd98049c9a87e87f",
            "_raise_first_nan_row": "957fe2169afa34e08bb34d577b665d9b86d924a85f177e69d3a9aa2e641f4f7e",
            "_raise_at": "c4b88def80c5e553cef7394a4f8d631bd13ac42d58faa4777c1bd2db07b3d120",
            "_reconstruct_next_regime_to_V_arr": "0a8edcc21ad6288ad9fa441453b8768d437ff4f5486687a239c4cf0ff9035314",
        },
    ),
    SUPPORT_FINGERPRINT_SOURCE: (
        "366eb2421a8d358c775780da953cf89e0b9cb469cdd562e6d0685cf86c7667a6",
        {
            "fingerprint_solution_support": "7b96d4f0c60f892df96d252b6dd5c8e3454db6da5509990e9f8d7d7e25e21174",
            "fingerprint_model": "04c8f2960e52d4144353b1e3b962edbc17b00eb05fc68a3207cf4826acbbda51",
            "_grid_support": "9709180334fc4732e2193f6b736667d6a96648f1346401a886bd7da0f41b65bf",
            "fingerprint_model_structure": "172561f90f3fb8181c86b0154bda45e3a5eb033d5e5c02a054d893544ff419a0",
            "_query_semantic_descriptor": "602544e2ed8bb315d1bd8a9425e486338810ba663c55197bebcfbe1e9eccac0d",
            "_project_user_regime_declaration": "a2b9cf5ca24d3082e5d5f333acb93fa6679ae026185b8f30f7bb1a1d8f83038b",
            "_semantic_fingerprint": "75f98beb48082f6fda273258837d03a38221af56812f8d621163ebfdfdda893b",
            "_SemanticHasher.visit": "eafce96fac57920b50fc278c6972607d1e829919f077de1acd7ab9ae47808f63",
        },
    ),
    SUPPORT_AUTHORITY_SOURCE: (
        "57c814bd160c51ce524d31b1c73b5d8c2bb59c541b637a985590312da239ab8f",
        {
            "supports_owned_negm_u_artifacts": "dcc3df548f590dac6b6a71ad8d41d079fa1e27c84e028ac27ff8031fb81f9d3b",
            "_is_installed_negm_u_continuation": "3e7e2b1b438e8ee67887ee733434bf702e1d1b4e94f8be1c008d086b2e7b2782",
            "build_solution_authority": "f16cfed0d51652c3d98b3993a797738f172e4471d5732f5a8927e96ef8965ca8",
            "_validate_model_query_spec": "63f481b0a92b03ba555d42819e74bd0392cfb53cb9edbb2fc1e89211bc6e32bf",
            "_egm_carry_artifact_layout": "f9cd96ea5b1097c6b84485477a0c9aed8164ec5bfc98f3f42ab2caff0ee8fb99",
            "_query_leaf_axis_names": "4802a4cb6701f3023e1fe6fe9d8a7253d06cfc8618a4923a4ec8870bbcfcd575",
            "_declared_carry_row_state_names": "835ba452926f8c55464f71282c057c64c690484767f67c9ca009db1e723b7f8e",
            "_declared_carry_trailing_state": "c60309b6bb614ecd89aeedaf6b3051df0d3c665d4a1e0ec4502f72a68079bc31",
            "_model_role_axes": "6e6dd672aa0cff5d3c2ee20541ea8c557362ca5ac2e6b52154222e827b84a01a",
            "_validate_query_policy_budget_authority": "8bc4e60840d1b564d845c212b318924a0b865d2e1403c037f8ee1d6745189495",
        },
    ),
    SUPPORT_PRECONDITIONS_SOURCE: (
        "2dde27fffaa89b3c7778a701e9906e7ba30a0094378cb2e08ada977d67f2003e",
        {
            "check_pareto_weights": "c59b54d3349a086083b2904e0f49d2396da35debce8b339ac1923eadf263be11",
            "_check_one_regimes_weights": "1eb8d8f9eefe3f49f32294f69c2b619404aa892fc477497704ce290facbb01ff",
        },
    ),
    SUPPORT_TRANSITION_CHECKS_SOURCE: (
        "fb9fa7fd03a6519542a1cc688a49bdaaa90b8c9f0b800c50681396f6fd85ce30",
        {
            "_ValidationSummary.state_action_space": "130e9abc335a298a37205deb98e1a3583e7632fd371ad0feae8413d91732cc4f",
            "validate_transitions": "e3cf9c7f61aa3bd979d51f625d31ea0bf1a554c2a18e06db0535bbf77100256b",
            "_validate_transition_sequence": "e6399ad3bbe56639ffdf5c7e7d27a9d2d41fb4026bc574ad25055c3bec811388",
            "validate_regime_transition_probs_all_periods": "bd4b49bcb10108dd06f79e01d59ee449cd7db25bfb9e15449561d23655392c02",
            "_validate_regime_transition_single": "689a067b3c00cd24ae46e69c12e1a2a0f8f6f530607b105de2403aee82bd13dd",
            "_evaluate_regime_probability_law": "c26148c3519d19ff749d06a65bb484f50fbf0a7ce0df36d09e1a1d98fbbf2e7f",
            "_regime_probability_law": "45faf8cf57fd2fb8a754589d2edcda7a1eb82910c7dd4f3c9fc8115caf6728a1",
            "_check_and_release_regime_probability": "f54a64b0239559ff0527404fe7dcaaf0783361ee24d5bf628d971f8037873b4b",
            "_validate_regime_transition_probs": "403ad994ea48ce5ee2982ffe904f3d57adbd1039d8fe9056e989019d0af19749",
            "validate_state_transitions_all_periods": "052d7def3951d720d31b8b46e20a8f9cb623d9f725f636bd6ffc1b869eaae03a",
            "validate_joint_transitions_all_periods": "27a6e3026d603052c37201fb8c406b4ca150c58ccbf0721db0ce698e1823dd90",
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
            "_validate_state_transition_single": "bd70fcd5f16e1fdaff02d044906eb873a8c732bb626e25e8144ced17510a1194",
            "_check_and_release_state_probability": "bbe5c53744d66283f4268a996341767fb18455c29017ca5bf6476e1f6ae1df6a",
            "_evaluate_state_probability_law": "91814f1439275d2c0af44fcff5cd2b6de9ce8f746a7b3945cf803120449e8e79",
            "_evaluate_admitted_transition_producer": "7656122b58624710c785c129472ba65d73bda0a334a8bdb9d7b9e7ee6ecdc6be",
            "_state_probability_law": "fddff0c03e9d046b20c4694816898d5d0e3fbf00007c8f618811b0f43f9cd0c8",
            "_abstract_transition_operand": "da4ed98dd978aa74b9e1e8886ab15e60f875198ad048d68ff23d143d370af33c",
            "_check_state_probs": "4e4e56e44907490ddbf993e53fd1f4e49611a505884128c66124d455143f23c7",
            "validate_off_grid_transitions": "efd8bfadc10f8acacb7eabbbcbadfcd7a23971a10750f5cabaa8df4afad6f11c",
            "_off_grid_target_counts": "ded21e0c8600d5e2d63879bb2c1dad732054b9470076bc083a8b4b96d30f94d7",
            "_off_grid_consumed_names": "2384e46328723bccb7e70a4c7321763c0fc83cbff27bd750d406afb080ce8aa0",
            "_repeat_off_grid_counts": "df360b8de56675f903c7888cfc32d30ee22d32c2cb461fc6c305f38fddedf496",
            "_merge_off_grid_counts": "1b42829f471bcc79cb076d1c992520feafbaf2013fa8a5c11d9d700d6b21c1fd",
            "_build_off_grid_cell": "2825dd7139d4eef370cce16496a88bfcc61610b0d944dd01661f355ba7d305a6",
            "_OffGridCell.__call__": "52f9aab90cd4d0728a0aead1db741209f0124a2e4a1e67f4b9c75f3b6734e767",
            "_off_grid_counts_law": "c5c02fba3ecfdf10a65282a2d43545c623da47872c97805ba9a0be632559e71d",
            "_NamedArgumentsCall.__call__": "9bf026a8bd55eb288277fd355e036b951a548eecdc4381f11526ab0ab06e0138",
            "_GridPointCall.__call__": "d98069d6ace823f41bdf5194a0dfe757b3325016eff363633dcfa3c35840709f",
            "_off_grid_block_candidates": "f2879912df63bc1d91a9997fd1abdf7a78f083bf6ffabb6f3a9dd964a4f9cb28",
            "_validate_off_grid_bound_status": "41744c1f7fe5902c34f130567e4817cf4102cd9c9f92ced7fe2c2de86df1f8f2",
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
        "cfff0672b5d027f91a08b25319bbc90d198d31d0694dd5bda6e9c2b5d8c8f15a",
        {
            "_shared_uniform_operations": "77e4fa81b76a13e05edecb8cbc14571226bb6e34a9a80842289ac345cef3ad6e",
            "SimulationProcessGrids.supports": "47c6617af89cdcd6ba3a99aae345a8c32f8aec740f838bd0b37817236306ab79",
            "SimulationProcessGrids.array_roots": "ddc99b73e50d34ce46e2c362900c62af7a6954d15f96dec516f0776be32f6101",
            "SimulationProcessGrids.__call__": "539b946733a550886073d1d4151262c786ca7c2adef0f72280de6db7feb85237",
            "SimulationProcessGrids.seal": "71dde0f63544df96a4c57b99a75f9dd844dd904be7c599ad30a3125262756e9d",
            "SimulationProcessGrids.close": "1fbde590b371243da9503b714cc55b822e333e3e7a64f100ed57051209076a5a",
            "SimulationProcessGrids._produce": "a72cb650f925971256c9ba339d57d2b3c260c6990565327cc4325b45cc9dc7a6",
            "SimulationProcessGrids.snapshot": "65d4d6df476276c3136bb02ba811eb5b8e1cfaddc248b6893375238f4800ca0b",
            "_uniform_parameters": "fc9a07f36c6a8882ac26609164df71175c8916a33484a464a4d5626e680b8851",
            "_parameter_bytes": "93e72d2f1aed3bfd4d0809cc8a5d47d0d1e7dc0f2664e926f53c1ece88cfaac5",
            "_abstract_grid_parameter": "8c1bd7f93b2aa50776102a6b2b2b733b80ddbd518f9d0b5eee47477bf3b8c8bf",
            "_compute_uniform_grid": "546e82063b47b05523ad23c64e681adaaea44dfda66071aa6b6f2ec94cde9d89",
            "SimulationProcessGrids._produce_normal": "5776e4f59b1eef6dbbcbacdb3d873321595222c3ab0d3939560c28d0eb1d6aed",
            "_normal_parameters": "893f4e4e80b34a1529672043f66cfbcef9989fd9730265858791d4d12321da1b",
            "_normal_fixed_identity": "467654dc343d02c419bd202358aad1e0ebad76ca7a754b8f82c708ee9ba4be1f",
            "_compute_normal_stage": "e49e8d6ed2760a94e39877a1717a6567220467d098de8f1670033411016d003f",
            "SimulationProcessGrids._produce_staged": "831f8107d8c1be270ce94a13d5f17e32b982f27758a2f6238bdecfc69950b0c2",
            "_complete_process_parameters": "b11dea89c5a1b5ea44b482574f74ae6cb50eae5072293fb5116ddc1cab88bead",
            "_process_fixed_identity": "e2e8cd965379829662477a34f9fdc0eea48e7b83b8ffd69e24a30b64791c4e98",
            "_staged_parameter_is_weak": "408c2ee847275e1c74b09648ec40d930051387d6c0d8d341fde63c359086f567",
            "_trace_process_jaxpr": "1601b7cd9c726c0e9bf4055e16122c551491ab87dd7933d55bfedfa8318dfc23",
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
        "1d03e7464cdb08148a3c117c32e3ea007d0b1afb91aeaf8e3ad650baaf7290f7",
        {
            "tiled_productmap": "fe04648c042f92bb2d261e5433022149ef6e21448f061676da08199256008a60",
            "_CountBroadcastExtentInWidth.__call__": "93f12196ae4af77225a6b8baea7f1e586d85c1ccabc642a21b8f3bf90efdd2cc",
            "_TiledProductMap.__call__": "f66967b618ac4fbd44efdc5872dd3ee3428796a41fa5236524ed1143322f282f",
            "_map_grouped_product": "cb92235746a936e0921a38dc8891ddfdcbda9b42246e1b8738025f708e0b6ef5",
            "_MapOverFinalCoordinate.__call__": "9c6b0311f6be53e322282a20e53787527e6cd8216a135831d27a32a25d2f5873",
            "_map_whole_product": "8c79554c80187a7a49ab0ce5928cd785f7ec73ee327ac7a4c2a1e5ebae7c213d",
            "_final_mapper": "1a807594263ba85bb7eca059502b225de599547a66faa5bfbf2296d2254cea55",
            "_MapWholeCoordinate.__call__": "19a3d8f40f917bcef5c326f034417fb774ca989d235054e8bfd7aa85aafa90c3",
            "_EvaluateTiledCell.__call__": "f0870bc3a6307da49db6563e1e8bdbdd753d5ba5b9fed9fd6b9b4e99fe75a1ca",
            "_restore_product_axes": "aefd91d6e6d5c1d49f1451435c80b04c616fc010784907adcf8e377508f087c2",
            "map_over_leading_axis": "99ff884f97c0d3f217db5616595c4a429bc641fca02a64487e11a1f845c3c5c8",
            "_RestoreProductAxisOrder.__call__": "0f8d87f5527a4e1f9dff72e29dc5dec70203b76275993f06971f5c476adf5434",
            "_transpose_product_axes": "54961f0159833bf5de6053053200dda0e3745caae17447c2aafd2e6f4faadae3",
            "_map_with_bound_summary": "41302938c06840ceec15e184d39e1c4cee159926f2b193cfd6502fd2a7ee9e41",
            "_BoundVmap.__call__": "6db6ab0d219f7b4b3732533d1f0046fba757a393b96f2b0b2b12803096c69e01",
            "vmap_1d": "2704fc9f35d1e4a95de4372bc36823cf83e9f05e3757bba8f97a4b7f4bc5275d",
            "simulation_spacemap": "bf1f6b6975837494a5096cbf851dadb8d1bd3238224a3578c2c324b04ed86560",
            "productmap": "73cac5b25853f490ebf3e016fbdaa07c471aa2746b55744e8496be9b8e0a18a4",
            "_restore_mapped_product": "c3ffd89df4e5486a21a7bbb81267eafe821bc4106f7ff0bc70a9d37064110597",
            "_ProductMapBatched.__call__": "838f1fc3f4f80b37eb6d02806b4082cf26a5f8fe1c0ef9a22cd1190d6c813cc4",
            "_MappedOverOneMoreAxis.__call__": "0507aa77048322386c81797b360a4c8527d4802cc68c34ecf5fc47571c9b9ceb",
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
        "7b5821e70c6b5b31b1bb51f764506c37a107fd5b64f17db65c7aafe99c7ed545",
        {
            "GridSearch.build_period_kernels": "39157034fc62043a68a064a93a38e31bcfc74918fb7f8a3201702c205309a970",
            "grid_search_query_inputs": "d8a6457be13e6374c1e659e6f6f3fbb35b1440afc779d00cbbc10c10136962fa",
            "_action_partition_mesh": "cab2381dbe9650e10620db20b6614ccb015eb147ff3356ab66e90cec58bf06fe",
            "_classify_action_streaming": "8165f079219429b82f955b4708874d438b3c9ce5e6415b277542c3857082c18b",
            "_select_action_width_keyword": "b45663df866d5a48c05b8955b6cdc68515697e8fa925566ae72afd06b3850104",
            "_select_cell_width_keyword": "f686d6cc7ae0d93dd1e3c301600872996943c7e3d6788c9d5098d39449793727",
            "_select_width_keyword": "00cd19cec6e137d7d9e044bc1625793b1d6f78bbdfc93d6858bb6f8e9d3c022f",
            "_edge_reference_regimes_for_targets": "9bd8c8de92411abd731020f62cb3c44ef814d4f3b64ab496b6299fcb925473e0",
            "_supports_action_streaming": "d93f977fad68ad528beb9d4b9e6d45e5eb95b53c9a0398ff6f6a62ec548bad11",
            "_value_reads": "9712ae402debbd0c37a12999b224e365c0ca1e1a8bcbb6e7bbb4043cbcfcacfc",
            "_value_read": "082a372c7e48bf7a32d390079e2dd60b9ed5aa56868cbf9fbccf0b8f924b3bb6",
            "_GridSearchArgumentBuilder.__call__": "b67346f6634f747bc76c90f257c0e994bffed85c232a024e5b00ecb19a9f9707",
            "_GridSearchArgumentBuilder._with_edge_substitution": "4968180e552973fcee84201ec802e668f9dc95ae023b405d123478d0040c089d",
            "_GridSearchArgumentBuilder._edge_reference_args": "bc98382ca2c7967c08e85d34106ce57ef4d395519e6058c5330e69a6af706580",
            "_GridSearchArgumentBuilder._same_period_params": "52279dcdea37c1ea9ad066794696d3f77a368539faaf12e3417fcfb439be7303",
            "_GridSearchPeriodKernel.__post_init__": "1e8c230f0845d9b8335c667f38593221a957c34955efe44921763bce6cd24ac7",
            "_GridSearchPeriodKernel.core_programs": "0d96f7bea814e419ef1dbdebbc3257d63c3c0d36d9e6e52e619f8f44ae5a8a56",
            "_GridSearchPeriodKernel.with_fixed_params": "5f5a26e02b136c760bd242cd91c22b76153adc9828a13d20de7a6a98f5877e14",
            "_GridSearchPeriodKernel.__call__": "ebe1ea0bc427736ff6b6c7921c3c4b48fed9eff51482bbe5ff52e3f73af22d5b",
            "_FlattenBoundedGridSearch.__call__": "114b0acaace961b15c442d9bfc59dcd41409c73962f3034f3e29546b79cf6ac0",
            "_grid_search_fold_weights": "9a9e0af52881e46a569335fd73c809b56c08f752ad38f01731769ca6e0c97f78",
            "_fail_if_decoupled_route_is_unsupported": "0cfd97f1c692b3f82b78d5a9403da93c902b88aa8805e26137930ad2b7dd301e",
        },
    ),
    MAX_Q_SOURCE: (
        "85c3e312c9073cc5b0f954e38fd69f11d3af35b3619bef4ac0278f15c5870696",
        {
            "get_action_partitioned_max_Q_over_a": "7e4b9b51294748528dbe9d26c80af48f4dc7ef8a8f75090e0d110d56eab6db80",
            "_ActionPartitionedMaxQOverA.__call__": "9252a1202b3781ff23df05c28e65d4585e16274e0ac6abd6bcc0c4a7dfdfdbdc",
            "_arguments_named": "95c0088a6f36474a87a6dd5d4f0a72745787638106bcbb911a0f5aad35101365",
            "_OnActionPartitionAxis.__call__": "ebae3ad5db6e89feb499866aa7ff673a8e6cd934e55346a77f051bbbd20ec64b",
            "_call_with_operands": "c26f9db85ab67d1b4d6c730076cba1d0c6b64e2d7dd98159d9c4d37b21c24c0c",
            "_get_extra_param_names": "cd97225f061bc5ef0bba82b75864b9cc7622174d10fe741c510ebba7d5efde0d",
            "_fail_if_action_width_keyword_collides": "d72946426a1e4a1d300fa4812f07eea8139236693ef8f79d0e8384b9d59a9f0d",
            "get_streaming_max_Q_over_a": "44e9f9671f8e8fad2f7a9a3c0b91c0d55543c864adbcf016835bc48e2f317c63",
            "_fail_if_full_V_streaming_route_is_unsupported": "cd4c96d572ec7df9dc269f5fa2bfc1ec5c16fe0a78de3be56adc28c15f065d2c",
            "_fail_if_streaming_co_map_layout_is_invalid": "59c06aedafc8bcbe31d7f2f7f7b7d94e1d8044bf529c6f11a05882c5bf1d7979",
            "_wrap_with_fold_reduction": "6070533814c5dcbe42cf8a4536de9990ea79a3ac71096afbcd4d61886041410b",
            "_StreamedMaxQOverA.__call__": "63ccbba0326478a8c341290f65c480d45d53b592cad8e769bf97821ea131f6ad",
            "get_max_Q_over_a": "c14b014393c8a99b8a7ba6be7fdd06135e49b65a44f2ef61e7b03b007237e415",
            "get_argmax_and_max_Q_over_a": "e963eef0544eccdf4f3add8197233301d068bca76660a711062e6c2df0308ff3",
            "_HardMaxQOverA.__call__": "461bfdb336128ac3f51ceda31703085bd87d06db3d0bd17af42956d96e25c873",
            "_HardMaxArgmaxQOverA.__call__": "8bafbe4d1f5d99962015e674980474e38b4a86c44c91d81bcd3f68b620e6ac66",
            "_SmoothedMaxQOverA.__call__": "b0f96f4b687109b1ed00ce6a7bfdb20f2b726fe3973d369358e75c634cb83a16",
            "_TasteShockArgmaxQOverA.__call__": "76e67e0e11ea6ff97a12fac34a1d7dcd3b88ad65a4ecc70c45eff17e225da395",
            "_with_bound_summary": "b013f2b5bf18ce8760c35c94ebd4058c4340d9ab306e7b4a834fe30d4a0063e7",
            "_FoldedMaxQOverA.__call__": "2687aa8d24a5181a1e1cacbb379c9a01e15fcb0bde9c55d52e111c31df0370a9",
        },
    ),
    ACTION_STREAMING_SOURCE: (
        "a9b4279ceb932a81798413192ad05f1b535ea6b6f17fce725046180676b62937",
        {
            "build_partitioned_streaming_max_Q_over_a": "2b1ab2a3075916b37d825af02061dd985063823e47aa8f57589a47c9dba86548",
            "merge_partition_accumulators": "490ff581c56b3c85bfbfc7d3824baed55adc1aa035970e5d15c6b433c6380098",
            "_fail_if_not_positive_int": "23eecceb2c4c24b75e70863f2a5f531c5a52e569400c090c382d5ba667cf6f9f",
            "ActionPartitionLayout.__post_init__": "9957631cd979a1ec48136310cb5e3cfb246b1c7367331544c68447b0b092539b",
            "ActionPartitionLayout.n_blocks": "269793628a54b8cd15b276fa360f884a97483ffbeb27bff2b68d4ba94ae055fe",
            "ActionPartitionLayout.blocks_per_partition": "b1109e10050ec6f186486a275a0467089c2c5101528b675c928ba0fd85d8b9b6",
            "ActionPartitionLayout.block_range": "c306308ded5a8642519248ce4bd7f400c8ea13b25b06a3fe2c01c24c90f03845",
            "ActionPartitionLayout.action_interval": "dbaa877fae3466c4ea5af7c2657fe18be813c4487ed3a26e6cbb06b1fac17aba",
            "_PartitionedStreamingHardMax.__call__": "e994a31485519e2bd6ac6449d6bfd9a3bff214d556db0ff1c0c746ae2a90e410",
            "_PartitionedStreamingHardMax.local": "974f0f5ae001649c22874bfccc8977564708c6d10e16c3c5eef1624631d207ad",
            "_evaluate_owned_block": "1140bd553314c405280402990a4e59d4acfa7d6b6b4d22d0b2a4b43a46ad4214",
            "_validate_streaming_configuration": "4f9f662fa73508293542ddd9e6ea428ac22f31f45376b80290f0e46b64a3f9b4",
            "_prepare_action_call": "27a8f85f8f16c7214846609213fdf9135bfb7118395aecdcf68ace17bea4e39c",
            "_evaluate_block": "6d0cbdb8fedaf491b8cf6fb942b9e680b6be28dca7db24e550d91bdde47ac401",
            "_evaluate_one_action": "c12a6ab8b6f9a29c2b985f2eda18f9ff2bd86ef0c3d2dbd17d6f739b0219bc07",
            "_decode_action": "cd4d0f5f01fe2a5648faab36deda62d09452116e3e7cf1e691fe6214b8a692c5",
            "_validate_block_Q_and_F": "7f00abbccfe23768df403596eb22c715eb3542b4e43dd67593f7bd3e487fdeef",
            "_trace_block": "f1d9a78339cd253ccd41e4e39611cdbb764c3855da2e1757c0101599d3d4b8bd",
            "_empty_reduction": "4db3f3eddf997264f026501228275cd64447b58489a72b86d674ddf1ae047189",
            "_typed_like": "fc7809b3ba41f45007a2464926696f92b1cf2f937c2933223c49ce4984d56b22",
            "_start_reduction": "034b3966dd04c0e0d66e085e8e2c4e16b127e1d8a9869ecfa039c3b5e7928b04",
            "_scan_one_block": "1901bdf24caccc5087081f15fc69e9138db76545ae7bb1794539d05adf5af7c9",
            "_add_block": "a5047bea80275b77727b69b06d563bcdfea7e80c0f99dda34f6570948ccd1a72",
            "build_streaming_max_Q_over_a": "a7ec46f647dedcfda11093c0ea2f51a1f80e7bae468fa0cabfbfffe667c1fcda",
            "build_streaming_collective_max_Q_over_a": "04bac0b32b19029d774e2b5a14593fecd7dd2f8ff8ddca60b8be08e58e45d5c6",
            "build_streaming_ev1_max_Q_over_a": "8498332dec9716b52427dabf6c22e0d624e33286b13327e22c5131f3e4c33fa8",
            "GridSearchEV1ActionReduction.semantic_key": "d1760804377fac087e3151a6dcff3c5293e984eb79b4f3f76cfc46caafdda923",
            "_StreamingHardMax.__call__": "ff4e10c15ce5e23f4d7a144035e2dc946baf0359a6fc2d1bde5c59c53a628134",
            "_StreamingCollectiveHardMax.__call__": "b0bb4bd13402c7624cf0cbd519b56c526c6d2de8cf7279c5f4434c2f757a67f7",
            "_StreamingEV1ExpectedMax.__call__": "c4a4d4240ebd92be8d0ea2b20d1825f1e8c787219a4895acc5bc3866704f729c",
            "_evaluate_ev1_branch_block": "a683329a9f55c515c5df50283b7b7af3f2733bf3ce474bc9169f1cb3e42c6bb4",
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
            "_scan_ev1_blocks": "b8ce9696b6664219799e0e48348c0c410aa9141a462579ad31ed755e4782ee8b",
            "_flush_ev1_branch_group": "cb87cd4217df5fb3e45806593d2013a2ba452f17cbb08097a9d0cacd1e6f9270",
            "_bounded_hard_max": "b34ac8a1ae8b38cc235e7e22983c9a46147ccc9ce66cb44c5d93081b7cb728ae",
            "_scan_one_ev1_block": "ba9b490e0b5b3f11aaff98f058b13947d42d5ff0c8f5a7fab842dbf6fc391857",
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
        "9806b3c7f5594db1bbfbee19e8cf24eb6db28a82e63881d864a617a7b0169703",
        {
            "_split_bound_arguments": "d31630c0bc8f65a362f33f5cf43b4a668cc63fbe6e9fdda820ec0e225308c62e",
            "allow_args": "aafbd21439f91d8e29c52097e4b0711469942a9ca6bafce071278ef69b3d78fe",
            "publish_signature": "5fc835acaa1bcc26f41d666c1d3a28fe82670f45916f59152aad1e77bbbfbc5b",
        },
    ),
    INTERNAL_OUTPUTS_SOURCE: (
        "17c81a2b6248f4c2f70e55954964af1628b65ef085e2e315836dd6ff11da269c",
        {
            "resolve_producer": "ca6ac0df113cc1a3eb2d3eae188ac842b33bf49434d6fd76a9d5a631d6348f7c",
            "assert_width_invariant_internal_outputs": "cf54d009a3abce9124ea579209b720ab06f7fe9bff641753f60753cc2ab1e975",
            "consumed_producer_names": "6c7019c97744a8bd73f34344a6dc45f195972e393dfeff1d1a206f0dce286ee2",
            "internal_input_templates": "6919cb41a458780ef41bc5bf08f5cad5f8097137461b73936dced1f0487699ce",
            "_select_path": "1d4c7b561538395a9e2499acec2627f5ae2b9281da9f6a555834293c475aaf5e",
            "_leaf_signature": "f2a4821d539a8c1f7a183e3f67d0d6a3bfc2b67244aa323a51d97c018d0b617f",
            "_published_signature": "0dd263a6830e3097f7414b5c8a674f42e339f094a2459d2a6999f6748d0619aa",
            "_tree_signature": "aea6cab8f3b6aeb31194c7fad76d2d1f061d338b1286662a07486009a5ca3c2f",
            "assert_internal_inputs": "87915fb8a1dfbd3a8663b3cde8df68dda0b51befcf67780f86b9d71c039cfd00",
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
        "67f7f90e6db59734f4e2c0b2f8603f3f5b022195b68c4af03138f01d8b1981e9",
        {
            "_build_per_subject_decisions_per_period": "b04fb50baadb61f7a5301b2ec1af99a6c53e10355574f1bddf3da94f606ac67a",
            "process_regimes": "288b4375dff6c1f924dc3e646f38e3c4471887ce440c6b99f9528bf26129c67f",
            "_TerminalCarryPeriodKernel.core_programs": "842c31af0bea766bfd410783881770152246a825505be327096c117f60ee65fa",
            "_TerminalCarryPeriodKernel.with_fixed_params": "4ee13dc7cbc4ebaa68102cc6eea4272791590fa1005080dbbbcf898e5f92a8f6",
            "_TerminalCarryPeriodKernel.__call__": "cdb11e9a08f14bc6d08289d277ca4d33d897553e16fa60eb7ff6ef36260a75a1",
            "_CanonicalRegimeBuilder.__call__": "c12af65038d06b64037f8eac3297a57775271f0c0981a83ba4649f88a41688a4",
            "_prepare_scalar_egm_query_models": "d8145cd202d993e8dc1e71a7f29042c46300fb558ff779b9b0c0b43e7274cf75",
            "_fail_if_a_declaration_moved_more_than_the_kernels": "77f7d0d6adebd110f143c0156321536d67434715df60d1c7375fbfc3d0ac269e",
            "_fail_if_a_continuation_demand_is_unmet": "7a16aafa14d7972d59e26cc793469342d6f8c7c046a48cb8e1f4d801bf5edc58",
            "_build_solution_phase": "efe9c64c7681afbee5cfca344005ed98f63758a1ecb27a8f6f56c636af13203f",
            "_process_regime_core": "77f41e5bdc1dc753f60717e47bdbd8a8eec70a66cf33f1aafa7f4b3b7c4aa03d",
            "_prepare_dcegm_query_models": "fb161609faa89d97ca8c85f2280b6999dd5604bb90db7c5e8dee7cd9e549ff77",
            "_canonical_constraint_routing_inputs": "b014cb91c503eb021cd7aee8b6ead91fa54a86b0ffc1b8fde29e80978b53327e",
            "_prepare_nbegm_query_models": "84d2dc8c636279381d611da2fab964b668dafe38ac6acf79bdeb27c99cffea54",
            "_nbegm_evaluated_query_predicates": "4a734bc9d0020b846a7ed8e56c318331ec1e414191354ee55194c39d0f5e6b98",
            "_nbegm_query_evaluator_targets": "4eca3820123c39974c20fbb85dffb68fd28286773f67a225af5a1e84aec36c33",
            "_nbegm_query_function_pools": "3d5549c36db6f139068d67a76c0234a3055f84989391576fe5615658ba39ce1b",
            "_nbegm_query_row_state_names": "f398c3a4c47b01a2fe7811f0c76f4ed01618129e38e7e86d703743fa24dea2ca",
            "_solver_with_query_authority": "0eb2703b05650da0b98779c17f9401b9d948debeed46d27068215ba1fe1a6385",
            "_build_simulation_phase": "bac3decf0820506992fc18cf4c6697c3dbe6bacd2e7dfa44f1712692f447f888",
        },
    ),
    SIMULATION_PROGRAMS_SOURCE: (
        "3596707300ed48dc65c6049119a3d038feff386bcd4115c97909feee26f339f9",
        {
            "build_simulation_programs": "783ca11c3cd92f66030c82cb8f83f703f0741e8b6d612df4091f5c53cf7cc203",
            "_decision_programs": "55e9564c10ff69ef2048e460e1560eb8e5d7b530c8ff7689376e32c59b890d18",
            "attach_gated_simulation_programs": "e92be1be5b6628706b2a2c7cdfb749b3936ade622d352161fb72d9d949a59cfa",
            "gated_simulation_programs_ready": "fe5419d6b84ba1929acf26cde8dc2feedf9589a2345925a1ae6de25fde674def",
            "budgeted_simulation_programs_ready": "f20390a48d72f2454deb9cb050493d3170092196cd6f57bf96380a09cb68c89c",
            "_GateFoldBody.__call__": "b24892f48c809e6aa25853831c72319e979078ae0dc0aa08a389755b2e2412e1",
            "_GateRouteBody.__call__": "bed23e58107f4a0b676644c86e50fc7fd545f1cb3ed8387126e735189740f73b",
            "_fail_if_the_streamed_reduction_is_wrong": "4393d8f3dc1dded01122dc6bf97e4f51c88e52f3804f79b08d600bc75496dd0a",
            "_decision_subject_arg_names": "bcf384bb7218cb328b3494a3eda67a40b1f23bd438663562315a47c47272f803",
            "_decision_value_reads": "9e61636fd11bb677abb92651de611ae99d7e561f4b61cc8e924ecc78db0edb8a",
            "_decision_body": "8ec19eb2ca794e8c88a1e492f08e9da80c3f4d973b1d340ce6fd86c6611a702d",
            "_StreamedArgmaxQOverA.__call__": "e2861ff5ccf4eb82ce58b4b4c043280952eb2d0ab29e657946bbe92b2ee456a8",
            "_StreamedArgmaxQOverA._fold": "c4cdc93b6c3412703e9b5edbf4a213bc9beaed8f5bdc9132600fe7de398a143c",
            "_SubjectTiled.__call__": "1f11a7951bae020b368431c354dc7294cc4d7778c567ac646eb95bd59fbc5e75",
            "_evaluate_subject_tile": "da5ea37bc7faea1405b83b856ca8701e178e0daf0a26223eb70146589d6d26de",
            "_ArgumentsBoundAtDispatch.__call__": "46ee403fdfc0849f77b0727c94064168ad21c00f5072d34d15247e939594babd",
            "_bound_output_roles": "cc97624ea9e47b53f685fcf6dc8c1e28e80313a6be1e3bc3c8bef51de964d091",
            "_flatten_subject_integrity": "af4f88646aa7bded3320e2df7163fe956644db738c12eae3007bc8207ba1369c",
            "with_kept_action_parts": "8629cf154b78299c070d03e96d74c5d8381988ec00f17ee68f950358686d2847",
            "_SelectedOutputs.__call__": "d8758fb93bbaf546eef01985ec1c6a9f753f822530e1da3bf18ab10f015d5950",
            "_decision_with_action_parts": "d41eb2e4f434e215d2e8f60580a365d83a2d85c23293f192c7fe43aadc59563f",
            "_DecisionWithActionParts.__call__": "2207b47224a48a0fc7e33e86c64f00ae48e1908ed29c37dd89d0b0da6ab8151c",
            "_BoundActionPartsBody.__call__": "1f249a93896b076c62dad990526688fafa23f6cb9ac51516bada50094c8d21e0",
            "_evaluated_action_parts": "eaa6a1fa9d1a3da4fd0d8a8e8f52446b9cee5186dec957f40309f1717f1e7fc7",
            "_lookup_body_and_roles": "22ddb336908ea49df7d212355dc896ee2c69f5b02fd51b6492c515f2be8adedf",
        },
    ),
    SIMULATION_PROGRAM_TYPES_SOURCE: (
        "3c983110936547c75f34c8853e3c74448462d7f9c6f4df2ea54e1f78e21d173b",
        {
            "SimulationProgramExecutor.dispatch": "cc42079ac6d204e3b4e22b058c24d7db4ac41c8269fd276a5ad224cb5bd2a7c7",
            "SimulationPrograms.declared_axis_names": "9843097cb1a6abbd4fa0121709a2356318fb4e99abe12fd549aaf7982577ee0f",
            "SimulationPrograms.forward_decision": "369b61edb30600fcabc910d0d6a333a939519fcda229f315d91101b96cb4de3f",
            "transition_output_roles": "ed772c2beff03f47113b71f5d3f0405469b6ef4bf67d4c3ad4df1a3f064e3fab",
            "route_output_roles": "81ccd2cdf1a29d1dcb3021d775c4abc8cb70364819a027303968afc69af19a2e",
            "subject_axis": "2d1dda5c95debf5b8c7d0a72c8fa71db6c42b2fb22763cc943ec494dfd3d942f",
        },
    ),
    NBEGM_SOURCE: (
        "15202b56e39bae9b4aaa7604dbddfbcbfc404a38b3931bbaf8518db8d155ad49",
        {
            "NBEGM.declare_continuation_reads": "b3aea84195a77cc3f4de92e284acd79cbb0e452f0f204dcd91b2aa930b4e632c",
            "_with_ride_marginal_reads": "8fe7ef4e91312fedb70de6d7bc72cb38fd80b50798b6d6e4a4ce712a7507cf56",
            "_NBEGMDiscreteCore.__call__": "d48acca0f3ca7acbc0c5e6b431e48075371006ebce3052f3b187a40ae4c0592d",
            "_NBEGMScheduleDiscreteCore.__call__": "b571219f4079410c84553a90a9b5038ebd7b1d34703edb8072c0d30980d5be68",
            "_collect_nbegm_discrete_spec": "00d3a9cb441c051e4f63317adab80d8e25c4681ee548a1134b11c62194152438",
            "_collect_nbegm_schedule_discrete_spec": "bb98fc777ebb9f11536ae88e5af2226bb0ba7912355d7439ce8656305c467b75",
            "_NBEGMOwnedSingleAxisCore.__call__": "2ace43ef2be36ddb3c0b1e2141b7078cc504d24e8178a65fafd92f30b99a5ca1",
            "_nbegm_bounded_output_roles": "19601c0b0c3e2da075cf93c7efbe8487e913fafd2a2fa44a1598e84fa294581d",
            "_ride_along_output_roles": "ca59e4477820cbadd82f07ff958bb5d0b5c2f3575ab88f54518ce50b5191ec50",
            "_with_nbegm_scalar_carry_reads": "91f139378fed563847b502c36c132335d6f32893f2a60a5e368b34d81e475459",
            "_with_nbegm_scalar_output_roles": "03378414f32c15bf7cffaf7225bef7e6c041642cfbb90a0e2cd1082259f80865",
            "_NBEGMActionEligibility.__call__": "3d19a478e3a429c3f7fd7f67747a7b250234b57681bd9ca01d84ae48cf496373",
            "_NBEGMScalarChildReads.__call__": "c8a9608a6feed12dcdda445db62f38600937a0c5bb15e37c2813ce0e7242cd28",
            "_NBEGMScalarChildReads.read_rows": "da3d0ac2ee97d230f5496f50806da4ddb30fcba6e869b2ea66a8ce872e3b370b",
            "_NBEGMTabulatedEulerLaw.__call__": "4fd15ffff0ae3fc44e0faddad861938cfe6b64341c595f04ccb9b92b9e51ef76",
            "_prepare_nbegm_scalar_child_reads": "c5c220b4466a0b3cc771b64e822c5df4d7f9f223c7f84c04a29619aaff0e2d25",
            "_scalar_conditional_nbegm_carry": "a42ee06742ab3a35e1baabce3245976d5b3fccc73aaca2a5510c9e4da46e000b",
            "_with_scalar_nbegm_sampled_run": "bf976784d73cd65045a661a11407bc7076854d1a28ed1ed507da3a344234f7df",
            "_RideAlongNBEGMPeriodKernel._consume_output": "ea8a7bd2399b369945b6d31bb35951d0ef863c8604745374bafcf48c49518767",
            "_bounded_nbegm_kernel_output": "dd01ddd425e43b5aed3dd7d186b4beae81f6cb1b78617ee0c641bc302e6f136c",
            "_fold_nbegm_scalar_read_records": "ee39b1847d388c84d876e51a9f0160eaca644372a7d1ba0d583b2deb89da6440",
            "_nbegm_cell_continuation_columns": "514ab277b764e5dfecfc321e0311d975ef558c99cdfd69edad0fb6e687dccc92",
            "_record_nbegm_scalar_child_slots": "82832fa81c78349fad64d63197dbd97b75774a01bfcef32169427f2e239de53c",
            "_validate_nbegm_integrity_slot_counts": "6b74be7ee8f8a7dbfcb6d3529964fec351e5da61eda773f8d0b1df7aea02fafa",
        },
    ),
    CONTINUATION_ARGUMENTS_SOURCE: (
        "0ff76c3bdc198903ae6a2cbcfaa1d4ddea3415a92c3ba29a77cea338812457ee",
        {
            "MarginalLeafArguments.__call__": "e9887274c66d0f3ae9491e7f10524aae4c0cb3a0536dbeff5dc17ab8abbaef27",
            "MarginalLeafCore.__call__": "aad3b455600f6cd5e695f44346904adb94700b92647c3fa20b275dbbd6d736d8",
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
    "src/_lcm/continuation.py": (
        "b610d33af1945219fe50710d5865d414a4d56010cd5713d036a2bee11bb34ea7",
        {
            "EGMQuerySpec.__post_init__": "89b66b576115d08be543418ed033b61c48aff06ffb1c4fdc8e16699311135ad1",
            "EGMContinuationSpec.__post_init__": "858421c8261189b44318132290da23dc53c63409cc4431c012f83b9aea63a480",
            "validate_query_bound_authority": "c9b892f5e998ceb1a11106c08a3bb327c251bccdf1f3488f93887834b5aa5194",
            "_same_bound_identity": "9112bff1752addf0eb915aeb1e2be156010a1ea7c17a46d7b46ccd593876610c",
            "validate_query_model_authority": "e8f3796a3f43cea3b1715011a00fb2f4449d4891c86346fa8617794cd7f029a4",
            "build_query_model_authority": "2dea949c44a328109a3f06660d93b25ec8e7558fbfe7c8a396e6c511ce0b0038",
            "_query_bound_evaluator_targets": "9f82f0ea98ba9b0b214a3e59a651a592b7a9185a549e4012620d89f90c9bdf45",
            "_query_evaluator_contract": "5b842073f2a991ae42b1f81aef724a26b7e60d94c6f9406c3fc04e2e84371e30",
            "validate_query_evaluator_authority": "798c68ca17d5b36414961b15a25825eeba5bf7ec801ca7be57a815416870adb9",
            "validate_query_template": "33061480152c70020b0c11a9a978eb8d53890822279b49d5dee7a431fdd81bb8",
            "EGMRowTopologySpec.__post_init__": "b8b3d2ec434b2182bd79c420dae4bedd4c144ae3ddaf2941e5e52a73772ba3ec",
            "_validate_row_topology": "002d0b993c0dc92edd86fc40dd102696c80447c9f3235fa9ae1f9f58ff909ca3",
            "_validate_row_topology_authority": "9a093fb013afd7594cb1d6d0abecf97c42f17cd2d20cebf6c24f3a3b0b7f4023",
        },
    ),
    "src/_lcm/egm/query_data.py": (
        "4c4199e26f39880bbde5659fee16a515e5c394a792b3f3ec805ef8003261e71b",
        {
            "EGMQueryData.__post_init__": "e24a38d87ee90f33f88db56e36018ffa4366d8032a9e98259c72ee8dfed268f7",
            "EGMQueryData.leaves": "59270bb8379faca4d29c34c474710fb38ee31888cd5ded73da165b329f8df5da",
            "_query_record_leaves": "c54dbae51cae94b3299c65fef284193643ac5722a639083ad95ff33bf73b4b7a",
            "_flatten_query_record": "8cac78187d8e9de027cde73c68bf20b2fb447a27a145a8d6ba5be3e6f777a9ab",
            "_flatten_query_record_with_keys": "2fdd9afcf893c8ec6ace15244f7f4ae5074f79a4ddd03f1801980e5c83a53836",
            "_unflatten_query_record": "e3892756c7f98ed7fa7ca2ef9ba126727ca4cbd9f8e9a16338bc0d280cfa0907",
            "build_template_egm_query_data": "a80a27c04c9142890a211c2ccc452fd65507a3e50a8ee51573f0a322dcb9b9d3",
            "select_query_data": "06b79c4d57ab75c9568bd8b9e5a771a0df9d2eacb20f47aa47995d6bd99d293e",
        },
    ),
    "src/_lcm/egm/carry.py": (
        "a97052072ee5f4fb7232d6a3033b89afc7ea2148516246bd4421091a786d84a8",
        {
            "_flatten_egm_carry": "63539b8a2847824f1b7ec52d33e8ffac5a8a8ab1905f9bd8be0d82baa8b38260",
            "_flatten_egm_carry_with_keys": "c6298ec459ae26b58df7921e6ababc6be9a04cd46c77726d8dad064ac02dec6f",
            "_unflatten_egm_carry": "f0c5176faf75f5f36f223b180c8917f5d56a18ce0ca42a7fcab1769cdb08f1e7",
            "_assemble_egm_carry": "393f264afe7531cbe47997a54ceb807221b9991925865e3974d250734fc9291c",
            "EGMCarry.leaves": "3d181c30a7098d13603d123e4df92ab034b423e9507bac29984b1c259d205f01",
            "EGMCarry.capabilities": "1d11858ae23e0a3476dae46784b7876b29051409a9f3309db9c7073bef20c5f3",
            "EGMCarry.value_at": "9195e6cc1eedd706bb39d583f73c465d9fe989677e44c7690ea0fa1d697ae98f",
            "EGMCarry.marginal_at": "878f8dc72bbef252a9ded576c64b6ed55c12ab1ff261e85071283710a5106996",
            "egm_carry_role_tree": "f4ebe40ea7cf992a69292073e7d132cf107064c75c74cc359d947641122c4d76",
            "egm_query_role_tree": "7f414a9d67be2308daf1bd88fd3eeec7d5b4d327ab051ac157a255d51b057a5f",
            "_query_output_role": "babdddeae20f9e18574f8ff73b8297689c75dd13219d49fdfcad992018d65cbb",
            "stack_egm_carries": "1d3b071519e08f7b6eed53b4dabe86fc3c832e578a10d0142f2ea74d460d99fa",
            "_stack_carry_leaf": "6e86df17be832cb07388adbf2641b7b5413e5421290c26556556afe6c3bb6136",
            "build_template_egm_carry": "6070b3e31f4c9b0eb249211bd62e6ef4ac151d77d13485209001691467557f9b",
            "shard_carry_template": "98bc2a38906b5d704e067ef8dc513dd25581d8ffef9663d2c94f820eb07aef0d",
            "_place_carry_leaf": "39d6f616dbb335dc878b283d6636ee09166271954e144d90a0d6d888834a240b",
            "_place_carry_template_on_one_device": "e39f96351821c5a50dea7cfffc858471f28c9b6c7777db28814f2d671e1e35f6",
            "EGMCarry.query_data_for_transport": "842f1b02b40dccce93a3c7d100d178d8d9f045db364c66b068b09203d0b2e4ab",
            "_flatten_owned_row_read": "3abf8e5794abe69e0f9d5b374e8e7bf33e2aa689c64cf3861a17622067e3d9b9",
            "_flatten_owned_row_read_with_keys": "232a25a9890c618746f9ac0ac6c88f37af5351c35b80aca0aa35170ba15733b5",
            "_read_owned_channels": "e25c42af4d22493b9ed05855d71d2d33faa498099473f7bb8a2258736aacf63e",
            "_unflatten_owned_row_read": "3547855dcaefc477c37b926122041c0609c0858d84ae87190db77c22f25cfbcc",
            "read_owned_row": "d6b96802842c1513e8d7bf8b6d44c5e241bf6d6dea4ee541f185da3348c0fce3",
            "read_marginal_row": "be6f84b6b72a770593c02a229ba0f3154b81baa823c904b7365bdf469b2a38a4",
            "read_value_and_slope_row": "0d8d41de61b40c01664403a6e43e2391149808cb6ef719e63d63782a1d295e9b",
            "read_value_row": "86acfbaff2132e44ab86f33d4671619fb3c95c0d605dbdaf061fb36ad5c01388",
            "_get_owned_local_piece": "ec4a2000545c6f7cab63b80314fc6e4f56a3f20fef5091282e06dac68cdaf734",
            "_read_owned_row_with_support": "8394d3df0bd08bb21219ce0bb8e5a9009622dfd3edbb7929c185333b332b5013",
        },
    ),
    "src/_lcm/solution/egm.py": (
        "c8bc5beafff52bdaf2243e6ae1ea1a338f823f299bc92e50b5a51f917fa7c84d",
        {
            "_EGMCore.__call__": "83fee11d9e754f13eb70c29bc825a42de7f1b1dac2ddf1ead4fbc5b66818df70",
            "_EGMArgumentBuilder.__call__": "6f801c142b046fa383a0129dc48e7e85c63fea66e8db27b1ba2213de4363c9f3",
            "EGM.build_period_kernels": "3a55fcebeb5b7a1143162b3d74b111e7b3a738b5ed5eb7afc77df293b52c927d",
            "_build_egm_period_kernel": "1e316f311fda317155246bcb25a9f8b102c93c6a7f4032270fef1363076b0302",
            "_EGMArgumentBuilder._law_readings": "f85bbf2978a4e71f1899cacca7cb2bbd9a5b54079daa786fee2735fb7d5652d3",
            "_EGMPeriodKernel.__call__": "68fc59f9fb1ec137a0a012bc1162bcb26d5d57df25f4d677d0f54c45a98c39f1",
            "_scalar_egm_output_role": "66b3813ab42722b10b28d2a57008fcc40cc5bcfbd83ffbecd3c9fd6d0ceea641",
            "_build_egm_core": "424fa2bd376638fd6cfa0b5612aead6acc711f7ee90d2475c288790414223ca4",
            "_RegimeMassGuardedCore.__call__": "c1627ad6fa12c501cc8bb00f0a524fcd3c6bf96c8e4e037cddbcb295d035883e",
            "_has_mandatory_child_failure": "45e3c79e9d7c78c7e260fb7312f54913b533080fbeb6f4339a6ea6890be51299",
            "_EGMCore._value_reconstructed_action": "ada1fc964a5168a23b1c2188e7d3bdb50e91b3106ccfb79206de99f3f7ee3c46",
            "_LiteralDCResourceTransport.__call__": "e7e5b6772745d7aeed3de26f6455db2712e76a0c8858712960a7d54764076602",
            "_LiteralDCResourceTransport._at_state": "38224bc9b104d78871986776f7c1adecea3b573dbf60a0e1cec9f69f6ba969e1",
            "_LiteralDCResourceTransport.value": "57824eddf242a5966f9fe3257daa65be276fff2a03f67c3067ce63d17977cbab",
            "_build_literal_dc_resource_transport": "1c1bf883cc393b830252d8100197fa878eed0aa0b15b4125393019ba936731bc",
            "_read_literal_dc_linear_child": "4c2b98228d468b240e31c41b4049b159d2ac099f1eb6cbe24a4fa8b104e174ed",
            "_read_one_row_dc_child": "847665c3d9a196cb052786b0b714c88019f99a76596ebeff958d909a4ffe3f24",
            "_EGMArgumentBuilder._retains_regime_mass": "e60931258e6d2a357a5495425865a6ba1b48c04b1026fc87daf5d8be0a43db31",
            "_EGMPeriodKernel.__post_init__": "da494ec15aad8e5570212813d4fe5d347abfb83dfa64f3e6f2656d53a6296f7b",
            "_EGMPeriodKernel.core_programs": "0d96f7bea814e419ef1dbdebbc3257d63c3c0d36d9e6e52e619f8f44ae5a8a56",
            "_EGMPeriodKernel.with_fixed_params": "b1dbe6505054130c1fdc98533ab637c8f40062b39f79bd42886e1329b409a8bd",
            "_build_one_asset_carry_template": "4a91c99f8f48f46e473ec6cc98c082bbce51379435d8d8a11dad538199336cb0",
            "_one_sided_boundary_savings_targets": "bb5854c2b079b1b20599c1ab903c0bc1b13bfbd542b556c0e85bd14fc240287d",
            "guard_regime_mass": "e9cba031ddd8b786eb0edf5116967b7b43d73d5b95a44752637ca9f2dc2b844a",
            "_EGMCore._prepare_child_read": "d98fdb0fce7084dbcd6eef24c1a2d94c1fd2b75b930d67e90e7f9b79185abe7b",
            "_unpack_scalar_child_read": "82c51619056c8571831c465d1f177d145bf7360b2bd1241fa075f601c4f1ade8",
        },
    ),
    "src/_lcm/regime_building/Q_and_F.py": (
        "3ac04bc3b6f1e71740a3a55e7344be239fdf72ae6d57e0fc4656e9c002034de0",
        {
            "runtime_bound_specs": "84f9e94e2f2fa9cf0f27f8900a244cf80b64681bf23b2973e5c08875c76155a0",
            "BoundedQAndF.__post_init__": "7a5a4b541ee6edb1d10dc24a715b93cab1a026b0f0d7151e6085fa00a1b92611",
            "BoundedQAndF.__call__": "31daf2f9c3ed33974bdfbf73f82438208f07d6fb9f66a8b0b3acd28aa4ad92f4",
            "_BoundUtilitiesAndF.__post_init__": "c1910d1df00ea9332bdc2e2f62a8fa205da5e2acafaf9c1a6cd17bb4845cc3b2",
            "_BoundUtilitiesAndF.evaluated": "b0328b23db1920d3396323688e198570861af372edaeb49ae571c337fe0a653a",
            "_get_U_and_F": "020611a2a21cea42d649067f79139c3c7ada6bdba76f8d29045d1b5374b7c5c4",
            "_BoundFeasibility.__call__": "7683f1d746db4874566047d2c4b3065d93d85a628297efcc8437f4e8af7e54ed",
            "_get_feasibility": "c7263cb59e61c0aa9196151b4be1dca447878fd62b49e09e2ef738a3fe1baf63",
            "_QAndF.evaluated": "bc56c7283a7e3f0e31b41e0d9a4c9fae176bcb691c52ebf81ba2fcae7a6ce4fa",
            "_TerminalQAndF.evaluated": "9773c0ae2b658ddab50c9bcaa10ba7838609e84a350ea9d3d7399a538f86e6fe",
            "_TerminalCollectiveQAndF.evaluated": "9b695227ba1f0b27b84597c466fa8ebde4042011e69505053c51eacbd738ca92",
            "_CollectiveQAndF.evaluated": "e8c2e972b2586a4b1481577583fe225b135349f625b57e7ff65987fe7d7b5b11",
            "_publish_signature": "29eed5ee787b943c47cfea245bd33535d213c0b2c21d44e05247fce03cf80980",
            "_QAndF._evaluate": "830431b717e7bbb7d3c4ec5f3d7c820054eba32a981f2a6546272e9fac6f2fbb",
            "_QAndF._parts_from_utilities": "dbc9eee3d9ecd100353bd807d0069028ed8793f46ffcccad0f42669209ee9a11",
            "_QAndF.parts": "976499debccbd87af6f279bf64a78cd042c7121f689fdc9d3bfa4fb3b9b8b8be",
            "get_Q_and_F_parts": "9a8de79067490b976265f47f0a5ba16de023377ac082f14fb84a1296ac655a90",
            "has_objective_profile": "fcc251061f52d4faead1c847883388539ce8a5aa1aeda708aa286a015148c81a",
        },
    ),
    "src/_lcm/state_action_space.py": (
        "a03e227e1d7080f4df8f58c85cf3a0f62569a4ae2e810b5760a0ebf8416d48ca",
        {},
    ),
    "src/_lcm/egm/budget.py": (
        "92b42d54a7fce80057798bbc80fc3f0c5c770c9efe4d73b772fd0d59102235ea",
        {
            "get_intrinsic_budget_constraint": "828ca3f365fcc19981783af5674990ab7a509a9447a8b4222b639ddd9f0208f4",
            "_IntrinsicBudgetConstraint.__call__": "5a80f9d6eba5d8a1b9801432e627da80598e79e536a31e0987265bc8f6692a44",
        },
    ),
    "src/_lcm/variables.py": (
        "b855682174aa1cfb906cf87c68147694dc25f879247518b77d01ae6c266e38d6",
        {},
    ),
    "src/_lcm/simulation/subject_parallel.py": (
        "b9b86ad8df5f8e75184b9fe4831cc72ced54dc08f96d252913aa1f4a04b9ac8a",
        {
            "shard_subject_function": "d92328e250cb52e6fb457939c3eb9eb676f747c982ba38add780147aba04cce1",
            "_evaluate_local_subjects": "bcf72e41a7312c0428173031dfd8c1a809a8ba2027966a2c6502dbbc5082351b",
        },
    ),
    "src/_lcm/solution/kernel_output.py": (
        "03a69ea63dd6701fd551d682b695d273c6216a1012a91213d8d514c39f49552f",
        {
            "consume_kernel_output": "26f9a541efb25f864997f90d8be23047eb0a1194c73f81abfb692103486b1ab4",
        },
    ),
    "src/_lcm/solution/landing_counts.py": (
        "c0b41de52c1eca64730c8afec2f0b88b096585a6ecfded9b2615123f3767c936",
        {
            "landing_checks_of": "5246312411fdfca9375fdc34b38e57a15620ff2a1da81d0d5d6c67bb0a74b9d9",
            "get_landing_counting_max_Q_over_a": "f57081013973a279fead4eaf296eba5ab3f9322c364b55a01ed7e37612779778",
            "_LandingCountingReduction.__call__": "faeeca7757e7ca66f394f6b72e1dde7ad58ca7ebd81f73d32830a3f056a30a3b",
            "_LandingCountingFold.__call__": "6d12d6fadc0bcea3f456324e0397e1e860a74cf1b4d5593919ae49287c445b42",
            "LandingCountedGridSearch.__call__": "a2e4c8877e1c1a2c7313ea3537866e5ece847751ba84ed0032d21ae0b9bb22d1",
            "_feasible_landing_counts": "1ad2cc2b01dfa37959918d4aabac29875d658ffc3b21924e1c981b2769226668",
            "fail_on_reached_landings": "6d2f6676a0fa1fd706be1baf0baf0e575c79b0a3068a8f86b94fe2056996418f",
            "_Evaluated.__call__": "9752355eca7af9b4a81c12bf72eecc6fb2e16f05e1f4f3b802cf0bd8f87fc046",
        },
    ),
    "src/_lcm/egm/outer_envelope.py": (
        "6e969e5b37ade307cfbdd88fe85db8b45f3fe5082a6b74536a34070cabc84a01",
        {
            "stack_candidate_carries": "8748c418485718ee5fde654aebacb42f41d62fdb721a59833a56c2918543c217",
        },
    ),
    "src/_lcm/egm/asset_row.py": (
        "767d5c2e352d6401dd58e1bf9c4932feda35a9145aead7102c2bb2a39dcfd03d",
        {
            "_SolveOneComboAssetRows.__call__": "3e5f29a1c3a32581da66ab54dc92ebc2268aaa57d97811c7052c941ebfc22404",
            "_SolveOneNode.__call__": "f440068fbd1882c4e935864b43612aab34eadbbcf3d662931792817dbb322c5f",
            "_asset_candidates_with_direct": "2b58e8449ea43421ed118df2523a85af293981a95e4f39357c852c167156d9a9",
            "_asset_candidates_without_direct": "f7ef520fb51ab568463ca9b2f68667459ca212a42699fca1ba1e43f984fef75b",
            "_asset_direct_gradient": "b0c214f13ed17dc0b8f95cc973439bfb156c1def010f68d46b5f4bcd086f02fe",
            "_conditional_asset_node": "56f1f5b20c1802a25f89faa8c48108ffa95318f423c400519afefd0d9073b3ea",
            "_conditional_asset_rows": "ea0d0990c9bfc3869480d3bad892cbc81788154365a214df01e9de32884f4508",
            "_finalize_asset_row_node": "6e6b583ef6687ccb58d739f00a7ad637177055cf093e6a30ebd60adc123f89d4",
            "_node_value_slope": "5bc5e2890ce90fe67fb992ec104c4d6783573a6f0b3b226b6ce6c20b368bc081",
            "_status_asset_rows": "402f4ecb9f972630c245c9493df3537fa1362654240438169c8ae0d8e14fb223",
            "_with_asset_child_status": "7b3499350a8d6c510e4b60eb52e44cc66135f63fbc0d1b41f1a5deb35f88e369",
            "publish_node_from_bracket": "1d1c24b85478cf62d5ae872ec4589ed30223d0c6207d8f92fc58c3d06b90a7c5",
            "_ExpectedContinuationValue.__call__": "b640419789c95b21164ba51bd8ce9cca24641825fa2116fd5392db900152c1f4",
            "_continuation_of_euler_state": "2ed89ef322008d0aa8094e81db68b6deb7a85bc78608abe4d27e03ee2bfe2ccd",
            "_get_expected_continuation_value": "86f45e9ee1584e84ed389330c8036f45a09ff478933a4fec2fc8ae700200cbdf",
            "_get_solve_one_combo_asset_rows": "87e630e054aa3f3ee1bb6f22100a5bd0aa8553f769d2290e30627980ac3f470d",
        },
    ),
    "src/_lcm/egm/continuation.py": (
        "83e62650a4d42b04d7c579c69d39de1b0a0dff4ef80e570b631912f3d1bb02fa",
        {
            "_child_row_values": "3a956663dd626460b776efa317ea5f1bdca72a19a6786a484dcaf370f0a30082",
            "_ActualChildQuery.__call__": "5502c0071b798d1ea39b340e99af4129f69217b977ef9b96d859fe0cbee41590",
            "_BoundContinuation.__call__": "af88073d02a40eaa303869f1d041a48c7b9e69030f6548ca13ae4f032b2a92cd",
            "_BoundContinuation._read_child": "131100c9c3a8878ba05a6a9a49668cdef21d4a8fa31dbd7aa55e8c711cbb9b66",
            "_BoundContinuation._retains_a_distribution": "43a450ba3413f120181f194dfae6c406042169c6db00d1159e87697172e7b354",
            "_BoundContinuation._with_query_status": "d45c483a993d2c779a87a937331483bcbff940878b0d02a27926803ded8789d9",
            "_ChildCarryReader.__call__": "764a35bf42f8d2c9e98e819b225565b0fccf7d927171a94313673021176d98fb",
            "_ChildEulerState.__call__": "239214c25754c91c9d8e9148f1c66483416f84732836ba5a82c62fa578fc8796",
            "_ChildEulerState.at_node": "2f27f807057c87607e8bffd82e8927875ccde3c8c9d8ed520d0ce45c09941438",
            "_ReadAtNodes.__call__": "3b50557079f43c4095bb78ef1d9e111371f641c0387c095e96079db6ada5a78d",
            "_RowQueriesAndGradients.__call__": "2f9dddf3b622a1447edf2761e05aa3e5a20b9105a1d4f8169f62f0393ca365a7",
            "_WeightedNodeBlockFold.__call__": "2259fe809e394bacbc348e7558292c3369166b99de8fd5a9ef0ee7bfb15ddb64",
            "_accumulate_ez_partials_over_blocks": "24ba6c2ce37945c51d349d9a6c4372ab9b83da52bbd85b097aea56ad1167feb5",
            "_aggregate_child_choices": "19672020613cfcb3f7c1de632bec6ad0a569ddf95927789126dc6dda0dc5eb1b",
            "_blend_passive_axes": "ee86b0e59259abe870c8f24c290fe1b9188fc17c87dd6c04a0e72cfceca154fd",
            "_build_child_reads": "ccf8da8dc5c853b47f790cea752d7f8eb9912730449f985fefd071c6e89f17fe",
            "_child_draw_reads": "245d5121170150b1d19ecfb78c05e2f0939e3d8a7072183ef92266fce4960be7",
            "_choose_blended_side": "04879417051fa01bb194684a3231428730691686c39c6fdc4b3bb8c9ac7a95d4",
            "_collapse_stacked_candidates": "e59cdbdba27c8a696ca116a3af69acf30132cd3fc28d85bb7f66e6b5fc035029",
            "_compute_row_queries_and_gradients": "896ade9d239fd56ad6922dcb33691891925e60ad16bec8f7918fbcb22882b17e",
            "_draw_axis_names": "eada09cbd8ef8a4c46b2fc9e58aa1ae723142fef40d4787a35864fbceee0d6e8",
            "_expect_over_stochastic_nodes": "8385931c4e3f8701b5fde4f54b0db1a4c45ce9f2e44c0755c34ad2992b444cb6",
            "_fail_if_carry_shape_mismatches_declaration": "bc3da4a413b8077a203b35ac1719533fee409cd66914ef6e832fe34e59a69b81",
            "_get_child_carry_reader": "b927d2b7de45e571c376c1eb3d36f904a6268a3ff4d6ee58c3bd9e8f2352379f",
            "_hard_max_and_one_hot": "2d0dc3e8a7958253577238e9c56cf1dc05f89da07387b3ce29304de688d260b8",
            "_interleave_child_index": "e4524eb5c0036dee7ff31a0539db5c33bb985af69993bc070ff54eea2f113904",
            "_is_runtime_process": "29a41a0fd7e57f305a64d89d737fdeebff52accab0667897e52144a087a7e468",
            "_is_stochastic_child_axis": "7b92f49106ba332f46b2fe9881f7fc268cb823da09770fce7db4b6f9221159e0",
            "_joint_node_weights": "9cad13fb36920eb93c9d5de9df8032df4b36995231b111fa9160dbda5e21454e",
            "_local_support_values": "c5b2dfc7d92bea5af544ac110a75a84d5e95e1c28080af1760cd971cd0ef7078",
            "_on_node_scale": "499ff177d987dbe6f8d25ca952bc7eeef76b3843cc20f1116c76e7f264d8ceb1",
            "_partials_on_node_scale": "6e3ed59421e6391ddfb2a11bfabdd1f34f428c0abfd8e455b8b147c86f8dd6b8",
            "_read_feature_child_choices": "3d6fbe16e188a48d837f643afaa114fa8960ec593db538a0a2e6c23aef8c620b",
            "_seed_child_query": "75327ca4560cbd192699fdc7b8019ee748344497ab8de58c9472c1afae9ba75e",
            "_stacked_blend_payloads": "244650d644e9f2eba160c6c6601d549c35829aeaf336821ff7741acfbd36bbaa",
            "_with_co_map_states": "2bc7faa0d63a672a221f061a44a9622e35baccb4a92f227355aff9e1b57553cb",
            "bind_continuation": "26d1a643a4796a1454179c77572035d9bde09f6c5b3e3a581ea47a0d766ef33d",
            "build_continuation_plan": "fbddb3c95240fcc9534f83c8a487f2efab80247cca84ab59910cbe0707c92aa7",
            "child_resources_params": "4fe1bd7918b429610e776cfc361527aa6203d49d0eb374dd1fb3ed005d067157",
            "_accumulate_ez_partials_block": "b466ad601d3c750221ee6f09ec1a3e8bd6269f7f7baef3217751e0c3e814c7f7",
            "_composed_row_resources": "74533a91ac290c152f7fc6385a3051c7cf6f73b317dc0a71646fc49b1b49e5cc",
            "_composed_simple_resources": "29e687065c9babe4b63f8b879111715e05904030e9ec6c639cab91450ed8bdb8",
            "_left_slope_row": "96cb4c92f9151a44b195a8e590dcf4b9f5d638e681074dd29503a70db474be69",
            "get_egm_continuation_targets": "b9ef5ac989fb93d2383969729d5b5feb77bc96fb391da4087abf2afff6542a30",
            "stochastic_node_axes": "82d34a340ad704f574d8137021361e8baed052d291344aee919c7862f1db2cd2",
            "_build_feature_child_query": "2a64dce34a2e53c8ffc0ccb013d3b508574b8192535b3d649faa59f7a350ec8e",
            "_select_feature_child_rows": "8c0df4a43c0775987f507bc5a312e52470ec34c2800ea5f3f503ac40566053e8",
        },
    ),
    "src/_lcm/egm/dcegm_query.py": (
        "45e1cfd8b7a58728327ccb6037ffbdac4b7992ed7f0dab59949611f5e4a3708c",
        {
            "DCEGMQuery.__call__": "dae1d967999ec126d88271c13d5b1fec67fa8a76478c8ca122a9eabfb4ade2ed",
            "DCEGMQuery._read_at_floor": "fecb3a2bb6ff544c4e28bacb9f99a1dd7fae8c93d75678f410eb4310ce17376f",
            "DCEGMQuery._read_liquid_corner": "edec6fa7338bfcf978954dac08016c4377477b35026a0e85a5b97f65b9622367",
            "DCEGMQuery._read_rows": "dc52f15dbef7ef687b8395a7da6da179df0c0ab688f4d7494d8c9d139a1cba78",
            "DCEGMQuery.capabilities": "97c1acf69c7dd500d9f263c861ecf832fe217552c85ae2923716fe48d32cdce7",
            "DCEGMQuery.evaluate_floor": "e6f1961f20820411a327f2106ea42d9c87af81b4b8580877020ffc1c3c984050",
            "DCEGMQuery.input_names": "77080dd8727ac51487da2c65e73ce843ce583e543925a4e976ccc89b8720c1a0",
            "DCEGMQuery.read_choices": "6bc81cb4f5c1ebe3771eca4deaafda365d77ebf0d2c9d41d74cbe45ce98ec9b8",
            "DCEGMQuery.read_prepared": "d101ef7bdcea7fa9d51953584d7bf638b8088909e3c495f1517f64b48a8465d8",
            "_evaluate": "b53e61ed416e8731e6c2aeb0efa9d82d47d2b3677de747eae3f0e06a85bf70ec",
            "_factor_product_bits": "fa8ef8a9ae6dfaca83e8d17adbfceced549a6e658d065d8972737aefcbc29e2a",
            "_factor_product_jvp": "e224baa675f1694fa37ae550dfcf58147b03c12fabe5ab9052bcca716815c0af",
            "_owned_at_query": "d3cde35a170d1c9be89ce67df9d332bf7dd735069fcdfec0193272070d87f472",
            "_passive_weights": "6700895e16aa9cf47da243442b184f5934dd4d1284a971a4989319dd02a8b0d9",
            "_read_compact_corner": "cc3a17969235760959c9bf7dbca2883c60b735fbab2cb54f2a639165f6d8855d",
            "_read_owned_derivative": "1a40fb7e66ae6067ff80e1e00606cef56d219e6de198c341b7f9c80fb4340e25",
            "_weighted_sum": "5fe3ca3e5abe5f2d291a9b534b22a09ccc9c0c7175f7f84c75709aef852223dc",
            "build_dcegm_query": "07d7b24c0c8f9d74ee3a300ee50630fa6ebe7e53b55ee63c4dcdbd858155bb7b",
            "DCEGMQuery.read_prepared_with_counts": "9ed2c050cb6803a5104d99f7f2ea21b7242fd1daae176170dc181e7e576ea26d",
            "DCEGMQuery._read_at_floor_with_summary": "17fc1c8b2e503fe2424e7eb9ec9958f10ce721d6111062c92cb30ae3f045ed92",
            "DCEGMQuery._read_rows_without_counts": "0c9f5360618501fdb69439f9d7838eeb21e9f23a2e49188aed3d8632db6fc5f1",
            "DCEGMQuery._read_rows_with_counts": "454c96a1bb6e18b7924bcc6990fa05a2af81329b3be19c8c79aabcb0c1e8d5e7",
            "_blend_owned_rows": "bbda81a9ead622e13f77687dabaca9928d29784f08fd8d7b325933d254d2f459",
            "_read_owned_derivative_with_support": "ed554cdaa8af45b9c812ed63c049bea02f6dfb5b77ee62d99a8ae45ca0f76313",
            "_owned_at_query_with_support": "44a3d9582317741dfd4b49d4ffed7c1dda646a2fb69caade6bdfcf07c2ff9736",
        },
    ),
    "src/_lcm/egm/dcegm_query_model.py": (
        "b01c28eb04463ddea49a3fba277911e83e4c4358e12813b5926754fea83a3067",
        {},
    ),
    "src/_lcm/egm/regime_introspection.py": (
        "85dbc7dab5269f9c357120319d8fbbd3a6eed82128ec98e6ab51a16999baf0ea",
        {
            "_IdentityResources.__call__": "8e4c0452a0f4991dbf0dacd01929a7ad2f1236e23a93eb244d8813ec554d37c3",
            "_KeeperNoAdjustment.__call__": "e2089c7184f51df1f99350ce1648f33fa3a6accd0457381ca064643076c1efc8",
            "_annotation_of_arg": "e553cc9544e4b813451de9e45ff651a72d4d13645a1b1eb4fd9e889a9a5a6209",
            "_as_dcegm": "ef53c53b759fddec0be64e442de77a74baaeb06f770b8822ab8d1cce9e8d3e0b",
            "_concatenate_child_resources": "44e4fdf6aa27a10f98064e10a6086253f65f84e46f1eb0f9e96a5eb0406250db",
            "_get_child_discrete_actions": "196de5336ce3aafb7f4a8a36b5121df38fecad588fc48c8e814cdc2d44f6648e",
            "_get_child_resources": "65f36bae04372e5cedfd01ef134b0413709d6985dd71c744146386680a594940",
            "_get_child_state_name": "6c707544ebbc74e91d4c5d2d4b87e12baf185f7dd1dbef5bc670461c4703250b",
            "_keeper_no_adjustment_function": "f43bc14daaa419f78acce503be0698cb114a80d1caafdb7217d4962226b1ee35",
            "_get_discrete_state_names": "b3f0d8886e1d8d924e80e9f6a016e8e2200de6682a135be390cb0e3769c69a9e",
            "_get_passive_state_names": "a6e6cf7327ab51802b73e35c1862dc64a7f02a1554396cdce5624c29aba6f8ca",
            "_get_process_state_names": "b72746336547658b9fa9b2d8d0a0a38ee7bfdae5bb0944ec52f52d8495e25c3b",
        },
    ),
    "src/_lcm/egm/step.py": (
        "f0b8e8b9770d840c4cdbe857b41ce733d4acd6def9ef895a208819511d425216",
        {
            "_ComboFeasibility.__call__": "61ef5afd190bc979e8759189e8a8b08cb15886e98c982d9cd1d4f293f99020ba",
            "_EGMStep.__call__": "f79213e5a875701ab7c037fc3b2b203e8fdce0c93841b43448e6a20c1d3fb641",
            "_SolveOneComboOverAxes.__call__": "8495a7d1c40d720e1ecb7ad82e73d87c77eeb5dfba32fa7cfb1eac6fbde2dc9b",
            "_build_feasibility_function": "04d448a1c1ac79f32be505bd70216dc59948545d694e323495b1b7a2eab92c01",
            "_build_kernel_pieces": "b78dd7d902aebcb454b16c9c8f2d9bf0d773a401f3e7ae9ead6ddac3e2578068",
            "_get_egm_step": "4bfa4d483c1d194b86dd926816a56dca6864a1160b0505939a676c6719c189dc",
            "_map_combo_product": "90c1612777c51c198b3c314b65dfe49180d702bec36d6bc4d292b2919ff4fb25",
            "_publish_conditional_combo": "f661a7eae88bd04016a3cd498857f79ebea1d06e79cc83493780c58b42c0b560",
            "_publish_conditional_product": "fb4dde6b5eaae7293a1997b169433cf14322a505bd698ddc23100ae67ef13c35",
            "_reorder_combo_axes": "c039213dc6f3bdf4d4dead84e8e158a378691816eb0f3e92a2b8ddd63d6e6c56",
            "build_egm_step_functions": "1d4ed663b1a768ebc589007ead2a73fdf351248be2e6ca697661eb2c626b79e1",
            "compute_egm_carry_length": "baad1daf9d8ba079aa22a0710673378848b812ad4df044f9ba6ede365734e18c",
            "_at_one_splayed_combo": "212968272192583e89f1e256610c39e098169da29dab431ceaf0f9bb8b4638b2",
        },
    ),
    "src/_lcm/egm/step_core.py": (
        "1aedd3ae90d581d006b3a78c738515db065341b36ce21e818114cddd5d548eb6",
        {
            "ResourcesOfState.__call__": "af9b9c1f641c364baf0aea48370d1a0b64f898096589dda9d3839d60f1bf9cde",
            "_ComputeNode.__call__": "eca99ef68065d3a3395e4b151b3bef654b73513c3db25832ecef58d28e6467c1",
            "_SolveOneCombo.__call__": "8ba4daef6d4412c63a996c64dfdc297d85588950920dda01b02c63c6fcb7a62e",
            "_candidate_supgradient": "73dd6f311b1b7e55bcd7b22e9723e3e0ebc664b2585c865863a947aa0a5da7b1",
            "_compute_constrained_candidates": "492fdfe24f6e4f6eb61cb1760861ed2911d772579130f2c8f9cdf12b25b8bd5b",
            "_compute_nodes_after_first": "61da7049c08cb91e48437889fc089c5f727cd626dde73ea84963f4e98606e5a5",
            "_compute_nodes_over_savings": "d20ff260cae042689e8a5036c251299302d6a05348a13267aa89bbace9bf2412",
            "_conditional_combo_output": "e696b1d5f2b6bfc5f6e7b135262bf3ac46705d8024502b314a69ecd7a8f9f90f",
            "_get_compute_node": "0eec95c376d9b1bf485d7f447df6ec13e2de1da3d8367d2ace0fb63538e940da",
            "_legacy_candidates_with_child_status": "46d7587846ef1f16f56a242c7e48508c015918ba8da9c9915e0f98dfd6543ec6",
            "_publish_V_and_carry_rows": "3e62dc314cc659ea70169a01afaa245dbf7bc9d79019bdec28af41cbd9c9d4de",
            "_refined_rows": "cc1205c707c2691c76875e0a916d0f1c24082736060f23851085fe5805ec6340",
            "_source_construction_status": "3752fbf4834cbc45bad60bf81245931ba01953f6d04531b35d25e7da73d4fc6c",
            "_with_literal_child_status": "717cf9cdcea25d828a88eac33f616635c0f4e08314bc50c11b64d26c787c7b90",
            "bind_cell_floor": "1ab70b889b27c7dd41e542eaa11c331a8885eff6dbe8b10ad31d7264d48583b5",
            "tile_block_size": "6f0a351163f6f4f39e06e75d4a234d28b9da8280360acc17e9033db3f5699766",
            "_get_solve_one_combo": "456d045dcc309364816335b5c63625f6a3d5d250fb8ee595b38b7c7594d5f15a",
            "_assemble_combo_candidates": "05fe527e25262707afb095a07fa4107717e9345ce7205893cc8b7e4dd78e795a",
            "_mask_dead_combo_rows": "ca74962ddb14231b6f7efacc4c61af4955e4de4e2596e022d965f60d1ea4c7cf",
        },
    ),
    "src/_lcm/probability.py": (
        "bb125dcad684e572c3ffb7ea7fd5ca19b7225662cd5fdebe0c070a71cacdd0dd",
        {
            "_encoded": "58d2dd01903a637adc4af5d18dcf3767eae9b16f685a0652e24c156164511bdd",
            "_exponent_bias": "c3a992302c901ac4ff4bb7c25cb2dc54965e8d08c9b8dd18fe98832431c469fe",
            "_int_dtype": "4bc9a3d08ad398298522fc4e860fc8c28bcd46c59e6a4b4ef03d0b60fb52e73e",
            "_magnitude_bits": "748b35291f6859e5f6bbbe7f6fc124637df152e16ccd339ecfcdcad0451009c1",
            "_mantissa_bits": "dcfdeff09086db4fce5a60b93e55f071bbe145ee10f5238035cf8b8fd4fdaf94",
            "_product_parts": "4577b0ec08ddf1cd64569028b8a9668a6d4bbe41d35891711eab704e3b032cbb",
            "_shift_to_normal": "7bfcaa26d2bbf24b87afd716307b36e8be0548183ee2f6c7cce652ce5753cf56",
            "_significand_in_unit_range": "df280a452049e3a304cf307d8ca743e0925c97c509162b0044f8a815d5e2d07c",
            "_smallest_normal_bits": "b2fbbb558948de8565fe0bc0e90a25ac834b387e808e60f860da28f726eae9d7",
            "_unbiased_exponent": "a39e94ca4b0c41c8416e5cad01bb8150565d5a1337cfa39ceda7ba5212939a8e",
            "exact_product": "edcddd28a2ba99b630a5d5c637953378e97bb2855d0e6cfb885f256b6e667081",
            "is_below_smallest_normal": "8406ed54c08d372e88d970f24d35f181d83c6e0103603cab368c21698988923e",
            "is_negative": "a21b73fcf4342065c86a6d3c86d33bf10500935356153aaf252506bf34280e7b",
            "is_represented_zero": "d0bb522a5bdd9b59a29082dc94f4f39c0d9a7f6434bffb789e530f6cf63184b5",
            "_scaled_bits": "370b33c8756f669c28d46159c12692943e20864c1e5a2525d8c8fcff2dab46ec",
            "_scaled_bits_jvp": "4d01940cf19ee38622cd6c37892842a8d5ffbee4751703943fbb2e4bd278e882",
            "is_live": "71802821da8f31311d00a04754a103a1b800a511542e5108f0baba1ac7892ca1",
            "probabilities_form_distribution": "ec4eed298e7434e2c85be32b67d20021b61e96eb102bffa8da09b6e301161928",
            "regime_mass_is_a_distribution": "77c20e7a228127a1b1bd072ba17a8f0e5140f8a87ab7b67e93c3304b514ea2c9",
            "scaled_by_power_of_two": "66ac45005852e867fe423327bcd3e61ae40217a4d27459d1f726d119ce6c7cb0",
            "_log_of_nonnegative_bits": "f08993db3a70bdf1d0d2a52f633994db0bcb327a0e31a62aabd3b31de4472bc7",
            "_log_of_nonnegative_jvp": "16da7f663340bad73c31f62493e755b0aa8f9584b253ab8f28fac11e908d6643",
            "log_of_nonnegative": "f1dc842ddb6d656c4422e2e36e7b1b33c54345aaa199afd112d04f05c39c813d",
            "rescaled_weight_pair": "d90ca0876a8a90af2ab72f50750d8bc499e2e5538c202181c0cb3896cec42244",
        },
    ),
    "src/_lcm/solution/dcegm.py": (
        "a31ebf4f844994e0d0489968e5b05c817d2f7b81373921cfbef1116554b33e74",
        {
            "DCEGM.build_period_kernels": "4dbe33765fd7e2f188d9985fcf07ab805d2c2190129964f414f5f3bdcece121a",
            "_DCEGMPeriodKernel.with_fixed_params": "576107fcdab8d47abbe0df04148ea500bc7595c77c33353f804d28576087536f",
            "ExactEnvelope.__post_init__": "4975eeacb21f66d64f1c324a6aa57fbffea03a0bc6452f27c4a6e837135b7c10",
            "DCEGM.__post_init__": "e6113e544fde0b166ac68d94deaacf78384e51f0360ad4f88d1c760b39e50698",
            "FUESEnvelope.__post_init__": "2e0cc917eb494bb33da6673a567c74762daa24c6d6903c1d38d9515667900beb",
            "_DCEGMArgumentBuilder.__call__": "83e73b3fa1733d949e557f7cb22dd0789c0e5c713d90c4f97f565228e559c1e2",
            "_DCEGMPeriodKernel.__call__": "0d58d7aeab84d00f85692ff9af0cd66a8a7c9094bad97c784d7511d5492ed202",
            "_DCEGMPeriodKernel.__post_init__": "db2eba075d8b69c2c9e6f3fa76eba27a19717abe2ea00c680fa75a599bfd54c9",
            "_DCEGMPeriodKernel.core_programs": "0d96f7bea814e419ef1dbdebbc3257d63c3c0d36d9e6e52e619f8f44ae5a8a56",
            "_carry_subset": "d3b50184286db7fdb24ed6b1176e4be0fc9a04935e167a98fbc0bb3ff476d91b",
            "_consume_bounded_dcegm_output": "83b7e20200f154c9f103130526bf33e0ca012edbdfb57c02b448729aa02d005a",
            "_dcegm_output_roles": "3608e79882f1dfe8ad011e2b70a9936c81eb984401bfafd5b08e1683d62c6f7a",
            "_dcegm_role_template": "3d05db38a723160cd4eb9c73c4b0b01295574ae39b98a0da16486513fba9713c",
            "_dcegm_status_role": "ad55b61dd70842363177f847a359f5a091246cb3cd1638dd843fb308faa20f9b",
            "_dcegm_values_core": "94547b418fae30ed7614e9037df0f7243cc7da3a6eed281bcd3de53a23043eeb",
            "_fail_if_envelope_max_runs_too_few": "81ba34c1807e953b64e5a80163ecda6f33037ffbe74a13e24b53fb7cc3bcbbc4",
            "_fail_if_fues_jump_thresh_non_positive": "1110ea554a2766f93856e3224bb05d8ee3ca1ccc7ee02b951fa32d9116f577fc",
            "_fail_if_fues_n_points_to_scan_too_few": "7137c349bd2cf8b975f20a1ae1972e4266186cee99dc276f755feb2135398d6e",
            "_fail_if_n_constrained_points_too_few": "5244d781261c2a67b3b0f8d30a158ed91d298c16f68976b8b24aa9cea004e794",
            "_fail_if_refined_grid_factor_too_small": "e9d01a635860eee512b35f96c5eafb5864f222caf6136e40149451d28bc3e7ad",
            "_fail_if_savings_grid_is_stochastic": "5854252ce336bdf030e8b424a2c3e3618e7d04e813198269653e58635cb509ba",
            "_stochastic_node_axis": "ce352befe0ff8dd4001c81b64dd4f1194a7f1c2a65ebace42fb9979f631ff3d2",
            "_tiled_axes": "80649afab5fb1b69191d98b113d6ad46770583e6a82def2d5f09acfbea7c6f54",
            "_DCEGMPeriodKernel.retains_current_feasibility": "2e3b2810e738123afd7a72641d4a42b8e9d5dedea4afdd9cd6a463e4a4da6512",
            "_counted_dcegm_output_roles": "cf47f2166083e9540d7a6bafb6e4033eed719e16823d80afcedd42988881580b",
            "_strip_nested_feasibility": "694117bac7ba42507a0e005c79f7c7c2e179ad4adbae5ba6328f19fdc3a82d6f",
            "_with_nested_feasibility_role": "cfdfa377aedebd2c93aa76b239013ca331db7a1c201b3f508bf30fdbcd18a1e0",
            "_dcegm_query_slots_and_capabilities": "39a97f8631b719779ec894900945ec73b019ea0b2f9e0cc8180b769afbaed0e4",
        },
    ),
    "src/lcm/_solver_api/replay.py": (
        "58a16d058c4e5e342bd83b5f267ec69cdd4e828a80fb45db9698bf593cc37f2e",
        {},
    ),
    "src/_lcm/egm/ez_kernel.py": (
        "56dadb97c4fbf48c2dd1def35ed0a104250ed1f0e6fa8bf0856260a12da6e426",
        {
            "ez_consumption_from_euler": "085242c1acc653e5624001827643fa5acf67f0624be144d50acafb2b0328899e",
            "ez_marginal_of_resource": "3eb273105d066cafbb7a1be233af83065c6a42aea37ca60ad3d8572651b3b6cf",
            "ez_period_value": "9970fcc5ec0831e979c2b7d37f002d9578d56c91a45d23cb38307cf635e1599d",
        },
    ),
    "src/_lcm/egm/framed_arithmetic.py": (
        "87144e91c01100a51599e08336a7dddfbffd16f3af10106cb811198a03db8598",
        {
            "_dyadic_parts": "2e338d297e534751fe05edba3f0d14df5d66732c646055fd89310f9b76e5bf2c",
            "_two_diff": "47bc35cc3141502578cedd4b1b05804de4768c8a3d60dc929cc8840f9ef3d982",
            "binade_exponent": "9d25d6b129d66b3b30615b1d8eac9192560778f152987953547de87681c61a00",
            "framed_difference": "e1b6751abb651987b8652dc2c02f91ee6827da0a32b0225c5f4e6ebf2a94fbea",
        },
    ),
    "src/_lcm/egm/nbegm_segments.py": (
        "5d035038fbae0a58c8da68927aad82b7d050125187f6619f5ce37117388a2eca",
        {
            "affords_an_action": "0b962d6e70b1d821fa22095724f373aacd53c41915296d4f4b96108e4b33a7c6",
            "mask_dead_candidates": "5a59c11a535462c34ed7f1cb275bd8e7586a85947d62254ac0c4f96c14c4887d",
            "segment_ids_from_folds": "a232e667913fb9ebffbad25957febccfa94b3fc5788a2672dac53d73376a2e88",
        },
    ),
    "src/_lcm/egm/nbegm_step.py": (
        "d53641c3c2b3e4937436410cc30653ded4bdff97f5f750a00c36fe7a523bdcc9",
        {
            "_IntervalCaseSolver.__call__": "9e8e6e6d6fcffbc0cec5c9ad8989131c66e0efd7e7e5ae8cba8e3954723fbf36",
            "_IntervalWinnerStep.__call__": "72707f2e7353bbeb5ffea6cf8c1725c775f190773d3497ee2835d3f58e4e7729",
            "_StreamedIntervalLayout.total_candidates": "45e95a6180217b19597ee16149adb1c26b8df68fb85383b7e2c99b889805db61",
            "_as_pairs": "9ebaee0b9aa88f8e523e5d817ed6bc1f74be2b1516e7653a7c09565b92ac2aa7",
            "_degenerate_inversion": "c45a20c732e9e12a2ddc130b597bf820465b82a7712d9bbafd32f00c4d2cf33f",
            "_discounted_continuation": "591bb2c3551df40fe5a372a7b747ae3f484764ca6f3f03c421310a97ccf4fe0a",
            "_ez_flow_power_structure": "286d32f0dd485c182b7701fa77d66fcb4d52642da794ad7554d31ae15d442c7b",
            "_fixed_savings_corner": "a8291f48c367281894d18b5e9fd83e532b88cf04cbdbe520cc2b6a203d0af265",
            "_interval_corner_candidates": "a484de163b226040b43732b479abbe5e27bd7e8523300fb1dfc2a41ea0e6ee70",
            "_invert_coh_with_linear_extension": "b908750cf32a30016e295606fd8fd9f5fd08ac43dc12e722dd514decab67dbbc",
            "_invert_euler_at_node": "91ad4d855f5f74dc76555b18112a97e8e83dab42ead221936d07f6b55efa4da3",
            "_invert_euler_over_savings": "a3338141686587ecbeb96f7558d81fc90d9bfe6b21442fabfde19c550803e50f",
            "_kill_padding": "2ae7399b876711f95ec6b05e628a944d282ac5b77a789ca41ceb73b5b7f8f926",
            "_linear_extension": "deb8bb7255ca344db8760d08d8885e39902ee1363a37ce855e25fba9f5876607",
            "_make_nbegm_query_data": "b51f1692ee86f9f978ba5ab9abc034b908b07bbee7849feaf08c1433679dbd91",
            "_next_segment_id": "9058b0199b5612fd7149eb36abf98e61e7e8a42b10b2fafddddd81c6c7cc613c",
            "_point_candidate_segment_bases": "099c1f90aa459965fcd8f5b5c53be8f3581fce2d76208ace8292fbc52c3c451f",
            "_publish_interval_step": "a4a062d3a8dfb61996e5dfd574c0b75bcf129bc21ef07fb61bf7da5025a1af4e",
            "_publish_no_action_carry": "8f392309f414027d1579e3ba64f2121d299d205aa1581f2bd66e17d4cfde7580",
            "_query_node_kind": "444a5e65acc1a48d34bc6c0c0f9f404ec15ca60f8a449ce95265120c5e7e1a3f",
            "_savings_node_point_candidates": "9159705d3dae69d3a3e47e462c59409ecc5d83d36e01ee8a4acd89547bdde449",
            "_savings_node_point_candidates_for_interval_block": "3fe5828dbad0d346af751d7a120efdf542d7cd270f1ecbae6167283fb7d645bc",
            "_solve_packed_interval": "39a0f0b595334ee12619f8131ebb97c32e10632869293d1b6a043e1add7ac0af",
            "_stack_flat": "1cce73fd366fc6c0c76439d175e4275b3a903639abb7acaeffc9324a83ceef0e",
            "_streamed_block_candidate_positions": "49b3a334f48b21ed0d219ce6c9c8e8c8b3930ed2dd32e82c2c3482b82e46d3be",
            "_streamed_interval_block_layout": "546fd0a7654e5c10a822336365f9e68af717db20221253aabde819d703abf602",
            "_streamed_interval_continuation_envelope": "fdf5a7a8370a2d21f0ad2ccfd684fa60a81d250379d6daaf6a86b0c1b2332a7e",
            "nbegm_multi_interval_step_savings": "1a9929a9b85339d5f1038b98afbb49344976d3f17aa8dd734235d0269bfe3381",
            "nbegm_per_interval_continuation_step_savings": "56554f49407f9e04e04418a7f01e2beaa4864e069f614d378a324662361eb743",
            "_single_row_nbegm_query_step": "ba4077ee1cdff8f953be04f7f5fae4896243e6513b698fc41d86d872c96819c6",
            "_prepare_nbegm_interval_block": "e13386fc7f5e82024ec0e7b964a49d5f740f2fbc5ed88282b9614797693c881d",
        },
    ),
    "src/_lcm/egm/upper_envelope/query.py": (
        "207319529445c9b72b98094f3650ecec3df703acfcdb6c15d4ed857cbc00251b",
        {
            "_along_link": "ad185ea9f330db56b8ee89e1ded66042f67a6dcbd103389842b33d931340d071",
            "_batched_link_terms": "5a1f8a56d71c43c7e7f914a52cf7d141b7e56067b6dc5291462581392890a978",
            "_block_query_terms": "92202932da6a42bb48529eaccd431a6ad340c125a7d3a60749cd4c7a17d77b5a",
            "_derived_subnormal_possible": "24d06e5cda74ae4dd4e3cb8caaaaacaaa5e4c1e6714afde4f5814af123a999a3",
            "_endpoint_order": "16c09ad66f06566c7961514bba7d29ddf3d6913e192952e71a421962d1b1612e",
            "_envelope_blocked_ordinary": "faad1c8204527a3e2b4141894be7cf58d5d8e642c1a17317a9d7e85a3b53516c",
            "_envelope_exact": "f18823d0ecb32f825dcd8ee05610a0fa4983a1e999aa237cfb756be5b1008022",
            "_link_blocks": "9e037eca2d0cc4b0c10a56c6c6aa08cae400a67d982dd73037acc31c5701ce5c",
            "_link_geometry": "21bbe995e458000066a7a97c2f680e24a44b4a29e7d4ea5d298176e939db5878",
            "_link_terms": "9d5a5e0024a7f231c1ff25f8af1b64d5223278f23c319148a6de4598aba5ab7a",
            "_mask_infeasible_queries": "bef6eb4d8bda8251496cc050fdf783667ea00650f092fd0139ea47e31ece0f9f",
            "_on_the_links_own_scale": "8dd84b0b6d817b1912f7f031385dcb3597286ca9d2f96395c384dda410ccdeb9",
            "_ordinary_block_best": "323fb481bb178996d08957564101ea105c8faa546f22cd42d19a853661f00168",
            "_ordinary_block_size": "e90df67dd1725f0452fd73e3eace2cdb8ee58d527ff211ad0825eae52842c27f",
            "_ordinary_block_step": "7b65eb09e46790ed448acd59ca8fadb910451b9727aea19cbbcf73389cafaecc",
            "_ordinary_rank_argmax": "e189fdf83a38a312fb3c95d9570c97d441161acd0848c987518e1b89301267c5",
            "_pad_column": "35594acba860cd4a15140d8b929b2b14ff2fad82b4ee127393fc4c6864a23ece",
            "_pick_column": "c2ed468e0e0cd70f668d29088ed946cb089e44b31c874007c06e0d0b439882d8",
            "_publish_decided": "1dd5b7a0fef2bd5b5715a7da8a74d3bfc8debeed73b600e40e9a0b4f952b74a8",
            "_quotient_is_subnormal": "9c32efeac8c418f831ec4ea1002869bf86065c5f4f5812491ba2522babb90a40",
            "_read_exact_channel": "83b48ca90be8be08f230ccd8e1245c8baa4fce3be2389b3c03751bb9d01c76ef",
            "_read_ordinary_channel": "5fb61fa2c0b3b5fcb0d2ed7b5ac16696115717ebbfc7830d0c2a408e693bcfb6",
            "_savings_link_endpoints": "0e681937fc35246895c96f195aa9b39a3cc3d93b033dac70dc00d5db5d4a5ba6",
            "_segment_links_from_candidates": "944d2c28268d374b432da64a500f2755f1451f666aee9e42b16b2ef40d6cfa2f",
            "_slope_words": "53307674bf7899065e43ede2d1a6af5cede1317b76e0cc01a23fe69bd309b7bb",
            "_smallest_finite_exponent": "e89b0a55c07f6836c39e00895bd79a814dd52b2b74b7adbd07cc6dd5943b5ef1",
            "_spread_over_queries": "325b744c37d290618688da4eaf93bb027753ece58145d9975624f121dacc2b95",
            "_subnormal_operand_present": "d9e8079427ea83b2b0cb03af192979fb2f47024666eeb099d36cce24484f7c69",
            "_take_rows": "8d4e055398c43d84a158e5a0927a33501bc07d879b01c44e1e24cc0500396352",
            "empty_envelope_winner": "9dfc3cd4ae616726cf588a52f9ecf1d3b8b563f292000063f0f759d6b699d2f2",
            "envelope_at_query": "669c476c23be935bb7ff46c253ec28ccb7c89719ea803f0d09de6e847120cc96",
            "finish_envelope_winner": "ce8cbdd55e02d678e6a46cde4352f970a690a3c14af46e5a709f9bbdcbc8895f",
            "merge_envelope_winner": "2aaba98de675e844dac477909f5c544cf23754c29ddf89feca6261232432acfd",
            "published_owner": "7a6063136bacdc55638a1e2cfd7e18f662b58ec25f8819b0dc24c1e1aa01c5f4",
        },
    ),
    "src/_lcm/power_mean.py": (
        "754874381d6e85669a5dbc30a450b1f987c53292c0b98ed083bae0e939204089",
        {
            "_applied_to_anchor": "51174b5a83bfd9f26290c8bf863f1930fd9232838c7c8c02c13c3a8dc54d999b",
            "_anchor_primal": "8f15f6000318be7e71bad6cac00992842d90e9ba4ec9a35c0d4ce85fff67955a",
            "_log_term": "759e0e60399615ea3a80b6d78e589c4432cce43d1947ade1e89a428d6afbfdaa",
            "weighted_power_mean_of_pair": "d2ccbc6e4a64e60f8e7ceed5107fbdd47898d0dc24c7b3f350deccb48b939306",
        },
    ),
    "src/_lcm/regime_building/decoupled.py": (
        "70f5fcda52ea91383e4be13f747c98033510baba317a980283ffbecb221b1aa0",
        {
            "DecoupledTasteShockLookup.__call__": "4cb697f078e503dab438fe5bab75631d3d2f31758b480bb707d920f95fafbbd9",
            "DecoupledWithActionParts.__call__": "44e85f717a656bc945256d5ad95ff6083b48cede4874fbc80a78cc9b493e29b7",
            "_DecoupledQOverA.__call__": "bee09450ab81c889a7ad7c61af2e3d6a9b8d400ac85a94856fa068259107b9b3",
            "argmax_and_readout": "bb970e9ef9a2356f598ae7400ee2a6e4e4c0338429d9f372c650e33da7caa34b",
            "decoupled_logsum": "76dee3957e4adf72e6524c280a97893dd6f19e000ba14be848c729f70d5a387c",
            "get_conditional_profiles": "0de48ea6866b2cf6c0fbf612ed308a90d82fa31adeadda93b905746042454413",
            "get_decoupled_Q_over_a": "119e3f97d9d92fbfaf6dcb2356c130994f55c5e78028b69fc8dede33f075bbaa",
            "get_decoupled_max_Q_over_a": "9f7be2a41daa00d2ff9848491fe97ebaf9d0a695830b3824a7c4226361eb49b6",
        },
    ),
    "src/_lcm/utils/counts.py": (
        "06ea17925d45274aae65f44a20471a73614a6e26779ce9912f72b91c3a3db106",
        {
            "add_counts": "2c0b3866b7b5f76b3a11301fa95ef082e10c68077781f38d92aa0057728dcff5",
            "sum_counts": "dce20fae5f7889d350808f10c1fe0934bae0b5045ba37b7f274bc7acd4c8a3d5",
        },
    ),
    "src/_lcm/egm/outer_interpolation.py": (
        "90041559a7660a3933b280258acb7e0a048eacddc1a7e51f0e139be1b50956d5",
        {
            "LocalCubicOuterInterpolant._evaluate": "a574b31ae3068dec4bbb0084a12bdc46f330184a8d7272a969c930f99a127c9f",
            "LocalCubicOuterInterpolant.evaluate": "3755c0a695ff20afab258d0afd328d45184d2a7d858b2f795e2377a1ef424e28",
            "LocalCubicOuterInterpolant.evaluate_validity": "657e8037bc045ac648c6871c65d119534142924eb6cee15550463386b2f279de",
            "LocalCubicOuterInterpolant.evaluate_with_derivative": "ddd928b6226bc8ec1ff09c543664125af18dae883a6ce535e98779cfdbf719e8",
            "_four_point_slopes": "c3a5a255001af28a1fda40786301dd6426f30140d76184e477cb730d0a86fb14",
            "_node_slopes": "dade18078563fed9aeee17b6e56ab3ea83b5c88e826b2fffdd69f898922470ff",
            "_one_sided_parabolic_slope": "9cdcf4748c8326d1dca665d9d47296d09ebbc8b9db87b03418a15d4743e537dc",
            "_pad_state_axes": "b407634890244a0f7aa330814c38aa7b8aa1efa12f0ed2ceeb78ec294ebb45f7",
            "_prefix_four_point_slopes": "7e729e8183295d1731467c9d0d31cb576c6a48454890f48b8e3caf80d69191ee",
            "_prefix_node_slopes": "06c1f17f9d047a2b0afa81d669ae9424e450f56b6ac8e2a63d72a07f40be09e7",
            "_prefix_three_point_slopes": "c3ddacde8c01c87f7d498d5d9cfd67dc004ca58e8108fb867a01c86f8e9e701b",
            "_three_point_slopes": "cfa89b8d2ec79e18746810909ef27a60809e5398b774fd5893ca6bd2ba85e1e7",
            "_validate_nodes": "0f0381bbd0cdf671c966063f7cb8f32308afdfe7b2e628588ac49ab71918b36b",
        },
    ),
    "src/_lcm/egm/outer_refinement.py": (
        "97cee47efd93412ab61abde4067aaa8a845dfb0966e62325b4fa4f5b390fb2ad",
        {
            "_consider": "c16244935263abcccdc0e3a37988926e0a262d711eacc3474a91059949bb9bdf",
            "_node_local_max_mask": "5ae84637770adc944c9179c687a3a7bf73a9d81734bfb26f3aaa6830a204f764",
            "_tie_band": "3f8ae5c48db15b4b9d898d1fea588edd2f11265d2eccf382f5fae1d13e693398",
            "safeguarded_continuous_argmax": "0f150e28a5214cd8e434397bc7013c7f54a5d045cada661f08c254ec037ad8c7",
            "_fold_exact_outer_nodes": "429f4e770249f7a3f78c01835d98148cfc12aaca374a808589af88b03b09c99e",
        },
    ),
    "src/_lcm/optimization/golden_section.py": (
        "bb48f126d2b7aace897d5315ecb48884cfc6fdbded837464eb28eb0436425c4b",
        {
            "_probe": "b91f74ccf5b88728cee45aca7c3241aa86223ec6d6f0d2b966aecd8cc52ecb57",
            "_shrink_bracket": "acc085ec83351c0e62320b1e80f02880ce0e92fd0e442d8cee8096253788e8fd",
            "maximize_golden_section": "8eb2ebfd4b243e851c33991f5dd072fda24de72ea4df4ec22d4230ec7c2d3804",
        },
    ),
    "src/_lcm/egm/adaptive_nbegm_producer.py": (
        "d8913492e7076f1ff881475346bf80d045845978e41b53f9f3a05223190dbf3c",
        {
            "PreparedFloorArguments.__call__": "b0fd1bc602c7d4a3b7fde7a28141726eba2a260110e832a758d066117de4bc69",
            "_floor_template": "24c730e2e09ffe4220564314afe5f381921a2481de97d7019726418c0eebe9f2",
            "_tuple_roles": "812f02a6cb84cec95e97419c06d6bc05b5e7cdbea468c493bd061f5b61ce3b4c",
            "fold_executed_summaries": "6b5176437b9cc44915952d2fe1265dc9d6b8785377e6e8c0db6cca4d35e54541",
            "prepare_nested_inner_kernels": "0f38580b4b92bd65d96634a953f420660e1f91a8536277b672cd8d81b1838da6",
            "with_keeper_floor": "53fdc8f38eb8fe6110e99f4da4f395737927cb4cef3cce1ccc603633296bf87f",
        },
    ),
    "src/_lcm/egm/finite_negm_query.py": (
        "cf5a3b6cc19c3062ef3798673cfd0b83c25ce0dc41185ac5544b6f63e3143d66",
        {
            "_collapse_finite_selection": "4a25b2d177a60c0136423d0f56dc008e48996dcbeaf6eb4dcd142bf128ddaf50",
            "_read_finite_corner_row": "63cb79d2f5e1d7b1d421a14515809553a3a5c9f85056f5e72ead8c73116a045e",
            "_read_finite_native_row": "bc20b4a9e05e263942ef24731092891cc1f8ede145db021f1962678c23646466",
            "_read_finite_owned_side": "7dc3f1cb84892e41774e7750a3e5ee7f431cc673d92986c2564fb585b0ad9d2e",
            "_read_finite_resource_gradient": "c07ff486bf9f5bdc28df1a33f057294b9aca05a425e202df05709122bad57720",
            "_select_finite_outer_profiles": "02b082e8677fca9d708c59d90f71d06ba28ee7100a70a68229405913d863b7ce",
            "_single_row_finite_corner_read": "90dff3ecb15311291f506dd61a9cb2b0117895783051a956ccac754c94a23aaf",
            "_validate_finite_outer_rows": "ba36937dcfd9018283a7230d6b91fe7095199ea3151950ba182be6f7a7d7f683",
            "read_finite_outer_choices": "287fb5b6eff1253d2a03eab7bceff901f7cfe79e9ee47fdf9409ab3488239dcc",
            "_finite_outer_publication_inputs": "4f16e5723551105dc96055286c98508e979aded1a52376dc93188170dff35387",
        },
    ),
    "src/_lcm/egm/nbegm_query.py": (
        "455cd82205e65ab7a13b05e86d71c3297650ae7c7f0f0cbd8aee9cf214c6e92a",
        {
            "NBEGMQuery._read_at_floor_with_summary": "ffd64fc50411da1887764e95e38ce5cc2eb6d25a6459cd1e69f47c609052929b",
            "NBEGMQuery._read_rows": "87617e069f0002cf9bd6e26d90ea9b4ed3a5245d8785978fbb1a1b7802a57ced",
            "NBEGMQuery._read_rows_with_counts": "9c7a1469081ea86e3c4046de81c9e860aae477a43e130e53d9644b6c1c406126",
            "NBEGMQuery._read_rows_without_counts": "0c9f5360618501fdb69439f9d7838eeb21e9f23a2e49188aed3d8632db6fc5f1",
            "NBEGMQuery.capabilities": "97c1acf69c7dd500d9f263c861ecf832fe217552c85ae2923716fe48d32cdce7",
            "NBEGMQuery.evaluate_floor": "4a84b59e6b78cc10176e4808ec1d6945926b5e0ea89c6642e81d924a15da3ed8",
            "NBEGMQuery.input_names": "2032c5e43a132e2d9dfee5a6ea3892162467c556fc8d5d561a0b23325426e5d1",
            "NBEGMQuery.read_choices": "96735ed8b3c8edb3b7dc75da417551577904c6b8e15e554e678e7aff0745fac2",
            "NBEGMQuery.read_prepared": "44ca733a87f8cb034a2e314268fb57b8052172f26f90dab7d5d09d451294aa08",
            "NBEGMQuery.read_prepared_with_counts": "7cff52b4c1fa4ec1327e3a88fa239a31fbe0e8dd209df721dea6a0f6e2081c82",
            "_read_nb_compact_corner": "63f8b63a8c657026fde6cb1ec94a1d057004d8b4a219db7a1ca5cb15a9c4ee67",
            "_read_nb_compact_corner_rows": "f22123a2f8c35afe938b04dd42400db8a4638edd581df73350e9ee449cd4fd84",
            "build_nbegm_query": "b3c52f323e5f9baf94e4c7cd2ec94d1ea0ceda60d51d82cc71506dfce4bd1ad9",
            "_read_scalar_nbegm_owned_row": "0a2a5b687867fa590727d99ef191c6cd59e4ddd914735fa12e8b5cd6b37ab3fc",
            "_read_scalar_nbegm_owned_row_with_support": "b6055d487ced9637bc220ae8f3ab128e8d4e4db5d446c7ee2e1d7c3a134b9878",
            "_with_scalar_nbegm_linear_value": "36c09828d3d89215d8ab99664cfe1dfe8499fa6b3a3574fcd8abbed9f8936dcd",
            "_validate_nbegm_query_recipe": "702f8614bde2466b3f8ad45de4b8134841ffc8ae6a63aac0e7ab93a357964f6c",
        },
    ),
    "src/_lcm/egm/nbegm_query_model.py": (
        "6de8915be8d1f12b6a0889378087af53ac69c2da8222c68539aa140c3cc29e81",
        {},
    ),
    "src/_lcm/egm/nested_floor.py": (
        "ac153105f85ba73895865d2f48a7e544202537780a2c598a1cce639e998defa1",
        {
            "nested_floor_at_cell": "170487dba7605e71d8213b64cbf3b0045c421dd40668a2e3463c31cdfaa7d1a9",
            "prepare_nested_floor": "d22aba07033bb3c7105f1a076aed71bc98e07af461b7da5cc12570fdf2122063",
        },
    ),
    "src/_lcm/egm/outer_candidates.py": (
        "59eeebea9e9142ab041aaa881ac6c730d636e409d957175d8f49f506d244aa5b",
        {
            "OuterCandidateBank.candidate_carry": "4ceab52bbb8ccecc4109e8f333cb5fa2e6815fcb34af25599a1a907f22ee4742",
            "OuterCandidateBank.candidate_v_arr": "8cde27eebfe2e6d356bb27a1f8fb05b7a735c9b319ab07b14881045f5530cd6e",
            "OuterCandidateBank.n_candidates": "c27b1083798dd3b210d7ea9b291ca6ea28fd7fe75046522b7b76724630433872",
            "_stack_leaves": "536a70a587df88efa7d6d3917cb3f7406ab1c0a0da48cdc38bf0f184273eda09",
            "build_outer_candidate_bank": "d34f95a40c3e8ee16c57fd88355b55e586df4559a5b1bde26c5104b2eaae2feb",
        },
    ),
    "src/_lcm/solution/negm.py": (
        "8f0c755881d09fa20bbda82dac97a60304eabc74df0d051030284e76e8657faf",
        {
            "_stack_owned_carry_template": "8ec66515c8b2a988e7541729eb37ac901b9204a8bc841c552c3265b04bebb4d6",
            "_broadcast_over_candidates": "15325b555d9e603b155f372a04471ceff959d7ab4035b6d9ab6f9bdbc940ac66",
            "NEGM.build_period_kernels": "953d4bd85c38fe78a5dc5d59aaa4fc9ae7065997354408869be6222f2b547425",
            "_NEGMPeriodKernel.__call__": "abda379507e50390a0a06e745ce6ebf96cc47f3d34dc92d11fd48cf10ffcbff6",
            "_NEGMPeriodKernel.__post_init__": "e5241f2618a6ee8cc775333d0a2eee41263f4f0a6ddb063f70d2a92339beab17",
            "_NEGMSweepArgumentBuilder.__call__": "b42523d6ec3756e19037bc931a86eb8f3df530d048014f4534e7a916e22acaaf",
            "_NestedNodeSolver.__call__": "2a3d448dc29ed1dd80aec464c435629dea7931225229b8f00c4878837c147174",
            "_build_negm_floor_preparation": "e1c55bbb5a99c38821725f6835257b4906cd73da367ff7ca36ce5e4866475c24",
            "_count_nested_current_feasibility": "ab797be77f5ef497ad590a99a407bf59ebeb7510b712992eecfea0dac5a64d23",
            "_derive_negm_inner_query_model": "ed61d9953c002b927ed1ebe659d92329369f02306f4c89a9a4aa490cd8f846f0",
            "_durable_values_at": "08077d4f948e51b0fbc2dbbc97d160c804c504b4460cb576a49a3e251449cafc",
            "_keeper_with_prepared_floor": "87ced247c2c491d7f4ae5f7ad47976ec8ae72331b760c1bbb955c4f650465f57",
            "_nested_feasibility_programs": "5f049652465883a607bd8d5f9c3cb4659dcd01bec0729bed08f0fcb4c8ce203f",
            "_nested_outer_sweep_program": "6a15e39ad851336460a6fc4b7300ffa707c54d0cc63ca35e589c09da881b09d9",
            "_nested_tuple_roles": "9b1f6c07c555ec4997f5ead3e834edd5681bab11ec7ee8d7a68c323d82a64c5a",
            "_periodized_function_groups": "13f6d1c80876f571be8dee37eac93b5303a5e39a15d9668f7cc4d46469b17dc2",
            "_run_nested_feasibility_kernel": "012b42d1f1f9bf281c3bbd8459f61d29cc98f2f0a70f75b0b1156ad804bb1af2",
            "_stacked_nested_carry_role": "5c34c027e38b806a8af80ef86c92bb72e1d9937ee43238764f4debfb0b501756",
            "_stacked_nested_carry_roles": "66a0b6530215123b306e07f7e9972b4d89abd6c750aa6bb28973972fa038899e",
            "_with_declared_keeper_reads": "19da2013f8d4d0933ad8b37cde27283ce4979bb2979d8979d583cf244e9f2d56",
        },
    ),
    "src/_lcm/solution/nnbegm.py": (
        "1098d32cf4b1c49742cd1bfd2f6dd08d0dff9e7f2f66970833d1bbf0910af090",
        {
            "NNBEGM.build_period_kernels": "d22aca7362f6648b7b101dfe14939709a9c5fded5b2cbdaf795eaa904f3409bb",
            "_AdaptiveNNBEGMPeriodKernel._solve_outer": "3201a6468622200d903998c56c7d83c638331a924338b609346f0d44f994bdbe",
            "_AdaptiveNodeSolver.__call__": "be0fd84de6d3bbf185d501c039fbf6720c85250e2b4b3f466ef749dd85e2cdc0",
            "_FiniteNNBEGMPeriodKernel._solve_outer": "917de456b989a4446bca20f79fd8963c290a9041d5f1551c5ba35ac909d6b5c0",
            "_NNBEGMPeriodKernel.__call__": "d4d0d638498e9fcfff852aa63742e0528e2e66548a4273a52834d5e3ddf7cc9a",
            "_NNBEGMPeriodKernel.__post_init__": "402ce22b33e568cf315a036a1a17f8d03d6a49cbf500dc629fd75ee7c128c955",
            "_NNBEGMPeriodKernel._solve_adjuster_node": "1378157bf1f0ac0348c9b77fa844a0e12f14479606724f22a9725bc818be4918",
            "_NNBEGMPeriodKernel._solve_keeper": "4ca3c8ab0f5658f1a2bf4fa318ffb9e51c705ec3096b492344aad8d2b60cdd35",
            "_NestedArgumentBuilder.__call__": "2487c8638632a8affb421ae4fd0726d153566a7c47355a1f6045d86dfa82e5c9",
            "_conditional_nnbegm_banks": "b5a4840292931ef0cb9fbd85c2fa52dd479ba64ba9f128660ab09e287d48d455",
            "_role_carry_reads": "6059679259758a56da264bcf7646ec89bc1f4b4c1a17af3b431c21a92284eca0",
            "_nnbegm_outer_replay_function": "2587eccd6afb167826e2810aca1e6636e755ef745b52264a7131dc766787344d",
            "_nnbegm_query_period_groups": "8dc56aeebf49b108bc73589d48fb77ad80bba30f96e9b9b531f7b2174a66bdfd",
            "_prepare_nnbegm_query_inner_pair": "36f77678b83be27ad0220c001cb0d9184d4a838e0d0a1eb71930d08140e7be9c",
        },
    ),
    "src/_lcm/egm/upper_envelope/__init__.py": (
        "7291f0c3effde40f47086175da5f5e6103e3791fea4ceee133ad8a22e9ccb2e2",
        {
            "_exact_backend": "3c59c8dd5a4810413c5782cec73a9d85bf90607cc1f687357f4c81593e3e4628",
        },
    ),
    "src/_lcm/egm/upper_envelope/segment_envelope.py": (
        "f15b6dd5fee38ce6d2b42f558dbad44c020b29dfdf2ec300673ae11707fff598",
        {
            "refine_envelope_exact": "f38c9eec39f60b607f5c98bfee1ac196d952b84baa85ef2b35ec6d31d6c4cbc3",
        },
    ),
}

_SELECTED_PINS: set[tuple[str, str]] = set()


type SourceText = str
type SourcePath = str


class DirectFlowReport(TypedDict):
    ok: bool
    result: str
    errors: list[str]
    offending_paths: list[str]
    routes: dict[str, str]
    certified_corridor_sources: list[SourcePath]
    source_seals: dict[str, str]


class MutationCase(TypedDict):
    path: str
    rejected: bool
    errors: list[str]
    offending_paths: list[str]


class DirectMutationControls(TypedDict):
    current_iid_mutations: dict[str, MutationCase]
    current_iid_names_match_expected: bool
    admitted_current_iid_mutations: list[str]
    clean: DirectFlowReport
    mutations: dict[str, MutationCase]
    mutation_count: int
    expected_mutation_count: int
    mutation_count_matches_expected: bool
    mutation_names_sha256: str
    expected_mutation_names_sha256: str
    mutation_names_match_expected: bool
    admitted_mutations: list[str]
    action_partition_mutations: dict[str, MutationCase]
    action_partition_mutation_count: int
    action_partition_names_match_expected: bool
    admitted_action_partition_mutations: list[str]
    normal_process_mutations: dict[str, MutationCase]
    normal_process_mutation_count: int
    normal_process_names_match_expected: bool
    admitted_normal_process_mutations: list[str]
    allocation_reservation_mutations: dict[str, MutationCase]
    allocation_reservation_mutation_count: int
    allocation_reservation_names_match_expected: bool
    admitted_allocation_reservation_mutations: list[str]
    feasibility_mutations: dict[str, MutationCase]
    feasibility_mutation_count: int
    feasibility_names_match_expected: bool
    admitted_feasibility_mutations: list[str]
    grouped_guard_mutations: dict[str, MutationCase]
    grouped_guard_mutation_count: int
    grouped_guard_names_match_expected: bool
    admitted_grouped_guard_mutations: list[str]
    grouped_mapper_mutations: dict[str, MutationCase]
    grouped_mapper_mutation_count: int
    grouped_mapper_names_match_expected: bool
    admitted_grouped_mapper_mutations: list[str]
    action_grid_mutations: dict[str, MutationCase]
    action_grid_mutation_count: int
    action_grid_names_match_expected: bool
    admitted_action_grid_mutations: list[str]
    uniform_process_mutations: dict[str, MutationCase]
    uniform_process_mutation_count: int
    uniform_process_names_match_expected: bool
    admitted_uniform_process_mutations: list[str]
    supplemental_mutations: dict[str, MutationCase]
    supplemental_mutation_count: int
    supplemental_names_match_expected: bool
    admitted_supplemental_mutations: list[str]
    s03_scalar_mutations: dict[str, MutationCase]
    s03_scalar_mutation_count: int
    s03_scalar_names_match_expected: bool
    admitted_s03_scalar_mutations: list[str]
    owned_query_helper_mutations: dict[str, MutationCase]
    owned_query_helper_mutation_count: int
    owned_query_helper_names_match_expected: bool
    admitted_owned_query_helper_mutations: list[str]
    associated_saving_mutations: dict[str, MutationCase]
    associated_saving_mutation_count: int
    associated_saving_names_match_expected: bool
    admitted_associated_saving_mutations: list[str]
    child_failure_mutations: dict[str, MutationCase]
    child_failure_mutation_count: int
    child_failure_names_match_expected: bool
    admitted_child_failure_mutations: list[str]
    reviewed_prefix_mutations: dict[str, MutationCase]
    reviewed_prefix_mutation_count: int
    reviewed_prefix_names_match_expected: bool
    admitted_reviewed_prefix_mutations: list[str]
    decoupled_bound_mutations: dict[str, MutationCase]
    decoupled_bound_mutation_count: int
    decoupled_bound_names_match_expected: bool
    admitted_decoupled_bound_mutations: list[str]
    retained_prefix_mutations: dict[str, MutationCase]
    retained_prefix_mutation_count: int
    retained_prefix_names_match_expected: bool
    admitted_retained_prefix_mutations: list[str]
    final_nested_mutations: dict[str, MutationCase]
    final_nested_mutation_count: int
    final_nested_names_match_expected: bool
    admitted_final_nested_mutations: list[str]
    outer_helper_mutations: dict[str, MutationCase]
    outer_helper_mutation_count: int
    outer_helper_names_match_expected: bool
    admitted_outer_helper_mutations: list[str]
    child_native_mutations: dict[str, MutationCase]
    child_native_mutation_count: int
    child_native_names_match_expected: bool
    admitted_child_native_mutations: list[str]
    additional_helper_mutations: dict[str, MutationCase]
    additional_helper_mutation_count: int
    additional_helper_names_match_expected: bool
    admitted_additional_helper_mutations: list[str]
    all_rejected: bool


def _surface_pin(source: SourcePath) -> str:
    """Select one certified source's pinned module transport surface."""
    surface = _CORRIDOR_PINS[source][0]
    if surface is None:
        raise ValueError(f"no module transport surface is pinned for {source}")
    _SELECTED_PINS.add((source, source))
    return surface


def _callable_pins(*, source: SourcePath, names: tuple[str, ...]) -> dict[str, str]:
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


_OUTER_HELPER_CONTRACTS = _contracts(_OUTER_HELPER_OWNER_SELECTION)

_ADDITIONAL_HELPER_CONTRACTS = _contracts(_ADDITIONAL_HELPER_OWNER_SELECTION)
_EXPECTED_ADDITIONAL_HELPER_MUTATION_NAMES = frozenset(
    (
        "additional_helper:body:src/_lcm/egm/continuation.py::_build_feature_child_query",
        "additional_helper:body:src/_lcm/egm/continuation.py::_select_feature_child_rows",
        "additional_helper:body:src/_lcm/egm/finite_negm_query.py::_finite_outer_publication_inputs",
        "additional_helper:body:src/_lcm/egm/nbegm_query.py::_read_scalar_nbegm_owned_row",
        "additional_helper:body:src/_lcm/egm/nbegm_query.py::_read_scalar_nbegm_owned_row_with_support",
        "additional_helper:body:src/_lcm/egm/nbegm_query.py::_validate_nbegm_query_recipe",
        "additional_helper:body:src/_lcm/egm/nbegm_query.py::_with_scalar_nbegm_linear_value",
        "additional_helper:body:src/_lcm/egm/nbegm_step.py::_prepare_nbegm_interval_block",
        "additional_helper:body:src/_lcm/egm/outer_refinement.py::_fold_exact_outer_nodes",
        "additional_helper:body:src/_lcm/egm/step_core.py::_assemble_combo_candidates",
        "additional_helper:body:src/_lcm/egm/step_core.py::_mask_dead_combo_rows",
        "additional_helper:body:src/_lcm/execution/internal_outputs.py::_leaf_signature",
        "additional_helper:body:src/_lcm/execution/internal_outputs.py::_published_signature",
        "additional_helper:body:src/_lcm/execution/internal_outputs.py::_tree_signature",
        "additional_helper:body:src/_lcm/execution/internal_outputs.py::assert_internal_inputs",
        "additional_helper:body:src/_lcm/model_processing.py::_validate_callable_borrowing_solver",
        "additional_helper:body:src/_lcm/regime_building/processing.py::_nbegm_evaluated_query_predicates",
        "additional_helper:body:src/_lcm/regime_building/processing.py::_nbegm_query_evaluator_targets",
        "additional_helper:body:src/_lcm/regime_building/processing.py::_nbegm_query_function_pools",
        "additional_helper:body:src/_lcm/regime_building/processing.py::_nbegm_query_row_state_names",
        "additional_helper:body:src/_lcm/regime_building/processing.py::_solver_with_query_authority",
        "additional_helper:body:src/_lcm/solution/dcegm.py::_dcegm_query_slots_and_capabilities",
        "additional_helper:body:src/_lcm/solution/egm.py::_unpack_scalar_child_read",
        "additional_helper:body:src/_lcm/solution/model_authority.py::_validate_query_policy_budget_authority",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_NBEGMActionEligibility.__call__",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_NBEGMScalarChildReads.__call__",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_NBEGMScalarChildReads.read_rows",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_NBEGMTabulatedEulerLaw.__call__",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_RideAlongNBEGMPeriodKernel._consume_output",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_bounded_nbegm_kernel_output",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_fold_nbegm_scalar_read_records",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_nbegm_cell_continuation_columns",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_prepare_nbegm_scalar_child_reads",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_record_nbegm_scalar_child_slots",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_scalar_conditional_nbegm_carry",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_validate_nbegm_integrity_slot_counts",
        "additional_helper:body:src/_lcm/solution/nbegm.py::_with_scalar_nbegm_sampled_run",
        "additional_helper:body:src/_lcm/solution/nnbegm.py::_nnbegm_outer_replay_function",
        "additional_helper:body:src/_lcm/solution/nnbegm.py::_nnbegm_query_period_groups",
        "additional_helper:body:src/_lcm/solution/nnbegm.py::_prepare_nnbegm_query_inner_pair",
        "additional_helper:semantic:invalid_numeric_laundered",
        "additional_helper:semantic:pad_bridge",
        "additional_helper:semantic:paired_marginal_changed",
        "additional_helper:semantic:same_owner_false_link_bridge",
        "additional_helper:semantic:schedule_guard_removed",
        "additional_helper:surface:src/_lcm/egm/nbegm_query.py",
        "additional_helper:surface:src/_lcm/execution/internal_outputs.py",
        "additional_helper:surface:src/_lcm/solution/nbegm.py",
    )
)


def _additional_helper_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Authenticate exact helper bodies and the declared three surface controls."""
    surface, callables = _ADDITIONAL_HELPER_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="additional helper dependency", contracts=callables
    )
    if (
        source in _ADDITIONAL_HELPER_SURFACE_SELECTION
        and _transport_module_surface(tree) != surface
    ):
        errors.append(
            f"additional helper dependency: {source} declaration surface changed"
        )
    return errors


def additional_helper_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Construct the declared forty-six-body, three-surface and five-attack family."""
    return _additional_helper_control_specs(
        repo_root=repo_root, checker=sys.modules[__name__]
    )


def _additional_helper_case_rejected(
    *,
    clean: DirectFlowReport,
    result: DirectFlowReport,
    source: SourcePath,
    diagnostic: str,
) -> bool:
    """Require clean input and every precise new or co-owned declaration refusal."""
    return (
        clean["ok"]
        and clean["result"] == "pass"
        and clean["errors"] == []
        and clean["offending_paths"] == []
        and not result["ok"]
        and result["result"] == "fail"
        and tuple(sorted(result["errors"]))
        == _additional_helper_expected_errors(source=source, diagnostic=diagnostic)
        and result["offending_paths"] == [source]
    )


_CHILD_NATIVE_CONTRACTS = _contracts(_CHILD_NATIVE_OWNER_SELECTION)
_EXPECTED_CHILD_NATIVE_MUTATION_NAMES = frozenset(
    (
        "child_native:body:src/_lcm/egm/upper_envelope/__init__.py::_exact_backend",
        "child_native:body:src/_lcm/egm/upper_envelope/segment_envelope.py::refine_envelope_exact",
        "child_native:overload:changed_stub",
        "child_native:overload:extra_function_binding",
        "child_native:overload:extra_stub",
        "child_native:overload:missing_stub",
        "child_native:overload:overload_alias",
        "child_native:overload:overload_import_removed",
        "child_native:overload:overload_rebinding",
        "child_native:overload:reordered_stubs",
        "child_native:overload:runtime_decorator",
        "child_native:surface:src/_lcm/egm/upper_envelope/__init__.py",
        "child_native:surface:src/_lcm/egm/upper_envelope/segment_envelope.py",
    )
)


def _child_native_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Authenticate native composition bodies and their complete declarations."""
    surface, callables = _CHILD_NATIVE_CONTRACTS[source]
    ordinary = {
        name: digest
        for name, digest in callables.items()
        if name != "refine_envelope_exact"
    }
    errors = _exact_callable_errors(
        tree=tree, label="child native dependency", contracts=ordinary
    )
    if "refine_envelope_exact" in callables:
        try:
            implementation = _select_refine_envelope_exact(
                tree=tree, checker=sys.modules[__name__]
            )
        except ValueError:
            errors.append(
                "child native dependency: refine_envelope_exact overload declarations changed"
            )
        else:
            if (
                _callable_ast_sha256(implementation)
                != callables["refine_envelope_exact"]
            ):
                errors.append(
                    "child native dependency: exact callable corridor 'refine_envelope_exact' changed"
                )
    if _transport_module_surface(tree) != surface:
        errors.append(f"child native dependency: {source} declaration surface changed")
    return errors


def child_native_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Construct the distinct two-body, two-surface and nine-declaration family."""
    return _child_native_control_specs(
        repo_root=repo_root, checker=sys.modules[__name__]
    )


def _child_native_case_rejected(
    *,
    clean: DirectFlowReport,
    result: DirectFlowReport,
    source: SourcePath,
    diagnostic: str,
) -> bool:
    """Require a clean baseline and exactly the native owner's complete refusal."""
    return (
        clean["ok"]
        and clean["result"] == "pass"
        and clean["errors"] == []
        and clean["offending_paths"] == []
        and not result["ok"]
        and result["result"] == "fail"
        and tuple(sorted(result["errors"]))
        == _child_native_expected_errors(source=source, diagnostic=diagnostic)
        and result["offending_paths"] == [source]
    )


_FINAL_NESTED_CONTRACTS = _contracts(
    {
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
            "_stack_owned_carry_template",
            "_broadcast_over_candidates",
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
)


def _final_nested_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Check each nested helper body and its complete declaration surface."""
    surface, callables = _FINAL_NESTED_CONTRACTS[source]
    overloaded = "NBEGMQuery.read_choices"
    ordinary = {
        name: digest
        for name, digest in callables.items()
        if source != "src/_lcm/egm/nbegm_query.py" or name != overloaded
    }
    errors = _exact_callable_errors(
        tree=tree, label="final nested dependency", contracts=ordinary
    )
    if source == "src/_lcm/egm/nbegm_query.py":
        try:
            implementation = _select_nbegm_query_read_choices(
                tree=tree, checker=sys.modules[__name__]
            )
            if _callable_ast_sha256(implementation) != callables[overloaded]:
                errors.append(
                    "final nested dependency: exact callable corridor "
                    "'NBEGMQuery.read_choices' changed"
                )
        except ValueError as error:
            errors.append(f"final nested dependency: {error}")
    if _transport_module_surface(tree) != surface:
        errors.append(f"final nested dependency: {source} declaration surface changed")
    return errors


EXPECTED_FINAL_NESTED_MUTATION_COUNT = 103
EXPECTED_FINAL_NESTED_MUTATION_NAMES_SHA256 = (
    "02728053a901a11ee91c4c8232a76fee558a45b5eee06da3aabe5f5c47887359"
)


def final_nested_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Keep all nested controls in a distinct canonically executed population."""
    return _final_nested_control_specs(
        repo_root=repo_root, checker=sys.modules[__name__]
    )


def _final_nested_case_rejected(
    *,
    clean: DirectFlowReport,
    result: DirectFlowReport,
    source: SourcePath,
    diagnostic: str,
) -> bool:
    """Require clean input and the exact new/co-owned error set on one source."""
    return (
        clean["ok"]
        and clean["result"] == "pass"
        and clean["errors"] == []
        and clean["offending_paths"] == []
        and not result["ok"]
        and result["result"] == "fail"
        and tuple(sorted(result["errors"]))
        == _final_nested_expected_errors(source=source, diagnostic=diagnostic)
        and result["offending_paths"] == [source]
    )


_EXPECTED_OUTER_HELPER_MUTATION_NAMES = frozenset(
    {
        "outer_helpers:body:interpolation:evaluate",
        "outer_helpers:body:interpolation:evaluate_derivative",
        "outer_helpers:body:interpolation:evaluate_validity",
        "outer_helpers:body:interpolation:evaluate_core",
        "outer_helpers:body:interpolation:pad_axes",
        "outer_helpers:body:interpolation:validate_nodes",
        "outer_helpers:body:interpolation:node_slopes",
        "outer_helpers:body:interpolation:four_point",
        "outer_helpers:body:interpolation:three_point",
        "outer_helpers:body:interpolation:one_sided",
        "outer_helpers:body:interpolation:prefix_node",
        "outer_helpers:body:interpolation:prefix_three",
        "outer_helpers:body:interpolation:prefix_four",
        "outer_helpers:body:search:safeguarded",
        "outer_helpers:body:search:local_max",
        "outer_helpers:body:search:consider",
        "outer_helpers:body:search:tie_band",
        "outer_helpers:body:golden:maximize",
        "outer_helpers:body:golden:probe",
        "outer_helpers:body:golden:shrink",
        "outer_helpers:prefix:capacity_domain",
        "outer_helpers:prefix:ordinary_slopes",
        "outer_helpers:prefix:cubic_early",
        "outer_helpers:prefix:nested_without_cubic",
        "outer_helpers:prefix:secant_reads_padding",
        "outer_helpers:prefix:endpoint_capacity",
        "outer_helpers:prefix:cubic_unclamped",
        "outer_helpers:search:pads_compete",
        "outer_helpers:search:bracket_capacity",
        "outer_helpers:search:objective_not_pinned",
        "outer_helpers:search:exact_nodes_omitted",
        "outer_helpers:search:duplicate_runner_up",
        "outer_helpers:golden:nan_not_sanitized",
        "outer_helpers:golden:interior_tie_right",
        "outer_helpers:golden:cached_value_reprobed",
        "outer_helpers:golden:upper_endpoint_omitted",
        "outer_helpers:golden:loop_callback_replaced",
        "outer_helpers:surface:golden_import_redirected",
        "outer_helpers:surface:prefix_minimum_changed",
        "outer_helpers:surface:golden_result_reordered",
    }
)


def _outer_helper_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Check the explicit primitive bodies beside their exact binding surface."""
    surface, callables = _OUTER_HELPER_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="outer helper dependency", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(f"outer helper dependency: {source} declaration surface changed")
    return errors


def outer_helper_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Keep all forty primitive body/binding controls in their own population."""
    return _outer_helper_control_specs(repo_root=repo_root)


def _outer_helper_case_rejected(
    *,
    clean: DirectFlowReport,
    result: DirectFlowReport,
    source: SourcePath,
    diagnostic: str,
) -> bool:
    """Require the exact non-seal owner error and path after a clean positive."""
    return (
        clean["ok"]
        and clean["result"] == "pass"
        and clean["errors"] == []
        and clean["offending_paths"] == []
        and not result["ok"]
        and result["result"] == "fail"
        and result["errors"] == [diagnostic]
        and result["offending_paths"] == [source]
    )


def canonical_json[PayloadValue](payload: Mapping[str, PayloadValue]) -> str:
    """Render deterministic JSON for command-line controls."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _mutation_name_digest(names: Sequence[str]) -> str:
    """Hash the exact sorted mutation-name family, including its cardinality."""
    payload = ("\n".join(sorted(names)) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _name(*, node: ast.AST | None, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _is_bounded_result_name(node: ast.AST | None) -> bool:
    """Recognize the exact nominal carrier, including explicit generic arguments."""
    return _name(node=node, expected="BoundedResult") or (
        isinstance(node, ast.Subscript)
        and _name(node=node.value, expected="BoundedResult")
    )


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Subscript) and _is_bounded_result_name(call.func):
        return "BoundedResult"
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
    if name == "_read_one_row_dc_child":
        return select_checked_dc_runtime(
            tree=tree, qualname=name, checker=sys.modules[__name__]
        )
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


def _expected_statements(source: SourceText) -> list[ast.stmt]:
    """Parse an allowlisted statement sequence under the running Python AST."""
    return ast.parse(source).body


def _body_matches(*, node: ast.FunctionDef, expected_source: SourceText) -> bool:
    observed = [_ast_key(item) for item in _body_without_docstring(node)]
    expected = [_ast_key(item) for item in _expected_statements(expected_source)]
    return observed == expected


def _expression_matches(*, node: ast.AST | None, source: SourceText) -> bool:
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
        annotation_source = (
            '"tuple[FloatND, BoolND]" if return_feasibility else "FloatND"'
        )
    else:
        annotation_source = (
            '"tuple[FloatND, BoolND]" '
            'if stakeholders is not None or return_feasibility else "FloatND"'
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


def _statements_match(
    *, observed: Sequence[ast.stmt], expected_source: SourceText
) -> bool:
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
    if class_name == "DCEGMQuery" and method_name == "read_choices":
        return cls, select_checked_dc_runtime(
            tree=tree, qualname="DCEGMQuery.read_choices", checker=sys.modules[__name__]
        )
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
    """Pin the live simulation caller to the params-completed base without wrapping.

    The first binding is a statement of the function body; a later one rebuilds
    the space from the same adapter after a dissolution route moved rows to a
    projected state in the same regime. Every binding is that exact call.
    """
    try:
        node = _definition(tree=tree, name="_simulate_regime_in_period")
    except ValueError as error:
        return [f"simulation state-action caller: {error}"]
    body = _body_without_docstring(node)
    bindings = _scope_binding_counts(body)
    top_level = [
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and _target_names(statement.targets[0]) == ("state_action_space",)
        and isinstance(statement.value, ast.Call)
    ]
    matches = [
        statement
        for statement in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and _target_names(statement.targets[0]) == ("state_action_space",)
    ]
    if len(top_level) != 1 or bindings.get("state_action_space") != len(matches):
        return [
            (
                "simulation state-action caller: expected exactly one live "
                "state_action_space binding"
            )
        ]
    for statement in matches:
        call = statement.value
        if not isinstance(call, ast.Call):
            return ["simulation state-action caller: adapter call changed"]
        if not (
            _call_name(call) == "create_regime_state_action_space"
            and not call.args
            and {item.arg for item in call.keywords}
            == {"regime", "regime_states", "base"}
            and _name(node=_keyword(call=call, name="regime"), expected="regime")
            and _expression_matches(
                node=_keyword(call=call, name="regime_states"),
                source="states[regime_name]",
            )
            and _name(
                node=_keyword(call=call, name="base"),
                expected="base_state_action_space",
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
        return _statements_match(
            observed=[statement],
            expected_source="return jnp.where(_dissolution, -1, argmax_flat), V_stacked",
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


def _candidate_value_projection(tree: ast.Module) -> ast.Module:
    """Expose ordinary Q/F identity beneath independently pinned status transport.

    Only exact status wrappers are erased. The source tree's callable and surface
    pins are checked separately; candidate-changing expressions remain visible to
    the route proof and its existing mutation corpus.
    """
    result = copy.deepcopy(tree)
    # This exact adapter is projected only when its independent guard proof holds.
    # A changed/missing guard remains visible to the old candidate-array proof.
    if not _current_iid_builder_guard_errors(tree):
        definition = _definition(tree=result, name="get_max_Q_over_a")
        definition.body = [
            statement
            for statement in definition.body
            if not _statements_match(
                observed=[statement],
                expected_source="if current_iid_plan is not None:\n    max_Q_over_a = current_iid_plan.wrap(max_Q_over_a)",
            )
        ]
        slots = [
            (argument, default)
            for argument, default in zip(
                definition.args.kwonlyargs, definition.args.kw_defaults, strict=True
            )
            if argument.arg != "current_iid_plan"
        ]
        definition.args.kwonlyargs = [argument for argument, _ in slots]
        definition.args.kw_defaults = [default for _, default in slots]
    builders = {"get_max_Q_over_a", "get_argmax_and_max_Q_over_a"}
    reducers = {
        "_HardMaxQOverA",
        "_HardMaxArgmaxQOverA",
        "_SmoothedMaxQOverA",
        "_TasteShockArgmaxQOverA",
        "_FoldedMaxQOverA",
    }

    def project(statements: list[ast.stmt]) -> list[ast.stmt]:
        projected: list[ast.stmt] = []
        for original_statement in statements:
            statement = original_statement
            if (
                isinstance(statement, ast.AnnAssign)
                and _name(node=statement.target, expected="output")
                and statement.value is not None
            ):
                statement = ast.Assign(
                    targets=[statement.target], value=statement.value
                )
            if isinstance(statement, ast.Assign) and isinstance(
                statement.value, ast.Call
            ):
                call = statement.value
                if (
                    _target_names(statement.targets[0]) == ("Q_arr", "F_arr")
                    and _call_name(call) == "cast"
                    and len(call.args) == 2
                    and not call.keywords
                    and _unparse(call.args[0]) == "'tuple[FloatND, BoolND]'"
                ):
                    statement.value = call.args[1]
            if isinstance(statement, ast.If):
                statement.body = project(statement.body)
                statement.orelse = project(statement.orelse)
            if isinstance(statement, ast.Assign) and projected:
                previous = projected[-1]
                if (
                    _target_names(statement.targets[0]) == ("Q_arr", "F_arr")
                    and _expression_matches(
                        node=statement.value,
                        source="evaluated.output if type(evaluated) is BoundedResult else evaluated",
                    )
                    and isinstance(previous, ast.Assign)
                    and _target_names(previous.targets[0]) == ("evaluated",)
                ):
                    projected[-1] = ast.Assign(
                        targets=statement.targets, value=previous.value
                    )
                    continue
            if isinstance(statement, ast.Return) and isinstance(
                statement.value, ast.Call
            ):
                call = statement.value
                if (
                    _call_name(call) == "_with_bound_summary"
                    and not call.args
                    and {keyword.arg for keyword in call.keywords}
                    == {"evaluated", "output"}
                    and _unparse(_keyword(call=call, name="evaluated"))
                    in {"evaluated", "out"}
                ):
                    output = _keyword(call=call, name="output")
                    if (
                        isinstance(output, ast.Call)
                        and _call_name(output) == "cast"
                        and len(output.args) == 2
                        and not output.keywords
                        and _unparse(output.args[0]) == "'MaxOutput'"
                    ):
                        output = output.args[1]
                    if _name(node=output, expected="output") and projected:
                        previous = projected[-1]
                        if isinstance(previous, ast.Assign) and _target_names(
                            previous.targets[0]
                        ) == ("output",):
                            output = previous.value
                            projected.pop()
                    statement.value = output
            projected.append(statement)
        return projected

    for definition in result.body:
        if isinstance(definition, ast.FunctionDef) and definition.name in builders:
            definition.body = [
                statement
                for statement in definition.body
                if not _statements_match(
                    observed=[statement],
                    expected_source="action_kernel: ActionFunction = BoundedQAndF(function=Q_and_F) if bound_summary else Q_and_F",
                )
            ]
            for node in ast.walk(definition):
                if isinstance(node, ast.Name) and node.id in {
                    "mapped_action_kernel",
                    "action_kernel",
                }:
                    node.id = "Q_and_F"
                if (
                    isinstance(node, ast.AnnAssign)
                    and _name(node=node.target, expected="max_Q_over_a")
                    and _unparse(node.annotation) == "Callable[..., BoundedMaxOutput]"
                ):
                    node.annotation = ast.parse(
                        "Callable[..., FloatND | tuple[FloatND, BoolND]]", mode="eval"
                    ).body
                if (
                    isinstance(node, ast.AnnAssign)
                    and _name(node=node.target, expected="argmax_and_max_Q_over_a")
                    and _unparse(node.annotation)
                    == "ArgmaxQOverAFunction[BoundedArgmaxOutput]"
                ):
                    node.annotation = ast.Name(
                        id="ArgmaxQOverAFunction", ctx=ast.Load()
                    )
                if (
                    isinstance(node, ast.Call)
                    and _call_name(node) == "cast"
                    and len(node.args) == 2
                ):
                    aliases = {
                        "Callable[..., BoundedMaxOutput]": "Callable[..., FloatND | tuple[FloatND, BoolND]]",
                        "MaxQOverAFunction[BoundedMaxOutput]": "MaxQOverAFunction",
                    }
                    if (
                        isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and node.args[0].value in aliases
                    ):
                        node.args[0].value = aliases[node.args[0].value]
            definition.body = [
                statement
                for statement in definition.body
                if not _statements_match(
                    observed=[statement],
                    expected_source="bound_summary = bool(has_action_integrity(function=Q_and_F))",
                )
                and not _statements_match(
                    observed=[statement],
                    expected_source="if bound_summary:\n    Q_and_F = BoundedQAndF(function=Q_and_F)",
                )
            ]
            for call in ast.walk(definition):
                if isinstance(call, ast.Call) and _call_name(call) in {
                    "productmap",
                    "tiled_productmap",
                    "vmap_1d",
                    "simulation_spacemap",
                }:
                    call.keywords = [
                        keyword
                        for keyword in call.keywords
                        if not (
                            keyword.arg == "bound_summary"
                            and _name(node=keyword.value, expected="bound_summary")
                        )
                    ]
        if isinstance(definition, ast.ClassDef) and definition.name in reducers:
            for method in definition.body:
                if isinstance(method, ast.FunctionDef) and method.name == "__call__":
                    method.body = project(method.body)
                    if definition.name == "_FoldedMaxQOverA":
                        method.body = [
                            statement
                            for statement in method.body
                            if not _statements_match(
                                observed=[statement],
                                expected_source="ordinary = out.output if type(out) is BoundedResult else out",
                            )
                        ]
                        for node in ast.walk(method):
                            if isinstance(node, ast.Name) and node.id == "ordinary":
                                node.id = "out"
                    if isinstance(
                        method.returns, ast.BinOp
                    ) and _is_bounded_result_name(method.returns.right):
                        method.returns = method.returns.left
    return ast.fix_missing_locations(result)


def _productmap_binding_errors(
    *, tree: ast.Module, outer_name: str, nested_name: str
) -> list[str]:
    """Require one unwrapped, unbatched action product and no later rebinding."""
    tree = _candidate_value_projection(tree)
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
    tree = _candidate_value_projection(tree)
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
        mapped=cast("Callable[..., FloatND | tuple[FloatND, BoolND]]", mapped),
        fold_state_names=fold_state_names,
        fold_weights=fold_weights,
        fold_conditioning=fold_conditioning,
        inner_state_names=inner_state_names,
        return_feasibility=return_feasibility,
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
                "return_feasibility",
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
                "False",
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
    errors = _exact_callable_errors(
        tree=tree,
        label="streamed max-Q builder",
        contracts=_STREAMED_MAX_BUILDER_PINS,
    )
    try:
        _, folded = _method_definition(
            tree=_candidate_value_projection(tree),
            class_name="_FoldedMaxQOverA",
            method_name="__call__",
        )
    except ValueError as error:
        return [*errors, f"streamed max-Q builder: {error}"]
    if not _body_matches(
        node=folded,
        expected_source="""out = self.mapped(
    next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params
)
if self.return_feasibility:
    V_arr, empty = cast("tuple[FloatND, BoolND]", out)
else:
    V_arr = cast("FloatND", out)
    empty = None
for name, axis, conditioning_axis in self.fold_steps:
    weights = self.fold_weights[name]
    if conditioning_axis is not None:
        weights = _broadcast_conditioned_rows(
            rows=weights,
            conditioning_axis=conditioning_axis,
            fold_axis=axis,
            shape=V_arr.shape,
        )
    if empty is not None:
        if weights.ndim == 1:
            weight_shape = [1] * V_arr.ndim
            weight_shape[axis] = weights.shape[0]
            support = weights.reshape(weight_shape) > 0
        else:
            support = weights > 0
        empty = jnp.any(empty & support, axis=axis)
    V_arr = self.fold_reducers[name](a=V_arr, axis=axis, weights=weights)
return V_arr if empty is None else (V_arr, empty)
""",
    ):
        errors.append(
            "streamed max-Q builder: folded value or support transport changed"
        )
    return errors


def _max_kernel_surface_errors(tree: ast.Module) -> list[str]:
    """Pin the six max-Q kernel classes to fields, one ``__call__``, and no more.

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
                "Q_and_F: ActionFunction",
                "n_discrete_action_axes: int",
                "return_feasibility: bool = False",
            ),
        ),
        (
            "hard-max max-Q kernel",
            "_HardMaxQOverA",
            (
                "__name__: ClassVar[str] = 'max_Q_over_a'",
                "Q_and_F: ActionFunction",
                "stakeholders: tuple[str, ...] | None",
                "pareto_weights: ParetoWeights | None",
                "return_feasibility: bool = False",
            ),
        ),
        (
            "streamed max-Q kernel",
            "_StreamedMaxQOverA",
            (
                "__name__: ClassVar[str] = 'streamed_max_Q_over_a'",
                "Q_and_F: RawActionFunction",
                "action_names: tuple[ActionName, ...]",
                "n_discrete_action_axes: int",
                "has_taste_shocks: bool",
                "stakeholders: tuple[str, ...] | None",
                "pareto_weights: ParetoWeights | None",
                "q_and_f_arg_names: frozenset[ReferenceName]",
                "action_width_keyword: str",
                "whole_product_Q_and_F: RawActionFunction",
                "return_feasibility: bool = False",
            ),
        ),
        (
            "folded max-Q kernel",
            "_FoldedMaxQOverA",
            (
                "mapped: Callable[..., BoundedMaxOutput]",
                "fold_steps: tuple[tuple[StateName, int, int | None], ...]",
                "fold_weights: Mapping[StateName, FloatND]",
                "fold_reducers: Mapping[StateName, Callable[..., FloatND]]",
                "return_feasibility: bool",
            ),
        ),
        (
            "taste-shock argmax kernel",
            "_TasteShockArgmaxQOverA",
            (
                "__name__: ClassVar[str] = 'argmax_and_max_Q_over_a'",
                "Q_and_F: ActionFunction",
                "n_discrete_action_axes: int",
            ),
        ),
        (
            "hard-max argmax kernel",
            "_HardMaxArgmaxQOverA",
            (
                "__name__: ClassVar[str] = 'argmax_and_max_Q_over_a'",
                "Q_and_F: ActionFunction",
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
                "internal_schedule: tuple[InternalDispatch, ...] = ()",
                "host_buffers: Mapping[str, ShapeDtypePytree] = MappingProxyType({})",
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
                "state_action_space: StateActionSpace | None",
                "next_regime_to_V_arr: Mapping[RegimeName, ValueTemplate]",
                "next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload]",
                "flat_params: Mapping[RegimeName, FlatRegimeParams | FlatEdgeParams]",
                "period: int",
                "ages: TimeAxis | None",
                "edge_regime_to_V_arr: Mapping[RegimeName, ValueTemplate] | None = None",
                "same_period_regime_to_V_arr: Mapping[RegimeName, ValueTemplate] | None = None",
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
                "checked_variant: _CheckedCoreVariant | None = None",
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
                "arguments: MappingProxyType[ReferenceName, PytreeValue | ShapeDtypePytree]",
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
                "arguments: MappingProxyType[str, ArgumentTree]",
                "static_kwargs: MappingProxyType[str, int]",
                "requirements: CoreExecutionRequirements",
                "output_roles: OutputRoleTree",
                "disposition: CoreExecutionDisposition",
                "donation_candidates: tuple[str, ...]",
                "tile_widths: MappingProxyType[str, int]",
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
        expected_source="_CORE_PROGRAM_VERSION = 8",
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
DENSE_OFF_GRID_RAISE_COUNT = "deliberately_dense:off_grid_raise_count"
DENSE_DECOUPLED = "deliberately_dense:decoupled"
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
                "current_iid_plan: CurrentIIDPlan | None = None",
                "target_regimes: tuple[RegimeName, ...]",
                "same_period_ref_regimes: tuple[RegimeName, ...] = ()",
                "edge_reference_regimes: tuple[RegimeName, ...] = ()",
                "edge_target_regimes: tuple[RegimeName, ...] = ()",
                "arrival_queries: Mapping[RegimeName, ArrivalQueryPlan] = field(default_factory=lambda: MappingProxyType({}))",
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
            fields=(
                "n_runtime_bounds: int = field(default=0, metadata={'fingerprint_omit_if_default': True})",
                "current_iid: bool = field(default=False, metadata={'fingerprint_omit_if_default': True})",
                "_core_programs: Mapping[str, CoreProgram]",
                "landing_checks: tuple[LandingCheck, ...] = field(default=(), metadata={'fingerprint_omit_if_default': True})",
            ),
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


_FEASIBILITY_SUMMARY_SURFACE = _surface_pin(FEASIBILITY_SOURCE)
_FEASIBILITY_SUMMARY_PINS = _callable_pins(
    source=FEASIBILITY_SOURCE,
    names=("CountedGridSearch.__call__", "FeasibilityKernelOutput.__post_init__"),
)


def _feasibility_summary_errors(tree: ast.Module) -> list[str]:
    """Require the diagnostic wrapper to publish its producer's unchanged value."""
    errors = _class_surface_errors(
        tree=tree,
        label="native feasibility summary",
        class_name="CountedGridSearch",
        fields=(
            "function: Callable[..., tuple[FloatND, BoolND] | tuple[FloatND, BoolND, ScalarBool] | BoundedResult[tuple[FloatND, BoolND]]]",
            "collective: bool",
            "current_iid: bool = False",
        ),
        methods=("__call__",),
        decorators=("dataclass(frozen=True, kw_only=True, eq=False)",),
    )
    if _transport_module_surface(tree) != _FEASIBILITY_SUMMARY_SURFACE:
        errors.append("native feasibility summary: module surface changed")
    errors.extend(
        _exact_callable_errors(
            tree=tree,
            label="native feasibility summary",
            contracts=_FEASIBILITY_SUMMARY_PINS,
        )
    )
    try:
        _, method = _method_definition(
            tree=tree, class_name="CountedGridSearch", method_name="__call__"
        )
    except ValueError as error:
        return [*errors, f"native feasibility summary: {error}"]
    if not _body_matches(
        node=method,
        expected_source="""evaluated = self.function(**kwargs)
bounded = evaluated if type(evaluated) is BoundedResult else None
if bounded is not None:
    evaluated = bounded.output
current_iid_invalid = None
if self.current_iid:
    value, empty, invalid = cast('tuple[FloatND, BoolND, ScalarBool]', evaluated)
    current_iid_invalid = jnp.any(invalid)
    evaluated = (value, empty)
value, empty = cast('tuple[FloatND, BoolND]', evaluated)
infinite = jnp.isinf(value)
if self.collective:
    infinite = jnp.any(infinite, axis=-1)
empty_count = jnp.sum(empty, dtype=jnp.int32)
other_inf_count = jnp.sum(infinite & ~empty, dtype=jnp.int32)
output: CountedGridSearchOutput = (value, empty, empty_count, other_inf_count) if self.collective else (value, empty_count, other_inf_count)
if self.current_iid:
    output = cast('CountedGridSearchOutput', (*output, current_iid_invalid))
return BoundedResult[CountedGridSearchOutput](output=output, bound_status=bounded.bound_status) if bounded is not None else output
""",
    ):
        errors.append("native feasibility summary: producer value or counters changed")
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
                "tile_widths: MappingProxyType[str, int]",
                "input_transfer_plan: tuple[ResolvedValueTransfer, ...] = ()",
                "internal_input_templates: MappingProxyType[ReferenceName, ShapeDtypePytree] = MappingProxyType({})",
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
ARRIVAL_REPLAY_VALUE = "arrival_replay_value"
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
                "from _lcm.typing import ArgumentTree, DataclassInstance, HostArray, RegimeName, StateName",
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
    current_iid_draw_plan=CurrentIIDDrawPlan(bindings=current_iid_bindings)
    if current_iid_bindings
    else None,
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
        "simulation_context": """simulation_context = dataclass_replace(
    solver_context,
    Q_and_F_functions=Q_and_F_functions,
    functions=simulate_functions,
    constraints=constraints,
    constraint_functions=core.constraint_functions,
    processed_constraints=core.processed_constraints,
    constraint_plan=routing.plan,
    grids=simulate_grids,
    state_action_space=create_state_action_space(
        variables=simulation_variables, grids=simulate_grids
    ),
    current_iid_plans=MappingProxyType({}),
    current_iid_draw_plan=CurrentIIDDrawPlan(bindings=current_iid_bindings)
    if current_iid_bindings
    else None,
)""",
        "programs": """programs = build_simulation_programs(
    context=dataclass_replace(
        simulation_context, state_action_space=state_action_space
    ),
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
    phase = [
        statement.value
        for statement in live_body
        if isinstance(statement, ast.Assign)
        and any(_target_names(target) == ("phase",) for target in statement.targets)
    ]
    if not (
        bindings.get("phase") == 1
        and len(phase) == 1
        and isinstance(phase[0], ast.Call)
        and _call_name(phase[0]) == "SimulationPhase"
        and _name(node=_keyword(call=phase[0], name="programs"), expected="programs")
        and len(returns) == 1
        and isinstance(returns[0].value, ast.Call)
        and _call_name(returns[0].value) == "_SimulationBuild"
        and _name(node=_keyword(call=returns[0].value, name="phase"), expected="phase")
        and _statements_match(
            observed=returns,
            expected_source="""return _SimulationBuild(
    phase=phase,
    context=(
        dataclass_replace(
            simulation_context,
            Q_and_F_functions=MappingProxyType(
                {**Q_and_F_functions, **query_Q_and_F_functions}
            ),
        )
        if query_Q_and_F_functions
        else simulation_context
    ),
)""",
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
            "_SimulationBuild",
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
                    "from _lcm.engine import CaseRReplaySpec, CaseUReplaySpec, EGMPolicyRead, FeasibilityPoolsByPeriod, NNBEGMPolicyRead, Regime, SimulationPhase, SolutionPhase, StateActionSpace, Variables, _fail_if_template_is_misplaced, placed_devices_for_ids"
                ),
                "from _lcm.regime_building.max_Q_over_a import BoundedArgmaxOutput, get_argmax_and_max_Q_over_a",
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
            "SimulationProgramExecutor.dispatch",
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


def _simulation_program_corridor_errors(
    *, tree: ast.Module, source: SourcePath
) -> list[str]:
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
            "transition_random_addresses",
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
            "Model._process_params",
            "Model._canonicalize_params",
            "Model._validate_canonical_params",
            "_simulation_programs",
        ),
        SIMULATION_RANDOM_SOURCE: (
            "canonicalize_simulation_seed",
            "create_simulation_key",
            "_create_simulation_key",
            "split_simulation_key",
            "_split_simulation_key",
            "generate_simulation_keys",
            "_generate_rowed_simulation_keys",
            "draw_random_seed",
        ),
        # The same typed domain/name/coordinate, uint64 original row and period
        # encoder feeds contiguous, grouped and destination-taste allocations.
        SIMULATION_RANDOM_ADDRESS_SOURCE: (
            "RandomAddress.__post_init__",
            "RandomAddress.words",
            "encode_subject_row",
            "draw_window_keys",
            "draw_row_keys",
            "scalar_draw_axis",
            "inverse_cdf_index",
            "draw_categorical_index",
            "_threefry_key_dtype",
            "_row_offset_words",
            "_fold_subject_key",
        ),
        ENGINE_SOURCE: (),
    }
)


def _simulation_dispatch_corridor_errors(
    *, tree: ast.Module, source: SourcePath
) -> list[str]:
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
        LCM_TYPING_SOURCE: ("CanonicalArrayWriter.__call__",),
        DTYPES_SOURCE: (
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
            "_regime_views_by_start_pool",
            "_constraints_read_age_specialized_functions",
            "_check_regime_feasibility",
            "_regime_feasibility_mask",
            "_run_profiled_feasibility",
            "_batched_feasibility_check",
            "_evaluate_constant_feasibility",
            "_admits_any_action",
            "_per_constraint_feasibility",
            "_initial_physical_joint_states",
            "_replace_joint_coordinates",
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
            "draw_taste_shock_keys",
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
            "run_simulation_producer",
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
            "_build_V_arr_template",
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
    "simulation_adapter:initial_physical_transform_changed": (
        INITIAL_CONDITIONS_SOURCE,
        "process.transform(standard=node, mean=mean, cov=cov)",
        "node",
    ),
    "simulation_adapter:initial_physical_mean_forwarding_omitted": (
        INITIAL_CONDITIONS_SOURCE,
        "process.transform(standard=node, mean=mean, cov=cov)",
        "process.transform(standard=node, cov=cov)",
    ),
    "simulation_adapter:initial_physical_cov_forwarding_omitted": (
        INITIAL_CONDITIONS_SOURCE,
        "process.transform(standard=node, mean=mean, cov=cov)",
        "process.transform(standard=node, mean=mean)",
    ),
    "simulation_adapter:initial_physical_mean_operand_changed": (
        INITIAL_CONDITIONS_SOURCE,
        'arguments["mean"] = cast("FloatND", regime_params[f"{group}__mean"])',
        'arguments["mean"] = jnp.zeros_like(regime_params[f"{group}__mean"])',
    ),
    "simulation_adapter:initial_physical_cov_operand_changed": (
        INITIAL_CONDITIONS_SOURCE,
        'arguments["cov"] = cast("FloatND", regime_params[f"{group}__cov"])',
        'arguments["cov"] = jnp.zeros_like(regime_params[f"{group}__cov"])',
    ),
    "simulation_adapter:initial_physical_fixed_law_selection_changed": (
        INITIAL_CONDITIONS_SOURCE,
        "if process.mean is None:",
        "if True:",
    ),
    "simulation_adapter:initial_physical_subject_order_changed": (
        INITIAL_CONDITIONS_SOURCE,
        '"standard": state_df[list(names)].to_numpy()',
        '"standard": state_df[list(names)].to_numpy()[::-1]',
    ),
    "simulation_adapter:initial_physical_admission_bypassed": (
        INITIAL_CONDITIONS_SOURCE,
        "physical = _evaluate_admitted_transition_producer(\n                function=transform, arguments=arguments, memory=memory\n            )",
        "physical = transform(**arguments)",
    ),
    "simulation_adapter:initial_physical_output_hold_omitted": (
        INITIAL_CONDITIONS_SOURCE,
        "            memory.hold(tree=physical)\n",
        "",
    ),
    "simulation_adapter:initial_physical_prefix_changed": (
        INITIAL_CONDITIONS_SOURCE,
        "        value = values.iloc[0]\n",
        "        value = 0.0\n",
    ),
    "simulation_adapter:initial_physical_internal_columns_retained": (
        INITIAL_CONDITIONS_SOURCE,
        "        state_df = state_df.drop(columns=list(names))\n",
        "",
    ),
    "simulation_adapter:initial_physical_table_column_changed": (
        INITIAL_CONDITIONS_SOURCE,
        "            state_df[name] = np.asarray(physical)[:, dimension]",
        "            state_df[name] = np.asarray(physical)[:, 0]",
    ),
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
        "_caller_arrays(values=(params, initial_conditions))",
        "_caller_arrays(values=(initial_conditions,))",
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


def _simulation_adapter_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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
        ),
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


def _finite_budget_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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


def _eager_input_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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


def _solve_readiness_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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


def _combined_input_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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


def _structural_blueprint_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Authenticate the structural-blueprint store and its key derivations."""
    surface, callables = _STRUCTURAL_BLUEPRINT_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="structural blueprint cache", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append("structural blueprint cache: module bindings changed")
    return errors


_BORROWING_BOUND_CONTRACTS = _contracts(
    {
        AGE_BOUND_NORMALIZATION_SOURCE: (
            "_representative_regime",
            "_regime_has_markers",
            "_collect_function_markers",
            "_restore_representative_floor_source",
        ),
        DYNAMIC_BOUND_SOURCE: (
            "ScalarFloor.__post_init__",
            "ScalarFloor.__call__",
            "_ScalarFloorFactory.__call__",
            "get_floor_node",
            "get_floor_source",
            "_bound_surface",
            "validate_declared_bounds",
            "_resolved_rhs_contract",
            "_walk_rhs",
            "_validate_scalar_trace",
        ),
        POST_DECISION_BOUND_SOURCE: (
            "grid_origin_bound_spec",
            "_PostDecisionLowerBound.__post_init__",
            "_PostDecisionLowerBound.__call__",
        ),
        BOUND_PROOF_SOURCE: (
            "lower_bound_declaration",
            "proves_the_savings_grids_lower_bound",
            "matches_installed_literal_bound",
            "_SavingsGridLowerBoundProof.__call__",
        ),
        BOUND_NORMALIZATION_SOURCE: ("_as_condition",),
        USER_REGIME_BOUND_SOURCE: (
            "Regime.decomposed_functions",
            "Regime.decomposed_constraints",
            "Regime.get_all_functions",
            "Regime._restore_floor_sources",
            "decompose_constraints",
            "Regime.with_engine_functions",
        ),
        LIQUID_BOUND_DECLARATION_SOURCE: ("post_decision_lower_bound",),
        MODEL_PROCESSING_SOURCE: (
            "validate_model_inputs",
            "_validate_borrowing_bound_routes",
            "_law_phase_varies",
            "_is_solve_proved_post_decision_lower_bound",
        ),
    }
)


def _floor_source_implementation(*, tree: ast.Module) -> ast.FunctionDef:
    """Authenticate two typing overloads and select their one runtime body."""
    expected = ast.parse(
        "@overload\n"
        "def get_floor_source(*, function: UserFunction) -> UserFunction: ...\n"
        "@overload\n"
        "def get_floor_source(*, function: None) -> None: ...\n"
    ).body
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_floor_source"
    ]
    bindings = _scope_binding_counts(tree.body)
    imports = _relevant_imports(tree=tree, names={"overload"})
    if (
        len(functions) != 3
        or bindings.get("get_floor_source") != 3
        or bindings.get("overload") != 1
        or imports
        != ["from typing import TYPE_CHECKING, Literal, cast, no_type_check, overload"]
        or [_ast_key(node) for node in functions[:2]]
        != [_ast_key(node) for node in expected]
        or functions[-1].decorator_list
    ):
        raise ValueError("get_floor_source overload declarations changed")
    return functions[-1]


def _borrowing_bound_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Authenticate named floor composition and its validation/proof boundary."""
    surface, callables = _BORROWING_BOUND_CONTRACTS[source]
    ordinary = (
        {
            name: digest
            for name, digest in callables.items()
            if name != "get_floor_source"
        }
        if source == DYNAMIC_BOUND_SOURCE
        else callables
    )
    errors = _exact_callable_errors(
        tree=tree, label="borrowing bound authority", contracts=ordinary
    )
    if source == DYNAMIC_BOUND_SOURCE:
        try:
            implementation = _floor_source_implementation(tree=tree)
        except ValueError as error:
            errors.append(f"borrowing bound authority: {error}")
        else:
            if _callable_ast_sha256(implementation) != callables["get_floor_source"]:
                errors.append(
                    "borrowing bound authority: exact callable corridor 'get_floor_source' changed"
                )
    if _transport_module_surface(tree) != surface:
        errors.append("borrowing bound authority: module or producer schema changed")
    return errors


_S03_SCALAR_CONTRACTS = _contracts(
    {
        S03_EULER_SOURCE: ("invert_euler",),
        "src/_lcm/constraints/bounds.py": ("_SavingsGridLowerBoundProof.__call__",),
        "src/_lcm/egm/budget.py": (
            "get_intrinsic_budget_constraint",
            "_IntrinsicBudgetConstraint.__call__",
        ),
        S03_DECLARED_LAW_SOURCE: (
            "build_declared_liquid_law",
            "_DeclaredLiquidLaw.__call__",
            "_landing_point",
            "_fail_if_law_reaches_past_the_post_decision",
            "fail_if_declared_law_is_not_increasing",
        ),
        S03_INTERP_SOURCE: ("locate_on_grid",),
        S03_NUMERIC_INVERSE_SOURCE: (
            "numeric_inverse_marginal_utility",
            "_expand_upper_bracket",
            "_LogMarginalUtility.__call__",
        ),
        S03_ONE_ASSET_STEP_SOURCE: (
            "egm_one_asset_step",
            "_solve_viable_feature_row",
            "_prepare_augmented_savings_support",
            "_flatten_egm_savings_support",
            "_flatten_egm_savings_support_with_keys",
            "_unflatten_egm_savings_support",
        ),
        S03_PREFERENCES_SOURCE: (
            "newton_action_ceiling",
            "get_numeric_inverse_marginal_utility",
            "preferences_from_utility",
            "get_preferences_builder",
            "get_discount_factor_reader",
            "concatenate_regime_function",
            "_elementwise",
            "_Elementwise.__call__",
            "BoundUtilityOfAction.__call__",
            "AnalyticInverseMarginalUtility.__call__",
            "_NumericInverseMarginalUtility.__call__",
            "_PreferencesBuilder.__call__",
            "_DiscountFactorReader.__call__",
        ),
        S03_SCALAR_QUERY_SOURCE: (
            "build_scalar_egm_query",
            "ScalarEGMQuery.capabilities",
            "ScalarEGMQuery.evaluate_floor",
            "ScalarEGMQuery.__call__",
            "_weighted_pair",
        ),
        "src/_lcm/engine.py": (),
        "src/_lcm/model_processing.py": ("_validate_borrowing_bound_routes",),
        "src/_lcm/regime_building/processing.py": (
            "_CanonicalRegimeBuilder.__call__",
            "_prepare_scalar_egm_query_models",
            "_fail_if_a_declaration_moved_more_than_the_kernels",
            "_fail_if_a_continuation_demand_is_unmet",
            "_build_solution_phase",
            "_process_regime_core",
        ),
        "src/_lcm/solution/backward_induction.py": (
            "solve",
            "_match_leaf_pair_sharding",
            "_match_leaf_template_sharding",
            "_release_closed_period_inputs",
        ),
        "src/_lcm/solution/contract.py": ("SolutionKernels.__post_init__",),
        "src/_lcm/solution/egm.py": (
            "EGM.build_period_kernels",
            "_build_egm_period_kernel",
            "_EGMArgumentBuilder.__call__",
            "_EGMArgumentBuilder._law_readings",
            "_EGMPeriodKernel.__call__",
            "_scalar_egm_output_role",
            "_build_egm_core",
            "_EGMCore.__call__",
            "_EGMCore._prepare_child_read",
            "_RegimeMassGuardedCore.__call__",
        ),
        "src/_lcm/solution/feasibility.py": ("BoundInputStatus.__reduce__",),
        "src/_lcm/solution/model_authority.py": ("_validate_model_query_spec",),
        "src/_lcm/zero_safe.py": ("zero_safe_weighted_term",),
    }
)


def _s03_scalar_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Authenticate scalar-query owners and imported numerical dependencies."""
    surface, callables = _S03_SCALAR_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="S03 scalar dependency", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(f"S03 scalar dependency: {source} declaration surface changed")
    return errors


_REVIEWED_PREFIX_CONTRACTS = _contracts(
    {
        "src/_lcm/egm/asset_row.py": (
            "_SolveOneComboAssetRows.__call__",
            "_SolveOneNode.__call__",
            "_asset_candidates_with_direct",
            "_asset_candidates_without_direct",
            "_asset_direct_gradient",
            "_conditional_asset_node",
            "_conditional_asset_rows",
            "_finalize_asset_row_node",
            "_node_value_slope",
            "_status_asset_rows",
            "_with_asset_child_status",
            "publish_node_from_bracket",
            "_ExpectedContinuationValue.__call__",
            "_continuation_of_euler_state",
            "_get_expected_continuation_value",
            "_get_solve_one_combo_asset_rows",
        ),
        "src/_lcm/egm/carry.py": (
            "_flatten_owned_row_read",
            "_flatten_owned_row_read_with_keys",
            "_read_owned_channels",
            "_unflatten_owned_row_read",
            "read_owned_row",
            "read_marginal_row",
            "read_value_and_slope_row",
            "read_value_row",
            "_read_owned_row_with_support",
        ),
        "src/_lcm/egm/continuation.py": (
            "_child_row_values",
            "_ActualChildQuery.__call__",
            "_BoundContinuation.__call__",
            "_BoundContinuation._read_child",
            "_BoundContinuation._retains_a_distribution",
            "_BoundContinuation._with_query_status",
            "_ChildCarryReader.__call__",
            "_ChildEulerState.__call__",
            "_ChildEulerState.at_node",
            "_ReadAtNodes.__call__",
            "_RowQueriesAndGradients.__call__",
            "_WeightedNodeBlockFold.__call__",
            "_accumulate_ez_partials_over_blocks",
            "_aggregate_child_choices",
            "_blend_passive_axes",
            "_build_child_reads",
            "_child_draw_reads",
            "_choose_blended_side",
            "_collapse_stacked_candidates",
            "_compute_row_queries_and_gradients",
            "_draw_axis_names",
            "_expect_over_stochastic_nodes",
            "_fail_if_carry_shape_mismatches_declaration",
            "_get_child_carry_reader",
            "_hard_max_and_one_hot",
            "_interleave_child_index",
            "_is_runtime_process",
            "_is_stochastic_child_axis",
            "_joint_node_weights",
            "_local_support_values",
            "_on_node_scale",
            "_partials_on_node_scale",
            "_read_feature_child_choices",
            "_seed_child_query",
            "_stacked_blend_payloads",
            "_with_co_map_states",
            "bind_continuation",
            "build_continuation_plan",
            "child_resources_params",
            "_accumulate_ez_partials_block",
            "_composed_row_resources",
            "_composed_simple_resources",
            "_left_slope_row",
            "get_egm_continuation_targets",
            "stochastic_node_axes",
        ),
        "src/_lcm/egm/dcegm_query.py": (
            "DCEGMQuery.__call__",
            "DCEGMQuery._read_at_floor",
            "DCEGMQuery._read_liquid_corner",
            "DCEGMQuery._read_rows",
            "DCEGMQuery.capabilities",
            "DCEGMQuery.evaluate_floor",
            "DCEGMQuery.input_names",
            "DCEGMQuery.read_choices",
            "DCEGMQuery.read_prepared",
            "_evaluate",
            "_factor_product_bits",
            "_factor_product_jvp",
            "_owned_at_query",
            "_passive_weights",
            "_read_compact_corner",
            "_read_owned_derivative",
            "_weighted_sum",
            "build_dcegm_query",
            "DCEGMQuery.read_prepared_with_counts",
            "DCEGMQuery._read_at_floor_with_summary",
            "DCEGMQuery._read_rows_without_counts",
            "DCEGMQuery._read_rows_with_counts",
            "_blend_owned_rows",
            "_read_owned_derivative_with_support",
            "_owned_at_query_with_support",
        ),
        "src/_lcm/egm/dcegm_query_model.py": (),
        "src/_lcm/egm/interp.py": (
            "_derivative_between_nodes",
            "_hermite_bracket_derivatives",
            "_hermite_coefficients",
            "_hermite_correction",
            "_hermite_prepared_read_jvp",
            "_hermite_prepared_read_primal",
            "_hermite_query_derivative",
            "_interp_between_nodes",
            "_limit_slope",
            "_linear_prepared_read_jvp",
            "_linear_prepared_read_primal",
            "_linear_query_derivative",
            "_read_prepared_row",
            "interp_and_derivative_on_prepared_grid",
            "interp_on_padded_grid",
            "interp_on_prepared_grid",
            "prepare_padded_grid",
            "_hermite_read_at_fixed_query",
            "_linear_read_at_fixed_query",
            "_sanitize_gathered",
            "interp_left_record_on_prepared_grid",
            "interp_on_regular_grid",
        ),
        "src/_lcm/egm/query_data.py": ("select_query_data",),
        "src/_lcm/egm/regime_introspection.py": (
            "_IdentityResources.__call__",
            "_KeeperNoAdjustment.__call__",
            "_annotation_of_arg",
            "_as_dcegm",
            "_concatenate_child_resources",
            "_get_child_discrete_actions",
            "_get_child_resources",
            "_get_child_state_name",
            "_keeper_no_adjustment_function",
            "_get_discrete_state_names",
            "_get_passive_state_names",
            "_get_process_state_names",
        ),
        "src/_lcm/egm/scalar_query.py": (
            "ScalarEGMQuery.prepare",
            "ScalarEGMQuery.read_prepared",
        ),
        "src/_lcm/egm/step.py": (
            "_ComboFeasibility.__call__",
            "_EGMStep.__call__",
            "_SolveOneComboOverAxes.__call__",
            "_build_feasibility_function",
            "_build_kernel_pieces",
            "_get_egm_step",
            "_map_combo_product",
            "_publish_conditional_combo",
            "_publish_conditional_product",
            "_reorder_combo_axes",
            "build_egm_step_functions",
            "compute_egm_carry_length",
            "_at_one_splayed_combo",
        ),
        "src/_lcm/egm/step_core.py": (
            "ResourcesOfState.__call__",
            "_ComputeNode.__call__",
            "_SolveOneCombo.__call__",
            "_candidate_supgradient",
            "_compute_constrained_candidates",
            "_compute_nodes_after_first",
            "_compute_nodes_over_savings",
            "_conditional_combo_output",
            "_get_compute_node",
            "_legacy_candidates_with_child_status",
            "_publish_V_and_carry_rows",
            "_refined_rows",
            "_source_construction_status",
            "_with_literal_child_status",
            "bind_cell_floor",
            "tile_block_size",
            "_get_solve_one_combo",
        ),
        "src/_lcm/execution/core_program.py": (
            "_CheckedCoreVariant.__post_init__",
            "_validate_checked_variant",
            "bind_core_program_fixed_params",
        ),
        "src/_lcm/execution/internal_outputs.py": ("_select_path",),
        "src/_lcm/probability.py": (
            "_encoded",
            "_exponent_bias",
            "_int_dtype",
            "_magnitude_bits",
            "_mantissa_bits",
            "_product_parts",
            "_shift_to_normal",
            "_significand_in_unit_range",
            "_smallest_normal_bits",
            "_unbiased_exponent",
            "exact_product",
            "is_below_smallest_normal",
            "is_negative",
            "is_represented_zero",
            "_scaled_bits",
            "_scaled_bits_jvp",
            "is_live",
            "probabilities_form_distribution",
            "regime_mass_is_a_distribution",
            "scaled_by_power_of_two",
        ),
        "src/_lcm/solution/dcegm.py": (
            "DCEGM.build_period_kernels",
            "_DCEGMPeriodKernel.with_fixed_params",
            "ExactEnvelope.__post_init__",
            "DCEGM.__post_init__",
            "FUESEnvelope.__post_init__",
            "_DCEGMArgumentBuilder.__call__",
            "_DCEGMPeriodKernel.__call__",
            "_DCEGMPeriodKernel.__post_init__",
            "_DCEGMPeriodKernel.core_programs",
            "_carry_subset",
            "_consume_bounded_dcegm_output",
            "_dcegm_output_roles",
            "_dcegm_role_template",
            "_dcegm_status_role",
            "_dcegm_values_core",
            "_fail_if_envelope_max_runs_too_few",
            "_fail_if_fues_jump_thresh_non_positive",
            "_fail_if_fues_n_points_to_scan_too_few",
            "_fail_if_n_constrained_points_too_few",
            "_fail_if_refined_grid_factor_too_small",
            "_fail_if_savings_grid_is_stochastic",
            "_stochastic_node_axis",
            "_tiled_axes",
        ),
        "src/_lcm/solution/egm.py": (
            "_EGMCore._value_reconstructed_action",
            "_LiteralDCResourceTransport.__call__",
            "_LiteralDCResourceTransport._at_state",
            "_LiteralDCResourceTransport.value",
            "_build_literal_dc_resource_transport",
            "_read_literal_dc_linear_child",
            "_read_one_row_dc_child",
            "_EGMArgumentBuilder._retains_regime_mass",
            "_EGMPeriodKernel.__post_init__",
            "_EGMPeriodKernel.core_programs",
            "_EGMPeriodKernel.with_fixed_params",
            "_build_one_asset_carry_template",
            "_one_sided_boundary_savings_targets",
            "guard_regime_mass",
        ),
        "src/_lcm/solution/feasibility.py": (
            "ContinuationReadCounts.__reduce__",
            "_flatten_continuation_read_counts",
            "_flatten_continuation_read_counts_with_keys",
            "_unflatten_continuation_read_counts",
            "count_current_feasibility",
        ),
        "src/_lcm/solution/nbegm.py": (
            "_NBEGMDiscreteCore.__call__",
            "_NBEGMScheduleDiscreteCore.__call__",
            "_collect_nbegm_discrete_spec",
            "_collect_nbegm_schedule_discrete_spec",
        ),
        SOLVER_API_SOURCE: (),
    }
)


def _reviewed_prefix_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Authenticate finite library-owned query arithmetic and checked transport."""
    surface, callables = _REVIEWED_PREFIX_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="reviewed S04/S05 dependency", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(
            f"reviewed S04/S05 dependency: {source} declaration surface changed"
        )
    return errors


_S02_QUERY_STATUS_CONTRACTS = _contracts(
    {
        "src/_lcm/continuation.py": (
            "EGMQuerySpec.__post_init__",
            "EGMContinuationSpec.__post_init__",
            "validate_query_bound_authority",
            "_same_bound_identity",
            "validate_query_model_authority",
            "build_query_model_authority",
            "_query_bound_evaluator_targets",
            "_query_evaluator_contract",
            "validate_query_evaluator_authority",
            "validate_query_template",
            "EGMRowTopologySpec.__post_init__",
            "_validate_row_topology",
            "_validate_row_topology_authority",
        ),
        "src/_lcm/egm/query_data.py": (
            "EGMQueryData.__post_init__",
            "EGMQueryData.leaves",
            "_query_record_leaves",
            "_flatten_query_record",
            "_flatten_query_record_with_keys",
            "_unflatten_query_record",
            "build_template_egm_query_data",
        ),
        "src/_lcm/egm/carry.py": (
            "_flatten_egm_carry",
            "_flatten_egm_carry_with_keys",
            "_unflatten_egm_carry",
            "_assemble_egm_carry",
            "EGMCarry.leaves",
            "EGMCarry.capabilities",
            "EGMCarry.value_at",
            "EGMCarry.marginal_at",
            "egm_carry_role_tree",
            "egm_query_role_tree",
            "_query_output_role",
            "stack_egm_carries",
            "_stack_carry_leaf",
            "build_template_egm_carry",
            "shard_carry_template",
            "_place_carry_template_on_one_device",
            "_place_carry_leaf",
            "EGMCarry.query_data_for_transport",
        ),
        "src/_lcm/solution/model_authority.py": (
            "supports_owned_negm_u_artifacts",
            "_is_installed_negm_u_continuation",
            "build_solution_authority",
            "_validate_model_query_spec",
            "_egm_carry_artifact_layout",
            "_query_leaf_axis_names",
            "_declared_carry_row_state_names",
            "_declared_carry_trailing_state",
            "_model_role_axes",
        ),
        "src/_lcm/solution/continuation_reads.py": ("continuation_leaf_reads",),
        "src/_lcm/solution/continuation_arguments.py": (
            "MarginalLeafArguments.__call__",
            "MarginalLeafCore.__call__",
            "marginal_leaf_reads",
        ),
        "src/_lcm/solution/fingerprint.py": (
            "fingerprint_model_structure",
            "_query_semantic_descriptor",
            "_project_user_regime_declaration",
            "_semantic_fingerprint",
            "_SemanticHasher.visit",
        ),
        "src/_lcm/solution/egm.py": (
            "_EGMCore.__call__",
            "_EGMArgumentBuilder.__call__",
            "_has_mandatory_child_failure",
        ),
        "src/_lcm/egm/budget.py": (
            "get_intrinsic_budget_constraint",
            "_IntrinsicBudgetConstraint.__call__",
        ),
        "src/_lcm/engine.py": (),
        "src/_lcm/dtypes.py": ("canonical_float_dtype",),
        "src/_lcm/post_decision_bound.py": (),
        "src/_lcm/variables.py": (),
        "src/_lcm/state_action_space.py": (),
        "src/_lcm/execution/output_layout.py": (),
        "src/_lcm/solution/feasibility.py": (
            "_flatten_bound_status",
            "_flatten_bound_status_with_keys",
            "_unflatten_bound_status",
            "reduce_bound_input_status",
            "summarize_bound_inputs",
            "shift_savings_grid",
            "merge_bound_input_status",
            "validate_runtime_bound_status",
            "consume_runtime_bound_outputs",
            "FeasibilityKernelOutput.__post_init__",
            "CountedGridSearch.__call__",
            "_flatten_bounded_result",
            "_flatten_bounded_result_with_keys",
            "_unflatten_bounded_result",
            "QueryInputStatus.__reduce__",
            "_flatten_query_input_status",
            "_flatten_query_input_status_with_keys",
            "_unflatten_query_input_status",
            "reduce_query_input_status",
            "merge_query_input_status",
            "validate_query_input_status",
        ),
        "src/_lcm/utils/dispatchers.py": (
            "map_over_leading_axis",
            "_map_with_bound_summary",
            "_BoundVmap.__call__",
            "vmap_1d",
            "simulation_spacemap",
            "productmap",
            "tiled_productmap",
            "_RestoreProductAxisOrder.__call__",
            "_restore_mapped_product",
            "_map_grouped_product",
            "_map_whole_product",
            "_ProductMapBatched.__call__",
            "_MappedOverOneMoreAxis.__call__",
        ),
        "src/_lcm/simulation/subject_parallel.py": (
            "shard_subject_function",
            "_evaluate_local_subjects",
        ),
        "src/_lcm/regime_building/Q_and_F.py": (
            "runtime_bound_specs",
            "BoundedQAndF.__post_init__",
            "BoundedQAndF.__call__",
            "_BoundUtilitiesAndF.__post_init__",
            "_BoundUtilitiesAndF.evaluated",
            "_get_U_and_F",
            "_BoundFeasibility.__call__",
            "_get_feasibility",
            "_QAndF.evaluated",
            "_TerminalQAndF.evaluated",
            "_TerminalCollectiveQAndF.evaluated",
            "_CollectiveQAndF.evaluated",
            "_publish_signature",
        ),
        "src/_lcm/regime_building/max_Q_over_a.py": (
            "get_streaming_max_Q_over_a",
            "_StreamedMaxQOverA.__call__",
            "_with_bound_summary",
            "_FoldedMaxQOverA.__call__",
            "get_max_Q_over_a",
            "get_argmax_and_max_Q_over_a",
            "_HardMaxQOverA.__call__",
            "_HardMaxArgmaxQOverA.__call__",
            "_SmoothedMaxQOverA.__call__",
            "_TasteShockArgmaxQOverA.__call__",
            "get_action_partitioned_max_Q_over_a",
            "_ActionPartitionedMaxQOverA.__call__",
            "_OnActionPartitionAxis.__call__",
            "_arguments_named",
            "_call_with_operands",
        ),
        "src/_lcm/solution/grid_search.py": (
            "_FlattenBoundedGridSearch.__call__",
            "_GridSearchPeriodKernel.__call__",
            "GridSearch.build_period_kernels",
            "_grid_search_fold_weights",
        ),
        "src/_lcm/solution/action_streaming.py": (
            "_bounded_hard_max",
            "_StreamingHardMax.__call__",
            "_PartitionedStreamingHardMax.__call__",
            "_PartitionedStreamingHardMax.local",
            "_StreamingEV1ExpectedMax.__call__",
            "_StreamingCollectiveHardMax.__call__",
            "_evaluate_owned_block",
            "_evaluate_ev1_branch_block",
            "_evaluate_collective_block",
            "_scan_ev1_blocks",
            "_scan_one_ev1_block",
            "_finalize_open_ev1_branch_group",
        ),
        "src/_lcm/solution/kernel_output.py": ("consume_kernel_output",),
        "src/_lcm/solution/backward_induction.py": (
            "solve",
            "_run_dispatch_unit",
            "_accumulate_landing_totals",
            "_query_block_status",
        ),
        "src/_lcm/simulation/programs.py": (
            "build_simulation_programs",
            "_decision_programs",
            "_bound_output_roles",
            "_flatten_subject_integrity",
            "_decision_body",
            "with_kept_action_parts",
            "_SelectedOutputs.__call__",
            "_decision_with_action_parts",
            "_DecisionWithActionParts.__call__",
            "_StreamedArgmaxQOverA.__call__",
            "_SubjectTiled.__call__",
        ),
        "src/_lcm/simulation/runtime.py": ("_SimulationCandidateCompiler._bound",),
        "src/_lcm/simulation/simulate.py": ("_canonical_Q_at_actions",),
        "src/_lcm/simulation/initial_conditions.py": (
            "_regime_feasibility_mask",
            "_is_any_action_feasible",
            "_is_feasible_without_actions",
            "_admits_any_action",
            "_validate_initial_bound_status",
        ),
        "src/_lcm/transition_checks.py": (
            "validate_off_grid_transitions",
            "_off_grid_target_counts",
            "_off_grid_consumed_names",
            "_repeat_off_grid_counts",
            "_merge_off_grid_counts",
            "_build_off_grid_cell",
            "_OffGridCell.__call__",
            "_off_grid_counts_law",
            "_NamedArgumentsCall.__call__",
            "_GridPointCall.__call__",
            "_off_grid_block_candidates",
            "_validate_off_grid_bound_status",
        ),
        "src/_lcm/simulation/chunk_profiles.py": (
            "profile_simulation_chunk",
            "_record_core",
        ),
        "src/_lcm/simulation/host_operations.py": (
            "ProfiledSimulationOperations.built",
            "_bound_identity",
            "_static_identity",
        ),
        "src/_lcm/simulation/forward_program_profiles.py": ("profile_forward_unit",),
        "src/lcm/model.py": ("Model._lookup_policy",),
        "src/_lcm/utils/functools.py": ("publish_signature",),
        "src/_lcm/solution/landing_counts.py": (
            "landing_checks_of",
            "get_landing_counting_max_Q_over_a",
            "_LandingCountingReduction.__call__",
            "_LandingCountingFold.__call__",
            "LandingCountedGridSearch.__call__",
            "_feasible_landing_counts",
            "fail_on_reached_landings",
            "_Evaluated.__call__",
        ),
        "src/_lcm/egm/outer_envelope.py": ("stack_candidate_carries",),
        "src/_lcm/regime_building/processing.py": ("_prepare_dcegm_query_models",),
    }
)


def _s02_query_status_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Authenticate selected imported query/status helpers and their declarations."""
    surface, callables = _S02_QUERY_STATUS_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="S02 query/status dependency", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(
            f"S02 query/status dependency: {source} declaration surface changed"
        )
    return errors


_ASSOCIATED_SAVING_OVERLOAD_COUNTS = {
    "envelope_at_query": 6,
    "empty_envelope_winner": 2,
    "merge_envelope_winner": 2,
    "finish_envelope_winner": 2,
}
_ASSOCIATED_SAVING_OVERLOAD_DECLARATIONS = "@overload\ndef envelope_at_query(\n    *,\n    endog_grid: Float1D,\n    policy: Float1D,\n    value: Float1D,\n    marginal: Float1D,\n    segment_id: Float1D,\n    stable_index: IntND | None = ...,\n    x_query: FloatND,\n    segment_block_size: int = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    return_owner: Literal[False] = ...,\n    candidate_savings: None = ...,\n) -> tuple[FloatND, FloatND, FloatND]: ...\n\n\n@overload\ndef envelope_at_query(\n    *,\n    endog_grid: Float1D,\n    policy: Float1D,\n    value: Float1D,\n    marginal: Float1D,\n    segment_id: Float1D,\n    stable_index: IntND | None = ...,\n    x_query: FloatND,\n    segment_block_size: int = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    return_owner: Literal[True],\n    candidate_savings: None = ...,\n) -> tuple[FloatND, FloatND, FloatND, IntND]: ...\n\n\n@overload\ndef envelope_at_query(\n    *,\n    endog_grid: Float1D,\n    policy: Float1D,\n    value: Float1D,\n    marginal: Float1D,\n    segment_id: Float1D,\n    stable_index: IntND | None = ...,\n    x_query: FloatND,\n    segment_block_size: int = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    return_owner: bool,\n    candidate_savings: None = ...,\n) -> tuple[FloatND, FloatND, FloatND] | tuple[FloatND, FloatND, FloatND, IntND]: ...\n\n\n@overload\ndef envelope_at_query(\n    *,\n    endog_grid: Float1D,\n    policy: Float1D,\n    value: Float1D,\n    marginal: Float1D,\n    segment_id: Float1D,\n    stable_index: IntND | None = ...,\n    x_query: FloatND,\n    segment_block_size: int = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    return_owner: Literal[False] = ...,\n    candidate_savings: Float1D,\n) -> tuple[FloatND, FloatND, FloatND, FloatND]: ...\n\n\n@overload\ndef envelope_at_query(\n    *,\n    endog_grid: Float1D,\n    policy: Float1D,\n    value: Float1D,\n    marginal: Float1D,\n    segment_id: Float1D,\n    stable_index: IntND | None = ...,\n    x_query: FloatND,\n    segment_block_size: int = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    return_owner: Literal[True],\n    candidate_savings: Float1D,\n) -> tuple[FloatND, FloatND, FloatND, FloatND, IntND]: ...\n\n\n@overload\ndef envelope_at_query(\n    *,\n    endog_grid: Float1D,\n    policy: Float1D,\n    value: Float1D,\n    marginal: Float1D,\n    segment_id: Float1D,\n    stable_index: IntND | None = ...,\n    x_query: FloatND,\n    segment_block_size: int = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    return_owner: bool = ...,\n    candidate_savings: Float1D,\n) -> (\n    tuple[FloatND, FloatND, FloatND, FloatND]\n    | tuple[FloatND, FloatND, FloatND, FloatND, IntND]\n): ...\n\n\n@overload\ndef empty_envelope_winner(\n    *, query: FloatND, retain_savings: Literal[False] = ...\n) -> EnvelopeWinner: ...\n\n\n@overload\ndef empty_envelope_winner(\n    *, query: FloatND, retain_savings: Literal[True]\n) -> _SavingsEnvelopeWinner: ...\n\n\n@overload\ndef merge_envelope_winner(\n    *,\n    held: EnvelopeWinner,\n    endog_grid: Float1D,\n    policy: Float1D,\n    value: Float1D,\n    marginal: Float1D,\n    segment_id: Float1D,\n    stable_index: IntND,\n    query: FloatND,\n    arithmetic: ComparisonArithmetic,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    candidate_savings: None = ...,\n) -> EnvelopeWinner: ...\n\n\n@overload\ndef merge_envelope_winner(\n    *,\n    held: _SavingsEnvelopeWinner,\n    endog_grid: Float1D,\n    policy: Float1D,\n    value: Float1D,\n    marginal: Float1D,\n    segment_id: Float1D,\n    stable_index: IntND,\n    query: FloatND,\n    arithmetic: ComparisonArithmetic,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    candidate_savings: Float1D,\n) -> _SavingsEnvelopeWinner: ...\n\n\n@overload\ndef finish_envelope_winner(\n    *,\n    winner: EnvelopeWinner,\n    query: FloatND,\n    arithmetic: ComparisonArithmetic,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n) -> tuple[FloatND, FloatND, FloatND]: ...\n\n\n@overload\ndef finish_envelope_winner(\n    *,\n    winner: _SavingsEnvelopeWinner,\n    query: FloatND,\n    arithmetic: ComparisonArithmetic,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n) -> tuple[FloatND, FloatND, FloatND, FloatND, IntND]: ...\n\n"
_ASSOCIATED_SAVING_CONTRACTS = _contracts(
    {
        ASSOCIATED_SAVING_SOURCE: (
            "envelope_at_query",
            "_savings_link_endpoints",
            "empty_envelope_winner",
            "merge_envelope_winner",
            "finish_envelope_winner",
            "_derived_subnormal_possible",
            "_envelope_exact",
            "_link_blocks",
            "_envelope_blocked_ordinary",
            "_ordinary_block_best",
            "_segment_links_from_candidates",
            "_link_geometry",
            "_along_link",
            "_endpoint_order",
            "_read_exact_channel",
            "_read_ordinary_channel",
            "_mask_infeasible_queries",
            "published_owner",
            "_subnormal_operand_present",
            "_quotient_is_subnormal",
            "_ordinary_block_size",
            "_pad_column",
            "_ordinary_block_step",
            "_block_query_terms",
            "_slope_words",
            "_batched_link_terms",
            "_ordinary_rank_argmax",
            "_pick_column",
            "_spread_over_queries",
            "_take_rows",
            "_publish_decided",
            "_link_terms",
            "_on_the_links_own_scale",
            "_smallest_finite_exponent",
        )
    }
)


_NB_PRODUCER_SOURCE = "src/_lcm/egm/nbegm_step.py"
_NB_PRODUCER_OVERLOAD_COUNTS = {
    "nbegm_multi_interval_step_savings": 5,
    "nbegm_per_interval_continuation_step_savings": 7,
}
_NB_PRODUCER_OVERLOAD_DECLARATIONS = "@overload\ndef nbegm_multi_interval_step_savings(\n    *,\n    cont_value: Float1D,\n    cont_marginal: Float1D,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    inverse_eis: ScalarFloat | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    query_grid: Float1D | None = ...,\n    return_owner: Literal[False] = ...,\n    retain_query: Literal[False] = ...,\n) -> tuple[Float1D, Float1D, Float1D]: ...\n\n\n@overload\ndef nbegm_multi_interval_step_savings(\n    *,\n    cont_value: Float1D,\n    cont_marginal: Float1D,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    inverse_eis: ScalarFloat | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    query_grid: Float1D | None = ...,\n    return_owner: Literal[False] = ...,\n    retain_query: Literal[True],\n) -> _NBEGMQueryStep: ...\n\n\n@overload\ndef nbegm_multi_interval_step_savings(\n    *,\n    cont_value: Float1D,\n    cont_marginal: Float1D,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    inverse_eis: ScalarFloat | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    query_grid: Float1D | None = ...,\n    return_owner: Literal[False] = ...,\n    retain_query: bool = ...,\n) -> tuple[Float1D, Float1D, Float1D] | _NBEGMQueryStep: ...\n\n\n@overload\ndef nbegm_multi_interval_step_savings(\n    *,\n    cont_value: Float1D,\n    cont_marginal: Float1D,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    inverse_eis: ScalarFloat | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    query_grid: Float1D | None = ...,\n    return_owner: Literal[True],\n    retain_query: Literal[True],\n) -> _OwnedNBEGMQueryStep: ...\n\n\n@overload\ndef nbegm_multi_interval_step_savings(\n    *,\n    cont_value: Float1D,\n    cont_marginal: Float1D,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    inverse_eis: ScalarFloat | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    query_grid: Float1D | None = ...,\n    return_owner: bool,\n    retain_query: bool,\n) -> tuple[Float1D, Float1D, Float1D] | _NBEGMQueryStep | _OwnedNBEGMQueryStep: ...\n\n\n@overload\ndef nbegm_per_interval_continuation_step_savings(\n    *,\n    cont_value: FloatND | None,\n    cont_marginal: FloatND | None,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    coh_grid: Float1D | None = ...,\n    envelope_segment_width: int = ...,\n    extra_savings: FloatND | None = ...,\n    extra_cont_value: FloatND | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    interval_block_reader: Callable[[IntND], _NBEGMIntervalRows] | None = ...,\n    interval_width: int = ...,\n    interval_indices: Int1D | None = ...,\n    query_status_slots: Literal[0] = ...,\n    collect_continuation_read_counts: Literal[False] = False,\n    return_owner: Literal[False] = ...,\n    retain_query: Literal[False] = ...,\n) -> tuple[Float1D, Float1D, Float1D]: ...\n\n\n@overload\ndef nbegm_per_interval_continuation_step_savings(\n    *,\n    cont_value: FloatND | None,\n    cont_marginal: FloatND | None,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    coh_grid: Float1D | None = ...,\n    envelope_segment_width: int = ...,\n    extra_savings: FloatND | None = ...,\n    extra_cont_value: FloatND | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    interval_block_reader: Callable[[IntND], _NBEGMIntervalRows] | None = ...,\n    interval_width: int = ...,\n    interval_indices: Int1D | None = ...,\n    query_status_slots: Literal[0] = ...,\n    collect_continuation_read_counts: Literal[False] = False,\n    return_owner: Literal[True],\n    retain_query: Literal[False] = ...,\n) -> tuple[Float1D, Float1D, Float1D, Int1D]: ...\n\n\n@overload\ndef nbegm_per_interval_continuation_step_savings(\n    *,\n    cont_value: FloatND | None,\n    cont_marginal: FloatND | None,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    coh_grid: Float1D | None = ...,\n    envelope_segment_width: int = ...,\n    extra_savings: FloatND | None = ...,\n    extra_cont_value: FloatND | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    interval_block_reader: Callable[[IntND], _NBEGMIntervalRows] | None = ...,\n    interval_width: int = ...,\n    interval_indices: Int1D | None = ...,\n    query_status_slots: Literal[0] = ...,\n    collect_continuation_read_counts: Literal[False] = False,\n    return_owner: bool,\n    retain_query: Literal[False] = ...,\n) -> tuple[Float1D, Float1D, Float1D] | tuple[Float1D, Float1D, Float1D, Int1D]: ...\n\n\n@overload\ndef nbegm_per_interval_continuation_step_savings(\n    *,\n    cont_value: FloatND | None,\n    cont_marginal: FloatND | None,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    coh_grid: Float1D | None = ...,\n    envelope_segment_width: int = ...,\n    extra_savings: FloatND | None = ...,\n    extra_cont_value: FloatND | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    interval_block_reader: Callable[[IntND], _NBEGMIntervalRows] | None = ...,\n    interval_width: int = ...,\n    interval_indices: Int1D | None = ...,\n    query_status_slots: Literal[0] = ...,\n    collect_continuation_read_counts: Literal[False] = False,\n    return_owner: Literal[False] = ...,\n    retain_query: Literal[True],\n) -> _NBEGMQueryStep: ...\n\n\n@overload\ndef nbegm_per_interval_continuation_step_savings(\n    *,\n    cont_value: FloatND | None,\n    cont_marginal: FloatND | None,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    coh_grid: Float1D | None = ...,\n    envelope_segment_width: int = ...,\n    extra_savings: FloatND | None = ...,\n    extra_cont_value: FloatND | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    interval_block_reader: Callable[[IntND], _NBEGMIntervalRows] | None = ...,\n    interval_width: int = ...,\n    interval_indices: Int1D | None = ...,\n    query_status_slots: Literal[0] = ...,\n    collect_continuation_read_counts: Literal[False] = False,\n    return_owner: Literal[True],\n    retain_query: Literal[True],\n) -> _OwnedNBEGMQueryStep: ...\n\n\n@overload\ndef nbegm_per_interval_continuation_step_savings(\n    *,\n    cont_value: FloatND | None,\n    cont_marginal: FloatND | None,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    coh_grid: Float1D | None = ...,\n    envelope_segment_width: int = ...,\n    extra_savings: FloatND | None = ...,\n    extra_cont_value: FloatND | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    interval_block_reader: Callable[[IntND], _NBEGMIntervalRows] | None = ...,\n    interval_width: int = ...,\n    interval_indices: Int1D | None = ...,\n    query_status_slots: int = ...,\n    collect_continuation_read_counts: bool = False,\n    return_owner: Literal[False] = ...,\n    retain_query: bool = ...,\n) -> _NBEGMUnownedIntervalResult: ...\n\n\n@overload\ndef nbegm_per_interval_continuation_step_savings(\n    *,\n    cont_value: FloatND | None,\n    cont_marginal: FloatND | None,\n    liquid_grid: Float1D,\n    savings_grid: Float1D,\n    discount_factor: ScalarFloat,\n    preferences: Preferences,\n    coh_slopes: Float1D,\n    coh_intercepts: Float1D,\n    breakpoints: Float1D,\n    coh_grid: Float1D | None = ...,\n    envelope_segment_width: int = ...,\n    extra_savings: FloatND | None = ...,\n    extra_cont_value: FloatND | None = ...,\n    arithmetic: ComparisonArithmetic = ...,\n    feasibility_partition: ResolvedAxisPartition | None = ...,\n    feasible_interval_mask: BoolND | None = ...,\n    interval_block_reader: Callable[[IntND], _NBEGMIntervalRows] | None = ...,\n    interval_width: int = ...,\n    interval_indices: Int1D | None = ...,\n    query_status_slots: int = ...,\n    collect_continuation_read_counts: bool = False,\n    return_owner: bool = ...,\n    retain_query: bool = ...,\n) -> _NBEGMIntervalResult: ...\n\n"
_RETAINED_PREFIX_CONTRACTS = _contracts(
    {
        "src/_lcm/utils/counts.py": ("sum_counts", "add_counts"),
        "src/_lcm/egm/nbegm_step.py": (
            "nbegm_multi_interval_step_savings",
            "nbegm_per_interval_continuation_step_savings",
            "_query_node_kind",
            "_make_nbegm_query_data",
            "_invert_euler_over_savings",
            "_invert_euler_at_node",
            "_ez_flow_power_structure",
            "_discounted_continuation",
            "_invert_coh_with_linear_extension",
            "_linear_extension",
            "_degenerate_inversion",
            "_next_segment_id",
            "_as_pairs",
            "_stack_flat",
            "_interval_corner_candidates",
            "_fixed_savings_corner",
            "_publish_no_action_carry",
            "_publish_interval_step",
            "_IntervalCaseSolver.__call__",
            "_solve_packed_interval",
            "_streamed_interval_block_layout",
            "_streamed_block_candidate_positions",
            "_StreamedIntervalLayout.total_candidates",
            "_streamed_interval_continuation_envelope",
            "_IntervalWinnerStep.__call__",
            "_kill_padding",
            "_point_candidate_segment_bases",
            "_savings_node_point_candidates_for_interval_block",
            "_savings_node_point_candidates",
        ),
        "src/_lcm/egm/nbegm_segments.py": (
            "affords_an_action",
            "mask_dead_candidates",
            "segment_ids_from_folds",
        ),
        "src/_lcm/egm/framed_arithmetic.py": (
            "framed_difference",
            "binade_exponent",
            "_dyadic_parts",
            "_two_diff",
        ),
        "src/_lcm/egm/ez_kernel.py": (
            "ez_consumption_from_euler",
            "ez_marginal_of_resource",
            "ez_period_value",
        ),
        "src/_lcm/power_mean.py": (
            "weighted_power_mean_of_pair",
            "_applied_to_anchor",
            "_anchor_primal",
            "_log_term",
        ),
        "src/_lcm/probability.py": (
            "log_of_nonnegative",
            "_log_of_nonnegative_bits",
            "_log_of_nonnegative_jvp",
            "rescaled_weight_pair",
        ),
    }
)


def _nb_producer_implementation(*, tree: ast.Module, name: str) -> ast.FunctionDef:
    """Select a declared NB runtime body beside exact ordered typing stubs."""
    if name not in _NB_PRODUCER_OVERLOAD_COUNTS:
        raise ValueError(f"{name} is not a reviewed retained producer overload")
    expected = [
        node
        for node in ast.parse(_NB_PRODUCER_OVERLOAD_DECLARATIONS).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    count = _NB_PRODUCER_OVERLOAD_COUNTS[name]
    bindings = _scope_binding_counts(tree.body)
    if (
        len(expected) != count
        or len(functions) != count + 1
        or bindings.get(name) != count + 1
        or bindings.get("overload") != 1
        or _relevant_imports(tree=tree, names={"overload"})
        != ["from typing import Literal, NamedTuple, cast, overload"]
        or [_ast_key(node) for node in functions[:count]]
        != [_ast_key(node) for node in expected]
        or functions[-1].decorator_list
    ):
        raise ValueError(f"{name} overload declarations changed")
    return functions[-1]


def _retained_prefix_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Check only newly selected bodies in the existing prefix canonical store."""
    surface, callables = _RETAINED_PREFIX_CONTRACTS[source]
    ordinary = {
        name: digest
        for name, digest in callables.items()
        if source != _NB_PRODUCER_SOURCE
        or (
            name not in _NB_PRODUCER_OVERLOAD_COUNTS
            and name != "_IntervalWinnerStep.__call__"
        )
    }
    errors = _exact_callable_errors(
        tree=tree, label="retained DC/NB dependency", contracts=ordinary
    )
    if source == _NB_PRODUCER_SOURCE:
        for name in _NB_PRODUCER_OVERLOAD_COUNTS:
            try:
                implementation = _nb_producer_implementation(tree=tree, name=name)
            except ValueError as error:
                errors.append(f"retained DC/NB dependency: {error}")
            else:
                if _callable_ast_sha256(implementation) != callables[name]:
                    errors.append(
                        f"retained DC/NB dependency: exact callable corridor {name!r} changed"
                    )
    if source == _NB_PRODUCER_SOURCE:
        name = "_IntervalWinnerStep.__call__"
        try:
            implementation = _select_interval_winner_call(
                tree=tree, checker=sys.modules[__name__]
            )
        except ValueError as error:
            errors.append(f"retained DC/NB dependency: {error}")
        else:
            if _callable_ast_sha256(implementation) != callables[name]:
                errors.append(
                    f"retained DC/NB dependency: exact callable corridor {name!r} changed"
                )
    if _transport_module_surface(tree) != surface:
        errors.append(
            f"retained DC/NB dependency: {source} declaration surface changed"
        )
    return errors


def _associated_saving_implementation(
    *, tree: ast.Module, name: str
) -> ast.FunctionDef:
    """Select one of four runtime bodies beside its exact ordered typing stubs."""
    if name not in _ASSOCIATED_SAVING_OVERLOAD_COUNTS:
        raise ValueError(f"{name} is not a reviewed associated saving overload")
    expected = [
        node
        for node in ast.parse(_ASSOCIATED_SAVING_OVERLOAD_DECLARATIONS).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    count = _ASSOCIATED_SAVING_OVERLOAD_COUNTS[name]
    bindings = _scope_binding_counts(tree.body)
    imports = _relevant_imports(tree=tree, names={"overload"})
    if (
        len(functions) != count + 1
        or len(expected) != count
        or bindings.get(name) != count + 1
        or bindings.get("overload") != 1
        or imports
        != [
            "from typing import Literal, NamedTuple, Protocol, cast, overload, runtime_checkable"
        ]
        or [_ast_key(node) for node in functions[:count]]
        != [_ast_key(node) for node in expected]
        or functions[-1].decorator_list
    ):
        raise ValueError(f"{name} overload declarations changed")
    return functions[-1]


def _associated_saving_errors(*, tree: ast.Module) -> list[str]:
    """Authenticate original savings through the existing selected envelope tuple."""
    surface, callables = _ASSOCIATED_SAVING_CONTRACTS[ASSOCIATED_SAVING_SOURCE]
    errors = _exact_callable_errors(
        tree=tree,
        label="associated saving dependency",
        contracts={
            name: digest
            for name, digest in callables.items()
            if name not in _ASSOCIATED_SAVING_OVERLOAD_COUNTS
        },
    )
    for name in _ASSOCIATED_SAVING_OVERLOAD_COUNTS:
        try:
            implementation = _associated_saving_implementation(tree=tree, name=name)
        except ValueError as error:
            errors.append(f"associated saving dependency: {error}")
        else:
            if _callable_ast_sha256(implementation) != callables[name]:
                errors.append(
                    f"associated saving dependency: exact callable corridor {name!r} changed"
                )
    if _transport_module_surface(tree) != surface:
        errors.append(
            f"associated saving dependency: {ASSOCIATED_SAVING_SOURCE} declaration surface changed"
        )
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


def _finite_replay_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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
        "_period_feasibility_counter_reservations",
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
                    _target_names(target) == ("output", "_blocked")
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
        expected = "8d3b408110025193608cc7e7a489eafd9f2def984ffcbfb9902ba9d72c03959e"
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
    tree = _candidate_value_projection(tree)
    label = "simulate" if simulate else "solve"
    errors: list[str] = []
    try:
        binding, nested = _ordinary_kernel(
            tree=tree,
            outer_name=outer_name,
            nested_name=nested_name,
            class_name=class_name,
            fields=fields if simulate else (*fields, "return_feasibility"),
        )
    except ValueError as error:
        return [f"{label}: {error}"]
    if not _exact_reducer_signature_call(
        call=binding, simulate=simulate, taste_shocks=False
    ):
        errors.append(f"{label}: ordinary reducer signature wrapper changed")
    body = _body_without_docstring(nested)
    if len(body) != 4:
        errors.append(
            f"{label}: ordinary reducer corridor must contain exactly origin, "
            f"collective branch, singleton reduction and return; found {len(body)} statements"
        )
        return errors
    origin, collective, *singleton = body
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
    singleton_ok = _statements_match(
        observed=singleton,
        expected_source=(
            "index, value = argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)\n"
            "return jnp.where(jnp.any(F_arr), index, -1), value"
            if simulate
            else "value = Q_arr.max(where=F_arr, initial=-jnp.inf)\n"
            "return (value, ~jnp.any(F_arr)) if self.return_feasibility else value"
        ),
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
    tree = _candidate_value_projection(tree)
    label = "taste-shock simulate" if simulate else "taste-shock solve"
    try:
        binding, nested = _taste_kernel(
            tree=tree,
            outer_name=outer_name,
            nested_name=nested_name,
            class_name=class_name,
            fields=fields if simulate else (*fields, "return_feasibility"),
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
return (smoothed, ~jnp.any(F_arr)) if self.return_feasibility else smoothed
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
return jnp.where(jnp.any(F_arr), flat_index, -1).astype(jnp.int32), Qc[discrete_argmax]
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
    expected_body = r"""if key.shape == (1,):
    gumbel = jax.custom_batching.sequential_vmap(
        functools.partial(jax.random.gumbel, shape=shape)
    )(key[0])
else:
    gumbel = jax.random.gumbel(key, shape)
return scale * (gumbel - EULER_GAMMA)
"""
    if not _body_matches(node=node, expected_source=expected_body):
        errors.append(
            "taste noise: body is not one scaled mean-zero Gumbel draw per cell"
        )

    imports = _relevant_imports(
        tree=tree,
        names={
            "cast",
            "functools",
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
            "import functools",
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
            "functools",
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
    return _argmax_and_max_of_one_candidate(a=a, where=where, initial=initial)
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


_CURRENT_IID_ADAPTER_CONTRACTS = _contracts(
    {
        "src/lcm/shocks.py": (
            "BaseInnovations.__post_init__",
            "PhysicalValues.__post_init__",
            "ShockGroup._correlation_factor",
            "ShockGroup._has_identity_correlation",
            "ShockGroup._realise",
            "ShockGroup.validate_runtime_correlation",
            "_continuous_value",
            "_correlation_pivots",
            "_law_validity",
            "_resolve_parameter",
            "_runtime_support_validity",
            "_validate_correlation",
            "_validate_override_array",
        ),
        "src/_lcm/shock_overrides.py": (
            "_flatten_override",
            "_flatten_override_with_keys",
            "_unflatten_override",
            "_inverse",
            "_recover_correlated",
            "apply_overrides",
        ),
        _CURRENT_IID_OVERRIDES_SOURCE: (
            "CurrentIIDOverrides.schema",
            "split_current_iid_inputs",
            "resolve_current_iid_overrides",
            "gather_current_iid_overrides",
            "current_iid_override_prototype",
        ),
        "src/_lcm/current_iid.py": (
            "CurrentIIDPlan.wrap",
            "CurrentIIDPlan.realise_nodes",
            "CurrentIIDPlan.realise_information_nodes",
            "law_parameters",
            "law_informativeness",
            "CurrentIIDDrawPlan.wrap",
            "_CurrentIIDExpectation.__call__",
            "_CurrentIIDDraw.__call__",
            "_evaluate_node",
            "CurrentIIDDrawPlan.realise",
            "_draw_base_value",
            "_law_inputs",
        ),
        "src/_lcm/simulation/current_iid.py": ("prepare_current_iid_keys",),
        "src/_lcm/regime_building/processing.py": (
            "_build_per_subject_decisions_per_period",
            "_build_simulation_phase",
        ),
        "src/_lcm/simulation/programs.py": (
            "_decision_programs",
            "_decision_subject_arg_names",
            "_decision_body",
        ),
        "src/_lcm/solution/feasibility.py": (
            "CountedGridSearch.__call__",
            "FeasibilityKernelOutput.__post_init__",
        ),
        "src/_lcm/solution/grid_search.py": (
            "grid_search_query_inputs",
            "_GridSearchArgumentBuilder.__call__",
            "_GridSearchPeriodKernel.__call__",
        ),
        "src/_lcm/solution/backward_induction.py": ("solve",),
        "src/_lcm/simulation/simulate.py": (
            "_current_iid_invalid",
            "_simulate_regime_in_period",
        ),
    }
)


def _current_iid_builder_guard_errors(tree: ast.Module) -> list[str]:
    """Check the exact optional adapter before permitting legacy projection."""
    errors: list[str] = []
    for name, reducer in (
        ("get_max_Q_over_a", "max_Q_over_a"),
        ("get_streaming_max_Q_over_a", "streamed_max_Q_over_a"),
    ):
        try:
            builder = _definition(tree=tree, name=name)
        except ValueError as error:
            errors.append(f"current IID adapter: {error}")
            continue
        slots = [
            (argument, default)
            for argument, default in zip(
                builder.args.kwonlyargs, builder.args.kw_defaults, strict=True
            )
            if argument.arg == "current_iid_plan"
        ]
        if (
            len(slots) != 1
            or _unparse(slots[0][1]) != "None"
            or _unparse(slots[0][0].annotation) != "CurrentIIDPlan | None"
        ):
            errors.append(
                f"current IID adapter: {name} optional plan signature changed"
            )
        expected = f"if current_iid_plan is not None:\n    {reducer} = current_iid_plan.wrap({reducer})"
        wrappers = [
            index
            for index, statement in enumerate(builder.body)
            if _statements_match(observed=[statement], expected_source=expected)
        ]
        mapping = [
            index
            for index, statement in enumerate(builder.body)
            if isinstance(statement, ast.Assign)
            and any(
                _target_names(target) == ("inner_state_names",)
                for target in statement.targets
            )
        ]
        if len(wrappers) != 1 or len(mapping) != 1 or wrappers[0] + 1 != mapping[0]:
            errors.append(
                f"current IID adapter: {name} must forward its canonical reducer immediately before real-state mapping"
            )
    return errors


def current_iid_adapter_errors(
    *,
    trees: Mapping[str, ast.Module],
    offending_paths: set[SourcePath] | None = None,
) -> list[str]:
    """Verify adapter identity and integrity, outside the law's semantic oracle.

    The adapter supplies physical inputs to the already-canonical full action
    reducer. This proof covers forwarding, guard/position, signatures and status;
    law transforms, quadrature and economic integration retain their semantic tests.
    """
    errors: list[str] = []
    offending = set() if offending_paths is None else offending_paths
    for source, (surface, callables) in _CURRENT_IID_ADAPTER_CONTRACTS.items():
        tree = trees.get(source)
        if tree is None:
            errors.append(f"current IID adapter: missing source {source}")
            offending.add(source)
            continue
        source_errors = _exact_callable_errors(
            tree=tree, label="current IID adapter", contracts=callables
        )
        if _transport_module_surface(tree) != surface:
            source_errors.append(
                f"current IID adapter: {source} declaration surface changed"
            )
        errors.extend(source_errors)
        if source_errors:
            offending.add(source)
    max_tree = trees.get(MAX_Q_SOURCE)
    if max_tree is None:
        offending.add(MAX_Q_SOURCE)
        return [*errors, "current IID adapter: missing canonical reducer builders"]
    guard_errors = _current_iid_builder_guard_errors(max_tree)
    errors.extend(guard_errors)
    if guard_errors:
        offending.add(MAX_Q_SOURCE)
    obligations = (
        (
            _CURRENT_IID_SOURCE,
            "_CurrentIIDDraw.__call__",
            "frame = self.plan.realise(keys=keys, group_params=group_params, arguments=arguments, overrides={name: BaseInnovations(values=cast('FloatND', item.values)) if item.kind == 'base' else PhysicalValues(values=cast('FloatND | IntND', item.values)) for name, item in overrides.items()}, override_active={name: cast('BoolND', item.active) for name, item in overrides.items()})",
        ),
        (
            _CURRENT_IID_SOURCE,
            "_CurrentIIDDraw.__call__",
            "physical, valid = frame.physical, frame.valid",
        ),
        (
            PROCESSING_SOURCE,
            "_build_per_subject_decisions_per_period",
            "if current_iid_draw_plan is not None:\n    built[q_id] = current_iid_draw_plan.wrap(built[q_id])",
        ),
        (
            SIMULATION_PROGRAMS_SOURCE,
            "_decision_body",
            "if context.current_iid_draw_plan is not None:\n    cell = context.current_iid_draw_plan.wrap(cell)",
        ),
        (
            _CURRENT_IID_SOURCE,
            "_CurrentIIDExpectation.__call__",
            "evaluated = jax.vmap(partial(_evaluate_node, function=self.function, arguments=arguments))(physical)",
        ),
        (
            _CURRENT_IID_SOURCE,
            "_integrate_nodes",
            "return jnp.sum(terms), jnp.any(support & empty), ~valid",
        ),
        (
            _CURRENT_IID_SOURCE,
            "_replay_output",
            "return index, value, dict(physical), ~valid",
        ),
        (
            FEASIBILITY_SOURCE,
            "CountedGridSearch.__call__",
            "if self.current_iid:\n    value, empty, invalid = cast('tuple[FloatND, BoolND, ScalarBool]', evaluated)\n    current_iid_invalid = jnp.any(invalid)\n    evaluated = (value, empty)",
        ),
        (
            GRID_SEARCH_SOURCE,
            "_GridSearchPeriodKernel.__call__",
            "if self.current_iid:\n    n_outputs = 4 + 2 * self.n_runtime_bounds + bool(self.landing_checks)\n    if not isinstance(out, tuple) or len(out) != n_outputs:\n        raise TypeError('Mandatory current-IID integrity output is missing.')\n    current_iid_invalid = cast('ScalarBool', out[3])\n    if current_iid_invalid.shape != () or current_iid_invalid.dtype != jnp.bool_:\n        raise TypeError('Current-IID integrity must be a scalar Boolean.')\n    out = (*out[:3], *out[4:])",
        ),
        (
            BACKWARD_INDUCTION_SOURCE,
            "solve",
            "if type(output) is FeasibilityKernelOutput and output.current_iid_invalid is not None and bool(output.current_iid_invalid):\n    raise ShockDeclarationError('Current-IID law is invalid or unresolved ' f'in {regime_name!r}, period {period}.')",
        ),
    )
    for source, owner, expected in obligations:
        tree = trees.get(source)
        if tree is None:
            errors.append(f"current IID adapter: missing source {source}")
            offending.add(source)
            continue
        try:
            if "." in owner:
                cls, method = owner.split(".", maxsplit=1)
                _, function = _method_definition(
                    tree=tree, class_name=cls, method_name=method
                )
            else:
                function = _definition(tree=tree, name=owner)
        except ValueError as error:
            errors.append(f"current IID adapter: {error}")
            offending.add(source)
            continue
        matches = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.stmt)
            and _statements_match(observed=[node], expected_source=expected)
        ]
        if len(matches) != 1:
            errors.append(
                f"current IID adapter: {owner} forwarding or mandatory integrity changed"
            )
            offending.add(source)
    kernel_tree = trees.get(GRID_SEARCH_SOURCE)
    if kernel_tree is not None:
        _, kernel = _method_definition(
            tree=kernel_tree,
            class_name="_GridSearchPeriodKernel",
            method_name="__call__",
        )
        publications = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and _call_name(node) == "FeasibilityKernelOutput"
            and _name(
                node=_keyword(call=node, name="current_iid_invalid"),
                expected="current_iid_invalid",
            )
        ]
        if len(publications) != 1:
            errors.append(
                "current IID adapter: solve publication omits mandatory integrity"
            )
            offending.add(GRID_SEARCH_SOURCE)
    simulation = trees.get(SIMULATION_SOURCE)
    if simulation is None:
        errors.append("current IID adapter: simulation publication is missing")
        offending.add(SIMULATION_SOURCE)
    else:
        function = _definition(tree=simulation, name="_simulate_regime_in_period")
        consumers = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _call_name(node) == "run_simulation_operation"
            and _name(
                node=_keyword(call=node, name="function"),
                expected="_current_iid_invalid",
            )
        ]
        if len(consumers) != 1:
            errors.append("current IID adapter: replay omits mandatory integrity")
            offending.add(SIMULATION_SOURCE)
    return errors


def verify_direct_candidate_flow(*, repo_root: Path) -> DirectFlowReport:
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
    adapter_errors = current_iid_adapter_errors(trees=parsed, offending_paths=offending)
    errors.extend(adapter_errors)
    decoupled_tree = parsed.get(DECOUPLED_BOUND_SOURCE)
    if decoupled_tree is not None:
        metadata_errors = _decoupled_bound_transport_errors(decoupled_tree)
        errors.extend(metadata_errors)
        if metadata_errors:
            offending.add(DECOUPLED_BOUND_SOURCE)
    for relative in _OUTER_HELPER_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _outer_helper_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _ADDITIONAL_HELPER_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _additional_helper_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _CHILD_NATIVE_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _child_native_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _FINAL_NESTED_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _final_nested_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
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
    feasibility_tree = parsed.get(FEASIBILITY_SOURCE)
    if feasibility_tree is not None:
        new_errors = _feasibility_summary_errors(feasibility_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(FEASIBILITY_SOURCE)
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
    for relative in _BORROWING_BOUND_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _borrowing_bound_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _REVIEWED_PREFIX_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _reviewed_prefix_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _S03_SCALAR_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _s03_scalar_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _UPSTREAM_COMPOSITION_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _upstream_composition_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    for relative in _RETAINED_PREFIX_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _retained_prefix_errors(tree=tree, source=relative)
            errors.extend(new_errors)
            if new_errors:
                offending.add(relative)
    associated_saving_tree = parsed.get(ASSOCIATED_SAVING_SOURCE)
    if associated_saving_tree is not None:
        new_errors = _associated_saving_errors(tree=associated_saving_tree)
        errors.extend(new_errors)
        if new_errors:
            offending.add(ASSOCIATED_SAVING_SOURCE)
    for relative in _S02_QUERY_STATUS_CONTRACTS:
        tree = parsed.get(relative)
        if tree is not None:
            new_errors = _s02_query_status_errors(tree=tree, source=relative)
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
        SIMULATION_RANDOM_ADDRESS_SOURCE,
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


def direct_flow_mutations(source: SourceText) -> dict[str, str]:
    """Generate the required route/value/support/shape/index perturbation family."""
    mutations: dict[str, str] = {}
    solve_singleton = "        value = Q_arr.max(where=F_arr, initial=-jnp.inf)"
    simulate_singleton = (
        "        index, value = argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)"
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
        "value = Q_arr.max(where=F_arr, initial=-jnp.inf)",
        "value = Q_arr.max(\n"
        "            where=F_arr.reshape(-1).at[0].set(False)\n"
        "            .reshape(F_arr.shape),\n"
        "            initial=-jnp.inf,\n"
        "        )",
        1,
    )
    mutations["singleton_simulate:inline_q_transform"] = source.replace(
        "index, value = argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)",
        "index, value = argmax_and_max(\n"
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
        "func=action_kernel,\n        variables=action_names,",
        "func=lambda **kwargs: action_kernel(**kwargs),\n        variables=action_names,",
        1,
    )
    # Replace the second productmap binding independently for simulate.
    first = source.find("func=action_kernel,\n        variables=action_names,")
    second = source.find(
        "func=action_kernel,\n        variables=action_names,", first + 1
    )
    if second < 0:
        raise ValueError("second Q_and_F productmap binding not found")
    mutations["simulate:wrapped_productmap_input"] = source[:second] + source[
        second:
    ].replace(
        "func=action_kernel,\n        variables=action_names,",
        "func=lambda **kwargs: action_kernel(**kwargs),\n        variables=action_names,",
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
        "output: MaxOutput = (\n            (smoothed, ~jnp.any(F_arr)) if self.return_feasibility else smoothed\n        )",
        "output = (Qc.reshape(-1)[0], ~jnp.any(F_arr)) if self.return_feasibility else Qc.reshape(-1)[0]",
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
        "output = (\n            jnp.where(jnp.any(F_arr), flat_index, -1).astype(jnp.int32),\n            Qc[discrete_argmax],\n        )",
        "output = jnp.where(jnp.any(F_arr), flat_index, -1).astype(jnp.int32), "
        "noisy_Qc[discrete_argmax]",
        1,
    )

    mutations["shared_taste_noise:shared_draw"] = source.replace(
        "gumbel = jax.random.gumbel(key, shape)",
        "gumbel = jnp.broadcast_to(jax.random.gumbel(key, (1,)), shape)",
        1,
    )
    mutations["shared_taste_noise:permuted_draw"] = source.replace(
        "gumbel = jax.random.gumbel(key, shape)",
        "gumbel = jax.random.gumbel(key, shape).reshape(-1)[::-1].reshape(shape)",
        1,
    )
    mutations["shared_taste_noise:first_candidate_zeroed"] = source.replace(
        "gumbel = jax.random.gumbel(key, shape)",
        "gumbel = jax.random.gumbel(key, shape).reshape(-1).at[0].set(0)"
        ".reshape(shape)",
        1,
    )
    mutations["shared_taste_noise:batched_independent_draw"] = source.replace(
        "gumbel = jax.custom_batching.sequential_vmap(\n"
        "            functools.partial(jax.random.gumbel, shape=shape)\n"
        "        )(key[0])",
        "gumbel = jax.random.gumbel(key[0], shape)",
        1,
    )
    mutations["shared_taste_noise:wrong_scale"] = source.replace(
        "return scale * (gumbel - EULER_GAMMA)",
        "return scale**2 * (gumbel - EULER_GAMMA)",
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
        marker="    mapped_action_kernel = productmap(",
        insertion="    action_names = action_names[:-1]\n",
        occurrence=1,
    )
    mutations["simulate:action_names_rebinding"] = _insert_before_nth(
        text=source,
        marker="    mapped_action_kernel = productmap(",
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
        marker="        evaluated = self.Q_and_F(",
        replacement="        evaluated = candidate_filter.Q_and_F(",
        occurrence=1,
    )
    mutations["taste_shock_simulate:attribute_q_and_f"] = _replace_nth(
        text=source,
        marker="        evaluated = self.Q_and_F(",
        replacement="        evaluated = candidate_filter.Q_and_F(",
        occurrence=3,
    )
    mutations["singleton_simulate:attribute_argmax_and_max"] = source.replace(
        "index, value = argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)",
        (
            "index, value = candidate_filter.argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)"
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
        marker="    mapped_action_kernel = productmap(",
        replacement="    mapped_action_kernel = candidate_filter.productmap(",
        occurrence=1,
    )
    mutations["simulate:attribute_productmap"] = _replace_nth(
        text=source,
        marker="    mapped_action_kernel = productmap(",
        replacement="    mapped_action_kernel = candidate_filter.productmap(",
        occurrence=2,
    )
    return mutations


_SUPPLEMENTAL_SOURCE_MUTATIONS = {
    "simulation_taste_stream:destination_name_ignored": (
        SIMULATION_TASTE_STREAM_SOURCE,
        'RandomAddress(kind="taste", name=regime).words()',
        'RandomAddress(kind="taste", name="ignored").words()',
    ),
    "s02_query:intrinsic_budget_owner_changed": (
        "src/_lcm/egm/budget.py",
        'DCEGM_BUDGET_CONSTRAINT_NAME: FunctionName = "dcegm_budget_constraint"',
        'DCEGM_BUDGET_CONSTRAINT_NAME: FunctionName = "unowned_budget"',
    ),
    "borrowing_bound:representative_marker_cache_identity_bypassed": (
        AGE_BOUND_NORMALIZATION_SOURCE,
        "value=user_regime.decomposed_functions.get(name, value)",
        "value=value",
    ),
    "borrowing_bound:scalar_shape_check_removed": (
        "src/_lcm/constraints/dynamic_bound.py",
        "value.ndim != 0 or not jnp.issubdtype(value.dtype, jnp.floating)",
        "not jnp.issubdtype(value.dtype, jnp.floating)",
    ),
    "borrowing_bound:parameter_owner_discarded": (
        "src/_lcm/constraints/dynamic_bound.py",
        '("parameter", regime_name, path[-1] if path else target, name)',
        '("parameter", regime_name, name)',
    ),
    "borrowing_bound:temporal_marker_discarded": (
        "src/_lcm/constraints/dynamic_bound.py",
        "return replace(\n            function,",
        "return AgeSpecializedFunction(",
    ),
    "borrowing_bound:forbidden_dependency_bypassed": (
        "src/_lcm/constraints/dynamic_bound.py",
        "if name in forbidden:",
        "if False and name in forbidden:",
    ),
    "borrowing_bound:literal_proof_value_substituted": (
        "src/_lcm/constraints/bounds.py",
        "jnp.asarray(declaration[1], dtype=dtype)",
        "jnp.asarray(bound_spec.literal_value, dtype=dtype)",
    ),
    "borrowing_bound:normalized_rhs_renamed": (
        "src/_lcm/constraints/processed.py",
        'Ref(f"{name}_lower")',
        'Ref(f"{name}_wrong")',
    ),
    "borrowing_bound:grid_origin_shifted": (
        "src/_lcm/post_decision_bound.py",
        "origin = float(grid.to_jax()[0])",
        "origin = float(grid.to_jax()[0]) + 1.0",
    ),
    "borrowing_bound:liquid_role_overwritten": (
        "src/lcm/consumption_savings_regime.py",
        "liquid_state=margin.state,",
        "liquid_state=margin.action,",
    ),
    "borrowing_bound:declared_producer_replaced_by_view": (
        "src/lcm/regime.py",
        "restored[name] = raw",
        "restored[name] = producer",
    ),
    "solve_completion:conflict_wait_bypassed": (
        "src/_lcm/execution/pending_work.py",
        "    owner.before(devices=devices)",
        "    owner.before(devices=frozenset())",
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
        (
            '"A decision chunk profile omitted its action decoder."\n'
            "                    )\n"
            "                actions = inventory.compiled(\n"
            '                    name="_lookup_values_from_indices",\n'
            "                    executable=decoder.executable,"
        ),
        (
            '"A decision chunk profile omitted its action decoder."\n'
            "                    )\n"
            "                actions = inventory.compiled(\n"
            '                    name="_lookup_values_from_indices",\n'
            "                    executable=decision.executable,"
        ),
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
        SIMULATION_RANDOM_ADDRESS_SOURCE,
        "jax.random.fold_in(jax.random.fold_in(key, high), low), period",
        "jax.random.fold_in(jax.random.fold_in(key, 0), low), period",
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
        (
            "        live=DeviceBufferFootprint(spans=MappingProxyType({})),\n"
            "    )\n"
            "    memory = SimulationMemory("
        ),
    ),
    "native_values:live_source_budget_omitted": (
        "src/_lcm/simulation/simulate.py",
        "                    live=inputs,\n                ),\n                inputs=inputs,",
        (
            "                    live=DeviceBufferFootprint(spans=MappingProxyType({})),\n"
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


_S02_HELPER_MUTATIONS = {
    "s02_query:auto_lower_target_omitted": (
        "src/_lcm/continuation.py",
        "_query_bound_evaluator_targets",
        "expression",
        '(*targets, ("lower_bound", bound.lower_target))',
        "targets",
        1,
    ),
    "s02_query:active_period_pool_omitted": (
        "src/_lcm/continuation.py",
        "_query_evaluator_contract",
        "expression",
        "tuple(sorted((functions_by_period or {}).items()))",
        "()",
        1,
    ),
    "s02_query:executable_ancestor_omitted": (
        "src/_lcm/continuation.py",
        "_query_evaluator_contract",
        "expression",
        "get_ancestors(pool, targets=target, include_targets=True)",
        "{target}",
        1,
    ),
    "s02_query:executable_aux_restored": (
        "src/_lcm/egm/query_data.py",
        "_flatten_query_record",
        "expression",
        "(tuple(getattr(record, item.name) for item in fields(record)), None)",
        "(tuple(getattr(record, item.name) for item in fields(record)), type(record))",
        1,
    ),
    "s02_query:expected_model_descriptor_bypassed": (
        "src/_lcm/solution/model_authority.py",
        "_validate_model_query_spec",
        "expression",
        "regime.solution.query_authority",
        "spec.layout.query_spec",
        1,
    ),
    "s02_status:active_lane_nonfinite_ignored": (
        "src/_lcm/solution/feasibility.py",
        "reduce_bound_input_status",
        "keyword",
        "nonfinite_floor",
        "jnp.any(~active & status.nonfinite_floor)",
        1,
    ),
    "s02_status:scan_history_retained": (
        "src/_lcm/utils/dispatchers.py",
        "_map_with_bound_summary",
        "expression",
        (
            "(BoundedResult(output=output, bound_status=statuses, query_statu"
            "s=merge_query_input_status(left=carry.query_status, right=reduce"
            "_query_input_status(statuses=result.query_status, active=valid))"
            ", continuation_read_counts=merge_continuation_read_counts(left="
            "carry.continuation_read_counts, right=reduce_continuation_read_co"
            "unts(counts=result.continuation_read_counts, active=valid))), No"
            "ne)"
        ),
        (
            "(BoundedResult(output=output, bound_status=statuses, query_statu"
            "s=merge_query_input_status(left=carry.query_status, right=reduce"
            "_query_input_status(statuses=result.query_status, active=valid))"
            ", continuation_read_counts=merge_continuation_read_counts(left="
            "carry.continuation_read_counts, right=reduce_continuation_read_co"
            "unts(counts=result.continuation_read_counts, active=valid))), st"
            "atuses)"
        ),
        1,
    ),
    "s02_status:subject_scalar_roles_ignored": (
        "src/_lcm/simulation/subject_parallel.py",
        "shard_subject_function",
        "keyword",
        "scalar_positions",
        "frozenset()",
        1,
    ),
    "s02_status:subject_collective_or_omitted": (
        "src/_lcm/simulation/subject_parallel.py",
        "_evaluate_local_subjects",
        "expression",
        'jax.lax.pmax(leaf.astype(jnp.int32), "X").astype(jnp.bool_)',
        "leaf",
        1,
    ),
    "s02_status:mandatory_host_gate_omitted": (
        "src/_lcm/solution/feasibility.py",
        "validate_runtime_bound_status",
        "expression",
        "nonfinite or invalid_grid",
        "False",
        1,
    ),
    "s02_query:typed_adapter_refusal_omitted": (
        "src/_lcm/solution/egm.py",
        "_EGMCore._prepare_child_read",
        "expression",
        "next_query_data is not None",
        "False",
        1,
    ),
    "s02_status:canonical_dag_declarations_omitted": (
        "src/_lcm/regime_building/Q_and_F.py",
        "runtime_bound_specs",
        "expression",
        "flow.bound_specs if type(flow) is _BoundUtilitiesAndF else ()",
        "()",
        1,
    ),
    "s02_status:kernel_status_channel_omitted": (
        "src/_lcm/solution/kernel_output.py",
        "consume_kernel_output",
        "keyword",
        "bound_status",
        "()",
        1,
    ),
    "s02_status:bounded_aux_executable": (
        "src/_lcm/solution/feasibility.py",
        "_flatten_bounded_result",
        "expression",
        "(children, None)",
        "(children, type(result))",
        1,
    ),
    "s02_status:bounded_keyed_leaf_swapped": (
        "src/_lcm/solution/feasibility.py",
        "_flatten_bounded_result_with_keys",
        "expression",
        'jax.tree_util.GetAttrKey("bound_status")',
        'jax.tree_util.GetAttrKey("output")',
        1,
    ),
    "s02_status:bounded_reconstruction_status_swapped": (
        "src/_lcm/solution/feasibility.py",
        "_unflatten_bounded_result",
        "expression",
        'object.__setattr__(result, "bound_status", leaves[1])',
        'object.__setattr__(result, "bound_status", leaves[0])',
        1,
    ),
    "s02_status:off_grid_host_helper_omits_status": (
        "src/_lcm/transition_checks.py",
        "_validate_off_grid_bound_status",
        "keyword",
        "statuses",
        "()",
        1,
    ),
    "s02_status:off_grid_host_call_seam_replaced": (
        "src/_lcm/transition_checks.py",
        "validate_off_grid_transitions",
        "expression",
        "_validate_off_grid_bound_status",
        "candidate_filter._validate_off_grid_bound_status",
        1,
    ),
    "s02_status:initial_host_helper_omits_status": (
        "src/_lcm/simulation/initial_conditions.py",
        "_validate_initial_bound_status",
        "keyword",
        "statuses",
        "()",
        1,
    ),
    "s02_status:initial_host_call_seam_replaced": (
        "src/_lcm/simulation/initial_conditions.py",
        "_regime_feasibility_mask",
        "expression",
        "_validate_initial_bound_status",
        "candidate_filter._validate_initial_bound_status",
        1,
    ),
    "s02_status:fold_weight_helper_changes_row": (
        "src/_lcm/solution/grid_search.py",
        "_grid_search_fold_weights",
        "expression",
        "process.get_transition_probs()[0]",
        "process.get_transition_probs()[-1]",
        1,
    ),
    "s02_status:fold_weight_call_seam_replaced": (
        "src/_lcm/solution/grid_search.py",
        "GridSearch.build_period_kernels",
        "expression",
        "_grid_search_fold_weights",
        "candidate_filter._grid_search_fold_weights",
        1,
    ),
    "s02_status:landing_total_helper_loses_left_bank": (
        "src/_lcm/solution/backward_induction.py",
        "_accumulate_landing_totals",
        "keyword",
        "left",
        "jnp.zeros_like(landing_totals)",
        1,
    ),
    "s02_status:landing_total_call_seam_replaced": (
        "src/_lcm/solution/backward_induction.py",
        "_run_dispatch_unit",
        "expression",
        "_accumulate_landing_totals",
        "candidate_filter._accumulate_landing_totals",
        1,
    ),
    "s02_status:signature_metadata_omitted": (
        "src/_lcm/utils/functools.py",
        "publish_signature",
        "expression",
        'object.__setattr__(target, "__signature__", signature)',
        "None",
        1,
    ),
    "s02_status:landing_reducer_status_omitted": (
        "src/_lcm/solution/landing_counts.py",
        "LandingCountedGridSearch.__call__",
        "keyword",
        "bound_status",
        "()",
        1,
    ),
    "s02_status:off_grid_eager_index_conversion_omitted": (
        "src/_lcm/transition_checks.py",
        "_off_grid_counts_law",
        "expression",
        "jnp.asarray(index, dtype=jnp.int32)",
        "index.astype(jnp.int32)",
        1,
    ),
    "s02_query:carry_query_record_discarded": (
        "src/_lcm/egm/carry.py",
        "EGMCarry.query_data_for_transport",
        "expression",
        "self.query_data",
        "None",
        1,
    ),
    "s02_query:outer_candidate_marginal_negated": (
        "src/_lcm/egm/outer_envelope.py",
        "stack_candidate_carries",
        "expression",
        "replace(candidate, value=jnp.where(jnp.isnan(leaf['value',]), -jnp.inf, leaf['value',]), marginal_utility=jnp.where(jnp.isnan(leaf['value',]), 0.0, leaf['marginal_utility',]))",
        "replace(candidate, value=jnp.where(jnp.isnan(leaf['value',]), -jnp.inf, leaf['value',]), marginal_utility=jnp.where(jnp.isnan(leaf['value',]), 0.0, -leaf['marginal_utility',]))",
        1,
    ),
    "s02_query:outer_query_record_call_seam_replaced": (
        "src/_lcm/egm/outer_envelope.py",
        "stack_candidate_carries",
        "expression",
        "nan_is_infeasible and candidate.query_data_for_transport() is None",
        "nan_is_infeasible and candidate_filter.query_data_for_transport() is None",
        1,
    ),
    "s02_query:builder_query_record_call_seam_replaced": (
        "src/_lcm/solution/egm.py",
        "_EGMArgumentBuilder.__call__",
        "expression",
        "next_carry.query_data_for_transport",
        "candidate_filter.query_data_for_transport",
        1,
    ),
}


# These contracts cover admitted Uniform support and its explicit consumer transport.
# Runtime tests establish numerical support, budget refusal and ownership lifetimes.
_UNIFORM_PROCESS_CONTRACTS: dict[str, tuple[str, dict[str, str]]] = _contracts(
    {
        "src/_lcm/engine.py": (
            "SolutionPhase.resolve_process_grids",
            "SolutionPhase.state_action_space",
            "resolve_phase_process_grids",
            "phase_state_action_space",
        ),
        "src/_lcm/simulation/eager_stages.py": (
            "_abstract_stage_operand",
            "run_eager_stage",
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


def _grouped_mapper_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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


def _action_grid_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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
        (
            "contextlib.closing("
            "PreflightActionGrids(memory=memory, build=_build_flat_action_grid))"
        ),
        (
            "contextlib.nullcontext("
            "PreflightActionGrids(memory=memory, build=_build_flat_action_grid))"
        ),
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


def _uniform_process_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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
        "resolve_phase_process_grids",
        "expression",
        "jax.sharding.NamedSharding(plan.mesh, jax.sharding.PartitionSpec())",
        "jax.sharding.SingleDeviceSharding(devices[0])",
        1,
    ),
    "uniform_grid:producer_output_layout_omitted": (
        EAGER_STAGE_SOURCE,
        "run_eager_stage",
        "keyword",
        "output_sharding",
        "None",
        1,
    ),
    "uniform_grid:selected_device_ignored": (
        ENGINE_SOURCE,
        "resolve_phase_process_grids",
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
        EAGER_STAGE_SOURCE,
        "run_eager_stage",
        "expression",
        "tuple(dict.fromkeys((*devices, *current.spans)))",
        "devices",
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
        EAGER_STAGE_SOURCE,
        "run_eager_stage",
        "before_assignment",
        "plan",
        "function(**placed, **static_arguments)",
        1,
    ),
    "uniform_grid:external_residency_ignored": (
        EAGER_STAGE_SOURCE,
        "run_eager_stage",
        "expression",
        "max(external.values())",
        "0",
        1,
    ),
    "uniform_grid:peak_admission_ignored": (
        EAGER_STAGE_SOURCE,
        "run_eager_stage",
        "keyword",
        "budget_bytes",
        "None",
        2,
    ),
    "uniform_grid:shared_cache_keeps_operands": (
        EAGER_STAGE_SOURCE,
        "run_eager_stage",
        "expression",
        "abstract",
        "placed",
        2,
    ),
    "uniform_grid:readiness_omitted": (
        EAGER_STAGE_SOURCE,
        "run_eager_stage",
        "expression",
        "jax.block_until_ready(result)",
        "result",
        1,
    ),
    "uniform_grid:caller_residency_omitted": (
        UNIFORM_PROCESS_GRID_SOURCE,
        "SimulationProcessGrids.snapshot",
        "expression",
        "self.live_footprint()",
        "DeviceBufferFootprint(spans=MappingProxyType({}))",
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
        "resolve_phase_process_grids",
        "expression",
        "process_grid_resolver.supports(spec)",
        "True",
        1,
    ),
    "uniform_grid:engine_recomputes_admitted_support": (
        ENGINE_SOURCE,
        "phase_state_action_space",
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
        "_raise_at",
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
        9,
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
        SIMULATION_MEMORY_SOURCE,
        "run_simulation_producer",
        "keyword",
        "budget_bytes",
        "None",
        2,
    ),
    "feasibility:live_owners_omitted": (
        SIMULATION_MEMORY_SOURCE,
        "run_simulation_producer",
        "expression",
        "max(external.values())",
        "0",
        1,
    ),
    "feasibility:concrete_inputs_closed_over": (
        SIMULATION_MEMORY_SOURCE,
        "run_simulation_producer",
        "expression",
        "dict(placed)",
        "{}",
        1,
    ),
    "feasibility:unprofiled_dispatch": (
        SIMULATION_MEMORY_SOURCE,
        "run_simulation_producer",
        "expression",
        "executable(**placed)",
        "function(**placed)",
        1,
    ),
    "feasibility:readiness_omitted": (
        SIMULATION_MEMORY_SOURCE,
        "run_simulation_producer",
        "expression",
        "jax.block_until_ready(result)",
        "None",
        1,
    ),
    "feasibility:result_lifetime_omitted": (
        SIMULATION_MEMORY_SOURCE,
        "run_simulation_producer",
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


def _action_partition_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
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
        "merge_partition_accumulators(accumulators=gathered, order=tuple(range(self.n_partitions)))",
        "merge_partition_accumulators(accumulators=gathered, order=tuple(reversed(range(self.n_partitions))))",
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
        if (
            relative == _NB_PRODUCER_SOURCE
            and qualname == "_IntervalWinnerStep.__call__"
        ):
            function = _select_interval_winner_call(
                tree=tree, checker=sys.modules[__name__]
            )
        elif "." in qualname:
            class_name, method_name = qualname.split(".", maxsplit=1)
            _, function = _method_definition(
                tree=tree, class_name=class_name, method_name=method_name
            )
        elif (
            relative == ASSOCIATED_SAVING_SOURCE
            and qualname in _ASSOCIATED_SAVING_OVERLOAD_COUNTS
        ):
            function = _associated_saving_implementation(tree=tree, name=qualname)
        elif (
            relative == _NB_PRODUCER_SOURCE and qualname in _NB_PRODUCER_OVERLOAD_COUNTS
        ):
            function = _nb_producer_implementation(tree=tree, name=qualname)
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


_REVIEWED_PREFIX_MUTATIONS = {
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
            "reduce_query_input_status(statuses=(status,), active=jnp.one"
            "s(liquid.shape, dtype=jnp.bool_))"
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


_REVIEWED_PREFIX_DIAGNOSTICS = {
    "reviewed_prefix:row_poison_not_legal": (
        "reviewed S04/S05 dependency: exact callable corridor 'DCEGM"
        "Query._read_at_floor_with_summary' changed"
    ),
    "reviewed_prefix:inactive_poison_not_reduced": (
        "reviewed S04/S05 dependency: exact callable corridor '_blend"
        "_owned_rows' changed"
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
        "reviewed S04/S05 dependency: exact callable corridor '_read_"
        "owned_row_with_support' changed"
    ),
    "reviewed_prefix:economic_rows_stay_live": (
        "reviewed S04/S05 dependency: exact callable corridor '_read_"
        "owned_row_with_support' changed"
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


_REVIEWED_PREFIX_SOURCE_MUTATIONS = {
    "reviewed_prefix:model_authority_immutable": (
        "src/_lcm/egm/dcegm_query_model.py",
        "@dataclass(frozen=True, kw_only=True)",
        "@dataclass(frozen=False, kw_only=True)",
    ),
}

EXPECTED_REVIEWED_PREFIX_MUTATION_COUNT = 28
EXPECTED_REVIEWED_PREFIX_MUTATION_NAMES_SHA256 = (
    "71a70a218ef33bcd2ac4506902d63d3a4fb1f5a51a021a56cb4ccc9f691726cf"
)


def reviewed_prefix_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build controls for owned row mathematics and immutable model transport."""
    specs = _callable_mutation_specs(
        repo_root=repo_root, mutations=_REVIEWED_PREFIX_MUTATIONS
    )
    for name, (relative, old, new) in _REVIEWED_PREFIX_SOURCE_MUTATIONS.items():
        source = (repo_root / relative).read_text(encoding="utf-8")
        if source.count(old) != 1:
            raise ValueError(f"{name}: expected one exact declaration anchor")
        specs[name] = {"path": relative, "source": source.replace(old, new, 1)}
    return specs


EXPECTED_S03_SCALAR_MUTATION_COUNT = 62
EXPECTED_S03_SCALAR_MUTATION_NAMES_SHA256 = (
    "41e1bded6af2f4b3c42c59c487f00c4dbe9c9e9e105ed12260b31a0cb1db30df"
)


def s03_scalar_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build separate named helper-body controls for the scalar-query corridor."""
    specs = {}
    for relative, (_, callables) in _S03_SCALAR_CONTRACTS.items():
        original = (repo_root / relative).read_text(encoding="utf-8")
        for qualname in callables:
            tree = ast.parse(original, filename=relative)
            if "." in qualname:
                class_name, method_name = qualname.split(".", maxsplit=1)
                _, function = _method_definition(
                    tree=tree, class_name=class_name, method_name=method_name
                )
            else:
                function = _definition(tree=tree, name=qualname)
            function.body = ast.parse("return None").body
            mutated = ast.unparse(tree) + "\n"
            if mutated == original:
                raise ValueError(f"S03 helper mutation did not change {qualname}")
            specs[f"s03_scalar:{relative}:{qualname}"] = {
                "path": relative,
                "source": mutated,
            }
    return specs


_OWNED_QUERY_HELPER_MUTATIONS = {
    "owned_query_status:pickle_wrong_record": (
        "src/_lcm/solution/feasibility.py",
        "QueryInputStatus.__reduce__",
        "expression",
        "_unflatten_query_input_status",
        "_unflatten_bound_status",
        1,
    ),
    "owned_query_status:flatten_floor_grid_swapped": (
        "src/_lcm/solution/feasibility.py",
        "_flatten_query_input_status",
        "expression",
        "tuple(status)",
        "(status.invalid_shifted_grid, status.nonfinite_floor, status.coverage_missing)",
        1,
    ),
    "owned_query_status:keyed_flatten_floor_grid_swapped": (
        "src/_lcm/solution/feasibility.py",
        "_flatten_query_input_status_with_keys",
        "expression",
        "zip(QueryInputStatus._fields, status, strict=True)",
        "zip(QueryInputStatus._fields, (status.invalid_shifted_grid, status.nonfinite_floor, status.coverage_missing), strict=True)",
        1,
    ),
    "owned_query_status:unflatten_floor_grid_swapped": (
        "src/_lcm/solution/feasibility.py",
        "_unflatten_query_input_status",
        "expression",
        "tuple.__new__(QueryInputStatus, leaves)",
        "tuple.__new__(QueryInputStatus, (leaves[1], leaves[0], leaves[2]))",
        1,
    ),
    "owned_query_status:reducer_inactive_lanes": (
        "src/_lcm/solution/feasibility.py",
        "reduce_query_input_status",
        "expression",
        "active & flag",
        "~active & flag",
        1,
    ),
    "owned_query_status:merge_requires_both_windows": (
        "src/_lcm/solution/feasibility.py",
        "merge_query_input_status",
        "expression",
        "a | b",
        "a & b",
        1,
    ),
    "owned_query_status:validator_arithmetic_gate_removed": (
        "src/_lcm/solution/feasibility.py",
        "validate_query_input_status",
        "expression",
        "nonfinite or invalid_grid",
        "False",
        1,
    ),
    "owned_query_status:validator_coverage_gate_removed": (
        "src/_lcm/solution/feasibility.py",
        "validate_query_input_status",
        "expression",
        "missing",
        "False",
        1,
    ),
    "owned_query_status:block_slot_count_ignored": (
        "src/_lcm/solution/backward_induction.py",
        "_query_block_status",
        "expression",
        "len(statuses or ()) != n_query_slots",
        "False",
        1,
    ),
    "owned_query_authority:topology_schema_not_checked": (
        "src/_lcm/continuation.py",
        "EGMRowTopologySpec.__post_init__",
        "expression",
        "self.schema_version != 1",
        "False",
        1,
    ),
    "owned_query_authority:topology_exact_type_not_checked": (
        "src/_lcm/continuation.py",
        "_validate_row_topology",
        "expression",
        "type(topology) is not EGMRowTopologySpec",
        "False",
        1,
    ),
    "owned_query_authority:topology_authority_not_compared": (
        "src/_lcm/continuation.py",
        "_validate_row_topology_authority",
        "expression",
        "not same",
        "False",
        1,
    ),
    "owned_query_authority:dc_later_periods_omitted": (
        "src/_lcm/regime_building/processing.py",
        "_prepare_dcegm_query_models",
        "keyword",
        "target_periods",
        "(periods[0],)",
        1,
    ),
}

_OWNED_QUERY_PRIORITY_MUTATION = (
    "owned_query_status:validator_coverage_before_arithmetic"
)
EXPECTED_OWNED_QUERY_HELPER_MUTATION_COUNT = 14
EXPECTED_OWNED_QUERY_HELPER_MUTATION_NAMES_SHA256 = (
    "0a4bd8a9400c984913740b6e58ac399e660bb5b39c41d1b6b2580388d196cb07"
)


def owned_query_helper_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build separate body controls for owned query integrity and authority."""
    specs = _callable_mutation_specs(
        repo_root=repo_root, mutations=_OWNED_QUERY_HELPER_MUTATIONS
    )
    relative = "src/_lcm/solution/feasibility.py"
    tree = ast.parse((repo_root / relative).read_text(encoding="utf-8"))
    function = _definition(tree=tree, name="validate_query_input_status")
    loop_indices = [
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.For)
    ]
    if len(loop_indices) != 2:
        raise ValueError("Query integrity priority requires exactly two owned loops.")
    first, second = loop_indices
    function.body[first], function.body[second] = (
        function.body[second],
        function.body[first],
    )
    specs[_OWNED_QUERY_PRIORITY_MUTATION] = {
        "path": relative,
        "source": ast.unparse(tree) + "\n",
    }
    return specs


_CHILD_FAILURE_MUTATIONS = {
    "owned_child_status:never_detected": False,
    "owned_child_status:always_detected": True,
}
EXPECTED_CHILD_FAILURE_MUTATION_COUNT = 2
EXPECTED_CHILD_FAILURE_MUTATION_NAMES_SHA256 = (
    "cfbf4195d3fb235d7ccec0a05d7d79712581053207d0ec36cfaeffaa9592b5df"
)


def child_failure_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build separate controls for mandatory-child and clean-coverage precedence."""
    relative = "src/_lcm/solution/egm.py"
    original = (repo_root / relative).read_text(encoding="utf-8")
    specs = {}
    for name, failure in _CHILD_FAILURE_MUTATIONS.items():
        tree = ast.parse(original, filename=relative)
        function = _definition(tree=tree, name="_has_mandatory_child_failure")
        original_body = tuple(_ast_key(statement) for statement in function.body)
        function.body = [ast.Return(value=ast.Constant(value=failure))]
        if tuple(_ast_key(statement) for statement in function.body) == original_body:
            raise ValueError(f"Child failure mutation did not change {name}")
        specs[name] = {
            "path": relative,
            "source": ast.unparse(ast.fix_missing_locations(tree)) + "\n",
        }
    return specs


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
    specs.update(
        _callable_mutation_specs(repo_root=repo_root, mutations=_S02_HELPER_MUTATIONS)
    )
    return specs


_ASSOCIATED_SAVING_MUTATIONS = {
    "associated_saving:endpoint_original_ordinates": (
        "src/_lcm/egm/upper_envelope/query.py",
        "_savings_link_endpoints",
        "expression",
        "candidate_savings[1:]",
        "candidate_savings[:-1]",
        1,
    ),
    "associated_saving:exact_same_selected_column": (
        "src/_lcm/egm/upper_envelope/query.py",
        "_envelope_exact",
        "expression",
        "take(column=savings_endpoints[0])",
        "take(column=links.left_policy)",
        1,
    ),
    "associated_saving:exact_status_coupled": (
        "src/_lcm/egm/upper_envelope/query.py",
        "_envelope_exact",
        "expression",
        "coupled_status | savings_status",
        "coupled_status",
        1,
    ),
    "associated_saving:ordinary_block_endpoint": (
        "src/_lcm/egm/upper_envelope/query.py",
        "_ordinary_block_best",
        "expression",
        "block[:, 9]",
        "block[:, 8]",
        1,
    ),
    "associated_saving:ordinary_payload_take": (
        "src/_lcm/egm/upper_envelope/query.py",
        "_ordinary_block_step",
        "expression",
        "jnp.where(take, offered_channel, standing_channel)",
        "offered_channel",
        1,
    ),
    "associated_saving:stream_endpoint_pairs": (
        "src/_lcm/egm/upper_envelope/query.py",
        "merge_envelope_winner",
        "expression",
        "zip(savings_held[1:], savings_links, strict=True)",
        "zip(savings_held[1:][::-1], savings_links[::-1], strict=True)",
        1,
    ),
    "associated_saving:descending_and_point_owner": (
        "src/_lcm/egm/upper_envelope/query.py",
        "_endpoint_order",
        "expression",
        "(lower, jnp.where(zero_width, lower, upper))",
        "(flat_left, flat_right)",
        1,
    ),
    "associated_saving:legacy_absence_shape": (
        "src/_lcm/egm/upper_envelope/query.py",
        "empty_envelope_winner",
        "expression",
        "retain_savings",
        "True",
        1,
    ),
}

_ASSOCIATED_SAVING_DIAGNOSTICS = {
    "associated_saving:endpoint_original_ordinates": "associated saving dependency: exact callable corridor '_savings_link_endpoints' changed",
    "associated_saving:exact_same_selected_column": "associated saving dependency: exact callable corridor '_envelope_exact' changed",
    "associated_saving:exact_status_coupled": "associated saving dependency: exact callable corridor '_envelope_exact' changed",
    "associated_saving:ordinary_block_endpoint": "associated saving dependency: exact callable corridor '_ordinary_block_best' changed",
    "associated_saving:ordinary_payload_take": "associated saving dependency: exact callable corridor '_ordinary_block_step' changed",
    "associated_saving:stream_endpoint_pairs": "associated saving dependency: exact callable corridor 'merge_envelope_winner' changed",
    "associated_saving:descending_and_point_owner": "associated saving dependency: exact callable corridor '_endpoint_order' changed",
    "associated_saving:legacy_absence_shape": "associated saving dependency: exact callable corridor 'empty_envelope_winner' changed",
}

EXPECTED_ASSOCIATED_SAVING_MUTATION_COUNT = 8
EXPECTED_ASSOCIATED_SAVING_MUTATION_NAMES_SHA256 = (
    "10413feac535a08f59a86aff734259d02f0b8b7e5ea609ecf41c32060402897a"
)


def associated_saving_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Build the eight exact original-saving association controls."""
    return _callable_mutation_specs(
        repo_root=repo_root, mutations=_ASSOCIATED_SAVING_MUTATIONS
    )


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
    lcm_typing_source = (root / LCM_TYPING_SOURCE).read_text(encoding="utf-8")
    namespace_source = (root / NAMESPACE_SOURCE).read_text(encoding="utf-8")
    pandas_utils_source = (root / PANDAS_UTILS_SOURCE).read_text(encoding="utf-8")
    model_processing_source = (root / MODEL_PROCESSING_SOURCE).read_text(
        encoding="utf-8"
    )
    specs: dict[str, dict[str, str]] = {
        name: {"path": MAX_Q_SOURCE, "source": mutated}
        for name, mutated in direct_flow_mutations(max_source).items()
    }

    def replace_once(*, source: SourceText, old: str, new: str, label: str) -> str:
        if source.count(old) != 1:
            raise ValueError(
                f"{label}: expected one mutation marker, found {source.count(old)}"
            )
        return source.replace(old, new, 1)

    feasibility_source = (root / FEASIBILITY_SOURCE).read_text(encoding="utf-8")
    specs["feasibility_summary:published_value_changed"] = {
        "path": FEASIBILITY_SOURCE,
        "source": replace_once(
            source=feasibility_source,
            old="else (value, empty_count, other_inf_count)",
            new="else (-value, empty_count, other_inf_count)",
            label="feasibility summary unchanged value",
        ),
    }
    specs["feasibility_summary:empty_count_reversed"] = {
        "path": FEASIBILITY_SOURCE,
        "source": replace_once(
            source=feasibility_source,
            old="empty_count = jnp.sum(empty, dtype=jnp.int32)",
            new="empty_count = jnp.sum(~empty, dtype=jnp.int32)",
            label="feasibility summary native count",
        ),
    }

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
            old="            period_kernels=MappingProxyType(result),",
            new="            period_kernels=MappingProxyType({}),",
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
            old="                    if context.stakeholders is not None\n",
            new="                    if False\n",
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
            old=(
                "            value=V_arr,\n"
                "            current_iid_invalid=current_iid_invalid,\n"
                "            feasibility_counts=(empty_count, other_inf_count),"
            ),
            new=(
                "            value=candidate_filter(V_arr),\n"
                "            current_iid_invalid=current_iid_invalid,\n"
                "            feasibility_counts=(empty_count, other_inf_count),"
            ),
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
        "value_access:grid_reachable_targets_dropped": replace_once(
            source=grid_source,
            old=(
                "    reads = _value_reads(\n"
                "        regime_name=context.regime_name,\n"
                "        period=period,\n"
                "        target_regimes=targets,"
            ),
            new=(
                "    reads = _value_reads(\n"
                "        regime_name=context.regime_name,\n"
                "        period=period,\n"
                "        target_regimes=targets[:-1],"
            ),
            label="GridSearch reachable value-read targets",
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
            old=("                value_reads=value_reads,"),
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
        '            mapped=cast("Callable[..., BoundedMaxOutput]", mapped),\n'
        "            fold_state_names=fold_state_names,\n"
        "            fold_weights=fold_weights,\n"
        "            fold_conditioning=fold_conditioning,\n"
        "            inner_state_names=inner_state_names,\n"
        "            return_feasibility=return_feasibility,\n"
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
            '    return cast(\n        "MaxQOverAFunction[BoundedMaxOutput]",\n        allow_only_kwargs(func=mapped, enforce=False),\n    )'
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
            marker='            mapped=cast("Callable[..., BoundedMaxOutput]", mapped),',
            replacement=(
                '            mapped=cast("Callable[..., BoundedMaxOutput]", '
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
            old='        ),\n        kwargs=[\n            "next_regime_to_V_arr",',
            new='        ),\n        args=[\n            "next_regime_to_V_arr",',
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
                "        out = self.mapped(\n"
                "            next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params\n"
                "        )"
            ),
            new=(
                "        out = self.mapped(\n"
                "            next_regime_to_V_arr=next_regime_to_V_arr,\n"
                "            **{\n"
                "                name: value\n"
                "                for name, value in states_actions_params.items()\n"
                "                if name != '_lcm_action_block_width'\n"
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
            old="                ev1_output = ev1_result.smoothed_value",
            new="                ev1_output = -ev1_result.smoothed_value",
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
            marker="    output = (values, feasible & valid, global_ids)",
            replacement="    output = (values, feasible, global_ids)",
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
                marker="    output = (values, feasible & valid, global_ids)",
                replacement=(
                    "    feasible = feasible & (global_ids != "
                    f"{index})\n    output = (values, feasible & valid, global_ids)"
                ),
                occurrence=1,
            )
        )
        action_streaming_cases[f"streaming_ev1:candidate_index_{index}"] = _replace_nth(
            text=action_streaming_source,
            marker="    output = (values, feasible & valid, global_ids)",
            replacement=(
                "    feasible = feasible & (global_ids != "
                f"{index})\n    output = (values, feasible & valid, global_ids)"
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
                        "return _SubjectTiled(\n            func=dense_reducer,\n            subject_arg_names=subject_arg_names,\n            n_bound_records=n_bound_records,\n            n_query_records=n_query_records,\n        )"
                    ),
                    new=(
                        "return _SubjectTiled(\n            func=candidate_filter(dense_reducer),\n            subject_arg_names=subject_arg_names,\n            n_bound_records=n_bound_records,\n            n_query_records=n_query_records,\n        )"
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
                    old="result.best_global_action_id.astype(jnp.int32)",
                    new="(result.best_global_action_id + 1).astype(jnp.int32)",
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
                    old="tiles = {name: kwargs.pop(name) for name in subject_arg_names}",
                    new="tiles = {name: kwargs.pop(name)[::-1] for name in subject_arg_names}",
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
                "path": SIMULATION_PROGRAMS_SOURCE,
                "source": replace_once(
                    source=simulation_programs_source,
                    old="        transition=MappingProxyType(transition),",
                    new="        transition=MappingProxyType(candidate_filter(transition)),",
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
                    old=(
                        "        program=programs.arrival_program(family=family, period=period)\n"
                        '        if family.startswith("arrival_entry:")\n'
                        "        else families[family][period],"
                    ),
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
            old="                action_names=state_action_space.action_names,\n"
            "                state_names=state_action_space.state_names,\n"
            "                n_discrete_action_axes=",
            new="                action_names=state_action_space.action_names[:-1],\n"
            "                state_names=state_action_space.state_names,\n"
            "                n_discrete_action_axes=",
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
            old="    phase = SimulationPhase(",
            new="    phase = candidate_filter.SimulationPhase(",
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
                old="xs=jnp.arange(n_cells, dtype=jnp.int32),\n            batch_size=width,",
                new="xs=jnp.arange(n_cells, dtype=jnp.int32),\n            batch_size=n_cells,",
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
                old="\n        base=base_state_action_space,",
                new=(
                    "\n        base=base_state_action_space.replace(continuous_actions=MappingProxyType({name: values.at[-1].set(values[0]) for name, values in base_state_action_space.continuous_actions.items()})),"
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
                    "    continuous_actions = {\n"
                    "        name: _grid_to_jax_or_placeholder(grids[name])\n"
                    "        if materializer is None\n"
                    "        else materializer.materialize_grid(spec=grids[name], placeholder=True)\n"
                    "        for name in variables.continuous_action_names\n"
                    "    }"
                ),
                new=(
                    "    continuous_actions = {\n"
                    "        name: _grid_to_jax_or_placeholder(grids[name])\n"
                    "        if materializer is None\n"
                    "        else materializer.materialize_grid(spec=grids[name], placeholder=True)\n"
                    "        for name in variables.continuous_action_names\n"
                    "    }\n"
                    "    continuous_actions = {name: values.at[-1].set(values[0]) for name, values in continuous_actions.items()}"
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
                old="            return self._simulate_runtime_regimes[key]",
                new="            return candidate_filter(\n"
                "                self._simulate_runtime_regimes[key]\n"
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
                old="row_low=rows[:, 1],",
                new=("row_low=jnp.roll(rows[:, 1], 1, axis=0),"),
                label="grouped subject innovation reassignment",
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
        "candidate_materialization:array_writer_dtype_widened": {
            "path": LCM_TYPING_SOURCE,
            "source": replace_once(
                source=lcm_typing_source,
                old="        self, *, value: np.ndarray | Array, dtype: np.dtype, name: str\n",
                new="        self, *, value: np.ndarray | Array, dtype: object, name: str\n",
                label="array writer accepts an uncanonical destination dtype",
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
        FEASIBILITY_SOURCE: feasibility_source,
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
        LCM_TYPING_SOURCE: lcm_typing_source,
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
    s03_scalar = s03_scalar_mutation_specs(repo_root=root)
    owned_query = owned_query_helper_mutation_specs(repo_root=root)
    child_failure = child_failure_mutation_specs(repo_root=root)
    reviewed_prefix = reviewed_prefix_mutation_specs(repo_root=root)
    decoupled_bound = decoupled_bound_mutation_specs(repo_root=root)
    retained_prefix = retained_prefix_mutation_specs(repo_root=root)
    associated_saving = associated_saving_mutation_specs(repo_root=root)
    outer_helpers = outer_helper_mutation_specs(repo_root=root)
    final_nested = final_nested_mutation_specs(repo_root=root)
    child_native = child_native_mutation_specs(repo_root=root)
    additional_helpers = additional_helper_mutation_specs(repo_root=root)
    current_iid = current_iid_mutation_specs(repo_root=root)
    if (set(specs) | set(supplemental)) & set(s03_scalar):
        raise ValueError("S03 scalar mutations overlap existing populations")
    if (set(specs) | set(supplemental) | set(s03_scalar)) & set(owned_query):
        raise ValueError("Owned query helper mutations overlap existing populations")
    if (set(specs) | set(supplemental) | set(s03_scalar) | set(owned_query)) & set(
        child_failure
    ):
        raise ValueError("Child failure mutations overlap existing populations")
    if set(reviewed_prefix) & (
        set(specs)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
    ):
        raise ValueError("Reviewed prefix mutations overlap existing populations")
    if set(decoupled_bound) & (
        set(specs)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
        | set(reviewed_prefix)
    ):
        raise ValueError("Decoupled bound mutations overlap existing populations")
    if (
        set(specs)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
        | set(reviewed_prefix)
        | set(decoupled_bound)
    ) & set(associated_saving):
        raise ValueError("Associated saving mutations overlap existing populations")
    if set(retained_prefix) & (
        set(specs)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
        | set(reviewed_prefix)
        | set(decoupled_bound)
        | set(associated_saving)
    ):
        raise ValueError("Retained prefix mutations overlap existing populations")
    if set(outer_helpers) & (
        set(specs)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
        | set(reviewed_prefix)
        | set(decoupled_bound)
        | set(associated_saving)
        | set(retained_prefix)
    ):
        raise ValueError("Outer helper mutations overlap existing populations")
    if set(final_nested) & (
        set(specs)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
        | set(reviewed_prefix)
        | set(decoupled_bound)
        | set(associated_saving)
        | set(retained_prefix)
        | set(outer_helpers)
    ):
        raise ValueError("Final nested mutations overlap existing populations")
    if set(child_native) & (
        set(specs)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
        | set(reviewed_prefix)
        | set(decoupled_bound)
        | set(associated_saving)
        | set(retained_prefix)
        | set(outer_helpers)
        | set(final_nested)
    ):
        raise ValueError("Child native mutations overlap existing populations")
    if set(additional_helpers) & (
        set(specs)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
        | set(reviewed_prefix)
        | set(decoupled_bound)
        | set(associated_saving)
        | set(retained_prefix)
        | set(outer_helpers)
        | set(final_nested)
        | set(child_native)
    ):
        raise ValueError("Additional helper mutations overlap existing populations")
    mutated_paths = {
        spec["path"]
        for spec in (
            *specs.values(),
            *supplemental.values(),
            *s03_scalar.values(),
            *owned_query.values(),
            *child_failure.values(),
            *reviewed_prefix.values(),
            *decoupled_bound.values(),
            *associated_saving.values(),
            *retained_prefix.values(),
            *outer_helpers.values(),
            *final_nested.values(),
            *child_native.values(),
            *additional_helpers.values(),
            *current_iid.values(),
        )
    }
    certified_paths = (
        set(_CERTIFIED_CORRIDOR_SOURCES)
        - set(_UNIFORM_PROCESS_SOURCES)
        - set(_ACTION_GRID_SOURCES)
    ) | {SUPPORT_AUTHORITY_SOURCE, SUPPORT_TRANSITION_CHECKS_SOURCE}
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


_CURRENT_IID_ADAPTER_MUTATIONS = (
    (
        "current_iid:missing_override_gather_proof",
        "src/_lcm/simulation/current_iid_overrides.py",
        "def gather_current_iid_overrides(",
        "def unavailable_override_gather_proof(",
    ),
    (
        "current_iid:missing_override_callable_proof",
        "src/_lcm/shock_overrides.py",
        "def apply_overrides(",
        "def unavailable_override_proof(",
    ),
    (
        "current_iid:missing_factor_callable_proof",
        "src/lcm/shocks.py",
        "def _correlation_factor(",
        "def _unavailable_factor_proof(",
    ),
    (
        "current_iid:omit_dense_wrap",
        "src/_lcm/regime_building/max_Q_over_a.py",
        "        max_Q_over_a = current_iid_plan.wrap(max_Q_over_a)",
        "        pass",
    ),
    (
        "current_iid:wrong_dense_guard",
        "src/_lcm/regime_building/max_Q_over_a.py",
        "if current_iid_plan is not None:",
        "if current_iid_plan is None:",
    ),
    (
        "current_iid:rewire_dense_reducer",
        "src/_lcm/regime_building/max_Q_over_a.py",
        "current_iid_plan.wrap(max_Q_over_a)",
        "current_iid_plan.wrap(Q_and_F)",
    ),
    (
        "current_iid:omit_stream_wrap",
        "src/_lcm/regime_building/max_Q_over_a.py",
        "        streamed_max_Q_over_a = current_iid_plan.wrap(streamed_max_Q_over_a)",
        "        pass",
    ),
    (
        "current_iid:rewire_physical_reducer",
        "src/_lcm/current_iid.py",
        "partial(_evaluate_node, function=self.function, arguments=arguments)",
        "partial(_evaluate_node, function=lambda **kwargs: (0.0, False), arguments=arguments)",
    ),
    (
        "current_iid:omit_solve_status",
        "src/_lcm/current_iid.py",
        "return jnp.sum(terms), jnp.any(support & empty), ~valid",
        "return jnp.sum(terms), jnp.any(support & empty), False",
    ),
    (
        "current_iid:omit_replay_status",
        "src/_lcm/current_iid.py",
        "return index, value, dict(physical), ~valid",
        "return index, value, dict(physical), False",
    ),
    (
        "current_iid:omit_publication_status",
        "src/_lcm/solution/grid_search.py",
        "current_iid_invalid=current_iid_invalid,",
        "current_iid_invalid=None,",
    ),
    (
        "current_iid:rewire_key_producer",
        "src/_lcm/simulation/current_iid.py",
        "function=draw_window_keys,",
        "function=lambda **kwargs: None,",
    ),
    (
        "current_iid:wrong_stream_guard",
        "src/_lcm/regime_building/max_Q_over_a.py",
        "if current_iid_plan is not None:\n        streamed_max_Q_over_a =",
        "if current_iid_plan is None:\n        streamed_max_Q_over_a =",
    ),
    (
        "current_iid:rewire_stream_reducer",
        "src/_lcm/regime_building/max_Q_over_a.py",
        "current_iid_plan.wrap(streamed_max_Q_over_a)",
        "current_iid_plan.wrap(Q_and_F)",
    ),
    (
        "current_iid:detour_host_solve_status",
        "src/_lcm/solution/backward_induction.py",
        "and bool(output.current_iid_invalid)",
        "and False and bool(output.current_iid_invalid)",
    ),
    (
        "current_iid:detour_host_replay_status",
        "src/_lcm/simulation/simulate.py",
        "function=_current_iid_invalid,",
        "function=lambda **kwargs: False,",
    ),
    (
        "current_iid:disable_occupant_status_reduction",
        "src/_lcm/simulation/simulate.py",
        "return jnp.any(invalid & in_regime)",
        "return jnp.asarray(False)",
    ),
    (
        "current_iid:drop_replay_physical_roles",
        "src/_lcm/simulation/programs.py",
        "if context.current_iid_draw_plan is not None\n                    else ()",
        "if False\n                    else ()",
    ),
    (
        "current_iid:missing_implementation_proof",
        "src/_lcm/current_iid.py",
        "__MISSING_IMPLEMENTATION_PROOF__",
        "",
    ),
)
_EXPECTED_CURRENT_IID_MUTATION_NAMES = frozenset(
    (
        "current_iid:missing_override_gather_proof",
        "current_iid:missing_override_callable_proof",
        "current_iid:missing_factor_callable_proof",
        "current_iid:detour_host_replay_status",
        "current_iid:detour_host_solve_status",
        "current_iid:disable_occupant_status_reduction",
        "current_iid:drop_replay_physical_roles",
        "current_iid:missing_implementation_proof",
        "current_iid:omit_dense_wrap",
        "current_iid:omit_publication_status",
        "current_iid:omit_replay_status",
        "current_iid:omit_solve_status",
        "current_iid:omit_stream_wrap",
        "current_iid:rewire_dense_reducer",
        "current_iid:rewire_key_producer",
        "current_iid:rewire_physical_reducer",
        "current_iid:rewire_stream_reducer",
        "current_iid:wrong_dense_guard",
        "current_iid:wrong_stream_guard",
    )
)
EXPECTED_CURRENT_IID_MUTATION_NAMES_SHA256 = (
    "e0c95ee607e1f8b3d1479ba048692d1b3e6e8ddcf081a59ff56da0875868bac5"
)


def current_iid_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Keep actual adapter defects distinct from all historical populations."""
    specs: dict[str, dict[str, str]] = {}
    for name, source, before, after in _CURRENT_IID_ADAPTER_MUTATIONS:
        original = (repo_root / source).read_text(encoding="utf-8")
        if name == "current_iid:missing_implementation_proof":
            mutated = ""
        else:
            if before not in original:
                raise ValueError(f"{name}: exact mutation anchor is missing")
            mutated = original.replace(before, after, 1)
        ast.parse(mutated, filename=source)
        specs[name] = {"path": source, "source": mutated}
    return specs


def run_direct_flow_mutation_controls(*, repo_root: Path) -> DirectMutationControls:
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
    s03_scalar = s03_scalar_mutation_specs(repo_root=root)
    owned_query = owned_query_helper_mutation_specs(repo_root=root)
    child_failure = child_failure_mutation_specs(repo_root=root)
    reviewed_prefix = reviewed_prefix_mutation_specs(repo_root=root)
    decoupled_bound = decoupled_bound_mutation_specs(repo_root=root)
    retained_prefix = retained_prefix_mutation_specs(repo_root=root)
    associated_saving = associated_saving_mutation_specs(repo_root=root)
    outer_helpers = outer_helper_mutation_specs(repo_root=root)
    final_nested = final_nested_mutation_specs(repo_root=root)
    child_native = child_native_mutation_specs(repo_root=root)
    additional_helpers = additional_helper_mutation_specs(repo_root=root)
    current_iid = current_iid_mutation_specs(repo_root=root)
    uniform = uniform_process_mutation_specs(repo_root=root)
    action_grid = action_grid_mutation_specs(repo_root=root)
    feasibility = feasibility_mutation_specs(repo_root=root)
    grouped = grouped_mapper_mutation_specs(repo_root=root)
    guard = grouped_guard_mutation_specs(repo_root=root)
    allocation = allocation_reservation_mutation_specs(repo_root=root)
    normal = normal_process_mutation_specs(repo_root=root)
    partition = action_partition_mutation_specs(repo_root=root)
    if set(final_nested) & (
        set(uniform)
        | set(action_grid)
        | set(feasibility)
        | set(grouped)
        | set(guard)
        | set(allocation)
        | set(normal)
        | set(partition)
    ):
        raise ValueError("Final nested mutations overlap additional populations")
    if set(child_native) & (
        set(registered)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
        | set(reviewed_prefix)
        | set(decoupled_bound)
        | set(associated_saving)
        | set(retained_prefix)
        | set(outer_helpers)
        | set(final_nested)
        | set(uniform)
        | set(action_grid)
        | set(feasibility)
        | set(grouped)
        | set(guard)
        | set(allocation)
        | set(normal)
        | set(partition)
    ):
        raise ValueError("Child native mutations overlap other populations")
    if set(additional_helpers) & (
        set(registered)
        | set(supplemental)
        | set(s03_scalar)
        | set(owned_query)
        | set(child_failure)
        | set(reviewed_prefix)
        | set(decoupled_bound)
        | set(associated_saving)
        | set(retained_prefix)
        | set(outer_helpers)
        | set(final_nested)
        | set(child_native)
        | set(uniform)
        | set(action_grid)
        | set(feasibility)
        | set(grouped)
        | set(guard)
        | set(allocation)
        | set(normal)
        | set(partition)
    ):
        raise ValueError("Additional helper mutations overlap other populations")
    existing_names = set().union(
        registered,
        supplemental,
        s03_scalar,
        owned_query,
        child_failure,
        reviewed_prefix,
        decoupled_bound,
        associated_saving,
        retained_prefix,
        outer_helpers,
        final_nested,
        child_native,
        additional_helpers,
        uniform,
        action_grid,
        feasibility,
        grouped,
        guard,
        allocation,
        normal,
        partition,
    )
    if set(current_iid) & existing_names:
        raise ValueError("Current-IID mutations overlap existing populations")
    cases: dict[str, MutationCase] = {}
    with tempfile.TemporaryDirectory() as raw:
        temp_root = Path(raw) / "repo"
        for relative, source in originals.items():
            target = temp_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
        for name, spec in (
            registered
            | supplemental
            | s03_scalar
            | owned_query
            | child_failure
            | reviewed_prefix
            | decoupled_bound
            | associated_saving
            | retained_prefix
            | outer_helpers
            | final_nested
            | child_native
            | additional_helpers
            | current_iid
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
            original_seal = _SOURCE_SEALS[relative]
            if (
                name in current_iid
                or name in outer_helpers
                or name in final_nested
                or name in child_native
                or name in additional_helpers
                or name in reviewed_prefix
                or name in decoupled_bound
                or name in retained_prefix
                or name.startswith(
                    (
                        "s02_query:",
                        "s02_status:",
                        "s03_scalar:",
                        "owned_query_status:",
                        "owned_query_authority:",
                        "owned_child_status:",
                        "reviewed_prefix:",
                        "associated_saving:",
                        "retained_prefix:",
                    )
                )
            ):
                _SOURCE_SEALS[relative] = sha256_file(target)
            try:
                result = verify_direct_candidate_flow(repo_root=temp_root)
            finally:
                _SOURCE_SEALS[relative] = original_seal
            rejected = not result["ok"]
            if name in child_failure:
                diagnostic = "S02 query/status dependency: exact callable corridor '_has_mandatory_child_failure' changed"
                rejected = (
                    diagnostic not in clean["errors"] and diagnostic in result["errors"]
                )
            if name in reviewed_prefix:
                diagnostic = _REVIEWED_PREFIX_DIAGNOSTICS[name]
                rejected = (
                    clean["ok"]
                    and diagnostic in result["errors"]
                    and not any(
                        "source seal mismatch" in error for error in result["errors"]
                    )
                )
            if name in decoupled_bound:
                rejected = (
                    clean["ok"]
                    and result["offending_paths"] == [DECOUPLED_BOUND_SOURCE]
                    and any(
                        "Decoupled bound metadata:" in error
                        for error in result["errors"]
                    )
                    and not any(
                        "source seal mismatch" in error for error in result["errors"]
                    )
                )
            if name in associated_saving:
                diagnostic = _ASSOCIATED_SAVING_DIAGNOSTICS[name]
                rejected = (
                    diagnostic not in clean["errors"]
                    and diagnostic in result["errors"]
                    and not any(
                        "source seal mismatch" in error for error in result["errors"]
                    )
                )
            if name in retained_prefix:
                diagnostic = _RETAINED_PREFIX_DIAGNOSTICS[name]
                rejected = (
                    clean["ok"]
                    and diagnostic in result["errors"]
                    and not any(
                        "source seal mismatch" in error for error in result["errors"]
                    )
                )
            if name in outer_helpers:
                control = _OUTER_HELPER_CONTROLS[name]
                rejected = _outer_helper_case_rejected(
                    clean=clean,
                    result=result,
                    source=control.source,
                    diagnostic=control.diagnostic,
                )
            if name in final_nested:
                rejected = _final_nested_case_rejected(
                    clean=clean,
                    result=result,
                    source=relative,
                    diagnostic=_final_nested_primary_diagnostic(name=name),
                )
            if name in child_native:
                rejected = _child_native_case_rejected(
                    clean=clean,
                    result=result,
                    source=relative,
                    diagnostic=_child_native_primary_diagnostic(name=name),
                )
            if name in additional_helpers:
                rejected = _additional_helper_case_rejected(
                    clean=clean,
                    result=result,
                    source=relative,
                    diagnostic=_additional_helper_primary_diagnostic(name=name),
                )
            if name in current_iid:
                # Byte seals were refreshed in this temporary copy. A clean full
                # proof and a substantive adapter/caller error are mandatory.
                rejected = (
                    clean["ok"]
                    and not result["ok"]
                    and any(error not in clean["errors"] for error in result["errors"])
                    and not any(
                        "source seal mismatch" in error for error in result["errors"]
                    )
                )
            cases[name] = {
                "path": relative,
                "rejected": rejected,
                "errors": result["errors"],
                "offending_paths": result["offending_paths"],
            }
            target.write_text(originals[relative], encoding="utf-8")
            if (
                name in current_iid
                or name in final_nested
                or name in child_native
                or name in additional_helpers
            ):
                restored = verify_direct_candidate_flow(repo_root=temp_root)
                if restored != clean:
                    raise ValueError(
                        f"{name}: nested control did not restore its clean baseline"
                    )
    current_iid_cases = {name: cases.pop(name) for name in current_iid}
    current_iid_admitted = sorted(
        name for name, case in current_iid_cases.items() if not case["rejected"]
    )
    current_iid_names_match = (
        set(current_iid_cases) == _EXPECTED_CURRENT_IID_MUTATION_NAMES
        and _mutation_name_digest(tuple(current_iid_cases))
        == EXPECTED_CURRENT_IID_MUTATION_NAMES_SHA256
    )
    additional_helper_cases = {name: cases.pop(name) for name in additional_helpers}
    additional_helper_admitted = sorted(
        name
        for name, result in additional_helper_cases.items()
        if not result["rejected"]
    )
    additional_helper_names_match = (
        set(additional_helper_cases) == _EXPECTED_ADDITIONAL_HELPER_MUTATION_NAMES
        and len(additional_helper_cases) == 48
    )
    child_native_cases = {name: cases.pop(name) for name in child_native}
    child_native_admitted = sorted(
        name for name, result in child_native_cases.items() if not result["rejected"]
    )
    child_native_names_match = (
        set(child_native_cases) == _EXPECTED_CHILD_NATIVE_MUTATION_NAMES
        and len(child_native_cases) == 13
    )
    final_nested_cases = {name: cases.pop(name) for name in final_nested}
    final_nested_admitted = sorted(
        name for name, result in final_nested_cases.items() if not result["rejected"]
    )
    final_nested_names_match = (
        len(final_nested_cases) == EXPECTED_FINAL_NESTED_MUTATION_COUNT
        and _mutation_name_digest(tuple(final_nested_cases))
        == EXPECTED_FINAL_NESTED_MUTATION_NAMES_SHA256
    )
    decoupled_bound_cases = {name: cases.pop(name) for name in decoupled_bound}
    decoupled_bound_admitted = sorted(
        name for name, result in decoupled_bound_cases.items() if not result["rejected"]
    )
    decoupled_bound_names_match = (
        len(decoupled_bound_cases) == 8
        and _mutation_name_digest(tuple(decoupled_bound_cases))
        == "96fe054af698aadee64e5bb7d06c9e2fb7160c382168bb5cee7942b3c58dfa24"
    )
    reviewed_prefix_cases = {name: cases.pop(name) for name in reviewed_prefix}
    reviewed_prefix_admitted = sorted(
        name for name, result in reviewed_prefix_cases.items() if not result["rejected"]
    )
    reviewed_prefix_names_match = (
        len(reviewed_prefix_cases) == EXPECTED_REVIEWED_PREFIX_MUTATION_COUNT
        and _mutation_name_digest(tuple(reviewed_prefix_cases))
        == EXPECTED_REVIEWED_PREFIX_MUTATION_NAMES_SHA256
    )
    outer_helper_cases = {name: cases.pop(name) for name in outer_helpers}
    outer_helper_admitted = sorted(
        name for name, result in outer_helper_cases.items() if not result["rejected"]
    )
    outer_helper_names_match = (
        set(outer_helper_cases)
        == set(_OUTER_HELPER_CONTROLS)
        == _EXPECTED_OUTER_HELPER_MUTATION_NAMES
    )
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
    s03_scalar_cases = {name: cases.pop(name) for name in s03_scalar}
    s03_scalar_admitted = sorted(
        name for name, result in s03_scalar_cases.items() if not result["rejected"]
    )
    s03_scalar_names_match = (
        len(s03_scalar_cases) == EXPECTED_S03_SCALAR_MUTATION_COUNT
        and _mutation_name_digest(tuple(s03_scalar_cases))
        == EXPECTED_S03_SCALAR_MUTATION_NAMES_SHA256
    )
    owned_query_cases = {name: cases.pop(name) for name in owned_query}
    owned_query_admitted = sorted(
        name for name, result in owned_query_cases.items() if not result["rejected"]
    )
    owned_query_names_match = (
        len(owned_query_cases) == EXPECTED_OWNED_QUERY_HELPER_MUTATION_COUNT
        and _mutation_name_digest(tuple(owned_query_cases))
        == EXPECTED_OWNED_QUERY_HELPER_MUTATION_NAMES_SHA256
    )
    child_failure_cases = {name: cases.pop(name) for name in child_failure}
    child_failure_admitted = sorted(
        name for name, result in child_failure_cases.items() if not result["rejected"]
    )
    child_failure_names_match = (
        len(child_failure_cases) == EXPECTED_CHILD_FAILURE_MUTATION_COUNT
        and _mutation_name_digest(tuple(child_failure_cases))
        == EXPECTED_CHILD_FAILURE_MUTATION_NAMES_SHA256
    )
    retained_prefix_cases = {name: cases.pop(name) for name in retained_prefix}
    retained_prefix_admitted = sorted(
        name for name, result in retained_prefix_cases.items() if not result["rejected"]
    )
    retained_prefix_names_match = (
        len(retained_prefix_cases) == EXPECTED_RETAINED_PREFIX_MUTATION_COUNT
        and _mutation_name_digest(tuple(retained_prefix_cases))
        == EXPECTED_RETAINED_PREFIX_MUTATION_NAMES_SHA256
    )
    associated_saving_cases = {name: cases.pop(name) for name in associated_saving}
    associated_saving_admitted = sorted(
        name
        for name, result in associated_saving_cases.items()
        if not result["rejected"]
    )
    associated_saving_names_match = (
        len(associated_saving_cases) == EXPECTED_ASSOCIATED_SAVING_MUTATION_COUNT
        and _mutation_name_digest(tuple(associated_saving_cases))
        == EXPECTED_ASSOCIATED_SAVING_MUTATION_NAMES_SHA256
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
        "s03_scalar_mutations": s03_scalar_cases,
        "s03_scalar_mutation_count": len(s03_scalar_cases),
        "s03_scalar_names_match_expected": s03_scalar_names_match,
        "admitted_s03_scalar_mutations": s03_scalar_admitted,
        "owned_query_helper_mutations": owned_query_cases,
        "owned_query_helper_mutation_count": len(owned_query_cases),
        "owned_query_helper_names_match_expected": owned_query_names_match,
        "admitted_owned_query_helper_mutations": owned_query_admitted,
        "associated_saving_mutations": associated_saving_cases,
        "associated_saving_mutation_count": len(associated_saving_cases),
        "associated_saving_names_match_expected": associated_saving_names_match,
        "admitted_associated_saving_mutations": associated_saving_admitted,
        "child_failure_mutations": child_failure_cases,
        "child_failure_mutation_count": len(child_failure_cases),
        "child_failure_names_match_expected": child_failure_names_match,
        "admitted_child_failure_mutations": child_failure_admitted,
        "reviewed_prefix_mutations": reviewed_prefix_cases,
        "reviewed_prefix_mutation_count": len(reviewed_prefix_cases),
        "reviewed_prefix_names_match_expected": reviewed_prefix_names_match,
        "admitted_reviewed_prefix_mutations": reviewed_prefix_admitted,
        "decoupled_bound_mutations": decoupled_bound_cases,
        "decoupled_bound_mutation_count": len(decoupled_bound_cases),
        "decoupled_bound_names_match_expected": decoupled_bound_names_match,
        "admitted_decoupled_bound_mutations": decoupled_bound_admitted,
        "retained_prefix_mutations": retained_prefix_cases,
        "retained_prefix_mutation_count": len(retained_prefix_cases),
        "retained_prefix_names_match_expected": retained_prefix_names_match,
        "admitted_retained_prefix_mutations": retained_prefix_admitted,
        "final_nested_mutations": final_nested_cases,
        "final_nested_mutation_count": len(final_nested_cases),
        "final_nested_names_match_expected": final_nested_names_match,
        "admitted_final_nested_mutations": final_nested_admitted,
        "outer_helper_mutations": outer_helper_cases,
        "outer_helper_mutation_count": len(outer_helper_cases),
        "outer_helper_names_match_expected": outer_helper_names_match,
        "admitted_outer_helper_mutations": outer_helper_admitted,
        "child_native_mutations": child_native_cases,
        "child_native_mutation_count": len(child_native_cases),
        "child_native_names_match_expected": child_native_names_match,
        "admitted_child_native_mutations": child_native_admitted,
        "additional_helper_mutations": additional_helper_cases,
        "additional_helper_mutation_count": len(additional_helper_cases),
        "additional_helper_names_match_expected": additional_helper_names_match,
        "admitted_additional_helper_mutations": additional_helper_admitted,
        "current_iid_mutations": current_iid_cases,
        "current_iid_names_match_expected": current_iid_names_match,
        "admitted_current_iid_mutations": current_iid_admitted,
        "all_rejected": (
            not current_iid_admitted
            and current_iid_names_match
            and not additional_helper_admitted
            and additional_helper_names_match
            and not child_native_admitted
            and child_native_names_match
            and not final_nested_admitted
            and final_nested_names_match
            and not outer_helper_admitted
            and not decoupled_bound_admitted
            and decoupled_bound_names_match
            and outer_helper_names_match
            and not retained_prefix_admitted
            and retained_prefix_names_match
            and not associated_saving_admitted
            and associated_saving_names_match
            and not reviewed_prefix_admitted
            and reviewed_prefix_names_match
            and not child_failure_admitted
            and child_failure_names_match
            and not owned_query_admitted
            and owned_query_names_match
            and not s03_scalar_admitted
            and s03_scalar_names_match
            and clean["ok"]
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


def _decoupled_metadata_binding(
    *, function: ast.FunctionDef, name: str, expected_source: SourceText
) -> bool:
    """Match the sole assignment of a named metadata owner in one function."""
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _name(node=node.targets[0], expected=name)
    ]
    return (
        _stored_name_count(node=function, name=name) == 1
        and len(assignments) == 1
        and _expression_matches(node=assignments[0].value, source=expected_source)
    )


def _decoupled_bound_transport_errors(tree: ast.Module) -> list[str]:
    """Check one evaluated bound record across dense Decoupled reductions."""
    prefix = "Decoupled bound metadata"
    try:
        action = _definition(tree=tree, name="get_decoupled_Q_over_a")
        state = _definition(tree=tree, name="get_decoupled_max_Q_over_a")
        _, readout = _method_definition(
            tree=tree, class_name="_DecoupledQOverA", method_name="__call__"
        )
    except ValueError as error:
        return [f"{prefix}: {error}"]
    errors = []
    bindings = _scope_binding_counts(tree.body)
    imports = {
        "BoundedQAndF": "_lcm.regime_building.Q_and_F",
        "runtime_bound_specs": "_lcm.regime_building.Q_and_F",
        "BoundedResult": "_lcm.solution.feasibility",
        "_with_bound_summary": "_lcm.regime_building.max_Q_over_a",
        "_wrap_with_fold_reduction": "_lcm.regime_building.max_Q_over_a",
        "productmap": "_lcm.utils.dispatchers",
        "tiled_productmap": "_lcm.utils.dispatchers",
        "with_signature": "dags",
        "cast": "typing",
    }
    for name, module in imports.items():
        owners = [
            alias
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == module
            for alias in node.names
            if alias.name == name and alias.asname is None
        ]
        if len(owners) != 1 or bindings.get(name) != 1:
            errors.append(f"{prefix}: imported owner {name} changed")
    checks = (
        (
            action,
            "bounded",
            "bool(runtime_bound_specs(function=Q_and_F))",
        ),
        (
            action,
            "parts",
            """productmap(
                func=BoundedQAndF(function=Q_and_F, parts=True)
                if bounded else get_Q_and_F_parts(Q_and_F=Q_and_F),
                variables=action_names,
                batch_sizes=dict.fromkeys(action_names, 0),
                bound_summary=bounded,
            )""",
        ),
        (
            readout,
            "evaluated",
            """self.parts(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **states_actions_params,
            )""",
        ),
        (
            readout,
            "parts",
            """cast(
                "QAndFParts",
                evaluated.output if type(evaluated) is BoundedResult else evaluated,
            )""",
        ),
    )
    for function, name, expression in checks:
        if not _decoupled_metadata_binding(
            function=function, name=name, expected_source=expression
        ):
            errors.append(f"{prefix}: {function.name}.{name} transport changed")
    constructors = [
        node
        for node in ast.walk(action)
        if isinstance(node, ast.Call) and _call_name(node) == "_DecoupledQOverA"
    ]
    if len(constructors) != 1 or not _expression_matches(
        node=_keyword(call=constructors[0], name="parts") if constructors else None,
        source='cast("Callable[..., QAndFParts | BoundedResult[QAndFParts]]", parts)',
    ):
        errors.append(f"{prefix}: action parts are not bound into the readout")
    mappings = [
        node
        for node in ast.walk(state)
        if isinstance(node, ast.Call)
        and _call_name(node) in {"productmap", "tiled_productmap"}
    ]
    if len(mappings) != 2:
        errors.append(f"{prefix}: state mappings changed")
    errors.extend(
        f"{prefix}: {_call_name(mapping)} loses state summaries"
        for mapping in mappings
        if not (
            _name(node=_keyword(call=mapping, name="func"), expected="cell")
            and _expression_matches(
                node=_keyword(call=mapping, name="bound_summary"),
                source="bool(runtime_bound_specs(function=Q_and_F))",
            )
        )
    )
    folds = [
        node
        for node in ast.walk(state)
        if isinstance(node, ast.Call)
        and _call_name(node) == "_wrap_with_fold_reduction"
    ]
    if len(folds) != 1 or not _expression_matches(
        node=_keyword(call=folds[0], name="mapped") if folds else None,
        source='cast("Callable[..., BoundedMaxOutput]", mapped)',
    ):
        errors.append(f"{prefix}: folded mappings lose their bound result")
    returns = [node for node in ast.walk(readout) if isinstance(node, ast.Return)]
    if len(returns) != 3:
        errors.append(f"{prefix}: hard and EV1 return owners changed")
    for returned in returns:
        call = returned.value
        if not (
            isinstance(call, ast.Call)
            and _call_name(call) == "_with_bound_summary"
            and not call.args
            and {item.arg for item in call.keywords} == {"evaluated", "output"}
            and len(call.keywords) == 2
            and _name(node=_keyword(call=call, name="evaluated"), expected="evaluated")
        ):
            errors.append(f"{prefix}: a readout return loses the actual summary")
    return errors


_DECOUPLED_BOUND_MUTATION_NAMES = (
    "action_adapter_bypassed",
    "action_summary_disabled",
    "state_tiled_summary_disabled",
    "state_untiled_summary_disabled",
    "readout_parts_recomputed",
    "hard_return_wrong_status_owner",
    "ev1_solve_return_wrong_status_owner",
    "ev1_simulate_return_wrong_status_owner",
)


def decoupled_bound_mutation_source(*, source: SourceText, mutation: str) -> SourceText:
    """Alter one named metadata link, leaving ordinary Decoupled arithmetic intact."""
    tree = ast.parse(source, filename=DECOUPLED_BOUND_SOURCE)
    action = _definition(tree=tree, name="get_decoupled_Q_over_a")
    state = _definition(tree=tree, name="get_decoupled_max_Q_over_a")
    _, readout = _method_definition(
        tree=tree, class_name="_DecoupledQOverA", method_name="__call__"
    )
    if mutation == "action_adapter_bypassed":
        for node in ast.walk(action):
            if isinstance(node, ast.Call) and _call_name(node) == "productmap":
                next(
                    item for item in node.keywords if item.arg == "func"
                ).value = ast.parse(
                    "get_Q_and_F_parts(Q_and_F=Q_and_F)", mode="eval"
                ).body
    elif mutation == "action_summary_disabled":
        for node in ast.walk(action):
            if isinstance(node, ast.Call) and _call_name(node) == "productmap":
                next(
                    item for item in node.keywords if item.arg == "bound_summary"
                ).value = ast.Constant(value=False)
    elif mutation in {"state_tiled_summary_disabled", "state_untiled_summary_disabled"}:
        owner = (
            "tiled_productmap"
            if mutation == "state_tiled_summary_disabled"
            else "productmap"
        )
        for node in ast.walk(state):
            if isinstance(node, ast.Call) and _call_name(node) == owner:
                next(
                    item for item in node.keywords if item.arg == "bound_summary"
                ).value = ast.Constant(value=False)
    elif mutation == "readout_parts_recomputed":
        assignment = next(
            node
            for node in ast.walk(readout)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and _name(node=node.targets[0], expected="parts")
        )
        assignment.value = ast.parse(
            "self.parts(next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params)",
            mode="eval",
        ).body
    elif mutation in _DECOUPLED_BOUND_MUTATION_NAMES[5:]:
        returns = sorted(
            (node for node in ast.walk(readout) if isinstance(node, ast.Return)),
            key=lambda node: node.lineno,
        )
        call = returns[_DECOUPLED_BOUND_MUTATION_NAMES[5:].index(mutation)].value
        if not isinstance(call, ast.Call):
            raise ValueError("Expected a metadata-preserving return call.")
        next(
            item for item in call.keywords if item.arg == "evaluated"
        ).value = ast.Name(id="parts", ctx=ast.Load())
    else:
        raise ValueError(f"Unknown bounded Decoupled mutation {mutation!r}")
    result = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    if _ast_key(ast.parse(result)) == _ast_key(ast.parse(source)):
        raise ValueError(f"Bounded Decoupled mutation {mutation!r} changed no AST.")
    return result


def decoupled_bound_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Register each actual Decoupled metadata mutation as a separate control."""
    original = (repo_root.resolve() / DECOUPLED_BOUND_SOURCE).read_text(
        encoding="utf-8"
    )
    return {
        f"decoupled_bound:{mutation}": {
            "path": DECOUPLED_BOUND_SOURCE,
            "source": decoupled_bound_mutation_source(
                source=original, mutation=mutation
            ),
        }
        for mutation in _DECOUPLED_BOUND_MUTATION_NAMES
    }


_RETAINED_PREFIX_OWNER_SELECTION = {
    "src/_lcm/egm/carry.py": ("_read_owned_row_with_support",),
    "src/_lcm/egm/dcegm_query.py": (
        "DCEGMQuery.read_prepared_with_counts",
        "DCEGMQuery._read_at_floor_with_summary",
        "DCEGMQuery._read_rows_without_counts",
        "DCEGMQuery._read_rows_with_counts",
        "_blend_owned_rows",
        "_read_owned_derivative_with_support",
        "_owned_at_query_with_support",
    ),
    "src/_lcm/utils/counts.py": ("sum_counts", "add_counts"),
    "src/_lcm/egm/nbegm_step.py": (
        "nbegm_multi_interval_step_savings",
        "nbegm_per_interval_continuation_step_savings",
        "_query_node_kind",
        "_make_nbegm_query_data",
        "_invert_euler_over_savings",
        "_invert_euler_at_node",
        "_ez_flow_power_structure",
        "_discounted_continuation",
        "_invert_coh_with_linear_extension",
        "_linear_extension",
        "_degenerate_inversion",
        "_next_segment_id",
        "_as_pairs",
        "_stack_flat",
        "_interval_corner_candidates",
        "_fixed_savings_corner",
        "_publish_no_action_carry",
        "_publish_interval_step",
        "_IntervalCaseSolver.__call__",
        "_solve_packed_interval",
        "_streamed_interval_block_layout",
        "_streamed_block_candidate_positions",
        "_StreamedIntervalLayout.total_candidates",
        "_streamed_interval_continuation_envelope",
        "_IntervalWinnerStep.__call__",
        "_kill_padding",
        "_point_candidate_segment_bases",
        "_savings_node_point_candidates_for_interval_block",
        "_savings_node_point_candidates",
    ),
    "src/_lcm/egm/nbegm_segments.py": (
        "affords_an_action",
        "mask_dead_candidates",
        "segment_ids_from_folds",
    ),
    "src/_lcm/egm/framed_arithmetic.py": (
        "framed_difference",
        "binade_exponent",
        "_dyadic_parts",
        "_two_diff",
    ),
    "src/_lcm/egm/ez_kernel.py": (
        "ez_consumption_from_euler",
        "ez_marginal_of_resource",
        "ez_period_value",
    ),
    "src/_lcm/power_mean.py": (
        "weighted_power_mean_of_pair",
        "_applied_to_anchor",
        "_log_term",
    ),
    "src/_lcm/probability.py": (
        "log_of_nonnegative",
        "_log_of_nonnegative_bits",
        "_log_of_nonnegative_jvp",
        "rescaled_weight_pair",
    ),
}
_RETAINED_PREFIX_SEMANTIC_MUTATIONS = {
    "retained_prefix:checked_selects_counted_reader": (
        "src/_lcm/egm/dcegm_query.py",
        "DCEGMQuery.read_prepared_with_counts",
        "keyword",
        "read_rows",
        "self._read_rows_without_counts",
        1,
    ),
    "retained_prefix:prepared_floor_not_repeated": (
        "src/_lcm/egm/dcegm_query.py",
        "DCEGMQuery._read_at_floor_with_summary",
        "expression",
        "floor_state",
        "self.evaluate_floor(pool=pool)",
        1,
    ),
    "retained_prefix:dead_support_not_eligible": (
        "src/_lcm/egm/dcegm_query.py",
        "DCEGMQuery._read_rows_with_counts",
        "expression",
        "active & has_first",
        "active",
        1,
    ),
    "retained_prefix:below_first_is_strict": (
        "src/_lcm/egm/dcegm_query.py",
        "DCEGMQuery._read_rows_with_counts",
        "expression",
        "row_queries.reshape(n_rows) < first",
        "row_queries.reshape(n_rows) <= first",
        1,
    ),
    "retained_prefix:first_original_node": (
        "src/_lcm/egm/dcegm_query.py",
        "DCEGMQuery._read_rows_with_counts",
        "expression",
        "jnp.argmax(supported_node, axis=-1)",
        "jnp.argmax(~supported_node, axis=-1)",
        1,
    ),
    "retained_prefix:JVP_primal_support": (
        "src/_lcm/egm/dcegm_query.py",
        "_read_owned_derivative_with_support",
        "expression",
        (
            "(result._replace(marginal=jnp.where(jnp.isneginf(result.value), "
            "0.0, tangent.value)), supported_node)"
        ),
        (
            "(result._replace(marginal=jnp.where(jnp.isneginf(result.value), "
            "0.0, tangent.value)), "
            "jnp.zeros_like(supported_node))"
        ),
        1,
    ),
    "retained_prefix:count_add_saturates": (
        "src/_lcm/utils/counts.py",
        "add_counts",
        "expression",
        "left > COUNT_CAPACITY - right",
        "False",
        1,
    ),
    "retained_prefix:count_sum_overflow_path": (
        "src/_lcm/utils/counts.py",
        "sum_counts",
        "expression",
        "n_elements * largest_element <= COUNT_CAPACITY",
        "True",
        1,
    ),
    "retained_prefix:ordinary_needs_original_A": (
        "src/_lcm/egm/nbegm_step.py",
        "_make_nbegm_query_data",
        "expression",
        "jnp.isfinite(savings)",
        "True",
        1,
    ),
    "retained_prefix:economic_kind_not_PAD": (
        "src/_lcm/egm/nbegm_step.py",
        "_query_node_kind",
        "expression",
        "supported & jnp.isneginf(value)",
        "False",
        1,
    ),
    "retained_prefix:link_same_budget_run": (
        "src/_lcm/egm/nbegm_step.py",
        "_make_nbegm_query_data",
        "expression",
        "budget_run[:-1] == budget_run[1:]",
        "jnp.ones_like(budget_run[:-1], dtype=bool)",
        1,
    ),
    "retained_prefix:flat_original_winner_A": (
        "src/_lcm/egm/nbegm_step.py",
        "_interval_corner_candidates",
        "expression",
        "jnp.where(flat, savings_grid[best_floor], savings_grid[0])",
        "savings_grid[0]",
        1,
    ),
    "retained_prefix:upper_original_A": (
        "src/_lcm/egm/nbegm_step.py",
        "_interval_corner_candidates",
        "expression",
        "jnp.where(smax_feasible, savings_grid[-1], jnp.nan)",
        "jnp.where(smax_feasible, savings_grid[0], jnp.nan)",
        1,
    ),
    "retained_prefix:Euler_original_A": (
        "src/_lcm/egm/nbegm_step.py",
        "_IntervalCaseSolver.__call__",
        "expression",
        "jnp.where(in_case, savings_grid, jnp.nan)",
        "jnp.where(in_case, coh_endog, jnp.nan)",
        1,
    ),
    "retained_prefix:dense_compact_floor_column": (
        "src/_lcm/egm/nbegm_step.py",
        "nbegm_per_interval_continuation_step_savings",
        "expression",
        "cont_value[:, 0]",
        "cont_value[:, -1]",
        1,
    ),
    "retained_prefix:stream_compact_floor_column": (
        "src/_lcm/egm/nbegm_step.py",
        "_IntervalWinnerStep.__call__",
        "expression",
        (
            "retained.corner_continuation.at[destinations].set(block_value[:, "
            '0], mode="drop")'
        ),
        (
            "retained.corner_continuation.at[destinations].set(block_value[:, "
            '-1], mode="drop")'
        ),
        1,
    ),
    "retained_prefix:padding_owns_no_compact_slot": (
        "src/_lcm/egm/nbegm_step.py",
        "_IntervalWinnerStep.__call__",
        "expression",
        "jnp.where(interval_live, indices, self.n_intervals)",
        "indices",
        1,
    ),
    "retained_prefix:one_callback_per_block": (
        "src/_lcm/egm/nbegm_step.py",
        "_IntervalWinnerStep.__call__",
        "expression",
        "self.interval_block_reader(indices)",
        "(self.interval_block_reader(indices), self.interval_block_reader(indices))[0]",
        1,
    ),
    "retained_prefix:dense_points_own_A": (
        "src/_lcm/egm/nbegm_step.py",
        "_savings_node_point_candidates",
        "expression",
        "jnp.where(node_feasible, savings_at_grid, jnp.nan)",
        "jnp.where(node_feasible, point_coh[:, None], jnp.nan)",
        1,
    ),
    "retained_prefix:block_points_own_A": (
        "src/_lcm/egm/nbegm_step.py",
        "_savings_node_point_candidates_for_interval_block",
        "expression",
        "jnp.where(feasible, savings_at_grid, jnp.nan)",
        "jnp.where(feasible, point_coh[:, None], jnp.nan)",
        1,
    ),
    "retained_prefix:node_support_not_tolerance": (
        "src/_lcm/egm/nbegm_segments.py",
        "affords_an_action",
        "expression",
        "budget > 0.0",
        "budget > 1e-6",
        1,
    ),
    "retained_prefix:segments_do_not_bridge_holes": (
        "src/_lcm/egm/nbegm_segments.py",
        "segment_ids_from_folds",
        "expression",
        "~ascending | dead[1:] | dead[:-1]",
        "~ascending",
        1,
    ),
    "retained_prefix:framed_difference_common_frame": (
        "src/_lcm/egm/framed_arithmetic.py",
        "framed_difference",
        "expression",
        "jnp.maximum(exponent_a, exponent_b)",
        "jnp.minimum(exponent_a, exponent_b)",
        1,
    ),
    "retained_prefix:EZ_resource_native_chain": (
        "src/_lcm/egm/ez_kernel.py",
        "ez_marginal_of_resource",
        "expression",
        "inverse_eis * jnp.log(value)",
        "jnp.log(value)",
        1,
    ),
    "retained_prefix:pair_anchor_exact_payoff": (
        "src/_lcm/power_mean.py",
        "_anchor_primal",
        "expression",
        "jnp.where(deviation == 0.0, payoff, payoff * jnp.exp(deviation))",
        "payoff * jnp.exp(deviation)",
        1,
    ),
    "retained_prefix:log_JVP_not_recomputed_bits": (
        "src/_lcm/probability.py",
        "_log_of_nonnegative_jvp",
        "expression",
        "values_dot / values",
        "values_dot",
        1,
    ),
}
_RETAINED_PREFIX_STUB_MUTATIONS = {
    "retained_prefix:multi_stub_type": (
        "nbegm_multi_interval_step_savings",
        0,
        "annotation",
    ),
    "retained_prefix:multi_stub_count": (
        "nbegm_multi_interval_step_savings",
        0,
        "duplicate",
    ),
    "retained_prefix:multi_stub_order": (
        "nbegm_multi_interval_step_savings",
        0,
        "swap",
    ),
    "retained_prefix:interval_stub_type": (
        "nbegm_per_interval_continuation_step_savings",
        0,
        "annotation",
    ),
    "retained_prefix:interval_stub_count": (
        "nbegm_per_interval_continuation_step_savings",
        0,
        "duplicate",
    ),
    "retained_prefix:interval_stub_order": (
        "nbegm_per_interval_continuation_step_savings",
        0,
        "swap",
    ),
}
_RETAINED_PREFIX_DIAGNOSTICS = {
    "retained_prefix:body:src/_lcm/egm/carry.py::_read_owned_row_with_support": "reviewed "
    "S04/S05 "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_read_owned_row_with_support' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/dcegm_query.py::DCEGMQuery.read_prepared_with_counts": "reviewed "
    "S04/S05 "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'DCEGMQuery.read_prepared_with_counts' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/dcegm_query.py::DCEGMQuery._read_at_floor_with_summary": "reviewed "
    "S04/S05 "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'DCEGMQuery._read_at_floor_with_summary' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/dcegm_query.py::DCEGMQuery._read_rows_without_counts": "reviewed "
    "S04/S05 "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'DCEGMQuery._read_rows_without_counts' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/dcegm_query.py::DCEGMQuery._read_rows_with_counts": "reviewed "
    "S04/S05 "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'DCEGMQuery._read_rows_with_counts' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/dcegm_query.py::_blend_owned_rows": "reviewed "
    "S04/S05 "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_blend_owned_rows' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/dcegm_query.py::_read_owned_derivative_with_support": "reviewed "
    "S04/S05 "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_read_owned_derivative_with_support' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/dcegm_query.py::_owned_at_query_with_support": "reviewed "
    "S04/S05 "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_owned_at_query_with_support' "
    "changed",
    "retained_prefix:body:src/_lcm/utils/counts.py::sum_counts": "retained DC/NB "
    "dependency: "
    "exact callable "
    "corridor "
    "'sum_counts' "
    "changed",
    "retained_prefix:body:src/_lcm/utils/counts.py::add_counts": "retained DC/NB "
    "dependency: "
    "exact callable "
    "corridor "
    "'add_counts' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::nbegm_multi_interval_step_savings": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'nbegm_multi_interval_step_savings' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::nbegm_per_interval_continuation_step_savings": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'nbegm_per_interval_continuation_step_savings' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_query_node_kind": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_query_node_kind' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_make_nbegm_query_data": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_make_nbegm_query_data' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_invert_euler_over_savings": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_invert_euler_over_savings' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_invert_euler_at_node": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_invert_euler_at_node' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_ez_flow_power_structure": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_ez_flow_power_structure' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_discounted_continuation": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_discounted_continuation' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_invert_coh_with_linear_extension": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_invert_coh_with_linear_extension' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_linear_extension": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_linear_extension' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_degenerate_inversion": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_degenerate_inversion' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_next_segment_id": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_next_segment_id' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_as_pairs": "retained DC/NB "
    "dependency: "
    "exact callable "
    "corridor "
    "'_as_pairs' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_stack_flat": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_stack_flat' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_interval_corner_candidates": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_interval_corner_candidates' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_fixed_savings_corner": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_fixed_savings_corner' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_publish_no_action_carry": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_publish_no_action_carry' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_publish_interval_step": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_publish_interval_step' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_IntervalCaseSolver.__call__": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_IntervalCaseSolver.__call__' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_solve_packed_interval": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_solve_packed_interval' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_streamed_interval_block_layout": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_streamed_interval_block_layout' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_streamed_block_candidate_positions": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_streamed_block_candidate_positions' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_StreamedIntervalLayout.total_candidates": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_StreamedIntervalLayout.total_candidates' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_streamed_interval_continuation_envelope": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_streamed_interval_continuation_envelope' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_IntervalWinnerStep.__call__": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_IntervalWinnerStep.__call__' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_kill_padding": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_kill_padding' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_point_candidate_segment_bases": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_point_candidate_segment_bases' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_savings_node_point_candidates_for_interval_block": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_savings_node_point_candidates_for_interval_block' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_step.py::_savings_node_point_candidates": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_savings_node_point_candidates' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_segments.py::affords_an_action": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'affords_an_action' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_segments.py::mask_dead_candidates": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'mask_dead_candidates' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/nbegm_segments.py::segment_ids_from_folds": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'segment_ids_from_folds' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/framed_arithmetic.py::framed_difference": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'framed_difference' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/framed_arithmetic.py::binade_exponent": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'binade_exponent' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/framed_arithmetic.py::_dyadic_parts": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_dyadic_parts' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/framed_arithmetic.py::_two_diff": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_two_diff' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/ez_kernel.py::ez_consumption_from_euler": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'ez_consumption_from_euler' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/ez_kernel.py::ez_marginal_of_resource": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'ez_marginal_of_resource' "
    "changed",
    "retained_prefix:body:src/_lcm/egm/ez_kernel.py::ez_period_value": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'ez_period_value' "
    "changed",
    "retained_prefix:body:src/_lcm/power_mean.py::weighted_power_mean_of_pair": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'weighted_power_mean_of_pair' "
    "changed",
    "retained_prefix:body:src/_lcm/power_mean.py::_applied_to_anchor": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_applied_to_anchor' "
    "changed",
    "retained_prefix:body:src/_lcm/power_mean.py::_log_term": "retained DC/NB "
    "dependency: exact "
    "callable corridor "
    "'_log_term' "
    "changed",
    "retained_prefix:body:src/_lcm/probability.py::log_of_nonnegative": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'log_of_nonnegative' "
    "changed",
    "retained_prefix:body:src/_lcm/probability.py::_log_of_nonnegative_bits": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_log_of_nonnegative_bits' "
    "changed",
    "retained_prefix:body:src/_lcm/probability.py::_log_of_nonnegative_jvp": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'_log_of_nonnegative_jvp' "
    "changed",
    "retained_prefix:body:src/_lcm/probability.py::rescaled_weight_pair": "retained "
    "DC/NB "
    "dependency: "
    "exact "
    "callable "
    "corridor "
    "'rescaled_weight_pair' "
    "changed",
    "retained_prefix:checked_selects_counted_reader": "reviewed S04/S05 "
    "dependency: exact callable "
    "corridor "
    "'DCEGMQuery.read_prepared_with_counts' "
    "changed",
    "retained_prefix:prepared_floor_not_repeated": "reviewed S04/S05 dependency: "
    "exact callable corridor "
    "'DCEGMQuery._read_at_floor_with_summary' "
    "changed",
    "retained_prefix:dead_support_not_eligible": "reviewed S04/S05 dependency: "
    "exact callable corridor "
    "'DCEGMQuery._read_rows_with_counts' "
    "changed",
    "retained_prefix:below_first_is_strict": "reviewed S04/S05 dependency: exact "
    "callable corridor "
    "'DCEGMQuery._read_rows_with_counts' "
    "changed",
    "retained_prefix:first_original_node": "reviewed S04/S05 dependency: exact "
    "callable corridor "
    "'DCEGMQuery._read_rows_with_counts' "
    "changed",
    "retained_prefix:JVP_primal_support": "reviewed S04/S05 dependency: exact "
    "callable corridor "
    "'_read_owned_derivative_with_support' "
    "changed",
    "retained_prefix:count_add_saturates": "retained DC/NB dependency: exact "
    "callable corridor 'add_counts' "
    "changed",
    "retained_prefix:count_sum_overflow_path": "retained DC/NB dependency: exact "
    "callable corridor 'sum_counts' "
    "changed",
    "retained_prefix:ordinary_needs_original_A": "retained DC/NB dependency: "
    "exact callable corridor "
    "'_make_nbegm_query_data' "
    "changed",
    "retained_prefix:economic_kind_not_PAD": "retained DC/NB dependency: exact "
    "callable corridor "
    "'_query_node_kind' changed",
    "retained_prefix:link_same_budget_run": "retained DC/NB dependency: exact "
    "callable corridor "
    "'_make_nbegm_query_data' changed",
    "retained_prefix:flat_original_winner_A": "retained DC/NB dependency: exact "
    "callable corridor "
    "'_interval_corner_candidates' "
    "changed",
    "retained_prefix:upper_original_A": "retained DC/NB dependency: exact "
    "callable corridor "
    "'_interval_corner_candidates' changed",
    "retained_prefix:Euler_original_A": "retained DC/NB dependency: exact "
    "callable corridor "
    "'_IntervalCaseSolver.__call__' changed",
    "retained_prefix:dense_compact_floor_column": "retained DC/NB dependency: "
    "exact callable corridor "
    "'nbegm_per_interval_continuation_step_savings' "
    "changed",
    "retained_prefix:stream_compact_floor_column": "retained DC/NB dependency: "
    "exact callable corridor "
    "'_IntervalWinnerStep.__call__' "
    "changed",
    "retained_prefix:padding_owns_no_compact_slot": "retained DC/NB dependency: "
    "exact callable corridor "
    "'_IntervalWinnerStep.__call__' "
    "changed",
    "retained_prefix:one_callback_per_block": "retained DC/NB dependency: exact "
    "callable corridor "
    "'_IntervalWinnerStep.__call__' "
    "changed",
    "retained_prefix:dense_points_own_A": "retained DC/NB dependency: exact "
    "callable corridor "
    "'_savings_node_point_candidates' "
    "changed",
    "retained_prefix:block_points_own_A": "retained DC/NB dependency: exact "
    "callable corridor "
    "'_savings_node_point_candidates_for_interval_block' "
    "changed",
    "retained_prefix:node_support_not_tolerance": "retained DC/NB dependency: "
    "exact callable corridor "
    "'affords_an_action' changed",
    "retained_prefix:segments_do_not_bridge_holes": "retained DC/NB dependency: "
    "exact callable corridor "
    "'segment_ids_from_folds' "
    "changed",
    "retained_prefix:framed_difference_common_frame": "retained DC/NB dependency: "
    "exact callable corridor "
    "'framed_difference' "
    "changed",
    "retained_prefix:EZ_resource_native_chain": "retained DC/NB dependency: exact "
    "callable corridor "
    "'ez_marginal_of_resource' "
    "changed",
    "retained_prefix:pair_anchor_exact_payoff": "retained DC/NB dependency: exact "
    "callable corridor "
    "'_anchor_primal' changed",
    "retained_prefix:log_JVP_not_recomputed_bits": "retained DC/NB dependency: "
    "exact callable corridor "
    "'_log_of_nonnegative_jvp' "
    "changed",
    "retained_prefix:multi_stub_type": "retained DC/NB dependency: "
    "nbegm_multi_interval_step_savings "
    "overload declarations changed",
    "retained_prefix:multi_stub_count": "retained DC/NB dependency: "
    "nbegm_multi_interval_step_savings "
    "overload declarations changed",
    "retained_prefix:multi_stub_order": "retained DC/NB dependency: "
    "nbegm_multi_interval_step_savings "
    "overload declarations changed",
    "retained_prefix:interval_stub_type": "retained DC/NB dependency: "
    "nbegm_per_interval_continuation_step_savings "
    "overload declarations changed",
    "retained_prefix:interval_stub_count": "retained DC/NB dependency: "
    "nbegm_per_interval_continuation_step_savings "
    "overload declarations changed",
    "retained_prefix:interval_stub_order": "retained DC/NB dependency: "
    "nbegm_per_interval_continuation_step_savings "
    "overload declarations changed",
    "combined_source:bounded_parts_owner": "combined upstream dependency: exact "
    "callable corridor "
    "'get_decoupled_Q_over_a' changed",
    "combined_source:disabled_constructor_absent": "combined upstream dependency: "
    "exact callable corridor "
    "'get_decoupled_Q_over_a' "
    "changed",
    "combined_source:ordinary_state_summary": "combined upstream dependency: "
    "exact callable corridor "
    "'get_decoupled_max_Q_over_a' "
    "changed",
    "combined_source:tiled_state_summary": "combined upstream dependency: exact "
    "callable corridor "
    "'get_decoupled_max_Q_over_a' changed",
    "combined_source:hard_return_summary": "combined upstream dependency: exact "
    "callable corridor "
    "'_DecoupledQOverA.__call__' changed",
    "combined_source:ev1_value_return_summary": "combined upstream dependency: "
    "exact callable corridor "
    "'_DecoupledQOverA.__call__' "
    "changed",
    "combined_source:ev1_sample_return_summary": "combined upstream dependency: "
    "exact callable corridor "
    "'_DecoupledQOverA.__call__' "
    "changed",
    "combined_source:single_action_evaluation": "combined upstream dependency: "
    "exact callable corridor "
    "'_DecoupledQOverA.__call__' "
    "changed",
    "combined_source:hard_readout_profile": "combined upstream dependency: exact "
    "callable corridor "
    "'argmax_and_readout' changed",
    "combined_source:conditional_stored_profile": "combined upstream dependency: "
    "exact callable corridor "
    "'get_conditional_profiles' "
    "changed",
    "combined_source:ev1_economic_infinite_support": "combined upstream "
    "dependency: exact callable "
    "corridor 'decoupled_logsum' "
    "changed",
    "combined_source:lookup_hard_readout": "combined upstream dependency: exact "
    "callable corridor "
    "'DecoupledWithActionParts.__call__' "
    "changed",
    "combined_source:lookup_runtime_scale": "combined upstream dependency: exact "
    "callable corridor "
    "'DecoupledTasteShockLookup.__call__' "
    "changed",
    "combined_source:objective_metadata_dispatch": "combined upstream dependency: "
    "exact callable corridor "
    "'has_objective_profile' "
    "changed",
    "combined_source:ordinary_complete_parts": "combined upstream dependency: "
    "exact callable corridor "
    "'get_Q_and_F_parts' changed",
    "combined_source:parts_output_channel": "combined upstream dependency: exact "
    "callable corridor '_QAndF.parts' "
    "changed",
    "combined_source:ordinary_flow_pool": "combined upstream dependency: exact "
    "callable corridor '_QAndF._evaluate' "
    "changed",
    "combined_source:objective_separate_from_stored": "combined upstream "
    "dependency: exact callable "
    "corridor "
    "'_QAndF._parts_from_utilities' "
    "changed",
    "combined_source:lookup_constructor_roles": "combined upstream dependency: "
    "exact callable corridor "
    "'_lookup_body_and_roles' changed",
    "combined_source:lookup_original_summary": "combined upstream dependency: "
    "exact callable corridor "
    "'_BoundActionPartsBody.__call__' "
    "changed",
    "combined_source:lookup_same_evaluated_callback": "combined upstream "
    "dependency: exact callable "
    "corridor "
    "'_evaluated_action_parts' "
    "changed",
    "combined_source:unsupported_route_refusal": "combined upstream dependency: "
    "exact callable corridor "
    "'_fail_if_decoupled_route_is_unsupported' "
    "changed",
    "combined_source:rounded_equal_support": "S03 scalar dependency: exact "
    "callable corridor "
    "'_solve_viable_feature_row' changed",
    "combined_source:rightmost_original_endpoint": "S03 scalar dependency: exact "
    "callable corridor "
    "'_solve_viable_feature_row' "
    "changed",
    "combined_source:complete_endpoint_association": "S03 scalar dependency: "
    "exact callable corridor "
    "'_solve_viable_feature_row' "
    "changed",
}
EXPECTED_RETAINED_PREFIX_MUTATION_COUNT = 113
EXPECTED_RETAINED_PREFIX_MUTATION_NAMES_SHA256 = (
    "ff7f730978f19c765f3f342e431b46e0ffff94a4817eebcdf8fd9bf9c92221f1"
)


def retained_prefix_mutation_specs(*, repo_root: Path) -> dict[str, dict[str, str]]:
    """Keep finite retained-read counterexamples separate from every old registry."""
    specs = _callable_mutation_specs(
        repo_root=repo_root, mutations=_RETAINED_PREFIX_SEMANTIC_MUTATIONS
    )
    for relative, names in _RETAINED_PREFIX_OWNER_SELECTION.items():
        source = (repo_root / relative).read_text(encoding="utf-8")
        for qualname in names:
            tree = ast.parse(source, filename=relative)
            if (
                relative == _NB_PRODUCER_SOURCE
                and qualname in _NB_PRODUCER_OVERLOAD_COUNTS
            ):
                function = _nb_producer_implementation(tree=tree, name=qualname)
            elif (
                relative == _NB_PRODUCER_SOURCE
                and qualname == "_IntervalWinnerStep.__call__"
            ):
                function = _select_interval_winner_call(
                    tree=tree, checker=sys.modules[__name__]
                )
            elif "." in qualname:
                cls, method = qualname.split(".", maxsplit=1)
                _, function = _method_definition(
                    tree=tree, class_name=cls, method_name=method
                )
            else:
                function = _definition(tree=tree, name=qualname)
            function.body = ast.parse("return None").body
            specs[f"retained_prefix:body:{relative}::{qualname}"] = {
                "path": relative,
                "source": ast.unparse(tree) + "\n",
            }
    relative = _NB_PRODUCER_SOURCE
    source = (repo_root / relative).read_text(encoding="utf-8")
    for name, (owner, index, mode) in _RETAINED_PREFIX_STUB_MUTATIONS.items():
        tree = ast.parse(source, filename=relative)
        _nb_producer_implementation(tree=tree, name=owner)
        stubs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == owner
        ][:-1]
        position = tree.body.index(stubs[index])
        if mode == "annotation":
            parameter = next(
                arg for arg in stubs[index].args.kwonlyargs if arg.arg == "retain_query"
            )
            parameter.annotation = ast.Name(id="bool", ctx=ast.Load())
        elif mode == "duplicate":
            tree.body.insert(position, copy.deepcopy(stubs[index]))
        elif mode == "swap":
            second = tree.body.index(stubs[index + 1])
            tree.body[position], tree.body[second] = (
                tree.body[second],
                tree.body[position],
            )
        else:
            raise ValueError(f"{name}: unknown reviewed declaration mutation")
        specs[name] = {"path": relative, "source": ast.unparse(tree) + "\n"}
    upstream = _upstream_composition_mutation_specs(repo_root=repo_root)
    if set(specs) & set(upstream):
        raise ValueError("Upstream controls overlap retained controls")
    specs.update(upstream)
    return specs


_UPSTREAM_COMPOSITION_CONTRACTS = _contracts(
    {
        "src/_lcm/regime_building/decoupled.py": (
            "get_decoupled_Q_over_a",
            "get_decoupled_max_Q_over_a",
            "_DecoupledQOverA.__call__",
            "argmax_and_readout",
            "get_conditional_profiles",
            "decoupled_logsum",
            "DecoupledWithActionParts.__call__",
            "DecoupledTasteShockLookup.__call__",
        ),
        "src/_lcm/regime_building/Q_and_F.py": (
            "has_objective_profile",
            "get_Q_and_F_parts",
            "_QAndF.parts",
            "_QAndF._evaluate",
            "_QAndF._parts_from_utilities",
        ),
        "src/_lcm/simulation/programs.py": (
            "_lookup_body_and_roles",
            "_BoundActionPartsBody.__call__",
            "_evaluated_action_parts",
        ),
        "src/_lcm/solution/grid_search.py": (
            "_fail_if_decoupled_route_is_unsupported",
        ),
    }
)
_UPSTREAM_COMPOSITION_MUTATIONS = {
    "combined_source:bounded_parts_owner": (
        "src/_lcm/regime_building/decoupled.py",
        "get_decoupled_Q_over_a",
        "expression",
        "BoundedQAndF(function=Q_and_F, parts=True)",
        "get_Q_and_F_parts(Q_and_F=Q_and_F)",
        1,
        None,
    ),
    "combined_source:disabled_constructor_absent": (
        "src/_lcm/regime_building/decoupled.py",
        "get_decoupled_Q_over_a",
        "condition",
        "bounded",
        "True",
        1,
        None,
    ),
    "combined_source:ordinary_state_summary": (
        "src/_lcm/regime_building/decoupled.py",
        "get_decoupled_max_Q_over_a",
        "keyword",
        "bound_summary",
        "False",
        1,
        "productmap",
    ),
    "combined_source:tiled_state_summary": (
        "src/_lcm/regime_building/decoupled.py",
        "get_decoupled_max_Q_over_a",
        "keyword",
        "bound_summary",
        "False",
        1,
        "tiled_productmap",
    ),
    "combined_source:hard_return_summary": (
        "src/_lcm/regime_building/decoupled.py",
        "_DecoupledQOverA.__call__",
        "expression",
        (
            "_with_bound_summary(evaluated=evaluated, "
            "output=(index, value) if "
            "self.publishes_index else (value, "
            "empty))"
        ),
        "(index, value) if self.publishes_index else (value, empty)",
        1,
        None,
    ),
    "combined_source:ev1_value_return_summary": (
        "src/_lcm/regime_building/decoupled.py",
        "_DecoupledQOverA.__call__",
        "expression",
        "_with_bound_summary(evaluated=evaluated, output=(value, empty))",
        "(value, empty)",
        1,
        None,
    ),
    "combined_source:ev1_sample_return_summary": (
        "src/_lcm/regime_building/decoupled.py",
        "_DecoupledQOverA.__call__",
        "expression",
        (
            "_with_bound_summary(evaluated=evaluated, "
            "output=(jnp.where(empty, -1, "
            "flat_index).astype(jnp.int32), "
            "jnp.where(empty, -jnp.inf, "
            "jnp.ravel(conditional.value)[discrete])))"
        ),
        (
            "(jnp.where(empty, -1, "
            "flat_index).astype(jnp.int32), "
            "jnp.where(empty, -jnp.inf, "
            "jnp.ravel(conditional.value)[discrete]))"
        ),
        1,
        None,
    ),
    "combined_source:single_action_evaluation": (
        "src/_lcm/regime_building/decoupled.py",
        "_DecoupledQOverA.__call__",
        "statement",
        (
            "evaluated = "
            "self.parts(next_regime_to_V_arr=next_regime_to_V_arr, "
            "**states_actions_params)"
        ),
        (
            "evaluated = "
            "self.parts(next_regime_to_V_arr=next_regime_to_V_arr, "
            "**states_actions_params)\n"
            "evaluated = "
            "self.parts(next_regime_to_V_arr=next_regime_to_V_arr, "
            "**states_actions_params)"
        ),
        1,
        None,
    ),
    "combined_source:hard_readout_profile": (
        "src/_lcm/regime_building/decoupled.py",
        "argmax_and_readout",
        "expression",
        "jnp.ravel(readout)[index]",
        "jnp.ravel(selection)[index]",
        1,
        None,
    ),
    "combined_source:conditional_stored_profile": (
        "src/_lcm/regime_building/decoupled.py",
        "get_conditional_profiles",
        "expression",
        "parts.Q.reshape(*shape, -1)",
        'cast("FloatND", parts.Q_objective).reshape(*shape, -1)',
        1,
        None,
    ),
    "combined_source:ev1_economic_infinite_support": (
        "src/_lcm/regime_building/decoupled.py",
        "decoupled_logsum",
        "expression",
        "jnp.isfinite(conditional.objective)",
        "probabilities > 0",
        1,
        None,
    ),
    "combined_source:lookup_hard_readout": (
        "src/_lcm/regime_building/decoupled.py",
        "DecoupledWithActionParts.__call__",
        "keyword",
        "readout",
        "Q_objective",
        1,
        "argmax_and_readout",
    ),
    "combined_source:lookup_runtime_scale": (
        "src/_lcm/regime_building/decoupled.py",
        "DecoupledTasteShockLookup.__call__",
        "expression",
        'cast("FloatND", kwargs[TASTE_SHOCK_SCALE_PARAM])',
        "jnp.asarray(1.0)",
        1,
        None,
    ),
    "combined_source:objective_metadata_dispatch": (
        "src/_lcm/regime_building/Q_and_F.py",
        "has_objective_profile",
        "expression",
        'getattr(Q_and_F, "objective_aggregator", None) is not None',
        "False",
        1,
        None,
    ),
    "combined_source:ordinary_complete_parts": (
        "src/_lcm/regime_building/Q_and_F.py",
        "get_Q_and_F_parts",
        "expression",
        "kernel.parts",
        "kernel.__call__",
        1,
        None,
    ),
    "combined_source:parts_output_channel": (
        "src/_lcm/regime_building/Q_and_F.py",
        "_QAndF.parts",
        "expression",
        (
            "self._evaluate(next_regime_to_V_arr=next_regime_to_V_arr, "
            "**states_actions_params)[0]"
        ),
        (
            "self._evaluate(next_regime_to_V_arr=next_regime_to_V_arr, "
            "**states_actions_params)[1]"
        ),
        1,
        None,
    ),
    "combined_source:ordinary_flow_pool": (
        "src/_lcm/regime_building/Q_and_F.py",
        "_QAndF._evaluate",
        "expression",
        "self.U_and_F(**states_actions_params)",
        "self.U_and_F(**{})",
        1,
        None,
    ),
    "combined_source:objective_separate_from_stored": (
        "src/_lcm/regime_building/Q_and_F.py",
        "_QAndF._parts_from_utilities",
        "keyword",
        "Q_objective",
        "Q_arr",
        1,
        "QAndFParts",
    ),
    "combined_source:lookup_constructor_roles": (
        "src/_lcm/simulation/programs.py",
        "_lookup_body_and_roles",
        "statement",
        "return DecoupledWithActionParts, DECOUPLED_LOOKUP_OUTPUT_ROLES",
        "return _DecisionWithActionParts, ACTION_VALUES_OUTPUT_ROLES",
        1,
        None,
    ),
    "combined_source:lookup_original_summary": (
        "src/_lcm/simulation/programs.py",
        "_BoundActionPartsBody.__call__",
        "keyword",
        "bound_status",
        "()",
        1,
        "BoundedResult",
    ),
    "combined_source:lookup_same_evaluated_callback": (
        "src/_lcm/simulation/programs.py",
        "_evaluated_action_parts",
        "statement",
        "return parts",
        "return ()",
        1,
        None,
    ),
    "combined_source:unsupported_route_refusal": (
        "src/_lcm/solution/grid_search.py",
        "_fail_if_decoupled_route_is_unsupported",
        "condition",
        "unsupported",
        "False",
        1,
        None,
    ),
    "combined_source:rounded_equal_support": (
        "src/_lcm/egm/one_asset_egm_step.py",
        "_solve_viable_feature_row",
        "expression",
        "support[1:] >= support[:-1]",
        "support[1:] > support[:-1]",
        1,
        None,
    ),
    "combined_source:rightmost_original_endpoint": (
        "src/_lcm/egm/one_asset_egm_step.py",
        "_solve_viable_feature_row",
        "statement",
        "selected_original = indices[lower]",
        "selected_original = indices[jnp.maximum(lower - 1, 0)]",
        1,
        None,
    ),
    "combined_source:complete_endpoint_association": (
        "src/_lcm/egm/one_asset_egm_step.py",
        "_solve_viable_feature_row",
        "expression",
        "value_endog[selected_original]",
        "value_endog[indices[0]]",
        1,
        None,
    ),
}


def _upstream_composition_mutation_specs(
    *, repo_root: Path
) -> dict[str, dict[str, str]]:
    """Apply one named upstream expression, condition, keyword or statement change."""
    specs = {}
    for name, (
        relative,
        qualname,
        mode,
        old,
        new,
        expected_count,
        callee,
    ) in _UPSTREAM_COMPOSITION_MUTATIONS.items():
        tree = ast.parse((repo_root / relative).read_text(encoding="utf-8"))
        if "." in qualname:
            class_name, method_name = qualname.split(".", maxsplit=1)
            _, function = _method_definition(
                tree=tree, class_name=class_name, method_name=method_name
            )
        else:
            function = _definition(tree=tree, name=qualname)
        if mode == "statement":
            expected = ast.parse(old).body
            if len(expected) != 1:
                raise ValueError(
                    f"{name}: a statement control needs one original anchor"
                )
            matches = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.stmt)
                and _ast_key(node) == _ast_key(expected[0])
            ]
        elif mode == "keyword":
            matches = [
                keyword
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == callee
                for keyword in node.keywords
                if keyword.arg == old
            ]
        elif mode == "condition":
            expected = ast.parse(old, mode="eval").body
            matches = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.If | ast.IfExp)
                and _ast_key(node.test) == _ast_key(expected)
            ]
        elif mode == "expression":
            expected = ast.parse(old, mode="eval").body
            matches = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.expr) and _ast_key(node) == _ast_key(expected)
            ]
        else:
            raise ValueError(f"{name}: unreviewed upstream mutation mode {mode!r}")
        if len(matches) != expected_count or expected_count != 1:
            raise ValueError(f"{name}: expected one anchor, found {len(matches)}")
        if mode == "statement":
            replacement = ast.parse(new).body
            for parent in ast.walk(function):
                for attribute, values in ast.iter_fields(parent):
                    if isinstance(values, list) and any(
                        value is matches[0] for value in values
                    ):
                        changed = []
                        for value in values:
                            changed.extend(
                                copy.deepcopy(replacement)
                                if value is matches[0]
                                else [value]
                            )
                        setattr(parent, attribute, changed)
        elif mode == "keyword":
            cast("ast.keyword", matches[0]).value = ast.parse(new, mode="eval").body
        elif mode == "condition":
            cast("ast.If | ast.IfExp", matches[0]).test = ast.parse(
                new, mode="eval"
            ).body
        else:
            replacement = ast.parse(new, mode="eval").body
            for parent in ast.walk(function):
                for attribute, value in ast.iter_fields(parent):
                    if value is matches[0]:
                        setattr(parent, attribute, copy.deepcopy(replacement))
                    elif isinstance(value, list):
                        setattr(
                            parent,
                            attribute,
                            [
                                copy.deepcopy(replacement)
                                if item is matches[0]
                                else item
                                for item in value
                            ],
                        )
        mutated = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        ast.parse(mutated, filename=relative)
        specs[name] = {"path": relative, "source": mutated}
    return specs


def _upstream_composition_errors(*, tree: ast.Module, source: SourcePath) -> list[str]:
    """Authenticate the finite upstream composition bodies in their canonical store."""
    surface, callables = _UPSTREAM_COMPOSITION_CONTRACTS[source]
    errors = _exact_callable_errors(
        tree=tree, label="combined upstream dependency", contracts=callables
    )
    if _transport_module_surface(tree) != surface:
        errors.append(
            f"combined upstream dependency: {source} declaration surface changed"
        )
    return errors
