"""The simulation certificate follows the published program into dispatch.

Refreshing byte seals cannot admit a changed candidate transport: each seeded
defect must also fail the independently pinned program corridor.
"""

from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.direct_flow import (
    direct_flow_mutation_specs,
    verify_direct_candidate_flow,
)
from tests.candidate_certificate.generate_sources import sha256_file

_PROGRAM_MUTATIONS = (
    "dense_reducer_replaced",
    "q_and_f_replaced",
    "action_coordinates_reversed",
    "action_order_changed",
    "hard_max_reduction_replaced",
    "streamed_width_ignored",
    "streamed_q_and_f_filtered",
    "streamed_index_shifted",
    "streamed_value_filtered",
    "subject_tiles_reversed",
    "argument_mapping_filtered",
    "decision_mapping_dropped",
    "streamed_guard_bypassed",
    "program_snapshot_filtered",
    "resolved_body_bypassed",
    "lowered_body_replaced",
    "dispatch_family_replaced",
    "dispatch_arguments_filtered",
    "body_cache_identity_dropped",
    "compiler_options_dropped",
    "duplicate_future_replaced",
    "resolved_widths_ignored",
    "prewarm_program_replaced",
    "prewarm_failure_hidden",
)

# Local controls for newly profiled helpers supplement the historical registry.
_RANDOM_HELPER_MUTATIONS = {
    "simulation_random:default_implementation_ignored": (
        '"impl": jax.config.jax_default_prng_impl,',
        '"impl": "threefry2x32",',
    ),
    "simulation_random:offset_trace_guard_bypassed": (
        "if seed_offset != jax.config.jax_random_seed_offset:",
        "if False and seed_offset != jax.config.jax_random_seed_offset:",
    ),
    "simulation_random:partition_context_ignored": (
        '{"partitionable": jax.config.jax_threefry_partitionable}',
        '{"partitionable": False}',
    ),
    "simulation_random:declared_split_mode_ignored": (
        (
            "with jax.threefry_partitionable(partitionable):\n"
            "        next_states_key, next_regime_key, next_key = "
            "jax.random.split(key=key, num=3)"
        ),
        (
            "with jax.threefry_partitionable(False):\n"
            "        next_states_key, next_regime_key, next_key = "
            "jax.random.split(key=key, num=3)"
        ),
    ),
    "simulation_random:population_split_context_ignored": (
        (
            '"partitionable": jax.config.jax_threefry_partitionable,\n'
            '            "subject_window":'
        ),
        '"partitionable": False,\n            "subject_window":',
    ),
    "simulation_random:declared_population_mode_ignored": (
        "with jax.threefry_partitionable(partitionable):\n        for name in names:",
        "with jax.threefry_partitionable(False):\n        for name in names:",
    ),
}

_PROFILED_HELPER_MUTATIONS = {
    "simulation_pandas:dataframe_writer_omitted": (
        "src/lcm/model.py",
        (
            "                regime_names_to_ids=self.regime_names_to_ids,\n"
            "                array_writer=entry_allocations,"
        ),
        (
            "                regime_names_to_ids=self.regime_names_to_ids,\n"
            "                array_writer=None,"
        ),
    ),
    "simulation_pandas:series_writer_omitted": (
        "src/lcm/model.py",
        (
            "                regime_names_to_ids=self.regime_names_to_ids,\n"
            "                array_writer=array_writer,"
        ),
        (
            "                regime_names_to_ids=self.regime_names_to_ids,\n"
            "                array_writer=None,"
        ),
    ),
    "simulation_pandas:nested_writer_omitted": (
        "src/_lcm/pandas_utils.py",
        (
            "        array_writer=array_writer,\n"
            "    )\n\n    if isinstance(value, pd.Series):"
        ),
        ("        array_writer=None,\n    )\n\n    if isinstance(value, pd.Series):"),
    ),
    "simulation_pandas:materialization_admission_bypassed": (
        "src/_lcm/pandas_utils.py",
        "return array_writer(value=value, dtype=dtype, name=name)",
        "return jnp.array(value, dtype=dtype)",
    ),
    "simulation_pandas:empty_series_admission_bypassed": (
        "src/_lcm/pandas_utils.py",
        "    if len(series) == 0:\n        if array_writer is not None:",
        "    if len(series) == 0:\n        if False and array_writer is not None:",
    ),
    "simulation_pandas:role_writer_omitted": (
        "src/_lcm/pandas_utils.py",
        (
            '            labels=df["own_stakeholder"],\n'
            "            user_regimes=user_regimes,\n"
            "            array_writer=array_writer,"
        ),
        (
            '            labels=df["own_stakeholder"],\n'
            "            user_regimes=user_regimes,\n"
            "            array_writer=None,"
        ),
    ),
    "simulation_pandas:completed_mapping_owner_omitted": (
        "src/lcm/model.py",
        (
            'array_writer.publish(stage="params", tree=flat_params)\n'
            "        flat_params = cast_params_to_canonical_dtypes("
        ),
        (
            'array_writer.publish(stage="params", tree={})\n'
            "        flat_params = cast_params_to_canonical_dtypes("
        ),
    ),
    "simulation_pandas:resolved_solution_owner_omitted": (
        "src/lcm/model.py",
        (
            "            period_to_regime_to_replay_reader = None\n"
            "        if entry_allocations is not None:\n"
            "            entry_allocations.update_solution("
        ),
        (
            "            period_to_regime_to_replay_reader = None\n"
            "        if entry_allocations is not None:\n"
            "            (lambda **kwargs: None)("
        ),
    ),
    "simulation_entry:automatic_solve_owners_omitted": (
        "src/lcm/model.py",
        "else entry_allocations.solve_input_roots()",
        "else ()",
    ),
    "simulation_entry:private_solve_owners_omitted": (
        "src/lcm/model.py",
        (
            "            collect_solver_diagnostics=True,\n"
            "            retained_input_arrays=retained_input_arrays,"
        ),
        (
            "            collect_solver_diagnostics=True,\n"
            "            retained_input_arrays=(),"
        ),
    ),
    "simulation_entry:compiled_solve_owners_omitted": (
        "src/lcm/model.py",
        "                retained_input_arrays=retained_input_arrays,",
        "                retained_input_arrays=(),",
    ),
    "simulation_entry:original_solve_inputs_omitted": (
        "src/_lcm/simulation/entry_allocations.py",
        "            self.original_inputs.arrays,",
        "            (),",
    ),
    "simulation_entry:normalized_solve_inputs_omitted": (
        "src/_lcm/simulation/entry_allocations.py",
        "            self.model_roots,\n            tuple(self._stages.values()),",
        "            self.model_roots,\n            (),",
    ),
    "simulation_entry:fixed_solve_inventory_omitted": (
        "src/_lcm/solution/backward_induction.py",
        "        fixed_input_arrays=(\n            retained_input_arrays,",
        "        fixed_input_arrays=(\n            (),",
    ),
    "simulation_entry:fixed_solve_inputs_rebound": (
        "src/_lcm/solution/backward_induction.py",
        "    capture_target = resolve_capture_target()",
        "    retained_input_arrays = ()\n    capture_target = resolve_capture_target()",
    ),
    "simulation_entry:model_owner_inventory_omitted": (
        "src/_lcm/simulation/entry_allocations.py",
        "                        self.model_roots,",
        "                        (),",
    ),
    "simulation_entry:preflight_owner_inventory_omitted": (
        "src/lcm/model.py",
        (
            "entry_allocations.snapshot()\n"
            "                if entry_allocations is not None "
            "and validation_enabled(log)"
        ),
        (
            "entry_inputs.footprint(solution=solution)\n"
            "                if entry_allocations is not None "
            "and validation_enabled(log)"
        ),
    ),
    "simulation_entry:completed_output_owner_omitted": (
        "src/_lcm/simulation/entry_allocations.py",
        "        self._pending.append(result)\n        return result",
        "        return result",
    ),
    "simulation_entry:resolved_view_inventory_omitted": (
        "src/_lcm/simulation/entry_allocations.py",
        "                        self._resolved_inputs,",
        "                        (),",
    ),
    "simulation_entry:canonical_padding_inputs_omitted": (
        "src/_lcm/simulation/entry_allocations.py",
        'self._stages["initial"] = initial_conditions',
        'self._stages["initial"] = MappingProxyType({})',
    ),
    "simulation_entry:padding_budget_bypassed": (
        "src/_lcm/simulation/entry_allocations.py",
        "                    budget_bytes=self.budget_bytes,",
        "                    budget_bytes=2**63 - 1,",
    ),
    "simulation_entry:excluded_source_inventory_omitted": (
        "src/_lcm/simulation/entry_allocations.py",
        "budget_devices = tuple(dict.fromkeys((*self.devices, *live.spans)))",
        "budget_devices = self.devices",
    ),
    "simulation_entry:padding_wrong_row": (
        "src/_lcm/simulation/entry_allocations.py",
        "jnp.repeat(array[-1:], pad, axis=0)",
        "jnp.repeat(array[:1], pad, axis=0)",
    ),
    "simulation_entry:parameter_writer_bypassed": (
        "src/lcm/model.py",
        "flat_params, array_writer=array_writer",
        "flat_params, array_writer=None",
    ),
    "simulation_entry:integer_upload_bypassed": (
        "src/_lcm/dtypes.py",
        "return array_writer(value=np_value, dtype=np.dtype(np.int32), name=name)",
        "return jnp.asarray(np_value, dtype=jnp.int32)",
    ),
    "simulation_finite_replay:diagnostic_bypassed": (
        "src/_lcm/simulation/simulate.py",
        "        dropped=live & ~represented,",
        "        dropped=jnp.zeros_like(live),",
    ),
    "simulation_finite_replay:canonical_ranking_bypassed": (
        "src/_lcm/simulation/simulate.py",
        "ranking_values = jnp.where(valid, canonical_values, -jnp.inf)",
        "ranking_values = jnp.where(valid, candidate_inner, -jnp.inf)",
    ),
    "simulation_preflight:initial_feasibility_bypassed": (
        "src/_lcm/simulation/initial_conditions.py",
        "            _collect_feasibility_errors(",
        "            (lambda **kwargs: None)(",
    ),
    "simulation_preflight:transition_validation_bypassed": (
        "src/_lcm/simulation/initial_conditions.py",
        "            validate_transitions(",
        "            (lambda **kwargs: None)(",
    ),
    "simulation_preflight:admission_refusal_retried": (
        "src/_lcm/simulation/initial_conditions.py",
        "except ExecutionPlanningError, MemoryError, jax.errors.JaxRuntimeError:",
        "except MemoryError, jax.errors.JaxRuntimeError:",
    ),
    "simulation_preflight:entry_residency_omitted": (
        "src/lcm/model.py",
        (
            "retained_footprint=(\n"
            "                entry_allocations.snapshot()\n"
            "                if entry_allocations is not None "
            "and validation_enabled(log)"
        ),
        (
            "retained_footprint=(\n"
            "                None\n"
            "                if entry_allocations is not None "
            "and validation_enabled(log)"
        ),
    ),
    "simulation_preflight:invalid_discrete_cohort_accepted": (
        "src/_lcm/simulation/initial_conditions.py",
        "or host[age_stop:].any()",
        "or False",
    ),
    "simulation_taste_stream:namespace_changed": (
        "src/_lcm/simulation/taste_stream.py",
        '"pylcm.taste-stream.v1",',
        '"pylcm.taste-stream.v2",',
    ),
    "simulation_taste_stream:action_domain_ignored": (
        "src/_lcm/simulation/taste_stream.py",
        "        domain,\n    )",
        "        (),\n    )",
    ),
    "simulation_taste_stream:ordinary_carry_not_advanced": (
        "src/_lcm/simulation/taste_stream.py",
        "return next_key, taste_keys",
        "return key, taste_keys",
    ),
    "simulation_taste_stream:ambient_implementation_used": (
        "src/_lcm/simulation/taste_stream.py",
        '"impl": "threefry2x32",',
        '"impl": jax.config.jax_default_prng_impl,',
    ),
    "simulation_taste_stream:dynamic_address_ignored": (
        "src/_lcm/simulation/taste_stream.py",
        '"address_words": np.asarray(address_words, dtype=np.uint32),',
        '"address_words": np.zeros(8, dtype=np.uint32),',
    ),
    "simulation_taste_stream:chunk_start_ignored": (
        "src/_lcm/simulation/taste_stream.py",
        '"subject_start": _encode_subject_row(row=start),',
        '"subject_start": _encode_subject_row(row=0),',
    ),
    "simulation_taste_stream:subject_output_layout_omitted": (
        "src/_lcm/simulation/taste_stream.py",
        "subject_outputs=True,",
        "subject_outputs=False,",
    ),
    "simulation_taste_stream:root_caller_owners_omitted": (
        "src/_lcm/simulation/taste_stream.py",
        "memory.hold(tree=live_inputs)",
        "memory.hold(tree=())",
    ),
    "simulation_host:unused_input_omitted_from_peak": (
        "src/_lcm/simulation/host_operations.py",
        "jax.jit(bound, keep_unused=True)",
        "jax.jit(bound, keep_unused=False)",
    ),
    "compiler_inputs:dynamic_tree_mismatch_accepted": (
        "src/_lcm/execution/compiler_inputs.py",
        "if actual_tree != sharding_tree:",
        "if False and actual_tree != sharding_tree:",
    ),
    "workspace:per_candidate_residency_ignored": (
        "src/_lcm/execution/workspace_planning.py",
        "resident_bytes_for=resident_bytes_for,",
        "resident_bytes_for=None,",
    ),
    "workspace:resident_lower_bound_bypassed": (
        "src/_lcm/execution/workspace_planning.py",
        "if resident < lower_bound:",
        "if False and resident < lower_bound:",
    ),
    "simulation_runtime:per_candidate_residency_ignored": (
        "src/_lcm/simulation/runtime.py",
        "resident_bytes_for=resident_lookup,",
        "resident_bytes_for=None,",
    ),
    "simulation_runtime:eliminated_argument_excluded": (
        "src/_lcm/simulation/runtime.py",
        "tree=tuple(leaf for path, leaf in with_paths if path in kept)",
        "tree=tuple(leaf for path, leaf in with_paths)",
    ),
    "solution_runtime:fixed_owner_inventory_omitted": (
        "src/_lcm/solution/backward_induction.py",
        "\n            fixed_bytes=fixed_bytes,",
        "\n            fixed_bytes={},",
    ),
    "solution_runtime:shared_copy_destinations_omitted": (
        "src/_lcm/solution/backward_induction.py",
        "                shared_copies=copies_by_period[period],",
        "                shared_copies={},",
    ),
    "solution_runtime:unshared_eliminated_copy_omitted": (
        "src/_lcm/solution/backward_induction.py",
        "elif not kept:",
        "elif False and not kept:",
    ),
    "solution_runtime:eliminated_aligned_read_excluded": (
        "src/_lcm/solution/backward_induction.py",
        "and _compiler_reads_source(shardings=input_shardings, source=read.source)",
        "and True",
    ),
    "solution_runtime:fixed_source_owners_omitted": (
        "src/_lcm/solution/backward_induction.py",
        (
            "        fixed_input_arrays=(\n"
            "            retained_input_arrays,\n"
            "            tuple(\n"
            "                (space.states, space.discrete_actions, "
            "space.continuous_actions)\n"
            "                for space in base_state_action_spaces.values()"
        ),
        (
            "        fixed_input_arrays=(\n"
            "            retained_input_arrays,\n"
            "            tuple(\n"
            "                (space.states, space.discrete_actions, "
            "space.continuous_actions)\n"
            "                for space in ()"
        ),
    ),
    "solution_runtime:internal_output_reservation_omitted": (
        "src/_lcm/solution/backward_induction.py",
        "internal_bytes=internal_bytes[triple[:2]],",
        "internal_bytes=0,",
    ),
    "solution_runtime:wrong_internal_cell": (
        "src/_lcm/solution/backward_induction.py",
        "cell = candidate[0][:2]",
        "cell = ('other_regime', 0)",
    ),
    "solution_inventory:fixed_owner_charge_omitted": (
        "src/_lcm/execution/footprint.py",
        "+ self.fixed_bytes.get(device, 0)",
        "+ 0",
    ),
    "solution_inventory:shared_copy_charge_omitted": (
        "src/_lcm/execution/footprint.py",
        "if key not in copies and device in footprint.device_ids",
        "if False and key not in copies and device in footprint.device_ids",
    ),
    "simulation_host:subject_output_layout_omitted": (
        "src/_lcm/simulation/host_operations.py",
        "else jax.jit(bound, keep_unused=True, out_shardings=output_sharding)",
        "else jax.jit(bound, keep_unused=True)",
    ),
    "simulation_host:subject_output_cache_identity_omitted": (
        "src/_lcm/simulation/host_operations.py",
        'layout_key=("simulation_host_operation", subject_outputs),',
        'layout_key=("simulation_host_operation", False),',
    ),
    "simulation_caller:initialized_state_ownership_omitted": (
        "src/_lcm/simulation/simulate.py",
        "memory.hold(tree=(states, key))",
        "memory.hold(tree=key)",
    ),
    "simulation_caller:placed_age_ownership_omitted": (
        "src/_lcm/simulation/simulate.py",
        "memory.hold(tree=age)",
        "memory.hold(tree=())",
    ),
}


_COMBINED_INPUT_MUTATIONS = {
    "foreign_copy:mixed_backend_admitted": (
        "src/_lcm/simulation/solution_copies.py",
        (
            "if len(platforms) != 1 or any(\n"
            "        device.platform not in platforms for device in lea"
            "f.sharding.device_set\n"
            "    ):"
        ),
        "if False:",
    ),
    "foreign_copy:source_devices_omitted": (
        "src/_lcm/simulation/solution_copies.py",
        "dict.fromkeys((*budget_devices, *live.spans, *devices))",
        "dict.fromkeys(budget_devices)",
    ),
    "foreign_copy:cache_layout_omitted": (
        "src/_lcm/simulation/solution_copies.py",
        'layout_key=("foreign_solution_copy", leaf.sharding),',
        'layout_key=("foreign_solution_copy",),',
    ),
    "foreign_copy:resident_bank_ignored": (
        "src/_lcm/simulation/solution_copies.py",
        "resident_bytes=max(external.values()),",
        "resident_bytes=0,",
    ),
    "foreign_copy:budget_ignored": (
        "src/_lcm/simulation/solution_copies.py",
        "        budget_bytes=budget_bytes,\n        resident_bytes=",
        "        budget_bytes=None,\n        resident_bytes=",
    ),
    "foreign_copy:source_returned": (
        "src/_lcm/simulation/solution_copies.py",
        "return jnp.array(value, copy=True)",
        "return value",
    ),
    "foreign_copy:physical_isolation_ignored": (
        "src/_lcm/simulation/solution_copies.py",
        "exclusive[device] != sum(stop - start for start, stop in spans)",
        "False",
    ),
    "foreign_owner:copy_not_retained": (
        "src/_lcm/simulation/entry_allocations.py",
        "self._foreign_copies.append(result)",
        "self._foreign_copies.extend(())",
    ),
    "foreign_owner:copy_bank_not_inventoried": (
        "src/_lcm/simulation/entry_allocations.py",
        "                        tuple(self._foreign_copies),",
        "                        (),",
    ),
    "foreign_model:foreign_copy_dependency_omitted": (
        "src/lcm/model.py",
        "array_copier=entry_allocations.copy_solution_leaf,",
        "array_copier=None,",
    ),
    "foreign_model:copy_released_before_publication": (
        "src/lcm/model.py",
        (
            "                entry_allocations.update_solution(\n"
            "                    solution=solution,"
        ),
        (
            "                entry_allocations.release_foreign_copies()"
            "\n"
            "                entry_allocations.update_solution(\n"
            "                    solution=solution,"
        ),
    ),
    "foreign_model:owner_persisted_in_memo": (
        "src/lcm/model.py",
        "consumed_views[memo_key] = resolved",
        "consumed_views[memo_key] = (resolved, entry_allocations)",
    ),
    "foreign_model:artifact_route_guard_bypassed": (
        "src/lcm/model.py",
        (
            "if array_copier is not None:\n"
            "            for regime_name, regime in self._regimes.items"
            "():"
        ),
        (
            "if False and array_copier is not None:\n"
            "            for regime_name, regime in self._regimes.items"
            "():"
        ),
    ),
    "foreign_model:envelope_copy_dependency_omitted": (
        "src/lcm/model.py",
        "solution=solution, array_copier=array_copier\n        )",
        "solution=solution, array_copier=None\n        )",
    ),
    "foreign_model:materialization_dependency_omitted": (
        "src/lcm/model.py",
        (
            "else value_store._materialize_with_copy(  # noqa: SLF001\n"
            "                    array_copier=array_copier"
        ),
        (
            "else value_store._materialize_with_copy(  # noqa: SLF001\n"
            "                    array_copier=None"
        ),
    ),
    "foreign_snapshot:host_upload_admitted": (
        "src/_lcm/solution/result_snapshot.py",
        "if not isinstance(entry.value, jax.Array):",
        "if False and not isinstance(entry.value, jax.Array):",
    ),
    "foreign_entry:source_copy_dependency_omitted": (
        "src/lcm/_solver_api/entries.py",
        "else value._fresh(array_copier=array_copier)",
        "else value._fresh(array_copier=None)",
    ),
    "foreign_entry:owner_stored": (
        "src/lcm/_solver_api/entries.py",
        "return _CanonicalValueEntry(value=private)",
        "return _CanonicalValueEntry(value=(private, array_copier))",
    ),
    "foreign_store:load_dependency_omitted": (
        "src/lcm/_solver_api/stores.py",
        "value = entry._fresh(array_copier=array_copier)",
        "value = entry._fresh(array_copier=None)",
    ),
    "solve_descriptors:metadata_validation_bypassed": (
        "src/_lcm/execution/abstract_program_inputs.py",
        (
            "_validate_transfer_argument_metadata(\n"
            "            program=program, read=read, transfer=transfer\n"
            "        )"
        ),
        "pass",
    ),
    "solve_descriptors:source_occurrences_conflated": (
        "src/_lcm/execution/abstract_program_inputs.py",
        "occurrence = _value_read_argument_leaf(program=described, read=read)",
        "occurrence = _value_read_argument_leaf(program=program, read=read)",
    ),
    "solve_descriptors:concrete_inputs_retained": (
        "src/_lcm/execution/abstract_program_inputs.py",
        "return replace(program, arguments=required)",
        "return program",
    ),
    "solve_descriptors:weak_type_ignored": (
        "src/_lcm/execution/abstract_program_inputs.py",
        "weak_type=abstract.weak_type,",
        "weak_type=False,",
    ),
    "solve_descriptors:planning_transfer_executed": (
        "src/_lcm/solution/backward_induction.py",
        "input_transfer_plan=transfer_plan,\n                abstract_inputs=True,",
        "input_transfer_plan=transfer_plan,\n                abstract_inputs=False,",
    ),
    "abstract_core:concrete_operand_admitted": (
        "src/_lcm/execution/core_program.py",
        "if abstract_inputs:\n        _validate_abstract_inputs(program=program)",
        (
            "if False and abstract_inputs:\n"
            "        _validate_abstract_inputs(program=program)"
        ),
    ),
    "chunk_model:resolved_input_inventory_omitted": (
        "src/lcm/model.py",
        (
            "retained_footprint=entry_allocations.snapshot(),\n"
            "                independent_taste="
        ),
        (
            "retained_footprint=DeviceBufferFootprint(spans={}),\n"
            "                independent_taste="
        ),
    ),
    "chunk_model:selected_profile_not_dispatched": (
        "src/lcm/model.py",
        "prepared_chunks=prepared_chunks,",
        "prepared_chunks=None,",
    ),
    "chunk_admission:retained_values_ignored": (
        "src/_lcm/simulation/chunk_admission.py",
        "                retained_footprint,",
        "                DeviceBufferFootprint(spans={}),",
    ),
    "chunk_admission:unpublished_reservation_fulfilled": (
        "src/_lcm/simulation/chunk_admission.py",
        "device: value - min(value, outputs.get(device, 0))",
        "device: 0",
    ),
    "chunk_admission:all_inner_choices_skipped": (
        "src/_lcm/simulation/chunk_admission.py",
        "for widths in choices:",
        "for widths in choices[:1]:",
    ),
    "chunk_planning:compiler_devices_not_checked": (
        "src/_lcm/simulation/chunk_planning.py",
        "if set(self.devices) != compiled_devices:",
        "if False and set(self.devices) != compiled_devices:",
    ),
    "chunk_inventory:logical_aliases_deduplicated": (
        "src/_lcm/simulation/chunk_profile_inventory.py",
        "for leaf in jax.tree.leaves(tree):",
        "for leaf in {id(item): item for item in jax.tree.leaves(tree)}.values():",
    ),
    "chunk_profiles:publication_slots_omitted": (
        "src/_lcm/simulation/chunk_profiles.py",
        "add_bytes(target=published, source=payload_bytes(tree=record))",
        "add_bytes(target=published, source={})",
    ),
    "chunk_profiles:whole_population_rng_changed": (
        "src/_lcm/simulation/chunk_profiles.py",
        '"n_initial_states": population,',
        '"n_initial_states": width,',
    ),
    "chunk_profiles:ordinary_taste_dtype_conflated": (
        "src/_lcm/simulation/chunk_profiles.py",
        'impl="threefry2x32")',
        "impl=jax.config.jax_default_prng_impl)",
    ),
    "chunk_profiles:period_transfer_reservation_omitted": (
        "src/_lcm/simulation/chunk_profiles.py",
        "required.device_set, transfer.cost.per_device_bytes",
        "required.device_set, 0",
    ),
    "chunk_profiles:outer_storage_omitted": (
        "src/_lcm/simulation/chunk_profiles.py",
        "output_reservation=output_bank,",
        "output_reservation={},",
    ),
    "chunk_dispatch:reserved_width_remaximized": (
        "src/_lcm/simulation/runtime.py",
        "fixed[axis.name] = selected",
        "fixed[axis.name] = axis.extent",
    ),
    "chunk_dispatch:explicit_width_conflict_ignored": (
        "src/_lcm/simulation/runtime.py",
        (
            "axis.name in configured\n"
            "            and min(configured[axis.name], axis.extent) !="
            " selected"
        ),
        "False",
    ),
    "chunk_dispatch:unit_width_handoff_omitted": (
        "src/_lcm/simulation/unit_executor.py",
        "axis_widths=self.axis_widths,",
        "axis_widths={},",
    ),
    "chunk_dispatch:readmission_before_slice_omitted": (
        "src/_lcm/simulation/simulate.py",
        (
            "prepared_chunks.require_chunk(\n"
            "                memory=memory, completed_setup=completed_s"
            "etup\n"
            "            )"
        ),
        "pass",
    ),
    "chunk_dispatch:published_chunks_not_owned": (
        "src/_lcm/simulation/simulate.py",
        "memory.replace_outputs(tree=chunk_results)",
        "memory.replace_outputs(tree=())",
    ),
    "chunk_diagnostics:live_carry_dropped": (
        "src/_lcm/simulation/simulate.py",
        (
            "                prev_regime_ids,\n"
            "                subject_regime_ids,\n"
            "                new_subject_regime_ids,"
        ),
        "                (),\n                (),\n                (),",
    ),
    "chunk_diagnostics:profiled_nonfinite_bypassed": (
        "src/_lcm/simulation/simulate.py",
        "non_finite_by_regime(**arguments)\n        if memory is None",
        "non_finite_by_regime(**arguments)\n        if True",
    ),
    "chunk_abstract:operation_concrete_operand_admitted": (
        "src/_lcm/simulation/host_operations.py",
        "if not isinstance(tree, jax.ShapeDtypeStruct):",
        "if False and not isinstance(tree, jax.ShapeDtypeStruct):",
    ),
    "foreign_model:supplied_resolution_owner_omitted": (
        "src/lcm/model.py",
        (
            "entry_allocations=entry_allocations,\n"
            "            )\n"
            "        else:\n"
            "            period_to_regime_to_V_arr = None"
        ),
        (
            "entry_allocations=None,\n"
            "            )\n"
            "        else:\n"
            "            period_to_regime_to_V_arr = None"
        ),
    ),
    "foreign_model:automatic_resolution_owner_omitted": (
        "src/lcm/model.py",
        (
            "entry_allocations=entry_allocations,\n"
            "            )\n"
            "        if (\n"
            "            period_to_regime_to_V_arr is None"
        ),
        (
            "entry_allocations=None,\n"
            "            )\n"
            "        if (\n"
            "            period_to_regime_to_V_arr is None"
        ),
    ),
    "foreign_model:value_snapshot_dependency_omitted": (
        "src/lcm/model.py",
        'cast("ValueStore", supplied_values), array_copier=array_copier',
        'cast("ValueStore", supplied_values), array_copier=None',
    ),
    "foreign_snapshot:store_dependency_omitted": (
        "src/_lcm/solution/result_snapshot.py",
        'entries=cast("Mapping[object, object]", entries), array_copier=array_copier',
        'entries=cast("Mapping[object, object]", entries), array_copier=None',
    ),
    "foreign_entry:owned_read_dependency_omitted": (
        "src/lcm/_solver_api/entries.py",
        'value=self.value, label="Owned solution value", array_copier=array_copier',
        'value=self.value, label="Owned solution value", array_copier=None',
    ),
    "foreign_entry:leaf_dependency_omitted": (
        "src/lcm/_solver_api/entries.py",
        "leaf=value, label=label, array_copier=array_copier",
        "leaf=value, label=label, array_copier=None",
    ),
    "foreign_store:admission_dependency_omitted": (
        "src/lcm/_solver_api/stores.py",
        "else _canonical_value_entry(value=value, array_copier=array_copier)",
        "else _canonical_value_entry(value=value, array_copier=None)",
    ),
    "foreign_store:flat_coordinate_dependency_omitted": (
        "src/lcm/_solver_api/stores.py",
        "value=value,\n                    array_copier=array_copier,",
        "value=value,\n                    array_copier=None,",
    ),
    "foreign_store:nested_coordinate_dependency_omitted": (
        "src/lcm/_solver_api/stores.py",
        "value=value,\n                        array_copier=array_copier,",
        "value=value,\n                        array_copier=None,",
    ),
    "foreign_store:materialize_dependency_omitted": (
        "src/lcm/_solver_api/stores.py",
        "period=period, regime=regime, array_copier=array_copier",
        "period=period, regime=regime, array_copier=None",
    ),
}


_FINITE_BUDGET_MUTATIONS = {
    "finite_budget:retained_policy_mapping_omitted": (
        "src/lcm/model.py",
        "                policies=period_to_regime_to_sim_policy,",
        "                policies=None,",
    ),
    "finite_budget:profile_payload_omitted": (
        "src/_lcm/simulation/forward_program_profiles.py",
        "payload=abstract_payload, states=states, params=params, age=age",
        "payload=None, states=states, params=params, age=age",
    ),
    "finite_budget:prepared_bank_schema_bypassed": (
        "src/_lcm/simulation/forward_program_profiles.py",
        "bank=preparation.executable.out_info,",
        "bank=abstract_payload,",
    ),
    "finite_budget:required_policy_layout_ignored": (
        "src/_lcm/simulation/forward_program_profiles.py",
        "\n            stored_sharding=leaf.sharding, devices=devices\n",
        (
            "\n            stored_sharding=leaf.sharding, devices=tupl"
            "e(leaf.sharding.device_set)\n"
        ),
    ),
    "finite_budget:preparation_storage_replaced": (
        "src/_lcm/simulation/chunk_profiles.py",
        'profile=cores["policy_prepare"],',
        'profile=cores["decision"],',
    ),
    "finite_budget:diagnostic_profile_replaced": (
        "src/_lcm/simulation/chunk_profiles.py",
        "function=dropped_candidate_counts,",
        "function=_empty_fallback,",
    ),
    "finite_budget:distinct_artifact_copies_merged": (
        "src/_lcm/simulation/chunk_profiles.py",
        "identity = (read.target, required)",
        "identity = (read.target.kind, required)",
    ),
    "finite_budget:policy_locator_value_replaced": (
        "src/_lcm/simulation/chunk_profiles.py",
        "(read.target, leaf) for read, leaf in zip(reads, leaves, strict=True)",
        "(read.target, leaves[0]) for read, leaf in zip(reads, leaves, strict=True)",
    ),
    "finite_budget:copy_scratch_omitted": (
        "src/_lcm/simulation/chunk_profiles.py",
        "transfer.cost.temporary_bytes,",
        "0,",
    ),
    "finite_budget:canonical_next_values_omitted": (
        "src/_lcm/simulation/program_arguments.py",
        (
            '        "age": age,\n        "next_regime_to_V_arr": next'
            "_values,\n        **references,"
        ),
        (
            '        "age": age,\n        "next_regime_to_V_arr": {},\n'
            "        **references,"
        ),
    ),
    "finite_budget:diagnostic_admission_bypassed": (
        "src/_lcm/simulation/simulate.py",
        "            memory=memory,\n            function=dropped_candidate_counts,",
        "            memory=None,\n            function=dropped_candidate_counts,",
    ),
}


_EAGER_PLACEMENT_MUTATIONS = {
    "eager_internal:declaration_guard_ignored": (
        "src/_lcm/execution/eager_core.py",
        (
            "internal_input_templates.keys() != program.requirements."
            "internal_inputs.keys()"
        ),
        "False",
    ),
    "eager_internal:producer_operand_omitted": (
        "src/_lcm/execution/eager_core.py",
        "            placed.update(\n",
        "            {}.update(\n",
    ),
    "eager_internal:producer_layout_replaced": (
        "src/_lcm/execution/eager_core.py",
        "                sharding=value.sharding,",
        "                sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),",
    ),
    "eager_internal:selected_templates_omitted": (
        "src/_lcm/solution/backward_induction.py",
        (
            "internal_input_templates=internal_templates[\n           "
            "             selected_candidates[triple]\n               "
            "     ],"
        ),
        "internal_input_templates={},",
    ),
    "solve_descriptors:nested_callback_reintroduced": (
        "src/_lcm/execution/abstract_program_inputs.py",
        (
            "    arguments = jax.tree.map(\n        _OperandDescriptor"
            "(default_sharding=shared), program.arguments\n    )"
        ),
        (
            "    def describe(value: object) -> jax.ShapeDtypeStruct:"
            "\n        return _OperandDescriptor(default_sharding=shar"
            "ed)(value)\n\n    arguments = jax.tree.map(describe, progr"
            "am.arguments)"
        ),
    ),
    "solve_descriptors:declared_operand_layout_ignored": (
        "src/_lcm/execution/abstract_program_inputs.py",
        "if isinstance(value, jax.ShapeDtypeStruct) and value.sharding is not None:",
        (
            "if False and isinstance(value, jax.ShapeDtypeStruct) and"
            " value.sharding is not None:"
        ),
    ),
    "eager_core:explicit_mesh_bypassed": (
        "src/_lcm/execution/eager_core.py",
        "if self.mesh is not None:",
        "if False and self.mesh is not None:",
    ),
    "eager_core:default_device_replaced": (
        "src/_lcm/execution/eager_core.py",
        "with jax.default_device(self.device):",
        "with jax.default_device(jax.devices()[0]):",
    ),
    "eager_core:descriptor_omitted_from_memo": (
        "src/_lcm/execution/eager_core.py",
        "key = (id(value), template)",
        "key = (id(value), None)",
    ),
    "eager_core:concrete_memo_retained": (
        "src/_lcm/execution/eager_core.py",
        "placement.results.clear()",
        "pass",
    ),
    "eager_core:committed_layout_guard_bypassed": (
        "src/_lcm/execution/eager_core.py",
        "if not runtime_shardings_match(",
        "if False and not runtime_shardings_match(",
    ),
    "eager_core:weak_typing_guard_bypassed": (
        "src/_lcm/execution/eager_core.py",
        "or placed.weak_type != template.weak_type",
        "or False",
    ),
    "eager_core:output_repaired_after_execution": (
        "src/_lcm/execution/eager_core.py",
        "                        return self.function(**placed)",
        (
            "                        return jax.device_put(self.funct"
            "ion(**placed), jax.devices()[0])"
        ),
    ),
    "eager_core:resolved_widths_omitted": (
        "src/_lcm/execution/eager_core.py",
        "function=functools.partial(program.function, **program.static_kwargs),",
        "function=program.function,",
    ),
    "runtime_sharding:mesh_shape_ignored": (
        "src/_lcm/execution/runtime_sharding.py",
        "actual.mesh.devices.shape == expected.mesh.devices.shape",
        "True",
    ),
    "runtime_sharding:axis_names_ignored": (
        "src/_lcm/execution/runtime_sharding.py",
        "and actual.mesh.axis_names == expected.mesh.axis_names",
        "and True",
    ),
}


_FINITE_POLICY_MUTATIONS = {
    "finite_policy:discrete_leaf_omitted": (
        "src/_lcm/simulation/policy_programs.py",
        (
            "n_arrays=5 if read.discrete_action_names else 4,\n"
            "            core=POLICY_PREPARE,"
        ),
        "n_arrays=4,\n            core=POLICY_PREPARE,",
    ),
    "finite_policy:smooth_leaf_count_wrong": (
        "src/_lcm/simulation/policy_programs.py",
        (
            "n_arrays=5 if read.discrete_action_names else 4,\n"
            "            core=POLICY_PREPARE,"
        ),
        "n_arrays=5,\n            core=POLICY_PREPARE,",
    ),
    "finite_policy:artifact_locator_shifted": (
        "src/_lcm/simulation/policy_programs.py",
        'leaf_path=(f"FlattenedIndexKey:{jax.tree_util.FlattenedIndexKey(i)}",),',
        'leaf_path=(f"FlattenedIndexKey:{jax.tree_util.FlattenedIndexKey(i + 1)}",),',
    ),
    "finite_policy:occurrence_core_aliased": (
        "src/_lcm/simulation/policy_programs.py",
        "core_key=core,",
        "core_key=POLICY_PREPARE,",
    ),
    "finite_policy:payload_arrays_made_static": (
        "src/_lcm/simulation/policy_programs.py",
        "return payload.arrays, payload.structure",
        "return (), (payload.arrays, payload.structure)",
    ),
    "finite_policy:payload_reconstruction_reversed": (
        "src/_lcm/simulation/policy_programs.py",
        "jax.tree_util.tree_unflatten(self.structure, self.arrays)",
        "jax.tree_util.tree_unflatten(self.structure, self.arrays[::-1])",
    ),
    "finite_policy:producer_reconstruction_swapped": (
        "src/_lcm/egm/published_policy.py",
        'object.__setattr__(policy, "candidate_inner_action", children[0])',
        'object.__setattr__(policy, "candidate_inner_action", children[1])',
    ),
    "finite_policy:fixed_binding_declaration_bypassed": (
        "src/_lcm/model_processing.py",
        "name: declare_finite_replay_programs(regime)",
        "name: regime",
    ),
    "finite_policy:output_action_order_changed": (
        "src/_lcm/simulation/policy_programs.py",
        "read.inner_action_name,\n                            read.outer_action_name,",
        "read.outer_action_name,\n                            read.inner_action_name,",
    ),
    "finite_policy:subject_bank_not_tiled": (
        "src/_lcm/simulation/policy_programs.py",
        'subject_names=("bank", "canonical_states"),',
        'subject_names=("canonical_states",),',
    ),
    "finite_policy:prepared_candidate_order_reversed": (
        "src/_lcm/simulation/policy_programs.py",
        "return tuple(value[:, 0] for value in bank)",
        "return tuple(value[::-1, 0] for value in bank)",
    ),
    "finite_policy:rank_outer_candidate_swapped": (
        "src/_lcm/simulation/policy_programs.py",
        "candidate_outer=bank[1][:, None],",
        "candidate_outer=bank[0][:, None],",
    ),
    "finite_policy:unrepresented_candidates_ranked": (
        "src/_lcm/simulation/policy_programs.py",
        "represented=bank[3][:, None],",
        "represented=bank[2][:, None],",
    ),
    "finite_policy:attained_value_discarded": (
        "src/_lcm/simulation/policy_programs.py",
        "            values[0],",
        "            jnp.zeros_like(values[0]),",
    ),
    "finite_policy:actual_diagnostic_bypassed": (
        "src/_lcm/simulation/simulate.py",
        'arguments={"live": bank[2], "represented": bank[3]},',
        'arguments={"live": bank[2], "represented": bank[2]},',
    ),
    "finite_policy:actual_rank_dispatch_replaced": (
        "src/_lcm/simulation/simulate.py",
        'family="policy_rank",',
        'family="policy_prepare",',
    ),
    "finite_policy:actual_preparation_payload_filtered": (
        "src/_lcm/simulation/simulate.py",
        "payload = ReplayPayload.from_policy(sim_policy)",
        "payload = ReplayPayload.from_policy(candidate_filter(sim_policy))",
    ),
    "finite_policy:prewarm_cartesian_template_admitted": (
        "src/_lcm/simulation/compile.py",
        "if period in programs.policy_rank:",
        "if False and period in programs.policy_rank:",
    ),
    "finite_policy:rank_occurrence_ownership_omitted": (
        "src/_lcm/simulation/period_inputs.py",
        "            regime.simulation.programs.policy_rank,",
        "",
    ),
    "finite_policy:preparation_axis_not_enumerated": (
        "src/lcm/model.py",
        "            programs.policy_prepare,",
        "",
    ),
}


_SOLVE_READINESS_MUTATIONS = {
    "solve_completion:discharged_records_retained": (
        "src/_lcm/execution/pending_work.py",
        (
            "        self._records[:] = [\n"
            "            record for record in self._records "
            "if not record.devices & devices\n"
            "        ]"
        ),
        "        self._records[:] = list(self._records)",
    ),
    "solve_completion:only_first_output_owned": (
        "src/_lcm/execution/pending_work.py",
        "for leaf in jax.tree.leaves(outputs)",
        "for leaf in jax.tree.leaves(outputs)[:1]",
    ),
    "solve_completion:actual_output_devices_ignored": (
        "src/_lcm/execution/pending_work.py",
        "devices=record.devices | actual_devices",
        "devices=record.devices",
    ),
    "solve_completion:close_retains_owners": (
        "src/_lcm/execution/pending_work.py",
        "        self._records.clear()",
        "        pass  # keep stale records",
    ),
    "solve_completion:cleanup_replaces_original_error": (
        "src/_lcm/execution/pending_work.py",
        (
            "            if active_error is None:\n                raise\n"
            "            active_error.add_note(\n"
            '                "Solve completion cleanup also failed: "'
        ),
        (
            "            if True:\n                raise\n"
            "            active_error.add_note(\n"
            '                "Solve completion cleanup also failed: "'
        ),
    ),
    "solve_completion:concrete_source_devices_ignored": (
        "src/_lcm/execution/pending_work.py",
        "for leaf in jax.tree.leaves(arguments)",
        "for leaf in ()",
    ),
    "solve_completion:transfer_endpoints_ignored": (
        "src/_lcm/execution/pending_work.py",
        "for transfer in transfers\n            for sharding",
        "for transfer in ()\n            for sharding",
    ),
    "solve_completion:selected_compiler_proof_ignored": (
        "src/_lcm/execution/pending_work.py",
        "if path in kept_paths and isinstance(leaf, jax.Array)",
        "if isinstance(leaf, jax.Array)",
    ),
    "solve_completion:donating_copy_wait_omitted": (
        "src/_lcm/execution/pending_work.py",
        "if donates or id(array) not in kept_arrays",
        "if id(array) not in kept_arrays",
    ),
    "solve_completion:dead_copy_wait_omitted": (
        "src/_lcm/execution/pending_work.py",
        "if donates or id(array) not in kept_arrays",
        "if donates",
    ),
    "solve_completion:partial_copy_cleanup_omitted": (
        "src/_lcm/execution/pending_work.py",
        "        copies.close(owner=owner, devices=devices)",
        "        pass  # drop returned transfer witnesses",
    ),
    "solve_completion:deleted_wrapper_treated_ready": (
        "src/_lcm/execution/pending_work.py",
        "        raise RuntimeError(msg)",
        "        return",
    ),
    "solve_completion:transfer_observation_omitted": (
        "src/_lcm/execution/value_transfer.py",
        "        on_materialized(transfer=transfer, array=copied)",
        "        pass  # drop observed copy",
    ),
    "solve_completion:transfer_cache_release_callback_omitted": (
        "src/_lcm/execution/scheduler.py",
        "            before_delete=self._before_delete,",
        "            before_delete=None,",
    ),
    "solve_completion:release_predelete_wait_omitted": (
        "src/_lcm/execution/scheduler.py",
        (
            "        before_delete(arrays=tuple("
            "candidate.array for candidate in to_delete))"
        ),
        "        pass  # invalidate pending witnesses",
    ),
    "solve_completion:donation_predelete_wait_omitted": (
        "src/_lcm/solution/backward_induction.py",
        "                before_delete(arrays=(donated.array,))",
        "                pass  # invalidate pending witnesses",
    ),
    "solve_completion:transient_core_owner_omitted": (
        "src/_lcm/solution/backward_induction.py",
        "core, transfer_cache=cache, pending_work=pending_work",
        "core, transfer_cache=cache, pending_work=None",
    ),
    "solve_completion:solve_finally_cleanup_omitted": (
        "src/_lcm/solution/backward_induction.py",
        "            pending_work.close()",
        "            pass  # abandon pending outputs",
    ),
    "solve_completion:unbudgeted_owner_installed": (
        "src/_lcm/solution/backward_induction.py",
        "if resolved_execution.device_memory_bytes is not None\n        else None",
        "if True\n        else None",
    ),
    "solve_completion:planned_core_owner_bypassed": (
        "src/_lcm/execution/output_layout.py",
        "        if self.pending_work is not None:",
        "        if False and self.pending_work is not None:",
    ),
}


@pytest.fixture(scope="module")
def program_mutations() -> dict[str, dict[str, str]]:
    """Build registry controls plus explicit random and profiling defects."""
    root = Path(__file__).parents[1]
    mutations = direct_flow_mutation_specs(repo_root=root)
    mutations.update(
        direct_flow.supplemental_direct_flow_mutation_specs(repo_root=root)
    )
    relative = "src/_lcm/simulation/random.py"
    source = (root / relative).read_text(encoding="utf-8")
    for name, (old, new) in _RANDOM_HELPER_MUTATIONS.items():
        assert source.count(old) == 1, name
        mutations[name] = {"path": relative, "source": source.replace(old, new)}
    for name, (relative, old, new) in (
        _PROFILED_HELPER_MUTATIONS
        | _FINITE_POLICY_MUTATIONS
        | _COMBINED_INPUT_MUTATIONS
        | _EAGER_PLACEMENT_MUTATIONS
        | _FINITE_BUDGET_MUTATIONS
        | _SOLVE_READINESS_MUTATIONS
    ).items():
        source = (root / relative).read_text(encoding="utf-8")
        assert source.count(old) == 1, name
        mutations[name] = {"path": relative, "source": source.replace(old, new)}
    return mutations


@pytest.mark.parametrize(
    "source",
    ["src/_lcm/solution/continuation_arguments.py", "src/_lcm/solution/nbegm.py"],
)
def test_donation_argument_adapter_is_in_the_reviewed_source_inventory(source: str):
    """The sole-marginal adapter and its installation are independently sealed."""
    result = verify_direct_candidate_flow(repo_root=Path(__file__).parents[1])

    assert source in result["certified_corridor_sources"]


def test_supplemental_sources_complete_the_pinned_registry_coverage():
    """Every added corridor has its own separate, nonoverlapping control."""
    root = Path(__file__).parents[1]
    registered = direct_flow_mutation_specs(repo_root=root)
    supplemental = direct_flow.supplemental_direct_flow_mutation_specs(repo_root=root)

    assert set(supplemental) == {
        "solve_completion:conflict_wait_bypassed",
        "eager_core:planned_operand_placement_bypassed",
        "runtime_sharding:physical_partition_ignored",
        "policy_diagnostics:represented_mask_ignored",
        "solve_descriptors:required_layout_replaced",
        "chunk_assembly:cpu_budget_exclusion_broadened",
        "chunk_admission:setup_fulfilled_by_unrelated_inputs",
        "chunk_offload:source_scratch_omitted",
        "chunk_operations:population_window_shifted",
        "chunk_planning:published_output_reservation_omitted",
        "chunk_inventory:compiler_output_owners_omitted",
        "chunk_profiles:action_decoder_profile_omitted",
        "chunk_diagnostics:ownership_mask_ignored",
        "forward_profiles:current_carrier_ignored",
        "population_operations:entry_period_shifted",
        "program_arguments:continuous_actions_omitted",
        "foreign_copy:source_layout_ignored",
        "foreign_snapshot:lazy_materializer_admitted",
        "diagnostic_error:partial_solution_owner_omitted",
        "diagnostic_logging:profiled_callback_bypassed",
        "foreign_authority:copy_admission_bypassed",
        "foreign_entry:canonical_copy_dependency_omitted",
        "foreign_store:copy_constructor_dependency_omitted",
        "simulation_finite_policy:consumer_locator_shifted",
        "simulation_finite_policy:producer_leaf_order_changed",
        "donation:unsupported_scope_admitted",
        "donation:replay_nomination_admitted",
        "donation:residual_duplicates_marginal_operand",
        "donation:ordinary_fallback_filtered",
        "donation:physical_alias_protection_bypassed",
        "donation:paired_residency_omitted",
        "compiler_inputs:eliminated_input_counted_by_compiler",
        "simulation_membership:entry_period_changed",
        "simulation_taste_stream:global_row_high_word_ignored",
        "simulation_entry:upload_budget_omitted",
    }
    assert not set(registered) & set(supplemental)
    assert {spec["path"] for spec in (registered | supplemental).values()} == set(
        direct_flow._CERTIFIED_CORRIDOR_SOURCES
    )


@pytest.mark.parametrize(
    "source",
    [
        "src/_lcm/simulation/programs.py",
        "src/_lcm/simulation/program_types.py",
        "src/_lcm/simulation/runtime.py",
        "src/_lcm/execution/compiler_inputs.py",
        "src/_lcm/execution/footprint.py",
        "src/_lcm/simulation/membership.py",
        "src/_lcm/simulation/taste_stream.py",
        "src/_lcm/simulation/entry_allocations.py",
        "src/_lcm/pandas_utils.py",
        "src/_lcm/simulation/policy_programs.py",
        "src/_lcm/egm/published_policy.py",
    ],
)
def test_live_simulation_program_sources_are_certified(source: str):
    """Declaration, argument binding, resolution, and dispatch are live obligations."""
    result = verify_direct_candidate_flow(repo_root=Path(__file__).parents[1])

    assert source in result["certified_corridor_sources"]


@pytest.mark.parametrize(
    "mutation",
    [f"simulation_program:{name}" for name in _PROGRAM_MUTATIONS]
    + [
        "caller_simulate:action_names_slice",
        "caller_simulate:wrong_discrete_axis_count",
        "caller_simulate:taste_flag_disabled",
        "caller_simulate:live_taste_flag_rebinding",
        "caller_simulate:published_empty_mapping",
        "caller_simulate:attribute_simulation_phase",
        "aot_compile:argmax_index_shift",
        "aot_model:compiled_regime_filter",
        "simulation_index_consumer:next_candidate",
    ]
    + list(direct_flow._SIMULATION_ADAPTER_MUTATIONS)
    + list(_RANDOM_HELPER_MUTATIONS)
    + list(_PROFILED_HELPER_MUTATIONS)
    + list(_FINITE_POLICY_MUTATIONS)
    + list(_COMBINED_INPUT_MUTATIONS)
    + list(_EAGER_PLACEMENT_MUTATIONS)
    + list(_FINITE_BUDGET_MUTATIONS)
    + list(_SOLVE_READINESS_MUTATIONS)
    + list(direct_flow._SUPPLEMENTAL_SOURCE_MUTATIONS),
)
def test_program_mutation_is_rejected_after_byte_seals_are_refreshed(
    *,
    mutation: str,
    program_mutations: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each transport defect fails its own semantic corridor after byte resealing."""
    root = Path(__file__).parents[1]
    sources = verify_direct_candidate_flow(repo_root=root)["certified_corridor_sources"]
    for relative in sources:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / relative, destination)
    spec = program_mutations[mutation]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {relative: sha256_file(tmp_path / relative) for relative in sources},
    )

    result = verify_direct_candidate_flow(repo_root=tmp_path)

    assert result["offending_paths"] == [spec["path"]], result["errors"]
