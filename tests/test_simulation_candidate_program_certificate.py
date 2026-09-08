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
        '"partitionable": jax.config.jax_threefry_partitionable,',
        '"partitionable": False,',
    ),
    "simulation_random:declared_population_mode_ignored": (
        "with jax.threefry_partitionable(partitionable):\n        for name in names:",
        "with jax.threefry_partitionable(False):\n        for name in names:",
    ),
}

_PROFILED_HELPER_MUTATIONS = {
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
        "dropped=bank[2] & ~bank[3],",
        "dropped=jnp.zeros_like(bank[2]),",
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
        _PROFILED_HELPER_MUTATIONS | _FINITE_POLICY_MUTATIONS
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
