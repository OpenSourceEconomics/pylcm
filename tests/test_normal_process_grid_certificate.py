"""Normal support mutations are rejected after independent byte resealing."""

import hashlib
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.generate_sources import sha256_file

# Sources the nested, scalar-query and state-dependent borrowing corridor adds
# to the original roster, in roster order.
_NESTED_CORRIDOR_SOURCES = (
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
    "src/_lcm/egm/outer_interpolation.py",
    "src/_lcm/egm/outer_refinement.py",
    "src/_lcm/optimization/golden_section.py",
    "src/_lcm/egm/upper_envelope/query.py",
    "src/_lcm/egm/asset_row.py",
    "src/_lcm/egm/continuation.py",
    "src/_lcm/egm/dcegm_query.py",
    "src/_lcm/egm/dcegm_query_model.py",
    "src/_lcm/egm/regime_introspection.py",
    "src/_lcm/egm/step.py",
    "src/_lcm/egm/step_core.py",
    "src/_lcm/solution/dcegm.py",
    "src/_lcm/regime_building/decoupled.py",
    "src/_lcm/egm/euler.py",
    "src/_lcm/egm/declared_law.py",
    "src/_lcm/egm/interp.py",
    "src/_lcm/egm/numeric_inverse.py",
    "src/_lcm/egm/one_asset_egm_step.py",
    "src/_lcm/egm/preferences.py",
    "src/_lcm/egm/scalar_query.py",
    "src/_lcm/egm/outer_envelope.py",
    "src/_lcm/solution/landing_counts.py",
    "src/_lcm/continuation.py",
    "src/_lcm/egm/budget.py",
    "src/_lcm/egm/carry.py",
    "src/_lcm/egm/query_data.py",
    "src/_lcm/regime_building/Q_and_F.py",
    "src/_lcm/simulation/subject_parallel.py",
    "src/_lcm/solution/egm.py",
    "src/_lcm/solution/kernel_output.py",
    "src/_lcm/constraints/dynamic_bound.py",
    "src/_lcm/post_decision_bound.py",
    "src/_lcm/constraints/bounds.py",
    "src/_lcm/constraints/processed.py",
    "src/lcm/regime.py",
    "src/lcm/consumption_savings_regime.py",
    "src/_lcm/regime_building/age_normalization.py",
    "src/_lcm/utils/counts.py",
    "src/_lcm/egm/nbegm_step.py",
    "src/_lcm/egm/nbegm_segments.py",
    "src/_lcm/egm/framed_arithmetic.py",
    "src/_lcm/egm/ez_kernel.py",
    "src/_lcm/power_mean.py",
)


# Exact additions from accepted S1, current-IID S4 and typed-override S11 corridors.
_S1_CORRIDOR_SOURCES = ("src/_lcm/simulation/random_address.py",)
_CURRENT_IID_CORRIDOR_SOURCES = (
    "src/_lcm/current_iid.py",
    "src/_lcm/simulation/current_iid.py",
)
_OVERRIDE_CORRIDOR_SOURCES = (
    "src/lcm/shocks.py",
    "src/_lcm/shock_overrides.py",
    "src/_lcm/simulation/current_iid_overrides.py",
)
_ENGINE_REDESIGN_CORRIDOR_SOURCES = (
    "src/_lcm/simulation/eager_stages.py",
    "src/lcm/typing.py",
)


def test_normal_controls_preserve_all_historical_populations() -> None:
    root = Path(__file__).parents[1]
    population = direct_flow.normal_process_mutation_specs(repo_root=root)
    assert len(population) == 17
    assert (
        hashlib.sha256(("\n".join(sorted(population)) + "\n").encode()).hexdigest()
        == direct_flow.EXPECTED_NORMAL_PROCESS_MUTATION_NAMES_SHA256
    )
    historical: set[str] = set()
    for generator in (
        direct_flow.direct_flow_mutation_specs,
        direct_flow.supplemental_direct_flow_mutation_specs,
        direct_flow.uniform_process_mutation_specs,
        direct_flow.action_grid_mutation_specs,
        direct_flow.feasibility_mutation_specs,
        direct_flow.grouped_mapper_mutation_specs,
        direct_flow.grouped_guard_mutation_specs,
        direct_flow.allocation_reservation_mutation_specs,
    ):
        names = set(generator(repo_root=root))
        assert historical.isdisjoint(names)
        assert set(population).isdisjoint(names)
        historical.update(names)
    added_feasibility_controls = {
        "feasibility_summary:published_value_changed",
        "feasibility_summary:empty_count_reversed",
    }
    assert added_feasibility_controls <= historical
    added_initial_controls = {
        "simulation_adapter:initial_physical_admission_bypassed",
        "simulation_adapter:initial_physical_cov_forwarding_omitted",
        "simulation_adapter:initial_physical_cov_operand_changed",
        "simulation_adapter:initial_physical_fixed_law_selection_changed",
        "simulation_adapter:initial_physical_internal_columns_retained",
        "simulation_adapter:initial_physical_mean_forwarding_omitted",
        "simulation_adapter:initial_physical_mean_operand_changed",
        "simulation_adapter:initial_physical_output_hold_omitted",
        "simulation_adapter:initial_physical_prefix_changed",
        "simulation_adapter:initial_physical_subject_order_changed",
        "simulation_adapter:initial_physical_table_column_changed",
        "simulation_adapter:initial_physical_transform_changed",
    }
    assert added_initial_controls <= historical
    # Accepted S1 adds this one destination-address control to the old roster.
    added_rng_controls = {"simulation_taste_stream:destination_name_ignored"}
    assert added_rng_controls <= historical
    # The engine redesign adds this one canonical-writer dtype control.
    added_engine_controls = {"candidate_materialization:array_writer_dtype_widened"}
    assert added_engine_controls <= historical
    assert len(historical) == 615
    historical_before_s1 = historical - added_rng_controls - added_engine_controls
    assert len(historical_before_s1) == 613
    assert len(historical_before_s1 - added_initial_controls) == 601
    assert (
        len(historical_before_s1 - added_initial_controls - added_feasibility_controls)
        == 599
    )
    assert "src/_lcm/solution/feasibility.py" in direct_flow._CERTIFIED_CORRIDOR_SOURCES
    assert len(direct_flow._CERTIFIED_CORRIDOR_SOURCES) == 179
    added_corridor_sources = (
        set(_S1_CORRIDOR_SOURCES)
        | set(_CURRENT_IID_CORRIDOR_SOURCES)
        | set(_OVERRIDE_CORRIDOR_SOURCES)
        | set(_ENGINE_REDESIGN_CORRIDOR_SOURCES)
    )
    assert len(added_corridor_sources) == 8
    assert added_corridor_sources <= set(direct_flow._CERTIFIED_CORRIDOR_SOURCES)
    assert added_corridor_sources.isdisjoint(_NESTED_CORRIDOR_SOURCES)
    assert len(set(_NESTED_CORRIDOR_SOURCES)) == 53
    assert set(_NESTED_CORRIDOR_SOURCES) <= set(direct_flow._CERTIFIED_CORRIDOR_SOURCES)
    original_corridor_sources = tuple(
        relative
        for relative in direct_flow._CERTIFIED_CORRIDOR_SOURCES
        if relative != "src/_lcm/solution/feasibility.py"
        and relative not in _NESTED_CORRIDOR_SOURCES
        and relative not in added_corridor_sources
    )
    assert len(original_corridor_sources) == 117
    # Exact ordered source roster from 0a39539fd628bb60dcae2d3cca601816fe4832f4.
    assert (
        hashlib.sha256(
            ("\n".join(original_corridor_sources) + "\n").encode()
        ).hexdigest()
        == "176867afe4246804efd4231a7a7a8439c6852d2bd099f6f537d3597473d70c85"
    )


@pytest.mark.parametrize("mutation", tuple(direct_flow._NORMAL_PROCESS_MUTATIONS))
def test_normal_mutation_is_rejected_after_independent_byte_reseal(
    *, mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1]
    clean = direct_flow.verify_direct_candidate_flow(repo_root=root)
    assert clean["ok"], clean["errors"]
    sources = clean["certified_corridor_sources"]
    for relative in sources:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / relative, destination)
    spec = direct_flow.normal_process_mutation_specs(repo_root=root)[mutation]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {relative: sha256_file(tmp_path / relative) for relative in sources},
    )
    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert not result["ok"]
    assert result["offending_paths"] == [spec["path"]], result["errors"]
