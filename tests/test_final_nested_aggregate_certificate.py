"""Require every nested helper control in the actual certificate aggregate."""

from pathlib import Path

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.runner_replay import DirectFlowControlsReplay

_ROOT = Path(__file__).parents[1]
_EXPECTED_NAMES = frozenset(
    (
        "final_nested:body:src/_lcm/egm/nested_floor.py::prepare_nested_floor",
        "final_nested:body:src/_lcm/egm/nested_floor.py::nested_floor_at_cell",
        "final_nested:body:src/_lcm/egm/finite_negm_query.py::read_finite_outer_choices",
        "final_nested:body:src/_lcm/egm/finite_negm_query.py::_single_row_finite_corner_read",
        "final_nested:body:src/_lcm/egm/finite_negm_query.py::_read_finite_resource_gradient",
        "final_nested:body:src/_lcm/egm/finite_negm_query.py::_read_finite_corner_row",
        "final_nested:body:src/_lcm/egm/finite_negm_query.py::_read_finite_native_row",
        "final_nested:body:src/_lcm/egm/finite_negm_query.py::_read_finite_owned_side",
        "final_nested:body:src/_lcm/egm/finite_negm_query.py::_validate_finite_outer_rows",
        "final_nested:body:src/_lcm/egm/finite_negm_query.py::_select_finite_outer_profiles",
        "final_nested:body:src/_lcm/egm/finite_negm_query.py::_collapse_finite_selection",
        "final_nested:body:src/_lcm/egm/adaptive_nbegm_producer.py::_tuple_roles",
        "final_nested:body:src/_lcm/egm/adaptive_nbegm_producer.py::_floor_template",
        "final_nested:body:src/_lcm/egm/adaptive_nbegm_producer.py::PreparedFloorArguments.__call__",
        "final_nested:body:src/_lcm/egm/adaptive_nbegm_producer.py::with_keeper_floor",
        "final_nested:body:src/_lcm/egm/adaptive_nbegm_producer.py::prepare_nested_inner_kernels",
        "final_nested:body:src/_lcm/egm/adaptive_nbegm_producer.py::fold_executed_summaries",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::build_nbegm_query",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery.capabilities",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery.input_names",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery.evaluate_floor",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery.read_prepared",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery.read_prepared_with_counts",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery._read_at_floor_with_summary",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery._read_rows_without_counts",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery._read_rows",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery._read_rows_with_counts",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::_read_nb_compact_corner_rows",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::_read_nb_compact_corner",
        "final_nested:body:src/_lcm/egm/nbegm_query.py::NBEGMQuery.read_choices",
        "final_nested:body:src/_lcm/egm/carry.py::_get_owned_local_piece",
        "final_nested:body:src/_lcm/egm/nbegm_step.py::_single_row_nbegm_query_step",
        "final_nested:body:src/_lcm/solution/negm.py::NEGM.build_period_kernels",
        "final_nested:body:src/_lcm/solution/negm.py::_derive_negm_inner_query_model",
        "final_nested:body:src/_lcm/solution/negm.py::_build_negm_floor_preparation",
        "final_nested:body:src/_lcm/solution/negm.py::_durable_values_at",
        "final_nested:body:src/_lcm/solution/negm.py::_NEGMPeriodKernel.__post_init__",
        "final_nested:body:src/_lcm/solution/negm.py::_NEGMPeriodKernel.__call__",
        "final_nested:body:src/_lcm/solution/negm.py::_nested_feasibility_programs",
        "final_nested:body:src/_lcm/solution/negm.py::_nested_tuple_roles",
        "final_nested:body:src/_lcm/solution/negm.py::_stacked_nested_carry_roles",
        "final_nested:body:src/_lcm/solution/negm.py::_stacked_nested_carry_role",
        "final_nested:body:src/_lcm/solution/negm.py::_keeper_with_prepared_floor",
        "final_nested:body:src/_lcm/solution/negm.py::_run_nested_feasibility_kernel",
        "final_nested:body:src/_lcm/solution/negm.py::_nested_outer_sweep_program",
        "final_nested:body:src/_lcm/solution/negm.py::_count_nested_current_feasibility",
        "final_nested:body:src/_lcm/solution/negm.py::_NestedNodeSolver.__call__",
        "final_nested:body:src/_lcm/solution/negm.py::_NEGMSweepArgumentBuilder.__call__",
        "final_nested:body:src/_lcm/solution/negm.py::_periodized_function_groups",
        "final_nested:body:src/_lcm/solution/negm.py::_with_declared_keeper_reads",
        "final_nested:body:src/_lcm/solution/nnbegm.py::NNBEGM.build_period_kernels",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_conditional_nnbegm_banks",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_NestedArgumentBuilder.__call__",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_NNBEGMPeriodKernel.__post_init__",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_NNBEGMPeriodKernel.__call__",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_NNBEGMPeriodKernel._solve_keeper",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_NNBEGMPeriodKernel._solve_adjuster_node",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_FiniteNNBEGMPeriodKernel._solve_outer",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_AdaptiveNNBEGMPeriodKernel._solve_outer",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_role_carry_reads",
        "final_nested:body:src/_lcm/solution/nnbegm.py::_AdaptiveNodeSolver.__call__",
        "final_nested:body:src/_lcm/solution/dcegm.py::_counted_dcegm_output_roles",
        "final_nested:body:src/_lcm/solution/dcegm.py::_with_nested_feasibility_role",
        "final_nested:body:src/_lcm/solution/dcegm.py::_DCEGMPeriodKernel.retains_current_feasibility",
        "final_nested:body:src/_lcm/solution/dcegm.py::_strip_nested_feasibility",
        "final_nested:body:src/_lcm/solution/nbegm.py::_ride_along_output_roles",
        "final_nested:body:src/_lcm/solution/nbegm.py::_nbegm_bounded_output_roles",
        "final_nested:body:src/_lcm/solution/nbegm.py::_with_nbegm_scalar_output_roles",
        "final_nested:body:src/_lcm/solution/nbegm.py::_with_nbegm_scalar_carry_reads",
        "final_nested:body:src/_lcm/solution/nbegm.py::_NBEGMOwnedSingleAxisCore.__call__",
        "final_nested:body:src/_lcm/solution/feasibility.py::reduce_continuation_read_counts",
        "final_nested:body:src/_lcm/solution/feasibility.py::merge_continuation_read_counts",
        "final_nested:body:src/_lcm/regime_building/processing.py::_prepare_nbegm_query_models",
        "final_nested:body:src/_lcm/regime_building/processing.py::_canonical_constraint_routing_inputs",
        "final_nested:body:src/_lcm/egm/outer_candidates.py::build_outer_candidate_bank",
        "final_nested:body:src/_lcm/egm/outer_candidates.py::_stack_leaves",
        "final_nested:body:src/_lcm/egm/outer_candidates.py::OuterCandidateBank.n_candidates",
        "final_nested:body:src/_lcm/egm/outer_candidates.py::OuterCandidateBank.candidate_v_arr",
        "final_nested:body:src/_lcm/egm/outer_candidates.py::OuterCandidateBank.candidate_carry",
        "final_nested:surface:src/_lcm/egm/nested_floor.py",
        "final_nested:surface:src/_lcm/egm/finite_negm_query.py",
        "final_nested:surface:src/_lcm/egm/adaptive_nbegm_producer.py",
        "final_nested:surface:src/_lcm/egm/nbegm_query_model.py",
        "final_nested:surface:src/_lcm/egm/nbegm_query.py",
        "final_nested:surface:src/_lcm/egm/carry.py",
        "final_nested:surface:src/_lcm/egm/nbegm_step.py",
        "final_nested:surface:src/_lcm/solution/negm.py",
        "final_nested:surface:src/_lcm/solution/nnbegm.py",
        "final_nested:surface:src/_lcm/solution/dcegm.py",
        "final_nested:surface:src/_lcm/solution/nbegm.py",
        "final_nested:surface:src/_lcm/solution/feasibility.py",
        "final_nested:surface:src/_lcm/egm/query_data.py",
        "final_nested:surface:src/_lcm/regime_building/processing.py",
        "final_nested:surface:src/_lcm/egm/outer_candidates.py",
        "final_nested:overload:missing_stub",
        "final_nested:overload:extra_stub",
        "final_nested:overload:reordered_stubs",
        "final_nested:overload:changed_stub",
        "final_nested:overload:runtime_decorator",
        "final_nested:overload:extra_method_binding",
        "final_nested:overload:extra_class_binding",
        "final_nested:overload:overload_alias",
        "final_nested:overload:overload_rebinding",
    )
)
_PROBE = (
    "final_nested:body:src/_lcm/egm/finite_negm_query.py::_collapse_finite_selection"
)
_PROBE_DIAGNOSTIC = (
    "final nested dependency: exact callable corridor "
    "'_collapse_finite_selection' changed"
)


@pytest.mark.xdist_group("direct_flow_runner")
@pytest.mark.timeout(3000)
def test_final_nested_runner_keeps_a_distinct_required_population(
    *, direct_flow_controls: DirectFlowControlsReplay
) -> None:
    """Every declared nested control executes without changing historical families."""
    assert callable(getattr(direct_flow, "final_nested_mutation_specs", None)), (
        "The canonical aggregate has no final nested mutation family"
    )
    specifications = direct_flow.final_nested_mutation_specs(repo_root=_ROOT)
    assert set(specifications) == _EXPECTED_NAMES
    report = direct_flow_controls.run(repo_root=_ROOT)
    assert report["clean"]["ok"]
    assert report["clean"]["errors"] == []
    assert report["mutation_count"] == 410
    assert report["supplemental_mutation_count"] == 94
    assert report["retained_prefix_mutation_count"] == 116
    assert report["outer_helper_mutation_count"] == 40
    cases = report["final_nested_mutations"]
    assert set(cases) == _EXPECTED_NAMES
    assert report["final_nested_mutation_count"] == 103
    assert report["final_nested_names_match_expected"]
    assert report["admitted_final_nested_mutations"] == []
    assert report["all_rejected"]
    for name, result in cases.items():
        assert result["rejected"], name
        assert result["errors"], name
        assert result["offending_paths"] == [result["path"]], name
        diagnostic = direct_flow._final_nested_primary_diagnostic(name=name)
        assert diagnostic in result["errors"], name
        assert tuple(
            sorted(result["errors"])
        ) == direct_flow._final_nested_expected_errors(
            source=result["path"], diagnostic=diagnostic
        ), name
        assert not any("source seal mismatch" in error for error in result["errors"]), (
            name
        )


@pytest.mark.xdist_group("direct_flow_runner")
@pytest.mark.timeout(3000)
@pytest.mark.parametrize("fault", ["admitted", "omitted"])
def test_final_nested_failure_prevents_the_actual_runner_aggregate(
    *,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
    direct_flow_controls: DirectFlowControlsReplay,
) -> None:
    """An admitted or missing nested case prevents canonical aggregate acceptance."""
    assert callable(getattr(direct_flow, "final_nested_mutation_specs", None)), (
        "The canonical aggregate has no final nested mutation family"
    )
    assert callable(getattr(direct_flow, "_final_nested_case_rejected", None)), (
        "The canonical aggregate has no scoped final nested rejection detector"
    )
    actual = direct_flow.final_nested_mutation_specs(repo_root=_ROOT)
    assert set(actual) == _EXPECTED_NAMES
    assert _PROBE in actual
    if fault == "admitted":
        original = direct_flow._final_nested_case_rejected

        def admit_probe(
            *,
            clean: direct_flow.DirectFlowReport,
            result: direct_flow.DirectFlowReport,
            source: direct_flow.SourcePath,
            diagnostic: str,
        ) -> bool:
            return diagnostic != _PROBE_DIAGNOSTIC and original(
                clean=clean, result=result, source=source, diagnostic=diagnostic
            )

        monkeypatch.setattr(direct_flow, "_final_nested_case_rejected", admit_probe)
    else:
        original_specs = direct_flow.final_nested_mutation_specs

        def omit_probe(*, repo_root: Path) -> dict[str, dict[str, str]]:
            return {
                name: specification
                for name, specification in original_specs(repo_root=repo_root).items()
                if name != _PROBE
            }

        monkeypatch.setattr(direct_flow, "final_nested_mutation_specs", omit_probe)
    report = direct_flow_controls.run(repo_root=_ROOT)
    assert report["clean"]["ok"]
    assert report["clean"]["errors"] == []
    assert report["mutation_count"] == 410
    assert report["supplemental_mutation_count"] == 94
    assert report["retained_prefix_mutation_count"] == 116
    assert report["outer_helper_mutation_count"] == 40
    assert not report["all_rejected"]
    if fault == "admitted":
        assert _PROBE in report["admitted_final_nested_mutations"]
        assert report["final_nested_names_match_expected"]
        assert report["final_nested_mutation_count"] == 103
    else:
        assert _PROBE not in report["final_nested_mutations"]
        assert not report["final_nested_names_match_expected"]
        assert report["final_nested_mutation_count"] == 102
