"""Require every additional helper source control in the canonical aggregate."""

from pathlib import Path

import pytest

from tests.candidate_certificate import direct_flow

_ROOT = Path(__file__).parents[1]
_EXPECTED_NAMES = frozenset(
    (
        "additional_helper:body:src/_lcm/egm/continuation.py::_build_feature_child_query",
        "additional_helper:body:src/_lcm/egm/continuation.py::_select_feature_child_rows",
        (
            "additional_helper:body:src/_lcm/egm/finite_negm_query.py::"
            "_finite_outer_publication_inputs"
        ),
        "additional_helper:body:src/_lcm/egm/nbegm_query.py::_read_scalar_nbegm_owned_row",
        (
            "additional_helper:body:src/_lcm/egm/nbegm_query.py::"
            "_read_scalar_nbegm_owned_row_with_support"
        ),
        "additional_helper:body:src/_lcm/egm/nbegm_query.py::_validate_nbegm_query_recipe",
        (
            "additional_helper:body:src/_lcm/egm/nbegm_query.py::"
            "_with_scalar_nbegm_linear_value"
        ),
        "additional_helper:body:src/_lcm/egm/nbegm_step.py::_prepare_nbegm_interval_block",
        "additional_helper:body:src/_lcm/egm/outer_refinement.py::_fold_exact_outer_nodes",
        "additional_helper:body:src/_lcm/egm/step_core.py::_assemble_combo_candidates",
        "additional_helper:body:src/_lcm/egm/step_core.py::_mask_dead_combo_rows",
        "additional_helper:body:src/_lcm/execution/internal_outputs.py::_leaf_signature",
        (
            "additional_helper:body:src/_lcm/execution/internal_outputs.py::"
            "_published_signature"
        ),
        "additional_helper:body:src/_lcm/execution/internal_outputs.py::_tree_signature",
        (
            "additional_helper:body:src/_lcm/execution/internal_outputs.py::"
            "assert_internal_inputs"
        ),
        (
            "additional_helper:body:src/_lcm/model_processing.py::"
            "_validate_callable_borrowing_solver"
        ),
        (
            "additional_helper:body:src/_lcm/regime_building/processing.py::"
            "_nbegm_evaluated_query_predicates"
        ),
        (
            "additional_helper:body:src/_lcm/regime_building/processing.py::"
            "_nbegm_query_evaluator_targets"
        ),
        (
            "additional_helper:body:src/_lcm/regime_building/processing.py::"
            "_nbegm_query_function_pools"
        ),
        (
            "additional_helper:body:src/_lcm/regime_building/processing.py::"
            "_nbegm_query_row_state_names"
        ),
        (
            "additional_helper:body:src/_lcm/regime_building/processing.py::"
            "_solver_with_query_authority"
        ),
        (
            "additional_helper:body:src/_lcm/solution/dcegm.py::"
            "_dcegm_query_slots_and_capabilities"
        ),
        "additional_helper:body:src/_lcm/solution/egm.py::_unpack_scalar_child_read",
        (
            "additional_helper:body:src/_lcm/solution/model_authority.py::"
            "_validate_query_policy_budget_authority"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_NBEGMActionEligibility.__call__"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_NBEGMScalarChildReads.__call__"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_NBEGMScalarChildReads.read_rows"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_NBEGMTabulatedEulerLaw.__call__"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_RideAlongNBEGMPeriodKernel._consume_output"
        ),
        "additional_helper:body:src/_lcm/solution/nbegm.py::_bounded_nbegm_kernel_output",
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_fold_nbegm_scalar_read_records"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_nbegm_cell_continuation_columns"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_prepare_nbegm_scalar_child_reads"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_record_nbegm_scalar_child_slots"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_scalar_conditional_nbegm_carry"
        ),
        (
            "additional_helper:body:src/_lcm/solution/nbegm.py::"
            "_validate_nbegm_integrity_slot_counts"
        ),
        "additional_helper:body:src/_lcm/solution/nbegm.py::_with_scalar_nbegm_sampled_run",
        "additional_helper:body:src/_lcm/solution/nnbegm.py::_nnbegm_outer_replay_function",
        "additional_helper:body:src/_lcm/solution/nnbegm.py::_nnbegm_query_period_groups",
        (
            "additional_helper:body:src/_lcm/solution/nnbegm.py::"
            "_prepare_nnbegm_query_inner_pair"
        ),
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
_PROBE = (
    "additional_helper:body:src/_lcm/execution/internal_outputs.py::"
    "assert_internal_inputs"
)


def _required_specs() -> dict[str, dict[str, str]]:
    """Require the distinct source-control family before running the aggregate."""
    assert callable(getattr(direct_flow, "additional_helper_mutation_specs", None)), (
        "The canonical aggregate has no additional helper mutation family"
    )
    specs = direct_flow.additional_helper_mutation_specs(repo_root=_ROOT)
    assert set(specs) == _EXPECTED_NAMES
    return specs


def _assert_historical_counts(*, report: direct_flow.DirectMutationControls) -> None:
    """Keep every independently declared family population intact."""
    assert report["mutation_count"] == 421
    assert report["supplemental_mutation_count"] == 96
    assert report["retained_prefix_mutation_count"] == 113
    assert report["outer_helper_mutation_count"] == 40
    assert report["final_nested_mutation_count"] == 104


def test_additional_helper_source_controls_execute_in_the_actual_aggregate() -> None:
    """All forty bodies, three surfaces and five source attacks execute once."""
    _required_specs()
    report = direct_flow.run_direct_flow_mutation_controls(repo_root=_ROOT)
    _assert_historical_counts(report=report)
    assert report["clean"]["ok"]
    assert report["clean"]["errors"] == []
    assert set(report["additional_helper_mutations"]) == _EXPECTED_NAMES
    assert report["additional_helper_mutation_count"] == 48
    assert report["additional_helper_names_match_expected"]
    assert report["admitted_additional_helper_mutations"] == []
    assert report["all_rejected"]
    for name, result in report["additional_helper_mutations"].items():
        assert result["rejected"], name
        diagnostic = direct_flow._additional_helper_primary_diagnostic(name=name)
        assert tuple(
            sorted(result["errors"])
        ) == direct_flow._additional_helper_expected_errors(
            source=result["path"], diagnostic=diagnostic
        ), name
        assert result["offending_paths"] == [result["path"]], name
        assert not any("source seal mismatch" in error for error in result["errors"]), (
            name
        )


@pytest.mark.parametrize("fault", ["admitted", "omitted"])
def test_an_admitted_or_missing_helper_control_prevents_aggregate_acceptance(
    *, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The genuine aggregate refuses detector or population omissions."""
    specs = _required_specs()
    assert _PROBE in specs
    if fault == "admitted":
        original = direct_flow._additional_helper_case_rejected
        probe_diagnostic = direct_flow._additional_helper_primary_diagnostic(
            name=_PROBE
        )

        def admit_probe(
            *,
            clean: direct_flow.DirectFlowReport,
            result: direct_flow.DirectFlowReport,
            source: direct_flow.SourcePath,
            diagnostic: str,
        ) -> bool:
            return diagnostic != probe_diagnostic and original(
                clean=clean, result=result, source=source, diagnostic=diagnostic
            )

        monkeypatch.setattr(
            direct_flow, "_additional_helper_case_rejected", admit_probe
        )
    else:
        original_specs = direct_flow.additional_helper_mutation_specs

        def omit_probe(*, repo_root: Path) -> dict[str, dict[str, str]]:
            return {
                name: spec
                for name, spec in original_specs(repo_root=repo_root).items()
                if name != _PROBE
            }

        monkeypatch.setattr(direct_flow, "additional_helper_mutation_specs", omit_probe)
    report = direct_flow.run_direct_flow_mutation_controls(repo_root=_ROOT)
    _assert_historical_counts(report=report)
    assert report["clean"]["ok"]
    assert report["clean"]["errors"] == []
    assert not report["all_rejected"]
    if fault == "admitted":
        assert _PROBE in report["admitted_additional_helper_mutations"]
        assert report["additional_helper_names_match_expected"]
        assert report["additional_helper_mutation_count"] == 48
    else:
        assert _PROBE not in report["additional_helper_mutations"]
        assert not report["additional_helper_names_match_expected"]
        assert report["additional_helper_mutation_count"] == 53
