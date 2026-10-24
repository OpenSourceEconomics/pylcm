"""Require the exact native-binding composition source controls in the aggregate."""

from pathlib import Path

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.runner_replay import DirectFlowControlsReplay

_ROOT = Path(__file__).parents[1]
_EXPECTED_NAMES = frozenset(
    (
        "child_native:body:src/_lcm/egm/upper_envelope/__init__.py::_exact_backend",
        (
            "child_native:body:src/_lcm/egm/upper_envelope/segment_envelope.py::"
            "refine_envelope_exact"
        ),
        "child_native:overload:changed_stub",
        "child_native:overload:overload_import_removed",
        "child_native:overload:extra_function_binding",
        "child_native:overload:extra_stub",
        "child_native:overload:missing_stub",
        "child_native:overload:overload_alias",
        "child_native:overload:overload_rebinding",
        "child_native:overload:reordered_stubs",
        "child_native:overload:runtime_decorator",
        "child_native:surface:src/_lcm/egm/upper_envelope/__init__.py",
        "child_native:surface:src/_lcm/egm/upper_envelope/segment_envelope.py",
    )
)
_PROBE = (
    "child_native:body:src/_lcm/egm/upper_envelope/segment_envelope.py::"
    "refine_envelope_exact"
)


def _required_specs() -> dict[str, dict[str, str]]:
    """Require the distinct source-control family before running the aggregate."""
    assert callable(getattr(direct_flow, "child_native_mutation_specs", None)), (
        "The canonical aggregate has no native composition mutation family"
    )
    specs = direct_flow.child_native_mutation_specs(repo_root=_ROOT)
    assert set(specs) == _EXPECTED_NAMES
    return specs


def _assert_historical_counts(*, report: direct_flow.DirectMutationControls) -> None:
    """Keep every independently declared family population intact."""
    assert report["mutation_count"] == 421
    assert report["supplemental_mutation_count"] == 97
    assert report["retained_prefix_mutation_count"] == 113
    assert report["outer_helper_mutation_count"] == 40
    assert report["final_nested_mutation_count"] == 104
    assert report["additional_helper_mutation_count"] == 48


@pytest.mark.xdist_group("direct_flow_runner")
@pytest.mark.timeout(3000)
def test_child_native_source_controls_execute_in_the_actual_aggregate(
    *, direct_flow_controls: DirectFlowControlsReplay
) -> None:
    """Both bodies, both surfaces and all nine declarations execute once."""
    _required_specs()
    report = direct_flow_controls.run(repo_root=_ROOT)
    _assert_historical_counts(report=report)
    assert report["clean"]["ok"]
    assert report["clean"]["errors"] == []
    assert set(report["child_native_mutations"]) == _EXPECTED_NAMES
    assert report["child_native_mutation_count"] == 13
    assert report["child_native_names_match_expected"]
    assert report["admitted_child_native_mutations"] == []
    assert report["all_rejected"]
    for name, result in report["child_native_mutations"].items():
        assert result["rejected"], name
        diagnostic = direct_flow._child_native_primary_diagnostic(name=name)
        assert tuple(
            sorted(result["errors"])
        ) == direct_flow._child_native_expected_errors(
            source=result["path"], diagnostic=diagnostic
        ), name
        assert result["offending_paths"] == [result["path"]], name
        assert not any("source seal mismatch" in error for error in result["errors"]), (
            name
        )


@pytest.mark.xdist_group("direct_flow_runner")
@pytest.mark.timeout(3000)
@pytest.mark.parametrize("fault", ["admitted", "omitted"])
def test_an_admitted_or_missing_helper_control_prevents_aggregate_acceptance(
    *,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
    direct_flow_controls: DirectFlowControlsReplay,
) -> None:
    """The genuine aggregate refuses detector or population omissions."""
    specs = _required_specs()
    assert _PROBE in specs
    if fault == "admitted":
        original = direct_flow._child_native_case_rejected
        probe_diagnostic = direct_flow._child_native_primary_diagnostic(name=_PROBE)

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

        monkeypatch.setattr(direct_flow, "_child_native_case_rejected", admit_probe)
    else:
        original_specs = direct_flow.child_native_mutation_specs

        def omit_probe(*, repo_root: Path) -> dict[str, dict[str, str]]:
            return {
                name: spec
                for name, spec in original_specs(repo_root=repo_root).items()
                if name != _PROBE
            }

        monkeypatch.setattr(direct_flow, "child_native_mutation_specs", omit_probe)
    report = direct_flow_controls.run(repo_root=_ROOT)
    _assert_historical_counts(report=report)
    assert report["clean"]["ok"]
    assert report["clean"]["errors"] == []
    assert not report["all_rejected"]
    if fault == "admitted":
        assert _PROBE in report["admitted_child_native_mutations"]
        assert report["child_native_names_match_expected"]
        assert report["child_native_mutation_count"] == 13
    else:
        assert _PROBE not in report["child_native_mutations"]
        assert not report["child_native_names_match_expected"]
        assert report["child_native_mutation_count"] == 12
