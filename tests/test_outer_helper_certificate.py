"""Outer helper bodies and bindings are explicit certificate dependencies."""

import ast
import hashlib
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.outer_helper_controls import (
    BODY_OWNERS,
    CONTROLS,
    OuterHelperControl,
    mutate_outer_helper,
    outer_helper_mutation_specs,
)

_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("mutation", tuple(CONTROLS))
def test_outer_helper_control_has_one_reversible_owner_delta(*, mutation: str) -> None:
    """Every named control constructs a live, valid mutation with an exact inverse."""
    control = CONTROLS[mutation]
    path = _ROOT / control.source
    original = path.read_bytes()
    mutated = mutate_outer_helper(source=original.decode("utf-8"), control=control)
    assert ast.dump(ast.parse(mutated)) != ast.dump(ast.parse(original))
    assert path.read_bytes() == original


def test_outer_helper_family_selects_exactly_the_declared_bodies_and_surfaces() -> None:
    """Every explicit helper is selected from the single canonical pin store."""
    assert hasattr(direct_flow, "_OUTER_HELPER_CONTRACTS"), (
        "The canonical checker has no outer-helper dependency family"
    )
    contracts = direct_flow._OUTER_HELPER_CONTRACTS
    assert {
        source: tuple(callables) for source, (_surface, callables) in contracts.items()
    } == BODY_OWNERS
    for source, (surface, callables) in contracts.items():
        # A source can also carry pins that another helper family selects, so
        # compare only this family's surface and callables against the store.
        stored_surface, stored_callables = direct_flow._CORRIDOR_PINS[source]
        assert stored_surface == surface
        assert {name: stored_callables[name] for name in callables} == callables
        assert source in direct_flow._CERTIFIED_CORRIDOR_SOURCES
    assert direct_flow._unselected_pins() == ()


def test_outer_helper_control_registry_keeps_every_explicit_name_and_association() -> (
    None
):
    """The executable family contains every reviewed body, callback and surface case."""
    assert hasattr(direct_flow, "outer_helper_mutation_specs"), (
        "The canonical checker does not execute the outer-helper controls"
    )
    actual = direct_flow.outer_helper_mutation_specs(repo_root=_ROOT)
    expected = outer_helper_mutation_specs(repo_root=_ROOT)
    assert actual == expected
    assert len(actual) == 40
    assert set(actual) == direct_flow._EXPECTED_OUTER_HELPER_MUTATION_NAMES
    assert (
        len(
            {
                (source, owner)
                for source, owners in BODY_OWNERS.items()
                for owner in owners
            }
        )
        == 20
    )
    assert {name for name in actual if ":surface:" in name} == {
        "outer_helpers:surface:golden_import_redirected",
        "outer_helpers:surface:prefix_minimum_changed",
        "outer_helpers:surface:golden_result_reordered",
    }


@pytest.mark.parametrize("mutation", tuple(CONTROLS))
def test_outer_helper_mutation_requires_its_non_seal_owner_error(
    *, mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean composed checker rejects each corruption at its actual source owner."""
    control = CONTROLS[mutation]
    baseline = direct_flow.verify_direct_candidate_flow(repo_root=_ROOT)
    assert baseline["ok"]
    assert baseline["errors"] == []
    assert baseline["offending_paths"] == []
    assert control.source in baseline["certified_corridor_sources"]
    originals = {}
    for source in baseline["certified_corridor_sources"]:
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(_ROOT / source, destination)
        originals[source] = destination.read_bytes()
    positive = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert positive == baseline
    target = tmp_path / control.source
    target.write_text(
        mutate_outer_helper(
            source=originals[control.source].decode("utf-8"), control=control
        ),
        encoding="utf-8",
    )
    # Byte anchoring cannot supply the body/binding rejection under test.
    monkeypatch.setitem(
        direct_flow._SOURCE_SEALS,
        control.source,
        hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert not result["ok"]
    assert result["errors"] == [control.diagnostic]
    assert result["offending_paths"] == [control.source]
    assert not any("source seal mismatch:" in error for error in result["errors"])
    assert all(
        (tmp_path / source).read_bytes() == original
        for source, original in originals.items()
        if source != control.source
    )


def test_outer_helper_runner_keeps_the_new_population_separate_and_required() -> None:
    """All forty named controls run without changing any original mutation family."""
    report = direct_flow.run_direct_flow_mutation_controls(repo_root=_ROOT)
    assert "outer_helper_mutations" in report, (
        "The canonical runner omits the outer-helper population"
    )
    cases = report["outer_helper_mutations"]
    assert set(cases) == set(CONTROLS)
    assert report["outer_helper_names_match_expected"]
    assert report["admitted_outer_helper_mutations"] == []
    assert report["all_rejected"]
    assert report["mutation_count"] == 421
    assert report["supplemental_mutation_count"] == 96
    assert report["retained_prefix_mutation_count"] == 113
    for name, case in cases.items():
        assert case["rejected"]
        assert case["errors"] == [CONTROLS[name].diagnostic]
        assert case["offending_paths"] == [CONTROLS[name].source]


def _change_clean_report(
    *, fault: str, clean: direct_flow.DirectFlowReport, control: OuterHelperControl
) -> None:
    """Apply one declared baseline corruption to the actual clean report."""
    if fault == "unclean_verdict":
        clean["ok"] = False
    elif fault == "baseline_result":
        clean["result"] = "fail"
    elif fault == "baseline_error":
        clean["errors"] = [control.diagnostic]
    elif fault == "baseline_path":
        clean["offending_paths"] = [control.source]


def _change_mutant_report(
    *, fault: str, result: direct_flow.DirectFlowReport, control: OuterHelperControl
) -> None:
    """Apply one declared mutant corruption to the actual failing report."""
    if fault == "mutant_passes":
        result["ok"] = True
    elif fault == "mutant_result":
        result["result"] = "pass"
    elif fault == "no_diagnostic":
        result["errors"] = []
    elif fault == "wrong_diagnostic":
        result["errors"] = ["unrelated helper changed"]
    elif fault == "seal_only":
        result["errors"] = [f"{control.source}: source seal mismatch: pending"]
    elif fault == "extra_error":
        result["errors"] = [control.diagnostic, "unrelated baseline error"]
    elif fault == "wrong_path":
        result["offending_paths"] = ["src/_lcm/egm/outer_refinement.py"]
    elif fault == "no_path":
        result["offending_paths"] = []
    elif fault == "extra_path":
        result["offending_paths"] = [control.source, "unrelated.py"]


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "unclean_verdict",
        "baseline_result",
        "baseline_error",
        "baseline_path",
        "mutant_passes",
        "mutant_result",
        "no_diagnostic",
        "wrong_diagnostic",
        "seal_only",
        "extra_error",
        "wrong_path",
        "no_path",
        "extra_path",
    ],
)
def test_outer_helper_rejection_requires_every_scoped_failure_condition(
    *, fault: str
) -> None:
    """Baseline errors, wrong verdicts and unrelated paths cannot supply sensitivity."""
    control = CONTROLS["outer_helpers:golden:nan_not_sanitized"]
    clean: direct_flow.DirectFlowReport = {
        "ok": True,
        "result": "pass",
        "errors": [],
        "offending_paths": [],
        "routes": {},
        "certified_corridor_sources": [],
        "source_seals": {},
    }
    result: direct_flow.DirectFlowReport = {
        **clean,
        "ok": False,
        "result": "fail",
        "errors": [control.diagnostic],
        "offending_paths": [control.source],
    }
    _change_clean_report(fault=fault, clean=clean, control=control)
    _change_mutant_report(fault=fault, result=result, control=control)
    assert direct_flow._outer_helper_case_rejected(
        clean=clean, result=result, source=control.source, diagnostic=control.diagnostic
    ) is (fault == "none")


@pytest.mark.parametrize("fault", ["admitted", "omitted"])
def test_outer_helper_failure_prevents_the_actual_runner_aggregate(
    *, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admitted or missing helper prevents aggregate all-rejected acceptance."""
    name = "outer_helpers:golden:nan_not_sanitized"
    if fault == "admitted":
        original = direct_flow._outer_helper_case_rejected

        def admit_probe(
            *,
            clean: direct_flow.DirectFlowReport,
            result: direct_flow.DirectFlowReport,
            source: direct_flow.SourcePath,
            diagnostic: str,
        ) -> bool:
            return diagnostic != CONTROLS[name].diagnostic and original(
                clean=clean, result=result, source=source, diagnostic=diagnostic
            )

        monkeypatch.setattr(direct_flow, "_outer_helper_case_rejected", admit_probe)
    else:
        original_specs = direct_flow.outer_helper_mutation_specs

        def omit_probe(*, repo_root: Path) -> dict[str, dict[str, str]]:
            return {
                key: value
                for key, value in original_specs(repo_root=repo_root).items()
                if key != name
            }

        monkeypatch.setattr(direct_flow, "outer_helper_mutation_specs", omit_probe)
    report = direct_flow.run_direct_flow_mutation_controls(repo_root=_ROOT)
    assert report["clean"]["ok"]
    assert report["clean"]["errors"] == []
    assert report["mutation_count"] == 421
    assert report["supplemental_mutation_count"] == 96
    assert report["retained_prefix_mutation_count"] == 113
    assert not report["all_rejected"]
    if fault == "admitted":
        assert name in report["admitted_outer_helper_mutations"]
    else:
        assert name not in report["outer_helper_mutations"]
        assert not report["outer_helper_names_match_expected"]
