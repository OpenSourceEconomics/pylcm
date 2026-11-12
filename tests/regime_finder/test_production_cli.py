"""Public metadata round trips, bounded failure, and full-relation views."""

from copy import deepcopy
import pytest
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from _lcm_regime_finder import find_regimes, verify_production_result
from _lcm_regime_finder.graph_view import typed_incidence_graph, boundary_maps

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "tests/regime_finder/examples/declarations/tiny-huge-grid.json"


def test_metadata_engine_imports_without_backend_packages() -> None:
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import _lcm_regime_finder.cli, _lcm_regime_finder.production; "
        "assert not {'lcm', '_lcm', 'jax'}.intersection(sys.modules)"
    )
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
    assert (result.returncode) == (0), result.stdout + result.stderr


def test_output_cannot_overwrite_original_input() -> None:
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "report.json"
        source.write_bytes(ORIGINAL.read_bytes())
        before = source.read_bytes()
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "_lcm_regime_finder",
                "analyze-model",
                "--search-mode",
                "exact",
                str(source),
                "--output",
                temp,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert (run.returncode) == (1)
        assert (source.read_bytes()) == (before)


def test_native_spec_schema_version_does_not_select_finite_engine() -> None:
    from tests.regime_finder.test_declaration_ingestion import _create_raw_spec

    original = _create_raw_spec()
    original["schema_version"] = 1
    with patch(
        "_lcm_regime_finder.core.find_regimes",
        side_effect=AssertionError("legacy route called"),
    ):
        report = find_regimes(original, search_mode="exact")
    assert (report["schema"]) == ("regime-production-result/v1")


def test_production_route_cannot_enter_dense_regression_engine() -> None:
    original = json.loads(ORIGINAL.read_text())
    with patch(
        "_lcm_regime_finder.core.prepare",
        side_effect=AssertionError("dense engine called"),
    ):
        report = find_regimes(original, search_mode="exact")
    assert (report["status"]) == ("COMPLETE")
    assert (
        report["analysis"]["production_execution"]["dense_cartesian_cells_enumerated"]
    ) == (0)
    assert (verify_production_result(original, report)["status"]) == ("PASS")


def test_public_saved_round_trip_and_projection_corruption() -> None:
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "result"
        command = [
            sys.executable,
            "-m",
            "_lcm_regime_finder",
            "analyze-model",
            "--search-mode",
            "exact",
            str(ORIGINAL),
            "--output",
            str(output),
        ]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert (run.returncode) == (0), run.stderr + run.stdout
        report = json.loads((output / "report.json").read_text())
        assert (report["status"]) == ("COMPLETE")
        assert (output / "report.html").is_file()
        assert (output / "report.md").is_file()
        assert list(output.rglob("*.mmd"))
        pngs = list(output.rglob("*.png"))
        assert pngs
        assert (pngs[0].read_bytes()[:8]) == (b"\x89PNG\r\n\x1a\n")
        resource = json.loads((output / "resource.json").read_text())
        assert resource["within_declared_budget"]
        assert (resource["budget"]) == ({"seconds": 900.0, "host_gib": 8.0})
        check = subprocess.run(
            [
                sys.executable,
                "-m",
                "_lcm_regime_finder",
                "check-model-result",
                str(ORIGINAL),
                str(output / "report.json"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert (check.returncode) == (0), check.stderr + check.stdout
        assert (json.loads(check.stdout)["status"]) == ("PASS")
        corrupted = deepcopy(report)
        option = next(
            option
            for cell in corrupted["analysis"]["interfaces"].values()
            for option in cell["options"]
            if any(
                p["kind"] == "characteristic"
                for p in option["state_projection"].values()
            )
        )
        projection = next(
            p
            for p in option["state_projection"].values()
            if p["kind"] == "characteristic"
        )
        projection["values"][0] = 987654321
        assert (
            verify_production_result(json.loads(ORIGINAL.read_text()), corrupted)[
                "status"
            ]
        ) == ("FAIL")


def test_resource_exhaustion_never_changes_original_or_claims_optimum() -> None:
    before = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as temp:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "_lcm_regime_finder",
                "analyze-model",
                "--search-mode",
                "exact",
                str(ORIGINAL),
                "--output",
                temp,
                "--seconds",
                "0",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert (run.returncode) == (2), run.stderr + run.stdout
        report = json.loads((Path(temp) / "report.json").read_text())
        assert (report["status"]) == ("INCOMPLETE")
        assert (report["established_lower_bound"]) == (
            {"work": 0, "storage": 0, "clarity": 0}
        )
    assert (hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()) == (before)


def test_full_relations_and_owned_metadata_are_preserved_in_graph_view() -> None:
    original = json.loads(ORIGINAL.read_text())
    original["relations"][0].update(
        kind="joint",
        joint_outputs={"x": "law-x", "y": "law-y"},
        ownership={"leg": "owned route A"},
    )
    graph = typed_incidence_graph(original)
    relation_nodes = [n for n in graph["nodes"] if "relation" in n]
    assert ([n["relation"] for n in relation_nodes]) == (original["relations"])
    assert (len(relation_nodes)) == (len(original["relations"]))


def test_declared_allocation_cannot_be_replaced_by_conditional_scenario() -> None:
    original = json.loads(ORIGINAL.read_text())
    result = find_regimes(original, search_mode="exact")
    result["scenarios"][0]["allocation_contract"] = {
        "status": "conditional_scenario",
        "kind": "global_union",
    }
    assert (verify_production_result(original, result)["status"]) == ("FAIL")


def test_boundary_maps_cover_every_original_relation() -> None:
    original = json.loads(ORIGINAL.read_text())
    result = find_regimes(original, search_mode="exact")
    plan = result["scenarios"][0]["frontier"][0]
    maps = boundary_maps(result["analysis"], plan)
    assert ([r["original_relation"] for r in maps]) == (
        [r["id"] for r in original["relations"]]
    )
    assert all(r["native_execution"] == "NOT_RUN" for r in maps)


def _write_evaluation_analysis(
    *,
    path: Path,
    wrapped: bool,
    complete: bool,
) -> bytes:
    from tests.regime_finder.existing_family_oracle import _mapping
    from tests.regime_finder.test_existing_layout_contract import _create_case

    analysis = _create_case("policies")
    analysis["schema"] = "regime-semantic-analysis/v1"
    analysis["discovery_complete"] = complete
    analysis.pop("clarity_weights")
    allocation = _mapping(_mapping(analysis["declarations"])["allocation"])
    analysis["allocation_scenario"] = {**allocation, "boundary_unit": 17}
    source = (
        {"schema": "regime-production-result/v1", "analysis": analysis}
        if wrapped
        else analysis
    )
    path.write_text(json.dumps(source))
    return path.read_bytes()


@pytest.mark.parametrize("wrapped", [False, True])
def test_evaluation_curve_uses_raw_grid_counts_without_scalar_recommendation(
    *,
    tmp_path: Path,
    wrapped: bool,
) -> None:
    from _lcm_regime_finder import cli, frontier
    from tests.regime_finder.existing_family_oracle import _mapping, _rows

    source = tmp_path / "analysis.json"
    before = _write_evaluation_analysis(path=source, wrapped=wrapped, complete=True)
    output = tmp_path / "result"

    with patch.object(
        frontier,
        "recommend",
        side_effect=AssertionError(
            "An evaluation curve must not call scalar recommendation"
        ),
    ):
        exit_code = cli.main(
            ["evaluation-curve", str(source), "--output", str(output), "--worker"]
        )
    assert exit_code == 0
    result = _mapping(json.loads((output / "report.json").read_text()))
    assert result["schema"] == "regime-evaluation-curve/v1"
    assert result["status"] == "COMPLETE"
    assert _mapping(result["independent_check"])["status"] == "PASS"
    assert "recommendation" not in result and "score" not in result
    points = _rows(result["frontier"])
    assert points
    assert all(point["link_charge_evaluations"] == 0 for point in points)
    assert all(point["evaluations"] == point["grid_evaluations"] for point in points)
    assert source.read_bytes() == before


def test_evaluation_curve_parent_preserves_a_partial_discovery_result(
    tmp_path: Path,
) -> None:
    from tests.regime_finder.existing_family_oracle import _mapping

    source = tmp_path / "analysis.json"
    before = _write_evaluation_analysis(path=source, wrapped=False, complete=False)
    output = tmp_path / "result"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "_lcm_regime_finder",
            "evaluation-curve",
            str(source),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = _mapping(json.loads((output / "report.json").read_text()))
    assert result["schema"] == "regime-evaluation-curve/v1"
    assert result["status"] == "PARTIAL"
    assert result["discovery_complete"] is False
    assert result["source_semantics"] == "NOT_ESTABLISHED_BY_COST_SEARCH"
    assert source.read_bytes() == before


@pytest.mark.parametrize("worker", [False, True])
def test_evaluation_curve_cannot_replace_its_saved_analysis(
    *,
    tmp_path: Path,
    worker: bool,
) -> None:
    source = tmp_path / "report.json"
    before = _write_evaluation_analysis(path=source, wrapped=False, complete=True)
    command = [
        sys.executable,
        "-m",
        "_lcm_regime_finder",
        "evaluation-curve",
        str(source),
        "--output",
        str(tmp_path),
    ]
    if worker:
        command.append("--worker")
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert run.returncode == 1, run.stdout + run.stderr
    assert source.read_bytes() == before
