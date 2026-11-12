"""A cost-ranked candidate must retain its separate adoption disposition."""

import json
from pathlib import Path
import tempfile

from _lcm_regime_finder import production_render as renderer
from tests.regime_finder.test_production_render import fixture


def test_boundary_exports_never_authorize_missing_or_rejected_decisions() -> None:
    from _lcm_regime_finder.cli import candidate_boundary_maps

    result = fixture()
    for interface in result["analysis"]["interfaces"].values():
        for option in interface["options"]:
            option["constants"] = {}
    assert not (candidate_boundary_maps(result)[0]["transformation_authorized"])
    for disposition in ("reject", "investigate", "adopt"):
        record = {
            "scenario_index": 0,
            "graph_id": "G0000",
            "source_snapshot_sha256": "source-anchor",
            "candidate_sha256": "candidate-anchor",
            "disposition": disposition,
        }
        result["proposal_decisions"] = [record]
        row = candidate_boundary_maps(result)[0]
        assert (row["decision"]) == (record)
        assert (row["transformation_authorized"]) == (disposition == "adopt")


def test_public_recheck_distinguishes_budget_exhaustion() -> None:
    from _lcm_regime_finder import find_regimes, verify_production_result
    from tests.regime_finder.tiny_metadata_family import original

    source = original()
    result = find_regimes(source, search_mode="exact")
    assert (result["status"]) == ("COMPLETE")
    checked = verify_production_result(source, result, limits={"seconds": 0})
    assert (checked["status"]) == ("INCOMPLETE")
    assert any(row.get("checking_complete") is False for row in checked["frontiers"])


def test_missing_local_certificate_requires_investigation() -> None:
    result = fixture()
    assert ("Disposition: investigate") in (renderer.render_html(result))
    assert ("Disposition: **investigate**") in (renderer.render_markdown(result))


def test_rejected_ranked_candidate_stays_rejected_in_every_view() -> None:
    result = fixture()
    records = [
        {
            "scenario_index": 0,
            "graph_id": "G0000",
            "disposition": "reject",
            "reasons": ["Reporting dependency lost."],
        }
    ]
    result["proposal_decisions"] = records
    with tempfile.TemporaryDirectory() as directory:
        # This tests report construction, not semantic validation. The
        # checker supplies decisions; the renderer must not reinterpret them.
        renderer.write_production_report(result, directory)
        output = Path(directory)
        assert (json.loads((output / "proposal-decisions.json").read_text())) == (
            records
        )
        assert ("Disposition: reject") in ((output / "report.html").read_text())
        assert ("Disposition: **reject**") in ((output / "report.md").read_text())
        assert ("Recommendation:") not in ((output / "report.md").read_text())
        assert ("recommended") not in ((output / "report.md").read_text())
        assert ("Recommendation:") not in ((output / "report.html").read_text())
        saved = json.loads((output / "report-data.json").read_text())
        assert (saved["scenarios"][0]["alternatives"][0]["decision"]) == (records[0])


def test_same_graph_id_in_different_scenarios_keeps_own_disposition() -> None:
    from copy import deepcopy

    result = fixture()
    result["scenarios"].append(deepcopy(result["scenarios"][0]))
    result["proposal_decisions"] = [
        {
            "scenario_index": 0,
            "graph_id": "G0000",
            "disposition": "adopt",
            "reasons": [],
        },
        {
            "scenario_index": 1,
            "graph_id": "G0000",
            "disposition": "investigate",
            "reasons": [],
        },
    ]
    view = renderer._view(result)
    assert (view["scenarios"][0]["points"][0]["decision"]["disposition"]) == ("adopt")
    assert (view["scenarios"][1]["points"][0]["decision"]["disposition"]) == (
        "investigate"
    )
