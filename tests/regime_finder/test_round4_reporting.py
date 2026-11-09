"""A cost-ranked candidate must retain its separate adoption disposition."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from _lcm_regime_finder import production_render as renderer
from tests.regime_finder.test_production_render import fixture


class ProposalReportingTests(unittest.TestCase):
    def test_boundary_exports_never_authorize_missing_or_rejected_decisions(self):
        from _lcm_regime_finder.cli import candidate_boundary_maps
        result = fixture()
        for interface in result['analysis']['interfaces'].values():
            for option in interface['options']:
                option['constants'] = {}
        self.assertFalse(candidate_boundary_maps(result)[0]['transformation_authorized'])
        for disposition in ('reject', 'investigate', 'adopt'):
            record = {'scenario_index': 0, 'graph_id': 'G0000',
                      'source_snapshot_sha256': 'source-anchor',
                      'candidate_sha256': 'candidate-anchor',
                      'disposition': disposition}
            result['proposal_decisions'] = [record]
            row = candidate_boundary_maps(result)[0]
            self.assertEqual(row['decision'], record)
            self.assertEqual(row['transformation_authorized'], disposition == 'adopt')

    def test_public_recheck_distinguishes_budget_exhaustion(self):
        from _lcm_regime_finder import find_regimes, verify_production_result
        from tests.regime_finder.tiny_metadata_family import original
        source = original()
        result = find_regimes(source, search_mode="exact")
        self.assertEqual(result['status'], 'COMPLETE')
        checked = verify_production_result(source, result, limits={'seconds': 0})
        self.assertEqual(checked['status'], 'INCOMPLETE')
        self.assertTrue(any(row.get('checking_complete') is False for row in checked['frontiers']))

    def test_missing_local_certificate_requires_investigation(self):
        result = fixture()
        self.assertIn("Disposition: investigate", renderer.render_html(result))
        self.assertIn("Disposition: **investigate**", renderer.render_markdown(result))

    def test_rejected_ranked_candidate_stays_rejected_in_every_view(self):
        result = fixture()
        records = [{"scenario_index": 0, "graph_id": "G0000",
                    "disposition": "reject", "reasons": ["Reporting dependency lost."]}]
        result["proposal_decisions"] = records
        with tempfile.TemporaryDirectory() as directory:
            # This tests report construction, not semantic validation. The
            # checker supplies decisions; the renderer must not reinterpret them.
            with patch.object(renderer, "_view", wraps=renderer._view) as view:
                renderer.write_production_report(result, directory)
                self.assertEqual(view.call_count, 1)
            output = Path(directory)
            self.assertEqual(json.loads((output / "proposal-decisions.json").read_text()), records)
            self.assertIn("Disposition: reject", (output / "report.html").read_text())
            self.assertIn("Disposition: **reject**", (output / "report.md").read_text())
            self.assertNotIn("Recommendation:", (output / "report.md").read_text())
            self.assertNotIn("recommended", (output / "report.md").read_text())
            self.assertNotIn("Recommendation:", (output / "report.html").read_text())
            saved = json.loads((output / "report-data.json").read_text())
            self.assertEqual(saved["scenarios"][0]["alternatives"][0]["decision"], records[0])

    def test_same_graph_id_in_different_scenarios_keeps_own_disposition(self):
        from copy import deepcopy
        result = fixture()
        result["scenarios"].append(deepcopy(result["scenarios"][0]))
        result["proposal_decisions"] = [
            {"scenario_index": 0, "graph_id": "G0000", "disposition": "adopt", "reasons": []},
            {"scenario_index": 1, "graph_id": "G0000", "disposition": "investigate", "reasons": []},
        ]
        view = renderer._view(result)
        self.assertEqual(view["scenarios"][0]["points"][0]["decision"]["disposition"], "adopt")
        self.assertEqual(view["scenarios"][1]["points"][0]["decision"]["disposition"], "investigate")


if __name__ == "__main__":
    unittest.main()
