"""Public metadata round trips, bounded failure, and full-relation views."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from _lcm_regime_finder import find_regimes, verify_production_result
from _lcm_regime_finder.graph_view import typed_incidence_graph, boundary_maps

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'tests/regime_finder/examples/declarations/tiny-huge-grid.json'


class ProductionCommandTests(unittest.TestCase):
    def test_metadata_engine_imports_without_backend_packages(self):
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
            "import _lcm_regime_finder.cli, _lcm_regime_finder.production; "
            "assert not {'lcm', '_lcm', 'jax'}.intersection(sys.modules)"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-S', '-c', script], cwd=directory,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_output_cannot_overwrite_original_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'report.json'; source.write_bytes(ORIGINAL.read_bytes())
            before = source.read_bytes()
            run = subprocess.run([sys.executable, '-m', '_lcm_regime_finder', 'analyze-model', "--search-mode", "exact",
                                  str(source), '--output', temp], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertEqual(source.read_bytes(), before)

    def test_native_spec_schema_version_does_not_select_finite_engine(self):
        from tests.regime_finder.test_declaration_ingestion import DeclarationIngestionTests
        original = DeclarationIngestionTests().raw_spec()
        original['schema_version'] = 1
        with patch('_lcm_regime_finder.core.find_regimes', side_effect=AssertionError('legacy route called')):
            report = find_regimes(original, search_mode="exact")
        self.assertEqual(report['schema'], 'regime-production-result/v1')

    def test_production_route_cannot_enter_dense_regression_engine(self):
        original = json.loads(ORIGINAL.read_text())
        with patch('_lcm_regime_finder.core.prepare', side_effect=AssertionError('dense engine called')):
            report = find_regimes(original, search_mode="exact")
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(report['analysis']['production_execution']['dense_cartesian_cells_enumerated'], 0)
        self.assertEqual(verify_production_result(original, report)['status'], 'PASS')

    def test_public_saved_round_trip_and_projection_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'result'
            command = [sys.executable, '-m', '_lcm_regime_finder', 'analyze-model', "--search-mode", "exact", str(ORIGINAL), '--output', str(output)]
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'COMPLETE')
            self.assertTrue((output / 'report.html').is_file())
            self.assertTrue((output / 'report.md').is_file())
            self.assertTrue(list(output.rglob('*.mmd')))
            pngs = list(output.rglob('*.png')); self.assertTrue(pngs)
            self.assertEqual(pngs[0].read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            resource = json.loads((output / 'resource.json').read_text())
            self.assertTrue(resource['within_declared_budget'])
            self.assertEqual(resource['budget'], {'seconds': 900.0, 'host_gib': 8.0})
            check = subprocess.run([sys.executable, '-m', '_lcm_regime_finder', 'check-model-result',
                                    str(ORIGINAL), str(output / 'report.json')], cwd=ROOT,
                                   capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr + check.stdout)
            self.assertEqual(json.loads(check.stdout)['status'], 'PASS')
            corrupted = deepcopy(report)
            option = next(option for cell in corrupted['analysis']['interfaces'].values()
                          for option in cell['options'] if any(p['kind'] == 'characteristic' for p in option['state_projection'].values()))
            projection = next(p for p in option['state_projection'].values() if p['kind'] == 'characteristic')
            projection['values'][0] = 987654321
            self.assertEqual(verify_production_result(json.loads(ORIGINAL.read_text()), corrupted)['status'], 'FAIL')

    def test_resource_exhaustion_never_changes_original_or_claims_optimum(self):
        before = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            run = subprocess.run([sys.executable, '-m', '_lcm_regime_finder', 'analyze-model', "--search-mode", "exact",
                                  str(ORIGINAL), '--output', temp, '--seconds', '0'],
                                 cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr + run.stdout)
            report = json.loads((Path(temp) / 'report.json').read_text())
            self.assertEqual(report['status'], 'INCOMPLETE')
            self.assertEqual(report['established_lower_bound'], {'work': 0, 'storage': 0, 'clarity': 0})
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), before)

    def test_full_relations_and_owned_metadata_are_preserved_in_graph_view(self):
        original = json.loads(ORIGINAL.read_text())
        original['relations'][0].update(kind='joint', joint_outputs={'x': 'law-x', 'y': 'law-y'},
                                        ownership={'leg': 'owned route A'})
        graph = typed_incidence_graph(original)
        relation_nodes = [n for n in graph['nodes'] if 'relation' in n]
        self.assertEqual([n['relation'] for n in relation_nodes], original['relations'])
        self.assertEqual(len(relation_nodes), len(original['relations']))

    def test_declared_allocation_cannot_be_replaced_by_conditional_scenario(self):
        original = json.loads(ORIGINAL.read_text()); result = find_regimes(original, search_mode="exact")
        result['scenarios'][0]['allocation_contract'] = {'status': 'conditional_scenario', 'kind': 'global_union'}
        self.assertEqual(verify_production_result(original, result)['status'], 'FAIL')

    def test_boundary_maps_cover_every_original_relation(self):
        original = json.loads(ORIGINAL.read_text()); result = find_regimes(original, search_mode="exact")
        plan = result['scenarios'][0]['frontier'][0]
        maps = boundary_maps(result['analysis'], plan)
        self.assertEqual([r['original_relation'] for r in maps], [r['id'] for r in original['relations']])
        self.assertTrue(all(r['native_execution'] == 'NOT_RUN' for r in maps))


if __name__ == '__main__':
    unittest.main()
