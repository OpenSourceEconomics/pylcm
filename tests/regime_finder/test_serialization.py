"""Saved certificates must be verifiable after a real JSON/CLI round trip."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from _lcm_regime_finder import find_regimes, verify_certificate
from tests.regime_finder.fixtures import fertility, endpoint_saving, observed_status


class SavedCertificateTests(unittest.TestCase):
    def test_saved_recommendations_and_incomplete_incumbent(self):
        for original, budget in ((fertility(), None), (endpoint_saving(), None),
                                 (observed_status(), None),
                                 (fertility(), {'candidate_segments': 0})):
            with self.subTest(name=original['name'], budget=budget):
                report = json.loads(json.dumps(find_regimes(original, budget)))
                saved_problem = json.loads(json.dumps(original))
                result = verify_certificate(saved_problem, report)
                self.assertEqual(result['status'],
                                 'INCOMPLETE' if budget else 'PASS', result)
                if not budget:
                    tampered = deepcopy(report)
                    tampered['recommended']['regimes'][0]['boundary_edges'] = []
                    self.assertEqual(verify_certificate(saved_problem, tampered)['status'],
                                     'FAIL')

    def test_public_find_then_check_commands(self):
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='finder-roundtrip-') as tmp:
            root = Path(tmp)
            source = root/'original.json'
            source.write_text(json.dumps(fertility()))
            result_dir = root/'result'
            found = subprocess.run([sys.executable, '-m', '_lcm_regime_finder', 'find-regimes',
                                    str(source), '--output', str(result_dir)],
                                   cwd=project, text=True, capture_output=True)
            self.assertEqual(found.returncode, 0, found.stderr + found.stdout)
            checked = subprocess.run([sys.executable, '-m', '_lcm_regime_finder', 'check-result',
                                      str(result_dir/'problem.json'), str(result_dir/'report.json')],
                                     cwd=project, text=True, capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stderr + checked.stdout)
            result = json.loads(checked.stdout)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['generated_transport']['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
