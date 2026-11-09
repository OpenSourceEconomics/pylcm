"""Resource exhaustion does not promote or materialize unchecked checkpoints."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from _lcm_regime_finder.resources import run_bounded

class Round4ResourceTests(unittest.TestCase):
    def test_monitoring_without_time_or_memory_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            code = run_bounded(['sleep', '0.1'], output=output, seconds=None, host_gib=None)
            self.assertEqual(code, 0)
            receipt = json.loads((output / 'resource.json').read_text())
            self.assertEqual(receipt['budget'], {'seconds': None, 'host_gib': None})
            self.assertIsNone(receipt['exhaustion_reason'])
            self.assertFalse((output / 'report.json').exists())

    def test_exhaustion_preserves_reference_and_zero_bound_without_adoption(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            checkpoint = output / 'partial-frontier.json'
            checkpoint.write_text(json.dumps({'status': 'INCOMPLETE',
                'frontier': [{'id': 'unverified-source-projection', 'cost': {'work': 1}}]}))
            before = checkpoint.read_bytes()
            code = run_bounded([sys.executable, '-c', 'import time; time.sleep(5)'],
                               output=output, seconds=0.03, host_gib=0.5)
            self.assertEqual(code, 2)
            result = json.loads((output / 'report.json').read_text())
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertIsNone(result['partial_frontier'])
            self.assertIsNone(result['recommendation'])
            self.assertEqual(result['partial_frontier_reference']['candidate_safety'], 'unverified')
            self.assertEqual(result['partial_frontier_reference']['size_bytes'], len(before))
            self.assertEqual(result['established_lower_bound'], {'work': 0, 'storage': 0, 'clarity': 0})
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_large_checkpoint_is_not_decoded_by_supervisor(self):
        # Sparse resource unit probe, not evidence for a production candidate.
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            checkpoint = output / 'partial-frontier.json'
            with checkpoint.open('wb') as stream:
                stream.truncate(128 * 1024**2)
            code = run_bounded([sys.executable, '-c', 'import time; time.sleep(5)'],
                               output=output, seconds=0.03, host_gib=0.5)
            self.assertEqual(code, 2)
            result = json.loads((output / 'report.json').read_text())
            self.assertEqual(result['partial_frontier_reference']['size_bytes'], 128 * 1024**2)
            self.assertLess((output / 'report.json').stat().st_size, 4096)

    def test_early_exhaustion_without_checkpoint_has_no_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            code = run_bounded([sys.executable, '-c', 'import time; time.sleep(5)'],
                               output=output, seconds=0, host_gib=0.5)
            self.assertEqual(code, 2)
            result = json.loads((output / 'report.json').read_text())
            self.assertIsNone(result['partial_frontier_reference'])
            self.assertIsNone(result['recommendation'])

if __name__ == '__main__':
    unittest.main()
