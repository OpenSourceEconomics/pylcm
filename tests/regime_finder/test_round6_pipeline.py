"""Whole metadata accounting and failure output, without native construction."""
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from _lcm_regime_finder.pytask_api import analyse_snapshot
from _lcm_regime_finder.resources import run_bounded, _tree_rss


SOURCE = Path(__file__).resolve().parents[2] / 'tests/regime_finder/examples/declarations/tiny-huge-grid.json'


class Round6MetadataBudgetTests(unittest.TestCase):
    def test_separate_phase_receipt_includes_checks_and_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'result'
            analyse_snapshot(SOURCE, output, seconds=10)
            receipt = json.loads((output / 'delivery-timings.json').read_text())
            self.assertTrue(all(receipt['phase_timings'][name] >= 0 for name in (
                'normalization_seconds', 'metadata_analysis_seconds', 'source_check_seconds',
                'frontier_search_seconds', 'frontier_check_seconds', 'proposal_check_seconds')))
            self.assertTrue(all(receipt['delivery_timings'][name] >= 0 for name in (
                'checkpoint_write_seconds', 'canonical_report_write_seconds',
                'graph_and_boundary_artifacts_seconds', 'readable_report_seconds')))
            progress = json.loads((output / 'metadata-progress.json').read_text())
            self.assertIn('source_check_seconds', progress['phase_timings'])
            self.assertGreater(progress['elapsed_seconds'], 0)

    def test_existing_capture_consumes_the_same_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'snapshot.json'
            source.write_bytes(SOURCE.read_bytes())
            receipt = {
                'schema': 'regime-metadata-capture-receipt/v1',
                'snapshot_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'status': 'COMPLETE', 'elapsed_seconds': 2.0,
            }
            (root / 'snapshot.metadata-capture.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError, 'INCOMPLETE'):
                analyse_snapshot(source, root / 'result', seconds=1)
            summary = json.loads((root / 'result/summary.json').read_text())
            self.assertEqual(summary['metadata_budget']['capture_seconds_charged'], 2.0)
            self.assertEqual(summary['metadata_budget']['remaining_worker_seconds'], 0)
            self.assertFalse(summary['metadata_budget']['within_time_budget'])
            self.assertNotIn('decisions', summary)

    def test_stale_success_is_replaced_on_early_exhaustion(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'result'
            output.mkdir()
            (output / 'summary.json').write_text('{"status":"COMPLETE","decisions":["stale"]}')
            (output / 'next-actions.html').write_text('stale complete optimum')
            (output / 'analysis.json').write_text('{"old_source":true}')
            (output / 'family-boundary.json').write_text('{"old_family":true}')
            (output / 'semantic-check.json.writing').write_text('incomplete old check')
            with self.assertRaisesRegex(RuntimeError, 'INCOMPLETE'):
                analyse_snapshot(SOURCE, output, seconds=0)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'INCOMPLETE')
            self.assertNotIn('decisions', summary)
            self.assertNotIn('stale complete optimum', (output / 'next-actions.html').read_text())
            self.assertIn('No optimum', (output / 'next-actions.md').read_text())
            self.assertFalse((output / 'analysis.json').exists())
            self.assertFalse((output / 'family-boundary.json').exists())
            self.assertFalse((output / 'semantic-check.json.writing').exists())

    def test_capture_receipt_is_bound_to_exact_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'snapshot.json'
            source.write_bytes(SOURCE.read_bytes())
            (root / 'snapshot.metadata-capture.json').write_text(json.dumps({
                'schema': 'regime-metadata-capture-receipt/v1',
                'snapshot_sha256': 'old snapshot identity',
                'status': 'COMPLETE', 'elapsed_seconds': 0.1,
            }))
            with self.assertRaisesRegex(ValueError, 'original snapshot bytes'):
                analyse_snapshot(source, root / 'result')

    def test_outer_supervisor_references_the_workers_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'partial-frontier.json'
            checkpoint.write_text('{"candidate":"unverified"}')
            code = run_bounded([sys.executable, '-c', 'import time; time.sleep(5)'],
                               output=root / 'outer-resource', seconds=0.03,
                               host_gib=0.5, checkpoint_path=checkpoint)
            report = json.loads((root / 'outer-resource/report.json').read_text())
            self.assertEqual(code, 2)
            reference = report['partial_frontier_reference']
            self.assertEqual((root / 'outer-resource' / reference['path']).resolve(), checkpoint)
            self.assertEqual(reference['candidate_safety'], 'unverified')
            self.assertIsNone(report['recommendation'])

    def test_late_budget_exit_cannot_publish_worker_success(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'result'
            # A supported command_prefix runs a minimal stand-in worker that
            # publishes a result and then exceeds its deadline. This tests the
            # supervisor boundary, not model or private-object manipulation.
            script = ('import json,time; from pathlib import Path; '
                      f'p=Path({str(output)!r}); '
                      '(p/"summary.json").write_text(json.dumps({"status":"COMPLETE","decisions":["old success"]})); '
                      '(p/"report.json").write_text(json.dumps({"status":"COMPLETE","recommendation":"old success"})); '
                      'time.sleep(5)')
            with self.assertRaisesRegex(RuntimeError, 'INCOMPLETE'):
                analyse_snapshot(SOURCE, output, seconds=0.12,
                                 command_prefix=[sys.executable, '-c', script])
            summary = json.loads((output / 'summary.json').read_text())
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(summary['status'], 'INCOMPLETE')
            self.assertNotIn('decisions', summary)
            self.assertEqual(report['status'], 'INCOMPLETE')
            self.assertIsNone(report['recommendation'])
            retained = json.loads((output / report['completed_worker_evidence']).read_text())
            self.assertEqual(retained['recommendation'], 'old success')

    def test_direct_cli_guards_the_entire_output_namespace(self):
        from _lcm_regime_finder.cli import main
        for name in ('metadata-progress.json', 'resource.json', 'report.json.writing'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                original = output / name
                original.write_bytes(SOURCE.read_bytes())
                before = original.read_bytes()
                with contextlib.redirect_stderr(io.StringIO()):
                    code = main(['analyze-model', str(original), '--output', str(output)])
                self.assertEqual(code, 1)
                self.assertEqual(original.read_bytes(), before)

    @unittest.skipUnless(Path('/proc/self/status').is_file(), 'Linux /proc is unavailable')
    def test_namespace_aware_rss_and_descendants_are_observed(self):
        import subprocess
        rss, count = _tree_rss(os.getpid())
        self.assertGreater(rss, 0)
        self.assertGreaterEqual(count, 1)
        # This subprocess allocates only a small bytearray. It exercises actual
        # descendant telemetry, including an ancestor-mounted /proc namespace;
        # it performs no model construction or economic calculation.
        command = [sys.executable, str(Path(__file__).with_name('resource_probe.py'))]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        measured = json.loads(result.stdout)
        record = measured['resource']
        self.assertEqual(measured['exit_code'], 0)
        self.assertTrue(record['process_tree_rss_available'])
        self.assertGreaterEqual(record['maximum_observed_processes'], 2)
        self.assertGreater(record['peak_host_gib'], 16 / 1024)
        self.assertTrue(record['within_declared_budget'])


if __name__ == '__main__':
    unittest.main()
