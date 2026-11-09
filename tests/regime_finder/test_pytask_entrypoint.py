"""Public outer-budget metadata runner, not a solver smoke fixture."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from _lcm_regime_finder.pytask_api import analyse_snapshot
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tests/regime_finder/examples/declarations/tiny-huge-grid.json'

class MetadataTaskEntrypointTests(unittest.TestCase):
    def test_actual_bounded_runner_produces_readable_source_bound_decisions(self):
        before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'result'
            summary=analyse_snapshot(SOURCE,output, search_mode="exact")
            self.assertEqual(summary['status'],'COMPLETE')
            self.assertTrue(summary['checker_modules_unchanged'])
            self.assertEqual(summary['source_snapshot_sha256'],before)
            self.assertTrue((output/'next-actions.html').is_file())
            self.assertTrue((output/'next-actions.md').is_file())
            self.assertEqual(summary['decisions'][0]['operation'],'retain_original')
            self.assertEqual(summary['decisions'][0]['decision'],'adopt')
            self.assertTrue(json.loads((output/'outer-resource/resource.json').read_text())['within_declared_budget'])
            self.assertIn('No recommendation is applied automatically',(output/'next-actions.md').read_text())
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),before)

    def test_exhaustion_has_no_adoption_and_preserves_original(self):
        before=SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'result'
            with self.assertRaisesRegex(RuntimeError,'INCOMPLETE'):
                analyse_snapshot(SOURCE,output,seconds=0, search_mode="exact")
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['status'],'INCOMPLETE')
            self.assertNotIn('decisions',summary)
        self.assertEqual(SOURCE.read_bytes(),before)

    def test_output_namespace_cannot_overwrite_original_summary_named_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            source=folder/'summary.json';source.write_bytes(SOURCE.read_bytes())
            before=source.read_bytes()
            with self.assertRaisesRegex(ValueError,'original input'):
                analyse_snapshot(source,folder, search_mode="exact")
            self.assertEqual(source.read_bytes(),before)

if __name__=='__main__':unittest.main()
