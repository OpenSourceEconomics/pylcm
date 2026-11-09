"""Exercise the independent public-entrypoint ownership matrix in normal tests."""
from pathlib import Path
import unittest
import os
from unittest.mock import patch

from . import rf10_checker_delivery_oracle as oracle
from .owned_write_cases import cases, one_case


class OwnedMutationIntegration(unittest.TestCase):
    def test_public_owned_mutation_matrix(self):
        project = Path(__file__).resolve().parents[2]
        for case in cases():
            with self.subTest(case=case):
                result = one_case(project, case, oracle)
                self.assertTrue(result['passed'], result.get('violations'))

    def test_hardlink_aliases_use_the_same_independent_oracle(self):
        project = Path(__file__).resolve().parents[2]
        def hardlink(path, target, target_is_directory=False):
            os.link(target, path)
        with patch.object(Path, 'symlink_to', hardlink):
            for case in cases():
                with self.subTest(case=case):
                    result = one_case(project, case, oracle)
                    self.assertTrue(result['passed'], result.get('violations'))

    def test_structural_input_artifacts_cannot_be_supervisor_log_targets(self):
        import contextlib
        import io
        import json
        import tempfile
        from _lcm_regime_finder import cli
        from .test_delivery_invalidation import DeliveryInvalidation
        for artifact in ('manifest', 'chunk'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, owner = DeliveryInvalidation().source(root)
                output = root/'analysis'
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.main(['analyze-model',str(source),'--output',str(output),'--worker']),0)
                report = output/'report.json'
                directory = source.parent/json.loads(source.read_text())['structural_store']['path']
                target = directory/'manifest.json' if artifact=='manifest' else owner.reader.payloads.path(next(iter(owner.reader.payloads)))
                protected = oracle.snapshot([source,report,target])
                run = output/'verification/verification-run';run.mkdir(parents=True)
                (run/'analysis-execution.log').symlink_to(target)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = cli.main(['check-model-result',str(source),str(report)])
                issues = oracle.assess(exit_code=code,result_path=output/'verification/check.json',protected=protected,success=False)
                self.assertFalse(issues,issues)
