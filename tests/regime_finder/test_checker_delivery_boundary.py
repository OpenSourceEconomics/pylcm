"""Check output delivery and failure handling against an independent oracle."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from _lcm_regime_finder import cli, resources
from .checker_delivery_oracle import assert_delivery, file_identity


class CheckerDeliveryBoundary(unittest.TestCase):
    def prepared(self, root, *, source_name='source.json'):
        source = root / source_name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((PROJECT / 'tests/regime_finder/examples/declarations/tiny-huge-grid.json').read_bytes())
        output = root / 'analysis'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(['analyze-model', str(source), '--output', str(output), '--worker'])
        self.assertEqual(code, 0)
        report = output / 'report.json'
        return source, report, {path: file_identity(path) for path in (source, report)}

    def parameters(self):
        return itertools.product(('check-result', 'check-model-result'), ('default', 'explicit'))

    def destination(self, root, report, mode):
        # A literal public contract, not an import of the production resolver.
        return report.parent / 'verification/check.json' if mode == 'default' else root / 'checks/selected.json'

    def invoke(self, command, source, report, mode, result):
        arguments = [command, str(source), str(report)]
        if mode == 'explicit':
            arguments += ['--output', str(result)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main(arguments)

    def test_normal_delivery_matrix(self):
        for command, mode in self.parameters():
            with self.subTest(command=command, output=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source, report, protected = self.prepared(root)
                result = self.destination(root, report, mode)
                code = self.invoke(command, source, report, mode, result)
                assert_delivery(code, result, protected, success=True)

    def test_actual_final_resource_write_failure_matrix(self):
        for command, mode in self.parameters():
            with self.subTest(command=command, output=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source, report, protected = self.prepared(root)
                result = self.destination(root, report, mode)
                # Actual worker, actual successful independent check, actual
                # supervisor receipt write failure. No fault-injection mock.
                (result.parent / 'verification-run/resource.json').mkdir(parents=True)
                code = self.invoke(command, source, report, mode, result)
                assert_delivery(code, result, protected, success=False)

    def test_success_then_parent_boundary_failure_matrix(self):
        runner = resources.run_bounded
        for command, mode, fault in itertools.product(
                ('check-result', 'check-model-result'), ('default', 'explicit'),
                ('exception', 'exhausted', 'worker_failed')):
            with self.subTest(command=command, output=mode, fault=fault), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source, report, protected = self.prepared(root)
                result = self.destination(root, report, mode)
                def finished_then_failed(*args, **kwargs):
                    self.assertEqual(runner(*args, **kwargs), 0)
                    self.assertEqual(json.loads(result.read_text())['status'], 'PASS')
                    if fault == 'exception':
                        raise OSError('controlled post-worker supervisor failure')
                    return 2 if fault == 'exhausted' else 1
                with patch.object(resources, 'run_bounded', side_effect=finished_then_failed):
                    code = self.invoke(command, source, report, mode, result)
                assert_delivery(code, result, protected, success=False)

    def test_protected_original_input_matrix(self):
        for command, mode, protected_kind in itertools.product(
                ('check-result', 'check-model-result'), ('default', 'explicit'),
                ('result', 'temporary')):
            with self.subTest(command=command, output=mode, protected=protected_kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                # First construct a valid report; then place the exact original
                # input at a protected result/temporary pathname.
                source, report, _ = self.prepared(root)
                result = self.destination(root, report, mode)
                guarded = result if protected_kind == 'result' else result.with_name(result.name + '.writing')
                guarded.parent.mkdir(parents=True, exist_ok=True)
                guarded.write_bytes(source.read_bytes())
                protected = {path: file_identity(path) for path in (guarded, report)}
                code = self.invoke(command, guarded, report, mode, result)
                self.assertEqual(code, 1)
                for path, digest in protected.items():
                    self.assertTrue(path.is_file(), str(path))
                    self.assertEqual(file_identity(path), digest)

    def test_direct_worker_temporary_symlink_preserves_input(self):
        for command in ('check-result', 'check-model-result'):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source, report, protected = self.prepared(root)
                result = root / 'checks/selected.json'; result.parent.mkdir()
                temporary = result.with_name(result.name + '.writing')
                temporary.symlink_to(source)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = cli.main([command, str(source), str(report), '--output', str(result), '--worker'])
                self.assertEqual(code, 1)
                for path, digest in protected.items():
                    self.assertEqual(file_identity(path), digest)
                self.assertTrue(temporary.is_symlink())

    def test_existing_run_namespace_symlink_cannot_delete_original_report(self):
        for command, mode in self.parameters():
            with self.subTest(command=command, output=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source, report, protected = self.prepared(root)
                result = self.destination(root, report, mode)
                result.parent.mkdir(parents=True)
                (result.parent / 'verification-run').symlink_to(report.parent, target_is_directory=True)
                code = self.invoke(command, source, report, mode, result)
                self.assertEqual(code, 1)
                for path, digest in protected.items():
                    self.assertTrue(path.is_file(), str(path))
                    self.assertEqual(file_identity(path), digest)

    def test_internal_worker_without_output_remains_stdout_only(self):
        for command in ('check-result', 'check-model-result'):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source, report, protected = self.prepared(root)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = cli.main([command, str(source), str(report), '--worker'])
                self.assertEqual(code, 0)
                self.assertFalse((report.parent / 'verification').exists())
                for path, digest in protected.items():
                    self.assertEqual(file_identity(path), digest)

    def test_reserved_checkpoint_alias_remains_protected(self):
        for command, mode in self.parameters():
            with self.subTest(command=command, output=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); source, report, protected = self.prepared(root)
                result = self.destination(root, report, mode)
                checkpoint = result.parent / 'verification-run/partial-frontier.json'
                checkpoint.parent.mkdir(parents=True)
                checkpoint.symlink_to(source)
                code = self.invoke(command, checkpoint, report, mode, result)
                self.assertEqual(code, 1)
                self.assertTrue(checkpoint.is_symlink())
                for path, digest in protected.items():
                    self.assertEqual(file_identity(path), digest)

    def test_analysis_result_alias_remains_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source, report, protected = self.prepared(root)
            output = root / 'second'; output.mkdir()
            alias = output / 'report.json'; alias.symlink_to(source)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(['analyze-model', str(source), '--output', str(output), '--worker'])
            self.assertEqual(code, 1)
            self.assertTrue(alias.is_symlink())
            for path, digest in protected.items():
                self.assertEqual(file_identity(path), digest)

    def test_end_of_options_keeps_canonical_output_before_positional_paths(self):
        for command, mode in self.parameters():
            with self.subTest(command=command, output=mode), tempfile.TemporaryDirectory() as folder:
                root=Path(folder); source,report,protected=self.prepared(root)
                result=self.destination(root,report,mode)
                arguments=[command]
                if mode=='explicit': arguments += ['--output='+str(result)]
                arguments += ['--',str(source),str(report)]
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    code=cli.main(arguments)
                assert_delivery(code,result,protected,success=True)


if __name__ == '__main__':
    unittest.main()
