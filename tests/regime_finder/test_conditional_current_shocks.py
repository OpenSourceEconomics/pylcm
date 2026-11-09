"""Run native capture regressions without retaining JAX in the metadata suite."""
import importlib.util
import os
from pathlib import Path
import subprocess
import unittest


@unittest.skipUnless(importlib.util.find_spec('lcm'), 'Native pylcm environment required')
class ConditionalCurrentShockCaptureTests(unittest.TestCase):
    def test_native_capture_to_recommendation_and_source_controls(self):
        # JAX's retained process memory would contaminate later RSS-budget tests.
        # The child inherits the caller's cap scope and exact Pixi environment.
        command = ['pixi', 'run', '--as-is', '--manifest-path', os.environ['PIXI_PROJECT_MANIFEST'],
                   '-e', os.environ.get('PIXI_ENVIRONMENT_NAME', 'tests-cpu'),
                   'python', '-m', 'unittest', 'tests.regime_finder.native_conditional_current_shocks', '-v']
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
