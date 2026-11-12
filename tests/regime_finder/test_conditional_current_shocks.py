"""Run native capture regressions without retaining JAX in the metadata suite."""

import pytest
import importlib.util
import os
from pathlib import Path
import subprocess


@pytest.mark.skipif(
    not (importlib.util.find_spec("lcm")), reason="Native pylcm environment required"
)
def test_native_capture_to_recommendation_and_source_controls() -> None:
    # JAX's retained process memory would contaminate later RSS-budget tests.
    # The child inherits the caller's cap scope and exact Pixi environment.
    command = [
        "pixi",
        "run",
        "--as-is",
        "--manifest-path",
        os.environ["PIXI_PROJECT_MANIFEST"],
        "-e",
        os.environ.get("PIXI_ENVIRONMENT_NAME", "tests-cpu"),
        "pytest",
        "tests/regime_finder/native_conditional_current_shocks.py",
        "-v",
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode) == (0), result.stdout + result.stderr
