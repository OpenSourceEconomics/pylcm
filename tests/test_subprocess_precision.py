"""A child interpreter spawned by a test runs at the suite's `--precision`."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import tests.conftest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_REPORT_X64 = "import jax; print(jax.config.read('jax_enable_x64'))"


@pytest.mark.parametrize(
    "copy_environment",
    [
        pytest.param(False, id="inherited_environment"),
        pytest.param(True, id="copied_environment"),
    ],
)
def test_child_interpreter_reports_the_requested_precision(
    *, copy_environment: bool
) -> None:
    """A fresh interpreter has `jax_enable_x64` equal to the suite's precision."""
    environment = {**os.environ, "JAX_PLATFORMS": "cpu"} if copy_environment else None
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _REPORT_X64],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tests.conftest.X64_ENABLED)
