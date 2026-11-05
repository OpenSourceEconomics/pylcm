"""A worker's collection usage error reaches the xdist controller's output.

Under xdist only the workers collect. A `pytest.UsageError` raised there, for
example by `pytest_policy.apply` refusing a duplicated policy marker, ends every
worker with status 4, and without a guard the controller reports only an
unrelated `assert not crashitem`.
"""

import os
import subprocess
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_SUITE = '''
import pytest


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_declares_coverage_twice() -> None:
    """Never runs: collection refuses it."""
'''


def test_worker_collection_usage_error_is_shown_by_the_controller(
    *, tmp_path: Path
) -> None:
    """The duplicated-marker refusal appears in a two-worker run's output."""
    (tmp_path / "test_duplicated_marker.py").write_text(_SUITE, encoding="utf-8")
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(_REPOSITORY_ROOT), os.environ.get("PYTHONPATH", ""))
        ),
    }
    completed = subprocess.run(  # noqa: S603 - fixed pytest argv on a temporary file
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--full-suite",
            "--hardware-profile=cpu",
            str(tmp_path / "test_duplicated_marker.py"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert "@pytest.mark.coverage is declared more than once" in output, output
