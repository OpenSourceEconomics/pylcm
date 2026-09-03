"""Tests for the GPU peak-memory measurement harness."""

import subprocess
import sys

from benchmarks.asv._gpu_mem import _PROJECT_ROOT, _subprocess_env

# Blocks the `resource` module before the harness is imported, so the fresh
# interpreter sees exactly what a host without that module (Windows) sees.
_WITHOUT_RESOURCE_PREAMBLE = "import sys; sys.modules['resource'] = None\n"


def _run_without_resource_module(*, code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter that cannot import `resource`."""
    return subprocess.run(
        [sys.executable, "-c", _WITHOUT_RESOURCE_PREAMBLE + code],
        capture_output=True,
        text=True,
        check=False,
        cwd=_PROJECT_ROOT,
    )


def test_subprocess_env_disables_autotuning():
    """GPU-mem subprocess disables XLA autotuning for a deterministic compile."""
    env = _subprocess_env({"PATH": "/usr/bin"})
    assert "--xla_gpu_autotune_level=0" in env["XLA_FLAGS"]


def test_subprocess_env_appends_to_existing_xla_flags():
    """Existing `XLA_FLAGS` are kept; the autotune flag is appended, not clobbered."""
    env = _subprocess_env({"XLA_FLAGS": "--xla_gpu_foo=1"})
    assert "--xla_gpu_foo=1" in env["XLA_FLAGS"]
    assert "--xla_gpu_autotune_level=0" in env["XLA_FLAGS"]


def test_subprocess_env_drops_mem_fraction():
    """The subprocess gets full GPU memory: the MEM_FRACTION cap is removed."""
    env = _subprocess_env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3"})
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env


def test_gpu_mem_imports_on_a_host_without_resource_module():
    """The harness imports on hosts whose Python has no `resource` module."""
    result = _run_without_resource_module(code="import benchmarks.asv._gpu_mem\n")
    assert result.returncode == 0, result.stderr


def test_cpu_peak_bytes_is_nan_on_a_host_without_resource_module():
    """Without `resource`, the host peak is reported as NaN — unknown, not zero."""
    result = _run_without_resource_module(
        code=(
            "from benchmarks.asv._gpu_mem import _get_cpu_peak_bytes\n"
            "print(repr(_get_cpu_peak_bytes()))\n"
        )
    )
    assert result.stdout.strip() == "nan", result.stderr
