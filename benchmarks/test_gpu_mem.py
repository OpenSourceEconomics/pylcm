"""Tests for the GPU peak-memory measurement harness."""

from benchmarks.asv._gpu_mem import _subprocess_env


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
