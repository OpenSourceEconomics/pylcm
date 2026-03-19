"""pytest-benchmark hooks that route results to machine-specific directories.

Guards against benchmarking uncommitted code so that every saved result is
tied to a specific commit SHA.
"""

import contextlib
import hashlib
import platform
import subprocess

import jax
import pytest


def _machine_hash() -> str:
    """Return a stable short hash identifying this machine and JAX backend.

    Results from different machines or backends must not mix — the hash
    partitions storage directories and the published dashboard.
    """
    parts = [platform.processor(), jax.default_backend()]
    with contextlib.suppress(RuntimeError):
        parts.append(str(jax.devices()[0]))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def _is_worktree_dirty() -> bool:
    """Return True if the git worktree has modified or untracked files.

    Benchmarks are tagged with the current commit SHA; uncommitted changes
    would make results misleading because the SHA would not reflect the
    actual code that was benchmarked.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def pytest_benchmark_update_machine_info(config, machine_info):  # noqa: ARG001
    """Add JAX backend and machine hash to benchmark JSON metadata.

    This ensures benchmark result files record what hardware and backend
    produced the numbers.
    """
    machine_info["jax_backend"] = jax.default_backend()
    machine_info["machine_hash"] = _machine_hash()


def pytest_configure(config):
    """Wire up machine-specific storage and enforce the dirty-worktree guard.

    Called at pytest startup. Sets the benchmark storage path to
    `.benchmarks/{machine_hash}/` and raises `UsageError` if the user
    attempts to autosave results while the worktree is dirty.
    """
    machine = _machine_hash()
    config.option.benchmark_storage = f".benchmarks/{machine}"

    if getattr(config.option, "benchmark_autosave", False) and _is_worktree_dirty():
        msg = (
            "Refusing to save benchmarks: git worktree is dirty. "
            "Commit or stash your changes first."
        )
        raise pytest.UsageError(msg)
