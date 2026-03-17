"""Shared fixtures and configuration for PyLCM benchmarks."""

import contextlib
import hashlib
import platform
import subprocess

import jax
import pytest


def _machine_hash() -> str:
    """Stable hash from CPU model + JAX backend + device."""
    parts = [platform.processor(), jax.default_backend()]
    with contextlib.suppress(RuntimeError):
        parts.append(str(jax.devices()[0]))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def _is_worktree_dirty() -> bool:
    """Check if the git worktree has modified or untracked files."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def pytest_benchmark_update_machine_info(config, machine_info):  # noqa: ARG001
    machine_info["jax_backend"] = jax.default_backend()
    machine_info["machine_hash"] = _machine_hash()


def pytest_configure(config):
    machine = _machine_hash()
    config.option.benchmark_storage = f".benchmarks/{machine}"

    if getattr(config.option, "benchmark_autosave", False) and _is_worktree_dirty():
        msg = (
            "Refusing to save benchmarks: git worktree is dirty. "
            "Commit or stash your changes first."
        )
        raise pytest.UsageError(msg)
