"""Shared fixtures and configuration for PyLCM benchmarks."""

import contextlib
import hashlib
import platform
import subprocess

import jax
import jax.numpy as jnp
import pytest

from lcm_examples import precautionary_savings


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


@pytest.fixture
def precautionary_model_factory():
    """Return a factory that builds precautionary savings models."""

    def _make(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_grid_type="lin",
        wealth_n_points=7,
        consumption_n_points=7,
    ):
        model = precautionary_savings.get_model(
            n_periods=n_periods,
            shock_type=shock_type,
            wealth_grid_type=wealth_grid_type,
            wealth_n_points=wealth_n_points,
            consumption_n_points=consumption_n_points,
        )
        params = precautionary_savings.get_params(
            shock_type=shock_type,
            sigma=0.2,
            rho=0.9,
        )
        return model, params

    return _make


@pytest.fixture
def precautionary_solved():
    """Return a solved precautionary savings model for simulation benchmarks."""
    model = precautionary_savings.get_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_n_points=10,
        consumption_n_points=10,
    )
    params = precautionary_savings.get_params(
        shock_type="rouwenhorst",
        sigma=0.2,
        rho=0.9,
    )
    V_arr_dict = model.solve(params, log_level="off")
    return model, params, V_arr_dict


@pytest.fixture
def initial_conditions_factory():
    """Return a factory that builds initial conditions for simulation."""

    def _make(n_subjects, regime_id=0, age=20.0, wealth=5.0, income=0.0):
        return {
            "age": jnp.full(n_subjects, age),
            "wealth": jnp.full(n_subjects, wealth),
            "income": jnp.full(n_subjects, income),
            "regime_id": jnp.full(n_subjects, regime_id, dtype=jnp.int32),
        }

    return _make
