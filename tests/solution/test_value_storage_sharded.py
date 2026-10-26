"""Host value storage of a sharded solve, on two CPU devices.

The values of a regime sharded over its devices are copied to pinned host memory
on the same sharding, each device copying its own slice, and read back onto that
sharding. Values and the simulated panel are those of device storage, bit for
bit. Two host devices need `XLA_FLAGS` set before JAX starts, so the check runs
in a fresh interpreter.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from _lcm.solution import backward_induction
    from tests.test_distributed_simulation_values import _build_model, _RegimeId

    assert len(jax.devices()) >= 2, jax.devices()
    params = {"discount_factor": 0.95}
    initial_conditions = {
        "wealth": jnp.full(6, 12.0),
        "age": jnp.zeros(6),
        "preference": jnp.arange(6, dtype=jnp.int32) % 4,
        "regime_id": jnp.full(6, _RegimeId.working),
    }

    stored_shardings = []
    copy_values_to_host = backward_induction.copy_values_to_host

    def _copy(*, values):
        copies = copy_values_to_host(values=values)
        stored_shardings.extend(value.sharding for value in copies.values())
        return copies

    backward_induction.copy_values_to_host = _copy

    solutions, panels = [], []
    for storage in ("device", "host"):
        model = _build_model(
            n_types=4, devices=(0, 1), sharded=True, value_storage=storage
        )
        solutions.append(model.solve(params=params, log_level="off"))
        panels.append(
            model.simulate(
                params=params,
                initial_conditions=initial_conditions,
                seed=0,
                log_level="off",
            ).to_dataframe()
        )

    assert stored_shardings
    assert all(s.memory_kind == "pinned_host" for s in stored_shardings)
    assert any(len(s.device_set) == 2 for s in stored_shardings), stored_shardings
    for period, regime_to_value in solutions[0].values.items():
        for regime_name, value in regime_to_value.items():
            read = solutions[1].value(period=period, regime=regime_name)
            assert read.sharding.memory_kind == "device"
            assert read.sharding.is_equivalent_to(value.sharding, value.ndim)
            np.testing.assert_array_equal(
                np.asarray(read), np.asarray(value), strict=True
            )
    assert panels[0].equals(panels[1])
    print("SHARDED-HOST-STORAGE-OK")
    """
)


def test_host_storage_of_a_sharded_solve_equals_device_storage() -> None:
    """Sharded values keep their sharding in host memory and read back exactly."""
    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "JAX_PLATFORMS": "cpu",
    }
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "SHARDED-HOST-STORAGE-OK" in result.stdout
