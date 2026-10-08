"""A stateless lookup is one row however many subject devices the model has.

An empty query stands for one row, and one row cannot be split across devices.
Each device count runs in its own subprocess with forced host devices, so the
witness does not depend on the ambient topology.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import jax
    import numpy as np

    from lcm import ExecutionConfig, LinSpacedGrid
    from tests.test_models.deterministic.regression import get_model, get_params

    n_devices = int(sys.argv[1])
    return_action_values = sys.argv[2] == "1"
    assert jax.device_count() == n_devices, jax.devices()

    grid = LinSpacedGrid(start=1, stop=3, n_points=3)
    model = get_model(
        n_periods=2,
        wealth_grid=grid,
        consumption_grid=grid,
        execution_config=ExecutionConfig(
            devices=tuple(device.id for device in jax.devices()),
            sharded_states=(),
            simulation_sharding="subjects",
            device_memory_bytes=2**30,
        ),
    )
    params = get_params(n_periods=2)
    solution = model.solve(params=params, log_level="off")
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=1,
        regime_name="dead",
        states={},
        return_action_values=return_action_values,
    )
    outputs = {"actions": sorted(got.actions), "value": np.asarray(got.value).tolist()}
    if return_action_values:
        outputs["Q"] = np.asarray(got.Q).tolist()
        outputs["F"] = np.asarray(got.F).tolist()
    print("LOOKUP-ROWS", json.dumps(outputs))
    """
)


def _lookup_on_devices(*, n_devices: int, return_action_values: bool) -> dict:
    env = {
        **os.environ,
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_devices}",
        "JAX_PLATFORMS": "cpu",
    }
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT, str(n_devices), str(int(return_action_values))],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    (line,) = (
        line for line in result.stdout.splitlines() if line.startswith("LOOKUP-ROWS ")
    )
    return json.loads(line.removeprefix("LOOKUP-ROWS "))


@pytest.mark.parametrize("n_devices", [1, 2, 8])
def test_budgeted_stateless_lookup_returns_one_row_of_value(*, n_devices: int) -> None:
    """The terminal `dead` regime's empty query is one row of zero on any devices."""
    got = _lookup_on_devices(n_devices=n_devices, return_action_values=False)
    assert got == {"actions": [], "value": [0.0]}


@pytest.mark.parametrize("n_devices", [1, 2, 8])
def test_budgeted_stateless_lookup_returns_one_row_of_action_values(
    *, n_devices: int
) -> None:
    """With action values requested, `Q` and `F` carry the same one row."""
    got = _lookup_on_devices(n_devices=n_devices, return_action_values=True)
    assert got == {"actions": [], "value": [0.0], "Q": [0.0], "F": [True]}
