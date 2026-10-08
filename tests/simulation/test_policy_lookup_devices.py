"""A policy lookup gives the same rows however many devices the model has.

An empty query stands for one row, and one row cannot be split across devices.
A stateful query reads the next period's values wherever the solve left them,
and gives its rows however many of them each subject device holds.
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


_STATEFUL_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import jax
    import jax.numpy as jnp
    import numpy as np

    from tests.test_models.deterministic.regression import get_model, get_params

    from lcm import ExecutionConfig

    n_devices = int(sys.argv[1])
    return_action_values = sys.argv[2] == "1"
    execution = sys.argv[3]
    n_rows = int(sys.argv[4])
    assert jax.device_count() == n_devices, jax.devices()

    execution_config = {
        "default": None,
        "subjects": ExecutionConfig(
            devices=tuple(device.id for device in jax.devices()),
            sharded_states=(),
            simulation_sharding="subjects",
        ),
        "subjects-budgeted": ExecutionConfig(
            devices=tuple(device.id for device in jax.devices()),
            sharded_states=(),
            simulation_sharding="subjects",
            device_memory_bytes=2**30,
        ),
    }[execution]
    model = (
        get_model(n_periods=5)
        if execution_config is None
        else get_model(n_periods=5, execution_config=execution_config)
    )
    params = get_params(n_periods=5)
    solution = model.solve(params=params, log_level="off")
    wealth = model.state_grid(
        params=params, regime_name="working_life", state_name="wealth"
    )
    assert len(wealth) >= n_rows
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=3,
        regime_name="working_life",
        states={"wealth": jnp.asarray(wealth[:n_rows])},
        return_action_values=return_action_values,
    )
    outputs = {
        "actions": {
            name: np.asarray(values).tolist() for name, values in got.actions.items()
        },
        "value": np.asarray(got.value).tolist(),
    }
    if return_action_values:
        outputs["Q"] = np.asarray(got.Q).tolist()
        outputs["F"] = np.asarray(got.F).tolist()
    print("LOOKUP-ROWS", json.dumps(outputs))
    """
)


def _lookup_on_devices(
    *,
    n_devices: int,
    return_action_values: bool,
    script: str = _SCRIPT,
    extra_args: tuple[str, ...] = (),
) -> dict:
    env = {
        **os.environ,
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_devices}",
        "JAX_PLATFORMS": "cpu",
    }
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            script,
            str(n_devices),
            str(int(return_action_values)),
            *extra_args,
        ],
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


@pytest.mark.parametrize("return_action_values", [False, True])
@pytest.mark.parametrize("n_devices", [2, 8])
def test_stateful_lookup_on_several_devices_matches_one_device(
    *, n_devices: int, return_action_values: bool
) -> None:
    """A default-config model's stateful lookup gives the one-device rows."""
    got = _lookup_on_devices(
        n_devices=n_devices,
        return_action_values=return_action_values,
        script=_STATEFUL_SCRIPT,
        extra_args=("default", "2"),
    )
    expected = _lookup_on_devices(
        n_devices=1,
        return_action_values=return_action_values,
        script=_STATEFUL_SCRIPT,
        extra_args=("default", "2"),
    )
    assert got == expected


@pytest.mark.parametrize("return_action_values", [False, True])
@pytest.mark.parametrize("execution", ["subjects", "subjects-budgeted"])
@pytest.mark.parametrize("n_devices", [2, 8])
def test_subject_sharded_lookup_of_uneven_rows_matches_one_device(
    *, n_devices: int, execution: str, return_action_values: bool
) -> None:
    """Three rows split over the subject devices give the three one-device rows."""
    got = _lookup_on_devices(
        n_devices=n_devices,
        return_action_values=return_action_values,
        script=_STATEFUL_SCRIPT,
        extra_args=(execution, "3"),
    )
    expected = _lookup_on_devices(
        n_devices=1,
        return_action_values=return_action_values,
        script=_STATEFUL_SCRIPT,
        extra_args=("default", "3"),
    )
    assert got == expected
