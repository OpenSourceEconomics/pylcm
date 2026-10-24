"""Given paths reach every subject on an eight-device subject mesh.

The cases skip unless the process has eight CPU host devices before JAX starts;
CI runs this file in its own fresh eight-device process.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import ExecutionConfig
from tests.simulation.test_shock_paths import RegimeId, _model

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@pytest.mark.parametrize("budget", [None, 100_000_000])
def test_supplied_paths_on_subject_mesh(*, budget: int | None) -> None:
    devices = tuple(range(8))
    if len(jax.devices()) < len(devices):
        pytest.skip("Select this control in a fresh eight-device CPU process.")
    paths = np.tile(np.array([1.0, -1.0, 2.0]), (9, 1))
    paths[1::2] *= -1
    initial = {
        "age": jnp.array([20, 21, 20, 21, 20, 21, 20, 21, 20]),
        "shock": jnp.zeros(9),
        "regime_id": jnp.full(9, RegimeId.alive),
    }
    model = _model(
        execution=ExecutionConfig(
            devices=devices,
            simulation_sharding="subjects",
            axis_widths={"subject": 3},
            device_memory_bytes=budget,
        )
    )
    result = model.simulate(
        params={"discount_factor": 0.9},
        initial_conditions=initial,
        shock_paths={"shock": paths},
        seed=0,
        log_level="off",
    )
    summary = result.plan_summary
    assert summary is not None
    assert (summary.route, summary.subject_device_ids, summary.subject_backend) == (
        "subjects",
        devices,
        "cpu",
    )
    rows = (
        result.to_dataframe(use_categorical_codes=True)
        .reset_index()
        .query("regime_name == 'alive'")
    )
    expected = paths[rows.subject_id.to_numpy(), rows.period.to_numpy()]
    np.testing.assert_array_equal(rows.shock, expected)
    np.testing.assert_array_equal(rows.choice, expected)
