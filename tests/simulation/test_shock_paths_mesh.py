"""Given paths reach every subject on an eight-device subject mesh.

The cases skip unless the process has eight CPU host devices before JAX starts;
CI runs this file in its own fresh eight-device process.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from lcm import ExecutionConfig, Model
from lcm.shocks import BaseInnovations, PhysicalValues
from tests.simulation.test_current_iid_overrides import _model as current_iid_model
from tests.simulation.test_current_iid_overrides import _Regimes as CurrentIIDRegimes
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


@pytest.mark.parametrize("physical", [False, True])
def test_tagged_current_paths_keep_original_rows_on_subject_mesh(
    *, physical: bool
) -> None:
    """A budgeted subject mesh preserves tagged paths through late entry and padding."""
    devices = tuple(range(8))
    if jax.default_backend() != "cpu" or len(jax.devices()) < len(devices):
        pytest.skip("Select this control in a fresh eight-device CPU process.")
    starts = np.asarray([2, 0, 1, 0, 2, 1, 0, 1, 0], dtype=np.int32)
    paths = np.arange(starts.size * 3, dtype=jnp.asarray(0.0).dtype).reshape(-1, 3) + 2
    paths[1::2] *= -1
    assert starts.size % len(devices) != 0
    assert np.unique(paths).size == paths.size
    initial = {"period": starts, "regime_id": starts}
    declaration = current_iid_model()
    frames = []
    for sharded in (False, True):
        model = Model(
            regimes=declaration.user_regimes,
            shocks=declaration.shocks,
            n_periods=3,
            initial_nodes=declaration.initial_nodes,
            edges=declaration.edges,
            regime_id_class=CurrentIIDRegimes,
            execution_config=ExecutionConfig(
                devices=devices if sharded else (0,),
                simulation_sharding="subjects" if sharded else "legacy",
                axis_widths={"subject": 3},
                device_memory_bytes=100_000_000,
            ),
        )
        result = model.simulate(
            params={"discount_factor": 0.5, "risk": {"eps__sigma": 2.0}},
            initial_conditions=initial,
            shock_paths={
                "eps": PhysicalValues(values=paths)
                if physical
                else BaseInnovations(values=paths)
            },
            seed=91,
            log_level="off",
            runtime_checks=False,
        )
        if sharded:
            summary = result.plan_summary
            assert summary is not None
            assert (
                summary.route,
                summary.subject_device_ids,
                summary.subject_backend,
            ) == (
                "subjects",
                devices,
                "cpu",
            )
            assert summary.budget_mode == "budgeted"
            assert summary.effective_device_memory_bytes is not None
            assert 0 < summary.effective_device_memory_bytes <= 100_000_000
            assert sum(summary.admitted_chunk_widths) > starts.size
            assert all(
                width % len(devices) == 0 for width in summary.admitted_chunk_widths
            )
        frame = (
            result.to_dataframe(use_categorical_codes=True)
            .sort_values(["subject_id", "period"])
            .reset_index(drop=True)
        )
        pairs = np.asarray(
            [
                (row, period)
                for row, start in enumerate(starts)
                for period in range(start, 3)
            ]
        )
        np.testing.assert_array_equal(frame[["subject_id", "period"]], pairs)
        expected = paths if physical else 1 + 2 * paths
        values = expected[frame["subject_id"], frame["period"]]
        np.testing.assert_array_equal(frame["eps"], values)
        np.testing.assert_array_equal(frame["action"], values > 0)
        frames.append(frame)
    assert_frame_equal(frames[1], frames[0], check_exact=True)
