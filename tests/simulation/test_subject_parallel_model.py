"""Public opt-in: unsharded solve states, eight forward devices, unchanged rows."""

# Seven rows per device at width three exercise full tiles and a padded remainder.
from typing import Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.simulation.runtime import SimulationRuntime
from _lcm.typing import PytreeValue
from lcm import ExecutionConfig, LinSpacedGrid
from tests.simulation._callback_types import RuntimeDispatch
from tests.test_models.deterministic.regression import (
    RegimeId,
    get_model,
    get_params,
)


@pytest.mark.parametrize("n_subjects", [1, 13, 56])
@pytest.mark.parametrize("budgeted", [False, True])
def test_public_subject_partition_matches_supplied_solution_reference(
    *, n_subjects: int, budgeted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if len(jax.devices()) < 8:
        pytest.skip("Select this test with eight visible devices.")
    ids = tuple(device.id for device in jax.devices()[:8])
    model_arguments = {
        "n_periods": 2,
        "wealth_grid": LinSpacedGrid(start=1, stop=3, n_points=3),
        "consumption_grid": LinSpacedGrid(start=1, stop=3, n_points=3),
    }
    reference = get_model(
        **model_arguments, execution_config=ExecutionConfig(devices=(ids[0],))
    )
    target = get_model(
        **model_arguments,
        execution_config=ExecutionConfig(
            devices=ids,
            sharded_states=(),
            simulation_sharding="subjects",
            device_memory_bytes=(1 << 28) if budgeted else None,
            axis_widths={"subject": 3} if budgeted else {},
        ),
    )
    params = get_params(n_periods=2)
    # A SolutionResult is bound to the Model that produced it, so each model
    # solves the identical unsharded problem and simulates with its own solution.
    reference_solution = reference.solve(params=params, log_level="off")
    solution = target.solve(params=params, log_level="off")
    initial = {
        "wealth": jnp.linspace(1.0, 3.0, n_subjects),
        "age": jnp.full(n_subjects, 18.0),
        "regime_id": jnp.full(n_subjects, RegimeId.working_life, dtype=jnp.int32),
    }
    expected = reference.simulate(
        params=params,
        solution=reference_solution,
        initial_conditions=initial,
        seed=17,
        log_level="off",
    )
    births = []
    dispatch = SimulationRuntime.dispatch

    def observe(
        self: SimulationRuntime, **kwargs: Unpack[RuntimeDispatch]
    ) -> PytreeValue:
        result = dispatch(self, **kwargs)
        if self.execution.simulation_sharding == "subjects":
            for leaf in jax.tree.leaves(result):
                if (
                    isinstance(leaf, jax.Array)
                    and leaf.ndim
                    and leaf.shape[0] == kwargs["n_subjects"]
                ):
                    births.append(
                        tuple(shard.data.shape[0] for shard in leaf.addressable_shards)
                    )
                    assert tuple(device.id for device in self.subject_devices) == ids
        return result

    monkeypatch.setattr(SimulationRuntime, "dispatch", observe)
    actual = target.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        seed=17,
        log_level="off",
    )
    assert actual.n_subjects == n_subjects
    assert births
    assert all(len(sizes) == 8 and len(set(sizes)) == 1 for sizes in births)
    _assert_panel_bitwise_equal(
        actual=actual.to_dataframe(), expected=expected.to_dataframe()
    )
    # Reuse the same published solution and runtime with the same seeded inputs.
    repeated = target.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        seed=17,
        log_level="off",
    )
    _assert_panel_bitwise_equal(
        actual=repeated.to_dataframe(), expected=actual.to_dataframe()
    )
    for leaf in jax.tree.leaves(initial):
        assert not leaf.is_deleted()
        assert np.asarray(leaf).shape == (n_subjects,)


def _assert_panel_bitwise_equal(
    *, actual: pd.DataFrame, expected: pd.DataFrame
) -> None:
    """Check schema and labels exactly, including floating storage bytes."""
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for name in actual.columns:
        left = actual[name].to_numpy()
        right = expected[name].to_numpy()
        if left.dtype.kind == "f":
            assert left.tobytes(order="C") == right.tobytes(order="C"), (
                f"Floating column {name!r} differs bitwise"
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mutation", ["one_ulp", "signed_zero"])
def test_panel_identity_comparator_rejects_float_bit_changes(
    *, dtype: type[np.float32 | np.float64], mutation: str
) -> None:
    """Panel identity rejects adjacent floats and opposite zero signs."""
    expected = pd.DataFrame({"value": np.array([0.0, 1.0], dtype=dtype)})
    actual = expected.copy()
    if mutation == "one_ulp":
        actual.loc[1, "value"] = np.nextafter(dtype(1.0), dtype(2.0))
    else:
        actual.loc[0, "value"] = dtype(-0.0)
    with pytest.raises(AssertionError):
        _assert_panel_bitwise_equal(actual=actual, expected=expected)
