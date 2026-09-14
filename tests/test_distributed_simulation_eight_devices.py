"""A public three-device solve feeds eight-device subject simulation.

Run this file alone with JAX_NUM_CPU_DEVICES=8 and
XLA_FLAGS=--xla_force_host_platform_device_count=8. It never changes topology at
import, so an ordinary one/four-device battery skips the isolated witness.
"""

import jax
import numpy as np
import pandas as pd
import pytest

from _lcm.execution.value_transfer import ResolvedValueTransfer, ValueTransferKind
from _lcm.simulation import value_reads
from _lcm.solution.artifacts import OwnedSolutionView
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.execution import ExecutionConfig
from lcm.typing import ScalarInt

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or jax.device_count() != 8,
    reason="Requires an isolated eight-device CPU process",
)


@categorical(ordered=False)
class _Kind:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt


def _model(*, devices: tuple[int, ...]) -> Model:
    """Keep economic grids and state order identical in both placements."""
    return Model(
        regimes={
            "working": Regime(
                active=lambda age: age < 2,
                transition=lambda age: jax.numpy.where(
                    age < 1, _RegimeId.working, _RegimeId.retired
                ),
                states={"wealth": LinSpacedGrid(start=1, stop=40, n_points=4)},
                actions={"consumption": LinSpacedGrid(start=1, stop=3, n_points=3)},
                functions={
                    "utility": lambda wealth, consumption, kind: (
                        -((consumption - (kind + 1)) ** 2) + wealth * 0.125
                    )
                },
                state_transitions={
                    "wealth": lambda wealth, consumption: wealth - consumption
                },
            ),
            "retired": Regime(
                transition=None,
                states={"wealth": LinSpacedGrid(start=1, stop=40, n_points=4)},
                functions={"utility": lambda wealth, kind: wealth * (kind + 1)},
            ),
        },
        states={"kind": DiscreteGrid(_Kind)},
        state_transitions={"kind": fixed_transition("kind")},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(
            devices=devices,
            sharded_states=("kind",),
            axis_widths={"subject": 16},
        ),
    )


def _assert_subject_partition(*, value: jax.Array) -> None:
    """Check actual local shard indices, not just the named device set."""
    assert value.shape[0] == 16
    shards = value.addressable_shards
    assert len(shards) == 8
    assert {shard.device.id for shard in shards} == set(range(8))
    rows = []
    for shard in shards:
        indices = np.arange(16)[shard.index[0]]
        assert len(indices) == 2
        assert shard.data.shape[0] == len(indices)
        rows.extend(indices.tolist())
    assert sorted(rows) == list(range(16))


def test_three_device_solve_simulates_on_eight_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight subject shards consume addressed copies without deleting solve values."""
    model = _model(devices=tuple(range(8)))
    reference = _model(devices=(0,))
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    reference_solution = reference.solve(params=params, log_level="off")
    originals = {
        (period, regime): (value, np.asarray(value).copy())
        for period, values in solution.values.items()
        for regime, value in values.items()
    }
    engine_view = solution._engine_view
    assert isinstance(engine_view, OwnedSolutionView)
    engine_originals = {
        (period, regime): (value, np.asarray(value).copy())
        for period, values in engine_view.values.items()
        for regime, value in values.items()
    }
    for (period, regime), (value, snapshot) in originals.items():
        assert len(value.addressable_shards) == 3
        assert not value.is_fully_replicated
        assert {shard.data.shape[0] for shard in value.addressable_shards} == {1}
        np.testing.assert_allclose(
            snapshot,
            np.asarray(reference_solution.values[period][regime]),
            rtol=8 * np.finfo(snapshot.dtype).eps,
            atol=8 * np.finfo(snapshot.dtype).eps,
        )

    initial = {
        "wealth": np.arange(20, 36, dtype=float),
        "age": np.zeros(16),
        "kind": np.arange(16, dtype=np.int32) % 3,
        "regime_id": np.zeros(16, dtype=np.int32),
    }
    transfers: list[ResolvedValueTransfer] = []
    apply = value_reads.apply_value_transfer

    def observe(*, value: jax.Array, transfer: ResolvedValueTransfer) -> jax.Array:
        copied = apply(value=value, transfer=transfer)
        transfers.append(transfer)
        assert transfer.kind is ValueTransferKind.CROSS_MESH_COPY
        assert len(transfer.stored_sharding.device_set) == 3
        assert copied.sharding.device_set == set(jax.devices())
        assert copied.is_fully_replicated
        original, snapshot = originals[(transfer.target.period, transfer.target.regime)]
        # Public ValueStore leaves and the model-owned dispatch arrays need not
        # be the same wrapper. The addressed read must use its actual owner.
        engine_original, engine_snapshot = engine_originals[
            (transfer.target.period, transfer.target.regime)
        ]
        assert value is engine_original
        assert value.sharding == original.sharding
        np.testing.assert_array_equal(np.asarray(value), engine_snapshot)
        np.testing.assert_array_equal(engine_snapshot, snapshot)
        np.testing.assert_array_equal(np.asarray(copied), snapshot)
        return copied

    with monkeypatch.context() as probe:
        probe.setattr(value_reads, "apply_value_transfer", observe)
        result = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="off",
            seed=42,
        )
    expected = reference.simulate(
        params=params,
        initial_conditions=initial,
        solution=reference_solution,
        log_level="off",
        seed=42,
    )
    assert transfers
    copy_addresses = [
        (transfer.source.source_period, transfer.target) for transfer in transfers
    ]
    assert len(copy_addresses) == len(set(copy_addresses))
    for period in (0, 1):
        data = result.raw_results["working"][period]
        _assert_subject_partition(value=data.V_arr)
        _assert_subject_partition(value=data.actions["consumption"])
        np.testing.assert_array_equal(
            np.asarray(data.in_regime), np.ones(16, dtype=bool)
        )
    frame = result.to_dataframe(use_labels=False)
    expected_frame = expected.to_dataframe(use_labels=False)
    tolerance = 8 * np.finfo(next(iter(originals.values()))[1].dtype).eps
    pd.testing.assert_frame_equal(frame, expected_frame, rtol=tolerance, atol=tolerance)
    for original, snapshot in (*originals.values(), *engine_originals.values()):
        assert not original.is_deleted()
        np.testing.assert_array_equal(np.asarray(original), snapshot)
