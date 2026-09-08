"""Simulation value-copy ownership on actual ordered four-device CPU layouts."""

from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import ValueRead
from _lcm.execution.scheduler import shares_a_buffer
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.simulation.gated_routing import simulation_gate_fold, simulation_gate_route
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.period_inputs import (
    GATE_FOLD,
    GATE_ROUTE,
    acquire_gate_inputs,
    gate_reads,
)
from _lcm.simulation.replay_inputs import place_replay_payload, replay_payload_reads
from _lcm.simulation.value_reads import PeriodSimulationReads
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.typing import StatesPerRegime
from lcm import ExecutionConfig
from lcm.solver_api import SIMULATION_POLICY
from lcm.typing import Bool1D, Int1D

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _TOPOLOGY_UNAVAILABLE = False
except RuntimeError:
    _TOPOLOGY_UNAVAILABLE = True

# This fixture defines categoricals and arrays at import, after topology setup.
from benchmarks.asv._simulation_witnesses import dissolution

pytestmark = pytest.mark.skipif(
    _TOPOLOGY_UNAVAILABLE, reason="Four-device topology requires an isolated process"
)


def _read(*, unit: str) -> ValueRead:
    """Read one solved target value from the active regime's decision family."""
    return ValueRead(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="target"
        ),
        source=ValueConsumerAddress(
            source_period=0,
            source_regime=unit,
            core_key="argmax",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            path=("target",),
        ),
    )


@pytest.mark.parametrize("subject_ids", [(3,), (3, 1)])
def test_host_replay_has_no_allocation_on_an_excluded_default_device(
    *,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    subject_ids: tuple[int, ...],
) -> None:
    """A NumPy artifact moves directly to the requested ordered destination."""
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    original = np.array([2.0, 5.0, 9.0], dtype=dtype)
    reads = replay_payload_reads(
        payload=original, key=SIMULATION_POLICY, period=0, regime="alive", core="replay"
    )
    devices = tuple(jax.devices()[index] for index in subject_ids)
    owner = PeriodSimulationReads(
        period=0, devices=devices, reads_by_unit={"alive": reads}, release_enabled=True
    )
    allocations: list[tuple[int, ...]] = []
    actual_put = jax.device_put

    def observe(*args: Any, **kwargs: Any) -> jax.Array:
        placed = actual_put(*args, **kwargs)
        if isinstance(placed.sharding, jax.NamedSharding):
            allocations.append(
                tuple(device.id for device in placed.sharding.mesh.devices.flat)
            )
        else:
            allocations.append(tuple(device.id for device in placed.devices()))
        return placed

    monkeypatch.setattr(jax, "device_put", observe)
    placed = cast(
        "jax.Array",
        place_replay_payload(
            payload=original,
            key=SIMULATION_POLICY,
            period=0,
            regime="alive",
            core="replay",
            owner=owner,
        ),
    )
    request.node.user_properties.append(("allocation_stages", repr(allocations)))
    assert allocations == [subject_ids]
    np.testing.assert_array_equal(placed, [2.0, 5.0, 9.0])
    owner.commit(unit="alive", outputs=placed + 1)
    owner.finish()
    assert placed.is_deleted()
    np.testing.assert_array_equal(original, [2.0, 5.0, 9.0])


def test_repeated_host_replay_occurrences_reuse_the_same_addressed_copy() -> None:
    """Distinct core reads of one NumPy leaf share one period-owned conversion."""
    original = np.array([True, False, True])
    reads = tuple(
        read
        for core in ("replay", "transition")
        for read in replay_payload_reads(
            payload=original, key=SIMULATION_POLICY, period=0, regime="alive", core=core
        )
    )
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[3], jax.devices()[1]),
        reads_by_unit={"alive": reads},
        release_enabled=True,
    )
    copied = cast(
        "jax.Array",
        place_replay_payload(
            payload=original,
            key=SIMULATION_POLICY,
            period=0,
            regime="alive",
            core="replay",
            owner=owner,
        ),
    )
    again = place_replay_payload(
        payload=original,
        key=SIMULATION_POLICY,
        period=0,
        regime="alive",
        core="transition",
        owner=owner,
    )
    assert again is copied
    owner.commit(unit="alive", outputs=~copied)
    owner.finish()
    assert copied.is_deleted()
    np.testing.assert_array_equal(original, [True, False, True])


@pytest.mark.parametrize("subject_ids", [(0, 1, 2, 3), (3, 1)])
def test_a_physical_copy_uses_the_ordered_subject_devices_and_last_regime_lifetime(
    *, subject_ids: tuple[int, ...]
) -> None:
    """Full gathers and proper-submesh copies preserve originals and share once."""
    devices = jax.devices()
    stored_devices = devices if len(subject_ids) == 4 else [devices[0], devices[2]]
    mesh = jax.sharding.Mesh(np.asarray(stored_devices), ("stored",))
    stored = jax.device_put(jnp.arange(8), jax.NamedSharding(mesh, jax.P("stored")))
    first = _read(unit="first")
    second = _read(unit="second")
    subject_devices = tuple(devices[index] for index in subject_ids)
    owner = PeriodSimulationReads(
        period=0,
        devices=subject_devices,
        reads_by_unit={"first": (first,), "second": (second,)},
        release_enabled=True,
    )
    copied = owner.read(unit="first", read=first, value=stored)
    assert isinstance(copied.sharding, jax.NamedSharding)
    assert tuple(copied.sharding.mesh.devices.flat) == subject_devices
    assert not shares_a_buffer(first=copied, second=stored)
    np.testing.assert_array_equal(copied, np.arange(8))
    owner.commit(unit="first", outputs=copied + 1)
    assert not copied.is_deleted()
    assert owner.read(unit="second", read=second, value=stored) is copied
    owner.commit(unit="second", outputs=copied + 2)
    owner.finish()

    assert copied.is_deleted()
    np.testing.assert_array_equal(stored, np.arange(8))


def test_a_copy_that_reuses_an_original_shard_is_never_explicitly_deleted() -> None:
    """A wider replica can own fresh shards while still aliasing its stored source."""
    devices = jax.devices()
    stored = jax.device_put(
        jnp.arange(8), jax.sharding.SingleDeviceSharding(devices[1])
    )
    read = _read(unit="alive")
    owner = PeriodSimulationReads(
        period=0,
        devices=(devices[3], devices[1], devices[2], devices[0]),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
    )
    copied = owner.read(unit="alive", read=read, value=stored)
    assert copied is not stored
    assert len(copied.addressable_shards) == 4
    assert shares_a_buffer(first=copied, second=stored)
    owner.commit(unit="alive", outputs=copied + 1)
    owner.finish()

    np.testing.assert_array_equal(copied, np.arange(8))
    np.testing.assert_array_equal(stored, np.arange(8))


def test_a_real_dissolution_gate_uses_raw_copies_on_an_ordered_submesh() -> None:
    """The raw Boolean artifact reaches both numerical adapters before casting."""
    model, user_params, _ = dissolution(execution_config=ExecutionConfig(devices=(0,)))
    solution = model.solve(params=user_params, log_level="off")
    owned = solution._engine_view
    assert isinstance(owned, OwnedSolutionView)
    regimes = model._regimes
    ids = model.regime_names_to_ids
    params = model._process_params(user_params)
    solved_values, solved_flags = owned.values, owned.dissolution_flags
    target = "married_with_participation"
    devices = jax.devices()
    stored_sharding = jax.sharding.SingleDeviceSharding(devices[2])
    subject_devices = (devices[3], devices[1])
    values = {
        1: {
            name: jax.device_put(array, stored_sharding)
            for name, array in solved_values[1].items()
        }
    }
    flags = {
        1: {
            name: jax.device_put(array, stored_sharding)
            for name, array in solved_flags[1].items()
        }
    }
    married = regimes["married"]
    reads = gate_reads(
        regime=married, name="married", period=0, values=values, flags=flags
    )
    owner = PeriodSimulationReads(
        period=0,
        devices=subject_devices,
        reads_by_unit={"married": reads},
        release_enabled=True,
    )
    fold_values, fold_flags = acquire_gate_inputs(
        reads=tuple(read for read in reads if read.source.core_key == GATE_FOLD),
        owner=owner,
        name="married",
        values=values,
        flags=flags,
    )
    copied_d = fold_flags[target]
    assert copied_d.dtype == jnp.dtype(bool)
    assert isinstance(copied_d.sharding, jax.NamedSharding)
    assert tuple(copied_d.sharding.mesh.devices.flat) == subject_devices
    assert not shares_a_buffer(first=copied_d, second=flags[1][target])
    copied_values = tuple(fold_values[target].values())
    for reference, array in fold_values[target].items():
        assert isinstance(array.sharding, jax.NamedSharding)
        assert tuple(array.sharding.mesh.devices.flat) == subject_devices
        assert not shares_a_buffer(first=array, second=values[1][reference])
    folded = simulation_gate_fold(
        regime=married,
        regime_name="married",
        regimes=regimes,
        period=0,
        next_regime_to_V_arr=values[1],
        base_state_action_spaces={
            name: regime.solution.state_action_space(regime_params=params[name])
            for name, regime in regimes.items()
        },
        edge_values=fold_values,
        edge_flags=fold_flags,
        flat_params=params,
        subject_devices=subject_devices,
    )
    np.testing.assert_array_equal(
        folded[target], [[2, 1, 0], [-np.inf, -np.inf, 1], [6, 3, 0]]
    )
    folded_sharding = folded[target].sharding
    assert isinstance(folded_sharding, jax.NamedSharding)
    assert tuple(folded_sharding.mesh.devices.flat) == subject_devices
    route_values, route_flags = acquire_gate_inputs(
        reads=tuple(read for read in reads if read.source.core_key == GATE_ROUTE),
        owner=owner,
        name="married",
        values=values,
        flags=flags,
    )
    assert route_flags[target] is copied_d
    for reference, array in fold_values[target].items():
        assert route_values[target][reference] is array
    arguments = place_simulation_arguments(
        arguments={
            "next_states": MappingProxyType(
                {
                    target: MappingProxyType(
                        {"wage": jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])}
                    ),
                    "single_f": MappingProxyType({"wage": jnp.full(6, -999.0)}),
                    "single_m": MappingProxyType({"wage": jnp.full(6, -999.0)}),
                }
            ),
            "ids": jnp.full(6, ids[target], dtype=jnp.int32),
            "membership": jnp.ones(6, dtype=bool),
            "roles": jnp.full(
                6, married.stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        },
        subject_arg_names=("next_states", "ids", "membership", "roles"),
        value_reads=(),
        devices=subject_devices,
    )
    routed = simulation_gate_route(
        regime=married,
        fold_period=1,
        edge_values=route_values,
        edge_flags=route_flags,
        next_states=cast("StatesPerRegime", arguments["next_states"]),
        regime_names_to_ids=ids,
        new_subject_regime_ids=cast("Int1D", arguments["ids"]),
        subjects_in_regime=cast("Bool1D", arguments["membership"]),
        flat_params=params,
        own_stakeholder=cast("Int1D", arguments["roles"]),
        new_own_stakeholder=cast("Int1D", arguments["roles"]),
        subject_devices=subject_devices,
    )
    states, routed_ids, _ = routed
    np.testing.assert_array_equal(
        routed_ids, [ids[target], ids["single_f"], ids[target]] * 2
    )
    np.testing.assert_array_equal(states["single_f"]["wage"], [-999.0, 2.0, -999.0] * 2)
    owner.commit(unit="married", outputs=(folded, routed))
    owner.finish()
    assert copied_d.is_deleted()
    assert all(array.is_deleted() for array in copied_values)
    np.testing.assert_array_equal(flags[1][target], [False, True, False])
    for reference, array in values[1].items():
        assert not array.is_deleted()
        np.testing.assert_array_equal(array, solved_values[1][reference])
