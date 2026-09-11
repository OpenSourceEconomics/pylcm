"""Stored replay artifacts have typed identities separate from continuation values."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import ValueRead
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    resolve_value_transfer,
)
from _lcm.simulation.replay_inputs import place_replay_payload, replay_payload_reads
from _lcm.simulation.value_reads import PeriodSimulationReads
from lcm.solver_api import DISSOLUTION_FLAG, SIMULATION_POLICY


def _dissolution_read() -> ValueRead:
    return ValueRead(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REPLAY_ARTIFACT_LEAF,
            period=2,
            regime="couple",
            artifact_key=DISSOLUTION_FLAG,
        ),
        source=ValueConsumerAddress(
            source_period=1,
            source_regime="single",
            core_key="simulation_gate_fold",
            channel=ValueInputChannel.NEXT_REPLAY_ARTIFACT,
            argument="flags",
            path=("couple",),
        ),
    )


def _resolve(read: ValueRead) -> ResolvedValueTransfer:
    value = jnp.array([True, False])
    return resolve_value_transfer(
        target=read.target,
        source=read.source,
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_template=value,
        source_sharding=value.sharding,
    )


def test_dissolution_transfer_keeps_the_stored_boolean_dtype() -> None:
    transfer = _resolve(_dissolution_read())
    assert transfer.expected_dtype == jnp.dtype(bool)
    assert transfer.target.artifact_key == DISSOLUTION_FLAG
    assert transfer.target.leaf_path == ()


@pytest.mark.parametrize(
    "channel",
    [ValueInputChannel.NEXT_REGIME_VALUE, ValueInputChannel.CONTINUATION_LEAF],
)
def test_dissolution_cannot_masquerade_as_a_value_or_continuation(
    channel: ValueInputChannel,
) -> None:
    read = _dissolution_read()
    with pytest.raises(ValueError, match="replay-artifact channel"):
        _resolve(replace(read, source=replace(read.source, channel=channel)))


@pytest.mark.parametrize("period", [1, 3])
def test_dissolution_period_is_bound_to_the_declared_landing_period(
    period: int,
) -> None:
    read = _dissolution_read()
    with pytest.raises(ValueError, match="period must match"):
        _resolve(replace(read, target=replace(read.target, period=period)))


def test_replay_payload_preserves_nested_leaf_identity_and_container_structure() -> (
    None
):
    payload = {"branches": (jnp.array([1.0, 2.0]), {"valid": jnp.array([True, False])})}
    reads = replay_payload_reads(
        payload=payload, key=SIMULATION_POLICY, period=2, regime="couple", core="replay"
    )
    assert [read.source.path for read in reads] == [
        ("branches", 0),
        ("branches", 1, "valid"),
    ]
    assert len({read.target for read in reads}) == 2
    assert all(read.target.artifact_key == SIMULATION_POLICY for read in reads)
    owner = PeriodSimulationReads(
        period=2,
        devices=(jax.devices()[0],),
        reads_by_unit={"couple": reads},
        release_enabled=True,
    )
    placed = place_replay_payload(
        payload=payload,
        key=SIMULATION_POLICY,
        period=2,
        regime="couple",
        core="replay",
        owner=owner,
    )
    assert jax.tree.structure(placed) == jax.tree.structure(payload)
    for actual, original in zip(
        jax.tree.leaves(placed), jax.tree.leaves(payload), strict=True
    ):
        assert actual is original
    owner.commit(unit="couple", outputs=placed)
    owner.finish()
    assert all(not leaf.is_deleted() for leaf in jax.tree.leaves(payload))
