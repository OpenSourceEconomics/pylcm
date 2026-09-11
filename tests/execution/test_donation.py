"""Static donation resolution: which candidate arguments a dispatch may donate."""

import dataclasses
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    ResolvedCoreProgram,
    ValueRead,
)
from _lcm.execution.donation import (
    DonatedBuffer,
    ResolvedDonation,
    resolve_donations,
)
from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
)
from lcm.solver_api import ArtifactKey

_KEY = ArtifactKey(type_id="tests.donation", schema_version=1)
_N_PERIODS = 4


def _leaf(*, period: int) -> ValueArtifactAddress:
    """The one continuation leaf every fixture in this module addresses."""
    return ValueArtifactAddress(
        kind=ValueArtifactKind.CONTINUATION_LEAF,
        period=period,
        regime="alive",
        artifact_key=_KEY,
        leaf_path=("count",),
    )


def _read(*, period: int) -> ValueRead:
    """The declared read of that leaf by the `count` argument of one dispatch."""
    return ValueRead(
        target=_leaf(period=period + 1),
        source=ValueConsumerAddress(
            source_period=period,
            source_regime="alive",
            core_key="main",
            channel=ValueInputChannel.CONTINUATION_LEAF,
            argument="count",
            path=(),
        ),
    )


def _function(*, count: jax.Array) -> jax.Array:
    """The fixture program: it reads its count argument and nothing else."""
    return count + 1.0


def _shardings(
    *, kind: ValueTransferKind
) -> tuple[jax.sharding.Sharding, jax.sharding.Sharding]:
    """Return stored and source layouts a transfer of `kind` classifies as."""
    source_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    if kind is ValueTransferKind.ALIGNED_LOCAL:
        return source_sharding, source_sharding
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("device",))
    return jax.sharding.NamedSharding(mesh=mesh, spec=jax.P()), source_sharding


def _program(
    *, period: int, kind: ValueTransferKind = ValueTransferKind.ALIGNED_LOCAL
) -> ResolvedCoreProgram:
    """A dense program declaring one read of the leaf, with `count` a candidate."""
    read = _read(period=period)
    count = jnp.zeros(3)
    stored_sharding, source_sharding = _shardings(kind=kind)
    return ResolvedCoreProgram(
        name="main",
        function=_function,
        arguments=MappingProxyType({"count": count}),
        static_kwargs=MappingProxyType({}),
        requirements=CoreExecutionRequirements(value_reads=(read,)),
        output_roles=None,
        disposition=CoreExecutionDisposition.DENSE,
        donation_candidates=("count",),
        tile_widths=MappingProxyType({}),
        specialization_key=("test",),
        input_transfer_plan=(
            ResolvedValueTransfer(
                target=read.target,
                source=read.source,
                kind=kind,
                stored_sharding=stored_sharding,
                source_sharding=source_sharding,
                expected_shape=count.shape,
                expected_dtype=count.dtype,
            ),
        ),
        disposition_reason="test",
    )


def _ledger(
    *,
    readers: int = 1,
    retained: bool = False,
    aliased: bool = False,
    pinned: bool = False,
) -> PlannedInputLiveness:
    """The plan the resolver reads: one reader unless a variant says otherwise."""
    accesses = {(2, "alive"): (_leaf(period=3),)}
    if readers == 2:
        accesses[(2, "other")] = (_leaf(period=3),)
    return PlannedInputLiveness(
        dispatch_accesses=accesses,
        pinned_artifacts=(_leaf(period=3),) if pinned else (),
        retained_artifacts=(_leaf(period=3),) if retained else (),
        aliases={_leaf(period=3): _leaf(period=4)} if aliased else {},
    )


def test_a_sole_consumer_of_an_aligned_unretained_leaf_donates_it() -> None:
    """The one reader of a fresh, unretained continuation leaf donates it."""
    donations = resolve_donations(
        program=_program(period=2),
        dispatch=(2, "alive"),
        ledger=_ledger(),
        n_periods=_N_PERIODS,
    )

    assert donations == (
        ResolvedDonation(
            argument="count",
            artifacts=(_leaf(period=3),),
            buffer=DonatedBuffer.STORED_ARTIFACT,
        ),
    )


@pytest.mark.parametrize(
    "ledger",
    [
        _ledger(readers=2),
        _ledger(retained=True),
        _ledger(aliased=True),
        _ledger(pinned=True),
    ],
    ids=["second reader", "retained", "aliased", "pinned"],
)
def test_a_shared_retained_aliased_or_pinned_artifact_is_not_donated(
    ledger: PlannedInputLiveness,
) -> None:
    """Every other reader, the result, an alias partner and a pin keep the buffer."""
    donations = resolve_donations(
        program=_program(period=2),
        dispatch=(2, "alive"),
        ledger=ledger,
        n_periods=_N_PERIODS,
    )

    assert not any(donation.donated for donation in donations)


def test_a_transferred_copy_is_recorded_but_not_donated() -> None:
    """Donating a copy frees nothing the ledger tracks, so the plan says so."""
    donations = resolve_donations(
        program=_program(period=2, kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT),
        dispatch=(2, "alive"),
        ledger=_ledger(),
        n_periods=_N_PERIODS,
    )

    assert [(donation.buffer, donation.donated) for donation in donations] == [
        (DonatedBuffer.TRANSFERRED_COPY, False)
    ]


def test_a_read_of_the_template_beyond_the_last_period_is_not_donated() -> None:
    """The last period reads the solve-lifetime template; nothing may delete it."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(_N_PERIODS - 1, "alive"): (_leaf(period=_N_PERIODS),)}
    )

    donations = resolve_donations(
        program=_program(period=_N_PERIODS - 1),
        dispatch=(_N_PERIODS - 1, "alive"),
        ledger=ledger,
        n_periods=_N_PERIODS,
    )

    assert not any(donation.donated for donation in donations)


def test_a_candidate_no_read_addresses_is_not_donated() -> None:
    """An argument the program reads outside its declarations is left alone."""
    program = dataclasses.replace(
        _program(period=2),
        requirements=CoreExecutionRequirements(),
        input_transfer_plan=(),
    )

    assert (
        resolve_donations(
            program=program,
            dispatch=(2, "alive"),
            ledger=_ledger(),
            n_periods=_N_PERIODS,
        )
        == ()
    )


def test_an_artifact_outside_the_plan_is_not_donated() -> None:
    """A read the ledger never registered is never handed to the executable."""
    ledger = PlannedInputLiveness(dispatch_accesses={(2, "alive"): ()})

    donations = resolve_donations(
        program=_program(period=2),
        dispatch=(2, "alive"),
        ledger=ledger,
        n_periods=_N_PERIODS,
    )

    assert not any(donation.donated for donation in donations)
