"""Abstract core preparation preserves declared transfer authority without copies."""

from collections.abc import Mapping
from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    MaterializedCoreProgram,
    ValueRead,
    resolve_core_program,
)
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    resolve_value_transfer,
)
from lcm.exceptions import ExecutionPlanningError


def _read_value(
    *, next_regime_to_V_arr: Mapping[str, jax.Array], extra: jax.Array
) -> jax.Array:
    return next_regime_to_V_arr["future"] + extra


def _inputs() -> tuple[MaterializedCoreProgram, ResolvedValueTransfer]:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    value = jax.ShapeDtypeStruct((3,), jnp.float32, sharding=sharding)
    target = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="future"
    )
    source = ValueConsumerAddress(
        channel=ValueInputChannel.NEXT_REGIME_VALUE,
        path=("future",),
        source_period=0,
        source_regime="current",
        core_key="main",
    )
    transfer = resolve_value_transfer(
        target=target,
        source=source,
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_template=value,
        source_sharding=sharding,
    )
    program = MaterializedCoreProgram(
        name="main",
        function=_read_value,
        arguments={"next_regime_to_V_arr": {"future": value}, "extra": value},
        requirements=CoreExecutionRequirements(
            value_reads=(ValueRead(target=target, source=source),)
        ),
        output_roles="value",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )
    return program, transfer


def test_abstract_resolution_keeps_transfer_metadata_without_device_put(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An abstract read retains its original exact plan and cost metadata."""
    program, transfer = _inputs()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Abstract resolution attempted a physical transfer")

    monkeypatch.setattr(jax, "device_put", forbidden)
    resolved = resolve_core_program(
        program=program, input_transfer_plan=(transfer,), abstract_inputs=True
    )
    assert resolved.input_transfer_plan == (transfer,)
    assert resolved.input_transfer_plan[0].cost == transfer.cost
    assert resolved.arguments["extra"] is program.arguments["extra"]


@pytest.mark.parametrize("use_required_layout", [False, True])
def test_abstract_copy_uses_destination_layout_and_keeps_source_cost(
    *, monkeypatch: pytest.MonkeyPatch, use_required_layout: bool
) -> None:
    """A copied descriptor names its destination and retains the source plan."""
    program, aligned = _inputs()
    mesh = jax.make_mesh((1,), ("destination",), devices=jax.devices()[:1])
    required = jax.NamedSharding(mesh, jax.P())
    transfer = resolve_value_transfer(
        target=aligned.target,
        source=aligned.source,
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
        stored_template=program.arguments["extra"],
        source_sharding=required,
    )
    descriptor = jax.ShapeDtypeStruct(
        (3,),
        jnp.float32,
        sharding=required if use_required_layout else aligned.stored_sharding,
    )
    candidate = replace(
        program,
        arguments={
            "next_regime_to_V_arr": {"future": descriptor},
            "extra": program.arguments["extra"],
        },
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Abstract copy preparation executed device_put")

    monkeypatch.setattr(jax, "device_put", forbidden)
    if not use_required_layout:
        with pytest.raises(ValueError, match="required-sharding"):
            resolve_core_program(
                program=candidate,
                input_transfer_plan=(transfer,),
                abstract_inputs=True,
            )
        return
    resolved = resolve_core_program(
        program=candidate, input_transfer_plan=(transfer,), abstract_inputs=True
    )
    assert resolved.input_transfer_plan[0] is transfer
    assert transfer.stored_sharding != required
    assert transfer.source_sharding == required
    assert transfer.cost.logical_bytes == 12
    assert transfer.cost.per_device_bytes == 12
    assert transfer.cost.temporary_bytes == 12


@pytest.mark.parametrize("argument", ["read", "extra"])
def test_abstract_resolution_rejects_concrete_arguments_everywhere(
    argument: str,
) -> None:
    """An unrelated dynamic argument cannot allocate outside abstract preparation."""
    program, transfer = _inputs()
    concrete = jnp.zeros(3, dtype=jnp.float32)
    replacement = dict(program.arguments)
    replacement["next_regime_to_V_arr" if argument == "read" else "extra"] = (
        {"future": concrete} if argument == "read" else concrete
    )
    with pytest.raises(ExecutionPlanningError, match=r"abstract|Abstract"):
        resolve_core_program(
            program=replace(program, arguments=replacement),
            input_transfer_plan=(transfer,),
            abstract_inputs=True,
        )


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "sharding", "missing_sharding"])
def test_abstract_read_requires_exact_destination_metadata(mismatch: str) -> None:
    """Shape, dtype and required placement remain authority at compile time."""
    program, transfer = _inputs()
    sharding = transfer.source_sharding
    shape, dtype = (3,), jnp.float32
    if mismatch == "shape":
        shape = (4,)
    elif mismatch == "dtype":
        dtype = jnp.int32
    elif mismatch == "sharding":
        mesh = jax.make_mesh((1,), ("test",), devices=jax.devices()[:1])
        sharding = jax.NamedSharding(mesh, jax.P())
    else:
        sharding = None
    wrong = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    with pytest.raises(
        (ValueError, TypeError, ExecutionPlanningError),
        match=r"shape|dtype|sharding|abstract|Abstract",
    ):
        resolve_core_program(
            program=replace(
                program,
                arguments={
                    "next_regime_to_V_arr": {"future": wrong},
                    "extra": program.arguments["extra"],
                },
            ),
            input_transfer_plan=(transfer,),
            abstract_inputs=True,
        )


def test_concrete_resolution_still_requires_concrete_transfer_inputs() -> None:
    """The default dispatch path never accepts a shape descriptor as an array."""
    program, transfer = _inputs()
    with pytest.raises(TypeError, match="concrete JAX array"):
        resolve_core_program(program=program, input_transfer_plan=(transfer,))
