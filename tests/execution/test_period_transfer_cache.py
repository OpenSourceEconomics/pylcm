"""A transfer several consumers of one period share is executed once."""

from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.execution.scheduler import PeriodTransferCache
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    apply_value_transfer_plan,
)


def _shardings() -> tuple[jax.sharding.Sharding, jax.sharding.Sharding]:
    # A `NamedSharding` stored layout copied onto a `SingleDeviceSharding` source
    # layout classifies as `COPY_TO_SOURCE_LAYOUT` even on one device, since the
    # catalogue keys on the layout *kind*, not the device count; two identical
    # `SingleDeviceSharding` instances would instead classify as `ALIGNED_LOCAL`.
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("device",))
    stored_sharding = jax.sharding.NamedSharding(mesh=mesh, spec=jax.P())
    source_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    return stored_sharding, source_sharding


def _copy_transfer(*, reused: bool) -> ResolvedValueTransfer:
    stored_sharding, source_sharding = _shardings()
    return ResolvedValueTransfer(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="target"
        ),
        source=ValueConsumerAddress(
            source_period=0,
            source_regime="source",
            core_key="main",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            path=("target",),
        ),
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
        stored_sharding=stored_sharding,
        source_sharding=source_sharding,
        expected_shape=(3,),
        expected_dtype=jnp.float64,
        reused_by_several_consumers=reused,
    )


def _arguments() -> MappingProxyType[str, object]:
    stored_sharding, _source_sharding = _shardings()
    stored_value = jax.device_put(jnp.arange(3.0), stored_sharding)
    return MappingProxyType(
        {"next_regime_to_V_arr": MappingProxyType({"target": stored_value})}
    )


def test_a_shared_transfer_is_served_from_the_cache_on_its_second_use() -> None:
    """Two consumers of one period receive the identical copied array."""
    cache = PeriodTransferCache()
    transfer = _copy_transfer(reused=True)
    arguments = _arguments()

    first = apply_value_transfer_plan(
        arguments=arguments, plan=(transfer,), cache=cache
    )
    second = apply_value_transfer_plan(
        arguments=arguments, plan=(transfer,), cache=cache
    )

    first_target_values = first["next_regime_to_V_arr"]
    second_target_values = second["next_regime_to_V_arr"]
    assert isinstance(first_target_values, Mapping)
    assert isinstance(second_target_values, Mapping)
    assert first_target_values["target"] is second_target_values["target"]


def test_an_unshared_transfer_is_not_cached() -> None:
    """A transfer one consumer reads is copied per dispatch and holds no cache row."""
    cache = PeriodTransferCache()
    transfer = _copy_transfer(reused=False)
    arguments = _arguments()

    apply_value_transfer_plan(arguments=arguments, plan=(transfer,), cache=cache)

    assert len(cache) == 0


def test_a_plan_without_a_cache_copies_as_before() -> None:
    """The cache is optional; a plan applied without one copies every time."""
    transfer = _copy_transfer(reused=True)
    arguments = _arguments()

    first = apply_value_transfer_plan(arguments=arguments, plan=(transfer,))
    second = apply_value_transfer_plan(arguments=arguments, plan=(transfer,))

    first_target_values = first["next_regime_to_V_arr"]
    second_target_values = second["next_regime_to_V_arr"]
    assert isinstance(first_target_values, Mapping)
    assert isinstance(second_target_values, Mapping)
    assert first_target_values["target"] is not second_target_values["target"]


def test_the_cache_keys_a_copy_by_artifact_and_required_layout() -> None:
    """One artifact copied onto one layout is one cache row."""
    cache = PeriodTransferCache()
    transfer = _copy_transfer(reused=True)
    cache.put(transfer=transfer, array=jnp.arange(3.0))

    assert cache.get(transfer=transfer) is not None and len(cache) == 1  # noqa: PT018
