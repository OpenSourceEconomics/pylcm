"""A transfer several consumers of one period share is executed once."""

from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.execution.scheduler import BufferRegistry, PeriodTransferCache
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


def _stored_value(*, sharding: jax.sharding.Sharding) -> jax.Array:
    return jax.device_put(jnp.arange(3.0), sharding)


def _copy_transfer(
    *,
    reused: bool,
    stored: jax.Array,
    source_sharding: jax.sharding.Sharding,
) -> ResolvedValueTransfer:
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
        stored_sharding=stored.sharding,
        source_sharding=source_sharding,
        expected_shape=stored.shape,
        expected_dtype=stored.dtype,
        reused_by_several_consumers=reused,
    )


def _arguments(*, stored: jax.Array) -> MappingProxyType[str, object]:
    return MappingProxyType(
        {"next_regime_to_V_arr": MappingProxyType({"target": stored})}
    )


def _key(*, transfer: ResolvedValueTransfer) -> tuple[object, object]:
    return (transfer.target, transfer.source_sharding)


def test_a_shared_transfer_is_served_from_the_cache_on_its_second_use() -> None:
    """Two consumers of one period receive the identical copied array."""
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    cache = PeriodTransferCache(
        registry=BufferRegistry(),
        consumer_counts=MappingProxyType({_key(transfer=transfer): 2}),
    )
    arguments = _arguments(stored=stored)

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
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=False, stored=stored, source_sharding=source_sharding
    )
    cache = PeriodTransferCache(
        registry=BufferRegistry(), consumer_counts=MappingProxyType({})
    )

    apply_value_transfer_plan(
        arguments=_arguments(stored=stored), plan=(transfer,), cache=cache
    )

    assert len(cache) == 0


def test_an_unshared_transfer_is_not_registered_with_the_buffer_registry() -> None:
    """A transfer one consumer reads never enters the buffer registry."""
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=False, stored=stored, source_sharding=source_sharding
    )
    registry = BufferRegistry()
    cache = PeriodTransferCache(registry=registry, consumer_counts=MappingProxyType({}))

    result = apply_value_transfer_plan(
        arguments=_arguments(stored=stored), plan=(transfer,), cache=cache
    )

    target_values = result["next_regime_to_V_arr"]
    assert isinstance(target_values, Mapping)
    produced = target_values["target"]
    assert isinstance(produced, jax.Array)
    assert not registry.artifacts_sharing(array=produced)


def test_a_plan_without_a_cache_copies_as_before() -> None:
    """The cache is optional; a plan applied without one copies every time."""
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    arguments = _arguments(stored=stored)

    first = apply_value_transfer_plan(arguments=arguments, plan=(transfer,))
    second = apply_value_transfer_plan(arguments=arguments, plan=(transfer,))

    first_target_values = first["next_regime_to_V_arr"]
    second_target_values = second["next_regime_to_V_arr"]
    assert isinstance(first_target_values, Mapping)
    assert isinstance(second_target_values, Mapping)
    assert first_target_values["target"] is not second_target_values["target"]


def test_the_cache_keys_a_copy_by_artifact_and_required_layout() -> None:
    """One artifact copied onto one layout is served from a stored cache row."""
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    cache = PeriodTransferCache(
        registry=BufferRegistry(), consumer_counts=MappingProxyType({})
    )
    copy = jax.device_put(stored, source_sharding)
    cache.put(transfer=transfer, array=copy, stored=copy)

    assert cache.get(transfer=transfer) is not None


def test_the_cache_holds_one_row_per_artifact_and_layout() -> None:
    """One artifact copied onto one layout is exactly one cache row."""
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    cache = PeriodTransferCache(
        registry=BufferRegistry(), consumer_counts=MappingProxyType({})
    )
    copy = jax.device_put(stored, source_sharding)
    cache.put(transfer=transfer, array=copy, stored=copy)

    assert len(cache) == 1


def test_a_period_cache_does_not_serve_a_copy_made_in_another_periods_cache() -> None:
    """A fresh `PeriodTransferCache` is built each period; none carries state across."""
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    key = _key(transfer=transfer)
    period_t_cache = PeriodTransferCache(
        registry=BufferRegistry(), consumer_counts=MappingProxyType({key: 1})
    )
    apply_value_transfer_plan(
        arguments=_arguments(stored=stored), plan=(transfer,), cache=period_t_cache
    )

    period_t_minus_1_cache = PeriodTransferCache(
        registry=BufferRegistry(), consumer_counts=MappingProxyType({key: 1})
    )

    assert period_t_minus_1_cache.get(transfer=transfer) is None


def test_a_genuinely_new_copy_is_registered_with_the_buffer_registry() -> None:
    """A copy whose buffer differs from the stored artifact's enters the registry."""
    registry = BufferRegistry()
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    key = _key(transfer=transfer)
    cache = PeriodTransferCache(
        registry=registry, consumer_counts=MappingProxyType({key: 2})
    )
    copy = jax.device_put(np.arange(3.0), source_sharding)

    cache.put(transfer=transfer, array=copy, stored=stored)

    assert registry.artifacts_sharing(array=copy)


def test_a_genuinely_new_copy_is_alive_after_one_of_two_sources_commits() -> None:
    """A copy two sources share stays alive once only one of them has committed."""
    registry = BufferRegistry()
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    key = _key(transfer=transfer)
    cache = PeriodTransferCache(
        registry=registry, consumer_counts=MappingProxyType({key: 2})
    )
    copy = jax.device_put(np.arange(3.0), source_sharding)
    cache.put(transfer=transfer, array=copy, stored=stored)

    cache.commit_consumer(key=key)

    assert not copy.is_deleted()


def test_a_genuinely_new_copy_is_deleted_after_both_sources_commit() -> None:
    """A copy two sources share is released once every sharing source has committed."""
    registry = BufferRegistry()
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    key = _key(transfer=transfer)
    cache = PeriodTransferCache(
        registry=registry, consumer_counts=MappingProxyType({key: 2})
    )
    copy = jax.device_put(np.arange(3.0), source_sharding)
    cache.put(transfer=transfer, array=copy, stored=stored)
    cache.commit_consumer(key=key)

    cache.commit_consumer(key=key)

    assert copy.is_deleted()


def test_a_same_buffer_device_put_is_not_registered_with_the_buffer_registry() -> None:
    """A `device_put` returning the stored buffer unchanged registers no new copy.

    `transfer` is a genuine `COPY_TO_SOURCE_LAYOUT` (its two layouts classify as
    a copy, not as `ALIGNED_LOCAL`); what is exercised here is the physical
    degeneracy where the copy JAX actually returns is the stored buffer itself,
    named directly rather than reproduced via a device-topology-specific
    `device_put` call.
    """
    registry = BufferRegistry()
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    key = _key(transfer=transfer)
    cache = PeriodTransferCache(
        registry=registry, consumer_counts=MappingProxyType({key: 1})
    )

    cache.put(transfer=transfer, array=stored, stored=stored)

    assert not registry.artifacts_sharing(array=stored)


def test_a_same_buffer_device_put_survives_the_caches_commit() -> None:
    """A `device_put` returning the stored buffer unchanged is not deleted by the cache.

    That buffer is the stored artifact itself, released under its own artifact's
    lifetime rather than the transfer cache's.
    """
    registry = BufferRegistry()
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    key = _key(transfer=transfer)
    cache = PeriodTransferCache(
        registry=registry, consumer_counts=MappingProxyType({key: 1})
    )
    cache.put(transfer=transfer, array=stored, stored=stored)

    cache.commit_consumer(key=key)

    assert not stored.is_deleted()
