"""A transfer several consumers of one period share is executed once."""

import logging
from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution import scheduler
from _lcm.execution.output_layout import VALUE, PlannedCore, resolve_output_layout
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
from _lcm.solution.backward_induction import _period_shared_transfer_plan
from lcm.exceptions import ExecutionPlanningError


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


def _produced_target_leaf(*, result: Mapping[str, object]) -> jax.Array:
    """Return a transfer-plan result's produced `target` leaf, type-narrowed."""
    next_regime_to_V_arr = result["next_regime_to_V_arr"]
    assert isinstance(next_regime_to_V_arr, Mapping)
    produced = next_regime_to_V_arr["target"]
    assert isinstance(produced, jax.Array)
    return produced


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

    assert not registry.artifacts_sharing(array=_produced_target_leaf(result=result))


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


def test_a_not_produced_registered_copy_survives_its_consumers_commit() -> None:
    """A registered copy declared as a buffer no dispatch produced is never released.

    `commit_consumer` routes a genuinely registered copy through
    `release_closed_artifacts`, which refuses to delete a buffer any of whose
    shards the registry marks as one no dispatch produced — the guard a
    manual delete-on-zero-count implementation could skip.
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
    copy = jax.device_put(np.arange(3.0), source_sharding)
    cache.put(transfer=transfer, array=copy, stored=stored)
    registry.declare_not_produced(tree=(copy,))

    cache.commit_consumer(key=key)

    assert not copy.is_deleted()


def _sole_blocked_on_argument(*, blocked_on: list[object]) -> object:
    """Return the single argument a barrier spy recorded, type-narrowed to a tuple."""
    assert len(blocked_on) == 1
    (recorded,) = blocked_on
    assert isinstance(recorded, tuple)
    return recorded


def test_commit_consumer_blocks_on_the_periods_pending_outputs_not_the_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Releasing a registered copy blocks on the period's pending outputs.

    A manual `jax.block_until_ready(array)` on the copy itself, instead of on
    the outputs the period has dispatched so far, is exactly the barrier
    `release_closed_artifacts` avoids: with no `pending_outputs` declared
    (the default, empty tuple), the barrier call observed here carries that
    empty tuple, never the copy.
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
    copy = jax.device_put(np.arange(3.0), source_sharding)
    cache.put(transfer=transfer, array=copy, stored=stored)
    blocked_on: list[object] = []
    monkeypatch.setattr(
        scheduler.jax, "block_until_ready", blocked_on.append, raising=True
    )

    cache.commit_consumer(key=key)

    assert _sole_blocked_on_argument(blocked_on=blocked_on) == ()


def test_commit_consumer_logs_a_release_record_for_a_registered_copy() -> None:
    """Releasing a registered copy is logged exactly like any other closed artifact."""
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
    copy = jax.device_put(np.arange(3.0), source_sharding)
    cache.put(transfer=transfer, array=copy, stored=stored)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # ty: ignore[invalid-assignment]
    release_logger = logging.getLogger(scheduler.__name__)
    release_logger.setLevel(logging.DEBUG)
    release_logger.addHandler(handler)
    try:
        cache.commit_consumer(key=key)
    finally:
        release_logger.removeHandler(handler)

    assert records


def _released_artifacts(*, records: object) -> tuple[object, ...]:
    """Return the artifacts a `commit_consumer` result named, type-narrowed."""
    assert isinstance(records, tuple)
    return tuple(record.artifact for record in records)


def test_commit_consumer_returns_a_release_record_naming_the_released_key() -> None:
    """The return value names the artifact a registered copy's release closed."""
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
    copy = jax.device_put(np.arange(3.0), source_sharding)
    cache.put(transfer=transfer, array=copy, stored=stored)

    records = cache.commit_consumer(key=key)

    assert _released_artifacts(records=records) == (key,)


def test_committing_more_consumers_than_the_period_declared_names_the_key() -> None:
    """One commit past the declared count is a planning error naming the key."""
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    key = _key(transfer=transfer)
    cache = PeriodTransferCache(
        registry=BufferRegistry(), consumer_counts=MappingProxyType({key: 1})
    )
    cache.commit_consumer(key=key)

    with pytest.raises(ExecutionPlanningError, match="declared consumer"):
        cache.commit_consumer(key=key)


def _planned_core(*, name: str, transfer: ResolvedValueTransfer) -> PlannedCore:
    """A core carrying one resolved input transfer and nothing else."""
    return PlannedCore(
        compiled=_unreachable_core,
        layout=resolve_output_layout(
            core_key=name,
            value_template=jnp.arange(3.0),
            state_order=("wealth",),
            output_roles=VALUE,
        ),
        tile_widths={},
        input_transfer_plan=(transfer,),
        name=name,
    )


def _unreachable_core(**_kwargs: object) -> object:
    """Stand in for a compiled core the plan never calls."""
    raise AssertionError


def test_a_two_core_regime_declares_one_consumer_of_a_shared_read() -> None:
    """A regime commits once per period, so its cores are one consumer together."""
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    cores = MappingProxyType(
        {name: _planned_core(name=name, transfer=transfer) for name in ("main", "tail")}
    )

    consumer_counts, _keys_by_regime = _period_shared_transfer_plan(
        compiled_cores_by_regime=MappingProxyType({"source": cores})
    )

    assert dict(consumer_counts) == {_key(transfer=transfer): 1}


def test_a_cache_that_does_not_release_keeps_its_copy_after_every_commit() -> None:
    """An eager solve's cache frees no buffer, however many consumers commit.

    An eager dispatch is an ordinary Python call whose result may be any object
    its arguments contained, so no buffer it touched is known to be one the
    engine produced — the same reason the period's other release paths do
    nothing when releasing is off.
    """
    registry = BufferRegistry()
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    key = _key(transfer=transfer)
    cache = PeriodTransferCache(
        registry=registry,
        consumer_counts=MappingProxyType({key: 1}),
        release_enabled=False,
    )
    copy = jax.device_put(np.arange(3.0), source_sharding)
    cache.put(transfer=transfer, array=copy, stored=stored)

    cache.commit_consumer(key=key)

    assert not copy.is_deleted()


def test_a_cache_that_does_not_release_still_refuses_an_undeclared_commit() -> None:
    """Consumer counting is a plan check, not a release decision."""
    stored_sharding, source_sharding = _shardings()
    stored = _stored_value(sharding=stored_sharding)
    transfer = _copy_transfer(
        reused=True, stored=stored, source_sharding=source_sharding
    )
    key = _key(transfer=transfer)
    cache = PeriodTransferCache(
        registry=BufferRegistry(),
        consumer_counts=MappingProxyType({key: 1}),
        release_enabled=False,
    )
    cache.commit_consumer(key=key)

    with pytest.raises(ExecutionPlanningError, match="declared consumer"):
        cache.commit_consumer(key=key)
