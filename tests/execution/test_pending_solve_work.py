"""Completion ownership visits every real output and clears its retained roots."""

import gc
import weakref
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution import pending_work
from _lcm.execution.compiler_inputs import compiler_input_paths
from _lcm.execution.output_layout import VALUE, PlannedCore, resolve_output_layout
from _lcm.execution.scheduler import BufferRegistry, PeriodTransferCache
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    resolve_value_transfer,
)


def _is_ready(*, array: jax.Array) -> bool:
    """Read the real JAX readiness method omitted from its current type stubs."""
    return cast("Callable[[], bool]", getattr(array, "is_ready"))()  # noqa: B009


@pytest.fixture
def completed_array_ids(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Observe real completion calls without retaining arrays or timing their work."""
    array_type = type(jnp.asarray(0.0))
    complete = array_type.block_until_ready
    completed: list[int] = []

    def observe(array: jax.Array) -> jax.Array:
        completed.append(id(array))
        return complete(array)

    monkeypatch.setattr(array_type, "block_until_ready", observe)
    return completed


def test_conflict_completes_auxiliary_beside_ready_pass_through_value(
    completed_array_ids: list[int],
) -> None:
    owner = pending_work.PendingSolveWork()
    ready_value = jnp.asarray(3.0).block_until_ready()
    auxiliary = jnp.arange(8.0).block_until_ready()
    completed_array_ids.clear()
    try:
        owner.record(
            outputs=(ready_value, {"auxiliary": auxiliary}), devices=auxiliary.devices()
        )
        assert _is_ready(array=ready_value)
        assert completed_array_ids == []
        owner.before(devices=auxiliary.devices())
        assert completed_array_ids == [id(ready_value), id(auxiliary)]
        assert _is_ready(array=auxiliary)
    finally:
        owner.close()


def test_before_delete_discharges_record_before_invalidating_wrapper(
    completed_array_ids: list[int],
) -> None:
    owner = pending_work.PendingSolveWork()
    result = jnp.arange(8.0).block_until_ready()
    devices = result.devices()
    completed_array_ids.clear()
    try:
        owner.record(outputs=result, devices=devices)
        assert completed_array_ids == []
        owner.before_delete(arrays=(result,))
        assert completed_array_ids == [id(result)]
        assert _is_ready(array=result)
        result.delete()
        owner.before(devices=devices)
        assert completed_array_ids == [id(result)]
    finally:
        owner.close()


@pytest.mark.parametrize("drain", ["before", "close"])
def test_drained_owner_keeps_no_output_reference(
    *, drain: str, completed_array_ids: list[int]
) -> None:
    owner = pending_work.PendingSolveWork()
    result = jnp.arange(8.0).block_until_ready()
    reference = weakref.ref(result)
    completed_array_ids.clear()
    owner.record(outputs=result, devices=result.devices())
    if drain == "before":
        owner.before(devices=result.devices())
    else:
        owner.close()
    assert completed_array_ids == [id(result)]
    assert _is_ready(array=result)
    del result
    gc.collect()
    assert reference() is None
    owner.close()


def test_close_preserves_active_exception_and_discards_its_owner_roots() -> None:
    owner = pending_work.PendingSolveWork()
    result = jnp.arange(8.0).block_until_ready()
    owner.record(outputs=result, devices=result.devices())
    # A missing release hook is not evidence of readiness: the cleanup must
    # report that invalid witness without replacing the original solver error.
    result.block_until_ready()
    result.delete()
    original = ValueError("original layout validation failure")
    with pytest.raises(ValueError, match="original layout") as caught:  # noqa: PT012
        try:
            raise original
        finally:
            owner.close()
    assert caught.value is original
    assert any("completion" in note for note in original.__notes__)
    owner.close()


def _copy_body(*, value: jax.Array, keep: bool) -> jax.Array:
    return value + 1 if keep else jnp.full_like(value, 2)


@pytest.mark.parametrize("keep", [False, True])
def test_real_new_copy_waits_before_dispatch_only_when_compiler_drops_its_operand(
    *,
    keep: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_copy(keep=keep, donates=False, monkeypatch=monkeypatch)


def test_real_donor_completes_fresh_copy_before_invalidating_its_wrapper(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_copy(keep=True, donates=True, monkeypatch=monkeypatch)


def test_shared_copy_cache_hit_and_release_do_not_leave_a_deleted_witness(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_copy(keep=True, donates=False, shared=True, monkeypatch=monkeypatch)


def _exercise_copy(  # noqa: PLR0915 -- one real transfer/dispatch/lifetime witness
    *,
    keep: bool,
    donates: bool,
    monkeypatch: pytest.MonkeyPatch,
    shared: bool = False,
) -> None:
    original = jax.device_put(
        jnp.arange(8.0), jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )
    required = jax.NamedSharding(jax.make_mesh((1,), ("destination",)), jax.P(None))
    output_template = jax.device_put(jnp.zeros_like(original), required)
    compiled = (
        jax.jit(
            partial(_copy_body, keep=keep),
            out_shardings=required,
            donate_argnames=("value",) if donates else (),
        )
        .lower(
            value=jax.ShapeDtypeStruct(
                original.shape, original.dtype, sharding=required
            )
        )
        .compile()
    )
    kept_paths = compiler_input_paths(compiled=compiled, arguments={"value": original})
    assert bool(kept_paths) is keep
    transfer = resolve_value_transfer(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="target"
        ),
        source=ValueConsumerAddress(
            source_period=0,
            source_regime="reader",
            core_key="main",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            argument="value",
            path=(),
        ),
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
        stored_template=original,
        source_sharding=required,
    )
    owner = pending_work.PendingSolveWork()
    outputs: list[jax.Array] = []
    transfer = replace(transfer, reused_by_several_consumers=shared)
    cache_key = (transfer.target, transfer.source_sharding)
    cache = (
        PeriodTransferCache(
            registry=BufferRegistry(),
            consumer_counts={cache_key: 2},
            pending_outputs=outputs,
            before_delete=owner.before_delete,
        )
        if shared
        else None
    )
    core = PlannedCore(
        compiled=compiled,
        layout=resolve_output_layout(
            core_key="main",
            value_template=output_template,
            state_order=(),
            output_roles=VALUE,
        ),
        tile_widths={},
        input_transfer_plan=(transfer,),
        name="main",
        pending_work=owner,
        transfer_cache=cache,
        donated_arguments=("value",) if donates else (),
    )
    copies: list[jax.Array] = []
    waited_ids: set[int] = set()
    dispatch_waits: list[bool] = []
    copy_pointers: list[int] = []
    put = jax.device_put
    wait = jax.block_until_ready
    call = jax.stages.Compiled.__call__

    # keyword-only-exempt: library-callback=jax.device_put
    def copy(value: object, device: object) -> object:
        placed = put(value, device)
        if value is original:
            # A one-device placement change may reuse the source. Force a real
            # fresh copy here and assert physical separation below; this is an
            # ownership fixture, not cross-device transfer evidence.
            fresh = jnp.array(placed, copy=True)
            copies.append(fresh)
            return fresh
        return placed

    def observe_wait(tree: object) -> object:
        waited_ids.update(
            id(leaf) for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)
        )
        return wait(tree)

    def observe_call(
        executable: jax.stages.Compiled, *args: object, **kwargs: object
    ) -> object:
        if executable is compiled:
            assert len(copies) == 1
            dispatch_waits.append(id(copies[0]) in waited_ids)
            copy_pointers.append(copies[0].unsafe_buffer_pointer())
        return call(executable, *args, **kwargs)

    monkeypatch.setattr(jax, "device_put", copy)
    monkeypatch.setattr(jax, "block_until_ready", observe_wait)
    monkeypatch.setattr(jax.stages.Compiled, "__call__", observe_call)
    try:
        result = core(value=original)
        assert isinstance(result, jax.Array)
        outputs.append(result)
        if shared:
            result = core(value=original)
            assert isinstance(result, jax.Array)
            outputs.append(result)
        assert dispatch_waits == [donates or not keep] * (2 if shared else 1)
        assert all(
            pointer != original.unsafe_buffer_pointer() for pointer in copy_pointers
        )
        assert len(set(copy_pointers)) == 1
        if donates:
            assert copies[0].is_deleted()
            assert result.unsafe_buffer_pointer() == copy_pointers[0]
        assert result.sharding == required
        assert jnp.array_equal(
            result, original + 1 if keep else jnp.full_like(original, 2)
        )
        assert jnp.array_equal(original, jnp.arange(8.0))
        if cache is not None:
            cache.commit_consumer(key=cache_key)
            assert not copies[0].is_deleted()
            released = cache.commit_consumer(key=cache_key)
            assert len(released) == 1
            assert copies[0].is_deleted()
    finally:
        owner.close()
