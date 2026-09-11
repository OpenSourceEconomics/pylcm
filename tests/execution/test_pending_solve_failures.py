"""Completion ownership survives nested operands and partial execution failures."""

import gc
import weakref
from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.output_layout import (
    VALUE,
    PlannedCore,
    StateAxesLeading,
    resolve_output_layout,
)
from _lcm.execution.pending_work import PendingSolveWork
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    resolve_value_transfer,
)


def _nested(*, inputs: Mapping[str, tuple[jax.Array]]) -> jax.Array:
    return inputs["payload"][0] + 1


def _two_outputs(*, value: jax.Array, matrix: jax.Array) -> tuple[jax.Array, jax.Array]:
    return value, matrix @ matrix


def test_nested_immutable_copy_is_matched_in_keyword_relative_compiler_paths(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = jnp.arange(8.0).block_until_ready()
    layout = jax.NamedSharding(jax.make_mesh((1,), ("unit",)), jax.P())
    template = jax.ShapeDtypeStruct(original.shape, original.dtype, sharding=layout)
    compiled = (
        jax.jit(_nested, out_shardings=layout)
        .lower(inputs=MappingProxyType({"payload": (template,)}))
        .compile()
    )
    transfer = resolve_value_transfer(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="target"
        ),
        source=ValueConsumerAddress(
            source_period=0,
            source_regime="reader",
            core_key="main",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            argument="inputs",
            path=("payload", 0),
        ),
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
        stored_template=original,
        source_sharding=layout,
    )
    owner = PendingSolveWork()
    core = PlannedCore(
        compiled=compiled,
        name="main",
        tile_widths={},
        layout=resolve_output_layout(
            core_key="main", value_template=template, state_order=(), output_roles=VALUE
        ),
        input_transfer_plan=(transfer,),
        pending_work=owner,
    )
    wait = jax.block_until_ready
    waits: list[object] = []

    def observe_wait(tree: object) -> object:
        waits.append(tree)
        return wait(tree)

    monkeypatch.setattr(jax, "block_until_ready", observe_wait)
    try:
        result = core(inputs=MappingProxyType({"payload": (original,)}))
        assert isinstance(result, jax.Array)
        assert waits == [], (
            "The exact nested kept occurrence must carry the copy dependency."
        )
        assert jnp.array_equal(result, original + 1)
        assert result.sharding == layout
        assert not original.is_deleted()
    finally:
        owner.close()


def test_full_output_tree_is_owned_before_layout_failure(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = jnp.asarray(3.0).block_until_ready()
    matrix = jnp.full((1024, 1024), 0.125).block_until_ready()
    compiled = jax.jit(_two_outputs).lower(value=value, matrix=matrix).compile()
    owner = PendingSolveWork()
    # The deliberate wrong auxiliary shape makes the ordinary layout guard fail
    # after real execution. Completion still owns the full returned tree.
    layout = resolve_output_layout(
        core_key="main",
        value_template=value,
        state_order=(),
        output_roles=(
            VALUE,
            StateAxesLeading(state_names=(), n_free_leading_axes=2, shape=(1, 1)),
        ),
    )
    core = PlannedCore(
        compiled=compiled,
        name="main",
        tile_widths={},
        layout=layout,
        pending_work=owner,
    )
    call = jax.stages.Compiled.__call__
    returned: list[jax.Array] = []

    def observe(
        executable: jax.stages.Compiled, *args: object, **kwargs: object
    ) -> object:
        output = call(executable, *args, **kwargs)
        if executable is compiled:
            returned.extend(jax.tree.leaves(output))
        return output

    monkeypatch.setattr(jax.stages.Compiled, "__call__", observe)
    try:
        with pytest.raises(AssertionError, match="shape mismatch"):
            core(value=value, matrix=matrix)
        assert len(returned) == 2
        reference = weakref.ref(returned[1])
        returned.clear()
        gc.collect()
        assert reference() is not None, (
            "The solve owner must retain the auxiliary witness."
        )
        owner.before(devices=matrix.devices())
        gc.collect()
        assert reference() is None
        assert not value.is_deleted()
        assert not matrix.is_deleted()
    finally:
        owner.close()


@pytest.mark.parametrize("fail_at", ["transfer_plan", "compiled_call"])
def test_failed_call_keeps_the_returned_copy_until_owner_close(
    *,
    fail_at: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = jnp.arange(8.0).block_until_ready()
    layout = jax.NamedSharding(jax.make_mesh((1,), ("unit",)), jax.P(None))
    template = jax.ShapeDtypeStruct(original.shape, original.dtype, sharding=layout)
    compiled = (
        jax.jit(_nested, out_shardings=layout)
        .lower(inputs=MappingProxyType({"payload": (template,)}))
        .compile()
    )
    transfer = resolve_value_transfer(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="target"
        ),
        source=ValueConsumerAddress(
            source_period=0,
            source_regime="reader",
            core_key="main",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            argument="inputs",
            path=("payload", 0),
        ),
        kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
        stored_template=original,
        source_sharding=layout,
    )
    owner = PendingSolveWork()
    core = PlannedCore(
        compiled=compiled,
        name="main",
        tile_widths={},
        layout=resolve_output_layout(
            core_key="main", value_template=template, state_order=(), output_roles=VALUE
        ),
        input_transfer_plan=(transfer, transfer)
        if fail_at == "transfer_plan"
        else (transfer,),
        pending_work=owner,
    )
    put = jax.device_put
    call = jax.stages.Compiled.__call__
    backend_error = RuntimeError(
        "observed compiled-call failure after materialized copy"
    )
    references: list[weakref.ReferenceType[jax.Array]] = []
    pointers: list[int] = []

    # keyword-only-exempt: library-callback=jax.device_put
    def observe_copy(value: object, device: object) -> object:
        placed = put(value, device)
        if value is original:
            fresh = jnp.array(placed, copy=True)
            references.append(weakref.ref(fresh))
            pointers.append(fresh.unsafe_buffer_pointer())
            return fresh
        return placed

    def fail_compiled_call(
        executable: jax.stages.Compiled, *args: object, **kwargs: object
    ) -> object:
        if executable is compiled:
            assert len(references) == 1
            assert references[0]() is not None
            raise backend_error
        return call(executable, *args, **kwargs)

    monkeypatch.setattr(jax, "device_put", observe_copy)
    if fail_at == "compiled_call":
        monkeypatch.setattr(jax.stages.Compiled, "__call__", fail_compiled_call)
    try:
        expected = ValueError if fail_at == "transfer_plan" else RuntimeError
        message = (
            "Duplicate value-transfer consumer path"
            if fail_at == "transfer_plan"
            else "observed compiled-call failure"
        )
        with pytest.raises(expected, match=message) as caught:
            core(inputs=MappingProxyType({"payload": (original,)}))
        if fail_at == "compiled_call":
            assert caught.value is backend_error
        # Retained Python tracebacks legitimately keep their argument tree.
        # Drop that independent owner before checking the solve owner's roots.
        caught.value.__traceback__ = None
        del caught
        assert len(references) == 1
        assert pointers != [original.unsafe_buffer_pointer()]
        gc.collect()
        assert references[0]() is not None
        owner.close()
        gc.collect()
        assert references[0]() is None
        assert not original.is_deleted()
        assert jnp.array_equal(original, jnp.arange(8.0))
    finally:
        owner.close()
