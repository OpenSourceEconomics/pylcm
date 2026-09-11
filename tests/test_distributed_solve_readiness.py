"""Actual complete device footprints control asynchronous solve completion."""

import jax
import pytest

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
except RuntimeError:
    pytest.skip(
        "Solve readiness topology requires a fresh JAX backend.",
        allow_module_level=True,
    )

import numpy as np

from _lcm.execution.output_layout import VALUE, PlannedCore, resolve_output_layout
from _lcm.execution.pending_work import PendingSolveWork
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    classify_value_transfer,
    resolve_value_transfer,
)
from tests.execution.test_pending_solve_work import _is_ready


def _dot(*, operand: jax.Array) -> jax.Array:
    return operand @ operand


def _increment(*, value: jax.Array) -> jax.Array:
    return value + 1


def _layout(*, device: jax.Device) -> jax.NamedSharding:
    return jax.NamedSharding(jax.make_mesh((1,), ("unit",), devices=(device,)), jax.P())


@pytest.mark.parametrize("budgeted", [True, False])
@pytest.mark.parametrize("footprint", ["same_execution", "shared_source", "disjoint"])
def test_actual_execution_and_copy_endpoints_determine_completion(  # noqa: PLR0915
    *,
    budgeted: bool,
    footprint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(jax.devices()) == 4
    first_device, other_device = jax.devices()[1:3]
    second_device = first_device if footprint == "same_execution" else other_device
    source_device = first_device if footprint == "shared_source" else second_device
    first_layout = _layout(device=first_device)
    second_layout = _layout(device=second_device)
    source_layout = _layout(device=source_device)
    work = jax.device_put(
        np.full((1024, 1024), 0.125), first_layout
    ).block_until_ready()
    source = jax.device_put(np.arange(8.0), source_layout).block_until_ready()
    output_template = jax.ShapeDtypeStruct(
        source.shape, source.dtype, sharding=second_layout
    )
    producer = jax.jit(_dot, out_shardings=first_layout).lower(operand=work).compile()
    consumer = (
        jax.jit(_increment, out_shardings=second_layout)
        .lower(value=output_template)
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
            argument="value",
            path=(),
        ),
        kind=classify_value_transfer(
            stored_sharding=source_layout, required_sharding=second_layout
        ),
        stored_template=source,
        source_sharding=second_layout,
    )
    owner = PendingSolveWork() if budgeted else None
    first = PlannedCore(
        compiled=producer,
        name="producer",
        tile_widths={},
        layout=resolve_output_layout(
            core_key="producer", value_template=work, state_order=(), output_roles=VALUE
        ),
        pending_work=owner,
    )
    second = PlannedCore(
        compiled=consumer,
        name="consumer",
        tile_widths={},
        layout=resolve_output_layout(
            core_key="consumer",
            value_template=output_template,
            state_order=(),
            output_roles=VALUE,
        ),
        input_transfer_plan=(transfer,),
        pending_work=owner,
    )
    assert work.devices() == {first_device}
    assert source.devices() == {source_device}
    assert set(consumer.output_shardings.device_set) == {second_device}
    complete_first = {first_device}
    complete_second = {source_device, second_device}
    assert complete_first.isdisjoint(complete_second) is (footprint == "disjoint")
    call = jax.stages.Compiled.__call__
    put = jax.device_put
    before_dispatch: list[bool] = []
    before_copy: list[bool] = []
    pending: jax.Array | None = None

    def observe(
        executable: jax.stages.Compiled, *args: object, **kwargs: object
    ) -> object:
        if executable is consumer:
            assert pending is not None
            before_dispatch.append(_is_ready(array=pending))
        return call(executable, *args, **kwargs)

    # keyword-only-exempt: library-callback=jax.device_put
    def observe_copy(value: object, device: object) -> object:
        if value is source:
            assert pending is not None
            before_copy.append(_is_ready(array=pending))
        return put(value, device)

    monkeypatch.setattr(jax.stages.Compiled, "__call__", observe)
    monkeypatch.setattr(jax, "device_put", observe_copy)
    try:
        produced = first(operand=work)
        assert isinstance(produced, jax.Array)
        pending = produced
        assert not _is_ready(array=pending), (
            "The control requires actual asynchronous work."
        )
        result = second(value=source)
        assert isinstance(result, jax.Array)
        expected_ready = budgeted and footprint != "disjoint"
        assert before_dispatch == [expected_ready]
        if footprint == "shared_source":
            assert before_copy == [expected_ready]
        else:
            assert before_copy == []
        assert result.sharding == second_layout
        np.testing.assert_array_equal(result, np.arange(8.0) + 1)
        np.testing.assert_array_equal(pending, np.full((1024, 1024), 16.0))
        np.testing.assert_array_equal(source, np.arange(8.0))
        np.testing.assert_array_equal(work, np.full((1024, 1024), 0.125))
    finally:
        if owner is not None:
            owner.close()
        if pending is not None:
            pending.block_until_ready()
