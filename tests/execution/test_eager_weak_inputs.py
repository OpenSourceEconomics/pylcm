"""Eager binding uses declared strong types without weakening operand guards."""

import gc
import weakref
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.eager_core import make_eager_core
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import ValueND
from tests.execution.test_eager_core import eager_program, internal_eager_program


@pytest.mark.parametrize("shape", [(), (3,)])
@pytest.mark.parametrize("fails", [False, True])
def test_weak_binding_shares_one_allocation_and_releases_call_owners(
    *, shape: tuple[int, ...], fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real weak originals and their distinct normalized copy survive the body."""
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    source = jax.device_put(jnp.full(shape, 2.0), target)
    assert source.weak_type
    assert source.committed
    source_ref = weakref.ref(source)
    normalized_refs: list[weakref.ReferenceType[ValueND]] = []
    outputs: list[object] = []
    calls: list[object] = []
    original_asarray = cast("Callable[..., ValueND]", jnp.asarray)

    def observe(*args: object, **kwargs: object) -> ValueND:
        calls.append(kwargs.get("dtype"))
        return original_asarray(*args, **kwargs)

    def body(*, first: ValueND, second: ValueND) -> object:
        original = source_ref()
        assert original is not None
        assert not original.is_deleted()
        assert first is second
        assert first is not original
        assert not first.weak_type
        assert original.weak_type
        assert first.sharding == target
        assert first.unsafe_buffer_pointer() != original.unsafe_buffer_pointer()
        np.testing.assert_array_equal(first, np.full(shape, 2.0))
        np.testing.assert_array_equal(original, np.full(shape, 2.0))
        normalized_refs.append(weakref.ref(first))
        if fails:
            raise RuntimeError("deliberate numerical failure")
        output = {"value": (first, second)}
        outputs.append(output)
        return output

    descriptor = jax.ShapeDtypeStruct(shape, source.dtype, sharding=target)
    adapter = make_eager_core(
        program=eager_program(
            function=body, arguments={"first": descriptor, "second": descriptor}
        ),
        execution_sharding=target,
    )
    monkeypatch.setattr(jnp, "asarray", observe)
    if fails:
        with pytest.raises(RuntimeError, match="deliberate numerical failure") as error:
            adapter(first=source, second=source)
        error.value.__traceback__ = None
        del error
    else:
        output = adapter(first=source, second=source)
        assert output is outputs.pop()
        del output
    assert calls == [source.dtype]
    assert source_ref() is source
    assert not source.is_deleted()
    del source
    gc.collect()
    assert source_ref() is None
    assert len(normalized_refs) == 1
    assert normalized_refs[0]() is None


def _mixed_promotion(*, value: ValueND, other: ValueND) -> ValueND:
    return value + other


def test_weak_binding_matches_actual_compiled_promotion_and_detects_raw_bypass() -> (
    None
):
    """Skipping declared strong binding changes the real arithmetic result dtype."""
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    source = jax.device_put(jnp.asarray(2.0), target)
    lower_dtype = jnp.float32 if jax.config.x64_enabled else jnp.float16
    other = jax.device_put(jnp.asarray(0.25, dtype=lower_dtype), target)
    descriptors = {
        name: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=target)
        for name, value in {"value": source, "other": other}.items()
    }
    compiled = jax.jit(_mixed_promotion).lower(**descriptors).compile()
    expected = compiled(value=source, other=other)
    raw = _mixed_promotion(value=source, other=other)
    assert expected.dtype == source.dtype
    assert raw.dtype == other.dtype
    assert raw.dtype != expected.dtype
    adapter = make_eager_core(
        program=eager_program(function=_mixed_promotion, arguments=descriptors),
        execution_sharding=target,
    )
    actual = cast("ValueND", adapter(value=source, other=other))
    assert actual.dtype == expected.dtype
    assert not actual.weak_type
    assert actual.sharding == target
    assert source.weak_type
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, 2.0)


def test_declared_internal_weak_value_uses_the_same_strong_binding() -> None:
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    source = jax.device_put(jnp.asarray(3.0), target)
    adapter = make_eager_core(
        program=internal_eager_program(function=lambda produced: produced),
        internal_input_templates={"produced": jax.ShapeDtypeStruct((), source.dtype)},
        execution_sharding=target,
    )
    actual = cast("ValueND", adapter(produced=source))
    assert not actual.weak_type
    assert source.weak_type
    assert actual.sharding == source.sharding == target
    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_weak_binding_never_repairs_invalid_shape_or_dtype(*, mismatch: str) -> None:
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    source = jax.device_put(jnp.asarray(3.0), target)

    def forbidden(*, value: ValueND) -> ValueND:
        del value
        pytest.fail("invalid weak operand reached numerical body")

    adapter = make_eager_core(
        program=eager_program(
            function=forbidden,
            arguments={
                "value": jax.ShapeDtypeStruct(
                    (1,) if mismatch == "shape" else (),
                    jnp.int32 if mismatch == "dtype" else source.dtype,
                    sharding=target,
                )
            },
        ),
        execution_sharding=target,
    )
    with pytest.raises(ExecutionPlanningError, match="shape, dtype or weak typing"):
        adapter(value=source)
    assert source.weak_type
    assert not source.is_deleted()
