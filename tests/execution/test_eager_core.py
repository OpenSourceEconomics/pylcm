"""Eager adapters preserve operands, output identity and call-local ownership."""

import dataclasses
import functools
import gc
import weakref
from collections.abc import Callable, Mapping
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    InternalInputRef,
    ResolvedCoreProgram,
)
from _lcm.execution.eager_core import make_eager_core
from _lcm.execution.output_layout import VALUE
from lcm.exceptions import ExecutionPlanningError


def internal_eager_program(*, function: Callable[..., object]) -> ResolvedCoreProgram:
    """Declare a real producer reference separate from ordinary builder arguments."""
    return dataclasses.replace(
        eager_program(function=function, arguments={}),
        requirements=CoreExecutionRequirements(
            internal_inputs={
                "produced": InternalInputRef(producer="keeper", label="carry")
            }
        ),
    )


def eager_program(
    *, function: Callable[..., object], arguments: Mapping[str, object]
) -> ResolvedCoreProgram:
    """Declare a dense numerical body independently of the eager adapter."""
    return ResolvedCoreProgram(
        name="main",
        function=function,
        arguments=arguments,
        static_kwargs={},
        requirements=CoreExecutionRequirements(),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="eager placement witness",
        donation_candidates=(),
        tile_widths={},
        specialization_key=(),
        input_transfer_plan=(),
    )


@pytest.mark.parametrize(
    "kind", ["weak_float", "weak_int", "strong_float", "strong_int"]
)
def test_eager_scalar_keeps_dtype_weak_type_and_original_tree(*, kind: str) -> None:
    source = {
        "weak_float": jnp.asarray(3.0),
        "weak_int": jnp.asarray(3),
        "strong_float": jnp.asarray(3.0, dtype=jnp.float32),
        "strong_int": jnp.asarray(3, dtype=jnp.int32),
    }[kind]
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    returned: list[object] = []

    def body(*, scalar: jax.Array) -> object:
        output = {"value": (scalar, None)}
        returned.append(output)
        return output

    descriptor = jax.ShapeDtypeStruct(
        source.shape, source.dtype, sharding=target, weak_type=source.weak_type
    )
    adapter = make_eager_core(
        program=eager_program(function=body, arguments={"scalar": descriptor}),
        execution_sharding=target,
    )
    result = adapter(scalar=source)
    assert result is returned[0]
    output = cast("dict[str, tuple[jax.Array, None]]", result)["value"][0]
    assert output.shape == ()
    assert output.dtype == source.dtype
    assert output.weak_type is source.weak_type
    assert output.sharding == target
    np.testing.assert_array_equal(output, source)
    assert not source.is_deleted()


def test_eager_partial_keeps_fixed_array_and_static_width() -> None:
    source = jnp.asarray([1.0, 2.0, 4.0])
    fixed = jnp.asarray([3.0, 5.0, 7.0])
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def body(*, value: jax.Array, offset: jax.Array, width: int) -> jax.Array:
        assert width == 2
        assert offset is fixed
        return value + offset

    program = dataclasses.replace(
        eager_program(
            function=functools.partial(body, offset=fixed),
            arguments={
                "value": jax.ShapeDtypeStruct(
                    source.shape, source.dtype, sharding=target
                )
            },
        ),
        static_kwargs={"width": 2},
    )
    adapter = make_eager_core(program=program, execution_sharding=target)
    np.testing.assert_array_equal(adapter(value=source), np.asarray([4.0, 7.0, 11.0]))
    np.testing.assert_array_equal(fixed, np.asarray([3.0, 5.0, 7.0]))
    np.testing.assert_array_equal(source, np.asarray([1.0, 2.0, 4.0]))


def test_eager_adapter_drops_runtime_owners_between_calls() -> None:
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    source = jax.device_put(jnp.arange(4, dtype=jnp.int32), target)
    source_ref = weakref.ref(source)
    adapter = make_eager_core(
        program=eager_program(
            function=lambda value: value,
            arguments={
                "value": jax.ShapeDtypeStruct(
                    source.shape, source.dtype, sharding=target
                )
            },
        ),
        execution_sharding=target,
    )
    output = adapter(value=source)
    assert output is source
    del output, source
    gc.collect()
    assert source_ref() is None
    replacement = jax.device_put(jnp.arange(4, dtype=jnp.int32) + 8, target)
    assert adapter(value=replacement) is replacement


def test_eager_exception_does_not_retain_runtime_owners() -> None:
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def body(*, value: jax.Array) -> object:
        assert value.shape == (4,)
        raise RuntimeError("numerical failure")

    adapter = make_eager_core(
        program=eager_program(
            function=body,
            arguments={"value": jax.ShapeDtypeStruct((4,), jnp.int32, sharding=target)},
        ),
        execution_sharding=target,
    )
    for offset in (0, 8):
        source = jnp.arange(4, dtype=jnp.int32) + offset
        reference = weakref.ref(source)
        with pytest.raises(RuntimeError, match="numerical failure"):
            adapter(value=source)
        assert not source.is_deleted()
        del source
        gc.collect()
        assert reference() is None


def test_repeated_original_does_not_hide_an_invalid_second_descriptor() -> None:
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    source = jnp.asarray(1.0, dtype=jnp.float32)

    def forbidden(**_arguments: object) -> object:
        pytest.fail("invalid alias metadata reached the numerical body")

    adapter = make_eager_core(
        program=eager_program(
            function=forbidden,
            arguments={
                "first": jax.ShapeDtypeStruct((), source.dtype, sharding=target),
                "second": jax.ShapeDtypeStruct(
                    (), source.dtype, sharding=target, weak_type=True
                ),
            },
        ),
        execution_sharding=target,
    )
    with pytest.raises(ExecutionPlanningError, match="shape, dtype or weak typing"):
        adapter(first=source, second=source)


@pytest.mark.parametrize("concrete", [False, True])
def test_eager_builder_refuses_missing_abstract_layout(*, concrete: bool) -> None:
    argument = jnp.ones(3) if concrete else jax.ShapeDtypeStruct((3,), jnp.float32)
    with pytest.raises(ExecutionPlanningError, match="abstract input layouts"):
        make_eager_core(
            program=eager_program(
                function=lambda value: value, arguments={"value": argument}
            ),
            execution_sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        )


@pytest.mark.parametrize("change", ["shape", "dtype", "weak_type"])
def test_eager_input_metadata_mismatch_fails_before_body(*, change: str) -> None:
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    source = jnp.asarray(1.0, dtype=jnp.float32)
    shape = (1,) if change == "shape" else ()
    dtype = jnp.int32 if change == "dtype" else jnp.float32

    def forbidden(*, value: object) -> object:
        pytest.fail(f"numerical body received invalid operand {value!r}")

    adapter = make_eager_core(
        program=eager_program(
            function=forbidden,
            arguments={
                "value": jax.ShapeDtypeStruct(
                    shape, dtype, sharding=target, weak_type=change == "weak_type"
                )
            },
        ),
        execution_sharding=target,
    )
    with pytest.raises(ExecutionPlanningError, match="shape, dtype or weak typing"):
        adapter(value=source)


def test_internal_eager_inputs_keep_identity_and_release_runtime_owners() -> None:
    """Missing abstract sharding does not allocate or retain a producer output."""
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    source = jax.device_put(jnp.arange(4, dtype=jnp.int32), target)
    source_ref = weakref.ref(source)
    adapter = make_eager_core(
        program=internal_eager_program(function=lambda produced: produced),
        internal_input_templates={"produced": jax.ShapeDtypeStruct((4,), jnp.int32)},
        execution_sharding=target,
    )
    output = adapter(produced=source)
    assert output is source
    np.testing.assert_array_equal(source, np.arange(4))
    del output, source
    gc.collect()
    assert source_ref() is None


@pytest.mark.parametrize("change", ["shape", "dtype", "weak_type"])
def test_internal_eager_metadata_corruption_stops_before_body(*, change: str) -> None:
    """The lack of template sharding never weakens numerical metadata checks."""
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    source = jax.device_put(jnp.asarray(2.0, dtype=jnp.float32), target)

    def forbidden(*, produced: object) -> object:
        pytest.fail(f"corrupted internal operand reached numerical body: {produced!r}")

    adapter = make_eager_core(
        program=internal_eager_program(function=forbidden),
        internal_input_templates={
            "produced": jax.ShapeDtypeStruct(
                (1,) if change == "shape" else (),
                jnp.int32 if change == "dtype" else jnp.float32,
                weak_type=change == "weak_type",
            )
        },
        execution_sharding=target,
    )
    with pytest.raises(ExecutionPlanningError, match="shape, dtype or weak typing"):
        adapter(produced=source)
    np.testing.assert_array_equal(source, 2.0)


@pytest.mark.parametrize("invalid", ["missing", "extra", "concrete", "overlap"])
def test_internal_eager_builder_requires_exact_abstract_declared_inputs(
    *, invalid: str
) -> None:
    """Only declared producer inputs may carry layout-free abstract metadata."""
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    template = jax.ShapeDtypeStruct((), jnp.float32)
    templates = {"produced": template}
    program = internal_eager_program(function=lambda produced: produced)
    if invalid == "missing":
        templates = {}
    elif invalid == "extra":
        templates["other"] = template
    elif invalid == "concrete":
        templates = {"produced": jnp.asarray(1.0)}
    else:
        program = dataclasses.replace(
            program,
            arguments={
                "produced": jax.ShapeDtypeStruct((), jnp.float32, sharding=target)
            },
        )
    with pytest.raises(ExecutionPlanningError, match="internal input"):
        make_eager_core(
            program=program,
            execution_sharding=target,
            internal_input_templates=templates,
        )
