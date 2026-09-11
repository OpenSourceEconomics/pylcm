"""Solve descriptors preserve concrete operands' numerical and placement contract."""

import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.abstract_program_inputs import abstract_program_inputs
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    MaterializedCoreProgram,
    ValueRead,
    resolve_core_program,
)
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    resolve_value_transfer,
)
from _lcm.solution.backward_induction import _abstract_arguments_key
from tests.test_dropped_models_release_nested_functions import _live_nested_functions


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Payload:
    values: object
    other: object


def _identity(*, payload: object, scalar: object) -> object:
    return payload, scalar


def _program(arguments: dict[str, object]) -> MaterializedCoreProgram:
    return MaterializedCoreProgram(
        name="main",
        function=_identity,
        arguments=arguments,
        requirements=CoreExecutionRequirements(),
        output_roles="value",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _describe_and_drop() -> tuple[weakref.ReferenceType[object], ...]:
    """Exercise real descriptor construction without retaining its input owners."""
    original = jnp.arange(4.0)
    program = _program({"payload": original, "scalar": jnp.asarray(1.0)})
    described = abstract_program_inputs(
        program=program,
        transfers=(),
        execution_sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    return weakref.ref(original), weakref.ref(program), weakref.ref(described)


def test_dropped_descriptor_build_retains_no_nested_function_or_input_owner() -> None:
    """A second real build leaves no per-call engine function in the claw cache."""
    source_roots = (Path(__file__).resolve().parents[2] / "src" / "_lcm",)
    _describe_and_drop()
    before = _live_nested_functions(source_roots=source_roots)

    references = _describe_and_drop()

    after = _live_nested_functions(source_roots=source_roots)
    assert all(reference() is None for reference in references)
    assert after == before


@pytest.mark.parametrize("weak", [False, True])
def test_numerical_argument_identity_agrees_for_arrays_and_descriptors(
    *,
    weak: bool,
) -> None:
    original = jnp.asarray(3.0) if weak else jnp.asarray(3.0, dtype=jnp.float32)
    layout = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    original = jax.device_put(original, layout)
    assert original.weak_type is weak
    concrete = _program({"payload": (original,), "scalar": original})

    described = abstract_program_inputs(
        program=concrete, transfers=(), execution_sharding=layout
    )

    assert _abstract_arguments_key(
        arguments=concrete.arguments
    ) == _abstract_arguments_key(arguments=described.arguments)
    scalar = described.arguments["scalar"]
    assert isinstance(scalar, jax.ShapeDtypeStruct)
    altered = {
        **described.arguments,
        "scalar": jax.ShapeDtypeStruct(
            scalar.shape, scalar.dtype, weak_type=not weak, sharding=layout
        ),
    }
    assert _abstract_arguments_key(arguments=altered) != _abstract_arguments_key(
        arguments=described.arguments
    )


def test_committed_layout_and_distinct_layout_identity_are_preserved() -> None:
    mesh = jax.make_mesh((1,), ("source",))
    original_layout = jax.NamedSharding(mesh, jax.P())
    required = jax.NamedSharding(jax.make_mesh((1,), ("different",)), jax.P())
    original = jax.device_put(jnp.arange(4.0), original_layout)
    program = _program({"payload": original, "scalar": original})

    described = abstract_program_inputs(
        program=program, transfers=(), execution_sharding=required
    )

    descriptor = described.arguments["payload"]
    assert isinstance(descriptor, jax.ShapeDtypeStruct)
    assert descriptor.sharding == original_layout
    changed = {
        **described.arguments,
        "payload": jax.ShapeDtypeStruct(
            original.shape, original.dtype, sharding=required
        ),
    }
    assert _abstract_arguments_key(arguments=changed) != _abstract_arguments_key(
        arguments=described.arguments
    )


def test_canonical_array_and_descriptor_metadata_requires_no_new_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = jnp.asarray(3.0)
    descriptor = jax.ShapeDtypeStruct(
        original.shape,
        original.dtype,
        weak_type=original.weak_type,
        sharding=original.sharding,
    )
    program = _program({"payload": original, "scalar": descriptor})

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Canonical descriptor construction attempted tracing.")

    monkeypatch.setattr(jax, "eval_shape", forbidden)
    abstract = abstract_program_inputs(
        program=program, transfers=(), execution_sharding=original.sharding
    )
    assert all(
        isinstance(leaf, jax.ShapeDtypeStruct)
        for leaf in jax.tree.leaves(abstract.arguments)
    )


@pytest.mark.parametrize(
    "value", [3, 3.0, np.int64(3), np.asarray([3.0], dtype=np.float64)]
)
def test_host_metadata_keeps_jax_dtype_and_weak_type_without_upload(
    *, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    expected = jax.eval_shape(lambda item: item, value)
    layout = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    program = _program({"payload": value, "scalar": value})

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Host descriptor construction attempted a device upload.")

    monkeypatch.setattr(jax, "device_put", forbidden)
    abstract = abstract_program_inputs(
        program=program, transfers=(), execution_sharding=layout
    )
    for leaf in jax.tree.leaves(abstract.arguments):
        assert leaf.shape == expected.shape
        assert leaf.dtype == expected.dtype
        assert leaf.weak_type == expected.weak_type
        assert leaf.sharding == layout


@pytest.mark.parametrize("container", ["tuple", "dataclass"])
def test_exact_occurrences_of_one_original_keep_distinct_required_layouts(
    *, monkeypatch: pytest.MonkeyPatch, container: str
) -> None:
    original = jnp.arange(4.0)
    source_layout = original.sharding
    destinations = tuple(
        jax.NamedSharding(jax.make_mesh((1,), (name,)), jax.P())
        for name in ("first", "second")
    )
    paths = ((0,), (1,)) if container == "tuple" else (("values",), ("other",))
    payload = (
        (original, original) if container == "tuple" else _Payload(original, original)
    )
    target = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="future"
    )
    reads = tuple(
        ValueRead(
            target=target,
            source=ValueConsumerAddress(
                channel=ValueInputChannel.NEXT_REGIME_VALUE,
                argument="payload",
                path=("nested", *path),
                source_period=0,
                source_regime="current",
                core_key="main",
            ),
        )
        for path in paths
    )
    transfers = tuple(
        resolve_value_transfer(
            target=target,
            source=read.source,
            kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
            stored_template=original,
            source_sharding=destination,
        )
        for read, destination in zip(reads, destinations, strict=True)
    )
    program = replace(
        _program(
            {"payload": MappingProxyType({"nested": payload}), "scalar": original}
        ),
        requirements=CoreExecutionRequirements(value_reads=reads),
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("A descriptor builder attempted a concrete allocation.")

    monkeypatch.setattr(jax, "device_put", forbidden)
    described = abstract_program_inputs(
        program=program, transfers=transfers, execution_sharding=source_layout
    )
    resolved = resolve_core_program(
        program=described, input_transfer_plan=transfers, abstract_inputs=True
    )

    leaves = jax.tree.leaves(described.arguments["payload"])
    assert tuple(leaf.sharding for leaf in leaves) == destinations
    assert leaves[0] is not leaves[1]
    assert all(
        isinstance(leaf, jax.ShapeDtypeStruct)
        for leaf in jax.tree.leaves(described.arguments)
    )
    assert jax.tree.structure(described.arguments) == jax.tree.structure(
        program.arguments
    )
    assert resolved.input_transfer_plan == transfers
    assert tuple(item.cost for item in resolved.input_transfer_plan) == tuple(
        item.cost for item in transfers
    )
    assert original.sharding == source_layout
    np.testing.assert_array_equal(original, np.arange(4.0))
