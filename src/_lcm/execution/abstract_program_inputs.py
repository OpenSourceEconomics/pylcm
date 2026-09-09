"""Describe a solve core's required operands without materializing its transfers."""

from collections.abc import Mapping
from dataclasses import replace

import jax

from _lcm.execution.core_program import (
    MaterializedCoreProgram,
    ValueRead,
    _validate_transfer_argument_metadata,
    _value_read_argument_leaf,
)
from _lcm.execution.value_transfer import ResolvedValueTransfer


def abstract_program_inputs(
    *,
    program: MaterializedCoreProgram,
    transfers: tuple[ResolvedValueTransfer, ...],
    execution_sharding: jax.sharding.Sharding,
) -> MaterializedCoreProgram:
    """Replace every occurrence with exact shape, weak type and required layout.

    Committed ordinary operands retain their layout. Uncommitted inputs may
    reside incidentally on another device; like concrete lowering, they use the
    source execution mesh, replicated until an actual input layout says otherwise.
    Declared reads always use their resolved transfer destination.

    Each occurrence gets its own descriptor, even when original leaves alias.
    Resolving the existing declared locator on that tree identifies the exact
    occurrence without inventing a second mapping/tuple/dataclass path grammar.
    """
    shared = (
        jax.NamedSharding(execution_sharding.mesh, jax.P())
        if isinstance(execution_sharding, jax.NamedSharding)
        else execution_sharding
    )

    def describe(value: object) -> jax.ShapeDtypeStruct:
        abstract = (
            value
            if isinstance(value, (jax.Array, jax.ShapeDtypeStruct))
            else jax.eval_shape(_identity, value)
        )
        if not isinstance(abstract, (jax.Array, jax.ShapeDtypeStruct)):
            raise TypeError("A numerical operand must describe one array leaf.")
        sharding = (
            value.sharding
            if isinstance(value, jax.Array) and value.committed
            else shared
        )
        if isinstance(value, jax.ShapeDtypeStruct) and value.sharding is not None:
            sharding = value.sharding
        return jax.ShapeDtypeStruct(
            abstract.shape,
            abstract.dtype,
            weak_type=abstract.weak_type,
            sharding=sharding,
        )

    arguments = jax.tree.map(describe, program.arguments)
    described = replace(program, arguments=arguments)
    replacements: dict[int, jax.ShapeDtypeStruct] = {}
    for transfer in transfers:
        read = ValueRead(target=transfer.target, source=transfer.source)
        _validate_transfer_argument_metadata(
            program=program, read=read, transfer=transfer
        )
        occurrence = _value_read_argument_leaf(program=described, read=read)
        if id(occurrence) in replacements:
            raise ValueError(
                f"Duplicate abstract transfer locator: {transfer.source!r}."
            )
        replacements[id(occurrence)] = jax.ShapeDtypeStruct(
            occurrence.shape,
            occurrence.dtype,
            weak_type=getattr(occurrence, "weak_type", False),
            sharding=transfer.source_sharding,
        )
    required: Mapping[str, object] = jax.tree.map(
        lambda leaf: replacements.get(id(leaf), leaf), arguments
    )
    return replace(program, arguments=required)


def _identity(value: object) -> object:
    """Canonicalize host scalar/NumPy metadata through one stable JAX trace."""
    return value
