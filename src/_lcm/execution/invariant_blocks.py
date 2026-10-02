"""Runtime pieces of solving an invariant state one code at a time.

A regime carrying a state that `ExecutionConfig.invariant_block_widths` blocks
declares one program per code, each bound by an `InvariantBinding`. These
helpers give such a program what the engine hands it:

- `block_state_action_space`: the state grid narrowed to the bound code, a
  slice of the original array, so the code enters arithmetic with the grid's
  own dtype and weak typing;
- `selected_block_view`: the view a read of a stored value carrying the state
  takes, which selects the bound code and removes that axis;
- `block_value_template`: the shape and layout a block's output is born in;
- `write_block`: the step that places a block's output into the regime's
  complete value.

The code is data of the shared executable, never a compilation constant.
"""

import dataclasses
import functools
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.execution.core_program import InvariantBinding
from _lcm.execution.value_transfer import (
    CoordinateSelection,
    ValueArtifactAddress,
    ValueViewDescriptor,
    ValueViewLeaf,
)
from _lcm.typing import StateName
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import FloatND


def block_state_action_space[SpaceT](
    *, space: SpaceT, binding: InvariantBinding
) -> SpaceT:
    """Return `space` with the bound state's grid narrowed to the bound code."""
    states = space.states  # ty: ignore[unresolved-attribute]
    return dataclasses.replace(
        space,  # ty: ignore[invalid-argument-type]
        states=MappingProxyType(
            {
                name: (
                    jax.lax.slice_in_dim(value, binding.start, binding.start + 1)
                    if name == binding.state_name
                    else value
                )
                for name, value in states.items()
            }
        ),
    )


def block_value_template(*, template: object, axis: int) -> jax.ShapeDtypeStruct:
    """Describe one block of a regime value: the bound axis at length one."""
    shape = tuple(int(size) for size in template.shape)  # ty: ignore[unresolved-attribute]
    return jax.ShapeDtypeStruct(
        (*shape[:axis], 1, *shape[axis + 1 :]),
        template.dtype,  # ty: ignore[unresolved-attribute]
        sharding=template.sharding,  # ty: ignore[unresolved-attribute]
    )


def selected_block_view(
    *,
    artifact: ValueArtifactAddress,
    binding: InvariantBinding,
    stored_axis_names: tuple[StateName, ...],
    stored_template: object,
    required_sharding: jax.sharding.Sharding,
) -> ValueViewDescriptor:
    """Describe the read of the bound code's block of a value that carries the state.

    The bound axis is removed; the consumer receives the remaining axes in
    their stored order.
    """
    shape = tuple(int(size) for size in stored_template.shape)  # ty: ignore[unresolved-attribute]
    axis = stored_axis_names.index(binding.state_name)
    return ValueViewDescriptor(
        artifact=artifact,
        leaf=ValueViewLeaf.SELECTED,
        stored_axis_names=stored_axis_names,
        stored_shape=shape,
        dtype=stored_template.dtype,  # ty: ignore[unresolved-attribute]
        weak_type=bool(getattr(stored_template, "weak_type", False)),
        consumer_shape=(*shape[:axis], *shape[axis + 1 :]),
        required_sharding=required_sharding,
        selections=(
            CoordinateSelection(
                state_name=binding.state_name,
                start=binding.start,
                width=1,
                codes=(binding.code,),
            ),
        ),
    )


def block_layout(
    *, layout: jax.sharding.Sharding, axis: int, ndim: int
) -> jax.sharding.Sharding:
    """Return the layout a stored value keeps once its unpartitioned `axis` is removed.

    Raises:
        ExecutionPlanningError: The axis is partitioned over devices.

    """
    if not isinstance(layout, jax.NamedSharding):
        return layout
    entries = [*layout.spec, *([None] * (ndim - len(layout.spec)))]
    if entries[axis] is not None:
        msg = (
            "An invariant block cannot be selected along an axis partitioned over "
            f"{entries[axis]!r}."
        )
        raise ExecutionPlanningError(msg)
    del entries[axis]
    return jax.NamedSharding(
        layout.mesh, jax.P(*entries), memory_kind=layout.memory_kind
    )


def write_block(
    *,
    value: FloatND | None,
    block: FloatND,
    binding: InvariantBinding,
    axis: int,
    n_codes: int,
    sharding: jax.sharding.Sharding,
) -> FloatND:
    """Place one block's output into the regime's complete value.

    The first block written allocates the complete value on `sharding`, the
    layout of the regime's value template; every later block overwrites its
    own position in that buffer, which is donated, so the complete value is
    never held twice.
    """
    if value is None:
        return _tile_block(block=block, n_codes=n_codes, axis=axis, sharding=sharding)
    return _write_block(
        value=value,
        block=block,
        start=jnp.int32(binding.start),
        axis=axis,
        sharding=sharding,
    )


@functools.partial(jax.jit, static_argnames=("n_codes", "axis", "sharding"))
def _tile_block(
    *, block: FloatND, n_codes: int, axis: int, sharding: jax.sharding.Sharding
) -> FloatND:
    """Allocate the complete value by repeating one block along its axis."""
    return jax.lax.with_sharding_constraint(
        jnp.repeat(block, n_codes, axis=axis), sharding
    )


@functools.partial(
    jax.jit, static_argnames=("axis", "sharding"), donate_argnames=("value",)
)
def _write_block(
    *,
    value: FloatND,
    block: FloatND,
    start: jax.Array,
    axis: int,
    sharding: jax.sharding.Sharding,
) -> FloatND:
    """Overwrite one position of the complete value."""
    return jax.lax.with_sharding_constraint(
        jax.lax.dynamic_update_slice_in_dim(value, block, start, axis=axis), sharding
    )
