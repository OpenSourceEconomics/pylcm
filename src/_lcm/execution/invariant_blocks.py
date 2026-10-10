"""Runtime pieces of solving an invariant state one code at a time.

A regime carrying a state that `ExecutionConfig.invariant_block_widths` blocks
declares one program per code, each bound by an `InvariantBinding`. These
helpers give such a program what the engine hands it:

- `block_state_action_space`: the state grid narrowed to the bound code,
  built on the host in the grid's own dtype and placed where the grid lives;
- `selected_block_view`: the view a read of a stored value carrying the state
  takes, which selects the bound code and removes that axis;
- `block_value_template`: the shape and layout a block's output is born in;
- `write_block`: the step that places a block's output into the regime's
  complete value.

The code is data of the shared executable, never a compilation constant. None
of these steps asks JAX for a program of its own except the block write.
"""

import dataclasses
import functools
from types import MappingProxyType

import jax
import numpy as np

from _lcm.execution.core_program import InvariantBinding
from _lcm.execution.runtime_sharding import runtime_shardings_match
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
    """Return `space` with the bound state's grid narrowed to the bound code.

    The one-element grid holds `binding.code`, the grid's element at
    `binding.start`, in the grid's dtype. It is put on the device from the
    host, so binding a code compiles no slicing program, and it is committed to
    the grid's sharding exactly when the grid is, so it joins any mesh the grid
    could join.

    Raises:
        TypeError: The grid is weakly typed, which a host-built array cannot
            reproduce.

    """
    states = space.states  # ty: ignore[unresolved-attribute]
    grid = states[binding.state_name]
    if getattr(grid, "weak_type", False):
        msg = (
            f"The grid of the invariant state {binding.state_name!r} is weakly "
            "typed; a bound code would change its arithmetic promotion."
        )
        raise TypeError(msg)
    return dataclasses.replace(
        space,  # ty: ignore[invalid-argument-type]
        states=MappingProxyType(
            {
                **states,
                binding.state_name: jax.device_put(
                    np.asarray([binding.code], dtype=grid.dtype),
                    grid.sharding if grid.committed else None,
                ),
            }
        ),
    )


def block_value_template(
    *, template: FloatND | jax.ShapeDtypeStruct, axis: int
) -> jax.ShapeDtypeStruct:
    """Describe one block of a regime value: the bound axis at length one."""
    shape = tuple(int(size) for size in template.shape)
    return jax.ShapeDtypeStruct(
        (*shape[:axis], 1, *shape[axis + 1 :]),
        template.dtype,
        sharding=template.sharding,
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
    template: FloatND,
) -> FloatND:
    """Place one block's output into the regime's complete value.

    The first block written goes into a device copy of `template`, the regime's
    value template, which `device_put` makes without compiling a program; every
    later block overwrites its own position in that buffer, which is donated, so
    the complete value is never held twice. Every position is written, so
    nothing of the template survives.
    """
    actual = block.sharding
    expected = template.sharding
    if (
        isinstance(actual, jax.NamedSharding)
        and isinstance(expected, jax.NamedSharding)
        and actual.mesh.axis_types != expected.mesh.axis_types
    ):
        if not runtime_shardings_match(
            actual=actual, expected=expected, ndim=block.ndim
        ):
            msg = "An invariant block's physical layout differs from its template."
            raise ExecutionPlanningError(msg)
        block = jax.device_put(
            block,
            jax.NamedSharding(
                expected.mesh, actual.spec, memory_kind=actual.memory_kind
            ),
        )
    return _write_block(
        value=(
            jax.device_put(template, template.sharding, may_alias=False)
            if value is None
            else value
        ),
        block=block,
        start=np.int32(binding.start),
        axis=axis,
        sharding=template.sharding,
    )


@functools.partial(
    jax.jit, static_argnames=("axis", "sharding"), donate_argnames=("value",)
)
def _write_block(
    *,
    value: FloatND,
    block: FloatND,
    start: jax.Array | np.integer,
    axis: int,
    sharding: jax.sharding.Sharding,
) -> FloatND:
    """Overwrite one position of the complete value."""
    return jax.lax.with_sharding_constraint(
        jax.lax.dynamic_update_slice_in_dim(value, block, start, axis=axis), sharding
    )
