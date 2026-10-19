"""One fixed-current floor bank shared by every nested outer/action producer."""

import inspect
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp

from _lcm.dtypes import canonical_float_dtype
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    shift_savings_grid,
)
from _lcm.typing import EconFunctionKwargs, PytreeValue
from lcm.typing import Float1D, FloatND, Int1D, ScalarFloat, ScalarInt, StateName

NESTED_FLOOR_INPUT = "_lcm_prepared_nested_floor"

type PreparedNestedFloor = BoundedResult[FloatND]


def prepare_nested_floor(
    *,
    lower: Callable[..., ScalarFloat],
    offsets: Float1D,
    state_names: tuple[StateName, ...],
    **arguments: PytreeValue,
) -> PreparedNestedFloor:
    """Evaluate and validate b(C), retaining only its C-shaped floor/status."""
    grids = tuple(jnp.asarray(arguments[name]) for name in state_names)
    shape = tuple(grid.size for grid in grids)
    coordinates = tuple(
        grid.reshape(-1) for grid in jnp.meshgrid(*grids, indexing="ij")
    )
    params = {
        name: value for name, value in arguments.items() if name not in state_names
    }
    names = tuple(inspect.signature(lower).parameters)

    def at_cell(index: ScalarInt) -> tuple[ScalarFloat, BoundInputStatus]:
        pool = {
            **params,
            **{
                name: coordinate[index]
                for name, coordinate in zip(state_names, coordinates, strict=True)
            },
        }
        raw_floor = jnp.asarray(lower(**{name: pool[name] for name in names}))
        if raw_floor.shape != () or not jnp.issubdtype(raw_floor.dtype, jnp.floating):
            raise TypeError("A nested current floor must be one real floating scalar.")
        floor = raw_floor.astype(canonical_float_dtype())
        _, status = shift_savings_grid(floor=floor, offsets=offsets)
        invalid = status.nonfinite_floor | status.invalid_shifted_grid
        return (
            jnp.where(invalid, 0.0, floor),
            status,
        )

    floors, status = jax.vmap(at_cell)(jnp.arange(math.prod(shape), dtype=jnp.int32))
    return BoundedResult(
        output=floors.reshape(shape),
        bound_status=(jax.tree.map(lambda flag: flag.reshape(shape), status),),
    )


def nested_floor_at_cell(
    *,
    prepared: PreparedNestedFloor,
    offsets: Float1D,
    axes: tuple[tuple[StateName, Float1D | Int1D], ...],
    pool: EconFunctionKwargs,
) -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
    """Index the bank only at the producer's actual fixed-current grid coordinates."""
    if type(prepared) is not BoundedResult or len(prepared.bound_status) != 1:
        raise TypeError(
            "A nested floor operand must retain its canonical input status."
        )
    positions = []
    matched = jnp.asarray(a=True)
    for name, grid in axes:
        matches = grid == pool[name]
        position = jnp.argmax(matches)
        # Category declarations may retain their own nonascending code order.
        matched &= jnp.sum(matches, dtype=jnp.int32) == 1
        positions.append(position)
    index = tuple(positions)
    floor = prepared.output[index]
    nodes, local_status = shift_savings_grid(floor=floor, offsets=offsets)
    status = prepared.bound_status[0]
    invalid_grid = (
        status.invalid_shifted_grid[index]
        | local_status.invalid_shifted_grid
        | ~matched
    )
    invalid_floor = status.nonfinite_floor[index] | local_status.nonfinite_floor
    return (
        floor,
        jnp.where(
            invalid_floor | invalid_grid,
            jnp.arange(offsets.size, dtype=offsets.dtype),
            nodes,
        ),
        BoundInputStatus(
            nonfinite_floor=invalid_floor,
            invalid_shifted_grid=invalid_grid,
        ),
    )
