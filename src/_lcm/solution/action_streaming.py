"""Blockwise-action equivalence experiment for ordinary singleton grid search.

This is deliberately not wired into the production GridSearch path.  It isolates one
planner-owned execution choice: enumerate the canonical action product in fixed-width
blocks and combine the results with a mergeable hard-max state. The experiment proves
only value, feasibility, and action-identity equivalence. It makes no runtime or peak-
memory claim: compiler fusion, rematerialization, and allocation still require direct
measurement. The experiment excludes taste shocks, collective scalarization, outer
state mapping, and folded state axes; those semantics must be layered on only after this
kernel is established as an exact ordinary-singleton reference.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from _lcm.solution.action_reduction import (
    HARD_MAX_REDUCTION,
    HardMaxAccumulator,
    HardMaxResult,
)

_INT32_MAX = 2_147_483_647
_Block = tuple[jax.Array, jax.Array, jax.Array]
_ScanCarry = tuple[HardMaxAccumulator, jax.Array]


def build_streaming_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    block_width: int,
) -> Callable[..., HardMaxResult]:
    """Build a private fixed-state blockwise hard-max equivalence callable.

    ``action_names`` defines the canonical product order: the final action is the
    fastest-moving coordinate, exactly as C-order flattening of the corresponding
    product-map output.  ``block_width`` is an internal build decision standing in for
    a future graph-wide planner choice.  It intentionally appears on neither a grid
    nor a solver's public configuration.

    The returned callable accepts one one-dimensional grid for each action name plus
    the scalar state, continuation, and parameter arguments consumed by ``Q_and_F``.
    It operates on one fixed state cell.  Callers may map it over state cells.

    At the source-program level, ``Q_and_F`` is vmapped over one block at a time and a
    padded final block is marked infeasible before reduction. The scan emits ``None``
    as its history. Those facts establish the equivalence program's structure, not an
    executable runtime or peak-memory bound after compiler transformation. The
    reduction deliberately retains GridSearch's legacy feasible-NaN behavior: a NaN
    maximum publishes action identity zero, even when that action is infeasible.
    """
    if (
        not isinstance(block_width, int)
        or isinstance(block_width, bool)
        or block_width <= 0
    ):
        raise ValueError("block_width must be positive")
    if block_width > _INT32_MAX:
        raise ValueError("block_width exceeds the int32 identity range")
    if len(set(action_names)) != len(action_names):
        raise ValueError("action_names must not contain duplicates")
    return _StreamingHardMax(
        Q_and_F=Q_and_F,
        action_names=action_names,
        block_width=block_width,
    )


@dataclass(frozen=True)
class _StreamingHardMax:
    """Configured private action-streaming callable."""

    Q_and_F: Callable[..., tuple[Any, Any]]
    action_names: tuple[str, ...]
    block_width: int

    def __call__(self, **kwargs: Any) -> HardMaxResult:  # noqa: ANN401
        if not self.action_names:
            return _reduce_no_action(Q_and_F=self.Q_and_F, kwargs=kwargs)

        action_grids, fixed_kwargs, action_sizes, n_actions = _prepare_action_call(
            action_names=self.action_names,
            kwargs=kwargs,
        )
        n_blocks = (n_actions + self.block_width - 1) // self.block_width
        evaluate_block = partial(
            _evaluate_block,
            Q_and_F=self.Q_and_F,
            action_names=self.action_names,
            action_grids=action_grids,
            action_sizes=action_sizes,
            fixed_kwargs=fixed_kwargs,
            n_actions=n_actions,
            block_width=self.block_width,
            block_offsets=jnp.arange(self.block_width, dtype=jnp.int32),
        )
        first_block = evaluate_block(block_index=jnp.asarray(0, dtype=jnp.int32))
        accumulator = _start_reduction(block=first_block)
        accumulator = _scan_remaining_blocks(
            accumulator=accumulator,
            evaluate_block=evaluate_block,
            n_remaining=n_blocks - 1,
        )
        return HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def _prepare_action_call(
    *, action_names: tuple[str, ...], kwargs: dict[str, Any]
) -> tuple[tuple[jax.Array, ...], dict[str, Any], tuple[int, ...], int]:
    """Validate grids and split them from scalar Q arguments."""
    missing = tuple(name for name in action_names if name not in kwargs)
    if missing:
        raise TypeError(f"Missing action-grid arguments: {missing}")

    action_grids = tuple(jnp.asarray(kwargs[name]) for name in action_names)
    for name, grid in zip(action_names, action_grids, strict=True):
        if grid.ndim != 1:
            raise ValueError(f"Action grid '{name}' must be one-dimensional")
        if grid.shape[0] == 0:
            raise ValueError(f"Action grid '{name}' must not be empty")

    fixed_kwargs = {
        name: value for name, value in kwargs.items() if name not in action_names
    }
    action_sizes = tuple(grid.shape[0] for grid in action_grids)
    n_actions = math.prod(action_sizes)
    if n_actions > _INT32_MAX:
        raise ValueError(
            "The canonical action product exceeds the int32 identity range"
        )
    return action_grids, fixed_kwargs, action_sizes, n_actions


def _evaluate_block(
    *,
    block_index: jax.Array,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    action_grids: tuple[jax.Array, ...],
    action_sizes: tuple[int, ...],
    fixed_kwargs: dict[str, Any],
    n_actions: int,
    block_width: int,
    block_offsets: jax.Array,
) -> _Block:
    """Evaluate one padded block, never the complete action product."""
    block_start = block_index * block_width
    remaining = n_actions - block_start
    valid = block_offsets < remaining
    safe_offsets = jnp.minimum(block_offsets, remaining - 1)
    global_ids = block_start + safe_offsets

    def evaluate_one(global_id: jax.Array) -> tuple[Any, Any]:
        action_kwargs = _decode_action(
            global_id=global_id,
            action_names=action_names,
            action_grids=action_grids,
            action_sizes=action_sizes,
        )
        return Q_and_F(**fixed_kwargs, **action_kwargs)

    values, feasible = jax.vmap(evaluate_one)(global_ids)
    values = jnp.asarray(values)
    feasible = jnp.asarray(feasible)
    _validate_block_Q_and_F(values=values, feasible=feasible)
    return values, feasible & valid, global_ids


def _start_reduction(*, block: _Block) -> HardMaxAccumulator:
    """Seed a hard-max reduction from the first evaluated block."""
    values, feasible, global_ids = block
    accumulator = HARD_MAX_REDUCTION.initialize(
        value_template=jnp.zeros_like(values[0])
    )
    return HARD_MAX_REDUCTION.add(
        accumulator=accumulator,
        values=values,
        feasible=feasible,
        action_ids=global_ids,
    )


def _scan_remaining_blocks(
    *,
    accumulator: HardMaxAccumulator,
    evaluate_block: Callable[..., _Block],
    n_remaining: int,
) -> HardMaxAccumulator:
    """Use a source-level scan whose returned history is the ``None`` pytree."""

    # keyword-only-exempt: library-callback=jax.lax.scan
    def scan_one_block(carry: _ScanCarry, _unused: None) -> tuple[_ScanCarry, None]:
        partial_accumulator, block_index = carry
        block = evaluate_block(block_index=block_index)
        partial_accumulator = _add_block(accumulator=partial_accumulator, block=block)
        return (partial_accumulator, block_index + 1), None

    (accumulator, _), _history = jax.lax.scan(
        scan_one_block,
        (accumulator, jnp.asarray(1, dtype=jnp.int32)),
        xs=None,
        length=n_remaining,
    )
    return accumulator


def _add_block(*, accumulator: HardMaxAccumulator, block: _Block) -> HardMaxAccumulator:
    """Merge one evaluated block into the hard-max state."""
    values, feasible, global_ids = block
    return HARD_MAX_REDUCTION.add(
        accumulator=accumulator,
        values=values,
        feasible=feasible,
        action_ids=global_ids,
    )


def _reduce_no_action(
    *, Q_and_F: Callable[..., tuple[Any, Any]], kwargs: dict[str, Any]
) -> HardMaxResult:
    """Treat an empty action product as the one-cell identity product."""
    value, feasible = Q_and_F(**kwargs)
    value = jnp.asarray(value)
    feasible = jnp.asarray(feasible)
    _validate_scalar_Q_and_F(value=value, feasible=feasible)
    block = (
        value[jnp.newaxis],
        feasible[jnp.newaxis],
        jnp.array([0], dtype=jnp.int32),
    )
    accumulator = _start_reduction(block=block)
    return HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def _decode_action(
    *,
    global_id: jax.Array,
    action_names: tuple[str, ...],
    action_grids: tuple[jax.Array, ...],
    action_sizes: tuple[int, ...],
) -> dict[str, jax.Array]:
    """Decode a C-order global identity without materializing the product."""
    stride = math.prod(action_sizes)
    out: dict[str, jax.Array] = {}
    for name, grid, size in zip(action_names, action_grids, action_sizes, strict=True):
        stride //= size
        coordinate = (global_id // stride) % size
        out[name] = grid[coordinate]
    return out


def _validate_scalar_Q_and_F(*, value: jax.Array, feasible: jax.Array) -> None:
    """Validate the no-action identity against the experiment's scalar contract."""
    if value.ndim != 0 or feasible.ndim != 0:
        raise ValueError(
            "The ordinary-singleton streaming experiment requires scalar Q and "
            "feasibility outputs at each action cell"
        )
    if feasible.dtype != jnp.bool_:
        raise TypeError("Q_and_F feasibility output must have boolean dtype")


def _validate_block_Q_and_F(*, values: jax.Array, feasible: jax.Array) -> None:
    """Validate a vmapped block against the experiment's scalar cell contract."""
    if values.ndim != 1 or feasible.ndim != 1:
        raise ValueError(
            "The ordinary-singleton streaming experiment requires scalar Q and "
            "feasibility outputs at each action cell"
        )
    if feasible.dtype != jnp.bool_:
        raise TypeError("Q_and_F feasibility output must have boolean dtype")
