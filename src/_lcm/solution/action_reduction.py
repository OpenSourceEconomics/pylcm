"""Exact blockwise reductions over a canonical global action order.

The execution planner may partition an action product into blocks and process those
blocks in any legal order.  This module owns the mathematical reduction state that
makes such a schedule observationally equivalent to one scalar pass over the complete
action order.  It deliberately knows nothing about grid construction, block width, or
device placement.
"""

from typing import NamedTuple

import jax.numpy as jnp

from lcm.typing import BoolND, FloatND, IntND


class HardMaxAccumulator(NamedTuple):
    """Mergeable state of an exact hard maximum over action blocks."""

    best_value: FloatND
    """Best feasible value seen so far, or ``-inf`` before one is seen."""

    best_global_action_id: IntND
    """Canonical global identity of the winner, or ``-1`` if none is feasible."""

    any_feasible: BoolND
    """Whether at least one feasible action has been seen."""


class HardMaxResult(NamedTuple):
    """Final value, replay identity, and explicit feasibility state."""

    best_value: FloatND
    best_global_action_id: IntND
    any_feasible: BoolND


class HardMaxReduction:
    """Exact hard maximum whose merge law is independent of block scheduling.

    Blocks occupy the last axis of ``values``. ``feasible`` and ``action_ids`` must
    broadcast to that shape; action identities are canonical global ``int32`` values,
    not positions within a block. Equal values select the smallest global identity.

    Feasibility is applied before comparison, so a feasible ``-inf`` remains distinct
    from an all-infeasible cell. A feasible NaN preserves GridSearch's existing
    full-array behavior: the maximum is NaN and the all-false equality mask makes
    ``argmax`` publish global identity zero, even if action zero is infeasible. This is
    a legacy-compatibility rule, not a preferred mathematical treatment of NaNs.
    """

    def initialize(self, *, value_template: FloatND) -> HardMaxAccumulator:
        """Create an empty accumulator with ``value_template``'s shape and dtype."""
        return HardMaxAccumulator(
            best_value=jnp.full_like(value_template, -jnp.inf),
            best_global_action_id=jnp.full(value_template.shape, -1, dtype=jnp.int32),
            any_feasible=jnp.zeros(value_template.shape, dtype=bool),
        )

    def add(
        self,
        *,
        accumulator: HardMaxAccumulator,
        values: FloatND,
        feasible: BoolND,
        action_ids: IntND,
    ) -> HardMaxAccumulator:
        """Reduce one action block and merge it into ``accumulator``."""
        if action_ids.dtype != jnp.dtype(jnp.int32):
            msg = (
                "HardMaxReduction.add action_ids must have dtype int32; "
                f"got {action_ids.dtype}."
            )
            raise TypeError(msg)
        block = _reduce_block(
            values=values,
            feasible=jnp.broadcast_to(feasible, values.shape),
            action_ids=jnp.broadcast_to(action_ids, values.shape),
        )
        return self.merge(left=accumulator, right=block)

    def merge(
        self,
        *,
        left: HardMaxAccumulator,
        right: HardMaxAccumulator,
    ) -> HardMaxAccumulator:
        """Merge partials independently of schedule, including legacy NaN id zero."""
        left_nan = jnp.isnan(left.best_value)
        right_nan = jnp.isnan(right.best_value)
        right_wins_both_feasible = (
            (right_nan & ~left_nan)
            | (
                right_nan
                & left_nan
                & (right.best_global_action_id < left.best_global_action_id)
            )
            | (
                ~right_nan
                & ~left_nan
                & (
                    (right.best_value > left.best_value)
                    | (
                        (right.best_value == left.best_value)
                        & (right.best_global_action_id < left.best_global_action_id)
                    )
                )
            )
        )
        choose_right = right.any_feasible & (
            ~left.any_feasible | (left.any_feasible & right_wins_both_feasible)
        )

        return HardMaxAccumulator(
            best_value=jnp.where(choose_right, right.best_value, left.best_value),
            best_global_action_id=jnp.where(
                choose_right,
                right.best_global_action_id,
                left.best_global_action_id,
            ),
            any_feasible=left.any_feasible | right.any_feasible,
        )

    def finalize(self, *, accumulator: HardMaxAccumulator) -> HardMaxResult:
        """Publish the winner while keeping the all-infeasible state explicit."""
        return HardMaxResult(
            best_value=jnp.where(
                accumulator.any_feasible,
                accumulator.best_value,
                jnp.full_like(accumulator.best_value, -jnp.inf),
            ),
            best_global_action_id=jnp.where(
                accumulator.any_feasible,
                accumulator.best_global_action_id,
                jnp.full_like(accumulator.best_global_action_id, -1),
            ),
            any_feasible=accumulator.any_feasible,
        )


def _reduce_block(
    *, values: FloatND, feasible: BoolND, action_ids: IntND
) -> HardMaxAccumulator:
    """Reduce one block without assuming its local order is canonical."""
    feasible_nan = feasible & jnp.isnan(values)
    any_feasible = jnp.any(feasible, axis=-1)
    any_nan = jnp.any(feasible_nan, axis=-1)

    comparable = jnp.where(feasible & ~feasible_nan, values, -jnp.inf)
    best_non_nan = jnp.max(comparable, axis=-1, initial=-jnp.inf)
    best_value = jnp.where(
        any_nan,
        jnp.full_like(best_non_nan, jnp.nan),
        best_non_nan,
    )

    winner = feasible & (values == jnp.expand_dims(best_non_nan, axis=-1))
    id_sentinel = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    best_global_action_id = jnp.min(
        jnp.where(winner, action_ids, id_sentinel),
        axis=-1,
        initial=id_sentinel,
    )
    # Full-array GridSearch obtains the identity from
    # ``argmax(feasible & (value == max_value))``. If max_value is NaN every
    # equality is false and argmax returns position zero. Keep that historical quirk
    # even for a block that does not contain global action zero so arbitrary block
    # schedules remain observationally equivalent after merging.
    best_global_action_id = jnp.where(any_nan, 0, best_global_action_id)

    return HardMaxAccumulator(
        best_value=jnp.where(
            any_feasible, best_value, jnp.full_like(best_value, -jnp.inf)
        ),
        best_global_action_id=jnp.where(
            any_feasible,
            best_global_action_id,
            jnp.full_like(best_global_action_id, -1),
        ),
        any_feasible=any_feasible,
    )


HARD_MAX_REDUCTION = HardMaxReduction()
# Shared exact hard-max reduction specification.
