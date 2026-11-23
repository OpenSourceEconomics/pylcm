"""Exact blockwise collective choice under one global household identity."""

from typing import Literal, NamedTuple

import jax.numpy as jnp

from _lcm.regime_building.argmax import NO_ID, max_and_smallest_id
from lcm.typing import BoolND, FloatND, IntND


class CollectiveHardMaxAccumulator(NamedTuple):
    """Mergeable household maximum and member values at one shared winner."""

    best_objective: FloatND
    best_member_values: FloatND
    best_global_action_id: IntND
    any_feasible: BoolND
    action_zero_member_values: FloatND
    action_zero_seen: BoolND
    any_feasible_nan: BoolND


class CollectiveHardMaxResult(NamedTuple):
    """Household winner, own-value readout, identity, and feasibility state."""

    best_objective: FloatND
    best_member_values: FloatND
    best_global_action_id: IntND
    any_feasible: BoolND


class CollectiveHardMaxReduction:
    """Merge collective action blocks without changing the household choice.

    ``objectives`` and ``feasible`` carry the action block on their final axis;
    ``member_values`` carries the same block on its penultimate axis and a
    trailing member axis. The household objective alone chooses the winner.
    Every member value is then read at that one global action identity.

    A feasible NaN preserves the dense GridSearch rule: the masked maximum is NaN,
    its equality mask is all false, and positional argmax publishes global action
    zero. The accumulator therefore retains action zero's member values even
    when they occur in a different block or action zero itself is infeasible.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Stable identity of the collective hard-max contract."""
        return ("collective-hard-max", 1)

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: the max with lowest-id tie-break is order independent."""
        return "exact"

    def initialize(self, *, member_template: FloatND) -> CollectiveHardMaxAccumulator:
        """Create an empty accumulator from a state-by-member template."""
        if member_template.ndim < 1 or member_template.shape[-1] == 0:
            raise ValueError(
                "member_template must have a non-empty trailing member axis"
            )
        value_template = member_template[..., 0]
        return CollectiveHardMaxAccumulator(
            best_objective=jnp.full_like(value_template, -jnp.inf),
            best_member_values=jnp.full_like(member_template, -jnp.inf),
            best_global_action_id=jnp.full(value_template.shape, -1, dtype=jnp.int32),
            any_feasible=jnp.zeros(value_template.shape, dtype=bool),
            action_zero_member_values=jnp.full_like(member_template, -jnp.inf),
            action_zero_seen=jnp.zeros(value_template.shape, dtype=bool),
            any_feasible_nan=jnp.zeros(value_template.shape, dtype=bool),
        )

    def add(
        self,
        *,
        accumulator: CollectiveHardMaxAccumulator,
        objectives: FloatND,
        member_values: FloatND,
        feasible: BoolND,
        action_ids: IntND,
    ) -> CollectiveHardMaxAccumulator:
        """Reduce one collective action block and merge it into ``accumulator``."""
        if action_ids.dtype != jnp.dtype(jnp.int32):
            msg = (
                "CollectiveHardMaxReduction.add action_ids must have dtype int32; "
                f"got {action_ids.dtype}."
            )
            raise TypeError(msg)
        _validate_block_shapes(
            objectives=objectives,
            member_values=member_values,
            accumulator=accumulator,
        )
        block = _reduce_block(
            objectives=objectives,
            member_values=member_values,
            feasible=jnp.broadcast_to(feasible, objectives.shape),
            action_ids=jnp.broadcast_to(action_ids, objectives.shape),
        )
        return self.merge(left=accumulator, right=block)

    def merge(
        self,
        *,
        left: CollectiveHardMaxAccumulator,
        right: CollectiveHardMaxAccumulator,
    ) -> CollectiveHardMaxAccumulator:
        """Merge partial household choices independently of block schedule."""
        left_nan = jnp.isnan(left.best_objective)
        right_nan = jnp.isnan(right.best_objective)
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
                    (right.best_objective > left.best_objective)
                    | (
                        (right.best_objective == left.best_objective)
                        & (right.best_global_action_id < left.best_global_action_id)
                    )
                )
            )
        )
        choose_right = right.any_feasible & (
            ~left.any_feasible | (left.any_feasible & right_wins_both_feasible)
        )
        choose_right_zero = right.action_zero_seen & ~left.action_zero_seen
        selected_best_objective = jnp.where(
            choose_right, right.best_objective, left.best_objective
        )
        both_feasible_zero = (
            left.any_feasible
            & right.any_feasible
            & (left.best_objective == 0)
            & (right.best_objective == 0)
        )
        signed_zero_max = jnp.where(
            jnp.signbit(left.best_objective) & jnp.signbit(right.best_objective),
            -jnp.zeros_like(left.best_objective),
            jnp.zeros_like(left.best_objective),
        )

        return CollectiveHardMaxAccumulator(
            best_objective=jnp.where(
                both_feasible_zero,
                signed_zero_max,
                selected_best_objective,
            ),
            best_member_values=jnp.where(
                choose_right[..., jnp.newaxis],
                right.best_member_values,
                left.best_member_values,
            ),
            best_global_action_id=jnp.where(
                choose_right,
                right.best_global_action_id,
                left.best_global_action_id,
            ),
            any_feasible=left.any_feasible | right.any_feasible,
            action_zero_member_values=jnp.where(
                choose_right_zero[..., jnp.newaxis],
                right.action_zero_member_values,
                left.action_zero_member_values,
            ),
            action_zero_seen=left.action_zero_seen | right.action_zero_seen,
            any_feasible_nan=left.any_feasible_nan | right.any_feasible_nan,
        )

    def finalize(
        self, *, accumulator: CollectiveHardMaxAccumulator
    ) -> CollectiveHardMaxResult:
        """Publish the household winner and every member's value there."""
        own_values = jnp.where(
            accumulator.any_feasible_nan[..., jnp.newaxis],
            accumulator.action_zero_member_values,
            accumulator.best_member_values,
        )
        return CollectiveHardMaxResult(
            best_objective=jnp.where(
                accumulator.any_feasible,
                accumulator.best_objective,
                jnp.full_like(accumulator.best_objective, -jnp.inf),
            ),
            best_member_values=jnp.where(
                accumulator.any_feasible[..., jnp.newaxis],
                own_values,
                jnp.full_like(own_values, -jnp.inf),
            ),
            best_global_action_id=jnp.where(
                accumulator.any_feasible,
                accumulator.best_global_action_id,
                jnp.full_like(accumulator.best_global_action_id, -1),
            ),
            any_feasible=accumulator.any_feasible,
        )


def _validate_block_shapes(
    *,
    objectives: FloatND,
    member_values: FloatND,
    accumulator: CollectiveHardMaxAccumulator,
) -> None:
    """Check the state, block, and member axes of one collective block."""
    if member_values.ndim != objectives.ndim + 1:
        raise ValueError(
            "member_values must add one trailing member axis to objectives"
        )
    if member_values.shape[:-2] != objectives.shape[:-1]:
        raise ValueError("objectives and member_values state axes must match")
    if member_values.shape[-2] != objectives.shape[-1]:
        raise ValueError("objectives and member_values block widths must match")
    expected = (*objectives.shape[:-1], member_values.shape[-1])
    if accumulator.best_member_values.shape != expected:
        raise ValueError("collective block does not match the accumulator template")


def _reduce_block(
    *,
    objectives: FloatND,
    member_values: FloatND,
    feasible: BoolND,
    action_ids: IntND,
) -> CollectiveHardMaxAccumulator:
    """Reduce one collective block and retain action zero's dense readout."""
    feasible_nan = feasible & jnp.isnan(objectives)
    any_feasible = jnp.any(feasible, axis=-1)
    any_feasible_nan = jnp.any(feasible_nan, axis=-1)

    comparable = feasible & ~feasible_nan
    best_non_nan, best_global_action_id = max_and_smallest_id(
        values=jnp.where(comparable, objectives, -jnp.inf),
        ids=jnp.where(comparable, action_ids, NO_ID),
        initial=-jnp.inf,
    )
    best_objective = jnp.where(
        any_feasible_nan,
        jnp.full_like(best_non_nan, jnp.nan),
        best_non_nan,
    )
    winner = comparable & (action_ids == best_global_action_id[..., jnp.newaxis])
    best_global_action_id = jnp.where(any_feasible_nan, 0, best_global_action_id)
    winner_position = jnp.asarray(
        jnp.argmax(
            winner & (action_ids == best_global_action_id[..., jnp.newaxis]),
            axis=-1,
        ),
        dtype=jnp.int32,
    )
    best_member_values = _take_member_values(
        member_values=member_values,
        positions=winner_position,
    )

    action_zero = action_ids == 0
    action_zero_seen = jnp.any(action_zero, axis=-1)
    action_zero_member_values = _take_member_values(
        member_values=member_values,
        positions=jnp.asarray(
            jnp.argmax(action_zero, axis=-1),
            dtype=jnp.int32,
        ),
    )
    action_zero_member_values = jnp.where(
        action_zero_seen[..., jnp.newaxis],
        action_zero_member_values,
        jnp.full_like(action_zero_member_values, -jnp.inf),
    )

    return CollectiveHardMaxAccumulator(
        best_objective=jnp.where(
            any_feasible,
            best_objective,
            jnp.full_like(best_objective, -jnp.inf),
        ),
        best_member_values=jnp.where(
            any_feasible[..., jnp.newaxis],
            best_member_values,
            jnp.full_like(best_member_values, -jnp.inf),
        ),
        best_global_action_id=jnp.where(
            any_feasible,
            best_global_action_id,
            jnp.full_like(best_global_action_id, -1),
        ),
        any_feasible=any_feasible,
        action_zero_member_values=action_zero_member_values,
        action_zero_seen=action_zero_seen,
        any_feasible_nan=any_feasible_nan,
    )


def _take_member_values(*, member_values: FloatND, positions: IntND) -> FloatND:
    """Take one action position while retaining the member axis."""
    indices = jnp.broadcast_to(
        positions[..., jnp.newaxis, jnp.newaxis],
        (*positions.shape, 1, member_values.shape[-1]),
    )
    return jnp.take_along_axis(member_values, indices, axis=-2)[..., 0, :]


COLLECTIVE_HARD_MAX_REDUCTION = CollectiveHardMaxReduction()
# Shared exact collective hard-max reduction specification.
