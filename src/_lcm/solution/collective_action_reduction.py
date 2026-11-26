"""Exact blockwise collective choice under one global action identity."""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from _lcm.regime_building.argmax import (
    NO_ID,
    max_and_any_nan,
    max_lexicographic_and_smallest_id,
)
from lcm.typing import BoolND, FloatND, IntND


class CollectiveHardMaxAccumulator(NamedTuple):
    """Mergeable collective maximum and member values at one shared winner."""

    best_objective: FloatND
    best_member_values: FloatND
    best_global_action_id: IntND
    """The winner's global identity, `NO_ID` before a feasible action is seen."""
    any_feasible: BoolND
    any_feasible_nan: BoolND


class CollectiveHardMaxResult(NamedTuple):
    """Collective winner, own-value readout, identity, and feasibility state."""

    best_objective: FloatND
    best_member_values: FloatND
    best_global_action_id: IntND
    any_feasible: BoolND


@dataclass(frozen=True, kw_only=True)
class CollectiveHardMaxReduction:
    """Merge collective action blocks without changing the collective choice.

    ``objectives`` and ``feasible`` carry the action block on their final axis;
    ``member_values`` carries the same block on its penultimate axis and a
    trailing member axis. The winner maximizes the collective objective; ties go
    to the members in declared order, then to the smallest global identity, as
    on the dense route. Every member value is read at that one winner.

    A feasible NaN objective has no collective value: the objective and every
    member value publish NaN and the identity is global action zero, as on the
    dense route.

    A regime with one member ranks its actions on that member's value alone,
    which is its objective (`ties_go_to_members` false).

    Where nothing reads the winner's identity, it keeps none
    (`keeps_identity` false): the identity is then `NO_ID`, and of a `-0` and a
    `+0` value it keeps `+0`.
    """

    ties_go_to_members: bool = True
    """Whether ties in the objective go to the member values.

    Without tie keys the value is the reduction's only key, so the compiler
    evaluates the action values inside the reduction instead of storing them
    for each action.
    """

    keeps_identity: bool = True
    """Whether the reduction carries the winner's identity.

    Without it, and without tie keys, a block reduces the value and the NaN
    flag alone.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Stable identity of the collective hard-max contract."""
        return ("collective-hard-max", 1)

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: the lexicographic max is order independent."""
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
            best_global_action_id=jnp.full(
                value_template.shape, NO_ID, dtype=jnp.int32
            ),
            any_feasible=jnp.zeros(value_template.shape, dtype=bool),
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
        feasible = jnp.broadcast_to(feasible, objectives.shape)
        block = _winner(
            objectives=objectives,
            member_values=member_values,
            ids=jnp.broadcast_to(action_ids, objectives.shape),
            where=feasible,
            ties_go_to_members=self.ties_go_to_members,
            keeps_identity=self.keeps_identity,
        )
        return self.merge(left=accumulator, right=block)

    def merge(
        self,
        *,
        left: CollectiveHardMaxAccumulator,
        right: CollectiveHardMaxAccumulator,
    ) -> CollectiveHardMaxAccumulator:
        """Merge partial collective choices independently of block schedule."""
        # A selection that never fires gives both partials the placement an
        # elementwise operation on them takes, so they can be stacked.
        never = jnp.zeros_like(left.any_feasible) & jnp.zeros_like(right.any_feasible)
        left, right = (
            jax.tree.map(
                lambda a, b: jnp.where(
                    jnp.reshape(never, never.shape + (1,) * (a.ndim - never.ndim)), b, a
                ),
                first,
                second,
            )
            for first, second in ((left, right), (right, left))
        )
        ids = jnp.stack(
            (left.best_global_action_id, right.best_global_action_id), axis=-1
        )
        return _winner(
            objectives=jnp.stack((left.best_objective, right.best_objective), axis=-1),
            member_values=jnp.stack(
                (left.best_member_values, right.best_member_values), axis=-2
            ),
            ids=ids,
            where=jnp.stack((left.any_feasible, right.any_feasible), axis=-1),
            ties_go_to_members=self.ties_go_to_members,
            keeps_identity=self.keeps_identity,
        )._replace(any_feasible_nan=left.any_feasible_nan | right.any_feasible_nan)

    def finalize(
        self, *, accumulator: CollectiveHardMaxAccumulator
    ) -> CollectiveHardMaxResult:
        """Publish the collective winner and every member's value there."""
        undefined = accumulator.any_feasible_nan
        dissolved = ~accumulator.any_feasible
        return CollectiveHardMaxResult(
            best_objective=jnp.where(
                undefined,
                jnp.nan,
                jnp.where(dissolved, -jnp.inf, accumulator.best_objective),
            ),
            best_member_values=jnp.where(
                undefined[..., jnp.newaxis],
                jnp.nan,
                jnp.where(
                    dissolved[..., jnp.newaxis],
                    -jnp.inf,
                    accumulator.best_member_values,
                ),
            ),
            best_global_action_id=jnp.where(
                undefined,
                0,
                jnp.where(dissolved, -1, accumulator.best_global_action_id),
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


def _winner(
    *,
    objectives: FloatND,
    member_values: FloatND,
    ids: IntND,
    where: BoolND,
    ties_go_to_members: bool,
    keeps_identity: bool,
) -> CollectiveHardMaxAccumulator:
    """Take the collective winner among the candidates on the last block axis.

    One reduction gives the winner, whether any candidate is inside `where`
    and whether one inside it has a NaN objective.
    """
    if not keeps_identity:
        best, any_nan = max_and_any_nan(values=member_values[..., 0], where=where)
        return CollectiveHardMaxAccumulator(
            best_objective=best,
            best_member_values=best[..., jnp.newaxis],
            best_global_action_id=jnp.full(best.shape, NO_ID, dtype=jnp.int32),
            any_feasible=~jnp.isnan(best) | any_nan,
            any_feasible_nan=any_nan,
        )
    (best_objective, *best_members), best_id, any_nan = (
        max_lexicographic_and_smallest_id(
            values=(objectives, *jnp.unstack(member_values, axis=-1)),
            ids=ids,
            where=where,
        )
        if ties_go_to_members
        else max_lexicographic_and_smallest_id(
            values=(member_values[..., 0],), ids=ids, where=where
        )
    )
    return CollectiveHardMaxAccumulator(
        best_objective=best_objective,
        best_member_values=(
            jnp.stack(best_members, axis=-1)
            if ties_go_to_members
            else best_objective[..., jnp.newaxis]
        ),
        best_global_action_id=best_id,
        any_feasible=best_id != NO_ID,
        any_feasible_nan=any_nan,
    )


COLLECTIVE_HARD_MAX_REDUCTION = CollectiveHardMaxReduction()
# Shared exact collective hard-max reduction specification.


def household_hard_max_reduction(
    *, members: tuple[str, ...], publishes_identity: bool
) -> CollectiveHardMaxReduction:
    """Return the hard-max reduction of a regime with `members`.

    The objective of a regime with one member is that member's value, and no
    tie key can change its winner: its ties go to the smallest identity alone,
    and where the caller does not publish the identity it is not kept.
    """
    several = len(members) > 1
    return CollectiveHardMaxReduction(
        ties_go_to_members=several, keeps_identity=several or publishes_identity
    )
