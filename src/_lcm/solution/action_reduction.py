"""Exact hard maxima over a canonical global action order.

The collective hard max in `collective_action_reduction` folds every action block
a regime's solve and simulation decision reduce. `HARD_MAX_REDUCTION` declares the
same exact contract for a reduced axis whose fold lives in its own step, such as
NB-EGM's branch axis.
"""

from typing import Literal, NamedTuple

from _lcm.solution.collective_action_reduction import (
    COLLECTIVE_HARD_MAX_REDUCTION,
    CollectiveHardMaxAccumulator,
    CollectiveHardMaxReduction,
    CollectiveHardMaxResult,
)
from lcm.typing import BoolND, FloatND, IntND

__all__ = [
    "COLLECTIVE_HARD_MAX_REDUCTION",
    "HARD_MAX_REDUCTION",
    "CollectiveHardMaxAccumulator",
    "CollectiveHardMaxReduction",
    "CollectiveHardMaxResult",
    "HardMaxReduction",
    "HardMaxResult",
]


class HardMaxResult(NamedTuple):
    """Final value, replay identity, and explicit feasibility state."""

    best_value: FloatND
    best_global_action_id: IntND
    any_feasible: BoolND


class HardMaxReduction:
    """Exact hard maximum over one value per action, as a declaration only.

    Equal values select the smallest canonical global identity, so the winner does
    not depend on how the axis is blocked or in which order blocks are merged.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Stable identity of the exact singleton hard-max contract."""
        return ("hard-max", 1)

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: the max with lowest-id tie-break is order independent."""
        return "exact"


HARD_MAX_REDUCTION = HardMaxReduction()
# Shared exact hard-max declaration.
