"""Shared-evaluation cell of a canonical coarse regime transition.

A leaf module with no dependency on `Regime`, the validators, or the
regime-building code, so that the phase grammar, the canonicalization stage,
and the engine-side processing can all import it without an import cycle.
"""

from dataclasses import dataclass

from lcm.transition import MarkovTransition
from lcm.typing import UserFunction


@dataclass(frozen=True)
class _CoarseTransitionCell:
    """One target's view of a coarse regime transition.

    Canonicalization rewrites a coarse regime transition (bare callable or
    `MarkovTransition`) into a per-target mapping whose every cell is the
    same `_CoarseTransitionCell` referencing the single underlying
    transition object. The engine recognizes the cell type, evaluates the
    underlying once, and indexes the result per target — cells are views on
    one evaluation, never per-cell re-evaluations.
    """

    underlying: UserFunction | MarkovTransition
    """The coarse transition object shared by every cell of the mapping."""
