"""The abstract `Grid` every outcome-space declaration implements.

A grid says which values a variable can take and nothing about how it moves;
laws of motion live in a regime's `state_transitions`.
"""

from abc import ABC, abstractmethod

from lcm.typing import Float1D, Int1D


class Grid(ABC):
    """Outcome-space definition shared by all LCM grids."""

    @abstractmethod
    def to_jax(self) -> Int1D | Float1D:
        """Convert the grid to a Jax array."""
