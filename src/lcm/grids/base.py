from abc import ABC, abstractmethod

from lcm.typing import Float1D, Int1D


class Grid(ABC):
    """LCM Grid base class."""

    @abstractmethod
    def to_jax(self) -> Int1D | Float1D:
        """Convert the grid to a Jax array."""
