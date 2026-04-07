from abc import ABC, abstractmethod

from lcm.typing import Float1D, Int1D


class Grid(ABC):
    """LCM Grid base class."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Size of the batches looped over during the solution.

        `ContinuousGrid` overrides this via its dataclass field.
        `DiscreteGrid` overrides this via its own property.

        """

    @abstractmethod
    def to_jax(self) -> Int1D | Float1D:
        """Convert the grid to a Jax array."""
