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

    @property
    @abstractmethod
    def distributed(self) -> bool:
        """Whether to shard the grid's state axis across available devices.

        `ContinuousGrid` exposes this as a dataclass field; `DiscreteGrid`
        exposes it as a property over a private field.

        """

    @abstractmethod
    def to_jax(self) -> Int1D | Float1D:
        """Convert the grid to a Jax array."""
