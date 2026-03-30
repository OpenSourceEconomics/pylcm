import jax.numpy as jnp

from lcm.grids.base import Grid
from lcm.grids.categorical import _validate_discrete_grid
from lcm.typing import Int1D
from lcm.utils.containers import get_field_names_and_values


class DiscreteGrid(Grid):
    """A discrete grid defining the outcome space of a categorical variable.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique int values.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with int
            fields.

    """

    def __init__(self, category_class: type, batch_size: int = 0) -> None:
        _validate_discrete_grid(category_class)
        names_and_values = get_field_names_and_values(category_class)
        self.__categories = tuple(names_and_values.keys())
        self.__codes = tuple(names_and_values.values())
        self.__ordered: bool = getattr(category_class, "_ordered", False)
        self.__batch_size: int = batch_size

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the list of category names."""
        return self.__categories

    @property
    def codes(self) -> tuple[int, ...]:
        """Return the list of category codes."""
        return self.__codes

    @property
    def ordered(self) -> bool:
        """Return whether the categories have a meaningful ordering."""
        return self.__ordered

    @property
    def batch_size(self) -> int:
        """Return batch size during solution."""
        return self.__batch_size

    def to_jax(self) -> Int1D:
        """Convert the grid to a Jax array."""
        return jnp.array(self.codes)
