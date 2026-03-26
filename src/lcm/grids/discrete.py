import jax.numpy as jnp

from lcm._utils.containers import get_field_names_and_values
from lcm.grids.categorical import _validate_discrete_grid
from lcm.grids.continuous import Grid
from lcm.typing import Int1D


class _DiscreteGridBase(Grid):
    """Base class for discrete grids: categories, codes, and JAX conversion."""

    def __init__(self, category_class: type) -> None:
        _validate_discrete_grid(category_class)
        names_and_values = get_field_names_and_values(category_class)
        self.__categories = tuple(names_and_values.keys())
        self.__codes = tuple(names_and_values.values())
        self.__ordered: bool = getattr(category_class, "_ordered", False)

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

    def to_jax(self) -> Int1D:
        """Convert the grid to a Jax array."""
        return jnp.array(self.codes)


class DiscreteGrid(_DiscreteGridBase):
    """A discrete grid defining the outcome space of a categorical variable.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique int values.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with int
            fields.

    """

    def __init__(self, category_class: type) -> None:
        super().__init__(category_class)
