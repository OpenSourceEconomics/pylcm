import jax.numpy as jnp
from beartype import beartype

from lcm._beartype_conf import GRID_CONF
from lcm.grids.base import Grid, _fail_if_batch_size_combined_with_distributed
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

    @beartype(conf=GRID_CONF)
    def __init__(
        self, category_class: type, batch_size: int = 0, *, distributed: bool = False
    ) -> None:
        _fail_if_batch_size_combined_with_distributed(
            batch_size=batch_size, distributed=distributed
        )
        _validate_discrete_grid(category_class)
        names_and_values = get_field_names_and_values(category_class)
        self.__categories = tuple(names_and_values.keys())
        # Coerce `ScalarInt` field values to Python `int` for the `codes`
        # property. `codes` is the Python-side API (the tuple flows into
        # dict/set operations that need hashable members); the JAX-side
        # representation comes from `to_jax()`.
        self.__codes = tuple(int(v) for v in names_and_values.values())
        self.__ordered: bool = getattr(category_class, "_ordered", False)
        self.__batch_size: int = batch_size
        self.__distributed: bool = distributed

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

    @property
    def distributed(self) -> bool:
        """Return whether the grid is sharded across available devices."""
        return self.__distributed

    def to_jax(self) -> Int1D:
        """Convert the grid to a Jax array.

        Discrete state/action codes are pinned to `int32` regardless of the
        ambient `jax_enable_x64` setting. A single integer dtype across
        transitions, V-array indexing, and action lookups keeps the JIT cache
        unsplit and lets AOT-compiled programs ship one signature. `int32`
        accommodates any realistic category count and matches the
        `MISSING_CAT_CODE` sentinel.
        """
        return jnp.array(self.codes, dtype=jnp.int32)
