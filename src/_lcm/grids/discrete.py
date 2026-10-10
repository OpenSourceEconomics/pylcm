"""`DiscreteGrid`: the outcome space of one categorical variable.

It holds the category class whose codes index that variable's axis; stochastic
movement over those codes is declared with `StochasticTransition`, not here.
"""

import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import GRID_CONF
from _lcm.grids.base import Grid
from _lcm.grids.categorical import (
    DataclassInstance,
    _validate_discrete_grid,
    get_category_codes,
)
from lcm.typing import Int1D


class DiscreteGrid(Grid):
    """A discrete grid defining the outcome space of a categorical variable.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique int values. The legacy
            single-positional-argument form remains supported for compatibility.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with int
            fields.

    """

    @beartype(conf=GRID_CONF)
    def __init__(
        self,
        *legacy_category_class: type[DataclassInstance],
        category_class: type[DataclassInstance] | None = None,
    ) -> None:
        if len(legacy_category_class) > 1:
            msg = "DiscreteGrid accepts at most one positional argument."
            raise TypeError(msg)
        if legacy_category_class:
            if category_class is not None:
                msg = "DiscreteGrid got multiple values for 'category_class'."
                raise TypeError(msg)
            category_class = legacy_category_class[0]
        elif category_class is None:
            msg = "DiscreteGrid missing required argument: 'category_class'."
            raise TypeError(msg)

        _validate_discrete_grid(category_class)
        category_codes = get_category_codes(category_class)
        self.__categories = tuple(category_codes)
        # Coerce `ScalarInt` codes to Python `int` for the `codes` property.
        # `codes` is the Python-side API (the tuple flows into dict/set
        # operations that need hashable members); the JAX-side representation
        # comes from `to_jax()`.
        self.__codes = tuple(int(code) for code in category_codes.values())
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
        """Convert the grid to a Jax array.

        Discrete state/action codes are pinned to `int32` regardless of the
        ambient `jax_enable_x64` setting. A single integer dtype across
        transitions, V-array indexing, and action lookups keeps the JIT cache
        unsplit and lets AOT-compiled programs ship one signature. `int32`
        accommodates any realistic category count and matches the
        `MISSING_CAT_CODE` sentinel.
        """
        return jnp.array(self.codes, dtype=jnp.int32)
