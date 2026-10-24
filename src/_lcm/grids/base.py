"""The abstract `Grid` every outcome-space declaration implements.

A grid says which values a variable can take and nothing about how it moves;
laws of motion live in a regime's `state_transitions`.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import numpy as np

from lcm.exceptions import GridInitializationError
from lcm.typing import Float1D, Int1D, ValueND

if TYPE_CHECKING:
    type UserVector = np.ndarray | ValueND
else:
    # Wider runtime form: `fail_if_not_a_numeric_vector` refuses a non-array with a
    # message that shows the array spelling, which beartype's own message cannot.
    type UserVector = object


def fail_if_not_a_numeric_vector(
    *, value: UserVector, argument: str, example: str
) -> None:
    """Raise unless `value` is a one-dimensional integer or floating array.

    Args:
        value: The numbers a user passed.
        argument: The argument name, for the message.
        example: A call that spells the argument as an array, for the message.

    Raises:
        GridInitializationError: If `value` is no NumPy or JAX array, is not
            one-dimensional, or has neither an integer nor a floating dtype.

    """
    if not isinstance(value, np.ndarray | jax.Array):
        raise GridInitializationError(
            f"`{argument}` must be a one-dimensional NumPy or JAX array, got "
            f"{type(value).__name__}. Write, for example, `{example}`."
        )
    if value.ndim != 1:
        raise GridInitializationError(
            f"`{argument}` must be one-dimensional, got shape {value.shape}."
        )
    if not (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ):
        raise GridInitializationError(
            f"`{argument}` must have an integer or floating dtype, got {value.dtype}."
        )


class Grid(ABC):
    """Outcome-space definition shared by all LCM grids."""

    @abstractmethod
    def to_jax(self) -> Int1D | Float1D:
        """Convert the grid to a Jax array."""
