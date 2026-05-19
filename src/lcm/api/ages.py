"""User-facing `AgeGrid` for lifecycle models."""

import functools
from collections.abc import Callable, Iterable
from fractions import Fraction
from typing import overload

import jax.numpy as jnp
from beartype import beartype

from lcm._ages import _is_integer_valued, _parse_step, _validate_age_grid
from lcm._beartype_conf import GRID_CONF
from lcm.exceptions import GridInitializationError
from lcm.typing import Float1D, Int1D, UserAge


class AgeGrid:
    """Age grid for life-cycle models.

    Automatically produces integer ages (`int32` array, `int` scalars) when all
    values are integer-valued, and float ages (`float32` array, `float` scalars)
    otherwise.

    """

    @overload
    def __init__(
        self,
        *,
        start: UserAge,
        stop: UserAge,
        step: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        exact_values: Iterable[UserAge],
    ) -> None: ...

    @beartype(conf=GRID_CONF)
    def __init__(
        self,
        *,
        start: UserAge | None = None,
        stop: UserAge | None = None,
        step: str | None = None,
        exact_values: Iterable[UserAge] | None = None,
    ) -> None:
        _validate_age_grid(start=start, stop=stop, step=step, exact_values=exact_values)

        if start is not None and stop is not None and step is not None:
            self._exact_step_size = _parse_step(step)
            self._step_size = float(self._exact_step_size)
            n_steps = int((stop - start) // self._exact_step_size) + 1
            self._exact_values = tuple(
                start + i * self._exact_step_size for i in range(n_steps)
            )
        elif exact_values is not None:
            self._exact_values = tuple(exact_values)
            self._step_size = None
            self._exact_step_size = None
        else:
            msg = "Must specify 'start/stop/step' or 'exact_values'."
            raise GridInitializationError(msg)

        self._is_integer = all(_is_integer_valued(v) for v in self._exact_values)
        if self._is_integer:
            self._exact_values = tuple(int(v) for v in self._exact_values)
            self._values = jnp.array(self._exact_values, dtype=jnp.int32)
        else:
            self._values = jnp.array([float(v) for v in self._exact_values])

    @property
    def is_integer(self) -> bool:
        """Whether all ages are integer-valued."""
        return self._is_integer

    @property
    def values(self) -> Int1D | Float1D:
        """Age values as a JAX array; indexed by period.

        `Int1D` when all ages are integer-valued, `Float1D` otherwise.

        """
        return self._values

    @property
    def exact_values(self) -> tuple[UserAge, ...]:
        """Exact ages; indexed by period.

        Could be:
        - An int if all ages are multiples of one year.
        - A Fraction if the ages are sub-annual.

        """
        return self._exact_values

    @property
    def n_periods(self) -> int:
        """Number of periods in the grid."""
        return int(self._values.shape[0])

    @property
    def step_size(self) -> float | None:
        """Step size in years, or None if using custom values."""
        return self._step_size

    @property
    def exact_step_size(self) -> int | Fraction | None:
        """Exact step size.

        Could be:
        - An int if the step size is a multiple of one year.
        - A Fraction if the step size is sub-annual.
        - None if using custom age values.

        """
        return self._exact_step_size

    def period_to_age(self, period: int) -> int | float:
        """Convert a period index to the corresponding age.

        Args:
            period: Zero-based period index.

        Returns:
            The age corresponding to the given period. `int` when all ages are
            integer-valued, `float` otherwise.

        Raises:
            IndexError: If period is out of bounds.

        """
        if period < 0 or period >= self.n_periods:
            raise IndexError(
                f"Period {period} out of bounds for grid with {self.n_periods} periods."
            )
        if self._is_integer:
            return int(self._values[period])
        return float(self._values[period])

    def age_to_period(self, age: float) -> int:
        """Convert an age to the corresponding period index.

        Args:
            age: Age value that must be a valid grid point.

        Returns:
            The zero-based period index corresponding to the given age.

        Raises:
            ValueError: If age is not a valid grid point.

        """
        try:
            return self._age_to_period_map[age]
        except KeyError:
            valid = sorted(self._age_to_period_map)
            msg = f"Age {age} is not a valid grid point. Valid ages: {valid}."
            raise ValueError(msg) from None

    @functools.cached_property
    def _age_to_period_map(self) -> dict[int | float, int]:
        if self._is_integer:
            return {int(v): i for i, v in enumerate(self._exact_values)}
        return {float(v): i for i, v in enumerate(self._exact_values)}

    def get_periods_where(
        self, predicate: Callable[[int | float], bool]
    ) -> tuple[int, ...]:
        """Get period indices where predicate is True.

        Args:
            predicate: A function that takes an age and returns True/False.

        Returns:
            Tuple of period indices where predicate(age) is True.

        """
        _convert: Callable[[object], int | float] = int if self._is_integer else float  # ty: ignore[invalid-assignment]
        return tuple(
            period
            for period in range(self.n_periods)
            if predicate(_convert(self._values[period]))
        )
