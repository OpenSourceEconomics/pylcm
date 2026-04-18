"""Age grid and step parsing utilities for lifecycle models."""

import functools
import re
from collections.abc import Callable, Iterable
from fractions import Fraction
from types import MappingProxyType
from typing import overload

import jax.numpy as jnp

from lcm.exceptions import GridInitializationError, format_messages
from lcm.typing import Age, Float1D, Int1D

STEP_UNITS: MappingProxyType[str, Fraction] = MappingProxyType(
    {
        "Y": Fraction(1, 1),
        "M": Fraction(1, 12),
        "Q": Fraction(1, 4),
    }
)

# Names that behave like states in initial conditions but are not declared on
# any `Regime.states`. `age` is required for every subject regardless of regime.
PSEUDO_STATE_NAMES: frozenset[str] = frozenset({"age"})


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
        start: int | Fraction,
        stop: int | Fraction,
        step: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        exact_values: Iterable[int | Fraction],
    ) -> None: ...

    def __init__(
        self,
        *,
        start: int | Fraction | None = None,
        stop: int | Fraction | None = None,
        step: str | None = None,
        exact_values: Iterable[int | Fraction] | None = None,
    ) -> None:
        _validate_age_grid(start=start, stop=stop, step=step, exact_values=exact_values)

        if start is not None and stop is not None and step is not None:
            self._exact_step_size = parse_step(step)
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
    def exact_values(self) -> tuple[int | Fraction, ...]:
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

    def period_to_age(self, period: int) -> Age:
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

    def age_to_period(self, age: Age) -> int:
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
    def _age_to_period_map(self) -> dict[Age, int]:
        if self._is_integer:
            return {int(v): i for i, v in enumerate(self._exact_values)}
        return {float(v): i for i, v in enumerate(self._exact_values)}

    def get_periods_where(self, predicate: Callable[[Age], bool]) -> tuple[int, ...]:
        """Get period indices where predicate is True.

        Args:
            predicate: A function that takes an age and returns True/False.

        Returns:
            Tuple of period indices where predicate(age) is True.

        """
        _convert: Callable[[object], Age] = int if self._is_integer else float  # ty: ignore[invalid-assignment]
        return tuple(
            period
            for period in range(self.n_periods)
            if predicate(_convert(self._values[period]))
        )


def parse_step(step: str) -> int | Fraction:
    """Parse a step string like 'Y', '2Y', 'M', '3M', 'Q' into int or Fraction."""
    match = re.match(r"^(\d+)?([YMQ])$", step, re.IGNORECASE)
    if not match:
        raise GridInitializationError(
            f"Invalid step format: '{step}'. "
            "Expected format like 'Y', '2Y', 'M', '3M', 'Q'."
        )

    multiplier_str, unit = match.groups()
    multiplier = int(multiplier_str) if multiplier_str else 1
    result = multiplier * STEP_UNITS[unit.upper()]
    return int(result) if result.denominator == 1 else result


def _is_integer_valued(value: int | Fraction) -> bool:
    """Check if a value is integer-valued (int or Fraction with unit denominator)."""
    if isinstance(value, int):
        return True
    return isinstance(value, Fraction) and value.denominator == 1


def _validate_age_grid(
    *,
    start: int | Fraction | None,
    stop: int | Fraction | None,
    step: str | None,
    exact_values: Iterable[int | Fraction] | None,
) -> None:
    error_messages: list[str] = []

    has_range = start is not None or stop is not None or step is not None
    has_values = exact_values is not None

    if has_values and has_range:
        error_messages.append("Cannot specify both 'values' and 'start/stop/step'.")
    elif exact_values is not None:
        error_messages.extend(_validate_values(exact_values))
    elif has_range:
        if start is None or stop is None or step is None:
            error_messages.append(
                "When using range, all of 'start', 'stop', 'step' must be provided."
            )
        else:
            error_messages.extend(_validate_range(start=start, stop=stop, step=step))
    else:
        error_messages.append("Must specify 'values' or 'start/stop/step'.")

    if error_messages:
        raise GridInitializationError(format_messages(error_messages))


def _validate_range(
    *, start: int | Fraction, stop: int | Fraction, step: str
) -> list[str]:
    errors: list[str] = []

    if start >= stop:
        errors.append(f"'start' ({start}) must be less than 'stop' ({stop}).")

    if start < 0:
        errors.append(f"'start' must be non-negative, got {start}.")

    try:
        exact_step_size = parse_step(step)
    except GridInitializationError as e:
        errors.append(str(e))
        return errors

    step_fraction = (
        Fraction(exact_step_size)
        if isinstance(exact_step_size, int)
        else exact_step_size
    )
    range_fraction = Fraction(stop) - Fraction(start)
    n_steps = range_fraction / step_fraction + 1
    if n_steps.denominator != 1:
        errors.append(
            f"Step size ({float(step_fraction)}) does not divide evenly into the range "
            f"({float(range_fraction)}). Number of steps would be {float(n_steps)}."
        )

    return errors


def _validate_values(values: Iterable[int | Fraction]) -> list[str]:
    errors: list[str] = []

    try:
        vals = tuple(values)
    except TypeError:
        return ["'values' must be iterable."]

    if not vals:
        return ["'values' cannot be empty."]

    if any(not isinstance(v, (int, Fraction)) for v in vals):
        return ["All values must be integers or fractions."]

    if any(v < 0 for v in vals):
        errors.append("All values must be non-negative.")

    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            errors.append(
                f"Values must be strictly increasing. "
                f"Found {vals[i]} <= {vals[i - 1]} at index {i}."
            )
            break

    return errors
