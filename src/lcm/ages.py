"""Age grid and step parsing utilities for lifecycle models."""

import re
from collections.abc import Callable, Iterable
from fractions import Fraction
from typing import overload

import jax.numpy as jnp

from lcm.exceptions import GridInitializationError, format_messages
from lcm.typing import Float1D

# ======================================================================================
# Step parsing
# ======================================================================================

STEP_UNITS: dict[str, Fraction] = {
    "Y": Fraction(1, 1),
    "M": Fraction(1, 12),
    "Q": Fraction(1, 4),
}


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


# ======================================================================================
# AgeGrid class
# ======================================================================================


class AgeGrid:
    """Age grid for life-cycle models."""

    @overload
    def __init__(
        self,
        *,
        start: int | Fraction,
        stop: int | Fraction,
        step: str,
        precise_values: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        precise_values: Iterable[int | Fraction] | None = None,
        start: None = None,
        stop: None = None,
        step: None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        start: int | Fraction | None = None,
        stop: int | Fraction | None = None,
        step: str | None = None,
        precise_values: Iterable[int | Fraction] | None = None,
    ) -> None:
        _validate_age_grid(start, stop, step, precise_values)

        if precise_values is not None:
            self._precise_values = tuple(precise_values)
            self._values = jnp.array(precise_values)
            self._step_size = None
            self._precise_step_size = None
        else:
            self._precise_step_size = parse_step(step)  # ty: ignore[invalid-argument-type]
            self._step_size = float(self._precise_step_size)
            n_steps = int((stop - start) // self._precise_step_size) + 1  # ty: ignore[unsupported-operator]
            self._precise_values = tuple(
                start + i * self._precise_step_size  # ty: ignore[unsupported-operator]
                for i in range(n_steps)
            )
            self._values = jnp.array([float(age) for age in self._precise_values])

    @property
    def values(self) -> Float1D:
        """Float ages; indexed by period."""
        return self._values

    @property
    def precise_values(self) -> tuple[int | Fraction, ...]:
        """Precise ages; indexed by period.

        Could be:
        - An int if all ages are multiples of one year.
        - A Fraction if the ages are sub-annual.

        """
        return self._precise_values

    @property
    def n_periods(self) -> int:
        """Number of periods in the grid."""
        return int(self._values.shape[0])

    @property
    def step_size(self) -> float | None:
        """Step size in years, or None if using custom values."""
        return self._step_size

    @property
    def precise_step_size(self) -> int | Fraction | None:
        """Precise step size.

        Could be:
        - An int if the step size is a multiple of one year.
        - A Fraction if the step size is sub-annual.
        - None if using custom age values.

        """
        return self._precise_step_size

    def period_to_age(self, period: int) -> float:
        """Convert a period index to the corresponding age.

        Args:
            period: Zero-based period index.

        Returns:
            The age corresponding to the given period.

        Raises:
            IndexError: If period is out of bounds.

        """
        if period < 0 or period >= self.n_periods:
            raise IndexError(
                f"Period {period} out of bounds for grid with {self.n_periods} periods."
            )
        return float(self._values[period])

    def get_periods_where(self, predicate: Callable[[float], bool]) -> tuple[int, ...]:
        """Get period indices where predicate is True.

        Args:
            predicate: A function that takes an age and returns True/False.

        Returns:
            Tuple of period indices where predicate(age) is True.

        """
        return tuple(
            period
            for period in range(self.n_periods)
            if predicate(float(self._values[period]))
        )


# ======================================================================================
# Validation
# ======================================================================================


def _validate_age_grid(
    start: int | Fraction | None,
    stop: int | Fraction | None,
    step: str | None,
    precise_values: Iterable[int | Fraction] | None,
) -> None:
    error_messages: list[str] = []

    has_range = start is not None or stop is not None or step is not None
    has_values = precise_values is not None

    if has_values and has_range:
        error_messages.append("Cannot specify both 'values' and 'start/stop/step'.")
    elif has_values:
        error_messages.extend(_validate_values(precise_values))  # ty: ignore[invalid-argument-type]
    elif has_range:
        if start is None or stop is None or step is None:
            error_messages.append(
                "When using range, all of 'start', 'stop', 'step' must be provided."
            )
        else:
            error_messages.extend(_validate_range(start, stop, step))
    else:
        error_messages.append("Must specify 'values' or 'start/stop/step'.")

    if error_messages:
        raise GridInitializationError(format_messages(error_messages))


def _validate_range(
    start: int | Fraction, stop: int | Fraction, step: str
) -> list[str]:
    errors: list[str] = []

    if start >= stop:
        errors.append(f"'start' ({start}) must be less than 'stop' ({stop}).")

    if start < 0:
        errors.append(f"'start' must be non-negative, got {start}.")

    try:
        precise_step_size = parse_step(step)
    except GridInitializationError as e:
        errors.append(str(e))
        return errors

    step_fraction = (
        Fraction(precise_step_size)
        if isinstance(precise_step_size, int)
        else precise_step_size
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
