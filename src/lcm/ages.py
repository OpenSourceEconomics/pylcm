"""Age grid and step parsing utilities for lifecycle models."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import jax.numpy as jnp

from lcm.exceptions import GridInitializationError, format_messages

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from lcm.typing import Float1D


# ======================================================================================
# Step parsing
# ======================================================================================

STEP_UNITS: dict[str, float] = {
    "Y": 1.0,
    "M": 1 / 12,
    "Q": 0.25,
}

_STEP_PATTERN = re.compile(r"^(\d+)?([YMQ])$", re.IGNORECASE)


def parse_step(step: str) -> float:
    """Parse a step string like 'Y', '2Y', 'M', '3M', 'Q' into years."""
    match = _STEP_PATTERN.match(step)
    if not match:
        raise GridInitializationError(
            f"Invalid step format: '{step}'. "
            "Expected format like 'Y', '2Y', 'M', '3M', 'Q'."
        )

    multiplier_str, unit = match.groups()
    multiplier = int(multiplier_str) if multiplier_str else 1
    return multiplier * STEP_UNITS[unit.upper()]


# ======================================================================================
# AgeGrid class
# ======================================================================================


class AgeGrid:
    """Age grid for lifecycle models."""

    def __init__(
        self,
        start: float | None = None,
        stop: float | None = None,
        step: str | None = None,
        values: tuple[float, ...] | None = None,
    ) -> None:
        _validate_age_grid(start, stop, step, values)

        self.start = start
        self.stop = stop
        self.step = step
        self.values = values

        if values is not None:
            self._ages = jnp.array(values)
            self._step_size: float | None = None
        else:
            self._step_size = parse_step(step)  # type: ignore[arg-type]
            self._ages = jnp.arange(start, stop, self._step_size)

    @property
    def ages(self) -> Float1D:
        """Array of ages for each period."""
        return self._ages

    @property
    def n_periods(self) -> int:
        """Number of periods in the grid."""
        return int(self._ages.shape[0])

    @property
    def step_size(self) -> float | None:
        """Step size in years, or None if using custom values."""
        return self._step_size

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
        return float(self._ages[period])

    def get_periods_where(self, predicate: Callable[[float], bool]) -> list[int]:
        """Get period indices where predicate is True.

        Args:
            predicate: A function that takes an age and returns True/False.

        Returns:
            List of period indices where predicate(age) is True.

        """
        return [
            period
            for period in range(self.n_periods)
            if predicate(float(self._ages[period]))
        ]


# ======================================================================================
# Validation
# ======================================================================================


def _validate_age_grid(
    start: float | None,
    stop: float | None,
    step: str | None,
    values: Iterable[float] | None,
) -> None:
    error_messages: list[str] = []

    has_range = start is not None or stop is not None or step is not None
    has_values = values is not None

    if has_values and has_range:
        error_messages.append("Cannot specify both 'values' and 'start/stop/step'.")
    elif has_values:
        assert values is not None  # has_values check guarantees this
        error_messages.extend(_validate_values(values))
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


def _validate_range(start: float, stop: float, step: str) -> list[str]:
    errors: list[str] = []

    if start >= stop:
        errors.append(f"'start' ({start}) must be less than 'stop' ({stop}).")

    if start < 0:
        errors.append(f"'start' must be non-negative, got {start}.")

    try:
        parse_step(step)
    except ValueError as e:
        errors.append(str(e))

    return errors


def _validate_values(values: Iterable[float]) -> list[str]:
    errors: list[str] = []

    try:
        vals = list(values)
    except TypeError:
        return ["'values' must be iterable."]

    if not vals:
        return ["'values' cannot be empty."]

    if any(not isinstance(v, int | float) for v in vals):
        return ["All values must be numbers."]

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
