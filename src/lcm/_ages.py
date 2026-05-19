"""Internal helpers and constants backing `lcm.api.ages.AgeGrid`."""

import re
from collections.abc import Iterable
from fractions import Fraction
from types import MappingProxyType

from lcm.exceptions import GridInitializationError, format_messages
from lcm.typing import UserAge

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


def _parse_step(step: str) -> int | Fraction:
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
    start: UserAge | None,
    stop: UserAge | None,
    step: str | None,
    exact_values: Iterable[UserAge] | None,
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


def _validate_range(*, start: UserAge, stop: UserAge, step: str) -> list[str]:
    errors: list[str] = []

    if start >= stop:
        errors.append(f"'start' ({start}) must be less than 'stop' ({stop}).")

    if start < 0:
        errors.append(f"'start' must be non-negative, got {start}.")

    try:
        exact_step_size = _parse_step(step)
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


def _validate_values(values: Iterable[UserAge]) -> list[str]:
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
