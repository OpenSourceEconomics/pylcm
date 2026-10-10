"""Normalize the public clock without assigning biological meaning to a period."""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import jax.numpy as jnp

from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.typing import Float1D, Int1D, UserAge


@dataclass(frozen=True, kw_only=True)
class ModelTime:
    """One model's coordinate kind, horizon and optional exact age labels."""

    n_periods: int
    """Number of computational slots, including any explicit terminal slot."""

    ages: AgeGrid | None = None
    """Real age labels, absent in a period model."""

    @classmethod
    def from_inputs(cls, *, ages: AgeGrid | None, n_periods: int | None) -> ModelTime:
        """Require exactly one public horizon declaration."""
        if (ages is None) == (n_periods is None):
            raise ModelInitializationError(
                "Exactly one of ages and n_periods is required."
            )
        if ages is not None:
            return cls(ages=ages, n_periods=ages.n_periods)
        if type(n_periods) is not int or n_periods <= 0:
            raise ModelInitializationError(
                "n_periods must be a positive integer, excluding booleans."
            )
        return cls(n_periods=n_periods)

    @property
    def kind(self) -> Literal["age", "period"]:
        """The coordinate names used at public data boundaries."""
        return "period" if self.ages is None else "age"

    @cached_property
    def values(self) -> Int1D | Float1D:
        """Coordinates indexed by computational period."""
        if self.ages is not None:
            return self.ages.values
        return jnp.arange(self.n_periods, dtype=jnp.int32)

    @property
    def exact_values(self) -> tuple[UserAge, ...]:
        """Exact public coordinates, with no float conversion."""
        return (
            tuple(range(self.n_periods))
            if self.ages is None
            else self.ages.exact_values
        )


# Internal helper entry points also accept the existing age-grid test/plugin inputs.
type TimeAxis = AgeGrid | ModelTime


def coordinate_kind(ages: TimeAxis) -> Literal["age", "period"]:
    """Name the axis without inferring meaning from its numeric values."""
    return ages.kind if isinstance(ages, ModelTime) else "age"


def coordinate_at(*, ages: TimeAxis, period: int) -> int | float:
    """Read a scalar time label for an internal execution context."""
    if period < 0 or period >= ages.n_periods:
        raise IndexError(f"Period {period} is outside the model horizon.")
    if isinstance(ages, ModelTime):
        return period if ages.ages is None else ages.ages.period_to_age(period)
    return ages.period_to_age(period)


def specialization_coordinate_at(*, ages: TimeAxis, period: int) -> int | float:
    """Preserve float age factories and integer period factories."""
    coordinate = coordinate_at(ages=ages, period=period)
    return coordinate if coordinate_kind(ages) == "period" else float(coordinate)


def age_at(*, ages: TimeAxis, period: int) -> float | None:
    """Report an age only when the model declares one."""
    return None if coordinate_kind(ages) == "period" else float(ages.values[period])
