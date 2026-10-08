"""Explicitly labelled starting nodes for age and period models."""

from dataclasses import dataclass
from fractions import Fraction

from lcm.exceptions import ModelInitializationError


@dataclass(frozen=True, kw_only=True)
class InitialNode:
    """One admissible starting regime at exactly one named time coordinate."""

    regime: str
    """Name of the starting regime."""

    age: int | float | Fraction | None = None
    """An exact grid age, only for an age model."""

    period: int | None = None
    """A zero-based computational period, only for a period model."""

    def __post_init__(self) -> None:
        if (self.age is None) == (self.period is None):
            raise ModelInitializationError(
                "InitialNode requires exactly one of age and period."
            )
        if self.period is not None and (
            type(self.period) is not int or self.period < 0
        ):
            raise ModelInitializationError(
                "InitialNode.period must be a nonnegative integer."
            )
