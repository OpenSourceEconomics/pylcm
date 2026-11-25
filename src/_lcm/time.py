"""Normalize the public clock without assigning biological meaning to a period."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import jax.numpy as jnp

from _lcm.typing import AgeValue
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.typing import (
    Float1D,
    Int1D,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    StageName,
    UserAge,
)

type Stages = tuple[tuple[StageName, tuple[RegimeName, ...]], ...]


@dataclass(frozen=True, kw_only=True)
class ModelTime:
    """One model's coordinate kind, horizon, stages and optional exact age labels.

    The engine runs over steps: step `s` is stage `s % n_stages` of period
    `s // n_stages`. Coordinates are indexed by step and repeat a period's label
    for each of its stages.
    """

    n_periods: int
    """Number of steps, including any explicit terminal step."""

    ages: AgeGrid | None = None
    """Real age labels, absent in a period model."""

    stages: Stages = ()
    """Each stage of a period, in order, with its regimes; empty for one stage."""

    pass_throughs: frozenset[RegimeName] = frozenset()
    """The generated regimes a subject is in where its regime sits out a stage."""

    @classmethod
    def from_inputs(
        cls,
        *,
        ages: AgeGrid | None,
        n_periods: int | None,
        stages: Mapping[StageName, Sequence[RegimeName]] | None = None,
        regime_names: Sequence[RegimeName] = (),
    ) -> ModelTime:
        """Require exactly one public horizon declaration and valid stages."""
        if (ages is None) == (n_periods is None):
            raise ModelInitializationError(
                "Exactly one of ages and n_periods is required."
            )
        declared = (
            ()
            if stages is None
            else _validated_stages(stages=stages, regime_names=regime_names)
        )
        n_stages = max(1, len(declared))
        if ages is not None:
            return cls(ages=ages, n_periods=ages.n_periods * n_stages, stages=declared)
        if type(n_periods) is not int or n_periods <= 0:
            raise ModelInitializationError(
                "n_periods must be a positive integer, excluding booleans."
            )
        return cls(n_periods=n_periods * n_stages, stages=declared)

    @property
    def kind(self) -> Literal["age", "period"]:
        """The coordinate names used at public data boundaries."""
        return "period" if self.ages is None else "age"

    @property
    def n_stages(self) -> int:
        """Number of stages in every period."""
        return max(1, len(self.stages))

    @property
    def economic(self) -> ModelTime:
        """The clock of periods, one step per period."""
        return ModelTime(n_periods=self.n_periods // self.n_stages, ages=self.ages)

    @cached_property
    def stage_of(self) -> Mapping[RegimeName, int]:
        """The position of each regime's stage within its period."""
        return {
            name: position
            for position, (_, names) in enumerate(self.stages)
            for name in names
        }

    @cached_property
    def values(self) -> Int1D | Float1D:
        """Coordinates indexed by step."""
        if self.ages is None:
            values = jnp.arange(self.n_periods // self.n_stages, dtype=jnp.int32)
        else:
            values = self.ages.values
        return values if self.n_stages == 1 else jnp.repeat(values, self.n_stages)

    @cached_property
    def exact_values(self) -> tuple[UserAge, ...]:
        """Exact public coordinates, with no float conversion."""
        labels = (
            tuple(range(self.n_periods // self.n_stages))
            if self.ages is None
            else self.ages.exact_values
        )
        return tuple(label for label in labels for _ in range(self.n_stages))


def _validated_stages(
    *,
    stages: Mapping[StageName, Sequence[RegimeName]],
    regime_names: Sequence[RegimeName],
) -> Stages:
    """Return the stages in order; every regime is in exactly one of them."""
    if not isinstance(stages, Mapping) or not stages:
        raise ModelInitializationError(
            "`stages` must be a nonempty mapping from stage names to regime names."
        )
    declared: list[tuple[StageName, tuple[RegimeName, ...]]] = []
    for name, names in stages.items():
        if (
            not isinstance(name, str)
            or isinstance(names, str)
            or not isinstance(names, Sequence)
            or not names
            or not all(isinstance(regime, str) for regime in names)
        ):
            raise ModelInitializationError(
                f"Stage {name!r} must be named by a string and list at least one "
                f"regime name, got {names!r}."
            )
        declared.append((name, tuple(names)))
    listed = [regime for _, names in declared for regime in names]
    unknown = sorted(set(listed) - set(regime_names))
    repeated = sorted({regime for regime in listed if listed.count(regime) > 1})
    missing = sorted(set(regime_names) - set(listed))
    if unknown or repeated or missing:
        raise ModelInitializationError(
            "Every regime must be in exactly one stage. "
            f"Unknown: {unknown}; in several stages: {repeated}; in none: {missing}."
        )
    return tuple(declared)


def stages_per_period(ages: TimeAxis) -> int:
    """Number of stages in every period of `ages`."""
    return ages.n_stages if isinstance(ages, ModelTime) else 1


def stage_position(*, ages: TimeAxis, regime: RegimeName) -> int:
    """Position of `regime`'s stage within every period of `ages`."""
    return ages.stage_of.get(regime, 0) if isinstance(ages, ModelTime) else 0


# Internal helper entry points also accept the existing age-grid test/plugin inputs.
type TimeAxis = AgeGrid | ModelTime


def coordinate_kind(ages: TimeAxis) -> Literal["age", "period"]:
    """Name the axis without inferring meaning from its numeric values."""
    return ages.kind if isinstance(ages, ModelTime) else "age"


def coordinate_at(*, ages: TimeAxis, period: int) -> AgeValue:
    """Read a scalar time label for an internal execution context."""
    if period < 0 or period >= ages.n_periods:
        raise IndexError(f"Period {period} is outside the model horizon.")
    if isinstance(ages, ModelTime):
        economic = period // ages.n_stages
        return economic if ages.ages is None else ages.ages.period_to_age(economic)
    return ages.period_to_age(period)


def specialization_coordinate_at(*, ages: TimeAxis, period: int) -> AgeValue:
    """Preserve float age factories and integer period factories."""
    coordinate = coordinate_at(ages=ages, period=period)
    return coordinate if coordinate_kind(ages) == "period" else float(coordinate)


def age_at(*, ages: TimeAxis, period: int) -> float | None:
    """Report an age only when the model declares one."""
    return None if coordinate_kind(ages) == "period" else float(ages.values[period])


def display_age_at(*, ages: TimeAxis, period: int) -> str:
    """Return a period's coordinate for a log line, read from the exact host values.

    The exact coordinates are Python `int` or `Fraction`, so formatting them
    never reads the device-held `values`.
    """
    return format_age(float(ages.exact_values[period]))


def format_age(age: AgeValue | ScalarInt | ScalarFloat) -> str:
    """Return an age for a message: an integer when whole, otherwise two decimals.

    A device-held age is read on the host, so pass only a concrete value.
    """
    value = float(age)
    return str(int(value)) if value.is_integer() else f"{value:.2f}"
