"""Explicit coordinates of admissible starting nodes."""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from beartype import beartype

from _lcm.beartype_conf import MODEL_CONF
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.transition import (
    AgeSelector,
    PeriodRange,
    Periods,
    _DeclaredSelector,
    _fail_if_invalid_age_selector,
    _fail_if_invalid_period_selector,
)
from lcm.typing import RegimeName, UserAge


@dataclass(frozen=True, init=False)
class InitialNodes:
    """Declare admissible starts using exactly one named coordinate mapping.

    `by_age` or `by_period` maps selectors to one regime name or a nonempty
    sequence or set of names. Containers are copied and frozen. The model unions
    overlapping selectors and publishes exact coordinates with unique regime
    tuples. Clock compatibility, grid membership and regime existence are checked
    when the model is built. These declarations carry no population weights.
    """

    by_age: MappingProxyType[AgeSelector, tuple[RegimeName, ...]] | None = None
    """Age selectors and sorted, unique regime names; absent in period mode."""

    by_period: (
        MappingProxyType[
            int | tuple[int, ...] | range | PeriodRange | Periods,
            tuple[RegimeName, ...],
        ]
        | None
    ) = None
    """Period selectors and sorted, unique regime names; absent in age mode."""

    @beartype(conf=MODEL_CONF)
    def __init__[K: _DeclaredSelector](
        self,
        *,
        by_age: Mapping[K, str | Sequence[str] | AbstractSet[str]] | None = None,
        by_period: Mapping[K, str | Sequence[str] | AbstractSet[str]] | None = None,
    ) -> None:
        if (by_age is None) == (by_period is None):
            raise ModelInitializationError(
                "InitialNodes requires exactly one of by_age and by_period."
            )
        kind = "age" if by_age is not None else "period"
        selected = by_age if by_age is not None else by_period
        if not selected:
            raise ModelInitializationError(
                f"`InitialNodes.by_{kind}` must be nonempty."
            )
        normalized: dict[K, tuple[RegimeName, ...]] = {}
        for selector, value in selected.items():
            if kind == "age" and isinstance(selector, PeriodRange | Periods):
                raise ModelInitializationError(
                    "InitialNodes.by_age requires age selectors."
                )
            try:
                if kind == "period":
                    _fail_if_invalid_period_selector(selector)
                else:
                    _fail_if_invalid_age_selector(selector)
            except RegimeInitializationError as error:
                raise ModelInitializationError(str(error)) from error
            names = (value,) if isinstance(value, str) else tuple(value)
            if not names or any(
                not isinstance(name, str) or not name for name in names
            ):
                raise ModelInitializationError(
                    f"`InitialNodes.by_{kind}` requires nonempty regime names; "
                    f"got {value!r}."
                )
            normalized[selector] = tuple(sorted(set(names)))
        object.__setattr__(self, "by_age", None)
        object.__setattr__(self, "by_period", None)
        object.__setattr__(self, f"by_{kind}", MappingProxyType(normalized))

    @classmethod
    def _from_pairs(
        cls,
        *,
        pairs: frozenset[tuple[UserAge, RegimeName]],
        kind: Literal["age", "period"],
    ) -> InitialNodes:
        """Group resolved pairs by exact coordinate for publication or restoration."""
        names_by_coordinate: dict[UserAge, list[RegimeName]] = {}
        for coordinate, name in sorted(pairs):
            names_by_coordinate.setdefault(coordinate, []).append(name)
        if kind == "age":
            return cls(by_age=names_by_coordinate)
        return cls(by_period=names_by_coordinate)


if TYPE_CHECKING:
    type UserInitialNodes = (
        InitialNodes
        | Sequence[tuple[UserAge | float, RegimeName]]
        | AbstractSet[tuple[UserAge | float, RegimeName]]
        | Mapping[AgeSelector, RegimeName | Sequence[RegimeName]]
    )
else:
    # The runtime check admits any start coordinate and selector, so that the
    # model refuses a malformed age naming it rather than with a type violation.
    type UserInitialNodes = (
        InitialNodes
        | Sequence[tuple[object, RegimeName]]
        | AbstractSet[tuple[object, RegimeName]]
        | Mapping[object, RegimeName | Sequence[RegimeName]]
    )
