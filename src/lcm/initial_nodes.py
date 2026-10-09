"""Explicit coordinates of admissible starting nodes."""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from beartype import beartype

from _lcm.beartype_conf import MODEL_CONF
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.transition import AgeSelector, _fail_if_invalid_age_selector
from lcm.typing import RegimeName


@dataclass(frozen=True, init=False)
class InitialNodes:
    """Declare admissible starting ages and regimes, without population weights.

    `by_age` maps an age selector to one regime name or a nonempty sequence or
    set of names. Its containers are copied and frozen. The model unions
    overlapping selectors and publishes exact grid ages with unique regime tuples.
    Grid membership and regime existence are checked when the model is built.
    """

    by_age: MappingProxyType[AgeSelector, tuple[RegimeName, ...]]
    """Age selectors and their sorted, unique regime names."""

    @beartype(conf=MODEL_CONF)
    def __init__[K](
        self,
        *,
        by_age: Mapping[K, str | Sequence[str] | AbstractSet[str]],
    ) -> None:
        if not by_age:
            raise ModelInitializationError("`InitialNodes.by_age` must be nonempty.")
        normalized: dict[AgeSelector, tuple[RegimeName, ...]] = {}
        for selector, value in by_age.items():
            try:
                _fail_if_invalid_age_selector(selector)
            except RegimeInitializationError as error:
                raise ModelInitializationError(str(error)) from error
            names = (value,) if isinstance(value, str) else tuple(value)
            if not names or any(
                not isinstance(name, str) or not name for name in names
            ):
                raise ModelInitializationError(
                    "`InitialNodes.by_age` requires nonempty regime names; "
                    f"got {value!r}."
                )
            normalized[cast("AgeSelector", selector)] = tuple(sorted(set(names)))
        object.__setattr__(self, "by_age", MappingProxyType(normalized))

    @classmethod
    def _from_pairs(cls, pairs: frozenset[tuple[object, RegimeName]]) -> InitialNodes:
        """Group resolved pairs by exact age for publication or restoration."""
        names_by_age: dict[object, list[RegimeName]] = {}
        for age, name in sorted(pairs):
            names_by_age.setdefault(age, []).append(name)
        return cls(by_age=names_by_age)


type UserInitialNodes = (
    InitialNodes
    | Sequence[tuple[object, str]]
    | AbstractSet[tuple[object, str]]
    | Mapping[object, str | Sequence[str]]
)
