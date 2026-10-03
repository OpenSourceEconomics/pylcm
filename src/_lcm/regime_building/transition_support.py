"""Private kernel declarations carrying support resolved from the model graph."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from lcm.transition import (
    AgeSelector,
    DeterministicTransition,
    StochasticTransition,
)

type _Support = tuple[str, ...] | Mapping[str, AgeSelector]


def _freeze_support(targets: _Support) -> _Support:
    """Copy already validated graph metadata, including unavailable phases."""
    return (
        MappingProxyType(dict(targets))
        if isinstance(targets, Mapping)
        else tuple(targets)
    )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class _SupportedDeterministicTransition(DeterministicTransition):
    """A deterministic kernel tagged with its graph-resolved support."""

    targets: _Support

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "targets", _freeze_support(self.targets))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class _SupportedStochasticTransition(StochasticTransition):
    """A probability kernel tagged with its graph-resolved support."""

    targets: _Support

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "targets", _freeze_support(self.targets))
