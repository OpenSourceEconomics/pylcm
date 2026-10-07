"""Private kernel declarations carrying support resolved from the model graph."""

from dataclasses import dataclass

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from lcm.transition import DeterministicTransition, StochasticTransition


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class _SupportedDeterministicTransition(DeterministicTransition):
    """A deterministic kernel tagged with its graph-resolved support."""

    targets: tuple[str, ...]


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class _SupportedStochasticTransition(StochasticTransition):
    """A probability kernel tagged with its graph-resolved support."""

    targets: tuple[str, ...]
