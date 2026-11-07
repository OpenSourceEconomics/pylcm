"""The base of the IID processes `lcm.processes` exposes.

`UniformIIDProcess`, `NormalIIDProcess`, `LogNormalIIDProcess` and
`NormalMixtureIIDProcess` each discretize a distribution whose draw does not
depend on the current node; `_IIDProcess` is what they share.
"""

from abc import abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

from _lcm.processes.base import (
    _ContinuousStochasticProcess,
)
from _lcm.typing import PRNGKeyND
from lcm.typing import ScalarFloat, ScalarInt

FOLD_REMOVED = (
    "The `fold=` argument was removed. Drop it: pylcm infers whether a shock "
    "carries an axis from how it is read. A shock read only as its draw "
    "`next_<shock>` by state transitions is integrated out of the continuation; "
    "one whose current value is read keeps its axis."
)


@dataclass(frozen=True, kw_only=True)
class _IIDProcess(_ContinuousStochasticProcess):
    """Base for IID processes — draw does not depend on previous value."""

    def __new__(
        cls,
        *args: object,  # noqa: PAN001 - forwarded to the base guard
        **kwargs: object,  # noqa: PAN001 - refuses removed keywords before any check
    ) -> Self:
        if "fold" in kwargs:
            raise TypeError(FOLD_REMOVED)
        return super().__new__(cls, *args, **kwargs)

    @abstractmethod
    def draw_shock(
        self, *, params: MappingProxyType[str, ScalarFloat | ScalarInt], key: PRNGKeyND
    ) -> ScalarFloat: ...
