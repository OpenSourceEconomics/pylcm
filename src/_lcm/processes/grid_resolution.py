"""Explicit call-owned admission of parameter-dependent process support."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from jax.sharding import Sharding

from _lcm.processes.base import _ContinuousStochasticProcess
from lcm.typing import Float1D, ScalarFloat, ScalarInt


@runtime_checkable
class ProcessGridResolver(Protocol):
    """Resolve a process support into its required physical layout."""

    def supports(self, spec: _ContinuousStochasticProcess) -> bool: ...

    def __call__(
        self,
        *,
        spec: _ContinuousStochasticProcess,
        parameters: Mapping[str, ScalarFloat | ScalarInt],
        required: Sharding,
    ) -> Float1D: ...
