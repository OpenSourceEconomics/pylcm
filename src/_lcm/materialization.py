"""Structural interface to the existing call-owned numeric allocation boundary.

No budget, cache or alternative allocation policy is defined here. The entry
owner already admits host uploads and compiler-profiled pure array producers.
"""

from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import jax

from _lcm.dtypes import CanonicalArrayWriter
from _lcm.grids import Grid
from _lcm.typing import PytreeValue
from lcm.typing import Float1D, Int1D, ReferenceName


@runtime_checkable
class DatedGenerationMaterializer(CanonicalArrayWriter, Protocol):
    """Borrow the same owner for validated uploads and real template producers."""

    def copy_solution_leaf(self, *, leaf: jax.Array, label: str) -> jax.Array:
        """Admit one independent original-layout device copy before allocation."""
        ...

    def materialize_grid(
        self, *, spec: Grid, placeholder: bool = False
    ) -> Float1D | Int1D:
        """Admit original grid producers with their existing eager boundaries."""
        ...

    def evaluate_validation[T: PytreeValue](
        self,
        *,
        function: Callable[..., T],
        arguments: Mapping[ReferenceName, PytreeValue],
    ) -> T:
        """Admit each original eager primitive, retaining callback captures."""
        ...

    def produce[T: PytreeValue](
        self,
        *,
        function: Callable[..., T],
        arguments: Mapping[ReferenceName, PytreeValue],
    ) -> T:
        """Admit the enclosing executable before dispatch; retain its ready leaves."""
        ...
