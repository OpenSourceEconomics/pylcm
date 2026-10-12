"""Native GridSearch feasibility summaries consumed only by solve diagnostics."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from _lcm.typing import QAndFArg
from lcm.solver_api import KernelOutput
from lcm.typing import BoolND, FloatND, IntND

type FeasibilityCounts = tuple[IntND, IntND]


@dataclass(frozen=True, kw_only=True)
class FeasibilityKernelOutput(KernelOutput):
    """Private scalar summaries beside the unchanged public artifact channels."""

    feasibility_counts: FeasibilityCounts
    """Counts of empty-action states and other infinite-valued states."""

    def __post_init__(self) -> None:
        """Require native scalar integer counters without a host transfer."""
        super().__post_init__()
        if len(self.feasibility_counts) != 2 or any(  # noqa: PLR2004
            value.ndim != 0 or value.dtype != jnp.int32
            for value in self.feasibility_counts
        ):
            raise TypeError(
                "GridSearch feasibility counts must be scalar int32 arrays."
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class CountedGridSearch:
    """Reduce canonical feasibility and final values inside the admitted program."""

    function: Callable[..., tuple[FloatND, BoolND]]
    collective: bool

    def __call__(
        self, **kwargs: QAndFArg
    ) -> tuple[FloatND, IntND, IntND] | tuple[FloatND, BoolND, IntND, IntND]:
        """Count state cells, treating a household as a single decision."""
        value, empty = self.function(**kwargs)
        infinite = jnp.isinf(value)
        if self.collective:
            infinite = jnp.any(infinite, axis=-1)
        empty_count = jnp.sum(empty, dtype=jnp.int32)
        other_inf_count = jnp.sum(infinite & ~empty, dtype=jnp.int32)
        if self.collective:
            return value, empty, empty_count, other_inf_count
        return value, empty_count, other_inf_count
