from abc import ABC, abstractmethod

from lcm.exceptions import GridInitializationError
from lcm.typing import Float1D, Int1D


def _fail_if_batch_size_combined_with_distributed(
    *,
    batch_size: int,
    distributed: bool,
) -> None:
    """Reject `batch_size > 0` paired with `distributed=True` on one axis.

    Each Python-level batch is its own `jax.jit` dispatch in the solve
    loop, and on a distributed axis every dispatch carries a cross-device
    collective. Batching therefore multiplies the per-period collective
    count by `ceil(n_per_device / batch_size)`; for small `batch_size`
    the collective overhead per kernel dwarfs the compute per kernel and
    sharding becomes a regression rather than a speedup. Reject the
    combination at construction time so the foot-gun never reaches the
    solve loop.
    """
    if batch_size > 0 and distributed:
        raise GridInitializationError(
            f"`batch_size={batch_size}` is incompatible with "
            "`distributed=True` on a single grid axis: every batch "
            "triggers a per-period cross-device collective, multiplying "
            "the synchronisation count by ceil(n_per_device / batch_size) "
            "and inverting the compute/communication ratio. Use "
            "`batch_size=0` (one kernel per period over the full "
            "per-device chunk) and, if memory is tight, reduce the chunk "
            "by adding devices or another distributed axis."
        )


def _fail_if_continuous_grid_distributed(
    *,
    grid_kind: str,
    distributed: bool,
) -> None:
    """Reject `distributed=True` on a continuous grid.

    Sharding a continuous state axis forces every next-period interpolation
    lookup to read across the full grid, which compiles to an `all-gather`
    of the full V-array per device — the same pathology that produces
    cross-device V-array materialisation in backward induction.

    Sharding is supported only on discrete state grids, where cross-shard
    probability mass is contracted via the cheap `all-reduce` (output-sized,
    not V-array-sized).

    Should we ever want to enable continuous-axis sharding, lifting this guard
    is not sufficient on its own: the solve-side `max_Q_over_a` lowering in
    backward induction would also need an explicit `out_shardings` so the
    emitted V-array carries the declared sharding. Natural jit output inference
    reproduces the declared sharding for discrete-axis distribution, but not for
    a sharded continuous axis — there the solved V would otherwise come back on
    a layout its next-period consumers were not lowered against. The abandoned
    PR #364 carries a reference implementation of that `out_shardings` path.
    """
    if distributed:
        raise GridInitializationError(
            f"`distributed=True` is not allowed on continuous grids "
            f"(got `{grid_kind}`). Continuous-axis sharding forces an "
            "`all-gather` of the full V-array per device on every "
            "interpolation lookup. Shard on a discrete state grid "
            "instead; cross-shard transitions over discrete states are "
            "served by `all-reduce`."
        )


class Grid(ABC):
    """LCM Grid base class."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Size of the batches looped over during the solution.

        `ContinuousGrid` overrides this via its dataclass field.
        `DiscreteGrid` overrides this via its own property.

        """

    @property
    @abstractmethod
    def distributed(self) -> bool:
        """Whether to shard the grid's state axis across available devices.

        `ContinuousGrid` exposes this as a dataclass field; `DiscreteGrid`
        exposes it as a property over a private field.

        """

    @abstractmethod
    def to_jax(self) -> Int1D | Float1D:
        """Convert the grid to a Jax array."""
