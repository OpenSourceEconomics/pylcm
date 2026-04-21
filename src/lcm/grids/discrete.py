import jax.numpy as jnp

from lcm.grids._dispatch import DispatchStrategy
from lcm.grids.base import Grid
from lcm.grids.categorical import _validate_discrete_grid
from lcm.typing import Int1D
from lcm.utils.containers import get_field_names_and_values


class DiscreteGrid(Grid):
    """A discrete grid defining the outcome space of a categorical variable.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique int values.
        dispatch: How this dim is compiled into the Bellman kernel. When
            `None` (the default), derived from `batch_size`: `0` →
            `DispatchStrategy.FUSED_VMAP`, `≥ 1` →
            `DispatchStrategy.CHUNKED_LAX_MAP`. Pass `DispatchStrategy.
            PARTITION_SCAN` or `DispatchStrategy.PARTITION_VMAP` to lift
            the dim out of the state-action space; see
            `docs/user_guide/dispatch.md` for the trade-offs.
        batch_size: Chunk size for `CHUNKED_LAX_MAP`. Must be `0` for
            every other strategy.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with int
            fields.
        ValueError: If `dispatch` and `batch_size` are inconsistent (e.g.
            `FUSED_VMAP` with `batch_size > 0`, `CHUNKED_LAX_MAP` with
            `batch_size = 0`, or any `PARTITION_*` with `batch_size > 0`).

    """

    def __init__(
        self,
        category_class: type,
        *,
        dispatch: DispatchStrategy | None = None,
        batch_size: int = 0,
    ) -> None:
        _validate_discrete_grid(category_class)
        names_and_values = get_field_names_and_values(category_class)
        self.__categories = tuple(names_and_values.keys())
        self.__codes = tuple(names_and_values.values())
        self.__ordered: bool = getattr(category_class, "_ordered", False)

        resolved = _resolve_dispatch(dispatch=dispatch, batch_size=batch_size)
        self.__dispatch: DispatchStrategy = resolved
        self.__batch_size: int = batch_size

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the list of category names."""
        return self.__categories

    @property
    def codes(self) -> tuple[int, ...]:
        """Return the list of category codes."""
        return self.__codes

    @property
    def ordered(self) -> bool:
        """Return whether the categories have a meaningful ordering."""
        return self.__ordered

    @property
    def batch_size(self) -> int:
        """Return the `jax.lax.map` chunk size for `CHUNKED_LAX_MAP`.

        Always `0` for any other `DispatchStrategy`.
        """
        return self.__batch_size

    @property
    def dispatch(self) -> DispatchStrategy:
        """Return how this dim is dispatched into the Bellman kernel."""
        return self.__dispatch

    def to_jax(self) -> Int1D:
        """Convert the grid to a Jax array."""
        return jnp.array(self.codes)


def _resolve_dispatch(
    *, dispatch: DispatchStrategy | None, batch_size: int
) -> DispatchStrategy:
    """Pick the `DispatchStrategy` from the user's kwargs and validate.

    `dispatch` is derived from `batch_size` when not passed explicitly —
    `batch_size=0` is `FUSED_VMAP`, `batch_size ≥ 1` is
    `CHUNKED_LAX_MAP`. An explicit `dispatch` with an inconsistent
    `batch_size` is an error.
    """
    if dispatch is None:
        if batch_size == 0:
            return DispatchStrategy.FUSED_VMAP
        if batch_size >= 1:
            return DispatchStrategy.CHUNKED_LAX_MAP
        msg = f"batch_size must be >= 0, got {batch_size}."
        raise ValueError(msg)

    if dispatch is DispatchStrategy.FUSED_VMAP and batch_size != 0:
        msg = (
            f"DispatchStrategy.FUSED_VMAP requires batch_size=0, "
            f"got batch_size={batch_size}."
        )
        raise ValueError(msg)
    if dispatch is DispatchStrategy.CHUNKED_LAX_MAP and batch_size < 1:
        msg = (
            f"DispatchStrategy.CHUNKED_LAX_MAP requires batch_size >= 1, "
            f"got batch_size={batch_size}."
        )
        raise ValueError(msg)
    if dispatch.is_partition_lifted and batch_size != 0:
        msg = (
            f"DispatchStrategy.{dispatch.name} requires batch_size=0, "
            f"got batch_size={batch_size}."
        )
        raise ValueError(msg)
    return dispatch
