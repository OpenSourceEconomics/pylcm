"""Collection of LCM marking decorators."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class StochasticInfo:
    """Information on the stochastic nature of user provided functions."""

    distribution_type: str = "custom"


def stochastic(
    func: Callable[..., Any] | None = None,
    distribution_type: str = "custom",
) -> Callable[..., Any]:
    """Decorator to mark a function as stochastic and add information.

    Args:
        func (callable): The function to be decorated.
        distribution_type: Type of the stochastic transitions distribution.

    Returns:
        The decorated function

    """
    stochastic_info = StochasticInfo(distribution_type=distribution_type)

    def decorator_stochastic(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper_mark_stochastic(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        wrapper_mark_stochastic._stochastic_info = stochastic_info  # ty: ignore[unresolved-attribute]
        return wrapper_mark_stochastic

    return decorator_stochastic(func) if callable(func) else decorator_stochastic
