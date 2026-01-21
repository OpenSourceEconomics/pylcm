"""Collection of LCM marking decorators."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class StochasticInfo:
    """Information on the stochastic nature of user provided functions."""


def stochastic(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to mark a function as stochastic.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function with stochastic metadata attached.

    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return func(*args, **kwargs)

    wrapper._stochastic_info = StochasticInfo()  # ty: ignore[unresolved-attribute]
    return wrapper
