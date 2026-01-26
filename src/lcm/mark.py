"""Collection of LCM marking decorators."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class StochasticInfo:
    """Information on the stochastic nature of user provided functions."""


def stochastic(
    func: Callable[..., Any],
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> Callable[..., Any]:
    """Decorator to mark a function as stochastic and add information.

    Args:
        func (callable): The function to be decorated.
        *args (list): Positional arguments to be passed to the StochasticInfo.
        **kwargs (dict): Keyword arguments to be passed to the StochasticInfo.

    Returns:
        The decorated function

    """
    stochastic_info = StochasticInfo(*args, **kwargs)

    def decorator_stochastic(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper_mark_stochastic(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        wrapper_mark_stochastic._stochastic_info = stochastic_info  # ty: ignore[unresolved-attribute] # noqa: SLF001
        return wrapper_mark_stochastic

    return decorator_stochastic(func) if callable(func) else decorator_stochastic
