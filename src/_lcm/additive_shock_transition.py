"""The law of motion backing `lcm.AdditiveShockTransition`.

A leaf module with no dependency on `Regime`, the validators, or the
regime-building code, so that the user-facing `lcm` namespace and the
engine-internal continuation builder can both import it without an import cycle.
"""

import inspect
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from _lcm.typing import FunctionName
from lcm.typing import ContinuousState, FloatND


class AdditiveShockTransition:
    """Law of motion `next_<state> = base + shock` for a continuous state.

    `base` and `shock` name functions of the regime. The shock may read the
    draws of next period's stochastic states; `base` must not. Declaring the law
    this way, rather than as a function computing the same sum, lets the solve
    average the target's value function over the shock once per period instead
    of interpolating it at every node of every draw the shock reads:

    $$W_\\theta(z) = \\sum_k w_k V(z + \\text{shock}(\\theta, \\varepsilon_k)),$$

    read at `z = base`. The conditioners `theta` are every input of the shock
    that varies across the source's state-action points; each is a function or
    discrete variable of the source regime with the finite support given here.
    Everything else the shock reads must be a draw, the period or age, or a
    parameter.

    With the value function linear between its grid points, `W` is linear
    between the merged points `{a_j - shock_k}`, so storing it there makes the
    average exact rather than an approximation on the target's own grid.
    """

    base: FunctionName
    """Function giving the state's next value before the shock."""
    shock: FunctionName
    """Function giving the shock added to the base, which may read next period's
    draws."""
    conditioners: Mapping[FunctionName, tuple[int | bool, ...]]
    """Every source-side input of the shock, with its finite support.

    Supports hold integer codes or booleans, of the type the input takes.
    """

    def __init__(
        self,
        *,
        base: FunctionName,
        shock: FunctionName,
        conditioners: Mapping[FunctionName, tuple[int | bool, ...]] = MappingProxyType(
            {}
        ),
    ) -> None:
        self.base = base
        self.shock = shock
        self.conditioners = MappingProxyType(
            {
                name: tuple(
                    bool(value) if isinstance(value, bool | np.bool_) else int(value)
                    for value in support
                )
                for name, support in conditioners.items()
            }
        )
        self.__name__ = f"{base}_plus_{shock}"
        self.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=FloatND
                )
                for name in (base, shock)
            ],
            return_annotation=ContinuousState,
        )
        self.__annotations__ = {
            base: FloatND,
            shock: FloatND,
            "return": ContinuousState,
        }

    def __call__(self, **kwargs: FloatND) -> ContinuousState:
        return kwargs[self.base] + kwargs[self.shock]
