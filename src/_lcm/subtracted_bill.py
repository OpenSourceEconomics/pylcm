"""The law of motion backing `lcm.SubtractedBill`.

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


class SubtractedBill:
    """Law of motion `next_<state> = resources - bill` for a continuous state.

    `resources` and `bill` name functions of the regime. The bill may read the
    draws of next period's stochastic states; `resources` must not. Declaring
    the law this way, rather than as a function computing the same difference,
    lets the solve average the target's value function over the bill once per
    period instead of interpolating it at every node of every draw the bill
    reads:

    $$W_\\theta(z) = \\sum_k w_k V(z - \\text{bill}(\\theta, \\varepsilon_k)),$$

    read at `z = resources`. The conditioners `theta` are every input of the
    bill that varies across the source's state-action points; each is a
    function or discrete variable of the source regime with the finite support
    given here. Everything else the bill reads must be a draw, the period or
    age, or a parameter.

    With the value function linear between its grid points, `W` is linear
    between the merged points `{a_j + bill_k}`, so storing it there makes the
    average exact rather than an approximation on the target's own grid.
    """

    resources: FunctionName
    """Function giving the state's next value before the bill is paid."""
    bill: FunctionName
    """Function giving the bill, which may read next period's draws."""
    conditioners: Mapping[FunctionName, tuple[int | bool, ...]]
    """Every source-side input of the bill, with its finite support.

    Supports hold integer codes or booleans, of the type the input takes.
    """

    def __init__(
        self,
        *,
        resources: FunctionName,
        bill: FunctionName,
        conditioners: Mapping[FunctionName, tuple[int | bool, ...]] = MappingProxyType(
            {}
        ),
    ) -> None:
        self.resources = resources
        self.bill = bill
        self.conditioners = MappingProxyType(
            {
                name: tuple(
                    bool(value) if isinstance(value, bool | np.bool_) else int(value)
                    for value in support
                )
                for name, support in conditioners.items()
            }
        )
        self.__name__ = f"{resources}_minus_{bill}"
        self.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=FloatND
                )
                for name in (resources, bill)
            ],
            return_annotation=ContinuousState,
        )
        self.__annotations__ = {
            resources: FloatND,
            bill: FloatND,
            "return": ContinuousState,
        }

    def __call__(self, **kwargs: FloatND) -> ContinuousState:
        return kwargs[self.resources] - kwargs[self.bill]
