from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lcm.exceptions import PyLCMError

if TYPE_CHECKING:
    from lcm.user_model import Model


def get_lcm_function(
    model: Model,  # noqa: ARG001
    *,
    targets: Literal["solve", "simulate", "solve_and_simulate"],  # noqa: ARG001
    debug_mode: bool = True,  # noqa: ARG001
    jit: bool = True,  # noqa: ARG001
) -> None:
    raise PyLCMError(
        "get_lcm_function() is deprecated. Use Model.solve(), Model.simulate(), "
        "or Model.solve_and_simulate() methods directly instead."
    )
