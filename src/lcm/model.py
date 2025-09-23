"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lcm.regime import Regime

if TYPE_CHECKING:
    from collections.abc import Sequence

from lcm.exceptions import ModelInitializationError, format_messages
from lcm.input_processing.regime_processing import InternalRegime, process_regimes


class Model:
    def __init__(
        self,
        regimes: Regime | Sequence[Regime],
        *,
        n_periods: int,
        description: str | None = None,
        jit: bool = True,
    ):
        _validate_input_types(
            regimes=regimes,
            n_periods=n_periods,
            description=description,
            jit=jit,
        )

        self.n_periods = n_periods
        self.description = description
        self.jit = jit

        self.internal_regimes: list[InternalRegime] = process_regimes(regimes)


def _validate_input_types(
    regimes: Regime | Sequence[Regime],
    n_periods: int,
    description: str | None,
    jit: bool,
) -> None:
    error_messages = []

    if isinstance(regimes, Regime):
        regimes = [regimes]

    if not isinstance(regimes, Sequence) or not all(
        isinstance(r, Regime) for r in regimes
    ):
        error_messages.append(
            f"'regimes' must be a Regime instance or a Sequence of Regime instances. "
            f"Got {regimes} of type {type(regimes)}."
        )

    active_periods = {p for regime in regimes for p in regime.active}
    if active_periods != set(range(n_periods)):
        error_messages.append(
            f"Regimes must cover all periods from 0 to {n_periods - 1}. "
            f"Got active periods: {active_periods}."
        )

    if not isinstance(n_periods, int) or n_periods <= 0:
        error_messages.append(
            f"'n_periods' must be a positive integer. Got {n_periods} of type "
            f"{type(n_periods)}."
        )

    if description is not None and not isinstance(description, str):
        error_messages.append(
            f"'description' must be a string or None. Got {description} of type "
            f"{type(description)}."
        )

    if not isinstance(jit, bool):
        error_messages.append(
            f"'jit' must be a boolean. Got {jit} of type {type(jit)}."
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitializationError(msg)
