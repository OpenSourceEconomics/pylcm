"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from lcm.exceptions import ModelInitializationError, format_messages
from lcm.interfaces import InternalRegime
from lcm.logging import get_logger
from lcm.regime import Regime
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve

if TYPE_CHECKING:
    from jax import Array

    from lcm.typing import (
        FloatND,
        ParamsDict,
    )

from lcm.input_processing.regime_processing import process_regimes


class Model:
    description: str | None = None
    n_periods: int
    enable_jit: bool = True
    internal_regimes: list[InternalRegime]

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

        self.internal_regimes: list[InternalRegime] = process_regimes(
            model=self, regimes=regimes
        )

    def solve(
        self,
        params: ParamsDict,
        *,
        debug_mode: bool = True,
    ) -> dict[int, FloatND]:
        """Solve the model using the pre-computed functions.

        Args:
            params: Model parameters matching the template from self.params_template
            debug_mode: Whether to enable debug logging

        Returns:
            Dictionary mapping period to value function arrays
        """
        return solve(
            params=params,
            state_action_spaces=self.state_action_spaces,
            max_Q_over_a_functions=self.max_Q_over_a_functions,
            logger=get_logger(debug_mode=debug_mode),
        )

    def simulate(
        self,
        params: ParamsDict,
        initial_states: dict[str, Array],
        V_arr_dict: dict[int, FloatND],
        *,
        additional_targets: list[str] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> pd.DataFrame:
        """Simulate the model forward using pre-computed functions.

        Args:
            params: Model parameters
            initial_states: Initial state values
            V_arr_dict: Value function arrays from solve()
            additional_targets: Additional targets to compute
            seed: Random seed
            debug_mode: Whether to enable debug logging

        Returns:
            Simulation results as DataFrame
        """
        logger = get_logger(debug_mode=debug_mode)

        return simulate(
            params=params,
            initial_states=initial_states,
            argmax_and_max_Q_over_a_functions=self.argmax_and_max_Q_over_a_functions,
            internal_model=self.internal_model,
            logger=logger,
            V_arr_dict=V_arr_dict,
            additional_targets=additional_targets,
            seed=seed,
        )

    def solve_and_simulate(
        self,
        params: ParamsDict,
        initial_states: dict[str, Array],
        *,
        additional_targets: list[str] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> pd.DataFrame:
        """Solve and then simulate the model in one call.

        Args:
            params: Model parameters
            initial_states: Initial state values
            additional_targets: Additional targets to compute
            seed: Random seed
            debug_mode: Whether to enable debug logging

        Returns:
            Simulation results as DataFrame
        """
        V_arr_dict = self.solve(params, debug_mode=debug_mode)
        return self.simulate(
            params=params,
            initial_states=initial_states,
            V_arr_dict=V_arr_dict,
            additional_targets=additional_targets,
            seed=seed,
            debug_mode=debug_mode,
        )

    def replace(self, **kwargs: Any) -> Model:
        """Replace the attributes of the model.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the model.

        Returns:
            A new model with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise ModelInitializationError(
                f"Failed to replace attributes of the model. The error was: {e}"
            ) from e


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
