"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from lcm.logging import get_logger
from lcm.regime import Regime
from lcm.simulation.simulate import simulate
from lcm.input_processing.regime_processing import process_regime
from lcm.solution.solve_brute import solve

if TYPE_CHECKING:
    from jax import Array

    from lcm.typing import (
        FloatND,
        ParamsDict,
    )

    from lcm.input_processing.regime_processing import InternalRegime


class Model:
    description: str | None = None
    n_periods: int
    enable_jit: bool = True
    regimes: Regime
    internal_regime: InternalRegime

    def __init__(
        self,
        regime: Regime,
        *,
        n_periods: int,
        description: str | None = None,
        enable_jit: bool = True,
    ):
        self.n_periods = n_periods
        self.description = description
        self.enable_jit = enable_jit
        self.regime = regime

        self.internal_regime = process_regime(regime=regime, enable_jit=enable_jit)

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
            state_action_spaces=self.internal_regime.state_action_spaces,
            max_Q_over_a_functions=self.internal_regime.max_Q_over_a_functions,
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
            argmax_and_max_Q_over_a_functions=self.internal_regime.argmax_and_max_Q_over_a_functions,
            internal_regime=self.internal_regime,
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