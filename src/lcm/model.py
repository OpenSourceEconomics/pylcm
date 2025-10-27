"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lcm.exceptions import ModelInitializationError, format_messages
from lcm.input_processing.regime_processing import process_regime
from lcm.logging import get_logger
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve

if TYPE_CHECKING:
    import pandas as pd
    from jax import Array

    from lcm.input_processing.regime_processing import InternalRegime
    from lcm.regime import Regime
    from lcm.typing import (
        FloatND,
        ParamsDict,
    )


class Model:
    """A model which is created from a regime.

    Upon initialization, an internal regime will be created which contains all
    the functions needed to solve and simulate the model.

    Attributes:
        description: Description of the model.
        n_periods: Number of periods in the model.
        enable_jit: Whether to jit the functions of the internal regime.
        regime: The user provided regime that contains the information
            about the models regime.
        internal_regime: The internal regime instance created by LCM, which allows
            to solve and simulate the model.

    """

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
    ) -> None:
        """Initialize the Model.

        Args:
            regime: User provided regime.
            n_periods: Numper of periods of the model.
            description: Description of the model.
            enable_jit: Whether to jit the functions of the internal regime.

        """
        _validate_model_consistency(regime, n_periods)
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
            next_state_simulation_functions=self.internal_regime.next_state_simulation_functions,
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


def _validate_model_consistency(regime: Regime, n_periods: int) -> None:
    error_messages = []
    # Just an example validation
    if regime.n_periods != n_periods:
        error_messages.append(
            "The Regime needs to have the same number of periods as the Model."
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitializationError(msg)
