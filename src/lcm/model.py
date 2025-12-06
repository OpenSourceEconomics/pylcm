"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from lcm.exceptions import ModelInitializationError, format_messages
from lcm.grids import _get_field_names_and_values, validate_category_class
from lcm.input_processing.regime_processing import (
    create_default_regime_id_cls,
    process_regimes,
)
from lcm.logging import get_logger
from lcm.regime import Regime
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve

if TYPE_CHECKING:
    import pandas as pd
    from jax import Array

    from lcm.input_processing.regime_processing import InternalRegime
    from lcm.typing import (
        FloatND,
        ParamsDict,
        RegimeName,
    )


class Model:
    """A model which is created from a regime.

    Upon initialization, an internal regime will be created which contains all
    the functions needed to solve and simulate the model.

    Attributes:
        description: Description of the model.
        n_periods: Number of periods in the model.
        enable_jit: Whether to jit the functions of the internal regime.
        regime_id_cls: The RegimeID class mapping regime names to indices.
        regimes: The user provided regimes that contain the information
            about the model's regimes.
        internal_regimes: The internal regime instances created by LCM, which allow
            to solve and simulate the model.
        params_template: Template for the model parameters.

    """

    description: str | None = None
    n_periods: int
    enable_jit: bool = True
    regime_id_cls: type
    regimes: dict[str, Regime]
    internal_regimes: dict[str, InternalRegime]
    params_template: ParamsDict

    def __init__(
        self,
        regimes: Regime | list[Regime],
        *,
        n_periods: int,
        regime_id_cls: type | None = None,
        description: str | None = None,
        enable_jit: bool = True,
    ) -> None:
        """Initialize the Model.

        Args:
            regimes: User provided regimes.
            n_periods: Number of periods of the model.
            regime_id_cls: A dataclass mapping regime names to integer indices.
                Required for multi-regime models. Must not be provided for single-regime
                models (will be auto-generated).
            description: Description of the model.
            enable_jit: Whether to jit the functions of the internal regime.

        """
        if not isinstance(regimes, list):
            regimes = [regimes]

        self.n_periods = n_periods
        self.description = description
        self.enable_jit = enable_jit
        self.regimes = {}
        self.internal_regimes = {}

        _validate_model_inputs(
            n_periods=n_periods,
            regimes=regimes,
            regime_id_cls=regime_id_cls,
        )

        # Auto-generate regime_id_cls for single-regime models
        if len(regimes) == 1:
            if regime_id_cls is not None:
                warnings.warn(
                    f"Single-regime model '{regimes[0].name}' has a user-provided "
                    "'regime_id_cls', but this will be ignored. For single-regime "
                    "models, the regime_id_cls is auto-generated internally.",
                    UserWarning,
                    stacklevel=2,
                )
            self.regime_id_cls = create_default_regime_id_cls(regimes[0].name)
        else:
            # Multi-regime: regime_id_cls is required and validated
            self.regime_id_cls = regime_id_cls  # type: ignore[assignment]

        self.regimes = {regime.name: regime for regime in regimes}
        self.internal_regimes = process_regimes(
            regimes=regimes,
            n_periods=n_periods,
            regime_id_cls=self.regime_id_cls,
            enable_jit=enable_jit,
        )
        self.params_template = {
            name: regime.params_template
            for name, regime in self.internal_regimes.items()
        }

    def solve(
        self,
        params: ParamsDict,
        *,
        debug_mode: bool = True,
    ) -> dict[int, dict[RegimeName, FloatND]]:
        """Solve the model using the pre-computed functions.

        Args:
            params: Model parameters matching the template from self.params_template
            debug_mode: Whether to enable debug logging

        Returns:
            Dictionary mapping period to a value function array for each regime.
        """
        return solve(
            params=params,
            n_periods=self.n_periods,
            internal_regimes=self.internal_regimes,
            logger=get_logger(debug_mode=debug_mode),
        )

    def simulate(
        self,
        params: ParamsDict,
        initial_states: dict[RegimeName, dict[str, Array]],
        initial_regimes: list[RegimeName],
        V_arr_dict: dict[int, dict[RegimeName, FloatND]],
        *,
        additional_targets: dict[RegimeName, list[str]] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> dict[RegimeName, pd.DataFrame]:
        """Simulate the model forward using pre-computed functions.

        Args:
            params: Model parameters
            initial_states: Initial state values
            initial_regimes: List containing the names of the regimes the subjects
                start in.
            V_arr_dict: Value function arrays from solve()
            additional_targets: Additional targets to compute
            seed: Random seed
            debug_mode: Whether to enable debug logging

        Returns:
            Simulation results as dict mapping regime name to DataFrame
        """
        logger = get_logger(debug_mode=debug_mode)

        return simulate(
            params=params,
            initial_states=initial_states,
            initial_regimes=initial_regimes,
            internal_regimes=self.internal_regimes,
            regime_id_cls=self.regime_id_cls,
            logger=logger,
            V_arr_dict=V_arr_dict,
            additional_targets=additional_targets,
            seed=seed,
        )

    def solve_and_simulate(
        self,
        params: ParamsDict,
        initial_states: dict[RegimeName, dict[str, Array]],
        initial_regimes: list[RegimeName],
        *,
        additional_targets: dict[RegimeName, list[str]] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> dict[RegimeName, pd.DataFrame]:
        """Solve and then simulate the model in one call.

        Args:
            params: Model parameters
            initial_states: Initial state values
            initial_regimes: List containing the names of the regimes the subjects
                start in.
            additional_targets: Additional targets to compute
            seed: Random seed
            debug_mode: Whether to enable debug logging

        Returns:
            Simulation results as dict mapping regime name to DataFrame
        """
        V_arr_dict = self.solve(params, debug_mode=debug_mode)
        return self.simulate(
            params=params,
            initial_states=initial_states,
            initial_regimes=initial_regimes,
            V_arr_dict=V_arr_dict,
            additional_targets=additional_targets,
            seed=seed,
            debug_mode=debug_mode,
        )


def _validate_model_inputs(
    n_periods: int,
    regimes: list[Regime],
    regime_id_cls: type | None,
) -> None:
    error_messages: list[str] = []

    if not isinstance(n_periods, int):
        error_messages.append("n_periods must be an integer.")
    elif n_periods <= 1:
        error_messages.append("n_periods must be at least 2.")

    error_messages.extend(
        [
            "regimes must be instances of lcm.Regime."
            for regime in regimes
            if not isinstance(regime, Regime)
        ]
    )

    # Single-regime: regime_id_cls warning is handled in Model.__init__

    # Multi-regime model validation
    if len(regimes) > 1:
        # Check next_regime is defined in each regime
        error_messages.extend(
            f"Multi-regime models require 'next_regime' in transitions for "
            f"each regime. Missing in regime '{regime.name}'."
            for regime in regimes
            if "next_regime" not in regime.transitions
        )

        # Check regime_id_cls is provided
        if regime_id_cls is None:
            error_messages.append(
                "regime_id_cls must be provided for multi-regime models."
            )
        else:
            # Validate regime_id_cls structure and names
            regime_names = [regime.name for regime in regimes]
            error_messages.extend(validate_regime_id_cls(regime_id_cls, regime_names))

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitializationError(msg)


def validate_regime_id_cls(
    regime_id_cls: type,
    regime_names: list[str],
) -> list[str]:
    """Validate RegimeID class against regime names.

    This validates that:
    - The class passes standard category class validation (dataclass, consecutive ints)
    - Attribute names exactly match the regime names

    Args:
        regime_id_cls: The user-provided RegimeID dataclass.
        regime_names: List of regime names from the model.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages: list[str] = []

    # Reuse category class validation (dataclass, scalar ints, consecutive from 0)
    category_errors = validate_category_class(regime_id_cls)
    error_messages.extend(f"regime_id_cls: {error}" for error in category_errors)

    # If basic validation failed, skip attribute name check
    if category_errors:
        return error_messages

    # Check attribute names match regime names
    regime_id_names = set(_get_field_names_and_values(regime_id_cls).keys())
    regime_name_set = set(regime_names)

    if regime_id_names != regime_name_set:
        missing = regime_name_set - regime_id_names
        extra = regime_id_names - regime_name_set
        if missing:
            error_messages.append(
                f"regime_id_cls is missing attributes for regimes: {missing}"
            )
        if extra:
            error_messages.append(
                f"regime_id_cls has extra attributes not matching any regime: {extra}"
            )

    return error_messages
