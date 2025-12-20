"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

from lcm.exceptions import ModelInitializationError, format_messages
from lcm.grids import _get_field_names_and_values, validate_category_class
from lcm.input_processing.regime_processing import process_regimes
from lcm.logging import get_logger
from lcm.regime import Regime
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve

if TYPE_CHECKING:
    from collections.abc import Iterable

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
        regime_id_cls: The RegimeId class mapping regime names to indices.
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
        regimes: Iterable[Regime],
        *,
        n_periods: int,
        regime_id_cls: type,
        description: str | None = None,
        enable_jit: bool = True,
    ) -> None:
        """Initialize the Model.

        Args:
            regimes: User provided regimes.
            n_periods: Number of periods of the model.
            regime_id_cls: A dataclass mapping regime names to integer indices.
            description: Description of the model.
            enable_jit: Whether to jit the functions of the internal regime.

        """
        regimes_list = list(regimes)

        self.n_periods = n_periods
        self.regime_id_cls = regime_id_cls
        self.description = description
        self.enable_jit = enable_jit
        self.regimes = {}
        self.internal_regimes = {}

        _validate_model_inputs(
            n_periods=n_periods,
            regimes=regimes_list,
            regime_id_cls=regime_id_cls,
        )

        self.internal_regimes = process_regimes(
            regimes=regimes_list,
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
        initial_states: dict[str, Array],
        initial_regimes: list[RegimeName],
        V_arr_dict: dict[int, dict[RegimeName, FloatND]],
        *,
        additional_targets: dict[RegimeName, list[str]] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> dict[RegimeName, pd.DataFrame]:
        """Simulate the model forward using pre-computed functions.

        Args:
            params: Model parameters matching the template from self.params_template
            initial_states: Dict mapping state names to arrays. All arrays must have the
                same length (number of subjects). Each state name should correspond to a
                state variable defined in at least one regime.
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
        initial_states: dict[str, Array],
        initial_regimes: list[RegimeName],
        *,
        additional_targets: dict[RegimeName, list[str]] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> dict[RegimeName, pd.DataFrame]:
        """Solve and then simulate the model in one call.

        Args:
            params: Model parameters matching the template from self.params_template
            initial_states: Dict mapping state names to arrays. All arrays must have the
                same length (number of subjects). Each state name should correspond to a
                state variable defined in at least one regime.
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


def _validate_model_inputs(  # noqa: C901
    n_periods: int,
    regimes: list[Regime],
    regime_id_cls: type | None,
) -> None:
    error_messages: list[str] = []

    if not isinstance(n_periods, int):
        error_messages.append("n_periods must be an integer.")
    elif n_periods <= 1:
        error_messages.append("n_periods must be at least 2.")

    if not regimes:
        error_messages.append(
            "At least one non-terminal and one terminal regime must be provided."
        )

    if not all(isinstance(regime, Regime) for regime in regimes):
        error_messages.append("All items in regimes must be instances of lcm.Regime.")

        # Early exit if regimes are invalid, as further checks require lcm.Regime
        # instances
        if error_messages:
            msg = format_messages(error_messages)
            raise ModelInitializationError(msg)

    # Assume all items in regimes are lcm.Regime instances beyond this point
    terminal_regimes = [r for r in regimes if r.terminal]
    if len(terminal_regimes) < 1:
        error_messages.append(
            "lcm.Model must have at least one terminal regime, but none found."
        )

    non_terminal_regimes = [r for r in regimes if not r.terminal]
    if len(non_terminal_regimes) < 1:
        error_messages.append(
            "lcm.Model must have at least one non-terminal regime, but none found."
        )
    else:
        non_terminal_regimes_without_next_regime = [
            r.name for r in non_terminal_regimes if "next_regime" not in r.transitions
        ]
        if non_terminal_regimes_without_next_regime:
            error_messages.append(
                "The following regimes are missing 'next_regime' in their transitions: "
                f"{non_terminal_regimes_without_next_regime}."
            )

    regime_names = [r.name for r in regimes]
    if regime_id_cls is not None:
        error_messages.extend(_validate_regime_id_cls(regime_id_cls, regime_names))
    error_messages.extend(_validate_transition_completeness(regimes))

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitializationError(msg)


def _validate_regime_id_cls(
    regime_id_cls: type,
    regime_names: list[str],
) -> list[str]:
    """Validate RegimeId class against regime names.

    This validates that:
    - The class passes standard category class validation (dataclass, consecutive ints)
    - Attribute names exactly match the regime names

    Args:
        regime_id_cls: The user-provided RegimeId dataclass.
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


def _validate_transition_completeness(regimes: list[Regime]) -> list[str]:
    """Validate that non-terminal regimes have complete transitions.

    Non-terminal regimes must have transition functions for ALL states across ALL
    regimes, since they can potentially transition to any other regime.

    Args:
        regimes: List of regimes to validate.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    all_states = set(chain.from_iterable(r.states.keys() for r in regimes))

    missing_transitions: dict[str, set[str]] = {}

    for regime in [r for r in regimes if not r.terminal]:
        states_from_transitions = {
            fn_key.removeprefix("next_") for fn_key in regime.transitions
        }

        missing = all_states - states_from_transitions
        if missing:
            missing_transitions[regime.name] = missing

    if missing_transitions:
        error = "Non-terminal regimes have missing transitions: "
        for regime_name, missing in sorted(missing_transitions.items()):
            missing_list = ", ".join(f"next_{s}" for s in sorted(missing))
            error += f"'{regime_name}': {missing_list}, "
        return [error]

    return []
