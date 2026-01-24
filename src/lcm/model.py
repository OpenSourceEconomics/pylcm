"""Collection of classes that are used by the user to define the model and grids."""

from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Any

from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, format_messages
from lcm.input_processing.process_params import process_params
from lcm.input_processing.regime_processing import InternalRegime, process_regimes
from lcm.logging import get_logger
from lcm.regime import Regime
from lcm.simulation.result import SimulationResult
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve
from lcm.typing import (
    FloatND,
    InternalParams,
    RegimeName,
    RegimeNamesToIds,
)
from lcm.utils import REGIME_SEPARATOR, get_field_names_and_values


class Model:
    """A model which is created from a regime.

    Upon initialization, an internal regime will be created which contains all
    the functions needed to solve and simulate the model.

    Attributes:
        description: Description of the model.
        n_periods: Number of periods in the model.
        enable_jit: Whether to jit the functions of the internal regime.
        regime_id: Immutable mapping from regime names to integer indices.
        regimes: The user provided regimes that contain the information
            about the model's regimes.
        internal_regimes: The internal regime instances created by LCM, which allow
            to solve and simulate the model.
        params_template: Template for the model parameters.

    """

    description: str | None = None
    ages: AgeGrid
    n_periods: int
    regime_names_to_ids: RegimeNamesToIds
    regimes: MappingProxyType[str, Regime]
    internal_regimes: MappingProxyType[str, InternalRegime]
    enable_jit: bool = True
    params_template: InternalParams

    def __init__(
        self,
        *,
        description: str | None = None,
        ages: AgeGrid,
        regimes: Mapping[str, Regime],
        regime_id_class: type,
        enable_jit: bool = True,
    ) -> None:
        """Initialize the Model.

        Args:
            regimes: Dict mapping regime names to Regime instances.
            ages: Age grid for the model.
            description: Description of the model.
            regime_id_class: Dataclass mapping regime names to integer indices.
            enable_jit: Whether to jit the functions of the internal regime.

        """
        # Create regime_id mapping from dict keys
        self.description = description
        self.ages = ages
        self.n_periods = ages.n_periods

        _validate_model_inputs(
            n_periods=self.n_periods, regimes=regimes, regime_id_class=regime_id_class
        )
        self.regime_names_to_ids = MappingProxyType(
            dict(
                sorted(
                    get_field_names_and_values(regime_id_class).items(),
                    key=lambda x: x[1],
                )
            )
        )
        self.regimes = MappingProxyType(dict(regimes))
        self.internal_regimes = MappingProxyType(
            process_regimes(
                regimes=regimes,
                ages=self.ages,
                regime_names_to_ids=self.regime_names_to_ids,
                enable_jit=enable_jit,
            )
        )
        self.enable_jit = enable_jit
        self.params_template = MappingProxyType(
            {
                name: regime.params_template
                for name, regime in self.internal_regimes.items()
            }
        )

    def solve(
        self,
        params: dict[str, Any],
        *,
        debug_mode: bool = True,
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
        """Solve the model using the pre-computed functions.

        Args:
            params: Model parameters matching the template from self.params_template
            debug_mode: Whether to enable debug logging

        Returns:
            Dictionary mapping period to a value function array for each regime.
        """
        internal_params = process_params(params, self.params_template)
        return solve(
            internal_params=internal_params,
            ages=self.ages,
            internal_regimes=self.internal_regimes,
            logger=get_logger(debug_mode=debug_mode),
        )

    def simulate(
        self,
        params: dict[str, Any],
        initial_states: Mapping[str, Array],
        initial_regimes: list[RegimeName],
        V_arr_dict: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
        *,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> SimulationResult:
        """Simulate the model forward using pre-computed value functions.

        Args:
            params: Model parameters matching the template from self.params_template.
            initial_states: Dict mapping state names to arrays. All arrays must have the
                same length (number of subjects). Each state name should correspond to a
                state variable defined in at least one regime.
            initial_regimes: List containing the names of the regimes the subjects
                start in.
            V_arr_dict: Value function arrays from solve().
            seed: Random seed.
            debug_mode: Whether to enable debug logging.

        Returns:
            SimulationResult object. Call .to_dataframe() to get a pandas DataFrame,
            optionally with additional_targets.

        """
        internal_params = process_params(params, self.params_template)
        return simulate(
            internal_params=internal_params,
            initial_states=initial_states,
            initial_regimes=initial_regimes,
            internal_regimes=self.internal_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            logger=get_logger(debug_mode=debug_mode),
            V_arr_dict=V_arr_dict,
            ages=self.ages,
            seed=seed,
        )

    def solve_and_simulate(
        self,
        params: dict[str, Any],
        initial_states: Mapping[str, Array],
        initial_regimes: list[RegimeName],
        *,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> SimulationResult:
        """Solve and then simulate the model in one call.

        Args:
            params: Model parameters matching the template from self.params_template.
            initial_states: Dict mapping state names to arrays. All arrays must have the
                same length (number of subjects). Each state name should correspond to a
                state variable defined in at least one regime.
            initial_regimes: List containing the names of the regimes the subjects
                start in.
            seed: Random seed.
            debug_mode: Whether to enable debug logging.

        Returns:
            SimulationResult object. Call .to_dataframe() to get a pandas DataFrame,
            optionally with additional_targets.

        """
        V_arr_dict = self.solve(params, debug_mode=debug_mode)
        return self.simulate(
            params=params,
            initial_states=initial_states,
            initial_regimes=initial_regimes,
            V_arr_dict=V_arr_dict,
            seed=seed,
            debug_mode=debug_mode,
        )


def _validate_model_inputs(  # noqa: C901
    n_periods: int,
    regimes: Mapping[str, Regime],
    regime_id_class: type,
) -> None:
    # Early exit if regimes are not lcm.Regime instances
    if not all(isinstance(regime, Regime) for regime in regimes.values()):
        raise ModelInitializationError(
            "All items in regimes must be instances of lcm.Regime."
        )

    error_messages: list[str] = []

    if not isinstance(n_periods, int):
        error_messages.append("n_periods must be an integer.")
    elif n_periods <= 1:
        error_messages.append("n_periods must be at least 2.")

    if not regimes:
        error_messages.append(
            "At least one non-terminal and one terminal regime must be provided."
        )

    # Validate regime names don't contain separator
    invalid_names = [name for name in regimes if REGIME_SEPARATOR in name]
    if invalid_names:
        error_messages.append(
            f"Regime names cannot contain the separator character "
            f"'{REGIME_SEPARATOR}'. The following names are invalid: {invalid_names}."
        )

    # Assume all items in regimes are lcm.Regime instances beyond this point
    terminal_regimes = [name for name, r in regimes.items() if r.terminal]
    if len(terminal_regimes) < 1:
        error_messages.append("lcm.Model must have at least one terminal regime.")

    non_terminal_regimes = {name: r for name, r in regimes.items() if not r.terminal}
    if len(non_terminal_regimes) < 1:
        error_messages.append("lcm.Model must have at least one non-terminal regime.")
    else:
        non_terminal_regimes_without_next_regime = [
            name
            for name, r in non_terminal_regimes.items()
            if "next_regime" not in r.transitions
        ]
        if non_terminal_regimes_without_next_regime:
            error_messages.append(
                "The following regimes are missing 'next_regime' in their transitions: "
                f"{non_terminal_regimes_without_next_regime}."
            )

    regime_id_fields = sorted(get_field_names_and_values(regime_id_class).keys())
    regime_names = sorted(regimes.keys())
    if regime_id_fields != regime_names:
        error_messages.append(
            f"regime_id_cls fields must match regime names.\n Got:"
            "regime_id_cls fields:\n"
            f"    {regime_id_fields}\n"
            "regime names:\n"
            f"    {regime_names}."
        )

    error_messages.extend(_validate_transition_completeness(regimes))

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitializationError(msg)


def _validate_transition_completeness(regimes: Mapping[str, Regime]) -> list[str]:
    """Validate that non-terminal regimes have complete transitions.

    Non-terminal regimes must have transition functions for ALL states across ALL
    regimes, since they can potentially transition to any other regime.

    Args:
        regimes: Mapping of regime names to regimes to validate.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    all_states = set(chain.from_iterable(r.states.keys() for r in regimes.values()))

    missing_transitions: dict[str, set[str]] = {}

    for name, regime in regimes.items():
        if regime.terminal:
            continue
        states_from_transitions = {
            fn_key.removeprefix("next_") for fn_key in regime.transitions
        }

        missing = all_states - states_from_transitions
        if missing:
            missing_transitions[name] = missing

    if missing_transitions:
        error = "Non-terminal regimes have missing transitions: "
        for regime_name, missing in sorted(missing_transitions.items()):
            missing_list = ", ".join(f"next_{s}" for s in sorted(missing))
            error += f"'{regime_name}': {missing_list}, "
        return [error]

    return []
