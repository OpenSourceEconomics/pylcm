"""Collection of classes that are used by the user to define the model and grids."""

from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType

from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, format_messages
from lcm.grids import ShockGrid
from lcm.input_processing.process_params import create_params_template, process_params
from lcm.input_processing.regime_processing import InternalRegime, process_regimes
from lcm.input_processing.util import get_variable_info
from lcm.logging import get_logger
from lcm.regime import Regime
from lcm.simulation.result import SimulationResult
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve
from lcm.typing import (
    FloatND,
    MutableParamsTemplate,
    ParamsTemplate,
    RegimeName,
    RegimeNamesToIds,
    UserParams,
)
from lcm.utils import (
    REGIME_SEPARATOR,
    ensure_containers_are_mutable,
    get_field_names_and_values,
)


class Model:
    """A model which is created from a regime.

    Upon initialization, an internal regime will be created which contains all
    the functions needed to solve and simulate the model.

    Attributes:
        description: Description of the model.
        n_periods: Number of periods in the model.
        enable_jit: Whether to jit the functions of the internal regime.
        regime_names_to_ids: Mapping from regime names to integer indices.
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
    internal_regimes: MappingProxyType[RegimeName, InternalRegime]
    enable_jit: bool = True
    fixed_params: UserParams
    params_template: ParamsTemplate

    def __init__(
        self,
        *,
        description: str = "",
        ages: AgeGrid,
        regimes: Mapping[str, Regime],
        regime_id_class: type,
        enable_jit: bool = True,
        fixed_params: UserParams = MappingProxyType({}),
    ) -> None:
        """Initialize the Model.

        Args:
            regimes: Dict mapping regime names to Regime instances.
            ages: Age grid for the model.
            description: Description of the model.
            regime_id_class: Dataclass mapping regime names to integer indices.
            enable_jit: Whether to jit the functions of the internal regime.
            fixed_params: Parameters that can be fixed at model initialization.

        """
        self.description = description
        self.ages = ages
        self.n_periods = ages.n_periods
        self.fixed_params = fixed_params

        _validate_model_inputs(
            n_periods=self.n_periods,
            regimes=regimes,
            regime_id_class=regime_id_class,
            fixed_params=self.fixed_params,
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
        self.internal_regimes = process_regimes(
            regimes=regimes,
            ages=self.ages,
            regime_names_to_ids=self.regime_names_to_ids,
            fixed_params=self.fixed_params,
            enable_jit=enable_jit,
        )
        self.enable_jit = enable_jit
        self.params_template = create_params_template(self.internal_regimes)

    def get_params_template(self) -> MutableParamsTemplate:
        """Get a mutable copy of the params template.

        Returns a deep copy of the params_template where all immutable containers
        (MappingProxyType, tuple, frozenset) are converted to their mutable
        equivalents (dict, list, set).

        Returns:
            A mutable nested dict with the same structure as params_template.

        """
        return ensure_containers_are_mutable(  # ty: ignore[invalid-return-type]
            self.params_template
        )

    def solve(
        self,
        params: UserParams,
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
        params: UserParams,
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
        params: UserParams,
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
    fixed_params: UserParams,
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
    error_messages.extend(_validate_all_variables_used(regimes))
    error_messages.extend(_validate_fixed_params_present(regimes, fixed_params))

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
    non_terminal_regimes = {n: r for n, r in regimes.items() if not r.terminal}

    missing_transitions: dict[str, set[str]] = {}
    for name, regime in non_terminal_regimes.items():
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


def _validate_all_variables_used(regimes: Mapping[str, Regime]) -> list[str]:
    """Validate that all states and actions are used somewhere in each regime.

    Each state or action must appear in at least one of:
    - The concurrent valuation (utility or constraints)
    - A transition function

    Args:
        regimes: Mapping of regime names to regimes to validate.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages = []

    for regime_name, regime in regimes.items():
        variable_info = get_variable_info(regime)
        is_used = (
            variable_info["enters_concurrent_valuation"]
            | variable_info["enters_transition"]
        )
        unused_variables = variable_info.index[~is_used].tolist()

        if unused_variables:
            unused_states = [
                v for v in unused_variables if variable_info.loc[v, "is_state"]
            ]
            unused_actions = [
                v for v in unused_variables if variable_info.loc[v, "is_action"]
            ]

            msg_parts = []
            if unused_states:
                state_word = "state" if len(unused_states) == 1 else "states"
                msg_parts.append(f"{state_word} {unused_states}")
            if unused_actions:
                action_word = "action" if len(unused_actions) == 1 else "actions"
                msg_parts.append(f"{action_word} {unused_actions}")

            error_messages.append(
                f"The following variables are defined but never used in regime "
                f"'{regime_name}': {' and '.join(msg_parts)}. "
                f"Each state and action must be used in at least one of: "
                f"utility, constraints, or transition functions."
            )

    return error_messages


def _validate_fixed_params_present(
    regimes: Mapping[str, Regime], fixed_params: UserParams
) -> list[str]:
    """Return error messages if params for shocks are missing."""
    error_messages = []
    for regime_name, regime in regimes.items():
        fixed_params_needed = set()
        for state_name, state in regime.states.items():
            if isinstance(state, ShockGrid) and state.distribution_type in [
                "tauchen",
                "rouwenhorst",
            ]:
                fixed_params_needed.add(state_name)
        if fixed_params_needed - set(fixed_params):
            error_messages.append(
                f"Regime {regime_name} is missing fixed params:\n"
                f"{fixed_params_needed.difference(fixed_params)}"
            )
    return error_messages
