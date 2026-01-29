"""Collection of classes that are used by the user to define the model and grids."""

from collections.abc import Mapping
from types import MappingProxyType

from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, format_messages
from lcm.grids import ShockGrid
from lcm.input_processing.regime_processing import InternalRegime, process_regimes
from lcm.logging import get_logger
from lcm.regime import Regime
from lcm.simulation.result import SimulationResult
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve
from lcm.typing import (
    FloatND,
    ParamsDict,
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
    fixed_params: ParamsDict
    params_template: ParamsDict

    def __init__(
        self,
        *,
        description: str | None = None,
        ages: AgeGrid,
        regimes: Mapping[str, Regime],
        regime_id_class: type,
        enable_jit: bool = True,
        fixed_params: ParamsDict | None = None,
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
        # Create regime_id mapping from dict keys
        self.description = description
        self.ages = ages
        self.n_periods = ages.n_periods
        self.fixed_params = fixed_params if fixed_params is not None else {}

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
        self.internal_regimes = MappingProxyType(
            process_regimes(
                regimes=regimes,
                ages=self.ages,
                regime_names_to_ids=self.regime_names_to_ids,
                fixed_params=self.fixed_params,
                enable_jit=enable_jit,
            )
        )
        self.enable_jit = enable_jit
        self.params_template = {
            name: regime.params_template
            for name, regime in self.internal_regimes.items()
        }

    def solve(
        self,
        params: ParamsDict,
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
        return solve(
            params=params,
            ages=self.ages,
            internal_regimes=self.internal_regimes,
            logger=get_logger(debug_mode=debug_mode),
        )

    def simulate(
        self,
        params: ParamsDict,
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
        return simulate(
            params=params,
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
        params: ParamsDict,
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
    fixed_params: ParamsDict,
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
    mising_fixed_params = _validate_fixed_params_present(regimes, fixed_params)
    if mising_fixed_params:
        error_messages.extend(mising_fixed_params)

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitializationError(msg)


def _validate_fixed_params_present(
    regimes: Mapping[str, Regime], fixed_params: ParamsDict
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
