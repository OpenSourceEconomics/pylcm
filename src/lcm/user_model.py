"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

import dataclasses
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

from lcm.exceptions import ModelInitilizationError, format_messages
from lcm.grids import Grid
from lcm.logging import get_logger
from lcm.model_initialization import initialize_model_components
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve

if TYPE_CHECKING:
    import pandas as pd
    from jax import Array

    from lcm.interfaces import InternalModel, StateActionSpace, StateSpaceInfo
    from lcm.typing import (
        ArgmaxQOverAFunction,
        FloatND,
        MaxQOverAFunction,
        ParamsDict,
        UserFunction,
    )


@dataclass(frozen=True)
class Model:
    """A user model which can be processed into an internal model.

    Attributes:
        description: Description of the model.
        n_periods: Number of periods in the model.
        functions: Dictionary of user provided functions that define the functional
            relationships between model variables. It must include at least a function
            called 'utility'.
        actions: Dictionary of user provided actions.
        states: Dictionary of user provided states.

    """

    # Model specification information (provided by the User)
    description: str | None = None
    _: KW_ONLY
    n_periods: int
    utility: UserFunction
    constraints: dict[str, UserFunction] = field(default_factory=dict)
    transitions: dict[str, UserFunction] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    enable_jit: bool = True

    # Computed model components (set in __post_init__)
    internal_model: InternalModel = field(init=False)
    params_template: ParamsDict = field(init=False)
    state_action_spaces: dict[int, StateActionSpace] = field(init=False)
    state_space_infos: dict[int, StateSpaceInfo] = field(init=False)
    max_Q_over_a_functions: dict[int, MaxQOverAFunction] = field(init=False)
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = field(
        init=False
    )

    def __post_init__(self) -> None:
        _validate_attribute_types(self)
        _validate_logical_consistency(self)
        initialize_model_components(self)

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

    def replace(self, **kwargs: Any) -> Model:  # noqa: ANN401
        """Replace the attributes of the model.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the model.

        Returns:
            A new model with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise ModelInitilizationError(
                f"Failed to replace attributes of the model. The error was: {e}"
            ) from e


def _validate_attribute_types(model: Model) -> None:  # noqa: C901, PLR0912
    """Validate the types of the model attributes."""
    error_messages = []

    # Validate types of states and actions
    # ----------------------------------------------------------------------------------
    for attr_name in ("actions", "states"):
        attr = getattr(model, attr_name)
        if isinstance(attr, dict):
            for k, v in attr.items():
                if not isinstance(k, str):
                    error_messages.append(f"{attr_name} key {k} must be a string.")
                if not isinstance(v, Grid):
                    error_messages.append(f"{attr_name} value {v} must be an LCM grid.")
        else:
            error_messages.append(f"{attr_name} must be a dictionary.")

    # Validate types of functions
    # ----------------------------------------------------------------------------------
    function_collections = [model.transitions, model.constraints, model.functions]
    for func_collection in function_collections:
        if isinstance(func_collection, dict):
            for k, v in func_collection.items():
                if not isinstance(k, str):
                    error_messages.append(
                        f"function keys must be a strings, but is {k}."
                    )
                if not callable(v):
                    error_messages.append(
                        f"function values must be a callable, but is {v}."
                    )
        else:
            error_messages.append(
                "transitions, constraints, and functions must be a dictionary of "
                "callables."
            )

    if not callable(model.utility):
        error_messages.append("utility must be a callable.")

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitilizationError(msg)


def _validate_logical_consistency(model: Model) -> None:
    """Validate the logical consistency of the model."""
    error_messages = []

    if model.n_periods < 1:
        error_messages.append("Number of periods must be a positive integer.")

    if "utility" in model.functions:
        error_messages.append(
            "The function name 'utility' is reserved and cannot be used in the "
            "functions dictionary. Please use the utility attribute instead.",
        )
    invalid_transitions = [
        tran_name
        for tran_name in model.transitions
        if not tran_name.startswith("next_")
    ]
    if invalid_transitions:
        error_messages.append(
            "Each transitions name must start with 'next_'. "
            "The following transition names are invalid:"
            f"{invalid_transitions}.",
        )

    states = set(model.states)
    states_via_transition = {s.removeprefix("next_") for s in model.transitions}

    if states - states_via_transition:
        error_messages.append(
            "Each state must have a corresponding transition function. For the "
            f"following states, no transition function was found: "
            f"{states - states_via_transition}.",
        )

    if states_via_transition - states:
        error_messages.append(
            "Each transition function must correspond to a state. For the following "
            f"transition functions, no corresponding state was found: "
            f"{states_via_transition - states}.",
        )

    states_and_actions_overlap = set(model.states) & set(model.actions)
    if states_and_actions_overlap:
        error_messages.append(
            "States and actions cannot have overlapping names. The following names "
            f"are used in both states and actions: {states_and_actions_overlap}.",
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitilizationError(msg)
