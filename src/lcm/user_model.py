"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

import dataclasses
import jax
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

from lcm.exceptions import ModelInitilizationError, format_messages
from lcm.grids import Grid

# Lazy imports to avoid circular imports
def _get_processing_imports() -> dict[str, Any]:
    """Get imports needed for model processing."""
    from lcm.input_processing import process_model
    from lcm.interfaces import StateSpaceInfo
    from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
    from lcm.Q_and_F import get_Q_and_F
    from lcm.state_action_space import create_state_action_space, create_state_space_info
    
    return {
        'process_model': process_model,
        'StateSpaceInfo': StateSpaceInfo,
        'get_argmax_and_max_Q_over_a': get_argmax_and_max_Q_over_a,
        'get_max_Q_over_a': get_max_Q_over_a,
        'get_Q_and_F': get_Q_and_F,
        'create_state_action_space': create_state_action_space,
        'create_state_space_info': create_state_space_info,
    }

if TYPE_CHECKING:
    from lcm.interfaces import InternalModel, StateActionSpace, StateSpaceInfo
    from lcm.typing import (
        ArgmaxQOverAFunction,
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

    description: str | None = None
    _: KW_ONLY
    n_periods: int
    functions: dict[str, UserFunction] = field(default_factory=dict)
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    enable_jit: bool = True
    
    # Computed attributes (set in __post_init__)
    internal_model: InternalModel = field(init=False)
    params_template: ParamsDict = field(init=False)
    state_action_spaces: dict[int, StateActionSpace] = field(init=False)
    state_space_infos: dict[int, StateSpaceInfo] = field(init=False)
    max_Q_over_a_functions: dict[int, MaxQOverAFunction] = field(init=False)
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = field(init=False)

    def __post_init__(self) -> None:
        _validate_attribute_types(self)
        _validate_logical_consistency(self)
        
        try:
            # Get required imports
            imports = _get_processing_imports()
            process_model = imports['process_model']
            StateSpaceInfo = imports['StateSpaceInfo']
            get_argmax_and_max_Q_over_a = imports['get_argmax_and_max_Q_over_a']
            get_max_Q_over_a = imports['get_max_Q_over_a']
            get_Q_and_F = imports['get_Q_and_F']
            create_state_action_space = imports['create_state_action_space']
            create_state_space_info = imports['create_state_space_info']
            
            # Process model to internal representation
            internal_model = process_model(self)
            object.__setattr__(self, 'internal_model', internal_model)
            object.__setattr__(self, 'params_template', internal_model.params)
            
            # Initialize containers
            last_period = internal_model.n_periods - 1
            state_action_spaces: dict[int, Any] = {}
            state_space_infos: dict[int, Any] = {}
            max_Q_over_a_functions: dict[int, Any] = {}
            argmax_and_max_Q_over_a_functions: dict[int, Any] = {}
            
            # Create last period's next state space info
            last_periods_next_state_space_info = StateSpaceInfo(
                states_names=(),
                discrete_states={},
                continuous_states={},
            )
            
            # Create functions for each period (reversed order like get_lcm_function)
            for period in reversed(range(internal_model.n_periods)):
                is_last_period = period == last_period
                
                # Create state action space
                state_action_space = create_state_action_space(
                    model=internal_model,
                    is_last_period=is_last_period,
                )
                
                # Create state space info  
                state_space_info = create_state_space_info(
                    model=internal_model,
                    is_last_period=is_last_period,
                )
                
                # Determine next state space info
                if is_last_period:
                    next_state_space_info = last_periods_next_state_space_info
                else:
                    next_state_space_info = state_space_infos[period + 1]
                
                # Create Q and F functions
                Q_and_F = get_Q_and_F(
                    model=internal_model,
                    next_state_space_info=next_state_space_info,
                    period=period,
                )
                
                # Create optimization functions
                max_Q_over_a = get_max_Q_over_a(
                    Q_and_F=Q_and_F,
                    actions_names=tuple(state_action_space.continuous_actions)
                    + tuple(state_action_space.discrete_actions),
                    states_names=tuple(state_action_space.states),
                )
                
                argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
                    Q_and_F=Q_and_F,
                    actions_names=tuple(state_action_space.discrete_actions)
                    + tuple(state_action_space.continuous_actions),
                )
                
                # Store results
                state_action_spaces[period] = state_action_space
                state_space_infos[period] = state_space_info
                max_Q_over_a_functions[period] = jax.jit(max_Q_over_a) if self.enable_jit else max_Q_over_a
                argmax_and_max_Q_over_a_functions[period] = (
                    jax.jit(argmax_and_max_Q_over_a) if self.enable_jit else argmax_and_max_Q_over_a
                )
            
            # Set computed attributes using object.__setattr__ (frozen dataclass)
            object.__setattr__(self, 'state_action_spaces', state_action_spaces)
            object.__setattr__(self, 'state_space_infos', state_space_infos) 
            object.__setattr__(self, 'max_Q_over_a_functions', max_Q_over_a_functions)
            object.__setattr__(self, 'argmax_and_max_Q_over_a_functions', argmax_and_max_Q_over_a_functions)
            
        except Exception as e:
            raise ModelInitilizationError(
                f"Failed to initialize Model. Error during function compilation: {e}"
            ) from e

    def solve(
        self,
        params: ParamsDict,
        *,
        debug_mode: bool = True,
    ) -> dict[int, Any]:
        """Solve the model using the pre-computed functions.
        
        Args:
            params: Model parameters matching the template from self.params_template
            debug_mode: Whether to enable debug logging
        
        Returns:
            Dictionary mapping period to value function arrays
        """
        from lcm.logging import get_logger
        from lcm.solution.solve_brute import solve
        
        logger = get_logger(debug_mode=debug_mode)
        
        return solve(
            params=params,
            state_action_spaces=self.state_action_spaces,
            max_Q_over_a_functions=self.max_Q_over_a_functions,
            logger=logger,
        )

    def simulate(
        self,
        params: ParamsDict,
        initial_states: dict[str, Any],
        V_arr_dict: dict[int, Any],
        *,
        additional_targets: list[str] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> Any:
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
        from lcm.logging import get_logger
        from lcm.simulation.simulate import simulate
        
        logger = get_logger(debug_mode=debug_mode)
        
        return simulate(
            params=params,
            initial_states=initial_states,
            argmax_and_max_Q_over_a_functions=self.argmax_and_max_Q_over_a_functions,
            model=self.internal_model,
            logger=logger,
            V_arr_dict=V_arr_dict,
            additional_targets=additional_targets,
            seed=seed,
        )

    def solve_and_simulate(
        self,
        params: ParamsDict,
        initial_states: dict[str, Any],
        *,
        additional_targets: list[str] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> Any:
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


def _validate_attribute_types(model: Model) -> None:  # noqa: C901
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
    if isinstance(model.functions, dict):
        for k, v in model.functions.items():
            if not isinstance(k, str):
                error_messages.append(f"function keys must be a strings, but is {k}.")
            if not callable(v):
                error_messages.append(
                    f"function values must be a callable, but is {v}."
                )
    else:
        error_messages.append("functions must be a dictionary.")

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitilizationError(msg)


def _validate_logical_consistency(model: Model) -> None:
    """Validate the logical consistency of the model."""
    error_messages = []

    if model.n_periods < 1:
        error_messages.append("Number of periods must be a positive integer.")

    if "utility" not in model.functions:
        error_messages.append(
            "Utility function is not defined. LCM expects a function called 'utility' "
            "in the functions dictionary.",
        )

    states_without_next_func = [
        state for state in model.states if f"next_{state}" not in model.functions
    ]
    if states_without_next_func:
        error_messages.append(
            "Each state must have a corresponding next state function. For the "
            "following states, no next state function was found: "
            f"{states_without_next_func}.",
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
