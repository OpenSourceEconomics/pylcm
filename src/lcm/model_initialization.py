"""Helper module for Model class initialization and utilities."""

from __future__ import annotations

import jax
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lcm.interfaces import InternalModel, StateActionSpace, StateSpaceInfo
    from lcm.typing import ArgmaxQOverAFunction, MaxQOverAFunction
    from lcm.user_model import Model


def set_frozen_attr(obj: Any, name: str, value: Any) -> None:
    """Robust attribute setting for frozen dataclasses.
    
    Args:
        obj: The frozen dataclass instance
        name: Name of the attribute to set
        value: Value to set
    """
    object.__setattr__(obj, name, value)


def get_processing_imports() -> dict[str, Any]:
    """Get imports needed for model processing.
    
    Returns:
        Dictionary of imported functions and classes needed for processing
    """
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


def initialize_model_functions(model: Model) -> None:
    """Initialize all pre-computed functions for the Model instance.
    
    This function handles the complex initialization logic that was previously
    in the Model.__post_init__ method. It processes the model, creates state-action
    spaces, and compiles optimization functions for each period.
    
    Args:
        model: The Model instance to initialize
        
    Raises:
        ModelInitilizationError: If initialization fails
    """
    from lcm.exceptions import ModelInitilizationError
    
    try:
        # Get required imports
        imports = get_processing_imports()
        process_model = imports['process_model']
        StateSpaceInfo = imports['StateSpaceInfo']
        get_argmax_and_max_Q_over_a = imports['get_argmax_and_max_Q_over_a']
        get_max_Q_over_a = imports['get_max_Q_over_a']
        get_Q_and_F = imports['get_Q_and_F']
        create_state_action_space = imports['create_state_action_space']
        create_state_space_info = imports['create_state_space_info']
        
        # Process model to internal representation
        internal_model = process_model(model)
        set_frozen_attr(model, 'internal_model', internal_model)
        set_frozen_attr(model, 'params_template', internal_model.params)
        
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
            max_Q_over_a_functions[period] = jax.jit(max_Q_over_a) if model.enable_jit else max_Q_over_a
            argmax_and_max_Q_over_a_functions[period] = (
                jax.jit(argmax_and_max_Q_over_a) if model.enable_jit else argmax_and_max_Q_over_a
            )
        
        # Set computed attributes using the robust setter
        set_frozen_attr(model, 'state_action_spaces', state_action_spaces)
        set_frozen_attr(model, 'state_space_infos', state_space_infos) 
        set_frozen_attr(model, 'max_Q_over_a_functions', max_Q_over_a_functions)
        set_frozen_attr(model, 'argmax_and_max_Q_over_a_functions', argmax_and_max_Q_over_a_functions)
        
    except Exception as e:
        raise ModelInitilizationError(
            f"Failed to initialize Model. Error during function compilation: {e}"
        ) from e