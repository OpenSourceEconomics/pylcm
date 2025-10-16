"""Helper module for regime class initialization and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax

from lcm.exceptions import RegimeInitializationError
from lcm.input_processing import process_regime
from lcm.interfaces import StateActionSpace, StateSpaceInfo
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.Q_and_F import get_Q_and_F
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)

if TYPE_CHECKING:
    from lcm.regime import Regime
    from lcm.typing import ArgmaxQOverAFunction, MaxQOverAFunction


def initialize_regime_components(regime: Regime) -> None:
    """Initialize all pre-computed components for the regime instance.

    This function handles the complex initialization logic that was previously
    in the regime.__post_init__ method. It processes the regime, creates state-action
    spaces, and compiles optimization components for each period.

    Args:
        regime: The regime instance to initialize

    Raises:
        regimeInitilizationError: If initialization fails

    """
    try:
        components = _get_regime_components(regime)
        for name, component in components.items():
            _set_frozen_attr(regime, name, component)
    except Exception as e:
        raise RegimeInitializationError(
            f"Failed to initialize regime components: {e}"
        ) from e


def _get_regime_components(regime: Regime) -> dict[str, Any]:
    # Process regime to internal representation
    internal_regime = process_regime(regime)

    # Initialize containers
    last_period = internal_regime.n_periods - 1
    state_action_spaces: dict[int, StateActionSpace] = {}
    state_space_infos: dict[int, StateSpaceInfo] = {}
    max_Q_over_a_functions: dict[int, MaxQOverAFunction] = {}
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = {}

    # Create last period's next state space info
    last_periods_next_state_space_info = StateSpaceInfo(
        states_names=(),
        discrete_states={},
        continuous_states={},
    )

    # Create functions for each period (reversed order following Backward induction)
    for period in reversed(range(internal_regime.n_periods)):
        is_last_period = period == last_period

        # Create state action space
        state_action_space = create_state_action_space(
            internal_regime=internal_regime,
            is_last_period=is_last_period,
        )

        # Create state space info
        state_space_info = create_state_space_info(
            internal_regime=internal_regime,
            is_last_period=is_last_period,
        )

        # Determine next state space info
        if is_last_period:
            next_state_space_info = last_periods_next_state_space_info
        else:
            next_state_space_info = state_space_infos[period + 1]

        # Create Q and F functions
        Q_and_F = get_Q_and_F(
            internal_regime=internal_regime,
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
        max_Q_over_a_functions[period] = (
            jax.jit(max_Q_over_a) if regime.enable_jit else max_Q_over_a
        )
        argmax_and_max_Q_over_a_functions[period] = (
            jax.jit(argmax_and_max_Q_over_a)
            if regime.enable_jit
            else argmax_and_max_Q_over_a
        )

    return {
        "internal_regime": internal_regime,
        "params_template": internal_regime.params,
        "state_action_spaces": state_action_spaces,
        "state_space_infos": state_space_infos,
        "max_Q_over_a_functions": max_Q_over_a_functions,
        "argmax_and_max_Q_over_a_functions": argmax_and_max_Q_over_a_functions,
    }


def _set_frozen_attr(obj: Any, name: str, value: Any) -> None:  # noqa: ANN401
    """Robust attribute setting for frozen dataclasses.

    Args:
        obj: The frozen dataclass instance
        name: Name of the attribute to set
        value: Value to set

    """
    object.__setattr__(obj, name, value)
