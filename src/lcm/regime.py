from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from lcm.exceptions import RegimeInitializationError, format_messages
from lcm.grids import Grid
from lcm.typing import UserFunction


@dataclass(frozen=True, kw_only=True)
class Regime:
    """A modular component defining a consistent state-action space and functions.

    Each Regime represents a distinct behavioral environment where the agent
    has a specific set of available states, actions, and functions.

    Attributes:
        name: Unique identifier for this regime.
        description: Optional description of what this regime represents.
        active: An iterable of periods during which this regime is active.
        actions: Dictionary of action variables and their grids for this regime.
        states: Dictionary of state variables and their grids for this regime.
        functions: Dictionary of functions specific to this regime.
        regime_transitions: Dictionary mapping the target regime names to a dictionary
            of functions that determine the state transition from this regime to the
            target regime. If empty, the regime is assumed to be absorbing.

    """

    name: str
    description: str | None = None
    active: Iterable[int]
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)
    regime_transitions: dict[str, dict[str, Callable[..., Any]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _validate_regime(self)


def _validate_regime(regime: Regime) -> None:
    """Validate the regime attributes.

    This function checks a Regime instance for basic specification errors.

    Raises:
        RegimeInitializationError: If any validation checks fail.

    """
    validators = [
        _validate_name_and_description_are_strings,
        _validate_active_periods_is_iterable_with_ints,
        _validate_states_and_actions_are_contain_grids,
        _validate_states_and_actions_no_overlap,
        _validate_functions_dict,
        _validate_utility_function_exists,
        _validate_each_state_has_next_state_function,
    ]

    error_messages = []
    for validator in validators:
        error_messages.extend(validator(regime))

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_name_and_description_are_strings(regime: Regime) -> list[str]:
    """Validate name and description attributes."""
    error_messages: list[str] = []

    if not isinstance(regime.name, str):
        error_messages.append("name must be a string.")

    if regime.description is not None and not isinstance(regime.description, str):
        error_messages.append("description must be a string or None.")

    return error_messages


def _validate_active_periods_is_iterable_with_ints(regime: Regime) -> list[str]:
    """Validate the active periods attribute."""
    error_messages = []

    if not isinstance(regime.active, Iterable):
        error_messages.append("active must be an iterable of integers.")
        return error_messages  # Skip further validation if not iterable

    non_int_periods: list[str] = []
    negative_periods_idx: list[int] = []
    for i, p in enumerate(regime.active):
        if not isinstance(p, int):
            non_int_periods.append(f"{p} (index: {i}, type: {type(p).__name__})")
        elif p < 0:
            negative_periods_idx.append(i)

    if non_int_periods:
        error_messages.append(
            "active must be an iterable of integers, but the following values are not: "
            f"{', '.join(non_int_periods)}."
        )
    if negative_periods_idx:
        error_messages.append(
            "active must be an iterable of non-negative integers, but the values at the"
            " following indices are negative: "
            f"{', '.join(map(str, negative_periods_idx))}."
        )

    return error_messages


def _validate_states_and_actions_are_contain_grids(regime: Regime) -> list[str]:
    """Validate states and actions dictionary attributes."""
    error_messages = []

    for attr_name in ("actions", "states"):
        attr = getattr(regime, attr_name)
        if isinstance(attr, dict):
            for k, v in attr.items():
                if not isinstance(k, str):
                    error_messages.append(f"{attr_name} key {k} must be a string.")
                if not isinstance(v, Grid):
                    error_messages.append(
                        f"{attr_name} value {v} must be a PyLCM grid, such as "
                        "lcm.DiscreteGrid or lcm.LinspaceGrid."
                    )
        else:
            error_messages.append(f"{attr_name} must be a dictionary.")

    return error_messages


def _validate_functions_dict(regime: Regime) -> list[str]:
    """Validate the functions dictionary attribute."""
    error_messages = []

    if isinstance(regime.functions, dict):
        non_string_keys: list[str] = []
        non_callable_values: list[str] = []
        for k, v in regime.functions.items():
            if not isinstance(k, str):
                non_string_keys.append(f"{k} (type: {type(k).__name__})")
            if not callable(v):
                non_callable_values.append(f"{v} (type: {type(v).__name__})")

        if non_string_keys:
            error_messages.append(
                "function keys must be strings, but the following keys are not: "
                f"{', '.join(non_string_keys)}."
            )
        if non_callable_values:
            error_messages.append(
                "function values must be callables, but the following values are not: "
                f"{', '.join(non_callable_values)}."
            )
    else:
        error_messages.append("functions must be a dictionary.")

    return error_messages


def _validate_utility_function_exists(regime: Regime) -> list[str]:
    """Validate that utility function is defined."""
    error_messages = []

    if "utility" not in regime.functions:
        error_messages.append(
            "Utility function is not defined. PyLCM expects a function with dictionary "
            "key 'utility' in the functions dictionary."
        )

    return error_messages


def _validate_each_state_has_next_state_function(regime: Regime) -> list[str]:
    """Validate that each state has a corresponding next state function."""
    error_messages = []

    states_without_next_func = [
        state for state in regime.states if f"next_{state}" not in regime.functions
    ]
    if states_without_next_func:
        error_messages.append(
            "Each state must have a corresponding next state function. For the "
            "following states, no next state function was found: "
            f"{states_without_next_func}."
        )

    return error_messages


def _validate_states_and_actions_no_overlap(regime: Regime) -> list[str]:
    """Validate that states and actions have no overlapping names."""
    error_messages = []

    states_and_actions_overlap = set(regime.states) & set(regime.actions)
    if states_and_actions_overlap:
        error_messages.append(
            "States and actions cannot have overlapping names. The following names "
            f"are used in both states and actions: {states_and_actions_overlap}."
        )

    return error_messages
