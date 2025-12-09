from __future__ import annotations

import dataclasses
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

from lcm.exceptions import RegimeInitializationError, format_messages
from lcm.grids import Grid
from lcm.utils import REGIME_SEPARATOR, flatten_regime_namespace

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lcm.typing import (
        UserFunction,
    )


@dataclass(frozen=True)
class Regime:
    """A user regime which can be processed into an internal regime.

    Attributes:
        name: Name of the regime.
        utility: Utility function for this regime.
        constraints: Dictionary of constraint functions.
        transitions: Dictionary of transition functions (keys must start with 'next_').
        functions: Dictionary of auxiliary functions.
        actions: Dictionary of action grids.
        states: Dictionary of state grids.
        absorbing: Whether this is an absorbing regime.
        terminal: Whether this is a terminal regime.
        active: Periods when regime is active. None means all periods.
        description: Description of the regime.

    """

    name: str
    _: KW_ONLY
    utility: UserFunction
    constraints: dict[str, UserFunction] = field(default_factory=dict)
    transitions: dict[str, UserFunction] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    absorbing: bool = False
    terminal: bool = False
    active: Iterable[int] | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        _validate_attribute_types(self)
        _validate_logical_consistency(self)

    def get_all_functions(self) -> dict[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Returns:
            Dictionary that maps names of all regime functions to the functions.

        """
        return (
            self.functions
            | {"utility": self.utility}
            | self.constraints
            | self.transitions
        )

    def replace(self, **kwargs: Any) -> Regime:  # noqa: ANN401
        """Replace the attributes of the regime.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the regime.

        Returns:
            A new regime with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise RegimeInitializationError(
                f"Failed to replace attributes of the regime. The error was: {e}"
            ) from e


def _validate_attribute_types(regime: Regime) -> None:  # noqa: C901, PLR0912
    """Validate the types of the regime attributes."""
    error_messages = []

    # Validate types of states and actions
    # ----------------------------------------------------------------------------------
    for attr_name in ("actions", "states"):
        attr = getattr(regime, attr_name)
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
    function_collections = [
        flatten_regime_namespace(regime.transitions),
        regime.constraints,
        regime.functions,
    ]
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

    if not callable(regime.utility):
        error_messages.append("utility must be a callable.")

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_logical_consistency(regime: Regime) -> None:
    """Validate the logical consistency of the regime."""
    error_messages: list[str] = []

    # Validate regime name does not contain the separator
    if REGIME_SEPARATOR in regime.name:
        error_messages.append(
            f"Regime name '{regime.name}' contains the reserved separator "
            f"'{REGIME_SEPARATOR}'. Please use a different name.",
        )

    # Validate function names do not contain the separator
    all_function_names = (
        list(regime.transitions.keys())
        + list(regime.constraints.keys())
        + list(regime.functions.keys())
    )
    invalid_function_names = [
        name for name in all_function_names if REGIME_SEPARATOR in name
    ]
    if invalid_function_names:
        error_messages.append(
            f"Function names cannot contain the reserved separator "
            f"'{REGIME_SEPARATOR}'. The following names are invalid: "
            f"{invalid_function_names}.",
        )

    # Validate state and action names do not contain the separator
    all_variable_names = list(regime.states.keys()) + list(regime.actions.keys())
    invalid_variable_names = [
        name for name in all_variable_names if REGIME_SEPARATOR in name
    ]
    if invalid_variable_names:
        error_messages.append(
            f"State and action names cannot contain the reserved separator "
            f"'{REGIME_SEPARATOR}'. The following names are invalid: "
            f"{invalid_variable_names}.",
        )

    if "utility" in regime.functions:
        error_messages.append(
            "The function name 'utility' is reserved and cannot be used in the "
            "functions dictionary. Please use the utility attribute instead.",
        )

    error_messages.extend(_validate_terminal_or_transitions(regime))
    error_messages.extend(_validate_active(regime.active))

    states_and_actions_overlap = set(regime.states) & set(regime.actions)
    if states_and_actions_overlap:
        error_messages.append(
            "States and actions cannot have overlapping names. The following names "
            f"are used in both states and actions: {states_and_actions_overlap}.",
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_terminal_or_transitions(regime: Regime) -> list[str]:
    """Validate terminal regime constraints or transition requirements."""
    errors: list[str] = []

    if regime.terminal:
        if regime.transitions:
            errors.append(
                "Terminal regimes cannot have transitions. Remove the transitions "
                "or set terminal=False.",
            )
        if not regime.states:
            errors.append(
                "Terminal regimes must have at least one state. The terminal utility "
                "function should depend on the states that agents bring into the "
                "terminal regime.",
            )
    else:
        # Validate transition function names start with 'next_'
        transitions_with_invalid_name = [
            fn_name for fn_name in regime.transitions if not fn_name.startswith("next_")
        ]
        if transitions_with_invalid_name:
            errors.append(
                "Each transitions name must start with 'next_'. The following "
                f"transition names are invalid: {transitions_with_invalid_name}.",
            )

        # Validate each state has a corresponding transition
        states = set(regime.states)
        states_via_transition = {
            fn_name.removeprefix("next_") for fn_name in regime.transitions
        }

        if states - states_via_transition:
            errors.append(
                "Each state must have a corresponding transition function. For the "
                "following states, no transition function was found: "
                f"{states - states_via_transition}.",
            )

    return errors


def _validate_active(active: Iterable[int] | None) -> list[str]:
    """Validate the active attribute."""
    if active is None:
        return []
    try:
        periods = list(active)
    except TypeError:
        return ["active must be iterable of ints or None."]
    errors: list[str] = []
    if not periods:
        errors.append("active cannot be empty. Use None for all periods.")
    elif not all(isinstance(p, int) for p in periods):
        errors.append("active must contain only integers.")
    elif any(p < 0 for p in periods):
        errors.append("active periods cannot be negative.")
    elif len(periods) != len(set(periods)):
        errors.append("active periods must be unique.")
    return errors
