from __future__ import annotations

import dataclasses
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

from lcm.exceptions import RegimeInitializationError, format_messages
from lcm.grids import Grid

if TYPE_CHECKING:
    from lcm.typing import (
        UserFunction,
    )


@dataclass(frozen=True)
class Regime:
    """A user regime which can be processed into an internal regime.

    Attributes:
        description: Description of the regime.
        n_periods: Number of periods in the regime.
        functions: Dictionary of user provided functions that define the functional
            relationships between regime variables. It must include at least a function
            called 'utility'.
        actions: Dictionary of user provided actions.
        states: Dictionary of user provided states.

    """

    # regime specification information (provided by the User)
    name: str
    description: str | None = None
    _: KW_ONLY
    utility: UserFunction
    constraints: dict[str, UserFunction] = field(default_factory=dict)
    transitions: dict[str, UserFunction] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)

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
    function_collections = [regime.transitions, regime.constraints, regime.functions]
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
    error_messages = []

    if "utility" in regime.functions:
        error_messages.append(
            "The function name 'utility' is reserved and cannot be used in the "
            "functions dictionary. Please use the utility attribute instead.",
        )
    invalid_transitions = [
        tran_name
        for tran_name in regime.transitions
        if not tran_name.startswith("next_")
    ]
    if invalid_transitions:
        error_messages.append(
            "Each transitions name must start with 'next_'. "
            "The following transition names are invalid:"
            f"{invalid_transitions}.",
        )

    states = set(regime.states)
    states_via_transition = {s.removeprefix("next_") for s in regime.transitions}

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

    states_and_actions_overlap = set(regime.states) & set(regime.actions)
    if states_and_actions_overlap:
        error_messages.append(
            "States and actions cannot have overlapping names. The following names "
            f"are used in both states and actions: {states_and_actions_overlap}.",
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)
