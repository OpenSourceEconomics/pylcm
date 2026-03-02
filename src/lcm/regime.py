import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dags.tree import QNAME_DELIMITER

from lcm.exceptions import RegimeInitializationError, format_messages
from lcm.grids import Grid
from lcm.shocks._base import _ShockGrid
from lcm.typing import (
    ActiveFunction,
    UserFunction,
)
from lcm.utils import (
    Unset,
    ensure_containers_are_immutable,
)


@dataclass(frozen=True)
class RegimeTransition:
    """Deterministic regime transition (returns a regime index)."""

    func: UserFunction

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise RegimeInitializationError(
                "RegimeTransition.func must be callable, "
                f"got {type(self.func).__name__}."
            )


@dataclass(frozen=True)
class MarkovRegimeTransition:
    """Stochastic regime transition (returns probability array over regimes)."""

    func: UserFunction

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise RegimeInitializationError(
                f"MarkovRegimeTransition.func must be callable, "
                f"got {type(self.func).__name__}."
            )


def _default_H(
    utility: float, continuation_value: float, discount_factor: float
) -> float:
    return utility + discount_factor * continuation_value


@dataclass(frozen=True, kw_only=True)
class Regime:
    """A user regime which can be processed into an internal regime.

    State transitions are attached directly to state grids via their `transition`
    parameter. A state with `transition=some_func` is time-varying; a state with
    `transition=None` (the default) is fixed and carried forward unchanged.
    ShockGrids have intrinsic transitions and do not need a `transition` parameter.

    The `transition` field on the regime itself is the *regime* transition function.
    A regime with `transition=None` is terminal — no separate `terminal` flag is
    needed.

    """

    transition: RegimeTransition | MarkovRegimeTransition | None
    """Regime transition wrapper, or `None` for terminal regimes."""

    active: ActiveFunction = lambda _age: True
    """Callable that takes age (float) and returns True if regime is active."""

    states: Mapping[str, Grid] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of state variable names to grid objects."""

    actions: Mapping[str, Grid] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of action variable names to grid objects."""

    functions: Mapping[str, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of function names to callables; must include 'utility'."""

    constraints: Mapping[str, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of constraint names to constraint functions."""

    description: str = ""
    """Description of the regime."""

    @property
    def terminal(self) -> bool:
        """Whether this is a terminal regime (derived from transition being None)."""
        return self.transition is None

    def __post_init__(self) -> None:
        _validate_attribute_types(self)
        _validate_logical_consistency(self)

        def make_immutable(name: str) -> None:
            value = ensure_containers_are_immutable(getattr(self, name))
            object.__setattr__(self, name, value)

        # Inject default aggregation function H if not provided by user.
        # Terminal regimes don't need H since Q = U directly (no continuation value).
        if not self.terminal and "H" not in self.functions:
            object.__setattr__(self, "functions", {**self.functions, "H": _default_H})
        make_immutable("functions")
        make_immutable("states")
        make_immutable("actions")
        make_immutable("constraints")

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
        if isinstance(attr, Mapping):
            for k, v in attr.items():
                if not isinstance(k, str):
                    error_messages.append(f"{attr_name} key {k} must be a string.")
                if not isinstance(v, Grid):
                    error_messages.append(f"{attr_name} value {v} must be an LCM grid.")
        else:
            error_messages.append(f"{attr_name} must be a mapping.")

    # Validate types of function mappings (constraints and functions)
    # ----------------------------------------------------------------------------------
    function_collections = [
        regime.constraints,
        regime.functions,
    ]
    for func_collection in function_collections:
        if isinstance(func_collection, Mapping):
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
                "constraints and functions must each be a mapping of callables."
            )

    # Validate regime transition type
    if regime.transition is not None and not isinstance(
        regime.transition, (RegimeTransition, MarkovRegimeTransition)
    ):
        error_messages.append(
            "transition must be a RegimeTransition, MarkovRegimeTransition, or None, "
            f"but is {type(regime.transition).__name__}."
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_logical_consistency(regime: Regime) -> None:
    """Validate the logical consistency of the regime."""
    error_messages: list[str] = []

    # Validate function names do not contain the separator
    all_function_names = [*regime.constraints.keys(), *regime.functions.keys()]
    invalid_function_names = [
        name for name in all_function_names if QNAME_DELIMITER in name
    ]
    if invalid_function_names:
        error_messages.append(
            f"Function names cannot contain the reserved separator "
            f"'{QNAME_DELIMITER}'. The following names are invalid: "
            f"{invalid_function_names}.",
        )

    # Validate state and action names do not contain the separator
    all_variable_names = [*regime.states.keys(), *regime.actions.keys()]
    invalid_variable_names = [
        name for name in all_variable_names if QNAME_DELIMITER in name
    ]
    if invalid_variable_names:
        error_messages.append(
            f"State and action names cannot contain the reserved separator "
            f"'{QNAME_DELIMITER}'. The following names are invalid: "
            f"{invalid_variable_names}.",
        )

    if "utility" not in regime.functions:
        error_messages.append(
            "A 'utility' function must be provided in the functions dictionary.",
        )

    error_messages.extend(_validate_active(regime.active))
    error_messages.extend(_validate_state_and_action_transitions(regime))

    states_and_actions_overlap = set(regime.states) & set(regime.actions)
    if states_and_actions_overlap:
        error_messages.append(
            "States and actions cannot have overlapping names. The following names "
            f"are used in both states and actions: {states_and_actions_overlap}.",
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_active(active: ActiveFunction) -> list[str]:
    """Validate the active attribute is a callable."""
    if not callable(active):
        return ["active must be a callable that takes age (float) and returns bool."]
    return []


def _validate_state_and_action_transitions(regime: Regime) -> list[str]:
    """Validate transition attributes on state and action grids."""
    error_messages: list[str] = []

    # State grids must have explicit transition
    for name, grid in regime.states.items():
        if not isinstance(grid, _ShockGrid):
            transition = getattr(grid, "transition", None)
            if isinstance(transition, Unset):
                error_messages.append(
                    f"State '{name}' must explicitly pass transition=<fn> or "
                    f"transition=None.",
                )

    # Action grids must not carry transitions
    for name, grid in regime.actions.items():
        transition = getattr(grid, "transition", Unset())
        if not isinstance(transition, Unset):
            error_messages.append(
                f"Action '{name}' must not have a transition (got "
                f"transition={transition!r}).",
            )

    return error_messages
