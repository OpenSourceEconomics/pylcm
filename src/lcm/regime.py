import dataclasses
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAliasType, overload

from dags.tree import QNAME_DELIMITER

from lcm.exceptions import RegimeInitializationError, format_messages
from lcm.grids import UNSET, DiscreteGrid, Grid, _Unset
from lcm.mark import stochastic
from lcm.shocks._base import _ShockGrid
from lcm.typing import (
    ActiveFunction,
    ContinuousState,
    DiscreteState,
    UserFunction,
)
from lcm.utils import (
    ensure_containers_are_immutable,
)


def _default_H(
    utility: float, continuation_value: float, discount_factor: float
) -> float:
    return utility + discount_factor * continuation_value


class _IdentityTransition:
    """Identity transition function for fixed states.

    Used by ``get_all_functions()`` so the params template includes fixed states.
    The ``_is_auto_identity`` attribute lets validation distinguish auto-generated
    identities from user-provided transitions.

    """

    _is_auto_identity: bool = True

    def __init__(self, state_name: str, *, annotation: TypeAliasType) -> None:
        self._state_name = state_name
        self.__name__ = f"next_{state_name}"
        param = inspect.Parameter(
            state_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
        )
        self.__signature__ = inspect.Signature(
            [param],
            return_annotation=annotation,
        )
        self.__annotations__ = {state_name: annotation, "return": annotation}

    @overload
    def __call__(self, **kwargs: DiscreteState) -> DiscreteState: ...
    @overload
    def __call__(self, **kwargs: ContinuousState) -> ContinuousState: ...
    def __call__(
        self, **kwargs: DiscreteState | ContinuousState
    ) -> DiscreteState | ContinuousState:
        return kwargs[self._state_name]


@dataclass(frozen=True)
class Regime:
    """A user regime which can be processed into an internal regime.

    State transitions are attached directly to state grids via their ``transition``
    parameter. A state with ``transition=some_fn`` is time-varying; a state with
    ``transition=None`` (the default) is fixed and carried forward unchanged.
    ShockGrids have intrinsic transitions and do not need a ``transition`` parameter.

    The ``transition`` field on the regime itself is the *regime* transition function.
    A regime with ``transition=None`` is terminal — no separate ``terminal`` flag is
    needed.

    Attributes:
        transition: Regime transition function, or ``None`` for terminal regimes.
        active: Callable that takes age (float) and returns True if regime is active.
        states: Dictionary of state grids (with transitions attached to grids).
        actions: Dictionary of action grids.
        functions: Dictionary of functions, must include a 'utility' function.
        constraints: Dictionary of constraint functions.
        description: Description of the regime.

    """

    transition: UserFunction | None
    active: ActiveFunction = lambda _age: True
    states: Mapping[str, Grid] = field(default_factory=lambda: MappingProxyType({}))
    actions: Mapping[str, Grid] = field(default_factory=lambda: MappingProxyType({}))
    functions: Mapping[str, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    constraints: Mapping[str, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    description: str = ""

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

    def get_all_functions(self) -> MappingProxyType[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Collects functions from three sources:
        - ``self.functions`` (utility, helpers, H)
        - ``self.constraints``
        - State transitions from grid ``transition`` attributes
        - The regime transition (``self.transition``, keyed as ``"next_regime"``)

        Returns:
            Read-only mapping of all regime functions.

        """
        result = (
            dict(self.functions)
            | dict(self.constraints)
            | _collect_state_transitions(self.states)
        )
        # Add regime transition
        if self.transition is not None:
            result["next_regime"] = self.transition
        return MappingProxyType(result)

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

    # Validate regime transition is callable if provided
    if regime.transition is not None and not callable(regime.transition):
        error_messages.append(
            "transition must be a callable or None, "
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
            if isinstance(transition, _Unset):
                error_messages.append(
                    f"State '{name}' must explicitly pass transition=<fn> or "
                    f"transition=None.",
                )

    # Action grids must not carry transitions
    for name, grid in regime.actions.items():
        transition = getattr(grid, "transition", UNSET)
        if not isinstance(transition, _Unset):
            error_messages.append(
                f"Action '{name}' must not have a transition (got "
                f"transition={transition!r}).",
            )

    return error_messages


def _make_identity_fn(
    state_name: str, *, annotation: TypeAliasType
) -> _IdentityTransition:
    """Create an identity transition for a fixed state.

    Convenience wrapper around ``_IdentityTransition``.

    """
    return _IdentityTransition(state_name, annotation=annotation)


def _collect_state_transitions(
    states: Mapping[str, Grid],
) -> dict[str, UserFunction]:
    """Collect state transition functions from grid objects.

    For each state grid, produces an entry ``f"next_{name}"`` mapped to:
    - A stochastic stub for ``ShockGrid`` types,
    - The grid's ``transition`` attribute if present, or
    - An auto-generated identity transition for fixed states.

    """
    transitions: dict[str, UserFunction] = {}
    for name, grid in states.items():
        if isinstance(grid, _ShockGrid):
            transitions[f"next_{name}"] = stochastic(lambda: None)
        elif callable(grid_transition := getattr(grid, "transition", None)):
            transitions[f"next_{name}"] = grid_transition
        else:
            ann = DiscreteState if isinstance(grid, DiscreteGrid) else ContinuousState
            transitions[f"next_{name}"] = _make_identity_fn(name, annotation=ann)
    return transitions
