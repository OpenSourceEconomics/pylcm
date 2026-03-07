import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAliasType, overload

from dags.tree import QNAME_DELIMITER

from lcm.exceptions import RegimeInitializationError, format_messages
from lcm.grids import DiscreteGrid, Grid
from lcm.shocks._base import _ShockGrid
from lcm.typing import (
    ActiveFunction,
    ContinuousState,
    DiscreteState,
    FloatND,
    UserFunction,
)
from lcm.utils import (
    ensure_containers_are_immutable,
)


@dataclass(frozen=True)
class MarkovTransition:
    """Wrapper marking a transition function as stochastic (Markov).

    Wrap a transition function in `MarkovTransition` to indicate that it returns
    a probability distribution over next states (for state transitions) or over
    next regimes (for regime transitions), rather than a deterministic next value.

    Use at both the state and regime level:

        # Stochastic state transition (in Regime.state_transitions)
        state_transitions={"health": MarkovTransition(health_probs)}

        # Stochastic regime transition
        Regime(transition=MarkovTransition(regime_probs), ...)

    A bare callable (without the wrapper) is deterministic at both levels.

    """

    func: Callable[..., FloatND]
    """The transition function returning a probability distribution."""

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise RegimeInitializationError(
                f"MarkovTransition requires a callable, "
                f"but got {type(self.func).__name__}: {self.func!r}"
            )
        # Copy __wrapped__ and __annotations__ from the wrapped function so
        # that inspect.signature and dags see the original signature. We use
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "__wrapped__", self.func)
        object.__setattr__(
            self, "__annotations__", getattr(self.func, "__annotations__", {})
        )

    def __call__(self, *args: Any, **kwargs: Any) -> FloatND:  # noqa: ANN401
        return self.func(*args, **kwargs)


def _default_H(
    utility: float, continuation_value: float, discount_factor: float
) -> float:
    return utility + discount_factor * continuation_value


class _IdentityTransition:
    """Identity transition function for fixed states.

    Used by `get_all_functions()` so the params template includes fixed states.
    The `_is_auto_identity` attribute lets validation distinguish auto-generated
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


@dataclass(frozen=True, kw_only=True)
class Regime:
    """A user regime which can be processed into an internal regime.

    State transitions are specified via `state_transitions`, mapping state names to
    transition functions. A bare callable is deterministic; wrap in `MarkovTransition`
    for stochastic transitions. `None` marks a fixed state (identity auto-generated).
    ShockGrids have intrinsic transitions and must not appear in `state_transitions`.

    The `transition` field on the regime itself is the *regime* transition function.
    A regime with `transition=None` is terminal — no separate `terminal` flag is
    needed.

    """

    transition: UserFunction | MarkovTransition | None
    """Regime transition function, or `None` for terminal regimes.

    A bare callable is deterministic. Wrap in `MarkovTransition` for stochastic
    regime transitions that return probability distributions.
    """

    active: ActiveFunction = lambda _age: True
    """Callable that takes age (float) and returns True if regime is active."""

    states: Mapping[str, Grid] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of state variable names to grid objects."""

    state_transitions: Mapping[
        str,
        UserFunction
        | MarkovTransition
        | None
        | Mapping[str, UserFunction | MarkovTransition],
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of state names to transition functions, `None`, or per-target dicts.

    Every non-shock state must have an entry — omitting a state raises an error.
    `None` marks a fixed state (identity auto-generated internally). Wrap in
    `MarkovTransition` for stochastic transitions. Per-target dicts map target
    regime names to transition functions — every reachable target must be listed.
    """

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

    @property
    def stochastic_regime_transition(self) -> bool:
        """Whether the regime transition is stochastic (MarkovTransition)."""
        return isinstance(self.transition, MarkovTransition)

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
        make_immutable("state_transitions")
        make_immutable("actions")
        make_immutable("constraints")

    def get_all_functions(self) -> MappingProxyType[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Collects functions from four sources:
        - `self.functions` (utility, helpers, H)
        - `self.constraints`
        - State transitions from `self.state_transitions`
        - The regime transition (`self.transition`, keyed as `"next_regime"`)

        Returns:
            Read-only mapping of all regime functions.

        """
        result = dict(self.functions) | dict(self.constraints)
        if not self.terminal:
            result |= _collect_state_transitions(self.states, self.state_transitions)
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

    # Validate state_transitions is a mapping
    if not isinstance(regime.state_transitions, Mapping):
        error_messages.append("state_transitions must be a mapping.")

    # Validate regime transition is callable, MarkovTransition, or None
    if regime.transition is not None and not (
        callable(regime.transition) or isinstance(regime.transition, MarkovTransition)
    ):
        error_messages.append(
            "transition must be a callable, MarkovTransition, or None, "
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
    error_messages.extend(_validate_state_transitions(regime))

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


def _validate_state_transitions(regime: Regime) -> list[str]:
    """Validate state_transitions against states."""
    error_messages: list[str] = []

    shock_names = {
        name for name, grid in regime.states.items() if isinstance(grid, _ShockGrid)
    }
    non_shock_names = set(regime.states) - shock_names

    # Keys not in states are allowed only with actual transitions (not None).
    # None means identity, which requires the state to exist in this regime.
    extra_keys = set(regime.state_transitions) - set(regime.states)
    for key in extra_keys:
        value = regime.state_transitions[key]
        if value is None:
            error_messages.append(
                f"state_transitions['{key}'] is None but '{key}' is not in states. "
                "Identity transitions require the state to exist in this regime.",
            )

    # ShockGrid names must NOT appear in state_transitions
    shock_in_transitions = shock_names & set(regime.state_transitions)
    if shock_in_transitions:
        error_messages.append(
            f"ShockGrid states have intrinsic transitions and must not appear "
            f"in state_transitions: {shock_in_transitions}.",
        )

    # Terminal regimes must have empty state_transitions
    if regime.terminal:
        if regime.state_transitions:
            error_messages.append(
                "Terminal regimes must have empty state_transitions.",
            )
        return error_messages

    # Every non-shock state must have an entry
    missing = non_shock_names - set(regime.state_transitions)
    if missing:
        error_messages.append(
            f"Every non-shock state must have an entry in state_transitions. "
            f"Missing: {missing}. Use None for fixed states.",
        )

    # Validate each value type
    for name, value in regime.state_transitions.items():
        if value is None or callable(value) or isinstance(value, MarkovTransition):
            continue
        if isinstance(value, Mapping):
            error_messages.extend(_validate_per_target_dict(name, value))
        else:
            error_messages.append(
                f"state_transitions['{name}'] must be callable, MarkovTransition, "
                f"None, or a per-target Mapping, got {type(value).__name__}.",
            )

    return error_messages


def _validate_per_target_dict(
    state_name: str, targets: Mapping[str, object]
) -> list[str]:
    """Validate a per-target transition dict for stochastic consistency and types."""
    error_messages: list[str] = []
    markov_count = 0
    for target_name, target_value in targets.items():
        if not isinstance(target_name, str):
            error_messages.append(
                f"state_transitions['{state_name}'] per-target dict key "
                f"{target_name!r} must be a string.",
            )
        if isinstance(target_value, MarkovTransition):
            markov_count += 1
        elif not callable(target_value):
            error_messages.append(
                f"state_transitions['{state_name}']['{target_name}'] must be "
                f"callable or MarkovTransition, got "
                f"{type(target_value).__name__}.",
            )
    # Check stochastic consistency
    if 0 < markov_count < len(targets):
        error_messages.append(
            f"state_transitions['{state_name}'] per-target dict must be "
            f"consistently stochastic: either all values are "
            f"MarkovTransition or none are.",
        )
    return error_messages


def _make_identity_fn(
    state_name: str, *, annotation: TypeAliasType
) -> _IdentityTransition:
    """Create an identity transition for a fixed state.

    Convenience wrapper around `_IdentityTransition`.

    """
    return _IdentityTransition(state_name, annotation=annotation)


def _add_raw_transition(
    transitions: dict[str, UserFunction],
    name: str,
    raw: UserFunction
    | MarkovTransition
    | Mapping[str, UserFunction | MarkovTransition],
) -> None:
    """Add a single raw transition entry to the transitions dict.

    Handles callables, MarkovTransition, and per-target dicts.

    """
    if isinstance(raw, MarkovTransition) or callable(raw):
        transitions[f"next_{name}"] = raw
    elif isinstance(raw, Mapping):
        for target_name, target_value in raw.items():
            key = f"next_{name}{QNAME_DELIMITER}{target_name}"
            transitions[key] = target_value


def _collect_state_transitions(
    states: Mapping[str, Grid],
    state_transitions: Mapping[
        str,
        UserFunction
        | MarkovTransition
        | None
        | Mapping[str, UserFunction | MarkovTransition],
    ],
) -> dict[str, UserFunction]:
    """Collect state transition functions from `state_transitions`.

    For each state, produces entries keyed as `f"next_{name}"`:
    - ShockGrid → stub `lambda: None`
    - `None` → auto-generated identity transition
    - Callable → used directly
    - `MarkovTransition` → used directly (callable via `__call__`)
    - Per-target dict → ALL variants with qualified names
      (e.g., `next_health__working`, `next_health__retired`)

    Target-only states (in `state_transitions` but not in `states`) are also
    collected. These have no grid in the source regime; `None` is rejected by
    validation, so only callables, MarkovTransition, and per-target dicts remain.

    """
    transitions: dict[str, UserFunction] = {}
    for name, grid in states.items():
        if isinstance(grid, _ShockGrid):
            transitions[f"next_{name}"] = lambda: None
            continue

        if name not in state_transitions:
            msg = (
                f"State '{name}' has no entry in state_transitions. "
                "Use None for fixed states."
            )
            raise RegimeInitializationError(msg)

        raw = state_transitions[name]
        if raw is None:
            ann = DiscreteState if isinstance(grid, DiscreteGrid) else ContinuousState
            transitions[f"next_{name}"] = _make_identity_fn(name, annotation=ann)
        else:
            _add_raw_transition(transitions, name, raw)

    # Second pass: target-only states (in state_transitions but not in states).
    for name, raw in state_transitions.items():
        if name not in states and raw is not None:
            _add_raw_transition(transitions, name, raw)

    return transitions
