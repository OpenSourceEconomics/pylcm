"""Regime input validation and state transition collection.

Called from `Regime.__post_init__` (validation) and `Regime.get_all_functions`
(state transition collection) to keep `regime.py` focused on the class definition.

"""

from collections.abc import Callable, Mapping
from typing import TypeAliasType

from dags.tree import QNAME_DELIMITER

from lcm.exceptions import RegimeInitializationError, format_messages
from lcm.grids import DiscreteGrid, Grid
from lcm.interfaces import SolveSimulateFunctionPair
from lcm.regime import MarkovTransition, Regime, _IdentityTransition
from lcm.shocks._base import _ShockGrid
from lcm.typing import (
    ActiveFunction,
    ContinuousState,
    DiscreteState,
    StateName,
    UserFunction,
)


def validate_attribute_types(regime: Regime) -> None:  # noqa: C901, PLR0912
    """Validate the types of the regime attributes."""
    error_messages = []

    # Validate types of states and actions
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
                if not callable(v) and not isinstance(v, SolveSimulateFunctionPair):
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

    # Validate regime transition is callable or None
    if not regime.terminal and not callable(regime.transition):
        error_messages.append(
            "transition must be callable or None, "
            f"but is {type(regime.transition).__name__}."
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def validate_logical_consistency(regime: Regime) -> None:
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

    # Validate function names do not start with "next_" (reserved for
    # auto-generated state transition functions)
    next_prefixed = [name for name in all_function_names if name.startswith("next_")]
    if next_prefixed:
        error_messages.append(
            f"Function names must not start with 'next_' — this prefix is "
            f"reserved for auto-generated state transition functions. "
            f"Invalid names: {next_prefixed}.",
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


def collect_state_transitions(
    states: Mapping[str, Grid],
    state_transitions: Mapping[
        str,
        UserFunction | Callable | None | Mapping[str, UserFunction | Callable],
    ],
) -> dict[str, UserFunction]:
    """Collect state transition functions from `state_transitions`.

    For each state, produces entries keyed as `f"next_{name}"`:
    - ShockGrid -> stub `lambda: None`
    - `None` -> auto-generated identity transition
    - Callable -> used directly
    - `MarkovTransition` -> used directly (callable via `__call__`)
    - Per-target dict -> ALL variants with qualified names
      (e.g., `next_health__working`, `next_health__retired`)

    Target-only states (in `state_transitions` but not in `states`) are also
    collected. These have no grid in the source regime; `None` is rejected by
    validation, so only callables, MarkovTransition, and per-target dicts remain.

    """
    transitions: dict[str, UserFunction] = {}
    for name, grid in states.items():
        # Shock transitions built directly in _process_regime_core
        if isinstance(grid, _ShockGrid):
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
        if value is None or callable(value):
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
    state_name: StateName, targets: Mapping[str, object]
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
    state_name: StateName, *, annotation: TypeAliasType
) -> _IdentityTransition:
    """Create an identity transition for a fixed state."""
    return _IdentityTransition(state_name, annotation=annotation)


def _add_raw_transition(
    transitions: dict[str, UserFunction],
    name: str,
    raw: UserFunction | Callable | Mapping[str, UserFunction | Callable],
) -> None:
    """Add a single raw transition entry to the transitions dict."""
    if callable(raw):
        transitions[f"next_{name}"] = raw
    elif isinstance(raw, Mapping):
        for target_name, target_value in raw.items():
            key = f"next_{name}{QNAME_DELIMITER}{target_name}"
            transitions[key] = target_value
