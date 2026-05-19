"""State transition collection from user-facing `state_transitions` mappings."""

from collections.abc import Callable, Mapping
from typing import TypeAliasType

from dags.tree import QNAME_DELIMITER

from lcm._grids import DiscreteGrid, Grid
from lcm.api.regime import _IdentityTransition
from lcm.exceptions import RegimeInitializationError
from lcm.shocks._base import _ShockGrid
from lcm.typing import (
    ContinuousState,
    DiscreteState,
    RegimeName,
    StateName,
    TransitionFunctionName,
    UserFunction,
)


def collect_state_transitions(
    states: Mapping[StateName, Grid | None],
    state_transitions: Mapping[
        StateName,
        UserFunction | Callable | None | Mapping[RegimeName, UserFunction | Callable],
    ],
) -> dict[TransitionFunctionName, UserFunction]:
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
    transitions: dict[TransitionFunctionName, UserFunction] = {}
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
            transitions[f"next_{name}"] = _make_identity_fn(
                state_name=name, annotation=ann
            )
        else:
            _add_raw_transition(transitions=transitions, name=name, raw=raw)

    # Second pass: target-only states (in state_transitions but not in states).
    for name, raw in state_transitions.items():
        if name not in states and raw is not None:
            _add_raw_transition(transitions=transitions, name=name, raw=raw)

    return transitions


def _make_identity_fn(
    *, state_name: StateName, annotation: TypeAliasType
) -> _IdentityTransition:
    """Create an identity transition for a fixed state."""
    return _IdentityTransition(state_name, annotation=annotation)


def _add_raw_transition(
    *,
    transitions: dict[TransitionFunctionName, UserFunction],
    name: StateName,
    raw: UserFunction | Callable | Mapping[RegimeName, UserFunction | Callable],
) -> None:
    """Add a single raw transition entry to the transitions dict."""
    if callable(raw):
        transitions[f"next_{name}"] = raw
    elif isinstance(raw, Mapping):
        for target_name, target_value in raw.items():
            key = f"next_{name}{QNAME_DELIMITER}{target_name}"
            transitions[key] = target_value
