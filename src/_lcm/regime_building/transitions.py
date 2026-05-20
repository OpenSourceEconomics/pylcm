"""Collect state transition functions from user-facing `state_transitions`.

`collect_state_transitions` walks a regime's `state_transitions` and returns
every state's transition *function* — a bare callable, a `MarkovTransition`
(callable via `__call__`), an auto-generated identity for `None`, or the
variants of a per-target dict.

The companion validation-metadata collector for the `MarkovTransition` entries
lives in `_lcm.regime_building.stochastic_state_transitions`; keeping it
separate lets this module stay free of any dependency on the user-facing
`Regime`.
"""

import inspect
from collections.abc import Callable, Mapping
from typing import TypeAliasType, overload

from dags.tree import QNAME_DELIMITER

from _lcm.grids import DiscreteGrid, Grid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import RegimeName, StateName, TransitionFunctionName
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousState, DiscreteState, UserFunction


class _IdentityTransition:
    """Identity transition function for fixed states.

    Used by `get_all_functions()` so the params template includes fixed states.
    The `_is_auto_identity` attribute lets validation distinguish auto-generated
    identities from user-provided transitions.

    """

    _is_auto_identity: bool = True

    def __init__(self, state_name: StateName, *, annotation: TypeAliasType) -> None:
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


def collect_state_transitions(
    states: Mapping[StateName, Grid | None],
    state_transitions: Mapping[
        StateName,
        UserFunction | Callable | None | Mapping[RegimeName, UserFunction | Callable],
    ],
) -> dict[TransitionFunctionName, UserFunction]:
    """Collect state transition functions from `state_transitions`.

    For each state, produces entries keyed as `f"next_{name}"`:
    - continuous stochastic process -> skipped (process transitions are built
      directly in `_process_regime_core`)
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
        # Process transitions built directly in _process_regime_core
        if isinstance(grid, _ContinuousStochasticProcess):
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
