"""Collect state transition functions from user-facing `state_transitions`.

`collect_state_transitions` walks a regime's `state_transitions` and returns
every state's transition *function* — a bare callable, a `MarkovTransition`
(callable via `__call__`), a grid-annotated identity for `fixed_transition`
entries, or the variants of a per-target dict.

The companion validation-metadata collector for the `MarkovTransition` entries
lives in `_lcm.regime_building.stochastic_state_transitions`; keeping it
separate lets this module stay free of any dependency on the user-facing
`Regime`.
"""

import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAliasType, cast

from dags.tree import QNAME_DELIMITER

from _lcm.engine import _StochasticStateTransition
from _lcm.grids import DiscreteGrid, Grid
from _lcm.identity_transition import _IdentityTransition
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import RegimeName, StateName, TransitionFunctionName
from _lcm.utils.ast_inspection import _get_func_indexing_params
from lcm.exceptions import (
    InvalidStateTransitionProbabilitiesError,
    RegimeInitializationError,
)
from lcm.phased import Phased
from lcm.transition import MarkovTransition
from lcm.typing import ContinuousState, DiscreteState, UserFunction

if TYPE_CHECKING:
    from lcm.regime import Regime as UserRegime


def collect_state_transitions(
    states: Mapping[StateName, Grid | Phased | None],
    state_transitions: Mapping[
        StateName,
        UserFunction
        | Callable
        | Phased
        | None
        | Mapping[RegimeName, UserFunction | Callable | Phased],
    ],
) -> dict[TransitionFunctionName, UserFunction | Phased]:
    """Collect state transition functions from `state_transitions`.

    For each state, produces entries keyed as `f"next_{name}"`:
    - continuous stochastic process -> skipped (process transitions are built
      directly in `_process_regime_core`)
    - `fixed_transition` entry -> rebuilt with the state's grid-matched
      annotation
    - Callable -> used directly
    - `MarkovTransition` -> used directly (callable via `__call__`)
    - Per-target dict -> ALL variants with qualified names
      (e.g., `next_health__working`, `next_health__retired`)

    Target-only states (in `state_transitions` but not in `states`) are also
    collected. These have no grid in the source regime; `fixed_transition` is
    rejected by validation there, so only callables, MarkovTransition, and
    per-target dicts remain.

    """
    transitions: dict[TransitionFunctionName, UserFunction | Phased] = {}
    for name, grid in states.items():
        # Process transitions built directly in _process_regime_core
        if isinstance(grid, _ContinuousStochasticProcess):
            continue

        if name not in state_transitions:
            msg = (
                f"State '{name}' has no entry in state_transitions. "
                "Use `fixed_transition(state_name)` for fixed states."
            )
            raise RegimeInitializationError(msg)

        raw = state_transitions[name]
        if isinstance(raw, _IdentityTransition):
            ann = DiscreteState if isinstance(grid, DiscreteGrid) else ContinuousState
            transitions[f"next_{name}"] = _make_identity_fn(
                state_name=name, annotation=ann
            )
        elif raw is not None:
            _add_raw_transition(transitions=transitions, name=name, raw=raw)

    # Second pass: target-only states (in state_transitions but not in states).
    for name, raw in state_transitions.items():
        if name not in states and raw is not None:
            _add_raw_transition(transitions=transitions, name=name, raw=raw)

    return transitions


def collect_stochastic_state_transitions(
    *,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> MappingProxyType[TransitionFunctionName, _StochasticStateTransition]:
    """Collect validation metadata for every `MarkovTransition` state transition.

    Walks `user_regime.state_transitions` and yields one entry per
    `MarkovTransition`. Per-target dict entries are flattened into
    `next_{state}__{target}` keys, mirroring the qname pattern used by
    `collect_state_transitions`. Returns an empty mapping for regimes with
    no stochastic state transitions (incl. terminal regimes).

    Args:
        user_regime: User-facing regime to inspect.
        user_regimes: All user regimes in the model. Needed to look up
            `n_outcomes` for target-only states whose `DiscreteGrid` lives
            on the target regime, not the source.

    Returns:
        Immutable mapping of qualified transition name to validation
        metadata.

    Raises:
        InvalidStateTransitionProbabilitiesError: If a `MarkovTransition`'s
            `probs_array` subscript order does not match the function's
            signature parameter order. Permissively skipped when the
            function does not use the `probs_array[...]` pattern.

    """
    entries: dict[TransitionFunctionName, _StochasticStateTransition] = {}

    for state_name, raw in user_regime.state_transitions.items():
        if isinstance(raw, MarkovTransition):
            _add_stochastic_entry(
                entries=entries,
                key=f"next_{state_name}",
                markov=raw,
                state_name=state_name,
                target_regime_name=None,
                user_regime=user_regime,
                user_regimes=user_regimes,
            )
        elif isinstance(raw, Mapping):
            for raw_target_name, target_value in raw.items():
                if not isinstance(target_value, MarkovTransition):
                    continue
                target_regime_name: RegimeName = cast("RegimeName", raw_target_name)
                _add_stochastic_entry(
                    entries=entries,
                    key=f"next_{state_name}__{target_regime_name}",
                    markov=target_value,
                    state_name=state_name,
                    target_regime_name=target_regime_name,
                    user_regime=user_regime,
                    user_regimes=user_regimes,
                )

    return MappingProxyType(entries)


def _make_identity_fn(
    *, state_name: StateName, annotation: TypeAliasType
) -> _IdentityTransition:
    """Create an identity transition for a fixed state."""
    return _IdentityTransition(state_name, annotation=annotation)


def _add_raw_transition(
    *,
    transitions: dict[TransitionFunctionName, UserFunction | Phased],
    name: StateName,
    raw: UserFunction
    | Callable
    | Phased
    | Mapping[RegimeName, UserFunction | Callable | Phased],
) -> None:
    """Add a single raw transition entry to the transitions dict.

    A `Phased` entry is registered as-is; consumers that need a single
    callable resolve it for their phase (`Regime.get_all_functions`), while
    the params-template collector unions both variants' parameters.
    """
    if callable(raw) or isinstance(raw, Phased):
        transitions[f"next_{name}"] = cast("UserFunction", raw)
    elif isinstance(raw, Mapping):
        for target_name, target_value in raw.items():
            key = f"next_{name}{QNAME_DELIMITER}{target_name}"
            transitions[key] = cast("UserFunction", target_value)


def _add_stochastic_entry(
    *,
    entries: dict[TransitionFunctionName, _StochasticStateTransition],
    key: TransitionFunctionName,
    markov: MarkovTransition,
    state_name: str,
    target_regime_name: RegimeName | None,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Static-check one MarkovTransition and append its metadata."""
    func = markov.func

    state_grid = _find_state_grid(
        state_name=state_name,
        target_regime_name=target_regime_name,
        user_regime=user_regime,
        user_regimes=user_regimes,
    )
    if not isinstance(state_grid, DiscreteGrid):
        # `MarkovTransition` on a continuous state is not a supported
        # pattern for the automatic validator. Static phase tolerates
        # the omission; the runtime phase skips it by absence from the
        # metadata. The subscript-order check is skipped too — it applies
        # only to the discrete `probs_array[...]` pattern this validator
        # covers.
        return

    indexing_params = tuple(
        _get_func_indexing_params(func=func, array_param_name="probs_array")
    )
    _check_subscript_order(
        func=func, indexing_params=indexing_params, state_name=state_name
    )
    n_outcomes = len(state_grid.categories)

    entries[key] = _StochasticStateTransition(
        func=func,
        state_name=state_name,
        target_regime_name=target_regime_name,
        n_outcomes=n_outcomes,
        indexing_params=indexing_params,
    )


def _find_state_grid(
    *,
    state_name: str,
    target_regime_name: RegimeName | None,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> object:
    """Look up the state's grid for outcome-axis sizing.

    For a per-target dict entry the **target** regime's grid is authoritative:
    the `MarkovTransition` returns a distribution over the target's state
    space, which may differ in size from the source's (cross-grid
    transitions). The source grid is never substituted in that case — if the
    target regime does not declare the state, `None` is returned so the
    caller skips metadata creation rather than sizing off a wrong grid.

    A plain `MarkovTransition` (no per-target dict) sizes off the source
    regime's grid.

    Returns `None` when no authoritative grid is found.
    """
    if target_regime_name is not None:
        target = user_regimes.get(target_regime_name)
        if target is not None and state_name in target.states:
            return target.states[state_name]
        return None
    if state_name in user_regime.states:
        return user_regime.states[state_name]
    return None


def _check_subscript_order(
    *,
    func: object,
    indexing_params: tuple[str, ...],
    state_name: str,
) -> None:
    """Raise if `probs_array[…]` subscripts don't match signature order.

    Permissive: when the function doesn't use the `probs_array[...]`
    pattern (`indexing_params` is empty), the check silently skips. The
    runtime numerical checks still cover such functions.
    """
    if not indexing_params:
        return
    sig = inspect.signature(func)  # ty: ignore[invalid-argument-type]
    sig_order = tuple(
        p for p in sig.parameters if p != "probs_array" and p in indexing_params
    )
    if indexing_params != sig_order:
        func_name = getattr(func, "__name__", "<unknown>")
        msg = (
            f"In MarkovTransition for state '{state_name}', function "
            f"'{func_name}' indexes `probs_array` as "
            f"`probs_array[{', '.join(indexing_params)}]` but the signature "
            f"order is `probs_array[{', '.join(sig_order)}]`. Swap the "
            f"subscript order or the signature so they match."
        )
        raise InvalidStateTransitionProbabilitiesError(msg)
