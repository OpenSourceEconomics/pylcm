"""Process-time static checks on stochastic state transitions.

Runs during `process_regimes` for every `MarkovTransition` state transition
(and per target of a per-target dict):

- **AST subscript-order check.** Parse the user function's source and confirm
  that `probs_array[...]` subscript order matches the signature parameter
  order. Permissive: if the function doesn't use a recognisable
  `probs_array[...]` pattern, the AST-side check silently skips and the
  runtime numerical checks still cover the function.
- **`n_outcomes` derivation.** Read the outcome-axis size off the state's
  `DiscreteGrid`. Cached on the canonical `Regime` so the runtime validator
  does not need to look it up per call.

The output is consumed by the pre-solve state-transition validator. Full
output-shape derivation is deferred to that runtime check, because it
depends on which of the function's indexing parameters resolve to grids in
the regime (states / actions) versus to scalar params (resolved at solve
time from `flat_params`).
The same function may be reused across regimes with different grid/param
splits.
"""

import inspect
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from lcm._grids import DiscreteGrid
from lcm.api.regime import MarkovTransition
from lcm.api.regime import Regime as UserRegime
from lcm.engine import _StochasticStateTransition
from lcm.exceptions import InvalidStateTransitionProbabilitiesError
from lcm.typing import RegimeName, TransitionFunctionName
from lcm.utils.ast_inspection import _get_func_indexing_params


def derive_stochastic_state_transitions(
    *,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> MappingProxyType[TransitionFunctionName, _StochasticStateTransition]:
    """Derive validation metadata for every `MarkovTransition` state transition.

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
            _add_entry(
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
                _add_entry(
                    entries=entries,
                    key=f"next_{state_name}__{target_regime_name}",
                    markov=target_value,
                    state_name=state_name,
                    target_regime_name=target_regime_name,
                    user_regime=user_regime,
                    user_regimes=user_regimes,
                )

    return MappingProxyType(entries)


def _add_entry(
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
