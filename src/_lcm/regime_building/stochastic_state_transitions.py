"""Collect validation metadata for `MarkovTransition` state transitions.

`collect_stochastic_state_transitions` walks a regime's `state_transitions`
and yields one `_StochasticStateTransition` entry per `MarkovTransition`: a
process-time AST subscript-order check plus `n_outcomes` derivation, cached on
the canonical `Regime` and consumed by the pre-solve state-transition
validator.

The deterministic counterpart — collecting the transition *functions* — lives
in `_lcm.regime_building.transitions` and carries no dependency on the
user-facing `Regime`.
"""

import inspect
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, cast

from _lcm.engine import _StochasticStateTransition
from _lcm.grids import DiscreteGrid
from _lcm.typing import RegimeName, TransitionFunctionName
from _lcm.utils.ast_inspection import _get_func_indexing_params
from lcm.exceptions import InvalidStateTransitionProbabilitiesError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import MarkovTransition


def collect_stochastic_state_transitions(
    *,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> MappingProxyType[TransitionFunctionName, _StochasticStateTransition]:
    """Collect validation metadata for every `MarkovTransition` state transition.

    Walks `user_regime.state_transitions` and yields one entry per
    `MarkovTransition`. Per-target dict entries are flattened into
    `next_{state}__{target}` keys, mirroring the qname pattern used by
    `collect_state_transitions`; each variant of a `Phased` law is further
    suffixed with `@{phase}`. Returns an empty mapping for regimes with
    no stochastic state transitions (incl. terminal regimes).

    Known limitation (pre-dates phase variance, and applies to bare laws just as
    much): the runtime numerical checks call the law with arguments drawn from
    grids and params, so a law that reads a named DAG *helper* cannot be invoked
    and is skipped with a warning by `validate_state_transitions_all_periods`.
    Closing that needs the validator to evaluate the compiled sub-DAG rather than
    the raw function.

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

    for state_name, entry in user_regime.state_transitions.items():
        # A `Phased` entry holds two laws for one state. Each is collected under its
        # OWN key: they are different functions, and a malformed perceived law is as
        # fatal as a malformed true one. Sharing one key would keep only whichever was
        # inserted last, so the belief law would reach the solver unchecked. Walking the
        # raw entry (ignoring `Phased`) would leave it with no metadata at all.
        for raw, phase in _phase_variants(entry):
            if isinstance(raw, MarkovTransition):
                _add_stochastic_entry(
                    entries=entries,
                    key=_phase_key(f"next_{state_name}", phase),
                    markov=raw,
                    state_name=state_name,
                    target_regime_name=None,
                    phase=phase,
                    user_regime=user_regime,
                    user_regimes=user_regimes,
                )
            elif isinstance(raw, Mapping):
                for raw_target_regime_name, law in raw.items():
                    if not isinstance(law, MarkovTransition):
                        continue
                    target_regime_name: RegimeName = cast(
                        "RegimeName", raw_target_regime_name
                    )
                    _add_stochastic_entry(
                        entries=entries,
                        key=_phase_key(
                            f"next_{state_name}__{target_regime_name}", phase
                        ),
                        markov=law,
                        state_name=state_name,
                        target_regime_name=target_regime_name,
                        phase=phase,
                        user_regime=user_regime,
                        user_regimes=user_regimes,
                    )

    return MappingProxyType(entries)


def _phase_key(
    base: str, phase: Literal["solve", "simulate"] | None
) -> TransitionFunctionName:
    """Key for one law's metadata, disambiguated by phase for `Phased` entries.

    A bare law keeps the plain `next_<state>` key, so nothing about phase-invariant
    models changes. The keys are internal to validation — no consumer resolves a
    transition function through this mapping.
    """
    return base if phase is None else f"{base}@{phase}"


def _phase_variants(
    entry: object,
) -> tuple[tuple[object, Literal["solve", "simulate"] | None], ...]:
    """The laws carried by one `state_transitions` entry, tagged by phase.

    A bare entry is phase-invariant and yields itself with no phase; a `Phased` entry
    yields its solve and simulate variants, so each is validated on its own.
    """
    if isinstance(entry, Phased):
        return ((entry.solve, "solve"), (entry.simulate, "simulate"))
    return ((entry, None),)


def _add_stochastic_entry(
    *,
    entries: dict[TransitionFunctionName, _StochasticStateTransition],
    key: TransitionFunctionName,
    markov: MarkovTransition,
    state_name: str,
    target_regime_name: RegimeName | None,
    phase: Literal["solve", "simulate"] | None,
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
        phase=phase,
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
