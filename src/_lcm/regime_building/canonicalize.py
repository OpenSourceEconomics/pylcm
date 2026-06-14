"""The model-level canonicalization stage.

`canonicalize_regimes` rewrites every regime's phase slices into the canonical
target-granular form: each `state_transitions` value becomes a
`Mapping[RegimeName, law]` over exactly the reachable target regimes that
carry the state in that phase:

- a bare law broadcasts over the reachable carriers
- a user per-target dict passes through (it must cover every reachable
  carrier and may not name anything else)
- a `fixed_transition` entry desugars into per-target identity laws with the
  source grid's dtype annotation

The regime transition itself is canonicalized into the same per-target form:

- a user per-target dict passes through
- a coarse callable / `MarkovTransition` maps every regime to one shared
  `_CoarseTransitionCell`, so the engine evaluates the underlying once and
  indexes per target
- `None` (terminal) stays `None`

Reachability has a single source of truth — the regime transition:

- per-target dict ⇒ its key set
- coarse callable / `MarkovTransition` ⇒ all regimes
- `None` (terminal) ⇒ empty

Everything downstream — function compilation, the per-target transition
bundles — reads the canonical mapping and never re-infers reachability.
"""

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from _lcm.coarse_transition import _CoarseTransitionCell
from _lcm.grids import DiscreteGrid, Grid
from _lcm.identity_transition import _IdentityTransition
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.phases import (
    PhasedRegimeSpec,
    RegimePhaseSpec,
    _PhaseRegimeTransition,
    normalize_regime_phases,
)
from _lcm.typing import RegimeName, StateName
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import ModelInitializationError
from lcm.transition import MarkovTransition
from lcm.typing import ContinuousState, DiscreteState, UserFunction

type _CanonicalLaw = UserFunction | MarkovTransition
type _CanonicalStateTransitions = MappingProxyType[
    StateName, MappingProxyType[RegimeName, _CanonicalLaw]
]
type _CanonicalRegimeTransition = (
    MappingProxyType[RegimeName, MarkovTransition | _CoarseTransitionCell] | None
)


def canonicalize_regimes(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
) -> MappingProxyType[RegimeName, PhasedRegimeSpec]:
    """Split every finalized regime into phases and canonicalize its laws.

    Args:
        user_regimes: Mapping of regime names to finalized regimes.

    Returns:
        Immutable mapping of regime names to per-phase specs whose every
        `state_transitions` value and every non-terminal regime transition
        is a canonical per-target mapping.

    Raises:
        ModelInitializationError: If a per-target regime transition or state
            law names an unknown regime, or a state law does not cover every
            reachable target carrying the state.

    """
    raw_specs = {
        regime_name: normalize_regime_phases(user_regime)
        for regime_name, user_regime in user_regimes.items()
    }
    all_regime_names = frozenset(user_regimes)
    errors: list[str] = []

    canonical_specs: dict[RegimeName, PhasedRegimeSpec] = {}
    for phase_name in ("solution", "simulation"):
        states_per_regime = {
            regime_name: frozenset(getattr(spec, phase_name).grid_states)
            for regime_name, spec in raw_specs.items()
        }
        for regime_name, spec in raw_specs.items():
            phase_slice: RegimePhaseSpec = getattr(spec, phase_name)
            canonical_transitions, slice_errors = _canonicalize_phase_transitions(
                phase_slice=phase_slice,
                states_per_regime=states_per_regime,
                all_regime_names=all_regime_names,
                source_label=f"regime '{regime_name}' ({phase_name})",
            )
            errors += slice_errors
            canonical_slice = dataclasses.replace(
                phase_slice,
                state_transitions=cast("MappingProxyType", canonical_transitions),
                regime_transition=_canonicalize_regime_transition(
                    regime_transition=phase_slice.regime_transition,
                    all_regime_names=all_regime_names,
                ),
            )
            base = canonical_specs.get(regime_name, spec)
            canonical_specs[regime_name] = dataclasses.replace(
                base, **{phase_name: canonical_slice}
            )

    if errors:
        raise ModelInitializationError(format_messages(sorted(set(errors))))

    return MappingProxyType(canonical_specs)


def _canonicalize_phase_transitions(
    *,
    phase_slice: RegimePhaseSpec,
    states_per_regime: Mapping[RegimeName, frozenset[StateName]],
    all_regime_names: frozenset[RegimeName] | None = None,
    source_label: str = "",
) -> tuple[_CanonicalStateTransitions, list[str]]:
    """Expand one phase slice's laws into the canonical per-target form.

    Returns the canonical mapping and the violations found along the way.
    """
    if all_regime_names is None:
        all_regime_names = frozenset(states_per_regime)
    if phase_slice.regime_transition is None:
        return MappingProxyType({}), []

    bare_laws, per_target_laws = _split_laws(phase_slice)

    reachable_targets, errors = _declared_reachable_targets(
        regime_transition=phase_slice.regime_transition,
        all_regime_names=all_regime_names,
        source_label=source_label,
    )

    canonical: dict[StateName, MappingProxyType[RegimeName, _CanonicalLaw]] = {}
    for state_name, law in bare_laws.items():
        carriers = {
            target_regime_name
            for target_regime_name in reachable_targets
            if state_name in states_per_regime.get(target_regime_name, frozenset())
        }
        if carriers:
            canonical[state_name] = MappingProxyType(
                dict.fromkeys(sorted(carriers), law)
            )
    for state_name, named in per_target_laws.items():
        required = {
            target_regime_name
            for target_regime_name in reachable_targets
            if state_name in states_per_regime.get(target_regime_name, frozenset())
        }
        errors += _per_target_law_errors(
            state_name=state_name,
            named_targets=frozenset(named),
            required=frozenset(required),
            reachable_targets=reachable_targets,
            all_regime_names=all_regime_names,
            source_label=source_label,
        )
        cells = {
            target_regime_name: law
            for target_regime_name, law in named.items()
            if target_regime_name in required
        }
        if cells:
            canonical[state_name] = MappingProxyType(cells)

    return MappingProxyType(canonical), errors


def _canonicalize_regime_transition(
    *,
    regime_transition: _PhaseRegimeTransition,
    all_regime_names: frozenset[RegimeName],
) -> _CanonicalRegimeTransition:
    """Rewrite one phase's regime transition into the per-target form.

    - a user per-target dict passes through
    - a coarse callable / `MarkovTransition` maps every regime to one shared
      `_CoarseTransitionCell`, so the engine evaluates the underlying once
      and indexes per target instead of re-evaluating per cell
    - `None` (terminal) stays `None`
    """
    if regime_transition is None:
        return None
    if isinstance(regime_transition, Mapping):
        cells = cast(
            "Mapping[RegimeName, MarkovTransition | _CoarseTransitionCell]",
            regime_transition,
        )
        return MappingProxyType(dict(cells.items()))
    cell = _CoarseTransitionCell(underlying=regime_transition)
    return MappingProxyType(dict.fromkeys(sorted(all_regime_names), cell))


def _split_laws(
    phase_slice: RegimePhaseSpec,
) -> tuple[
    dict[StateName, _CanonicalLaw],
    dict[StateName, Mapping[RegimeName, _CanonicalLaw]],
]:
    """Split a slice's laws into bare and per-target groups, identities desugared.

    Stochastic-process states are skipped: their transitions are generated
    from the grid's intrinsic transition logic, not from `state_transitions`.
    """
    bare_laws: dict[StateName, _CanonicalLaw] = {}
    per_target_laws: dict[StateName, Mapping[RegimeName, _CanonicalLaw]] = {}
    for state_name, raw in phase_slice.state_transitions.items():
        if isinstance(
            phase_slice.grid_states.get(state_name), _ContinuousStochasticProcess
        ):
            continue
        if isinstance(raw, Mapping):
            named = cast("Mapping[RegimeName, _CanonicalLaw]", raw)
            per_target_laws[state_name] = {
                target_regime_name: _desugar_identity(
                    law=law,
                    state_name=state_name,
                    grid=phase_slice.grid_states.get(state_name),
                )
                for target_regime_name, law in named.items()
            }
        else:
            bare_laws[state_name] = _desugar_identity(
                law=cast("_CanonicalLaw", raw),
                state_name=state_name,
                grid=phase_slice.grid_states.get(state_name),
            )
    return bare_laws, per_target_laws


def _per_target_law_errors(
    *,
    state_name: StateName,
    named_targets: frozenset[RegimeName],
    required: frozenset[RegimeName],
    reachable_targets: frozenset[RegimeName],
    all_regime_names: frozenset[RegimeName],
    source_label: str,
) -> list[str]:
    """Check a per-target state law's key set against declared reachability."""
    error_messages: list[str] = []
    missing = required - named_targets
    if missing:
        error_messages.append(
            f"{source_label}: state_transitions['{state_name}'] does not "
            f"cover reachable target(s) {sorted(missing)}. Provide a law "
            f"for each, or narrow reachability by declaring per-target "
            f"regime transitions (`transition={{target: "
            f"MarkovTransition(...)}}`).",
        )
    unknown = named_targets - all_regime_names
    if unknown:
        error_messages.append(
            f"{source_label}: state_transitions['{state_name}'] names "
            f"unknown regime(s) {sorted(unknown)}.",
        )
    unreachable = (named_targets & all_regime_names) - reachable_targets
    if unreachable:
        error_messages.append(
            f"{source_label}: state_transitions['{state_name}'] names "
            f"target(s) {sorted(unreachable)} that the regime transition "
            f"declares unreachable.",
        )
    return error_messages


def _declared_reachable_targets(
    *,
    regime_transition: object,
    all_regime_names: frozenset[RegimeName],
    source_label: str,
) -> tuple[frozenset[RegimeName], list[str]]:
    """Read the reachable target set off the regime transition.

    - per-target dict ⇒ its key set (unknown regime names are errors)
    - coarse callable / `MarkovTransition` ⇒ all regimes

    Returns the reachable targets and the violations found along the way.
    """
    if isinstance(regime_transition, Mapping):
        declared = frozenset(cast("Mapping[RegimeName, object]", regime_transition))
        unknown = declared - all_regime_names
        errors = (
            [
                f"{source_label}: the per-target regime transition names "
                f"unknown regime(s) {sorted(unknown)}."
            ]
            if unknown
            else []
        )
        return declared & all_regime_names, errors
    return all_regime_names, []


def _desugar_identity(
    *,
    law: _CanonicalLaw,
    state_name: StateName,
    grid: Grid | None,
) -> _CanonicalLaw:
    """Rebuild a `fixed_transition` identity with the source grid's annotation."""
    if not isinstance(law, _IdentityTransition):
        return law
    annotation = DiscreteState if isinstance(grid, DiscreteGrid) else ContinuousState
    return _IdentityTransition(state_name, annotation=annotation)
