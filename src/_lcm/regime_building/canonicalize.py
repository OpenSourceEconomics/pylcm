"""The model-level canonicalization stage.

`canonicalize_regimes` rewrites every regime's phase slices into the canonical
target-granular form: each `state_transitions` value becomes a
`Mapping[RegimeName, law]` over exactly the reachable target regimes that
carry the state in that phase:

- a bare law broadcasts over the reachable carriers
- a user per-target dict passes through restricted to its named targets
- a `fixed_transition` entry desugars into per-target identity laws with the
  source grid's dtype annotation

Reachability is resolved here, once per phase, from the per-target dicts'
named targets plus every target whose state needs are fully covered by bare
laws (all regimes when no per-target dict exists). Everything downstream —
function compilation, the per-target transition bundles — reads the canonical
mapping and never re-infers reachability.
"""

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from _lcm.grids import DiscreteGrid, Grid
from _lcm.identity_transition import _IdentityTransition
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building.effective import EffectiveUserRegime
from _lcm.regime_building.phases import (
    PhasedRegimeSpec,
    RegimePhaseSpec,
    normalize_regime_phases,
)
from _lcm.typing import RegimeName, StateName
from lcm.transition import MarkovTransition
from lcm.typing import ContinuousState, DiscreteState, UserFunction

type _CanonicalLaw = UserFunction | MarkovTransition
type _CanonicalStateTransitions = MappingProxyType[
    StateName, MappingProxyType[RegimeName, _CanonicalLaw]
]


def canonicalize_regimes(
    *,
    user_regimes: Mapping[RegimeName, EffectiveUserRegime],
) -> MappingProxyType[RegimeName, PhasedRegimeSpec]:
    """Split every effective regime into phases and canonicalize its laws.

    Args:
        user_regimes: Mapping of regime names to effective regimes.

    Returns:
        Immutable mapping of regime names to per-phase specs whose every
        `state_transitions` value is a canonical per-target mapping.

    """
    raw_specs = {
        regime_name: normalize_regime_phases(user_regime)
        for regime_name, user_regime in user_regimes.items()
    }

    canonical_specs: dict[RegimeName, PhasedRegimeSpec] = {}
    for phase_name in ("solution", "simulation"):
        states_per_regime = {
            regime_name: frozenset(getattr(spec, phase_name).grid_states)
            for regime_name, spec in raw_specs.items()
        }
        for regime_name, spec in raw_specs.items():
            phase_slice: RegimePhaseSpec = getattr(spec, phase_name)
            canonical_slice = dataclasses.replace(
                phase_slice,
                state_transitions=cast(
                    "MappingProxyType",
                    _canonicalize_phase_transitions(
                        phase_slice=phase_slice,
                        states_per_regime=states_per_regime,
                    ),
                ),
            )
            base = canonical_specs.get(regime_name, spec)
            canonical_specs[regime_name] = dataclasses.replace(
                base, **{phase_name: canonical_slice}
            )

    return MappingProxyType(canonical_specs)


def _canonicalize_phase_transitions(
    *,
    phase_slice: RegimePhaseSpec,
    states_per_regime: Mapping[RegimeName, frozenset[StateName]],
) -> _CanonicalStateTransitions:
    """Expand one phase slice's laws into the canonical per-target form."""
    if phase_slice.regime_transition is None:
        return MappingProxyType({})

    bare_laws: dict[StateName, _CanonicalLaw] = {}
    per_target_laws: dict[StateName, Mapping[RegimeName, _CanonicalLaw]] = {}
    for state_name, raw in phase_slice.state_transitions.items():
        if isinstance(
            phase_slice.grid_states.get(state_name), _ContinuousStochasticProcess
        ):
            # Process transitions are generated from the grid's intrinsic
            # transition logic, not from `state_transitions`.
            continue
        if isinstance(raw, Mapping):
            named = cast("Mapping[RegimeName, _CanonicalLaw]", raw)
            per_target_laws[state_name] = {
                target_name: _desugar_identity(
                    law=cell,
                    state_name=state_name,
                    grid=phase_slice.grid_states.get(state_name),
                )
                for target_name, cell in named.items()
            }
        else:
            bare_laws[state_name] = _desugar_identity(
                law=cast("_CanonicalLaw", raw),
                state_name=state_name,
                grid=phase_slice.grid_states.get(state_name),
            )

    reachable_targets = _get_reachable_targets(
        bare_laws=bare_laws,
        per_target_laws=per_target_laws,
        states_per_regime=states_per_regime,
    )

    canonical: dict[StateName, MappingProxyType[RegimeName, _CanonicalLaw]] = {}
    for state_name, law in bare_laws.items():
        carriers = {
            target_name
            for target_name in reachable_targets
            if state_name in states_per_regime[target_name]
        }
        if carriers:
            canonical[state_name] = MappingProxyType(
                dict.fromkeys(sorted(carriers), law)
            )
    for state_name, named in per_target_laws.items():
        cells = {
            target_name: law
            for target_name, law in named.items()
            if target_name in reachable_targets
            and state_name in states_per_regime.get(target_name, frozenset())
        }
        if cells:
            canonical[state_name] = MappingProxyType(cells)

    return MappingProxyType(canonical)


def _get_reachable_targets(
    *,
    bare_laws: Mapping[StateName, _CanonicalLaw],
    per_target_laws: Mapping[StateName, Mapping[RegimeName, _CanonicalLaw]],
    states_per_regime: Mapping[RegimeName, frozenset[StateName]],
) -> set[RegimeName]:
    """Infer which target regimes need transition bundles.

    When per-target dicts exist, start from their explicitly named targets and
    add any target whose state needs are fully covered by bare laws. Without
    per-target dicts, all regimes are reachable.

    """
    if not per_target_laws:
        return set(states_per_regime)

    targets: set[RegimeName] = set()
    for named in per_target_laws.values():
        targets |= named.keys()
    for target_name, target_states in states_per_regime.items():
        if (
            target_name not in targets
            and target_states
            and target_states <= bare_laws.keys()
        ):
            targets.add(target_name)
    return targets


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
