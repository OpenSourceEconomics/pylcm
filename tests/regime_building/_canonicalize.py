"""The two canonicalization steps as one call, for the canonicalization tests.

The model path (`process_regimes`) calls `normalize_all_regime_phases` and
`canonicalize_phased_regimes` itself, so that model-level age normalization can sit
between the two steps.
"""

from collections.abc import Mapping
from types import MappingProxyType

from _lcm.regime_building.canonicalize import canonicalize_phased_regimes
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.phases import PhasedRegimeSpec, normalize_all_regime_phases
from _lcm.regime_law import RegimeLaws
from lcm.typing import RegimeName


def canonicalize_regimes(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    laws: RegimeLaws,
) -> MappingProxyType[RegimeName, PhasedRegimeSpec]:
    """Split every finalized regime into phases and canonicalize its laws.

    Normalizes every regime's `Phased` slots and then canonicalizes the resulting
    specs, with no model-level age normalization between the two steps.

    Args:
        user_regimes: Mapping of regime names to finalized regimes.
        laws: Each regime's law, bound from `Model(edges=...)`, by regime name.

    Returns:
        Immutable mapping of regime names to per-phase specs whose every
        `state_transitions` value and every non-terminal regime transition
        is a canonical per-target mapping.

    Raises:
        ModelInitializationError: If a per-target regime transition or state
            law names an unknown regime, or a state law does not cover every
            retained temporal target carrying the state.

    """
    raw_specs = normalize_all_regime_phases(user_regimes=user_regimes, laws=laws)
    return canonicalize_phased_regimes(
        raw_specs=raw_specs,
        all_regime_names=frozenset(user_regimes),
    )
