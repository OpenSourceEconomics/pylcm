"""Finalize user regimes at model build.

`finalize_regimes` turns each user `Regime` into the complete form the model
runs: model-level `derived_categoricals` are merged in, the default Bellman
aggregator `H` is injected for non-terminal regimes that supply none, and
completeness is validated (a `utility` entry, state-transition coverage, no
state/action overlap, distributed-grid rules). The result is a plain
`lcm.regime.Regime`, still in user vocabulary — coarse laws, `Phased`
containers, and per-target dicts survive untouched, so the params template
reads the user's coarseness off it.
"""

from collections.abc import Mapping
from types import MappingProxyType

from _lcm.grids import DiscreteGrid
from _lcm.typing import FunctionName, RegimeName
from _lcm.user_regime_validation import _validate_completeness
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.regime import _default_H

# A user `Regime` after model-build finalization. Runtime-equivalent to
# `lcm.regime.Regime`; internal signatures use this alias to mark values
# produced by `finalize_regimes` (model-level slots merged, default `H`
# injected, completeness validated).
type FinalizedUserRegime = UserRegime


def finalize_regimes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> MappingProxyType[RegimeName, FinalizedUserRegime]:
    """Finalize every user regime for the model build.

    Merges model-level `derived_categoricals` into each regime (a regime
    entry with identical categories is tolerated; conflicting categories
    raise), injects the default `H` into non-terminal regimes that supply
    none, and validates completeness.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.
        derived_categoricals: Model-level categorical grids to broadcast.

    Returns:
        Immutable mapping of regime names to finalized regimes.

    Raises:
        ModelInitializationError: If a regime has a `derived_categoricals`
            entry conflicting with a model-level one.
        RegimeInitializationError: If a regime is incomplete (e.g. missing
            `utility` or state-transition coverage), with the regime name
            prefixed.

    """
    result: dict[RegimeName, FinalizedUserRegime] = {}
    for regime_name, user_regime in user_regimes.items():
        merged = _merge_derived_categoricals(
            regime_name=regime_name,
            user_regime=user_regime,
            derived_categoricals=derived_categoricals,
        )
        functions = dict(user_regime.functions)
        # Terminal regimes don't need H since Q = U directly (no E_next_V).
        if user_regime.transition is not None and "H" not in functions:
            functions["H"] = _default_H
        finalized = user_regime.replace(
            functions=functions, derived_categoricals=merged
        )
        error_messages = _validate_completeness(finalized)
        if error_messages:
            raise RegimeInitializationError(
                f"In regime '{regime_name}': {format_messages(error_messages)}"
            )
        result[regime_name] = finalized
    return MappingProxyType(result)


def _merge_derived_categoricals(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> dict[FunctionName, DiscreteGrid]:
    """Merge model-level derived categoricals into one regime's mapping."""
    merged = dict(user_regime.derived_categoricals)
    for var, grid in derived_categoricals.items():
        existing = merged.get(var)
        if existing is not None and existing.categories != grid.categories:
            msg = (
                f"Model-level derived_categoricals['{var}'] conflicts "
                f"with regime '{regime_name}': {grid.categories} vs "
                f"{existing.categories}."
            )
            raise ModelInitializationError(msg)
        merged[var] = grid
    return merged
