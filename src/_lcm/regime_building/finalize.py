"""Finalize user regimes at model build.

`finalize_regimes` turns each user `Regime` into the complete form the model
runs: model-level `derived_categoricals` are merged in, the default Bellman
aggregator `H` and the model-level certainty equivalent are injected into
non-terminal regimes that supply none, and completeness is validated (a
`utility` entry, state-transition coverage, no state/action overlap,
distributed-grid rules). The result is a plain
`lcm.regime.Regime`, still in user vocabulary — coarse laws, `Phased`
containers, and per-target dicts survive untouched, so the params template
reads the user's coarseness off it.
"""

from collections.abc import Mapping
from types import MappingProxyType

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.grids import DiscreteGrid
from _lcm.typing import FunctionName, RegimeName
from _lcm.user_regime_validation import _validate_completeness
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.temporal_aggregation import H_linear

# A user `Regime` after model-build finalization. Runtime-equivalent to
# `lcm.regime.Regime`; internal signatures use this alias to mark values
# produced by `finalize_regimes` (model-level slots merged, default `H`
# injected, completeness validated).
type FinalizedUserRegime = UserRegime


def finalize_regimes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
    certainty_equivalent: CertaintyEquivalent,
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
        certainty_equivalent: Model-level continuation aggregation, given to
            every non-terminal regime that declares none of its own.

    Returns:
        Immutable mapping of regime names to finalized regimes.

    Raises:
        ModelInitializationError: If a regime has a `derived_categoricals`
            entry conflicting with a model-level one.
        RegimeInitializationError: If a regime is incomplete (e.g. missing
            `utility` or state-transition coverage), with the regime name
            prefixed.

    """
    _fail_if_certainty_equivalent_is_mixed(user_regimes)
    result: dict[RegimeName, FinalizedUserRegime] = {}
    for regime_name, user_regime in user_regimes.items():
        merged = _merge_derived_categoricals(
            regime_name=regime_name,
            user_regime=user_regime,
            derived_categoricals=derived_categoricals,
        )
        functions = dict(user_regime.functions)
        # Terminal regimes have no continuation, so they need neither an
        # aggregator (Q = U directly) nor a certainty equivalent. Their slot is
        # carried through untouched so that a declared one still reaches the
        # completeness check below rather than being silently discarded.
        if user_regime.terminal:
            regime_certainty_equivalent = user_regime.certainty_equivalent
        else:
            if "H" not in functions:
                functions["H"] = H_linear
            regime_certainty_equivalent = (
                user_regime.certainty_equivalent
                if user_regime.certainty_equivalent is not None
                else certainty_equivalent
            )
        finalized = user_regime.replace(
            functions=functions,
            derived_categoricals=merged,
            certainty_equivalent=regime_certainty_equivalent,
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
    """Merge model-level derived categoricals into one regime's mapping.

    Follows the exactly-one-level rule of the other model-level regime
    slots: a name is defined at model level or regime level, never both.
    """
    merged = dict(user_regime.derived_categoricals)
    for var, grid in derived_categoricals.items():
        if var in merged:
            msg = (
                f"Ambiguous specification for derived_categoricals['{var}'] "
                f"in regime '{regime_name}': defined at model level and "
                f"regime level. Remove one."
            )
            raise ModelInitializationError(msg)
        merged[var] = grid
    return merged


def _fail_if_certainty_equivalent_is_mixed(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Reject a model that declares the certainty equivalent at both levels.

    It is declared once for the model, or once in every regime that has a
    continuation — never some of each. A partial declaration reads as if the
    silent regimes had opted out of something they never saw, when in fact
    they are still taking the model-level value.
    """
    with_continuation = {
        name for name, regime in user_regimes.items() if not regime.terminal
    }
    declaring = {
        name
        for name in with_continuation
        if user_regimes[name].certainty_equivalent is not None
    }
    silent = with_continuation - declaring
    if declaring and silent:
        msg = (
            f"Ambiguous specification for `certainty_equivalent`: declared in "
            f"regime(s) {sorted(declaring)} but not in {sorted(silent)}, which "
            f"take the model-level value. Declare it once on the `Model`, or "
            f"in every regime that has a continuation."
        )
        raise ModelInitializationError(msg)
