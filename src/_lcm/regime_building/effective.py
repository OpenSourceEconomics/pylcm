"""The effective regime — a regime as the model actually runs it.

`build_effective_regimes` merges model-level slots (`derived_categoricals`)
into every user `Regime` and constructs one `EffectiveUserRegime` per regime:
complete (default `H` injected, completeness validated), immutable, and still
in user vocabulary — coarse laws, `Phased` containers, and per-target dicts
survive untouched, so the params template reads the user's coarseness off it.
"""

from collections.abc import Mapping
from types import MappingProxyType

from _lcm.grids import DiscreteGrid
from _lcm.typing import FunctionName, RegimeName
from _lcm.user_regime_validation import _validate_completeness
from _lcm.utils.containers import ensure_containers_are_immutable
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.regime import _default_H


class EffectiveUserRegime(UserRegime):
    """A complete, validated regime spec in user vocabulary.

    Subclasses the user `Regime` so every engine boundary typed against the
    user form accepts it, but is constructed only by
    `build_effective_regimes` — never by users. Construction bypasses the
    user constructor (no re-validation of value-shape properties, which the
    user `Regime` already checked) and instead:

    - injects the default Bellman aggregator `H` when a non-terminal regime
      supplies none
    - validates completeness: a `utility` entry, state-transition coverage,
      no state/action overlap, distributed-grid rules

    """

    def __init__(
        self,
        *,
        user_regime: UserRegime | None = None,
        derived_categoricals: Mapping[FunctionName, DiscreteGrid] | None = None,
        **field_values: object,
    ) -> None:
        # `dataclasses.replace` reconstructs via `type(self)(**fields)`, so a
        # replaced effective regime arrives as field values: rebuild the user
        # regime from them (re-running its value-shape validation), then
        # proceed exactly like the factory path.
        if user_regime is None:
            if derived_categoricals is not None:
                field_values["derived_categoricals"] = derived_categoricals
            user_regime = UserRegime(**field_values)  # ty: ignore[invalid-argument-type]
            derived_categoricals = None
        elif field_values:
            msg = (
                "Pass either `user_regime` or the full set of regime fields, not both."
            )
            raise TypeError(msg)
        # `derived_categoricals` is the final (model-merged) mapping; `None`
        # keeps the regime's own entries.
        if derived_categoricals is None:
            derived_categoricals = user_regime.derived_categoricals
        for field_name in (
            "transition",
            "active",
            "states",
            "state_transitions",
            "actions",
            "constraints",
            "solver",
            "taste_shocks",
            "description",
        ):
            object.__setattr__(self, field_name, getattr(user_regime, field_name))

        functions = dict(user_regime.functions)
        # Terminal regimes don't need H since Q = U directly (no E_next_V).
        if user_regime.transition is not None and "H" not in functions:
            functions["H"] = _default_H
        object.__setattr__(
            self, "functions", ensure_containers_are_immutable(functions)
        )
        object.__setattr__(
            self,
            "derived_categoricals",
            ensure_containers_are_immutable(dict(derived_categoricals)),
        )

        error_messages = _validate_completeness(self)
        if error_messages:
            raise RegimeInitializationError(format_messages(error_messages))


def build_effective_regimes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> MappingProxyType[RegimeName, EffectiveUserRegime]:
    """Build the effective regime for every user regime.

    Merges model-level `derived_categoricals` into each regime (a regime
    entry with identical categories is tolerated; conflicting categories
    raise) and constructs the validated `EffectiveUserRegime`.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.
        derived_categoricals: Model-level categorical grids to broadcast.

    Returns:
        Immutable mapping of regime names to effective regimes.

    Raises:
        ModelInitializationError: If a regime has a `derived_categoricals`
            entry conflicting with a model-level one.
        RegimeInitializationError: If a regime is incomplete (e.g. missing
            `utility` or state-transition coverage), with the regime name
            prefixed.

    """
    result: dict[RegimeName, EffectiveUserRegime] = {}
    for regime_name, user_regime in user_regimes.items():
        merged = _merge_derived_categoricals(
            regime_name=regime_name,
            user_regime=user_regime,
            derived_categoricals=derived_categoricals,
        )
        try:
            result[regime_name] = EffectiveUserRegime(
                user_regime=user_regime, derived_categoricals=merged
            )
        except RegimeInitializationError as error:
            raise RegimeInitializationError(
                f"In regime '{regime_name}': {error}"
            ) from None
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
