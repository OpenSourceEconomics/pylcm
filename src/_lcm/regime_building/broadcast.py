"""Model-level regime slots: merge and DAG-reachability pruning.

`merge_model_slots` merges `Model(functions=..., constraints=..., states=...,
state_transitions=..., actions=...)` into every regime under the
exactly-one-level rule — a name is defined at model level or regime level,
never both — with regime-level `None` masking the model entry.

`prune_broadcast_variables` then weeds the broadcast states and actions per
regime by DAG reachability: a broadcast variable survives in a regime only if
it is a transitive input of that regime's root computations in either phase
slice. Regime-level declarations are never pruned. The needed-set is a
cross-regime fixed point: a state unused inside a regime is still required
when a reachable target keeps it and the law of motion toward that target
reads it.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from dags import get_ancestors

from _lcm.grids import Grid
from _lcm.regime_building.phases import (
    PhasedRegimeSpec,
    RegimePhaseSpec,
    normalize_regime_phases,
)
from _lcm.typing import RegimeName, StateOrActionName
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import UserFunction

_BROADCASTABLE_SLOTS = (
    "functions",
    "constraints",
    "states",
    "state_transitions",
    "actions",
)


def merge_model_slots(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    model_slots: Mapping[str, Mapping[str, object]],
) -> tuple[
    MappingProxyType[RegimeName, UserRegime],
    MappingProxyType[RegimeName, frozenset[StateOrActionName]],
]:
    """Merge model-level slots into every regime.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.
        model_slots: Mapping of slot names (`functions`, `constraints`,
            `states`, `state_transitions`, `actions`) to model-level entries.

    Returns:
        Tuple of the merged regimes and, per regime, the names of broadcast
        states and actions (the pruning candidates).

    Raises:
        ModelInitializationError: If a name is defined at both levels, or a
            regime masks a name no model-level slot provides.

    """
    errors: list[str] = []
    merged_regimes: dict[RegimeName, UserRegime] = {}
    broadcast_variables: dict[RegimeName, frozenset[StateOrActionName]] = {}

    for regime_name, user_regime in user_regimes.items():
        replacements: dict[str, Mapping[str, object]] = {}
        variable_names: set[StateOrActionName] = set()
        for slot_name in _BROADCASTABLE_SLOTS:
            regime_slot = dict(getattr(user_regime, slot_name))
            model_slot = dict(model_slots.get(slot_name, {}))
            if slot_name == "state_transitions" and user_regime.terminal:
                # Terminal regimes consume no laws of motion; broadcast laws
                # are inert there and must not violate the empty-transitions
                # rule.
                model_slot = {}
            errors.extend(
                _merge_one_slot(
                    slot_name=slot_name,
                    regime_name=regime_name,
                    regime_slot=regime_slot,
                    model_slot=model_slot,
                )
            )
            if slot_name == "states":
                # Sharding is a cross-regime device-layout property; one
                # model-level declaration keeps every regime consistent.
                errors.extend(
                    f"states['{name}'] in regime '{regime_name}' has "
                    f"`distributed=True` — sharding is declared at the model "
                    f"level (`Model(states=...)`)."
                    for name, grid in regime_slot.items()
                    if isinstance(grid, Grid) and grid.distributed
                )
            if slot_name in ("states", "actions"):
                variable_names |= model_slot.keys() & regime_slot.keys()
            replacements[slot_name] = {**model_slot, **regime_slot}
        # A masked state's broadcast law is dropped with it.
        masked_states = {
            name
            for name, value in user_regime.states.items()
            if value is None and name in model_slots.get("states", {})
        }
        for slot_name in _BROADCASTABLE_SLOTS:
            replacements[slot_name] = {
                name: value
                for name, value in replacements[slot_name].items()
                if value is not None
                and not (slot_name == "state_transitions" and name in masked_states)
            }
        if not errors:
            merged_regimes[regime_name] = user_regime.replace(**replacements)
            broadcast_variables[regime_name] = frozenset(
                (
                    set(model_slots.get("states", {}))
                    | set(model_slots.get("actions", {}))
                )
                & (set(replacements["states"]) | set(replacements["actions"]))
            )

    if errors:
        raise ModelInitializationError(format_messages(errors))

    return MappingProxyType(merged_regimes), MappingProxyType(broadcast_variables)


def prune_broadcast_variables(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    broadcast_variables: Mapping[RegimeName, frozenset[StateOrActionName]],
) -> tuple[
    MappingProxyType[RegimeName, UserRegime],
    MappingProxyType[RegimeName, frozenset[StateOrActionName]],
]:
    """Weed broadcast states and actions per regime by DAG reachability.

    A broadcast variable is pruned from a regime when no root computation of
    either phase slice transitively reads it — in that regime or through a
    law of motion toward a reachable target that keeps it. Pruning drops the
    variable's grid, and for states also the regime's law entry for it.

    Args:
        user_regimes: Mapping of regime names to merged `Regime` instances.
        broadcast_variables: Per regime, the broadcast state/action names.

    Returns:
        Tuple of the pruned regimes and, per regime, the pruned names.

    Raises:
        ModelInitializationError: If a `distributed=True` model-level state is
            pruned from a non-terminal regime.

    """
    specs = {
        regime_name: normalize_regime_phases(user_regime)
        for regime_name, user_regime in user_regimes.items()
    }
    all_regime_names = frozenset(user_regimes)

    kept: dict[RegimeName, frozenset[StateOrActionName]] = {}
    for regime_name, user_regime in user_regimes.items():
        declared = (
            set(user_regime.states) | set(user_regime.actions)
        ) - broadcast_variables[regime_name]
        kept[regime_name] = frozenset(declared)

    for phase_name in ("solution", "simulation"):
        kept = _phase_fixed_point(
            specs=specs,
            user_regimes=user_regimes,
            broadcast_variables=broadcast_variables,
            kept=kept,
            phase_name=phase_name,
            all_regime_names=all_regime_names,
        )

    pruned_regimes: dict[RegimeName, UserRegime] = {}
    pruned_variables: dict[RegimeName, frozenset[StateOrActionName]] = {}
    errors: list[str] = []
    for regime_name, user_regime in user_regimes.items():
        pruned = broadcast_variables[regime_name] - kept[regime_name]
        pruned_variables[regime_name] = frozenset(pruned)
        if not pruned:
            pruned_regimes[regime_name] = user_regime
            continue
        errors.extend(
            _sharded_pruned_errors(
                user_regime=user_regime, regime_name=regime_name, pruned=pruned
            )
        )
        pruned_regimes[regime_name] = user_regime.replace(
            states={
                name: grid
                for name, grid in user_regime.states.items()
                if name not in pruned
            },
            actions={
                name: grid
                for name, grid in user_regime.actions.items()
                if name not in pruned
            },
            state_transitions={
                name: law
                for name, law in user_regime.state_transitions.items()
                if name not in pruned
            },
        )

    if errors:
        raise ModelInitializationError(format_messages(errors))

    return MappingProxyType(pruned_regimes), MappingProxyType(pruned_variables)


def _phase_fixed_point(
    *,
    specs: Mapping[RegimeName, PhasedRegimeSpec],
    user_regimes: Mapping[RegimeName, UserRegime],
    broadcast_variables: Mapping[RegimeName, frozenset[StateOrActionName]],
    kept: Mapping[RegimeName, frozenset[StateOrActionName]],
    phase_name: str,
    all_regime_names: frozenset[RegimeName],
) -> dict[RegimeName, frozenset[StateOrActionName]]:
    """Grow the kept-sets to this phase slice's least fixed point.

    Per iteration, each regime's needed-set is the DAG ancestry of its root
    computations plus the laws of motion toward reachable targets that
    currently keep the law's state; a broadcast variable joins the kept-set
    when needed. Monotone in the kept-sets, so iteration terminates. The
    input mapping is left untouched; the grown kept-sets are returned.
    """
    reachable: dict[RegimeName, frozenset[RegimeName]] = {}
    for regime_name, spec in specs.items():
        phase_slice: RegimePhaseSpec = getattr(spec, phase_name)
        regime_transition = phase_slice.regime_transition
        if regime_transition is None:
            reachable[regime_name] = frozenset()
        elif isinstance(regime_transition, Mapping):
            reachable[regime_name] = (
                frozenset(cast("Mapping[RegimeName, object]", regime_transition))
                & all_regime_names
            )
        else:
            reachable[regime_name] = all_regime_names

    grown = dict(kept)
    while True:
        changed = False
        for regime_name, user_regime in user_regimes.items():
            candidates = broadcast_variables[regime_name] - grown[regime_name]
            if not candidates:
                continue
            phase_slice = getattr(specs[regime_name], phase_name)
            needed = _needed_names(
                phase_slice=phase_slice,
                user_regime=user_regime,
                reachable_targets=reachable[regime_name],
                kept=grown,
            )
            newly_kept = candidates & needed
            if newly_kept:
                grown[regime_name] = grown[regime_name] | newly_kept
                changed = True
        if not changed:
            return grown


def _needed_names(
    *,
    phase_slice: RegimePhaseSpec,
    user_regime: UserRegime,
    reachable_targets: frozenset[RegimeName],
    kept: Mapping[RegimeName, frozenset[StateOrActionName]],
) -> set[str]:
    """Collect every name this phase slice's root computations read.

    Roots:

    - `utility` and `H` (when present)
    - every constraint
    - every derived-categorical function
    - the regime transition (coarse inputs, or every granular cell)
    - the laws of motion toward reachable targets that keep the law's state

    """
    pool: dict[str, UserFunction] = dict(phase_slice.functions)
    # A collective regime carries a per-stakeholder `utility_<s>` in place of a
    # single `utility`; a broadcast variable read only by those must survive.
    utility_names: tuple[str, ...] = (
        tuple(f"utility_{s}" for s in user_regime.stakeholders)
        if user_regime.stakeholders is not None
        else ("utility",)
    )
    targets = [name for name in (*utility_names, "H") if name in pool]
    targets += [name for name in user_regime.derived_categoricals if name in pool]

    roots = {
        f"__constraint_{name}": func for name, func in phase_slice.constraints.items()
    }
    roots |= _regime_transition_roots(phase_slice)
    roots |= _law_roots(
        phase_slice=phase_slice, reachable_targets=reachable_targets, kept=kept
    )
    pool |= roots
    targets += list(roots)

    if not targets:
        return set()
    return set(get_ancestors(pool, targets=targets, include_targets=True))


def _regime_transition_roots(
    phase_slice: RegimePhaseSpec,
) -> dict[str, UserFunction]:
    """Key the regime transition's callables as pruning roots.

    A per-target dict contributes every cell; a coarse transition contributes
    itself; a terminal regime contributes nothing.
    """
    regime_transition = phase_slice.regime_transition
    if isinstance(regime_transition, Mapping):
        return {
            f"__next_regime_{target_regime_name}": cast("UserFunction", cell)
            for target_regime_name, cell in regime_transition.items()
        }
    if regime_transition is not None:
        return {"__next_regime": cast("UserFunction", regime_transition)}
    return {}


def _law_roots(
    *,
    phase_slice: RegimePhaseSpec,
    reachable_targets: frozenset[RegimeName],
    kept: Mapping[RegimeName, frozenset[StateOrActionName]],
) -> dict[str, UserFunction]:
    """Key the laws of motion that rescue a state as pruning roots.

    A law counts only toward a reachable target that currently keeps the
    law's state — that target needs the handed-over value, so whatever the
    law reads stays alive in this regime.
    """
    roots: dict[str, UserFunction] = {}
    for state_name, raw in phase_slice.state_transitions.items():
        laws: dict[RegimeName, object] = (
            dict(cast("Mapping[RegimeName, object]", raw))
            if isinstance(raw, Mapping)
            else dict.fromkeys(reachable_targets, raw)
        )
        for target_regime_name, law in laws.items():
            if (
                law is not None
                and target_regime_name in reachable_targets
                and state_name in kept.get(target_regime_name, frozenset())
            ):
                roots[f"__law_{state_name}_{target_regime_name}"] = cast(
                    "UserFunction", law
                )
    return roots


def _sharded_pruned_errors(
    *,
    user_regime: UserRegime,
    regime_name: RegimeName,
    pruned: frozenset[StateOrActionName],
) -> list[str]:
    """Reject pruning a `distributed=True` state from a non-terminal regime."""
    if user_regime.terminal:
        return []
    return [
        f"Sharded state '{name}' is pruned from non-terminal regime "
        f"'{regime_name}' — its DAG never reads the state, so the sharded "
        f"V-array axis would disappear there. Remove `distributed=True` "
        f"from the model-level declaration, or make the regime use the state."
        for name in sorted(pruned)
        if isinstance(grid := user_regime.states.get(name), Grid) and grid.distributed
    ]


def _merge_one_slot(
    *,
    slot_name: str,
    regime_name: RegimeName,
    regime_slot: Mapping[str, object],
    model_slot: Mapping[str, object],
) -> list[str]:
    """Apply the exactly-one-level rule to one slot of one regime."""
    errors: list[str] = []
    for name, value in regime_slot.items():
        if value is None:
            if name not in model_slot:
                errors.append(
                    f"{slot_name}['{name}'] in regime '{regime_name}' is "
                    f"`None`, but no model-level entry provides '{name}' — "
                    f"there is nothing to mask.",
                )
        elif name in model_slot:
            errors.append(
                f"Ambiguous specification for {slot_name}['{name}'] in "
                f"regime '{regime_name}': defined at model level and regime "
                f"level. Remove one, or mask the model entry with `None`.",
            )
    return errors


def _model_slot_value_errors(
    *,
    model_slots: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Reject `None` values in model-level slots (masks are regime-level).

    Per-value grammar (grids, callables, law vocabulary, `Phased` placement)
    is validated when the merged regimes are constructed; only the
    merge-specific vocabulary is checked here.
    """
    errors: list[str] = []
    for slot_name, slot in model_slots.items():
        for name, value in slot.items():
            if value is None:
                errors.append(
                    f"Model-level {slot_name}['{name}'] cannot be `None` — "
                    f"masks are regime-level.",
                )
    return errors


def validate_model_slots(*, model_slots: Mapping[str, Mapping[str, object]]) -> None:
    """Raise on merge-specific vocabulary errors in model-level slots."""
    errors = _model_slot_value_errors(model_slots=model_slots)
    if errors:
        raise ModelInitializationError(format_messages(errors))
