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

from collections.abc import Callable, Mapping
from typing import TypeAliasType, cast

from dags.tree import QNAME_DELIMITER

from _lcm.grids import DiscreteGrid, Grid
from _lcm.identity_transition import _IdentityTransition
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import RegimeName, StateName, TransitionFunctionName
from lcm.exceptions import (
    RegimeInitializationError,
)
from lcm.phased import Phased
from lcm.typing import ContinuousState, DiscreteState, UserFunction


def collect_state_transitions(
    states: Mapping[StateName, Grid | Phased | AgeSpecializedGrid | None],
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

    An outer `Phased` with a per-target dict on at least one side is NORMALIZED
    into the inner form — one qualified key per target, each carrying a `Phased` of
    that target's two laws. `Phased` of dicts is not a value any consumer
    understands: registered as-is it would reach `Regime.get_all_functions` as a
    raw dict where a callable is required. The two forms mean the same thing, and
    the per-key one is what the engine already consumes.

    A **bare** law on one side (map-vs-bare) BROADCASTS over the per-target side's
    targets — the same meaning a bare state law has outside `Phased`. The
    per-target side's key set defines the targets; the shape validator has already
    rejected two per-target dicts over different targets, so when both sides are
    dicts their keys match and either set works.

    Note this produces a `Phased` value *under a per-target key*, which is exactly
    what `_validate_per_target_dict` forbids a USER to write (`Phased` is
    outermost-only). No contradiction: that rule governs the user's spelling, this
    is the internal normal form the outer spelling is rewritten INTO.
    """
    if isinstance(raw, Phased) and (
        isinstance(raw.solve, Mapping) or isinstance(raw.simulate, Mapping)
    ):

        def _cell(side: object, target: RegimeName) -> UserFunction:
            # A per-target dict yields its cell; a bare law broadcasts over targets.
            if isinstance(side, Mapping):
                by_target = cast("Mapping[RegimeName, object]", side)
                return cast("UserFunction", by_target[target])
            return cast("UserFunction", side)

        target_source = raw.solve if isinstance(raw.solve, Mapping) else raw.simulate
        targets = cast("Mapping[RegimeName, object]", target_source)
        for target_regime_name in targets:
            key = f"next_{name}{QNAME_DELIMITER}{target_regime_name}"
            transitions[key] = Phased(
                solve=_cell(raw.solve, target_regime_name),
                simulate=_cell(raw.simulate, target_regime_name),
            )
    elif callable(raw) or isinstance(raw, Phased):
        transitions[f"next_{name}"] = cast("UserFunction", raw)
    elif isinstance(raw, Mapping):
        for target_regime_name, law in raw.items():
            key = f"next_{name}{QNAME_DELIMITER}{target_regime_name}"
            transitions[key] = cast("UserFunction", law)
