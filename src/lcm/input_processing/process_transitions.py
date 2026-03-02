"""Extract and resolve state transitions across regime boundaries.

Collects all functions (utility, constraints, helpers, transitions) for each
regime, resolving per-boundary transitions using grid `transition` attributes.
Transition functions are target-prefixed with `QNAME_DELIMITER`
(e.g., `"retired__next_wealth"`). The regime transition (`next_regime`) is
included under its plain name.

"""

import inspect
from collections.abc import Mapping
from typing import TypeAliasType, cast, overload

from dags.tree import QNAME_DELIMITER

from lcm.exceptions import ModelInitializationError
from lcm.grids import Grid, _DiscreteGridBase
from lcm.regime import Regime
from lcm.shocks._base import _ShockGrid
from lcm.typing import (
    ContinuousState,
    DiscreteState,
    UserFunction,
)


class _IdentityTransition:
    """Identity transition function for fixed states.

    Used so the params template includes fixed states. The `_is_auto_identity`
    attribute lets validation distinguish auto-generated identities from
    user-provided transitions.

    """

    _is_auto_identity: bool = True

    def __init__(self, state_name: str, *, annotation: TypeAliasType) -> None:
        self._state_name = state_name
        self.__name__ = f"next_{state_name}"
        param = inspect.Parameter(
            state_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
        )
        self.__signature__ = inspect.Signature(
            [param],
            return_annotation=annotation,
        )
        self.__annotations__ = {state_name: annotation, "return": annotation}

    @overload
    def __call__(self, **kwargs: DiscreteState) -> DiscreteState: ...
    @overload
    def __call__(self, **kwargs: ContinuousState) -> ContinuousState: ...
    def __call__(
        self, **kwargs: DiscreteState | ContinuousState
    ) -> DiscreteState | ContinuousState:
        return kwargs[self._state_name]


def _make_identity_fn(
    state_name: str, *, annotation: TypeAliasType
) -> _IdentityTransition:
    """Create an identity transition for a fixed state."""
    return _IdentityTransition(state_name, annotation=annotation)


def collect_all_regime_functions(
    regimes: Mapping[str, Regime],
) -> tuple[
    dict[str, dict[str, UserFunction]],
    dict[str, frozenset[str]],
]:
    """Collect all functions for all regimes with boundary-encoded transitions.

    Transition function names are target-prefixed with `QNAME_DELIMITER`
    (e.g., `"retired__next_wealth"`). `next_regime` is included as-is.

    Non-transition functions (utility, constraints, H, helpers) are included
    under their original names.

    Returns:
        Tuple of two dicts keyed by regime name:
        - all_functions: flat mapping from function name to callable
        - transition_keys: frozenset of keys that are transitions

    """
    nested_transitions = _extract_transitions(regimes)

    all_functions: dict[str, dict[str, UserFunction]] = {}
    transition_keys: dict[str, frozenset[str]] = {}

    for regime_name, regime in regimes.items():
        funcs: dict[str, UserFunction] = {}
        trans_keys: set[str] = set()

        # Non-transition functions
        funcs.update(regime.functions)
        funcs.update(regime.constraints)

        # Transitions
        nested = nested_transitions[regime_name]
        for key, value in nested.items():
            if key == "next_regime":
                funcs["next_regime"] = cast("UserFunction", value)
                trans_keys.add("next_regime")
            else:
                # key is a target regime name; value is dict of state transitions
                target_name = key
                target_transitions = cast("dict[str, UserFunction]", value)
                for next_name, func in target_transitions.items():
                    flat_name = f"{target_name}{QNAME_DELIMITER}{next_name}"
                    funcs[flat_name] = func
                    trans_keys.add(flat_name)

        all_functions[regime_name] = funcs
        transition_keys[regime_name] = frozenset(trans_keys)

    return all_functions, transition_keys


def collect_regime_functions(
    *,
    regime_name: str,
    regimes: Mapping[str, Regime],
) -> tuple[dict[str, UserFunction], frozenset[str]]:
    """Single-regime convenience wrapper around `collect_all_regime_functions`."""
    all_funcs, all_keys = collect_all_regime_functions(regimes)
    return all_funcs[regime_name], all_keys[regime_name]


def _extract_transitions(
    regimes: Mapping[str, Regime],
) -> dict[str, dict[str, dict[str, UserFunction] | UserFunction]]:
    """Extract nested transitions for all regimes at once.

    Returns per-regime nested transitions dict:
    `{source: {target: {next_state: func}, "next_regime": func}, ...}`
    Terminal regimes get empty dicts.

    """
    states_per_regime = {name: set(r.states.keys()) for name, r in regimes.items()}
    result: dict[str, dict[str, dict[str, UserFunction] | UserFunction]] = {}
    for regime_name, regime in regimes.items():
        if regime.terminal:
            result[regime_name] = {}
            continue
        nested = _build_regime_transitions(
            regime_name=regime_name,
            regimes=regimes,
            states_per_regime=states_per_regime,
        )
        result[regime_name] = nested
    return result


def _build_regime_transitions(
    *,
    regime_name: str,
    regimes: Mapping[str, Regime],
    states_per_regime: Mapping[str, set[str]],
) -> dict[str, dict[str, UserFunction] | UserFunction]:
    """Build nested transitions for a single source regime.

    Returns nested dict: `{target: {next_state: func}, "next_regime": func}`.

    """
    regime = regimes[regime_name]
    assert regime.transition is not None  # noqa: S101

    nested: dict[str, dict[str, UserFunction] | UserFunction] = {}
    nested["next_regime"] = regime.transition.func

    for target_name, target_state_names in states_per_regime.items():
        target_regime = regimes[target_name]
        boundary_key = (regime_name, target_name)
        boundary_transitions: dict[str, UserFunction] = {}
        missing_states: list[str] = []

        for state_name in target_state_names:
            result = _resolve_state_transition(
                state_name=state_name,
                boundary_key=boundary_key,
                source_regime=regime,
                target_regime=target_regime,
            )
            if isinstance(result, _Unresolved):
                missing_states.append(state_name)
            else:
                boundary_transitions[f"next_{state_name}"] = result

        if missing_states:
            continue

        _validate_discrete_category_compatibility(
            boundary_key=boundary_key,
            boundary_transitions=boundary_transitions,
            source_regime=regime,
            target_regime=target_regime,
        )
        nested[target_name] = boundary_transitions

    return nested


class _Unresolved:
    """Sentinel type for unresolved transitions."""


_UNRESOLVED = _Unresolved()


def _resolve_state_transition(
    *,
    state_name: str,
    boundary_key: tuple[str, str],
    source_regime: Regime,
    target_regime: Regime,
) -> UserFunction | _Unresolved:
    """Resolve the transition function for one state in a `(source, target)` boundary.

    Priority (highest to lowest):

    1. Target grid mapping with `(source, target)` key
    2. Source grid mapping with `(source, target)` key
    3. Source grid single-callable
    4. Source grid `None` (identity)
    5. Target grid single-callable
    6. Target grid `None` (identity)
    7. Target or source has mapping but boundary not listed -> identity

    Returns:
        Resolved function, or the `_UNRESOLVED` sentinel.

    """
    source_grid = source_regime.states.get(state_name)
    target_grid = target_regime.states.get(state_name)

    # ShockGrids have intrinsic transitions handled separately by
    # _get_internal_functions; return a placeholder so the target stays reachable.
    if isinstance(source_grid, _ShockGrid) or isinstance(target_grid, _ShockGrid):
        return lambda: None

    source_trans = _get_grid_transition(source_grid)
    target_trans = _get_grid_transition(target_grid)

    identity = _make_identity_for_target(state_name, target_regime)

    # Priority 1-2: Mapping with this boundary key (target wins over source).
    for trans in (target_trans, source_trans):
        if isinstance(trans, Mapping) and boundary_key in trans:
            fn = trans[boundary_key]  # ty: ignore[invalid-argument-type]
            return identity if fn is None else fn  # ty: ignore[invalid-return-type]

    # Priority 3-6: Source then target — single-callable or None
    for trans in (source_trans, target_trans):
        if callable(trans):
            return trans
        if trans is None:
            return identity

    # Priority 7: Mapping exists but boundary not listed → identity
    if isinstance(target_trans, Mapping) or isinstance(source_trans, Mapping):
        return identity
    return _UNRESOLVED


def _get_grid_transition(grid: Grid | None) -> object:
    """Extract the raw transition attribute from a grid, or return `_UNRESOLVED`."""
    if grid is None:
        return _UNRESOLVED
    if isinstance(grid, _ShockGrid):
        return _UNRESOLVED
    return getattr(grid, "transition", _UNRESOLVED)


def _make_identity_for_target(state_name: str, target_regime: Regime) -> UserFunction:
    """Create an identity transition using the target regime's type for a state."""
    ann = (
        DiscreteState
        if state_name in target_regime.states
        and isinstance(target_regime.states[state_name], _DiscreteGridBase)
        else ContinuousState
    )
    return _make_identity_fn(state_name, annotation=ann)


def _validate_discrete_category_compatibility(
    *,
    boundary_key: tuple[str, str],
    boundary_transitions: dict[str, UserFunction],
    source_regime: Regime,
    target_regime: Regime,
) -> None:
    """Validate discrete states with different categories have explicit transitions.

    Raise `ModelInitializationError` if a discrete state has different category sets
    in source and target regimes but no per-boundary transition was provided.

    """
    source_name, target_name = boundary_key
    for state_name in target_regime.states:
        source_grid = source_regime.states.get(state_name)
        target_grid = target_regime.states.get(state_name)

        if not (
            isinstance(source_grid, _DiscreteGridBase)
            and isinstance(target_grid, _DiscreteGridBase)
        ):
            continue

        if source_grid.categories == target_grid.categories:
            continue

        # Categories differ — check if an explicit per-boundary transition was provided
        next_name = f"next_{state_name}"
        transition_fn = boundary_transitions.get(next_name)
        if transition_fn is None or getattr(transition_fn, "_is_auto_identity", False):
            raise ModelInitializationError(
                f"State '{state_name}' has different discrete categories in regimes "
                f"'{source_name}' and '{target_name}' "
                f"({list(source_grid.categories)} vs {list(target_grid.categories)}) "
                f"but no per-boundary transition was provided. "
                f"Use a mapping transition on the target regime's grid, e.g.: "
                f'DiscreteGrid(..., transition={{("{source_name}", "{target_name}"): '
                f"map_fn}})"
            )
