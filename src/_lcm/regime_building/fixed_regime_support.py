"""Remove ordinary regime edges proved to have constant zero probability.

Only construction-time fixed leaves may feed the probability or its ordinary DAG
ancestors. States, actions, time, transition outputs and free parameters make an
edge conditional; no state probes or runtime values narrow the graph.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from dags.tree import qname_from_tree_path

from _lcm.params.processing import (
    cast_params_to_canonical_dtypes,
    find_param_candidates,
)
from _lcm.typing import FlatParams, RegimeName
from _lcm.utils.namespace import flatten_regime_namespace
from lcm.exceptions import InvalidNameError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import ByAge, StochasticTransition
from lcm.typing import UserParams

type Side = Literal["solve", "simulate"]


@dataclass(frozen=True, kw_only=True)
class FixedRegimeSupport:
    """Keep the reduced declarations and exact consumed fixed-key provenance."""

    user_regimes: MappingProxyType[RegimeName, UserRegime]
    """Regimes with constant-zero ordinary transition cells removed."""

    consumed_param_keys: frozenset[str]
    """Supplied flat keys used to prove a removed cell constant and zero."""


def prune_fixed_regime_support(
    *, user_regimes: Mapping[RegimeName, UserRegime], fixed_params: UserParams
) -> FixedRegimeSupport:
    """Remove exactly zero cells whose entire dependency graph is fixed.

    Per-target state laws toward a target removed from their phase are omitted
    too. Bare laws and source variables remain declared. Fixed keys consumed by
    removed cells retain their provenance for later unknown-key validation.
    """
    fixed_flat = flatten_regime_namespace(fixed_params)
    consumed: set[str] = set()
    result: dict[RegimeName, UserRegime] = {}
    for name, regime in user_regimes.items():
        transition = _prune_law(
            law=regime.regime_transitions,
            side=None,
            regime_name=name,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
        )
        removed: dict[Side, frozenset[str]] = {
            side: _targets(law=regime.regime_transitions, side=side)
            - _targets(law=transition, side=side)
            for side in ("solve", "simulate")
        }
        state_transitions = {
            state: _trim_state_law(
                law=law,
                removed=removed,
                regime_name=name,
                state=state,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
            )
            for state, law in regime.state_transitions.items()
        }
        result[name] = regime.replace(
            regime_transitions=transition, state_transitions=state_transitions
        )
    return FixedRegimeSupport(
        user_regimes=MappingProxyType(result), consumed_param_keys=frozenset(consumed)
    )


def _prune_law(
    *,
    law: object,
    side: Side | None,
    regime_name: RegimeName,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    consumed: set[str],
) -> object:
    if isinstance(law, ByAge):
        return law.with_mapped_laws(
            func=lambda case: _prune_law(
                law=case,
                side=side,
                regime_name=regime_name,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
            )
        )
    if isinstance(law, Phased):
        return Phased(
            solve=_prune_law(
                law=law.solve,
                side="solve",
                regime_name=regime_name,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
            ),
            simulate=_prune_law(
                law=law.simulate,
                side="simulate",
                regime_name=regime_name,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
            ),
        )
    if not isinstance(law, Mapping):
        return law
    if side is None:
        # A shared law can read phase-specific helpers, so its structural zeros
        # must be proved independently in each phase.
        solve = _prune_law(
            law=law,
            side="solve",
            regime_name=regime_name,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
        )
        simulate = _prune_law(
            law=law,
            side="simulate",
            regime_name=regime_name,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
        )
        return (
            solve
            if set(cast("Mapping[str, object]", solve))
            == set(cast("Mapping[str, object]", simulate))
            else Phased(solve=solve, simulate=simulate)
        )
    retained: dict[str, object] = {}
    removed_keys: set[str] = set()
    for target, cell in law.items():
        evaluated = (
            _evaluate_fixed_function(
                func=cell.func,
                path=(regime_name, target, "next_regime"),
                side=side,
                regime=regime,
                fixed_flat=fixed_flat,
                ancestors=(),
            )
            if isinstance(cell, StochasticTransition)
            else None
        )
        if evaluated is None:
            retained[target] = cell
            continue
        value, keys = evaluated
        scalar = np.asarray(value)
        if (
            scalar.shape != ()
            or scalar.dtype.kind not in "fiu"
            or not np.isfinite(scalar)
        ):
            retained[target] = cell
            continue
        # Exact zero is a structural predicate, never a tolerance comparison.
        if np.array_equal(scalar, np.zeros_like(scalar)):
            removed_keys.update(keys)
        else:
            retained[target] = cell
    if not retained:
        # Keep an invalid zero-mass row available to the ordinary probability
        # validator; it must not become an empty law or a terminal regime.
        return law
    consumed.update(removed_keys)
    return MappingProxyType(retained)


def _evaluate_fixed_function(
    *,
    func: Callable[..., Any],
    path: tuple[str, ...],
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    ancestors: tuple[str, ...],
) -> tuple[Any, frozenset[str]] | None:
    """Evaluate a complete fixed DAG, or retain an unresolved conditional edge."""
    arguments: dict[str, object] = {}
    keys: set[str] = set()
    for arg_name, parameter in inspect.signature(func).parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return None
        if _is_runtime_argument(arg_name=arg_name, regime=regime):
            return None
        evaluated = _resolve_fixed_argument(
            arg_name=arg_name,
            path=path,
            side=side,
            regime=regime,
            fixed_flat=fixed_flat,
            ancestors=ancestors,
        )
        if evaluated is None:
            return None
        arguments[arg_name], argument_keys = evaluated
        keys.update(argument_keys)
    try:
        value = func(**arguments)
    except Exception:  # noqa: BLE001 -- failure proves no structural zero
        # The ordinary compilation/evaluation route owns errors in a user law.
        # Failed evaluation proves no structural zero.
        return None
    return value, frozenset(keys)


def _resolve_fixed_argument(
    *,
    arg_name: str,
    path: tuple[str, ...],
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    ancestors: tuple[str, ...],
) -> tuple[Any, frozenset[str]] | None:
    """Resolve one argument through its helper DAG or a fixed parameter leaf."""
    helper = regime.functions.get(arg_name)
    if isinstance(helper, Phased):
        helper = helper.solve if side == "solve" else helper.simulate
    if helper is not None:
        if not callable(helper) or arg_name in ancestors:
            return None
        return _evaluate_fixed_function(
            func=helper,
            path=(path[0], arg_name),
            side=side,
            regime=regime,
            fixed_flat=fixed_flat,
            ancestors=(*ancestors, arg_name),
        )
    qname = qname_from_tree_path((*path, arg_name))
    candidates = find_param_candidates(qname=qname, params_flat=fixed_flat)
    if len(candidates) > 1:
        raise InvalidNameError(
            f"Ambiguous parameter specification for {qname!r}. Found values "
            f"at: {candidates}"
        )
    if not candidates:
        return None
    key = candidates[0]
    # A Series declares coordinate-indexed values; fixed parameters may
    # therefore still vary by age or model state.
    if isinstance(fixed_flat[key], pd.Series):
        return None
    return _canonicalize_fixed_leaf(
        regime_name=path[0], qname=qname, value=fixed_flat[key]
    ), frozenset((key,))


def _is_runtime_argument(*, arg_name: str, regime: UserRegime) -> bool:
    """Identify leaves whose value depends on a model problem or realization."""
    return (
        arg_name in regime.states
        or arg_name in regime.actions
        or arg_name in regime.derived_categoricals
        or arg_name in {"age", "period", "CE", "utility"}
        or arg_name.startswith("next_")
    )


def _canonicalize_fixed_leaf(
    *, regime_name: RegimeName, qname: str, value: object
) -> object:
    """Apply the parameter dtype boundary before executing a fixed callable."""
    canonical = cast_params_to_canonical_dtypes(
        cast(
            "FlatParams",
            MappingProxyType({regime_name: MappingProxyType({qname: value})}),
        )
    )
    return canonical[regime_name][qname]


def _targets(*, law: object, side: Side) -> frozenset[str]:
    """Collect one phase's target names across every age case."""
    if isinstance(law, ByAge):
        return frozenset(
            target for case in law.laws for target in _targets(law=case, side=side)
        )
    if isinstance(law, Phased):
        return _targets(law=law.solve if side == "solve" else law.simulate, side=side)
    if isinstance(law, Mapping):
        return frozenset(law)
    if isinstance(law, str):
        return frozenset((law,))
    return frozenset(getattr(law, "targets", ()) or ())


def _trim_state_law(
    *,
    law: object,
    removed: Mapping[Side, frozenset[str]],
    regime_name: RegimeName,
    state: str,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    consumed: set[str],
) -> object:
    """Omit explicit laws toward targets removed from the relevant phase."""
    if isinstance(law, Phased):
        for side, variant in (("solve", law.solve), ("simulate", law.simulate)):
            _record_removed_state_keys(
                law=variant,
                removed=removed[side],
                regime_name=regime_name,
                state=state,
                side=side,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
            )
        return Phased(
            solve=_trim_state_side(law=law.solve, removed=removed["solve"]),
            simulate=_trim_state_side(law=law.simulate, removed=removed["simulate"]),
        )
    for side in ("solve", "simulate"):
        _record_removed_state_keys(
            law=law,
            removed=removed[side],
            regime_name=regime_name,
            state=state,
            side=side,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
        )
    solve = _trim_state_side(law=law, removed=removed["solve"])
    simulate = _trim_state_side(law=law, removed=removed["simulate"])
    return (
        solve
        if removed["solve"] == removed["simulate"]
        else Phased(solve=solve, simulate=simulate)
    )


def _trim_state_side(*, law: object, removed: frozenset[str]) -> object:
    """Keep a bare law, or the explicit target cells still needed in this phase."""
    if isinstance(law, Mapping):
        return MappingProxyType(
            {target: cell for target, cell in law.items() if target not in removed}
        )
    return law


def _record_removed_state_keys(
    *,
    law: object,
    removed: frozenset[str],
    regime_name: RegimeName,
    state: str,
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    consumed: set[str],
) -> None:
    """Keep fixed keys on declared handoff cells omitted with their zero edges."""
    if not isinstance(law, Mapping):
        return
    for target in removed & law.keys():
        cell = law[target]
        if callable(cell):
            consumed.update(
                _get_declared_fixed_keys(
                    func=cell,
                    path=(regime_name, target, f"next_{state}"),
                    side=side,
                    regime=regime,
                    fixed_flat=fixed_flat,
                    ancestors=(),
                )
            )


def _get_declared_fixed_keys(
    *,
    func: Callable[..., Any],
    path: tuple[str, ...],
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    ancestors: tuple[str, ...],
) -> frozenset[str]:
    """Resolve supplied fixed leaves without evaluating an omitted handoff law."""
    result: set[str] = set()
    for arg_name in inspect.signature(func).parameters:
        if _is_runtime_argument(arg_name=arg_name, regime=regime):
            continue
        helper = regime.functions.get(arg_name)
        if isinstance(helper, Phased):
            helper = helper.solve if side == "solve" else helper.simulate
        if callable(helper) and arg_name not in ancestors:
            result.update(
                _get_declared_fixed_keys(
                    func=helper,
                    path=(path[0], arg_name),
                    side=side,
                    regime=regime,
                    fixed_flat=fixed_flat,
                    ancestors=(*ancestors, arg_name),
                )
            )
        elif helper is None:
            qname = qname_from_tree_path((*path, arg_name))
            candidates = find_param_candidates(qname=qname, params_flat=fixed_flat)
            if len(candidates) > 1:
                raise InvalidNameError(
                    f"Ambiguous parameter specification for {qname!r}. "
                    f"Found values at: {candidates}"
                )
            for candidate in candidates:
                if not isinstance(fixed_flat[candidate], pd.Series):
                    _canonicalize_fixed_leaf(
                        regime_name=path[0], qname=qname, value=fixed_flat[candidate]
                    )
            result.update(candidates)
    return frozenset(result)
