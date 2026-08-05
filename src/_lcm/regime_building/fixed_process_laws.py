"""Bind process laws supplied through `fixed_params` into the process grids.

A stochastic process leaves `None` in every distribution field the user has not
filled, and reports those fields as `params_to_pass_at_runtime`. Two public
routes fill them, and they mean the same thing: passing the value to the
process constructor, and naming it in `Model(fixed_params=...)`.

`bind_fixed_process_laws` makes them literally the same object. It runs before
the model structure is prepared, so every consumer downstream — the handoff and
entry-law validation, intrinsic-entry synthesis, the target's own nodes,
diagnostics, and simulation — reads one resolved process rather than a
half-specified one that a later stage would have patched. A law that genuinely
arrives at runtime keeps its `None` fields and stays a runtime parameter.

Where a law may be written is not decided here. The binder resolves against the
project's single rule, `find_param_candidates`, so a process law obeys exactly
the levels every other fixed parameter obeys: under its state, under its regime,
or at model level, most specific winning and a value at two levels an error.
Anything narrower would reject a model the params template calls fully
specified — and entry into a process requires its law to be fixed *here*.
"""

import dataclasses
import numbers
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from dags.tree import qname_from_tree_path, tree_path_from_qname

from _lcm.grids import Grid
from _lcm.params.processing import find_param_candidates
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import RegimeName, StateName
from _lcm.utils.namespace import ParamsQnameDepth, flatten_regime_namespace
from lcm.exceptions import InvalidNameError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedGrid
from lcm.typing import UserParams

# A state as the user declares it on a `Regime`.
type StateDeclaration = Grid | Phased | AgeSpecializedGrid | None


def bind_fixed_process_laws(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    fixed_params: UserParams,
) -> tuple[MappingProxyType[RegimeName, UserRegime], UserParams, frozenset[str]]:
    """Bake `fixed_params` entries naming a process's law into that process.

    Args:
        user_regimes: Mapping of regime names to regimes whose states may hold
            half-specified stochastic processes.
        fixed_params: Parameters fixed at model initialization, in the
            user-facing nested form.

    Returns:
        Tuple of the regimes with every bindable process law baked in, the
        `fixed_params` still to be resolved against the params template, and the
        flat keys that bound a law by broadcast. A broadcast may serve slots
        beyond the process it pinned, so it stays in the residual; the third
        element is what tells the template resolver not to call it unknown when
        it does not.

    Raises:
        InvalidNameError: If a process parameter is written at two levels.

    """
    if not fixed_params:
        return MappingProxyType(dict(user_regimes)), fixed_params, frozenset()

    params_flat = flatten_regime_namespace(fixed_params)
    bound_values, consumed_keys = _resolve_process_law_params(
        user_regimes=user_regimes, params_flat=params_flat
    )
    if not bound_values:
        return MappingProxyType(dict(user_regimes)), fixed_params, frozenset()

    bound_regimes = {
        regime_name: _bind_regime(
            user_regime=user_regime,
            regime_values=bound_values.get(regime_name, {}),
        )
        for regime_name, user_regime in user_regimes.items()
    }
    # A key naming the state exactly pinned that process and nothing else, so it
    # leaves the residual. A coarser key is a broadcast, which may serve slots
    # beyond the one it bound, so it stays — and is reported instead.
    exact_keys = {
        key
        for key in consumed_keys
        if len(tree_path_from_qname(key)) == ParamsQnameDepth.REGIME__FUNC__PARAM
    }
    residual = _drop_flat_keys(params=fixed_params, drop=exact_keys)
    return (
        MappingProxyType(bound_regimes),
        residual,
        frozenset(consumed_keys - exact_keys),
    )


def _resolve_process_law_params(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    params_flat: Mapping[str, Any],
) -> tuple[dict[RegimeName, dict[StateName, dict[str, Any]]], set[str]]:
    """Resolve every runtime process parameter against the user's `fixed_params`.

    The slots come from the regimes as declared, before any binding: binding only
    ever removes slots, so resolving against the pre-bind set is complete.

    Args:
        user_regimes: Mapping of regime names to regimes as the user declared
            them.
        params_flat: Flattened `fixed_params`, keyed by qualified name.

    Returns:
        Tuple of the resolved values, nested by regime and state, and the flat
        keys they came from.

    Raises:
        InvalidNameError: If a process parameter is written at two levels.

    """
    resolved: dict[RegimeName, dict[StateName, dict[str, Any]]] = {}
    consumed: set[str] = set()
    for regime_name, user_regime in user_regimes.items():
        for state_name, declaration in user_regime.states.items():
            for process in _processes_in(declaration):
                for param_name in process.params_to_pass_at_runtime:
                    qname = qname_from_tree_path((regime_name, state_name, param_name))
                    candidates = find_param_candidates(
                        qname=qname, params_flat=params_flat
                    )
                    if len(candidates) > 1:
                        msg = (
                            f"Ambiguous parameter specification for {qname!r}. "
                            f"Found values at: {candidates}"
                        )
                        raise InvalidNameError(msg)
                    if not candidates:
                        continue
                    value = params_flat[candidates[0]]
                    scalar = _as_bindable_scalar(value)
                    if scalar is None:
                        continue
                    resolved.setdefault(regime_name, {}).setdefault(state_name, {})[
                        param_name
                    ] = scalar
                    consumed.add(candidates[0])
    return resolved, consumed


def _processes_in(declaration: StateDeclaration) -> list[_ContinuousStochasticProcess]:
    """Return the stochastic processes a state declaration holds.

    Args:
        declaration: A state as the user declared it, possibly `Phased`.

    Returns:
        List of the processes in it, empty when it holds none.

    """
    if isinstance(declaration, Phased):
        return [
            process
            for member in (declaration.solve, declaration.simulate)
            for process in _processes_in(cast("StateDeclaration", member))
        ]
    if isinstance(declaration, _ContinuousStochasticProcess):
        return [declaration]
    return []


def _as_bindable_scalar(value: Any) -> Any | None:  # noqa: ANN401
    """Return `value` as a Python scalar a process field can take, or `None`.

    A process computes its nodes eagerly with numpy, so a 0-d array has to be
    materialized before it can be a field. Anything that is not a numeric scalar
    — an array with axes, a container, a string — is left alone and stays a
    runtime parameter.

    Args:
        value: A leaf of the user's `fixed_params`.

    Returns:
        The scalar, or `None` when the value cannot pin a process field.

    """
    if isinstance(value, numbers.Real):
        return value
    array = getattr(value, "shape", None)
    if array is None or array != ():
        return None
    dtype_kind = getattr(getattr(value, "dtype", None), "kind", "")
    if dtype_kind == "b":
        return bool(value)
    if dtype_kind in "iu":
        return int(value)
    if dtype_kind == "f":
        return float(np.asarray(value))
    return None


def _bind_regime(
    *,
    user_regime: UserRegime,
    regime_values: Mapping[StateName, Mapping[str, Any]],
) -> UserRegime:
    """Return one regime with every resolved process law baked into its grids.

    Args:
        user_regime: Regime whose states may hold half-specified processes.
        regime_values: Mapping of state names to the process fields resolved for
            them.

    Returns:
        The regime, unchanged when nothing bound.

    """
    if not regime_values:
        return user_regime
    states: dict[StateName, StateDeclaration] = dict(user_regime.states)
    for state_name, values in regime_values.items():
        states[state_name] = _bind_declaration(
            declaration=states[state_name], values=values
        )
    return user_regime.replace(states=MappingProxyType(states))


def _bind_declaration(
    *, declaration: StateDeclaration, values: Mapping[str, Any]
) -> StateDeclaration:
    """Bind a state declaration, which may be `Phased` over two grids.

    Args:
        declaration: A state as the user declared it.
        values: Process fields resolved for this state.

    Returns:
        The declaration with every process in it replaced by a bound one. A
        field a given process does not take is skipped, so the two members of a
        `Phased` may take different subsets.

    """
    if isinstance(declaration, Phased):
        return Phased(
            solve=_bind_declaration(
                declaration=cast("StateDeclaration", declaration.solve), values=values
            ),
            simulate=_bind_declaration(
                declaration=cast("StateDeclaration", declaration.simulate),
                values=values,
            ),
        )
    if not isinstance(declaration, _ContinuousStochasticProcess):
        return declaration
    bindable = frozenset(declaration.params_to_pass_at_runtime)
    to_bind = {name: value for name, value in values.items() if name in bindable}
    if not to_bind:
        return declaration
    return dataclasses.replace(declaration, **to_bind)


def _drop_flat_keys(*, params: UserParams, drop: set[str]) -> UserParams:
    """Return `params` without the entries named by their flat qualified names.

    Args:
        params: Parameters in the user-facing nested form.
        drop: Qualified names to remove.

    Returns:
        The nested parameters with those leaves gone, and every branch left
        empty by the removal gone with them.

    """
    if not drop:
        return params

    def _prune(branch: Mapping[str, Any], prefix: tuple[str, ...]) -> dict[str, Any]:
        kept: dict[str, Any] = {}
        for key, value in branch.items():
            path = (*prefix, key)
            if qname_from_tree_path(path) in drop:
                continue
            if isinstance(value, Mapping):
                inner = _prune(cast("Mapping[str, Any]", value), path)
                if inner:
                    kept[key] = inner
                continue
            kept[key] = value
        return kept

    return MappingProxyType(_prune(params, ()))
