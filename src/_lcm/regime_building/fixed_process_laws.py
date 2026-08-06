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
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags.tree import qname_from_tree_path, tree_path_from_qname

from _lcm.grids import Grid
from _lcm.params.processing import (
    cast_params_to_canonical_dtypes,
    find_param_candidates,
)
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import FlatParams, RegimeName, StateName
from _lcm.utils.namespace import ParamsQnameDepth, flatten_regime_namespace
from lcm.exceptions import InvalidNameError, InvalidParamsError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedGrid
from lcm.typing import UserParams

# A state as the user declares it on a `Regime`, or one member of a `Phased`
# declaration. The two are one type because a carried state is
# `Phased(solve=callable, simulate=Grid)`, so a member may be a plain function
# where the outer slot may not.
type StateDeclaration = Grid | Phased | AgeSpecializedGrid | Callable[..., Any] | None


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
        InvalidParamsError: If a resolved leaf is not a numeric scalar, or is a
            leaf type the canonical boundary cast rejects.

    """
    raw: dict[RegimeName, dict[str, Any]] = {name: {} for name in user_regimes}
    slot_owner: dict[tuple[RegimeName, str], tuple[StateName, str]] = {}
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
                    slot = qname_from_tree_path((state_name, param_name))
                    raw[regime_name][slot] = params_flat[candidates[0]]
                    slot_owner[regime_name, slot] = (state_name, param_name)
                    consumed.add(candidates[0])

    # A varying leaf is rejected before the boundary cast, which speaks to the
    # dtype contract rather than to the process's own one-number-per-field law.
    _fail_if_any_process_law_field_varies(raw=raw, slot_owner=slot_owner)

    # The same boundary cast every other fixed parameter goes through, so a
    # process law has one dtype and leaf-type contract rather than a second one.
    # It also raises on the leaf types no parameter may take, naming the qname.
    canonical = cast_params_to_canonical_dtypes(
        cast(
            "FlatParams",
            MappingProxyType({k: MappingProxyType(v) for k, v in raw.items()}),
        )
    )

    resolved: dict[RegimeName, dict[StateName, dict[str, Any]]] = {}
    for regime_name, regime_slots in canonical.items():
        for slot, value in regime_slots.items():
            state_name, param_name = slot_owner[regime_name, slot]
            resolved.setdefault(regime_name, {}).setdefault(state_name, {})[
                param_name
            ] = _as_process_field(
                value=value,
                qname=qname_from_tree_path((regime_name, state_name, param_name)),
            )
    return resolved, consumed


def _processes_in(declaration: StateDeclaration) -> list[_ContinuousStochasticProcess]:
    """Return the stochastic processes a state declaration holds.

    Args:
        declaration: A state as the user declared it, possibly `Phased`, or one
            member of such a pair — for a carried state that member is the
            solve-phase callable.

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


def _fail_if_any_process_law_field_varies(
    *,
    raw: Mapping[RegimeName, Mapping[str, Any]],
    slot_owner: Mapping[tuple[RegimeName, str], tuple[StateName, str]],
) -> None:
    """Reject every process law field that was given more than one number.

    Args:
        raw: Mapping of regime names to the process law slots collected for them.
        slot_owner: Mapping of `(regime, slot)` to the state and field it fills.

    Raises:
        InvalidParamsError: If any value holds more than one number.

    """
    for regime_name, regime_slots in raw.items():
        for slot, value in regime_slots.items():
            state_name, param_name = slot_owner[regime_name, slot]
            _fail_if_a_process_law_field_varies(
                value=value,
                qname=qname_from_tree_path((regime_name, state_name, param_name)),
            )


def _fail_if_a_process_law_field_varies(*, value: Any, qname: str) -> None:  # noqa: ANN401
    """Reject a process law field given more than one number.

    Args:
        value: The value supplied for the field, before any dtype cast.
        qname: Qualified name of the parameter, named in the rejection.

    Raises:
        InvalidParamsError: If the value holds more than one number.

    """
    shape = getattr(value, "shape", ())
    if not shape:
        return
    msg = (
        f"The fixed parameter {qname!r} pins a stochastic process's law, which "
        f"is one number per field, but its value has shape {shape}. A process's "
        f"law is fixed at construction, so this field takes one number. A law "
        f"that varies with age is not supported: `AgeSpecializedGrid` builds a "
        f"plain grid per age and rejects a stochastic process. Fix the law "
        f"across ages and carry the age variation in a deterministic function "
        f"that reads `age` and the process's draw. A quantity varying across "
        f"subjects belongs in such a function too, never in the process's law."
    )
    raise InvalidParamsError(msg)


def _as_process_field(*, value: Any, qname: str) -> float | int | bool:  # noqa: ANN401
    """Return a canonically cast leaf as the Python scalar a process field takes.

    A process's distribution fields are Python scalars, and it computes its nodes
    eagerly, so the 0-d array the boundary cast produces is materialized here.

    Args:
        value: A leaf of `cast_params_to_canonical_dtypes`' output.
        qname: Qualified name of the parameter, named in the rejection.

    Returns:
        The scalar to bake into the process.

    Raises:
        InvalidParamsError: If the leaf is not a scalar. A process's law is one
            number per field, so an array — an age-varying or per-subject one —
            has no field to become.

    """
    array = jnp.asarray(value)
    _fail_if_a_process_law_field_varies(value=value, qname=qname)
    if array.dtype.kind == "b":
        return bool(array)
    if array.dtype.kind in "iu":
        return int(array)
    return float(array)


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
