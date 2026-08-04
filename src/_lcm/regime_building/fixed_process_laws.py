"""Bind process laws supplied through `fixed_params` into the process grids.

A stochastic process leaves `None` in every distribution field the user has not
filled, and reports those fields as `params_to_pass_at_runtime`. Two public
routes fill them, and they mean the same thing: passing the value to the
process constructor, and naming it under the state in `Model(fixed_params=...)`.

`bind_fixed_process_laws` makes them literally the same object. It runs before
the model structure is prepared, so every consumer downstream — the handoff and
entry-law validation, intrinsic-entry synthesis, the target's own nodes,
diagnostics, and simulation — reads one resolved process rather than a
half-specified one that a later stage would have patched. A law that genuinely
arrives at runtime keeps its `None` fields and stays a runtime parameter.
"""

import dataclasses
import numbers
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from _lcm.grids import Grid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import RegimeName, StateName
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
) -> tuple[MappingProxyType[RegimeName, UserRegime], UserParams]:
    """Bake `fixed_params` entries naming a process's law into that process.

    Args:
        user_regimes: Mapping of regime names to regimes whose states may hold
            half-specified stochastic processes.
        fixed_params: Parameters fixed at model initialization, in the
            user-facing nested form.

    Returns:
        Tuple of the regimes with every bindable process law baked in, and the
        `fixed_params` entries that were not consumed.

    """
    if not fixed_params:
        return MappingProxyType(dict(user_regimes)), fixed_params

    bound_regimes: dict[RegimeName, UserRegime] = {}
    residual: dict[str, Any] = {
        key: value for key, value in fixed_params.items() if key not in user_regimes
    }
    for regime_name, user_regime in user_regimes.items():
        regime_params = fixed_params.get(regime_name)
        if not isinstance(regime_params, Mapping):
            bound_regimes[regime_name] = user_regime
            if regime_params is not None:
                residual[regime_name] = regime_params
            continue
        typed_regime_params = cast("Mapping[str, Any]", regime_params)
        states, consumed = _bind_regime_states(
            states=user_regime.states, regime_params=typed_regime_params
        )
        bound_regimes[regime_name] = (
            user_regime if not consumed else user_regime.replace(states=states)
        )
        remaining: dict[str, Any] = {}
        for name, value in typed_regime_params.items():
            unbound = consumed.get(name)
            if unbound is None:
                remaining[name] = value
            elif unbound:
                remaining[name] = unbound
        if remaining:
            residual[regime_name] = remaining
    return MappingProxyType(bound_regimes), MappingProxyType(residual)


def _bind_regime_states(
    *,
    states: Mapping[StateName, StateDeclaration],
    regime_params: Mapping[str, Any],
) -> tuple[MappingProxyType[StateName, StateDeclaration], dict[str, dict[str, Any]]]:
    """Bind one regime's process states; report what each state left unbound.

    Returns:
        Tuple of the regime's states with bound processes substituted, and a
        mapping from every state name whose entry was touched to the params of
        that entry that stay runtime parameters.

    """
    bound: dict[StateName, StateDeclaration] = dict(states)
    consumed: dict[str, dict[str, Any]] = {}
    for state_name, declaration in states.items():
        state_params = regime_params.get(state_name)
        if not isinstance(state_params, Mapping):
            continue
        substituted, unbound = _bind_declaration(
            declaration=declaration, state_params=state_params
        )
        if substituted is declaration:
            continue
        bound[state_name] = substituted
        consumed[state_name] = unbound
    return MappingProxyType(bound), consumed


def _bind_declaration(
    *, declaration: StateDeclaration, state_params: Mapping[str, Any]
) -> tuple[StateDeclaration, dict[str, Any]]:
    """Bind a state declaration, which may be `Phased` over two grids.

    Returns:
        Tuple of the declaration with every bindable process replaced, and the
        params that no process in it could take.

    """
    if isinstance(declaration, Phased):
        # `Phased` members are declared `object`; only a process member can
        # take a law, and every other value comes back unchanged.
        solve, unbound_solve = _bind_declaration(
            declaration=cast("StateDeclaration", declaration.solve),
            state_params=state_params,
        )
        simulate, unbound_simulate = _bind_declaration(
            declaration=cast("StateDeclaration", declaration.simulate),
            state_params=state_params,
        )
        if solve is declaration.solve and simulate is declaration.simulate:
            return declaration, dict(state_params)
        # A param binds for the phase whose member is a process; it stays a
        # runtime param only when neither member could take it.
        unbound = {
            name: value
            for name, value in unbound_solve.items()
            if name in unbound_simulate
        }
        return Phased(solve=solve, simulate=simulate), unbound
    if not isinstance(declaration, _ContinuousStochasticProcess):
        return declaration, dict(state_params)

    bindable = frozenset(declaration.params_to_pass_at_runtime)
    to_bind = {
        name: value
        for name, value in state_params.items()
        if name in bindable and isinstance(value, numbers.Real)
    }
    if not to_bind:
        return declaration, dict(state_params)
    unbound = {
        name: value for name, value in state_params.items() if name not in to_bind
    }
    return dataclasses.replace(declaration, **to_bind), unbound
