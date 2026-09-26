"""Factor an annotated fixed component out of a product-coded Markov state.

A discrete state whose `MarkovTransition` declares `fixed_component` is rewritten,
before model slots are merged, into three pieces of the ordinary vocabulary:

- `<state>_fixed`, a model-level `DiscreteGrid` over the groups with `fixed_transition`,
  so it can be named in `ExecutionConfig.sharded_states`;
- `<state>_rest`, a `DiscreteGrid` over the position within a group, whose Markov law
  is the user's law restricted to the current group;
- a function `<state>` that recombines the two into the original code, so every other
  function, constraint and law keeps reading the code it was written against.

The continuation then gathers next-period values over one group instead of every code:
the identity law turns the group into an index, which is exactly what a hand-split
model gets.
"""

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import make_dataclass
from types import MappingProxyType
from typing import cast, no_type_check

import jax.numpy as jnp
import numpy as np

from _lcm.grids import DiscreteGrid
from _lcm.grids.categorical import categorical
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import DiscreteState, FloatND, ScalarInt, UserParams


def factor_fixed_components(
    *,
    regimes: Mapping[str, Regime],
    fixed_params: UserParams,
    states: Mapping[str, object],
    state_transitions: Mapping[str, object],
) -> tuple[
    Mapping[str, Regime], UserParams, Mapping[str, object], Mapping[str, object]
]:
    """Return regimes, fixed params and model-level states and laws, factored."""
    from lcm.transition import MarkovTransition, fixed_transition  # noqa: PLC0415

    model_states = dict(states)
    model_laws = dict(state_transitions)

    new_regimes = dict(regimes)
    new_fixed: dict[str, object] = dict(fixed_params)
    for regime_name, regime in regimes.items():
        transitions = regime.state_transitions
        annotated = {
            name: (law.func, law.fixed_component)
            for name, law in transitions.items()
            if isinstance(law, MarkovTransition) and law.fixed_component is not None
        }
        if not annotated:
            continue
        regime_states = dict(regime.states)
        laws = dict(transitions)
        functions = dict(regime.functions)
        regime_fixed = dict(
            cast("Mapping[str, object]", new_fixed.get(regime_name, {}))
        )
        for name, (func, fixed_component) in annotated.items():
            grid = regime_states.get(name)
            if not isinstance(grid, DiscreteGrid):
                msg = (
                    f"MarkovTransition.fixed_component on {name!r} in regime "
                    f"{regime_name!r} needs a DiscreteGrid state declared there."
                )
                raise RegimeInitializationError(msg)
            taken = sorted(
                {f"{name}_rest", f"{name}_fixed"}
                & (regime_states.keys() | functions.keys())
                | {f"{name}_rest"} & model_states.keys()
            )
            if taken:
                msg = (
                    f"Factoring the fixed component of {name!r} needs the names "
                    f"{taken}, which regime {regime_name!r} already uses."
                )
                raise RegimeInitializationError(msg)
            fixed_of_code, code_by_parts = _group_codes(
                name=name,
                fixed_component=fixed_component,
                n_codes=len(grid.to_jax()),
            )
            n_rest, n_fixed = code_by_parts.shape
            del regime_states[name], laws[name]
            regime_states[f"{name}_rest"] = DiscreteGrid(
                _labels(prefix=f"{name}_rest", n=n_rest)
            )
            _add_model_fixed_state(
                model_states=model_states,
                model_laws=model_laws,
                name=f"{name}_fixed",
                n_fixed=n_fixed,
                law=fixed_transition(f"{name}_fixed"),
            )
            laws[f"{name}_rest"] = MarkovTransition(
                _restricted_law(
                    func=func,
                    state_name=name,
                    fixed_of_code=fixed_of_code,
                    code_by_parts=code_by_parts,
                )
            )
            functions[name] = _recombine(name=name, code_by_parts=code_by_parts)
            if f"next_{name}" in regime_fixed:
                regime_fixed[f"next_{name}_rest"] = regime_fixed.pop(f"next_{name}")
        new_regimes[regime_name] = dataclasses.replace(
            regime,
            states=MappingProxyType(regime_states),
            state_transitions=MappingProxyType(laws),
            functions=MappingProxyType(functions),
        )
        if regime_name in new_fixed:
            new_fixed[regime_name] = MappingProxyType(regime_fixed)
    return (
        MappingProxyType(new_regimes),
        cast("UserParams", MappingProxyType(new_fixed)),
        MappingProxyType(model_states),
        MappingProxyType(model_laws),
    )


def _add_model_fixed_state(
    *,
    model_states: dict[str, object],
    model_laws: dict[str, object],
    name: str,
    n_fixed: int,
    law: object,
) -> None:
    """Declare the group state once at model level; regimes must agree on its size."""
    existing = model_states.get(name)
    if existing is None:
        model_states[name] = DiscreteGrid(_labels(prefix=name, n=n_fixed))
        model_laws[name] = law
        return
    if not isinstance(existing, DiscreteGrid) or len(existing.to_jax()) != n_fixed:
        msg = (
            f"Factoring a fixed component needs the model-level state {name!r} with "
            f"{n_fixed} groups; the model already declares {name!r} differently."
        )
        raise RegimeInitializationError(msg)


def _group_codes(
    *, name: str, fixed_component: tuple[int, ...], n_codes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map each code to its group, and (position, group) back to the code."""
    fixed_of_code = np.asarray(fixed_component, dtype=np.int32)
    if fixed_of_code.shape != (n_codes,):
        msg = (
            f"MarkovTransition.fixed_component for {name!r} has {fixed_of_code.size} "
            f"entries; the state has {n_codes} codes."
        )
        raise RegimeInitializationError(msg)
    groups, sizes = np.unique(fixed_of_code, return_counts=True)
    if groups.tolist() != list(range(groups.size)) or len(set(sizes.tolist())) != 1:
        msg = (
            f"MarkovTransition.fixed_component for {name!r} must label groups 0..k-1 "
            f"of equal size; found groups {groups.tolist()} with sizes "
            f"{sizes.tolist()}."
        )
        raise RegimeInitializationError(msg)
    code_by_parts = np.empty((int(sizes[0]), groups.size), dtype=np.int32)
    for group in range(groups.size):
        members = np.flatnonzero(fixed_of_code == group)
        code_by_parts[:, group] = members
    return fixed_of_code, code_by_parts


def _restricted_law(
    *,
    func: Callable[..., FloatND],
    state_name: str,
    fixed_of_code: np.ndarray,
    code_by_parts: np.ndarray,
) -> Callable[..., FloatND]:
    """The user's law, read at the targets that share the current code's group."""
    fixed_table = jnp.asarray(fixed_of_code)
    parts_table = jnp.asarray(code_by_parts)

    signature = inspect.signature(func)
    names = tuple(signature.parameters)

    # Generated per model, so the claw must not wrap it: model fingerprinting reads
    # a plain closure, but refuses a beartype wrapper it did not capture at import.
    @no_type_check
    def restricted(*args: object, **kwargs: object) -> FloatND:
        arguments = dict(zip(names, args, strict=False)) | kwargs
        full = func(**arguments)
        return full[..., parts_table[:, fixed_table[arguments[state_name]]]]

    if state_name not in signature.parameters:
        msg = (
            f"MarkovTransition.fixed_component needs the law for {state_name!r} to "
            f"read {state_name!r}, to know the current group."
        )
        raise RegimeInitializationError(msg)
    restricted.__signature__ = signature  # ty: ignore[unresolved-attribute]
    restricted.__annotations__ = dict(getattr(func, "__annotations__", {}))
    restricted.__name__ = f"next_{state_name}_rest"
    return restricted


def _recombine(*, name: str, code_by_parts: np.ndarray) -> Callable[..., DiscreteState]:
    """A DAG function named after the state, returning the original code."""
    table = jnp.asarray(code_by_parts)
    rest, fixed = f"{name}_rest", f"{name}_fixed"

    @no_type_check
    def recombine(**kwargs: DiscreteState) -> DiscreteState:
        return table[kwargs[rest], kwargs[fixed]]

    recombine.__signature__ = inspect.Signature(  # ty: ignore[unresolved-attribute]
        [
            inspect.Parameter(
                rest, inspect.Parameter.KEYWORD_ONLY, annotation=DiscreteState
            ),
            inspect.Parameter(
                fixed, inspect.Parameter.KEYWORD_ONLY, annotation=DiscreteState
            ),
        ],
        return_annotation=DiscreteState,
    )
    recombine.__annotations__ = {
        rest: DiscreteState,
        fixed: DiscreteState,
        "return": DiscreteState,
    }
    recombine.__name__ = name
    return recombine


def _labels(*, prefix: str, n: int) -> type:
    return categorical(ordered=False)(
        make_dataclass(
            prefix.title().replace("_", ""), [(f"c{i}", ScalarInt) for i in range(n)]
        )
    )
