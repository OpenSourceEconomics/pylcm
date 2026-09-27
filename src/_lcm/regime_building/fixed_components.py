"""Lower fixed-component Markov states consistently across all model carriers.

Every original code is represented by its group and position within that group.
The original state name remains a DAG function, while initial observations retain
original codes or labels until the simulation input boundary.
"""

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, make_dataclass
from types import MappingProxyType
from typing import cast, no_type_check

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from _lcm.grids import DiscreteGrid
from _lcm.grids.categorical import categorical
from _lcm.identity_transition import _IdentityTransition
from _lcm.regime_building.broadcast import merge_model_slots
from _lcm.simulation.initial_conditions import MISSING_CAT_CODE
from _lcm.typing import RegimeNamesToIds
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.regime import Regime
from lcm.transition import MarkovTransition, fixed_transition
from lcm.typing import (
    DiscreteState,
    FloatND,
    IntND,
    ScalarInt,
    UserInitialConditions,
    UserParams,
)


@dataclass(frozen=True)
class FixedComponentSplit:
    """A code bijection and the original observation domains of its carriers."""

    grids: Mapping[str, DiscreteGrid]
    """Original simulation grids by regime, before broadcast pruning."""

    rest_grid: DiscreteGrid
    """Grid of positions within each group."""

    fixed_grid: DiscreteGrid
    """Grid of groups."""

    rest_of_code: tuple[int, ...]
    """Position of each original code within its group."""

    fixed_of_code: tuple[int, ...]
    """Group of each original code."""


def split_initial_conditions(
    *,
    initial_conditions: UserInitialConditions | pd.DataFrame,
    splits: Mapping[str, FixedComponentSplit],
    user_regimes: Mapping[str, Regime],
    regime_names_to_ids: RegimeNamesToIds,
) -> UserInitialConditions | pd.DataFrame:
    """Split active observations on the host before canonical numeric uploads."""
    present = [name for name in splits if name in initial_conditions]
    if not present:
        return initial_conditions
    is_frame = isinstance(initial_conditions, pd.DataFrame)
    if isinstance(initial_conditions, pd.DataFrame):
        if "regime_name" not in initial_conditions:
            raise ValueError("DataFrame must contain a 'regime_name' column.")
        regime_column = initial_conditions["regime_name"].to_numpy()
        invalid = set(regime_column) - regime_names_to_ids.keys()
        if invalid:
            raise ValueError(
                f"Invalid regime names in 'regime_name' column: {sorted(invalid)}."
            )
        out = initial_conditions.copy()
    else:
        regime_column = np.asarray(initial_conditions["regime_id"])
        out = dict(initial_conditions)
    for name in present:
        split = splits[name]
        active = np.zeros(len(regime_column), dtype=bool)
        code = np.full(len(regime_column), MISSING_CAT_CODE, dtype=np.int32)
        for regime_name, grid in split.grids.items():
            if f"{name}_rest" not in user_regimes[regime_name].states:
                continue
            rows = regime_column == (
                regime_name if is_frame else int(regime_names_to_ids[regime_name])
            )
            active |= rows
            code[rows] = _read_initial_codes(
                initial_conditions=initial_conditions,
                rows=rows,
                name=name,
                grid=grid,
                regime_name=regime_name,
            )
        out.pop(name)
        # A wholly state-absent cohort needs no factor columns; state assembly
        # supplies missing-state sentinels for later entry into a carrier.
        if not active.any():
            continue
        for suffix, table, grid in (
            ("rest", split.rest_of_code, split.rest_grid),
            ("fixed", split.fixed_of_code, split.fixed_grid),
        ):
            factor = (
                np.full(len(code), None, dtype=object)
                if is_frame
                else np.full(len(code), MISSING_CAT_CODE, dtype=np.int32)
            )
            positions = np.asarray(table)[code[active]]
            factor[active] = (
                np.asarray(grid.categories)[positions] if is_frame else positions
            )
            out[f"{name}_{suffix}"] = factor
    return out if isinstance(out, pd.DataFrame) else MappingProxyType(out)


def factor_fixed_components(  # noqa: C901
    *,
    regimes: Mapping[str, Regime],
    fixed_params: UserParams,
    states: Mapping[str, object],
    state_transitions: Mapping[str, object],
    functions: Mapping[str, object],
    constraints: Mapping[str, object],
    actions: Mapping[str, object],
    derived_categoricals: Mapping[str, object],
) -> tuple[
    Mapping[str, Regime],
    UserParams,
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, FixedComponentSplit],
]:
    """Inventory declarations, then lower each grid and law in its original slot."""
    groups = _collect_groups(regimes=regimes, state_transitions=state_transitions)
    if not groups:
        return regimes, fixed_params, states, state_transitions, MappingProxyType({})
    # Resolve masks and the exactly-one-level rule without changing ownership.
    merged, _ = merge_model_slots(
        user_regimes=regimes,
        model_slots={
            "states": states,
            "state_transitions": state_transitions,
            "functions": functions,
            "constraints": constraints,
            "actions": actions,
        },
    )
    occupied = (
        set(states)
        | set(state_transitions)
        | set(functions)
        | set(actions)
        | set(constraints)
        | set(derived_categoricals)
    )
    for regime in regimes.values():
        occupied.update(
            set(regime.states)
            | set(regime.functions)
            | set(regime.actions)
            | set(regime.constraints)
            | set(regime.derived_categoricals)
            | set(regime.state_transitions)
        )
    splits, parts = _create_splits(regimes=merged, groups=groups, occupied=occupied)
    model_states = dict(states)
    model_laws = dict(state_transitions)
    for name, split in splits.items():
        if name in model_states:
            del model_states[name]
            model_states[f"{name}_rest"] = split.rest_grid
        model_states[f"{name}_fixed"] = split.fixed_grid
        if name in model_laws:
            model_laws[f"{name}_rest"] = _lower_law(
                law=model_laws.pop(name),
                name=name,
                split=split,
                code_by_parts=parts[name],
            )
        model_laws[f"{name}_fixed"] = fixed_transition(f"{name}_fixed")
    new_regimes: dict[str, Regime] = {}
    for regime_name, regime in regimes.items():
        regime_states = dict(regime.states)
        laws: dict[str, object] = dict(regime.state_transitions)
        regime_functions = dict(regime.functions)
        for name, split in splits.items():
            if name in regime_states:
                grid = regime_states.pop(name)
                regime_states[f"{name}_rest"] = (
                    None if grid is None else split.rest_grid
                )
            if name in laws:
                laws[f"{name}_rest"] = _lower_law(
                    law=laws.pop(name),
                    name=name,
                    split=split,
                    code_by_parts=parts[name],
                )
            if regime_name in split.grids:
                regime_functions[name] = _recombine(
                    name=name, code_by_parts=parts[name]
                )
        new_regimes[regime_name] = dataclasses.replace(
            regime,
            states=regime_states,
            state_transitions=laws,
            functions=regime_functions,
        )
    return (
        MappingProxyType(new_regimes),
        rename_split_params(params=fixed_params, splits=splits),
        MappingProxyType(model_states),
        MappingProxyType(model_laws),
        MappingProxyType(splits),
    )


def rename_split_params(
    *, params: UserParams, splits: Mapping[str, FixedComponentSplit]
) -> UserParams:
    """Retain original transition parameter names at the public boundary."""
    if not splits:
        return params
    regime_names = {regime for split in splits.values() for regime in split.grids}
    renamed: dict[str, object] = dict(params)
    for regime_name in regime_names & params.keys():
        block = params[regime_name]
        if not isinstance(block, Mapping):
            continue
        regime_params = _rename_law_params(params=block, splits=splits)
        for target in regime_names & block.keys():
            target_params = block[target]
            if isinstance(target_params, Mapping):
                regime_params[target] = _rename_law_params(
                    params=target_params, splits=splits
                )
        renamed[regime_name] = MappingProxyType(regime_params)
    return cast("UserParams", MappingProxyType(renamed))


def _rename_law_params(
    *, params: Mapping[str, object], splits: Mapping[str, FixedComponentSplit]
) -> dict[str, object]:
    """Rename function slots without inspecting their parameter payloads."""
    renamed = dict(params)
    for name in splits:
        key, replacement = f"next_{name}", f"next_{name}_rest"
        if key not in params or not isinstance(params[key], Mapping):
            continue
        if replacement in params:
            raise ValueError(f"Parameters declare both {key!r} and {replacement!r}.")
        renamed[replacement] = renamed.pop(key)
    return renamed


def _collect_groups(
    *, regimes: Mapping[str, Regime], state_transitions: Mapping[str, object]
) -> dict[str, tuple[int, ...]]:
    """Require one declared grouping across all law leaves."""
    groups: dict[str, tuple[int, ...]] = {}
    for laws in (
        state_transitions,
        *(regime.state_transitions for regime in regimes.values()),
    ):
        for name, law in laws.items():
            for leaf in _law_leaves(law):
                if (
                    isinstance(leaf, MarkovTransition)
                    and leaf.fixed_component is not None
                ):
                    previous = groups.setdefault(name, leaf.fixed_component)
                    if previous != leaf.fixed_component:
                        raise RegimeInitializationError(
                            f"MarkovTransition.fixed_component for {name!r} differs "
                            "between declarations; one state needs one grouping."
                        )
    return groups


def _create_splits(
    *,
    regimes: Mapping[str, Regime],
    groups: Mapping[str, tuple[int, ...]],
    occupied: set[str],
) -> tuple[dict[str, FixedComponentSplit], dict[str, np.ndarray]]:
    """Resolve every original carrier before replacing any grid or law."""
    splits: dict[str, FixedComponentSplit] = {}
    parts: dict[str, np.ndarray] = {}
    for name, grouping in groups.items():
        taken = sorted({f"{name}_rest", f"{name}_fixed"} & occupied)
        if taken:
            raise RegimeInitializationError(
                f"Factoring the fixed component of {name!r} needs the names "
                f"{taken}, which the model or a regime already uses."
            )
        grids: dict[str, DiscreteGrid] = {}
        for regime_name, regime in regimes.items():
            if name not in regime.states:
                if name in regime.state_transitions:
                    raise RegimeInitializationError(
                        f"Fixed component of {name!r} cannot preserve an ingress/reset "
                        f"from regime {regime_name!r} without that state."
                    )
                continue
            grid = regime.states[name]
            if not isinstance(grid, DiscreteGrid):
                raise RegimeInitializationError(
                    f"MarkovTransition.fixed_component on {name!r} in regime "
                    f"{regime_name!r} needs a DiscreteGrid state."
                )
            _, table = _group_codes(
                name=name, fixed_component=grouping, n_codes=len(grid.codes)
            )
            grids[regime_name] = grid
            parts[name] = table
        if not grids:
            raise RegimeInitializationError(
                f"Fixed component of {name!r} has no state carrier."
            )
        table = parts[name]
        rest_of_code = np.empty(len(grouping), dtype=np.int32)
        for rest, codes in enumerate(table):
            rest_of_code[codes] = rest
        splits[name] = FixedComponentSplit(
            grids=MappingProxyType(grids),
            rest_grid=DiscreteGrid(_labels(prefix=f"{name}_rest", n=table.shape[0])),
            fixed_grid=DiscreteGrid(_labels(prefix=f"{name}_fixed", n=table.shape[1])),
            rest_of_code=tuple(int(code) for code in rest_of_code),
            fixed_of_code=grouping,
        )
    return splits, parts


def _read_initial_codes(
    *,
    initial_conditions: UserInitialConditions | pd.DataFrame,
    rows: np.ndarray,
    name: str,
    grid: DiscreteGrid,
    regime_name: str,
) -> np.ndarray:
    """Read only meaningful observations using their original regime's domain."""
    if not rows.any():
        return np.empty(0, dtype=np.int32)
    if isinstance(initial_conditions, pd.DataFrame):
        values = initial_conditions.loc[rows, name]
        mapped = values.map(dict(zip(grid.categories, grid.codes, strict=True)))
        if mapped.isna().any():
            bad = sorted(set(values.loc[mapped.isna()].astype(str)))
            raise ValueError(
                f"Invalid labels for {name!r} in regime {regime_name!r}: {bad}."
            )
        return mapped.to_numpy(dtype=np.int32)
    values = np.asarray(initial_conditions[name])[rows]
    if not np.all(
        np.isfinite(values)
        & (values >= 0)
        & (values < len(grid.codes))
        & (values == np.floor(values))
    ):
        raise ValueError(
            f"Invalid categorical codes for {name!r} in regime {regime_name!r}."
        )
    return values.astype(np.int32)


def _law_leaves(law: object) -> tuple[object, ...]:
    """Flatten declaration containers without changing phase or target ownership."""
    if isinstance(law, Phased):
        return _law_leaves(law.solve) + _law_leaves(law.simulate)
    if isinstance(law, Mapping):
        return tuple(leaf for value in law.values() for leaf in _law_leaves(value))
    return (law,)


def _lower_law(
    *, law: object, name: str, split: FixedComponentSplit, code_by_parts: np.ndarray
) -> object:
    """Restrict every supported leaf, preserving its phase and target containers."""
    if isinstance(law, Phased):
        return Phased(
            solve=_lower_law(
                law=law.solve, name=name, split=split, code_by_parts=code_by_parts
            ),
            simulate=_lower_law(
                law=law.simulate, name=name, split=split, code_by_parts=code_by_parts
            ),
        )
    if isinstance(law, Mapping):
        return MappingProxyType(
            {
                target: _lower_law(
                    law=leaf, name=name, split=split, code_by_parts=code_by_parts
                )
                for target, leaf in law.items()
            }
        )
    if law is None:
        return None
    if isinstance(law, _IdentityTransition):
        return fixed_transition(f"{name}_rest")
    if (
        not isinstance(law, MarkovTransition)
        or law.fixed_component != split.fixed_of_code
    ):
        raise RegimeInitializationError(
            f"Every law of {name!r} must declare the same fixed_component "
            "or use fixed_transition; an unannotated/reset law cannot "
            "establish group preservation."
        )
    return MarkovTransition(
        _restricted_law(
            func=law.func,
            state_name=name,
            fixed_of_code=np.asarray(split.fixed_of_code),
            code_by_parts=code_by_parts,
        )
    )


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
    table = jnp.asarray(code_by_parts).reshape(-1)
    n_fixed = code_by_parts.shape[1]
    rest, fixed = f"{name}_rest", f"{name}_fixed"

    @no_type_check
    def recombine(**kwargs: DiscreteState) -> DiscreteState:
        index = kwargs[rest] * n_fixed + kwargs[fixed]
        return _gather_codes(table=table, index=index)

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


def _gather_codes(*, table: IntND, index: IntND) -> IntND:
    """Read code-table entries with the index's explicit device layout."""
    # A vector index preserves the explicit gather layout through mapped kernels.
    indices = jnp.expand_dims(index, axis=-1)
    gathered = table.at[indices].get(out_sharding=jax.typeof(indices).sharding)  # noqa: PD008
    return jnp.squeeze(gathered, axis=-1)


def _labels(*, prefix: str, n: int) -> type:
    return categorical(ordered=False)(
        make_dataclass(
            prefix.title().replace("_", ""), [(f"c{i}", ScalarInt) for i in range(n)]
        )
    )
