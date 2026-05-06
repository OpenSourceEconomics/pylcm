"""Generate function that compute the next states for solution and simulation."""

import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType

import jax
import pandas as pd
from dags import concatenate_functions, with_signature
from dags.tree import qname_from_tree_path
from jax import Array

from lcm.grids import Grid
from lcm.shocks import _ShockGrid
from lcm.shocks.ar1 import _ShockGridAR1
from lcm.shocks.iid import _ShockGridIID
from lcm.typing import (
    ContinuousState,
    DiscreteState,
    FloatND,
    FunctionsMapping,
    NextStateSimulationFunction,
    RegimeName,
    ShockName,
    StateName,
    StateOrActionName,
    StochasticNextFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.utils.namespace import flatten_regime_namespace


def get_next_state_function_for_solution(
    *,
    transitions: FunctionsMapping,
    functions: FunctionsMapping,
) -> NextStateSimulationFunction:
    """Get function that computes the next states during the solution.

    Args:
        transitions: Transitions to the next states of a regime.
        functions: Immutable mapping of auxiliary functions of a regime.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the regime parameters (as flat kwargs). If target
        is "simulate", the function also depends on the dictionary of random keys
        ("keys"), which corresponds to the names of stochastic next functions.

    """
    functions_to_concatenate = dict(transitions) | dict(functions)

    return concatenate_functions(
        functions=functions_to_concatenate,
        targets=list(transitions.keys()),
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def get_next_state_function_for_simulation(
    *,
    transitions: TransitionFunctionsMapping,
    functions: FunctionsMapping,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    variable_info: pd.DataFrame,
    stochastic_transition_names: frozenset[TransitionFunctionName] = frozenset(),
) -> NextStateSimulationFunction:
    """Get function that computes the next states during the simulation.

    Builds one DAG per target regime using unqualified `next_<state>` keys, mirroring
    the per-target structure of {func}`get_next_state_function_for_solution`. This
    lets a transition function or auxiliary regime function consume another
    transition's `next_<state>` output via plain name resolution within the same
    target's DAG. Per-target outputs are then merged into a single flat dict keyed
    by `<target>__next_<state>`, matching the shape consumed downstream by
    {func}`lcm.simulation.transitions._update_states_for_subjects`.

    Stochastic-transition wrappers expose `key_<target>__next_<state>` and
    `weight_<target>__next_<state>` as external arguments so callers can pass a
    distinct random key and pre-computed weight per target.

    Args:
        transitions: Nested mapping of target regime names to transition functions.
        functions: Immutable mapping of auxiliary functions of a regime.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        variable_info: Variable info of a regime.
        stochastic_transition_names: Frozenset of stochastic transition function names.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the regime parameters ("params"). The function also
        depends on the dictionary of random keys ("keys") for stochastic transitions.

    """
    per_target_funcs: dict[RegimeName, Callable[..., dict[str, Array]]] = {}
    for target, target_trans in transitions.items():
        extended = _extend_target_transitions_for_simulation(
            target=target,
            target_trans=target_trans,
            all_grids=all_grids,
            variable_info=variable_info,
            stochastic_transition_names=stochastic_transition_names,
        )
        per_target_funcs[target] = concatenate_functions(
            functions=dict(extended) | dict(functions),
            targets=list(extended.keys()),
            return_type="dict",
            enforce_signature=False,
            set_annotations=True,
        )

    return _build_combined_simulation_function(per_target_funcs=per_target_funcs)


def get_next_stochastic_weights_function(
    *,
    regime_name: RegimeName,
    functions: FunctionsMapping,
    transitions: FunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
) -> Callable[..., dict[str, Array]]:
    """Get function that computes the weights for the next stochastic states.

    Args:
        regime_name: Name of the regime that the transitions target.
        functions: Immutable mapping of auxiliary functions of the model.
        transitions: Transitions to the target regime.
        stochastic_transition_names: Frozenset of stochastic transition function names.

    Returns:
        Function that computes the weights for the next stochastic states.

    """
    targets = [
        f"weight_{regime_name}__{func_name}"
        for func_name in transitions
        if func_name in stochastic_transition_names
    ]
    return concatenate_functions(
        functions=functions,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def _extend_target_transitions_for_simulation(
    *,
    target: RegimeName,
    target_trans: MappingProxyType[TransitionFunctionName, Callable[..., Array]],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    variable_info: pd.DataFrame,
    stochastic_transition_names: frozenset[TransitionFunctionName],
) -> dict[TransitionFunctionName, Callable[..., Array]]:
    """Replace stochastic transitions for one target with realisation wrappers.

    Deterministic transitions are passed through unchanged. Stochastic transitions
    are replaced by wrappers that draw a realisation from a precomputed weight
    vector and a random key. The wrapper's external argument names use
    target-qualified form (`key_<target>__<next_state>`,
    `weight_<target>__<next_state>`) so multi-target callers can supply distinct
    random keys per target. The dict key keeps the unqualified `next_<state>` so
    other transitions or regime functions in the same target's DAG can resolve
    it by name.

    Args:
        target: Target regime name.
        target_trans: Mapping of unqualified `next_<state>` transition names to
            functions, restricted to one target regime.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        variable_info: Variable info of the current regime.
        stochastic_transition_names: Frozenset of stochastic transition function names.

    Returns:
        Extended transitions dictionary keyed by unqualified `next_<state>` names.

    """
    shock_names: set[ShockName] = set(variable_info.query("is_shock").index.to_list())
    flat_grids = flatten_regime_namespace(all_grids)
    extended: dict[TransitionFunctionName, Callable[..., Array]] = dict(target_trans)
    for next_state_name in target_trans:
        if next_state_name not in stochastic_transition_names:
            continue
        qname = qname_from_tree_path((target, next_state_name))
        raw_state_name = next_state_name.removeprefix("next_")
        if raw_state_name in shock_names:
            extended[next_state_name] = _create_continuous_stochastic_next_func(
                name=qname, flat_grids=flat_grids
            )
        else:
            extended[next_state_name] = _create_discrete_stochastic_next_func(
                name=qname,
                labels=flat_grids[
                    qname_from_tree_path((target, raw_state_name))
                ].to_jax(),
            )
    return extended


def _build_combined_simulation_function(
    *,
    per_target_funcs: dict[RegimeName, Callable[..., dict[str, Array]]],
) -> NextStateSimulationFunction:
    """Combine per-target simulation DAGs into one function with qualified outputs.

    Each per-target callable returns `{next_<state>: array}` (unqualified). The
    combined callable returns `{<target>__next_<state>: array}` after dispatching
    inputs to the relevant per-target function based on its signature.

    Args:
        per_target_funcs: Mapping of target regime names to per-target simulation
            DAGs returning unqualified `{next_<state>: array}` outputs.

    Returns:
        A single callable that takes the union of all per-target inputs and
        returns target-qualified outputs.

    """
    target_args: dict[RegimeName, tuple[str, ...]] = {
        target: tuple(inspect.signature(func).parameters)
        for target, func in per_target_funcs.items()
    }
    all_args: tuple[str, ...] = tuple(
        sorted({arg for args in target_args.values() for arg in args})
    )

    def _dispatch(kwargs: Mapping[str, Array]) -> dict[str, Array]:
        out: dict[str, Array] = {}
        for target, func in per_target_funcs.items():
            target_kwargs = {arg: kwargs[arg] for arg in target_args[target]}
            target_out = func(**target_kwargs)
            for next_state_name, value in target_out.items():
                out[qname_from_tree_path((target, next_state_name))] = value
        return out

    # Generate a real function with named positional-or-keyword parameters so
    # `vmap_1d` (and any other introspecting caller) sees a faithful signature
    # rather than a `(*args, **kwargs)` shim. Mirrors the strategy used by
    # `dataclasses` and `attrs` to synthesise typed `__init__` methods.
    src = (
        f"def combined({', '.join(all_args)}) -> 'dict[str, Array]':\n"
        f"    return _dispatch({{{', '.join(f'{a!r}: {a}' for a in all_args)}}})\n"
    )
    namespace: dict[str, object] = {"_dispatch": _dispatch}
    exec(src, namespace)  # noqa: S102
    combined = namespace["combined"]
    combined.__annotations__ = {
        **dict.fromkeys(all_args, "Array"),
        "return": "dict[str, Array]",
    }
    return combined  # ty: ignore[invalid-return-type]


def _create_discrete_stochastic_next_func(
    *, name: str, labels: DiscreteState
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    Args:
        name: Name of the stochastic variable.
        labels: 1d array of labels.

    Returns:
        A function that simulates the next state of the stochastic variable. The
        function must be called with keyword arguments:
        - weight_{name}: 2d array of weights. The first dimension corresponds to the
          number of simulation units. The second dimension corresponds to the number of
          grid points (labels).
        - key_{name}: PRNG key for the stochastic next function, e.g. 'next_health'.

    """

    @with_signature(
        args={f"weight_{name}": "FloatND", f"key_{name}": "dict[str, Array]"},
        return_annotation="DiscreteState",
    )
    def next_stochastic_state(**kwargs: FloatND) -> DiscreteState:
        return jax.random.choice(
            key=kwargs[f"key_{name}"],
            a=labels,
            p=kwargs[f"weight_{name}"],
        )

    return next_stochastic_state


def _create_continuous_stochastic_next_func(
    *, name: str, flat_grids: MappingProxyType[str, Grid]
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    For shocks whose params are supplied at runtime, the runtime params are
    accepted as additional keyword arguments and merged with fixed shock_params
    before calling the shock calculation function.

    Args:
        name: Name of the stochastic variable (e.g. `"regime__next_shock"`).
        flat_grids: Flattened immutable mapping of regime-qualified names to Grid spec
            objects.

    Returns:
        A function that simulates the next state of the stochastic variable.

    """
    prev_state_name = name.split("next_")[1]
    flat_key = name.replace("next_", "")
    grid: _ShockGrid = flat_grids[flat_key]  # ty: ignore [invalid-assignment]

    if isinstance(grid, _ShockGridAR1):
        return _create_ar1_next_func(
            name=name, prev_state_name=prev_state_name, grid=grid
        )
    if isinstance(grid, _ShockGridIID):
        return _create_iid_next_func(
            name=name, prev_state_name=prev_state_name, grid=grid
        )

    msg = f"Expected _ShockGridIID or _ShockGridAR1, got {type(grid)}"
    raise TypeError(msg)


def _create_ar1_next_func(
    *, name: str, prev_state_name: StateName, grid: _ShockGridAR1
) -> StochasticNextFunction:
    fixed_params = dict(grid.params)
    runtime_param_names = {
        qname_from_tree_path((prev_state_name, p)): p
        for p in grid.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{name}": "dict[str, Array]",
        prev_state_name: "ContinuousState",
        **dict.fromkeys(runtime_param_names, "float"),
    }
    _draw_shock = grid.draw_shock

    @with_signature(args=args, return_annotation="ContinuousState")
    def next_stochastic_state(**kwargs: FloatND) -> ContinuousState:
        params = MappingProxyType(
            {
                **fixed_params,
                **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
            }
        )
        return _draw_shock(
            params=params,
            key=kwargs[f"key_{name}"],
            current_value=kwargs[prev_state_name],
        )

    return next_stochastic_state


def _create_iid_next_func(
    *, name: str, prev_state_name: StateName, grid: _ShockGridIID
) -> StochasticNextFunction:
    fixed_params = dict(grid.params)
    runtime_param_names = {
        qname_from_tree_path((prev_state_name, p)): p
        for p in grid.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{name}": "dict[str, Array]",
        **dict.fromkeys(runtime_param_names, "float"),
    }
    _draw_shock = grid.draw_shock

    @with_signature(args=args, return_annotation="ContinuousState")
    def next_stochastic_state(**kwargs: FloatND) -> ContinuousState:
        params = MappingProxyType(
            {
                **fixed_params,
                **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
            }
        )
        return _draw_shock(
            params=params,
            key=kwargs[f"key_{name}"],
        )

    return next_stochastic_state
