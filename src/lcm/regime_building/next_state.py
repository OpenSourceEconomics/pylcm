"""Generate function that compute the next states for solution and simulation."""

from collections.abc import Callable
from types import MappingProxyType

import jax
from dags import concatenate_functions, with_signature
from dags.tree import qname_from_tree_path

from lcm.grids import Grid
from lcm.shocks import _ShockGrid
from lcm.shocks._base import _params_to_jax
from lcm.shocks.ar1 import _ShockGridAR1
from lcm.shocks.iid import _ShockGridIID
from lcm.typing import (
    ContinuousState,
    DiscreteState,
    FloatND,
    FunctionsMapping,
    IntND,
    NextStateSimulationFunction,
    RegimeName,
    ShockName,
    StateName,
    StateOrActionName,
    StochasticNextFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.variables import Variables


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
    variables: Variables,
    stochastic_transition_names: frozenset[TransitionFunctionName] = frozenset(),
) -> NextStateSimulationFunction:
    """Get function that computes the next states during the simulation.

    Builds one DAG per target regime using unqualified `next_<state>` keys, mirroring
    the per-target structure of {func}`get_next_state_function_for_solution`. This
    lets a transition function or auxiliary regime function consume another
    transition's `next_<state>` output via plain name resolution within the same
    target's DAG. The combined function returns a nested mapping keyed by target
    regime name, with each inner dict using unqualified `next_<state>` keys.

    Stochastic-transition wrappers expose `key_<target>__next_<state>` and
    `weight_<target>__next_<state>` as external arguments so callers can pass a
    distinct random key and pre-computed weight per target.

    Args:
        transitions: Nested mapping of target regime names to transition functions.
        functions: Immutable mapping of auxiliary functions of a regime.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        variables: States and actions of the regime with kind/topology/shock tags.
        stochastic_transition_names: Frozenset of stochastic transition function names.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the regime parameters ("params"). The function also
        depends on the dictionary of random keys ("keys") for stochastic transitions.
        Returns `{target_regime_name: {next_<state>: array}}`.

    """
    per_target_funcs: dict[RegimeName, Callable[..., dict[str, FloatND | IntND]]] = {}
    for target, target_transitions in transitions.items():
        extended = _extend_target_transitions_for_simulation(
            target=target,
            target_transitions=target_transitions,
            all_grids=all_grids,
            variables=variables,
            stochastic_transition_names=stochastic_transition_names,
        )
        per_target_funcs[target] = concatenate_functions(
            functions=dict(extended) | dict(functions),
            targets=list(extended.keys()),
            return_type="dict",
            enforce_signature=False,
            set_annotations=True,
        )

    return concatenate_functions(
        functions=per_target_funcs,
        targets=list(per_target_funcs.keys()),
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def get_next_stochastic_weights_function(
    *,
    regime_name: RegimeName,
    functions: FunctionsMapping,
    transitions: FunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
) -> Callable[..., dict[str, FloatND | IntND]]:
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
    target_transitions: MappingProxyType[
        TransitionFunctionName, Callable[..., FloatND | IntND]
    ],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    variables: Variables,
    stochastic_transition_names: frozenset[TransitionFunctionName],
) -> dict[TransitionFunctionName, Callable[..., FloatND | IntND]]:
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
        target_transitions: Mapping of unqualified `next_<state>` transition names
            to functions, restricted to one target regime.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        variables: States and actions of the current regime with
            kind/topology/shock tags.
        stochastic_transition_names: Frozenset of stochastic transition function names.

    Returns:
        Extended transitions dictionary keyed by unqualified `next_<state>` names.

    """
    shock_names: frozenset[ShockName] = frozenset(variables.shock_names)
    extended: dict[TransitionFunctionName, Callable[..., FloatND | IntND]] = dict(
        target_transitions
    )
    for next_state_name in target_transitions:
        if next_state_name not in stochastic_transition_names:
            continue
        state_name = next_state_name.removeprefix("next_")
        if state_name in shock_names:
            extended[next_state_name] = _create_continuous_stochastic_next_func(
                target=target,
                next_state_name=next_state_name,
                all_grids=all_grids,
            )
        else:
            extended[next_state_name] = _create_discrete_stochastic_next_func(
                target=target,
                next_state_name=next_state_name,
                labels=all_grids[target][state_name].to_jax(),
            )
    return extended


def _create_discrete_stochastic_next_func(
    *,
    target: RegimeName,
    next_state_name: TransitionFunctionName,
    labels: DiscreteState,
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    Args:
        target: Target regime name.
        next_state_name: Transition function name with the `next_` prefix
            (e.g. `next_health`).
        labels: Category codes the discrete state can take (the DiscreteGrid
            rendered as a 1d JAX array). The simulated realisation is one of
            these, drawn via `jax.random.choice` weighted by `weight_<qname>`.

    Returns:
        A function that simulates the next state of the stochastic variable. The
        function must be called with keyword arguments:
        - weight_{qname}: 2d array of weights. The first dimension corresponds to the
          number of simulation units. The second dimension corresponds to the number of
          grid points (one slot per `labels` entry).
        - key_{qname}: PRNG key for the stochastic next function. `qname` is the
          dags-qualified `<target>__<next_state>`.

    """
    qname = qname_from_tree_path((target, next_state_name))

    @with_signature(
        args={f"weight_{qname}": "FloatND", f"key_{qname}": "PRNGKeyND"},
        return_annotation="DiscreteState",
    )
    def next_stochastic_state(**kwargs: FloatND) -> DiscreteState:
        return jax.random.choice(
            key=kwargs[f"key_{qname}"],
            a=labels,
            p=kwargs[f"weight_{qname}"],
        )

    return next_stochastic_state


def _create_continuous_stochastic_next_func(
    *,
    target: RegimeName,
    next_state_name: TransitionFunctionName,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    For shocks whose params are supplied at runtime, the runtime params are
    accepted as additional keyword arguments and merged with fixed shock_params
    before calling the shock calculation function.

    Args:
        target: Target regime name.
        next_state_name: Transition function name with the `next_` prefix
            (e.g. `next_shock`).
        all_grids: Immutable mapping of regime names to Grid spec objects.

    Returns:
        A function that simulates the next state of the stochastic variable.

    """
    state_name = next_state_name.removeprefix("next_")
    grid: _ShockGrid = all_grids[target][state_name]  # ty: ignore [invalid-assignment]
    qname = qname_from_tree_path((target, next_state_name))

    if isinstance(grid, _ShockGridAR1):
        return _create_ar1_next_func(qname=qname, state_name=state_name, grid=grid)
    if isinstance(grid, _ShockGridIID):
        return _create_iid_next_func(qname=qname, state_name=state_name, grid=grid)

    msg = f"Expected _ShockGridIID or _ShockGridAR1, got {type(grid)}"
    raise TypeError(msg)


def _create_ar1_next_func(
    *, qname: str, state_name: StateName, grid: _ShockGridAR1
) -> StochasticNextFunction:
    fixed_params = dict(grid.params)
    runtime_param_names = {
        qname_from_tree_path((state_name, p)): p for p in grid.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{qname}": "PRNGKeyND",
        state_name: "ContinuousState",
        **dict.fromkeys(runtime_param_names, "float"),
    }
    _draw_shock = grid.draw_shock

    @with_signature(args=args, return_annotation="ContinuousState")
    def next_stochastic_state(**kwargs: FloatND) -> ContinuousState:
        params = _params_to_jax(
            MappingProxyType(
                {
                    **fixed_params,
                    **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
                }
            )
        )
        return _draw_shock(
            params=params,
            key=kwargs[f"key_{qname}"],
            current_value=kwargs[state_name],
        )

    return next_stochastic_state


def _create_iid_next_func(
    *, qname: str, state_name: StateName, grid: _ShockGridIID
) -> StochasticNextFunction:
    fixed_params = dict(grid.params)
    runtime_param_names = {
        qname_from_tree_path((state_name, p)): p for p in grid.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{qname}": "PRNGKeyND",
        **dict.fromkeys(runtime_param_names, "float"),
    }
    _draw_shock = grid.draw_shock

    @with_signature(args=args, return_annotation="ContinuousState")
    def next_stochastic_state(**kwargs: FloatND) -> ContinuousState:
        params = _params_to_jax(
            MappingProxyType(
                {
                    **fixed_params,
                    **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
                }
            )
        )
        return _draw_shock(
            params=params,
            key=kwargs[f"key_{qname}"],
        )

    return next_stochastic_state
