"""Generate function that compute the next states for solution and simulation."""

from collections.abc import Callable
from types import MappingProxyType

import jax
from dags import concatenate_functions, with_signature
from dags.tree import qname_from_tree_path

from _lcm.engine import Variables
from _lcm.grids import Grid
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.processes.ar1 import _AR1Process
from _lcm.processes.iid import _IIDProcess
from _lcm.typing import (
    EconFunctionsMapping,
    NextStateSimulationFunction,
    ProcessName,
    RegimeName,
    StateName,
    StateOrActionName,
    StochasticNextFunction,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.typing import ContinuousState, DiscreteState, FloatND, IntND


def get_next_state_function_for_solution(
    *,
    transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
    functions: EconFunctionsMapping,
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
    functions: EconFunctionsMapping,
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
        variables: States and actions of the regime with kind/topology/process tags.
        stochastic_transition_names: Frozenset of stochastic transition function names.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the regime parameters ("params"). The function also
        depends on the dictionary of random keys ("keys") for stochastic transitions.
        Returns `{target_regime_name: {next_<state>: array}}`.

    """
    per_target_funcs: dict[RegimeName, Callable[..., dict[str, FloatND | IntND]]] = {}
    for target_regime_name, bundle in transitions.items():
        extended = _extend_bundle_for_simulation(
            target_regime_name=target_regime_name,
            bundle=bundle,
            all_grids=all_grids,
            variables=variables,
            stochastic_transition_names=stochastic_transition_names,
        )
        per_target_funcs[target_regime_name] = concatenate_functions(
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
    functions: EconFunctionsMapping,
    transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
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
    # A stochastic weight law may read another transition's *deterministic*
    # `next_<state>` output within the same target's DAG -- the supported
    # transition-reads-transition composition that the solution next-state builder
    # (`get_next_state_function_for_solution`) already relies on. Those producers live
    # in `transitions`, not `functions`, so include the deterministic transitions in
    # the weight DAG; otherwise the read is left as an unsupplied argument and the Q
    # build fails with a missing input (round-12 F2).
    # Stochastic stubs are excluded on purpose: they are the realised draws, not
    # closed-form producers, and a weight depending on another stochastic next-state
    # would need a conditional joint kernel the product-of-marginals form cannot
    # represent -- leaving it unresolved surfaces that unsupported composition loudly.
    deterministic_transitions = {
        name: func
        for name, func in transitions.items()
        if name not in stochastic_transition_names
    }
    return concatenate_functions(
        functions=dict(deterministic_transitions) | dict(functions),
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def _extend_bundle_for_simulation(
    *,
    target_regime_name: RegimeName,
    bundle: MappingProxyType[TransitionFunctionName, Callable[..., FloatND | IntND]],
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
        target_regime_name: Target regime name.
        bundle: Mapping of unqualified `next_<state>` transition names
            to functions, restricted to one target regime.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        variables: States and actions of the current regime with
            kind/topology/process tags.
        stochastic_transition_names: Frozenset of stochastic transition function names.

    Returns:
        Extended transitions dictionary keyed by unqualified `next_<state>` names.

    """
    process_names: frozenset[ProcessName] = frozenset(variables.process_names)
    extended: dict[TransitionFunctionName, Callable[..., FloatND | IntND]] = dict(
        bundle
    )
    for next_state_name in bundle:
        if next_state_name not in stochastic_transition_names:
            continue
        state_name = next_state_name.removeprefix("next_")
        if state_name in process_names:
            extended[next_state_name] = _create_continuous_stochastic_next_func(
                target_regime_name=target_regime_name,
                next_state_name=next_state_name,
                all_grids=all_grids,
            )
        else:
            extended[next_state_name] = _create_discrete_stochastic_next_func(
                target_regime_name=target_regime_name,
                next_state_name=next_state_name,
                labels=all_grids[target_regime_name][state_name].to_jax(),
            )
    return extended


def _create_discrete_stochastic_next_func(
    *,
    target_regime_name: RegimeName,
    next_state_name: TransitionFunctionName,
    labels: DiscreteState,
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    Args:
        target_regime_name: Target regime name.
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
    qname = qname_from_tree_path((target_regime_name, next_state_name))

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
    target_regime_name: RegimeName,
    next_state_name: TransitionFunctionName,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    For processes whose params are supplied at runtime, the runtime params are
    accepted as additional keyword arguments and merged with fixed process
    params before calling the process calculation function.

    Args:
        target_regime_name: Target regime name.
        next_state_name: Transition function name with the `next_` prefix
            (e.g. `next_<process>`).
        all_grids: Immutable mapping of regime names to Grid spec objects.

    Returns:
        A function that simulates the next state of the stochastic variable.

    """
    state_name = next_state_name.removeprefix("next_")
    grid: _ContinuousStochasticProcess = all_grids[target_regime_name][state_name]  # ty: ignore [invalid-assignment]
    qname = qname_from_tree_path((target_regime_name, next_state_name))

    if isinstance(grid, _AR1Process):
        return _create_ar1_next_func(qname=qname, state_name=state_name, grid=grid)
    if isinstance(grid, _IIDProcess):
        return _create_iid_next_func(qname=qname, state_name=state_name, grid=grid)

    msg = f"Expected _IIDProcess or _AR1Process, got {type(grid)}"
    raise TypeError(msg)


def _create_ar1_next_func(
    *, qname: str, state_name: StateName, grid: _AR1Process
) -> StochasticNextFunction:
    fixed_params = dict(grid.params)
    runtime_param_names = {
        qname_from_tree_path((state_name, p)): p for p in grid.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{qname}": "PRNGKeyND",
        state_name: "ContinuousState",
        **dict.fromkeys(runtime_param_names, "FloatND"),
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
            key=kwargs[f"key_{qname}"],
            current_value=kwargs[state_name],
        )

    return next_stochastic_state


def _create_iid_next_func(
    *, qname: str, state_name: StateName, grid: _IIDProcess
) -> StochasticNextFunction:
    fixed_params = dict(grid.params)
    runtime_param_names = {
        qname_from_tree_path((state_name, p)): p for p in grid.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{qname}": "PRNGKeyND",
        **dict.fromkeys(runtime_param_names, "FloatND"),
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
            key=kwargs[f"key_{qname}"],
        )

    return next_stochastic_state
