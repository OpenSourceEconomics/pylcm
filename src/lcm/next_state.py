"""Generate function that compute the next states for solution and simulation."""

from collections.abc import Callable
from types import MappingProxyType

import jax
import pandas as pd
from dags import concatenate_functions
from dags.signature import with_signature
from dags.tree import QNAME_DELIMITER
from jax import Array

from lcm.grids import Grid
from lcm.shocks.ar1 import _ShockGridAR1
from lcm.shocks.iid import _ShockGridIID
from lcm.typing import (
    ContinuousState,
    DiscreteState,
    FloatND,
    GridsDict,
    InternalUserFunction,
    NextStateSimulationFunction,
    RegimeName,
    StochasticNextFunction,
)
from lcm.utils import flatten_regime_namespace


def get_next_state_function_for_solution(
    *,
    transitions: MappingProxyType[str, InternalUserFunction],
    functions: MappingProxyType[str, InternalUserFunction],
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
    transitions: MappingProxyType[str, InternalUserFunction],
    functions: MappingProxyType[str, InternalUserFunction],
    grids: GridsDict,
    gridspecs: MappingProxyType[str, Grid],
    variable_info: pd.DataFrame,
    stochastic_transition_names: frozenset[str] = frozenset(),
) -> NextStateSimulationFunction:
    """Get function that computes the next states during the simulation.

    Args:
        grids: Grids of a regime.
        gridspecs: The specifications of the current regimes grids.
        variable_info: Variable info of a regime.
        transitions: Transitions to the next states of a regime.
        functions: Immutable mapping of auxiliary functions of a regime.
        stochastic_transition_names: Frozenset of stochastic transition function names.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the regime parameters ("params"). If target is "simulate",
        the function also depends on the dictionary of random keys ("keys"), which
        corresponds to the names of stochastic next functions.

    """
    # For the simulation target, we need to extend the functions dictionary with
    # stochastic next states functions and their weights.
    extended_transitions = _extend_transitions_for_simulation(
        grids=grids,
        gridspecs=gridspecs,
        transitions=transitions,
        variable_info=variable_info,
        stochastic_transition_names=stochastic_transition_names,
    )
    functions_to_concatenate = extended_transitions | dict(functions)

    return concatenate_functions(
        functions=functions_to_concatenate,
        targets=list(transitions.keys()),
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def get_next_stochastic_weights_function(
    *,
    regime_name: RegimeName,
    functions: MappingProxyType[str, InternalUserFunction],
    transitions: MappingProxyType[str, InternalUserFunction],
    stochastic_transition_names: frozenset[str],
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


def _extend_transitions_for_simulation(
    *,
    grids: GridsDict,
    gridspecs: MappingProxyType[str, Grid],
    transitions: MappingProxyType[str, InternalUserFunction],
    variable_info: pd.DataFrame,
    stochastic_transition_names: frozenset[str],
) -> dict[str, Callable[..., Array]]:
    """Extend the functions dictionary for the simulation target.

    Args:
        grids: Immutable mapping of grids.
        gridspecs: The specifications of the current regimes grids.
        transitions: Immutable mapping of transitions to extend.
        variable_info: Variable info of the current regime.
        stochastic_transition_names: Frozenset of stochastic transition function names.

    Returns:
        Extended functions dictionary.

    """
    shock_names = set(variable_info.query("is_shock").index.to_list())
    flat_grids = flatten_regime_namespace(grids)
    discrete_stochastic_targets = [
        func_name
        for func_name in transitions
        if func_name.split(QNAME_DELIMITER)[-1] in stochastic_transition_names
        and func_name.split(QNAME_DELIMITER)[-1].replace("next_", "") not in shock_names
    ]
    continuous_stochastic_targets = [
        func_name
        for func_name in transitions
        if func_name.split(QNAME_DELIMITER)[-1] in stochastic_transition_names
        and func_name.split(QNAME_DELIMITER)[-1].replace("next_", "") in shock_names
    ]
    # Handle stochastic next states functions
    # ----------------------------------------------------------------------------------
    # We generate stochastic next states functions that simulate the next state given
    # a random key (think of a seed) and the weights corresponding to the labels of the
    # stochastic variable. The weights are computed using the stochastic weight
    # functions, which we add the to functions dict. `dags.concatenate_functions` then
    # generates a function that computes the weights and simulates the next state in
    # one go.
    # ----------------------------------------------------------------------------------
    discrete_stochastic_next = {
        name: _create_discrete_stochastic_next_func(
            name, labels=flat_grids[name.replace("next_", "")]
        )
        for name in discrete_stochastic_targets
    }
    continuous_stochastic_next = {
        name: _create_continuous_stochastic_next_func(name, gridspecs=gridspecs)
        for name in continuous_stochastic_targets
    }

    # Overwrite regime transitions with generated stochastic next states functions
    # ----------------------------------------------------------------------------------
    return dict(transitions) | discrete_stochastic_next | continuous_stochastic_next


def _create_discrete_stochastic_next_func(
    name: str, *, labels: DiscreteState
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
    name: str, *, gridspecs: MappingProxyType[str, Grid]
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    For shocks whose params are supplied at runtime, the runtime params are
    accepted as additional keyword arguments and merged with fixed shock_params
    before calling the shock calculation function.

    Args:
        name: Name of the stochastic variable.
        gridspecs: The specifications of the current regimes grids.

    Returns:
        A function that simulates the next state of the stochastic variable.

    """
    prev_state_name = name.split("next_")[1]
    gridspec = gridspecs[prev_state_name]

    if isinstance(gridspec, _ShockGridAR1):
        return _create_ar1_next_func(name, prev_state_name, gridspec=gridspec)
    if isinstance(gridspec, _ShockGridIID):
        return _create_iid_next_func(name, prev_state_name, gridspec=gridspec)

    msg = f"Expected _ShockGridIID or _ShockGridAR1, got {type(gridspec)}"
    raise TypeError(msg)


def _create_ar1_next_func(
    name: str, prev_state_name: str, *, gridspec: _ShockGridAR1
) -> StochasticNextFunction:
    fixed_params = dict(gridspec.params)
    regime_prefix = name.split(f"{QNAME_DELIMITER}next_", maxsplit=1)[0]
    runtime_param_names = {
        f"{regime_prefix}{QNAME_DELIMITER}{prev_state_name}{QNAME_DELIMITER}{p}": p
        for p in gridspec.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{name}": "dict[str, Array]",
        prev_state_name: "ContinuousState",
        **dict.fromkeys(runtime_param_names, "float"),
    }
    _draw_shock = gridspec.draw_shock

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
    name: str, prev_state_name: str, *, gridspec: _ShockGridIID
) -> StochasticNextFunction:
    fixed_params = dict(gridspec.params)
    regime_prefix = name.split(f"{QNAME_DELIMITER}next_", maxsplit=1)[0]
    runtime_param_names = {
        f"{regime_prefix}{QNAME_DELIMITER}{prev_state_name}{QNAME_DELIMITER}{p}": p
        for p in gridspec.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{name}": "dict[str, Array]",
        **dict.fromkeys(runtime_param_names, "float"),
    }
    _draw_shock = gridspec.draw_shock

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
