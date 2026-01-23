"""Generate function that compute the next states for solution and simulation."""

from collections.abc import Callable
from types import MappingProxyType

import jax
import pandas as pd
from dags import concatenate_functions
from dags.signature import with_signature
from jax import Array

from lcm.grids import Grid
from lcm.input_processing.util import is_stochastic_transition
from lcm.shocks import SHOCK_CALCULATION_FUNCTIONS
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
        functions: Dict of auxiliary functions of a regime.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the regime parameters ("params"). If target is "simulate",
        the function also depends on the dictionary of random keys ("keys"), which
        corresponds to the names of stochastic next functions.

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
    grids: GridsDict,
    gridspecs: MappingProxyType[str, Grid],
    variable_info: pd.DataFrame,
    transitions: MappingProxyType[str, InternalUserFunction],
    functions: MappingProxyType[str, InternalUserFunction],
) -> NextStateSimulationFunction:
    """Get function that computes the next states during the simultion.

    Args:
        grids: Grids of a regime.
        gridspecs: The specifications of the current regimes grids.
        variable_info: Variable info of a regime.
        transitions: Transitions to the next states of a regime.
        functions: Dict of auxiliary functions of a regime.

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
    regime_name: RegimeName,
    functions: MappingProxyType[str, InternalUserFunction],
    transitions: MappingProxyType[str, InternalUserFunction],
) -> Callable[..., dict[str, Array]]:
    """Get function that computes the weights for the next stochastic states.

    Args:
        regime_name: Name of the regime that the transitions target.
        functions: Dict containing the auxiliary functions of the model.
        transitions: Transitions to the target regime.

    Returns:
        Function that computes the weights for the next stochastic states.

    """
    targets = [
        f"weight_{regime_name}__{fn_name}"
        for fn_name, fn in transitions.items()
        if is_stochastic_transition(fn)
    ]
    return concatenate_functions(
        functions=functions,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def _extend_transitions_for_simulation(
    grids: GridsDict,
    gridspecs: MappingProxyType[str, Grid],
    transitions: MappingProxyType[str, InternalUserFunction],
    variable_info: pd.DataFrame,
) -> dict[str, Callable[..., Array]]:
    """Extend the functions dictionary for the simulation target.

    Args:
        grids: Dictionary of grids.
        gridspecs: The specifications of the current regimes grids.
        transitions: A dictonary of transitions to extend.
        variable_info: Variable info of the current regime.

    Returns:
        Extended functions dictionary.

    """
    shock_names = set(variable_info.query("is_shock").index.to_list())
    flat_grids = flatten_regime_namespace(grids)
    discrete_stochastic_targets = [
        fn_name
        for fn_name, fn in transitions.items()
        if is_stochastic_transition(fn)
        and fn_name.split("__")[-1].replace("next_", "") not in shock_names
    ]
    continuous_stochastic_targets = [
        (fn_name, fn)
        for fn_name, fn in transitions.items()
        if is_stochastic_transition(fn)
        and fn_name.split("__")[-1].replace("next_", "") in shock_names
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
        name: _create_continuous_stochastic_next_func(name, gridspecs)
        for name, fn in continuous_stochastic_targets
    }

    # Overwrite regime transitions with generated stochastic next states functions
    # ----------------------------------------------------------------------------------
    return dict(transitions) | discrete_stochastic_next | continuous_stochastic_next


def _create_discrete_stochastic_next_func(
    name: str, labels: DiscreteState
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
    name: str, gridspecs: MappingProxyType[str, Grid]
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    Args:
        name: Name of the stochastic variable.
        gridspecs: The specifications of the current regimes grids.

    Returns:
        A function that simulates the next state of the stochastic variable. The
        function must be called with keyword arguments:
        - weight_{name}: 2d array of weights. The first dimension corresponds to the
          number of simulation units. The second dimension corresponds to the number of
          grid points (labels).
        - key_{name}: PRNG key for the stochastic next function, e.g. 'next_health'.

    """
    prev_state_name = name.split("next_")[1]
    distribution_type = gridspecs[prev_state_name].distribution_type  # ty: ignore[unresolved-attribute]
    args = {
        "params": "ParamsDict",
        f"key_{name}": "dict[str, Array]",
        prev_state_name: "Array",
    }

    @with_signature(
        args=args,
        return_annotation="ContinuousState",
    )
    def next_stochastic_state(
        **kwargs: FloatND,
    ) -> ContinuousState:
        return SHOCK_CALCULATION_FUNCTIONS[distribution_type](
            params=gridspecs[prev_state_name].shock_params,  # ty: ignore[unresolved-attribute]
            key=kwargs[f"key_{name}"],
            prev_value=kwargs[prev_state_name],
        )

    return next_stochastic_state
