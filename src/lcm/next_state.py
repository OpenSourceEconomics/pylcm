"""Generate function that compute the next states for solution and simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from dags import concatenate_functions
from dags.signature import with_signature

from lcm.input_processing.util import is_stochastic_transition
from lcm.interfaces import Target

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from lcm.typing import (
        DiscreteState,
        FloatND,
        NextStateSimulationFunction,
        StochasticNextFunction,
    )


def get_next_state_function(
    *,
    grids: dict[str, Array],
    transitions,
    functions,
    next_states: tuple[str, ...],
    target: Target,
) -> NextStateSimulationFunction:
    """Get function that computes the next states during the solution.

    Args:
        grids: Grids of a regime.
        internal_functions: Internal functions of a regime.
        next_states: Names of the next states to compute. These states are relevant for
            the next state space.
        target: Whether to generate the function for the solve or simulate target.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the regime parameters ("params"). If target is "simulate",
        the function also depends on the dictionary of random keys ("keys"), which
        corresponds to the names of stochastic next functions.

    """
    if target == Target.SOLVE:
        functions = transitions | functions
    elif target == Target.SIMULATE:
        # For the simulation target, we need to extend the functions dictionary with
        # stochastic next states functions and their weights.
        extended_transitions = _extend_transitions_for_simulation(
            grids=grids, transitions=transitions
        )
        functions = extended_transitions | functions
    else:
        raise ValueError(f"Invalid target: {target}")

    requested_next_states = [
        next_fn_name
        for next_fn_name in transitions
        if next_fn_name.removeprefix("next_") in next_states
    ]

    return concatenate_functions(
        functions=functions,
        targets=requested_next_states,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def get_next_stochastic_weights_function(
    functions,
    next_stochastic_states: tuple[str, ...],
) -> Callable[..., dict[str, Array]]:
    """Get function that computes the weights for the next stochastic states.

    Args:
        internal_functions: Internal functions instance.
        next_stochastic_states: Names of the stochastic states for which to compute the
            weights. These variables are relevant for the next state space.

    Returns:
        Function that computes the weights for the next stochastic states.

    """
    targets = [f"weight_next_{name}" for name in next_stochastic_states]

    return concatenate_functions(
        functions=functions,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def _extend_transitions_for_simulation(
    grids: dict[str, Array],
    transitions,
) -> dict[str, Callable[..., Array]]:
    """Extend the functions dictionary for the simulation target.

    Args:
        grids: Dictionary of grids.
        internal_functions: Internal functions instance.

    Returns:
        Extended functions dictionary.

    """
    stochastic_targets = [
        key for key, next_fn in transitions.items() if is_stochastic_transition(next_fn)
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
    stochastic_next = {
        name: _create_stochastic_next_func(
            name, labels=grids[name.removeprefix("next_")]
        )
        for name in stochastic_targets
    }

    # Overwrite regime transitions with generated stochastic next states functions
    # ----------------------------------------------------------------------------------
    return transitions | stochastic_next


def _create_stochastic_next_func(
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
