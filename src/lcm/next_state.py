"""Generate function that compute the next states for solution and simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dags import concatenate_functions
from dags.signature import with_signature

from lcm.interfaces import Target
from lcm.random import random_choice

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from lcm.interfaces import InternalModel
    from lcm.typing import (
        ContinuousState,
        DiscreteState,
        FloatND,
        StochasticNextFunction,
    )


def get_next_state_function(
    *,
    internal_model: InternalModel,
    next_states: tuple[str, ...],
    target: Target,
) -> Callable[..., dict[str, DiscreteState | ContinuousState]]:
    """Get function that computes the next states during the solution.

    Args:
        internal_model: Internal model instance.
        next_states: Names of the next states to compute. These states are relevant for
            the next state space.
        target: Whether to generate the function for the solve or simulate target.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the model parameters ("params"). If target is "simulate",
        the function also depends on the dictionary of random keys ("keys"), which
        corresponds to the names of stochastic next functions.

    """
    if target == Target.SOLVE:
        functions_dict = internal_model.functions
    elif target == Target.SIMULATE:
        # For the simulation target, we need to extend the functions dictionary with
        # stochastic next states functions and their weights.
        functions_dict = _extend_functions_dict_for_simulation(internal_model)
    else:
        raise ValueError(f"Invalid target: {target}")

    requested_next_states = [
        next_state
        for next_state in internal_model.function_info.query("is_next").index
        if next_state.replace("next_", "") in next_states
    ]

    return concatenate_functions(
        functions=functions_dict,
        targets=requested_next_states,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def get_next_stochastic_weights_function(
    internal_model: InternalModel,
    next_stochastic_states: tuple[str, ...],
) -> Callable[..., dict[str, Array]]:
    """Get function that computes the weights for the next stochastic states.

    Args:
        internal_model: Internal model instance.
        next_stochastic_states: Names of the stochastic states for which to compute the
            weights. These variables are relevant for the next state space.

    Returns:
        Function that computes the weights for the next stochastic states.

    """
    targets = [f"weight_next_{name}" for name in next_stochastic_states]

    return concatenate_functions(
        functions=internal_model.functions,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def _extend_functions_dict_for_simulation(
    internal_model: InternalModel,
) -> dict[str, Callable[..., Array]]:
    """Extend the functions dictionary for the simulation target.

    Args:
        internal_model: Internal model instance.

    Returns:
        Extended functions dictionary.

    """
    stochastic_targets = internal_model.function_info.query("is_stochastic_next").index

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
            name, labels=internal_model.grids[name.removeprefix("next_")]
        )
        for name in stochastic_targets
    }

    stochastic_weights = {
        f"weight_{name}": internal_model.functions[f"weight_{name}"]
        for name in stochastic_targets
    }

    # Overwrite model.functions with generated stochastic next states functions
    # ----------------------------------------------------------------------------------
    return internal_model.functions | stochastic_next | stochastic_weights


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
        - keys: Dictionary with random key arrays. Dictionary keys correspond to the
          names of stochastic next functions, e.g. 'next_health'.

    """

    @with_signature(
        args={f"weight_{name}": "FloatND", "keys": "dict[str, Array]"},
        return_annotation="DiscreteState",
    )
    def next_stochastic_state(
        keys: dict[str, Array], **kwargs: FloatND
    ) -> DiscreteState:
        return random_choice(
            labels=labels,
            probs=kwargs[f"weight_{name}"],
            key=keys[name],
        )

    return next_stochastic_state
