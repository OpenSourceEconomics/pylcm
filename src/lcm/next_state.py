"""Generate function that compute the next states for solution and simulation."""

from __future__ import annotations

from collections.abc import Callable

from dags import concatenate_functions
from dags.signature import with_signature
from jax import Array

from lcm.interfaces import InternalModel
from lcm.random import random_choice
from lcm.typing import Scalar, StochasticNextFunction, Target


def get_next_state_function(
    model: InternalModel,
    target: Target,
) -> Callable[..., dict[str, Scalar]]:
    """Get function that computes the next states during the solution.

    Args:
        model: Internal model instance.
        target: Whether to generate the function for the solve or simulate target.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the model parameters ("params"). If target is "simulate",
        the function also depends on the dictionary of random keys ("keys"), which
        corresponds to the names of stochastic next functions.

    """
    targets = model.function_info.query("is_next").index.tolist()

    if target == Target.SOLVE:
        functions_dict = model.functions
    elif target == Target.SIMULATE:
        # For the simulation target, we need to extend the functions dictionary with
        # stochastic next states functions and their weights.
        functions_dict = _extend_functions_dict_for_simulation(model)
    else:
        raise ValueError(f"Invalid target: {target}")

    return concatenate_functions(
        functions=functions_dict,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def get_next_stochastic_weights_function(
    model: InternalModel,
) -> Callable[..., dict[str, Array]]:
    """Get function that computes the weights for the next stochastic states.

    Args:
        model: Internal model instance.

    Returns:
        Function that computes the weights for the next stochastic states.

    """
    targets = [
        f"weight_{name}"
        for name in model.function_info.query("is_stochastic_next").index.tolist()
    ]

    return concatenate_functions(
        functions=model.functions,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def _extend_functions_dict_for_simulation(
    model: InternalModel,
) -> dict[str, Callable[..., Scalar]]:
    """Extend the functions dictionary for the simulation target.

    Args:
        model: Internal model instance.

    Returns:
        Extended functions dictionary.

    """
    stochastic_targets = model.function_info.query("is_stochastic_next").index

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
            name, labels=model.grids[name.removeprefix("next_")]
        )
        for name in stochastic_targets
    }

    stochastic_weights = {
        f"weight_{name}": model.functions[f"weight_{name}"]
        for name in stochastic_targets
    }

    # Overwrite model.functions with generated stochastic next states functions
    # ----------------------------------------------------------------------------------
    return model.functions | stochastic_next | stochastic_weights


def _create_stochastic_next_func(name: str, labels: Array) -> StochasticNextFunction:
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
        args={f"weight_{name}": "Array", "keys": "dict[str, Array]"},
        return_annotation="DiscreteState",
    )
    def next_stochastic_state(keys: dict[str, Array], **kwargs: Array) -> Array:
        return random_choice(
            labels=labels,
            probs=kwargs[f"weight_{name}"],
            key=keys[name],
        )

    return next_stochastic_state
