import inspect
from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
from dags.signature import with_signature
from jax import Array

from lcm.argmax import argmax_and_max
from lcm.dispatchers import productmap
from lcm.typing import (
    ArgmaxQOverAFunction,
    BoolND,
    FloatND,
    IntND,
    MaxQOverAFunction,
    RegimeName,
)


def get_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    actions_names: tuple[str, ...],
    states_names: tuple[str, ...],
) -> MaxQOverAFunction:
    r"""Get the function returning the maximum of Q over all actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  H(U(x, a), \mathbb{E}[V(x', a') | x, a]),
    ```
    with $H(U, v) = u + \beta \cdot v$ as the leading case (which is the only one that
    is pre-implemented in LCM).

    Fixing a state, maximizing over all feasible actions,
    we get the $V$ function:

    ```{math}
    V(x) = \max_{a} Q(x, a).
    ```

    This last step is handled by the function returned here.

    Args:
        Q_and_F: A function that takes a state-action combination and returns the action
            value of that combination and whether the state-action combination is
            feasible.
        actions_names: Tuple of action variable names.
        states_names: Tuple of state names.

    Returns:
        V, i.e., the function that calculates the maximum of the Q-function over all
        feasible actions.

    """
    # Extract extra param names from Q_and_F's signature (flat regime params)
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, actions_names=actions_names, states_names=states_names
    )

    Q_and_F = productmap(
        func=Q_and_F,
        variables=actions_names,
    )

    @with_signature(
        args=["next_V_arr", *actions_names, *states_names, *extra_param_names],
        return_annotation="FloatND",
        enforce=False,
    )
    def max_Q_over_a(
        next_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: Array,
    ) -> FloatND:
        Q_arr, F_arr = Q_and_F(
            next_V_arr=next_V_arr,
            **states_actions_params,
        )
        return Q_arr.max(where=F_arr, initial=-jnp.inf)

    return productmap(func=max_Q_over_a, variables=states_names)


def get_argmax_and_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    actions_names: tuple[str, ...],
    states_names: tuple[str, ...],
) -> ArgmaxQOverAFunction:
    r"""Get the function returning the arguments maximizing Q over all actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  H(U(x, a), \mathbb{E}[V(x', a') | x, a]),
    ```
    with $H(U, v) = u + \beta \cdot v$ as the leading case (which is the only one that
    is pre-implemented in LCM).

    Fixing a state but choosing the feasible actions that maximize Q, we get the optimal
    policy

    ```{math}
    \pi(x) = \argmax_{a} Q(x, a).
    ```

    This last step is handled by the function returned here.

    Args:
        Q_and_F: A function that takes a state-action combination and returns the action
            value of that combination and whether the state-action combination is
            feasible.
        actions_names: Tuple of action variable names.
        states_names: Tuple of state names.

    Returns:
        Function that calculates the argument maximizing Q over the feasible continuous
        actions and the maximum iteself. The argument maximizing Q is the policy
        function of the continuous actions, conditional on the states and discrete
        actions. The maximum corresponds to the Qc-function.

    """
    # Extract extra param names from Q_and_F's signature (flat regime params)
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, actions_names=actions_names, states_names=states_names
    )

    Q_and_F = productmap(
        func=Q_and_F,
        variables=actions_names,
    )

    @with_signature(
        args=[
            "next_V_arr",
            *actions_names,
            *states_names,
            *extra_param_names,
        ],
        return_annotation="tuple[IntND, FloatND]",
        enforce=False,
    )
    def argmax_and_max_Q_over_a(
        next_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: Array,
    ) -> tuple[IntND, FloatND]:
        Q_arr, F_arr = Q_and_F(
            next_V_arr=next_V_arr,
            **states_actions_params,
        )
        return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)

    return argmax_and_max_Q_over_a


def _get_extra_param_names(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    actions_names: tuple[str, ...],
    states_names: tuple[str, ...],
) -> list[str]:
    """Get param names from Q_and_F that are not actions, states, or next_V_arr."""
    sig = inspect.signature(Q_and_F)
    known_names = {"next_V_arr", *actions_names, *states_names}
    return sorted(
        name
        for name, param in sig.parameters.items()
        if name not in known_names
        and param.kind
        not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    )
