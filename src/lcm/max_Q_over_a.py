from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from lcm.argmax import argmax_and_max
from lcm.dispatchers import productmap

if TYPE_CHECKING:
    from collections.abc import Callable

    from lcm.typing import (
        ArgmaxQOverAFunction,
        BoolND,
        FloatND,
        IntND,
        MaxQOverAFunction,
        ParamsDict,
    )


def get_max_Q_over_a(
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
    Q_and_F = productmap(
        func=Q_and_F,
        variables=actions_names,
    )

    @functools.wraps(Q_and_F)
    def max_Q_over_a(
        next_V_arr: FloatND, params: ParamsDict, **states_and_actions: Array
    ) -> FloatND:
        Q_arr, F_arr = Q_and_F(params=params, next_V_arr=next_V_arr, **states_and_actions)
        return Q_arr.max(where=F_arr, initial=-jnp.inf)

    return productmap(max_Q_over_a, variables=states_names)


def get_argmax_and_max_Q_over_a(
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    actions_names: tuple[str, ...],
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

    Returns:
        Function that calculates the argument maximizing Q over the feasible continuous
        actions and the maximum iteself. The argument maximizing Q is the policy
        function of the continuous actions, conditional on the states and discrete
        actions. The maximum corresponds to the Qc-function.

    """
    Q_and_F = productmap(
        func=Q_and_F,
        variables=actions_names,
    )

    @functools.wraps(Q_and_F)
    def argmax_and_max_Q_over_a(
        next_V_arr: FloatND, params: ParamsDict, **states_and_actions: Array
    ) -> tuple[IntND, FloatND]:
        Q_arr, F_arr = Q_and_F(params=params, next_V_arr=next_V_arr, **states_and_actions)
        return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)

    return argmax_and_max_Q_over_a
