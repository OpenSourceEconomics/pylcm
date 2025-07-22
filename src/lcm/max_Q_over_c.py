import functools
from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from lcm.argmax import argmax_and_max
from lcm.dispatchers import productmap
from lcm.typing import ArgmaxQOverCFunction, MaxQOverCFunction, ParamsDict, Scalar


def get_max_Q_over_c(
    Q_and_F: Callable[..., tuple[Array, Array]],
    continuous_actions_names: tuple[str, ...],
    states_and_discrete_actions_names: tuple[str, ...],
) -> MaxQOverCFunction:
    r"""Get the function returning the maximum of Q over continuous actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  H(U(x, a), \mathbb{E}[V(x', a') | x, a]),
    ```
    with $H(U, v) = u + \beta \cdot v$ as the leading case (which is the only one that
    is pre-implemented in LCM).

    Fixing a state and discrete action, maximizing over the feasible continuous actions,
    we get the $Q^c$ function:

    ```{math}
    Q^{c}(x, a^d) = \max_{a^c} Q(x, a^d, a^c).
    ```

    This last step is handled by the function returned here.

    Args:
        Q_and_F: A function that takes a state-action combination and returns the action
            value of that combination and whether the state-action combination is
            feasible.
        continuous_actions_names: Tuple of action variable names that are continuous.
        states_and_discrete_actions_names: Tuple of state and discrete action variable
            names.

    Returns:
        Qc, i.e., the function that calculates the maximum of the Q-function over the
        feasible continuous actions.

    """
    if continuous_actions_names:
        Q_and_F = productmap(
            func=Q_and_F,
            variables=continuous_actions_names,
        )

    @functools.wraps(Q_and_F)
    def max_Q_over_c(next_V_arr: Array, params: ParamsDict, **kwargs: Scalar) -> Array:
        Q_arr, F_arr = Q_and_F(params=params, next_V_arr=next_V_arr, **kwargs)
        return Q_arr.max(where=F_arr, initial=-jnp.inf)

    return productmap(max_Q_over_c, variables=states_and_discrete_actions_names)


def get_argmax_and_max_Q_over_c(
    Q_and_F: Callable[..., tuple[Array, Array]],
    continuous_actions_names: tuple[str, ...],
) -> ArgmaxQOverCFunction:
    r"""Get the function returning the arguments maximizing Q over continuous actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  H(U(x, a), \mathbb{E}[V(x', a') | x, a]),
    ```
    with $H(U, v) = u + \beta \cdot v$ as the leading case (which is the only one that
    is pre-implemented in LCM).

    Fixing a state and discrete action but choosing the feasible continuous actions that
    maximizes Q, we get

    ```{math}
    \pi^{c}(x, a^d) = \argmax_{a^c} Q(x, a^d, a^c).
    ```

    This last step is handled by the function returned here.

    Args:
        Q_and_F: A function that takes a state-action combination and returns the action
            value of that combination and whether the state-action combination is
            feasible.
        continuous_actions_names: Tuple of action variable names that are continuous.

    Returns:
        Function that calculates the argument maximizing Q over the feasible continuous
        actions and the maximum iteself. The argument maximizing Q is the policy
        function of the continuous actions, conditional on the states and discrete
        actions. The maximum corresponds to the Qc-function.

    """
    if continuous_actions_names:
        Q_and_F = productmap(
            func=Q_and_F,
            variables=continuous_actions_names,
        )

    @functools.wraps(Q_and_F)
    def argmax_and_max_Q_over_c(
        next_V_arr: Array, params: ParamsDict, **kwargs: Scalar
    ) -> tuple[Array, Array]:
        Q_arr, F_arr = Q_and_F(params=params, next_V_arr=next_V_arr, **kwargs)
        return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)

    return argmax_and_max_Q_over_c
