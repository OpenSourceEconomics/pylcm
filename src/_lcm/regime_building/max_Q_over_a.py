import inspect
from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
from dags import with_signature

from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.inclusive_value import smoothed_choice_probabilities
from _lcm.typing import (
    ActionName,
    ArgmaxQOverAFunction,
    MaxQOverAFunction,
    RegimeName,
    StateName,
)
from _lcm.utils.dispatchers import productmap
from lcm.typing import BoolND, FloatND, IntND


def get_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    batch_sizes: dict[StateName, int],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
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
        batch_sizes: Mapping of state variable names to batch sizes for the outer
            productmap over states. A batch size of 0 means no batching.
        action_names: Tuple of action variable names.
        state_names: Tuple of state names.

    Returns:
        V, i.e., the function that calculates the maximum of the Q-function over all
        feasible actions.

    """
    # Extract extra param names from Q_and_F's signature (flat regime params)
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
    )

    # Actions are the inner optimization axis — batching applies only to the
    # outer state loop.
    Q_and_F = productmap(
        func=Q_and_F,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )

    @with_signature(
        args=["next_regime_to_V_arr", *action_names, *state_names, *extra_param_names],
        return_annotation="FloatND",
        enforce=False,
    )
    def max_Q_over_a(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: FloatND | IntND | BoolND,
    ) -> FloatND:
        Q_arr, F_arr = Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        return Q_arr.max(where=F_arr, initial=-jnp.inf)

    return productmap(func=max_Q_over_a, variables=state_names, batch_sizes=batch_sizes)


def get_argmax_and_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
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
        action_names: Tuple of action variable names.
        state_names: Tuple of state names.

    Returns:
        Function that calculates the argument maximizing Q over the feasible continuous
        actions and the maximum itself. The argument maximizing Q is the policy
        function of the continuous actions, conditional on the states and discrete
        actions. The maximum corresponds to the Qc-function.

    """
    # Extract extra param names from Q_and_F's signature (flat regime params)
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
    )

    Q_and_F = productmap(
        func=Q_and_F,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )

    @with_signature(
        args=[
            "next_regime_to_V_arr",
            *action_names,
            *state_names,
            *extra_param_names,
        ],
        return_annotation="tuple[IntND, FloatND]",
        enforce=False,
    )
    def argmax_and_max_Q_over_a(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: FloatND | IntND | BoolND,
    ) -> tuple[IntND, FloatND]:
        Q_arr, F_arr = Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)

    return argmax_and_max_Q_over_a


def get_smoothed_choice_probs_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    choice_action: ActionName,
    tau: float,
) -> Callable[..., FloatND]:
    r"""Get the function returning smoothed choice probabilities over `choice_action`.

    The opt-in counterpart to `get_argmax_and_max_Q_over_a`: instead of the hard
    `argmax` over the joint action grid it returns the `τ`-smoothed probability of
    each level of `choice_action`, marginalizing the masked joint softmax over the
    other actions (`smoothed_choice_probabilities`). Nothing in the solve/simulate
    path calls it; the smoothing diagnostic builds it on demand from the retained
    `Q_and_F`.

    Args:
        Q_and_F: A function that takes a state-action combination and returns the
            action value of that combination and whether it is feasible.
        action_names: Tuple of action variable names; their order is the action-axis
            order of `Q_arr`.
        state_names: Tuple of state names.
        choice_action: The discrete action whose level probabilities are returned.
        tau: Smoothing temperature; must be strictly positive.

    Returns:
        Function returning the probability vector over the levels of `choice_action`
        for a single state (the subject axis is added by the outer spacemap).

    """
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
    )
    choice_axis = action_names.index(choice_action)

    Q_and_F = productmap(
        func=Q_and_F,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )

    @with_signature(
        args=["next_regime_to_V_arr", *action_names, *state_names, *extra_param_names],
        return_annotation="FloatND",
        enforce=False,
    )
    def smoothed_choice_probs_Q_over_a(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: FloatND | IntND | BoolND,
    ) -> FloatND:
        Q_arr, F_arr = Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        return smoothed_choice_probabilities(
            Q_arr, feasible=F_arr, tau=tau, choice_axis=choice_axis
        )

    return smoothed_choice_probs_Q_over_a


def _get_extra_param_names(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
) -> list[str]:
    """Get param names from Q_and_F not in actions, states, or next_regime_to_V_arr."""
    sig = inspect.signature(Q_and_F)
    known_names = {"next_regime_to_V_arr", *action_names, *state_names}
    return sorted(
        name
        for name, param in sig.parameters.items()
        if name not in known_names
        and param.kind
        not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    )
