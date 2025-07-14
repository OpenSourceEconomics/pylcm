from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
from jax import Array

from lcm.argmax import argmax_and_max
from lcm.typing import (
    ArgmaxQcOverDFunction,
    MaxQcOverDFunction,
    ParamsDict,
    ShockType,
)

if TYPE_CHECKING:
    import pandas as pd


def get_max_Qc_over_d(
    *,
    random_utility_shock_type: ShockType,
    variable_info: pd.DataFrame,
    is_last_period: bool,
) -> MaxQcOverDFunction:
    r"""Get the function returning the maximum of Qc over discrete actions.

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

    And maximizing over the discrete actions, we get the value function:

    ```{math}
    V(x) = \max_{a^d} Q^{c}(x, a^d).
    ```

    This last step is handled by the function returned here.

    Args:
        random_utility_shock_type: Type of action shock. Currently only Shock.NONE is
            supported. Work for "extreme_value" is in progress.
        variable_info: DataFrame with information about the variables.
        is_last_period: Whether the function is created for the last period.

    Returns:
        Function that returns the argument that maximize the Qc-function over the
        discrete actions. The maximizing argument corresponds to the policy function of
        the discrete actions.

    """
    if is_last_period:
        variable_info = variable_info.query("~is_auxiliary")

    discrete_action_axes = _determine_discrete_action_axes_solution(variable_info)

    if random_utility_shock_type == ShockType.NONE:
        func = _max_Qc_over_d_no_shocks
    elif random_utility_shock_type == ShockType.EXTREME_VALUE:
        raise NotImplementedError("Extreme value shocks are not yet implemented.")
    else:
        raise ValueError(f"Invalid shock_type: {random_utility_shock_type}.")

    return partial(func, discrete_action_axes=discrete_action_axes)


def get_argmax_and_max_Qc_over_d(
    *,
    variable_info: pd.DataFrame,
) -> ArgmaxQcOverDFunction:
    r"""Get the function returning the arguments maximizing Qc over discrete actions.

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

    Taking the argmax over the discrete actions, we get the policy function of the
    discrete actions:

    ```{math}
    \pi^{d}(x) = \argmax_{a^d} Q^{c}(x, a^d).
    ```

    This last step is handled by the function returned here.

    Args:
        variable_info: DataFrame with information about the variables.

    Returns:
        Function that returns the arguments that maximize the Qc-function over the
        discrete actions and the maximum itself, i.e., policy function of the discrete
        actions. The maximum corresponds to the value function.

    """
    discrete_action_axes = _determine_discrete_action_axes_simulation(variable_info)

    def argmax_and_max_Qc_over_d(
        Qc_arr: Array,
        discrete_action_axes: tuple[int, ...],
        params: ParamsDict,  # noqa: ARG001
    ) -> tuple[Array, Array]:
        return argmax_and_max(Qc_arr, axis=discrete_action_axes)

    return partial(argmax_and_max_Qc_over_d, discrete_action_axes=discrete_action_axes)


# ======================================================================================
# Discrete problem with no shocks
# ======================================================================================


def _max_Qc_over_d_no_shocks(
    Qc_arr: Array,
    discrete_action_axes: tuple[int, ...],
    params: ParamsDict,  # noqa: ARG001
) -> Array:
    """Take the maximum of the Qc-function over the discrete actions.

    Args:
        Qc_arr: The maximum of the state-action value function (Q) over the continuous
            actions, conditional on the discrete action. This has one axis for each
            state and discrete action variable.
        discrete_action_axes: Tuple of indices representing the axes in the value
            function that correspond to discrete actions.
        params: See `get_solve_discrete_problem`.

    Returns:
        The maximum of Qc_arr over the discrete action axes.

    """
    return Qc_arr.max(axis=discrete_action_axes)


# ======================================================================================
# Discrete problem with extreme value shocks
# --------------------------------------------------------------------------------------
# The following is currently *NOT* supported.
# ======================================================================================


def _max_Qc_over_d_extreme_value_shocks(
    Qc_arr: Array, discrete_action_axes: tuple[int, ...], params: ParamsDict
) -> Array:
    """Take the expected maximum of the Qc-function over the discrete actions.

    Args:
        Qc_arr: The maximum of the state-action value function (Q) over the continuous
            actions, conditional on the discrete action. This has one axis for each
            state and discrete action variable.
        discrete_action_axes: Tuple of indices representing the axes in the value
            function that correspond to discrete actions.
        params: See `get_solve_discrete_problem`.

    Returns:
        The expected maximum of Qc_arr over the discrete action axes.

    """
    scale = params["additive_utility_shock"]["scale"]
    return scale * jax.scipy.special.logsumexp(
        Qc_arr / scale, axis=discrete_action_axes
    )


# ======================================================================================
# Auxiliary functions
# ======================================================================================


def _determine_discrete_action_axes_solution(
    variable_info: pd.DataFrame,
) -> tuple[int, ...]:
    """Get axes of state-action-space that correspond to discrete actions in solution.

    Args:
        variable_info: DataFrame with information about the variables.

    Returns:
        A tuple of indices representing the axes' positions in the value function that
        correspond to discrete actions.

    """
    discrete_action_vars = set(
        variable_info.query("is_action & is_discrete").index.tolist()
    )
    return tuple(
        i for i, ax in enumerate(variable_info.index) if ax in discrete_action_vars
    )


def _determine_discrete_action_axes_simulation(
    variable_info: pd.DataFrame,
) -> tuple[int, ...]:
    """Get axes of state-action-space that correspond to discrete actions in simulation.

    Args:
        variable_info: DataFrame with information about the variables.

    Returns:
        A tuple of indices representing the axes' positions in the value function that
        correspond to discrete actions.

    """
    discrete_action_vars = set(
        variable_info.query("is_action & is_discrete").index.tolist()
    )

    # The first dimension corresponds to the simulated states, so add 1.
    return tuple(1 + i for i in range(len(discrete_action_vars)))
