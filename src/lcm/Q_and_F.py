from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from dags import concatenate_functions
from dags.dag import DagsWarning
from dags.signature import with_signature

from lcm.dispatchers import productmap
from lcm.function_representation import get_value_function_representation
from lcm.functools import get_union_of_arguments
from lcm.interfaces import Target
from lcm.next_state import get_next_state_function, get_next_stochastic_weights_function
from lcm.interfaces import InternalRegime

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from lcm.interfaces import StateSpaceInfo
    from lcm.typing import (
        BoolND,
        Float1D,
        FloatND,
        InternalUserFunction,
        ParamsDict,
    )


type RegimeName = str


def get_Q_and_F(
    internal_regime: InternalRegime,
    next_state_space_info: dict[RegimeName, StateSpaceInfo],
    period: int,
    n_periods: int,
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Get the state-action (Q) and feasibility (F) function for a given period.

    Args:
        internal_regime: Internal regime instance.
        next_state_space_info: The state space information of the next period.
        period: The current period.
        n_periods: Total number of periods in the model.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for the given period for the given regime.

    """
    is_last_period = period == n_periods - 1

    if is_last_period:
        Q_and_F = get_Q_and_F_terminal(internal_regime, period=period)
    else:
        Q_and_F = get_Q_and_F_non_terminal(
            internal_regime, next_state_space_info=next_state_space_info, period=period
        )

    return Q_and_F


def get_Q_and_F_non_terminal(
    internal_regime: InternalRegime,
    next_state_space_info: dict[RegimeName, StateSpaceInfo],
    period: int,
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    Args:
        internal_regime: Internal regime instance.
        next_state_space_info: The state space information of the next period.
        period: The current period.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    stochastic_variables = internal_regime.variable_info.query(
        "is_stochastic"
    ).index.tolist()
    # As we compute the expecation of the next period's value function, we only need the
    # stochastic variables that are relevant for the next state space.
    next_stochastic_variables: dict[RegimeName, tuple[str]] = {
        rn: tuple(
            set(stochastic_variables) & set(nssi.states_names)
        )
        for rn, nssi in next_state_space_info.items()
    }

    # ----------------------------------------------------------------------------------
    # Generate dynamic functions
    # ----------------------------------------------------------------------------------

    # Function required to calculate instantaneous utility and feasibility
    U_and_F = _get_U_and_F(internal_regime)

    # Functions required to calculate the expected continuation values
    state_transitions: dict[RegimeName, Callable[..., dict[str, Array]]] = {
        rn: get_next_state_function(
            internal_regime=internal_regime,
            next_states=nssi.states_names,
            target_regime=rn,
            target=Target.SOLVE,
        )
        for rn, nssi in next_state_space_info.items()
    }
    next_stochastic_states_weights = get_next_stochastic_weights_function(
        internal_regime, next_stochastic_states=next_stochastic_variables
    )
    joint_weights_from_marginals = _get_joint_weights_function(
        next_stochastic_variables
    )
    _scalar_next_V: dict[RegimeName, Callable[..., Array]] = {
        rn: get_value_function_representation(nssi)
        for rn, nssi in next_state_space_info.items()
    }

    next_V: dict[RegimeName, Callable[..., FloatND]] = {}
    for rn, nssi in next_state_space_info.items():
        next_V[rn] = productmap(
            _scalar_next_V[rn],
            variables=tuple(f"next_{var}" for var in next_stochastic_variables[rn]),
        )

    # ----------------------------------------------------------------------------------
    # Create the state-action value and feasibility function
    # ----------------------------------------------------------------------------------
    _arg_names_of_Q_and_F_relevant_functions = [
        U_and_F,
        *list(state_transitions.values()),
        *list(next_stochastic_states_weights.values()),
    ]
    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        _arg_names_of_Q_and_F_relevant_functions,
        include={"params", "next_V_arr"},
        exclude={"_period"},
    )

    regime_names = list(next_state_space_info.keys())

    regime_transition_probs_fn = internal_regime.regime_transition_probs

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        params: ParamsDict, next_V_arr: FloatND, **states_and_actions: Array
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action value and feasibility for a non-terminal period.

        Args:
            params: The parameters.
            next_V_arr: The next period's value function array.
            **states_and_actions: The current states and actions.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        next_V_expected_arrays: dict[RegimeName, FloatND] = {}

        for target_regime in regime_names:
            # --------------------------------------------------------------------------
            # Calculate the expected continuation values
            # --------------------------------------------------------------------------
            next_states = state_transitions[target_regime](
                **states_and_actions,
                _period=period,
                params=params,
            )

            marginal_next_stochastic_states_weights = next_stochastic_states_weights[target_regime](
                **states_and_actions,
                _period=period,
                params=params,
            )

            joint_next_stochastic_states_weights = joint_weights_from_marginals[target_regime](
                **marginal_next_stochastic_states_weights
            )

            # As we productmap'd the value function over the stochastic variables, the
            # resulting next value function gets a new dimension for each stochastic
            # variable.
            next_V_at_stochastic_states_arr = next_V[target_regime](**next_states, next_V_arr=next_V_arr)

            # We then take the weighted average of the next value function at
            # the stochastic states to get the expected next value function.
            next_V_expected_arr = jnp.average(
                next_V_at_stochastic_states_arr,
                weights=joint_next_stochastic_states_weights,
            )
            next_V_expected_arrays[target_regime] = next_V_expected_arr

        # ------------------------------------------------------------------------------
        # Calculate the instantaneous utility and feasibility
        # ------------------------------------------------------------------------------
        U_arr, F_arr = U_and_F(
            **states_and_actions,
            _period=period,
            params=params,
        )

        regime_transition_probs = regime_transition_probs_fn(
            **states_and_actions, _period=period, params=params
        )

        weighted_V_expected_arrays = [
            regime_transition_probs[rn] * next_V_expected_arrays[rn]
            for rn in regime_names 
        ]

        Q_arr = U_arr + params["beta"] * jnp.sum(weighted_V_expected_arrays, axis=0)

        return Q_arr, F_arr

    return Q_and_F


def get_Q_and_F_terminal(
    internal_regime: InternalRegime,
    period: int,
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Get the state-action (Q) and feasibility (F) function for the terminal period.

    Args:
        internal_regime: Internal regime instance.
        period: The current period.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for the terminal period for the given regime.

    """
    U_and_F = _get_U_and_F(internal_regime)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        [U_and_F],
        # While the terminal period does not depend on the value function array, we
        # include it in the signature, such that we can treat all periods uniformly
        # during the solution and simulation.
        include={"params", "next_V_arr"},
        exclude={"_period"},
    )

    args = dict.fromkeys(arg_names_of_Q_and_F, "Array")
    args["params"] = "ParamsDict"
    args["next_V_arr"] = "FloatND"

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        params: ParamsDict,
        next_V_arr: FloatND,  # noqa: ARG001
        **states_and_actions: Array,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action values and feasibilities for the terminal period.

        Args:
            params: The parameters.
            next_V_arr: The next period's value function array (unused here).
            **states_and_actions: The current states and actions.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        return U_and_F(
            **states_and_actions,
            _period=period,
            params=params,
        )

    return Q_and_F


# ======================================================================================
# Helper functions
# ======================================================================================


def _get_arg_names_of_Q_and_F(
    deps: list[Callable[..., Any]],
    include: set[str] = set(),  # noqa: B006
    exclude: set[str] = set(),  # noqa: B006
) -> list[str]:
    """Get the argument names of the dependencies.

    Args:
        deps: List of dependencies.
        include: Set of argument names to include.
        exclude: Set of argument names to exclude.

    Returns:
        The union of the argument names in deps and include, except for those in
        exclude.

    """
    deps_arg_names = get_union_of_arguments(deps)
    return list(include | deps_arg_names - exclude)


def _get_joint_weights_function(
    stochastic_variables: tuple[str, ...],
) -> Callable[..., FloatND]:
    """Get function that calculates the joint weights.

    This function takes the weights of the individual stochastic variables and
    multiplies them together to get the joint weights on the product space of the
    stochastic variables.

    Args:
        stochastic_variables: List of stochastic variables.

    Returns:
        A function that computes the outer product of the weights of the stochastic
        variables.

    """
    arg_names = [f"weight_next_{var}" for var in stochastic_variables]

    @with_signature(args=arg_names)
    def _outer(**kwargs: Float1D) -> FloatND:
        weights = jnp.array(list(kwargs.values()))
        return jnp.prod(weights)

    return productmap(_outer, variables=tuple(arg_names))


def _get_U_and_F(
    internal_regime: InternalRegime,
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Get the instantaneous utility and feasibility function.

    Note:
    -----
    U may depend on all kinds of other functions (taxes, transfers, ...), which will be
    executed if they matter for the value of U.

    Args:
        internal_regime: Internal regime instance.

    Returns:
        The instantaneous utility and feasibility function.

    """
    functions = {
        "feasibility": _get_feasibility(internal_regime),
        **internal_regime.functions,
    }
    return concatenate_functions(
        functions=functions,
        targets=["utility", "feasibility"],
        enforce_signature=False,
        set_annotations=True,
    )


def _get_feasibility(internal_regime: InternalRegime) -> InternalUserFunction:
    """Create a function that combines all constraint functions into a single one.

    Args:
        internal_regime: Internal regime instance.

    Returns:
        The combined constraint function (feasibility).

    """
    constraints = internal_regime.function_info.query("is_constraint").index.tolist()

    if constraints:
        with warnings.catch_warnings():
            # set annotations does not set the return type when concatenate_functions is
            # called with an aggregator and raises a warning.
            warnings.simplefilter("ignore", category=DagsWarning)
            combined_constraint = concatenate_functions(
                functions=internal_regime.functions,
                targets=constraints,
                aggregator=jnp.logical_and,
                set_annotations=True,
            )
        combined_constraint.__annotations__["return"] = "Feasibility"

    else:

        def combined_constraint() -> bool:
            """Dummy feasibility function that always returns True."""
            return True

    return combined_constraint
