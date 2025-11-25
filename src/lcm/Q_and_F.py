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
from lcm.input_processing.util import is_stochastic_transition
from lcm.interfaces import InternalFunctions, Target
from lcm.next_state import get_next_state_function, get_next_stochastic_weights_function

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from lcm import Regime
    from lcm.interfaces import StateSpaceInfo
    from lcm.typing import (
        BoolND,
        Float1D,
        FloatND,
        InternalUserFunction,
        ParamsDict,
        Period,
        QAndFFunction,
        RegimeName,
    )


def get_Q_and_F(
    regime: Regime,
    internal_functions: InternalFunctions,
    next_state_space_infos: dict[str, StateSpaceInfo],
    grids: dict[RegimeName, Any],
    *,
    is_last_period: bool,
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a given period.

    Args:
        regime: Regime object containing all infos about the pre-processed regime.
        internal_functions: Internal functions of the regime.
        next_state_space_infos: The state space information of the next period.
        grids: Dict containing the state frids for all regimes.
        is_last_period: True if this period is the last.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for the given period.

    """
    if is_last_period:
        Q_and_F = get_Q_and_F_terminal(regime, internal_functions=internal_functions)
    else:
        Q_and_F = get_Q_and_F_non_terminal(
            regime,
            internal_functions=internal_functions,
            next_state_space_infos=next_state_space_infos,
            grids=grids,
        )

    return Q_and_F


def get_Q_and_F_non_terminal(
    regime: Regime,
    internal_functions: InternalFunctions,
    next_state_space_infos: dict[str, StateSpaceInfo],
    grids: dict[RegimeName, Any],
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    Args:
        regime: Regime instance.
        internal_functions: Internal functions instance.
        next_state_space_infos: The state space information of the next period.
        grids: Dict containing the state frids for all regimes.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    # ----------------------------------------------------------------------------------
    # Generate dynamic functions
    # ----------------------------------------------------------------------------------

    # Function required to calculate instantaneous utility and feasibility
    U_and_F = _get_U_and_F(internal_functions)
    regime_transition_prob_func = internal_functions.regime_transition_probs.solve
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    for regime_name, transitions in internal_functions.transitions.items():
        # Functions required to calculate the expected continuation values
        state_transitions[regime_name] = get_next_state_function(
            grids=grids[regime_name],
            functions=internal_functions.functions,
            transitions=transitions,
            target=Target.SOLVE,
        )
        next_stochastic_states_weights[regime_name] = (
            get_next_stochastic_weights_function(
                regime_name=regime.name,
                functions=internal_functions.functions,
                transitions=transitions,
            )
        )
        joint_weights_from_marginals[regime_name] = _get_joint_weights_function(
            regime_name=regime.name, transitions=transitions
        )
        _scalar_next_V = get_value_function_representation(
            next_state_space_infos[regime_name]
        )
        next_V[regime_name] = productmap(
            _scalar_next_V,
            variables=tuple(
                key
                for key, value in transitions.items()
                if is_stochastic_transition(value)
            ),
        )

    # ----------------------------------------------------------------------------------
    # Create the state-action value and feasibility function
    # ----------------------------------------------------------------------------------
    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        [
            U_and_F,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
        ],
        include={"params", "next_V_arr", "period"},
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_V_arr: FloatND,
        params: ParamsDict,
        period: Period,
        **states_and_actions: Array,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action value and feasibility for a non-terminal period.

        Args:
            params: The parameters.
            period: The current period.
            next_V_arr: The next period's value function array.
            **states_and_actions: The current states and actions.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        # ------------------------------------------------------------------------------
        # Calculate the expected continuation values
        # ------------------------------------------------------------------------------
        regime_transition_prob = regime_transition_prob_func(
            **states_and_actions, period=period, params=params
        )
        U_arr, F_arr = U_and_F(
            **states_and_actions,
            period=period,
            params=params[regime.name],
        )
        Q_arr = U_arr
        for regime_name in internal_functions.transitions:
            next_states = state_transitions[regime_name](
                **states_and_actions,
                period=period,
                params=params[regime_name],
            )

            marginal_next_stochastic_states_weights = next_stochastic_states_weights[
                regime_name
            ](
                **states_and_actions,
                period=period,
                params=params[regime_name],
            )

            joint_next_stochastic_states_weights = joint_weights_from_marginals[
                regime_name
            ](**marginal_next_stochastic_states_weights)

            # As we productmap'd the value function over the stochastic variables, the
            # resulting next value function gets a new dimension for each stochastic
            # variable.
            next_V_at_stochastic_states_arr = next_V[regime_name](
                **next_states, next_V_arr=next_V_arr[regime_name]
            )

            # We then take the weighted average of the next value function at the
            # stochastic states to get the expected next value function.
            next_V_expected_arr = jnp.average(
                next_V_at_stochastic_states_arr,
                weights=joint_next_stochastic_states_weights,
            )
            Q_arr = (
                Q_arr
                + params[regime_name]["beta"]
                * regime_transition_prob[regime_name]
                * next_V_expected_arr
            )

        # ------------------------------------------------------------------------------
        # Calculate the instantaneous utility and feasibility
        # ------------------------------------------------------------------------------

        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_Q_and_F_terminal(
    regime: Regime,
    internal_functions: InternalFunctions,
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for the terminal period.

    Args:
        regime: The current regime.
        internal_functions: Internal functions instance.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for the terminal period.

    """
    U_and_F = _get_U_and_F(internal_functions)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        [U_and_F],
        # While the terminal period does not depend on the value function array, we
        # include it in the signature, such that we can treat all periods uniformly
        # during the solution and simulation.
        include={"params", "next_V_arr", "period"},
    )

    args = dict.fromkeys(arg_names_of_Q_and_F, "Array")
    args["params"] = "ParamsDict"
    args["next_V_arr"] = "FloatND"

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_V_arr: FloatND,  # noqa: ARG001
        params: ParamsDict,
        period: Period,
        **states_and_actions: Array,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action values and feasibilities for the terminal period.

        Args:
            params: The parameters.
            period: The current period.
            next_V_arr: The next period's value function array (unused here).
            **states_and_actions: The current states and actions.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        U_arr, F_arr = U_and_F(
            **states_and_actions,
            period=period,
            params=params[regime.name],
        )

        return jnp.asarray(U_arr), jnp.asarray(F_arr)

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
    regime_name: RegimeName,
    transitions: dict[RegimeName, InternalUserFunction],
) -> Callable[..., FloatND]:
    """Get function that calculates the joint weights.

    This function takes the weights of the individual stochastic variables and
    multiplies them together to get the joint weights on the product space of the
    stochastic variables.

    Args:
        regime_name: Name of the target regime.
        transitions: Transitions of the target regime.

    Returns:
        A function that computes the outer product of the weights of the stochastic
        variables.

    """
    arg_names = [
        f"weight_{regime_name}__{key}"
        for key, value in transitions.items()
        if is_stochastic_transition(value)
    ]

    @with_signature(args=arg_names)
    def _outer(**kwargs: Float1D) -> FloatND:
        weights = jnp.array(list(kwargs.values()))
        return jnp.prod(weights)

    return productmap(_outer, variables=tuple(arg_names))


def _get_U_and_F(
    internal_functions: InternalFunctions,
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Get the instantaneous utility and feasibility function.

    Note:
    -----
    U may depend on all kinds of other functions (taxes, transfers, ...), which will be
    executed if they matter for the value of U.

    Args:
        internal_functions: Internal functions instance.

    Returns:
        The instantaneous utility and feasibility function.

    """
    functions = {
        "feasibility": _get_feasibility(internal_functions),
        "utility": internal_functions.utility,
        **internal_functions.functions,
    }
    return concatenate_functions(
        functions=functions,
        targets=["utility", "feasibility"],
        enforce_signature=False,
        set_annotations=True,
    )


def _get_feasibility(internal_functions: InternalFunctions) -> InternalUserFunction:
    """Create a function that combines all constraint functions into a single one.

    Args:
        internal_functions: Internal functions instance.

    Returns:
        The combined constraint function (feasibility).

    """
    if internal_functions.constraints:
        with warnings.catch_warnings():
            # set annotations does not set the return type when concatenate_functions is
            # called with an aggregator and raises a warning.
            warnings.simplefilter("ignore", category=DagsWarning)
            combined_constraint = concatenate_functions(
                functions=internal_functions.constraints | internal_functions.functions,
                targets=list(internal_functions.constraints),
                aggregator=jnp.logical_and,
                set_annotations=True,
            )
        combined_constraint.__annotations__["return"] = "Feasibility"

    else:

        def combined_constraint() -> bool:
            """Dummy feasibility function that always returns True."""
            return True

    return combined_constraint
