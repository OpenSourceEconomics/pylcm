from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags import concatenate_functions
from dags.signature import with_signature
from jax import Array

from lcm.dispatchers import productmap
from lcm.function_representation import get_value_function_representation
from lcm.functools import get_union_of_arguments
from lcm.input_processing.util import is_stochastic_transition
from lcm.interfaces import InternalFunctions, StateSpaceInfo, Target
from lcm.next_state import get_next_state_function, get_next_stochastic_weights_function
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    GridsDict,
    InternalUserFunction,
    ParamsDict,
    QAndFFunction,
    RegimeName,
)
from lcm.utils import normalize_regime_transition_probs


def get_Q_and_F(
    regime_name: str,
    regimes_to_active_periods: Mapping[RegimeName, tuple[int, ...]],
    period: int,
    age: float,
    next_state_space_infos: Mapping[RegimeName, StateSpaceInfo],
    grids: GridsDict,
    internal_functions: InternalFunctions,
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    Args:
        regime_name: The name of the regime.
        regimes_to_active_periods: Mapping regime names to their active periods.
        period: The current period.
        age: The age corresponding to the current period.
        next_state_space_infos: The state space information of the next period.
        grids: Dict containing the state grids for all regimes.
        internal_functions: Internal functions instance.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    # ----------------------------------------------------------------------------------
    # Generate dynamic functions
    # ----------------------------------------------------------------------------------
    U_and_F = _get_U_and_F(internal_functions)
    regime_transition_prob_func = internal_functions.regime_transition_probs.solve  # ty: ignore[possibly-missing-attribute]
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    target_regimes = tuple(internal_functions.transitions)
    active_target_regimes = tuple(
        target_name
        for target_name in target_regimes
        if period + 1 in regimes_to_active_periods[target_name]
    )

    for target_regime in active_target_regimes:
        # Transitions from the current regime to the target regime
        transitions = internal_functions.transitions[target_regime]

        # Functions required to calculate the expected continuation values
        # Note: grids is not used for Target.SOLVE, but we pass the full dict for typing
        state_transitions[target_regime] = get_next_state_function(
            grids=grids,
            functions=internal_functions.functions,
            transitions=transitions,
            target=Target.SOLVE,
        )
        next_stochastic_states_weights[target_regime] = (
            get_next_stochastic_weights_function(
                regime_name=regime_name,
                functions=internal_functions.functions,
                transitions=transitions,
            )
        )
        joint_weights_from_marginals[target_regime] = _get_joint_weights_function(
            regime_name=regime_name, transitions=transitions
        )
        _scalar_next_V = get_value_function_representation(
            next_state_space_infos[target_regime]
        )
        next_V[target_regime] = productmap(
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
            regime_transition_prob_func,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
        ],
        include={"params", "next_V_arr"},
        exclude={"period", "age"},
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_V_arr: FloatND,
        params: ParamsDict,
        **states_and_actions: Array,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action value and feasibility for a non-terminal period.

        Args:
            params: The parameters.
            next_V_arr: The next period's value function array.
            **states_and_actions: The current states and actions.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        regime_transition_prob: MappingProxyType[str, Array] = (  # ty: ignore[invalid-assignment]
            regime_transition_prob_func(
                **states_and_actions, period=period, age=age, params=params[regime_name]
            )
        )
        U_arr, F_arr = U_and_F(
            **states_and_actions,
            period=period,
            age=age,
            params=params[regime_name],
        )
        Q_arr = U_arr
        # Normalize probabilities over active regimes
        normalized_regime_transition_prob = normalize_regime_transition_probs(
            regime_transition_prob, active_target_regimes
        )

        for target_regime_name in active_target_regimes:
            next_states = state_transitions[target_regime_name](
                **states_and_actions,
                period=period,
                age=age,
                params=params[regime_name],
            )

            marginal_next_stochastic_states_weights = next_stochastic_states_weights[
                target_regime_name
            ](
                **states_and_actions,
                period=period,
                age=age,
                params=params[regime_name],
            )

            joint_next_stochastic_states_weights = joint_weights_from_marginals[
                target_regime_name
            ](**marginal_next_stochastic_states_weights)

            # As we productmap'd the value function over the stochastic variables, the
            # resulting next value function gets a new dimension for each stochastic
            # variable.
            next_V_at_stochastic_states_arr = next_V[target_regime_name](
                **next_states, next_V_arr=next_V_arr[target_regime_name]
            )

            # We then take the weighted average of the next value function at the
            # stochastic states to get the expected next value function.
            next_V_expected_arr = jnp.average(
                next_V_at_stochastic_states_arr,
                weights=joint_next_stochastic_states_weights,
            )
            Q_arr = (
                Q_arr
                + params[regime_name]["discount_factor"]
                * normalized_regime_transition_prob[target_regime_name]
                * next_V_expected_arr
            )

        # Handle cases when there is only one state. (Q_arr and F_arr are then scalars
        # but we require arrays as output).
        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_Q_and_F_terminal(
    regime_name: RegimeName,
    internal_functions: InternalFunctions,
    period: int,
    age: float,
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for the terminal period.

    Args:
        regime_name: The name of the regime.
        internal_functions: Internal functions instance.
        period: The current period.
        age: The age corresponding to the current period.

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
        include={"params", "next_V_arr"},
        exclude={"period", "age"},
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
        U_arr, F_arr = U_and_F(
            **states_and_actions,
            period=period,
            age=age,
            params=params[regime_name],
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
    transitions: Mapping[str, InternalUserFunction],
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
        combined_constraint = concatenate_functions(
            functions=dict(internal_functions.constraints)
            | dict(internal_functions.functions),
            targets=list(internal_functions.constraints),
            aggregator=jnp.logical_and,
            aggregator_return_type="Feasibility",
            set_annotations=True,
        )

    else:

        def combined_constraint() -> bool:
            """Dummy feasibility function that always returns True."""
            return True

    return cast("InternalUserFunction", combined_constraint)
