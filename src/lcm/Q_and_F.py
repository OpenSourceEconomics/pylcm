from collections.abc import Callable
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags import concatenate_functions, with_signature
from jax import Array

from lcm.dispatchers import productmap
from lcm.function_representation import get_V_interpolator
from lcm.functools import get_union_of_args
from lcm.interfaces import StateSpaceInfo
from lcm.next_state import (
    get_next_state_function_for_solution,
    get_next_stochastic_weights_function,
)
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    InternalUserFunction,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    TransitionFunctionsMapping,
)


def get_Q_and_F(
    *,
    flat_param_names: frozenset[str],
    age: float,
    period: int,
    functions: MappingProxyType[str, InternalUserFunction],
    constraints: MappingProxyType[str, InternalUserFunction],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[str],
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_transition_probs: RegimeTransitionFunction,
    next_state_space_infos: MappingProxyType[RegimeName, StateSpaceInfo],
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        age: The age corresponding to the current period.
        period: The current period.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        transitions: Immutable mapping of transition names to transition functions.
        stochastic_transition_names: Frozenset of stochastic transition function names.
        regimes_to_active_periods: Mapping regime names to their active periods.
        regime_transition_probs: Regime transition probability function for solve.
        next_state_space_infos: The state space information of the next period.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)
    regime_transition_probs_func = regime_transition_probs
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    target_regime_names = tuple(transitions)
    active_regimes_next_period = tuple(
        target_regime_name
        for target_regime_name in target_regime_names
        if period + 1 in regimes_to_active_periods[target_regime_name]
    )
    next_V_extra_param_names: dict[str, frozenset[str]] = {}

    for target_regime_name in active_regimes_next_period:
        # Transitions from the current regime to the target regime
        target_transitions = transitions[target_regime_name]

        # Functions required to calculate the expected continuation values
        state_transitions[target_regime_name] = get_next_state_function_for_solution(
            functions=functions,
            transitions=target_transitions,
        )
        next_stochastic_states_weights[target_regime_name] = (
            get_next_stochastic_weights_function(
                functions=functions,
                transitions=target_transitions,
                stochastic_transition_names=stochastic_transition_names,
                regime_name=target_regime_name,
            )
        )
        joint_weights_from_marginals[target_regime_name] = _get_joint_weights_function(
            transitions=target_transitions,
            stochastic_transition_names=stochastic_transition_names,
            regime_name=target_regime_name,
        )
        V_arr_name = "next_V_arr"
        next_V_interpolator = get_V_interpolator(
            state_space_info=next_state_space_infos[target_regime_name],
            state_prefix="next_",
            V_arr_name=V_arr_name,
        )
        # Determine extra kwargs needed by next_V beyond next_states and next_V_arr
        # (e.g. wealth__points for IrregSpacedGrid with runtime-supplied points).
        next_V_extra_param_names[target_regime_name] = frozenset(
            get_union_of_args([next_V_interpolator])
            - set(target_transitions)
            - {V_arr_name}
        )
        next_V[target_regime_name] = productmap(
            func=next_V_interpolator,
            variables=tuple(
                key for key in target_transitions if key in stochastic_transition_names
            ),
        )

    # ----------------------------------------------------------------------------------
    # Create the state-action value and feasibility function
    # ----------------------------------------------------------------------------------

    # Determine which qname params H accepts so we can filter at runtime.
    # This is necessary when the params template is the union of multiple
    # PhaseVariant signatures but each variant only uses its own subset.
    _H_func = functions["H"]
    _H_accepted_params = frozenset(
        get_union_of_args([_H_func]) - {"utility", "E_next_V"}
    )

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        [
            U_and_F,
            regime_transition_probs_func,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
        ],
        include=frozenset({"next_regime_to_V_arr"} | flat_param_names),
        exclude=frozenset({"period", "age"}),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: FloatND,
        **states_actions_params: Array,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action value and feasibility for a non-terminal period.

        Args:
            next_regime_to_V_arr: The next period's value function array.
            **states_actions_params: States, actions, and flat regime params.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        regime_transition_probs: MappingProxyType[str, Array] = (  # ty: ignore[invalid-assignment]
            regime_transition_probs_func(
                **states_actions_params,
                period=period,
                age=age,
            )
        )
        U_arr, F_arr = U_and_F(
            **states_actions_params,
            period=period,
            age=age,
        )
        # Filter to active regimes only — inactive regimes must have 0
        # probability (validated before solve).
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in active_regimes_next_period}
        )

        E_next_V = jnp.zeros_like(U_arr)
        for target_regime_name in active_regimes_next_period:
            next_states = state_transitions[target_regime_name](
                **states_actions_params,
                period=period,
                age=age,
            )
            marginal_next_stochastic_states_weights = next_stochastic_states_weights[
                target_regime_name
            ](
                **states_actions_params,
                period=period,
                age=age,
            )
            joint_next_stochastic_states_weights = joint_weights_from_marginals[
                target_regime_name
            ](**marginal_next_stochastic_states_weights)

            # As we productmap'd the value function over the stochastic variables, the
            # resulting next value function gets a new dimension for each stochastic
            # variable.
            extra_kw = {
                k: states_actions_params[k]
                for k in next_V_extra_param_names[target_regime_name]
            }
            next_V_at_stochastic_states_arr = next_V[target_regime_name](
                **next_states,
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )

            # We then take the weighted average of the next value function at the
            # stochastic states to get the expected next value function.
            next_V_expected_arr = jnp.average(
                next_V_at_stochastic_states_arr,
                weights=joint_next_stochastic_states_weights,
            )
            E_next_V = (
                E_next_V + active_regime_probs[target_regime_name] * next_V_expected_arr
            )

        H_kwargs = {
            k: v for k, v in states_actions_params.items() if k in _H_accepted_params
        }
        Q_arr = _H_func(utility=U_arr, E_next_V=E_next_V, **H_kwargs)

        # Handle cases when there is only one state.
        # In that case, Q_arr and F_arr are scalars, but we require arrays as output.
        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_Q_and_F_terminal(
    *,
    flat_param_names: frozenset[str],
    age: float,
    period: int,
    functions: MappingProxyType[str, InternalUserFunction],
    constraints: MappingProxyType[str, InternalUserFunction],
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for the terminal period.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        age: The age corresponding to the current period.
        period: The current period.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for the terminal period.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        [U_and_F],
        # While the terminal period does not depend on the value function array, we
        # include it in the signature, such that we can treat all periods uniformly
        # during the solution and simulation.
        include=frozenset({"next_regime_to_V_arr"} | flat_param_names),
        exclude=frozenset({"period", "age"}),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: FloatND,  # noqa: ARG001
        **states_actions_params: Array,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action values and feasibilities for the terminal period.

        Args:
            next_regime_to_V_arr: The next period's value function array (unused here).
            **states_actions_params: States, actions, and flat regime params.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        U_arr, F_arr = U_and_F(
            **states_actions_params,
            period=period,
            age=age,
        )

        return jnp.asarray(U_arr), jnp.asarray(F_arr)

    return Q_and_F


def _get_arg_names_of_Q_and_F(
    deps: list[Callable[..., Any]],
    *,
    include: frozenset[str] = frozenset(),
    exclude: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Get the argument names of the dependencies.

    Args:
        deps: List of dependencies.
        include: Set of argument names to include.
        exclude: Set of argument names to exclude.

    Returns:
        The union of the argument names in deps and include, except for those in
        exclude.

    """
    return tuple((get_union_of_args(deps) | include) - exclude)


def _get_joint_weights_function(
    *,
    transitions: MappingProxyType[str, InternalUserFunction],
    stochastic_transition_names: frozenset[str],
    regime_name: RegimeName,
) -> Callable[..., FloatND]:
    """Get function that calculates the joint weights.

    This function takes the weights of the individual stochastic variables and
    multiplies them together to get the joint weights on the product space of the
    stochastic variables.

    Args:
        transitions: Transitions of the target regime.
        stochastic_transition_names: Frozenset of stochastic transition function names.
        regime_name: Name of the target regime.

    Returns:
        A function that computes the outer product of the weights of the stochastic
        variables.

    """
    arg_names = [
        f"weight_{regime_name}__{key}"
        for key in transitions
        if key in stochastic_transition_names
    ]

    @with_signature(args=arg_names)
    def _outer(**kwargs: Float1D) -> FloatND:
        weights = jnp.array(list(kwargs.values()))
        return jnp.prod(weights)

    return productmap(func=_outer, variables=tuple(arg_names))


def _get_U_and_F(
    *,
    functions: MappingProxyType[str, InternalUserFunction],
    constraints: MappingProxyType[str, InternalUserFunction],
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Get the instantaneous utility and feasibility function.

    Note:
    -----
    U may depend on all kinds of other functions (taxes, transfers, ...), which will be
    executed if they matter for the value of U.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.

    Returns:
        The instantaneous utility and feasibility function.

    """
    combined = {
        "feasibility": _get_feasibility(functions=functions, constraints=constraints),
        **{k: v for k, v in functions.items() if k != "H"},
    }
    return concatenate_functions(
        functions=combined,
        targets=["utility", "feasibility"],
        enforce_signature=False,
        set_annotations=True,
    )


def _get_feasibility(
    *,
    functions: MappingProxyType[str, InternalUserFunction],
    constraints: MappingProxyType[str, InternalUserFunction],
) -> InternalUserFunction:
    """Create a function that combines all constraint functions into a single one.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.

    Returns:
        The combined constraint function (feasibility).

    """
    if constraints:
        combined_constraint = concatenate_functions(
            functions=dict(constraints) | dict(functions),
            targets=list(constraints),
            aggregator=jnp.logical_and,
            aggregator_return_type="Feasibility",
            set_annotations=True,
        )

    else:

        def combined_constraint() -> bool:
            """Dummy feasibility function that always returns True."""
            return True

    return cast("InternalUserFunction", combined_constraint)
