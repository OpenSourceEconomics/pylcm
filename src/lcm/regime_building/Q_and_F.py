from collections.abc import Callable
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags import concatenate_functions, with_signature
from jax import Array

from lcm.regime_building.h_dag import _get_build_H_kwargs
from lcm.regime_building.next_state import (
    get_next_state_function_for_solution,
    get_next_stochastic_weights_function,
)
from lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    FunctionsMapping,
    InternalUserFunction,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    TransitionFunctionsMapping,
)
from lcm.utils.dispatchers import productmap
from lcm.utils.functools import get_union_of_args


def get_Q_and_F(
    *,
    flat_param_names: frozenset[str],
    functions: FunctionsMapping,
    constraints: FunctionsMapping,
    complete_targets: tuple[str, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[str],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    `age` and `period` are runtime arguments (via `**states_actions_params`),
    not closure constants. This allows periods with the same target
    configuration to share a single JIT-compiled function.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        complete_targets: Target regimes with all required stochastic transitions.
        transitions: Immutable mapping of transition names to transition functions.
        stochastic_transition_names: Frozenset of stochastic transition function names.
        compute_regime_transition_probs: Regime transition probability function
            for solve.
        regime_to_v_interpolation_info: Mapping of regime names to V-interpolation
            info.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[str, frozenset[str]] = {}

    for target_regime_name in complete_targets:
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
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
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
        stochastic_variables = tuple(
            key for key in target_transitions if key in stochastic_transition_names
        )
        next_V[target_regime_name] = productmap(
            func=next_V_interpolator,
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    # ----------------------------------------------------------------------------------
    # Create the state-action value and feasibility function
    # ----------------------------------------------------------------------------------

    _build_H_kwargs = _get_build_H_kwargs(functions)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        [
            U_and_F,
            compute_regime_transition_probs,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
        ],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
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
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        regime_transition_probs: MappingProxyType[str, Array] = (  # ty: ignore[invalid-assignment]
            compute_regime_transition_probs(**states_actions_params)
        )
        U_arr, F_arr = U_and_F(**states_actions_params)
        # Use only complete targets for the traced function — incomplete
        # target validation happens outside JIT to keep the HLO (and thus
        # the persistent compilation cache key) independent of the
        # partition.
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in complete_targets}
        )

        E_next_V = jnp.zeros_like(U_arr)
        for target_regime_name in complete_targets:
            next_states = state_transitions[target_regime_name](
                **states_actions_params,
            )
            marginal_next_stochastic_states_weights = next_stochastic_states_weights[
                target_regime_name
            ](**states_actions_params)
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

        Q_arr = functions["H"](
            utility=U_arr,
            E_next_V=E_next_V,
            **_build_H_kwargs(states_actions_params),
        )

        # Handle cases when there is only one state.
        # In that case, Q_arr and F_arr are scalars, but we require arrays as output.
        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_compute_intermediates(
    *,
    flat_param_names: frozenset[str],
    functions: FunctionsMapping,
    constraints: FunctionsMapping,
    complete_targets: tuple[str, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[str],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> Callable:
    """Build a closure that computes Q_and_F intermediates for diagnostics.

    Mirrors `get_Q_and_F` but returns all intermediates instead of just
    `(Q, F)`. The caller productmaps and JIT-compiles the closure; it runs
    only in the error path when `validate_V` detects NaN. `age` and `period`
    are runtime arguments (passed via `states_actions_params`) so that
    periods sharing the same target configuration share a single
    JIT-compiled function.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to constraint functions.
        complete_targets: Target regimes with all required stochastic transitions.
        transitions: Immutable mapping of target regime names to state transition
            functions.
        stochastic_transition_names: Frozenset of stochastic transition function
            names.
        compute_regime_transition_probs: Callable returning regime transition
            probabilities for the current regime.
        regime_to_v_interpolation_info: Immutable mapping of regime names to
            V-interpolation info.

    Returns:
        Closure returning `(U_arr, F_arr, E_next_V, Q_arr, active_regime_probs)`.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[str, frozenset[str]] = {}

    for target_regime_name in complete_targets:
        target_transitions = transitions[target_regime_name]
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
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
            state_prefix="next_",
            V_arr_name=V_arr_name,
        )
        next_V_extra_param_names[target_regime_name] = frozenset(
            get_union_of_args([next_V_interpolator])
            - set(target_transitions)
            - {V_arr_name}
        )
        stochastic_variables = tuple(
            key for key in target_transitions if key in stochastic_transition_names
        )
        next_V[target_regime_name] = productmap(
            func=next_V_interpolator,
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    _build_H_kwargs = _get_build_H_kwargs(functions)

    arg_names_of_compute_intermediates = _get_arg_names_of_Q_and_F(
        [
            U_and_F,
            compute_regime_transition_probs,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
        ],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_compute_intermediates,
        return_annotation=(
            "tuple[FloatND, FloatND, FloatND, FloatND, "
            "MappingProxyType[RegimeName, Array]]"
        ),
    )
    def compute_intermediates(
        next_regime_to_V_arr: FloatND,
        **states_actions_params: Array,
    ) -> tuple[FloatND, FloatND, FloatND, FloatND, MappingProxyType[RegimeName, Array]]:
        """Compute all Q_and_F intermediates."""
        regime_transition_probs: MappingProxyType[str, Array] = (  # ty: ignore[invalid-assignment]
            compute_regime_transition_probs(**states_actions_params)
        )
        U_arr, F_arr = U_and_F(**states_actions_params)
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in complete_targets}
        )

        E_next_V = jnp.zeros_like(U_arr)
        for target_regime_name in complete_targets:
            next_states = state_transitions[target_regime_name](
                **states_actions_params,
            )
            marginal = next_stochastic_states_weights[target_regime_name](
                **states_actions_params,
            )
            joint = joint_weights_from_marginals[target_regime_name](**marginal)
            extra_kw = {
                k: states_actions_params[k]
                for k in next_V_extra_param_names[target_regime_name]
            }
            next_V_stoch = next_V[target_regime_name](
                **next_states,
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )
            contribution = jnp.average(next_V_stoch, weights=joint)
            E_next_V = E_next_V + active_regime_probs[target_regime_name] * contribution

        Q_arr = functions["H"](
            utility=U_arr,
            E_next_V=E_next_V,
            **_build_H_kwargs(states_actions_params),
        )

        return U_arr, F_arr, E_next_V, Q_arr, active_regime_probs

    return compute_intermediates


def get_Q_and_F_terminal(
    *,
    flat_param_names: frozenset[str],
    functions: FunctionsMapping,
    constraints: FunctionsMapping,
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a terminal period.

    `age` and `period` are runtime arguments (via `**states_actions_params`).

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a terminal period.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        [U_and_F],
        # While the terminal period does not depend on the value function array, we
        # include it in the signature, such that we can treat all periods uniformly
        # during the solution and simulation.
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: FloatND,  # noqa: ARG001
        **states_actions_params: Array,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action values and feasibilities for a terminal period.

        Args:
            next_regime_to_V_arr: Unused in the terminal period; accepted so that
                solve and simulate treat all periods uniformly.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple of the state-action value array (Q) and the feasibility
            mask (F).

        """
        U_arr, F_arr = U_and_F(**states_actions_params)
        return jnp.asarray(U_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_complete_targets(
    *,
    period: int,
    transitions: TransitionFunctionsMapping,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    stochastic_transition_names: frozenset[str],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> tuple[RegimeName, ...]:
    """Return active target regimes whose stochastic needs are fully covered.

    Enumerates every regime active in the next period (from
    `regime_to_v_interpolation_info`) and keeps those whose stochastic
    state needs are all covered by `transitions`. Targets missing stochastic
    transitions (including those entirely absent from `transitions`) are
    dropped; `validate_regime_transitions_all_periods` (via
    `_validate_no_reachable_incomplete_targets` in
    `lcm.utils.error_handling`) raises pre-solve if any dropped target has
    non-zero transition probability.

    Args:
        period: The period to enumerate active targets for.
        transitions: Immutable mapping of target regime names to their
            state transition functions.
        regimes_to_active_periods: Immutable mapping of regime names to
            their active period tuples.
        stochastic_transition_names: Frozenset of stochastic transition
            function names.
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info.

    Returns:
        Tuple of complete target regime names.

    """
    all_active = tuple(
        regime_name
        for regime_name in regime_to_v_interpolation_info
        if period + 1 in regimes_to_active_periods.get(regime_name, ())
    )

    complete: list[RegimeName] = []
    for regime_name in all_active:
        target_stochastic_needs = {
            f"next_{s}"
            for s in regime_to_v_interpolation_info[regime_name].state_names
            if f"next_{s}" in stochastic_transition_names
        }
        if regime_name in transitions and target_stochastic_needs.issubset(
            transitions[regime_name]
        ):
            complete.append(regime_name)

    return tuple(complete)


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
    transitions: FunctionsMapping,
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

    variables = tuple(arg_names)
    return productmap(
        func=_outer, variables=variables, batch_sizes=dict.fromkeys(variables, 0)
    )


def _get_U_and_F(
    *,
    functions: FunctionsMapping,
    constraints: FunctionsMapping,
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
    functions: FunctionsMapping,
    constraints: FunctionsMapping,
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
