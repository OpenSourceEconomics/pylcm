from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags import concatenate_functions, with_signature

from _lcm.certainty_equivalent import CertaintyEquivalent, LinearExpectation
from _lcm.regime_building.next_state import (
    get_next_state_function_for_solution,
    get_next_stochastic_weights_function,
)
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.regime_building.w_dag import _get_build_W_kwargs
from _lcm.transition_laws import TransitionLaws, is_stochastic
from _lcm.typing import (
    ConstraintFunction,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import get_union_of_args
from lcm.typing import BoolND, Float1D, FloatND, IntND


def get_Q_and_F(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    koopmans_aggregator: EconFunction,
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...] = (),
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    `age` and `period` are runtime arguments (via `**states_actions_params`),
    not closure constants. This allows periods with the same target
    configuration to share a single JIT-compiled function.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        period_targets: Graph targets whose continuation enters the certainty
            equivalent this period.
        transitions: Immutable mapping of transition names to transition functions.
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        compute_regime_transition_probs: Regime transition probability function
            for solve.
        regime_to_v_interpolation_info: Mapping of regime names to V-interpolation
            info.
        koopmans_aggregator: The regime's Bellman aggregator, combining
            utility and the certainty equivalent into `Q`.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.
        co_map_state_names: Tuple of state names co-mapped with the continuation V —
            their axes are sliced off each `next_V_arr` leaf by the backward-induction
            co-map, so their coordinates are dropped from the interpolation. Only fixed
            (never-transitioning) distributed states qualify.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)
    compute_CE, continuation_deps = _get_compute_CE(
        functions=functions,
        period_targets=period_targets,
        transitions=transitions,
        transition_laws=transition_laws,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
    )
    _build_W_kwargs = _get_build_W_kwargs(functions, koopmans_aggregator)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[U_and_F, *continuation_deps],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: FloatND,
        **states_actions_params: FloatND | IntND | BoolND,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action value and feasibility for a non-terminal period.

        Args:
            next_regime_to_V_arr: The next period's value function array.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        U_arr, F_arr = U_and_F(**states_actions_params)
        CE, _ = compute_CE(
            next_regime_to_V_arr=next_regime_to_V_arr,
            zero=jnp.zeros_like(U_arr),
            states_actions_params=states_actions_params,
        )

        Q_arr = koopmans_aggregator(
            utility=U_arr,
            CE=CE,
            **_build_W_kwargs(states_actions_params),
        )

        # Handle cases when there is only one state.
        # In that case, Q_arr and F_arr are scalars, but we require arrays as output.
        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_compute_intermediates(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    koopmans_aggregator: EconFunction,
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...],
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
        period_targets: Graph targets whose continuation enters the certainty
            equivalent this period.
        transitions: Immutable mapping of target regime names to state transition
            functions.
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        compute_regime_transition_probs: Callable returning regime transition
            probabilities for the current regime.
        regime_to_v_interpolation_info: Immutable mapping of regime names to
            V-interpolation info.
        koopmans_aggregator: The regime's Bellman aggregator, combining
            utility and the certainty equivalent into `Q`.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.
        co_map_state_names: Tuple of state names co-mapped with the
            continuation V in the solve kernel. The diagnostics pass an empty
            tuple: they are handed the full, un-sliced value arrays and map
            over every state, so no axis has been sliced off to compensate for.

    Returns:
        Closure returning `(U_arr, F_arr, CE, Q_arr, active_regime_probs)`.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)
    compute_CE, continuation_deps = _get_compute_CE(
        functions=functions,
        period_targets=period_targets,
        transitions=transitions,
        transition_laws=transition_laws,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
    )
    _build_W_kwargs = _get_build_W_kwargs(functions, koopmans_aggregator)

    arg_names_of_compute_intermediates = _get_arg_names_of_Q_and_F(
        deps=[U_and_F, *continuation_deps],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_compute_intermediates,
        return_annotation=(
            "tuple[FloatND, FloatND, FloatND, FloatND, "
            "MappingProxyType[RegimeName, FloatND]]"
        ),
    )
    def compute_intermediates(
        next_regime_to_V_arr: FloatND,
        **states_actions_params: FloatND | IntND | BoolND,
    ) -> tuple[
        FloatND, FloatND, FloatND, FloatND, MappingProxyType[RegimeName, FloatND]
    ]:
        """Compute all Q_and_F intermediates."""
        U_arr, F_arr = U_and_F(**states_actions_params)
        CE, active_regime_probs = compute_CE(
            next_regime_to_V_arr=next_regime_to_V_arr,
            zero=jnp.zeros_like(U_arr),
            states_actions_params=states_actions_params,
        )

        Q_arr = koopmans_aggregator(
            utility=U_arr,
            CE=CE,
            **_build_W_kwargs(states_actions_params),
        )

        return U_arr, F_arr, CE, Q_arr, active_regime_probs

    return compute_intermediates


def get_Q_and_F_terminal(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
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
        deps=[U_and_F],
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
        **states_actions_params: FloatND | IntND | BoolND,
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


def _get_compute_CE(
    *,
    functions: EconFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...],
) -> tuple[
    Callable[..., tuple[FloatND, MappingProxyType[RegimeName, FloatND]]],
    tuple[Callable[..., Any], ...],
]:
    """Build the closure that aggregates next period's value into `CE`.

    The single continuation-aggregation site of the engine: both the Bellman
    `Q` and the NaN diagnostics call the closure this returns, so they cannot
    disagree. The continuation is a lottery over the stochastic nodes of every
    reachable target regime, weighted by that target's regime-transition
    probability:

    - Without a certainty equivalent, it is aggregated as the linear
      expectation `Σ_r p_r · E_w[V'_r]`.
    - With one, the whole joint lottery is handed to
      `CertaintyEquivalent.aggregate` in one piece, which applies the
      transform before every expectation and inverts exactly once. Flattening
      before aggregating is what lets `PowerMean` anchor the transform, which
      a per-target `transform -> reduce -> inverse` decomposition cannot.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        period_targets: Graph targets whose continuation enters the certainty
            equivalent this period.
        transitions: Immutable mapping of transition names to transition functions.
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        compute_regime_transition_probs: Regime transition probability function
            for solve.
        regime_to_v_interpolation_info: Immutable mapping of regime names to
            V-interpolation info.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.
        co_map_state_names: Tuple of state names co-mapped with the continuation V.

    Returns:
        Tuple of the closure returning `(CE, active_regime_probs)` and the
        dependencies whose arguments must enter the calling closure's signature.

    """
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[RegimeName, frozenset[str]] = {}
    next_V_has_stochastic_states: dict[RegimeName, bool] = {}

    for target_regime_name in period_targets:
        # Transitions from the current regime to the target regime
        bundle = transitions.get(target_regime_name, MappingProxyType({}))

        # Functions required to calculate the expected continuation values
        state_transitions[target_regime_name] = get_next_state_function_for_solution(
            functions=functions,
            transitions=bundle,
        )
        next_stochastic_states_weights[target_regime_name] = (
            get_next_stochastic_weights_function(
                functions=functions,
                transitions=bundle,
                transition_laws=transition_laws,
                regime_name=target_regime_name,
            )
        )
        joint_weights_from_marginals[target_regime_name] = _get_joint_weights_function(
            transitions=bundle,
            transition_laws=transition_laws,
            regime_name=target_regime_name,
        )
        V_arr_name = "next_V_arr"
        next_V_interpolator = get_V_interpolator(
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
            state_prefix="next_",
            V_arr_name=V_arr_name,
            co_map_state_names=co_map_state_names,
        )
        # Determine extra kwargs needed by next_V beyond next_states and next_V_arr
        # (e.g. wealth__points for IrregSpacedGrid with runtime-supplied points).
        next_V_extra_param_names[target_regime_name] = frozenset(
            get_union_of_args([next_V_interpolator]) - set(bundle) - {V_arr_name}
        )
        stochastic_variables = tuple(
            key
            for key in bundle
            if is_stochastic(transition_laws, target_regime_name, key)
        )
        next_V_has_stochastic_states[target_regime_name] = bool(stochastic_variables)
        next_V[target_regime_name] = productmap(
            func=next_V_interpolator,
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    # The plain expectation reduces each target on its own; every other
    # certainty equivalent needs the whole joint lottery in one piece, because
    # its transform has to be applied before any expectation is taken.
    # `LinearExpectation.aggregate` states the same quantity over the flattened
    # lottery, but reducing per target is materially cheaper.
    # Exact type, not `isinstance`: a subclass overriding `aggregate` states a
    # different quantity, and the per-target route would silently discard the
    # override.
    reduces_per_target = (
        certainty_equivalent is None or type(certainty_equivalent) is LinearExpectation
    )
    ce_flat_param_names = (
        MappingProxyType({})
        if certainty_equivalent is None
        else certainty_equivalent.flat_param_names
    )

    # Co-mapped states are sliced off each `next_V_arr` leaf by the backward-
    # induction co-map, so their `next_`-prefixed coordinates are not passed to
    # the interpolator (which no longer indexes those axes).
    co_map_next_names = frozenset(f"next_{name}" for name in co_map_state_names)

    def compute_CE(
        *,
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        zero: FloatND,
        states_actions_params: Mapping[str, Any],
    ) -> tuple[FloatND, MappingProxyType[RegimeName, FloatND]]:
        """Aggregate the continuation lottery into `CE` at one state-action point.

        Args:
            next_regime_to_V_arr: Immutable mapping of target regime names to
                next period's value function arrays.
            zero: Zero at the shape and dtype of the value being built up.
            states_actions_params: Mapping of states, actions, age, period, and
                flat regime params. Forwarded verbatim to the transition and
                probability functions, so it carries whatever the caller
                supplies — including params that never passed through
                `cast_params_to_canonical_dtypes`.

        Returns:
            Tuple of the aggregated continuation value and the regime transition
            probabilities of the reachable targets.

        """
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in period_targets}
        )

        CE = zero
        probability_mass = zero
        lottery_values: list[FloatND] = []
        lottery_weights: list[FloatND] = []
        for target_regime_name in period_targets:
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
                **{
                    name: val
                    for name, val in next_states.items()
                    if name not in co_map_next_names
                },
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )

            if reduces_per_target:
                # We then take the weighted average of the next value function at the
                # stochastic states to get the expected next value function.
                if next_V_has_stochastic_states[target_regime_name]:
                    next_V_expected_arr = _expectation_over_stochastic_nodes(
                        values=next_V_at_stochastic_states_arr,
                        weights=joint_next_stochastic_states_weights,
                    )
                else:
                    next_V_expected_arr = jnp.average(next_V_at_stochastic_states_arr)
                target_probability = active_regime_probs[target_regime_name]
                CE = CE + target_probability * next_V_expected_arr
                probability_mass = probability_mass + target_probability
            else:
                values, node_weights = _as_lottery(
                    values=next_V_at_stochastic_states_arr,
                    weights=joint_next_stochastic_states_weights,
                    has_stochastic_states=next_V_has_stochastic_states[
                        target_regime_name
                    ],
                )
                lottery_values.append(values)
                lottery_weights.append(
                    active_regime_probs[target_regime_name] * node_weights
                )

        if reduces_per_target and period_targets:
            # The per-target route accumulates `Σ p·E[V]`, so it has to divide by
            # the represented mass to state the same quantity as
            # `LinearExpectation.aggregate`. Regime-transition validation accepts
            # any mass within `jnp.allclose` of one, and at the top of that
            # tolerance the undivided sum reverses the Bellman argmax. Dividing is
            # exact whenever the mass is exactly one, so a well-formed lottery
            # keeps its floating-point association.
            #
            # A regime with no target at all carries no continuation, and its
            # `CE` stays at zero rather than becoming `0 / 0` — matching the
            # lottery route, which leaves `CE` at zero when it collects no nodes.
            # A represented mass of zero across targets that do exist is a
            # massless lottery, and NaN there is the same answer both routes give.
            CE = CE / probability_mass
        elif certainty_equivalent is not None and lottery_values:
            CE = certainty_equivalent.aggregate(
                values=jnp.concatenate(lottery_values),
                weights=jnp.concatenate(lottery_weights),
                # The params template types every certainty-equivalent
                # parameter as a float, so its runtime values are float arrays.
                params=cast(
                    "Mapping[str, FloatND]",
                    {
                        arg: states_actions_params[flat_name]
                        for arg, flat_name in ce_flat_param_names.items()
                    },
                ),
            )

        return CE, active_regime_probs

    deps = (
        compute_regime_transition_probs,
        *state_transitions.values(),
        *next_stochastic_states_weights.values(),
    )
    return compute_CE, deps


def _expectation_over_stochastic_nodes(*, values: FloatND, weights: FloatND) -> FloatND:
    """Return the weighted mean of one target's continuation over its nodes.

    Normalized explicitly rather than with `jnp.average`, for the reason
    `_as_lottery` states: a target whose joint weights carry no mass
    contributes no branch, and must not contribute NaN either — every target
    enters the same continuation, so a NaN here would destroy the
    well-specified targets beside it.
    """
    weight_sum = jnp.sum(weights)
    safe_weight_sum = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
    return jnp.sum(values * weights) / safe_weight_sum


def _as_lottery(
    *,
    values: FloatND,
    weights: FloatND,
    has_stochastic_states: bool,
) -> tuple[Float1D, Float1D]:
    """Flatten one target regime's continuation into a unit-mass lottery.

    Args:
        values: Next period's value at this target's stochastic nodes.
        weights: Joint weights over those nodes; ignored when the target has
            no stochastic states.
        has_stochastic_states: Whether the target's transition draws stochastic
            states.

    Returns:
        Tuple of the flattened values and their probabilities, which sum to
        one — or to zero for a target whose weights carry no mass at all.

    """
    flat_values = jnp.ravel(values)
    if has_stochastic_states:
        flat_weights = jnp.ravel(weights)
        weight_sum = jnp.sum(flat_weights)
        # A target whose joint weights carry no mass contributes no branch. It
        # must not contribute NaN either: every target's nodes are concatenated
        # into one lottery, so a NaN here would destroy the certainty
        # equivalent of the well-specified targets alongside it.
        safe_weight_sum = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
        return flat_values, flat_weights / safe_weight_sum
    uniform = jnp.full(
        flat_values.shape, 1.0 / flat_values.size, dtype=flat_values.dtype
    )
    return flat_values, uniform


def _get_arg_names_of_Q_and_F(
    *,
    deps: list[Callable[..., Any]],
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
    transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
    transition_laws: TransitionLaws,
    regime_name: RegimeName,
) -> Callable[..., FloatND]:
    """Get function that calculates the joint weights.

    This function takes the weights of the individual stochastic variables and
    multiplies them together to get the joint weights on the product space of the
    stochastic variables.

    Args:
        transitions: Transitions of the target regime.
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        regime_name: Name of the target regime.

    Returns:
        A function that computes the outer product of the weights of the stochastic
        variables.

    """
    arg_names = [
        f"weight_{regime_name}__{key}"
        for key in transitions
        if is_stochastic(transition_laws, regime_name, key)
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
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
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
        **functions,
    }
    return concatenate_functions(
        functions=combined,
        targets=["utility", "feasibility"],
        enforce_signature=False,
        set_annotations=True,
    )


def _get_feasibility(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
) -> ConstraintFunction:
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

    return cast("ConstraintFunction", combined_constraint)
