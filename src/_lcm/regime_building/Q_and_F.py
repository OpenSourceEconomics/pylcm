import dataclasses
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags import (
    concatenate_functions,
    get_ancestors,
    get_annotations,
    with_signature,
)

from _lcm.certainty_equivalent import CertaintyEquivalent, LinearExpectation
from _lcm.regime_building.next_state import (
    get_next_interpolation_basis_weights_function,
    get_next_state_function_for_solution,
    get_next_stochastic_weights_function,
)
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.regime_building.w_dag import _get_build_W_kwargs
from _lcm.transition_laws import (
    SupportAxes,
    TransitionLaws,
    is_interpolation_basis,
    is_stochastic,
)
from _lcm.typing import (
    ConstraintFunction,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    FunctionName,
    NextStateSimulationFunction,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
    _ParamsLeaf,
)
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import get_union_of_args
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND


def get_Q_and_F(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
    support_axes: SupportAxes,
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
        period_targets: Carry targets — reachable, active next period, and
            carrying at least one state, so their continuation is read at the
            next states their laws produce.
        scalar_targets: Graph targets active next period that carry no state.
            Their value function is rank-zero, so it enters `E[V]` as a single
            degenerate lottery node weighted only by the regime transition
            probability.
        transitions: Immutable mapping of transition names to transition functions.
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        support_axes: Immutable mapping of target regime names to their private
            node axes.
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
    deterministic_transitions, conflicting_deterministic_transition_names = (
        _get_deterministic_transitions(
            transitions=transitions,
            transition_laws=transition_laws,
        )
    )
    U_and_F = _get_U_and_F(
        functions=functions,
        constraints=constraints,
        deterministic_transitions=deterministic_transitions,
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
    )
    compute_CE, continuation_deps = _get_compute_CE(
        functions=functions,
        period_targets=period_targets,
        scalar_targets=scalar_targets,
        transitions=transitions,
        transition_laws=transition_laws,
        support_axes=support_axes,
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
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: _ParamsLeaf,
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
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
    support_axes: SupportAxes,
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
        period_targets: Carry targets — reachable, active next period, and
            carrying at least one state.
        scalar_targets: Graph stateless targets active next period, whose
            rank-zero value enters `E[V]` weighted only by the regime transition
            probability. Must match what `get_Q_and_F` was built with, or the
            diagnostics disagree with the solve they explain.
        transitions: Immutable mapping of target regime names to state transition
            functions.
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        support_axes: Immutable mapping of target regime names to their private
            node axes.
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
    deterministic_transitions, conflicting_deterministic_transition_names = (
        _get_deterministic_transitions(
            transitions=transitions,
            transition_laws=transition_laws,
        )
    )
    U_and_F = _get_U_and_F(
        functions=functions,
        constraints=constraints,
        deterministic_transitions=deterministic_transitions,
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
    )
    compute_CE, continuation_deps = _get_compute_CE(
        functions=functions,
        period_targets=period_targets,
        scalar_targets=scalar_targets,
        transitions=transitions,
        transition_laws=transition_laws,
        support_axes=support_axes,
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
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: _ParamsLeaf,
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
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],  # noqa: ARG001
        **states_actions_params: _ParamsLeaf,
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


def partition_continuation_targets(
    *,
    targets: tuple[RegimeName, ...],
    regime_to_v_interpolation_info: Mapping[RegimeName, VInterpolationInfo],
) -> tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]]:
    """Partition canonical graph targets into stateful and stateless tuples.

    Membership comes entirely from `targets`. Interpolation metadata classifies how
    each continuation is read without adding or removing graph edges.

    Args:
        targets: Canonical graph targets for one source and period.
        regime_to_v_interpolation_info: Mapping of regime names to interpolation
            metadata whose state names determine the continuation representation.

    Returns:
        Tuple of `(stateful_targets, scalar_targets)` preserving graph order.

    """
    stateful_targets = tuple(
        target
        for target in targets
        if regime_to_v_interpolation_info[target].state_names
    )
    scalar_targets = tuple(
        target
        for target in targets
        if not regime_to_v_interpolation_info[target].state_names
    )
    return stateful_targets, scalar_targets


def _get_compute_CE(
    *,
    functions: EconFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
    support_axes: SupportAxes,
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

    A target carrying no state joins that same lottery as a single degenerate
    node, so it is transformed with every other node rather than on its own.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        period_targets: Carry targets whose continuation is read at the next
            states their laws produce.
        scalar_targets: Targets carrying no state, whose rank-zero value enters
            as one degenerate lottery node.
        transitions: Immutable mapping of transition names to transition functions.
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        support_axes: Immutable mapping of target regime names to their private
            node axes.
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
    continuations = {
        target_regime_name: _build_target_continuation(
            target_regime_name=target_regime_name,
            functions=functions,
            bundle=transitions.get(target_regime_name, MappingProxyType({})),
            transition_laws=transition_laws,
            support_axes=support_axes,
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
            co_map_state_names=co_map_state_names,
        )
        for target_regime_name in period_targets
    }

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
            {r: regime_transition_probs[r] for r in (*period_targets, *scalar_targets)}
        )

        CE, lottery_values, lottery_weights, probability_mass = (
            _scalar_target_contribution(
                scalar_targets=scalar_targets,
                next_regime_to_V_arr=next_regime_to_V_arr,
                active_regime_probs=active_regime_probs,
                as_lottery=not reduces_per_target,
                zero=zero,
            )
        )
        for target_regime_name in period_targets:
            continuation = continuations[target_regime_name]
            next_states = continuation.next_states(**states_actions_params)
            joint_next_stochastic_states_weights = continuation.joint_lottery_weights(
                **continuation.lottery_weights(**states_actions_params)
            )

            # As we productmap'd the value function over the stochastic variables, the
            # resulting next value function gets a new dimension for each stochastic
            # variable.
            extra_kw = {
                k: states_actions_params[k] for k in continuation.extra_param_names
            }
            # The interpolator is indexed, not evaluated, along a node axis. A
            # declared entry publishes its physical value under the public name
            # every other law reads; what the target's value function is indexed
            # by is the private axis, substituted here and nowhere else — so the
            # physical value stays intact in `next_states` for the dependent
            # laws, diagnostics, and simulation.
            interpolator_coordinates = {
                name: val
                for name, val in next_states.items()
                if name not in co_map_next_names
            } | dict(continuation.support_axes)
            next_V_at_stochastic_states_arr = continuation.next_V(
                **interpolator_coordinates,
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )

            if continuation.n_basis_axes:
                # A declared entry names one value; the basis axes are only how
                # the target's nodes can express it. Contracting them here states
                # that value as the single number `Σ_j w_j · V(node_j)` -- the
                # linear interpolation of the target's value function -- before
                # any lottery is formed. The coefficients sum to one by
                # construction, so normalizing here would mask a malformed basis
                # rather than protect against one.
                next_V_at_stochastic_states_arr = jnp.tensordot(
                    next_V_at_stochastic_states_arr,
                    continuation.joint_basis_weights(
                        **continuation.basis_weights(**states_actions_params)
                    ),
                    axes=continuation.n_basis_axes,
                )

            target_probability = active_regime_probs[target_regime_name]
            probability_mass = probability_mass + target_probability

            if reduces_per_target:
                # We then take the weighted average of the next value function at the
                # stochastic states to get the expected next value function.
                if continuation.has_lottery_axes:
                    next_V_expected_arr = _expectation_over_stochastic_nodes(
                        values=next_V_at_stochastic_states_arr,
                        weights=joint_next_stochastic_states_weights,
                    )
                else:
                    next_V_expected_arr = jnp.average(next_V_at_stochastic_states_arr)
                # A target carrying no mass here is never consulted, so whatever
                # its entry law names at this point -- a value off the target's
                # support, or nothing meaningful at all -- must not reach the
                # aggregate.
                CE = CE + zero_safe_weighted_term(
                    target_probability, next_V_expected_arr
                )
            else:
                values, node_weights = _as_lottery(
                    values=next_V_at_stochastic_states_arr,
                    weights=joint_next_stochastic_states_weights,
                    has_stochastic_states=continuation.has_lottery_axes,
                )
                # Same rule, applied to the value rather than to a product: the
                # aggregate reduces values and weights together, so an
                # unreachable target's value has to be neutral before it is
                # collected.
                lottery_values.append(jnp.where(target_probability == 0, zero, values))
                lottery_weights.append(target_probability * node_weights)

        if reduces_per_target and (period_targets or scalar_targets):
            # The per-target route accumulates `Σ p·E[V]`, so it has to divide by
            # the represented mass to state the same quantity as
            # `LinearExpectation.aggregate`. Regime-transition validation accepts
            # any mass within `jnp.allclose` of one, and at the top of that
            # tolerance the undivided sum reverses the Bellman argmax. Dividing is
            # exact whenever the mass is exactly one, so a well-formed lottery
            # keeps its floating-point association.
            #
            # A regime with no target at all — neither carrying state nor
            # stateless — carries no continuation, and its `CE` stays at zero
            # rather than becoming `0 / 0`, matching the lottery route, which
            # leaves `CE` at zero when it collects no nodes.
            # A represented mass of zero across targets that do exist is a
            # massless lottery, and NaN there is the same answer both routes give.
            CE = CE / _unit_regime_mass_or_nan(probability_mass)
        elif certainty_equivalent is not None:
            # `aggregate` normalizes by the weight sum itself, so the lottery
            # route has no division to attach the check to. Selecting between
            # the aggregate and NaN leaves the well-formed path free of any
            # arithmetic at all, which a multiplication by `1.0` would not.
            #
            # With no node collected there is nothing to hand `aggregate` — it
            # would reduce over an empty axis — so the selection falls back to
            # the initialized `CE`. The mask is `False` there regardless, since
            # a mass of zero is not unit mass.
            CE = jnp.where(
                _regime_mass_is_unit(probability_mass),
                _aggregate_joint_lottery(
                    certainty_equivalent=certainty_equivalent,
                    lottery_values=lottery_values,
                    lottery_weights=lottery_weights,
                    ce_flat_param_names=ce_flat_param_names,
                    states_actions_params=states_actions_params,
                )
                if lottery_values
                else CE,
                jnp.nan,
            )

        return CE, active_regime_probs

    deps = (
        compute_regime_transition_probs,
        *(c.next_states for c in continuations.values()),
        *(c.lottery_weights for c in continuations.values()),
        *(c.basis_weights for c in continuations.values()),
    )
    return compute_CE, deps


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TargetContinuation:
    """Everything built once for one reachable target's continuation."""

    next_states: NextStateSimulationFunction
    """Next-period states of this target at one state-action point."""

    lottery_weights: Callable[..., dict[str, FloatND | IntND]]
    """Marginal probabilities of the target's stochastic laws."""

    basis_weights: Callable[..., dict[str, FloatND | IntND]]
    """Marginal node-basis coefficients of the target's declared entry laws."""

    joint_lottery_weights: Callable[..., FloatND]
    """Outer product of the lottery marginals, over the leading node axes."""

    joint_basis_weights: Callable[..., FloatND]
    """Outer product of the basis marginals, over the trailing node axes."""

    next_V: Callable[..., FloatND]
    """Target's value function, product-mapped over its node axes.

    The axes come in the order `(lottery..., basis...)`, so the basis block is
    the tail and contracts away in one `tensordot`.
    """

    support_axes: MappingProxyType[TransitionFunctionName, Int1D]
    """Private node axes to index `next_V` along, keyed by public next-state name.

    One entry per declared entry law. The public name produces that law's
    physical value everywhere else; only the interpolator sees this axis.
    """

    extra_param_names: frozenset[str]
    """Arguments `next_V` needs beyond the next states and the value array.

    A grid whose points arrive at runtime is the case — `wealth__points` for an
    `IrregSpacedGrid`.
    """

    has_lottery_axes: bool
    """Whether the target draws anything, i.e. whether `next_V` has lottery axes."""

    n_basis_axes: int
    """Number of trailing axes to contract against the basis weights."""

    draw_dependent_names: frozenset[TransitionFunctionName] = frozenset()
    """Laws the interpolator resolves itself, one value per node of a sibling draw."""


def _draw_dependent_law_names(
    *,
    bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    functions: EconFunctionsMapping,
    stochastic_names: tuple[TransitionFunctionName, ...],
) -> tuple[TransitionFunctionName, ...]:
    """Return the target's deterministic laws that read one of its own draws.

    A law reading `next_<state>` of a stochastic sibling depends on which node the
    draw lands on, so its value is one number per node rather than one number. The
    dependence travels through helpers, so the walk is transitive.

    Args:
        bundle: This target's unqualified `next_<state>` transition functions.
        functions: Immutable mapping of function names to internal user functions.
        stochastic_names: This target's stochastic `next_<state>` names.

    Returns:
        Tuple of the deterministic law names that resolve on the node axis, in
        `bundle` order.

    """
    carries_a_draw = set(stochastic_names)
    candidates = {
        name: func
        for name, func in (dict(functions) | dict(bundle)).items()
        if name not in carries_a_draw
    }
    growing = True
    while growing:
        growing = False
        for name, func in candidates.items():
            if name in carries_a_draw:
                continue
            args = (arg for arg in get_annotations(func) if arg != "return")
            if any(arg in carries_a_draw for arg in args):
                carries_a_draw.add(name)
                growing = True
    return tuple(
        name
        for name in bundle
        if name in carries_a_draw and name not in stochastic_names
    )


def _fail_if_a_read_draw_has_no_nodes_yet(
    *,
    target_regime_name: RegimeName,
    stochastic_names: tuple[TransitionFunctionName, ...],
    draw_dependent_names: tuple[TransitionFunctionName, ...],
    v_interpolation_info: VInterpolationInfo,
) -> None:
    """Check that every draw a law reads has a support while the model builds.

    Resolving a dependent law on the node axis reads the nodes themselves, so a
    process whose law arrives at runtime has nothing to resolve against — its
    nodes are not numbers yet.

    Args:
        target_regime_name: Regime whose laws are being built, named in the message.
        stochastic_names: This target's stochastic `next_<state>` names.
        draw_dependent_names: Laws resolved on the node axis, named in the message.
        v_interpolation_info: The target's V-interpolation info, holding the grids.

    Raises:
        ModelInitializationError: If a read draw's process is not fully specified.

    """
    for next_state_name in stochastic_names:
        state_name = next_state_name.removeprefix("next_")
        grid = v_interpolation_info.discrete_states[state_name]
        if getattr(grid, "is_fully_specified", True):
            continue
        msg = (
            f"{', '.join(sorted(draw_dependent_names))} of regime "
            f"'{target_regime_name}' reads the draw '{next_state_name}', but that "
            f"process is parameterized at runtime, so its nodes are not known "
            f"while the model builds. A law reading a draw is resolved on that "
            f"draw's own nodes, which requires them to be fixed at construction — "
            f"through the process constructor or `fixed_params`."
        )
        raise ModelInitializationError(msg)


def _get_interpolator_resolving_draws(
    *,
    next_V_interpolator: Callable[..., FloatND],
    bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    functions: EconFunctionsMapping,
    stochastic_names: tuple[TransitionFunctionName, ...],
    draw_dependent_names: tuple[TransitionFunctionName, ...],
    node_values: MappingProxyType[TransitionFunctionName, FloatND],
) -> Callable[..., FloatND]:
    """Wrap the interpolator so draw-dependent laws resolve on the node axis.

    The caller product-maps the result over the target's node axes, so one call
    sees one node per stochastic law. The draw's *index* is what indexes the value
    function; the draw's *value* is what a dependent law reads. Both come from the
    same node, which is why resolving them here — inside the axis the process
    already contributes — needs no second axis and no parameter for the draw.

    Args:
        next_V_interpolator: The target's value-function interpolator.
        bundle: This target's unqualified `next_<state>` transition functions.
        functions: Immutable mapping of function names to internal user functions.
        stochastic_names: This target's stochastic `next_<state>` names.
        draw_dependent_names: Laws to resolve here rather than ahead of the axes.
        node_values: Immutable mapping of each stochastic law to its nodes, indexed
            by the value its next-state function yields.

    Returns:
        A callable with the interpolator's signature, minus the laws it now
        resolves itself, plus whatever resolving them reads.

    """
    resolve = concatenate_functions(
        functions={
            name: func
            for name, func in (dict(bundle) | dict(functions)).items()
            if name not in stochastic_names
        },
        targets=list(draw_dependent_names),
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )
    resolver_args = get_union_of_args([resolve])
    interpolator_args = get_union_of_args([next_V_interpolator])
    read_as_a_draw = tuple(name for name in stochastic_names if name in resolver_args)
    arg_names = sorted((interpolator_args - set(draw_dependent_names)) | resolver_args)

    @with_signature(args=arg_names)
    def interpolate_at_this_node(**kwargs: FloatND) -> FloatND:
        drawn = {
            name: node_values[name][kwargs[name].astype(jnp.int32)]
            for name in read_as_a_draw
        }
        resolved = resolve(
            **{
                k: v for k, v in kwargs.items() if k in resolver_args and k not in drawn
            },
            **drawn,
        )
        return next_V_interpolator(
            **{k: v for k, v in kwargs.items() if k in interpolator_args},
            **resolved,
        )

    return interpolate_at_this_node


def _build_target_continuation(
    *,
    target_regime_name: RegimeName,
    functions: EconFunctionsMapping,
    bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    transition_laws: TransitionLaws,
    support_axes: SupportAxes,
    v_interpolation_info: VInterpolationInfo,
    co_map_state_names: tuple[StateName, ...],
) -> _TargetContinuation:
    """Build one target's continuation machinery.

    A law that carries weights contributes a node axis to the interpolated value
    function either way, but only a lottery's weights are probabilities. The two
    groups are kept apart here, with the lottery axes product-mapped first, so
    the basis axes sit at the tail and can be contracted before the certainty
    equivalent ever sees the surface.

    Args:
        target_regime_name: Regime the continuation leads into.
        functions: Immutable mapping of function names to internal user functions.
        bundle: This target's unqualified `next_<state>` transition functions.
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        support_axes: Immutable mapping of target regime names to their private
            node axes.
        v_interpolation_info: The target's V-interpolation info.
        co_map_state_names: Tuple of state names co-mapped with the continuation V.

    Returns:
        The target's continuation machinery.

    """
    lottery_variables = tuple(
        key for key in bundle if is_stochastic(transition_laws, target_regime_name, key)
    )
    basis_variables = tuple(
        key
        for key in bundle
        if is_interpolation_basis(transition_laws, target_regime_name, key)
    )
    node_variables = (*lottery_variables, *basis_variables)

    V_arr_name = "next_V_arr"
    next_V_interpolator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        state_prefix="next_",
        V_arr_name=V_arr_name,
        co_map_state_names=co_map_state_names,
    )

    # A law reading one of this target's own draws has one value per node, so it
    # is resolved inside the node axes rather than once ahead of them.
    draw_dependent_names = _draw_dependent_law_names(
        bundle=bundle, functions=functions, stochastic_names=lottery_variables
    )
    if draw_dependent_names:
        _fail_if_a_read_draw_has_no_nodes_yet(
            target_regime_name=target_regime_name,
            stochastic_names=lottery_variables,
            draw_dependent_names=draw_dependent_names,
            v_interpolation_info=v_interpolation_info,
        )
        next_V_interpolator = _get_interpolator_resolving_draws(
            next_V_interpolator=next_V_interpolator,
            bundle=bundle,
            functions=functions,
            stochastic_names=lottery_variables,
            draw_dependent_names=draw_dependent_names,
            node_values=MappingProxyType(
                {
                    name: v_interpolation_info.discrete_states[
                        name.removeprefix("next_")
                    ].to_jax()
                    for name in lottery_variables
                }
            ),
        )

    return _TargetContinuation(
        next_states=get_next_state_function_for_solution(
            functions=functions,
            transitions=bundle,
            targets=[key for key in bundle if key not in draw_dependent_names],
        ),
        lottery_weights=get_next_stochastic_weights_function(
            functions=functions,
            transitions=bundle,
            transition_laws=transition_laws,
            regime_name=target_regime_name,
        ),
        basis_weights=get_next_interpolation_basis_weights_function(
            functions=functions,
            transitions=bundle,
            transition_laws=transition_laws,
            regime_name=target_regime_name,
        ),
        joint_lottery_weights=_get_joint_weights_function(
            regime_name=target_regime_name, variables=lottery_variables
        ),
        joint_basis_weights=_get_joint_weights_function(
            regime_name=target_regime_name, variables=basis_variables
        ),
        next_V=productmap(
            func=next_V_interpolator,
            variables=node_variables,
            batch_sizes=dict.fromkeys(node_variables, 0),
        ),
        support_axes=support_axes.get(target_regime_name, MappingProxyType({})),
        extra_param_names=frozenset(
            get_union_of_args([next_V_interpolator]) - set(bundle) - {V_arr_name}
        ),
        has_lottery_axes=bool(lottery_variables),
        n_basis_axes=len(basis_variables),
        draw_dependent_names=frozenset(draw_dependent_names),
    )


def _scalar_target_contribution(
    *,
    scalar_targets: tuple[RegimeName, ...],
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    active_regime_probs: Mapping[RegimeName, FloatND],
    as_lottery: bool,
    zero: FloatND,
) -> tuple[FloatND, list[FloatND], list[FloatND], FloatND]:
    """Seed the continuation accumulators with the stateless targets.

    A target carrying no state has a rank-zero value function: there is no next
    state to evaluate it at and no stochastic node to average over, so it
    contributes exactly one node whose only weight is the probability of going
    there. Under a certainty equivalent that node joins the joint lottery, so it
    is transformed together with every other target's nodes rather than on its
    own; under the linear expectation it is added straight to the running
    certainty equivalent.

    Args:
        scalar_targets: Targets active next period that carry no state.
        next_regime_to_V_arr: Mapping of target regime names to next period's
            value function arrays.
        active_regime_probs: Mapping of target regime names to their regime
            transition probabilities.
        as_lottery: Whether a nonlinear certainty equivalent aggregates the
            continuation, so the nodes must be handed over unaggregated.
        zero: Zero at the shape and dtype of the value being built up.

    Returns:
        Tuple of the seeded certainty equivalent, the lottery values, their
        weights, and the probability mass these targets represent.

    """
    CE = zero
    values: list[FloatND] = []
    weights: list[FloatND] = []
    probability_mass = zero
    for target_regime_name in scalar_targets:
        scalar_V = next_regime_to_V_arr[target_regime_name]
        prob = active_regime_probs[target_regime_name]
        # A stateless target contributes to the represented mass on either
        # route, so the linear fast path divides by the mass of *every* target
        # it summed, not just the ones carrying state.
        probability_mass = probability_mass + prob
        # Same rule on either route: a target reached with no mass contributes
        # nothing, and its value is neutralized rather than its product masked.
        # `-inf` is the ordinary value of a state where every action is
        # infeasible, so `0 * -inf` is otherwise how a single unreachable target
        # takes the reachable ones down with it.
        if as_lottery:
            node = jnp.ravel(scalar_V)
            values.append(jnp.where(prob == 0, jnp.zeros_like(node), node))
            weights.append(prob * jnp.ones_like(node))
        else:
            CE = CE + zero_safe_weighted_term(prob, scalar_V)
    return CE, values, weights, probability_mass


def _expectation_over_stochastic_nodes(*, values: FloatND, weights: FloatND) -> FloatND:
    """Return the weighted mean of one target's continuation over its nodes.

    Normalized explicitly rather than with `jnp.average`, for the reason
    `_as_lottery` states: a target whose joint weights carry no mass
    contributes no branch, and must not contribute NaN either — every target
    enters the same continuation, so a NaN here would destroy the
    well-specified targets beside it.

    A single node of probability exactly zero is dropped the same way, and for
    the same reason: an event that cannot occur says nothing about the
    expectation, so whatever it carries -- `-inf` at a state where every action
    is infeasible is the ordinary case -- must not reach the sum. A node with
    *positive* probability keeps its value, infinite or not.
    """
    weight_sum = jnp.sum(weights)
    safe_weight_sum = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
    return jnp.sum(zero_safe_weighted_term(weights, values)) / safe_weight_sum


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
    regime_name: RegimeName,
    variables: tuple[TransitionFunctionName, ...],
) -> Callable[..., FloatND]:
    """Get function that calculates the joint weights over one group of laws.

    This function takes the weights of the individual variables and multiplies
    them together to get the joint weights on their product space.

    The group is passed in as an ordered tuple rather than re-derived from the
    transition laws, because the caller productmaps the value function over the
    same tuple: one ordering fixes both the axes of the value surface and the
    axes of the weights, so the two cannot drift apart.

    Args:
        regime_name: Name of the target regime.
        variables: Ordered unqualified `next_<state>` names whose weights to
            multiply, in the order their axes appear on the value surface.

    Returns:
        A function that computes the outer product of the variables' weights.

    """
    arg_names = [f"weight_{regime_name}__{key}" for key in variables]

    @with_signature(args=arg_names)
    def _outer(**kwargs: Float1D) -> FloatND:
        weights = jnp.array(list(kwargs.values()))
        return jnp.prod(weights)

    variables = tuple(arg_names)
    return productmap(
        func=_outer, variables=variables, batch_sizes=dict.fromkeys(variables, 0)
    )


def _get_deterministic_transitions(
    *,
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
) -> tuple[
    Mapping[TransitionFunctionName, TransitionFunction],
    frozenset[TransitionFunctionName],
]:
    """Merge the deterministic `next_<state>` transitions across all targets.

    Iterates every target bundle, not just this period's targets: the within-
    period durable law (`next_<durable>`) lives in the source regime's own
    self-transition bundle and is needed even in periods bound for a terminal
    target that does not carry it. Own-regime within-period laws are
    target-independent, so the first occurrence of each `next_<state>` name is
    kept. Stochastic transitions are excluded — a within-period utility or
    constraint cannot read an unrealised stochastic next state.

    Returns the merged mapping and the set of `next_<state>` names that appear in
    more than one target bundle with non-identical implementations. The merge
    keeps one of them, so a within-period utility or constraint reading such a
    name would silently bind one target's law; the caller rejects the model if a
    conflicting name is actually read by the decision evaluation.

    Non-identity is tested by object identity (`is not`), not structural
    equality. This is a conservative proxy that relies on the canonicalization
    pipeline installing the *same* function object for a target-independent
    own-regime within-period law across every bundle: a shared reference is
    correctly seen as non-conflicting, and a distinct object genuinely signals a
    different target's law. Two behaviourally-equal but distinct objects would be
    over-reported as conflicting — harmless, since the conflict set only matters
    for names the decision evaluation actually reads.

    Returns:
        Tuple of the immutable merged `next_<state>` mapping and the frozenset of
        conflicting `next_<state>` names.
    """
    merged: dict[TransitionFunctionName, TransitionFunction] = {}
    conflicting: set[TransitionFunctionName] = set()
    for target_regime_name, bundle in transitions.items():
        for name, func in bundle.items():
            if is_stochastic(transition_laws, target_regime_name, name):
                continue
            if name in merged and merged[name] is not func:
                conflicting.add(name)
            merged.setdefault(name, func)
    return MappingProxyType(merged), frozenset(conflicting)


def _get_U_and_F(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction] = (
        MappingProxyType({})
    ),
    conflicting_deterministic_transition_names: frozenset[
        TransitionFunctionName
    ] = frozenset(),
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Get the instantaneous utility and feasibility function.

    Note:
    -----
    U may depend on all kinds of other functions (taxes, transfers, ...), which will be
    executed if they matter for the value of U.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        deterministic_transitions: Mapping of `next_<state>` names to deterministic
            own-regime transition functions, made available so within-period utility
            or feasibility that reads a chosen next state (the NEGM service-flow
            `next_<durable>`, or a budget constraint reading it) resolves it from the
            current states and actions. Pruned away when unread, so the grid-search
            path is unchanged.
        conflicting_deterministic_transition_names: Frozenset of `next_<state>`
            names whose deterministic law differs across target bundles. A model is
            rejected if any of them is read by the within-period decision (utility
            or feasibility), because the merged law would disagree with the
            simulate state-update.

    Returns:
        The instantaneous utility and feasibility function.

    """
    combined = {
        "feasibility": _get_feasibility(
            functions=functions,
            constraints=constraints,
            deterministic_transitions=deterministic_transitions,
        ),
        **dict(deterministic_transitions),
        **dict(functions),
    }
    _fail_if_conflicting_transition_is_read(
        combined=combined,
        targets=["utility", "feasibility"],
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
    )
    return concatenate_functions(
        functions=combined,
        targets=["utility", "feasibility"],
        enforce_signature=False,
        set_annotations=True,
    )


def _fail_if_conflicting_transition_is_read(
    *,
    combined: Mapping[FunctionName, Callable[..., Any]],
    targets: list[FunctionName],
    conflicting_deterministic_transition_names: frozenset[TransitionFunctionName],
) -> None:
    """Reject a model whose decision reads a target-dependent `next_<state>` law.

    A `next_<state>` whose deterministic law differs across target bundles is
    merged down to one implementation; binding it into the decision DAG while the
    simulate state-update uses the per-target law produces a silent disagreement.
    Raise naming each such state actually read by `targets`.

    Args:
        combined: Mapping of function names to the functions assembled for the
            decision DAG.
        targets: List of target function names the decision evaluates.
        conflicting_deterministic_transition_names: Frozenset of `next_<state>`
            names with non-identical implementations across target bundles.
    """
    if not conflicting_deterministic_transition_names:
        return
    read_names = get_ancestors(combined, targets, include_targets=True)
    offending = sorted(conflicting_deterministic_transition_names & read_names)
    if offending:
        names = ", ".join(offending)
        msg = (
            "Within-period utility or feasibility reads a target-dependent "
            f"deterministic state law ({names}), but its implementation differs "
            "across target regimes. The decision DAG would bind one target's law "
            "while the simulate state-update uses the right one, so they would "
            "disagree silently. Make the law identical across all targets that "
            "carry the state, or stop reading the chosen next state in the "
            "within-period utility/feasibility."
        )
        raise ValueError(msg)


def _get_feasibility(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction] = (
        MappingProxyType({})
    ),
) -> ConstraintFunction:
    """Create a function that combines all constraint functions into a single one.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        deterministic_transitions: Mapping of `next_<state>` names to deterministic
            transition functions, so a constraint reading a chosen next state (the
            NEGM budget constraint reading `next_<durable>`) resolves it. Pruned when
            unread.

    Returns:
        The combined constraint function (feasibility).

    """
    if constraints:
        combined_constraint = concatenate_functions(
            functions=dict(deterministic_transitions)
            | dict(constraints)
            | dict(functions),
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


# Gross departures from unit regime mass are a specification error, not rounding:
# every aggregation route divides the continuation by the mass it received, so the
# lost mass is divided straight back out and the survivors renormalize. The solved
# value function then comes back finite, plausible, and independent of what went
# missing. `validate_transitions` catches it, but `log_level="off"` skips that, so
# this poisons the arithmetic itself and cannot be gated away.
#
# The tolerance is deliberately loose. It is a backstop against a wrong model, not
# a numerical check: `1e-3` never fires on accumulated float error over a handful
# of targets, while a mass of 0.977 — small enough to look plausible, large enough
# to move every value in the model — becomes NaN.
_MAX_REGIME_MASS_DEVIATION = 1.0e-3


def _regime_mass_is_unit(probability_mass: FloatND) -> BoolND:
    """Whether the represented regime mass is unit mass, within tolerance."""
    return jnp.abs(probability_mass - 1.0) <= _MAX_REGIME_MASS_DEVIATION


def _aggregate_joint_lottery(
    *,
    certainty_equivalent: CertaintyEquivalent,
    lottery_values: Sequence[FloatND],
    lottery_weights: Sequence[FloatND],
    ce_flat_param_names: Mapping[str, str],
    states_actions_params: Mapping[str, Any],
) -> FloatND:
    """Aggregate the continuation nodes of every retained target in one piece.

    Args:
        certainty_equivalent: The regime's certainty equivalent.
        lottery_values: Sequence of per-target continuation values.
        lottery_weights: Sequence of per-target node weights, already scaled by
            the target's regime-transition probability.
        ce_flat_param_names: Mapping of certainty-equivalent argument names to
            their flat parameter names.
        states_actions_params: Mapping of states, actions, age, period, and flat
            regime params.

    Returns:
        The aggregated continuation value.

    """
    return certainty_equivalent.aggregate(
        values=jnp.concatenate(list(lottery_values)),
        weights=jnp.concatenate(list(lottery_weights)),
        # The params template types every certainty-equivalent parameter as a
        # float, so its runtime values are float arrays.
        params=cast(
            "Mapping[str, FloatND]",
            {
                arg: states_actions_params[flat_name]
                for arg, flat_name in ce_flat_param_names.items()
            },
        ),
    )


def _unit_regime_mass_or_nan(probability_mass: FloatND) -> FloatND:
    """Return the mass itself, or NaN where it is not unit mass.

    For the per-target route, which divides by the mass it accumulated.
    """
    return jnp.where(_regime_mass_is_unit(probability_mass), probability_mass, jnp.nan)
