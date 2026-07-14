from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags import concatenate_functions, get_ancestors, with_signature

from _lcm.certainty_equivalent import CertaintyEquivalent, resolve_certainty_equivalent
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.h_dag import _get_build_H_kwargs
from _lcm.regime_building.next_state import (
    get_next_state_function_for_solution,
    get_next_stochastic_weights_function,
)
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.regime_building.zero_safe import zero_safe_average, zero_safe_weighted_term
from _lcm.typing import (
    ConstraintFunction,
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
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
from lcm.typing import BoolND, Float1D, FloatND


def get_Q_and_F(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    co_map_state_names: tuple[StateName, ...] = (),
    certainty_equivalent: CertaintyEquivalent | None = None,
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    `age` and `period` are runtime arguments (via `**states_actions_params`),
    not closure constants. This allows periods with the same target
    configuration to share a single JIT-compiled function.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        period_targets: Target regimes whose continuation enters E[V]
            this period (reachable, with state laws, active next period).
        transitions: Immutable mapping of transition names to transition functions.
        stochastic_transition_names: Frozenset of stochastic transition function names.
        compute_regime_transition_probs: Regime transition probability function
            for solve.
        regime_to_v_interpolation_info: Mapping of regime names to V-interpolation
            info.
        co_map_state_names: Tuple of state names co-mapped with the continuation V —
            their axes are sliced off each `next_V_arr` leaf by the backward-induction
            co-map, so their coordinates are dropped from the interpolation. Only fixed
            (never-transitioning) distributed states qualify.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    deterministic_transitions, conflicting_deterministic_transition_names = (
        _get_deterministic_transitions(
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
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
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[RegimeName, frozenset[str]] = {}

    for target_regime_name in period_targets:
        # Transitions from the current regime to the target regime
        bundle = transitions[target_regime_name]

        # Functions required to calculate the expected continuation values
        state_transitions[target_regime_name] = get_next_state_function_for_solution(
            functions=functions,
            transitions=bundle,
        )
        next_stochastic_states_weights[target_regime_name] = (
            get_next_stochastic_weights_function(
                functions=functions,
                transitions=bundle,
                stochastic_transition_names=stochastic_transition_names,
                regime_name=target_regime_name,
            )
        )
        joint_weights_from_marginals[target_regime_name] = _get_joint_weights_function(
            transitions=bundle,
            stochastic_transition_names=stochastic_transition_names,
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
            key for key in bundle if key in stochastic_transition_names
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
    ce, ce_transform_flat_names, ce_inverse_flat_names = resolve_certainty_equivalent(
        certainty_equivalent
    )

    # Co-mapped states are sliced off each `next_V_arr` leaf by the backward-
    # induction co-map, so their `next_`-prefixed coordinates are not passed to
    # the interpolator (which no longer indexes those axes).
    _co_map_next_names = frozenset(f"next_{name}" for name in co_map_state_names)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[
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
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        # COLLECTIVE-REGIMES (E2): F_arr is built here, before and independently
        # of Q (it never reads E_next_V). A value-aware mask cannot stay here:
        # it needs per-stakeholder Q^s, so E2 splits this into (i) build the
        # state-independent F here, (ii) compute Q^s, (iii) `mask = F ∧ g(...)`
        # applied in max_Q_over_a. This site also returns the explicit dissolution
        # flag D = 1[mask empty], distinct from a numeric -inf. See design doc
        # §2 (E2) / §3.
        U_arr, F_arr = U_and_F(**states_actions_params)
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in period_targets}
        )

        E_next_V = jnp.zeros_like(U_arr)
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
                    if name not in _co_map_next_names
                },
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )
            if ce is not None:
                next_V_at_stochastic_states_arr = ce.transform(
                    value=next_V_at_stochastic_states_arr,
                    **{
                        arg: states_actions_params[flat_name]
                        for arg, flat_name in ce_transform_flat_names.items()
                    },
                )

            # We then take the weighted average of the next value function at the
            # stochastic states to get the expected next value function.
            # Zero-safe: a zero-probability stochastic node next to an
            # admissible on-path -inf must not turn the average into a nan.
            next_V_expected_arr = zero_safe_average(
                next_V_at_stochastic_states_arr,
                weights=joint_next_stochastic_states_weights,
            )
            # Zero-safe: an inactive regime-transition target (probability
            # exactly 0) next to an admissible on-path -inf continuation must
            # not turn this mixture term into a nan either.
            E_next_V = E_next_V + zero_safe_weighted_term(
                active_regime_probs[target_regime_name], next_V_expected_arr
            )

        if ce is not None:
            E_next_V = ce.inverse(
                value=E_next_V,
                **{
                    arg: states_actions_params[flat_name]
                    for arg, flat_name in ce_inverse_flat_names.items()
                },
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
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    certainty_equivalent: CertaintyEquivalent | None = None,
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
        period_targets: Target regimes whose continuation enters E[V]
            this period (reachable, with state laws, active next period).
        transitions: Immutable mapping of target regime names to state transition
            functions.
        stochastic_transition_names: Frozenset of stochastic transition function
            names.
        compute_regime_transition_probs: Callable returning regime transition
            probabilities for the current regime.
        regime_to_v_interpolation_info: Immutable mapping of regime names to
            V-interpolation info.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.

    Returns:
        Closure returning `(U_arr, F_arr, E_next_V, Q_arr, active_regime_probs)`.

    """
    deterministic_transitions, conflicting_deterministic_transition_names = (
        _get_deterministic_transitions(
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
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
    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[RegimeName, frozenset[str]] = {}

    for target_regime_name in period_targets:
        bundle = transitions[target_regime_name]
        state_transitions[target_regime_name] = get_next_state_function_for_solution(
            functions=functions,
            transitions=bundle,
        )
        next_stochastic_states_weights[target_regime_name] = (
            get_next_stochastic_weights_function(
                functions=functions,
                transitions=bundle,
                stochastic_transition_names=stochastic_transition_names,
                regime_name=target_regime_name,
            )
        )
        joint_weights_from_marginals[target_regime_name] = _get_joint_weights_function(
            transitions=bundle,
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
            get_union_of_args([next_V_interpolator]) - set(bundle) - {V_arr_name}
        )
        stochastic_variables = tuple(
            key for key in bundle if key in stochastic_transition_names
        )
        next_V[target_regime_name] = productmap(
            func=next_V_interpolator,
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    ce, ce_transform_flat_names, ce_inverse_flat_names = resolve_certainty_equivalent(
        certainty_equivalent
    )

    arg_names_of_compute_intermediates = _get_arg_names_of_Q_and_F(
        deps=[
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
            "MappingProxyType[RegimeName, FloatND]]"
        ),
    )
    def compute_intermediates(
        next_regime_to_V_arr: FloatND,
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[
        FloatND, FloatND, FloatND, FloatND, MappingProxyType[RegimeName, FloatND]
    ]:
        """Compute all Q_and_F intermediates."""
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        U_arr, F_arr = U_and_F(**states_actions_params)
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in period_targets}
        )

        E_next_V = jnp.zeros_like(U_arr)
        for target_regime_name in period_targets:
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
            if ce is not None:
                next_V_stoch = ce.transform(
                    value=next_V_stoch,
                    **{
                        arg: states_actions_params[flat_name]
                        for arg, flat_name in ce_transform_flat_names.items()
                    },
                )
            # Zero-safe, mirroring `get_Q_and_F` above: see the guards there.
            contribution = zero_safe_average(next_V_stoch, weights=joint)
            E_next_V = E_next_V + zero_safe_weighted_term(
                active_regime_probs[target_regime_name], contribution
            )

        if ce is not None:
            E_next_V = ce.inverse(
                value=E_next_V,
                **{
                    arg: states_actions_params[flat_name]
                    for arg, flat_name in ce_inverse_flat_names.items()
                },
            )

        Q_arr = functions["H"](
            utility=U_arr,
            E_next_V=E_next_V,
            **_get_build_H_kwargs(functions)(states_actions_params),
        )

        return U_arr, F_arr, E_next_V, Q_arr, active_regime_probs

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


def get_Q_and_F_terminal_collective(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    stakeholders: tuple[str, ...],
) -> QAndFFunction:
    """Terminal (Q, F) for a collective regime — stacked per-stakeholder U + shared F.

    COLLECTIVE-REGIMES (E1). Separate from `get_Q_and_F_terminal` so the singleton
    terminal path (shared with the simulate / compute-intermediates machinery) is
    byte-identical; this builder is used only at the collective solve site.

    Builds one `U^s`-and-`F` closure per stakeholder from its own `utility_<s>`
    DAG target (feasibility is regime-level, so it is identical across
    stakeholders — the first one is kept). The returned `Q_and_F` stacks the
    per-stakeholder utilities on a trailing stakeholder axis: for a scalar
    (state, action) cell it returns `U` of shape `(n_stakeholders,)` and a scalar
    `F`. After the action product-map in `get_max_Q_over_a`, `U` has shape
    `(*action_axes, n_stakeholders)` and `F` `(*action_axes,)`; the stakeholder
    branch there splits `U` by stakeholder and calls `collective_readout`.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions;
            carries `utility_<s>` for each stakeholder in place of `utility`.
        constraints: Immutable mapping of constraint names to internal user functions.
        stakeholders: Ordered stakeholder names; fixes the trailing-axis order.

    Returns:
        A function computing the stacked per-stakeholder utilities (Q) and the
        shared feasibility mask (F) for a terminal collective period.

    """
    U_and_F_by_stakeholder = {
        stakeholder: _get_U_and_F(
            functions=functions,
            constraints=constraints,
            utility_name=f"utility_{stakeholder}",
        )
        for stakeholder in stakeholders
    }

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=list(U_and_F_by_stakeholder.values()),
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: FloatND,  # noqa: ARG001
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[FloatND, BoolND]:
        """Stacked per-stakeholder utilities and the shared feasibility mask.

        Args:
            next_regime_to_V_arr: Unused in a terminal period; accepted so solve
                treats all periods uniformly.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple of the stacked per-stakeholder utility array (trailing
            stakeholder axis) and the shared feasibility mask.

        """
        U_arrays: list[FloatND] = []
        F_arr: BoolND | None = None
        for u_and_f in U_and_F_by_stakeholder.values():
            U_s, F_arr = u_and_f(**states_actions_params)
            U_arrays.append(jnp.asarray(U_s))
        U_stack = jnp.stack(U_arrays, axis=-1)
        return U_stack, jnp.asarray(F_arr)

    return Q_and_F


# COLLECTIVE-REGIMES (E2): the name under which the mapping of same-period
# reference regimes to their current-period V arrays enters the kernel
# signature. Only regimes declaring `same_period_refs` carry it.
SAME_PERIOD_V_ARG = "same_period_regime_to_V_arr"

# Internal argument names of the same-period reference interpolation; never
# surfaced in the kernel signature.
_REF_STATE_PREFIX = "__same_period_ref__"
_REF_V_ARR_NAME = "__same_period_ref_V_arr__"


@dataclass(frozen=True, kw_only=True)
class ResolvedSamePeriodRef:
    """Engine-side form of a user `SamePeriodRef`, resolved at model processing.

    COLLECTIVE-REGIMES (E2). The user declaration names a stakeholder; the
    engine resolves it to the index on the reference regime's trailing
    stakeholder axis (`None` for a singleton reference, whose V has no such
    axis).
    """

    regime: RegimeName
    """Name of the reference regime whose same-period V is read."""

    projection: Mapping[StateName, Callable[..., Any]]
    """Per-reference-state projection functions (user vocabulary, DAG-resolved)."""

    stakeholder_index: int | None
    """Index into the reference V's trailing stakeholder axis, or `None`."""


def _build_same_period_ref_reader(
    *,
    ref: ResolvedSamePeriodRef,
    v_interpolation_info: VInterpolationInfo,
    functions: EconFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction],
) -> Callable[..., FloatND]:
    """Build the reader of one same-period reference value at a (state, action) cell.

    COLLECTIVE-REGIMES (E2). Each projection entry is concatenated with the
    regime's function DAG (so it may read states, actions, helper functions,
    and the merged deterministic `next_<state>` laws), producing one coordinate
    per reference state; the reference regime's CURRENT-period V array — passed
    per solve step under `SAME_PERIOD_V_ARG` — is then interpolated at those
    coordinates with the ordinary V-interpolation machinery
    (`get_V_interpolator`), sliced to the named stakeholder first when the
    reference is collective. The returned callable's signature carries only
    user-level names (states / actions / params reached by the projections,
    plus `SAME_PERIOD_V_ARG`), so the kernel signature stays clean.

    A projection produces a genuine VALUE for every reference state
    (interpolation-worthy, possibly off-grid) — unlike the ordinary
    continuation-value path, which always feeds a process axis its exact
    on-grid Markov-chain index. When the reference regime carries a
    non-folded process state (`_ContinuousStochasticProcess`, classified
    `discrete_states` for the Markov-chain solve path but read here as a
    genuine value), `get_V_interpolator`'s process-aware mode
    (`interpolate_process_axes=True`) is used so that axis is linearly
    interpolated instead of integer-looked-up; a reference regime without a
    process state is unaffected (`interpolate_process_axes=False`, the
    ordinary path, byte-identical).
    """
    _reference_has_process_axis = any(
        isinstance(grid, _ContinuousStochasticProcess)
        for grid in v_interpolation_info.discrete_states.values()
    )
    interpolator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        state_prefix=_REF_STATE_PREFIX,
        V_arr_name=_REF_V_ARR_NAME,
        interpolate_process_axes=_reference_has_process_axis,
    )
    dag_pool = {
        **dict(deterministic_transitions),
        **{k: v for k, v in functions.items() if k != "H"},
    }
    projection_funcs: dict[StateName, Callable[..., FloatND]] = {}
    projection_args: dict[StateName, tuple[str, ...]] = {}
    for state_name in v_interpolation_info.state_names:
        target = f"{_REF_STATE_PREFIX}{state_name}"
        projection_funcs[state_name] = concatenate_functions(
            functions={**dag_pool, target: ref.projection[state_name]},
            targets=target,
            enforce_signature=False,
            set_annotations=True,
        )
        projection_args[state_name] = tuple(
            get_union_of_args([projection_funcs[state_name]])
        )
    coordinate_names = {
        f"{_REF_STATE_PREFIX}{state}" for state in v_interpolation_info.state_names
    }
    # Extra interpolator inputs beyond the coordinates and the V array (e.g.
    # runtime-supplied irregular-grid points) pass through from the cell kwargs.
    interpolator_extra = tuple(
        get_union_of_args([interpolator]) - coordinate_names - {_REF_V_ARR_NAME}
    )
    arg_names = sorted(
        {arg for args in projection_args.values() for arg in args}
        | set(interpolator_extra)
        | {SAME_PERIOD_V_ARG}
    )

    @with_signature(args=arg_names, return_annotation="FloatND")
    def read_reference_value(**kwargs: _ParamsLeaf) -> FloatND:
        same_period_V = cast("Mapping[RegimeName, FloatND]", kwargs[SAME_PERIOD_V_ARG])
        V_ref = same_period_V[ref.regime]
        if ref.stakeholder_index is not None:
            # A collective reference V carries a trailing stakeholder axis;
            # read the declared stakeholder's slice (state axes only remain).
            V_ref = V_ref[..., ref.stakeholder_index]
        coordinates = {
            f"{_REF_STATE_PREFIX}{state}": projection_funcs[state](
                **{arg: kwargs[arg] for arg in projection_args[state]}
            )
            for state in v_interpolation_info.state_names
        }
        return interpolator(
            **coordinates,
            **{arg: kwargs[arg] for arg in interpolator_extra},
            **{_REF_V_ARR_NAME: V_ref},
        )

    return read_reference_value


def get_Q_and_F_collective(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    stakeholders: tuple[str, ...],
    co_map_state_names: tuple[StateName, ...] = (),
    value_constraints: ConstraintFunctionsMapping = MappingProxyType({}),
    same_period_refs: Mapping[str, ResolvedSamePeriodRef] = MappingProxyType({}),
) -> QAndFFunction:
    """Non-terminal (Q, F) for a collective regime — per-stakeholder continuation.

    COLLECTIVE-REGIMES (E1, slice 2). Separate from `get_Q_and_F` so the
    singleton path is byte-identical; this builder is used only at the
    collective solve site.

    Per stakeholder `s`, computes `Q^s = H(u^s, E[V'^s])` with the shared
    Bellman aggregator `H` (the default `H_linear` applies `u + beta * E[V']`
    elementwise, so every stakeholder is discounted with the SAME beta). Each
    transition target must itself be a collective regime with the identical
    `stakeholders` tuple (validated at model processing), so its
    `next_V_arr` leaf carries the trailing stakeholder axis. The continuation
    interpolates the target's V over STATE axes only: the interpolator is
    evaluated once per stakeholder on the leaf's slice `next_V_arr[..., s]` and
    the results are re-stacked on a trailing axis, so the stakeholder axis
    provably rides through the stochastic-node product-map (which stacks its
    mapped axes at the front) as the last axis. For a scalar (state, action)
    cell the returned `Q` has shape `(n_stakeholders,)` while `F` is scalar;
    after the action product-map in `get_max_Q_over_a`, `Q` is
    `(*action_axes, n_stakeholders)` and `F` `(*action_axes,)` — exactly what
    the stakeholder branch there (`collective_readout`) consumes.

    No taste shocks and no nonlinear certainty equivalent: both are rejected at
    regime construction for collective regimes.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user
            functions; carries `utility_<s>` for each stakeholder in place of
            `utility`, plus the shared `H`.
        constraints: Immutable mapping of constraint names to internal user
            functions.
        period_targets: Target regimes whose continuation enters E[V^s] this
            period (all collective with the identical stakeholder tuple).
        transitions: Immutable mapping of transition names to transition
            functions.
        stochastic_transition_names: Frozenset of stochastic transition function
            names.
        compute_regime_transition_probs: Regime transition probability function
            for solve (stakeholder-independent — per-stakeholder gates are E3').
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info (state axes only; the stakeholder axis is not
            an interpolation axis).
        stakeholders: Ordered stakeholder names; fixes the trailing-axis order.
        co_map_state_names: Tuple of state names co-mapped with the continuation
            V (see `get_Q_and_F`).
        value_constraints: Immutable mapping of value-constraint names to
            predicates (params already renamed to qnames). COLLECTIVE-REGIMES
            (E2): evaluated AFTER the per-stakeholder `Q^s`, each predicate may
            read `Q_<s>` per stakeholder, the `same_period_refs` reference
            values, and ordinary states / actions / functions / params via the
            DAG; the results are ANDed into the feasibility mask, so the
            household argmax runs over `F ∧ g(Q^s, V_ref, ...)` and an
            all-infeasible cell publishes the dissolution flag `D` downstream.
        same_period_refs: Immutable mapping of reference-value names to resolved
            same-period reference declarations. When non-empty, the returned
            `Q_and_F` carries the extra argument `SAME_PERIOD_V_ARG` — the
            mapping of reference regime names to their CURRENT-period V arrays,
            supplied per period by the solve loop (which orders the period's
            regimes so references are solved first).

    Returns:
        A function computing the stacked per-stakeholder state-action values
        (trailing stakeholder axis) and the shared feasibility mask for a
        non-terminal collective period.

    """
    deterministic_transitions, conflicting_deterministic_transition_names = (
        _get_deterministic_transitions(
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
        )
    )
    U_and_F_by_stakeholder = {
        stakeholder: _get_U_and_F(
            functions=functions,
            constraints=constraints,
            deterministic_transitions=deterministic_transitions,
            conflicting_deterministic_transition_names=(
                conflicting_deterministic_transition_names
            ),
            utility_name=f"utility_{stakeholder}",
        )
        for stakeholder in stakeholders
    }
    n_stakeholders = len(stakeholders)

    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[RegimeName, frozenset[str]] = {}

    for target_regime_name in period_targets:
        bundle = transitions[target_regime_name]
        state_transitions[target_regime_name] = get_next_state_function_for_solution(
            functions=functions,
            transitions=bundle,
        )
        next_stochastic_states_weights[target_regime_name] = (
            get_next_stochastic_weights_function(
                functions=functions,
                transitions=bundle,
                stochastic_transition_names=stochastic_transition_names,
                regime_name=target_regime_name,
            )
        )
        joint_weights_from_marginals[target_regime_name] = _get_joint_weights_function(
            transitions=bundle,
            stochastic_transition_names=stochastic_transition_names,
            regime_name=target_regime_name,
        )
        V_arr_name = "next_V_arr"
        next_V_interpolator = get_V_interpolator(
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
            state_prefix="next_",
            V_arr_name=V_arr_name,
            co_map_state_names=co_map_state_names,
        )
        next_V_extra_param_names[target_regime_name] = frozenset(
            get_union_of_args([next_V_interpolator]) - set(bundle) - {V_arr_name}
        )
        stochastic_variables = tuple(
            key for key in bundle if key in stochastic_transition_names
        )
        next_V[target_regime_name] = productmap(
            func=_get_stakeholder_sliced_interpolator(
                base_interpolator=next_V_interpolator,
                V_arr_name=V_arr_name,
                n_stakeholders=n_stakeholders,
            ),
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    _build_H_kwargs = _get_build_H_kwargs(functions)
    _co_map_next_names = frozenset(f"next_{name}" for name in co_map_state_names)

    # COLLECTIVE-REGIMES (E2): build the same-period reference readers and the
    # value-constraint evaluators once; their engine-supplied arguments —
    # `Q_<s>` and the reference-value names — are excluded from the kernel
    # signature and bound per (state, action) cell inside `Q_and_F`.
    value_constraint_machinery = _build_value_constraint_machinery(
        value_constraints=value_constraints,
        same_period_refs=same_period_refs,
        stakeholders=stakeholders,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        functions=functions,
        deterministic_transitions=deterministic_transitions,
    )

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[
            *list(U_and_F_by_stakeholder.values()),
            compute_regime_transition_probs,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
            *list(value_constraint_machinery.evaluators.values()),
            *list(value_constraint_machinery.reference_readers.values()),
        ],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=value_constraint_machinery.engine_supplied_names,
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: FloatND,
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[FloatND, BoolND]:
        """Per-stakeholder state-action values and the shared feasibility mask.

        Args:
            next_regime_to_V_arr: The next period's value function arrays, each
                target leaf carrying a trailing stakeholder axis.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple of the stacked per-stakeholder state-action value array
            (trailing stakeholder axis) and the shared feasibility mask.

        """
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        U_arrays: list[FloatND] = []
        F_arr: BoolND | None = None
        for u_and_f in U_and_F_by_stakeholder.values():
            U_s, F_arr = u_and_f(**states_actions_params)
            U_arrays.append(jnp.asarray(U_s))
        U_stack = jnp.stack(U_arrays, axis=-1)
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in period_targets}
        )

        E_next_V = jnp.zeros_like(U_stack)
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

            extra_kw = {
                k: states_actions_params[k]
                for k in next_V_extra_param_names[target_regime_name]
            }
            # Shape (*stochastic_axes, n_stakeholders): the product-map stacks
            # the stochastic-node axes at the front, the stakeholder axis stays
            # trailing.
            next_V_at_stochastic_states_arr = next_V[target_regime_name](
                **{
                    name: val
                    for name, val in next_states.items()
                    if name not in _co_map_next_names
                },
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )

            # Per-stakeholder weighted average over the stochastic nodes only —
            # never over the trailing stakeholder axis. Zero-safe: see the
            # guards in `get_Q_and_F` above.
            next_V_expected_arr = zero_safe_average(
                next_V_at_stochastic_states_arr.reshape(-1, n_stakeholders),
                axis=0,
                weights=jnp.asarray(joint_next_stochastic_states_weights).reshape(-1),
            )
            E_next_V = E_next_V + zero_safe_weighted_term(
                active_regime_probs[target_regime_name], next_V_expected_arr
            )

        # H applied on the stacked arrays is H per stakeholder: `utility` and
        # `E_next_V` share the trailing stakeholder axis and H's parameters
        # (e.g. the default `H_linear`'s discount factor) are shared across
        # stakeholders, so the elementwise aggregation is exactly
        # Q^s = H(u^s, E[V'^s], beta) with the same beta for every s.
        Q_arr = functions["H"](
            utility=U_stack,
            E_next_V=E_next_V,
            **_build_H_kwargs(states_actions_params),
        )

        # COLLECTIVE-REGIMES (E2): value-aware feasibility. Evaluated AFTER
        # Q^s — this is the reorder the singleton path never needs (there,
        # F is built before and independently of Q). Interpolate each declared
        # same-period reference value at the projected coordinates, then AND
        # every predicate — reading its own `Q_<s>` gathers, the reference
        # values, and ordinary cell kwargs — into the mask. The household
        # argmax downstream runs over the masked set; an all-infeasible cell
        # sets the dissolution flag D there (`collective_readout`).
        if value_constraint_machinery.evaluators:
            F_arr = _apply_value_constraints(
                machinery=value_constraint_machinery,
                Q_arr=jnp.asarray(Q_arr),
                # A constraint-less regime's F is the Python `True` scalar.
                F_arr=jnp.asarray(F_arr),
                states_actions_params=states_actions_params,
            )

        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


@dataclass(frozen=True, kw_only=True)
class _ValueConstraintMachinery:
    """Prebuilt E2 evaluation machinery closed over by a collective `Q_and_F`."""

    reference_readers: Mapping[str, Callable[..., FloatND]]
    """Per reference-value name, the same-period reference reader."""

    reference_reader_args: Mapping[str, tuple[str, ...]]
    """Each reader's argument names (fetched off the cell kwargs)."""

    evaluators: Mapping[str, Callable[..., BoolND]]
    """Per value-constraint name, the DAG-concatenated predicate."""

    evaluator_args: Mapping[str, tuple[str, ...]]
    """Each evaluator's argument names (split engine-supplied vs cell kwargs)."""

    q_value_index: Mapping[str, int]
    """`Q_<s>` argument name -> index on the trailing stakeholder axis."""

    engine_supplied_names: frozenset[str]
    """Names bound by the engine per cell — excluded from the kernel signature."""


def _build_value_constraint_machinery(
    *,
    value_constraints: ConstraintFunctionsMapping,
    same_period_refs: Mapping[str, ResolvedSamePeriodRef],
    stakeholders: tuple[str, ...],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    functions: EconFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction],
) -> _ValueConstraintMachinery:
    """Build the E2 reference readers and value-constraint evaluators once.

    COLLECTIVE-REGIMES (E2). Each evaluator is the predicate concatenated with
    the regime's function DAG (so it may read helper functions and the merged
    deterministic `next_<state>` laws, exactly like ordinary constraints); its
    engine-supplied arguments — `Q_<s>` and the reference-value names — are
    bound per (state, action) cell by `_apply_value_constraints`.
    """
    reference_readers: dict[str, Callable[..., FloatND]] = {}
    reference_reader_args: dict[str, tuple[str, ...]] = {}
    for ref_name, ref in same_period_refs.items():
        reader = _build_same_period_ref_reader(
            ref=ref,
            v_interpolation_info=regime_to_v_interpolation_info[ref.regime],
            functions=functions,
            deterministic_transitions=deterministic_transitions,
        )
        reference_readers[ref_name] = reader
        reference_reader_args[ref_name] = tuple(get_union_of_args([reader]))

    dag_pool = {
        **dict(deterministic_transitions),
        **{k: v for k, v in functions.items() if k != "H"},
    }
    evaluators: dict[str, Callable[..., BoolND]] = {}
    evaluator_args: dict[str, tuple[str, ...]] = {}
    for constraint_name, predicate in value_constraints.items():
        evaluator = concatenate_functions(
            functions={**dag_pool, constraint_name: predicate},
            targets=constraint_name,
            enforce_signature=False,
            set_annotations=True,
        )
        evaluators[constraint_name] = evaluator
        evaluator_args[constraint_name] = tuple(get_union_of_args([evaluator]))

    q_value_index = {f"Q_{s}": index for index, s in enumerate(stakeholders)}
    return _ValueConstraintMachinery(
        reference_readers=MappingProxyType(reference_readers),
        reference_reader_args=MappingProxyType(reference_reader_args),
        evaluators=MappingProxyType(evaluators),
        evaluator_args=MappingProxyType(evaluator_args),
        q_value_index=MappingProxyType(q_value_index),
        engine_supplied_names=(frozenset(q_value_index) | frozenset(reference_readers)),
    )


def _apply_value_constraints(
    *,
    machinery: _ValueConstraintMachinery,
    Q_arr: FloatND,
    F_arr: BoolND,
    # `object` values: besides ordinary `_ParamsLeaf` leaves, the cell kwargs
    # carry the same-period V mapping under `SAME_PERIOD_V_ARG`.
    states_actions_params: Mapping[str, object],
) -> BoolND:
    """AND every value constraint into the feasibility of one (state, action) cell.

    COLLECTIVE-REGIMES (E2). Reads each declared same-period reference value at
    the projected coordinates (the readers pull the current-period reference V
    arrays off `states_actions_params[SAME_PERIOD_V_ARG]`), then evaluates each
    predicate with its `Q_<s>` arguments gathered from the trailing stakeholder
    axis of `Q_arr`, its reference-value arguments, and its remaining arguments
    from the cell kwargs.
    """
    reference_values = {
        ref_name: reader(
            **{
                arg: states_actions_params[arg]
                for arg in machinery.reference_reader_args[ref_name]
            }
        )
        for ref_name, reader in machinery.reference_readers.items()
    }
    for constraint_name, evaluate in machinery.evaluators.items():
        predicate_kwargs: dict[str, object] = {}
        for arg in machinery.evaluator_args[constraint_name]:
            if arg in machinery.q_value_index:
                predicate_kwargs[arg] = Q_arr[..., machinery.q_value_index[arg]]
            elif arg in reference_values:
                predicate_kwargs[arg] = reference_values[arg]
            else:
                predicate_kwargs[arg] = states_actions_params[arg]
        F_arr = jnp.logical_and(F_arr, evaluate(**predicate_kwargs))
    return F_arr


def _get_stakeholder_sliced_interpolator(
    *,
    base_interpolator: Callable[..., FloatND],
    V_arr_name: str,
    n_stakeholders: int,
) -> Callable[..., FloatND]:
    """Evaluate a V-interpolator per stakeholder slice of a stacked V array.

    COLLECTIVE-REGIMES (E1, slice 2). The target regime's `next_V_arr` leaf has
    shape `(*target_state_axes, n_stakeholders)`; the base interpolator
    interpolates over the state axes of a plain `(*target_state_axes,)` array.
    Calling it once per stakeholder on the slice `next_V_arr[..., s]` and
    re-stacking on a trailing axis keeps the interpolation semantics untouched
    and puts the stakeholder axis last by construction — no axis bookkeeping
    can reorder it. The wrapper carries the base interpolator's exact argument
    names so the stochastic-variable product-map and the extra-param discovery
    treat it like the singleton interpolator.

    Args:
        base_interpolator: The singleton V-interpolator from
            `get_V_interpolator` (state axes only).
        V_arr_name: Name of the interpolator's value-array argument.
        n_stakeholders: Number of stakeholder slices on the trailing axis.

    Returns:
        A callable with the base interpolator's signature returning the
        per-stakeholder interpolated values, stakeholder axis trailing.

    """
    arg_names = tuple(get_union_of_args([base_interpolator]))

    @with_signature(args=arg_names, return_annotation="FloatND")
    def next_V_per_stakeholder(**kwargs: _ParamsLeaf) -> FloatND:
        stacked_V_arr = cast("FloatND", kwargs.pop(V_arr_name))
        return jnp.stack(
            [
                base_interpolator(**kwargs, **{V_arr_name: stacked_V_arr[..., s]})
                for s in range(n_stakeholders)
            ],
            axis=-1,
        )

    return next_V_per_stakeholder


def get_period_targets(
    *,
    period: int,
    transitions: TransitionFunctionsMapping,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
) -> tuple[RegimeName, ...]:
    """Return the target regimes whose continuation enters E[V] this period.

    The canonical transition bundles (`transitions` keys) carry exactly the
    reachable targets with at least one state law; the period filter keeps
    those active in the next period. A reachable target absent from the
    bundles has no states (its V is identically zero) and contributes
    nothing to the continuation.

    Args:
        period: The period to enumerate targets for.
        transitions: Immutable mapping of target regime names to their
            state transition functions.
        regimes_to_active_periods: Immutable mapping of regime names to
            their active period tuples.

    Returns:
        Tuple of this period's target regime names.

    """
    return tuple(
        regime_name
        for regime_name in transitions
        if period + 1 in regimes_to_active_periods.get(regime_name, ())
    )


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
    stochastic_transition_names: frozenset[TransitionFunctionName],
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


def _get_deterministic_transitions(
    *,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
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

    Returns:
        Tuple of the immutable merged `next_<state>` mapping and the frozenset of
        conflicting `next_<state>` names.
    """
    merged: dict[TransitionFunctionName, TransitionFunction] = {}
    conflicting: set[TransitionFunctionName] = set()
    for bundle in transitions.values():
        for name, func in bundle.items():
            if name in stochastic_transition_names:
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
    utility_name: str = "utility",
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
        utility_name: DAG target name of the felicity function. `"utility"` (the
            default) is the singleton case; a collective regime passes a
            per-stakeholder `"utility_<s>"` so this builder returns that
            stakeholder's own `U^s` alongside the shared feasibility.

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
        **{k: v for k, v in functions.items() if k != "H"},
    }
    _fail_if_conflicting_transition_is_read(
        combined=combined,
        targets=[utility_name, "feasibility"],
        conflicting_deterministic_transition_names=(
            conflicting_deterministic_transition_names
        ),
    )
    return concatenate_functions(
        functions=combined,
        targets=[utility_name, "feasibility"],
        enforce_signature=False,
        set_annotations=True,
    )


def _fail_if_conflicting_transition_is_read(
    *,
    combined: Mapping[str, Callable[..., Any]],
    targets: list[str],
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
