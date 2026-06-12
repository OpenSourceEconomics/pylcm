"""The per-(period, regime) DC-EGM solve kernel.

`build_egm_step_functions` turns a regime's processed functions, transitions,
and `DCEGM` solver configuration into per-period kernels that replace the
brute-force `max_Q_over_a` during backward induction. Each kernel runs the
DC-EGM step per combination of the regime's discrete states and discrete
actions, with the exogenous savings grid as the inner axis:

1. compute child states at every savings node (the post-decision function is
   removed from the DAG so the savings node enters as an external input),
2. map the child state into the child's resources space, select the carry
   rows matching the child's discrete-state values, and interpolate the
   child's value and marginal utility there per child discrete-action combo,
3. aggregate over the child's discrete-action rows with the child's EV1
   taste-shock scale: the smoothed value is the logsum and the smoothed
   marginal is the choice-probability-weighted marginal
   $\\sum_{d'} P_{d'} \\mu_{d'}$ (exact for EV1 by Danskin's theorem; scale
   zero degrades to the hard max / one-hot argmax),
4. take the regime-transition-probability-weighted expectation, multiplying
   by the composed-gradient factor $\\partial R'/\\partial A$ of the map
   $A \\mapsto R'(\\mathcal{T}(A))$,
5. invert the Euler equation per savings node (with the degenerate-inversion
   guard) to obtain the optimal action and the endogenous resources grid,
6. add the closed-form credit-constrained segment as additional candidates,
7. refine the candidate correspondence through the configured upper-envelope
   backend (one envelope per discrete combo),
8. publish the value function on the regime's exogenous state grid —
   discrete-state combos remain axes of the value-function array, the
   regime's own discrete-action combos are aggregated with the regime's own
   taste-shock scale — and assemble the per-combo carry rows for the
   regime's parents.

Discrete-only constraints mask infeasible discrete combos: their value rows
are $-\\infty$ and their marginal-utility rows are exactly zero, so they
carry zero choice probability and stay finite inside the parent's
probability-weighted expectation.

Out of scope: process states (own or in a carry target), stochastic
transitions into a carry target, terminal carry targets with discrete states
or actions. Such configurations build kernels that raise
`NotImplementedError` at solve time, so `Model` construction always succeeds
for a validated DC-EGM regime.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
from dags import concatenate_functions

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EgmCarry, build_template_egm_carry
from _lcm.egm.euler import invert_euler
from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope import get_upper_envelope
from _lcm.engine import StateActionSpace
from _lcm.logsum import logsum_and_softmax
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.h_dag import _get_build_H_kwargs
from _lcm.regime_building.max_Q_over_a import TASTE_SHOCK_SCALE_PARAM
from _lcm.regime_building.next_state import get_next_state_function_for_solution
from _lcm.regime_building.Q_and_F import get_period_targets
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import (
    ActionName,
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    EgmStepFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from _lcm.utils.functools import get_union_of_args
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    Float1D,
    FloatND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    UserFunction,
)

# Smallest constrained-segment action as a fraction of the segment's span.
# The constrained candidates are geometrically spaced from this offset toward
# the borrowing limit (additive, so a non-positive limit stays well-defined).
CONSTRAINED_OFFSET_FRACTION = 1e-4


def build_egm_step_functions(
    *,
    solver: DCEGM,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    flat_param_names: frozenset[str],
    state_action_space: StateActionSpace,
    has_taste_shocks: bool,
) -> tuple[MappingProxyType[int, EgmStepFunction], EgmCarry]:
    """Build per-period DC-EGM kernels and the regime's carry template.

    Periods sharing the same continuation-target configuration share one
    kernel (and hence one compiled program), mirroring the per-period
    grouping of the Q-and-F builders.

    Args:
        solver: The regime's DC-EGM solver configuration.
        regime_name: Name of the regime the kernels solve.
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances (the carry targets' resources functions are read from
            here).
        functions: The regime's processed functions (params renamed to
            qualified names).
        constraints: Immutable mapping of the regime's constraint names to
            constraint functions (discrete-only after DC-EGM validation).
        transitions: Immutable mapping of target regime names to their state
            transition functions.
        stochastic_transition_names: Frozenset of stochastic transition
            function names.
        compute_regime_transition_probs: Regime transition probability
            function for solve.
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info.
        regimes_to_active_periods: Immutable mapping of regime names to their
            active period tuples.
        flat_param_names: Frozenset of flat parameter names for the regime.
        state_action_space: The regime's state-action space (source of the
            discrete-action grids and their canonical order).
        has_taste_shocks: Whether the regime declares EV1 taste shocks on its
            discrete actions.

    Returns:
        Tuple of the per-period kernel mapping and the regime's all-finite
        carry template (leading axes: discrete states, then discrete
        actions).

    """
    n_pad = compute_egm_carry_length(solver=solver)
    own_discrete_state_names = _get_discrete_state_names(
        v_interpolation_info=regime_to_v_interpolation_info[regime_name]
    )
    own_discrete_action_values = MappingProxyType(
        dict(state_action_space.discrete_actions)
    )
    leading_shape = tuple(
        int(
            regime_to_v_interpolation_info[regime_name]
            .discrete_states[name]
            .to_jax()
            .shape[0]
        )
        for name in own_discrete_state_names
    ) + tuple(int(v.shape[0]) for v in own_discrete_action_values.values())
    carry_template = build_template_egm_carry(n_rows=n_pad, leading_shape=leading_shape)

    configs: dict[tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]], list[int]] = {}
    for period in regimes_to_active_periods[regime_name]:
        target_split = get_egm_continuation_targets(
            period=period,
            transitions=transitions,
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )
        configs.setdefault(target_split, []).append(period)

    built: dict[
        tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]], EgmStepFunction
    ] = {}
    for carry_targets, scalar_targets in configs:
        unsupported = _find_unsupported_feature(
            solver=solver,
            regime_name=regime_name,
            user_regimes=user_regimes,
            functions=functions,
            constraints=constraints,
            carry_targets=carry_targets,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            flat_param_names=flat_param_names,
            own_discrete_state_names=own_discrete_state_names,
            own_discrete_action_names=tuple(own_discrete_action_values),
        )
        if unsupported is not None:
            kernel = _get_raising_egm_step(reason=unsupported)
        else:
            kernel = _get_egm_step(
                solver=solver,
                user_regimes=user_regimes,
                functions=functions,
                constraints=constraints,
                transitions=transitions,
                compute_regime_transition_probs=compute_regime_transition_probs,
                carry_targets=carry_targets,
                scalar_targets=scalar_targets,
                n_pad=n_pad,
                own_discrete_state_names=own_discrete_state_names,
                own_discrete_action_values=own_discrete_action_values,
                has_taste_shocks=has_taste_shocks,
                regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            )
        built[(carry_targets, scalar_targets)] = kernel

    result: dict[int, EgmStepFunction] = {}
    for target_split, periods in configs.items():
        for period in periods:
            result[period] = built[target_split]

    return MappingProxyType(dict(sorted(result.items()))), carry_template


def get_egm_continuation_targets(
    *,
    period: int,
    transitions: TransitionFunctionsMapping,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]]:
    """Split next-period-active targets into carry-interpolated and scalar ones.

    This adapter is the single place where the EGM step derives "which target
    regimes / which transition functions" from the engine regime's
    transitions; changes to the transition representation swap this body
    without touching the kernel.

    - *Carry targets* have state-transition entries; their continuation is
      interpolated from their `EgmCarry` rows.
    - *Scalar targets* are stateless (no transition entries, no states; e.g.
      a `dead` regime); their continuation is the constant value of their
      carry rows and their marginal continuation is zero.

    Args:
        period: The period the kernel solves.
        transitions: Immutable mapping of target regime names to their state
            transition functions.
        regimes_to_active_periods: Immutable mapping of regime names to their
            active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info.

    Returns:
        Tuple of carry-target names and scalar-target names.

    """
    carry_targets = get_period_targets(
        period=period,
        transitions=transitions,
        regimes_to_active_periods=regimes_to_active_periods,
    )
    scalar_targets = tuple(
        name
        for name in regime_to_v_interpolation_info
        if period + 1 in regimes_to_active_periods.get(name, ())
        and not regime_to_v_interpolation_info[name].state_names
        and name not in carry_targets
    )
    return carry_targets, scalar_targets


def compute_egm_carry_length(*, solver: DCEGM) -> int:
    """Static carry-row length for a DC-EGM regime.

    Covers the Euler candidates (one per savings node), the closed-form
    constrained candidates, and the headroom factor for envelope-kink
    insertions (each costs two slots).

    Args:
        solver: The regime's DC-EGM solver configuration.

    Returns:
        Number of slots in the regime's carry rows.

    """
    n_savings = int(solver.savings_grid.to_jax().shape[0])
    n_candidates = n_savings + solver.n_constrained_points
    return math.ceil(solver.refined_grid_factor * n_candidates)


def _get_egm_step(
    *,
    solver: DCEGM,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    compute_regime_transition_probs: RegimeTransitionFunction,
    carry_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...],
    n_pad: int,
    own_discrete_state_names: tuple[StateName, ...],
    own_discrete_action_values: MappingProxyType[ActionName, Any],
    has_taste_shocks: bool,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> EgmStepFunction:
    """Build the EGM kernel for one continuation-target configuration."""
    pieces = _build_kernel_pieces(
        solver=solver,
        user_regimes=user_regimes,
        functions=functions,
        constraints=constraints,
        transitions=transitions,
        compute_regime_transition_probs=compute_regime_transition_probs,
        carry_targets=carry_targets,
        scalar_targets=scalar_targets,
        n_pad=n_pad,
        own_discrete_state_names=own_discrete_state_names,
        own_discrete_action_values=own_discrete_action_values,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
    )

    def egm_step(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],  # noqa: ARG001
        next_regime_to_egm_carry: MappingProxyType[RegimeName, EgmCarry],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[FloatND, EgmCarry]:
        """Run the DC-EGM step and publish V on the exogenous grid.

        Args:
            next_regime_to_V_arr: The next period's value-function arrays;
                accepted so solve treats all kernels uniformly (continuation
                values come from the carries).
            next_regime_to_egm_carry: The next period's EGM carries.
            **kwargs: The regime's state grids, flat params, `period`, and
                `age`.

        Returns:
            Tuple of the value-function array on the exogenous state grid
            (discrete-state axes leading, the continuous state last) and the
            regime's carry (one row per discrete-state x discrete-action
            combo).

        """
        dtype = canonical_float_dtype()
        own_state_names = {pieces.euler_state_name, *own_discrete_state_names}
        pool = {k: v for k, v in kwargs.items() if k not in own_state_names}
        state_grid = jnp.asarray(kwargs[pieces.euler_state_name], dtype=dtype)
        own_taste_shock_scale = (
            jnp.asarray(kwargs[TASTE_SHOCK_SCALE_PARAM], dtype=dtype)
            if has_taste_shocks
            else jnp.asarray(0.0, dtype=dtype)
        )
        solve_one_combo = _get_solve_one_combo(
            pieces=pieces,
            pool=pool,
            state_grid=state_grid,
            next_regime_to_egm_carry=next_regime_to_egm_carry,
        )

        if pieces.combo_names:
            combo_grids = tuple(
                jnp.asarray(kwargs[name]) for name in own_discrete_state_names
            ) + tuple(own_discrete_action_values.values())
            mesh = jnp.meshgrid(*combo_grids, indexing="ij")
            flat_combos = tuple(m.ravel() for m in mesh)
            V_rows, grid_rows, policy_rows, value_rows, marginal_rows = jax.vmap(
                solve_one_combo
            )(flat_combos)
            dims = tuple(int(g.shape[0]) for g in combo_grids)
            V_stack = V_rows.reshape(*dims, state_grid.shape[0])
            action_axes = tuple(range(len(own_discrete_state_names), len(combo_grids)))
            if action_axes:
                V_arr, _ = logsum_and_softmax(
                    values=V_stack, scale=own_taste_shock_scale, axes=action_axes
                )
            else:
                V_arr = V_stack
            carry = EgmCarry(
                endog_grid=grid_rows.reshape(*dims, n_pad),
                policy=policy_rows.reshape(*dims, n_pad),
                value=value_rows.reshape(*dims, n_pad),
                marginal_utility=marginal_rows.reshape(*dims, n_pad),
                taste_shock_scale=own_taste_shock_scale,
            )
        else:
            V_arr, grid_row, policy_row, value_row, marginal_row = solve_one_combo(())
            carry = EgmCarry(
                endog_grid=grid_row,
                policy=policy_row,
                value=value_row,
                marginal_utility=marginal_row,
                taste_shock_scale=own_taste_shock_scale,
            )
        return V_arr, carry

    return egm_step


@dataclass(frozen=True, kw_only=True)
class _EgmKernelPieces:
    """Build-time statics shared by every per-combo EGM computation."""

    euler_state_name: StateName
    """Name of the regime's continuous (Euler) state."""

    action_name: ActionName
    """Name of the regime's continuous action."""

    post_decision_name: str
    """Name of the post-decision function (the savings node's input slot)."""

    savings_nodes: Float1D
    """The exogenous end-of-period savings grid."""

    borrowing_limit: ScalarFloat
    """Lower bound of the savings grid."""

    n_constrained: int
    """Number of closed-form credit-constrained candidate points."""

    constrained_ratio: float
    """Static geometric spacing ratio of the constrained candidates."""

    n_pad: int
    """Static length of the refined carry rows."""

    combo_names: tuple[StateName | ActionName, ...]
    """Discrete-state names, then discrete-action names (carry-axis order)."""

    carry_targets: tuple[RegimeName, ...]
    """Targets whose continuation is interpolated from their carry rows."""

    scalar_targets: tuple[RegimeName, ...]
    """Stateless targets contributing a constant continuation value."""

    next_state_funcs: Mapping[RegimeName, Callable[..., Any]]
    """Per-target next-state functions (post-decision function removed)."""

    child_resources_funcs: Mapping[RegimeName, Callable[[ScalarFloat], ScalarFloat]]
    """Per-target closed-over child resources maps."""

    child_next_state_keys: Mapping[RegimeName, str]
    """Per-target `next_<state>` key of the child's continuous state."""

    child_discrete_state_names: Mapping[RegimeName, tuple[StateName, ...]]
    """Per-target child discrete-state names in carry-axis order."""

    utility_func: UserFunction
    """The regime's concatenated utility function."""

    inverse_marginal_utility_func: UserFunction
    """The regime's concatenated inverse-marginal-utility function."""

    own_resources_func: UserFunction
    """The regime's concatenated resources function."""

    feasibility_func: Callable[..., ScalarBool] | None
    """Discrete-feasibility predicate of a combo, or `None`."""

    build_H_kwargs: Callable[[Mapping[str, Any]], dict[str, Any]]
    """Closure assembling the Bellman aggregator's keyword arguments."""

    refine: Callable[..., tuple[Float1D, Float1D, Float1D, ScalarInt]]
    """The configured upper-envelope backend."""

    compute_regime_transition_probs: RegimeTransitionFunction
    """Regime transition probability function for solve."""


def _build_kernel_pieces(
    *,
    solver: DCEGM,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    compute_regime_transition_probs: RegimeTransitionFunction,
    carry_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...],
    n_pad: int,
    own_discrete_state_names: tuple[StateName, ...],
    own_discrete_action_values: MappingProxyType[ActionName, Any],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> _EgmKernelPieces:
    """Assemble the build-time statics of the EGM kernel."""
    savings_nodes = jnp.asarray(
        solver.savings_grid.to_jax(), dtype=canonical_float_dtype()
    )
    n_constrained = solver.n_constrained_points
    (
        next_state_funcs,
        child_resources_funcs,
        child_next_state_keys,
        child_discrete_state_names,
    ) = _build_target_closures(
        user_regimes=user_regimes,
        functions=functions,
        transitions=transitions,
        carry_targets=carry_targets,
        post_decision_name=solver.post_decision_function,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
    )
    return _EgmKernelPieces(
        euler_state_name=solver.continuous_state,
        action_name=solver.continuous_action,
        post_decision_name=solver.post_decision_function,
        savings_nodes=savings_nodes,
        borrowing_limit=savings_nodes[0],
        n_constrained=n_constrained,
        # Static geometric ratio: the constrained actions run from
        # `span * CONSTRAINED_OFFSET_FRACTION` up to `span`, so the ratio
        # depends only on the offset fraction and the point count.
        constrained_ratio=(1.0 / CONSTRAINED_OFFSET_FRACTION)
        ** (1.0 / max(n_constrained - 1, 1)),
        n_pad=n_pad,
        combo_names=own_discrete_state_names + tuple(own_discrete_action_values),
        carry_targets=carry_targets,
        scalar_targets=scalar_targets,
        next_state_funcs=next_state_funcs,
        child_resources_funcs=child_resources_funcs,
        child_next_state_keys=child_next_state_keys,
        child_discrete_state_names=child_discrete_state_names,
        utility_func=_concatenate_regime_function(
            functions=functions, target="utility"
        ),
        inverse_marginal_utility_func=_concatenate_regime_function(
            functions=functions, target="inverse_marginal_utility"
        ),
        own_resources_func=_concatenate_regime_function(
            functions=functions, target=solver.resources
        ),
        feasibility_func=_build_feasibility_function(
            functions=functions, constraints=constraints
        ),
        build_H_kwargs=_get_build_H_kwargs(functions),
        refine=get_upper_envelope(solver=solver, n_refined=n_pad),
        compute_regime_transition_probs=compute_regime_transition_probs,
    )


def _get_solve_one_combo(
    *,
    pieces: _EgmKernelPieces,
    pool: dict[str, Any],
    state_grid: Float1D,
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EgmCarry],
) -> Callable[
    [tuple[ScalarInt, ...]], tuple[Float1D, Float1D, Float1D, Float1D, Float1D]
]:
    """Build the per-combo EGM computation for one kernel invocation."""
    dtype = state_grid.dtype

    def solve_one_combo(
        combo_values: tuple[ScalarInt, ...],
    ) -> tuple[Float1D, Float1D, Float1D, Float1D, Float1D]:
        """Run the EGM step for one discrete (state x action) combo.

        Takes the combo's discrete values positionally so `jax.vmap` can
        batch over flattened combo arrays.

        Returns:
            Tuple of the combo's value row on the exogenous state grid and
            its refined endogenous grid, policy, value, and marginal-utility
            carry rows.

        """
        combo_pool = {
            **pool,
            **dict(zip(pieces.combo_names, combo_values, strict=True)),
        }
        # Validation pins the default Bellman aggregator, whose single
        # non-(utility, E_next_V) parameter is the discount factor.
        (discount_factor,) = tuple(pieces.build_H_kwargs(combo_pool).values())

        def utility_of_action(action_value: ScalarFloat) -> ScalarFloat:
            return pieces.utility_func(
                **{pieces.action_name: action_value}, **combo_pool
            )

        compute_node = _get_compute_node(
            pieces=pieces,
            combo_pool=combo_pool,
            discount_factor=discount_factor,
            utility_of_action=utility_of_action,
            next_regime_to_egm_carry=next_regime_to_egm_carry,
            dtype=dtype,
        )
        actions, endog_grid, values, expected_values = jax.vmap(compute_node)(
            pieces.savings_nodes
        )

        def own_resources_of_state(state_value: ScalarFloat) -> ScalarFloat:
            return pieces.own_resources_func(
                **{pieces.euler_state_name: state_value}, **combo_pool
            )

        publish_resources = jax.vmap(own_resources_of_state)(state_grid)

        constrained_actions, constrained_values = _compute_constrained_candidates(
            first_endogenous_point=endog_grid[0],
            publish_resources=publish_resources,
            borrowing_limit=pieces.borrowing_limit,
            n_constrained=pieces.n_constrained,
            constrained_ratio=pieces.constrained_ratio,
            utility_of_action=utility_of_action,
            discounted_expected_value_at_limit=discount_factor * expected_values[0],
        )

        refined_grid, refined_policy, refined_value, n_kept = pieces.refine(
            endog_grid=jnp.concatenate(
                [pieces.borrowing_limit + constrained_actions, endog_grid]
            ),
            policy=jnp.concatenate([constrained_actions, actions]),
            value=jnp.concatenate([constrained_values, values]),
        )

        V_row, value_row, marginal_utility_row = _publish_V_and_carry_rows(
            refined_grid=refined_grid,
            refined_policy=refined_policy,
            refined_value=refined_value,
            n_kept=n_kept,
            n_pad=pieces.n_pad,
            publish_resources=publish_resources,
            borrowing_limit=pieces.borrowing_limit,
            utility_of_action=utility_of_action,
            discounted_expected_value_at_limit=discount_factor * expected_values[0],
        )

        if pieces.feasibility_func is not None:
            # Infeasible discrete combos: -inf value rows so they win no
            # maximum and carry zero choice probability; exactly-zero
            # marginal utility so probability-weighted sums stay finite.
            feasible = pieces.feasibility_func(**combo_pool)
            V_row = jnp.where(feasible, V_row, -jnp.inf)
            value_row = jnp.where(feasible, value_row, -jnp.inf)
            marginal_utility_row = jnp.where(feasible, marginal_utility_row, 0.0)

        return (
            V_row,
            refined_grid.astype(dtype),
            refined_policy.astype(dtype),
            value_row,
            marginal_utility_row,
        )

    return solve_one_combo


def _get_compute_node(
    *,
    pieces: _EgmKernelPieces,
    combo_pool: dict[str, Any],
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EgmCarry],
    dtype: Any,  # noqa: ANN401
) -> Callable[[ScalarFloat], tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]]:
    """Build the per-savings-node Euler inversion for one discrete combo."""
    regime_transition_probs = pieces.compute_regime_transition_probs(**combo_pool)

    def inverse_marginal_utility(
        marginal_continuation: ScalarFloat,
    ) -> ScalarFloat:
        return pieces.inverse_marginal_utility_func(
            marginal_continuation=marginal_continuation, **combo_pool
        )

    def compute_node(
        savings_value: ScalarFloat,
    ) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]:
        """Euler-invert one savings node against the continuation."""
        expected_marginal = jnp.asarray(0.0, dtype=dtype)
        expected_value = jnp.asarray(0.0, dtype=dtype)
        for target in pieces.carry_targets:
            carry = next_regime_to_egm_carry[target]

            def composed_resources(
                savings: ScalarFloat, *, target: RegimeName = target
            ) -> tuple[ScalarFloat, tuple[ScalarInt, ...]]:
                """Map a savings node into the child's resources space.

                Returns the child's resources (differentiated) and the
                child's discrete-state values (auxiliary; their transitions
                are independent of the savings node).
                """
                next_states = pieces.next_state_funcs[target](
                    **combo_pool, **{pieces.post_decision_name: savings}
                )
                # The solution-phase next-state function returns a flat
                # mapping of `next_<state>` names to scalars; the shared
                # protocol's nested return type is the simulation form.
                next_state_value = cast(
                    "ScalarFloat", next_states[pieces.child_next_state_keys[target]]
                )
                child_index = tuple(
                    cast("ScalarInt", next_states[f"next_{name}"])
                    for name in pieces.child_discrete_state_names[target]
                )
                return (
                    pieces.child_resources_funcs[target](next_state_value),
                    child_index,
                )

            (child_resources, child_index), child_dr_da = jax.value_and_grad(
                composed_resources, has_aux=True
            )(savings_value)
            smoothed_value, smoothed_marginal = _aggregate_child_choices(
                carry=carry,
                child_index=child_index,
                child_resources=child_resources,
            )
            prob = regime_transition_probs[target]
            # Zero unreachable-target contributions on the results, never by
            # multiplying into a possibly non-finite value.
            expected_marginal = expected_marginal + jnp.where(
                prob > 0.0, prob * smoothed_marginal * child_dr_da, 0.0
            )
            expected_value = expected_value + jnp.where(
                prob > 0.0, prob * smoothed_value, 0.0
            )
        for target in pieces.scalar_targets:
            prob = regime_transition_probs[target]
            constant_value = next_regime_to_egm_carry[target].value[0]
            expected_value = expected_value + jnp.where(
                prob > 0.0, prob * constant_value, 0.0
            )

        action = invert_euler(
            expected_marginal_continuation=expected_marginal,
            discount_factor=discount_factor,
            inverse_marginal_utility=inverse_marginal_utility,
        )
        endog_point = savings_value + action
        value = utility_of_action(action) + discount_factor * expected_value
        return action, endog_point, value, expected_value

    return compute_node


def _aggregate_child_choices(
    *,
    carry: EgmCarry,
    child_index: tuple[ScalarInt, ...],
    child_resources: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Interpolate one child's carry and aggregate its discrete-action rows.

    The carry rows matching the child's discrete-state values are selected
    by integer indexing on the leading state axes (discrete codes equal grid
    positions); the remaining leading axes are the child's discrete-action
    combos. Each row is interpolated at the child's resources value, then
    aggregated with the child's taste-shock scale: the smoothed value is the
    logsum and the smoothed marginal is $\\sum_{d'} P_{d'} \\mu_{d'}$ —
    exact for EV1 by Danskin's theorem, no $\\partial P/\\partial R$ terms.
    Scale zero yields the hard max / one-hot argmax through the same code
    path. Rows that are $-\\infty$ everywhere (infeasible child combos) get
    zero probability and contribute exactly zero marginal utility.

    Args:
        carry: The child's EGM carry.
        child_index: The child's discrete-state values at this savings node.
        child_resources: The child's resources value at this savings node.

    Returns:
        Tuple of the smoothed continuation value and the smoothed marginal
        continuation $\\partial W/\\partial R'$.

    """
    n_pad = carry.value.shape[-1]
    grid_rows = carry.endog_grid[child_index].reshape(-1, n_pad)
    value_rows = carry.value[child_index].reshape(-1, n_pad)
    marginal_rows = carry.marginal_utility[child_index].reshape(-1, n_pad)

    # The marginal-utility row is the value row's exact slope (envelope
    # theorem), upgrading the value read to cubic Hermite; the mu read itself
    # stays linear (a policy-grade quantity, and its interpolation error
    # enters the value only at second order through the Euler inversion).
    def interp_value_row(xp: Float1D, fp: Float1D, fp_slopes: Float1D) -> ScalarFloat:
        """Interpolate one carry value row; positional per `jax.vmap`."""
        return interp_on_padded_grid(
            x_query=child_resources, xp=xp, fp=fp, fp_slopes=fp_slopes
        )

    def interp_row(xp: Float1D, fp: Float1D) -> ScalarFloat:
        """Interpolate one carry row; positional per `jax.vmap`."""
        return interp_on_padded_grid(x_query=child_resources, xp=xp, fp=fp)

    value_at_child = jax.vmap(interp_value_row)(grid_rows, value_rows, marginal_rows)
    marginal_at_child = jax.vmap(interp_row)(grid_rows, marginal_rows)
    # A row that is -inf everywhere yields NaN under linear interpolation
    # (-inf minus -inf); restore the -inf / exact-zero pair on the results.
    row_infeasible = jnp.isneginf(value_rows[:, 0])
    value_at_child = jnp.where(row_infeasible, -jnp.inf, value_at_child)
    marginal_at_child = jnp.where(row_infeasible, 0.0, marginal_at_child)

    smoothed_value, choice_probs = logsum_and_softmax(
        values=value_at_child, scale=carry.taste_shock_scale, axes=(0,)
    )
    smoothed_marginal = jnp.sum(choice_probs * marginal_at_child)
    return smoothed_value, smoothed_marginal


def _build_target_closures(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    carry_targets: tuple[RegimeName, ...],
    post_decision_name: str,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> tuple[
    dict[RegimeName, Callable[..., Any]],
    dict[RegimeName, Callable[[ScalarFloat], ScalarFloat]],
    dict[RegimeName, str],
    dict[RegimeName, tuple[StateName, ...]],
]:
    """Build the per-carry-target closures of the EGM kernel.

    The post-decision function is removed from the DAG, so its output (the
    savings node) becomes an external input of the next-state functions.
    Passing the full `functions` mapping would let the DAG compute savings
    internally from the (unknown) state and action leaves — it runs, but is
    silently wrong.

    Returns:
        Tuple of four dicts keyed by carry-target name: the next-state
        function, the closed-over child resources map, the child's
        `next_<state>` key for its continuous state, and the child's
        discrete-state names in carry-axis order.

    """
    functions_without_post = MappingProxyType(
        {name: func for name, func in functions.items() if name != post_decision_name}
    )
    next_state_funcs: dict[RegimeName, Callable[..., Any]] = {
        target: get_next_state_function_for_solution(
            transitions=transitions[target],
            functions=functions_without_post,
        )
        for target in carry_targets
    }
    child_resources_funcs = {
        target: _get_child_resources_function(user_regime=user_regimes[target])
        for target in carry_targets
    }
    child_next_state_keys = {
        target: f"next_{_get_child_state_name(user_regime=user_regimes[target])}"
        for target in carry_targets
    }
    child_discrete_state_names = {
        target: _get_discrete_state_names(
            v_interpolation_info=regime_to_v_interpolation_info[target]
        )
        for target in carry_targets
    }
    return (
        next_state_funcs,
        child_resources_funcs,
        child_next_state_keys,
        child_discrete_state_names,
    )


def _get_discrete_state_names(
    *,
    v_interpolation_info: VInterpolationInfo,
) -> tuple[StateName, ...]:
    """Discrete-state names of a regime in carry-axis (V state) order."""
    return tuple(
        name
        for name in v_interpolation_info.state_names
        if name in v_interpolation_info.discrete_states
    )


def _concatenate_regime_function(
    *,
    functions: EconFunctionsMapping,
    target: str,
) -> UserFunction:
    """Concatenate one regime-function target from the H-free DAG."""
    return concatenate_functions(
        functions={name: func for name, func in functions.items() if name != "H"},
        targets=target,
        enforce_signature=False,
        set_annotations=True,
    )


def _build_feasibility_function(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
) -> Callable[..., ScalarBool] | None:
    """Build the discrete-feasibility predicate of a combo, or `None`.

    DC-EGM validation guarantees that no constraint reaches the continuous
    state or action, so every constraint is evaluable per discrete combo.

    Returns:
        Callable mapping a combo's pool (discrete values plus flat params)
        to a scalar feasibility indicator, or `None` without constraints.

    """
    if not constraints:
        return None
    constraints_func = concatenate_functions(
        functions={
            **{name: func for name, func in functions.items() if name != "H"},
            **dict(constraints),
        },
        targets=list(constraints),
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )

    def feasibility(**combo_pool: Any) -> ScalarBool:  # noqa: ANN401
        """Evaluate all constraints for one combo and combine them."""
        outputs = constraints_func(**combo_pool)
        return jnp.all(jnp.stack([jnp.asarray(out) for out in outputs.values()]))

    return feasibility


def _compute_constrained_candidates(
    *,
    first_endogenous_point: ScalarFloat,
    publish_resources: FloatND,
    borrowing_limit: ScalarFloat,
    n_constrained: int,
    constrained_ratio: float,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
) -> tuple[FloatND, FloatND]:
    """Compute the closed-form credit-constrained candidate points.

    Below the first endogenous point the borrowing limit binds: the action is
    `R - borrowing_limit` and the continuation is the limit-node expectation.
    The candidate actions are geometrically spaced toward the limit; the span
    is capped at the publish range so a degenerate (huge) first endogenous
    point cannot stretch the sample beyond where queries land.

    Args:
        first_endogenous_point: The endogenous resources point of the lowest
            savings node.
        publish_resources: Resources at the regime's exogenous state grid.
        borrowing_limit: Lower bound of the savings grid.
        n_constrained: Number of constrained candidate points.
        constrained_ratio: Static geometric spacing ratio of the candidates.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of constrained candidate actions and values.

    """
    dtype = publish_resources.dtype
    span = jnp.maximum(
        jnp.minimum(first_endogenous_point, jnp.max(publish_resources))
        - borrowing_limit,
        jnp.finfo(dtype).tiny,
    )
    constrained_actions = (
        span
        * CONSTRAINED_OFFSET_FRACTION
        * constrained_ratio ** jnp.arange(n_constrained, dtype=dtype)
    )
    constrained_values = (
        jax.vmap(utility_of_action)(constrained_actions)
        + discounted_expected_value_at_limit
    )
    return constrained_actions, constrained_values


def _publish_V_and_carry_rows(
    *,
    refined_grid: Float1D,
    refined_policy: Float1D,
    refined_value: Float1D,
    n_kept: ScalarInt,
    n_pad: int,
    publish_resources: FloatND,
    borrowing_limit: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
) -> tuple[FloatND, Float1D, Float1D]:
    """Interpolate V onto the exogenous grid and finish the carry value rows.

    Below the lowest refined envelope point the interpolant edge-clamps, so
    the published value there is the closed-form constrained value — the
    exact value of saving exactly the borrowing limit. Above it, the refined
    envelope is read with the cubic Hermite interpolant (the marginal-utility
    row is the value row's exact slope by the envelope theorem), floored at
    the constrained value, which remains a feasible-policy lower bound
    everywhere.

    Envelope overflow is not silent: the outputs are NaN-poisoned so the
    solve loop's NaN diagnostics surface the offending (regime, period).

    Args:
        refined_grid: Refined endogenous grid row from the envelope backend.
        refined_policy: Refined policy row.
        refined_value: Refined value row.
        n_kept: Number of envelope points the backend kept.
        n_pad: Static length of the refined rows.
        publish_resources: Resources at the regime's exogenous state grid.
        borrowing_limit: Lower bound of the savings grid.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of the value row on the exogenous state grid, the carry value
        row, and the carry marginal-utility row.

    """
    dtype = publish_resources.dtype
    overflowed = n_kept > n_pad

    marginal_utility = jax.vmap(jax.grad(utility_of_action))(
        jnp.where(jnp.isnan(refined_policy), 1.0, refined_policy)
    )
    marginal_utility = jnp.where(jnp.isnan(refined_policy), jnp.nan, marginal_utility)
    marginal_utility = jnp.where(jnp.isneginf(refined_value), 0.0, marginal_utility)

    value_interpolated = interp_on_padded_grid(
        x_query=publish_resources,
        xp=refined_grid,
        fp=refined_value,
        fp_slopes=marginal_utility,
    )
    closed_form_actions = publish_resources - borrowing_limit
    value_constrained = jnp.where(
        closed_form_actions > 0.0,
        jax.vmap(utility_of_action)(
            jnp.maximum(closed_form_actions, jnp.finfo(dtype).tiny)
        )
        + discounted_expected_value_at_limit,
        -jnp.inf,
    )
    # Below the lowest refined point the interpolant edge-clamps, so the
    # closed-form constrained value (the exact value of saving exactly the
    # borrowing limit) is published outright there. Everywhere else it is a
    # feasible-policy floor under the Hermite read of the refined envelope:
    # forcing it further up — e.g. to the first Euler point — would discard
    # envelope information wherever degenerate inversions push that point
    # right (a zero-ish marginal continuation makes it ~1/eps), and the
    # constrained candidates inside the envelope already carry exact slopes.
    refined_grid_start = refined_grid[0]
    V_row = jnp.where(
        (closed_form_actions > 0.0) & (publish_resources <= refined_grid_start),
        value_constrained,
        jnp.maximum(value_interpolated, value_constrained),
    )
    V_row = jnp.where(overflowed, jnp.nan, V_row).astype(dtype)

    value_row = jnp.where(overflowed, jnp.nan, refined_value).astype(dtype)
    return V_row, value_row, marginal_utility.astype(dtype)


def _get_raising_egm_step(*, reason: str) -> EgmStepFunction:
    """Build a kernel that raises at solve time for unsupported configurations.

    `Model` construction with a validated DC-EGM regime always succeeds;
    features the kernel does not cover yet surface as `NotImplementedError`
    when the model is solved.
    """

    def raising_egm_step(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        next_regime_to_egm_carry: MappingProxyType[RegimeName, EgmCarry],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[FloatND, EgmCarry]:
        raise NotImplementedError(reason)

    return raising_egm_step


def _find_unsupported_feature(
    *,
    solver: DCEGM,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    carry_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    flat_param_names: frozenset[str],
    own_discrete_state_names: tuple[StateName, ...],
    own_discrete_action_names: tuple[ActionName, ...],
) -> str | None:
    """Return a message naming the first feature outside the kernel's scope.

    Returns `None` when the configuration is fully supported.
    """
    own_process_states = _get_process_state_names(
        v_interpolation_info=regime_to_v_interpolation_info[regime_name]
    )
    message: str | None = None
    if own_process_states:
        message = f"it has process states {list(own_process_states)}."

    for target in carry_targets:
        if message is not None:
            break
        message = _find_unsupported_target_feature(
            target=target,
            user_regimes=user_regimes,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )

    if message is None:
        message = _find_unsupported_function_args(
            solver=solver,
            functions=functions,
            constraints=constraints,
            compute_regime_transition_probs=compute_regime_transition_probs,
            flat_param_names=flat_param_names,
            own_discrete_state_names=own_discrete_state_names,
            own_discrete_action_names=own_discrete_action_names,
        )

    if message is None:
        return None
    return (
        f"The DC-EGM solver cannot solve regime '{regime_name}' yet: {message} "
        "This configuration is outside the DC-EGM kernel's current scope."
    )


def _find_unsupported_target_feature(
    *,
    target: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> str | None:
    """Return a message naming the first unsupported feature of one target."""
    target_info = regime_to_v_interpolation_info[target]
    target_process_states = _get_process_state_names(v_interpolation_info=target_info)
    if target_process_states:
        return (
            f"its target regime '{target}' has process states "
            f"{list(target_process_states)}."
        )
    if user_regimes[target].terminal:
        terminal_message = _find_unsupported_terminal_target_feature(
            target=target,
            user_regime=user_regimes[target],
            target_info=target_info,
        )
        if terminal_message is not None:
            return terminal_message
    stochastic = sorted(set(transitions[target]) & stochastic_transition_names)
    if stochastic:
        return f"the transitions {stochastic} into regime '{target}' are stochastic."
    child_state_name = _get_child_state_name(user_regime=user_regimes[target])
    extra_resources_args = sorted(
        _get_child_resources_arg_names(user_regime=user_regimes[target])
        - {child_state_name}
    )
    if extra_resources_args:
        return (
            f"the resources function of target regime '{target}' depends on "
            f"{extra_resources_args} in addition to '{child_state_name}'."
        )
    return None


def _find_unsupported_terminal_target_feature(
    *,
    target: RegimeName,
    user_regime: UserRegime,
    target_info: VInterpolationInfo,
) -> str | None:
    """Return a message naming the first unsupported feature of a terminal target."""
    if target_info.discrete_states:
        return (
            f"its terminal target regime '{target}' has discrete states "
            f"{list(target_info.discrete_states)}; terminal carries cover "
            "a single continuous state only."
        )
    if len(target_info.state_names) != 1:
        return (
            f"its terminal target regime '{target}' has states "
            f"{list(target_info.state_names)}; exactly one continuous "
            "state is supported."
        )
    if user_regime.actions:
        return (
            f"its terminal target regime '{target}' has actions "
            f"{list(user_regime.actions)}, so its carry is not "
            "its utility on the state grid."
        )
    return None


def _find_unsupported_function_args(
    *,
    solver: DCEGM,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    compute_regime_transition_probs: RegimeTransitionFunction,
    flat_param_names: frozenset[str],
    own_discrete_state_names: tuple[StateName, ...],
    own_discrete_action_names: tuple[ActionName, ...],
) -> str | None:
    """Return a message naming the first function with out-of-scope arguments."""
    allowed_discrete = set(own_discrete_state_names) | set(own_discrete_action_names)
    allowed_params = flat_param_names | {"age", "period"}
    utility_func = _concatenate_regime_function(functions=functions, target="utility")
    arg_requirements: list[tuple[str, frozenset[str], set[str]]] = [
        (
            "the utility function",
            frozenset(get_union_of_args([utility_func])),
            {solver.continuous_action} | allowed_discrete | allowed_params,
        ),
        (
            "the regime transition probability function",
            frozenset(get_union_of_args([compute_regime_transition_probs])),
            allowed_discrete | allowed_params,
        ),
    ]
    for constraint_name in constraints:
        constraint_func = _concatenate_regime_function(
            functions=MappingProxyType({**dict(functions), **dict(constraints)}),
            target=constraint_name,
        )
        arg_requirements.append(
            (
                f"the constraint '{constraint_name}'",
                frozenset(get_union_of_args([constraint_func])),
                allowed_discrete | allowed_params,
            )
        )
    for label, needed, allowed in arg_requirements:
        extra = sorted(needed - allowed)
        if extra:
            return f"{label} depends on {extra}."
    return None


def _get_process_state_names(
    *,
    v_interpolation_info: VInterpolationInfo,
) -> tuple[StateName, ...]:
    """Names of a regime's process states (node-valued discrete dimensions)."""
    return tuple(
        name
        for name, grid in v_interpolation_info.discrete_states.items()
        if isinstance(grid, _ContinuousStochasticProcess)
    )


def _get_child_state_name(*, user_regime: UserRegime) -> StateName:
    """Name of a carry target's continuous (Euler) state.

    For a DC-EGM target this is its configured Euler state; for a terminal
    target it is its only state (uniqueness is checked by
    `_find_unsupported_feature`).
    """
    if isinstance(user_regime.solver, DCEGM):
        return user_regime.solver.continuous_state
    return next(iter(user_regime.states))


def _get_child_resources_function(
    *, user_regime: UserRegime
) -> Callable[[ScalarFloat], ScalarFloat]:
    """Build the closed-over resources map of one carry target.

    For a DC-EGM target the map is its declared resources function (resolved
    to the solve-phase variant); for a terminal target the carry lives in
    M-space and the map is the identity. The returned callable takes the
    child's state value positionally so the kernel can compose it with the
    state transition and differentiate the composition.
    """
    if isinstance(user_regime.solver, DCEGM):
        child_state_name = user_regime.solver.continuous_state
        resources_func = _concatenate_child_resources(user_regime=user_regime)

        def child_resources(state_value: ScalarFloat) -> ScalarFloat:
            return resources_func(**{child_state_name: state_value})

        return child_resources

    def identity_resources(state_value: ScalarFloat) -> ScalarFloat:
        return state_value

    return identity_resources


def _get_child_resources_arg_names(*, user_regime: UserRegime) -> set[str]:
    """Argument names of a carry target's resources map."""
    if isinstance(user_regime.solver, DCEGM):
        return set(
            get_union_of_args([_concatenate_child_resources(user_regime=user_regime)])
        )
    return {next(iter(user_regime.states))}


def _concatenate_child_resources(*, user_regime: UserRegime) -> UserFunction:
    """Concatenate a DC-EGM target's resources function from its user DAG."""
    solver = cast("DCEGM", user_regime.solver)
    resolved: dict[str, UserFunction] = {}
    for name, func in user_regime.functions.items():
        if isinstance(func, Phased):
            resolved[name] = cast("UserFunction", func.solve)
        elif func is not None:
            resolved[name] = func
    return concatenate_functions(
        functions=resolved,
        targets=solver.resources,
        enforce_signature=False,
        set_annotations=True,
    )
