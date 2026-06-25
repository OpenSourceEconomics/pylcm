"""The per-(period, regime) DC-EGM solve kernel.

`build_egm_step_functions` turns a regime's processed functions, transitions,
and `DCEGM` solver configuration into per-period kernels that replace the
brute-force `max_Q_over_a` during backward induction. Each kernel runs the
DC-EGM step per combination of the regime's discrete states, passive
continuous states (deterministic non-process states whose transitions are
independent of the decision, e.g. an AIME-like skill level — one node per
combo), and discrete actions, with the exogenous savings grid as the inner
axis:

1. compute child states at every savings node (the post-decision function is
   removed from the DAG so the savings node enters as an external input),
2. map the child state into the child's resources space, select the carry
   rows matching the child's discrete-state values, and interpolate the
   child's value and marginal utility there per child discrete-action combo.
   The child's resources space is *per combo*: when the child's resources
   function reads its discrete states, passive states, or discrete actions,
   each carry row is queried at its own $R'$ and carries its own composed
   gradient $\\partial R'/\\partial A$ of the map
   $A \\mapsto R'(\\mathcal{T}(A), z', d', p')$ — the non-Euler inputs are
   savings-independent (validated) and ride as constants through the
   gradient. When the resources function reads only the child's Euler state,
   a single query and gradient is computed and broadcast across the rows.
   The child's passive values land off-grid, so the read is *mixed*: linear
   weights on the two neighboring nodes of the child's passive grid
   (edge-clamped), each neighbor's row interpolated 1-D in its own node's
   resources space, then blended per discrete-action row — before the choice
   aggregation, so the logsum sees blended choice-specific values,
3. aggregate over the child's discrete-action rows with the child's EV1
   taste-shock scale: the smoothed value is the logsum and the smoothed
   marginal is the choice-probability-weighted marginal
   $\\sum_{d'} P_{d'} \\mu_{d'} (\\partial R'/\\partial A)_{d'}$ — the
   composed factor sits inside the aggregation because each choice's
   envelope lives in its own resources space (exact for EV1 by Danskin's
   theorem; scale zero degrades to the hard max / one-hot argmax),
4. take the stochastic-node and regime-transition expectations: a child
   stochastic state's node — a continuous AR(1) process state or a
   Markov-discrete state — is distributed per its intrinsic transition weights
   $w(\\text{node}' \\mid \\text{node}, \\text{params})$, so steps 2-3 run
   per child node combo and the results are weight-summed — the stochastic
   expectation sits *outside* the action aggregation (the shock realizes
   before the next period's choice), matching the brute-force solver's
   weighted average of the already action-aggregated next-period V — and the
   per-target results are regime-transition-probability weighted,
5. invert the Euler equation per savings node (with the degenerate-inversion
   guard) to obtain the optimal action and the endogenous resources grid,
6. add the closed-form credit-constrained segment as additional candidates,
7. refine the candidate correspondence through the configured upper-envelope
   backend (one envelope per discrete combo),
8. publish the value function on the regime's exogenous state grid —
   discrete-state and passive-state combos remain axes of the value-function
   array (the Euler axis is moved to its canonical position among the
   continuous states), the regime's own discrete-action combos are
   aggregated with the regime's own taste-shock scale — and assemble the
   per-combo carry rows for the regime's parents. The publish step never
   evaluates anything at off-grid passive values: each combo's rows are
   interpolated in resources only, on the regime's own grid axes.

When any savings-stage function reads the current Euler state — the Euler
state's own law (an additive residual like a means-tested capital-income
supplement), the regime-transition probabilities, a stochastic transition
weight, or a passive state's law — the kernel solves *per exogenous asset
node* instead: conditional on one node of the Euler grid every such read is
a per-combo constant, so the single-post-state pipeline above is exact
within the node's row, and the row publishes only its own node. This is
brute-force-equivalent by construction — the brute solver evaluates the
same decision-time functions at the same exogenous nodes. The per-combo
carry then holds the per-node published points (one row per combo, no asset
axis): abscissa the node resources, policy the published optimal action,
value the published V, and marginal
$dV/dR = u'(c^*) + \\beta\\, (\\partial W/\\partial a)\\big|_{A^*} / R'(a)$
— the envelope term $u'(c^*)$ plus the savings-stage functions' direct
Euler-state channels through the continuation $W$, divided by the resources
slope. The Euler-state channel is differentiated through the *whole*
continuation closure: regime-transition probabilities and transition
weights are evaluated inside it, so first-order terms like
$\\sum_{targets} \\partial P/\\partial a \\cdot EV_{target}$ survive — the
probabilities are not the softmax of the values they weight, so Danskin
does not cancel them, and hoisting them out of the differentiated closure
would silently drop them. The mode requires every savings-stage Euler-state
read to be *continuous* in the Euler state at the resolution of the Euler
grid (kinks are fine): a residual that jumps makes the child's value
function discontinuous, the true policy bunches next-period wealth exactly
at the discontinuity — an Euler-equation-free corner outside EGM's
candidate families (interior Euler inversions plus the closed-form
credit-constrained segment) — and a jump in the probabilities or weights
breaks the smoothness-at-node-resolution assumption the per-node solve
relies on, so such functions are rejected at build time rather than solved
approximately. The switch is a build-time Python branch; the default path
is untouched and costs nothing when the mode is unused.

The canonical combo-axis order — single-sourced through
`_EgmKernelPieces.combo_names` and the carry template's leading shape — is
discrete states (V state order, process states included as node-valued
discrete dimensions), then passive states (V continuous-state order), then
discrete actions (action-grid order), then the carry's grid axis. Discrete
states lead so the child carry can be selected by integer indexing; passive
axes follow so the mixed read blends them away next, leaving exactly the
discrete-action rows the choice aggregation consumes.

Discrete-only constraints mask infeasible discrete combos: their value rows
are $-\\infty$ and their marginal-utility rows are exactly zero, so they
carry zero choice probability and stay finite inside the parent's
probability-weighted expectation.

A terminal carry target may carry discrete states, but only those shared
with the parent's own discrete combo axes and reached by the identity
(fixed) transition — at a parent combo with the state at value $k$ the
terminal carry row is its utility evaluated at $k$, selected by the parent's
own integer combo index, exactly as the non-terminal child read selects its
combo. A fixed `pref_type` whose terminal bequest differs by type is the
motivating case.

A child Markov-discrete state — one whose `next_<name>` is a stochastic
transition into the target — is integrated exactly like a process state: its
node axis is the *child's* discrete grid (codes $0, 1, \\dots$, which also
index the carry's leading axis), distributed per the intrinsic weights
$w(\\text{node}' \\mid \\text{node}, \\text{params})$, summed outside the
action aggregation. The child grid need not match the source's: a 3-state
health remapped onto a 2-state target carries a length-2 weight vector, and
the integration ranges over the child axis.

Out of scope:
terminal carry targets with actions, with a discrete state the parent does
not carry, or with a non-identity transition into a shared discrete state,
child resources functions reading anything beyond the child's states and
discrete actions (e.g. free child params), and child process states whose
grid points are supplied at runtime while feeding the child's resources
function. Such configurations build kernels that raise `NotImplementedError`
at solve time, so `Model` construction always succeeds for a validated
DC-EGM regime.
"""

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
from dags import concatenate_functions, with_signature

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.asset_row import _get_solve_one_combo_asset_rows
from _lcm.egm.carry import EGMCarry, build_template_egm_carry
from _lcm.egm.continuation import (
    ContinuationPlan,
    _build_child_reads,
    _is_runtime_process,
    get_egm_continuation_targets,
)
from _lcm.egm.kernel_scope import _find_unsupported_feature
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.egm.regime_introspection import (
    _concatenate_regime_function,
    _get_discrete_state_names,
    _get_passive_state_names,
    _get_process_state_names,
)
from _lcm.egm.step_core import (
    CONSTRAINED_OFFSET_FRACTION,
    _EgmKernelPieces,
    _get_solve_one_combo,
)
from _lcm.egm.upper_envelope import get_bracket_finder, get_upper_envelope
from _lcm.egm.validation import _reachable_target_names, savings_stage_reads_euler_state
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid, Grid
from _lcm.logsum import logsum_and_softmax
from _lcm.regime_building.h_dag import _get_build_H_kwargs
from _lcm.regime_building.max_Q_over_a import TASTE_SHOCK_SCALE_PARAM
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import (
    ActionName,
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    EGMStepFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from _lcm.utils.dispatchers import productmap
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    Float1D,
    FloatND,
    IntND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


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
    regime_to_flat_param_names: MappingProxyType[RegimeName, frozenset[str]],
    state_action_space: StateActionSpace,
    has_taste_shocks: bool,
) -> tuple[MappingProxyType[int, EGMStepFunction], EGMCarry, frozenset[RegimeName]]:
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
        regime_to_flat_param_names: Immutable mapping of every regime name to
            its flat parameter names. A carry target's resources / transition
            functions read the target regime's params, so the validator admits
            and the kernel binds the union of the source and its reachable
            carry targets' params.
        state_action_space: The regime's state-action space (source of the
            discrete-action grids and their canonical order).
        has_taste_shocks: Whether the regime declares EV1 taste shocks on its
            discrete actions.

    Returns:
        Tuple of the per-period kernel mapping, the regime's all-finite carry
        template (leading axes: discrete states, then passive states, then
        discrete actions), and the regime's reachable-target names — the only
        carry keys any of its kernels read, used to filter the rolling carry
        mapping the solve loop hands each kernel.

    """
    n_pad = compute_egm_carry_length(solver=solver)
    # `batch_size` on the Euler-state grid splays the per-asset-node solve into
    # blocks (`lax.map`) to shed peak working-set memory; 0 keeps the fused
    # vmap. Only the asset-row kernel has a per-node axis to splay.
    euler_grid = cast(
        "ContinuousGrid", user_regimes[regime_name].states[solver.continuous_state]
    )
    euler_batch_size = euler_grid.batch_size
    # `batch_size` on the exogenous savings grid splays the inner per-savings-node
    # continuation computation into `lax.map` blocks, shedding the dominant
    # egm_step working buffer (savings x child-stochastic mesh x combos); 0 keeps
    # the fused vmap. The upper envelope still runs on the gathered full grid.
    savings_batch_size = solver.savings_grid.batch_size
    own_v_info = regime_to_v_interpolation_info[regime_name]
    # Any savings-stage Euler-state read (the Euler law's residual, regime
    # transition probabilities, stochastic transition weights, non-Euler
    # laws) switches the kernel to the per-exogenous-asset-node solve; the
    # single-post-state kernel has no defined Euler value at the savings
    # stage, so dispatching it would be silently wrong.
    asset_row_mode = savings_stage_reads_euler_state(
        user_regime=user_regimes[regime_name], solver=solver
    )
    # The persisted carry length is split from the envelope-workspace length
    # `n_pad`: in asset-row mode the stored row holds one published point per
    # exogenous Euler node and the dense workspace is transient (FUES/RFC runs
    # per node and publishes a single point), so the carry needs only
    # `n_euler_nodes` rows. The padding to `n_pad` it would otherwise carry is
    # pure dead storage (`interp_on_padded_grid` masks the NaN tail). In the
    # single-post-state kernel the carry *is* the refined envelope, so its
    # length stays `n_pad`.
    n_carry_rows = n_pad
    if asset_row_mode:
        n_euler_nodes = int(
            own_v_info.continuous_states[solver.continuous_state].to_jax().shape[0]
        )
        n_pad = max(n_pad, n_euler_nodes)
        n_carry_rows = n_euler_nodes
    own_discrete_state_names = _get_discrete_state_names(
        v_interpolation_info=own_v_info
    )
    own_passive_state_names = _get_passive_state_names(
        v_interpolation_info=own_v_info,
        euler_state_name=solver.continuous_state,
    )
    own_discrete_action_values = MappingProxyType(
        dict(state_action_space.discrete_actions)
    )
    # Process states whose distribution params arrive at runtime have their
    # grids resolved per solve; thread those resolved grids into the
    # continuation so a process-reading resources function integrates over the
    # solve-time nodes rather than the build-time NaN placeholder.
    own_runtime_process_names = tuple(
        name
        for name in _get_process_state_names(v_interpolation_info=own_v_info)
        if _is_runtime_process(own_v_info.discrete_states[name])
    )
    # `batch_size` on a discrete-state, process, or passive-state grid splays
    # that combo axis (per-axis `productmap` blocks) to shed memory; 0 keeps
    # the fused vmap. Discrete-action axes are never split (the discrete-action
    # logsum needs every action value at once), so they map to 0.
    combo_state_batch_sizes = MappingProxyType(
        {
            name: cast("Grid", user_regimes[regime_name].states[name]).batch_size
            for name in own_discrete_state_names + own_passive_state_names
        }
    )
    # Canonical position of the Euler axis in the published V array: after
    # the discrete-state axes, at its slot within the continuous-state order.
    euler_axis_in_V = len(own_discrete_state_names) + tuple(
        own_v_info.continuous_states
    ).index(solver.continuous_state)
    leading_shape = (
        tuple(
            int(own_v_info.discrete_states[name].to_jax().shape[0])
            for name in own_discrete_state_names
        )
        + tuple(
            int(own_v_info.continuous_states[name].to_jax().shape[0])
            for name in own_passive_state_names
        )
        + tuple(int(v.shape[0]) for v in own_discrete_action_values.values())
    )
    carry_template = build_template_egm_carry(
        n_rows=n_carry_rows, leading_shape=leading_shape
    )

    reachable_targets = frozenset(
        _reachable_target_names(
            user_regime=user_regimes[regime_name], user_regimes=user_regimes
        )
    )
    configs: dict[tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]], list[int]] = {}
    for period in regimes_to_active_periods[regime_name]:
        target_split = get_egm_continuation_targets(
            period=period,
            transitions=transitions,
            reachable_targets=reachable_targets,
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        )
        configs.setdefault(target_split, []).append(period)

    built: dict[
        tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]], EGMStepFunction
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
            regime_to_flat_param_names=regime_to_flat_param_names,
            own_discrete_state_names=own_discrete_state_names,
            own_passive_state_names=own_passive_state_names,
            own_discrete_action_names=tuple(own_discrete_action_values),
            asset_row_mode=asset_row_mode,
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
                stochastic_transition_names=stochastic_transition_names,
                compute_regime_transition_probs=compute_regime_transition_probs,
                carry_targets=carry_targets,
                scalar_targets=scalar_targets,
                n_pad=n_pad,
                n_carry_rows=n_carry_rows,
                own_discrete_state_names=own_discrete_state_names,
                own_passive_state_names=own_passive_state_names,
                own_discrete_action_values=own_discrete_action_values,
                own_runtime_process_names=own_runtime_process_names,
                euler_axis_in_V=euler_axis_in_V,
                has_taste_shocks=has_taste_shocks,
                regime_to_v_interpolation_info=regime_to_v_interpolation_info,
                asset_row_mode=asset_row_mode,
                euler_batch_size=euler_batch_size,
                savings_batch_size=savings_batch_size,
                combo_state_batch_sizes=combo_state_batch_sizes,
            )
        built[(carry_targets, scalar_targets)] = kernel

    result: dict[int, EGMStepFunction] = {}
    for target_split, periods in configs.items():
        for period in periods:
            result[period] = built[target_split]

    return (
        MappingProxyType(dict(sorted(result.items()))),
        carry_template,
        reachable_targets,
    )


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
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    carry_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...],
    n_pad: int,
    n_carry_rows: int,
    own_discrete_state_names: tuple[StateName, ...],
    own_passive_state_names: tuple[StateName, ...],
    own_discrete_action_values: MappingProxyType[ActionName, Any],
    own_runtime_process_names: tuple[StateName, ...],
    euler_axis_in_V: int,
    has_taste_shocks: bool,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    asset_row_mode: bool,
    euler_batch_size: int,
    savings_batch_size: int,
    combo_state_batch_sizes: MappingProxyType[StateName, int],
) -> EGMStepFunction:
    """Build the EGM kernel for one continuation-target configuration.

    `asset_row_mode` selects the per-combo computation at build time: the
    per-exogenous-asset-node solve when any savings-stage function reads the
    current Euler state, the single-post-state default otherwise.

    `euler_batch_size` (the Euler grid's `batch_size`) splays the asset-row
    per-node solve into `lax.map` blocks to shed peak memory; it has no effect
    in the single-post-state (non-asset-row) kernel, which has no per-node axis.
    """
    get_solve_one_combo = (
        _get_solve_one_combo_asset_rows if asset_row_mode else _get_solve_one_combo
    )
    pieces = _build_kernel_pieces(
        solver=solver,
        user_regimes=user_regimes,
        functions=functions,
        constraints=constraints,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        carry_targets=carry_targets,
        scalar_targets=scalar_targets,
        n_pad=n_pad,
        n_carry_rows=n_carry_rows,
        own_discrete_state_names=own_discrete_state_names,
        own_passive_state_names=own_passive_state_names,
        own_discrete_action_values=own_discrete_action_values,
        euler_axis_in_V=euler_axis_in_V,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
    )

    def egm_step(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],  # noqa: ARG001
        next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[FloatND, EGMCarry, EGMSimPolicy]:
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
            (discrete-state axes leading, continuous states in canonical
            order) and the regime's carry (one row per discrete-state x
            passive-node x discrete-action combo).

        """
        dtype = canonical_float_dtype()
        own_state_names = {
            pieces.euler_state_name,
            *own_discrete_state_names,
            *own_passive_state_names,
        }
        pool = {k: v for k, v in kwargs.items() if k not in own_state_names}
        state_grid = jnp.asarray(kwargs[pieces.euler_state_name], dtype=dtype)
        own_taste_shock_scale = (
            jnp.asarray(kwargs[TASTE_SHOCK_SCALE_PARAM], dtype=dtype)
            if has_taste_shocks
            else jnp.asarray(0.0, dtype=dtype)
        )
        resolved_process_grids = MappingProxyType(
            {
                name: jnp.asarray(kwargs[name], dtype=dtype)
                for name in own_runtime_process_names
            }
        )
        solve_one_combo = get_solve_one_combo(
            pieces=pieces,
            pool=pool,
            state_grid=state_grid,
            next_regime_to_egm_carry=next_regime_to_egm_carry,
            euler_batch_size=euler_batch_size,
            savings_batch_size=savings_batch_size,
            resolved_process_grids=resolved_process_grids,
        )

        if pieces.combo_names:
            combo_var_names = (
                own_discrete_state_names
                + own_passive_state_names
                + tuple(own_discrete_action_values)
            )

            @with_signature(args=list(combo_var_names))
            def solve_one_combo_over_axes(
                **combo_values: ScalarFloat | ScalarInt,
            ) -> tuple[Float1D, Float1D, Float1D, Float1D, Float1D]:
                return solve_one_combo(
                    tuple(combo_values[name] for name in combo_var_names)
                )

            # Map the per-combo solve over the Cartesian product of the combo
            # axes: a discrete state / process / passive grid's `batch_size`
            # splays its axis (shedding memory), actions stay fused (the
            # discrete-action logsum needs every value at once). Splayed axes
            # share one `lax.map` (one scan carry) rather than nesting one per
            # axis. Outputs come back with the combo axes in `combo_var_names`
            # order, preserving whole discrete axes for the carry.
            combo_axis_values = {
                **{
                    name: jnp.asarray(kwargs[name])
                    for name in own_discrete_state_names + own_passive_state_names
                },
                **{
                    name: jnp.asarray(values)
                    for name, values in own_discrete_action_values.items()
                },
            }
            V_stack, grid_stack, policy_stack, value_stack, marginal_stack = (
                _map_combo_product(
                    func=solve_one_combo_over_axes,
                    combo_var_names=combo_var_names,
                    combo_axis_values=combo_axis_values,
                    batch_sizes={
                        **dict(combo_state_batch_sizes),
                        **dict.fromkeys(own_discrete_action_values, 0),
                    },
                )
            )
            n_state_axes = len(own_discrete_state_names) + len(own_passive_state_names)
            action_axes = tuple(range(n_state_axes, len(combo_var_names)))
            if action_axes and has_taste_shocks:
                V_arr, _ = logsum_and_softmax(
                    values=V_stack, scale=own_taste_shock_scale, axes=action_axes
                )
            elif action_axes:
                # Without taste shocks the smoothed maximum is the hard maximum
                # over the discrete-action axes (the logsum is for `scale > 0`).
                V_arr = jnp.max(V_stack, axis=action_axes)
            else:
                V_arr = V_stack
            # The combo layout puts the Euler axis last; the canonical V
            # layout interleaves it with the passive axes in V state order.
            V_arr = jnp.moveaxis(V_arr, -1, pieces.euler_axis_in_V)
            carry = EGMCarry(
                endog_grid=grid_stack,
                value=value_stack,
                marginal_utility=marginal_stack,
                taste_shock_scale=own_taste_shock_scale,
            )
            sim_policy = EGMSimPolicy(endog_grid=grid_stack, policy=policy_stack)
        else:
            V_arr, grid_row, policy_row, value_row, marginal_row = solve_one_combo(())
            carry = EGMCarry(
                endog_grid=grid_row,
                value=value_row,
                marginal_utility=marginal_row,
                taste_shock_scale=own_taste_shock_scale,
            )
            sim_policy = EGMSimPolicy(endog_grid=grid_row, policy=policy_row)
        return V_arr, carry, sim_policy

    return egm_step


def _map_combo_product(
    *,
    func: Callable[..., tuple[Float1D, ...]],
    combo_var_names: tuple[StateOrActionName, ...],
    combo_axis_values: dict[StateOrActionName, FloatND | IntND],
    batch_sizes: dict[StateOrActionName, int],
) -> tuple[FloatND, ...]:
    """Map the per-combo solve over the Cartesian product of the combo axes.

    `func` has a `combo_var_names` keyword signature and returns one tuple of
    1-D arrays per combo. Each combo axis with `batch_size == 0` is vmapped;
    axes with `batch_size > 0` are splayed (run in `lax.map` blocks) to shed
    peak memory. Returns the stacked outputs with the combo axes as leading
    dims in `combo_var_names` order (the canonical carry layout).

    With ≤1 splayed axis this is plain `productmap` (one `lax.map`, no
    nesting). With ≥2 splayed axes, `productmap` would nest one `lax.map`
    per axis and stack a scan carry per level; instead the splayed axes are
    flattened into a *single* `lax.map` (one carry) with the unsplayed axes
    vmapped within each step, then the result is transposed back into
    `combo_var_names` order. Numerically identical to the nested form — only
    the schedule (and its peak resident) differs.
    """
    splayed = tuple(name for name in combo_var_names if batch_sizes[name] > 0)
    vmapped = tuple(name for name in combo_var_names if batch_sizes[name] == 0)

    if len(splayed) <= 1:
        mapped = productmap(
            func=func,  # ty: ignore[invalid-argument-type]
            variables=combo_var_names,
            batch_sizes=batch_sizes,
        )
        return mapped(**combo_axis_values)

    # One `lax.map` over the flattened splayed product, unsplayed axes vmapped
    # within each step (all-`batch_size=0` `productmap` lowers to nested vmaps,
    # no scan carry).
    inner = productmap(
        func=func,  # ty: ignore[invalid-argument-type]
        variables=vmapped,
        batch_sizes=dict.fromkeys(vmapped, 0),
    )
    vmapped_values = {name: combo_axis_values[name] for name in vmapped}
    splayed_dims = tuple(int(combo_axis_values[name].shape[0]) for name in splayed)
    mesh = jnp.meshgrid(*(combo_axis_values[name] for name in splayed), indexing="ij")
    flat_splayed = tuple(grid.ravel() for grid in mesh)
    block = 1
    for name in splayed:
        block *= batch_sizes[name]

    def at_one_splayed_combo(
        splayed_values: tuple[ScalarFloat | ScalarInt, ...],
    ) -> tuple[FloatND, ...]:
        return inner(
            **dict(zip(splayed, splayed_values, strict=True)), **vmapped_values
        )

    stacked = jax.lax.map(at_one_splayed_combo, flat_splayed, batch_size=block)

    # Restore `combo_var_names` order: `stacked` axes are
    # (flattened splayed, *vmapped, *per-element); reshape the flat axis back
    # to the splayed dims, then transpose the combo axes into canonical order.
    current_order = splayed + vmapped
    perm = tuple(current_order.index(name) for name in combo_var_names)
    n_combo = len(combo_var_names)

    def _reorder(arr: FloatND) -> FloatND:
        arr = arr.reshape(*splayed_dims, *arr.shape[1:])
        trailing = tuple(range(n_combo, arr.ndim))
        return jnp.transpose(arr, (*perm, *trailing))

    return tuple(_reorder(arr) for arr in stacked)


def _build_kernel_pieces(
    *,
    solver: DCEGM,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    carry_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...],
    n_pad: int,
    n_carry_rows: int,
    own_discrete_state_names: tuple[StateName, ...],
    own_passive_state_names: tuple[StateName, ...],
    own_discrete_action_values: MappingProxyType[ActionName, Any],
    euler_axis_in_V: int,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> _EgmKernelPieces:
    """Assemble the build-time statics of the EGM kernel."""
    savings_nodes = jnp.asarray(
        solver.savings_grid.to_jax(), dtype=canonical_float_dtype()
    )
    n_constrained = solver.n_constrained_points
    child_reads = _build_child_reads(
        user_regimes=user_regimes,
        functions=functions,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        carry_targets=carry_targets,
        post_decision_name=solver.post_decision_function,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
    )
    continuation_plan = ContinuationPlan(
        carry_targets=carry_targets,
        scalar_targets=scalar_targets,
        child_reads=child_reads,
        compute_regime_transition_probs=compute_regime_transition_probs,
        post_decision_name=solver.post_decision_function,
        stochastic_node_batch_size=solver.stochastic_node_batch_size,
    )
    return _EgmKernelPieces(
        euler_state_name=solver.continuous_state,
        action_name=solver.continuous_action,
        savings_nodes=savings_nodes,
        borrowing_limit=savings_nodes[0],
        n_constrained=n_constrained,
        # Static geometric ratio: the constrained actions run from
        # `span * CONSTRAINED_OFFSET_FRACTION` up to `span`, so the ratio
        # depends only on the offset fraction and the point count.
        constrained_ratio=(1.0 / CONSTRAINED_OFFSET_FRACTION)
        ** (1.0 / max(n_constrained - 1, 1)),
        n_pad=n_pad,
        n_carry_rows=n_carry_rows,
        combo_names=own_discrete_state_names
        + own_passive_state_names
        + tuple(own_discrete_action_values),
        euler_axis_in_V=euler_axis_in_V,
        utility_func=_concatenate_regime_function(
            functions=functions, target="utility"
        ),
        inverse_marginal_utility_func=(
            _concatenate_regime_function(
                functions=functions, target="inverse_marginal_utility"
            )
            if "inverse_marginal_utility" in functions
            else None
        ),
        own_resources_func=_concatenate_regime_function(
            functions=functions, target=solver.resources
        ),
        feasibility_func=_build_feasibility_function(
            functions=functions, constraints=constraints
        ),
        build_H_kwargs=_get_build_H_kwargs(functions),
        refine=get_upper_envelope(solver=solver, n_refined=n_pad),
        refine_to_bracket=get_bracket_finder(solver=solver, n_refined=n_pad),
        continuation_plan=continuation_plan,
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


def _get_raising_egm_step(*, reason: str) -> EGMStepFunction:
    """Build a kernel that raises at solve time for unsupported configurations.

    `Model` construction with a validated DC-EGM regime always succeeds;
    features the kernel does not cover yet surface as `NotImplementedError`
    when the model is solved.
    """

    def raising_egm_step(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[FloatND, EGMCarry, EGMSimPolicy]:
        raise NotImplementedError(reason)

    return raising_egm_step
