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
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
from dags import concatenate_functions, with_signature

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry, build_template_egm_carry
from _lcm.egm.euler import invert_euler
from _lcm.egm.interp import (
    interp_on_padded_grid,
    interp_on_prepared_grid,
    locate_on_grid,
    prepare_padded_grid,
)
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.egm.upper_envelope import get_upper_envelope
from _lcm.egm.validation import _reachable_target_names, savings_stage_reads_euler_state
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid, Grid
from _lcm.logsum import logsum_and_softmax
from _lcm.params.regime_template import create_regime_params_template
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
    EGMStepFunction,
    FunctionName,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import get_union_of_args
from _lcm.variables import from_regime, get_grids
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    Float1D,
    FloatND,
    IntND,
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


def get_egm_continuation_targets(
    *,
    period: int,
    transitions: TransitionFunctionsMapping,
    reachable_targets: frozenset[RegimeName],
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]]:
    """Split next-period-active targets into carry-interpolated and scalar ones.

    This adapter is the single place where the EGM step derives "which target
    regimes / which transition functions" from the engine regime's
    transitions; changes to the transition representation swap this body
    without touching the kernel.

    - *Carry targets* have state-transition entries; their continuation is
      interpolated from their `EGMCarry` rows.
    - *Scalar targets* are stateless (no transition entries, no states; e.g.
      a `dead` regime); their continuation is the constant value of their
      carry rows and their marginal continuation is zero. Only declared-
      reachable regimes qualify: a stateless regime the model contains for
      other regimes' sake has no transition-probability cell here, and the
      regime transition is the single source of truth for reachability.

    Args:
        period: The period the kernel solves.
        transitions: Immutable mapping of target regime names to their state
            transition functions.
        reachable_targets: The regime's declared-reachable target names.
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
        if name in reachable_targets
        and period + 1 in regimes_to_active_periods.get(name, ())
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
        solve_one_combo = get_solve_one_combo(
            pieces=pieces,
            pool=pool,
            state_grid=state_grid,
            next_regime_to_egm_carry=next_regime_to_egm_carry,
            euler_batch_size=euler_batch_size,
            savings_batch_size=savings_batch_size,
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


@dataclass(frozen=True, kw_only=True)
class _EgmKernelPieces:
    """Build-time statics shared by every per-combo EGM computation."""

    euler_state_name: StateName
    """Name of the regime's continuous (Euler) state."""

    action_name: ActionName
    """Name of the regime's continuous action."""

    post_decision_name: FunctionName
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
    """Static length of the envelope-refinement workspace (FUES/RFC, overflow)."""

    n_carry_rows: int
    """Static length of the persisted carry rows.

    Equals `n_pad` in the single-post-state kernel (the carry is the refined
    envelope) and `n_euler_nodes` in asset-row mode (the carry is one published
    point per exogenous Euler node, with no envelope-workspace padding)."""

    stochastic_node_batch_size: int
    """Block size for splaying the child stochastic-node expectation (0 = fused)."""

    combo_names: tuple[StateName | ActionName, ...]
    """Discrete-state, passive-state, then discrete-action names (carry-axis order)."""

    euler_axis_in_V: int
    """Canonical axis of the Euler state in the published value-function array."""

    carry_targets: tuple[RegimeName, ...]
    """Targets whose continuation is interpolated from their carry rows."""

    scalar_targets: tuple[RegimeName, ...]
    """Stateless targets contributing a constant continuation value."""

    child_reads: Mapping[RegimeName, _ChildRead]
    """Per-carry-target statics of the child carry read."""

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


@dataclass(frozen=True, kw_only=True)
class _ChildRead:
    """Build-time statics for reading one carry target's rows.

    The row block of a child carry — after the deterministic discrete-state
    and stochastic-node indices are applied — has the child's passive nodes as
    leading axes and its discrete-action combos as trailing axes; the
    per-row binding values and the block shape are precomputed here so the
    kernel's per-savings-node read is pure array work.
    """

    next_state_func: Callable[..., Any]
    """The target's next-state function (post-decision function removed)."""

    next_state_key: TransitionFunctionName
    """`next_<state>` key of the child's continuous (Euler) state."""

    euler_state_name: StateName
    """Name of the child's continuous (Euler) state."""

    has_taste_shocks: bool
    """Whether the target regime declares EV1 taste shocks.

    Selects the child's discrete-action aggregation: the `scale > 0` logsum
    when set, the hard maximum (one-hot argmax) when not.
    """

    resources_func: Callable[..., ScalarFloat]
    """The child's concatenated resources function (kwargs-based)."""

    resources_arg_names: frozenset[str]
    """Leaf argument names of the child's resources function."""

    resources_param_names: frozenset[str]
    """Qualified param leaves of the child's resources function.

    Bound per node from the combo pool (the regime's flat params, plus `age`
    / `period`). Constant in the savings node, so they ride through the
    composed resources gradients without contributing a savings derivative.
    """

    resources_is_simple: bool
    """Whether the resources function reads only the child's Euler state.

    The simple case computes one query and one composed gradient per savings
    node and broadcasts them across the carry rows; the general case
    evaluates both per row. Params and `age` / `period` are constants, so a
    resources function reading only the Euler state and params still counts
    as simple.
    """

    discrete_state_names: tuple[StateName, ...]
    """Child discrete-state names (stochastic states included) in carry-axis order."""

    stochastic_flags: tuple[bool, ...]
    """Per discrete-state dimension: whether it is a stochastic node axis.

    A dimension is stochastic when its next-period node is distributed by a
    transition law: a continuous AR(1) process state, or a Markov-discrete
    state whose `next_<name>` is a stochastic transition into the target. Both
    are integrated over the child's node axis with the intrinsic weights.
    """

    stochastic_state_names: tuple[StateName, ...]
    """Child stochastic node-axis names (process or Markov) in carry-axis order."""

    stochastic_node_values: tuple[FloatND | IntND, ...]
    """Per stochastic dimension: the node values fed into the resources query.

    Process dimensions carry the continuous AR(1) grid points (NaN when
    supplied at runtime); Markov-discrete dimensions carry the integer
    category codes (which equal the carry's leading-axis indices).
    """

    weight_keys: tuple[str, ...]
    """`weight_<target>__next_<state>` keys aligned with the stochastic dims."""

    weights_func: Callable[..., Any] | None
    """Concatenated intrinsic-weights function, or `None` without stochastic dims."""

    passive_state_names: tuple[StateName, ...]
    """Child passive-state names in carry-axis order."""

    passive_grids: tuple[Float1D, ...]
    """Child passive grids, aligned with `passive_state_names`."""

    row_arg_names: tuple[StateName | ActionName, ...]
    """Names bound per carry row: passive states, then discrete actions."""

    row_values: tuple[FloatND | IntND, ...]
    """Flattened row-binding value meshes, aligned with `row_arg_names`.

    Passive entries are float node values; discrete-action entries are
    integer codes.
    """

    row_block_shape: tuple[int, ...]
    """Shape of the carry's row block: passive sizes, then action sizes."""


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
        n_carry_rows=n_carry_rows,
        stochastic_node_batch_size=solver.stochastic_node_batch_size,
        combo_names=own_discrete_state_names
        + own_passive_state_names
        + tuple(own_discrete_action_values),
        euler_axis_in_V=euler_axis_in_V,
        carry_targets=carry_targets,
        scalar_targets=scalar_targets,
        child_reads=child_reads,
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
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
    euler_batch_size: int,
    savings_batch_size: int,
) -> Callable[
    [tuple[ScalarInt | ScalarFloat, ...]],
    tuple[Float1D, Float1D, Float1D, Float1D, Float1D],
]:
    """Build the per-combo EGM computation for one kernel invocation.

    `euler_batch_size` is accepted for a uniform builder signature but unused:
    the single-post-state kernel solves once per combo, with no per-asset-node
    axis to splay (only the asset-row kernel honors it).
    """
    del euler_batch_size
    dtype = state_grid.dtype

    def solve_one_combo(
        combo_values: tuple[ScalarInt | ScalarFloat, ...],
    ) -> tuple[Float1D, Float1D, Float1D, Float1D, Float1D]:
        """Run the EGM step for one (discrete x passive-node) combo.

        Takes the combo's values (discrete codes and passive node values)
        positionally so `jax.vmap` can batch over flattened combo arrays.

        Returns:
            Tuple of the combo's value row on the exogenous state grid and
            its refined endogenous grid, the published consumption policy on
            that grid, and the value and marginal-utility carry rows.

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
        actions, endog_grid, values, expected_values = _compute_nodes_over_savings(
            compute_node=compute_node,
            savings_nodes=pieces.savings_nodes,
            savings_batch_size=savings_batch_size,
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

        candidate_grid = jnp.concatenate(
            [pieces.borrowing_limit + constrained_actions, endog_grid]
        )
        candidate_policy = jnp.concatenate([constrained_actions, actions])
        candidate_value = jnp.concatenate([constrained_values, values])
        # A `-inf`-valued candidate (e.g. a corner whose continuation is
        # `-inf`) is dominated by every finite candidate and would inject
        # `-inf - (-inf) = NaN` into the envelope scan's gradient arithmetic.
        # Mask the triple to NaN — the scan's absent form. NaN-valued
        # candidates stay as they are: genuine poison must keep propagating.
        candidate_dead = jnp.isneginf(candidate_value)
        refined_grid, refined_policy, refined_value, n_kept = pieces.refine(
            endog_grid=jnp.where(candidate_dead, jnp.nan, candidate_grid),
            policy=jnp.where(candidate_dead, jnp.nan, candidate_policy),
            value=jnp.where(candidate_dead, jnp.nan, candidate_value),
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

        # A combo with no live candidate (its entire continuation is `-inf`)
        # is worth `-inf` everywhere, like an infeasible combo.
        no_live_candidate = jnp.all(candidate_dead)
        V_row = jnp.where(no_live_candidate, -jnp.inf, V_row)
        value_row = jnp.where(no_live_candidate, -jnp.inf, value_row)
        marginal_utility_row = jnp.where(no_live_candidate, 0.0, marginal_utility_row)

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


def _get_solve_one_combo_asset_rows(
    *,
    pieces: _EgmKernelPieces,
    pool: dict[str, Any],
    state_grid: Float1D,
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
    euler_batch_size: int,
    savings_batch_size: int,
) -> Callable[
    [tuple[ScalarInt | ScalarFloat, ...]],
    tuple[Float1D, Float1D, Float1D, Float1D, Float1D],
]:
    """Build the per-combo EGM computation solving per exogenous asset node.

    Used when any savings-stage function (the Euler state's law, the
    regime-transition probabilities, a stochastic transition weight, a
    passive state's law) reads the current Euler state: conditional on one
    node of the Euler grid every such read is a per-combo constant, so the
    single-post-state pipeline (Euler inversion over the savings nodes,
    constrained candidates, upper-envelope refinement) is exact within the
    node's row, and each row publishes only its own node — exactly where
    the brute-force oracle evaluates the same decision-time functions. The
    per-combo carry row holds the per-node published points: abscissa the
    node resources (weakly ascending by the resources monotonicity check),
    value the published V, and marginal the corrected
    $dV/dR = u'(c^*) + \\beta\\, (\\partial W/\\partial a)|_{A^*} / R'(a)$,
    NaN-padded to the carry length. The Euler-state gradient
    $\\partial W/\\partial a$ rebuilds the full continuation closure from
    the traced node value, so it carries every direct channel: the law's
    residual, $\\sum \\partial P/\\partial a \\cdot EV$ from the
    probabilities, the weights' and passive laws' derivatives.
    """
    dtype = state_grid.dtype
    n_state = int(state_grid.shape[0])

    def solve_one_combo(
        combo_values: tuple[ScalarInt | ScalarFloat, ...],
    ) -> tuple[Float1D, Float1D, Float1D, Float1D, Float1D]:
        """Run the per-asset-node EGM step for one (discrete x passive) combo.

        Takes the combo's values (discrete codes and passive node values)
        positionally so `jax.vmap` can batch over flattened combo arrays.

        Returns:
            Tuple of the combo's value row on the exogenous state grid and
            its per-node endogenous grid, the published consumption policy on
            that grid, and the value and marginal-utility carry
            rows.

        """
        combo_pool = {
            **pool,
            **dict(zip(pieces.combo_names, combo_values, strict=True)),
        }
        # Validation pins the default Bellman aggregator, whose single
        # non-(utility, E_next_V) parameter is the discount factor.
        (discount_factor,) = tuple(pieces.build_H_kwargs(combo_pool).values())

        def own_resources_of_state(state_value: ScalarFloat) -> ScalarFloat:
            return pieces.own_resources_func(
                **{pieces.euler_state_name: state_value}, **combo_pool
            )

        def continuation_of_euler_state(
            state_value: ScalarFloat, savings_value: ScalarFloat
        ) -> ScalarFloat:
            """Expected continuation with the Euler slot as the grad argument.

            Rebuilds the per-combo continuation closure with the Euler slot
            of the combo pool bound to `state_value` (positional, so
            `jax.grad` differentiates the law's direct Euler-state channel)
            and evaluates it at fixed savings.
            """
            node_pool = {**combo_pool, pieces.euler_state_name: state_value}
            expected_continuation = _get_expected_continuation_value(
                pieces=pieces,
                combo_pool=node_pool,
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                dtype=dtype,
            )
            return expected_continuation(savings_value)

        def solve_one_node(
            node_value: ScalarFloat,
        ) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat]:
            """Run the single-post-state pipeline conditional on one node."""
            node_pool = {**combo_pool, pieces.euler_state_name: node_value}

            def utility_of_action(action_value: ScalarFloat) -> ScalarFloat:
                return pieces.utility_func(
                    **{pieces.action_name: action_value}, **node_pool
                )

            compute_node = _get_compute_node(
                pieces=pieces,
                combo_pool=node_pool,
                discount_factor=discount_factor,
                utility_of_action=utility_of_action,
                next_regime_to_egm_carry=next_regime_to_egm_carry,
                dtype=dtype,
            )
            actions, endog_grid, values, expected_values = _compute_nodes_over_savings(
                compute_node=compute_node,
                savings_nodes=pieces.savings_nodes,
                savings_batch_size=savings_batch_size,
            )

            resources_at_node, resources_gradient = jax.value_and_grad(
                own_resources_of_state
            )(node_value)

            constrained_actions, constrained_values = _compute_constrained_candidates(
                first_endogenous_point=endog_grid[0],
                publish_resources=resources_at_node,
                borrowing_limit=pieces.borrowing_limit,
                n_constrained=pieces.n_constrained,
                constrained_ratio=pieces.constrained_ratio,
                utility_of_action=utility_of_action,
                discounted_expected_value_at_limit=discount_factor * expected_values[0],
            )

            candidate_grid = jnp.concatenate(
                [pieces.borrowing_limit + constrained_actions, endog_grid]
            )
            candidate_policy = jnp.concatenate([constrained_actions, actions])
            candidate_value = jnp.concatenate([constrained_values, values])
            # Same `-inf` masking as the default per-combo computation: dead
            # candidates become the envelope scan's absent form (NaN).
            candidate_dead = jnp.isneginf(candidate_value)
            refined_grid, refined_policy, refined_value, n_kept = pieces.refine(
                endog_grid=jnp.where(candidate_dead, jnp.nan, candidate_grid),
                policy=jnp.where(candidate_dead, jnp.nan, candidate_policy),
                value=jnp.where(candidate_dead, jnp.nan, candidate_value),
            )

            V_node, policy_node = _publish_node_V_and_policy(
                refined_grid=refined_grid,
                refined_policy=refined_policy,
                refined_value=refined_value,
                n_kept=n_kept,
                n_pad=pieces.n_pad,
                resources_at_node=resources_at_node,
                borrowing_limit=pieces.borrowing_limit,
                utility_of_action=utility_of_action,
                discounted_expected_value_at_limit=discount_factor * expected_values[0],
            )

            # The carry marginal at this node:
            # dV/dR = u'(c*) + discount_factor * (dW/da at fixed A*) / R'(a),
            # with A* = R(a) - c*(a). The envelope term u'(c*) covers the
            # savings channel; the second term is the law's direct
            # Euler-state channel through the continuation, mapped into
            # resources space by the resources slope.
            marginal_utility_node = jax.grad(utility_of_action)(
                jnp.where(jnp.isnan(policy_node), 1.0, policy_node)
            )
            savings_at_optimum = resources_at_node - policy_node
            continuation_gradient = jax.grad(continuation_of_euler_state)(
                node_value, savings_at_optimum
            )
            mu_node = (
                marginal_utility_node
                + discount_factor * continuation_gradient / resources_gradient
            )
            mu_node = jnp.where(jnp.isnan(policy_node), jnp.nan, mu_node)

            # A node with no live candidate (its entire continuation is
            # `-inf`) is worth `-inf`, like an infeasible combo; its
            # marginal is exactly zero so probability-weighted sums stay
            # finite.
            no_live_candidate = jnp.all(candidate_dead)
            V_node = jnp.where(no_live_candidate, -jnp.inf, V_node)
            mu_node = jnp.where(jnp.isneginf(V_node) | no_live_candidate, 0.0, mu_node)

            return V_node, policy_node, mu_node

        # Splay the per-asset-node solve into `lax.map` blocks of
        # `euler_batch_size` to shed peak working-set memory; `0` (or a size
        # covering the whole grid) keeps the fused vmap. The two are
        # numerically identical — only the schedule differs.
        if 0 < euler_batch_size < n_state:
            V_vec, policy_vec, mu_vec = jax.lax.map(
                solve_one_node, state_grid, batch_size=euler_batch_size
            )
        else:
            V_vec, policy_vec, mu_vec = jax.vmap(solve_one_node)(state_grid)
        publish_resources = jax.vmap(own_resources_of_state)(state_grid)

        pad = jnp.full((pieces.n_carry_rows - n_state,), jnp.nan, dtype=dtype)
        grid_row = jnp.concatenate([publish_resources.astype(dtype), pad])
        policy_row = jnp.concatenate([policy_vec.astype(dtype), pad])
        value_row = jnp.concatenate([V_vec.astype(dtype), pad])
        marginal_row = jnp.concatenate([mu_vec.astype(dtype), pad])

        if pieces.feasibility_func is not None:
            # Infeasible discrete combos: -inf value rows so they win no
            # maximum and carry zero choice probability; exactly-zero
            # marginal utility so probability-weighted sums stay finite.
            feasible = pieces.feasibility_func(**combo_pool)
            V_vec = jnp.where(feasible, V_vec, -jnp.inf)
            value_row = jnp.where(feasible, value_row, -jnp.inf)
            marginal_row = jnp.where(feasible, marginal_row, 0.0)

        return (
            V_vec.astype(dtype),
            grid_row,
            policy_row,
            value_row,
            marginal_row,
        )

    return solve_one_combo


def _compute_nodes_over_savings(
    *,
    compute_node: Callable,
    savings_nodes: Float1D,
    savings_batch_size: int,
) -> tuple[FloatND, FloatND, FloatND, FloatND]:
    """Run `compute_node` over every savings node, optionally splayed.

    A positive `savings_batch_size` below the grid length splays the
    per-savings-node continuation computation — the dominant egm_step working
    buffer (savings nodes by the child stochastic mesh by the combo block) — into
    `lax.map` blocks, shedding peak memory; 0 (or a size covering the whole
    grid) keeps the fused vmap. The output is identical either way: the per-node
    `(action, endogenous resources, value, expected continuation)` candidates
    stacked along the savings axis, which the constrained-region assembly and
    the upper envelope then consume on the full grid.
    """
    n_savings = savings_nodes.shape[0]
    if 0 < savings_batch_size < n_savings:
        return jax.lax.map(compute_node, savings_nodes, batch_size=savings_batch_size)
    return jax.vmap(compute_node)(savings_nodes)


def _get_compute_node(
    *,
    pieces: _EgmKernelPieces,
    combo_pool: dict[str, Any],
    discount_factor: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
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

    child_readers = {
        target: _get_child_carry_reader(
            read=pieces.child_reads[target],
            carry=next_regime_to_egm_carry[target],
            combo_pool=combo_pool,
            post_decision_name=pieces.post_decision_name,
            stochastic_node_batch_size=pieces.stochastic_node_batch_size,
        )
        for target in pieces.carry_targets
    }

    def compute_node(
        savings_value: ScalarFloat,
    ) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]:
        """Euler-invert one savings node against the continuation."""
        expected_marginal = jnp.asarray(0.0, dtype=dtype)
        expected_value = jnp.asarray(0.0, dtype=dtype)
        for target in pieces.carry_targets:
            # The smoothed marginal is already in savings space: the composed
            # gradient factor is applied per carry row inside the read.
            smoothed_value, smoothed_marginal = child_readers[target](savings_value)
            prob = regime_transition_probs[target]
            # Zero unreachable-target contributions on the results, never by
            # multiplying into a possibly non-finite value. The else branch
            # is `prob * 0.0` (not `0.0`) so a NaN probability poisons the
            # sum instead of vanishing.
            expected_marginal = expected_marginal + jnp.where(
                prob > 0.0, prob * smoothed_marginal, prob * 0.0
            )
            expected_value = expected_value + jnp.where(
                prob > 0.0, prob * smoothed_value, prob * 0.0
            )
        for target in pieces.scalar_targets:
            prob = regime_transition_probs[target]
            constant_value = next_regime_to_egm_carry[target].value[0]
            expected_value = expected_value + jnp.where(
                prob > 0.0, prob * constant_value, prob * 0.0
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


def _get_expected_continuation_value(
    *,
    pieces: _EgmKernelPieces,
    combo_pool: dict[str, Any],
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
    dtype: Any,  # noqa: ANN401
) -> Callable[[ScalarFloat], ScalarFloat]:
    """Build the expected-continuation map $W(A)$ for one combo pool.

    Mirrors the expected-value aggregation of the per-savings-node Euler
    inversion: per-target smoothed carry reads, regime-transition-probability
    weighted, plus the scalar targets' constant values. The asset-row mode
    differentiates this map in the Euler slot of the combo pool to obtain
    the direct Euler-state channel $\\partial W/\\partial a$. The
    probabilities, transition weights, and child next-state reads are all
    evaluated from the combo pool *inside* this builder, so when the pool's
    Euler slot is a traced value the gradient carries their first-order
    terms (e.g. $\\sum \\partial P/\\partial a \\cdot EV$) — precomputing
    them outside the differentiated closure would silently drop those
    terms (Danskin does not cancel them: the probabilities are not the
    softmax of the values they weight).
    """
    regime_transition_probs = pieces.compute_regime_transition_probs(**combo_pool)
    child_readers = {
        target: _get_child_carry_reader(
            read=pieces.child_reads[target],
            carry=next_regime_to_egm_carry[target],
            combo_pool=combo_pool,
            post_decision_name=pieces.post_decision_name,
            stochastic_node_batch_size=pieces.stochastic_node_batch_size,
        )
        for target in pieces.carry_targets
    }

    def expected_continuation(savings_value: ScalarFloat) -> ScalarFloat:
        expected_value = jnp.asarray(0.0, dtype=dtype)
        for target in pieces.carry_targets:
            smoothed_value, _ = child_readers[target](savings_value)
            prob = regime_transition_probs[target]
            # Zero unreachable-target contributions on the results, never by
            # multiplying into a possibly non-finite value. The else branch
            # is `prob * 0.0` (not `0.0`) so a NaN probability poisons the
            # sum instead of vanishing.
            expected_value = expected_value + jnp.where(
                prob > 0.0, prob * smoothed_value, prob * 0.0
            )
        for target in pieces.scalar_targets:
            prob = regime_transition_probs[target]
            constant_value = next_regime_to_egm_carry[target].value[0]
            expected_value = expected_value + jnp.where(
                prob > 0.0, prob * constant_value, prob * 0.0
            )
        return expected_value

    return expected_continuation


def _get_child_carry_reader(
    *,
    read: _ChildRead,
    carry: EGMCarry,
    combo_pool: dict[str, Any],
    post_decision_name: FunctionName,
    stochastic_node_batch_size: int,
) -> Callable[[ScalarFloat], tuple[ScalarFloat, ScalarFloat]]:
    """Build the per-savings-node carry read of one target for one combo.

    The returned callable maps a savings node to the target's smoothed
    continuation value and smoothed marginal continuation in savings space
    (the composed gradient $\\partial R'/\\partial A$ is applied per carry
    row inside the read). With child stochastic states (a continuous AR(1)
    process state or a Markov-discrete state), the read runs per child node
    combo and the per-node results are summed with the intrinsic transition
    weights $w(\\text{node}' \\mid \\text{node})$ — *outside* the
    discrete-action aggregation, matching the brute-force expectation over
    the already action-aggregated next-period V. The weights are evaluated
    once per combo (they depend on the current node values, params, and — in
    asset-row mode — the combo pool's Euler value, never on the savings
    node — validated).
    """
    weight_vecs: tuple[Float1D, ...] = ()
    if read.weights_func is not None:
        weights = read.weights_func(**combo_pool)
        weight_vecs = tuple(weights[key] for key in read.weight_keys)
    resources_reads_stochastic = bool(
        set(read.stochastic_state_names) & read.resources_arg_names
    )

    # The child's carry rows are fixed for the period; prepare each row's `+inf`
    # search key and valid prefix length once here — above the per-savings-node
    # and per-stochastic-node reads — so the per-query interpolation never
    # recomputes the row's NaN mask. That mask, recomputed and held for every
    # query lane, is the grid-length working buffer that dominates `egm_step` at
    # scale; preparing it once collapses it to a carry-sized array.
    n_carry_rows = carry.endog_grid.shape[-1]
    flat_search, flat_valid = jax.vmap(prepare_padded_grid)(
        carry.endog_grid.reshape(-1, n_carry_rows)
    )
    prepared_search_grid = flat_search.reshape(carry.endog_grid.shape)
    prepared_valid_length = flat_valid.reshape(carry.endog_grid.shape[:-1])

    def read_child(savings_value: ScalarFloat) -> tuple[ScalarFloat, ScalarFloat]:
        """Read the child's carry at one savings node."""
        # The solution-phase next-state function returns a flat mapping of
        # `next_<state>` names to scalars; the shared protocol's nested
        # return type is the simulation form. Everything but the child's
        # Euler state is savings-independent (validated), so these values
        # ride as constants through the composed gradients below.
        next_states = read.next_state_func(
            **combo_pool, **{post_decision_name: savings_value}
        )
        deterministic_index = tuple(
            cast("ScalarInt", next_states[f"next_{name}"])
            for name, is_stochastic in zip(
                read.discrete_state_names, read.stochastic_flags, strict=True
            )
            if not is_stochastic
        )
        child_passive_values = tuple(
            cast("ScalarFloat", next_states[f"next_{name}"])
            for name in read.passive_state_names
        )
        deterministic_resources_kwargs = {
            name: next_states[f"next_{name}"]
            for name, is_stochastic in zip(
                read.discrete_state_names, read.stochastic_flags, strict=True
            )
            if not is_stochastic and name in read.resources_arg_names
        }
        # The child resources function may read the regime's flat params and
        # `age` / `period` (e.g. a capital-income return rate); bind them from
        # the combo pool once. They are constant in the savings node, so they
        # ride through the composed gradients as constants.
        resources_param_kwargs = {
            name: combo_pool[name] for name in read.resources_param_names
        }

        def child_euler_state(savings: ScalarFloat) -> ScalarFloat:
            inner = read.next_state_func(**combo_pool, **{post_decision_name: savings})
            return cast("ScalarFloat", inner[read.next_state_key])

        def queries_and_gradients(
            stochastic_values: tuple[ScalarFloat | ScalarInt, ...],
        ) -> tuple[FloatND, FloatND]:
            return _compute_row_queries_and_gradients(
                read=read,
                child_euler_state=child_euler_state,
                deterministic_resources_kwargs=deterministic_resources_kwargs,
                resources_param_kwargs=resources_param_kwargs,
                savings_value=savings_value,
                stochastic_values=stochastic_values,
            )

        if not read.stochastic_state_names:
            queries, gradients = queries_and_gradients(())
            return _aggregate_child_choices(
                carry=carry,
                prepared_search_grid=prepared_search_grid,
                prepared_valid_length=prepared_valid_length,
                has_taste_shocks=read.has_taste_shocks,
                child_index=deterministic_index,
                child_passive_values=child_passive_values,
                child_passive_grids=read.passive_grids,
                row_queries=queries,
                row_gradients=gradients,
            )

        return _expect_over_stochastic_nodes(
            read=read,
            carry=carry,
            prepared_search_grid=prepared_search_grid,
            prepared_valid_length=prepared_valid_length,
            weight_vecs=weight_vecs,
            deterministic_index=deterministic_index,
            child_passive_values=child_passive_values,
            queries_and_gradients=queries_and_gradients,
            resources_reads_stochastic=resources_reads_stochastic,
            stochastic_node_batch_size=stochastic_node_batch_size,
        )

    return read_child


def _compute_row_queries_and_gradients(
    *,
    read: _ChildRead,
    child_euler_state: Callable[[ScalarFloat], ScalarFloat],
    deterministic_resources_kwargs: dict[str, Any],
    resources_param_kwargs: dict[str, Any],
    savings_value: ScalarFloat,
    stochastic_values: tuple[ScalarFloat | ScalarInt, ...],
) -> tuple[FloatND, FloatND]:
    """Per-row $R'$ queries and composed gradients for one node combo.

    The composed map differentiated per row is
    $A \\mapsto R'(\\mathcal{T}(A), z', d', p'; \\theta)$ — only the child's
    Euler state depends on the savings node; the discrete-state codes,
    stochastic node values, passive node values, action codes, and model
    params $\\theta$ ride as constants. With a simple resources function
    (Euler state and params only), one query and gradient is computed and
    broadcast across the row block.
    """
    if read.resources_is_simple:

        def composed(savings: ScalarFloat) -> ScalarFloat:
            return read.resources_func(
                **{read.euler_state_name: child_euler_state(savings)},
                **resources_param_kwargs,
            )

        query, gradient = jax.value_and_grad(composed)(savings_value)
        return (
            jnp.broadcast_to(query, read.row_block_shape),
            jnp.broadcast_to(gradient, read.row_block_shape),
        )

    # Empty when the resources function reads no stochastic state: the shared
    # (node-independent) computation passes no node values.
    stochastic_kwargs = (
        dict(zip(read.stochastic_state_names, stochastic_values, strict=True))
        if stochastic_values
        else {}
    )

    def composed_row(
        savings: ScalarFloat, row_values: tuple[ScalarFloat | ScalarInt, ...]
    ) -> ScalarFloat:
        bound = {
            read.euler_state_name: child_euler_state(savings),
            **deterministic_resources_kwargs,
            **stochastic_kwargs,
            **dict(zip(read.row_arg_names, row_values, strict=True)),
        }
        return read.resources_func(
            **{k: v for k, v in bound.items() if k in read.resources_arg_names},
            **resources_param_kwargs,
        )

    if read.row_values:
        queries, gradients = jax.vmap(
            jax.value_and_grad(composed_row), in_axes=(None, 0)
        )(savings_value, read.row_values)
        return (
            queries.reshape(read.row_block_shape),
            gradients.reshape(read.row_block_shape),
        )
    query, gradient = jax.value_and_grad(composed_row)(savings_value, ())
    return (
        jnp.broadcast_to(query, read.row_block_shape),
        jnp.broadcast_to(gradient, read.row_block_shape),
    )


def _expect_over_stochastic_nodes(
    *,
    read: _ChildRead,
    carry: EGMCarry,
    prepared_search_grid: FloatND,
    prepared_valid_length: IntND,
    weight_vecs: tuple[Float1D, ...],
    deterministic_index: tuple[ScalarInt, ...],
    child_passive_values: tuple[ScalarFloat, ...],
    queries_and_gradients: Callable[
        [tuple[ScalarFloat | ScalarInt, ...]], tuple[FloatND, FloatND]
    ],
    resources_reads_stochastic: bool,
    stochastic_node_batch_size: int,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Weight the carry read over the child's stochastic-node combos.

    Runs the full read (per-row queries, mixed passive interpolation, choice
    aggregation) at every child node combo and sums the per-node smoothed
    values and marginals with the joint intrinsic weights — the stochastic
    expectation sits *outside* the discrete-action aggregation, matching the
    brute-force solver's weighted average of the already action-aggregated
    next-period V. The node axes are the child's continuous AR(1) process
    states and Markov-discrete states alike; a Markov node feeds its integer
    code into the resources query (when read) and selects the carry's leading
    discrete axis by that code.
    """
    # The queries depend on the node combo only when the resources function
    # reads a stochastic state; otherwise compute them once and share.
    if not resources_reads_stochastic:
        shared_queries, shared_gradients = queries_and_gradients(())

    def read_at_nodes(
        node_indices: tuple[ScalarInt, ...],
    ) -> tuple[ScalarFloat, ScalarFloat]:
        """Run the full carry read at one child stochastic-node combo."""
        if resources_reads_stochastic:
            stochastic_values = tuple(
                values[index]
                for values, index in zip(
                    read.stochastic_node_values, node_indices, strict=True
                )
            )
            queries, gradients = queries_and_gradients(stochastic_values)
        else:
            queries, gradients = shared_queries, shared_gradients
        return _aggregate_child_choices(
            carry=carry,
            prepared_search_grid=prepared_search_grid,
            prepared_valid_length=prepared_valid_length,
            has_taste_shocks=read.has_taste_shocks,
            child_index=_interleave_child_index(
                deterministic_index=deterministic_index,
                node_indices=node_indices,
                stochastic_flags=read.stochastic_flags,
            ),
            child_passive_values=child_passive_values,
            child_passive_grids=read.passive_grids,
            row_queries=queries,
            row_gradients=gradients,
        )

    node_index_mesh = jnp.meshgrid(
        *(
            jnp.arange(values.shape[0], dtype=jnp.int32)
            for values in read.stochastic_node_values
        ),
        indexing="ij",
    )
    flat_node_indices = tuple(mesh.ravel() for mesh in node_index_mesh)
    # The expectation mesh (the product of the child's stochastic-node counts)
    # is the dominant `egm_step` working buffer's child-node axis. A positive
    # `stochastic_node_batch_size` below the mesh length runs the per-node reads
    # in `lax.map` blocks, shedding that buffer's peak; `0` (or a size covering
    # the whole mesh) keeps the fused vmap. The per-node results are summed with
    # the joint weights below either way, so the value function is identical.
    n_nodes = flat_node_indices[0].shape[0]
    if 0 < stochastic_node_batch_size < n_nodes:
        node_values, node_marginals = jax.lax.map(
            read_at_nodes, flat_node_indices, batch_size=stochastic_node_batch_size
        )
    else:
        node_values, node_marginals = jax.vmap(read_at_nodes)(flat_node_indices)
    joint_weights = weight_vecs[0][flat_node_indices[0]]
    for vec, indices in zip(weight_vecs[1:], flat_node_indices[1:], strict=True):
        joint_weights = joint_weights * vec[indices]
    # Weight on results: a zero-weight node contributes exactly 0.0 even
    # when its smoothed value is -inf (never 0 * inf = NaN). The else branch
    # is `weights * 0.0` (not `0.0`) so a NaN weight poisons the sum instead
    # of vanishing.
    smoothed_value = jnp.sum(
        jnp.where(joint_weights > 0.0, joint_weights * node_values, joint_weights * 0.0)
    )
    smoothed_marginal = jnp.sum(
        jnp.where(
            joint_weights > 0.0, joint_weights * node_marginals, joint_weights * 0.0
        )
    )
    return smoothed_value, smoothed_marginal


def _interleave_child_index(
    *,
    deterministic_index: tuple[ScalarInt, ...],
    node_indices: tuple[ScalarInt, ...],
    stochastic_flags: tuple[bool, ...],
) -> tuple[ScalarInt, ...]:
    """Merge deterministic codes and stochastic node indices in carry-axis order."""
    deterministic_iter = iter(deterministic_index)
    node_iter = iter(node_indices)
    return tuple(
        next(node_iter) if is_stochastic else next(deterministic_iter)
        for is_stochastic in stochastic_flags
    )


def _hard_max_and_one_hot(
    *, values: FloatND, axes: tuple[int, ...]
) -> tuple[FloatND, FloatND]:
    """Hard maximum over `axes` and a one-hot indicator of the (first) argmax.

    The no-taste-shocks aggregation: the smoothed maximum a regime without
    declared EV1 taste shocks uses in place of the `scale > 0` logsum. Ties
    break toward the first flat index. A slice that is `-inf` everywhere yields
    a `-inf` maximum and a one-hot at index 0, so the marginal it weights (zero
    on infeasible rows) stays consistent.

    Args:
        values: Choice-specific values; infeasible entries are `-inf`.
        axes: Axes to aggregate over (the discrete-choice axes).

    Returns:
        Tuple of the hard maximum (shape of `values` with `axes` removed) and
        the one-hot argmax indicator (shape of `values`).

    """
    hard_max = jnp.max(values, axis=axes)
    moved = jnp.moveaxis(values, axes, tuple(range(len(axes))))
    lead_shape = moved.shape[: len(axes)]
    flat = moved.reshape((-1, *moved.shape[len(axes) :]))
    one_hot = (
        jnp.arange(flat.shape[0]).reshape((-1,) + (1,) * (flat.ndim - 1))
        == jnp.argmax(flat, axis=0)
    ).astype(values.dtype)
    one_hot = jnp.moveaxis(
        one_hot.reshape(lead_shape + moved.shape[len(axes) :]),
        tuple(range(len(axes))),
        axes,
    )
    return hard_max, one_hot


def _aggregate_child_choices(
    *,
    carry: EGMCarry,
    prepared_search_grid: FloatND,
    prepared_valid_length: IntND,
    has_taste_shocks: bool,
    child_index: tuple[ScalarInt, ...],
    child_passive_values: tuple[ScalarFloat, ...],
    child_passive_grids: tuple[Float1D, ...],
    row_queries: FloatND,
    row_gradients: FloatND,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Read one child's carry with mixed interpolation and aggregate its choices.

    The carry rows matching the child's discrete-state values are selected
    by integer indexing on the leading state axes (discrete codes equal grid
    positions, stochastic dims indexed at one node); the remaining leading axes
    are the child's passive nodes, then its discrete-action combos. Every
    row is interpolated 1-D at its own resources query and its marginal is
    multiplied by its own composed gradient $(\\partial R'/\\partial A)$ —
    per row, because each row's envelope lives in its own resources space.
    The passive axes are then blended away with edge-clamped linear weights
    on the two neighboring nodes of each passive grid — *before* the choice
    aggregation, so the logsum sees blended choice-specific values. Finally
    the discrete-action rows are aggregated with the child's taste-shock
    scale: the smoothed value is the logsum and the smoothed marginal is
    $\\sum_{d'} P_{d'} \\mu_{d'} (\\partial R'/\\partial A)_{d'}$ — exact
    for EV1 by Danskin's theorem, no $\\partial P/\\partial R$ terms. Scale
    zero yields the hard max / one-hot argmax through the same code path.
    Rows that are $-\\infty$ everywhere (infeasible child combos) get zero
    probability and contribute exactly zero marginal utility; a zero-weight
    passive neighbor contributes exactly zero even when its row is
    $-\\infty$ (`jnp.where` on results, never `0 \\cdot \\infty`).

    Args:
        carry: The child's EGM carry.
        prepared_search_grid: The whole carry's `+inf`-padded search key
            (`carry.endog_grid`'s shape), prepared once above the read fan-out.
        prepared_valid_length: The whole carry's per-row valid prefix lengths
            (`carry.endog_grid`'s shape without the row axis).
        child_index: The child's discrete-state values at this savings node
            (stochastic dims: the node index of this read).
        child_passive_values: The child's passive values at this savings
            node, aligned with `child_passive_grids`.
        child_passive_grids: The child's passive grids in carry-axis order.
        row_queries: Per-row resources queries with the row block's shape
            (passive dims, then action dims).
        row_gradients: Per-row composed gradients $\\partial R'/\\partial A$
            with the row block's shape.

    Returns:
        Tuple of the smoothed continuation value and the smoothed marginal
        continuation $\\partial W/\\partial A$.

    """
    n_pad = carry.value.shape[-1]
    grid_block = carry.endog_grid[child_index]
    value_block = carry.value[child_index]
    marginal_block = carry.marginal_utility[child_index]
    # The prepared search key and valid length are indexed by the same
    # `child_index` as the carry rows, so each row reads its own precomputed
    # pair instead of recomputing the NaN mask per query.
    search_block = prepared_search_grid[child_index]
    valid_block = prepared_valid_length[child_index]
    # Leading axes of the blocks: the child's passive nodes, then its
    # discrete-action combos.
    block_shape = value_block.shape[:-1]
    grid_rows = grid_block.reshape(-1, n_pad)
    value_rows = value_block.reshape(-1, n_pad)
    marginal_rows = marginal_block.reshape(-1, n_pad)
    search_rows = search_block.reshape(-1, n_pad)
    valid_rows = valid_block.reshape(-1)
    queries_flat = row_queries.reshape(-1)
    gradients_flat = row_gradients.reshape(-1)

    # The marginal-utility row is the value row's exact slope (envelope
    # theorem), upgrading the value read to cubic Hermite; the mu read itself
    # stays linear (a policy-grade quantity, and its interpolation error
    # enters the value only at second order through the Euler inversion).
    def interp_value_row(
        search_grid: Float1D,
        valid_length: ScalarInt,
        xp: Float1D,
        fp: Float1D,
        fp_slopes: Float1D,
        x_query: ScalarFloat,
    ) -> ScalarFloat:
        """Interpolate one carry value row at its query; positional per `jax.vmap`."""
        return interp_on_prepared_grid(
            x_query=x_query,
            search_grid=search_grid,
            valid_length=valid_length,
            xp=xp,
            fp=fp,
            fp_slopes=fp_slopes,
        )

    def interp_row(
        search_grid: Float1D,
        valid_length: ScalarInt,
        xp: Float1D,
        fp: Float1D,
        x_query: ScalarFloat,
    ) -> ScalarFloat:
        """Interpolate one carry row at its own query; positional per `jax.vmap`."""
        return interp_on_prepared_grid(
            x_query=x_query,
            search_grid=search_grid,
            valid_length=valid_length,
            xp=xp,
            fp=fp,
        )

    value_at_child = jax.vmap(interp_value_row)(
        search_rows, valid_rows, grid_rows, value_rows, marginal_rows, queries_flat
    )
    marginal_at_child = jax.vmap(interp_row)(
        search_rows, valid_rows, grid_rows, marginal_rows, queries_flat
    )
    # `-inf` entries interpolate pointwise to `-inf` (never NaN) and carry
    # exactly-zero marginal utility, so an infeasible-everywhere row reads as
    # the `-inf` / zero pair while a row with isolated `-inf` nodes (e.g. a
    # bequest at zero wealth) keeps its finite region intact. A `-inf` value
    # read pins the marginal read to zero so the pair stays consistent at
    # queries clamped onto a `-inf` node.
    marginal_at_child = jnp.where(
        jnp.isneginf(value_at_child), 0.0, marginal_at_child * gradients_flat
    )
    value_at_child = value_at_child.reshape(block_shape)
    marginal_at_child = marginal_at_child.reshape(block_shape)

    for passive_value, passive_grid in zip(
        child_passive_values, child_passive_grids, strict=True
    ):
        lower, upper, weight_upper = locate_on_grid(
            x_query=passive_value, grid=passive_grid
        )
        weight_lower = 1.0 - weight_upper
        # Blend on results: a zero-weight neighbor contributes exactly 0.0,
        # so an on-node read reproduces the node rows and a -inf neighbor
        # never turns into 0 * inf = NaN; a positive-weight -inf neighbor
        # correctly forces the blend to -inf.
        value_at_child = jnp.where(
            weight_lower > 0.0, weight_lower * value_at_child[lower], 0.0
        ) + jnp.where(weight_upper > 0.0, weight_upper * value_at_child[upper], 0.0)
        # Marginal rows are finite everywhere (exactly 0.0 on infeasible
        # rows), so a plain blend is safe.
        marginal_at_child = (
            weight_lower * marginal_at_child[lower]
            + weight_upper * marginal_at_child[upper]
        )

    value_at_child = value_at_child.reshape(-1)
    marginal_at_child = marginal_at_child.reshape(-1)
    if has_taste_shocks:
        smoothed_value, choice_probs = logsum_and_softmax(
            values=value_at_child, scale=carry.taste_shock_scale, axes=(0,)
        )
    else:
        smoothed_value, choice_probs = _hard_max_and_one_hot(
            values=value_at_child, axes=(0,)
        )
    smoothed_marginal = jnp.sum(choice_probs * marginal_at_child)
    return smoothed_value, smoothed_marginal


def _build_child_reads(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    carry_targets: tuple[RegimeName, ...],
    post_decision_name: FunctionName,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
) -> MappingProxyType[RegimeName, _ChildRead]:
    """Build the per-carry-target statics of the EGM kernel's child reads.

    The post-decision function is removed from the DAG, so its output (the
    savings node) becomes an external input of the next-state functions.
    Passing the full `functions` mapping would let the DAG compute savings
    internally from the (unknown) state and action leaves — it runs, but is
    silently wrong.

    Returns:
        Immutable mapping of carry-target names to their read statics.

    """
    functions_without_post = MappingProxyType(
        {name: func for name, func in functions.items() if name != post_decision_name}
    )
    reads: dict[RegimeName, _ChildRead] = {}
    for target in carry_targets:
        target_info = regime_to_v_interpolation_info[target]
        target_regime = user_regimes[target]
        euler_state_name = _get_child_state_name(user_regime=target_regime)
        discrete_state_names = _get_discrete_state_names(
            v_interpolation_info=target_info
        )
        # A discrete dimension is a stochastic node axis when its next-period
        # node is distributed by a transition law: a continuous AR(1) process
        # state, or a Markov-discrete state whose `next_<name>` is a stochastic
        # transition into this target. Both are integrated over the child's
        # node axis with the intrinsic weights `weight_<target>__next_<name>`.
        target_transition_names = frozenset(transitions[target])

        def _is_stochastic(
            name: StateName,
            target_info: VInterpolationInfo = target_info,
            target_transition_names: frozenset[
                TransitionFunctionName
            ] = target_transition_names,
        ) -> bool:
            if isinstance(
                target_info.discrete_states[name], _ContinuousStochasticProcess
            ):
                return True
            transition_name = f"next_{name}"
            return (
                transition_name in target_transition_names
                and transition_name in stochastic_transition_names
            )

        stochastic_flags = tuple(_is_stochastic(name) for name in discrete_state_names)
        stochastic_state_names = tuple(
            name
            for name, is_stochastic in zip(
                discrete_state_names, stochastic_flags, strict=True
            )
            if is_stochastic
        )
        # Process axes feed the continuous AR(1) grid points into the resources
        # query; Markov axes feed their integer category codes (`to_jax()`
        # returns `[0, 1, ...]`, which also equal the carry's leading-axis
        # indices). Both serve as the node-axis range of the integration mesh.
        stochastic_node_values = tuple(
            (
                jnp.asarray(
                    target_info.discrete_states[name].to_jax(),
                    dtype=canonical_float_dtype(),
                )
                if isinstance(
                    target_info.discrete_states[name], _ContinuousStochasticProcess
                )
                else jnp.asarray(target_info.discrete_states[name].to_jax())
            )
            for name in stochastic_state_names
        )
        weight_keys = tuple(
            f"weight_{target}__next_{name}" for name in stochastic_state_names
        )
        weights_func = None
        if weight_keys:
            weights_func = concatenate_functions(
                functions={
                    name: func
                    for name, func in functions_without_post.items()
                    if name != "H"
                },
                targets=list(weight_keys),
                return_type="dict",
                enforce_signature=False,
                set_annotations=True,
            )
        passive_state_names = _get_passive_state_names(
            v_interpolation_info=target_info,
            euler_state_name=euler_state_name,
        )
        passive_grids = tuple(
            jnp.asarray(
                target_info.continuous_states[name].to_jax(),
                dtype=canonical_float_dtype(),
            )
            for name in passive_state_names
        )
        action_names, action_values = _get_child_discrete_actions(
            user_regime=target_regime
        )
        resources_func = _get_child_resources_function(user_regime=target_regime)
        resources_arg_names = frozenset(
            _get_child_resources_arg_names(user_regime=target_regime)
        )
        # Everything the resources function reads beyond the child's own
        # states and discrete actions is a (qualified) param or `age` /
        # `period`: a per-node constant bound from the combo pool.
        child_binding_names = (
            {euler_state_name}
            | set(discrete_state_names)
            | set(passive_state_names)
            | set(action_names)
        )
        resources_param_names = resources_arg_names - child_binding_names
        row_grids = passive_grids + action_values
        if row_grids:
            row_mesh = jnp.meshgrid(*row_grids, indexing="ij")
            row_values = tuple(mesh.ravel() for mesh in row_mesh)
            row_block_shape = tuple(int(grid.shape[0]) for grid in row_grids)
        else:
            row_values = ()
            row_block_shape = ()
        reads[target] = _ChildRead(
            next_state_func=get_next_state_function_for_solution(
                transitions=transitions[target],
                functions=functions_without_post,
            ),
            next_state_key=f"next_{euler_state_name}",
            euler_state_name=euler_state_name,
            has_taste_shocks=target_regime.taste_shocks is not None,
            resources_func=resources_func,
            resources_arg_names=resources_arg_names,
            resources_param_names=resources_param_names,
            resources_is_simple=(resources_arg_names - resources_param_names)
            <= {euler_state_name},
            discrete_state_names=discrete_state_names,
            stochastic_flags=stochastic_flags,
            stochastic_state_names=stochastic_state_names,
            stochastic_node_values=stochastic_node_values,
            weight_keys=weight_keys,
            weights_func=weights_func,
            passive_state_names=passive_state_names,
            passive_grids=passive_grids,
            row_arg_names=passive_state_names + action_names,
            row_values=row_values,
            row_block_shape=row_block_shape,
        )
    return MappingProxyType(reads)


def _get_child_discrete_actions(
    *, user_regime: UserRegime
) -> tuple[tuple[ActionName, ...], tuple[Any, ...]]:
    """Discrete-action names and grid values of a carry target, in combo order.

    The order matches the target's own kernel combos (its state-action
    space's discrete actions), so per-row bindings line up with the carry's
    action axes. Terminal targets have no actions (guarded).
    """
    variables = from_regime(user_regime)
    grids = get_grids(user_regime)
    names = tuple(variables.discrete_action_names)
    return names, tuple(grids[name].to_jax() for name in names)


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


def _get_passive_state_names(
    *,
    v_interpolation_info: VInterpolationInfo,
    euler_state_name: StateName,
) -> tuple[StateName, ...]:
    """Passive-state names of a regime in carry-axis (V continuous-state) order.

    Every continuous state other than the Euler state is passive — DC-EGM
    validation enforces this for DC-EGM regimes, and terminal carry targets
    are restricted to a single continuous state.
    """
    return tuple(
        name
        for name in v_interpolation_info.continuous_states
        if name != euler_state_name
    )


def _concatenate_regime_function(
    *,
    functions: EconFunctionsMapping,
    target: FunctionName,
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

    # The carry value row is the refined upper envelope itself — the correct
    # object for a parent to interpolate. The constrained floor applies only to
    # the published `V_row` on the exogenous grid; flooring the carry would lift
    # finite envelope nodes above the true envelope wherever the closed-form
    # constrained value exceeds them, overstating the parent's continuation.
    value_row = jnp.where(overflowed, jnp.nan, refined_value).astype(dtype)
    return V_row, value_row, marginal_utility.astype(dtype)


def _publish_node_V_and_policy(
    *,
    refined_grid: Float1D,
    refined_policy: Float1D,
    refined_value: Float1D,
    n_kept: ScalarInt,
    n_pad: int,
    resources_at_node: ScalarFloat,
    borrowing_limit: ScalarFloat,
    utility_of_action: Callable[[ScalarFloat], ScalarFloat],
    discounted_expected_value_at_limit: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Publish one asset node's value and optimal action at its single query.

    The scalar-query counterpart of `_publish_V_and_carry_rows`: below the
    lowest refined envelope point the closed-form constrained value is
    published outright; everywhere else the Hermite read of the refined
    envelope is floored at the constrained value, which remains a
    feasible-policy lower bound. The published action follows the winning
    branch — the closed-form action `R - borrowing_limit` where the
    constrained value wins, the interpolated refined policy otherwise.
    Envelope overflow NaN-poisons both outputs so the solve loop's NaN
    diagnostics surface the offending (regime, period).

    Args:
        refined_grid: Refined endogenous grid row from the envelope backend.
        refined_policy: Refined policy row.
        refined_value: Refined value row.
        n_kept: Number of envelope points the backend kept.
        n_pad: Static length of the refined rows.
        resources_at_node: Resources at this exogenous Euler node (the row's
            single publish query).
        borrowing_limit: Lower bound of the savings grid.
        utility_of_action: Utility with everything but the continuous action
            bound.
        discounted_expected_value_at_limit: Discounted expected continuation
            value at the lowest savings node.

    Returns:
        Tuple of the node's published value and published optimal action.

    """
    dtype = resources_at_node.dtype
    overflowed = n_kept > n_pad

    marginal_utility = jax.vmap(jax.grad(utility_of_action))(
        jnp.where(jnp.isnan(refined_policy), 1.0, refined_policy)
    )
    marginal_utility = jnp.where(jnp.isnan(refined_policy), jnp.nan, marginal_utility)
    marginal_utility = jnp.where(jnp.isneginf(refined_value), 0.0, marginal_utility)

    value_interpolated = interp_on_padded_grid(
        x_query=resources_at_node,
        xp=refined_grid,
        fp=refined_value,
        fp_slopes=marginal_utility,
    )
    policy_interpolated = interp_on_padded_grid(
        x_query=resources_at_node,
        xp=refined_grid,
        fp=refined_policy,
    )
    closed_form_action = resources_at_node - borrowing_limit
    value_constrained = jnp.where(
        closed_form_action > 0.0,
        utility_of_action(jnp.maximum(closed_form_action, jnp.finfo(dtype).tiny))
        + discounted_expected_value_at_limit,
        -jnp.inf,
    )
    below_refined = (closed_form_action > 0.0) & (resources_at_node <= refined_grid[0])
    constrained_wins = below_refined | (value_constrained >= value_interpolated)
    V_node = jnp.where(
        below_refined,
        value_constrained,
        jnp.maximum(value_interpolated, value_constrained),
    )
    policy_node = jnp.where(constrained_wins, closed_form_action, policy_interpolated)
    V_node = jnp.where(overflowed, jnp.nan, V_node).astype(dtype)
    policy_node = jnp.where(overflowed, jnp.nan, policy_node).astype(dtype)
    return V_node, policy_node


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


def _find_unsupported_feature(
    *,
    solver: DCEGM,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    carry_targets: tuple[RegimeName, ...],
    transitions: TransitionFunctionsMapping,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    flat_param_names: frozenset[str],
    regime_to_flat_param_names: MappingProxyType[RegimeName, frozenset[str]],
    own_discrete_state_names: tuple[StateName, ...],
    own_passive_state_names: tuple[StateName, ...],
    own_discrete_action_names: tuple[ActionName, ...],
    asset_row_mode: bool,
) -> str | None:
    """Return a message naming the first feature outside the kernel's scope.

    Returns `None` when the configuration is fully supported.
    """
    message: str | None = None
    for target in carry_targets:
        message = _find_unsupported_target_feature(
            target=target,
            user_regimes=user_regimes,
            functions=functions,
            transitions=transitions,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            own_discrete_state_names=own_discrete_state_names,
            allowed_param_names=flat_param_names | regime_to_flat_param_names[target],
        )
        if message is not None:
            break

    if message is None:
        message = _find_unsupported_function_args(
            solver=solver,
            functions=functions,
            constraints=constraints,
            carry_targets=carry_targets,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            flat_param_names=flat_param_names,
            own_discrete_state_names=own_discrete_state_names,
            own_passive_state_names=own_passive_state_names,
            own_discrete_action_names=own_discrete_action_names,
            asset_row_mode=asset_row_mode,
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
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    own_discrete_state_names: tuple[StateName, ...],
    allowed_param_names: frozenset[str],
) -> str | None:
    """Return a message naming the first unsupported feature of one target.

    `allowed_param_names` is the union of the source regime's flat params and
    the target regime's flat params: a cross-regime carry evaluates the
    target's resources / transition functions, which read the *target's*
    params (e.g. a pension factor the source never reads), so admitting the
    target's params mirrors the kernel's runtime reach.
    """
    target_info = regime_to_v_interpolation_info[target]
    target_process_states = _get_process_state_names(v_interpolation_info=target_info)
    if user_regimes[target].terminal:
        terminal_message = _find_unsupported_terminal_target_feature(
            target=target,
            user_regime=user_regimes[target],
            target_info=target_info,
            target_transitions=transitions[target],
            own_discrete_state_names=own_discrete_state_names,
        )
        if terminal_message is not None:
            return terminal_message
    for process_name in target_process_states:
        # The child's node distribution comes from the intrinsic transition
        # of the shared process state; without it (the source regime does
        # not carry the process) there is nothing to weight the child's
        # node axis with.
        has_transition = f"next_{process_name}" in transitions[target]
        has_weights = f"weight_{target}__next_{process_name}" in functions
        if not (has_transition and has_weights):
            return (
                f"the process state '{process_name}' of target regime "
                f"'{target}' has no intrinsic transition from this regime "
                "(both regimes must carry the same process state)."
            )
    child_state_name = _get_child_state_name(user_regime=user_regimes[target])
    resources_arg_names = _get_child_resources_arg_names(
        user_regime=user_regimes[target]
    )
    child_action_names, _ = _get_child_discrete_actions(
        user_regime=user_regimes[target]
    )
    # Mirror the function-args allowance (`_find_unsupported_function_args`):
    # beyond the child's states and discrete actions, the resources function
    # may read the regime's flat params and the lifecycle `age` / `period`.
    # Solve-phase imputed intermediates (a carried state's solve law) are
    # baked into the resources DAG, so their leaf inputs (the passive states
    # and params they read) are what surfaces here — the imputed output is
    # computed, never demanded as a leaf.
    allowed_resources_args = (
        {child_state_name}
        | set(target_info.state_names)
        | set(child_action_names)
        | set(allowed_param_names)
        | {"age", "period"}
    )
    extra_resources_args = sorted(resources_arg_names - allowed_resources_args)
    if extra_resources_args:
        return (
            f"the resources function of target regime '{target}' depends on "
            f"{extra_resources_args}; beyond the Euler state it may read "
            "only the child's states and discrete actions, the regime's "
            "params, and age/period."
        )
    for process_name in target_process_states:
        grid = target_info.discrete_states[process_name]
        if (
            process_name in resources_arg_names
            and not cast("_ContinuousStochasticProcess", grid).is_fully_specified
        ):
            return (
                f"the resources function of target regime '{target}' reads "
                f"the process state '{process_name}', whose grid points are "
                "supplied at runtime."
            )
    return None


def _find_unsupported_terminal_target_feature(
    *,
    target: RegimeName,
    user_regime: UserRegime,
    target_info: VInterpolationInfo,
    target_transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
    own_discrete_state_names: tuple[StateName, ...],
) -> str | None:
    """Return a message naming the first unsupported feature of a terminal target.

    A terminal carry covers exactly one continuous state and no actions. It
    may additionally carry discrete states, but only those that are *shared*
    with the parent's own discrete combo axes and reached by the *identity*
    (fixed) transition: at a parent combo with the state at value $k$ the
    terminal carry row is its utility evaluated at $k$, so the parent selects
    its own combo by integer indexing the carry's leading axis. A terminal
    discrete state that the parent does not carry has no parent axis to align
    to; one reached by a stochastic or non-identity transition is a separate,
    unsupported gap.
    """
    own_discrete = set(own_discrete_state_names)
    for name, grid in target_info.discrete_states.items():
        if isinstance(grid, _ContinuousStochasticProcess):
            return (
                f"its terminal target regime '{target}' has the process state "
                f"'{name}'; terminal carries cover discrete states shared with "
                "the parent via an identity transition only."
            )
        if name not in own_discrete:
            return (
                f"its terminal target regime '{target}' has the discrete state "
                f"'{name}', which the parent does not carry; a terminal carry's "
                "discrete states must be shared with the parent's own discrete "
                "combo axes."
            )
        transition = target_transitions.get(f"next_{name}")
        if not getattr(transition, "_is_auto_identity", False):
            return (
                f"its terminal target regime '{target}' is reached by a "
                f"non-identity transition into the discrete state '{name}'; a "
                "terminal carry's discrete states must be shared with the "
                "parent via an identity (fixed) transition."
            )
    n_continuous = len(target_info.continuous_states)
    if n_continuous != 1:
        return (
            f"its terminal target regime '{target}' has continuous states "
            f"{list(target_info.continuous_states)}; exactly one continuous "
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
    carry_targets: tuple[RegimeName, ...],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    flat_param_names: frozenset[str],
    own_discrete_state_names: tuple[StateName, ...],
    own_passive_state_names: tuple[StateName, ...],
    own_discrete_action_names: tuple[ActionName, ...],
    asset_row_mode: bool,
) -> str | None:
    """Return a message naming the first function with out-of-scope arguments."""
    # Combo inputs are bound per (discrete state, passive node, discrete
    # action) combination, so any of them may feed these functions.
    allowed_combo_inputs = (
        set(own_discrete_state_names)
        | set(own_passive_state_names)
        | set(own_discrete_action_names)
    )
    # In asset-row mode the combo pool carries the Euler node's value, so
    # savings-stage functions (regime transition probabilities, transition
    # weights) may read the Euler state; the single-post-state kernel has no
    # Euler value in the pool.
    allowed_savings_stage_inputs = allowed_combo_inputs | (
        {solver.continuous_state} if asset_row_mode else set()
    )
    allowed_params = flat_param_names | {"age", "period"}
    utility_func = _concatenate_regime_function(functions=functions, target="utility")
    arg_requirements: list[tuple[str, frozenset[str], set[str]]] = [
        (
            "the utility function",
            frozenset(get_union_of_args([utility_func])),
            {solver.continuous_action} | allowed_combo_inputs | allowed_params,
        ),
        (
            "the regime transition probability function",
            frozenset(get_union_of_args([compute_regime_transition_probs])),
            allowed_savings_stage_inputs | allowed_params,
        ),
    ]
    # Intrinsic process-weight functions are evaluated per combo at the
    # savings-node stage, mirroring the savings-stage independence the
    # validation requires of every other stochastic weight function.
    for target in carry_targets:
        target_process_states = _get_process_state_names(
            v_interpolation_info=regime_to_v_interpolation_info[target]
        )
        for process_name in target_process_states:
            weight_key = f"weight_{target}__next_{process_name}"
            if weight_key not in functions:
                continue
            weight_func = _concatenate_regime_function(
                functions=functions, target=weight_key
            )
            arg_requirements.append(
                (
                    f"the transition-weight function '{weight_key}'",
                    frozenset(get_union_of_args([weight_func])),
                    allowed_savings_stage_inputs | allowed_params,
                )
            )
    for constraint_name in constraints:
        constraint_func = _concatenate_regime_function(
            functions=MappingProxyType({**dict(functions), **dict(constraints)}),
            target=constraint_name,
        )
        arg_requirements.append(
            (
                f"the constraint '{constraint_name}'",
                frozenset(get_union_of_args([constraint_func])),
                allowed_combo_inputs | allowed_params,
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
) -> Callable[..., ScalarFloat]:
    """Build the closed-over resources map of one carry target.

    For a DC-EGM target the map is its declared resources function (resolved
    to the solve-phase variant); for a terminal target the carry lives in
    M-space and the map is the identity. The returned callable takes the
    child's state, passive, and discrete-action values as keyword arguments
    (child names) so the kernel can compose it with the state transition and
    differentiate the composition per carry row.
    """
    if isinstance(user_regime.solver, DCEGM):
        return _concatenate_child_resources(user_regime=user_regime)

    state_name = next(iter(user_regime.states))

    def identity_resources(**kwargs: ScalarFloat) -> ScalarFloat:
        return kwargs[state_name]

    return identity_resources


def _get_child_resources_arg_names(*, user_regime: UserRegime) -> set[str]:
    """Argument names of a carry target's resources map."""
    if isinstance(user_regime.solver, DCEGM):
        return set(
            get_union_of_args([_concatenate_child_resources(user_regime=user_regime)])
        )
    return {next(iter(user_regime.states))}


def _concatenate_child_resources(*, user_regime: UserRegime) -> UserFunction:
    """Concatenate a DC-EGM target's resources function from its user DAG.

    Each user function's params are renamed to their qualified names
    (`<func>__<param>`) before concatenation, matching the engine's binding
    vocabulary so the kernel feeds them straight from the combo pool (which
    carries the regime's flat params). Solve-phase imputed intermediates (a
    `Phased` function or a carried state's solve law) are resolved to their
    solve variant and baked into the DAG, so their outputs are computed from
    leaf states and params rather than demanded as leaves.
    """
    # Imported lazily: `regime_building.processing` imports the solver
    # registry, which imports this module, so a top-level import would cycle.
    from _lcm.regime_building import processing as _proc  # noqa: PLC0415

    solver = cast("DCEGM", user_regime.solver)
    regime_params_template = create_regime_params_template(user_regime)
    resolved: dict[str, UserFunction] = {}
    for name, func in user_regime.functions.items():
        if isinstance(func, Phased):
            resolved[name] = cast("UserFunction", func.solve)
        elif func is not None:
            resolved[name] = func
    # A carried state contributes a solve-phase imputation function under its
    # own name; include it so resources may read the imputed value.
    for name, value in user_regime.states.items():
        if isinstance(value, Phased) and name not in resolved:
            resolved[name] = cast("UserFunction", value.solve)
    qnamed = {
        name: _proc._rename_params_to_qnames(  # noqa: SLF001
            func=func,
            regime_params_template=regime_params_template,
            param_key=name,
        )
        for name, func in resolved.items()
    }
    return concatenate_functions(
        functions=qnamed,
        targets=solver.resources,
        enforce_signature=False,
        set_annotations=True,
    )
