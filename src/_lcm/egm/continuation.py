"""The DC-EGM continuation: expected next-period value and marginal over targets.

Both the per-savings-node Euler inversion and the asset-row Euler-state gradient
need one quantity — the expected continuation as a function of end-of-period
savings, aggregated over the regime's reachable next-period targets. This module
builds that aggregation from the period's carry: it selects each target's carry
rows by the child's discrete state, maps the child state into the child's
resources space, interpolates each row at its own next-period resources, blends
over passive states, smooths the child's discrete choices (hard max or EV1
logsum), and weights stochastic-process nodes by their intrinsic transition. The
core EGM step and the asset-row step consume the per-target reader it produces;
everything multi-target, passive-state, taste-shock, and stochastic-node lives
behind that boundary.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
from dags import concatenate_functions

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry
from _lcm.egm.ez_kernel import (
    ez_blend_partials,
    ez_invert_partials,
    ez_transform_partials,
    ez_transform_scalar,
)
from _lcm.egm.interp import (
    interp_and_derivative_on_prepared_grid,
    interp_on_prepared_grid,
    interp_right_germ_on_prepared_grid,
    locate_on_grid,
    prepare_padded_grid,
)
from _lcm.egm.nbegm import jump_moving_state_names
from _lcm.egm.outer_envelope import right_germ_winner
from _lcm.egm.regime_introspection import (
    _get_child_discrete_actions,
    _get_child_resources_arg_names,
    _get_child_resources_function,
    _get_child_state_name,
    _get_discrete_state_names,
    _get_passive_state_names,
)
from _lcm.grids import Grid
from _lcm.logsum import logsum_and_softmax
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.next_state import get_next_state_function_for_solution
from _lcm.regime_building.Q_and_F import get_period_targets
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import (
    ActionName,
    EconFunctionsMapping,
    FunctionName,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)

# The anchored Epstein-Zin partials `(a, W, E, b, T~)` a child reader returns
# when a certainty equivalent is active.
type _EZPartials = tuple[
    ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat
]


def _is_runtime_process(grid: Grid) -> bool:
    """Whether the grid is a process whose nodes resolve only at solve time."""
    return (
        isinstance(grid, _ContinuousStochasticProcess) and not grid.is_fully_specified
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

    foldable_stochastic_flags: tuple[bool, ...]
    """Per stochastic dimension: whether its expectation folds into the carry.

    A dimension folds when its nodes cannot move the per-row resources
    queries or the aggregation those queries feed, so its expectation
    commutes with the row-linear carry interpolation and pre-folds into an
    expected carry once per cell (instead of looping its nodes per savings
    query). All three conditions are static model topology:

    - the child's resources DAG does not read the dimension's node value;
    - the child's carry rows share the state grid as abscissae
      (`Solver.carry_rows_share_state_grid`), so every node row interpolates
      with the same bracket structure;
    - the carry keeps no per-discrete-action rows
      (`Solver.carry_retains_discrete_action_rows` is `False`), so no
      per-node choice aggregation sits between interpolation and the fold;
    - for a topology-publishing child, no jump source reads the dimension's
      node value, so the published jump preimages (and the rows' duplicated
      abscissae) are identical along it.

    The fold commutes with the read exactly wherever the value read's
    monotone slope limiter is inactive; where it binds (near jumps) the
    folded read is a different valid interpolant of the same data, deviating
    at interpolation-error order.
    """

    stochastic_node_values: tuple[FloatND | IntND, ...]
    """Per stochastic dimension: the node values fed into the resources query.

    Process dimensions carry the continuous AR(1) grid points (NaN when
    supplied at runtime — `process_grid_names` then names the runtime grid
    that overrides the placeholder); Markov-discrete dimensions carry the
    integer category codes (which equal the carry's leading-axis indices).
    """

    process_grid_names: tuple[StateName | None, ...]
    """Per stochastic dimension: the process state's name, or `None`.

    A continuous AR(1) process state with runtime-supplied distribution params
    has its grid points resolved only at solve time; this names the state whose
    resolved grid the kernel substitutes for the build-time placeholder in
    `stochastic_node_values`. `None` for Markov-discrete dimensions and for
    fully-specified processes (whose build-time grid is already final)."""

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

    n_outer_candidates: int
    """Candidate-axis length of a stacked NEGM child carry (`0` = none).

    A NEGM child publishes every outer durable candidate — the keeper plus one
    per outer-grid node — lifted into common cash-on-hand and stacked on an
    axis just before the grid axis. The read takes the exact hard max over
    that axis at the query, per durable node, *before* the passive blend, so
    the blended value interpolates the nodewise outer maximum. `0` for every
    other child (no candidate axis in its carry).
    """

    co_map_state_names: tuple[StateName, ...] = ()
    """Fixed, distributed child states co-mapped with the carry's leading axes.

    A co-mapped state's carry axis is sliced off by the caller's outer `vmap`
    before the read runs, so it names no discrete index here: its `next_<name>`
    coordinate is dropped from the carry indexing (the axis is gone), while the
    resources read still binds it from the combo pool. Empty in the replicated
    (non-co-mapped) read.
    """


@dataclass(frozen=True, kw_only=True)
class ContinuationPlan:
    """Build-time statics for the per-savings-node continuation aggregation.

    Binding a plan to one combo pool and the next period's carries (via
    `bind_continuation`) yields the regime's expected continuation value and
    marginal continuation as a function of end-of-period savings, aggregated
    over all reachable targets. The EGM step and the asset-row step read only
    that bound callable, never these statics directly.
    """

    carry_targets: tuple[RegimeName, ...]
    """Targets whose continuation is interpolated from their carry rows."""

    scalar_targets: tuple[RegimeName, ...]
    """Stateless targets contributing a constant continuation value."""

    child_reads: Mapping[RegimeName, _ChildRead]
    """Per-carry-target statics of the child carry read."""

    compute_regime_transition_probs: RegimeTransitionFunction
    """Regime transition probability function for solve."""

    post_decision_name: FunctionName
    """Name of the post-decision function (the savings node's input slot)."""

    stochastic_node_batch_size: int
    """Block size for splaying the child stochastic-node expectation (0 = fused)."""

    risk_aversion_param_name: str | None = None
    """Flat-param name of the certainty equivalent's risk-aversion coefficient.

    `None` for the linear (expected-utility) continuation. When set, the child
    stochastic-node expectation is the Epstein-Zin power mean of the next-period
    value and its risk-reweighted savings derivative rather than the plain
    weighted sum.
    """


def bind_continuation(
    *,
    plan: ContinuationPlan,
    combo_pool: dict[str, Any],
    next_regime_to_egm_carry: MappingProxyType[RegimeName, EGMCarry],
    dtype: Any,  # noqa: ANN401
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
    co_map_state_names: tuple[StateName, ...] = (),
) -> Callable[[ScalarFloat], tuple[ScalarFloat, ScalarFloat]]:
    """Bind a continuation plan to one combo pool and the next period's carries.

    Returns a map from an end-of-period savings node to the regime's expected
    continuation value and expected marginal continuation (both in savings
    space): per-carry-target smoothed carry reads weighted by the
    regime-transition probabilities, plus the scalar targets' constant values.
    The probabilities, transition weights, and child next-state reads are all
    evaluated from `combo_pool` *inside* this builder, so when the pool's Euler
    slot is a traced value (asset-row mode) the value's gradient carries their
    first-order terms (e.g. $\\sum \\partial P/\\partial a \\cdot EV$) —
    precomputing them outside the differentiated closure would silently drop
    those terms (Danskin does not cancel them: the probabilities are not the
    softmax of the values they weight).

    `resolved_process_grids` maps each runtime-resolved process state name to
    its solve-time grid (the node values its distribution params imply). A
    child read whose stochastic dimension is such a process substitutes that
    grid for the build-time NaN placeholder, so a resources function reading
    the process node integrates over the resolved nodes.

    `co_map_state_names` names the fixed distributed child states whose carry
    axes the caller has already sliced off each `next_regime_to_egm_carry` leaf
    with an outer `vmap` (one device-local slice per co-mapped value). The read
    drops those states from the carry indexing — the axis is gone — while still
    binding them for the child resources from `combo_pool`, so the continuation
    is read from the device-local slice and XLA inserts no all-gather.
    """
    regime_transition_probs = plan.compute_regime_transition_probs(**combo_pool)
    # Carry rows are indexed here along their whole leading discrete axes (the
    # carry-producing target's combo axes). A target state sharded across devices
    # (`distributed=True`) would split those axes per device, so this index could
    # not cross shards — and unlike the continuation V-array (co-mapped device-local
    # via `co_mapped_in_axes` in `max_Q_over_a`), the carry channel has no such
    # co-map, so a sharded carry into a DCEGM parent is unsupported. It is currently
    # unreachable, not merely unhandled: a distributed state cannot reach a DCEGM
    # regime or its carry-producing targets — regime-level `distributed` is rejected
    # (must be model-level), a model-level sharded state pruned from a non-terminal
    # regime is rejected, and a sharded discrete state surviving on a DCEGM regime is
    # rejected by grid hygiene. Lifting the restriction is not infeasible: extend the
    # continuation-V co-map to `next_regime_to_egm_carry` (map each carry's leading
    # discrete axis device-local, as for the V-array); or gather carry rows
    # shard-aware before indexing (an all-gather on just the carry's leading axis);
    # or carve out sharded axes that appear in no carry via a validation check.
    # The Epstein-Zin continuation reweights the child expectation by risk; its
    # coefficient is a flat param resolved from the pool here (runtime), keyed by
    # the plan's build-time name. `None` keeps the linear expected-utility read.
    risk_aversion = (
        combo_pool[plan.risk_aversion_param_name]
        if plan.risk_aversion_param_name is not None
        else None
    )
    child_readers = {
        target: _get_child_carry_reader(
            read=_with_co_map_states(plan.child_reads[target], co_map_state_names),
            carry=next_regime_to_egm_carry[target],
            combo_pool=combo_pool,
            post_decision_name=plan.post_decision_name,
            stochastic_node_batch_size=plan.stochastic_node_batch_size,
            resolved_process_grids=resolved_process_grids,
            risk_aversion=risk_aversion,
        )
        for target in plan.carry_targets
    }

    def continuation(
        savings_value: ScalarFloat,
    ) -> tuple[ScalarFloat, ScalarFloat]:
        """Blend the reachable targets' continuations at one savings node.

        Linear (expected-utility) mode returns the regime-probability-weighted
        expected value and marginal. Epstein-Zin mode blends each target's
        anchored transform-space partials `(a_r, S~_r, b_r, T~_r)` with the
        regime probabilities (`ez_blend_partials`) and inverts once, so the
        certainty equivalent spans the joint (regime x shock) lottery — the
        regime split (e.g. survival) is inside the CE, not a linear average of
        per-regime certainty equivalents.
        """
        if risk_aversion is not None:
            anchors: list[ScalarFloat] = []
            weight_sums: list[ScalarFloat] = []
            scaled_values: list[ScalarFloat] = []
            marginal_log_scales: list[ScalarFloat] = []
            marginal_mantissas: list[ScalarFloat] = []
            probs: list[ScalarFloat] = []
            for target in plan.carry_targets:
                # The reader returns the anchored quint whenever risk_aversion
                # is set (this branch); ty cannot correlate the union's arity
                # with the mode.
                (
                    anchor,
                    weight_sum,
                    scaled_value,
                    marginal_log_scale,
                    marginal_mantissa,
                ) = cast(
                    "_EZPartials",
                    child_readers[target](savings_value),
                )
                anchors.append(anchor)
                weight_sums.append(weight_sum)
                scaled_values.append(scaled_value)
                marginal_log_scales.append(marginal_log_scale)
                marginal_mantissas.append(marginal_mantissa)
                probs.append(regime_transition_probs[target])
            for target in plan.scalar_targets:
                # A stateless target (a constant bequest) contributes only to
                # the value channel; its constant enters transform space as its
                # own anchor, and its marginal channel is exactly zero.
                constant_value = next_regime_to_egm_carry[target].value[0]
                anchor, weight_sum, scaled_value = ez_transform_scalar(
                    value=constant_value, risk_aversion=risk_aversion
                )
                anchors.append(anchor)
                weight_sums.append(weight_sum)
                scaled_values.append(scaled_value)
                marginal_log_scales.append(jnp.asarray(0.0, dtype=dtype))
                marginal_mantissas.append(jnp.asarray(0.0, dtype=dtype))
                probs.append(regime_transition_probs[target])
            (
                joint_anchor,
                blended_weight,
                blended_value,
                joint_marginal_scale,
                blended_mantissa,
            ) = ez_blend_partials(
                log_anchors=jnp.stack(anchors),
                weight_sums=jnp.stack(weight_sums),
                scaled_values=jnp.stack(scaled_values),
                marginal_log_scales=jnp.stack(marginal_log_scales),
                marginal_mantissas=jnp.stack(marginal_mantissas),
                probs=jnp.stack(probs),
                risk_aversion=risk_aversion,
            )
            return ez_invert_partials(
                log_anchor=joint_anchor,
                weight_sum=blended_weight,
                scaled_value=blended_value,
                marginal_log_scale=joint_marginal_scale,
                marginal_mantissa=blended_mantissa,
                risk_aversion=risk_aversion,
            )
        blended_marginal = jnp.asarray(0.0, dtype=dtype)
        blended_value = jnp.asarray(0.0, dtype=dtype)
        for target in plan.carry_targets:
            # Linear mode: the reader returns the plain (value, marginal) pair.
            target_value, target_marginal = cast(
                "tuple[ScalarFloat, ScalarFloat]",
                child_readers[target](savings_value),
            )
            prob = regime_transition_probs[target]
            # Zero unreachable-target contributions on the results, never by
            # multiplying into a possibly non-finite value. The else branch is
            # `prob * 0.0` (not `0.0`) so a NaN probability poisons the sum
            # instead of vanishing.
            blended_marginal = blended_marginal + jnp.where(
                prob > 0.0, prob * target_marginal, prob * 0.0
            )
            blended_value = blended_value + jnp.where(
                prob > 0.0, prob * target_value, prob * 0.0
            )
        for target in plan.scalar_targets:
            prob = regime_transition_probs[target]
            constant_value = next_regime_to_egm_carry[target].value[0]
            blended_value = blended_value + jnp.where(
                prob > 0.0, prob * constant_value, prob * 0.0
            )
        return blended_value, blended_marginal

    return continuation


def build_continuation_plan(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    carry_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...],
    compute_regime_transition_probs: RegimeTransitionFunction,
    post_decision_name: FunctionName,
    stochastic_node_batch_size: int,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    risk_aversion_param_name: str | None = None,
) -> ContinuationPlan:
    """Assemble a `ContinuationPlan` from the regime's continuation statics.

    The plan binds (via `bind_continuation`) to one combo pool and the next
    period's carries to yield the expected continuation as a function of
    end-of-period savings. The post-decision function names the savings slot the
    child next-state reads consume; `stochastic_node_batch_size` splays the child
    stochastic-node expectation. Both the DC-EGM kernel and the NBEGM case-piece
    solver build their plan through this seam, so the continuation construction
    lives in one place.

    Returns:
        The assembled continuation plan.

    """
    child_reads = _build_child_reads(
        user_regimes=user_regimes,
        functions=functions,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        carry_targets=carry_targets,
        post_decision_name=post_decision_name,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
    )
    return ContinuationPlan(
        carry_targets=carry_targets,
        scalar_targets=scalar_targets,
        child_reads=child_reads,
        compute_regime_transition_probs=compute_regime_transition_probs,
        post_decision_name=post_decision_name,
        stochastic_node_batch_size=stochastic_node_batch_size,
        risk_aversion_param_name=risk_aversion_param_name,
    )


def _fold_stochastic_dims(
    *,
    read: _ChildRead,
    carry: EGMCarry,
    stochastic_node_values: tuple[FloatND | IntND, ...],
    weight_vecs: tuple[Float1D, ...],
) -> tuple[_ChildRead, EGMCarry, tuple[FloatND | IntND, ...], tuple[Float1D, ...]]:
    """Pre-apply the foldable stochastic dims' expectation to the carry rows.

    Each foldable dimension's intrinsic weights are savings-independent, its
    node values never reach the resources queries, and (under the fold gates)
    every node row shares the abscissae — so its expectation commutes with the
    per-row interpolation and folds into the carry once per cell. The folded
    carry's value and marginal rows are the guarded weighted sums over the
    dimension's node axis (`w * 0` on zero-weight nodes, so a `-inf`
    infeasible row on a zero-weight node contributes exactly zero and a NaN
    weight still poisons the sum); the abscissae are taken from the first
    node (identical across nodes under the gate). The read's per-dimension
    tuples shrink to the unfolded dims, so the remaining node loop — or, when
    everything folds, the loop-free branch — runs unchanged downstream.

    Returns:
        Tuple of the reduced read, the folded carry, and the unfolded dims'
        node values and weight vectors.

    """
    folded_names = frozenset(
        name
        for name, foldable in zip(
            read.stochastic_state_names, read.foldable_stochastic_flags, strict=True
        )
        if foldable
    )

    def fold_rows(rows: FloatND, axis: int, weights: Float1D) -> FloatND:
        moved = jnp.moveaxis(rows, axis, 0)
        broadcast_weights = weights.reshape(
            (weights.shape[0],) + (1,) * (moved.ndim - 1)
        )
        return jnp.sum(
            jnp.where(
                broadcast_weights > 0.0,
                broadcast_weights * moved,
                broadcast_weights * 0.0,
            ),
            axis=0,
        )

    weight_by_name = dict(zip(read.stochastic_state_names, weight_vecs, strict=True))
    endog_grid = carry.endog_grid
    value = carry.value
    marginal_utility = carry.marginal_utility
    breakpoints = carry.breakpoints

    def _carry_axis(name: StateName) -> int:
        """Axis of a discrete state in the carry rows.

        A co-mapped fixed distributed state is sliced off the carry's leading
        axis before the continuation reads it, so its axis is absent from the
        rows. The carry axis of any other state is its position in
        `discrete_state_names` less the co-mapped states that precede it.
        """
        position = read.discrete_state_names.index(name)
        co_mapped_before = sum(
            1
            for earlier in read.discrete_state_names[:position]
            if earlier in read.co_map_state_names
        )
        return position - co_mapped_before

    for name in sorted(folded_names, key=_carry_axis, reverse=True):
        axis = _carry_axis(name)
        node_weights = weight_by_name[name]
        endog_grid = jnp.take(endog_grid, 0, axis=axis)
        value = fold_rows(value, axis, node_weights)
        marginal_utility = fold_rows(marginal_utility, axis, node_weights)
        if breakpoints is not None:
            # Jump locations are identical along a foldable dim (its rows
            # share the duplicated abscissae), so the first node's row stands
            # for all of them.
            breakpoints = jnp.take(breakpoints, 0, axis=axis)
    folded_carry = replace(
        carry,
        endog_grid=endog_grid,
        value=value,
        marginal_utility=marginal_utility,
        breakpoints=breakpoints,
    )

    keep_stochastic = tuple(
        name not in folded_names for name in read.stochastic_state_names
    )

    def kept[T](entries: tuple[T, ...]) -> tuple[T, ...]:
        return tuple(
            entry for entry, keep in zip(entries, keep_stochastic, strict=True) if keep
        )

    reduced_read = replace(
        read,
        discrete_state_names=tuple(
            name for name in read.discrete_state_names if name not in folded_names
        ),
        stochastic_flags=tuple(
            flag
            for name, flag in zip(
                read.discrete_state_names, read.stochastic_flags, strict=True
            )
            if name not in folded_names
        ),
        stochastic_state_names=kept(read.stochastic_state_names),
        foldable_stochastic_flags=kept(read.foldable_stochastic_flags),
        stochastic_node_values=kept(read.stochastic_node_values),
        process_grid_names=kept(read.process_grid_names),
        weight_keys=kept(read.weight_keys),
    )
    return (
        reduced_read,
        folded_carry,
        kept(stochastic_node_values),
        kept(weight_vecs),
    )


def _with_co_map_states(
    read: _ChildRead, co_map_state_names: tuple[StateName, ...]
) -> _ChildRead:
    """Tag a child read with the co-mapped states among its discrete-state axes.

    Only the co-mapped states this read carries as discrete axes are tagged; the
    read then drops their (caller-sliced) carry axes from its deterministic index
    while still binding them for the resources read from the combo pool.
    """
    present = tuple(
        name for name in co_map_state_names if name in read.discrete_state_names
    )
    if not present:
        return read
    return replace(read, co_map_state_names=present)


def _get_child_carry_reader(
    *,
    read: _ChildRead,
    carry: EGMCarry,
    combo_pool: dict[str, Any],
    post_decision_name: FunctionName,
    stochastic_node_batch_size: int,
    resolved_process_grids: Mapping[StateName, FloatND] = MappingProxyType({}),
    risk_aversion: FloatND | None = None,
) -> Callable[
    [ScalarFloat],
    tuple[ScalarFloat, ScalarFloat]
    | tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat],
]:
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

    A runtime-resolved process dimension reads its node values from
    `resolved_process_grids` (keyed by the process state name) rather than the
    build-time NaN placeholder in `read.stochastic_node_values`, so a resources
    function reading the process node integrates over the resolved nodes.
    """
    stochastic_node_values = tuple(
        resolved_process_grids[name]
        if name is not None and name in resolved_process_grids
        else values
        for name, values in zip(
            read.process_grid_names, read.stochastic_node_values, strict=True
        )
    )
    weight_vecs: tuple[Float1D, ...] = ()
    if read.weights_func is not None:
        weights = read.weights_func(**combo_pool)
        weight_vecs = tuple(weights[key] for key in read.weight_keys)
    # A foldable dim of a topology-bearing carry shares the duplicated jump
    # abscissae across its rows (no jump source reads it — enforced by the
    # fold flags), so averaging the rows preserves both one-sided limits and
    # the fold applies exactly as for a smooth carry. Folding is a linear
    # expectation over the dim's rows, so it is valid only for the linear
    # expected-utility read: a nonlinear certainty equivalent must transform
    # every node's value before the lottery sum (`E[g(V)] != g(E[V])`), so
    # under Epstein-Zin the node axis stays and the per-node loop transforms
    # each row.
    if risk_aversion is None and any(read.foldable_stochastic_flags):
        read, carry, stochastic_node_values, weight_vecs = _fold_stochastic_dims(
            read=read,
            carry=carry,
            stochastic_node_values=stochastic_node_values,
            weight_vecs=weight_vecs,
        )
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

    def read_child(
        savings_value: ScalarFloat,
    ) -> (
        tuple[ScalarFloat, ScalarFloat]
        | tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]
    ):
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
            if not is_stochastic and name not in read.co_map_state_names
        )
        # A passive next-state is normally produced by `next_state_func`. Under
        # NEGM the outer post-decision's transition is stripped (it is bound per
        # outer-grid node, not recomputed), so that margin's next value is read
        # from the combo pool instead.
        child_passive_values = tuple(
            cast(
                "ScalarFloat",
                next_states[f"next_{name}"]
                if f"next_{name}" in next_states
                else combo_pool[f"next_{name}"],
            )
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
            value, marginal = _aggregate_child_choices(
                carry=carry,
                prepared_search_grid=prepared_search_grid,
                prepared_valid_length=prepared_valid_length,
                has_taste_shocks=read.has_taste_shocks,
                child_index=deterministic_index,
                child_passive_values=child_passive_values,
                child_passive_grids=read.passive_grids,
                row_queries=queries,
                row_gradients=gradients,
                paired_marginal_read=risk_aversion is not None,
                n_outer_candidates=read.n_outer_candidates,
            )
            if risk_aversion is not None:
                # A target whose continuation carries no shock lottery at this
                # seam — a stateless terminal value, or one whose stochastic
                # states are ride-along co-states integrated in the outer
                # envelope — contributes its anchored Epstein-Zin partials to
                # the joint regime blend: `ez_transform_partials` on a single
                # unit-weight node.
                return ez_transform_partials(
                    child_values=value[..., None],
                    child_marginals=marginal[..., None],
                    weights=jnp.ones(1, dtype=value.dtype),
                    risk_aversion=risk_aversion,
                )
            return value, marginal

        return _expect_over_stochastic_nodes(
            read=read,
            carry=carry,
            prepared_search_grid=prepared_search_grid,
            prepared_valid_length=prepared_valid_length,
            stochastic_node_values=stochastic_node_values,
            weight_vecs=weight_vecs,
            deterministic_index=deterministic_index,
            child_passive_values=child_passive_values,
            queries_and_gradients=queries_and_gradients,
            resources_reads_stochastic=resources_reads_stochastic,
            stochastic_node_batch_size=stochastic_node_batch_size,
            risk_aversion=risk_aversion,
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
    stochastic_node_values: tuple[FloatND | IntND, ...],
    weight_vecs: tuple[Float1D, ...],
    deterministic_index: tuple[ScalarInt, ...],
    child_passive_values: tuple[ScalarFloat, ...],
    queries_and_gradients: Callable[
        [tuple[ScalarFloat | ScalarInt, ...]], tuple[FloatND, FloatND]
    ],
    resources_reads_stochastic: bool,
    stochastic_node_batch_size: int,
    risk_aversion: FloatND | None = None,
) -> (
    tuple[ScalarFloat, ScalarFloat]
    | tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]
):
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
                    stochastic_node_values, node_indices, strict=True
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
            paired_marginal_read=risk_aversion is not None,
            n_outer_candidates=read.n_outer_candidates,
        )

    node_index_mesh = jnp.meshgrid(
        *(
            jnp.arange(values.shape[0], dtype=jnp.int32)
            for values in stochastic_node_values
        ),
        indexing="ij",
    )
    flat_node_indices = tuple(mesh.ravel() for mesh in node_index_mesh)
    joint_weights = weight_vecs[0][flat_node_indices[0]]
    for vec, indices in zip(weight_vecs[1:], flat_node_indices[1:], strict=True):
        joint_weights = joint_weights * vec[indices]

    def _weighted_node_sum(values: FloatND, weights: FloatND) -> ScalarFloat:
        # A zero-weight node contributes exactly 0.0 even when its smoothed
        # value is -inf (never 0 * inf = NaN). The else branch is `weights *
        # 0.0` (not a bare `0.0`) so a NaN weight poisons the sum instead of
        # vanishing.
        return jnp.sum(jnp.where(weights > 0.0, weights * values, weights * 0.0))

    # The expectation mesh (the product of the child's stochastic-node counts)
    # is the dominant `egm_step` working buffer's child-node axis. A positive
    # `stochastic_node_batch_size` below the mesh length accumulates the
    # weighted expectation in `lax.scan` blocks: each block reads only its
    # `batch_size` nodes (shedding the per-node gather working-set) AND folds
    # the weighted sum into the scan carry, so the full node-stacked
    # `(..., n_nodes)` result is never materialised — the savings the single
    # fused vmap below cannot reach, because there the reduction is downstream
    # of the materialised stack. `0` (or a size covering the whole mesh) keeps
    # that fused vmap + reduction. The weighted sum is associative, so the
    # value function matches the fused solve to numerical tolerance (the block
    # reduction reorders the floating-point adds).
    n_nodes = flat_node_indices[0].shape[0]
    if 0 < stochastic_node_batch_size < n_nodes:
        n_blocks = -(-n_nodes // stochastic_node_batch_size)
        pad = n_blocks * stochastic_node_batch_size - n_nodes
        blocked_indices = tuple(
            jnp.concatenate([indices, jnp.zeros(pad, dtype=indices.dtype)]).reshape(
                n_blocks, stochastic_node_batch_size
            )
            for indices in flat_node_indices
        )
        # Pad weights with 0.0, not the pad slots' real weights: the pad slots
        # reuse node index 0, so their values are read but zero-weighted, and
        # contribute exactly 0.0 to every block sum.
        blocked_weights = jnp.concatenate(
            [joint_weights, jnp.zeros(pad, dtype=joint_weights.dtype)]
        ).reshape(n_blocks, stochastic_node_batch_size)
        zero = jnp.zeros((), dtype=joint_weights.dtype)

        if risk_aversion is not None:
            return _accumulate_ez_partials_over_blocks(
                read_at_nodes=read_at_nodes,
                blocked_indices=blocked_indices,
                blocked_weights=blocked_weights,
                risk_aversion=risk_aversion,
            )

        def accumulate(
            carry: tuple[ScalarFloat, ScalarFloat],
            block: tuple[tuple[IntND, ...], FloatND],
        ) -> tuple[tuple[ScalarFloat, ScalarFloat], None]:
            block_indices, block_weights = block
            block_values, block_marginals = jax.vmap(read_at_nodes)(block_indices)
            acc_value, acc_marginal = carry
            return (
                acc_value + _weighted_node_sum(block_values, block_weights),
                acc_marginal + _weighted_node_sum(block_marginals, block_weights),
            ), None

        (smoothed_value, smoothed_marginal), _ = jax.lax.scan(
            accumulate, (zero, zero), (blocked_indices, blocked_weights)
        )
        return smoothed_value, smoothed_marginal

    node_values, node_marginals = jax.vmap(read_at_nodes)(flat_node_indices)
    if risk_aversion is not None:
        # Epstein-Zin: reduce this target's shock lottery into the certainty
        # equivalent's transform space — the transformed value and marginal
        # partials `(S, T)`, NOT the inverted `(nu, dnu/ds)`. The regime blend sums
        # these per-target partials with the regime probabilities and inverts once
        # (`ez_invert_partials`), so the joint certainty equivalent spans the full
        # (regime x shock) lottery. A single reachable target recovers M1's
        # per-regime power mean.
        return ez_transform_partials(
            child_values=node_values,
            child_marginals=node_marginals,
            weights=joint_weights,
            risk_aversion=risk_aversion,
        )
    smoothed_value = _weighted_node_sum(node_values, joint_weights)
    smoothed_marginal = _weighted_node_sum(node_marginals, joint_weights)
    return smoothed_value, smoothed_marginal


def _accumulate_ez_partials_over_blocks(
    *,
    read_at_nodes: Callable[
        [tuple[ScalarInt, ...] | tuple[IntND, ...]], tuple[FloatND, FloatND]
    ],
    blocked_indices: tuple[IntND, ...],
    blocked_weights: FloatND,
    risk_aversion: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]:
    """Accumulate one target's Epstein-Zin transform partials in node blocks.

    The anchored partials `(a, W, E, b, T~)` are additive across node blocks —
    each block is a partial sum of the same lottery — so each scan step
    transforms its block and folds it into the carry with a unit-probability
    blend. The single inversion happens downstream of the regime blend,
    exactly as in the fused path, so the block scan is a memory lever only.
    """
    dtype = blocked_weights.dtype
    exponent = 1.0 - risk_aversion
    neutral_anchor = jnp.where(
        exponent == 0.0,
        0.0,
        jnp.where(exponent >= 0.0, -jnp.inf, jnp.inf),
    ).astype(dtype)
    unit_probs = jnp.ones(2, dtype=dtype)
    zero = jnp.zeros((), dtype=dtype)

    def accumulate_partials(
        carry: tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat],
        block: tuple[tuple[IntND, ...], FloatND],
    ) -> tuple[
        tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat], None
    ]:
        block_indices, block_weights = block
        block_values, block_marginals = jax.vmap(read_at_nodes)(block_indices)
        block_quint = ez_transform_partials(
            child_values=block_values,
            child_marginals=block_marginals,
            weights=block_weights,
            risk_aversion=risk_aversion,
        )
        combined = ez_blend_partials(
            log_anchors=jnp.stack([carry[0], block_quint[0]]),
            weight_sums=jnp.stack([carry[1], block_quint[1]]),
            scaled_values=jnp.stack([carry[2], block_quint[2]]),
            marginal_log_scales=jnp.stack([carry[3], block_quint[3]]),
            marginal_mantissas=jnp.stack([carry[4], block_quint[4]]),
            probs=unit_probs,
            risk_aversion=risk_aversion,
        )
        return combined, None

    # The neutral carry: an empty partial sum whose anchor sits on the
    # non-dominating side, so the first real block's anchor wins the joint
    # extremum and the empty term contributes exactly zero.
    quint, _ = jax.lax.scan(
        accumulate_partials,
        (neutral_anchor, zero, zero, zero, zero),
        (blocked_indices, blocked_weights),
    )
    return quint


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
    paired_marginal_read: bool = False,
    n_outer_candidates: int = 0,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Read one child's carry with mixed interpolation and aggregate its choices.

    The carry rows matching the child's discrete-state values are selected
    by integer indexing on the leading state axes (discrete codes equal grid
    positions, stochastic dims indexed at one node); the remaining leading axes
    are the child's passive nodes, then its discrete-action combos. Every
    row is interpolated 1-D at its own resources query and its marginal is
    multiplied by its own composed gradient $(\\partial R'/\\partial A)$ —
    per row, because each row's envelope lives in its own resources space.
    The passive axes are then blended away (`_blend_passive_axes`) with
    edge-clamped linear weights on the two neighboring nodes of each passive
    grid — *before* the choice aggregation, so the logsum sees blended
    choice-specific values. Finally
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
        n_outer_candidates: Candidate-axis length of a stacked NEGM child
            carry, `0` for a child without one. When positive, the carry's row
            block carries a trailing candidate axis (the outer durable
            candidates lifted into common cash-on-hand); each candidate row is
            read at the *same* query (they share the coh axis), masked to
            `-inf` below its own first finite coh node (its
            borrowing-constrained support has not started — the lift moves
            adjuster supports up by the credited cost, so the edge clamp would
            otherwise let an infeasible candidate win on a boundary value),
            and the axis is collapsed by the exact hard max at the query with
            the winner's marginal (Danskin) — *before* the passive blend, so
            the blend interpolates the nodewise outer maximum
            `sum_k w_k max_j W_j(q; d_k)` rather than the lower bound
            `max_j sum_k w_k W_j(q; d_k)`.

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
    # discrete-action combos (then the candidate axis of a stacked NEGM child).
    block_shape = value_block.shape[:-1]
    _fail_if_carry_shape_mismatches_declaration(
        block_shape=block_shape,
        row_queries_shape=row_queries.shape,
        n_outer_candidates=n_outer_candidates,
    )
    if n_outer_candidates:
        # The stacked candidates share the lifted common-coh axis, so every
        # candidate row of a block cell is read at that cell's single query and
        # scaled by its single gradient.
        row_queries = jnp.broadcast_to(
            row_queries[..., None], (*row_queries.shape, n_outer_candidates)
        )
        row_gradients = jnp.broadcast_to(
            row_gradients[..., None], (*row_gradients.shape, n_outer_candidates)
        )
    grid_rows = grid_block.reshape(-1, n_pad)
    value_rows = value_block.reshape(-1, n_pad)
    marginal_rows = marginal_block.reshape(-1, n_pad)
    search_rows = search_block.reshape(-1, n_pad)
    valid_rows = valid_block.reshape(-1)
    queries_flat = row_queries.reshape(-1)
    gradients_flat = row_gradients.reshape(-1)

    # The marginal-utility row is the value row's exact slope (envelope
    # theorem), upgrading the value read to cubic Hermite. The marginal read
    # comes in two conventions:
    # - linear expected utility reads the marginal row itself linearly (a
    #   policy-grade quantity whose interpolation error enters the value only
    #   at second order through the Euler inversion);
    # - a paired read (`paired_marginal_read`, the Epstein-Zin transform
    #   marginal) differentiates the value interpolant itself, so the marginal
    #   is exactly the derivative of the value the certainty equivalent
    #   carries — where the slope limiter binds, the two conventions differ at
    #   leading order.
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

    if paired_marginal_read:

        def value_and_slope_row(
            search_grid: Float1D,
            valid_length: ScalarInt,
            xp: Float1D,
            fp: Float1D,
            fp_slopes: Float1D,
            x_query: ScalarFloat,
        ) -> tuple[ScalarFloat, ScalarFloat]:
            """Value read and its analytic derivative; positional per `jax.vmap`.

            The closed-form derivative of the selected piece — not autodiff
            through the bracket-selection program, whose `searchsorted`/`clip`
            representation returns arbitrary subgradients at exact grid nodes
            (a routine alignment: a zero-savings corner on a child grid that
            starts at zero).
            """
            return interp_and_derivative_on_prepared_grid(
                x_query=x_query,
                search_grid=search_grid,
                valid_length=valid_length,
                xp=xp,
                fp=fp,
                fp_slopes=fp_slopes,
            )

        value_at_child, marginal_at_child = jax.vmap(value_and_slope_row)(
            search_rows, valid_rows, grid_rows, value_rows, marginal_rows, queries_flat
        )
    else:
        value_at_child = jax.vmap(interp_value_row)(
            search_rows, valid_rows, grid_rows, value_rows, marginal_rows, queries_flat
        )
        marginal_at_child = jax.vmap(interp_row)(
            search_rows, valid_rows, grid_rows, marginal_rows, queries_flat
        )
    if n_outer_candidates:
        # Below a candidate's own first finite coh node its support has not
        # started: mask the read to `-inf` so the edge clamp cannot hand an
        # infeasible lifted candidate a boundary value that wins the max. The
        # `-inf` also pins the marginal to zero below. The right germ of each
        # candidate's value read feeds the tie rule at the candidate max; it
        # needs no support mask of its own — a below-support candidate enters
        # the tie set only when every candidate is below support, where all
        # published marginals are exactly zero regardless of the winner.
        def right_germ_row(
            search_grid: Float1D,
            valid_length: ScalarInt,
            xp: Float1D,
            fp: Float1D,
            fp_slopes: Float1D,
            x_query: ScalarFloat,
        ) -> tuple[ScalarBool, ScalarFloat, ScalarFloat, ScalarFloat]:
            """Right germ of one value row at its query; positional per `jax.vmap`."""
            return interp_right_germ_on_prepared_grid(
                x_query=x_query,
                search_grid=search_grid,
                valid_length=valid_length,
                xp=xp,
                fp=fp,
                fp_slopes=fp_slopes,
            )

        right_germ_at_child = jax.vmap(right_germ_row)(
            search_rows, valid_rows, grid_rows, value_rows, marginal_rows, queries_flat
        )
        row_lower = jnp.min(
            jnp.where(jnp.isfinite(grid_rows), grid_rows, jnp.inf), axis=1
        )
        # Only rows with a finite first node have a support to fall below;
        # `row_lower` is `+inf` on an all-NaN (poisoned) row, whose NaN read
        # must reach the candidate maximum fail-loud instead of becoming an
        # ordinary infeasible `(-inf, 0)` pair.
        below_row_support = (queries_flat < row_lower) & jnp.isfinite(row_lower)
        value_at_child = jnp.where(below_row_support, -jnp.inf, value_at_child)
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
    if n_outer_candidates:
        # Collapse the candidate axis *before* the passive blend, so the blend
        # interpolates the nodewise outer maximum
        # `sum_k w_k max_j W_j(q; d_k)` rather than the lower bound
        # `max_j sum_k w_k W_j(q; d_k)`.
        value_at_child, marginal_at_child = _collapse_stacked_candidates(
            value_at_child=value_at_child,
            marginal_at_child=marginal_at_child,
            right_germ_at_child=tuple(
                component.reshape(block_shape) for component in right_germ_at_child
            ),
        )

    value_at_child, marginal_at_child = _blend_passive_axes(
        value_at_child=value_at_child,
        marginal_at_child=marginal_at_child,
        child_passive_values=child_passive_values,
        child_passive_grids=child_passive_grids,
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


def _blend_passive_axes(
    *,
    value_at_child: FloatND,
    marginal_at_child: FloatND,
    child_passive_values: tuple[ScalarFloat, ...],
    child_passive_grids: tuple[Float1D, ...],
) -> tuple[FloatND, FloatND]:
    """Blend each passive axis away with edge-clamped linear node weights.

    Runs before the choice aggregation, so the logsum sees blended
    choice-specific values.

    Args:
        value_at_child: Read values with the row block's shape (passive dims,
            then action dims).
        marginal_at_child: Read marginals with the same shape.
        child_passive_values: The child's passive values at this savings node,
            aligned with `child_passive_grids`.
        child_passive_grids: The child's passive grids in carry-axis order.

    Returns:
        Tuple of the value and marginal with every passive axis blended away.

    """
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
        # rows), so a plain blend never produces NaN.
        marginal_at_child = (
            weight_lower * marginal_at_child[lower]
            + weight_upper * marginal_at_child[upper]
        )
    # A positive-weight blend of a dead node with a live one yields a `-inf`
    # value alongside a finite averaged marginal; re-pin the marginal to zero
    # so the `(-inf, 0)` infeasible pair survives the passive blend and no
    # finite marginal of an infeasible cell reaches the Euler inversion.
    marginal_at_child = jnp.where(jnp.isneginf(value_at_child), 0.0, marginal_at_child)
    return value_at_child, marginal_at_child


def _fail_if_carry_shape_mismatches_declaration(
    *,
    block_shape: tuple[int, ...],
    row_queries_shape: tuple[int, ...],
    n_outer_candidates: int,
) -> None:
    """Check the carry block's leading shape against the stacking declaration.

    The declaration and the published carry must agree before any
    broadcasting: a mismatch would otherwise surface as an opaque vmap
    axis-size error deep in the batched interpolation instead of naming the
    violated solver contract.
    """
    expected_block_shape = (
        (*row_queries_shape, n_outer_candidates)
        if n_outer_candidates
        else row_queries_shape
    )
    if block_shape != expected_block_shape:
        msg = (
            "The child's published carry block has leading shape "
            f"{block_shape}, but its solver declares "
            f"n_stacked_carry_candidates={n_outer_candidates}, which requires "
            f"{expected_block_shape}. The solver's declaration must match the "
            "candidate-axis structure of the carry it publishes."
        )
        raise ValueError(msg)


def _collapse_stacked_candidates(
    *,
    value_at_child: FloatND,
    marginal_at_child: FloatND,
    right_germ_at_child: tuple[BoolND, FloatND, FloatND, FloatND],
) -> tuple[FloatND, FloatND]:
    """Collapse the candidate axis by the exact hard max at the query.

    Publishes the winner's marginal (Danskin). This runs *before* the passive
    blend so the blend interpolates the nodewise outer maximum. An exact value
    tie resolves right-continuously in the value read itself: the tied
    candidates are compared by the complete right germ of their own value
    interpolants (`right_germ_winner` — right-finiteness, then the first three
    one-sided derivatives, then the lowest index among locally identical
    pieces), so the branch whose read actually wins immediately to the right
    owns the (economic) marginal the parent's Euler inversion consumes. (The
    germ uses the unscaled interpolant derivatives: the composed gradient is
    shared by all candidates of a cell and positive, so scaling could never
    reorder them.) A cell whose candidates are all `-inf` (no live support)
    keeps the `(-inf, 0)` infeasible contract: every masked marginal is
    exactly zero.

    Args:
        value_at_child: Candidate value reads; the candidate axis is last.
        marginal_at_child: Gradient-scaled candidate marginal reads, same shape.
        right_germ_at_child: Tuple of the right-finiteness flag and the first
            three right derivatives of the candidate value reads, same shape.

    Returns:
        Tuple of the winner's value and marginal, with the candidate axis
        collapsed.

    """
    winner = right_germ_winner(value=value_at_child, right_germ=right_germ_at_child)
    winner_marginal = jnp.take_along_axis(marginal_at_child, winner, axis=-1)[..., 0]
    # The published value is the maximum itself: identical to the winner's read
    # at any tie, and NaN-propagating when a poisoned candidate row (whose NaN
    # empties the tie set) must surface fail-loud. The marginal is pinned to
    # NaN alongside so the published pair never looks healthy over a poisoned
    # cell — the germ selector falls back to a finite competitor's index when
    # the NaN empties the tie set, and that competitor's marginal must not
    # ship beside a NaN value.
    collapsed_value = jnp.max(value_at_child, axis=-1)
    return (
        collapsed_value,
        jnp.where(jnp.isnan(collapsed_value), jnp.nan, winner_marginal),
    )


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
        # A process state whose distribution params arrive at runtime has its
        # `to_jax()` grid as a NaN placeholder above; name it so the kernel
        # substitutes the resolved grid (the same node values the regime's own
        # combo axes iterate, shared with the source regime) at solve time. A
        # fully-specified process keeps its final build-time grid (`None`).
        process_grid_names = tuple(
            name if _is_runtime_process(target_info.discrete_states[name]) else None
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
        # A value-only child (brute `GridSearch`, the case-piece `NBEGM`)
        # publishes a value array already maxed over its discrete actions, so the
        # read carries no per-action rows; only a choice-retaining child (DC-EGM)
        # leaves them for the parent to aggregate.
        if target_regime.solver.carry_retains_discrete_action_rows:
            action_names, action_values = _get_child_discrete_actions(
                user_regime=target_regime
            )
        else:
            action_names, action_values = (), ()
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
        child_carry_rows_uniform = (
            target_regime.solver.carry_rows_share_state_grid
            and not target_regime.solver.carry_retains_discrete_action_rows
        )
        # Under a topology-publishing read, a dim is only foldable when no
        # jump source reads its node value — otherwise the published jump
        # preimages (and so the rows' duplicated abscissae) vary along it.
        if getattr(target_regime.solver, "jump_read", None) == "one_sided":
            jump_moving = jump_moving_state_names(
                functions=target_regime.functions,
                state_names=frozenset(target_regime.states),
                euler_state_name=euler_state_name,
            )
        else:
            jump_moving = frozenset()
        foldable_stochastic_flags = tuple(
            child_carry_rows_uniform
            and name not in resources_arg_names
            and name not in jump_moving
            for name in stochastic_state_names
        )
        row_grids = passive_grids + action_values
        if row_grids:
            row_mesh = jnp.meshgrid(*row_grids, indexing="ij")
            row_values = tuple(mesh.ravel() for mesh in row_mesh)
            row_block_shape = tuple(int(grid.shape[0]) for grid in row_grids)
        else:
            row_values = ()
            row_block_shape = ()
        # The candidate-axis length comes from the child solver's own
        # declaration: only a solver whose published carry actually stacks
        # outer candidates (NEGM's keeper + per-node rows) reports a nonzero
        # count. Merely having an outer margin is not enough — a solver that
        # folds its outer axis before publishing (the nested N-NB-EGM upper
        # envelope) carries no candidate axis, and broadcasting queries over a
        # phantom axis would desync rows and queries in the read.
        n_outer_candidates = target_regime.solver.n_stacked_carry_candidates
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
            foldable_stochastic_flags=foldable_stochastic_flags,
            stochastic_node_values=stochastic_node_values,
            process_grid_names=process_grid_names,
            weight_keys=weight_keys,
            weights_func=weights_func,
            passive_state_names=passive_state_names,
            passive_grids=passive_grids,
            row_arg_names=passive_state_names + action_names,
            row_values=row_values,
            row_block_shape=row_block_shape,
            n_outer_candidates=n_outer_candidates,
        )
    return MappingProxyType(reads)
