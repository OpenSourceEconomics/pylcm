"""Reduce a period's Q over its action axes into that period's value.

`get_max_Q_over_a` builds the dense reducer — a hard max, or the EV1 smoothed
maximum when the regime declares taste shocks. `get_streaming_max_Q_over_a` is
the blockwise route that never materializes the whole action product,
`get_action_partitioned_max_Q_over_a` the blockwise route whose action product
several devices share, and `get_argmax_and_max_Q_over_a` the variant simulation
needs, which returns the winning action alongside the value.
"""

import functools
import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, cast

import jax
import jax.numpy as jnp
from dags import with_signature
from jax import Array

from _lcm.current_iid import CurrentIIDDrawOutput, CurrentIIDMaxOutput, CurrentIIDPlan
from _lcm.execution.placement import ACTION_PARTITION_AXIS
from _lcm.logsum import EULER_GAMMA
from _lcm.regime_building.collective import (
    ParetoWeights,
    collective_argmax_and_readout,
    collective_readout,
    decider_choice,
    decider_smoothed_readout,
)
from _lcm.regime_building.Q_and_F import (
    BoundedActionOutput,
    BoundedQAndF,
    has_action_integrity,
)
from _lcm.solution.action_reduction import CollectiveHardMaxResult
from _lcm.solution.action_streaming import (
    build_partitioned_streaming_max_Q_over_a,
    build_streaming_collective_max_Q_over_a,
)
from _lcm.solution.feasibility import BoundedResult
from _lcm.typing import (
    ActionName,
    ArgmaxQOverAFunction,
    MaxQOverAFunction,
    PRNGKeyND,
    PytreeValue,
    QAndFArg,
    QAndFKwargs,
    QualifiedName,
    RegimeName,
    StateName,
)
from _lcm.utils.dispatchers import productmap, tiled_productmap, vmap_1d
from _lcm.utils.functools import allow_args, allow_only_kwargs
from lcm.typing import BoolND, FloatND, IntND, ReferenceName, ScalarFloat

# Flat param name of the EV1 taste-shock scale (template pseudo-function entry).
TASTE_SHOCK_SCALE_PARAM = "taste_shocks__scale"

type MaxOutput = FloatND | tuple[FloatND, BoolND] | CurrentIIDMaxOutput
type BoundedMaxOutput = (
    MaxOutput | BoundedResult[MaxOutput] | BoundedResult[CurrentIIDMaxOutput]
)
type ArgmaxOutput = tuple[IntND, FloatND]
type BoundedArgmaxOutput = (
    ArgmaxOutput | BoundedResult[ArgmaxOutput] | CurrentIIDDrawOutput
)
type ActionFunction = Callable[
    ..., tuple[FloatND, BoolND] | BoundedResult[BoundedActionOutput]
]
type RawActionFunction = Callable[..., tuple[FloatND, BoolND]]


def _with_bound_summary[Output: PytreeValue](
    *,
    evaluated: PytreeValue,
    output: Output,
) -> Output | BoundedResult[Output]:
    """Preserve scalar integrity while the action owner reduces ordinary channels."""
    return (
        BoundedResult(
            output=output,
            bound_status=evaluated.bound_status,
            query_status=evaluated.query_status,
        )
        if type(evaluated) is BoundedResult
        else output
    )


def _evaluate_pareto_weights(
    *,
    pareto_weights: ParetoWeights,
    states_actions_params: QAndFKwargs,
) -> dict[str, FloatND]:
    """Evaluate the household's Pareto weights at one cell.

    The kernel's own signature carries every argument the declaration reads, so
    the weights are read out of the cell rather than closed over — which is
    what makes a state-dependent or estimated weight ordinary.
    """
    return pareto_weights.compute(
        **{name: states_actions_params[name] for name in pareto_weights.arg_names}
    )


def get_max_Q_over_a(
    *,
    current_iid_plan: CurrentIIDPlan | None = None,
    Q_and_F: RawActionFunction,
    batch_sizes: dict[StateName, int],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
    co_map_state_names: tuple[StateName, ...] = (),
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = (),
    members: tuple[str, ...],
    pareto_weights: ParetoWeights,
    cell_width_keyword: str | None = None,
    untiled_state_names: tuple[StateName, ...] = (),
    broadcast_state_names: tuple[StateName, ...] = (),
) -> MaxQOverAFunction[BoundedMaxOutput]:
    r"""Get the function returning the maximum of Q over all actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  W(U(x, a), \mathbb{E}[V(x', a') | x, a]),
    ```
    with $W(U, v) = u + \beta \cdot v$ as the leading case (which is the only one that
    is pre-implemented in LCM).

    Fixing a state, maximizing over all feasible actions,
    we get the $V$ function:

    ```{math}
    V(x) = \max_{a} Q(x, a).
    ```

    This last step is handled by the function returned here.

    Args:
        Q_and_F: A function that takes a state-action combination and returns the action
            value of that combination and whether the state-action combination is
            feasible.
        batch_sizes: Mapping of state variable names to batch sizes for the outer
            productmap over states when no cell-width keyword is supplied. A batch
            size of 0 means no batching.
        cell_width_keyword: Optional static keyword sizing the flattened inner
            state product. The execution planner binds it before lowering.
        untiled_state_names: Non-co-mapped states evaluated by ordinary outer
            vmaps, keeping sharded coordinate axes outside the cell loop.
        broadcast_state_names: Non-co-mapped states the continuation never
            reads, evaluated by ordinary outer vmaps whose extent counts toward
            the cell width. The continuation is then computed once per remaining
            cell and broadcast along them.
        action_names: Tuple of action variable names (discrete first, continuous
            last — the `StateActionSpace.action_names` order).
        state_names: Tuple of state names.
        n_discrete_action_axes: Number of leading discrete-action axes in the
            Q array. Only used when `has_taste_shocks` is set.
        has_taste_shocks: Whether the regime declares EV1 taste shocks. When
            set, the hard maximum over the discrete-action axes is replaced by
            the smoothed expected maximum with the runtime scale param
            `taste_shocks__scale`; in a household, the expected maximum of the
            member with the positive Pareto weight, every other member's value
            read off its choice (`decider_smoothed_readout`).
        co_map_state_names: Tuple of fixed (never-transitioning) distributed state
            names, the leading axes of the value-function array. Each is mapped by an
            outer `vmap` that co-maps the matching axis of every `next_regime_to_V_arr`
            leaf carrying it, so the continuation-V interpolation reads only the
            device-local slice and XLA inserts no all-gather. Must be a leading prefix
            of `state_names`.
        co_map_v_arr_in_axes: Per-co-map-state `in_axes` for `next_regime_to_V_arr`,
            aligned with `co_map_state_names`. Each entry is an immutable mapping of
            regime name to `0` (the leaf carries that state as its current leading
            axis — slice it) or `None` (the leaf does not carry it — pass it through,
            e.g. a target regime where the state is pruned).
        members: Ordered member names. `Q_and_F` returns a stacked per-member
            `Q` (trailing member axis) and the inner reduction reads off each
            member's own value at the shared household argmax
            (`collective_readout`).
        pareto_weights: The household's Pareto weight evaluator, called at each
            cell with the states and parameters its declaration reads.

    Returns:
        The function returning the pair `(V, D)`: the member-axis value array of
        the maximum of Q over the feasible actions, and the boolean dissolution
        flag `D = 1[mask empty]` on the state axes — distinct from a numeric
        `-inf`, which occurs on-path.

    """
    _fail_if_co_map_states_not_leading(
        state_names=state_names, co_map_state_names=co_map_state_names
    )
    # Extract extra param names from Q_and_F's signature (flat regime params)
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
    )
    # A Pareto weight is evaluated at the cell, so its free parameters ride in
    # this kernel's own signature alongside `Q_and_F`'s.
    extra_param_names = list(
        dict.fromkeys((*extra_param_names, *pareto_weights.param_names))
    )

    bound_summary = bool(has_action_integrity(function=Q_and_F))
    action_kernel: ActionFunction = (
        BoundedQAndF(function=Q_and_F) if bound_summary else Q_and_F
    )

    # Actions are the inner optimization axis — batching applies only to the
    # outer state loop.
    mapped_action_kernel = productmap(
        func=action_kernel,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
        bound_summary=bound_summary,
    )

    max_Q_over_a: Callable[..., BoundedMaxOutput]
    if has_taste_shocks:
        max_Q_over_a = with_signature(
            _SmoothedMaxQOverA(
                Q_and_F=mapped_action_kernel,
                n_discrete_action_axes=n_discrete_action_axes,
                members=members,
                pareto_weights=pareto_weights,
            ),
            args=[
                "next_regime_to_V_arr",
                *action_names,
                *state_names,
                *extra_param_names,
            ],
            return_annotation="tuple[FloatND, BoolND]",
            enforce=False,
        )
    else:
        max_Q_over_a = with_signature(
            _HardMaxQOverA(
                Q_and_F=mapped_action_kernel,
                members=members,
                pareto_weights=pareto_weights,
            ),
            args=[
                "next_regime_to_V_arr",
                *action_names,
                *state_names,
                *extra_param_names,
            ],
            return_annotation="tuple[FloatND, BoolND]",
            enforce=False,
        )

    if current_iid_plan is not None:
        max_Q_over_a = current_iid_plan.wrap(max_Q_over_a)

    inner_state_names = tuple(
        name for name in state_names if name not in co_map_state_names
    )
    mapped = (
        tiled_productmap(
            func=max_Q_over_a,
            variables=inner_state_names,
            width_keyword=cell_width_keyword,
            untiled_variables=untiled_state_names,
            broadcast_variables=broadcast_state_names,
            bound_summary=bound_summary,
        )
        if cell_width_keyword is not None
        else productmap(
            func=max_Q_over_a,
            variables=inner_state_names,
            batch_sizes={name: batch_sizes[name] for name in inner_state_names},
            bound_summary=bound_summary,
        )
    )

    if not co_map_state_names:
        return mapped

    # Co-map each fixed distributed state — the leading V-array axes — with the
    # matching axis of every `next_regime_to_V_arr` leaf that carries it, outermost
    # state first. Each map peels the state's leading axis off both the state grid and
    # the continuation V, so the interpolation reads the device-local slice and
    # produces axes in `state_names` order. A leaf that does not carry the state (e.g.
    # a target regime where it is pruned) maps with `None` and passes through. The
    # vmaps need positional dispatch, so `allow_args` first and restore the kwargs
    # interface afterwards.
    mapped = allow_args(mapped)
    for state_name, v_arr_in_axes in zip(
        reversed(co_map_state_names), reversed(co_map_v_arr_in_axes), strict=True
    ):
        mapped = vmap_1d(
            func=mapped,
            variables=(state_name,),
            co_mapped_in_axes=MappingProxyType({"next_regime_to_V_arr": v_arr_in_axes}),
            callable_with="only_args",
            bound_summary=bound_summary,
        )
    return cast(
        "MaxQOverAFunction[BoundedMaxOutput]",
        allow_only_kwargs(func=mapped, enforce=False),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _SmoothedMaxQOverA:
    """The EV1-smoothed maximum of Q over the actions, at one state cell.

    Every value the reduction reads is an explicit field, so the regime's
    action kernel stays reachable only through this instance, which the model
    build drops with the value function it was built for. The caller wraps it
    in `dags.with_signature` to advertise the arguments the state productmap
    binds by name.
    """

    __name__: ClassVar[str] = "max_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    Q_and_F: ActionFunction
    """The regime's action value and feasibility, mapped over the action product."""

    n_discrete_action_axes: int
    """Number of leading discrete-action axes the smoothed maximum reduces."""

    members: tuple[str, ...]
    """Ordered member names of the household."""

    pareto_weights: ParetoWeights
    """The household's Pareto weight evaluator."""

    def __call__(
        self,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> BoundedMaxOutput:
        """Return each member's expected maximum and the dissolution flag."""
        evaluated = self.Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        Q_arr, F_arr = cast(
            "tuple[FloatND, BoolND]",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        # Q_arr carries a trailing member axis. The member with the positive
        # Pareto weight decides; every member's value is read off its smoothed
        # choice, beside the dissolution flag D.
        values, dissolution = decider_smoothed_readout(
            Q=Q_arr,
            feasibility=F_arr,
            weights=_stacked_pareto_weights(
                pareto_weights=self.pareto_weights,
                members=self.members,
                states_actions_params=states_actions_params,
            ),
            n_discrete_action_axes=self.n_discrete_action_axes,
            scale=cast("ScalarFloat", states_actions_params[TASTE_SHOCK_SCALE_PARAM]),
        )
        return _with_bound_summary(
            evaluated=evaluated, output=cast("MaxOutput", (values, dissolution))
        )


def _stacked_pareto_weights(
    *,
    pareto_weights: ParetoWeights,
    members: tuple[str, ...],
    states_actions_params: QAndFKwargs,
) -> FloatND:
    """Evaluate the household's Pareto weights at one cell, in member order."""
    weights = _evaluate_pareto_weights(
        pareto_weights=pareto_weights, states_actions_params=states_actions_params
    )
    return jnp.stack([jnp.asarray(weights[name]) for name in members])


@dataclass(frozen=True, kw_only=True, eq=False)
class _HardMaxQOverA:
    """The hard maximum of Q over the feasible actions, at one state cell.

    Every value the reduction reads is an explicit field, so the regime's
    action kernel stays reachable only through this instance, which the model
    build drops with the value function it was built for. The caller wraps it
    in `dags.with_signature` to advertise the arguments the state productmap
    binds by name.
    """

    __name__: ClassVar[str] = "max_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    Q_and_F: ActionFunction
    """The regime's action value and feasibility, mapped over the action product."""

    members: tuple[str, ...]
    """Ordered member names of the household."""

    pareto_weights: ParetoWeights
    """The household's Pareto weight evaluator."""

    def __call__(
        self,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> BoundedMaxOutput:
        """Return each member's value at the cell and the dissolution flag."""
        evaluated = self.Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        Q_arr, F_arr = cast(
            "tuple[FloatND, BoolND]",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        # Q_arr carries a trailing member axis (the action product-map keeps it
        # last); F_arr does not. Split Q_arr per member, take the household
        # argmax of the scalarization over the masked action axes, and read off
        # each member's OWN value at that shared argmax. The returned pair is
        # the member value vector (re-stacked on a trailing axis, which the
        # outer state product-map turns into `(*states, n_members)`) plus the
        # dissolution flag D — `True` where NO action is feasible (empty mask),
        # published alongside V and never conflated with a numeric -inf value
        # (which occurs on-path).
        action_axes = tuple(range(F_arr.ndim))
        member_Q = {name: Q_arr[..., index] for index, name in enumerate(self.members)}
        values, dissolution = collective_readout(
            member_Q=member_Q,
            feasibility=F_arr,
            weights=_evaluate_pareto_weights(
                pareto_weights=self.pareto_weights,
                states_actions_params=states_actions_params,
            ),
            action_axes=action_axes,
        )
        output: MaxOutput = (
            jnp.stack([values[name] for name in self.members], axis=-1),
            dissolution,
        )
        return _with_bound_summary(
            evaluated=evaluated, output=cast("MaxOutput", output)
        )


def get_streaming_max_Q_over_a(
    *,
    current_iid_plan: CurrentIIDPlan | None = None,
    Q_and_F: RawActionFunction,
    batch_sizes: dict[StateName, int],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    co_map_state_names: tuple[StateName, ...] = (),
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = (),
    members: tuple[str, ...],
    pareto_weights: ParetoWeights,
    action_width_keyword: str = "_lcm_action_block_width",
    cell_width_keyword: str | None = None,
    untiled_state_names: tuple[StateName, ...] = (),
    broadcast_state_names: tuple[StateName, ...] = (),
) -> MaxQOverAFunction[BoundedMaxOutput]:
    """Build a hard-max V kernel that streams the action product.

    The returned raw callable has the same dynamic argument layout as
    `get_max_Q_over_a` plus one required static keyword named by
    `action_width_keyword`. The execution planner selects a collision-free name and
    binds that width before tracing. For each state cell, the fixed-cell reducer
    evaluates actions in canonical C order and publishes each member's value at
    one shared household winner plus the empty-feasible-set flag. Ordinary
    states use the existing state productmap. Fixed distributed states retain the
    dense route co-map access law: each state axis is mapped in lockstep with axis
    zero of every continuation leaf that carries it.
    """
    _fail_if_streaming_co_map_layout_is_invalid(
        state_names=state_names,
        co_map_state_names=co_map_state_names,
        co_map_v_arr_in_axes=co_map_v_arr_in_axes,
    )
    extra_param_names = list(
        dict.fromkeys(
            (
                *_get_extra_param_names(
                    Q_and_F=Q_and_F,
                    action_names=action_names,
                    state_names=state_names,
                ),
                *pareto_weights.param_names,
            )
        )
    )
    _fail_if_action_width_keyword_collides(
        action_width_keyword=action_width_keyword,
        action_names=action_names,
        state_names=state_names,
        extra_param_names=extra_param_names,
    )

    bound_summary = bool(has_action_integrity(function=Q_and_F))
    streamed_max_Q_over_a = with_signature(
        _StreamedMaxQOverA(
            Q_and_F=Q_and_F,
            action_names=action_names,
            members=members,
            pareto_weights=pareto_weights,
            q_and_f_arg_names=frozenset(inspect.signature(Q_and_F).parameters),
            action_width_keyword=action_width_keyword,
        ),
        args=[
            "next_regime_to_V_arr",
            *action_names,
            *state_names,
            *extra_param_names,
            action_width_keyword,
        ],
        return_annotation="tuple[FloatND, BoolND]",
        enforce=False,
    )

    if current_iid_plan is not None:
        streamed_max_Q_over_a = current_iid_plan.wrap(streamed_max_Q_over_a)

    inner_state_names = tuple(
        name for name in state_names if name not in co_map_state_names
    )
    mapped = (
        tiled_productmap(
            func=streamed_max_Q_over_a,
            variables=inner_state_names,
            width_keyword=cell_width_keyword,
            untiled_variables=untiled_state_names,
            broadcast_variables=broadcast_state_names,
            bound_summary=bound_summary,
        )
        if cell_width_keyword is not None
        else productmap(
            func=streamed_max_Q_over_a,
            variables=inner_state_names,
            batch_sizes={name: batch_sizes[name] for name in inner_state_names},
            bound_summary=bound_summary,
        )
    )
    if not co_map_state_names:
        return cast("MaxQOverAFunction[BoundedMaxOutput]", mapped)

    # Preserve the dense route's device-local continuation access exactly. Build
    # the maps from the innermost co-map axis outward: at runtime the outer map
    # removes the original leading V axis first, leaving the next co-map axis at
    # position zero for the nested map. Leaves that do not carry a state use
    # ``None`` and pass through unchanged.
    mapped = allow_args(mapped)
    for state_name, v_arr_in_axes in zip(
        reversed(co_map_state_names), reversed(co_map_v_arr_in_axes), strict=True
    ):
        mapped = vmap_1d(
            func=mapped,
            variables=(state_name,),
            co_mapped_in_axes=MappingProxyType({"next_regime_to_V_arr": v_arr_in_axes}),
            callable_with="only_args",
            bound_summary=bound_summary,
        )
    return cast(
        "MaxQOverAFunction[BoundedMaxOutput]",
        allow_only_kwargs(func=mapped, enforce=False),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _StreamedMaxQOverA:
    """The maximum of Q over a streamed action product, at one state cell.

    Every value the reduction reads is an explicit field, so the regime's
    action kernel stays reachable only through this instance, which the model
    build drops with the value function it was built for. The caller wraps it
    in `dags.with_signature` to advertise the arguments the state productmap
    binds by name, the planner-owned action width included.
    """

    __name__: ClassVar[str] = "streamed_max_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    Q_and_F: RawActionFunction
    """The regime's action value and feasibility, evaluated per action block."""

    action_names: tuple[ActionName, ...]
    """Action variable names, discrete first, spanning the streamed product."""

    members: tuple[str, ...]
    """Ordered member names of the household."""

    pareto_weights: ParetoWeights
    """The household's Pareto weight evaluator."""

    q_and_f_arg_names: frozenset[ReferenceName]
    """The argument names `Q_and_F` declares, which select what it is handed."""

    action_width_keyword: str
    """Name of the planner-bound static action-block width in the call."""

    def __call__(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> BoundedMaxOutput:
        """Return each member's value at the cell and the dissolution flag."""
        collective_cell = build_streaming_collective_max_Q_over_a(
            Q_and_F=self.Q_and_F,
            action_names=self.action_names,
            block_width=cast("int", states_actions_params[self.action_width_keyword]),
            members=self.members,
            publishes_identity=False,
            weights=_evaluate_pareto_weights(
                pareto_weights=self.pareto_weights,
                states_actions_params=states_actions_params,
            ),
        )
        collective_result = collective_cell(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **_arguments_named(
                arguments=states_actions_params, names=self.q_and_f_arg_names
            ),
        )
        if isinstance(collective_result, BoundedResult):
            result = collective_result.output
            return BoundedResult(
                output=(result.best_member_values, ~result.any_feasible),
                bound_status=collective_result.bound_status,
                query_status=collective_result.query_status,
            )
        return (
            collective_result.best_member_values,
            ~collective_result.any_feasible,
        )


def get_action_partitioned_max_Q_over_a(
    *,
    Q_and_F: RawActionFunction,
    batch_sizes: dict[StateName, int],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    n_partitions: int,
    mesh: jax.sharding.Mesh,
    action_width_keyword: str = "_lcm_action_block_width",
    cell_width_keyword: str | None = None,
    untiled_state_names: tuple[StateName, ...] = (),
    broadcast_state_names: tuple[StateName, ...] = (),
    members: tuple[str, ...],
    pareto_weights: ParetoWeights,
) -> MaxQOverAFunction[BoundedMaxOutput]:
    """Build a hard-max V kernel whose action product devices share.

    The returned callable has the argument layout of
    `get_streaming_max_Q_over_a`. It runs its body once per device of `mesh`'s
    `ACTION_PARTITION_AXIS`, with every input replicated along that axis and
    the regime's state axes left to the compiler. Each device streams the
    contiguous run of whole action blocks `ActionPartitionLayout` assigns it at
    the planner-bound width, gathers one hard-max accumulator per state cell
    from every device of the axis and merges them in partition order, so all
    devices publish the same value: every member's value at the household's
    one shared winner and the dissolution flag.

    Taste shocks and co-mapped states are not served; model construction
    refuses them before this is called.
    """
    if n_partitions < 2:  # noqa: PLR2004
        raise ValueError("An action-partitioned kernel needs at least two partitions.")
    if ACTION_PARTITION_AXIS not in mesh.axis_names:
        raise ValueError(
            f"The mesh {mesh.axis_names!r} has no {ACTION_PARTITION_AXIS!r} axis."
        )
    if mesh.shape[ACTION_PARTITION_AXIS] != n_partitions:
        raise ValueError(
            f"The mesh action axis has {mesh.shape[ACTION_PARTITION_AXIS]} devices, "
            f"not the {n_partitions} partitions requested."
        )
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F,
        action_names=action_names,
        state_names=state_names,
    )
    extra_param_names = list(
        dict.fromkeys((*extra_param_names, *pareto_weights.param_names))
    )
    # The keyword order differs from the streamed builder's call on purpose:
    # that call is a certified mutation anchor and must stay unique.
    _fail_if_action_width_keyword_collides(
        extra_param_names=extra_param_names,
        action_names=action_names,
        state_names=state_names,
        action_width_keyword=action_width_keyword,
    )
    partitioned_max_Q_over_a = with_signature(
        _ActionPartitionedMaxQOverA(
            Q_and_F=Q_and_F,
            action_names=action_names,
            q_and_f_arg_names=frozenset(inspect.signature(Q_and_F).parameters),
            action_width_keyword=action_width_keyword,
            n_partitions=n_partitions,
            members=members,
            pareto_weights=pareto_weights,
        ),
        args=[
            "next_regime_to_V_arr",
            *action_names,
            *state_names,
            *extra_param_names,
            action_width_keyword,
        ],
        return_annotation="tuple[FloatND, BoolND]",
        enforce=False,
    )
    mapped = (
        tiled_productmap(
            func=partitioned_max_Q_over_a,
            variables=state_names,
            width_keyword=cell_width_keyword,
            untiled_variables=untiled_state_names,
            broadcast_variables=broadcast_state_names,
            bound_summary=bool(has_action_integrity(function=Q_and_F)),
        )
        if cell_width_keyword is not None
        else productmap(
            func=partitioned_max_Q_over_a,
            variables=state_names,
            batch_sizes={name: batch_sizes[name] for name in state_names},
            bound_summary=bool(has_action_integrity(function=Q_and_F)),
        )
    )
    static_names = (
        action_width_keyword,
        *((cell_width_keyword,) if cell_width_keyword is not None else ()),
    )
    return cast(
        "MaxQOverAFunction[BoundedMaxOutput]",
        with_signature(
            _OnActionPartitionAxis(
                function=mapped, mesh=mesh, static_names=static_names
            ),
            args=list(inspect.signature(mapped).parameters),
            return_annotation="tuple[FloatND, BoolND]",
            enforce=False,
        ),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _ActionPartitionedMaxQOverA:
    """The maximum of Q over this device's run of the action product, merged.

    Called inside `_OnActionPartitionAxis`, so the action axis is bound and
    every device of it returns the same merged value.
    """

    __name__: ClassVar[str] = "action_partitioned_max_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    Q_and_F: RawActionFunction
    """The regime's action value and feasibility, evaluated per action block."""

    action_names: tuple[ActionName, ...]
    """Action variable names, discrete first, spanning the partitioned product."""

    q_and_f_arg_names: frozenset[ReferenceName]
    """The argument names `Q_and_F` declares, which select what it is handed."""

    action_width_keyword: str
    """Name of the planner-bound static action-block width in the call."""

    n_partitions: int
    """Devices along the action axis, each owning one run of blocks."""

    members: tuple[str, ...]
    """Ordered member names of the household."""

    pareto_weights: ParetoWeights
    """The household's Pareto weight evaluator."""

    def __call__(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> BoundedMaxOutput:
        """Return every member's value and the dissolution flag after the merge."""
        reduce_cell = build_partitioned_streaming_max_Q_over_a(
            Q_and_F=self.Q_and_F,
            action_names=self.action_names,
            block_width=cast("int", states_actions_params[self.action_width_keyword]),
            n_partitions=self.n_partitions,
            axis_name=ACTION_PARTITION_AXIS,
            members=self.members,
            weights=_evaluate_pareto_weights(
                pareto_weights=self.pareto_weights,
                states_actions_params=states_actions_params,
            ),
        )
        result = reduce_cell(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **_arguments_named(
                arguments=states_actions_params, names=self.q_and_f_arg_names
            ),
        )
        if isinstance(result, BoundedResult):
            return BoundedResult(
                output=self._published(reduced=result.output),
                bound_status=result.bound_status,
                query_status=result.query_status,
            )
        return self._published(reduced=result)

    def _published(self, *, reduced: CollectiveHardMaxResult) -> MaxOutput:
        """Return what the kernel publishes from the merged reduction."""
        return reduced.best_member_values, ~reduced.any_feasible


def _arguments_named[V](
    *, arguments: Mapping[ReferenceName, V], names: frozenset[str]
) -> dict[ReferenceName, V]:
    """Return the entries of `arguments` whose names `names` lists."""
    return {name: value for name, value in arguments.items() if name in names}


@dataclass(frozen=True, kw_only=True, eq=False)
class _OnActionPartitionAxis:
    """Run a kernel once per device of the mesh's action axis.

    Only the action axis is manual: every input is replicated along it, and
    the regime's state axes retain the enclosing execution context's typing.
    The merged value is identical on every device of the axis, so the output
    is replicated along it.
    """

    __name__: ClassVar[str] = "on_action_partition_axis"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    function: Callable[..., BoundedMaxOutput]
    """The state-mapped kernel, reading the action axis inside its reduction."""

    mesh: jax.sharding.Mesh
    """The regime's mesh, whose trailing axis is the action axis."""

    static_names: tuple[str, ...]
    """Planner-bound widths, passed as compile-time constants, not operands."""

    def __call__(self, **kwargs: QAndFArg) -> BoundedMaxOutput:
        """Evaluate the kernel on every device of the action axis."""
        static = {name: kwargs.pop(name) for name in self.static_names}
        mesh = self.mesh
        context = jax.sharding.get_abstract_mesh()
        if context.axis_names == mesh.axis_names and context.shape == mesh.shape:
            mesh = mesh.update(axis_types=context.axis_types)
        return jax.shard_map(
            functools.partial(_call_with_operands, function=self.function, **static),
            mesh=mesh,
            in_specs=jax.P(),
            out_specs=jax.P(),
            axis_names=frozenset({ACTION_PARTITION_AXIS}),
            # Every device merges the same gathered accumulators in the same
            # order, so the replicated output holds one value per cell.
            check_vma=False,
        )(kwargs)


# keyword-only-exempt: library-callback=jax.shard_map
def _call_with_operands(
    operands: QAndFKwargs,
    *,
    function: Callable[..., BoundedMaxOutput],
    **static: int,
) -> BoundedMaxOutput:
    """Call `function` with its traced operands and its static widths."""
    return function(**operands, **static)


def _fail_if_action_width_keyword_collides(
    *,
    action_width_keyword: str,
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    extra_param_names: list[QualifiedName],
) -> None:
    """Keep the planner-owned width outside the model runtime namespace."""
    runtime_arg_names = frozenset(
        ("next_regime_to_V_arr", *action_names, *state_names, *extra_param_names)
    )
    if action_width_keyword in runtime_arg_names:
        raise ValueError(
            f"Action width keyword {action_width_keyword!r} collides with a "
            "streamed core argument."
        )


def _fail_if_streaming_co_map_layout_is_invalid(
    *,
    state_names: tuple[StateName, ...],
    co_map_state_names: tuple[StateName, ...],
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...],
) -> None:
    """Validate the complete named-axis contract of the streamed co-map.

    One ``in_axes`` mapping belongs to each leading co-map state. A carrying
    continuation leaf is sliced only on its current leading axis (``0``); a
    leaf without that state is passed through (``None``). Refusing every other
    spelling prevents a malformed internal layout from degrading into an
    ordinary state productmap or silently slicing a different continuation
    coordinate.
    """
    _fail_if_co_map_states_not_leading(
        state_names=state_names, co_map_state_names=co_map_state_names
    )
    if len(co_map_state_names) != len(co_map_v_arr_in_axes):
        raise ValueError(
            "Streaming co-map state names and continuation in_axes must have "
            "the same length."
        )

    if not co_map_state_names:
        return

    target_names = frozenset(co_map_v_arr_in_axes[0])
    if not target_names:
        raise ValueError(
            "Streaming co-map continuation in_axes must name at least one target."
        )
    inconsistent_target_names = [
        (state_name, tuple(target_axes))
        for state_name, target_axes in zip(
            co_map_state_names, co_map_v_arr_in_axes, strict=True
        )
        if frozenset(target_axes) != target_names
    ]
    if inconsistent_target_names:
        raise ValueError(
            "Streaming co-map continuation in_axes must name the same target keys "
            f"for every state; got {inconsistent_target_names}."
        )

    invalid = [
        (state_name, target, axis)
        for state_name, target_axes in zip(
            co_map_state_names, co_map_v_arr_in_axes, strict=True
        )
        for target, axis in target_axes.items()
        if axis is not None
        and (not isinstance(axis, int) or isinstance(axis, bool) or axis != 0)
    ]
    if invalid:
        raise ValueError(
            "Streaming co-map continuation in_axes must contain only 0 or None; "
            f"got {invalid}."
        )


def get_argmax_and_max_Q_over_a(
    *,
    Q_and_F: RawActionFunction,
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
    members: tuple[str, ...],
    pareto_weights: ParetoWeights,
) -> ArgmaxQOverAFunction[BoundedArgmaxOutput]:
    r"""Get the function returning the arguments maximizing Q over all actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  W(U(x, a), \mathbb{E}[V(x', a') | x, a]),
    ```
    with $W(U, v) = u + \beta \cdot v$ as the leading case (which is the only one that
    is pre-implemented in LCM).

    Fixing a state but choosing the feasible actions that maximize Q, we get the optimal
    policy

    ```{math}
    \pi(x) = \argmax_{a} Q(x, a).
    ```

    This last step is handled by the function returned here.

    Args:
        Q_and_F: A function that takes a state-action combination and returns the action
            value of that combination and whether the state-action combination is
            feasible.
        action_names: Tuple of action variable names (discrete first, continuous
            last — the `StateActionSpace.action_names` order).
        state_names: Tuple of state names.
        n_discrete_action_axes: Number of leading discrete-action axes in the
            Q array. Only used when `has_taste_shocks` is set.
        has_taste_shocks: Whether the regime declares EV1 taste shocks. When
            set, the returned function takes a leading `taste_shock_key`
            argument and draws the discrete action by Gumbel-max: per-discrete-
            combination mean-zero `scale * (Gumbel(0, 1) - EULER_GAMMA)` noise
            is added to the masked maxima over the continuous axes before the
            discrete argmax — exactly logit-consistent with the smoothed solve.
        members: Ordered member names. `Q_and_F` returns a stacked per-member
            `Q` (trailing member axis); the household argmax of the weighted
            scalarization is computed once (`collective_argmax_and_readout`)
            and each member's own value is gathered at that shared index —
            mirrors the solve-side `get_max_Q_over_a` so simulate recomputes the
            identical argmax. With `has_taste_shocks`, the member with the
            positive Pareto weight draws the action, and each member's own value
            is gathered at the drawn index.
        pareto_weights: The household's Pareto weight evaluator, called at each
            cell with the states and parameters its declaration reads.

    Returns:
        Function that calculates the argument maximizing Q over the feasible continuous
        actions and the maximum itself. The argument maximizing Q is the policy
        function of the continuous actions, conditional on the states and discrete
        actions. The maximum corresponds to the Qc-function.

    """
    # Extract extra param names from Q_and_F's signature (flat regime params)
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
    )
    # A Pareto weight is evaluated at the cell, so its free parameters ride in
    # this kernel's own signature alongside `Q_and_F`'s.
    extra_param_names = list(
        dict.fromkeys((*extra_param_names, *pareto_weights.param_names))
    )

    bound_summary = bool(has_action_integrity(function=Q_and_F))
    action_kernel: ActionFunction = (
        BoundedQAndF(function=Q_and_F) if bound_summary else Q_and_F
    )
    mapped_action_kernel = productmap(
        func=action_kernel,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
        bound_summary=bound_summary,
    )

    argmax_and_max_Q_over_a: ArgmaxQOverAFunction[BoundedArgmaxOutput]
    if has_taste_shocks:
        argmax_and_max_Q_over_a = with_signature(
            _TasteShockArgmaxQOverA(
                Q_and_F=mapped_action_kernel,
                n_discrete_action_axes=n_discrete_action_axes,
                members=members,
                pareto_weights=pareto_weights,
            ),
            args=[
                "next_regime_to_V_arr",
                "taste_shock_key",
                *action_names,
                *state_names,
                *extra_param_names,
            ],
            return_annotation="tuple[IntND, FloatND]",
            enforce=False,
        )
    else:
        argmax_and_max_Q_over_a = with_signature(
            _HardMaxArgmaxQOverA(
                Q_and_F=mapped_action_kernel,
                members=members,
                pareto_weights=pareto_weights,
            ),
            args=[
                "next_regime_to_V_arr",
                *action_names,
                *state_names,
                *extra_param_names,
            ],
            return_annotation="tuple[IntND, FloatND]",
            enforce=False,
        )

    return argmax_and_max_Q_over_a


@dataclass(frozen=True, kw_only=True, eq=False)
class _TasteShockArgmaxQOverA:
    """The Gumbel-max discrete draw and its value, at one state cell.

    Every value the draw reads is an explicit field, so the regime's action
    kernel stays reachable only through this instance, which the model build
    drops with the policy function it was built for. The caller wraps it in
    `dags.with_signature` to advertise the arguments the simulation space map
    binds by name.
    """

    __name__: ClassVar[str] = "argmax_and_max_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    Q_and_F: ActionFunction
    """The regime's action value and feasibility, mapped over the action product."""

    n_discrete_action_axes: int
    """Number of leading discrete-action axes the noisy draw chooses among."""

    members: tuple[str, ...]
    """Ordered member names of the household."""

    pareto_weights: ParetoWeights
    """The household's Pareto weight evaluator."""

    def __call__(
        self,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg | PRNGKeyND,
    ) -> BoundedArgmaxOutput:
        """Return the flat index of the drawn action and its noise-free value."""
        taste_shock_key = cast("Array", states_actions_params.pop("taste_shock_key"))
        evaluated = self.Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        Q_arr, F_arr = cast(
            "tuple[FloatND, BoolND]",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        return _with_bound_summary(
            evaluated=evaluated,
            output=_household_taste_shock_draw(
                Q_arr=Q_arr,
                F_arr=F_arr,
                weights=_stacked_pareto_weights(
                    pareto_weights=self.pareto_weights,
                    members=self.members,
                    states_actions_params=cast("QAndFKwargs", states_actions_params),
                ),
                n_discrete_action_axes=self.n_discrete_action_axes,
                taste_shock_key=taste_shock_key,
                scale=cast("FloatND", states_actions_params[TASTE_SHOCK_SCALE_PARAM]),
            ),
        )


def _household_taste_shock_draw(
    *,
    Q_arr: FloatND,
    F_arr: BoolND,
    weights: FloatND,
    n_discrete_action_axes: int,
    taste_shock_key: Array,
    scale: FloatND,
) -> ArgmaxOutput:
    """Draw a household's action as its decider's, and read every member there.

    The decider is the member with the positive Pareto weight. Its draw adds
    one mean-zero Gumbel term per discrete action to its own conditional maxima
    over the continuous actions. The decider publishes its conditional maximum
    at the drawn action, and every other member its own noise-free action
    value there.
    """
    decider, Qc, continuous_argmax, at_choice = decider_choice(
        Q=Q_arr,
        feasibility=F_arr,
        weights=weights,
        n_discrete_action_axes=n_discrete_action_axes,
    )
    noise = draw_taste_shock_noise(key=taste_shock_key, shape=Qc.shape, scale=scale)
    # An infeasible discrete cell stays infeasible: the noise is finite, so
    # `-inf + noise` is still `-inf`.
    discrete_argmax = jnp.argmax(Qc + noise)
    flat_index = (
        discrete_argmax * math.prod(F_arr.shape[n_discrete_action_axes:])
        + continuous_argmax[discrete_argmax]
    )
    values = jnp.where(
        jnp.arange(at_choice.shape[-1]) == decider,
        Qc[discrete_argmax],
        at_choice[discrete_argmax],
    )
    # Without an alternative worth more than `-inf` to the decider there is
    # nothing to draw: the cell publishes no action, as one without a feasible
    # action does.
    worthless = jnp.all(jnp.isneginf(Qc))
    return (
        jnp.where(worthless, -1, flat_index).astype(jnp.int32),
        jnp.where(worthless, -jnp.inf, values),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _HardMaxArgmaxQOverA:
    """The argmax of Q over the feasible actions and its value, at one state cell.

    Every value the argmax reads is an explicit field, so the regime's action
    kernel stays reachable only through this instance, which the model build
    drops with the policy function it was built for. The caller wraps it in
    `dags.with_signature` to advertise the arguments the simulation space map
    binds by name.
    """

    __name__: ClassVar[str] = "argmax_and_max_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    Q_and_F: ActionFunction
    """The regime's action value and feasibility, mapped over the action product."""

    members: tuple[str, ...]
    """Ordered member names of the household."""

    pareto_weights: ParetoWeights
    """The household's Pareto weight evaluator."""

    def __call__(
        self,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> BoundedArgmaxOutput:
        """Return the flat index of the chosen action and the value it attains."""
        evaluated = self.Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        Q_arr, F_arr = cast(
            "tuple[FloatND, BoolND]",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        # Mirrors the solve-side `get_max_Q_over_a` — split the stacked Q by
        # member, argmax the household scalarization once over the value-masked
        # feasible action set, and gather each member's OWN value at that shared
        # index. The simulate-only addition vs. the solve readout
        # (`collective_readout`) is the argmax index itself, needed to look up
        # which JOINT action the members actually took
        # (`_lookup_values_from_indices` in `simulate.py`).
        action_axes = tuple(range(F_arr.ndim))
        member_Q = {name: Q_arr[..., index] for index, name in enumerate(self.members)}
        argmax_flat, values, _dissolution = collective_argmax_and_readout(
            member_Q=member_Q,
            feasibility=F_arr,
            weights=_evaluate_pareto_weights(
                pareto_weights=self.pareto_weights,
                states_actions_params=states_actions_params,
            ),
            action_axes=action_axes,
        )
        V_stacked = jnp.stack([values[name] for name in self.members], axis=-1)
        return _with_bound_summary(
            evaluated=evaluated,
            output=(jnp.where(_dissolution, -1, argmax_flat), V_stacked),
        )


def draw_taste_shock_noise(
    *,
    key: Array,
    shape: tuple[int, ...],
    scale: FloatND,
) -> FloatND:
    """Draw the additive, mean-zero EV1 taste-shock noise for discrete choices.

    The draw is `scale * (jax.random.gumbel(key, shape) - EULER_GAMMA)`. A raw
    Gumbel(0, 1) has mean `EULER_GAMMA`, so subtracting it makes the shock
    mean-zero — the condition under which the solve's smoothed maximum
    `scale * logsumexp(Qc / scale)` equals the expected realized maximum.

    A key with a trailing length-one axis is an independent taste key whose
    batched draws would not follow each key, so its draw runs as a scalar
    computation for every key under any enclosing `jax.vmap`. A scalar key
    keeps the vectorized draw.

    Args:
        key: JAX PRNG key for the Gumbel draw, scalar or of shape `(1,)`.
        shape: Shape of the noise array (one draw per discrete-action cell).
        scale: Taste-shock scale; broadcasts against the draw.

    Returns:
        Mean-zero additive noise of the given shape.

    """
    if key.shape == (1,):
        gumbel = jax.custom_batching.sequential_vmap(
            functools.partial(jax.random.gumbel, shape=shape)
        )(key[0])
    else:
        gumbel = jax.random.gumbel(key, shape)
    return scale * (gumbel - EULER_GAMMA)


def _fail_if_co_map_states_not_leading(
    *,
    state_names: tuple[StateName, ...],
    co_map_state_names: tuple[StateName, ...],
) -> None:
    """Fail if the co-mapped states are not a leading prefix of `state_names`.

    The co-map peels axes off the front of each `next_regime_to_V_arr` leaf, so the
    co-mapped states must be exactly the leading axes of the value-function array, in
    order.
    """
    leading = state_names[: len(co_map_state_names)]
    if tuple(co_map_state_names) != leading:
        msg = (
            "Co-mapped states must be the leading axes of the value-function array, "
            f"in order. Got co_map_state_names={co_map_state_names} but the leading "
            f"state_names are {leading}."
        )
        raise ValueError(msg)


def _get_extra_param_names(
    *,
    Q_and_F: ActionFunction,
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
) -> list[str]:
    """Get param names from Q_and_F not in actions, states, or next_regime_to_V_arr."""
    sig = inspect.signature(Q_and_F)
    known_names = {"next_regime_to_V_arr", *action_names, *state_names}
    return sorted(
        name
        for name, param in sig.parameters.items()
        if name not in known_names
        and param.kind
        not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    )
