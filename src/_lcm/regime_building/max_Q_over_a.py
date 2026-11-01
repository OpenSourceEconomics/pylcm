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
from _lcm.logsum import EULER_GAMMA, logsum_and_softmax
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.collective import (
    ParetoWeights,
    collective_argmax_and_readout,
    collective_readout,
)
from _lcm.regime_building.Q_and_F import (
    BoundedActionOutput,
    BoundedQAndF,
    has_action_integrity,
)
from _lcm.solution.action_reduction import HardMaxResult
from _lcm.solution.action_streaming import (
    build_partitioned_streaming_max_Q_over_a,
    build_streaming_collective_max_Q_over_a,
    build_streaming_ev1_max_Q_over_a,
    build_streaming_max_Q_over_a,
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
    pareto_weights: ParetoWeights | None,
    states_actions_params: QAndFKwargs,
) -> dict[str, FloatND]:
    """Evaluate the household's Pareto weights at one cell.

    The kernel's own signature carries every argument the declaration reads, so
    the weights are read out of the cell rather than closed over — which is
    what makes a state-dependent or estimated weight ordinary.
    """
    weights = cast("ParetoWeights", pareto_weights)
    return weights.compute(
        **{name: states_actions_params[name] for name in weights.arg_names}
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
    stakeholders: tuple[str, ...] | None = None,
    pareto_weights: ParetoWeights | None = None,
    cell_width_keyword: str | None = None,
    untiled_state_names: tuple[StateName, ...] = (),
    broadcast_state_names: tuple[StateName, ...] = (),
    return_feasibility: bool = False,
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
            `taste_shocks__scale`.
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
        stakeholders: Ordered stakeholder names for a collective regime, or `None`
            (the singleton default). When set, `Q_and_F` returns a stacked
            per-stakeholder `Q` (trailing stakeholder axis) and the inner reduction
            reads off each stakeholder's own value at the shared household argmax
            (`collective_readout`) instead of the plain masked max; the returned
            function then yields the pair `(V, D)` — the stakeholder-axis value
            array plus the boolean dissolution flag `D = 1[mask empty]` on the state
            axes — distinct from a numeric `-inf`, which occurs on-path.
        pareto_weights: The household's Pareto weight evaluator; required (and
            only used) when `stakeholders` is set. Called at each cell with the
            states and parameters its declaration reads.

    Returns:
        V, i.e., the function that calculates the maximum of the Q-function over all
        feasible actions — or, for a collective regime, the pair `(V, D)`.

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
    if pareto_weights is not None:
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
                return_feasibility=return_feasibility,
            ),
            args=[
                "next_regime_to_V_arr",
                *action_names,
                *state_names,
                *extra_param_names,
            ],
            return_annotation=(
                "tuple[FloatND, BoolND]" if return_feasibility else "FloatND"
            ),
            enforce=False,
        )
    else:
        max_Q_over_a = with_signature(
            _HardMaxQOverA(
                Q_and_F=mapped_action_kernel,
                stakeholders=stakeholders,
                pareto_weights=pareto_weights,
                return_feasibility=return_feasibility,
            ),
            args=[
                "next_regime_to_V_arr",
                *action_names,
                *state_names,
                *extra_param_names,
            ],
            return_annotation=(
                "tuple[FloatND, BoolND]"
                if stakeholders is not None or return_feasibility
                else "FloatND"
            ),
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

    return_feasibility: bool = False
    """Whether to return the empty-action-set mask beside the value."""

    def __call__(
        self,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> BoundedMaxOutput:
        """Return the expected maximum over the discrete actions at this cell."""
        evaluated = self.Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        Q_arr, F_arr = cast(
            "tuple[FloatND, BoolND]",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)
        continuous_axes = tuple(range(self.n_discrete_action_axes, Q_arr.ndim))
        Qc = Q_masked.max(axis=continuous_axes) if continuous_axes else Q_masked
        smoothed, _ = logsum_and_softmax(
            values=Qc,
            scale=cast("ScalarFloat", states_actions_params[TASTE_SHOCK_SCALE_PARAM]),
            axes=tuple(range(Qc.ndim)),
        )
        output: MaxOutput = (
            (smoothed, ~jnp.any(F_arr)) if self.return_feasibility else smoothed
        )
        return _with_bound_summary(
            evaluated=evaluated, output=cast("MaxOutput", output)
        )


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

    stakeholders: tuple[str, ...] | None
    """Ordered stakeholder names of a collective regime, or `None` for a singleton."""

    pareto_weights: ParetoWeights | None
    """The household's Pareto weight evaluator, or `None` for a singleton."""

    return_feasibility: bool = False
    """Whether to return the empty-action-set mask beside a singleton value."""

    def __call__(
        self,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> BoundedMaxOutput:
        """Return the cell's value, plus the dissolution flag for a household."""
        evaluated = self.Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        Q_arr, F_arr = cast(
            "tuple[FloatND, BoolND]",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        if self.stakeholders is not None:
            # Q_arr carries a trailing
            # stakeholder axis (the action product-map keeps it last);
            # F_arr does not — where the regime declares value constraints
            # F_arr already includes them, ANDed in by Q_and_F AFTER computing
            # Q^s. Split Q_arr per stakeholder, take the household argmax
            # of the scalarization over the masked action axes, and read
            # off each stakeholder's OWN value at that shared argmax. The
            # returned pair is the stakeholder value vector (re-stacked on
            # a trailing axis, which the outer state product-map turns
            # into `(*states, n_stakeholders)`) plus the dissolution flag D —
            # `True` where NO action is feasible (empty mask), published
            # alongside V and never conflated with a numeric -inf value
            # (which occurs on-path).
            action_axes = tuple(range(F_arr.ndim))
            stakeholder_Q = {
                name: Q_arr[..., index] for index, name in enumerate(self.stakeholders)
            }
            values, dissolution = collective_readout(
                stakeholder_Q=stakeholder_Q,
                feasibility=F_arr,
                weights=_evaluate_pareto_weights(
                    pareto_weights=self.pareto_weights,
                    states_actions_params=states_actions_params,
                ),
                action_axes=action_axes,
            )
            output: MaxOutput = (
                jnp.stack([values[name] for name in self.stakeholders], axis=-1),
                dissolution,
            )
            return _with_bound_summary(
                evaluated=evaluated, output=cast("MaxOutput", output)
            )
        value = Q_arr.max(where=F_arr, initial=-jnp.inf)
        output = (value, ~jnp.any(F_arr)) if self.return_feasibility else value
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
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
    co_map_state_names: tuple[StateName, ...] = (),
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = (),
    stakeholders: tuple[str, ...] | None = None,
    pareto_weights: ParetoWeights | None = None,
    action_width_keyword: str = "_lcm_action_block_width",
    cell_width_keyword: str | None = None,
    untiled_state_names: tuple[StateName, ...] = (),
    broadcast_state_names: tuple[StateName, ...] = (),
    return_feasibility: bool = False,
) -> MaxQOverAFunction[BoundedMaxOutput]:
    """Build a singleton or collective V kernel that streams the action product.

    The returned raw callable has the same dynamic argument layout as
    `get_max_Q_over_a` plus one required static keyword named by
    `action_width_keyword`. The execution planner selects a collision-free name and
    binds that width before tracing. For each state cell, the fixed-cell reducer
    evaluates actions in canonical C order. A hard-max singleton publishes its best
    value; an EV1 singleton first hard-maxes each discrete-prefix branch and then
    log-sums the branch values; a collective regime publishes each stakeholder's
    value at one shared household winner plus the empty-feasible-set flag. Ordinary
    states use the existing state productmap. Fixed distributed states retain the
    dense route co-map access law: each state axis is mapped in lockstep with axis
    zero of every continuation leaf that carries it.
    """
    _fail_if_streaming_co_map_layout_is_invalid(
        state_names=state_names,
        co_map_state_names=co_map_state_names,
        co_map_v_arr_in_axes=co_map_v_arr_in_axes,
    )
    _fail_if_full_V_streaming_route_is_unsupported(
        has_taste_shocks=has_taste_shocks,
        stakeholders=stakeholders,
        pareto_weights=pareto_weights,
    )
    if has_taste_shocks and return_feasibility:
        raise ValueError("Feasibility output for EV1 uses the canonical dense reducer.")
    if has_taste_shocks and not 1 <= n_discrete_action_axes <= len(action_names):
        raise ValueError(
            "EV1 action streaming requires a non-empty leading discrete-action prefix"
        )

    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F,
        action_names=action_names,
        state_names=state_names,
    )
    if has_taste_shocks and TASTE_SHOCK_SCALE_PARAM not in extra_param_names:
        extra_param_names.append(TASTE_SHOCK_SCALE_PARAM)
    q_and_f_arg_names = frozenset(inspect.signature(Q_and_F).parameters)

    if pareto_weights is not None:
        extra_param_names = list(
            dict.fromkeys((*extra_param_names, *pareto_weights.param_names))
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
            n_discrete_action_axes=n_discrete_action_axes,
            has_taste_shocks=has_taste_shocks,
            stakeholders=stakeholders,
            pareto_weights=pareto_weights,
            q_and_f_arg_names=q_and_f_arg_names,
            action_width_keyword=action_width_keyword,
            return_feasibility=return_feasibility,
            whole_product_Q_and_F=productmap(
                func=Q_and_F,
                variables=action_names,
                batch_sizes=dict.fromkeys(action_names, 0),
            ),
        ),
        args=[
            "next_regime_to_V_arr",
            *action_names,
            *state_names,
            *extra_param_names,
            action_width_keyword,
        ],
        return_annotation=(
            "tuple[FloatND, BoolND]"
            if stakeholders is not None or return_feasibility
            else "FloatND"
        ),
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

    n_discrete_action_axes: int
    """Number of leading discrete-action axes an EV1 route log-sums over."""

    has_taste_shocks: bool
    """Whether the regime declares EV1 taste shocks on its discrete actions."""

    stakeholders: tuple[str, ...] | None
    """Ordered stakeholder names of a collective regime, or `None` for a singleton."""

    pareto_weights: ParetoWeights | None
    """The household's Pareto weight evaluator, or `None` for a singleton."""

    q_and_f_arg_names: frozenset[ReferenceName]
    """The argument names `Q_and_F` declares, which select what it is handed."""

    action_width_keyword: str
    """Name of the planner-bound static action-block width in the call."""

    whole_product_Q_and_F: RawActionFunction
    """`Q_and_F` mapped over the named action grids, for a block covering them."""

    return_feasibility: bool = False
    """Whether to return the empty-action-set mask beside a singleton value."""

    def __call__(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> BoundedMaxOutput:
        """Return the cell's value, plus the dissolution flag for a household."""
        action_block_width = cast(
            "int", states_actions_params[self.action_width_keyword]
        )
        q_and_f_params = {
            name: value
            for name, value in states_actions_params.items()
            if name in self.q_and_f_arg_names
        }
        if (
            has_action_integrity(function=self.Q_and_F)
            and not self.has_taste_shocks
            and self.stakeholders is None
        ):
            fixed_cell = build_streaming_max_Q_over_a(
                Q_and_F=self.Q_and_F,
                action_names=self.action_names,
                block_width=action_block_width,
            )
            bounded = cast(
                "BoundedResult[HardMaxResult]",
                fixed_cell(
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    **q_and_f_params,
                ),
            )
            result = bounded.output
            return BoundedResult(
                output=(
                    (result.best_value, ~result.any_feasible)
                    if self.return_feasibility
                    else result.best_value
                ),
                bound_status=bounded.bound_status,
                query_status=bounded.query_status,
            )
        if self.has_taste_shocks:
            ev1_cell = build_streaming_ev1_max_Q_over_a(
                Q_and_F=self.Q_and_F,
                action_names=self.action_names,
                n_discrete_action_axes=self.n_discrete_action_axes,
                block_width=action_block_width,
                scale=cast(
                    "ScalarFloat",
                    states_actions_params[TASTE_SHOCK_SCALE_PARAM],
                ),
            )
            ev1_result = ev1_cell(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **q_and_f_params,
            )
            if isinstance(ev1_result, BoundedResult):
                ev1_output = BoundedResult(
                    output=ev1_result.output.smoothed_value,
                    bound_status=ev1_result.bound_status,
                    query_status=ev1_result.query_status,
                )
            else:
                ev1_output = ev1_result.smoothed_value
            return cast("BoundedMaxOutput", ev1_output)

        if self.stakeholders is None:
            n_actions = math.prod(
                jnp.shape(cast("Array", states_actions_params[name]))[0]
                for name in self.action_names
            )
            if action_block_width >= n_actions:
                # One block covers the product: map the named grids, as the dense
                # route does, instead of decoding a flat index into each of them.
                # The streamed reduction's value contract is stated explicitly: a
                # feasible NaN publishes NaN. XLA:CPU's vectorized maximum drops NaN
                # once a mapped reduction spans a few thousand elements.
                Q_arr, F_arr = self.whole_product_Q_and_F(
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    **q_and_f_params,
                )
                # One variadic reduce yields the maximum and the NaN flag, so the
                # backend fuses Q into a single pass instead of materialising it
                # for two consumers.
                best, any_feasible_nan = jax.lax.reduce(
                    (
                        jnp.where(F_arr, Q_arr, -jnp.inf),
                        F_arr & jnp.isnan(Q_arr),
                    ),
                    (jnp.asarray(-jnp.inf, Q_arr.dtype), jnp.zeros((), dtype=bool)),
                    lambda left, right: (
                        jax.lax.max(left[0], right[0]),
                        jax.lax.bitwise_or(left[1], right[1]),
                    ),
                    tuple(range(jnp.ndim(Q_arr))),
                )
                value = jnp.where(any_feasible_nan, jnp.nan, best)
                return (value, ~jnp.any(F_arr)) if self.return_feasibility else value
            fixed_cell = build_streaming_max_Q_over_a(
                Q_and_F=self.Q_and_F,
                action_names=self.action_names,
                block_width=action_block_width,
            )
            result = cast(
                "HardMaxResult",
                fixed_cell(
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    **q_and_f_params,
                ),
            )
            return (
                (result.best_value, ~result.any_feasible)
                if self.return_feasibility
                else result.best_value
            )

        collective_cell = build_streaming_collective_max_Q_over_a(
            Q_and_F=self.Q_and_F,
            action_names=self.action_names,
            block_width=action_block_width,
            stakeholders=self.stakeholders,
            weights=_evaluate_pareto_weights(
                pareto_weights=self.pareto_weights,
                states_actions_params=states_actions_params,
            ),
        )
        collective_result = collective_cell(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **q_and_f_params,
        )
        if isinstance(collective_result, BoundedResult):
            result = collective_result.output
            return BoundedResult(
                output=(result.best_stakeholder_values, ~result.any_feasible),
                bound_status=collective_result.bound_status,
                query_status=collective_result.query_status,
            )
        return (
            collective_result.best_stakeholder_values,
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
    return_feasibility: bool = False,
) -> MaxQOverAFunction[BoundedMaxOutput]:
    """Build a singleton hard-max V kernel whose action product devices share.

    The returned callable has the argument layout of
    `get_streaming_max_Q_over_a` for an ordinary singleton regime. It runs its
    body once per device of `mesh`'s `ACTION_PARTITION_AXIS`, with every input
    replicated along that axis and the regime's state axes left to the
    compiler. Each device streams the contiguous run of whole action blocks
    `ActionPartitionLayout` assigns it at the planner-bound width, gathers one
    hard-max accumulator per state cell from every device of the axis and
    merges them in partition order, so all devices publish the same value.

    Taste shocks, collective regimes and co-mapped states are
    not served; model construction refuses them before this is called.
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
            return_feasibility=return_feasibility,
        ),
        args=[
            "next_regime_to_V_arr",
            *action_names,
            *state_names,
            *extra_param_names,
            action_width_keyword,
        ],
        return_annotation=(
            "tuple[FloatND, BoolND]" if return_feasibility else "FloatND"
        ),
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
            return_annotation=(
                "tuple[FloatND, BoolND]" if return_feasibility else "FloatND"
            ),
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

    return_feasibility: bool = False
    """Whether to return the merged empty-action-set mask beside the value."""

    def __call__(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> BoundedMaxOutput:
        """Return the cell's value after merging every device's accumulator."""
        reduce_cell = build_partitioned_streaming_max_Q_over_a(
            Q_and_F=self.Q_and_F,
            action_names=self.action_names,
            block_width=cast("int", states_actions_params[self.action_width_keyword]),
            n_partitions=self.n_partitions,
            axis_name=ACTION_PARTITION_AXIS,
        )
        result = reduce_cell(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **_arguments_named(
                arguments=states_actions_params, names=self.q_and_f_arg_names
            ),
        )
        if isinstance(result, BoundedResult):
            reduced = result.output
            return BoundedResult(
                output=(
                    (reduced.best_value, ~reduced.any_feasible)
                    if self.return_feasibility
                    else reduced.best_value
                ),
                bound_status=result.bound_status,
                query_status=result.query_status,
            )
        return (
            (result.best_value, ~result.any_feasible)
            if self.return_feasibility
            else result.best_value
        )


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


def _fail_if_full_V_streaming_route_is_unsupported(
    *,
    has_taste_shocks: bool,
    stakeholders: tuple[str, ...] | None,
    pareto_weights: ParetoWeights | None,
) -> None:
    """Reject routes unsupported by full-value action streaming."""
    if has_taste_shocks and stakeholders is not None:
        raise NotImplementedError(
            "Full-V action streaming does not support collective EV1 regimes."
        )
    if (stakeholders is None) != (pareto_weights is None):
        raise ValueError(
            "Collective action streaming requires stakeholders and Pareto weights."
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
    stakeholders: tuple[str, ...] | None = None,
    pareto_weights: ParetoWeights | None = None,
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
        stakeholders: Ordered stakeholder names for a collective regime, or
            `None` (the singleton default). When
            set, `Q_and_F` returns a stacked per-stakeholder `Q` (trailing
            stakeholder axis); the household argmax of the weighted
            scalarization is computed once (`collective_argmax_and_readout`)
            and each stakeholder's own value is gathered at that shared
            index — mirrors the solve-side `get_max_Q_over_a` collective
            branch so simulate recomputes the identical argmax. Mutually
            exclusive with `has_taste_shocks` (rejected at regime
            construction for collective regimes).
        pareto_weights: The household's Pareto weight evaluator; required (and
            only used) when `stakeholders` is set. Called at each cell with the
            states and parameters its declaration reads.

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
    if pareto_weights is not None:
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
                stakeholders=stakeholders,
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
        Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)
        n_discrete_cells = math.prod(Q_arr.shape[: self.n_discrete_action_axes])
        n_continuous_cells = math.prod(Q_arr.shape[self.n_discrete_action_axes :])
        Q_flat = Q_masked.reshape(n_discrete_cells, n_continuous_cells)
        continuous_argmax = jnp.argmax(Q_flat, axis=1)
        Qc = Q_flat.max(axis=1)
        scale = cast("FloatND", states_actions_params[TASTE_SHOCK_SCALE_PARAM])
        noise = draw_taste_shock_noise(key=taste_shock_key, shape=Qc.shape, scale=scale)
        # An infeasible discrete cell stays infeasible: the noise is
        # finite, so `-inf + noise` is still `-inf`.
        noisy_Qc = Qc + noise
        discrete_argmax = jnp.argmax(noisy_Qc)
        flat_index = (
            discrete_argmax * n_continuous_cells + continuous_argmax[discrete_argmax]
        )
        output = (
            jnp.where(jnp.any(F_arr), flat_index, -1).astype(jnp.int32),
            Qc[discrete_argmax],
        )
        return _with_bound_summary(evaluated=evaluated, output=output)


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

    stakeholders: tuple[str, ...] | None
    """Ordered stakeholder names of a collective regime, or `None` for a singleton."""

    pareto_weights: ParetoWeights | None
    """The household's Pareto weight evaluator, or `None` for a singleton."""

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
        if self.stakeholders is not None:
            # Mirrors the solve-side collective
            # branch in `get_max_Q_over_a` — split the stacked Q by
            # stakeholder, argmax the household scalarization once over
            # the value-masked feasible action set, and gather
            # each stakeholder's OWN value at that shared index. The
            # simulate-only addition vs. the solve readout
            # (`collective_readout`) is the argmax index itself, needed
            # to look up which JOINT action both stakeholders actually
            # took (`_lookup_values_from_indices` in `simulate.py`).
            action_axes = tuple(range(F_arr.ndim))
            stakeholder_Q = {
                name: Q_arr[..., index] for index, name in enumerate(self.stakeholders)
            }
            argmax_flat, values, _dissolution = collective_argmax_and_readout(
                stakeholder_Q=stakeholder_Q,
                feasibility=F_arr,
                weights=_evaluate_pareto_weights(
                    pareto_weights=self.pareto_weights,
                    states_actions_params=states_actions_params,
                ),
                action_axes=action_axes,
            )
            V_stacked = jnp.stack([values[name] for name in self.stakeholders], axis=-1)
            return _with_bound_summary(
                evaluated=evaluated,
                output=(jnp.where(_dissolution, -1, argmax_flat), V_stacked),
            )
        index, value = argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)
        return _with_bound_summary(
            evaluated=evaluated,
            output=(jnp.where(jnp.any(F_arr), index, -1), value),
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
