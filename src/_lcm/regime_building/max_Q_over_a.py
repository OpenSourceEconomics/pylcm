import functools
import inspect
import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from dags import with_signature
from jax import Array

from _lcm.logsum import EULER_GAMMA, logsum_and_softmax
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.collective import (
    ParetoWeights,
    collective_argmax_and_readout,
    collective_readout,
)
from _lcm.regime_building.zero_safe import zero_safe_average
from _lcm.solution.action_streaming import (
    build_streaming_collective_max_Q_over_a,
    build_streaming_ev1_max_Q_over_a,
    build_streaming_max_Q_over_a,
)
from _lcm.typing import (
    ActionName,
    ArgmaxQOverAFunction,
    MaxQOverAFunction,
    RegimeName,
    StateName,
    _ParamsLeaf,
)
from _lcm.utils.dispatchers import productmap, vmap_1d
from _lcm.utils.functools import allow_args, allow_only_kwargs
from lcm.typing import BoolND, FloatND, IntND, ScalarFloat

# Flat param name of the EV1 taste-shock scale (template pseudo-function entry).
TASTE_SHOCK_SCALE_PARAM = "taste_shocks__scale"


def _evaluate_pareto_weights(
    *,
    pareto_weights: ParetoWeights | None,
    states_actions_params: Mapping[str, _ParamsLeaf],
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
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    batch_sizes: dict[StateName, int],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
    co_map_state_names: tuple[StateName, ...] = (),
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = (),
    stakeholders: tuple[str, ...] | None = None,
    pareto_weights: ParetoWeights | None = None,
    fold_state_names: tuple[StateName, ...] = (),
    fold_weights: Mapping[StateName, FloatND] = MappingProxyType({}),
    fold_conditioning: Mapping[StateName, StateName] = MappingProxyType({}),
) -> MaxQOverAFunction:
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
            productmap over states. A batch size of 0 means no batching.
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
        fold_state_names: IID-process states declared `fold=True`, or empty (the
            default). Each is still an ordinary inner (non-co-mapped) productmap
            axis THROUGH the max-over-actions — every node is evaluated — but its
            axis is then weighted-averaged away (with `fold_weights[name]`)
            before the result is returned, so the caller never sees it. Only a
            singleton regime may fold: a collective regime's `-inf` dissolution
            sentinel is not a value quadrature can average, so the combination
            is rejected at model build.
        fold_weights: Quadrature weights per name in `fold_state_names`. A name
            absent from `fold_conditioning` carries a 1-D array matching that
            state's node count and summing to 1; a name present there carries a
            `(n_categories, n_points)` array whose row `c` is the quadrature the
            conditioning state's category `c` selects. Must be CONCRETE (not
            traced) — `_select_fold_reducer` reads them at kernel-build time to
            pick each axis's reduction kernel. Ignored when `fold_state_names`
            is empty.
        fold_conditioning: Mapping of a folded state to the discrete state its
            quadrature is conditioned on, for the folded processes that declare
            a `StateConditioned` parameter. Absent names fold against one shared
            row. The conditioning state must itself be an inner (non-co-mapped)
            productmap axis, since the reduction gathers its row along that axis.

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

    # Actions are the inner optimization axis — batching applies only to the
    # outer state loop.
    Q_and_F = productmap(
        func=Q_and_F,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )

    if has_taste_shocks:

        @with_signature(
            args=[
                "next_regime_to_V_arr",
                *action_names,
                *state_names,
                *extra_param_names,
            ],
            return_annotation="FloatND",
            enforce=False,
        )
        def max_Q_over_a(
            next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
            **states_actions_params: _ParamsLeaf,
        ) -> FloatND:
            Q_arr, F_arr = Q_and_F(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **states_actions_params,
            )
            Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)
            continuous_axes = tuple(range(n_discrete_action_axes, Q_arr.ndim))
            Qc = Q_masked.max(axis=continuous_axes) if continuous_axes else Q_masked
            smoothed, _ = logsum_and_softmax(
                values=Qc,
                scale=cast(
                    "ScalarFloat", states_actions_params[TASTE_SHOCK_SCALE_PARAM]
                ),
                axes=tuple(range(Qc.ndim)),
            )
            return smoothed

    else:

        @with_signature(
            args=[
                "next_regime_to_V_arr",
                *action_names,
                *state_names,
                *extra_param_names,
            ],
            return_annotation=(
                "tuple[FloatND, BoolND]" if stakeholders is not None else "FloatND"
            ),
            enforce=False,
        )
        def max_Q_over_a(
            next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
            **states_actions_params: _ParamsLeaf,
        ) -> FloatND | tuple[FloatND, BoolND]:
            Q_arr, F_arr = Q_and_F(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **states_actions_params,
            )
            if stakeholders is not None:
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
                    name: Q_arr[..., index] for index, name in enumerate(stakeholders)
                }
                values, dissolution = collective_readout(
                    stakeholder_Q=stakeholder_Q,
                    feasibility=F_arr,
                    weights=_evaluate_pareto_weights(
                        pareto_weights=pareto_weights,
                        states_actions_params=states_actions_params,
                    ),
                    action_axes=action_axes,
                )
                return (
                    jnp.stack([values[name] for name in stakeholders], axis=-1),
                    dissolution,
                )
            return Q_arr.max(where=F_arr, initial=-jnp.inf)

    inner_state_names = tuple(
        name for name in state_names if name not in co_map_state_names
    )
    mapped = productmap(
        func=max_Q_over_a,
        variables=inner_state_names,
        batch_sizes={name: batch_sizes[name] for name in inner_state_names},
    )

    if fold_state_names:
        _fail_if_collective(
            fold_state_names=fold_state_names, stakeholders=stakeholders
        )
        mapped = _wrap_with_fold_reduction(
            mapped=cast("Callable[..., FloatND]", mapped),
            fold_state_names=fold_state_names,
            fold_weights=fold_weights,
            fold_conditioning=fold_conditioning,
            inner_state_names=inner_state_names,
            action_names=action_names,
            state_names=state_names,
            extra_param_names=extra_param_names,
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
        )
    return cast("MaxQOverAFunction", allow_only_kwargs(func=mapped, enforce=False))


def get_streaming_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    batch_sizes: dict[StateName, int],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
    co_map_state_names: tuple[StateName, ...] = (),
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = (),
    stakeholders: tuple[str, ...] | None = None,
    pareto_weights: ParetoWeights | None = None,
    fold_state_names: tuple[StateName, ...] = (),
    fold_weights: Mapping[StateName, FloatND] = MappingProxyType({}),
    fold_conditioning: Mapping[StateName, StateName] = MappingProxyType({}),
) -> MaxQOverAFunction:
    """Build a singleton or collective V kernel that streams the action product.

    The returned raw callable has the same dynamic argument layout as
    get_max_Q_over_a plus one required static keyword,
    _lcm_action_block_width. The execution planner binds that width before
    tracing. For each state cell, the fixed-cell reducer evaluates actions in
    canonical C order. A hard-max singleton publishes its best value; an EV1
    singleton first hard-maxes each discrete-prefix branch and then log-sums the
    branch values; a collective regime publishes every stakeholder's value at one
    shared household winner plus the empty-feasible-set flag. Ordinary states use
    the existing state productmap. Fixed distributed states instead retain the
    dense route's co-map access law: each state axis is mapped in lockstep with
    axis zero of every continuation leaf that carries it. Folded singleton routes
    stream their action product, then apply the existing quadrature reduction to the
    still-materialized fold axes before any co-map wrapper.
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
        fold_state_names=fold_state_names,
    )
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

    @with_signature(
        args=[
            "next_regime_to_V_arr",
            *action_names,
            *state_names,
            *extra_param_names,
            "_lcm_action_block_width",
        ],
        return_annotation=(
            "tuple[FloatND, BoolND]" if stakeholders is not None else "FloatND"
        ),
        enforce=False,
    )
    def streamed_max_Q_over_a(
        *,
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        _lcm_action_block_width: int,
        **states_actions_params: _ParamsLeaf,
    ) -> FloatND | tuple[FloatND, BoolND]:
        q_and_f_params = {
            name: value
            for name, value in states_actions_params.items()
            if name in q_and_f_arg_names
        }
        if has_taste_shocks:
            ev1_cell = build_streaming_ev1_max_Q_over_a(
                Q_and_F=Q_and_F,
                action_names=action_names,
                n_discrete_action_axes=n_discrete_action_axes,
                block_width=_lcm_action_block_width,
                scale=cast(
                    "ScalarFloat",
                    states_actions_params[TASTE_SHOCK_SCALE_PARAM],
                ),
            )
            ev1_result = ev1_cell(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **q_and_f_params,
            )
            return ev1_result.smoothed_value

        if stakeholders is None:
            fixed_cell = build_streaming_max_Q_over_a(
                Q_and_F=Q_and_F,
                action_names=action_names,
                block_width=_lcm_action_block_width,
            )
            result = fixed_cell(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **q_and_f_params,
            )
            return result.best_value

        collective_cell = build_streaming_collective_max_Q_over_a(
            Q_and_F=Q_and_F,
            action_names=action_names,
            block_width=_lcm_action_block_width,
            stakeholders=stakeholders,
            weights=_evaluate_pareto_weights(
                pareto_weights=pareto_weights,
                states_actions_params=states_actions_params,
            ),
        )
        collective_result = collective_cell(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **q_and_f_params,
        )
        return (
            collective_result.best_stakeholder_values,
            ~collective_result.any_feasible,
        )

    inner_state_names = tuple(
        name for name in state_names if name not in co_map_state_names
    )
    mapped = productmap(
        func=streamed_max_Q_over_a,
        variables=inner_state_names,
        batch_sizes={name: batch_sizes[name] for name in inner_state_names},
    )
    if fold_state_names:
        _fail_if_collective(
            fold_state_names=fold_state_names, stakeholders=stakeholders
        )
        mapped = _wrap_with_fold_reduction(
            mapped=cast("Callable[..., FloatND]", mapped),
            fold_state_names=fold_state_names,
            fold_weights=fold_weights,
            fold_conditioning=fold_conditioning,
            inner_state_names=inner_state_names,
            action_names=action_names,
            state_names=state_names,
            extra_param_names=[*extra_param_names, "_lcm_action_block_width"],
        )
    if not co_map_state_names:
        return cast("MaxQOverAFunction", mapped)

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
        )
    return cast("MaxQOverAFunction", allow_only_kwargs(func=mapped, enforce=False))


def _fail_if_full_V_streaming_route_is_unsupported(
    *,
    has_taste_shocks: bool,
    stakeholders: tuple[str, ...] | None,
    pareto_weights: ParetoWeights | None,
    fold_state_names: tuple[StateName, ...],
) -> None:
    """Reject routes unsupported by full-value action streaming."""
    if has_taste_shocks and stakeholders is not None:
        raise NotImplementedError(
            "Full-V action streaming does not support collective EV1 regimes."
        )
    if has_taste_shocks and fold_state_names:
        raise NotImplementedError(
            "Full-V action streaming does not support EV1 taste shocks "
            "with fold states."
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


def _wrap_with_fold_reduction(
    *,
    mapped: Callable[..., FloatND],
    fold_state_names: tuple[StateName, ...],
    fold_weights: Mapping[StateName, FloatND],
    fold_conditioning: Mapping[StateName, StateName],
    inner_state_names: tuple[StateName, ...],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    extra_param_names: list[str],
) -> Callable[..., FloatND]:
    """Wrap the (still fold-axis-carrying) inner productmap with the fold average.

    `mapped`'s output axes are exactly `inner_state_names`, in order (the
    `productmap`'s `variables` order) — this runs BEFORE any co-map wrapping,
    so no co-map axis is present yet. Fold axes are reduced from the highest
    inner-position down, so removing one axis never shifts the position of a
    not-yet-reduced one. The wrapper redeclares the post-`productmap` keyword-only call
    interface with `with_signature`, so `allow_args` can adapt it safely for the
    co-map `vmap_1d` wrapping that may follow. On a streamed route,
    `extra_param_names` also carries the planner-bound action-width keyword;
    the wrapper forwards it unchanged to the mapped action reducer.

    A folded state named in `fold_conditioning` averages against a different row
    per category of its conditioning state, so its weights are broadcast to the
    value array with the category dimension on that state's own axis. Reducing
    one fold axis shifts the position of every later axis, so each axis pair is
    resolved against the axis order as it stands at that step, not the original.

    `mapped` is a singleton regime's value kernel: a collective regime may not
    declare a folded state at all, so there is no stakeholder axis here and no
    dissolution flag to reduce (`_fail_if_collective`).

    The per-axis reducer is bound HERE, at kernel-build time, from the axis's
    own quadrature weights — see `_select_fold_reducer`.
    """
    _fail_if_conditioning_state_not_inner(
        fold_conditioning=fold_conditioning, inner_state_names=inner_state_names
    )
    fold_steps = _plan_fold_steps(
        fold_state_names=fold_state_names,
        fold_conditioning=fold_conditioning,
        inner_state_names=inner_state_names,
    )
    fold_reducers = {
        name: _select_fold_reducer(weight=fold_weights[name], name=name)
        for name in fold_state_names
    }

    @with_signature(
        kwargs=[
            "next_regime_to_V_arr",
            *action_names,
            *state_names,
            *extra_param_names,
        ],
        return_annotation="FloatND",
        enforce=False,
    )
    def folded(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: _ParamsLeaf,
    ) -> FloatND:
        V_arr = mapped(
            next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params
        )
        for name, axis, conditioning_axis in fold_steps:
            weights = fold_weights[name]
            if conditioning_axis is not None:
                weights = _broadcast_conditioned_rows(
                    rows=weights,
                    conditioning_axis=conditioning_axis,
                    fold_axis=axis,
                    shape=V_arr.shape,
                )
            V_arr = fold_reducers[name](a=V_arr, axis=axis, weights=weights)
        return V_arr

    return folded


def _plan_fold_steps(
    *,
    fold_state_names: tuple[StateName, ...],
    fold_conditioning: Mapping[StateName, StateName],
    inner_state_names: tuple[StateName, ...],
) -> tuple[tuple[StateName, int, int | None], ...]:
    """Resolve `(folded state, its axis, its conditioning axis)` per reduction step.

    Axes are resolved against the axis order as it stands when that step runs,
    because each reduction removes one axis and shifts everything after it. The
    order is highest fold axis first, so a fold axis's own position is stable,
    but a conditioning axis sitting after an already-reduced fold axis is not.
    """
    remaining = list(inner_state_names)
    ordered = sorted(fold_state_names, key=remaining.index, reverse=True)
    steps: list[tuple[StateName, int, int | None]] = []
    for name in ordered:
        conditioning_state = fold_conditioning.get(name)
        steps.append(
            (
                name,
                remaining.index(name),
                None
                if conditioning_state is None
                else remaining.index(conditioning_state),
            )
        )
        remaining.remove(name)
    return tuple(steps)


def _broadcast_conditioned_rows(
    *,
    rows: FloatND,
    conditioning_axis: int,
    fold_axis: int,
    shape: tuple[int, ...],
) -> FloatND:
    """Spread one quadrature row per category over the value array's full shape.

    `rows` is `(n_categories, n_points)`. The reduction kernels take weights
    either 1-D along the reduced axis or the same shape as the value array, and
    a conditioned row set is neither, so it is placed on its two axes and
    broadcast. Under `jit` the broadcast is a view XLA fuses into the reduction;
    the eager path materializes it, which is the same size the unfolded model
    would have carried as an axis anyway.
    """
    placed = rows if conditioning_axis < fold_axis else rows.T
    target = [1] * len(shape)
    target[conditioning_axis] = rows.shape[0]
    target[fold_axis] = rows.shape[1]
    return jnp.broadcast_to(placed.reshape(target), shape)


def _fail_if_conditioning_state_not_inner(
    *,
    fold_conditioning: Mapping[StateName, StateName],
    inner_state_names: tuple[StateName, ...],
) -> None:
    """Refuse a conditioned fold whose conditioning state has no inner axis.

    The reduction gathers a category's row along the conditioning state's own
    productmap axis, so a conditioning state that was co-mapped away (a fixed
    distributed state) or pruned leaves the fold nothing to index by.
    """
    missing = sorted(
        f"'{name}' conditioned on '{conditioning_state}'"
        for name, conditioning_state in fold_conditioning.items()
        if conditioning_state not in inner_state_names
    )
    if missing:
        msg = (
            f"Folded state(s) {missing} condition their quadrature on a state "
            "that is not an inner productmap axis of the value kernel. The fold "
            "selects a category's quadrature row along that state's own axis, "
            "so the conditioning state must be an ordinary (non-distributed) "
            f"state of the same regime. Inner axes are {list(inner_state_names)}."
        )
        raise ValueError(msg)


def _fail_if_collective(
    *, fold_state_names: tuple[StateName, ...], stakeholders: tuple[str, ...] | None
) -> None:
    """Refuse to build a fold reduction for a collective regime.

    A collective regime's value carries a dissolution flag beside it: where no
    action satisfies every stakeholder's participation constraint the cell is
    flagged and written `-inf`, a not-sustainable sentinel a gated edge
    resolves to the outside option. Quadrature cannot average a sentinel, so
    the combination is refused at model build by
    `_fail_if_collective_regime_folds`. Reaching here means that guarantee was
    bypassed.
    """
    if stakeholders is not None:
        msg = (
            f"fold=True on state(s) {sorted(fold_state_names)} reached the "
            f"value kernel of a collective regime (stakeholders "
            f"{list(stakeholders)}). Folding a shock out of a collective "
            "regime's stored value is not supported — the combination is "
            "rejected at model build, so reaching here means that rejection "
            "was bypassed."
        )
        raise ValueError(msg)


def _select_fold_reducer(*, weight: FloatND, name: StateName) -> Callable[..., FloatND]:
    """Pick the weighted-average kernel for ONE fold axis, at kernel-build time.

    A fold axis's weights are the folded process's own quadrature marginal.
    `_validate_fold_declarations` rejects a runtime-parameterized process, so
    `solvers.py` computes them ONCE as a concrete constant before the core is
    ever traced — which makes this a plain Python branch on a concrete value,
    not a traced predicate. (A `jax.lax.cond` could not do this job: its
    predicate would be a single GLOBAL runtime value, unable to select
    per-axis at trace time.)

    Why branch at all: `zero_safe_average` exists so a zero-weight node next
    to an admissible on-path `-inf` cannot inject a `nan` via `0 * -inf`. It
    buys that with a per-term `jnp.where`, which blocks XLA from contracting
    the `multiply` into the reduction's FMA — so it rounds twice where
    `jnp.average` rounds once, and the two can differ by ~1 ULP (see
    `zero_safe.zero_safe_average`). On an axis whose weights are ALL strictly
    positive the guard protects against nothing, yet still costs that extra
    rounding, which is enough to make the fold non-exact against the
    unfolded-then-averaged oracle on the non-jitted path.

    The motivation is EXACTNESS, not speed. MEASURED (jax 0.10.1, CPU,
    float64, 1e6x4): with the weights closed over as constants — which is
    exactly how the fold calls it — the guard costs 1.04x, since XLA
    constant-folds the `select` away; 1.20x for traced weights. Only the
    EAGER path pays materially (5.3x), because nothing folds the `select`
    there. So this branch buys real time only for `enable_jit=False`.

    CAVEAT. Selecting `jnp.average` restores bit-exactness against the
    unfolded-then-averaged oracle on the NON-JITTED path, but it does NOT make
    the JITTED fold bit-identical to that oracle: under `jit` the fold's
    average is fused into the mapped value kernel while the oracle averages a
    materialized unfolded array, so XLA may reassociate the two reductions
    differently. The jitted contract is therefore numerical equivalence within
    a SCALE-AWARE tolerance, not bit-identity, and NOT a fixed ULP count: ULP
    is a result-space spacing metric and is unstable near CANCELLATION. A
    supported 18-node uniform-IID float32 fold has a fused value and a
    materialized oracle differing by only ~2.62e-7 in absolute terms (a
    summand-scale float32 reduction floor) yet 287,557 ULP in the small
    (~1e-5) cancelled result, so a `<= 2 * spacing(oracle)` claim is false
    there. The honest contract is `|fold - oracle| <= atol + rtol *
    max|summand|` (see `test_fold_jitted_matches_unfolded_then_averaged_to_
    summand_scale_tolerance`). Do not describe the jitted fold as bit-identical
    or few-ULP. This branch is still correct and necessary: it removes the
    LARGER `zero_safe_average` drift on all-positive axes and keeps the
    zero-weight guard where a zero can occur.

    This is a per-AXIS decision on that axis's own weights, so a model that
    folds one zero-weight axis and one all-positive axis gets the right
    kernel for each.
    """
    weight_arr = jnp.asarray(weight)
    try:
        has_zero = bool(jnp.any(weight_arr == 0))
    except jax.errors.ConcretizationTypeError as exc:
        msg = (
            f"Fold weights for state '{name}' are not concrete at kernel-build "
            "time. The fold reduction picks its weighted-average kernel from "
            "the quadrature weights themselves, which requires them to be "
            "known before the solve core is traced; "
            "`_validate_fold_declarations` is supposed to guarantee this by "
            "rejecting a fold on a process with runtime-supplied distribution "
            "parameters. Reaching this means that guarantee was bypassed."
        )
        raise ValueError(msg) from exc
    if not has_zero:
        return jnp.average
    # `shifts` is bound here rather than at the two call sites because only this
    # function knows which kernel it handed back, and `jnp.average` has no such
    # parameter. `None` is the answer for a fold axis: its weights are the folded
    # process's own quadrature marginal, which never passed through
    # `scaled_joint_weight` and so carries no base-two scale to reconcile.
    return functools.partial(zero_safe_average, shifts=None)


def get_argmax_and_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
    stakeholders: tuple[str, ...] | None = None,
    pareto_weights: ParetoWeights | None = None,
) -> ArgmaxQOverAFunction:
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

    Q_and_F = productmap(
        func=Q_and_F,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )

    if has_taste_shocks:

        @with_signature(
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
        def argmax_and_max_Q_over_a(
            next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
            **states_actions_params: _ParamsLeaf,
        ) -> tuple[IntND, FloatND]:
            taste_shock_key = cast(
                "Array", states_actions_params.pop("taste_shock_key")
            )
            Q_arr, F_arr = Q_and_F(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **states_actions_params,
            )
            Q_masked = jnp.where(F_arr, Q_arr, -jnp.inf)
            n_discrete_cells = math.prod(Q_arr.shape[:n_discrete_action_axes])
            n_continuous_cells = math.prod(Q_arr.shape[n_discrete_action_axes:])
            Q_flat = Q_masked.reshape(n_discrete_cells, n_continuous_cells)
            continuous_argmax = jnp.argmax(Q_flat, axis=1)
            Qc = Q_flat.max(axis=1)
            scale = cast("FloatND", states_actions_params[TASTE_SHOCK_SCALE_PARAM])
            noise = draw_taste_shock_noise(
                key=taste_shock_key, shape=Qc.shape, scale=scale
            )
            # An infeasible discrete cell stays infeasible: the noise is
            # finite, so `-inf + noise` is still `-inf`.
            noisy_Qc = Qc + noise
            discrete_argmax = jnp.argmax(noisy_Qc)
            flat_index = (
                discrete_argmax * n_continuous_cells
                + continuous_argmax[discrete_argmax]
            )
            return flat_index.astype(jnp.int32), Qc[discrete_argmax]

    else:

        @with_signature(
            args=[
                "next_regime_to_V_arr",
                *action_names,
                *state_names,
                *extra_param_names,
            ],
            return_annotation="tuple[IntND, FloatND]",
            enforce=False,
        )
        def argmax_and_max_Q_over_a(
            next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
            **states_actions_params: _ParamsLeaf,
        ) -> tuple[IntND, FloatND]:
            Q_arr, F_arr = Q_and_F(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **states_actions_params,
            )
            if stakeholders is not None:
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
                    name: Q_arr[..., index] for index, name in enumerate(stakeholders)
                }
                argmax_flat, values, _dissolution = collective_argmax_and_readout(
                    stakeholder_Q=stakeholder_Q,
                    feasibility=F_arr,
                    weights=_evaluate_pareto_weights(
                        pareto_weights=pareto_weights,
                        states_actions_params=states_actions_params,
                    ),
                    action_axes=action_axes,
                )
                V_stacked = jnp.stack([values[name] for name in stakeholders], axis=-1)
                return argmax_flat, V_stacked
            return argmax_and_max(a=Q_arr, where=F_arr, initial=-jnp.inf)

    return argmax_and_max_Q_over_a


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

    Args:
        key: JAX PRNG key for the Gumbel draw.
        shape: Shape of the noise array (one draw per discrete-action cell).
        scale: Taste-shock scale; broadcasts against the draw.

    Returns:
        Mean-zero additive noise of the given shape.

    """
    return scale * (jax.random.gumbel(key, shape) - EULER_GAMMA)


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
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
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
