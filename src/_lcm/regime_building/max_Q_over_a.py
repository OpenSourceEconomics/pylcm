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
    collective_argmax_and_readout,
    collective_readout,
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
    weights: Mapping[str, float] | None = None,
    fold_state_names: tuple[StateName, ...] = (),
    fold_weights: Mapping[StateName, FloatND] = MappingProxyType({}),
) -> MaxQOverAFunction:
    r"""Get the function returning the maximum of Q over all actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  H(U(x, a), \mathbb{E}[V(x', a') | x, a]),
    ```
    with $H(U, v) = u + \beta \cdot v$ as the leading case (which is the only one that
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
            array plus the boolean divorce flag `D = 1[mask empty]` on the state
            axes (E2; distinct from a numeric `-inf`, which occurs on-path).
        weights: Household Pareto weights per stakeholder; required (and only used)
            when `stakeholders` is set.
        fold_state_names: IID-process states declared `fold=True`, or empty (the
            default). Each is still an ordinary inner (non-co-mapped) productmap
            axis THROUGH the max-over-actions / collective readout — every node is
            evaluated exactly as today — but its axis is then weighted-averaged
            away (`jnp.average(..., weights=fold_weights[name])`) before the
            result is returned, so the caller never sees it. A collective
            regime's divorce flag `D` is folded by `jnp.any` instead (stays
            strictly boolean; see `_wrap_with_fold_reduction`).
        fold_weights: Quadrature weights per name in `fold_state_names` (each a
            1-D array matching that state's node count, summing to 1). Ignored
            when `fold_state_names` is empty.

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
                # COLLECTIVE-REGIMES (E1 + E2): Q_arr carries a trailing
                # stakeholder axis (the action product-map keeps it last);
                # F_arr does not — for an E2 regime it already includes the
                # value constraints, which Q_and_F ANDed in AFTER computing
                # Q^s. Split Q_arr per stakeholder, take the household argmax
                # of the scalarization over the masked action axes, and read
                # off each stakeholder's OWN value at that shared argmax. The
                # returned pair is the stakeholder value vector (re-stacked on
                # a trailing axis, which the outer state product-map turns
                # into `(*states, n_stakeholders)`) plus the divorce flag D —
                # `True` where NO action is feasible (empty mask), published
                # alongside V and never conflated with a numeric -inf value
                # (which occurs on-path).
                action_axes = tuple(range(F_arr.ndim))
                stakeholder_Q = {
                    name: Q_arr[..., index] for index, name in enumerate(stakeholders)
                }
                values, divorce = collective_readout(
                    stakeholder_Q=stakeholder_Q,
                    feasibility=F_arr,
                    weights=cast("Mapping[str, float]", weights),
                    action_axes=action_axes,
                )
                return (
                    jnp.stack([values[name] for name in stakeholders], axis=-1),
                    divorce,
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
        mapped = _wrap_with_fold_reduction(
            mapped,
            fold_state_names=fold_state_names,
            fold_weights=fold_weights,
            inner_state_names=inner_state_names,
            action_names=action_names,
            state_names=state_names,
            extra_param_names=extra_param_names,
            stakeholders=stakeholders,
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
    return cast("MaxQOverAFunction", allow_only_kwargs(mapped, enforce=False))


def _wrap_with_fold_reduction(
    mapped: Callable[..., FloatND | tuple[FloatND, BoolND]],
    *,
    fold_state_names: tuple[StateName, ...],
    fold_weights: Mapping[StateName, FloatND],
    inner_state_names: tuple[StateName, ...],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    extra_param_names: list[str],
    stakeholders: tuple[str, ...] | None,
) -> Callable[..., FloatND | tuple[FloatND, BoolND]]:
    """Wrap the (still fold-axis-carrying) inner productmap with the fold average.

    `mapped`'s output axes are exactly `inner_state_names`, in order (the
    `productmap`'s `variables` order) — this runs BEFORE any co-map wrapping,
    so no co-map axis is present yet. A collective `mapped` additionally
    carries a TRAILING stakeholder axis on `V` only (not on `D`); since it is
    trailing, it never shifts a fold axis's position. Fold axes are reduced
    from the highest inner-position down, so removing one axis never shifts
    the position of a not-yet-reduced one. The wrapper re-declares EXACTLY
    `mapped`'s own call signature (`with_signature`, matching
    `max_Q_over_a`'s pre-productmap signature, which `productmap` preserves)
    so it composes transparently with the co-map `vmap_1d` wrapping that may
    follow.

    The divorce flag `D` (collective only) stays strictly boolean — reduced
    by `jnp.any` ("divorced at any folded node"), not a weighted average — so
    it keeps its `BoolND` contract for every downstream reader (the gated-edge
    fold, `KernelResult.divorce`). This is a conservative, not an exact,
    reduction; `_validate_fold_declarations` already rejects a fold state a
    same-period gate reads DIRECTLY, but a gate reading `D` itself (of a
    regime that happens to also fold an UNRELATED state) is not caught — out
    of scope for this slice.
    """
    fold_axis_positions = sorted(
        ((name, inner_state_names.index(name)) for name in fold_state_names),
        key=lambda item: -item[1],
    )

    @with_signature(
        args=["next_regime_to_V_arr", *action_names, *state_names, *extra_param_names],
        return_annotation=(
            "tuple[FloatND, BoolND]" if stakeholders is not None else "FloatND"
        ),
        enforce=False,
    )
    def folded(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: _ParamsLeaf,
    ) -> FloatND | tuple[FloatND, BoolND]:
        out = mapped(next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params)
        if stakeholders is not None:
            V_arr, divorce = cast("tuple[FloatND, BoolND]", out)
            for name, axis in fold_axis_positions:
                V_arr = jnp.average(V_arr, axis=axis, weights=fold_weights[name])
                divorce = jnp.any(divorce, axis=axis)
            return V_arr, divorce
        V_arr = cast("FloatND", out)
        for name, axis in fold_axis_positions:
            V_arr = jnp.average(V_arr, axis=axis, weights=fold_weights[name])
        return V_arr

    return folded


def get_argmax_and_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
    stakeholders: tuple[str, ...] | None = None,
    weights: Mapping[str, float] | None = None,
) -> ArgmaxQOverAFunction:
    r"""Get the function returning the arguments maximizing Q over all actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  H(U(x, a), \mathbb{E}[V(x', a') | x, a]),
    ```
    with $H(U, v) = u + \beta \cdot v$ as the leading case (which is the only one that
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
            `None` (the singleton default). COLLECTIVE-REGIMES (E4): when
            set, `Q_and_F` returns a stacked per-stakeholder `Q` (trailing
            stakeholder axis); the household argmax of the weighted
            scalarization is computed once (`collective_argmax_and_readout`)
            and each stakeholder's own value is gathered at that shared
            index — mirrors the solve-side `get_max_Q_over_a` collective
            branch so simulate recomputes the identical argmax. Mutually
            exclusive with `has_taste_shocks` (rejected at regime
            construction for collective regimes).
        weights: Household Pareto weights per stakeholder; required (and only
            used) when `stakeholders` is set.

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
                # COLLECTIVE-REGIMES (E4): mirrors the solve-side collective
                # branch in `get_max_Q_over_a` — split the stacked Q by
                # stakeholder, argmax the household scalarization once over
                # the (value-masked, per E2) feasible action set, and gather
                # each stakeholder's OWN value at that shared index. The
                # simulate-only addition vs. the solve readout
                # (`collective_readout`) is the argmax index itself, needed
                # to look up which JOINT action both stakeholders actually
                # took (`_lookup_values_from_indices` in `simulate.py`).
                action_axes = tuple(range(F_arr.ndim))
                stakeholder_Q = {
                    name: Q_arr[..., index] for index, name in enumerate(stakeholders)
                }
                argmax_flat, values, _divorce = collective_argmax_and_readout(
                    stakeholder_Q=stakeholder_Q,
                    feasibility=F_arr,
                    weights=cast("Mapping[str, float]", weights),
                    action_axes=action_axes,
                )
                V_stacked = jnp.stack([values[name] for name in stakeholders], axis=-1)
                return argmax_flat, V_stacked
            return argmax_and_max(Q_arr, where=F_arr, initial=-jnp.inf)

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
