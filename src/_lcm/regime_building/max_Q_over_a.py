import inspect
import math
from collections.abc import Callable
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from dags import with_signature
from jax import Array

from _lcm.logsum import EULER_GAMMA, logsum_and_softmax
from _lcm.regime_building.argmax import argmax_and_max
from _lcm.typing import (
    ActionName,
    ArgmaxQOverAFunction,
    MaxQOverAFunction,
    RegimeName,
    StateName,
)
from _lcm.utils.dispatchers import productmap, vmap_1d
from _lcm.utils.functools import allow_args, allow_only_kwargs
from lcm.typing import BoolND, FloatND, IntND

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

    Returns:
        V, i.e., the function that calculates the maximum of the Q-function over all
        feasible actions.

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
            **states_actions_params: FloatND | IntND | BoolND,
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
                scale=states_actions_params[TASTE_SHOCK_SCALE_PARAM],
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
            return_annotation="FloatND",
            enforce=False,
        )
        def max_Q_over_a(
            next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
            **states_actions_params: FloatND | IntND | BoolND,
        ) -> FloatND:
            Q_arr, F_arr = Q_and_F(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **states_actions_params,
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


def get_argmax_and_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
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
            **states_actions_params: FloatND | IntND | BoolND,
        ) -> tuple[IntND, FloatND]:
            taste_shock_key = states_actions_params.pop("taste_shock_key")
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
            scale = states_actions_params[TASTE_SHOCK_SCALE_PARAM]
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
            **states_actions_params: FloatND | IntND | BoolND,
        ) -> tuple[IntND, FloatND]:
            Q_arr, F_arr = Q_and_F(
                next_regime_to_V_arr=next_regime_to_V_arr,
                **states_actions_params,
            )
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
