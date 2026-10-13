"""Reduce over actions for a regime that chooses with one aggregator and stores another.

A `DecisionValue` Koopmans aggregator gives each cell two profiles from one
kernel evaluation: `Q_decision`, which the action maximizes, and `Q`, which is
stored and published at that action. The reductions here mirror the dense ones
in `max_Q_over_a` for that pair:

- `argmax_and_readout` is the reduction itself, at one cell;
- `get_decision_value_Q_over_a` is the reduction over the action product, as
  simulation and policy lookup run it;
- `get_decision_value_max_Q_over_a` maps it over the state product, with the
  same tiling and fold `get_max_Q_over_a` takes, as the solve runs it.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, cast

import jax.numpy as jnp
from dags import with_signature

from _lcm.regime_building.argmax import argmax_and_max
from _lcm.regime_building.max_Q_over_a import (
    _get_extra_param_names,
    _wrap_with_fold_reduction,
)
from _lcm.regime_building.Q_and_F import QAndFParts, get_Q_and_F_parts
from _lcm.simulation.program_types import (
    ACTION_INDEX,
    ACTION_PARTS_ROLES,
    DECISION_PROFILE,
    DECISION_VALUE,
)
from _lcm.typing import (
    ActionName,
    MaxQOverAFunction,
    QAndFArg,
    RegimeName,
    StateName,
)
from _lcm.utils.dispatchers import productmap, tiled_productmap
from lcm.typing import BoolND, FloatND, IntND


def argmax_and_readout(
    *, selection: FloatND, readout: FloatND, where: BoolND
) -> tuple[IntND, FloatND]:
    """Choose the first feasible maximizer of `selection` and read `readout` there.

    Both arrays share one shape and are reduced over all axes:

    - the index is the flat index of the first feasible maximizer of `selection`;
    - the value is `readout` at that index;
    - no feasible element gives index `-1` and value `-inf`;
    - a NaN at any feasible element, in either array, makes the value NaN.

    Args:
        selection: The values the element is chosen on.
        readout: The values published at the chosen element.
        where: Feasibility of each element.

    Returns:
        The flat index of the chosen element and the readout there.

    """
    index, _ = argmax_and_max(a=selection, where=where, initial=-jnp.inf)
    any_feasible = jnp.any(where)
    feasible_nan = jnp.any(where & (jnp.isnan(selection) | jnp.isnan(readout)))
    value = jnp.where(any_feasible, jnp.ravel(readout)[index], -jnp.inf)
    return (
        jnp.where(any_feasible, index, -1).astype(jnp.int32),
        jnp.where(feasible_nan, jnp.nan, value),
    )


def get_decision_value_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    publishes_index: bool,
) -> Callable[..., tuple[IntND, FloatND] | tuple[FloatND, BoolND]]:
    """Get the reduction over actions of a `DecisionValue` regime, at one cell.

    Args:
        Q_and_F: A kernel built by `get_Q_and_F` with a decision aggregator.
        action_names: Tuple of action names, discrete first.
        state_names: Tuple of state names.
        publishes_index: Whether to return `(index, value)`, as simulation
            needs, instead of `(value, empty)`, as the counted solve needs.

    Returns:
        The reduction, taking the kernel's arguments.

    """
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
    )
    parts = productmap(
        func=get_Q_and_F_parts(Q_and_F=Q_and_F),
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )
    return with_signature(
        _DecisionValueQOverA(parts=parts, publishes_index=publishes_index),
        args=["next_regime_to_V_arr", *action_names, *state_names, *extra_param_names],
        return_annotation=(
            "tuple[IntND, FloatND]" if publishes_index else "tuple[FloatND, BoolND]"
        ),
        enforce=False,
    )


def get_decision_value_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    batch_sizes: dict[StateName, int],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    cell_width_keyword: str | None = None,
    untiled_state_names: tuple[StateName, ...] = (),
    broadcast_state_names: tuple[StateName, ...] = (),
    fold_state_names: tuple[StateName, ...] = (),
    fold_weights: Mapping[StateName, FloatND] = MappingProxyType({}),
    fold_conditioning: Mapping[StateName, StateName] = MappingProxyType({}),
) -> MaxQOverAFunction:
    """Get the solve's value function of a `DecisionValue` regime.

    Maps `get_decision_value_Q_over_a` over the state product, then folds
    exactly as `get_max_Q_over_a` does, and returns `(V, empty)` with `empty`
    marking the cells without a feasible action.
    """
    cell = get_decision_value_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=action_names,
        state_names=state_names,
        publishes_index=False,
    )
    mapped = (
        tiled_productmap(
            func=cell,
            variables=state_names,
            width_keyword=cell_width_keyword,
            untiled_variables=untiled_state_names,
            broadcast_variables=broadcast_state_names,
        )
        if cell_width_keyword is not None
        else productmap(
            func=cell,
            variables=state_names,
            batch_sizes={name: batch_sizes[name] for name in state_names},
        )
    )
    if fold_state_names:
        mapped = _wrap_with_fold_reduction(
            mapped=cast("Callable[..., FloatND | tuple[FloatND, BoolND]]", mapped),
            fold_state_names=fold_state_names,
            fold_weights=fold_weights,
            fold_conditioning=fold_conditioning,
            inner_state_names=state_names,
            return_feasibility=True,
            action_names=action_names,
            state_names=state_names,
            extra_param_names=[
                *_get_extra_param_names(
                    Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
                ),
                *((cell_width_keyword,) if cell_width_keyword is not None else ()),
            ],
        )
    return cast("MaxQOverAFunction", mapped)


@dataclass(frozen=True, kw_only=True, eq=False)
class _DecisionValueQOverA:
    """The stored value at the decision's choice, at one state cell.

    The caller wraps it in `dags.with_signature` to advertise the arguments the
    state productmap binds by name.
    """

    __name__: ClassVar[str] = "decision_value_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    parts: Callable[..., QAndFParts]
    """The regime's kernel parts, mapped over the action product."""

    publishes_index: bool
    """Whether to return the chosen flat index in place of the empty flag."""

    def __call__(
        self,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> tuple[IntND, FloatND] | tuple[FloatND, BoolND]:
        """Return `(index, value)` or `(value, empty)` at this cell."""
        parts = self.parts(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        index, value = argmax_and_readout(
            selection=cast("FloatND", parts.Q_decision),
            readout=parts.Q,
            where=parts.F,
        )
        if self.publishes_index:
            return index, value
        return value, ~jnp.any(parts.F)


# What the dense lookup of a `DecisionValue` regime publishes, in the order its
# body returns them: the decision's pair, the kernel's parts over the action
# product, then the decision profile.
DECISION_VALUE_LOOKUP_OUTPUT_ROLES = (
    ACTION_INDEX,
    DECISION_VALUE,
    *ACTION_PARTS_ROLES,
    DECISION_PROFILE,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class DecisionValueWithActionParts:
    """The dense lookup decision of a `DecisionValue` regime at one state cell.

    Returns the chosen flat index and the stored value there, followed by `Q`,
    `F`, `U`, `CE` and `Q_decision` over the action product, all from one
    evaluation of the regime's kernel.
    """

    __name__: ClassVar[str] = "decision_value_Q_over_a_with_action_parts"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    parts: Callable[..., QAndFParts]
    """The kernel's parts, mapped over the action product."""

    def __call__(
        self, **kwargs: QAndFArg
    ) -> tuple[IntND, FloatND, FloatND, BoolND, FloatND, FloatND, FloatND]:
        """Return the index, the value, and `Q`, `F`, `U`, `CE`, `Q_decision`."""
        parts = self.parts(**kwargs)
        Q_decision = cast("FloatND", parts.Q_decision)
        index, value = argmax_and_readout(
            selection=Q_decision, readout=parts.Q, where=parts.F
        )
        return index, value, parts.Q, parts.F, parts.U, parts.CE, Q_decision
