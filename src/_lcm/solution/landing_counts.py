"""Count a GridSearch solve's raw raise-rule landings in the pass that maximizes.

A regime whose value constraints or gated edges decide, only once values are
known, which actions and branches reach a landing has its `Q_and_F` return the
raw `off_grid="raise"` landing counts of every action as a third output. The
builders here carry those counts through the same reductions the value takes:

- over the actions, summed where the feasibility mask keeps an action, value
  constraints included;
- through the fold and the state product, alongside the value;
- over every state cell, into one saturating int32 total per check, which the period
  kernel reads and raises on.

Each wrapper evaluates the counting `Q_and_F` once and hands the value
reduction it wraps the `(Q, F)` it would have computed itself, so that
reduction is the unchanged dense one a regime without counts runs.
"""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, cast

import jax
import jax.numpy as jnp
from dags import with_signature

from _lcm.off_grid import off_grid_message
from _lcm.regime_building.collective import ParetoWeights
from _lcm.regime_building.max_Q_over_a import (
    _fail_if_co_map_states_not_leading,
    _fail_if_collective,
    _get_extra_param_names,
    _HardMaxQOverA,
    _SmoothedMaxQOverA,
    _wrap_with_fold_reduction,
)
from _lcm.regime_building.Q_and_F import BoundedQAndF, runtime_bound_specs
from _lcm.solution.feasibility import (
    BoundedResult,
    CountedGridSearch,
    CountedGridSearchOutput,
)
from _lcm.typing import (
    ActionName,
    ArrayTree,
    MaxQOverAFunction,
    QAndFArg,
    QAndFFunction,
    RegimeName,
    StateName,
)
from _lcm.utils.counts import sum_counts
from _lcm.utils.dispatchers import productmap, tiled_productmap, vmap_1d
from _lcm.utils.functools import allow_args, allow_only_kwargs
from lcm.exceptions import OffGridStateError
from lcm.typing import BoolND, FloatND, IntND

type LandingCheck = tuple[RegimeName, StateName]


def landing_checks_of(Q_and_F: QAndFFunction) -> tuple[LandingCheck, ...]:
    """Return the `(target, state)` landings a kernel counts, in output order."""
    return getattr(Q_and_F, "landing_checks", ())


def get_landing_counting_max_Q_over_a(
    *,
    Q_and_F: QAndFFunction,
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
    cell_width_keyword: str | None = None,
    untiled_state_names: tuple[StateName, ...] = (),
    broadcast_state_names: tuple[StateName, ...] = (),
    return_feasibility: bool = True,
) -> MaxQOverAFunction:
    """Build `get_max_Q_over_a`'s dense reduction for a landing-counting kernel.

    The arguments and the composition are `get_max_Q_over_a`'s: the action
    product, the hard or EV1-smoothed reducer, the state product, the fold and
    the co-map. The reducer and the fold are wrapped so the landing counts ride
    beside the value, and the result returns them after the feasibility output.

    Raises:
        ValueError: `return_feasibility` is off; only the solve, which always
            returns feasibility, counts landings.

    """
    if not return_feasibility:
        msg = "A landing-counting reduction returns feasibility beside the value."
        raise ValueError(msg)
    _fail_if_co_map_states_not_leading(
        state_names=state_names, co_map_state_names=co_map_state_names
    )
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
    )
    if pareto_weights is not None:
        extra_param_names = list(
            dict.fromkeys((*extra_param_names, *pareto_weights.param_names))
        )
    bounded = bool(runtime_bound_specs(function=Q_and_F))
    action_mapped = productmap(
        func=BoundedQAndF(function=Q_and_F, landings=True) if bounded else Q_and_F,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
        bound_summary=bounded,
    )
    reducer = (
        _SmoothedMaxQOverA(
            Q_and_F=action_mapped,
            n_discrete_action_axes=n_discrete_action_axes,
            return_feasibility=True,
        )
        if has_taste_shocks
        else _HardMaxQOverA(
            Q_and_F=action_mapped,
            stakeholders=stakeholders,
            pareto_weights=pareto_weights,
            return_feasibility=True,
        )
    )
    max_Q_over_a = with_signature(
        _LandingCountingReduction(reducer=reducer),
        args=["next_regime_to_V_arr", *action_names, *state_names, *extra_param_names],
        return_annotation="tuple[FloatND, BoolND, IntND]",
        enforce=False,
    )
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
            bound_summary=bounded,
        )
        if cell_width_keyword is not None
        else productmap(
            func=max_Q_over_a,
            variables=inner_state_names,
            batch_sizes={name: batch_sizes[name] for name in inner_state_names},
            bound_summary=bounded,
        )
    )
    if fold_state_names:
        _fail_if_collective(
            fold_state_names=fold_state_names, stakeholders=stakeholders
        )
        extra_fold_names = [
            *extra_param_names,
            *((cell_width_keyword,) if cell_width_keyword is not None else ()),
        ]
        folded = _wrap_with_fold_reduction(
            mapped=cast("Callable[..., FloatND | tuple[FloatND, BoolND]]", mapped),
            fold_state_names=fold_state_names,
            fold_weights=fold_weights,
            fold_conditioning=fold_conditioning,
            inner_state_names=inner_state_names,
            return_feasibility=True,
            action_names=action_names,
            state_names=state_names,
            extra_param_names=extra_fold_names,
        )
        mapped = with_signature(
            _LandingCountingFold(fold=folded.__wrapped__),  # ty: ignore[unresolved-attribute]
            kwargs=[
                "next_regime_to_V_arr",
                *action_names,
                *state_names,
                *extra_fold_names,
            ],
            return_annotation="tuple[FloatND, BoolND, IntND]",
            enforce=False,
        )
    if not co_map_state_names:
        return cast("MaxQOverAFunction", mapped)
    mapped = allow_args(mapped)
    for state_name, v_arr_in_axes in zip(
        reversed(co_map_state_names), reversed(co_map_v_arr_in_axes), strict=True
    ):
        mapped = vmap_1d(
            func=mapped,
            variables=(state_name,),
            co_mapped_in_axes=MappingProxyType({"next_regime_to_V_arr": v_arr_in_axes}),
            callable_with="only_args",
            bound_summary=bounded,
        )
    return cast("MaxQOverAFunction", allow_only_kwargs(func=mapped, enforce=False))


@dataclass(frozen=True, kw_only=True, eq=False)
class LandingCountedGridSearch:
    """Total a counting reduction's landings over the state cells, after its counts."""

    counted: CountedGridSearch
    """The feasibility summary, whose reduction also returns landing counts."""

    def __call__(
        self, **kwargs: QAndFArg
    ) -> tuple[FloatND | BoolND | IntND, ...] | BoundedResult[ArrayTree]:
        """Return the feasibility summary's outputs, then the landing totals."""
        evaluated = self.counted.function(**kwargs)
        value, empty, landing_counts = cast(
            "tuple[FloatND, BoolND, IntND]",
            evaluated.output if isinstance(evaluated, BoundedResult) else evaluated,
        )
        summary = dataclasses.replace(
            self.counted, function=_Evaluated(outputs=(value, empty))
        )(**kwargs)
        output = (
            *cast("CountedGridSearchOutput", summary),
            sum_counts(
                counts=landing_counts, axis=tuple(range(landing_counts.ndim - 1))
            ),
        )
        return (
            BoundedResult(output=output, bound_status=evaluated.bound_status)
            if isinstance(evaluated, BoundedResult)
            else output
        )


def fail_on_reached_landings(
    *,
    totals: IntND,
    checks: tuple[LandingCheck, ...],
    regime_name: RegimeName,
    period: int,
) -> None:
    """Raise for the first check whose raw landings the solve reached.

    Raises:
        OffGridStateError: A reached landing lies outside, or is not a finite
            point of, a grid declaring `off_grid="raise"`.

    """
    for (target, state), count in zip(checks, jax.device_get(totals), strict=True):
        if int(count):
            raise OffGridStateError(
                off_grid_message(
                    count=int(count),
                    regime=regime_name,
                    target=target,
                    period=period,
                    state=state,
                    phase="solve",
                )
            )


def _feasible_landing_counts(*, F_arr: BoolND, landing_counts: IntND) -> IntND:
    """Sum per-action landing counts over the actions the mask keeps."""
    feasible = jnp.broadcast_to(F_arr, landing_counts.shape[:-1])
    return sum_counts(
        counts=jnp.where(feasible[..., None], landing_counts, 0),
        axis=tuple(range(landing_counts.ndim - 1)),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _Evaluated:
    """Return outputs already evaluated at the cell, whatever it is called with."""

    outputs: tuple[FloatND | BoolND, ...]
    """The outputs to return."""

    def __call__(self, **kwargs: QAndFArg) -> tuple[FloatND | BoolND, ...]:  # noqa: ARG002
        return self.outputs


@dataclass(frozen=True, kw_only=True, eq=False)
class _LandingCountingReduction:
    """Reduce one cell's actions and sum its landing counts over feasible actions."""

    __name__: ClassVar[str] = "max_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    reducer: _HardMaxQOverA | _SmoothedMaxQOverA
    """The dense reduction, whose `Q_and_F` also returns landing counts."""

    def __call__(
        self,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> tuple[FloatND, BoolND, IntND] | BoundedResult[ArrayTree]:
        """Return the reduction's value and feasibility, then the landing counts."""
        evaluated = self.reducer.Q_and_F(
            next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params
        )
        Q_arr, F_arr, landing_counts = cast(
            "tuple[FloatND, BoolND, IntND]",
            evaluated.output if isinstance(evaluated, BoundedResult) else evaluated,
        )
        value, feasibility = cast(
            "tuple[FloatND, BoolND]",
            dataclasses.replace(
                self.reducer, Q_and_F=_Evaluated(outputs=(Q_arr, F_arr))
            )(next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params),
        )
        output = (
            value,
            feasibility,
            _feasible_landing_counts(F_arr=F_arr, landing_counts=landing_counts),
        )
        return (
            BoundedResult(output=output, bound_status=evaluated.bound_status)
            if isinstance(evaluated, BoundedResult)
            else output
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _LandingCountingFold:
    """Fold the value and its empty-set cause, passing the landing counts through.

    The counts keep the folded axes: every state cell is counted, and the period
    kernel totals them over all cells anyway.
    """

    fold: Callable[..., tuple[FloatND, BoolND]]
    """The fold reduction, whose `mapped` also returns landing counts."""

    def __call__(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg,
    ) -> tuple[FloatND, BoolND, IntND] | BoundedResult[ArrayTree]:
        """Return the folded value and empty-set cause, then the landing counts."""
        mapped = self.fold.mapped  # ty: ignore[unresolved-attribute]
        evaluated = mapped(
            next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params
        )
        value, empty, landing_counts = cast(
            "tuple[FloatND, BoolND, IntND]",
            evaluated.output if isinstance(evaluated, BoundedResult) else evaluated,
        )
        folded_value, folded_empty = dataclasses.replace(
            self.fold,  # ty: ignore[invalid-argument-type]
            mapped=_Evaluated(outputs=(value, empty)),
        )(next_regime_to_V_arr=next_regime_to_V_arr, **states_actions_params)
        output = (folded_value, folded_empty, landing_counts)
        return (
            BoundedResult(output=output, bound_status=evaluated.bound_status)
            if isinstance(evaluated, BoundedResult)
            else output
        )
