"""Mandatory solve/simulation input integrity and independent feasibility summaries."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.typing import PytreeChild, PytreeValue, QAndFArg, ShapeDtypeTree, ShardingTree
from _lcm.utils.counts import add_counts, sum_counts
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import KernelOutput
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    RegimeName,
    ScalarBool,
    ScalarFloat,
)

type FeasibilityCounts = tuple[IntND, IntND]
type CountedGridSearchOutput = (
    tuple[FloatND, IntND, IntND] | tuple[FloatND, BoolND, IntND, IntND]
)


class BoundInputStatus(NamedTuple):
    """Mandatory input integrity reduced over actual, independently active lanes."""

    nonfinite_floor: ScalarBool
    """At least one active canonical floor is nonfinite."""
    invalid_shifted_grid: ScalarBool
    """An actual shifted grid has nonfinite nodes or nonstrict successive order."""

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[PytreeChild]], BoundInputStatus],
        tuple[None, tuple[PytreeChild, ...]],
    ]:
        """Persist native leaves and role placeholders through pytree reconstruction."""
        return _unflatten_bound_status, (None, tuple(self))


def _flatten_bound_status(
    status: BoundInputStatus,
) -> tuple[tuple[PytreeChild, ...], None]:
    return (status.nonfinite_floor, status.invalid_shifted_grid), None


def _flatten_bound_status_with_keys(
    status: BoundInputStatus,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return (
        (
            (jax.tree_util.GetAttrKey("nonfinite_floor"), status.nonfinite_floor),
            (
                jax.tree_util.GetAttrKey("invalid_shifted_grid"),
                status.invalid_shifted_grid,
            ),
        ),
        None,
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_bound_status(
    _aux: None, children: Iterable[PytreeChild]
) -> BoundInputStatus:
    """Restore temporary batch axes and role placeholders without scalar checks."""
    leaves = tuple(children)
    if len(leaves) != 2:  # noqa: PLR2004
        raise TypeError("A bound-integrity record has exactly two leaves.")
    return cast("BoundInputStatus", tuple.__new__(BoundInputStatus, leaves))


jax.tree_util.register_pytree_with_keys(
    BoundInputStatus,
    _flatten_bound_status_with_keys,
    _unflatten_bound_status,
    _flatten_bound_status,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class BoundedResult[Output: PytreeValue]:
    """Ordinary mapped outputs beside separately reduced bound-integrity records."""

    output: Output
    """Value, policy or economic predicate channels retaining their ordinary axes."""
    bound_status: tuple[BoundInputStatus, ...]
    """Scalar records at an owning reduction; temporary lanes stay inside its window."""


def _flatten_bounded_result[Output: PytreeValue](
    result: BoundedResult[Output],
) -> tuple[
    tuple[
        Output | ShapeDtypeTree | ShardingTree | PytreeChild,
        tuple[BoundInputStatus, ...],
    ],
    None,
]:
    return (result.output, result.bound_status), None


def _flatten_bounded_result_with_keys[Output: PytreeValue](
    result: BoundedResult[Output],
) -> tuple[
    tuple[
        tuple[
            jax.tree_util.GetAttrKey,
            Output | ShapeDtypeTree | ShardingTree | PytreeChild,
        ],
        tuple[jax.tree_util.GetAttrKey, tuple[BoundInputStatus, ...]],
    ],
    None,
]:
    return (
        (
            (jax.tree_util.GetAttrKey("output"), result.output),
            (jax.tree_util.GetAttrKey("bound_status"), result.bound_status),
        ),
        None,
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_bounded_result(
    _aux: None,
    children: Iterable[PytreeValue | ShapeDtypeTree | ShardingTree | PytreeChild],
) -> BoundedResult[PytreeValue]:
    """Restore numeric, abstract and role trees before their owning trust seam."""
    leaves = tuple(children)
    if len(leaves) != 2:  # noqa: PLR2004
        raise TypeError("A bounded result has ordinary output and one status bank.")
    result = object.__new__(BoundedResult)
    object.__setattr__(result, "output", leaves[0])
    object.__setattr__(result, "bound_status", leaves[1])
    return cast("BoundedResult[PytreeValue]", result)


jax.tree_util.register_pytree_with_keys(
    BoundedResult,
    _flatten_bounded_result_with_keys,
    _unflatten_bounded_result,
    flatten_func=_flatten_bounded_result,
)


def reduce_bound_input_status(
    *, statuses: tuple[BoundInputStatus, ...], active: BoolND
) -> tuple[BoundInputStatus, ...]:
    """Fold one actual window before its temporary lane flags leave their owner."""
    if active.dtype != jnp.bool_ or any(
        type(status) is not BoundInputStatus
        or any(flag.dtype != jnp.bool_ or flag.shape != active.shape for flag in status)
        for status in statuses
    ):
        raise TypeError("Boolean summaries require exactly one flag per owned lane.")
    return tuple(
        BoundInputStatus(
            nonfinite_floor=jnp.any(active & status.nonfinite_floor),
            invalid_shifted_grid=jnp.any(active & status.invalid_shifted_grid),
        )
        for status in statuses
    )


def summarize_bound_inputs(
    *,
    floor_values: FloatND,
    active: BoolND,
    invalid_shifted_grid: BoolND | None = None,
) -> BoundInputStatus:
    """OR invalid inputs independently of feasibility, utility and action choice."""
    return BoundInputStatus(
        nonfinite_floor=jnp.any(active & ~jnp.isfinite(floor_values)),
        invalid_shifted_grid=(
            jnp.asarray(a=False)
            if invalid_shifted_grid is None
            else jnp.any(active & invalid_shifted_grid)
        ),
    )


def shift_savings_grid(
    *, floor: ScalarFloat, offsets: Float1D
) -> tuple[Float1D, BoundInputStatus]:
    """Keep floor-plus-offset arithmetic and report invalid represented order."""
    nodes = floor + offsets
    return nodes, summarize_bound_inputs(
        floor_values=floor,
        active=jnp.asarray(a=True),
        invalid_shifted_grid=(
            ~jnp.all(jnp.isfinite(nodes)) | ~jnp.all(nodes[1:] > nodes[:-1])
        ),
    )


def merge_bound_input_status(
    *, left: BoundInputStatus, right: BoundInputStatus
) -> BoundInputStatus:
    """Merge bounded records without constructing a history of lane flags."""
    return BoundInputStatus(
        nonfinite_floor=left.nonfinite_floor | right.nonfinite_floor,
        invalid_shifted_grid=left.invalid_shifted_grid | right.invalid_shifted_grid,
    )


def validate_runtime_bound_status(
    *,
    statuses: tuple[BoundInputStatus, ...],
    bound_specs: tuple[PostDecisionBoundSpec, ...],
    regime_name: RegimeName,
    period: int,
) -> None:
    """Accept bounded integrity records using independently supplied model names."""
    if len(statuses) != len(bound_specs):
        raise TypeError("Runtime bound status must match model declaration authority.")
    if any(
        type(status) is not BoundInputStatus
        or any(flag.shape != () or flag.dtype != jnp.bool_ for flag in status)
        for status in statuses
    ):
        raise TypeError("Runtime integrity must contain scalar Boolean arrays.")
    if not statuses:
        return
    flags = jax.device_get(
        jnp.stack(
            tuple(
                jnp.stack((status.nonfinite_floor, status.invalid_shifted_grid))
                for status in statuses
            )
        )
    )
    for spec, (nonfinite, invalid_grid) in zip(bound_specs, flags, strict=True):
        if nonfinite or invalid_grid:
            declarations = ", ".join(spec.declaration_names)
            reason = (
                "nonfinite borrowing floor"
                if nonfinite
                else "invalid shifted savings grid"
            )
            raise InvalidValueFunctionError(
                f"Invalid borrowing bound '{declarations}': {reason} in "
                f"regime '{regime_name}', period {period}."
            )


def consume_runtime_bound_outputs(
    *,
    outputs: tuple[QAndFArg, ...],
    bound_specs: tuple[PostDecisionBoundSpec, ...],
    regime_name: RegimeName,
    period: int,
) -> tuple[QAndFArg, ...]:
    """Accept declared subject summaries before ordinary decision interpretation."""
    runtime_bounds = tuple(bound for bound in bound_specs if bound.kind == "dag")
    n_flags = 2 * len(runtime_bounds)
    if not n_flags:
        return outputs
    if len(outputs) <= n_flags:
        raise TypeError("A runtime-bound subject output is missing mandatory records.")
    flags = outputs[-n_flags:]
    statuses = tuple(
        BoundInputStatus(
            nonfinite_floor=cast("ScalarBool", flags[index]),
            invalid_shifted_grid=cast("ScalarBool", flags[index + 1]),
        )
        for index in range(0, n_flags, 2)
    )
    validate_runtime_bound_status(
        statuses=statuses,
        bound_specs=runtime_bounds,
        regime_name=regime_name,
        period=period,
    )
    return outputs[:-n_flags]


@dataclass(frozen=True, kw_only=True)
class FeasibilityKernelOutput(KernelOutput):
    """Private scalar summaries beside the unchanged public artifact channels."""

    feasibility_counts: FeasibilityCounts
    """Counts of empty-action states and other infinite-valued states."""
    bound_status: tuple[BoundInputStatus, ...] = ()
    """Mandatory scalar integrity records, independently of economic counts."""
    landing_totals: IntND | None = None
    """Reached landing totals retained until mandatory integrity is accepted."""
    landing_checks: tuple[tuple[str, str], ...] = ()
    """Canonical qualified landing checks corresponding to the totals."""

    def __post_init__(self) -> None:
        """Require native scalar integer counters without a host transfer."""
        super().__post_init__()
        if bool(self.landing_checks) != (self.landing_totals is not None) or (
            self.landing_totals is not None
            and (
                self.landing_totals.dtype != jnp.int32
                or self.landing_totals.shape != (len(self.landing_checks),)
            )
        ):
            raise TypeError(
                "Deferred landings require one int32 total per declared check."
            )
        if any(
            type(status) is not BoundInputStatus
            or any(flag.shape != () or flag.dtype != jnp.bool_ for flag in status)
            for status in self.bound_status
        ):
            raise TypeError(
                "Kernel bound integrity must contain scalar Boolean arrays."
            )
        if len(self.feasibility_counts) != 2 or any(  # noqa: PLR2004
            value.ndim != 0 or value.dtype != jnp.int32
            for value in self.feasibility_counts
        ):
            raise TypeError(
                "GridSearch feasibility counts must be scalar int32 arrays."
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class CountedGridSearch:
    """Reduce canonical feasibility and final values inside the admitted program."""

    function: Callable[
        ..., tuple[FloatND, BoolND] | BoundedResult[tuple[FloatND, BoolND]]
    ]
    collective: bool

    def __call__(
        self, **kwargs: QAndFArg
    ) -> CountedGridSearchOutput | BoundedResult[CountedGridSearchOutput]:
        """Count state cells, treating a household as a single decision."""
        evaluated = self.function(**kwargs)
        value, empty = cast(
            "tuple[FloatND, BoolND]",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        infinite = jnp.isinf(value)
        if self.collective:
            infinite = jnp.any(infinite, axis=-1)
        empty_count = jnp.sum(empty, dtype=jnp.int32)
        other_inf_count = jnp.sum(infinite & ~empty, dtype=jnp.int32)
        output: CountedGridSearchOutput = (
            (value, empty, empty_count, other_inf_count)
            if self.collective
            else (value, empty_count, other_inf_count)
        )
        return (
            BoundedResult[CountedGridSearchOutput](
                output=output, bound_status=evaluated.bound_status
            )
            if type(evaluated) is BoundedResult
            else output
        )


@jax.jit
def add_feasibility_counts(
    *, total: FeasibilityCounts, block: FeasibilityCounts
) -> FeasibilityCounts:
    """Add one invariant block's counts to the counts of the blocks before it."""
    return (
        add_counts(left=total[0], right=block[0]),
        add_counts(left=total[1], right=block[1]),
    )


def count_current_feasibility(
    *,
    budget_feasible: BoolND,
    predicate_feasible: BoolND,
    value: FloatND,
    active_states: BoolND,
) -> FeasibilityCounts:
    """Count states after jointly testing each action's budget and predicates.

    Budget axes are `(*states, actions)` and predicate axes are
    `(*states, actions, predicates)`. An empty predicate axis admits every
    budget-feasible action. Padding contributes to neither count.
    """
    feasible = budget_feasible & jnp.all(predicate_feasible, axis=-1)
    empty = ~jnp.any(feasible, axis=-1)
    return (
        sum_counts(counts=active_states & empty),
        sum_counts(counts=active_states & ~empty & jnp.isinf(value)),
    )
