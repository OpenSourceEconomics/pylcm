"""Mandatory solve/simulation input integrity and independent feasibility summaries."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.egm.projected_continuation import ProjectedChildAuthority
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.typing import (
    DataclassInstance,
    LandingCheck,
    PytreeChild,
    PytreeValue,
    QAndFArg,
    ShapeDtypeTree,
    ShardingTree,
)
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
    ScalarInt,
)

type FeasibilityCounts = tuple[IntND, IntND]
type CountedGridSearchOutput = (
    tuple[FloatND, IntND, IntND]
    | tuple[FloatND, BoolND, IntND, IntND]
    | tuple[FloatND, IntND, IntND, ScalarBool]
)

if TYPE_CHECKING:
    from _lcm.execution.output_layout import OutputRoleTree
else:
    # Importing the role resolver here cycles through core_program. The claw
    # checks the recursive role-tree shape; the existing resolver validates
    # its exact enum/dataclass leaves before lowering.
    type OutputRoleTree = (
        Enum
        | str
        | DataclassInstance
        | tuple[OutputRoleTree, ...]
        | Mapping[str, OutputRoleTree]
        | None
    )

# The same registered record represents numerical outputs and their pure roles.
type _BoundedPayload = PytreeValue | OutputRoleTree


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


class QueryInputStatus(NamedTuple):
    """Mandatory actual-child input and represented-coverage summary."""

    nonfinite_floor: ScalarBool
    """An active canonical target floor is nonfinite."""
    invalid_shifted_grid: ScalarBool
    """An active target floor fails represented saving-offset integrity."""
    coverage_missing: ScalarBool
    """An eligible actual child query lacks a supported numerical candidate."""

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[PytreeChild]], QueryInputStatus],
        tuple[None, tuple[PytreeChild, ...]],
    ]:
        """Persist native leaves and role placeholders through pytree reconstruction."""
        return _unflatten_query_input_status, (None, tuple(self))


def _flatten_query_input_status(
    status: QueryInputStatus,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(status), None


def _flatten_query_input_status_with_keys(
    status: QueryInputStatus,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), flag)
        for name, flag in zip(QueryInputStatus._fields, status, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_query_input_status(
    _aux: None, children: Iterable[PytreeChild]
) -> QueryInputStatus:
    """Restore temporary lane axes and output-role placeholders losslessly."""
    leaves = tuple(children)
    if len(leaves) != 3:  # noqa: PLR2004
        raise TypeError("A query-integrity record has exactly three leaves.")
    return cast("QueryInputStatus", tuple.__new__(QueryInputStatus, leaves))


jax.tree_util.register_pytree_with_keys(
    QueryInputStatus,
    _flatten_query_input_status_with_keys,
    _unflatten_query_input_status,
    _flatten_query_input_status,
)


class ContinuationReadCounts(NamedTuple):
    """Optional int32 totals for paired continuation reads and first-live support."""

    below_first_live: ScalarInt
    """Number of eligible reads strictly below their first supported coordinate."""
    eligible: ScalarInt
    """Total number of independently eligible paired reads."""

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[PytreeChild]], ContinuationReadCounts],
        tuple[None, tuple[PytreeChild, ...]],
    ]:
        """Persist native leaves and role placeholders through pytree reconstruction."""
        return _unflatten_continuation_read_counts, (None, tuple(self))


def _flatten_continuation_read_counts(
    status: ContinuationReadCounts,
) -> tuple[tuple[PytreeChild, ...], None]:
    return (status.below_first_live, status.eligible), None


def _flatten_continuation_read_counts_with_keys(
    status: ContinuationReadCounts,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return (
        (
            (jax.tree_util.GetAttrKey("below_first_live"), status.below_first_live),
            (
                jax.tree_util.GetAttrKey("eligible"),
                status.eligible,
            ),
        ),
        None,
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_continuation_read_counts(
    _aux: None, children: Iterable[PytreeChild | DataclassInstance]
) -> ContinuationReadCounts:
    """Restore arrays or declared dataclass roles without allocating numeric leaves."""
    leaves = tuple(children)
    if len(leaves) != 2:  # noqa: PLR2004
        raise TypeError("A continuation-read record has exactly two leaves.")
    return cast("ContinuationReadCounts", tuple.__new__(ContinuationReadCounts, leaves))


jax.tree_util.register_pytree_with_keys(
    ContinuationReadCounts,
    _flatten_continuation_read_counts_with_keys,
    _unflatten_continuation_read_counts,
    _flatten_continuation_read_counts,
)


def reduce_continuation_read_counts(
    *, counts: ContinuationReadCounts | None, active: BoolND
) -> ContinuationReadCounts | None:
    """Fold actual active read events without weighting their integer populations."""
    if counts is None:
        return None
    return ContinuationReadCounts(
        *(sum_counts(counts=jnp.where(active, count, 0)) for count in counts)
    )


def merge_continuation_read_counts(
    *, left: ContinuationReadCounts | None, right: ContinuationReadCounts | None
) -> ContinuationReadCounts | None:
    """Add bounded event summaries while preserving absent diagnostic work."""
    if left is None:
        return right
    if right is None:
        return left
    return ContinuationReadCounts(
        *(add_counts(left=a, right=b) for a, b in zip(left, right, strict=True))
    )


type _BoundedResultAux = tuple[Literal["continuation_read_counts"], bool] | None


@dataclass(frozen=True, kw_only=True, eq=False)
class BoundedResult[Output: _BoundedPayload]:
    """Ordinary mapped outputs beside separately reduced bound-integrity records."""

    output: Output
    """Value, policy or economic predicate channels retaining their ordinary axes."""
    bound_status: tuple[BoundInputStatus, ...]
    """Scalar records at an owning reduction; temporary lanes stay inside its window."""
    query_status: tuple[QueryInputStatus, ...] | None = None
    """Separate inherited query records; absent on the feature-free layout."""
    continuation_read_counts: ContinuationReadCounts | None = None
    """Optional diagnostic summary, independent of mandatory integrity records."""


def _flatten_bounded_result[Output: _BoundedPayload](
    result: BoundedResult[Output],
) -> tuple[
    tuple[
        Output
        | ShapeDtypeTree
        | ShardingTree
        | PytreeChild
        | tuple[BoundInputStatus, ...]
        | tuple[QueryInputStatus, ...]
        | ContinuationReadCounts,
        ...,
    ],
    _BoundedResultAux,
]:
    children = (result.output, result.bound_status)
    if result.query_status is not None:
        children = (*children, result.query_status)
    if result.continuation_read_counts is None:
        return children, None
    return (*children, result.continuation_read_counts), (
        "continuation_read_counts",
        result.query_status is not None,
    )


def _flatten_bounded_result_with_keys[Output: _BoundedPayload](
    result: BoundedResult[Output],
) -> tuple[
    tuple[
        tuple[
            jax.tree_util.GetAttrKey,
            Output
            | ShapeDtypeTree
            | ShardingTree
            | PytreeChild
            | tuple[BoundInputStatus, ...]
            | tuple[QueryInputStatus, ...]
            | ContinuationReadCounts,
        ],
        ...,
    ],
    _BoundedResultAux,
]:
    children = (
        (jax.tree_util.GetAttrKey("output"), result.output),
        (jax.tree_util.GetAttrKey("bound_status"), result.bound_status),
    )
    if result.query_status is not None:
        children = (
            *children,
            (jax.tree_util.GetAttrKey("query_status"), result.query_status),
        )
    if result.continuation_read_counts is None:
        return children, None
    return (
        *children,
        (
            jax.tree_util.GetAttrKey("continuation_read_counts"),
            result.continuation_read_counts,
        ),
    ), ("continuation_read_counts", result.query_status is not None)


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_bounded_result(
    _aux: _BoundedResultAux,
    children: Iterable[_BoundedPayload | ShapeDtypeTree | ShardingTree | PytreeChild],
) -> BoundedResult[_BoundedPayload]:
    """Restore numeric, abstract and role trees before their owning trust seam."""
    leaves = tuple(children)
    if _aux is None:
        if len(leaves) not in (2, 3):
            raise TypeError(
                "A bounded result has ordinary output and declared summaries."
            )
        has_query = len(leaves) == 3  # noqa: PLR2004
        counts = None
    else:
        if (
            type(_aux) is not tuple
            or len(_aux) != 2  # noqa: PLR2004
            or _aux[0] != "continuation_read_counts"
            or type(_aux[1]) is not bool
        ):
            raise TypeError("A checked bounded result requires its declared layout.")
        has_query = _aux[1]
        if len(leaves) != (4 if has_query else 3) or not isinstance(
            leaves[-1], ContinuationReadCounts
        ):
            raise TypeError("A checked bounded result has one separate count pair.")
        counts = leaves[-1]
    result = object.__new__(BoundedResult)
    object.__setattr__(result, "output", leaves[0])
    object.__setattr__(result, "bound_status", leaves[1])
    # The optional query bank follows the two ordinary children.
    query_status = leaves[2] if has_query else None
    object.__setattr__(result, "query_status", query_status)
    object.__setattr__(result, "continuation_read_counts", counts)
    return cast("BoundedResult[_BoundedPayload]", result)


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


def reduce_query_input_status(
    *, statuses: tuple[QueryInputStatus, ...] | None, active: BoolND
) -> tuple[QueryInputStatus, ...] | None:
    """Reduce one owning query window without retaining its lane flags."""
    if statuses is None:
        return None
    if (
        not statuses
        or active.dtype != jnp.bool_
        or any(
            type(status) is not QueryInputStatus
            or any(
                flag.dtype != jnp.bool_ or flag.shape != active.shape for flag in status
            )
            for status in statuses
        )
    ):
        raise TypeError("Query summaries require declared Boolean records per lane.")
    return tuple(
        QueryInputStatus(*(jnp.any(active & flag) for flag in status))
        for status in statuses
    )


def merge_query_input_status(
    *,
    left: tuple[QueryInputStatus, ...] | None,
    right: tuple[QueryInputStatus, ...] | None,
) -> tuple[QueryInputStatus, ...] | None:
    """Merge the same static child slots without merging different attributions."""
    if left is None and right is None:
        return None
    if (
        left is None
        or right is None
        or len(left) != len(right)
        or any(type(status) is not QueryInputStatus for status in (*left, *right))
    ):
        raise TypeError("Query-summary windows must preserve declaration slots.")
    return tuple(
        QueryInputStatus(*(a | b for a, b in zip(lhs, rhs, strict=True)))
        for lhs, rhs in zip(left, right, strict=True)
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


def _is_projected_status(
    *,
    authority: PostDecisionBoundSpec | ProjectedChildAuthority,
    target: RegimeName,
    period: int,
    nonfinite: bool,
    invalid_grid: bool,
) -> bool:
    """Authenticate projected provenance and prohibit fictitious floor statuses."""
    if not isinstance(authority, ProjectedChildAuthority):
        return False
    if target != authority.producer or period != authority.period:
        raise TypeError("Projected child status has a different producer/date.")
    if nonfinite or invalid_grid:
        raise TypeError("Projected child reads cannot report a floor read.")
    return True


def validate_query_input_status(
    *,
    statuses: tuple[QueryInputStatus, ...] | None,
    authorities: tuple[
        tuple[RegimeName, int, PostDecisionBoundSpec | ProjectedChildAuthority], ...
    ],
) -> None:
    """Accept all child arithmetic before interpreting any query coverage failure."""
    if len(statuses or ()) != len(authorities):
        raise TypeError("Query summaries must match model-owned declaration slots.")
    if statuses is None:
        return
    if not statuses or any(
        type(status) is not QueryInputStatus
        or any(flag.shape != () or flag.dtype != jnp.bool_ for flag in status)
        for status in statuses
    ):
        raise TypeError("Query integrity requires scalar Boolean records.")
    flags = jax.device_get(jnp.stack(tuple(jnp.stack(status) for status in statuses)))
    for (target, period, bound), (nonfinite, invalid_grid, _) in zip(
        authorities, flags, strict=True
    ):
        if isinstance(bound, ProjectedChildAuthority):
            _is_projected_status(
                authority=bound,
                target=target,
                period=period,
                nonfinite=bool(nonfinite),
                invalid_grid=bool(invalid_grid),
            )
            continue
        if nonfinite or invalid_grid:
            declarations = ", ".join(bound.declaration_names)
            reason = (
                "nonfinite borrowing floor"
                if nonfinite
                else "invalid shifted savings grid"
            )
            raise InvalidValueFunctionError(
                f"Invalid borrowing bound '{declarations}': {reason} in "
                f"regime '{target}', period {period}."
            )
    for (target, period, bound), (_, _, missing) in zip(
        authorities, flags, strict=True
    ):
        if missing:
            if isinstance(bound, ProjectedChildAuthority):
                raise InvalidValueFunctionError(
                    "Invalid projected continuation or unsupported pre-shock "
                    f"domain in regime '{target}', period {period}, "
                    f"liquid state '{bound.liquid_state}'. One cause is a "
                    "producer row that cannot meet its value and marginal "
                    f"budgets within its projection capacity of {bound.capacity} "
                    "records; raise `current_iid_projection_capacity` on the "
                    f"solver of regime '{bound.producer}' if so."
                )
            declarations = ", ".join(bound.declaration_names)
            raise InvalidValueFunctionError(
                f"query_coverage_missing for borrowing bound '{declarations}' in "
                f"regime '{target}', period {period}."
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
class ContinuationCountedKernelOutput(KernelOutput):
    """A checked literal result with only its independent paired-read summary."""

    continuation_read_counts: ContinuationReadCounts
    """Exactly two native int32 scalars, without a fabricated feasibility pair."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.continuation_read_counts) is not ContinuationReadCounts or any(
            count.shape != () or count.dtype != jnp.int32
            for count in self.continuation_read_counts
        ):
            raise TypeError("Continuation-read summaries require two int32 scalars.")


@dataclass(frozen=True, kw_only=True)
class ChildStatusKernelOutput(KernelOutput):
    """A literal-floor result with only the child summaries its solves executed.

    A source without an own query floor still reads callable-floor children
    through their queries. It publishes those declared slots, and optionally the
    paired-read counts, without a fabricated feasibility pair.
    """

    query_status: tuple[QueryInputStatus, ...]
    """Mandatory child summaries in independently declared target/period slots."""
    continuation_read_counts: ContinuationReadCounts | None = None
    """Optional paired-read population, materialized independently of feasibility."""

    def __post_init__(self) -> None:
        """Require the declared slots as native scalar Boolean records."""
        super().__post_init__()
        _fail_if_invalid_child_summaries(
            query_status=self.query_status,
            continuation_read_counts=self.continuation_read_counts,
        )


@dataclass(frozen=True, kw_only=True)
class FeasibilityKernelOutput(KernelOutput):
    """Private scalar summaries beside the unchanged public artifact channels."""

    feasibility_counts: FeasibilityCounts
    """Counts of empty-action states and other infinite-valued states."""
    current_iid_invalid: ScalarBool | None = None
    """Mandatory unresolved-current-law summary on integrated routes."""
    bound_status: tuple[BoundInputStatus, ...] = ()
    """Mandatory scalar integrity records, independently of economic counts."""
    query_status: tuple[QueryInputStatus, ...] | None = None
    """Mandatory child summaries in independently declared target/period slots."""
    landing_totals: IntND | None = None
    """Reached landing totals retained until mandatory integrity is accepted."""
    landing_checks: tuple[LandingCheck, ...] = ()
    """Canonical qualified landing checks corresponding to the totals."""
    continuation_read_counts: ContinuationReadCounts | None = None
    """Optional paired-read population, materialized independently of feasibility."""

    def __post_init__(self) -> None:
        """Require native scalar integer counters without a host transfer."""
        super().__post_init__()
        _fail_if_invalid_child_summaries(
            query_status=self.query_status,
            continuation_read_counts=self.continuation_read_counts,
        )
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


def _fail_if_invalid_child_summaries(
    *,
    query_status: tuple[QueryInputStatus, ...] | None,
    continuation_read_counts: ContinuationReadCounts | None,
) -> None:
    """Require scalar Boolean query records and int32 read counts when present."""
    if continuation_read_counts is not None and (
        type(continuation_read_counts) is not ContinuationReadCounts
        or any(
            count.shape != () or count.dtype != jnp.int32
            for count in continuation_read_counts
        )
    ):
        raise TypeError("Continuation-read summaries require two int32 scalars.")
    if query_status is not None and (
        not query_status
        or any(
            type(status) is not QueryInputStatus
            or any(flag.shape != () or flag.dtype != jnp.bool_ for flag in status)
            for status in query_status
        )
    ):
        raise TypeError("Kernel query integrity requires scalar Boolean records.")


@dataclass(frozen=True, kw_only=True, eq=False)
class CountedGridSearch:
    """Reduce canonical feasibility and final values inside the admitted program."""

    function: Callable[
        ...,
        tuple[FloatND, BoolND]
        | tuple[FloatND, BoolND, ScalarBool]
        | BoundedResult[tuple[FloatND, BoolND]],
    ]
    collective: bool
    current_iid: bool = False
    """Consume and reduce the current-law integrity leaf independently of F."""

    def __call__(
        self, **kwargs: QAndFArg
    ) -> CountedGridSearchOutput | BoundedResult[CountedGridSearchOutput]:
        """Count state cells, treating a household as a single decision."""
        evaluated = self.function(**kwargs)
        current_iid_invalid = None
        if self.current_iid:
            value, empty, invalid = cast(
                "tuple[FloatND, BoolND, ScalarBool]", evaluated
            )
            current_iid_invalid = jnp.any(invalid)
            evaluated = (value, empty)
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
        if self.current_iid:
            return cast("CountedGridSearchOutput", (*output, current_iid_invalid))
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
