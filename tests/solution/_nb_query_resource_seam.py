"""A solved ordinary NB child read through the installed query at chosen lanes.

The child has passive states `stock` and `z` on the nodes `{0, 1}` and a
discrete `job` whose `work` alternative always wins: its utility carries a
bonus of ten. Every child resource fixture is finite at all four stored
passive nodes for both alternatives, so the solved bank is ordinary; the
nonfinite fixtures are NaN or infinite only for `stay` at the reached interior
stock 1/2. Each lane fixes the actual query `(liquid, stock, z)`:

- `z` chooses the sign class of the passive stencil factors: positive inside
  `[0, 1]`, one exact zero on a node, one negative outside;
- `z = 0.9` makes `stay` ineligible under the child's own predicate;
- `stock = 0` puts the actual query on a node, where every budget is finite.

The child bank is captured from a real solve (`LCM_CAPTURE_PERIOD`) and read
through the installed `NBEGMQuery` by the production `_aggregate_child_choices`
seam, which reduces the alternatives into one scalar Boolean host slot.
"""

import functools
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from unittest import mock

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import _aggregate_child_choices
from _lcm.egm.dcegm_query import _evaluate, _passive_weights
from _lcm.egm.nbegm_query import NBEGMQuery, _read_nb_compact_corner
from _lcm.egm.projected_continuation import ProjectedChildAuthority
from _lcm.egm.query_data import select_query_data
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.solution.continuation_target import union_free_params
from _lcm.solution.feasibility import (
    BoundedResult,
    QueryInputStatus,
    validate_query_input_status,
)
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.solution.period_replay import _load_capture_payload
from _lcm.typing import EconFunctionKwargs
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
    post_decision_lower_bound,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.solvers import NBEGM
from lcm.transition import fixed_transition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    IntND,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    UserFunction,
)
from tests.solution._nb_query_resource_oracle import ResourceKind

PASSIVE_NODES = (0.0, 1.0)
JOB_CODES = (0, 1)
WORK_BONUS = 10.0


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


@categorical(ordered=True)
class Job:
    stay: ScalarInt
    work: ScalarInt


@dataclass(frozen=True)
class Lane:
    """One actual child query point."""

    liquid: float
    """The actual liquid state."""
    stock: float
    """The actual passive stock, whose interior 1/2 is where fixtures fail."""
    z: float
    """The actual second passive state, which sets the stencil sign class."""


LANES = MappingProxyType(
    {
        "positive_factors": Lane(liquid=0.0, stock=0.5, z=0.5),
        "exact_zero_factor": Lane(liquid=0.0, stock=0.5, z=0.0),
        "negative_factor": Lane(liquid=0.0, stock=0.5, z=-0.25),
        "predicate_false": Lane(liquid=0.0, stock=0.5, z=0.9),
        "finite_node": Lane(liquid=0.0, stock=0.0, z=0.5),
    }
)


def nan_resources(
    *, liquid: ContinuousState, stock: ContinuousState, job: DiscreteAction
) -> FloatND:
    return liquid + 1.25 + jnp.sqrt((stock - 0.5) ** 2 - 0.01 * (1 - job))


def positive_infinity_resources(
    *, liquid: ContinuousState, stock: ContinuousState, job: DiscreteAction
) -> FloatND:
    return liquid + 1.25 + 1.0 / ((stock - 0.5) ** 2 + job)


def negative_infinity_resources(
    *, liquid: ContinuousState, stock: ContinuousState, job: DiscreteAction
) -> FloatND:
    return liquid + 8.0 - 1.0 / ((stock - 0.5) ** 2 + job)


def positive_resources(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return liquid + 1.25 + 4.0 * stock * (1.0 - stock)


def zero_resources(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return liquid + 1.25 - 5.0 * stock * (1.0 - stock)


def negative_resources(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return liquid + 1.25 - 64.0 * stock * (1.0 - stock)


RESOURCES: MappingProxyType[ResourceKind, UserFunction] = MappingProxyType(
    {
        ResourceKind.NAN: nan_resources,
        ResourceKind.POSITIVE_INFINITY: positive_infinity_resources,
        ResourceKind.NEGATIVE_INFINITY: negative_infinity_resources,
        ResourceKind.POSITIVE: positive_resources,
        ResourceKind.ZERO_AT_FLOOR: zero_resources,
        ResourceKind.NEGATIVE_FINITE: negative_resources,
    }
)


def literal_resources(
    *, kind: ResourceKind, liquid: float, stock: float, job: int
) -> float:
    """Evaluate a fixture in scalar Python arithmetic, independent of JAX."""
    gap = (stock - 0.5) ** 2
    match kind:
        case ResourceKind.NAN:
            radicand = gap - 0.01 * (1 - job)
            return math.nan if radicand < 0.0 else liquid + 1.25 + math.sqrt(radicand)
        case ResourceKind.POSITIVE_INFINITY:
            return math.inf if gap + job == 0.0 else liquid + 1.25 + 1.0 / (gap + job)
        case ResourceKind.NEGATIVE_INFINITY:
            return -math.inf if gap + job == 0.0 else liquid + 8.0 - 1.0 / (gap + job)
        case ResourceKind.POSITIVE:
            return liquid + 1.25 + 4.0 * stock * (1.0 - stock)
        case ResourceKind.ZERO_AT_FLOOR:
            return liquid + 1.25 - 5.0 * stock * (1.0 - stock)
        case ResourceKind.NEGATIVE_FINITE:
            return liquid + 1.25 - 64.0 * stock * (1.0 - stock)
        case _:
            msg = f"The seam has no fixture {kind}."
            raise ValueError(msg)


def expected_lane_flag(*, kind: ResourceKind, lane: Lane) -> bool:
    """A lane flags exactly when some alternative's actual budget is nonfinite."""
    return any(
        not math.isfinite(
            literal_resources(kind=kind, liquid=lane.liquid, stock=lane.stock, job=job)
        )
        for job in JOB_CODES
    )


def fail_if_bank_nodes_nonfinite(*, kind: ResourceKind) -> None:
    """Fail unless every stored passive node has a finite budget for every job."""
    bad = [
        (stock, job)
        for stock in PASSIVE_NODES
        for job in JOB_CODES
        if not math.isfinite(
            literal_resources(kind=kind, liquid=0.0, stock=stock, job=job)
        )
    ]
    if bad:
        msg = f"Fixture {kind} is nonfinite at stored nodes {bad}."
        raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class Seam:
    """The installed reader beside its captured production bank."""

    query: NBEGMQuery
    """The parent's installed ordinary NB child reader."""
    carry: EGMCarry
    """The child's solved carry, captured at the parent's kernel entry."""
    pool: EconFunctionKwargs
    """The child's processed parameters and clock, without any state."""
    authorities: tuple[
        tuple[RegimeName, int, PostDecisionBoundSpec | ProjectedChildAuthority], ...
    ]
    """The parent's declared query slots, for host validation."""


def utility(*, consumption: ContinuousAction, job: DiscreteAction) -> FloatND:
    return jnp.log(consumption) + WORK_BONUS * job


def parent_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def parent_resources(*, liquid: ContinuousState) -> FloatND:
    return liquid + 1.25


def savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def eligibility(*, job: DiscreteAction, z: ContinuousState, cutoff: float) -> BoolND:
    return (job == 1) | (z <= cutoff)


def next_liquid(*, savings: FloatND) -> ContinuousState:
    return savings


def next_stock(*, stock: ContinuousState) -> ContinuousState:
    return stock


def next_z(*, z: ContinuousState) -> ContinuousState:
    return z


def terminal(
    *, liquid: ContinuousState, stock: ContinuousState, z: ContinuousState
) -> FloatND:
    return jnp.log(liquid) + 0.0 * stock + 0.0 * z


PARAMS = MappingProxyType(
    {
        "parent": {"koopmans_aggregator": {"discount_factor": 0.5}},
        "child": {
            "koopmans_aggregator": {"discount_factor": 0.5},
            "eligibility": {"cutoff": 0.75},
        },
        "done": {},
    }
)


def make_model(*, kind: ResourceKind) -> Model:
    """Build an NB parent whose ordinary NB child has two passive states and a job."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    passive = {
        "stock": IrregSpacedGrid(points=jnp.array(PASSIVE_NODES)),
        "z": IrregSpacedGrid(points=jnp.array(PASSIVE_NODES)),
    }
    liquid = IrregSpacedGrid(points=jnp.array((0.0, 0.25, 0.5, 1.0, 2.0, 4.0)))
    common: dict[str, UserFunction] = {
        "savings": savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    }
    borrowing = post_decision_lower_bound(margin=margin, lower=0.0)
    child = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": liquid, **passive},
        actions={
            "job": DiscreteGrid(category_class=Job),
            "consumption": IrregSpacedGrid(
                points=jnp.array((0.125, 0.25, 0.5, 1.0, 2.0, 4.0))
            ),
        },
        functions={**common, "utility": utility, "resources": RESOURCES[kind]},
        constraints={"borrowing": borrowing, "eligibility": eligibility},
        state_transitions={
            "liquid": next_liquid,
            "stock": fixed_transition("stock"),
            "z": fixed_transition("z"),
        },
        solver=NBEGM(
            savings_grid=IrregSpacedGrid(points=jnp.array((0.0, 0.25, 0.5, 1.0, 2.0)))
        ),
    )
    parent = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": liquid, **passive},
        actions={
            "consumption": IrregSpacedGrid(points=jnp.array((0.125, 0.25, 0.5, 1.0)))
        },
        functions={
            **common,
            "utility": parent_utility,
            "resources": parent_resources,
        },
        constraints={"borrowing": borrowing},
        state_transitions={"liquid": next_liquid, "stock": next_stock, "z": next_z},
        solver=NBEGM(
            savings_grid=IrregSpacedGrid(points=jnp.array((0.0, 0.25, 0.5, 1.0)))
        ),
    )
    return Model(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={
                    "liquid": IrregSpacedGrid(
                        points=jnp.array((0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0))
                    ),
                    **passive,
                },
                functions={"utility": terminal},
            ),
        },
        ages=AgeGrid(exact_values=(0, 10, 20)),
        edges={"parent": {"child": (0,)}, "child": {"done": (10,)}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
    )


@functools.cache
def build_seam(*, kind: ResourceKind, x64: bool) -> Seam:
    """Solve once, capturing the parent's real inputs, and bind its installed reader.

    The parent's passive states transition to themselves, so the solve reaches
    the child only at stored nodes, where every fixture is finite; it must
    succeed for every fixture. The interior is reached only by `read_lane`.
    """
    del x64  # Part of the cache key only: banks differ by precision.
    fail_if_bank_nodes_nonfinite(kind=kind)
    model = make_model(kind=kind)
    params = {name: dict(values) for name, values in PARAMS.items()}
    with tempfile.TemporaryDirectory() as directory:
        environment = {
            "LCM_CAPTURE_PERIOD": "parent@0",
            "LCM_CAPTURE_DIR": directory,
        }
        with mock.patch.dict(os.environ, environment):
            model.solve(params=params, log_level="off", runtime_checks=False)
        payload = _load_capture_payload(directory=Path(directory) / "parent@0")
    carry = payload["kernel_kwargs"]["next_regime_to_continuation"]["child"]
    if not isinstance(carry, EGMCarry):
        raise TypeError("The captured child continuation is not an EGM carry.")
    kernel = model._regimes["parent"].solution.period_kernels[0]
    if not isinstance(kernel, _RideAlongNBEGMPeriodKernel):
        raise TypeError("The parent kernel is not the ride-along NB kernel.")
    query = kernel.continuation_plan.child_reads["child"].query
    if not isinstance(query, NBEGMQuery):
        msg = f"The installed child reader is {type(query)}, not an NB query."
        raise TypeError(msg)
    _fail_if_not_ordinary_nb_reader(query=query)
    authorities = model._regimes["parent"].solution.query_input_authorities
    if authorities is None:
        raise TypeError("The parent has no declared query slots.")
    flat = model._process_params(params, runtime_checks=False)
    projected = union_free_params(
        flat_params=flat, regime_name="parent", transition_target_names=("child",)
    )
    if model.ages is None:
        raise TypeError("The model has no age grid.")
    pool = {
        **{
            name.removeprefix("child__"): value
            for name, value in projected.items()
            if name.startswith("child__")
        },
        "period": jnp.int32(1),
        "age": model.ages.values[1],
    }
    return Seam(query=query, carry=carry, pool=pool, authorities=authorities[0])


def _fail_if_not_ordinary_nb_reader(*, query: NBEGMQuery) -> None:
    """Fail unless the installed reader is the ordinary model-bound NB query."""
    authority = query.model.authority
    if (
        authority.outer_mode != "none"
        or authority.finite_outer is not None
        or authority.state_names != ("stock", "z")
        or authority.action_names != ("job",)
        or authority.bound.kind != "literal"
    ):
        msg = f"The installed reader is not the ordinary NB query: {authority}."
        raise TypeError(msg)


def permute_jobs(*, carry: EGMCarry, order: tuple[int, ...]) -> EGMCarry:
    """Reorder or repeat the carry's job rows, keeping every row's own payload."""
    index = jnp.asarray(order, dtype=jnp.int32)
    n_jobs = len(JOB_CODES)

    def take(leaf: jax.Array) -> jax.Array:
        if leaf.ndim == 4 and leaf.shape[2] == n_jobs:
            return jnp.take(leaf, index, axis=2)
        if leaf.ndim >= 3:
            msg = f"Unexpected carry leaf shape {leaf.shape}."
            raise TypeError(msg)
        return leaf

    return jax.tree.map(take, carry)


def read_lane(
    *,
    seam: Seam,
    liquid: ScalarFloat | float,
    stock: ScalarFloat | float,
    z: ScalarFloat | float,
    collect_counts: bool = False,
    carry: EGMCarry | None = None,
    jobs: tuple[int, ...] = JOB_CODES,
) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
    """Read the child at one actual lane through the production choice seam."""
    carry = seam.carry if carry is None else carry
    dtype = carry.endog_grid.dtype
    pool = {
        **seam.pool,
        "liquid": jnp.asarray(liquid, dtype=dtype),
        "stock": jnp.asarray(stock, dtype=dtype),
        "z": jnp.asarray(z, dtype=dtype),
    }
    action_values = jnp.asarray(jobs, dtype=jnp.int32)
    grids = (
        jnp.asarray(PASSIVE_NODES, dtype=dtype),
        jnp.asarray(PASSIVE_NODES, dtype=dtype),
    )
    queries = _node_queries(
        resources=seam.query.resources, pool=pool, grids=grids, jobs=action_values
    )
    shape = carry.endog_grid.shape[:-1]
    result = _aggregate_child_choices(
        carry=carry,
        prepared_search_grid=carry.endog_grid,
        prepared_valid_length=jnp.zeros(shape, dtype=jnp.int32),
        has_taste_shocks=False,
        child_index=(),
        child_passive_values=(pool["stock"], pool["z"]),
        child_passive_grids=grids,
        row_queries=queries,
        row_gradients=jnp.ones_like(queries),
        query=seam.query,
        query_pool=pool,
        query_euler_gradient=jnp.asarray(1.0, dtype=dtype),
        query_action_names=("job",),
        query_action_values=(action_values,),
        collect_continuation_read_counts=collect_counts,
    )
    if not isinstance(result, BoundedResult):
        raise TypeError("An installed NB child read omitted its mandatory slot.")
    return result


def _node_queries(
    *,
    resources: Callable[..., FloatND],
    pool: EconFunctionKwargs,
    grids: tuple[jax.Array, jax.Array],
    jobs: IntND,
) -> FloatND:
    """Evaluate the child's own budget at each stored passive node and job."""

    def at(*, stock: ScalarFloat, z: ScalarFloat, job: ScalarInt) -> ScalarFloat:
        return _evaluate(
            function=resources, pool={**pool, "stock": stock, "z": z, "job": job}
        )

    return jax.vmap(
        lambda stock: jax.vmap(
            lambda z: jax.vmap(lambda job: at(stock=stock, z=z, job=job))(jobs)
        )(grids[1])
    )(grids[0])


def host_rejects(*, seam: Seam, status: QueryInputStatus) -> bool:
    """Return whether the mandatory host validation rejects one child slot."""
    try:
        validate_query_input_status(statuses=(status,), authorities=seam.authorities)
    except InvalidValueFunctionError as error:
        if "query_coverage_missing" not in str(error):
            raise
        return True
    return False


@dataclass(frozen=True)
class LaneConditions:
    """What the production readers see for one alternative at one lane."""

    eligible: bool
    """Every declared child predicate holds."""
    row_supported: bool
    """The blended ordinary row read is supported."""
    corner_supported: bool
    """The held compact corner read is supported."""
    corner_value: float
    """The blended corner continuation."""
    min_abs_factor: float
    """The smallest absolute passive stencil factor."""
    min_factor: float
    """The smallest signed passive stencil factor."""


def lane_conditions(*, seam: Seam, lane: Lane, job: int) -> LaneConditions:
    """Run the installed reader's own row, corner and predicate reads for one job."""
    query = seam.query
    dtype = seam.carry.endog_grid.dtype
    pool = {
        **seam.pool,
        "liquid": jnp.asarray(lane.liquid, dtype=dtype),
        "stock": jnp.asarray(lane.stock, dtype=dtype),
        "z": jnp.asarray(lane.z, dtype=dtype),
        "job": jnp.int32(job),
    }
    grids = (
        jnp.asarray(PASSIVE_NODES, dtype=dtype),
        jnp.asarray(PASSIVE_NODES, dtype=dtype),
    )
    positions = (slice(None), slice(None), job)
    data = seam.carry.query_data_for_transport()
    if data is None:
        raise TypeError("The captured child carry has no query bank.")
    data = select_query_data(data=data, positions=positions, floor_positions=())
    queries = _node_queries(
        resources=query.resources,
        pool=pool,
        grids=grids,
        jobs=jnp.asarray([job], dtype=jnp.int32),
    )[..., 0]
    weights, active, negative = _passive_weights(
        values=(pool["stock"], pool["z"]), grids=grids
    )
    weights = tuple(weight.reshape(-1) for weight in weights)
    active, negative = active.reshape(-1), negative.reshape(-1)
    row = query._read_rows(
        grid=seam.carry.endog_grid[positions],
        value=seam.carry.value[positions],
        marginal=seam.carry.marginal_utility[positions],
        data=data,
        row_queries=queries,
        row_gradients=jnp.ones_like(queries),
        weights=weights,
        active=active,
        negative=negative,
        paired_marginal_read=False,
    )
    corner = _read_nb_compact_corner(
        recipe=query.model.intervals,
        data=data,
        pool=pool,
        passive_names=query.model.passive_state_names,
        passive_grids=grids,
        weights=weights,
        active=active,
        negative=negative,
    )
    eligible = jnp.asarray(a=True)
    for predicate in query.predicates:
        eligible &= _evaluate(function=predicate, pool=pool)
    factors = jnp.concatenate(weights)
    return LaneConditions(
        eligible=bool(eligible),
        row_supported=bool(row.supported),
        corner_supported=bool(corner.supported),
        corner_value=float(corner.value),
        min_abs_factor=float(jnp.min(jnp.abs(factors))),
        min_factor=float(jnp.min(factors)),
    )
