"""A direct scalar oracle for one ride-along NB-EGM period.

The oracle solves one regime-period of a ride-along NB-EGM model the way the
method is written down — one ride cell at a time, one discrete branch at a time,
one savings node at a time, in NumPy scalars — and returns the same objects the
production kernel publishes: the value array on the state grid, the continuation
carry, the consumption policy, and the conditional branch banks.

Independence basis. The oracle shares with the production kernel only the model's
*declarations*, as the engine composes them from the user's functions: the
cash-on-hand schedule as a function of the liquid state, the period utility (or
Epstein-Zin flow), the discount factor, the regime-transition probabilities, each
carry target's next-state law and resources map, the stochastic-node weights, and
each breakpoint's threshold and derived variable. Everything the solver *computes*
is reimplemented here from the written contract:

- the child carry read (monotone cubic Hermite value interpolation with the
  one-sided duplicated-abscissa convention, linear marginal interpolation — or the
  value interpolant's own derivative under Epstein-Zin — the passive-axis blend,
  the child's discrete-choice aggregation, the stochastic-node expectation or
  power-mean certainty equivalent, and the regime blend);
- each breakpoint's liquid preimage, its ownership advance, and the jump-case
  partition with its one-sided row publication;
- the Euler inversion (a bracketed bisection in `log c` run to convergence, not the
  production Newton loop), for the additive and the recursive Euler equation;
- the candidate families (interior Euler roots, savings-node points, per-case and
  per-interval corners, and save-to-cliff points);
- the upper envelope (a brute pointwise maximum over every bracketing link at every
  liquid node, ranked by value, right extension, slope, and stable index) and the
  hard maximum over discrete branches.

`bind_continuation` and `envelope_at_query` are never called; a test patches them to
raise and the oracle still runs.
"""

import contextlib
import inspect
import itertools
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.solution.negm import _with_outer_post_decision
from _lcm.solution.period_replay import _load_capture_payload
from lcm import Model

NEWTON_ACTION_FLOOR = 1e-8


@dataclass(frozen=True, kw_only=True)
class HostCarry:
    """An EGM carry's rows as host float64 arrays, in the carry's field order."""

    endog_grid: np.ndarray
    value: np.ndarray
    marginal_utility: np.ndarray
    taste_shock_scale: np.ndarray
    breakpoints: np.ndarray | None
    policy: np.ndarray | None


@dataclass(frozen=True, kw_only=True)
class OraclePeriodResult:
    """One regime-period solved by the direct oracle, in the kernel's layout."""

    value: np.ndarray
    """Value array in canonical productmap state order."""

    carry: HostCarry
    """The continuation carry in working layout (ride axes leading the row axis)."""

    policy: np.ndarray
    """Consumption on the state grid, in canonical productmap state order."""

    policy_alternatives: np.ndarray
    """Per state, the consumption of every candidate whose value lies within the
    tie tolerance of the published value (an object array of float arrays). Where
    two candidates are that close the oracle cannot say which one wins."""

    carry_marginal_alternatives: np.ndarray
    """Per carry row entry (working layout), the marginal of every candidate tied
    in value with the published one, as an object array of float arrays."""

    branch_value: np.ndarray | None
    """Every discrete branch's value on the state grid, branch axis leading, or
    `None` for a regime without discrete actions."""

    branch_inner_action: np.ndarray | None
    """Every discrete branch's consumption, laid out like `branch_value`."""

    branch_inner_action_alternatives: np.ndarray | None
    """Per branch and state, the consumption of every candidate tied in value
    with that branch's published one, laid out like `branch_value`."""


def ride_along_kernel(
    *,
    model: Model,
    params: Mapping[str, Any],
    regime_name: str = "alive",
    period: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return one regime-period's kernel and the inputs the solve handed it.

    The model is solved once with the engine's period capture switched on for the
    requested regime-period, so the returned context carries the *solved* child
    carries and value arrays of the following period rather than templates. The
    period defaults to the regime's middle active period.
    """
    regime = model._regimes[regime_name]
    if period is None:
        period = regime.active_periods[len(regime.active_periods) // 2]
    with tempfile.TemporaryDirectory() as directory:
        environment = {
            "LCM_CAPTURE_PERIOD": f"{regime_name}@{period}",
            "LCM_CAPTURE_DIR": directory,
        }
        with _environment(environment):
            model.solve(params=params, log_level="off")
        payload = _load_capture_payload(
            directory=Path(directory) / f"{regime_name}@{period}"
        )
    kernel_kwargs = payload["kernel_kwargs"]
    kernel = regime.solution.period_kernels[period]
    return kernel, {
        "state_action_space": kernel_kwargs["state_action_space"],
        "next_regime_to_V_arr": kernel_kwargs["next_regime_to_V_arr"],
        "next_regime_to_continuation": kernel_kwargs["next_regime_to_continuation"],
        "flat_params": kernel_kwargs["flat_params"],
        "period": period,
        "ages": kernel_kwargs["ages"],
    }


def nnbegm_inner_contexts(
    *,
    model: Model,
    params: Mapping[str, Any],
    regime_name: str = "alive",
    period: int | None = None,
) -> tuple[tuple[str, Any, dict[str, Any]], ...]:
    """Return the keeper and adjuster contexts of one NNBEGM regime-period.

    The nested solver hands both inner kernels the period's own inputs; the
    adjuster additionally sees the outer post-decision value bound into the flat
    params at each outer node. The first and the middle outer node are returned.
    """
    kernel, context = ride_along_kernel(
        model=model, params=params, regime_name=regime_name, period=period
    )
    nodes = np.asarray(kernel.outer_grid_values)
    contexts: list[tuple[str, Any, dict[str, Any]]] = [
        ("keeper", kernel.keeper_kernel, context)
    ]
    for position in sorted({0, len(nodes) // 2}):
        node_context = {
            **context,
            "flat_params": _with_outer_post_decision(
                flat_params=context["flat_params"],
                regime_name=kernel.regime_name,
                outer_post_decision=kernel.outer_post_decision,
                value=jnp.asarray(nodes[position]),
            ),
        }
        contexts.append(
            (f"adjuster@node{position}", kernel.adjuster_kernel, node_context)
        )
    return tuple(contexts)


@contextlib.contextmanager
def _environment(variables: Mapping[str, str]) -> Iterator[None]:
    """Set environment variables for the duration of the block."""
    previous = {name: os.environ.get(name) for name in variables}
    os.environ.update(variables)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value


def run_production_kernel(*, kernel: Any, context: Mapping[str, Any]) -> tuple:
    """Run the kernel's production core on the context and return its raw outputs."""
    args = kernel.build_lower_args(core_key="main", **context)
    return tuple(jax.jit(kernel.cores()["main"])(**args))


def direct_oracle_period(  # noqa: PLR0915
    *, kernel: Any, context: Mapping[str, Any], tie_tolerance: float = 0.0
) -> OraclePeriodResult:
    """Solve one ride-along regime-period with the direct scalar oracle.

    `tie_tolerance` is the absolute value gap below which two candidates count
    as tied; every tied candidate's consumption is published in
    `policy_alternatives`.
    """
    statics = kernel.statics
    spec = kernel.schedule_spec
    plan = kernel.continuation_plan

    kwargs = dict(kernel.build_lower_args(core_key="main", **context))
    kwargs.update(getattr(kernel.tiled_core, "keywords", None) or {})
    carries = {
        name: _numpy_carry(carry)
        for name, carry in kwargs["next_regime_to_continuation"].items()
    }
    dtype = np.asarray(kwargs[statics.liquid_name]).dtype
    liquid_grid = np.asarray(kwargs[statics.liquid_name], dtype=np.float64)
    ride_grids = [jnp.asarray(kwargs[name]) for name in statics.ride_names]
    ride_shape = tuple(len(grid) for grid in ride_grids)
    savings_grid = np.asarray(kernel.savings_grid, dtype=np.float64)
    param_pool = {
        key: value for key, value in kwargs.items() if key not in statics.state_names
    }
    del param_pool["next_regime_to_continuation"]
    del param_pool["next_regime_to_V_arr"]
    inverse_eis = (
        1.0
        / _scalar(
            kwargs["koopmans_aggregator__intertemporal_elasticity_of_substitution"]
        )
        if plan.risk_aversion_param_name is not None
        else None
    )
    bindings = _branch_bindings(spec.discrete_actions)
    n_branches = len(spec.discrete_actions) and len(bindings)
    functions = _period_functions(spec=spec, statics=statics, kwargs=kwargs)

    n_liquid = len(liquid_grid)
    n_published_jumps = statics.n_published_jumps
    n_row = n_liquid + 2 * n_published_jumps
    endog_rows = np.empty((*ride_shape, n_row))
    value_rows = np.empty((*ride_shape, n_row))
    marginal_rows = np.empty((*ride_shape, n_row))
    breakpoint_rows = np.empty((*ride_shape, n_published_jumps))
    value_at_liquid = np.empty((*ride_shape, n_liquid))
    policy_at_liquid = np.empty((*ride_shape, n_liquid))
    alternatives_at_liquid = np.empty((*ride_shape, n_liquid), dtype=object)
    marginal_alternatives_rows = np.empty((*ride_shape, n_row), dtype=object)
    branch_values = np.empty((*ride_shape, len(bindings), n_liquid))
    branch_policies = np.empty((*ride_shape, len(bindings), n_liquid))
    branch_alternatives = np.empty((*ride_shape, len(bindings), n_liquid), dtype=object)
    for index in itertools.product(*(range(size) for size in ride_shape)):
        cell = {
            name: grid[position]
            for name, grid, position in zip(
                statics.ride_names, ride_grids, index, strict=True
            )
        }
        geometry = _cell_geometry(
            statics=statics,
            kwargs=kwargs,
            cell=cell,
            liquid_grid=liquid_grid,
            dtype=dtype,
            action_binding={},
        )
        branch_rows = []
        for binding in bindings:
            combo_pool = {**param_pool, **cell, **binding}
            budget = _cell_budget(
                functions=functions,
                spec=spec,
                statics=statics,
                cell=cell,
                kwargs=kwargs,
                action_binding=binding,
                inverse_eis=inverse_eis,
                action_ceiling=float(savings_grid[-1]) * 1000.0 + 1000.0,
            )
            branch_geometry = _cell_geometry(
                statics=statics,
                kwargs=kwargs,
                cell=cell,
                liquid_grid=liquid_grid,
                dtype=dtype,
                action_binding=binding,
            )
            continuation = _cell_continuation(
                kernel=kernel,
                statics=statics,
                plan=plan,
                combo_pool=combo_pool,
                carries=carries,
                jumps=geometry.jumps,
                breakpoints=branch_geometry.breakpoints,
                liquid_grid=liquid_grid,
                savings_grid=savings_grid,
                dtype=dtype,
            )
            branch_rows.append(
                _solve_cell_step(
                    budget=budget,
                    statics=statics,
                    liquid_grid=liquid_grid,
                    query_grid=geometry.query_grid,
                    savings_grid=savings_grid,
                    breakpoints=branch_geometry.breakpoints,
                    jump_positions=branch_geometry.jump_positions,
                    continuation=continuation,
                )
            )
        value_stack = np.stack([rows[0] for rows in branch_rows])
        marginal_stack = np.stack([rows[1] for rows in branch_rows])
        policy_stack = np.stack([rows[2] for rows in branch_rows])
        modal = np.argmax(value_stack, axis=0)
        columns = np.arange(value_stack.shape[1])
        value_row = value_stack[modal, columns]
        marginal_row = marginal_stack[modal, columns]
        policy_row = policy_stack[modal, columns]
        alternatives_row = np.empty(len(value_row), dtype=object)
        branch_alternatives_row = np.empty(
            (len(bindings), len(value_row)), dtype=object
        )
        for point, liquid in enumerate(geometry.query_grid):
            alternatives_row[point] = np.concatenate(
                [
                    rows[3].near_ties(
                        query=float(liquid),
                        best_value=float(value_row[point]),
                        tolerance=tie_tolerance,
                    )
                    for rows in branch_rows
                ],
                axis=0,
            )
            for branch, rows in enumerate(branch_rows):
                branch_alternatives_row[branch, point] = rows[3].near_ties(
                    query=float(liquid),
                    best_value=float(value_stack[branch, point]),
                    tolerance=tie_tolerance,
                )[:, 0]
        endog_rows[index] = geometry.endog_row
        value_rows[index] = value_row
        marginal_rows[index] = marginal_row
        breakpoint_rows[index] = geometry.jumps[:n_published_jumps]
        value_at_liquid[index] = value_row[geometry.unsort][:n_liquid]
        policy_at_liquid[index] = policy_row[geometry.unsort][:n_liquid]
        marginal_alternatives_rows[index] = [ties[:, 1] for ties in alternatives_row]
        alternatives_at_liquid[index] = [
            ties[:, 0] for ties in alternatives_row[geometry.unsort][:n_liquid]
        ]
        branch_values[index] = value_stack[:, geometry.unsort][:, :n_liquid]
        branch_policies[index] = policy_stack[:, geometry.unsort][:, :n_liquid]
        branch_alternatives[index] = branch_alternatives_row[:, geometry.unsort][
            :, :n_liquid
        ]

    axis = spec.liquid_axis_pos
    carry = HostCarry(
        endog_grid=endog_rows,
        value=value_rows,
        marginal_utility=marginal_rows,
        taste_shock_scale=np.asarray(0.0),
        breakpoints=breakpoint_rows if n_published_jumps else None,
        policy=(
            policy_at_liquid.copy()
            if n_published_jumps == 0 and n_branches == 0
            else None
        ),
    )

    def bank(stack: np.ndarray) -> np.ndarray:
        return np.moveaxis(np.moveaxis(stack, -2, 0), -1, axis + 1)

    return OraclePeriodResult(
        value=np.moveaxis(value_at_liquid, -1, axis),
        carry=carry,
        policy=np.moveaxis(policy_at_liquid, -1, axis),
        policy_alternatives=np.moveaxis(alternatives_at_liquid, -1, axis),
        carry_marginal_alternatives=marginal_alternatives_rows,
        branch_value=bank(branch_values) if n_branches else None,
        branch_inner_action=bank(branch_policies) if n_branches else None,
        branch_inner_action_alternatives=(
            bank(branch_alternatives) if n_branches else None
        ),
    )


def _branch_bindings(discrete_actions: Any) -> tuple[dict[str, Any], ...]:
    """Every combination of the declared discrete actions' codes, product order."""
    names = tuple(name for name, _ in discrete_actions)
    code_sets = tuple(codes for _, codes in discrete_actions)
    return tuple(
        {
            name: jnp.asarray(code, dtype=jnp.int32)
            for name, code in zip(names, combination, strict=True)
        }
        for combination in itertools.product(*code_sets)
    )


def _numpy_carry(carry: EGMCarry) -> HostCarry:
    """Copy every array leaf of a carry to the host as float64."""

    def to_host(leaf: Any) -> Any:
        return None if leaf is None else np.asarray(leaf, dtype=np.float64)

    return HostCarry(
        endog_grid=to_host(carry.endog_grid),
        value=to_host(carry.value),
        marginal_utility=to_host(carry.marginal_utility),
        taste_shock_scale=to_host(carry.taste_shock_scale),
        breakpoints=to_host(carry.breakpoints),
        policy=to_host(carry.policy),
    )


def _scalar(value: Any) -> float:
    return float(np.asarray(value))


def _nextafter(*, values: Any, direction: float, dtype: Any) -> Any:
    """Step each value to its neighbour in the working dtype, as float64."""
    stepped = np.nextafter(
        np.asarray(values, dtype=dtype), np.asarray(direction, dtype)
    )
    return np.asarray(stepped, dtype=np.float64)


@dataclass(frozen=True, kw_only=True)
class _CellGeometry:
    """One ride cell's breakpoint partition and its liquid query grid."""

    breakpoints: np.ndarray
    """Every declared breakpoint's liquid preimage in this cell, sorted ascending."""

    jump_positions: tuple[int, ...]
    """Indices into `breakpoints` of the jump breakpoints."""

    jumps: np.ndarray
    """The jump preimages in sorted order."""

    query_grid: np.ndarray
    """Liquid points the cell is solved at: the grid, plus one point just inside
    each side of every published jump."""

    endog_row: np.ndarray
    """The published abscissae of `query_grid`: each jump-side point relabeled to
    the exact jump location, so the row carries the jump as a duplicated node."""

    unsort: np.ndarray
    """Permutation from `query_grid` order back to (grid, left sides, right sides)."""


def _cell_geometry(
    *,
    statics: Any,
    kwargs: Mapping[str, Any],
    cell: Mapping[str, Any],
    liquid_grid: np.ndarray,
    dtype: Any,
    action_binding: Mapping[str, Any],
) -> _CellGeometry:
    """Map every declared breakpoint to its liquid preimage and partition the cell.

    A breakpoint declared on the liquid state is its threshold, advanced one float
    when the side below the threshold owns the equality; a breakpoint on an affine
    derived variable is inverted through the variable's slope and intercept at zero
    assets, with the ownership advance applied on the liquid axis. Non-finite
    preimages collapse to a margin outside the grid.
    """
    preimages = np.array(
        [
            _source_preimage(
                source=source,
                statics=statics,
                kwargs=kwargs,
                cell=cell,
                dtype=dtype,
                action_binding=action_binding,
            )
            for source in statics.sources
        ],
        dtype=np.float64,
    )
    margin = max(liquid_grid[-1] - liquid_grid[0], 1.0)
    preimages = np.where(np.isnan(preimages), liquid_grid[-1] + margin, preimages)
    preimages = np.clip(preimages, liquid_grid[0] - margin, liquid_grid[-1] + margin)
    order = np.argsort(preimages, kind="stable")
    flags = np.asarray(statics.jump_flags_arr, dtype=bool)
    breakpoints = preimages[order]
    jump_positions = tuple(
        int(position) for position, source in enumerate(order) if flags[source]
    )
    jumps = breakpoints[list(jump_positions)]
    if statics.n_published_jumps == 0:
        return _CellGeometry(
            breakpoints=breakpoints,
            jump_positions=jump_positions,
            jumps=jumps,
            query_grid=liquid_grid,
            endog_row=liquid_grid,
            unsort=np.arange(len(liquid_grid)),
        )
    evaluation_points = np.concatenate(
        [
            liquid_grid,
            _nextafter(values=jumps, direction=-np.inf, dtype=dtype),
            _nextafter(values=jumps, direction=np.inf, dtype=dtype),
        ]
    )
    published = np.concatenate([liquid_grid, jumps, jumps])
    sort_order = np.argsort(evaluation_points, kind="stable")
    return _CellGeometry(
        breakpoints=breakpoints,
        jump_positions=jump_positions,
        jumps=jumps,
        query_grid=evaluation_points[sort_order],
        endog_row=published[sort_order],
        unsort=np.argsort(sort_order),
    )


def _source_preimage(
    *,
    source: Any,
    statics: Any,
    kwargs: Mapping[str, Any],
    cell: Mapping[str, Any],
    dtype: Any,
    action_binding: Mapping[str, Any],
) -> float:
    """One breakpoint's liquid preimage in one cell, ownership resolved."""
    table = kwargs[source.threshold_param_name]
    if source.threshold_subkey is not None:
        table = table.data[source.threshold_subkey]
    if source.threshold_index_state is not None:
        table = table[cell[source.threshold_index_state]]
    if source.threshold_static_index is not None:
        table = table[source.threshold_static_index]
    threshold = float(np.asarray(table).astype(dtype))
    if source.derived_of_liquid_dag is None:
        if source.equality_owner == "above":
            return threshold
        return float(_nextafter(values=threshold, direction=np.inf, dtype=dtype))
    dag = source.derived_of_liquid_dag
    dag_arg_names = frozenset(inspect.signature(dag).parameters)

    def derived_of_liquid(liquid: Any) -> Any:
        return dag(
            **{statics.liquid_name: liquid},
            **{name: cell[name] for name in source.derived_state_names},
            **{name: kwargs[name] for name in source.derived_param_names},
            **{k: v for k, v in action_binding.items() if k in dag_arg_names},
        )

    slope = _scalar(jax.grad(derived_of_liquid)(jnp.zeros(())))
    intercept = _scalar(derived_of_liquid(jnp.zeros(())))
    preimage = (threshold - intercept) / slope
    liquid_right_owns = (slope > 0.0) == (source.equality_owner == "above")
    if liquid_right_owns:
        return preimage
    return float(_nextafter(values=preimage, direction=np.inf, dtype=dtype))


@dataclass(frozen=True, kw_only=True)
class _CellContinuation:
    """The expected continuation one branch of one cell reads.

    Rows are indexed `(interval, savings node)` when the continuation reads the
    liquid state and `(savings node,)` otherwise; the cliff channels carry one
    save-to-cliff target per jump side, NaN where the target is dead.
    """

    value: np.ndarray
    marginal: np.ndarray
    cliff_savings: np.ndarray
    cliff_value: np.ndarray


def _cell_continuation(
    *,
    kernel: Any,
    statics: Any,
    plan: Any,
    combo_pool: Mapping[str, Any],
    carries: Mapping[str, HostCarry],
    jumps: np.ndarray,
    breakpoints: np.ndarray,
    liquid_grid: np.ndarray,
    savings_grid: np.ndarray,
    dtype: Any,
) -> _CellContinuation:
    """Read one branch's continuation rows, per interval when the law reads liquid."""
    risk_aversion = (
        _scalar(combo_pool[plan.risk_aversion_param_name])
        if plan.risk_aversion_param_name is not None
        else None
    )
    pools: list[Mapping[str, Any]]
    if statics.continuation_reads_liquid:
        lower_edges = np.concatenate([liquid_grid[:1], breakpoints])
        upper_edges = np.concatenate([breakpoints, liquid_grid[-1:]])
        midpoints = 0.5 * (lower_edges + upper_edges)
        pools = [
            {**combo_pool, statics.liquid_name: jnp.asarray(midpoint, dtype=dtype)}
            for midpoint in midpoints
        ]
    else:
        pools = [combo_pool]
    values, marginals, cliff_savings, cliff_values = [], [], [], []
    for pool in pools:
        targets = (
            _cliff_savings_targets(
                plan=plan,
                regime_name=kernel.regime_name,
                combo_pool=pool,
                jumps=jumps,
                savings_grid=savings_grid,
                dtype=dtype,
            )
            if kernel.cliff_candidates
            else np.zeros(0)
        )
        value, marginal = _expected_continuation(
            plan=plan,
            combo_pool=pool,
            carries=carries,
            savings_grid=np.concatenate([savings_grid, targets]),
            risk_aversion=risk_aversion,
        )
        n_nodes = len(savings_grid)
        values.append(value[:n_nodes])
        marginals.append(marginal[:n_nodes])
        cliff_savings.append(targets)
        cliff_values.append(value[n_nodes:])
    if statics.continuation_reads_liquid:
        return _CellContinuation(
            value=np.stack(values),
            marginal=np.stack(marginals),
            cliff_savings=np.stack(cliff_savings),
            cliff_value=np.stack(cliff_values),
        )
    return _CellContinuation(
        value=values[0],
        marginal=marginals[0],
        cliff_savings=cliff_savings[0],
        cliff_value=cliff_values[0],
    )


def _cliff_savings_targets(
    *,
    plan: Any,
    regime_name: str,
    combo_pool: Mapping[str, Any],
    jumps: np.ndarray,
    savings_grid: np.ndarray,
    dtype: Any,
) -> np.ndarray:
    """Savings targets a few float steps inside each side of every own-regime jump.

    The self-read child's liquid law is affine in savings; each jump's savings
    preimage is offered from both sides, displaced by four units of the law's
    rounding (in savings units) but never more than a quarter of the distance to
    the nearest other preimage. Targets off the savings grid, or under a
    non-increasing law, are NaN.
    """
    read = plan.child_reads[regime_name]

    def next_euler_state(savings: float) -> float:
        next_states = read.next_state_func(
            **combo_pool, **{plan.post_decision_name: jnp.asarray(savings, dtype=dtype)}
        )
        return _scalar(next_states[read.next_state_key])

    intercept = next_euler_state(0.0)
    slope = next_euler_state(1.0) - intercept
    s_star = (jumps - intercept) / slope
    rounding = np.finfo(dtype).eps * (np.abs(slope * s_star) + abs(intercept))
    margin = 4.0 * rounding / abs(slope)
    separation = np.abs(s_star[:, None] - s_star[None, :])
    np.fill_diagonal(separation, np.inf)
    margin = np.minimum(margin, 0.25 * separation.min(axis=-1))
    candidates = np.stack([s_star - margin, s_star + margin], axis=-1).reshape(-1)
    valid = (
        (candidates >= savings_grid[0])
        & (candidates <= savings_grid[-1])
        & (slope > 0.0)
    )
    return np.where(valid, candidates, np.nan)


def _expected_continuation(
    *,
    plan: Any,
    combo_pool: Mapping[str, Any],
    carries: Mapping[str, HostCarry],
    savings_grid: np.ndarray,
    risk_aversion: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected continuation value and marginal at every savings value of one cell.

    With a linear certainty equivalent both are probability-weighted sums over
    the regime targets and each target's stochastic nodes. Under Epstein-Zin the
    value is the power-mean certainty equivalent `nu` of the joint regime-and-node
    lottery and the marginal is its savings derivative
    `nu^gamma * E[V'^(-gamma) dV'/ds]`. A NaN savings value reads NaN.
    """
    probs = {
        target: _scalar(prob)
        for target, prob in plan.compute_regime_transition_probs(**combo_pool).items()
    }
    readers = {
        target: _ChildReader(
            read=plan.child_reads[target],
            carry=carries[target],
            combo_pool=combo_pool,
            post_decision_name=plan.post_decision_name,
        )
        for target in plan.stateful_targets
        if probs[target] > 0.0
    }
    expected_value = np.full(len(savings_grid), np.nan)
    expected_marginal = np.full(len(savings_grid), np.nan)
    for node, savings in enumerate(savings_grid):
        if math.isnan(savings):
            continue
        lottery: list[tuple[float, float, float]] = []
        for target, reader in readers.items():
            lottery.extend(
                (probs[target] * weight, value, marginal)
                for weight, value, marginal in reader.lottery(
                    savings=float(savings),
                    paired_marginal_read=risk_aversion is not None,
                )
            )
        for target in plan.scalar_targets:
            prob = probs[target]
            if prob > 0.0:
                lottery.append((prob, float(carries[target].value.reshape(-1)[0]), 0.0))
        if risk_aversion is None:
            expected_value[node] = sum(w * v for w, v, _ in lottery)
            expected_marginal[node] = sum(w * m for w, _, m in lottery)
        else:
            expected_value[node], expected_marginal[node] = _certainty_equivalent(
                lottery=lottery, risk_aversion=risk_aversion
            )
    return expected_value, expected_marginal


def _certainty_equivalent(
    *, lottery: list[tuple[float, float, float]], risk_aversion: float
) -> tuple[float, float]:
    """The power-mean certainty equivalent of a lottery and its savings derivative."""
    exponent = 1.0 - risk_aversion
    mass = sum(w for w, _, _ in lottery if w > 0.0)
    mean_power = sum(w * v**exponent for w, v, _ in lottery if w > 0.0) / mass
    certainty_equivalent = mean_power ** (1.0 / exponent)
    derivative = (
        certainty_equivalent**risk_aversion
        * sum(w * v ** (-risk_aversion) * m for w, v, m in lottery if w > 0.0)
        / mass
    )
    return certainty_equivalent, derivative


class _ChildReader:
    """One target's carry read for one cell branch, its declarations compiled once.

    The next-state law and the composed resources map (with its savings
    derivative) take the savings value, the stochastic node values, the row
    values, and the deterministic next-state codes as traced arguments.
    """

    def __init__(
        self,
        *,
        read: Any,
        carry: HostCarry,
        combo_pool: Mapping[str, Any],
        post_decision_name: str,
    ) -> None:
        self.read = read
        self.carry = carry
        self.combo_pool = dict(combo_pool)
        self.post_decision_name = post_decision_name
        self.code_names = tuple(
            name
            for name, is_stochastic in zip(
                read.discrete_state_names, read.stochastic_flags, strict=True
            )
            if not is_stochastic
        )
        self.node_values = [jnp.asarray(v) for v in read.stochastic_node_values]
        self.row_values = [jnp.asarray(v).reshape(-1) for v in read.row_values]
        self.weight_vectors: list[np.ndarray] = []
        if read.weights_func is not None:
            weights = read.weights_func(**combo_pool)
            self.weight_vectors = [
                np.asarray(weights[key], dtype=np.float64) for key in read.weight_keys
            ]
        resources_params = {
            name: combo_pool[name] for name in read.resources_param_names
        }

        def next_states(savings: Any) -> Any:
            return read.next_state_func(**combo_pool, **{post_decision_name: savings})

        # keyword-only-exempt: library-callback=jax.value_and_grad
        def resources(
            savings: Any,
            stochastic_values: tuple[Any, ...],
            row_values: tuple[Any, ...],
            codes: tuple[Any, ...],
        ) -> Any:
            bound = {
                read.euler_state_name: next_states(savings)[read.next_state_key],
                **dict(zip(self.code_names, codes, strict=True)),
                **dict(
                    zip(read.stochastic_state_names, stochastic_values, strict=True)
                ),
                **dict(zip(read.row_arg_names, row_values, strict=True)),
            }
            return read.resources_func(
                **{k: v for k, v in bound.items() if k in read.resources_arg_names},
                **resources_params,
            )

        self.next_states = jax.jit(next_states)
        self.query_and_gradient = jax.jit(jax.value_and_grad(resources))

    def lottery(
        self, *, savings: float, paired_marginal_read: bool
    ) -> list[tuple[float, float, float]]:
        """`(weight, value, marginal)` per stochastic node at one savings value.

        The child's discrete-choice aggregation happens per node combo, inside
        the lottery: every node combo is read and aggregated on its own.
        """
        read = self.read
        next_states = self.next_states(jnp.asarray(savings))
        codes = tuple(
            jnp.asarray(next_states[f"next_{name}"], dtype=jnp.int32)
            for name in self.code_names
        )
        deterministic_codes = {
            name: int(np.asarray(code))
            for name, code in zip(self.code_names, codes, strict=True)
        }
        passive_values = tuple(
            _scalar(
                next_states[f"next_{name}"]
                if f"next_{name}" in next_states
                else self.combo_pool[f"next_{name}"]
            )
            for name in read.passive_state_names
        )
        lottery: list[tuple[float, float, float]] = []
        for node_indices in itertools.product(
            *(range(len(v)) for v in self.node_values)
        ):
            weight = 1.0
            for vector, node in zip(self.weight_vectors, node_indices, strict=True):
                weight *= float(vector[node])
            if weight == 0.0:
                continue
            stochastic_values = tuple(
                self.node_values[position][node]
                for position, node in enumerate(node_indices)
            )
            child_index = _child_carry_index(
                read=read,
                deterministic_codes=deterministic_codes,
                node_indices=node_indices,
            )

            def queries_and_gradients(
                *, row: int, stochastic_values: tuple[Any, ...] = stochastic_values
            ) -> tuple[float, float]:
                query, gradient = self.query_and_gradient(
                    jnp.asarray(savings),
                    stochastic_values,
                    tuple(values[row] for values in self.row_values),
                    codes,
                )
                return _scalar(query), _scalar(gradient)

            value, marginal = _aggregate_child_rows(
                read=read,
                carry=self.carry,
                child_index=child_index,
                passive_values=passive_values,
                paired_marginal_read=paired_marginal_read,
                queries_and_gradients=queries_and_gradients,
            )
            lottery.append((weight, value, marginal))
        return lottery


def _child_carry_index(
    *,
    read: Any,
    deterministic_codes: Mapping[str, int],
    node_indices: tuple[int, ...],
) -> tuple[int, ...]:
    """Index into the child's leading discrete carry axes for one node combo."""
    nodes = iter(node_indices)
    return tuple(
        next(nodes) if is_stochastic else deterministic_codes[name]
        for name, is_stochastic in zip(
            read.discrete_state_names, read.stochastic_flags, strict=True
        )
    )


def _aggregate_child_rows(
    *,
    read: Any,
    carry: HostCarry,
    child_index: tuple[int, ...],
    passive_values: tuple[float, ...],
    paired_marginal_read: bool,
    queries_and_gradients: Callable[..., tuple[float, float]],
) -> tuple[float, float]:
    """Read one child node combo's rows, blend passive axes, aggregate choices."""
    grid_block = carry.endog_grid[child_index]
    value_block = carry.value[child_index]
    marginal_block = carry.marginal_utility[child_index]
    block_shape = value_block.shape[:-1]
    n_rows = int(np.prod(block_shape)) if block_shape else 1
    values = np.empty(n_rows)
    marginals = np.empty(n_rows)
    for row in range(n_rows):
        query, gradient = queries_and_gradients(row=row)
        xp = grid_block.reshape(n_rows, -1)[row]
        value_row = value_block.reshape(n_rows, -1)[row]
        marginal_row = marginal_block.reshape(n_rows, -1)[row]
        value = interpolate_row(query=query, xp=xp, fp=value_row, slopes=marginal_row)
        if paired_marginal_read:
            marginal = interpolate_row_derivative(
                query=query, xp=xp, fp=value_row, slopes=marginal_row
            )
        else:
            marginal = interpolate_row(query=query, xp=xp, fp=marginal_row, slopes=None)
        values[row] = value
        marginals[row] = 0.0 if value == -np.inf else marginal * gradient
    values = values.reshape(block_shape or (1,))
    marginals = marginals.reshape(block_shape or (1,))
    for passive_value, passive_grid in zip(
        passive_values, read.passive_grids, strict=True
    ):
        lower, upper, weight_upper = _locate_on_grid(
            query=passive_value, grid=np.asarray(passive_grid, dtype=np.float64)
        )
        values = _blend(
            lower=values[lower], upper=values[upper], weight_upper=weight_upper
        )
        marginals = _blend(
            lower=marginals[lower], upper=marginals[upper], weight_upper=weight_upper
        )
        marginals = np.where(values == -np.inf, 0.0, marginals)
    values = values.reshape(-1)
    marginals = marginals.reshape(-1)
    if read.has_taste_shocks:
        scale = float(carry.taste_shock_scale)
        smoothed, probabilities = _logsum_and_softmax(values=values, scale=scale)
    else:
        winner = int(np.argmax(values))
        smoothed = float(values[winner])
        probabilities = np.zeros(len(values))
        probabilities[winner] = 1.0
    return smoothed, float(np.sum(probabilities * marginals))


def _blend(*, lower: np.ndarray, upper: np.ndarray, weight_upper: float) -> np.ndarray:
    """Blend two neighbours with exact-zero weights contributing exactly nothing."""
    weight_lower = 1.0 - weight_upper
    result = np.zeros(np.broadcast(lower, upper).shape)
    if weight_lower != 0.0:
        result = result + weight_lower * lower
    if weight_upper != 0.0:
        result = result + weight_upper * upper
    return result


def _locate_on_grid(*, query: float, grid: np.ndarray) -> tuple[int, int, float]:
    """Nearest-segment bracket and (possibly extrapolating) upper weight."""
    if len(grid) == 1:
        return 0, 0, 0.0
    upper = int(np.clip(np.searchsorted(grid, query, side="right"), 1, len(grid) - 1))
    lower = upper - 1
    return lower, upper, (query - grid[lower]) / (grid[upper] - grid[lower])


def _logsum_and_softmax(
    *, values: np.ndarray, scale: float
) -> tuple[float, np.ndarray]:
    """The EV1 logsum and its choice probabilities at a positive scale."""
    if scale <= 0.0:
        winner = int(np.argmax(values))
        probabilities = np.zeros(len(values))
        probabilities[winner] = 1.0
        return float(values[winner]), probabilities
    shifted = values / scale
    top = np.max(shifted)
    weights = np.exp(shifted - top)
    total = np.sum(weights)
    return float(scale * (top + np.log(total))), weights / total


@dataclass(frozen=True, kw_only=True)
class _Bracket:
    """The two-node bracket a padded carry row selects for one query."""

    x_lower: float
    x_upper: float
    f_lower: float
    f_upper: float
    slope_lower: float
    slope_upper: float
    raw_position: float

    @property
    def width(self) -> float:
        return self.x_upper - self.x_lower

    @property
    def finite_pair(self) -> bool:
        return math.isfinite(self.f_lower) and math.isfinite(self.f_upper)

    def hermite_coefficients(self) -> tuple[float, float] | None:
        """Fritsch-Carlson limited cubic-correction coefficients, if applicable."""
        if not (
            self.finite_pair
            and math.isfinite(self.slope_lower)
            and math.isfinite(self.slope_upper)
            and self.width != 0.0
        ):
            return None
        delta = self.f_upper - self.f_lower
        secant = delta / self.width

        def limited(slope: float) -> float:
            if slope * secant > 0.0:
                return math.copysign(min(abs(slope), 3.0 * abs(secant)), secant)
            return 0.0

        return (
            self.width * limited(self.slope_lower) - delta,
            delta - self.width * limited(self.slope_upper),
        )


def _select_bracket(
    *, query: float, xp: np.ndarray, fp: np.ndarray, slopes: np.ndarray | None
) -> _Bracket | float:
    """Select the query's right-continuous bracket, or return the degenerate read."""
    valid = int(np.sum(~np.isnan(xp)))
    if valid == 0 or math.isnan(query):
        return math.nan
    if valid == 1:
        return float(fp[0])
    search = np.where(np.isnan(xp), np.inf, xp)
    upper = int(np.clip(np.searchsorted(search, query, side="right"), 1, valid - 1))
    lower = upper - 1
    width = float(xp[upper]) - float(xp[lower])
    return _Bracket(
        x_lower=float(xp[lower]),
        x_upper=float(xp[upper]),
        f_lower=float(fp[lower]),
        f_upper=float(fp[upper]),
        slope_lower=math.nan if slopes is None else float(slopes[lower]),
        slope_upper=math.nan if slopes is None else float(slopes[upper]),
        raw_position=(query - float(xp[lower])) / (width if width != 0.0 else 1.0),
    )


def interpolate_row(
    *, query: float, xp: np.ndarray, fp: np.ndarray, slopes: np.ndarray | None
) -> float:
    """Read one NaN-padded carry row at a query.

    Linear between nodes, cubic Hermite (Fritsch-Carlson limited node slopes) when
    `slopes` is given and the bracket is finite. A duplicated abscissa yields the
    right record; below the first node the first bracket's secant continues; at or
    above the last node the boundary value applies.
    """
    bracket = _select_bracket(query=query, xp=xp, fp=fp, slopes=slopes)
    if not isinstance(bracket, _Bracket):
        return bracket
    if bracket.width == 0.0:
        return bracket.f_upper
    raw = bracket.raw_position
    position = min(max(raw, 0.0), 1.0)
    linear = 0.0
    if 1.0 - position > 0.0:
        linear += (1.0 - position) * bracket.f_lower
    if position > 0.0:
        linear += position * bracket.f_upper
    if bracket.finite_pair and raw < 0.0:
        linear += (raw - position) * (bracket.f_upper - bracket.f_lower)
    coefficients = bracket.hermite_coefficients()
    if coefficients is None:
        return linear
    coefficient_lower, coefficient_upper = coefficients
    correction = (
        position
        * (1.0 - position)
        * ((1.0 - position) * coefficient_lower + position * coefficient_upper)
    )
    return linear + correction


def interpolate_row_derivative(
    *, query: float, xp: np.ndarray, fp: np.ndarray, slopes: np.ndarray
) -> float:
    """The analytic derivative of `interpolate_row`'s value read.

    Below the first node the secant continues; at or above the last node the
    value clamps so the derivative is zero; a zero-width bracket or a `-inf`
    endpoint reads zero.
    """
    bracket = _select_bracket(query=query, xp=xp, fp=fp, slopes=slopes)
    if not isinstance(bracket, _Bracket):
        return math.nan if math.isnan(bracket) else 0.0
    if bracket.width == 0.0 or bracket.f_lower == -np.inf or bracket.f_upper == -np.inf:
        return 0.0
    secant = (bracket.f_upper - bracket.f_lower) / bracket.width
    raw = bracket.raw_position
    if raw < 0.0:
        return secant if bracket.finite_pair else 0.0
    if raw > 1.0:
        return 0.0
    coefficients = bracket.hermite_coefficients()
    if coefficients is None:
        return secant
    coefficient_lower, coefficient_upper = coefficients
    correction_slope = (
        coefficient_lower
        + 2.0 * (coefficient_upper - 2.0 * coefficient_lower) * raw
        + 3.0 * (coefficient_lower - coefficient_upper) * raw**2
    ) / bracket.width
    return secant + correction_slope


@dataclass(frozen=True, kw_only=True)
class _CellBudget:
    """One branch of one cell's period declarations as scalar callables.

    Under the additive aggregator the Euler equation is `u'(c) = beta * m`, the
    period value `u(c) + beta * nu`, and the flow marginal `u'(c)`. Under
    Epstein-Zin (`inverse_eis` set) the utility declaration is the period flow
    `q`, the Euler equation `(1-beta) q^(-rho) q'(c) = beta nu^(-rho) dnu/ds`, the
    period value the CES aggregate of `(q, nu)`, and the flow marginal
    `(1-beta) V^rho q^(-rho) q'(c)`.
    """

    discount_factor: float
    utility: Callable[[float], float]
    marginal_utility: Callable[[float], float]
    coh_of_liquid: Callable[[Any], Any]
    coh_slope_of_liquid: Callable[[Any], Any]
    action_ceiling: float
    inverse_eis: float | None

    def coh(self, liquid: float) -> float:
        return _scalar(self.coh_of_liquid(jnp.asarray(liquid)))

    def coh_slope(self, liquid: float) -> float:
        return _scalar(self.coh_slope_of_liquid(jnp.asarray(liquid)))

    def euler_consumption(self, *, cont_value: float, cont_marginal: float) -> float:
        """The consumption solving the Euler equation at one savings node."""
        beta = self.discount_factor
        if self.inverse_eis is None:
            return invert_euler(
                target=beta * cont_marginal,
                marginal_utility=self.marginal_utility,
                action_lower=NEWTON_ACTION_FLOOR,
                action_upper=self.action_ceiling,
            )
        rho = self.inverse_eis
        if not cont_marginal > 0.0:
            return math.nan
        return invert_euler(
            target=beta * cont_value ** (-rho) * cont_marginal / (1.0 - beta),
            marginal_utility=lambda c: (
                self.utility(c) ** (-rho) * self.marginal_utility(c)
            ),
            action_lower=NEWTON_ACTION_FLOOR,
            action_upper=self.action_ceiling,
        )

    def period_value(self, *, consumption: float, cont_value: float) -> float:
        beta = self.discount_factor
        flow = self.utility(consumption)
        if self.inverse_eis is None:
            return flow + beta * cont_value
        exponent = 1.0 - self.inverse_eis
        if exponent == 0.0:
            return flow ** (1.0 - beta) * cont_value**beta
        return ((1.0 - beta) * flow**exponent + beta * cont_value**exponent) ** (
            1.0 / exponent
        )

    def flow_marginal(self, *, consumption: float, value: float) -> float:
        """The marginal value of cash-on-hand before the schedule slope."""
        if self.inverse_eis is None:
            return self.marginal_utility(consumption)
        rho = self.inverse_eis
        return (
            (1.0 - self.discount_factor)
            * value**rho
            * self.utility(consumption) ** (-rho)
            * self.marginal_utility(consumption)
        )


@dataclass(frozen=True, kw_only=True)
class _PeriodFunctions:
    """The period's declarations compiled once, taking the cell and branch as data."""

    utility: Callable[..., Any]
    marginal_utility: Callable[..., Any]
    coh: Callable[..., Any]
    coh_slope: Callable[..., Any]
    utility_arg_names: frozenset[str]


def _period_functions(
    *, spec: Any, statics: Any, kwargs: Mapping[str, Any]
) -> _PeriodFunctions:
    """Compile the schedule and utility declarations with the params bound.

    The cell's ride-state values and the branch's action codes are traced
    arguments, so one compilation serves every cell and branch of the period.
    """
    coh_params = {name: kwargs[name] for name in spec.coh_param_names}
    utility_params = {name: kwargs[name] for name in statics.utility_param_names}

    # keyword-only-exempt: library-callback=jax.grad
    def coh_of(liquid: Any, cell: Mapping[str, Any], binding: Mapping[str, Any]) -> Any:
        return spec.coh_of_liquid_dag(
            **{statics.liquid_name: liquid},
            **{name: cell[name] for name in statics.coh_state_names},
            **coh_params,
            **binding,
        )

    # keyword-only-exempt: library-callback=jax.grad
    def utility_of(
        consumption: Any, cell: Mapping[str, Any], binding: Mapping[str, Any]
    ) -> Any:
        return spec.utility_dag(
            **{statics.consumption_action_name: consumption},
            **{name: cell[name] for name in statics.utility_state_names},
            **utility_params,
            **binding,
        )

    return _PeriodFunctions(
        utility=jax.jit(utility_of),
        marginal_utility=jax.jit(jax.grad(utility_of)),
        coh=jax.jit(coh_of),
        coh_slope=jax.jit(jax.grad(coh_of)),
        utility_arg_names=frozenset(inspect.signature(spec.utility_dag).parameters),
    )


def _cell_budget(
    *,
    functions: _PeriodFunctions,
    spec: Any,
    statics: Any,
    cell: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    action_binding: Mapping[str, Any],
    inverse_eis: float | None,
    action_ceiling: float,
) -> _CellBudget:
    """Bind the compiled declarations and the discount factor to one cell branch.

    The discrete action binds into cash-on-hand always and into the period utility
    only when the utility declaration names it.
    """
    discount_params = {name: kwargs[name] for name in statics.discount_param_names}
    if spec.discount_factor_dag is None:
        discount_factor = _scalar(kwargs["koopmans_aggregator__discount_factor"])
    else:
        discount_factor = _scalar(
            spec.discount_factor_dag(
                **{name: cell[name] for name in statics.discount_state_names},
                **discount_params,
            )
        )
    cell = dict(cell)
    coh_binding = dict(action_binding)
    utility_binding = {
        name: value
        for name, value in action_binding.items()
        if name in functions.utility_arg_names
    }
    return _CellBudget(
        discount_factor=discount_factor,
        utility=lambda consumption: _scalar(
            functions.utility(jnp.asarray(consumption), cell, utility_binding)
        ),
        marginal_utility=lambda consumption: _scalar(
            functions.marginal_utility(jnp.asarray(consumption), cell, utility_binding)
        ),
        coh_of_liquid=lambda liquid: functions.coh(liquid, cell, coh_binding),
        coh_slope_of_liquid=lambda liquid: functions.coh_slope(
            liquid, cell, coh_binding
        ),
        action_ceiling=action_ceiling,
        inverse_eis=inverse_eis,
    )


def _solve_cell_step(
    *,
    budget: _CellBudget,
    statics: Any,
    liquid_grid: np.ndarray,
    query_grid: np.ndarray,
    savings_grid: np.ndarray,
    breakpoints: np.ndarray,
    jump_positions: tuple[int, ...],
    continuation: _CellContinuation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _Candidates]:
    """Solve one branch's 1-D step: value, marginal, and consumption on the query grid.

    Every candidate is a (liquid, value, consumption, marginal) point; the Euler
    roots of consecutive savings nodes chain into a link when both are live and
    the chain still ascends. The pointwise maximum over every bracketing link at
    every query point is the branch's solution. The candidate set is returned
    alongside so near-tied alternatives can be read off it.
    """
    if statics.continuation_reads_liquid:
        candidates = _per_interval_step_candidates(
            budget=budget,
            liquid_grid=liquid_grid,
            query_grid=query_grid,
            savings_grid=savings_grid,
            breakpoints=breakpoints,
            continuation=continuation,
        )
    elif statics.has_jump:
        candidates = _jump_step_candidates(
            budget=budget,
            liquid_grid=liquid_grid,
            query_grid=query_grid,
            savings_grid=savings_grid,
            breakpoints=breakpoints,
            jump_positions=jump_positions,
            continuation=continuation,
        )
    else:
        candidates = _kink_step_candidates(
            budget=budget,
            query_grid=query_grid,
            savings_grid=savings_grid,
            continuation=continuation,
        )
    value_row = np.empty(len(query_grid))
    marginal_row = np.empty(len(query_grid))
    policy_row = np.empty(len(query_grid))
    for point, liquid in enumerate(query_grid):
        value_row[point], policy_row[point], marginal_row[point] = candidates.envelope(
            query=float(liquid)
        )
    return value_row, marginal_row, policy_row, candidates


def _euler_chain(
    *,
    budget: _CellBudget,
    candidates: _Candidates,
    savings_grid: np.ndarray,
    cont_value: np.ndarray,
    cont_marginal: np.ndarray,
    liquid_of_coh: Callable[..., float],
    slope_at: Callable[..., float],
    in_case: Callable[..., bool],
) -> None:
    """Add one chain of interior Euler-root candidates, one per live savings node."""
    previous_liquid = math.nan
    previous_added = False
    for node, savings in enumerate(savings_grid):
        consumption = budget.euler_consumption(
            cont_value=float(cont_value[node]), cont_marginal=float(cont_marginal[node])
        )
        if not (cont_marginal[node] > 0.0 and math.isfinite(consumption)):
            previous_added = False
            continue
        liquid = liquid_of_coh(coh=consumption + float(savings))
        if not in_case(liquid=liquid):
            previous_added = False
            continue
        value = budget.period_value(
            consumption=consumption, cont_value=float(cont_value[node])
        )
        added = candidates.add(
            liquid=liquid,
            value=value,
            policy=consumption,
            marginal=slope_at(liquid=liquid)
            * budget.flow_marginal(consumption=consumption, value=value),
            linked_to_previous=previous_added and liquid > previous_liquid,
        )
        previous_liquid = liquid
        previous_added = added


def _point_candidate(
    *,
    budget: _CellBudget,
    candidates: _Candidates,
    liquid: float,
    coh: float,
    savings: float,
    cont_value: float,
    slope: float,
    linked_to_previous: bool = False,
) -> bool:
    """Add one zero-width candidate consuming `coh - savings` at a liquid point."""
    consumption = coh - savings
    if not consumption > 0.0:
        return False
    value = budget.period_value(consumption=consumption, cont_value=cont_value)
    return candidates.add(
        liquid=liquid,
        value=value,
        policy=consumption,
        marginal=slope * budget.flow_marginal(consumption=consumption, value=value),
        linked_to_previous=linked_to_previous,
    )


def _kink_step_candidates(
    *,
    budget: _CellBudget,
    query_grid: np.ndarray,
    savings_grid: np.ndarray,
    continuation: _CellContinuation,
) -> _Candidates:
    """Candidates of a continuous (kink-only) budget.

    Interior candidates place each Euler root at the liquid value whose
    cash-on-hand equals consumption plus savings; at every liquid node, every
    post-decision node that leaves positive consumption is its own zero-width
    point candidate.
    """
    coh_grid = np.array([budget.coh(liquid) for liquid in query_grid])
    candidates = _Candidates()
    _euler_chain(
        budget=budget,
        candidates=candidates,
        savings_grid=savings_grid,
        cont_value=continuation.value,
        cont_marginal=continuation.marginal,
        liquid_of_coh=lambda *, coh: invert_cash_on_hand(
            target=coh, coh_grid=coh_grid, liquid_grid=query_grid
        ),
        slope_at=lambda *, liquid: budget.coh_slope(liquid),
        in_case=lambda *, liquid: True,  # noqa: ARG005
    )
    for node, savings in enumerate(savings_grid):
        for point, liquid in enumerate(query_grid):
            _point_candidate(
                budget=budget,
                candidates=candidates,
                liquid=float(liquid),
                coh=float(coh_grid[point]),
                savings=float(savings),
                cont_value=float(continuation.value[node]),
                slope=budget.coh_slope(float(liquid)),
            )
    return candidates


def _jump_step_candidates(
    *,
    budget: _CellBudget,
    liquid_grid: np.ndarray,
    query_grid: np.ndarray,
    savings_grid: np.ndarray,
    breakpoints: np.ndarray,
    jump_positions: tuple[int, ...],
    continuation: _CellContinuation,
) -> _Candidates:
    """Candidates of a budget with jump breakpoints.

    The jumps cut the liquid axis into cases on each of which cash-on-hand is
    continuous. Each case extends its own affine pieces over the whole grid,
    inverts that continuous map for the Euler roots, and keeps only the roots
    landing inside the case (lower-closed, upper-open); it adds one chain of
    hard-borrowing corners over its own query points. Every finite save-to-cliff
    target is offered as a point candidate at every query point, consuming that
    point's own-interval cash-on-hand minus the target.
    """
    slopes, intercepts = _interval_pieces(
        budget=budget, liquid_grid=liquid_grid, breakpoints=breakpoints
    )
    grid_interval = np.searchsorted(breakpoints, query_grid, side="right")
    last_interval = len(slopes) - 1
    case_starts = (0, *(position + 1 for position in jump_positions))
    case_ends = (*jump_positions, last_interval)
    n_cases = len(case_starts)
    candidates = _Candidates()
    for case, (start, end) in enumerate(zip(case_starts, case_ends, strict=True)):
        case_grid_interval = np.clip(grid_interval, start, end)
        coh_case_grid = (
            slopes[case_grid_interval] * query_grid + intercepts[case_grid_interval]
        )
        lower = -math.inf if case == 0 else float(breakpoints[start - 1])
        upper = math.inf if case == n_cases - 1 else float(breakpoints[end])

        def slope_in_case(
            *, liquid: float, start: int = start, end: int = end
        ) -> float:
            interval = int(
                np.clip(np.searchsorted(breakpoints, liquid, side="right"), start, end)
            )
            return float(slopes[interval])

        _euler_chain(
            budget=budget,
            candidates=candidates,
            savings_grid=savings_grid,
            cont_value=continuation.value,
            cont_marginal=continuation.marginal,
            liquid_of_coh=lambda *, coh, coh_case_grid=coh_case_grid: (
                invert_cash_on_hand(
                    target=coh, coh_grid=coh_case_grid, liquid_grid=query_grid
                )
            ),
            slope_at=slope_in_case,
            in_case=lambda *, liquid, lower=lower, upper=upper: lower <= liquid < upper,
        )
        previous_liquid = math.nan
        previous_added = False
        for point, liquid in enumerate(query_grid):
            if not lower <= liquid < upper:
                previous_added = False
                continue
            added = _point_candidate(
                budget=budget,
                candidates=candidates,
                liquid=float(liquid),
                coh=float(coh_case_grid[point]),
                savings=0.0,
                cont_value=float(continuation.value[0]),
                slope=float(slopes[case_grid_interval[point]]),
                linked_to_previous=previous_added and liquid > previous_liquid,
            )
            previous_liquid = float(liquid)
            previous_added = added
    for point, liquid in enumerate(query_grid):
        interval = grid_interval[point]
        point_coh = float(slopes[interval] * liquid + intercepts[interval])
        for target, target_value in zip(
            continuation.cliff_savings, continuation.cliff_value, strict=True
        ):
            if math.isnan(target):
                continue
            _point_candidate(
                budget=budget,
                candidates=candidates,
                liquid=float(liquid),
                coh=point_coh,
                savings=float(target),
                cont_value=float(target_value),
                slope=float(slopes[interval]),
            )
    return candidates


def _per_interval_step_candidates(
    *,
    budget: _CellBudget,
    liquid_grid: np.ndarray,
    query_grid: np.ndarray,
    savings_grid: np.ndarray,
    breakpoints: np.ndarray,
    continuation: _CellContinuation,
) -> _Candidates:
    """Candidates of a budget whose continuation differs per liquid interval.

    Each interval is its own case: its single affine cash-on-hand piece, extended
    over the whole grid, is inverted against the interval's own continuation row,
    the roots are masked to the interval, and the interval's lower- and
    upper-savings corners (consuming true cash-on-hand minus the first and the
    last savings node) are added. Every savings node and every finite
    save-to-cliff target is a point candidate at every query point, reading the
    query point's own interval's continuation.
    """
    slopes, intercepts = _interval_pieces(
        budget=budget, liquid_grid=liquid_grid, breakpoints=breakpoints
    )
    coh_true = np.array([budget.coh(liquid) for liquid in query_grid])
    grid_interval = np.searchsorted(breakpoints, query_grid, side="right")
    lowers = np.concatenate([[-np.inf], breakpoints])
    uppers = np.concatenate([breakpoints, [np.inf]])
    candidates = _Candidates()
    for interval in range(len(slopes)):
        coh_case_grid = slopes[interval] * query_grid + intercepts[interval]
        span = np.max(coh_case_grid) - np.min(coh_case_grid)
        if span <= 1e-6 * max(1.0, np.max(np.abs(coh_case_grid))):
            msg = "The direct oracle does not cover a flat cash-on-hand interval."
            raise NotImplementedError(msg)
        lower, upper = float(lowers[interval]), float(uppers[interval])
        _euler_chain(
            budget=budget,
            candidates=candidates,
            savings_grid=savings_grid,
            cont_value=continuation.value[interval],
            cont_marginal=continuation.marginal[interval],
            liquid_of_coh=lambda *, coh, coh_case_grid=coh_case_grid: (
                invert_cash_on_hand(
                    target=coh, coh_grid=coh_case_grid, liquid_grid=query_grid
                )
            ),
            slope_at=lambda *, liquid, interval=interval: float(  # noqa: ARG005
                slopes[interval]
            ),
            in_case=lambda *, liquid, lower=lower, upper=upper: lower <= liquid < upper,
        )
        for point, liquid in enumerate(query_grid):
            if not lower <= liquid < upper:
                continue
            for savings, cont_value in (
                (float(savings_grid[0]), float(continuation.value[interval, 0])),
                (float(savings_grid[-1]), float(continuation.value[interval, -1])),
            ):
                _point_candidate(
                    budget=budget,
                    candidates=candidates,
                    liquid=float(liquid),
                    coh=float(coh_true[point]),
                    savings=savings,
                    cont_value=cont_value,
                    slope=float(slopes[interval]),
                )
    for point, liquid in enumerate(query_grid):
        interval = grid_interval[point]
        node_savings = [
            (float(s), float(v))
            for s, v in zip(savings_grid, continuation.value[interval], strict=True)
        ]
        cliff = [
            (float(s), float(v))
            for s, v in zip(
                continuation.cliff_savings[interval],
                continuation.cliff_value[interval],
                strict=True,
            )
            if not math.isnan(s)
        ]
        for savings, cont_value in [*node_savings, *cliff]:
            _point_candidate(
                budget=budget,
                candidates=candidates,
                liquid=float(liquid),
                coh=float(coh_true[point]),
                savings=savings,
                cont_value=cont_value,
                slope=float(slopes[interval]),
            )
    return candidates


def _interval_pieces(
    *, budget: _CellBudget, liquid_grid: np.ndarray, breakpoints: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The active affine cash-on-hand piece of every interval, read at its midpoint."""
    lower_edges = np.concatenate([liquid_grid[:1], breakpoints])
    upper_edges = np.concatenate([breakpoints, liquid_grid[-1:]])
    midpoints = 0.5 * (lower_edges + upper_edges)
    slopes = np.array([budget.coh_slope(float(midpoint)) for midpoint in midpoints])
    values = np.array([budget.coh(float(midpoint)) for midpoint in midpoints])
    return slopes, values - slopes * midpoints


def invert_euler(
    *,
    target: float,
    marginal_utility: Callable[[float], float],
    action_lower: float,
    action_upper: float,
) -> float:
    """Solve `marginal_utility(c) = target` by bisection in `log c`.

    A negative target has no root; a target below machine epsilon is raised to it
    (the degenerate-inversion guard). A target above the marginal at
    `action_lower` returns the floor; a target the expanded bracket cannot reach
    returns NaN.
    """
    if target < 0.0:
        return math.nan
    target = max(target, np.finfo(np.float64).eps)
    lower = action_lower
    upper = action_upper
    for _ in range(32):
        if marginal_utility(upper) <= target:
            break
        upper *= 4.0
    marginal_lower = marginal_utility(lower)
    marginal_upper = marginal_utility(upper)
    if not (math.isfinite(marginal_lower) and math.isfinite(marginal_upper)):
        return math.nan
    if target > marginal_lower:
        return lower
    if marginal_upper > target:
        return math.nan
    log_lower, log_upper = math.log(lower), math.log(upper)
    for _ in range(200):
        log_middle = 0.5 * (log_lower + log_upper)
        if log_middle in (log_lower, log_upper):
            break
        if marginal_utility(math.exp(log_middle)) > target:
            log_lower = log_middle
        else:
            log_upper = log_middle
    return math.exp(0.5 * (log_lower + log_upper))


def invert_cash_on_hand(
    *, target: float, coh_grid: np.ndarray, liquid_grid: np.ndarray
) -> float:
    """Liquid value whose cash-on-hand equals `target`.

    Piecewise linear through the grid nodes, continued along the end brackets'
    secants outside the grid.
    """
    if target < coh_grid[0]:
        lower, upper = 0, 1
    elif target > coh_grid[-1]:
        lower, upper = len(coh_grid) - 2, len(coh_grid) - 1
    else:
        return float(np.interp(target, coh_grid, liquid_grid))
    if not coh_grid[upper] > coh_grid[lower] and not coh_grid[lower] > coh_grid[upper]:
        return float(np.interp(target, coh_grid, liquid_grid))
    slope = (liquid_grid[upper] - liquid_grid[lower]) / (
        coh_grid[upper] - coh_grid[lower]
    )
    return float(liquid_grid[lower] + (target - coh_grid[lower]) * slope)


class _Candidates:
    """Candidate points with their consecutive links, and the envelope over them."""

    def __init__(self) -> None:
        self.liquid: list[float] = []
        self.value: list[float] = []
        self.policy: list[float] = []
        self.marginal: list[float] = []
        self.links: list[tuple[int, int]] = []

    def add(
        self,
        *,
        liquid: float,
        value: float,
        policy: float,
        marginal: float,
        linked_to_previous: bool,
    ) -> bool:
        """Record a candidate; a non-finite abscissa or value is dead and skipped."""
        if not (math.isfinite(liquid) and math.isfinite(value)):
            return False
        index = len(self.liquid)
        self.liquid.append(liquid)
        self.value.append(value)
        self.policy.append(policy)
        self.marginal.append(marginal)
        if linked_to_previous:
            self.links.append((index - 1, index))
        return True

    def near_ties(
        self, *, query: float, best_value: float, tolerance: float
    ) -> np.ndarray:
        """`(consumption, marginal)` of every bracketing link within `tolerance`
        of `best_value`, as a `(n_ties, 2)` array."""
        ties = [
            (policy, marginal)
            for value, policy, marginal in self._bracketing_reads(query=query)
            if value >= best_value - tolerance
        ]
        return np.asarray(ties, dtype=np.float64).reshape(-1, 2)

    def _bracketing_reads(
        self, *, query: float
    ) -> Iterator[tuple[float, float, float]]:
        """Every link bracketing the query, read at it, in candidate order."""
        all_links = [*self.links, *((i, i) for i in range(len(self.liquid)))]
        for left, right in all_links:
            x_left, x_right = self.liquid[left], self.liquid[right]
            if not min(x_left, x_right) <= query <= max(x_left, x_right):
                continue
            value_read, policy_read, marginal_read = (
                _along_link(
                    query=query,
                    x_left=x_left,
                    x_right=x_right,
                    left=channel[left],
                    right=channel[right],
                )
                for channel in (self.value, self.policy, self.marginal)
            )
            yield value_read, policy_read, marginal_read

    def envelope(self, *, query: float) -> tuple[float, float, float]:
        """Value, policy, and marginal of the best bracketing link at the query.

        A link is ranked by its value at the query, then by whether it extends
        strictly right of the query, then by its slope, then by its position in
        the candidate order; consecutive links precede self-brackets.
        """
        best_rank: tuple[float, float, float, int] | None = None
        best: tuple[float, float, float] = (math.nan, math.nan, math.nan)
        all_links = [*self.links, *((i, i) for i in range(len(self.liquid)))]
        for position, (left, right) in enumerate(all_links):
            x_left, x_right = self.liquid[left], self.liquid[right]
            low, high = min(x_left, x_right), max(x_left, x_right)
            if not low <= query <= high:
                continue
            value_read, policy_read, marginal_read = (
                _along_link(
                    query=query,
                    x_left=x_left,
                    x_right=x_right,
                    left=channel[left],
                    right=channel[right],
                )
                for channel in (self.value, self.policy, self.marginal)
            )
            channels = (value_read, policy_read, marginal_read)
            slope = (
                0.0
                if x_right == x_left
                else (self.value[right] - self.value[left]) / (x_right - x_left)
            )
            rank = (channels[0], float(query < high), slope, -position)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = channels
        return best


def _along_link(
    *, query: float, x_left: float, x_right: float, left: float, right: float
) -> float:
    """Read an affine link at the query, returning an endpoint exactly on it."""
    if x_left in (query, x_right):
        return left
    if query == x_right:
        return right
    return ((x_right - query) * left + (query - x_left) * right) / (x_right - x_left)
