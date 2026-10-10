"""Average a target's value function over a shock added to one of its states.

A law declared as `lcm.AdditiveShockTransition` reads `next_<state> = base + shock`,
where the shock reads draws of next period's stochastic states. The continuation
then needs `E_k[V(base + shock_k, ...)]` at every source point, and every node of
every draw the shock reads moves the interpolation coordinate.

This module takes the draws the target does not store — transition-local draws,
such as a folded shock that only the transition reads — out of that per-point
expectation. Once per call it forms, for every combination of the shock's
conditioners and of the stored draws the shock reads,

$$W(z) = \\sum_k w_k V(z + \\text{shock}_k),$$

and the continuation reads `W` at `z = base`: one coordinate per stored node rather
than one per node of every draw. Linear interpolation makes `V` linear between its
grid points `a_j`, so `W` is linear between the merged points `{a_j - shock_k}`.
Storing `W` there is exact, and reading it with the same linear interpolation
reproduces the per-node expectation up to rounding.
"""

import dataclasses
import inspect
import itertools
from collections.abc import Callable, Mapping
from typing import NoReturn, no_type_check

import jax
import jax.numpy as jnp
import numpy as np
from dags import concatenate_functions
from dags.tree import qname_from_tree_path

from _lcm.additive_shock_transition import AdditiveShockTransition
from _lcm.grids.continuous import IrregSpacedGrid
from _lcm.probability import is_represented_zero
from _lcm.regime_building.V import (
    VInterpolationInfo,
    _get_coordinate_finder,
    _get_interpolator,
    _get_lookup_function,
    _publish_signature,
)
from _lcm.transition_plans import (
    LotteryLifetime,
    TargetTransitionPlan,
    TargetTransitionPlans,
)
from _lcm.typing import (
    EconFunctionArg,
    EconFunctionsMapping,
    FunctionName,
    QualifiedName,
    ReferenceName,
    RegimeName,
    StateName,
    TransitionFunction,
    TransitionFunctionName,
)
from _lcm.utils.functools import get_union_of_args
from lcm.exceptions import ModelInitializationError
from lcm.typing import Float1D, FloatND, Int1D, ScalarInt, UserFunction

_TIME_NAMES = frozenset({"period", "age"})
# Knots `_merged_knots` adds beyond the merged points, one on each side.
_N_ANCHORS = 2
_HIGHEST = jax.lax.Precision.HIGHEST


def average_over_shock(
    *,
    values: FloatND,
    points: Float1D,
    coordinate: Callable[[FloatND], FloatND],
    shocks: Float1D,
    weights: Float1D,
) -> tuple[Float1D, FloatND]:
    """Average values over a shock added to the last axis's coordinate.

    Args:
        values: Values on a grid, the shocked axis last.
        points: The grid points of the last axis, ascending.
        coordinate: Map from a value of the last axis to its fractional grid
            coordinate, as the value function's interpolator computes it.
        shocks: One shock per node of the averaged draws.
        weights: The nodes' probabilities.

    Returns:
        Tuple of the merged points, ascending, and the average at those points,
        the last axis of `values` replaced by them. A point that coincides with
        another is kept once; its copy moves above the largest point, where the
        average is linear, so there are always `len(points) * len(shocks)` merged
        points, plus one anchor below and one above them (`_merged_knots`).

    """
    knots, averaging = shock_averaging_matrix(
        points=points, coordinate=coordinate, shocks=shocks, weights=weights
    )
    return knots, apply_averaging(
        values=values, averaging=averaging, subscripts="...n,nm->...m"
    )


def shock_averaging_matrix(
    *,
    points: Float1D,
    coordinate: Callable[[FloatND], FloatND],
    shocks: Float1D,
    weights: Float1D,
) -> tuple[Float1D, FloatND]:
    """Return the merged points and the matrix averaging a grid onto them.

    Row `j` and column `m` hold the weight the grid's node `j` receives in the
    average at merged point `m`: each shock's linear-interpolation stencil at
    `m + shock_k`, times the shock's probability, normalized by the
    probabilities' sum. A second slab marks the entries some stencil reaches
    under a nonzero weight, which decides whether a non-finite node reaches the
    average.

    Args:
        points: The grid points, ascending.
        coordinate: Map from a value to its fractional grid coordinate.
        shocks: One shock per node of the averaged draws.
        weights: The nodes' probabilities.

    Returns:
        Tuple of the merged points `[M]` and the matrix `[2, n, M]`: the
        averaging weights and the reach indicator.

    """
    knots = _merged_knots(points=points, shocks=shocks)
    coordinates = jax.vmap(lambda shock: coordinate(knots + shock))(shocks)
    n_points = points.shape[0]
    lower = jnp.clip(jnp.floor(coordinates), 0, n_points - 2).astype(jnp.int32)
    upper_weight = coordinates - lower
    lower_weight = 1 - upper_weight
    nodes = jnp.arange(n_points)[:, None, None]
    is_lower = nodes == lower[None]
    is_upper = nodes == (lower + 1)[None]
    stencil = jnp.where(is_lower, lower_weight[None], 0) + jnp.where(
        is_upper, upper_weight[None], 0
    )
    reach = (is_lower & ~is_represented_zero(lower_weight)[None]) | (
        is_upper & ~is_represented_zero(upper_weight)[None]
    )
    live = ~is_represented_zero(weights)
    averaging = jnp.einsum(
        "k,nkm->nm", weights / jnp.sum(weights), stencil, precision=_HIGHEST
    )
    reached = jnp.any(reach & live[None, :, None], axis=1)
    return knots, jnp.stack([averaging, reached.astype(averaging.dtype)])


def apply_averaging(*, values: FloatND, averaging: FloatND, subscripts: str) -> FloatND:
    """Contract values with an averaging matrix, keeping infeasibility exact.

    The contraction runs on the finite values. An average that reaches a `-inf`
    node under a nonzero weight is `-inf`, as an interpolated read touching it
    is; one that reaches any other non-finite node is NaN.

    Args:
        values: The values, holding the grid axis named `n` in `subscripts`.
        averaging: The `[2, ..., n, M]` output of `shock_averaging_matrix`, with
            any batch axes it shares with `values`.
        subscripts: The `einsum` contraction of `values` with one slab.

    Returns:
        The averaged values.

    """
    finite = jnp.isfinite(values)
    averaged = jnp.einsum(
        subscripts,
        jnp.where(finite, values, 0),
        averaging[0],
        precision=_HIGHEST,
    )
    reach = averaging[1]
    infeasible = (
        jnp.einsum(
            subscripts,
            jnp.isneginf(values).astype(reach.dtype),
            reach,
            precision=_HIGHEST,
        )
        > 0
    )
    undefined = (
        jnp.einsum(
            subscripts,
            (~finite & ~jnp.isneginf(values)).astype(reach.dtype),
            reach,
            precision=_HIGHEST,
        )
        > 0
    )
    averaged = jnp.where(infeasible, -jnp.inf, averaged)
    return jnp.where(undefined, jnp.nan, averaged)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShockAveragePlan:
    """How one target's continuation averages its value over an additive shock."""

    law: AdditiveShockTransition
    """The declared law."""
    law_name: TransitionFunctionName
    """The `next_<state>` name of the law."""
    state_name: StateName
    """The target's continuous state the shock is added to."""
    conditioners: tuple[tuple[str, tuple[int | bool, ...]], ...]
    """The declared conditioners the shock reads, each with its support."""
    function_conditioners: tuple[str, ...]
    """The conditioners that are functions, evaluated with the next states."""
    integrated_draws: tuple[TransitionFunctionName, ...]
    """Transition-local draws the shock reads, averaged into the stored value."""
    kept_draws: tuple[TransitionFunctionName, ...]
    """Stored draws the shock reads, each an axis of the target's value."""
    shock_param_names: tuple[QualifiedName, ...]
    """What the shock reads besides its conditioners and the draws."""


def plan_shock_average(
    *,
    target_regime_name: RegimeName,
    bundle: Mapping[TransitionFunctionName, TransitionFunction],
    functions: EconFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    v_interpolation_info: VInterpolationInfo,
    lottery_variables: tuple[TransitionFunctionName, ...],
    dependencies_by_law: Mapping[TransitionFunctionName, frozenset[str]],
    allowed_inputs: frozenset[str] | None,
    co_map_state_names: tuple[StateName, ...],
) -> ShockAveragePlan | None:
    """Return how a target averages over its additive shock, or `None` without one.

    Args:
        target_regime_name: Regime the continuation leads into.
        bundle: This target's unqualified `next_<state>` transition functions.
        functions: Immutable mapping of function names to internal user functions.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.
        v_interpolation_info: The target's V-interpolation info.
        lottery_variables: The target's stochastic `next_<state>` names.
        dependencies_by_law: Per draw-dependent law, the draws it reads.
        allowed_inputs: Names the shock may read besides its conditioners, the
            draws and the time coordinates: the source's parameters. `None`
            skips that check.
        co_map_state_names: States whose axes are sliced off the value array.

    Returns:
        The plan, or `None` when no law of the bundle is an additive shock
        transition.

    Raises:
        ModelInitializationError: If the declaration cannot be averaged exactly.

    """
    laws = {
        name: law
        for name, law in bundle.items()
        if isinstance(law, AdditiveShockTransition)
    }
    if not laws:
        return None
    if len(laws) > 1:
        _fail(
            target_regime_name=target_regime_name,
            reason=f"is declared on several laws, {sorted(laws)}",
        )
    ((law_name, law),) = laws.items()
    state_name = law_name.removeprefix("next_")
    if state_name not in v_interpolation_info.continuous_states:
        _fail(
            target_regime_name=target_regime_name,
            reason=(
                f"is added to '{state_name}', which the target does not carry "
                "as a continuous state"
            ),
        )
    if state_name in co_map_state_names:
        _fail(
            target_regime_name=target_regime_name,
            reason=f"is added to the sliced axis '{state_name}'",
        )

    integrated, kept = _split_shock_draws(
        target_regime_name=target_regime_name,
        law_name=law_name,
        plans=transition_plans[target_regime_name],
        v_interpolation_info=v_interpolation_info,
        lottery_variables=lottery_variables,
        dependencies_by_law=dependencies_by_law,
        allowed_inputs=allowed_inputs,
        co_map_state_names=co_map_state_names,
    )

    base_reads = _reads(functions=functions, bundle=bundle, targets=(law.base,))
    if base_reads & set(lottery_variables):
        _fail(
            target_regime_name=target_regime_name,
            reason=(
                f"has a base '{law.base}' that reads the draws "
                f"{sorted(base_reads & set(lottery_variables))}"
            ),
        )

    shock_args = get_union_of_args([_shock_function(functions=functions, law=law)])
    conditioners = tuple(
        (name, support)
        for name, support in law.conditioners.items()
        if name in shock_args
    )
    shock_param_names = tuple(
        sorted(shock_args - set(law.conditioners) - set(lottery_variables))
    )
    if allowed_inputs is not None:
        unexplained = sorted(set(shock_param_names) - allowed_inputs - _TIME_NAMES)
        if unexplained:
            _fail(
                target_regime_name=target_regime_name,
                reason=(
                    f"reads the source's {unexplained} through '{law.shock}'; declare "
                    "each such input, or a function of it, as a conditioner with its "
                    "finite support"
                ),
            )
    function_conditioners = tuple(name for name, _ in conditioners if name in functions)
    conditioner_reads = _reads(
        functions=functions, bundle=bundle, targets=function_conditioners
    )
    if conditioner_reads & set(lottery_variables):
        _fail(
            target_regime_name=target_regime_name,
            reason=(
                f"has conditioners {list(function_conditioners)} that read the draws "
                f"{sorted(conditioner_reads & set(lottery_variables))}"
            ),
        )
    return ShockAveragePlan(
        law=law,
        law_name=law_name,
        state_name=state_name,
        conditioners=conditioners,
        function_conditioners=function_conditioners,
        integrated_draws=integrated,
        kept_draws=kept,
        shock_param_names=shock_param_names,
    )


def _split_shock_draws(
    *,
    target_regime_name: RegimeName,
    law_name: TransitionFunctionName,
    plans: TargetTransitionPlan,
    v_interpolation_info: VInterpolationInfo,
    lottery_variables: tuple[TransitionFunctionName, ...],
    dependencies_by_law: Mapping[TransitionFunctionName, frozenset[str]],
    allowed_inputs: frozenset[str] | None,
    co_map_state_names: tuple[StateName, ...],
) -> tuple[tuple[TransitionFunctionName, ...], tuple[TransitionFunctionName, ...]]:
    """Split the draws a shock reads into those averaged and those kept as axes.

    Returns:
        Tuple of the averaged draws and the kept draws, in lottery order.

    Raises:
        ModelInitializationError: If a kept draw has no nodes to index, or an
            averaged draw is read by another law too.

    """
    read_draws = dependencies_by_law.get(law_name, frozenset())
    # A draw the transition alone makes, with probabilities no source point
    # moves, is averaged into the stored value. Every other draw the shock reads
    # keeps its node axis, and the averaged value gets one axis per node.
    integrated = tuple(
        name
        for name in lottery_variables
        if name in read_draws
        and plans.lotteries[name].lifetime is LotteryLifetime.TRANSITION_LOCAL
        and _reads_only(
            func=plans.lotteries[name].probabilities, allowed_inputs=allowed_inputs
        )
    )
    kept = tuple(
        name
        for name in lottery_variables
        if name in read_draws and name not in integrated
    )
    for name in kept:
        state = name.removeprefix("next_")
        is_axis = (
            plans.lotteries[name].lifetime is not LotteryLifetime.TRANSITION_LOCAL
            and state in v_interpolation_info.discrete_states
            and state not in co_map_state_names
        )
        is_local = (
            plans.lotteries[name].lifetime is LotteryLifetime.TRANSITION_LOCAL
            and plans.lotteries[name].support_provider_name is not None
            and _reads_only(
                func=plans.lotteries[name].support_provider,
                allowed_inputs=allowed_inputs,
            )
        )
        if not (is_axis or is_local):
            _fail(
                target_regime_name=target_regime_name,
                reason=(
                    f"reads the draw '{name}', which is neither an axis of the "
                    "target's value nor drawn on nodes no source point moves"
                ),
            )
    for other, draws in dependencies_by_law.items():
        shared = sorted(draws & set(integrated))
        if other != law_name and shared:
            _fail(
                target_regime_name=target_regime_name,
                reason=(
                    f"averages {shared} into the stored value, but '{other}' reads "
                    "them too, so the draw would no longer be shared"
                ),
            )

    return integrated, kept


def get_shock_average_reader(
    *,
    plan: ShockAveragePlan,
    functions: EconFunctionsMapping,
    target_regime_name: RegimeName,
    transition_plans: TargetTransitionPlans,
    v_interpolation_info: VInterpolationInfo,
    co_map_state_names: tuple[StateName, ...],
    V_arr_name: str,
) -> _ShockAverageReader:
    """Build the reader of one target's value averaged over its shock.

    Args:
        plan: How the target averages over its shock.
        functions: Immutable mapping of function names to internal user functions.
        target_regime_name: Regime the continuation leads into.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.
        v_interpolation_info: The target's V-interpolation info.
        co_map_state_names: States whose axes are sliced off the value array.
        V_arr_name: Name under which the target's value array arrives.

    Returns:
        The reader, called at one node of the target's remaining lottery axes.

    """
    plans = transition_plans[target_regime_name]
    grid = v_interpolation_info.continuous_states[plan.state_name]
    runtime_points = isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime
    n_points = grid.n_points if runtime_points else len(grid.to_jax())
    n_nodes = int(
        np.prod(
            [
                plans.lotteries[name].support_signature.size
                for name in plan.integrated_draws
            ]
        )
    )
    array_state_names = tuple(
        name
        for name in v_interpolation_info.state_names
        if name not in co_map_state_names
    )
    averaged_name = f"averaged_{plan.state_name}"
    averaged_coordinate = f"next_{averaged_name}"
    averaged_points = qname_from_tree_path((averaged_name, "points"))
    condition_index_names = tuple(f"__{name}_index__" for name, _ in plan.conditioners)
    # The averaged value carries the conditioners' axes ahead of the target's
    # own, all read by index, so one gather picks the cell a read needs.
    funcs: dict[str, Callable[..., FloatND]] = {
        "__interpolation_data__": _get_lookup_function(
            array_name=V_arr_name,
            axis_names=[
                *condition_index_names,
                *(
                    name
                    for name in plan.kept_draws
                    if plans.lotteries[name].lifetime
                    is LotteryLifetime.TRANSITION_LOCAL
                ),
                *(
                    f"next_{name}"
                    for name in array_state_names
                    if name in v_interpolation_info.discrete_states
                ),
            ],
        )
    }
    continuous_coordinates = []
    for name in array_state_names:
        if name == plan.state_name:
            funcs[f"__{averaged_name}_coord__"] = _get_coordinate_finder(
                in_name=averaged_coordinate,
                grid=IrregSpacedGrid(n_points=n_points * n_nodes + _N_ANCHORS),
            )
            continuous_coordinates.append(f"__{averaged_name}_coord__")
        elif name in v_interpolation_info.continuous_states:
            funcs[f"__{name}_coord__"] = _get_coordinate_finder(
                in_name=f"next_{name}",
                grid=v_interpolation_info.continuous_states[name],
            )
            continuous_coordinates.append(f"__{name}_coord__")
    funcs["__fval__"] = _get_interpolator(
        name_of_values_on_grid="__interpolation_data__",
        axis_names=continuous_coordinates,
    )
    inner = concatenate_functions(
        functions=funcs, targets="__fval__", set_annotations=True
    )
    inner_args = frozenset(get_union_of_args([inner]))
    points_param = (
        qname_from_tree_path((plan.state_name, "points")) if runtime_points else None
    )
    integrated = tuple(
        _IntegratedDraw(
            name=name,
            weight_name=plans.lotteries[name].weight_name,
            support_name=_support_name(
                target_regime_name=target_regime_name,
                name=name,
                support_name=plans.lotteries[name].support_provider_name,
            ),
        )
        for name in plan.integrated_draws
    )
    kept = tuple(
        _KeptDraw(
            name=name,
            state_name=None,
            size=plans.lotteries[name].support_signature.size,
            node_values=None,
            support_name=plans.lotteries[name].support_provider_name,
        )
        if plans.lotteries[name].lifetime is LotteryLifetime.TRANSITION_LOCAL
        else _KeptDraw(
            name=name,
            state_name=name.removeprefix("next_"),
            size=len(
                v_interpolation_info.discrete_states[
                    name.removeprefix("next_")
                ].to_jax()
            ),
            node_values=v_interpolation_info.discrete_states[
                name.removeprefix("next_")
            ].to_jax(),
            support_name=None,
        )
        for name in plan.kept_draws
    )
    arg_names = sorted(
        (inner_args - {averaged_coordinate, averaged_points, *condition_index_names})
        | {plan.law.base, *(name for name, _ in plan.conditioners)}
        | {draw.weight_name for draw in integrated}
        | {draw.support_name for draw in integrated}
        | {draw.support_name for draw in kept if draw.support_name is not None}
        | {draw.name for draw in kept}
        | set(plan.shock_param_names)
        | ({points_param} if points_param is not None else set())
    )
    return _ShockAverageReader(
        inner=inner,
        inner_args=inner_args,
        V_arr_name=V_arr_name,
        averaged_coordinate=averaged_coordinate,
        averaged_points=averaged_points,
        condition_index_names=condition_index_names,
        base_name=plan.law.base,
        array_state_names=array_state_names,
        state_name=plan.state_name,
        find_coordinate=_get_coordinate_finder(
            in_name=f"next_{plan.state_name}", grid=grid
        ),
        grid_points=None if runtime_points else grid.to_jax(),
        points_param=points_param,
        conditioners=plan.conditioners,
        kept=kept,
        integrated=integrated,
        shock=_shock_function(functions=functions, law=plan.law),
        shock_param_names=plan.shock_param_names,
        arg_names=tuple(arg_names),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _IntegratedDraw:
    """A transition-local draw averaged into the stored value."""

    name: TransitionFunctionName
    """The draw's `next_<state>` name, under which the shock reads its value."""
    weight_name: str
    """Argument carrying the draw's node probabilities."""
    support_name: str
    """Argument carrying the draw's node values."""


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _KeptDraw:
    """A stored draw the shock reads, an axis of the target's value."""

    name: TransitionFunctionName
    """The draw's `next_<state>` name; its argument is the node index."""
    state_name: StateName | None
    """The target state whose axis the draw indexes, or `None` for a draw the
    transition alone makes, which gets an axis of its own."""
    size: int
    """The number of the draw's nodes."""
    node_values: Float1D | Int1D | None
    """A stored draw's node values (category codes or process nodes)."""
    support_name: str | None
    """Argument carrying a transition-local draw's node values."""


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class _ShockAverageReader:
    """Read a target's value averaged over an additive shock, at one node."""

    inner: Callable[..., FloatND]
    """Interpolator of the averaged value on the merged points."""
    inner_args: frozenset[ReferenceName]
    """Every argument `inner` reads."""
    V_arr_name: str
    """Argument carrying the target's value array."""
    averaged_coordinate: str
    """Argument of `inner` taking the base."""
    averaged_points: str
    """Argument of `inner` taking the merged points."""
    condition_index_names: tuple[str, ...]
    """Arguments of `inner` taking each conditioner's index in its support."""
    base_name: str
    """Argument carrying the base."""
    array_state_names: tuple[StateName, ...]
    """States of the received value array, in axis order."""
    state_name: StateName
    """The state the shock is added to."""
    find_coordinate: Callable[..., FloatND]
    """The coordinate finder of the state's grid."""
    grid_points: Float1D | None
    """The state's grid points, or `None` when they arrive at runtime."""
    points_param: str | None
    """Argument carrying the state's runtime grid points, if any."""
    conditioners: tuple[tuple[str, tuple[int | bool, ...]], ...]
    """The shock's conditioners with their supports."""
    kept: tuple[_KeptDraw, ...]
    """Stored draws the shock reads."""
    integrated: tuple[_IntegratedDraw, ...]
    """Transition-local draws averaged into the stored value."""
    shock: Callable[..., FloatND]
    """The shock as a function of conditioners, draws and parameters."""
    shock_param_names: tuple[QualifiedName, ...]
    """What the shock reads besides its conditioners and the draws."""
    arg_names: tuple[ReferenceName, ...]
    """The published argument names."""

    def __post_init__(self) -> None:
        _publish_signature(
            target=self,
            args=dict.fromkeys(self.arg_names, "FloatND"),
            return_annotation="FloatND",
            name="read_shock_average",
        )

    @no_type_check
    def __call__(self, **kwargs: EconFunctionArg) -> FloatND:
        averaged, knots = self._average(kwargs)
        condition_index = []
        known = jnp.ones((), dtype=bool)
        for name, support in self.conditioners:
            matches = jnp.asarray(kwargs[name]) == jnp.asarray(support)
            condition_index.append(jnp.argmax(matches).astype(jnp.int32))
            known = known & jnp.any(matches)
        kept_index = [
            jnp.asarray(kwargs[draw.name]).astype(jnp.int32) for draw in self.kept
        ]
        inner_kwargs = {
            name: kwargs[name] for name in self.inner_args if name in kwargs
        }
        inner_kwargs[self.V_arr_name] = averaged
        inner_kwargs[self.averaged_coordinate] = kwargs[self.base_name]
        inner_kwargs[self.averaged_points] = knots[(*condition_index, *kept_index)]
        inner_kwargs.update(
            zip(self.condition_index_names, condition_index, strict=True)
        )
        return jnp.where(known, self.inner(**inner_kwargs), jnp.nan)

    @no_type_check
    def _average(
        self, kwargs: Mapping[ReferenceName, EconFunctionArg]
    ) -> tuple[FloatND, FloatND]:
        """Return the averaged value and its merged points for every combination.

        The averaged value carries the conditioners' supports as leading axes,
        then the received value array's axes with the shock's state replaced by
        the merged points. The merged points carry the conditioners' axes and
        one axis per stored draw the shock reads. Neither depends on the source
        point, so under the point map both are formed once.
        """
        values = kwargs[self.V_arr_name]
        fixed = {name: kwargs[name] for name in self.shock_param_names}

        supports = [support for _, support in self.conditioners]
        combos = list(itertools.product(*supports))
        # Integer codes take the engine's integer dtype, booleans stay booleans,
        # matching the values the conditioners take at a source point.
        condition_values = {
            name: jnp.asarray(
                [combo[i] for combo in combos],
                dtype=bool if isinstance(support[0], bool) else jnp.int32,
            )
            for i, (name, support) in enumerate(self.conditioners)
        }
        kept_sizes = [draw.size for draw in self.kept]
        kept_combos = list(itertools.product(*(range(n) for n in kept_sizes)))
        kept_values = {
            draw.name: (
                draw.node_values
                if draw.support_name is None
                else kwargs[draw.support_name]
            )[np.asarray([combo[i] for combo in kept_combos], dtype=np.int32)]
            for i, draw in enumerate(self.kept)
        }
        if self.integrated:
            node_values = jnp.meshgrid(
                *(kwargs[draw.support_name] for draw in self.integrated),
                indexing="ij",
            )
            node_weights = jnp.meshgrid(
                *(kwargs[draw.weight_name] for draw in self.integrated),
                indexing="ij",
            )
            draw_values = {
                draw.name: grid.ravel()
                for draw, grid in zip(self.integrated, node_values, strict=True)
            }
            weights = jnp.prod(jnp.stack([w.ravel() for w in node_weights]), axis=0)
        else:
            draw_values = {}
            weights = jnp.ones(1, dtype=values.dtype)

        def shock_at(index: Mapping[str, ScalarInt]) -> FloatND:
            return self.shock(
                **{name: v[index["c"]] for name, v in condition_values.items()},
                **{name: v[index["j"]] for name, v in kept_values.items()},
                **{name: v[index["k"]] for name, v in draw_values.items()},
                **fixed,
            )

        shape = (len(combos), len(kept_combos), weights.shape[0])
        grid = jnp.meshgrid(*(jnp.arange(n) for n in shape), indexing="ij")
        shocks = jax.vmap(shock_at)(
            {key: axis.ravel() for key, axis in zip("cjk", grid, strict=True)}
        ).reshape(shape)
        points = (
            kwargs[self.points_param]
            if self.points_param is not None
            else self.grid_points
        )
        points_kwargs = (
            {self.points_param: points} if self.points_param is not None else {}
        )

        def coordinate(value: FloatND) -> FloatND:
            return self.find_coordinate(
                **{f"next_{self.state_name}": value}, **points_kwargs
            )

        def matrix(shocks_row: Float1D) -> tuple[Float1D, FloatND]:
            return shock_averaging_matrix(
                points=points, coordinate=coordinate, shocks=shocks_row, weights=weights
            )

        knots, averaging = jax.vmap(jax.vmap(matrix))(shocks)
        condition_sizes = [len(support) for support in supports]
        n_knots = knots.shape[-1]
        knots = knots.reshape(*condition_sizes, *kept_sizes, n_knots)
        averaging = jnp.moveaxis(averaging, 2, 0).reshape(
            2, *condition_sizes, *kept_sizes, len(points), n_knots
        )
        return (
            apply_averaging(
                values=values,
                averaging=averaging,
                subscripts=self._subscripts(n_conditions=len(condition_sizes)),
            ),
            knots,
        )

    def _subscripts(self, *, n_conditions: int) -> str:
        """Return the contraction of the value array with one averaging slab."""
        letters = iter("abcdefghijklopqrstuvwxyABCDEFGHIJKLOPQRSTUVWXY")
        condition_letters = "".join(next(letters) for _ in range(n_conditions))
        axis_letters = {name: next(letters) for name in self.array_state_names}
        axis_letters[self.state_name] = "n"
        value_letters = "".join(axis_letters[name] for name in self.array_state_names)
        kept_letters = "".join(
            axis_letters[draw.state_name]
            if draw.state_name is not None
            else next(letters)
            for draw in self.kept
        )
        local_letters = "".join(
            letter
            for letter, draw in zip(kept_letters, self.kept, strict=True)
            if draw.state_name is None
        )
        output_letters = (
            condition_letters + local_letters + value_letters.replace("n", "m")
        )
        return f"{value_letters},{condition_letters}{kept_letters}nm->{output_letters}"


def _merged_knots(*, points: Float1D, shocks: Float1D) -> Float1D:
    """Return `{a_j - shock_k}` ascending, framed by two anchors a span away.

    Below the smallest merged point every term of the average extrapolates along
    the grid's first segment, and above the largest along its last, so the
    average is linear in both regions and a point placed there costs no
    exactness. Every coincident copy moves above the top. One anchor sits a
    span below the smallest point and one a span above the largest: a read
    beyond the merged points then extrapolates along a segment as long as the
    grid, not along the shortest gap between two merged points, whose rounded
    endpoint values would set the slope.
    """
    merged = jnp.sort((points[:, None] - shocks[None, :]).ravel())
    is_copy = jnp.concatenate([jnp.zeros(1, dtype=bool), merged[1:] <= merged[:-1]])
    bottom = merged[0]
    top = merged[-1]
    span = top - bottom + 1
    lifted = top + span * jnp.cumsum(is_copy).astype(merged.dtype)
    knots = jnp.sort(jnp.where(is_copy, lifted, merged))
    return jnp.concatenate([(bottom - span)[None], knots, (knots[-1] + span)[None]])


def _shock_function(
    *, functions: EconFunctionsMapping, law: AdditiveShockTransition
) -> Callable[..., FloatND]:
    """Return the shock as a function of its conditioners, draws and parameters."""
    return concatenate_functions(
        functions={
            name: func
            for name, func in functions.items()
            if name not in law.conditioners
        },
        targets=law.shock,
        enforce_signature=False,
        set_annotations=True,
    )


def _reads(
    *,
    functions: EconFunctionsMapping,
    bundle: Mapping[TransitionFunctionName, TransitionFunction],
    targets: tuple[FunctionName | TransitionFunctionName, ...],
) -> frozenset[str]:
    """Return every input the targets read, through helpers."""
    if not targets:
        return frozenset()
    func = concatenate_functions(
        functions=dict(bundle) | dict(functions),
        targets=list(targets),
        return_type="dict",
        enforce_signature=False,
    )
    return frozenset(inspect.signature(func).parameters)


def _reads_only(
    *, func: UserFunction | None, allowed_inputs: frozenset[str] | None
) -> bool:
    """Return whether a function reads nothing but parameters and time."""
    if allowed_inputs is None or not callable(func):
        return False
    return get_union_of_args([func]) <= allowed_inputs | _TIME_NAMES


def _support_name(
    *, target_regime_name: RegimeName, name: str, support_name: str | None
) -> str:
    """Return the argument carrying a transition-local draw's node values."""
    if support_name is None:
        _fail(
            target_regime_name=target_regime_name,
            reason=(
                f"averages the draw '{name}', whose node values no function supplies"
            ),
        )
    return support_name


def _fail(*, target_regime_name: RegimeName, reason: str) -> NoReturn:
    msg = (
        f"The additive shock transition toward regime '{target_regime_name}' "
        f"{reason}. The average over the shock is exact only when the law is "
        "`base + shock`, the base reads no draw, and the shock reads nothing but "
        "its declared conditioners, next period's draws, the period or age, and "
        "parameters."
    )
    raise ModelInitializationError(msg)
