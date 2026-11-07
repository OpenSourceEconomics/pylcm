"""Sum a target's pure-index draws into its value array once per kernel call.

A target's continuation at one state-action point averages its value function
over the joint nodes of every draw toward it:

$$E[V'] = \\sum_{d} p(d \\mid s, a)\\, V'(g(s, a, d), d).$$

Some draws only select a slice of $V'$. A draw `next_<state>` qualifies when

- its weights read no state or action of the source other than `<state>` itself,
  so they are the same at every cell sharing a value of `<state>`;
- no law of the target reads the draw, so it moves no interpolation coordinate;
- it is a genuine draw onto the target's grid of `<state>`, persisted as that
  state, and the source carries `<state>` on a grid with static nodes: a
  stochastic process whose nodes come from runtime parameters keeps its axis;
- no later stage rebinds the target's reader, as a participation check does.

Weights that do not read `<state>` itself, as an IID draw's, give every row the
same expectation; the continuation then reads the first row and no current value,
so it stays constant along `<state>`.

For such a draw the expectation over its nodes commutes with the read of every
other coordinate: interpolation is linear in the node values, so

$$\\sum_{z'} P(z' \\mid z)\\, \\operatorname{interp}(V'(\\cdot, z'), y)
= \\operatorname{interp}(\\tilde V(\\cdot, z), y), \\qquad
\\tilde V(\\cdot, z) = \\sum_{z'} P(z' \\mid z)\\, V'(\\cdot, z').$$

The contracted table $\\tilde V$ replaces the draw's axis of $V'$ by one row per
node of the SOURCE's grid of `<state>`, which may differ from the target's. The
continuation then reads it at the current node instead of mapping over the draw,
so the sum over $z'$ leaves the action loop. Each row is reduced with
`zero_safe_average` over weights scaled by `scaled_joint_weight`, the arithmetic
the per-node expectation uses, so a node of zero probability still drops out
rather than multiplying a non-finite value. The mean is held between its
smallest and largest live value, so a draw summed on its own at near-maximal
values stays as finite as the joint expectation it replaces.

Only the linear expectation reduces per target, and only there does a
per-draw sum commute with the certainty equivalent; the caller enables the
contraction on that route alone.
"""

import dataclasses
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import no_type_check

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.grids import Grid
from _lcm.probability import is_live
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.next_state import get_next_stochastic_weights_function
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.regime_building.zero_safe import zero_safe_average
from _lcm.transition_plans import (
    LotteryIndexCoordinate,
    LotteryLifetime,
    PhysicalCoordinate,
    SupportOrigin,
    TargetTransitionPlan,
    TargetTransitionPlans,
)
from _lcm.typing import (
    EconFunctionArg,
    EconFunctionsMapping,
    QAndFKwargs,
    RegimeName,
    StateName,
    TransitionFunction,
    TransitionFunctionName,
)
from _lcm.zero_safe import scaled_joint_weight
from lcm.exceptions import GridInitializationError
from lcm.grids import DiscreteGrid
from lcm.typing import (
    ActionName,
    ContinuousState,
    DiscreteState,
    FloatND,
    IntND,
    ReferenceName,
)

type StochasticWeightsFunction = Callable[..., dict[str, FloatND | IntND]]


@dataclass(frozen=True, kw_only=True)
class LotteryContractionSource:
    """What the source regime contributes to deciding a contraction."""

    state_grids: MappingProxyType[StateName, Grid]
    """Immutable mapping of the source's state names to their grids."""

    action_names: frozenset[ActionName]
    """The source's action names."""

    rebound_targets: frozenset[RegimeName] = frozenset()
    """Targets whose continuation a later stage rebinds around the built reader.

    A participation check replaces the reader of the continuation it is built
    on, so that continuation must keep every draw as a node axis.
    """


@dataclass(frozen=True, kw_only=True, eq=False)
class ContractedLottery:
    """One draw summed into its target's value array."""

    state_name: StateName
    """The drawn state, current and next."""

    law_name: TransitionFunctionName
    """The draw's `next_<state>` law."""

    weight_name: str
    """The `weight_<target>__next_<state>` output of `weights`."""

    axis: int
    """Position of the state's axis in the value array as the target receives it."""

    source_nodes: tuple[int, ...] | tuple[float, ...]
    """The values the current state takes on the source's grid, in node order."""

    is_discrete: bool
    """Whether the state is discrete, its nodes being its integer codes; a
    stochastic process otherwise."""

    weights: StochasticWeightsFunction
    """The draw's marginal over the target's nodes, at one current value."""

    weight_arg_names: tuple[ReferenceName, ...]
    """The arguments `weights` reads: the state, the period, the age, parameters."""

    @property
    def reads_state(self) -> bool:
        """Whether the weights read the drawn state's current value.

        Weights that do not read it, as an IID draw's, give every row of the table
        the same expectation, so one row stands for all of them.
        """
        return self.state_name in self.weight_arg_names

    @property
    def source_node_array(self) -> FloatND | IntND:
        """The source nodes as the cell holds the state: codes as `int32`."""
        if self.is_discrete:
            return jnp.asarray(self.source_nodes, dtype=jnp.int32)
        return jnp.asarray(self.source_nodes)

    def contract(
        self, *, values: FloatND, states_actions_params: QAndFKwargs
    ) -> FloatND:
        """Replace the draw's axis of `values` by one expected row per source node.

        Args:
            values: The target's value array, the draw's axis at `axis`.
            states_actions_params: The kernel's arguments, supplying everything
                `weights` reads besides the state.

        Returns:
            The array with the axis at `axis` indexed by the source's nodes.

        """
        invariant = {
            name: states_actions_params[name]
            for name in self.weight_arg_names
            if name != self.state_name
        }
        if not self.reads_state:
            # Every row is the same expectation: compute it once and broadcast
            # it, so the table still has one row per source node without the
            # rows being stored apart.
            row = self.weights(**invariant)[self.weight_name]
            single = _contract_rows(
                values=values, rows=jnp.expand_dims(row, 0), axis=self.axis
            )
            shape = list(single.shape)
            shape[self.axis] = len(self.source_nodes)
            return jnp.broadcast_to(single, tuple(shape))

        def _row(node: EconFunctionArg) -> FloatND:
            return self.weights(**invariant, **{self.state_name: node})[
                self.weight_name
            ]

        rows = jax.vmap(_row)(self.source_node_array)
        return _contract_rows(values=values, rows=rows, axis=self.axis)


@dataclass(frozen=True, kw_only=True, eq=False)
class LotteryContraction:
    """Every draw contracted into one target's value array."""

    lotteries: tuple[ContractedLottery, ...]
    """The contracted draws, in the target's law order."""

    def __call__(
        self, *, values: FloatND, states_actions_params: QAndFKwargs
    ) -> FloatND:
        """Contract every draw of `lotteries` into `values`.

        Args:
            values: The target's value array as the continuation receives it.
            states_actions_params: The kernel's arguments.

        Returns:
            The contracted value array, read by the rewritten continuation.

        """
        for lottery in self.lotteries:
            values = lottery.contract(
                values=values, states_actions_params=states_actions_params
            )
        return values

    @property
    def weight_functions(self) -> tuple[StochasticWeightsFunction, ...]:
        """The weight functions whose arguments the kernel must carry."""
        return tuple(lottery.weights for lottery in self.lotteries)

    def rewrite_bundle(
        self, *, bundle: MappingProxyType[TransitionFunctionName, TransitionFunction]
    ) -> MappingProxyType[TransitionFunctionName, TransitionFunction]:
        """Return `bundle` with each contracted draw read at the current node.

        Args:
            bundle: The target's `next_<state>` laws.

        Returns:
            The laws, each contracted draw replaced by the current node's index.

        """
        replaced = {
            lottery.law_name: _current_node_index(lottery=lottery)
            for lottery in self.lotteries
        }
        return MappingProxyType(
            {name: replaced.get(name, law) for name, law in bundle.items()}
        )

    def rewrite_plans(
        self,
        *,
        target_regime_name: RegimeName,
        transition_plans: TargetTransitionPlans,
        bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    ) -> TargetTransitionPlans:
        """Return `transition_plans` with the contracted draws no longer lotteries.

        Args:
            target_regime_name: The target whose plan is rewritten.
            transition_plans: Immutable mapping of targets to their plans.
            bundle: The rewritten laws of the target.

        Returns:
            The plans, the target's contracted draws turned into deterministic
            outputs read off the rewritten laws.

        """
        plan = transition_plans[target_regime_name]
        contracted = {lottery.law_name: lottery for lottery in self.lotteries}
        outputs = {
            name: dataclasses.replace(
                output,
                continuation_coordinate=PhysicalCoordinate(),
                physical_resolver=bundle[output.next_state_name],
                lottery_dependencies=frozenset(),
            )
            if output.next_state_name in contracted
            else output
            for name, output in plan.outputs.items()
        }
        rewritten = dataclasses.replace(
            plan,
            lotteries=MappingProxyType(
                {
                    name: info
                    for name, info in plan.lotteries.items()
                    if name not in contracted
                }
            ),
            outputs=MappingProxyType(outputs),
        )
        return MappingProxyType({**transition_plans, target_regime_name: rewritten})

    def rewrite_v_interpolation_info(
        self,
        *,
        v_interpolation_info: VInterpolationInfo,
        source: LotteryContractionSource,
    ) -> VInterpolationInfo:
        """Return the target's info with each contracted axis on the source's grid.

        Args:
            v_interpolation_info: The target's V-interpolation info.
            source: The source regime's grids.

        Returns:
            The info the contracted table is read with.

        """
        return dataclasses.replace(
            v_interpolation_info,
            discrete_states=MappingProxyType(
                {
                    **v_interpolation_info.discrete_states,
                    **{
                        lottery.state_name: source.state_grids[lottery.state_name]
                        for lottery in self.lotteries
                    },
                }
            ),
        )


def plan_lottery_contraction(
    *,
    target_regime_name: RegimeName,
    bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    functions: EconFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    v_interpolation_info: VInterpolationInfo,
    received_state_names: tuple[StateName, ...],
    read_draws: frozenset[TransitionFunctionName],
    source: LotteryContractionSource,
) -> LotteryContraction | None:
    """Return the draws toward one target that can be summed into its value array.

    Args:
        target_regime_name: The target regime.
        bundle: The target's unqualified `next_<state>` laws.
        functions: Immutable mapping of the continuation's function pool.
        transition_plans: Immutable mapping of targets to their plans.
        v_interpolation_info: The target's V-interpolation info.
        received_state_names: The target's value axes as the continuation
            receives them, after co-mapped axes are sliced off.
        read_draws: The target's draws that some law reads, directly or through
            helpers, so that they move a coordinate.
        source: The source regime's state grids and action names.

    Returns:
        The contraction, or `None` when no draw qualifies.

    """
    plan = transition_plans[target_regime_name]
    available = MappingProxyType(
        {name: law for name, law in bundle.items() if not plan.is_lottery(name)}
    )
    lotteries = tuple(
        lottery
        for law_name in bundle
        if plan.is_lottery(law_name)
        and (
            lottery := _plan_one_lottery(
                target_regime_name=target_regime_name,
                law_name=law_name,
                bundle=bundle,
                available=available,
                functions=functions,
                transition_plans=transition_plans,
                plan=plan,
                v_interpolation_info=v_interpolation_info,
                received_state_names=received_state_names,
                read_draws=read_draws,
                source=source,
            )
        )
        is not None
    )
    return LotteryContraction(lotteries=lotteries) if lotteries else None


def _plan_one_lottery(
    *,
    target_regime_name: RegimeName,
    law_name: TransitionFunctionName,
    bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    available: MappingProxyType[TransitionFunctionName, TransitionFunction],
    functions: EconFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    plan: TargetTransitionPlan,
    v_interpolation_info: VInterpolationInfo,
    received_state_names: tuple[StateName, ...],
    read_draws: frozenset[TransitionFunctionName],
    source: LotteryContractionSource,
) -> ContractedLottery | None:
    """Return one draw's contraction, or `None` when a condition fails."""
    info = plan.lotteries[law_name]
    state_name = info.persisted_state
    if (
        state_name is None
        or info.lifetime is not LotteryLifetime.PERSISTED_STATE
        or info.support_origin is not SupportOrigin.TARGET_GRID
        or info.original_layout is not None
        or state_name not in received_state_names
        or state_name not in v_interpolation_info.discrete_states
        or not _is_pure_index(plan=plan, law_name=law_name, state_name=state_name)
        or law_name in read_draws
    ):
        return None
    source_grid = source.state_grids.get(state_name)
    source_nodes = _static_source_nodes(
        source_grid=source_grid,
        target_grid=v_interpolation_info.discrete_states[state_name],
    )
    if source_nodes is None:
        return None
    weights = get_next_stochastic_weights_function(
        regime_name=target_regime_name,
        functions=functions,
        transitions=MappingProxyType({**available, law_name: bundle[law_name]}),
        transition_plans=transition_plans,
    )
    weight_arg_names = tuple(inspect.signature(weights).parameters)
    cell_inputs = (set(source.state_grids) | source.action_names) - {state_name}
    if any(
        name in cell_inputs or name.startswith("next_") for name in weight_arg_names
    ):
        return None
    return ContractedLottery(
        state_name=state_name,
        law_name=law_name,
        weight_name=info.weight_name,
        axis=received_state_names.index(state_name),
        source_nodes=source_nodes,
        is_discrete=isinstance(source_grid, DiscreteGrid),
        weights=weights,
        weight_arg_names=weight_arg_names,
    )


def _is_pure_index(
    *,
    plan: TargetTransitionPlan,
    law_name: TransitionFunctionName,
    state_name: StateName,
) -> bool:
    """Return whether the draw's only consumer is its own state's coordinate."""
    output = plan.outputs.get(state_name)
    if output is None or not (
        isinstance(output.continuation_coordinate, LotteryIndexCoordinate)
        and output.continuation_coordinate.lottery_name == law_name
    ):
        return False
    return not any(
        law_name in other.lottery_dependencies
        or state_name in other.output_dependencies
        for name, other in plan.outputs.items()
        if name != state_name
    )


def _contract_rows(*, values: FloatND, rows: FloatND, axis: int) -> FloatND:
    """Replace `axis` of `values` by one weighted mean per row of `rows`.

    Args:
        values: The value array, the draw's target nodes along `axis`.
        rows: One weight vector over the target nodes per output row.
        axis: The draw's axis.

    Returns:
        The array with `axis` holding one mean per row of `rows`.

    """
    moved = jnp.moveaxis(values, axis, -1)
    last = moved.ndim - 1

    def _average(row: FloatND) -> FloatND:
        # Each node's weight carries its own base-two scale, exactly as a node of
        # the joint lottery does, so a rare node against a large value and an
        # ordinary node against a near-maximal one both stay inside the format.
        coefficients, shifts = jax.vmap(_scaled_node_weight)(row)
        mean = zero_safe_average(
            a=moved, weights=coefficients, shifts=shifts, axis=last
        )
        # A mean lies between its smallest and largest live value. The rounded
        # sum of near-maximal terms can land past the largest finite value, which
        # the joint lottery never forms on its own, since there every term also
        # carries the other draws' weights. Holding the mean to its exact bounds
        # removes only that rounding; a NaN or an infinity at a live node passes
        # through unchanged.
        live = is_live(row)
        lower = jnp.min(jnp.where(live, moved, jnp.inf), axis=last)
        upper = jnp.max(jnp.where(live, moved, -jnp.inf), axis=last)
        return jnp.minimum(jnp.maximum(mean, lower), upper)

    contracted = jax.vmap(_average, out_axes=-1)(rows)
    return jnp.moveaxis(contracted, -1, axis)


def _scaled_node_weight(weight: FloatND) -> tuple[FloatND, IntND]:
    """Return one node's weight as a coefficient and its base-two scale."""
    return scaled_joint_weight(jnp.expand_dims(weight, 0))


def _static_source_nodes(
    *, source_grid: Grid | None, target_grid: Grid
) -> tuple[int, ...] | tuple[float, ...] | None:
    """Return the source grid's node values, or `None` when they are not static.

    A discrete grid's codes must be its node positions, and a stochastic process
    must not take its nodes from runtime parameters: at build time those nodes are
    unknown, held as non-finite placeholders or refused outright. Source and target
    must be grids of the same kind, so the contracted axis keeps the target's reader.
    """
    if isinstance(source_grid, DiscreteGrid) and isinstance(target_grid, DiscreteGrid):
        codes = np.asarray(source_grid.to_jax())
        if not np.array_equal(codes, np.arange(codes.size)):
            return None
        return tuple(int(code) for code in codes)
    if isinstance(source_grid, _ContinuousStochasticProcess) and isinstance(
        target_grid, _ContinuousStochasticProcess
    ):
        try:
            nodes = np.asarray(source_grid.to_jax())
        except GridInitializationError:
            return None
        # Nodes computed from runtime parameters are placeholders at build time.
        if not np.all(np.isfinite(nodes)):
            return None
        return tuple(float(node) for node in nodes)
    return None


def _current_node_index(*, lottery: ContractedLottery) -> TransitionFunction:
    """Return the law reading the contracted table at the current node."""
    if not lottery.reads_state:
        return _FirstNodeIndex(
            state_name=lottery.state_name, is_discrete=lottery.is_discrete
        )
    if lottery.is_discrete:
        return _CurrentNodeIndex(state_name=lottery.state_name, process_nodes=None)
    return _CurrentNodeIndex(
        state_name=lottery.state_name,
        process_nodes=tuple(float(node) for node in lottery.source_nodes),
    )


class _CurrentNodeIndex:
    """The node index of a contracted state's current value on the source's grid.

    A continuation reads a discrete or process axis by node index. A discrete
    state's code is its index. A process state takes exactly the values of its
    grid in the solve, so the nearest node is the node itself.
    """

    def __init__(
        self, *, state_name: StateName, process_nodes: tuple[float, ...] | None
    ) -> None:
        self._state_name = state_name
        self._process_nodes = process_nodes
        argument = DiscreteState if process_nodes is None else ContinuousState
        returned = DiscreteState if process_nodes is None else IntND
        self.__name__ = f"next_{state_name}"
        self.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    state_name, inspect.Parameter.KEYWORD_ONLY, annotation=argument
                )
            ],
            return_annotation=returned,
        )
        self.__annotations__ = {state_name: argument, "return": returned}

    @no_type_check
    def __call__(
        self, *args: EconFunctionArg, **kwargs: EconFunctionArg
    ) -> FloatND | IntND:
        del args
        value = kwargs[self._state_name]
        if self._process_nodes is None:
            return value
        nodes = jnp.asarray(self._process_nodes, dtype=jnp.asarray(value).dtype)
        return jnp.argmin(jnp.abs(nodes - value)).astype(jnp.int32)


class _FirstNodeIndex:
    """The first row of a table whose rows all coincide.

    Reads no state, so the continuation does not depend on the drawn state's
    current value and is not batched along its axis.
    """

    def __init__(self, *, state_name: StateName, is_discrete: bool) -> None:
        returned = DiscreteState if is_discrete else IntND
        self.__name__ = f"next_{state_name}"
        self.__signature__ = inspect.Signature([], return_annotation=returned)
        self.__annotations__ = {"return": returned}

    @no_type_check
    def __call__(
        self, *args: EconFunctionArg, **kwargs: EconFunctionArg
    ) -> FloatND | IntND:
        del args, kwargs
        return jnp.int32(0)
