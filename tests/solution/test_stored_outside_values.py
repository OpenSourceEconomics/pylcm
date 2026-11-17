"""Outside values a regime reads from its destination's stored nodes.

Each row of a query's information frame realises one node of every shock group the
frame enumerates, and the frame names that node. A stored outside value is gathered
at the node its caller's row names, so re-realising a group at the named node must
give back the row's physical shocks.

The stored array holds the destination's problem value at every grid cell and at
every node of the groups its readers observe, integrating the other groups.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import exit_queries, outside_values, policy_queries, query_information
from _lcm.current_iid import (
    CURRENT_IID_ARGUMENT,
    CURRENT_IID_NODES_ARGUMENT,
    AdmittedGroupParams,
    CurrentIIDDrawFrame,
    CurrentIIDPlan,
    NodeOperands,
)
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import CoreBuildContext
from _lcm.outside_values import outside_value_program
from _lcm.policy_queries import PolicyAnswer, QueryKind
from _lcm.query_information import InformationQueryCell, QueryArguments
from _lcm.solution.contract import ContinuationPayload
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PytreeValue, QAndFKwargs
from lcm import (
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Periods,
    Regime,
    Unilateral,
    categorical,
)
from lcm.shocks import (
    GaussHermite,
    Normal,
    Product,
    QuasiMonteCarlo,
    ShockGroup,
    ShockGroupName,
    ShockName,
)
from lcm.solver_api import KernelOutput
from lcm.typing import BoolND, FloatND, IntND, RegimeName, ScalarInt
from tests.conftest import DECIMAL_PRECISION, assert_agrees_to_ulp
from tests.solution.test_formation_read_classes import _model as _formation_model
from tests.solution.test_formation_read_classes import _solve

type _Label = tuple[str, str, str]
type _Tally = dict[_Label, np.ndarray]


def _group_physical(
    *, plan: CurrentIIDPlan, group: ShockGroupName, fixed: QueryArguments, codes: IntND
) -> Mapping[ShockName, FloatND | IntND]:
    """The physical shocks of `group` realised afresh at each of `codes`."""
    _, binding, nodes = next(entry for entry in plan.bindings if entry[0] == group)
    site = CurrentIIDPlan(bindings=((group, binding, nodes),), indices=codes[:, None])
    frame, _ = site.realise_information_nodes(
        arguments=cast("QAndFKwargs", fixed),
        group_params={
            group: cast("AdmittedGroupParams", fixed[CURRENT_IID_ARGUMENT])[group]
        },
        node_operands={
            group: cast("NodeOperands", fixed[CURRENT_IID_NODES_ARGUMENT])[group]
        },
        indices=site.indices,
    )
    return frame.physical


def _differs(*, got: FloatND | IntND, expected: FloatND | IntND) -> BoolND:
    """Whether two realisations of a shock name different values, beyond rounding."""
    if jnp.issubdtype(expected.dtype, jnp.integer):
        return got != expected
    spacing = 4 * jnp.finfo(expected.dtype).eps
    return jnp.abs(got - expected) > spacing * jnp.maximum(1.0, jnp.abs(expected))


def _check_codes(
    *,
    cell: InformationQueryCell,
    key: str,
    frame: CurrentIIDDrawFrame,
    codes: Mapping[ShockGroupName, IntND],
    previous: CurrentIIDDrawFrame | None,
    tally: _Tally,
) -> None:
    """Count rows whose named node does not realise the frame's physical shocks.

    A group the frame copies from its caller is realised at the caller's node, so
    only groups the frame enumerates itself are checked: its own groups none of
    whose shocks it observes, and the groups it observes through another query,
    where that observation is available. The same count at the next node is the
    control, which must find differences.
    """
    node = cell.definitions[key]
    observed = node.read.price_information & (
        frozenset() if previous is None else previous.physical.keys()
    )
    for group, code in codes.items():
        producer = node.observation_sources.get(group, key)
        target = cell.definitions[producer]
        plan = cast(
            "CurrentIIDPlan", target.context.current_iid_plans[target.read.period]
        )
        fixed = (
            cell.fixed
            if producer == cell.name
            else cell.information_arguments[producer]
        )
        shocks = next(
            binding.group.shocks for name, binding, _ in plan.bindings if name == group
        )
        if group not in node.observation_sources and shocks.keys() & observed:
            continue
        size = next(
            nodes.weights.size for name, _, nodes in plan.bindings if name == group
        )
        named = _group_physical(plan=plan, group=group, fixed=fixed, codes=code)
        shifted = _group_physical(
            plan=plan, group=group, fixed=fixed, codes=(code + 1) % size
        )
        compared = jnp.zeros((), jnp.int32)
        mismatched = jnp.zeros((), jnp.int32)
        control = jnp.zeros((), jnp.int32)
        for shock in shocks.keys() & frame.physical.keys():
            value = frame.physical[shock]
            rows = jnp.broadcast_to(
                frame.information_available.get(shock, jnp.asarray(a=True)), value.shape
            )
            compared += jnp.sum(rows)
            mismatched += jnp.sum(rows & _differs(got=named[shock], expected=value))
            control += jnp.sum(rows & _differs(got=shifted[shock], expected=value))
        label = (
            node.read.kind.name,
            node.read.target,
            "observation" if group in node.observation_sources else "own",
        )
        jax.debug.callback(
            partial(_record, tally=tally, label=label), compared, mismatched, control
        )


# keyword-only-exempt: library-callback=jax.debug.callback
def _record(
    compared: IntND,
    mismatched: IntND,
    control: IntND,
    *,
    tally: _Tally,
    label: _Label,
) -> None:
    tally[label] += np.asarray(
        [np.sum(compared), np.sum(mismatched), np.sum(control)], dtype=np.int64
    )


def _record_frame_codes(*, monkeypatch: pytest.MonkeyPatch) -> _Tally:
    """Check every realised frame and the codes each row's descendants receive."""
    tally: _Tally = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    realise = InformationQueryCell._realise
    own = InformationQueryCell._own

    # keyword-only-exempt: primary-argument=self
    def checked_realise(
        self: InformationQueryCell,
        *,
        node: query_information.QueryNode,
        fixed: QueryArguments,
        source: query_information.SourceStates,
        frames: query_information.Frames,
        guard: BoolND,
    ) -> tuple[CurrentIIDDrawFrame, FloatND, Mapping[ShockGroupName, IntND]]:
        frame, factors, codes = realise(
            self, node=node, fixed=fixed, source=source, frames=frames, guard=guard
        )
        _check_codes(
            cell=self,
            key=next(name for name, other in self.definitions.items() if other is node),
            frame=frame,
            codes=codes,
            previous=frames[node.ancestors[-1]] if node.ancestors else None,
            tally=tally,
        )
        return frame, factors, codes

    # keyword-only-exempt: primary-argument=self
    def checked_own(
        self: InformationQueryCell,
        *,
        frames: query_information.Frames,
        codes: query_information.FrameCodes,
        indices: tuple[IntND, ...],
        live: BoolND,
        source: query_information.SourceStates,
        dependencies: query_information.Dependencies,
    ) -> PolicyAnswer:
        for position, key in enumerate(self.node.ancestors):
            _check_codes(
                cell=self,
                key=key,
                frame=jax.tree.map(lambda leaf: jnp.asarray(leaf)[None], frames[key]),
                codes={group: code[None] for group, code in codes[key].items()},
                previous=frames[self.node.ancestors[position - 1]]
                if position
                else None,
                tally=tally,
            )
        return own(
            self,
            frames=frames,
            codes=codes,
            indices=indices,
            live=live,
            source=source,
            dependencies=dependencies,
        )

    monkeypatch.setattr(InformationQueryCell, "_realise", checked_realise)
    monkeypatch.setattr(InformationQueryCell, "_own", checked_own)
    return tally


@pytest.mark.parametrize("draws", [False, True], ids=["enumerated", "drawn"])
def test_frame_codes_name_the_node_each_row_realises(
    *, draws: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-realising each group at the node a row names gives the row's shocks back.

    This holds for every row of each single's frame and of the couple's frame, for
    the observation axes a single crosses its rows with, and for the row each
    descendant receives from its ancestors. Shifting every code to the next node
    breaks it, so the check can fire.
    """
    tally = _record_frame_codes(monkeypatch=monkeypatch)
    _solve(
        _formation_model(
            dissolution=True, nodes=QuasiMonteCarlo(n=4, seed=0) if draws else None
        )
    )
    required = {
        ("FULL", "single_f", "own"),
        ("FULL", "single_f", "observation"),
        ("FULL", "single_m", "own"),
        ("FULL", "single_m", "observation"),
        ("FULL", "couple", "own"),
    }
    failures = {
        label: (int(compared), int(mismatched), int(control))
        for label, (compared, mismatched, control) in tally.items()
        if mismatched or (label in required and not (compared and control))
    } | dict.fromkeys(required - tally.keys(), "unchecked")
    assert not failures, failures


@categorical(ordered=False)
class _Pair:
    source: ScalarInt
    destination: ScalarInt


@categorical(ordered=False)
class _PairOverTwoPeriods:
    source: ScalarInt
    destination: ScalarInt
    source_terminal: ScalarInt
    destination_terminal: ScalarInt


_SEEN_SIGMA = 1.0
_UNSEEN_SIGMA = 0.5


def _wealth_grid() -> LinSpacedGrid:
    return LinSpacedGrid(start=0, stop=1, n_points=3)


def _work() -> DiscreteGrid:
    return DiscreteGrid(values=jnp.arange(2))


def _gauss(*, name: str, n: int, sigma: float) -> ShockGroup:
    return ShockGroup(
        shocks={name: Normal(mu=0.0, sigma=sigma)},
        nodes=Product(**{name: GaussHermite(n=n)}),
    )


def _wealth(wealth: FloatND) -> FloatND:
    return wealth


def _half(wealth: FloatND) -> FloatND:
    return 0.5 + 0 * wealth


def _keep_wealth(wealth: FloatND) -> FloatND:
    return wealth


def _source_utility(*, wealth: FloatND, seen: FloatND) -> FloatND:
    return 1.0 + wealth + 0.1 * seen


def _plain_source_utility(wealth: FloatND) -> FloatND:
    return 1.0 + wealth


def _earnings(*, wealth: FloatND, work: IntND, unseen: FloatND) -> FloatND:
    return wealth + work * (0.8 + unseen) + (1 - work) * 0.6 + unseen**2


def _seen_earnings(
    *, wealth: FloatND, work: IntND, seen: FloatND, unseen: FloatND
) -> FloatND:
    return wealth + work * (1.0 + seen) + (1 - work) * 0.5 + unseen**2


def _fragile_earnings(*, wealth: FloatND, work: IntND, unseen: FloatND) -> FloatND:
    return jnp.sqrt(wealth - 0.25) + work * 0.3 + unseen**2


def _exit_to_destination(
    *, projection: Callable[..., FloatND] = _wealth, observes: tuple[str, ...] = ()
) -> Exit:
    return Exit(
        targets={
            "self": ExitTarget(regime="destination", projection={"wealth": projection})
        },
        rule=Unilateral(),
        observes=observes,
    )


def _terminal_pair(
    *,
    destination_utility: Callable[..., FloatND],
    source_utility: Callable[..., FloatND] = _plain_source_utility,
    source_shocks: tuple[str, ...] = (),
    destination_shocks: tuple[str, ...] = ("unseen",),
    projection: Callable[..., FloatND] = _wealth,
    observes: tuple[str, ...] = (),
) -> Model:
    """A source leaves unilaterally for a destination; one period, both solved."""
    return Model(
        regimes={
            "source": Regime(
                states={"wealth": _wealth_grid()},
                functions={"utility": source_utility},
                shocks=source_shocks,
                exit=_exit_to_destination(projection=projection, observes=observes),
            ),
            "destination": Regime(
                states={"wealth": _wealth_grid()},
                actions={"work": _work()},
                functions={"utility": destination_utility},
                shocks=destination_shocks,
            ),
        },
        shocks={
            "seen": _gauss(name="seen", n=3, sigma=_SEEN_SIGMA),
            "unseen": _gauss(name="unseen", n=2, sigma=_UNSEEN_SIGMA),
        },
        regime_id_class=_Pair,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("source", "destination")}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _pair_over_two_periods() -> Model:
    """The source's exit reads a destination that integrates `unseen` and carries
    its wealth into a terminal period.
    """
    source = Regime(
        states={"wealth": _wealth_grid()},
        functions={"utility": _plain_source_utility},
        state_transitions={"wealth": _keep_wealth},
        exit=_exit_to_destination(),
    )
    destination = Regime(
        states={"wealth": _wealth_grid()},
        actions={"work": _work()},
        functions={"utility": _earnings},
        shocks=("unseen",),
        state_transitions={"wealth": _keep_wealth},
    )
    return Model(
        regimes={
            "source": source,
            "destination": destination,
            "source_terminal": replace(source, state_transitions={}, exit=None),
            "destination_terminal": replace(destination, state_transitions={}),
        },
        shocks={"unseen": _gauss(name="unseen", n=2, sigma=_UNSEEN_SIGMA)},
        regime_id_class=_PairOverTwoPeriods,
        n_periods=2,
        edges={
            name: {f"{name}_terminal": Periods(values=(0,))}
            for name in ("source", "destination")
        },
        initial_nodes=InitialNodes(by_period={0: ("source", "destination")}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _tabulate(
    *,
    model: Model,
    kernel: policy_queries._PolicyQueryKernel,
    context: CoreBuildContext,
    target: RegimeName,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the producer of the reader's aligned read of `target` on its inputs."""
    (read,) = (
        node
        for node in kernel.definitions.values()
        if node.read.kind is QueryKind.RAW and node.read.target == target
    )
    assert read.outside_value is not None
    definitions, key = exit_queries.bind_exit_query_graph(
        contexts={target: read.context},
        canonical_regimes=model._regimes,
        root=target,
        period=kernel.period,
        kind=QueryKind.RAW,
    )
    program = outside_value_program(
        read=read.outside_value,
        definitions=definitions,
        key=key,
        reader=kernel.root_context.regime_name,
        width=kernel.chunk_size,
    )
    # The reader's kernel binds its fixed parameters into every argument builder.
    builder = cast("outside_values._TableArguments", program.argument_builder)
    arguments = replace(
        builder,
        builder=replace(builder.builder, fixed_flat_params=kernel.fixed_flat_params),
    )(context)
    values, support, _ = cast(
        "tuple[FloatND, BoolND, IntND]", program.function(**arguments)
    )
    return np.asarray(values), np.asarray(support)


def _solve_with_table(
    *,
    model: Model,
    reader: RegimeName,
    target: RegimeName,
    monkeypatch: pytest.MonkeyPatch,
    params: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, Mapping[int, Mapping[RegimeName, np.ndarray]]]:
    """Solve `model`; tabulate `target` with the inputs of `reader`'s period-0 kernel.

    The table is built while the reader's kernel runs, so its continuation inputs
    are the ones the solve hands that kernel.
    """
    tables = []
    original = policy_queries._PolicyQueryKernel.__call__

    # keyword-only-exempt: primary-argument=self
    def tabulating(
        self: policy_queries._PolicyQueryKernel,
        *,
        compiled_cores: Mapping[str, Callable[..., PytreeValue]],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: TimeAxis,
        logger: logging.Logger,
        same_period_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> KernelOutput:
        if self.root_context.regime_name == reader and self.period == 0:
            context = CoreBuildContext(
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
                period=period,
                ages=ages,
                same_period_regime_to_V_arr=same_period_regime_to_V_arr,
                edge_regime_to_V_arr=edge_regime_to_V_arr,
            )
            tables.append(
                _tabulate(model=model, kernel=self, context=context, target=target)
            )
        return original(
            self,
            compiled_cores=compiled_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            logger=logger,
            same_period_regime_to_V_arr=same_period_regime_to_V_arr,
            edge_regime_to_V_arr=edge_regime_to_V_arr,
        )

    monkeypatch.setattr(policy_queries._PolicyQueryKernel, "__call__", tabulating)
    solution = model.solve(params=params or {}, log_level="off", runtime_checks=False)
    ((values, support),) = tables
    return (
        values,
        support,
        {
            period: {name: np.asarray(value) for name, value in by_regime.items()}
            for period, by_regime in solution.values.items()
        },
    )


def test_a_table_without_observed_shocks_is_the_problem_s_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source observes none of the destination's shocks, and the destination
    has no exit, so its table is its solved value at every grid cell.
    """
    values, _, solved = _solve_with_table(
        model=_pair_over_two_periods(),
        reader="source",
        target="destination",
        monkeypatch=monkeypatch,
        params={"discount_factor": 0.9},
    )
    expected = solved[0]["destination"]
    assert_agrees_to_ulp(
        got=values,
        expected=expected,
        n_ulp=16,
        operand_magnitude=float(np.max(np.abs(expected))),
    )


def _normal_nodes(*, n: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights of a centred normal, ascending."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    return sigma * nodes, weights / np.sqrt(2 * np.pi)


def test_a_table_entry_is_the_problem_value_at_its_observed_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source observes `seen`, so the table keeps its nodes as an axis.

    At wealth `w` and node `s`, working pays `w + 1 + s`, not working `w + 0.5`,
    and both add `unseen**2`, whose expectation over its nodes is `0.25`.
    """
    values, _, _ = _solve_with_table(
        model=_terminal_pair(
            destination_utility=_seen_earnings,
            source_utility=_source_utility,
            source_shocks=("seen",),
            destination_shocks=("seen", "unseen"),
            observes=("seen",),
        ),
        reader="source",
        target="destination",
        monkeypatch=monkeypatch,
    )
    seen, _ = _normal_nodes(n=3, sigma=_SEEN_SIGMA)
    unseen, weights = _normal_nodes(n=2, sigma=_UNSEEN_SIGMA)
    wealth = np.linspace(0, 1, 3)
    expected = (
        wealth[:, None]
        + np.maximum(1.0 + seen, 0.5)[None, :]
        + np.sum(weights * unseen**2)
    )
    np.testing.assert_array_almost_equal(values, expected, decimal=DECIMAL_PRECISION)


def test_a_table_marks_a_failed_cell_nan_and_keeps_its_neighbours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The destination's value is NaN at wealth 0, where `sqrt(wealth - 0.25)` is.

    The source reads only wealth 0.5, so the solve goes through. The table holds
    NaN at wealth 0 and the problem's value `sqrt(w - 0.25) + 0.3 + 0.25` at
    wealth 0.5 and 1; every cell has a feasible action, so every cell has support.
    """
    values, support, _ = _solve_with_table(
        model=_terminal_pair(destination_utility=_fragile_earnings, projection=_half),
        reader="source",
        target="destination",
        monkeypatch=monkeypatch,
    )
    unseen, weights = _normal_nodes(n=2, sigma=_UNSEEN_SIGMA)
    wealth = np.asarray([0.0, 0.5, 1.0])
    with np.errstate(invalid="ignore"):
        expected = np.sqrt(wealth - 0.25) + 0.3 + np.sum(weights * unseen**2)
    # A cell without support would read minus infinity instead of its value.
    np.testing.assert_array_almost_equal(
        np.where(support, values, -np.inf), expected, decimal=DECIMAL_PRECISION
    )
