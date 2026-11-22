"""Outside values a regime reads from its destination's stored nodes.

Each row of a query's information frame realises one node of every shock group the
frame enumerates, and the frame names that node. A stored outside value is gathered
at the node its caller's row names, so re-realising a group at the named node must
give back the row's physical shocks.

The stored array holds the destination's problem value at every grid cell and at
every node of the groups its readers observe, integrating the other groups.

A solve that reads stored values gives the values of one that evaluates every outside
read at its point, to a few units in the last place.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
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
from _lcm.regime_building.Q_and_F import EvaluatedQAndF
from _lcm.solution.contract import ContinuationPayload
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PytreeValue, QAndFKwargs
from lcm import (
    CollectiveUtility,
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
    ValueDependentConstraint,
    categorical,
)
from lcm.exceptions import ExecutionPlanningError, OffGridStateError
from lcm.shocks import (
    Discrete,
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
from tests.solution.test_outside_value_alignment import _marriage_model
from tests.solution.test_outside_value_truncation import _Chain, _chain_model

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
    destination_constraints: Mapping[str, Callable[..., BoolND]] | None = None,
    shocks: Mapping[str, ShockGroup] | None = None,
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
                constraints=destination_constraints or {},
                shocks=destination_shocks,
            ),
        },
        shocks=shocks
        or {
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
        row_batch=policy_queries._row_batch(
            chunk_size=kernel.chunk_size, halvings=kernel.halvings
        ),
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


@dataclass(frozen=True, kw_only=True)
class _Arm:
    """What one solve computed and how it read its outside values."""

    values: dict[str, np.ndarray] = field(default_factory=dict)
    """Solved values by `period/regime`."""
    tables: set[str] = field(default_factory=set)
    """Names of the programs that tabulate a destination, over every kernel."""
    gathers: list[RegimeName] = field(default_factory=list)
    """The destination of every stored read the solve traced."""
    evaluated: set[RegimeName] = field(default_factory=set)
    """Every regime whose action problem the solve's queries evaluated."""
    rebuilt: set[RegimeName] = field(default_factory=set)
    """Destinations of stored reads whose node evaluated their action problem."""
    failure: str = ""
    """The message of the solve's `ExecutionPlanningError`, if it raised one."""


def _solve_arm(
    *,
    model: Callable[[], Model],
    stored: bool,
    monkeypatch: pytest.MonkeyPatch,
    params: Mapping[str, float] | None = None,
    scale: float = 1.0,
    fails: bool = False,
    chunk: int | None = None,
    runtime_checks: bool = False,
) -> _Arm:
    """Build and solve `model` with stored or point outside reads.

    `scale` multiplies every stored value as it is read, `fails` expects the solve
    to raise `ExecutionPlanningError`, `chunk` fixes the source cells per chunk of
    every query kernel whose model leaves that width to the engine, and
    `runtime_checks` is passed to the solve.
    """
    arm = _Arm()
    read = outside_values.read_outside_value
    evaluate = policy_queries._evaluate_query_action

    def gathered(
        *,
        node: policy_queries.QueryNode,
        table: tuple[FloatND, BoolND],
        live: BoolND,
        available: Mapping[str, PytreeValue],
        codes: query_information.FrameCodes,
    ) -> PolicyAnswer:
        arm.gathers.append(node.read.target)
        return read(
            node=node,
            table=(table[0] * scale, table[1]),
            live=live,
            available=available,
            codes=codes,
        )

    def evaluated(
        *,
        node: policy_queries.QueryNode,
        period: int,
        arguments: Mapping[str, PytreeValue],
    ) -> EvaluatedQAndF:
        arm.evaluated.add(node.read.target)
        if node.outside_value is not None:
            arm.rebuilt.add(node.read.target)
        return evaluate(node=node, period=period, arguments=arguments)

    def chunk_size(*, definitions: Mapping[str, policy_queries.QueryNode]) -> int:
        del definitions
        return cast("int", chunk)

    with monkeypatch.context() as patch:
        patch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
        patch.setattr(outside_values, "read_outside_value", gathered)
        patch.setattr(policy_queries, "_evaluate_query_action", evaluated)
        if chunk is not None:
            patch.setattr(exit_queries, "default_chunk_size", chunk_size)
        built = model()
        arm.tables.update(
            name
            for regime in built._regimes.values()
            for kernel in regime.solution.period_kernels.values()
            if isinstance(kernel, policy_queries._PolicyQueryKernel)
            for name in kernel.programs
            if name.startswith("_outside:")
        )
        if fails:
            with pytest.raises(ExecutionPlanningError) as raised:
                built.solve(
                    params=params or {},
                    log_level="off",
                    runtime_checks=runtime_checks,
                )
            return replace(arm, failure=str(raised.value))
        solution = built.solve(
            params=params or {}, log_level="off", runtime_checks=runtime_checks
        )
    arm.values.update(
        {
            f"{period}/{regime}": np.asarray(value)
            for period, by_regime in solution.values.items()
            for regime, value in by_regime.items()
        }
    )
    return arm


def _assert_agree(
    *, got: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]
) -> None:
    """Every array agrees to 16 units in the last place at its own largest magnitude."""
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        finite = np.abs(value[np.isfinite(value)])
        assert_agrees_to_ulp(
            got=got[key],
            expected=value,
            n_ulp=16,
            operand_magnitude=float(np.max(finite, initial=0.0)),
            err_msg=key,
        )


def _observed_pair() -> Model:
    """The source observes `seen`, which the destination keeps as an axis; the
    destination integrates `unseen` and chooses whether to work.
    """
    return _terminal_pair(
        destination_utility=_seen_earnings,
        source_utility=_source_utility,
        source_shocks=("seen",),
        destination_shocks=("seen", "unseen"),
        observes=("seen",),
    )


@categorical(ordered=True)
class _Offer:
    none: ScalarInt
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=True)
class _Luck:
    bad: ScalarInt
    good: ScalarInt


def _offer_source_utility(*, wealth: FloatND, offer: IntND) -> FloatND:
    return 1.0 + wealth + 0.1 * offer


def _offer_earnings(
    *, wealth: FloatND, work: IntND, offer: IntND, unseen: FloatND
) -> FloatND:
    return wealth + work * (0.3 + 0.4 * offer) + (1 - work) * 0.5 + unseen**2


def _luck_source_utility(*, wealth: FloatND, luck: IntND, seen: FloatND) -> FloatND:
    return 1.0 + wealth + 0.1 * luck + 0.1 * seen


def _luck_earnings(
    *, wealth: FloatND, work: IntND, luck: IntND, seen: FloatND, unseen: FloatND
) -> FloatND:
    return wealth + work * (0.7 + 0.3 * luck + seen) + (1 - work) * 0.5 + unseen**2


def _categorical_pair(*, mixed: bool) -> Model:
    """The source observes a group the destination keeps as an axis: one `Discrete`
    shock with three categories, or, `mixed`, a two-category `Discrete` shock beside
    a normal one. The probabilities are not dyadic.
    """
    shocks = ("luck", "seen") if mixed else ("offer",)
    group = (
        ShockGroup(
            shocks={
                "luck": Discrete(categories=_Luck, probs=(0.3, 0.7)),
                "seen": Normal(mu=0.0, sigma=_SEEN_SIGMA),
            },
            nodes=Product(seen=GaussHermite(n=3)),
        )
        if mixed
        else ShockGroup(
            shocks={"offer": Discrete(categories=_Offer, probs=(0.1, 0.2, 0.7))}
        )
    )
    return _terminal_pair(
        destination_utility=_luck_earnings if mixed else _offer_earnings,
        source_utility=_luck_source_utility if mixed else _offer_source_utility,
        source_shocks=("kept",),
        destination_shocks=("kept", "unseen"),
        observes=shocks,
        shocks={
            "kept": group,
            "unseen": _gauss(name="unseen", n=2, sigma=_UNSEEN_SIGMA),
        },
    )


def _solved_chain() -> Model:
    """Source -> destination -> onward, each occupied and solved in the one period,
    so the source's read of the destination and the destination's read of onward
    are both stored.
    """
    return Model(
        regimes=_chain_model().user_regimes,
        regime_id_class=_Chain,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("source", "destination", "onward")}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


_AGREEMENT_CASES: dict[str, tuple[Callable[[], Model], Mapping[str, float]]] = {
    "irregular knots, aligned and unaligned reads": (_marriage_model, {}),
    "chain": (_solved_chain, {}),
    "observed group kept": (_observed_pair, {}),
    "categorical group kept": (partial(_categorical_pair, mixed=False), {}),
    "categorical and normal group kept": (partial(_categorical_pair, mixed=True), {}),
    "continuation": (_pair_over_two_periods, {"discount_factor": 0.9}),
    "formation with dissolution": (
        lambda: _formation_model(dissolution=True, query_cell=None),
        {"discount_factor": 0.9},
    ),
}


@pytest.mark.parametrize(
    ("model", "params"), list(_AGREEMENT_CASES.values()), ids=list(_AGREEMENT_CASES)
)
def test_stored_reads_agree_with_point_reads(
    *,
    model: Callable[[], Model],
    params: Mapping[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading stored outside values gives the values of reading every outside value
    at its point, to 16 units in the last place.

    The stored solve tabulates a destination and gathers from the table, and no
    stored read evaluates its destination's action problem, which only the table
    evaluates; the point solve does neither.
    """
    point = _solve_arm(
        model=model, stored=False, params=params, monkeypatch=monkeypatch
    )
    stored = _solve_arm(
        model=model, stored=True, params=params, monkeypatch=monkeypatch
    )
    assert (bool(stored.tables), bool(stored.gathers), stored.rebuilt) == (
        True,
        True,
        set(),
    )
    assert set(stored.gathers) <= stored.evaluated
    assert (point.tables, point.gathers) == (set(), [])
    _assert_agree(got=stored.values, expected=point.values)


@pytest.mark.parametrize(
    ("model", "params"), list(_AGREEMENT_CASES.values()), ids=list(_AGREEMENT_CASES)
)
def test_point_reads_agree_with_themselves_across_chunk_widths(
    *,
    model: Callable[[], Model],
    params: Mapping[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: point reads in one-cell chunks agree with point reads in the
    engine's chunks under the comparison stored reads are held to.
    """
    wide = _solve_arm(model=model, stored=False, params=params, monkeypatch=monkeypatch)
    narrow = _solve_arm(
        model=model, stored=False, params=params, monkeypatch=monkeypatch, chunk=1
    )
    _assert_agree(got=narrow.values, expected=wide.values)


@pytest.mark.parametrize("mixed", [False, True], ids=["categorical", "mixed"])
def test_a_categorical_group_is_kept_as_an_axis_of_the_table(
    *, mixed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The destination's table keeps the group holding a `Discrete` shock as an
    axis, and the source's read gathers from it.
    """
    stored = _solve_arm(
        model=partial(_categorical_pair, mixed=mixed),
        stored=True,
        monkeypatch=monkeypatch,
    )
    assert stored.tables == {"_outside:destination:kept"}
    assert set(stored.gathers) == {"destination"}


def test_a_perturbed_stored_value_fails_the_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source leaves for the destination's stored value 3, so stored values
    moved by 64 units in the last place move the source's value as far.
    """
    point = _solve_arm(model=_solved_chain, stored=False, monkeypatch=monkeypatch)
    eps = float(jnp.finfo(jnp.asarray(0.0).dtype).eps)
    perturbed = _solve_arm(
        model=_solved_chain, stored=True, monkeypatch=monkeypatch, scale=1 + 64 * eps
    )
    assert perturbed.gathers
    with pytest.raises(AssertionError):
        _assert_agree(got=perturbed.values, expected=point.values)


def test_a_failed_stored_cell_fails_only_the_read_that_takes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The destination's value is NaN at wealth 0.

    A source that reads wealth 0 fails as the point read does, with the same
    message. A source that reads only wealth 0.5 solves, from a table that holds
    the NaN.
    """

    def taken() -> Model:
        return _terminal_pair(destination_utility=_fragile_earnings)

    def avoided() -> Model:
        return _terminal_pair(destination_utility=_fragile_earnings, projection=_half)

    failed = {
        stored: _solve_arm(
            model=taken, stored=stored, monkeypatch=monkeypatch, fails=True
        )
        for stored in (False, True)
    }
    point = _solve_arm(model=avoided, stored=False, monkeypatch=monkeypatch)
    stored = _solve_arm(model=avoided, stored=True, monkeypatch=monkeypatch)
    assert (bool(failed[True].gathers), bool(stored.gathers)) == (True, True)
    assert failed[True].failure == failed[False].failure
    assert "unresolved numerical status" in failed[True].failure
    _assert_agree(got=stored.values, expected=point.values)


def _signed(code: IntND) -> FloatND:
    """The `_Luck` code 0 as -1 and 1 as +1."""
    return 2.0 * code - 1.0


def _seen_source_utility_codes(*, wealth: FloatND, seen: IntND) -> FloatND:
    return 1.0 + wealth + 0.1 * _signed(seen)


def _two_seen_source_utility(
    *, wealth: FloatND, seen: FloatND, other: FloatND
) -> FloatND:
    return 1.0 + wealth + 0.1 * seen + 0.05 * other


def _root_payoff(*, wealth: FloatND, seen: FloatND) -> FloatND:
    return 2.0 + wealth + jnp.sqrt(seen)


def _root_payoff_codes(*, wealth: FloatND, seen: IntND) -> FloatND:
    return 2.0 + wealth + jnp.sqrt(_signed(seen))


def _two_root_payoff(*, wealth: FloatND, seen: FloatND, other: FloatND) -> FloatND:
    return 2.0 + wealth + jnp.sqrt(seen) + 0.2 * other


def _hidden_root_payoff(*, wealth: FloatND, seen: FloatND, unseen: FloatND) -> FloatND:
    return 2.0 + wealth + jnp.sqrt(seen) + unseen**2


def _absolute_root_payoff(*, wealth: FloatND, seen: FloatND) -> FloatND:
    return 2.0 + wealth + jnp.sqrt(jnp.abs(seen))


def _positive_seen(seen: FloatND) -> BoolND:
    return seen > 0


def _seen_past_two(seen: FloatND) -> BoolND:
    return seen > 2


def _seen_past_minus_two(seen: FloatND) -> BoolND:
    return seen > -2


def _kept_pair(
    *, case: str = "normal", available: Callable[..., BoolND] = _positive_seen
) -> Model:
    """A source may leave for a destination where `available` holds; it observes
    `seen`, which the destination keeps as an axis of its table.

    `seen` takes -1 and +1 with mass 1/2 each, as two Gauss-Hermite nodes or as the
    codes of `_Luck`; the destination's payoff is NaN at -1 and `3 + wealth` at +1.
    The cases vary that model:

    - "normal": as described
    - "categorical": `seen` is a `Discrete` shock
    - "zero mass": `seen` is a `Discrete` shock whose bad category has probability 0
    - "two kept": the source also observes `other`, +/-1 with mass 1/2, which adds
      `0.2 * other` to the destination's payoff
    - "hidden": the destination also integrates `unseen`, which the source does not
      observe and which adds `unseen**2`, of mean 0.25
    - "finite": the destination's payoff is finite at -1 too
    """
    codes = case in {"categorical", "zero mass"}
    seen = (
        ShockGroup(
            shocks={
                "seen": Discrete(
                    categories=_Luck,
                    probs=(0.0, 1.0) if case == "zero mass" else (0.5, 0.5),
                )
            }
        )
        if codes
        else _gauss(name="seen", n=2, sigma=1.0)
    )
    shocks = {"seen": seen}
    observes = destination_shocks = ("seen",)
    source_utility = _seen_source_utility_codes if codes else _source_utility
    destination_utility = _root_payoff_codes if codes else _root_payoff
    if case == "two kept":
        shocks["other"] = _gauss(name="other", n=2, sigma=1.0)
        observes = destination_shocks = ("seen", "other")
        source_utility, destination_utility = (
            _two_seen_source_utility,
            _two_root_payoff,
        )
    elif case == "hidden":
        shocks["unseen"] = _gauss(name="unseen", n=2, sigma=_UNSEEN_SIGMA)
        destination_shocks = ("seen", "unseen")
        destination_utility = _hidden_root_payoff
    elif case == "finite":
        destination_utility = _absolute_root_payoff
    return Model(
        regimes={
            "source": Regime(
                states={"wealth": _wealth_grid()},
                functions={"utility": source_utility},
                shocks=observes,
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="destination", projection={"wealth": _wealth}
                        )
                    },
                    rule=Unilateral(),
                    observes=observes,
                    available=available,
                ),
            ),
            "destination": Regime(
                states={"wealth": _wealth_grid()},
                functions={"utility": destination_utility},
                shocks=destination_shocks,
            ),
        },
        shocks=shocks,
        regime_id_class=_Pair,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("source", "destination")}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


# Each case: the model, the solve's runtime checks, and the source's value at
# wealth 0 in closed form.
_KEPT_NODE_CASES: dict[str, tuple[Callable[[], Model], bool, float]] = {
    "positive mass": (_kept_pair, False, 1.95),
    "categorical": (partial(_kept_pair, case="categorical"), False, 1.95),
    "zero mass": (partial(_kept_pair, case="zero mass"), False, 3.0),
    "zero mass, runtime checks": (partial(_kept_pair, case="zero mass"), True, 3.0),
    "two kept groups": (partial(_kept_pair, case="two kept"), False, 1.95),
    "hidden group": (partial(_kept_pair, case="hidden"), False, 2.075),
    "read never made": (partial(_kept_pair, available=_seen_past_two), False, 1.0),
    "finite": (partial(_kept_pair, case="finite"), True, 1.95),
}


@pytest.mark.parametrize(
    ("model", "runtime_checks", "intercept"),
    list(_KEPT_NODE_CASES.values()),
    ids=list(_KEPT_NODE_CASES),
)
def test_a_failed_kept_node_fails_no_read_of_another_node(
    *,
    model: Callable[[], Model],
    runtime_checks: bool,
    intercept: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source may leave only at `seen = +1`, and in "read never made" nowhere, so
    it never reads the destination's failed node at `seen = -1`.

    Where it may leave, leaving for `3 + wealth` beats staying; elsewhere it stays
    for `1 + wealth + 0.1 * seen`. Its value is the closed form `intercept + wealth`
    in both routes, and the stored solve gives the point solve's values.
    """
    point = _solve_arm(
        model=model,
        stored=False,
        runtime_checks=runtime_checks,
        monkeypatch=monkeypatch,
    )
    stored = _solve_arm(
        model=model, stored=True, runtime_checks=runtime_checks, monkeypatch=monkeypatch
    )
    assert (bool(stored.tables), point.tables) == (True, set())
    np.testing.assert_array_almost_equal(
        point.values["0/source"],
        intercept + np.linspace(0, 1, 3),
        decimal=DECIMAL_PRECISION,
    )
    _assert_agree(got=stored.values, expected=point.values)


def test_a_read_that_takes_a_failed_kept_node_fails_in_both_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source may leave at `seen = -1`, where the destination's payoff is NaN."""
    failed = {
        stored: _solve_arm(
            model=partial(_kept_pair, available=_seen_past_minus_two),
            stored=stored,
            monkeypatch=monkeypatch,
            fails=True,
        )
        for stored in (False, True)
    }
    assert bool(failed[True].tables)
    assert failed[True].failure == failed[False].failure
    assert "unresolved numerical status" in failed[True].failure


@pytest.mark.parametrize("case", ["normal", "two kept"])
def test_a_table_marks_only_the_failed_kept_node_nan(
    *, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The destination's table is NaN at `seen = -1` and its problem's value
    `3 + wealth + 0.2 * other` at `seen = +1`; `other` and `seen` are the table's
    kept axes in that order.
    """
    values, support, _ = _solve_with_table(
        model=_kept_pair(case=case),
        reader="source",
        target="destination",
        monkeypatch=monkeypatch,
    )
    wealth = np.linspace(0, 1, 3)
    other = np.asarray([-1.0, 1.0]) if case == "two kept" else np.zeros(())
    expected = np.stack(
        np.broadcast_arrays(np.nan, 3.0 + np.add.outer(wealth, 0.2 * other)), axis=-1
    )
    assert support.all()
    np.testing.assert_array_almost_equal(values, expected, decimal=DECIMAL_PRECISION)


def test_a_failed_law_fails_every_kept_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every payoff is finite but the draw frame's law fails, so the source's read
    at `seen = +1` fails too.
    """
    kept_answer = InformationQueryCell._kept_answer

    # keyword-only-exempt: primary-argument=self
    def failed_law(
        self: InformationQueryCell,
        *,
        answers: PolicyAnswer,
        factors: FloatND,
        frame: CurrentIIDDrawFrame,
    ) -> PolicyAnswer:
        return kept_answer(
            self,
            answers=answers,
            factors=factors,
            frame=frame._replace(valid=jnp.zeros_like(frame.valid)),
        )

    monkeypatch.setattr(InformationQueryCell, "_kept_answer", failed_law)
    failed = _solve_arm(
        model=partial(_kept_pair, case="finite"),
        stored=True,
        monkeypatch=monkeypatch,
        fails=True,
    )
    assert bool(failed.tables)
    assert "unresolved numerical status" in failed.failure


@categorical(ordered=False)
class _Leaving:
    source: ScalarInt
    destination: ScalarInt
    elsewhere: ScalarInt
    source_terminal: ScalarInt
    destination_terminal: ScalarInt
    elsewhere_terminal: ScalarInt


def _past_the_top(wealth: FloatND) -> FloatND:
    return jnp.where(wealth > 0.75, 1.5, wealth)


def _poor_at_the_top(*, wealth: FloatND, work: IntND) -> FloatND:
    return 2.0 + wealth + 0.1 * work - 10.0 * (wealth > 0.75)


def _poor_at_the_top_f(*, wealth: FloatND, work: IntND) -> FloatND:
    return _poor_at_the_top(wealth=wealth, work=work)


def _poor_at_the_top_m(*, wealth: FloatND, work: IntND) -> FloatND:
    return 1.0 + 0.5 * _poor_at_the_top(wealth=wealth, work=work)


def _admit_all(Q_f: FloatND) -> BoolND:
    return Q_f >= -1000.0


def _leaving_destination(
    *, collective: bool, landing: Callable[..., FloatND]
) -> Regime:
    """The destination works for its next wealth by `landing`; it stays at wealth
    0 and 0.5 and is so poor at wealth 1 that it leaves there for `elsewhere`.

    A `collective` destination keeps every action by a value constraint, so it
    counts its landings in its kernel rather than in the preflight.
    """
    members = ("f", "m") if collective else ("self",)
    return Regime(
        states={"wealth": _wealth_grid()},
        actions={"work": _work()},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _poor_at_the_top_f, "m": _poor_at_the_top_m}
            )
            if collective
            else _poor_at_the_top
        },
        constraints={"admit_all": ValueDependentConstraint(predicate=_admit_all)}
        if collective
        else {},
        state_transitions={"wealth": landing},
        exit=Exit(
            targets={
                member: ExitTarget(regime="elsewhere", projection={"wealth": _wealth})
                for member in members
            },
            rule=Unilateral(),
        ),
    )


def _leaving_model(
    *,
    collective: bool,
    projection: Callable[..., FloatND] = _half,
    landing: Callable[..., FloatND] = _past_the_top,
) -> Model:
    """The source reads the destination at `projection` of its wealth, the
    destination's next wealth must stay on a raise-rule grid, and the destination
    leaves for `elsewhere` at wealth 1, where `landing` takes it past the grid.
    """
    source = Regime(
        states={"wealth": _wealth_grid()},
        functions={"utility": _plain_source_utility},
        state_transitions={"wealth": _keep_wealth},
        exit=Exit(
            targets={
                "self": ExitTarget(
                    regime="destination",
                    role="f" if collective else None,
                    projection={"wealth": projection},
                )
            },
            rule=Unilateral(),
        ),
    )
    elsewhere = replace(source, exit=None)
    destination = _leaving_destination(collective=collective, landing=landing)
    terminal = replace(
        destination,
        states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=3, off_grid="raise")},
        constraints={},
        state_transitions={},
        exit=None,
    )
    return Model(
        regimes={
            "source": source,
            "destination": destination,
            "elsewhere": elsewhere,
            "source_terminal": replace(source, state_transitions={}, exit=None),
            "destination_terminal": terminal,
            "elsewhere_terminal": replace(elsewhere, state_transitions={}),
        },
        regime_id_class=_Leaving,
        n_periods=2,
        edges={
            name: {f"{name}_terminal": Periods(values=(0,))}
            for name in ("source", "destination", "elsewhere")
        },
        initial_nodes=InitialNodes(
            by_period={0: ("source", "destination", "elsewhere")}
        ),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _refusal(
    *, model: Callable[[], Model], stored: bool, monkeypatch: pytest.MonkeyPatch
) -> str:
    """The message of the solve's `OffGridStateError`."""
    with pytest.raises(OffGridStateError) as raised:
        _solve_arm(model=model, stored=stored, monkeypatch=monkeypatch, params=_BETA)
    return str(raised.value)


_BETA = {"discount_factor": 0.9}


@pytest.mark.parametrize("collective", [False, True], ids=["preflight", "kernel"])
def test_a_landing_past_the_grid_where_the_destination_leaves_fails_both_modes(
    *, collective: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The destination's stay problem at wealth 1 lands past its raise-rule grid,
    and the destination leaves there; the source reads it only at wealth 0.5.

    The destination's own solve refuses that landing: a plain destination in the
    preflight, a destination with a value constraint in its kernel, which counts
    the stay problem's landings where it leaves too. So both modes refuse alike,
    and the table's check over the whole grid refuses nothing more.
    """
    model = partial(_leaving_model, collective=collective)
    point = _refusal(model=model, stored=False, monkeypatch=monkeypatch)
    stored = _refusal(model=model, stored=True, monkeypatch=monkeypatch)
    assert stored == point
    assert "regime='destination', target='destination_terminal'" in point
    assert "state='wealth', phase='solve'" in point


@pytest.mark.parametrize("collective", [False, True], ids=["preflight", "kernel"])
def test_the_leaving_destination_on_the_grid_solves_both_modes_alike(
    *, collective: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: with the landing kept on the grid, the same model solves, from
    a table in the stored mode, to the point mode's values. Every member of the
    destination takes `elsewhere`'s value at wealth 1 and more at wealth 0 and 0.5.
    """
    model = partial(_leaving_model, collective=collective, landing=_keep_wealth)
    point = _solve_arm(model=model, stored=False, monkeypatch=monkeypatch, params=_BETA)
    stored = _solve_arm(model=model, stored=True, monkeypatch=monkeypatch, params=_BETA)
    destination = point.values["0/destination"].reshape(3, -1)
    elsewhere = point.values["0/elsewhere"][:, None]
    assert (destination[2] == elsewhere[2]).all()
    assert (destination[:2] > elsewhere[:2]).all()
    assert "_outside:destination" in stored.tables
    assert "destination" in stored.gathers
    _assert_agree(got=stored.values, expected=point.values)


def _positive_wealth(wealth: FloatND) -> BoolND:
    return wealth > 0


def _read_supports(
    *, stored: bool, monkeypatch: pytest.MonkeyPatch
) -> dict[float, bool]:
    """The support of the source's read of the destination at each live source cell."""
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    original = policy_queries._QueryCore.__call__

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record(wealth: FloatND, live: BoolND, support: BoolND) -> None:
        records.append((np.asarray(wealth), np.asarray(live), np.asarray(support)))

    def recorded(
        self: policy_queries._QueryCore, **arguments: PytreeValue
    ) -> PolicyAnswer:
        answer = original(self, **arguments)
        if (
            self.node.read.kind is QueryKind.RAW
            and self.node.read.target == "destination"
        ):
            source = cast("Mapping[str, FloatND]", arguments[policy_queries._SOURCE])
            jax.debug.callback(
                record, source["wealth"], arguments[policy_queries._LIVE], answer[2]
            )
        return answer

    with monkeypatch.context() as patch:
        patch.setattr(policy_queries._QueryCore, "__call__", recorded)
        arm = _solve_arm(
            model=lambda: _terminal_pair(
                destination_utility=_earnings,
                destination_constraints={"positive_wealth": _positive_wealth},
            ),
            stored=stored,
            monkeypatch=monkeypatch,
        )
    assert bool(arm.gathers) == stored
    return {
        float(wealth): bool(support)
        for wealths, lives, supports in records
        for wealth, live, support in zip(
            np.ravel(wealths), np.ravel(lives), np.ravel(supports), strict=True
        )
        if live
    }


def test_a_stored_read_has_the_point_read_s_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No action of the destination is feasible at wealth 0, so neither read of it
    has support there; both have support at wealth 0.5 and 1.
    """
    point = _read_supports(stored=False, monkeypatch=monkeypatch)
    stored = _read_supports(stored=True, monkeypatch=monkeypatch)
    assert point == {0.0: False, 0.5: True, 1.0: True}
    assert stored == point
