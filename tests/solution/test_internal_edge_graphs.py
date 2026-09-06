"""A multi-program graph's typed internal edges lower and publish their values.

The engine visits producers before consumers and traces each producer with the
complete invocation it is lowered with: the dynamic arguments its builder
returned, the templates of the internal inputs it reads itself, and the widths
the planner owns. A chain, a fork and join, two labels of one producer, and a
nested output path therefore all reach dispatch with exact shapes and dtypes.

The chain cases restate the public acceptance graph in the repository's own
test-model style, carrying 28 of its 32 items: the four combinations that pair a
planned root with a one-program chain are dropped deliberately, since a graph of
one program declares no typed edge and so states nothing about one.

Ten items are controls rather than defect witnesses: the eight dense chains of
depth one and two, whose producers read no internal input and own no width, and
the two-label and nested-path graphs, whose producers are roots. They pass
whether or not a producer is traced with its complete invocation and show that
the harness itself publishes the expected rows; the remaining items fail at
lowering unless it is.
"""

import pathlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.execution.core_program import (
    CoreArgumentBuilder,
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalInputRef,
    InternalOutputSpec,
)
from _lcm.execution.internal_outputs import topological_program_order
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.solution.period_replay import replay_period
from _lcm.typing import FlatParams, FloatND
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.execution import ExecutionConfig
from lcm.solver_api import (
    KernelOutput,
    ResultRetention,
    SolverIdentity,
)
from lcm.solvers import GridSearch, StreamableProductAxis
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_N_PERIODS = 3
_CANDIDATES = 5


def _root_body(*, x: FloatND) -> FloatND:
    """Publish one row per state node, one above the state itself."""
    return x + 1.0


def _step_body(*, upstream: FloatND) -> FloatND:
    """Publish one row per state node from the row its producer published."""
    return upstream * 2.0 + 1.0


def _triple_body(*, upstream: FloatND) -> FloatND:
    """Publish a second row per state node, distinguishable from `_step_body`."""
    return upstream * 3.0


def _join_body(*, left: FloatND, right: FloatND) -> FloatND:
    """Publish the sum of two producers' rows."""
    return left + right


def _pair_body(*, x: FloatND) -> tuple[FloatND, dict[str, FloatND]]:
    """Publish two separately labelled subtrees of one raw output."""
    return x + 1.0, {"carry": x * 2.0}


def _pair_consumer(*, first: FloatND, second: Mapping[str, FloatND]) -> FloatND:
    """Publish the sum of two labels of one producer."""
    return first + second["carry"]


def _nested_body(*, x: FloatND) -> tuple[FloatND, dict[str, dict[str, FloatND]]]:
    """Publish a raw output whose second element nests two mapping steps."""
    return x + 1.0, {"inner": {"deep": x * 3.0}}


def _nested_consumer(*, deep: FloatND) -> FloatND:
    """Publish the row selected by a tuple-then-mapping-then-mapping path."""
    return deep + 1.0


def _planned_root(*, x: FloatND, candidate: FloatND, width: int) -> FloatND:
    """Publish the state row shifted by the maximum of a padded candidate block.

    The block count depends on the planner's width and the padded tail is
    `-inf`, so the published row is the same at every legal width.
    """
    pad = (-candidate.shape[0]) % width
    blocks = jnp.pad(candidate, (0, pad), constant_values=-jnp.inf).reshape((-1, width))
    best = jnp.asarray(-jnp.inf, dtype=x.dtype)
    for index in range(blocks.shape[0]):
        best = jnp.maximum(best, jnp.max(blocks[index]))
    return x + best


@dataclass(frozen=True, kw_only=True)
class _MaxReduction:
    """Semantics of the streamed candidate axis's reduction."""

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the durable identity of this reduction."""
        return ("tests.internal_edges.max", 1)


@dataclass(frozen=True, kw_only=True)
class _StateArguments:
    """Build the wealth row every root program reads."""

    planned: bool
    """Whether the root also receives the streamed candidate coordinate."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Return the wealth row, plus the candidate row for a planned root."""
        x = cast("Any", context.state_action_space).states["wealth"]
        if not self.planned:
            return {"x": x}
        return {"x": x, "candidate": jnp.arange(_CANDIDATES, dtype=x.dtype)}


@dataclass(frozen=True, kw_only=True)
class _NoArguments:
    """Build the empty argument tree of a program that only reads its producers."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Return no arguments, ignoring the build context."""
        del context
        return {}


def _spec(*, producer: CoreProgram, label: str) -> InternalOutputSpec:
    """Return the producer's output declaration one label names."""
    for spec in producer.internal_outputs:
        if spec.label == label:
            return spec
    msg = f"Producer {producer.name!r} does not declare {label!r}."
    raise ValueError(msg)


def _select(*, tree: object, path: tuple[int | str, ...]) -> object:
    """Index a producer's real output down to the subtree one label publishes."""
    node: Any = tree
    for step in path:
        node = node[step]
    return node


@dataclass(frozen=True, kw_only=True)
class _GraphKernel:
    """Dispatch one program graph in producer order, threading its typed edges."""

    programs: Mapping[str, CoreProgram]
    """The graph this kernel publishes, in its declaration order."""

    order: tuple[str, ...]
    """The graph's keys with every producer before its consumers."""

    value_from: str
    """Graph key of the program whose output is the period's value."""

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return the graph as the sole execution authority."""
        return self.programs

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _GraphKernel:
        """Return this kernel; its programs read no fixed regime params."""
        del fixed_flat_params
        return self

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Any],
        state_action_space: object,
        next_regime_to_V_arr: Mapping[str, object],
        next_regime_to_continuation: Mapping[str, object],
        flat_params: FlatParams,
        period: int,
        ages: object,
        **unused: object,
    ) -> KernelOutput:
        """Run every program once, handing each producer's labelled subtree on."""
        del unused
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        produced: dict[str, Any] = {}
        for name in self.order:
            program = self.programs[name]
            arguments = dict(program.argument_builder(context))
            for argument, ref in program.requirements.internal_inputs.items():
                arguments[argument] = _select(
                    tree=produced[ref.producer],
                    path=_spec(
                        producer=self.programs[ref.producer], label=ref.label
                    ).path,
                )
            produced[name] = compiled_cores[name](**arguments)
        return KernelOutput(value=produced[self.value_from])


@dataclass(frozen=True, kw_only=True)
class _GraphSolver(GridSearch):
    """Publish one fixed program graph at every active period of its regime.

    The graph is named rather than held: a solver's declared fields enter the
    model's durable structure fingerprint, which admits plain declarative values
    and refuses a program declaration carried on one.
    """

    graph_kind: str
    """Which of this module's graphs every period of this regime dispatches."""

    depth: int = 0
    """Number of programs a chain graph declares; unread by the other graphs."""

    planned: bool = False
    """Whether a chain's root streams a planner-owned candidate axis."""

    reverse: bool = False
    """Whether a chain is declared with its consumers before its producers."""

    value_from: str
    """Graph key of the program whose output is the period's value."""

    @property
    def identity(self) -> SolverIdentity:
        """Return this test solver's durable plugin identity."""
        return SolverIdentity(
            plugin_id="tests.internal_edge_graphs", plugin_version="1.0.0"
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Share one kernel object across every active period of the regime."""
        programs = _graph_programs(
            graph_kind=self.graph_kind,
            depth=self.depth,
            planned=self.planned,
            reverse=self.reverse,
        )
        kernel = _GraphKernel(
            programs=programs,
            order=topological_program_order(graph=programs),
            value_from=self.value_from,
        )
        return SolutionKernels(
            period_kernels=MappingProxyType(
                dict.fromkeys(
                    context.regimes_to_active_periods[context.regime_name], kernel
                )
            )
        )


def _program(
    *,
    name: str,
    function: Callable[..., object],
    builder: CoreArgumentBuilder,
    internal_inputs: Mapping[str, InternalInputRef],
    internal_outputs: tuple[InternalOutputSpec, ...],
    output_roles: object,
    planned: bool,
) -> CoreProgram:
    """Declare one program of a test graph, dense unless it streams candidates."""
    axes = (
        (
            StreamableProductAxis(
                name="candidate",
                coordinate_names=("candidate",),
                coordinate_extents=(_CANDIDATES,),
                canonical_order="c",
                reduction=_MaxReduction(),
                width_keyword="width",
            ),
        )
        if planned
        else ()
    )
    return CoreProgram(
        name=name,
        function=function,
        argument_builder=builder,
        requirements=CoreExecutionRequirements(
            streamable_axes=axes, internal_inputs=internal_inputs
        ),
        output_roles=output_roles,
        disposition=(
            CoreExecutionDisposition.PLANNED
            if planned
            else CoreExecutionDisposition.DENSE
        ),
        disposition_reason=None if planned else "deliberately_dense:test_graph",
        internal_outputs=internal_outputs,
    )


def _chain(*, depth: int, planned: bool, reverse: bool) -> Mapping[str, CoreProgram]:
    """Declare a chain of `depth` programs, each reading the one before it."""
    programs: dict[str, CoreProgram] = {}
    for index in range(depth):
        is_root = index == 0
        programs[f"p{index}"] = _program(
            name=f"p{index}",
            function=(
                (_planned_root if planned else _root_body) if is_root else _step_body
            ),
            builder=_StateArguments(planned=planned) if is_root else _NoArguments(),
            internal_inputs=(
                {}
                if is_root
                else {
                    "upstream": InternalInputRef(
                        producer=f"p{index - 1}", label="value"
                    )
                }
            ),
            internal_outputs=(InternalOutputSpec(label="value", path=()),),
            output_roles=VALUE,
            planned=planned and is_root,
        )
    if reverse:
        return MappingProxyType(dict(reversed(tuple(programs.items()))))
    return MappingProxyType(programs)


def _fork_join() -> Mapping[str, CoreProgram]:
    """Declare one root read by two programs whose outputs a fourth joins."""
    root = _program(
        name="root",
        function=_root_body,
        builder=_StateArguments(planned=False),
        internal_inputs={},
        internal_outputs=(InternalOutputSpec(label="value", path=()),),
        output_roles=VALUE,
        planned=False,
    )
    branches = {
        name: _program(
            name=name,
            function=function,
            builder=_NoArguments(),
            internal_inputs={
                "upstream": InternalInputRef(producer="root", label="value")
            },
            internal_outputs=(InternalOutputSpec(label="value", path=()),),
            output_roles=VALUE,
            planned=False,
        )
        for name, function in (("left", _step_body), ("right", _triple_body))
    }
    join = _program(
        name="join",
        function=_join_body,
        builder=_NoArguments(),
        internal_inputs={
            "left": InternalInputRef(producer="left", label="value"),
            "right": InternalInputRef(producer="right", label="value"),
        },
        internal_outputs=(),
        output_roles=VALUE,
        planned=False,
    )
    return MappingProxyType({"root": root, **branches, "join": join})


def _two_labels() -> Mapping[str, CoreProgram]:
    """Declare one producer publishing two labels that one consumer reads."""
    producer = _program(
        name="pair",
        function=_pair_body,
        builder=_StateArguments(planned=False),
        internal_inputs={},
        internal_outputs=(
            InternalOutputSpec(label="value", path=(0,)),
            InternalOutputSpec(label="carry", path=(1,)),
        ),
        output_roles=(VALUE, {"carry": StateAxesLeading(state_names=("wealth",))}),
        planned=False,
    )
    consumer = _program(
        name="sum",
        function=_pair_consumer,
        builder=_NoArguments(),
        internal_inputs={
            "first": InternalInputRef(producer="pair", label="value"),
            "second": InternalInputRef(producer="pair", label="carry"),
        },
        internal_outputs=(),
        output_roles=VALUE,
        planned=False,
    )
    return MappingProxyType({"pair": producer, "sum": consumer})


def _nested_path() -> Mapping[str, CoreProgram]:
    """Declare a label whose path indexes a tuple and then two mappings."""
    producer = _program(
        name="nested",
        function=_nested_body,
        builder=_StateArguments(planned=False),
        internal_inputs={},
        internal_outputs=(InternalOutputSpec(label="deep", path=(1, "inner", "deep")),),
        output_roles=(
            VALUE,
            {"inner": {"deep": StateAxesLeading(state_names=("wealth",))}},
        ),
        planned=False,
    )
    consumer = _program(
        name="reader",
        function=_nested_consumer,
        builder=_NoArguments(),
        internal_inputs={"deep": InternalInputRef(producer="nested", label="deep")},
        internal_outputs=(),
        output_roles=VALUE,
        planned=False,
    )
    return MappingProxyType({"nested": producer, "reader": consumer})


def _graph_programs(
    *, graph_kind: str, depth: int, planned: bool, reverse: bool
) -> Mapping[str, CoreProgram]:
    """Return the graph one solver declaration names."""
    if graph_kind == "chain":
        return _chain(depth=depth, planned=planned, reverse=reverse)
    if graph_kind == "fork_join":
        return _fork_join()
    if graph_kind == "two_labels":
        return _two_labels()
    if graph_kind == "nested_path":
        return _nested_path()
    msg = f"Unknown test graph {graph_kind!r}."
    raise ValueError(msg)


def _model(
    *,
    graph_kind: str,
    value_from: str,
    enable_jit: bool,
    n_wealth: int,
    depth: int = 0,
    planned: bool = False,
    reverse: bool = False,
) -> Model:
    """Build the regression regime with its solver replaced by a graph solver."""
    last_age = START_AGE + _N_PERIODS - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= last_age,
                states={
                    "wealth": LinSpacedGrid(
                        start=1, stop=float(n_wealth), n_points=n_wealth
                    )
                },
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                solver=_GraphSolver(
                    graph_kind=graph_kind,
                    depth=depth,
                    planned=planned,
                    reverse=reverse,
                    value_from=value_from,
                ),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=last_age + 1, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


def _wealth(*, n_wealth: int) -> np.ndarray:
    """Return the wealth grid the graph programs read."""
    return np.linspace(1.0, float(n_wealth), n_wealth)


def _chain_expectation(*, depth: int, planned: bool, n_wealth: int) -> np.ndarray:
    """Return the chain's published row, computed independently of the engine."""
    expected = _wealth(n_wealth=n_wealth) + (_CANDIDATES - 1 if planned else 1)
    for _ in range(depth - 1):
        expected = expected * 2.0 + 1.0
    return expected


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_a_dense_chain_publishes_the_row_its_programs_compose(
    *, depth: int, reverse: bool, enable_jit: bool
) -> None:
    """Each program of a chain is lowered against the row it will receive."""
    model = _model(
        graph_kind="chain",
        depth=depth,
        planned=False,
        reverse=reverse,
        value_from=f"p{depth - 1}",
        enable_jit=enable_jit,
        n_wealth=3,
    )

    result = model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    aaae(
        np.asarray(result.values[0]["working_life"]),
        _chain_expectation(depth=depth, planned=False, n_wealth=3),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("depth", [2, 3, 5])
def test_a_planned_producer_feeds_a_dense_consumer(
    *, depth: int, reverse: bool, enable_jit: bool
) -> None:
    """A producer's planner-owned width is bound before its consumer is lowered."""
    model = _model(
        graph_kind="chain",
        depth=depth,
        planned=True,
        reverse=reverse,
        value_from=f"p{depth - 1}",
        enable_jit=enable_jit,
        n_wealth=3,
    )

    result = model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    aaae(
        np.asarray(result.values[0]["working_life"]),
        _chain_expectation(depth=depth, planned=True, n_wealth=3),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("n_wealth", [2, 3, 5])
def test_a_chain_publishes_one_row_per_state_node_at_every_extent(
    *, n_wealth: int
) -> None:
    """The templates carry the state extent, so a wider grid needs no new rule."""
    model = _model(
        graph_kind="chain",
        depth=3,
        value_from="p2",
        enable_jit=True,
        n_wealth=n_wealth,
    )

    result = model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    aaae(
        np.asarray(result.values[0]["working_life"]),
        _chain_expectation(depth=3, planned=False, n_wealth=n_wealth),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("period", [0, 1])
def test_every_regime_period_cell_of_a_chain_publishes_its_own_row(
    *, period: int
) -> None:
    """Each cell resolves its own producers, so both solved periods agree."""
    model = _model(
        graph_kind="chain",
        depth=3,
        value_from="p2",
        enable_jit=True,
        n_wealth=3,
    )

    result = model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    aaae(
        np.asarray(result.values[period]["working_life"]),
        _chain_expectation(depth=3, planned=False, n_wealth=3),
        decimal=DECIMAL_PRECISION,
    )


def test_a_fork_and_join_graph_publishes_the_sum_of_both_branches() -> None:
    """One producer read by two consumers is traced once and reaches both."""
    model = _model(
        graph_kind="fork_join", value_from="join", enable_jit=True, n_wealth=3
    )

    result = model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    root = _wealth(n_wealth=3) + 1.0
    aaae(
        np.asarray(result.values[0]["working_life"]),
        (root * 2.0 + 1.0) + root * 3.0,
        decimal=DECIMAL_PRECISION,
    )


def test_two_labels_of_one_producer_reach_one_consumer() -> None:
    """Two labels select two subtrees of one traced abstract output."""
    model = _model(
        graph_kind="two_labels", value_from="sum", enable_jit=True, n_wealth=3
    )

    result = model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    wealth = _wealth(n_wealth=3)
    aaae(
        np.asarray(result.values[0]["working_life"]),
        (wealth + 1.0) + wealth * 2.0,
        decimal=DECIMAL_PRECISION,
    )


def test_a_nested_tuple_and_mapping_path_reaches_its_consumer() -> None:
    """A label whose path crosses a tuple and two mappings selects one leaf."""
    model = _model(
        graph_kind="nested_path", value_from="reader", enable_jit=True, n_wealth=3
    )

    result = model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    aaae(
        np.asarray(result.values[0]["working_life"]),
        _wealth(n_wealth=3) * 3.0 + 1.0,
        decimal=DECIMAL_PRECISION,
    )


def test_a_budgeted_solve_resolves_every_width_candidate_of_a_producer() -> None:
    """A budget offers the planned root four widths; all publish the one row.

    The frontier over an extent-5 candidate axis is 1, 2, 4 and 5, so two of the
    four widths are padded products. Every candidate is resolved and traced
    before any is compiled, and the invariance check admits them because the
    reduction is what the axis publishes.
    """
    model = _model(
        graph_kind="chain",
        depth=2,
        planned=True,
        value_from="p1",
        enable_jit=True,
        n_wealth=3,
    )

    result = model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=ResultRetention.VALUES,
        execution_config=ExecutionConfig(device_memory_bytes=2**32),
    )

    aaae(
        np.asarray(result.values[0]["working_life"]),
        _chain_expectation(depth=2, planned=True, n_wealth=3),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize(
    "retention", [ResultRetention.VALUES, ResultRetention.VALUES_AND_REPLAY]
)
def test_a_captured_period_replays_the_planned_chain_it_solved(
    *,
    retention: ResultRetention,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """A replay lowers the captured widths against the same producer records."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "working_life@0")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = _model(
        graph_kind="chain",
        depth=3,
        planned=True,
        value_from="p2",
        enable_jit=True,
        n_wealth=3,
    )
    solution = model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        retention=retention,
    )

    replay = replay_period(directory=tmp_path / "working_life@0")

    aaae(
        np.asarray(replay.output.value),
        np.asarray(solution.values[0]["working_life"]),
        decimal=DECIMAL_PRECISION,
    )
