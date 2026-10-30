"""Typed current-period reads use mapped canonical action problems.

The private adapter installed below is a stage fixture. Public Exit declarations
and integrated collective publication are qualified by their own later stages.
"""

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from _lcm import policy_queries as queries
from _lcm.arrival_queries import bind_arrival_query_graph
from _lcm.execution.core_program import (
    CoreArgumentBuilder,
    CoreBuildContext,
    ValueRead,
)
from _lcm.execution.footprint import ResidentBreakdown
from _lcm.execution.output_layout import StateAxesLeading, resolve_output_layout
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.policy_queries import (
    PolicyAnswer,
    QueryKind,
    QueryNode,
    QueryRead,
    build_policy_query_kernel,
    demanded_query_order,
)
from _lcm.query_information import InformationQueryCell
from _lcm.solution import backward_induction as backward
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.typing import PytreeValue
from lcm import (
    CollectiveUtility,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
)
from lcm.arrivals import ArrivalDecision, ArrivalRead
from lcm.collective import ProjectedRegimeValue, StakeholderRoute
from lcm.shocks import Normal, ShockGroup, UserNodes
from lcm.solvers import EGM, GridSearch
from lcm.typing import BoolND, FloatND, IntND, RegimeName, ScalarInt
from tests.conftest import DECIMAL_PRECISION


def _identity(wealth: FloatND) -> FloatND:
    return wealth


def _read(*, kind: QueryKind, target: RegimeName, view: str = "own") -> QueryRead:
    return QueryRead(
        kind=kind,
        target=target,
        phase="solve",
        projection={"wealth": _identity},
        state_inputs=frozenset({"wealth"}),
        physical_inputs=frozenset(),
        operand_paths={"wealth": (view, "wealth")},
        policy_information=frozenset(),
        price_information=frozenset(),
    )


def test_typed_query_order_allows_reciprocal_regime_reads() -> None:
    """Full roots can price a reciprocal regime without a query cycle."""
    reads = {
        "single_raw": _read(kind=QueryKind.RAW, target="single"),
        "couple_participating": _read(kind=QueryKind.PARTICIPATING, target="couple"),
        "single_full": _read(kind=QueryKind.FULL, target="single"),
        "couple_full": _read(kind=QueryKind.FULL, target="couple"),
    }
    dependencies = {
        "single_raw": (),
        "couple_participating": ("single_raw",),
        "single_full": ("couple_participating",),
        "couple_full": ("single_raw", "couple_participating"),
    }
    assert demanded_query_order(
        reads=reads, dependencies=dependencies, roots=("single_full", "couple_full")
    ) == ("single_raw", "couple_participating", "single_full", "couple_full")


def test_query_cycles_and_foreign_full_reads_refuse() -> None:
    """A query cannot recurse or read another root's Full publication."""
    reads = {
        "a": _read(kind=QueryKind.PARTICIPATING, target="a"),
        "b": _read(kind=QueryKind.RAW, target="b"),
    }
    with pytest.raises(ValueError, match="cycle"):
        demanded_query_order(
            reads=reads, dependencies={"a": ("b",), "b": ("a",)}, roots=("a",)
        )
    with pytest.raises(ValueError, match="Full"):
        demanded_query_order(
            reads={"a": reads["a"], "b": replace(reads["b"], kind=QueryKind.FULL)},
            dependencies={"a": ("b",), "b": ()},
            roots=("a",),
        )


def test_integrated_value_reader_is_only_an_arrival_comparison_operand() -> None:
    """A represented Full payload cannot become a policy/support dependency."""
    reads = {
        "arrival": _read(kind=QueryKind.ARRIVAL, target="target"),
        "stored": _read(kind=QueryKind.INTEGRATED_FULL, target="target"),
    }
    assert demanded_query_order(
        reads=reads,
        dependencies={"arrival": ("stored",), "stored": ()},
        roots=("arrival",),
    ) == ("stored", "arrival")
    with pytest.raises(ValueError, match="Only an arrival comparison"):
        demanded_query_order(
            reads={
                **reads,
                "arrival": replace(reads["arrival"], kind=QueryKind.PARTICIPATING),
            },
            dependencies={"arrival": ("stored",), "stored": ()},
            roots=("arrival",),
        )


def test_context_identity_preserves_roles_information_and_operand_paths() -> None:
    """Equal mapped values do not identify differently bound policy contexts."""
    read = _read(kind=QueryKind.RAW, target="single")
    variants = (
        replace(read, kind=QueryKind.PARTICIPATING),
        replace(read, phase="simulate"),
        replace(read, operand_paths={"wealth": ("partner", "wealth")}),
        replace(read, policy_information=frozenset({"job"})),
        replace(read, price_information=frozenset({"job"})),
    )
    assert all(read.identity != other.identity for other in variants)
    assert isinstance(read.projection, MappingProxyType)


def test_transient_query_layout_has_its_own_chunk_prototype() -> None:
    """Mapped query tuples have their own exact shape on the occupant placement."""
    roles = tuple(
        StateAxesLeading(state_names=(), shape=shape, dtype=dtype)
        for shape, dtype in (
            ((2, 2), jnp.asarray(0.0).dtype),
            ((2,), jnp.int32),
            ((2,), jnp.bool_),
            ((2,), jnp.bool_),
        )
    )
    layout = resolve_output_layout(
        core_key="target_participating",
        value_template=jax.device_put(jnp.zeros((3,))),
        state_order=("wealth",),
        output_roles=roles,
    )
    assert tuple(leaf.shape for leaf in layout.expected_leaves) == (
        (2, 2),
        (2,),
        (2,),
        (2,),
    )
    with pytest.raises(ValueError, match="explicit"):
        resolve_output_layout(
            core_key="unchecked_query",
            value_template=jax.device_put(jnp.zeros((3,))),
            state_order=("wealth",),
            output_roles=(StateAxesLeading(state_names=()),),
        )


_CONTEXTS: dict[str, SolverBuildContext] = {}


@pytest.fixture(scope="module", autouse=True)
def _source_binding(record_testsuite_property) -> None:
    """Bind native receipts to this worktree and the actually imported runtime."""
    assert (
        Path(lcm.__file__).resolve().parents[2] == Path(__file__).resolve().parents[2]
    )
    record_testsuite_property("lcm_file", lcm.__file__)
    record_testsuite_property("jax_version", jax.__version__)
    record_testsuite_property("backend", jax.default_backend())


@categorical(ordered=False)
class _RegimeId:
    root: ScalarInt
    target: ScalarInt


@dataclass(frozen=True)
class _CaptureGridSearch(GridSearch):
    """Test-owned adapter collecting the model's actual canonical build inputs."""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        _CONTEXTS[context.regime_name] = context
        return super().build_period_kernels(context=context)


def _target_f(*, wealth: FloatND, action: IntND) -> FloatND:
    return 4 * (1 - action) * (1 - wealth) + action * wealth


def _target_m(*, wealth: FloatND, action: IntND) -> FloatND:
    return (1 - action) * (1 - wealth) + 4 * action * wealth


def _root_f(*, wealth: FloatND, action: IntND) -> FloatND:
    return wealth + 2 * (1 - action)


def _root_m(*, wealth: FloatND, action: IntND) -> FloatND:
    return wealth + 0.5 * (1 - action) + 3 * action


def _root_f_gain(*, wealth: FloatND, action: IntND, gain: FloatND) -> FloatND:
    return _root_f(wealth=wealth, action=action) + gain


def _root_m_gain(*, wealth: FloatND, action: IntND, gain: FloatND) -> FloatND:
    return _root_m(wealth=wealth, action=action) + gain


@categorical(ordered=False)
class _TargetWealth:
    low: ScalarInt
    high: ScalarInt


def _target_code_f(*, wealth_code: IntND, action: IntND) -> FloatND:
    return _target_f(wealth=jnp.where(wealth_code == 0, 0.0, 1.0), action=action)


def _target_code_m(*, wealth_code: IntND, action: IntND) -> FloatND:
    return _target_m(wealth=jnp.where(wealth_code == 0, 0.0, 1.0), action=action)


def _invalid_code(wealth: FloatND) -> IntND:
    return jnp.int32(99) + jnp.int32(0) * wealth.astype(jnp.int32)


def _half(wealth: FloatND) -> FloatND:
    return jnp.asarray(0.5) + 0 * wealth


def _keep(*, q: FloatND, outside: PolicyAnswer) -> FloatND:
    return jnp.all(q >= outside[0])


def _native_model(
    *,
    enable_jit: bool,
    chunk_size: int = 2,
    source_size: int = 3,
    budget: int | None = None,
    parameterized: bool = False,
    fixed_gain: float | None = None,
    invalid_mapped_code: bool = False,
    target_roles: int = 2,
) -> Model:
    _CONTEXTS.clear()
    regimes = {}
    for name, utilities, stop, size in (
        (
            "root",
            {"f": _root_f_gain, "m": _root_m_gain}
            if parameterized
            else {"f": _root_f, "m": _root_m},
            source_size - 1,
            source_size,
        ),
        (
            "target",
            {"f": _target_code_f, "m": _target_code_m}
            if invalid_mapped_code
            else {"f": _target_f, "m": _target_m},
            1,
            2,
        ),
    ):
        grid = (
            DiscreteGrid(category_class=_TargetWealth)
            if name == "target" and invalid_mapped_code
            else LinSpacedGrid(start=0, stop=stop, n_points=size)
        )
        selected_utilities = (
            {f"person_{index}": _target_f for index in range(target_roles)}
            if name == "target" and target_roles != 2
            else utilities
        )
        regimes[name] = Regime(
            states={"wealth": grid},
            actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
            functions={"utility": CollectiveUtility(utilities=selected_utilities)},
            solver=_CaptureGridSearch(),
        )
    model = Model(
        regimes=regimes,
        regime_id_class=_RegimeId,
        n_periods=1,
        initial_nodes=InitialNodes(by_period={0: ("root", "target")}),
        edges={},
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=budget),
        fixed_params={}
        if fixed_gain is None
        else {
            "root": {
                "utility_f": {"gain": fixed_gain},
                "utility_m": {"gain": fixed_gain},
            }
        },
    )
    target = replace(
        _read(kind=QueryKind.PARTICIPATING, target="target"),
        projection={"wealth": _invalid_code if invalid_mapped_code else _half},
    )
    root = _read(kind=QueryKind.FULL, target="root")
    kernel = build_policy_query_kernel(
        root_context=_CONTEXTS["root"],
        period=0,
        definitions={
            "target": QueryNode(read=target, context=_CONTEXTS["target"]),
            "root": QueryNode(
                read=root,
                context=_CONTEXTS["root"],
                inputs={"outside": "target"},
                participation=_keep,
            ),
        },
        root="root",
        chunk_size=chunk_size,
    )
    canonical = model._regimes["root"]
    kernel = kernel.with_fixed_params(
        fixed_flat_params=MappingProxyType(
            {
                name: regime.solution.resolved_fixed_params
                for name, regime in model._regimes.items()
            }
        )
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            dict(model._regimes)
            | {
                "root": replace(
                    canonical,
                    solution=replace(
                        canonical.solution, period_kernels=MappingProxyType({0: kernel})
                    ),
                )
            }
        ),
    )
    return model


def test_arrival_consumer_uses_boolean_rule_and_source_owned_answer_shape() -> None:
    """A singleton arrival on a household context does not rerun its argmax."""
    _native_model(enable_jit=False)

    def decide(*, target: PolicyAnswer, outside: PolicyAnswer) -> PolicyAnswer:
        accept = (target[0][0] > 0.75) & target[2]
        return (
            jnp.where(accept, target[0][0], outside[0]),
            jnp.asarray(-1, jnp.int32),
            jnp.where(accept, target[2], outside[2]),
            target[3] & outside[3],
            jnp.zeros((0,), dtype=jnp.int32),
        )

    node = QueryNode(
        read=_read(kind=QueryKind.ARRIVAL, target="target"),
        context=_CONTEXTS["target"],
        inputs={"target": "part", "outside": "outside"},
        operand_builder=lambda _context: {},
        decision=decide,
        answer_shape=(),
    )
    answer = queries._query_cell(
        jnp.asarray(a=True),
        {"wealth": jnp.asarray(0.5)},
        {
            "target": (
                jnp.asarray([0.5, 2.0]),
                jnp.asarray(0, jnp.int32),
                jnp.asarray(a=True),
                jnp.asarray(a=True),
                jnp.zeros((0,), dtype=jnp.int32),
            ),
            "outside": (
                jnp.asarray(-2.0),
                jnp.asarray(0, jnp.int32),
                jnp.asarray(a=True),
                jnp.asarray(a=True),
                jnp.zeros((0,), dtype=jnp.int32),
            ),
        },
        node=node,
        period=0,
        fixed={"period": jnp.asarray(0), "age": jnp.asarray(0)},
        projection_operands={},
    )
    np.testing.assert_array_equal(np.asarray(answer[0]), -2.0)
    assert bool(answer[2])
    assert bool(answer[3])


def test_bound_arrival_strict_hurdle_uses_one_household_participating_tuple() -> None:
    """An exact threshold can reject a target even when a member's fallback is worse."""
    model = _native_model(enable_jit=False)
    declaration = ArrivalDecision(
        predicate=lambda benchmark: benchmark > 2.0,
        reads={
            "benchmark": ArrivalRead(
                regime="target",
                projection={"wealth": _identity},
                stakeholder="f",
                kind="Participating",
            )
        },
        routes={
            member: StakeholderRoute(
                target_stakeholder=member,
                fallback=ProjectedRegimeValue(
                    regime="root",
                    projection={"wealth": _identity},
                    stakeholder=member,
                ),
            )
            for member in ("f", "m")
        },
    )
    plan = bind_arrival_query_graph(
        contexts=_CONTEXTS,
        canonical_regimes=model._regimes,
        source="root",
        target="target",
        period=0,
        phase="solve",
        declaration=declaration,
    )
    arguments = plan.queries.arguments(
        context=CoreBuildContext(
            state_action_space=_CONTEXTS["target"].state_action_space,
            next_regime_to_V_arr={},
            next_regime_to_continuation={},
            flat_params={},
            period=0,
            ages=model._time,
        )
    )
    answer = plan.queries(
        states={"wealth": jnp.asarray(0.5)},
        arguments=cast("Mapping[str, Mapping[str, PytreeValue]]", arguments),
    )
    np.testing.assert_array_equal(answer[0], [0.5, 3.5])
    assert int(answer[1]) == 1
    assert bool(answer[2])
    assert bool(answer[3])


@pytest.mark.parametrize("enable_jit", [False, True])
def test_native_query_selects_one_mapped_target_tuple(*, enable_jit: bool) -> None:
    """A mapped target tie selects action zero before the root participation test."""
    model = _native_model(enable_jit=enable_jit)
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_array_almost_equal(
        result.values[0]["root"],
        [[2, 0.5], [3, 1.5], [2, 5]],
        decimal=DECIMAL_PRECISION,
    )
    # Stored target values interpolate to (2.5,2.5), an unattainable tuple.
    np.testing.assert_array_almost_equal(
        result.values[0]["target"],
        [[4, 1], [1, 4]],
        decimal=DECIMAL_PRECISION,
    )


def test_noninformation_native_and_scalar_queries_keep_action_ids() -> None:
    """Observed action IDs remain available on ordinary and acting policies."""
    model = _native_model(enable_jit=True)
    context = _CONTEXTS["target"]
    node = QueryNode(
        read=_read(kind=QueryKind.PARTICIPATING, target="target"), context=context
    )
    scalar = queries.ScalarPolicyQueries(
        definitions={"target": node}, root="target", period=0
    )
    arguments = scalar.arguments(
        context=CoreBuildContext(
            state_action_space=context.state_action_space,
            next_regime_to_V_arr={},
            next_regime_to_continuation={},
            flat_params={},
            period=0,
            ages=model._time,
        )
    )
    native = jax.jit(queries._QueryCore(node=node, period=0))(
        **{
            **arguments["target"],
            queries._SOURCE: {"wealth": jnp.asarray([0.0, 1.0])},
            queries._LIVE: jnp.ones((2,), dtype=jnp.bool_),
        }
    )
    np.testing.assert_array_equal(native[0], [[4.0, 1.0], [1.0, 4.0]])
    np.testing.assert_array_equal(native[1], [0, 1])
    for wealth in (0, 1):
        answer = scalar(
            states={"wealth": jnp.asarray(float(wealth))},
            arguments=cast("Mapping[str, Mapping[str, PytreeValue]]", arguments),
        )
        np.testing.assert_array_equal(answer[0], native[0][wealth])
        assert int(answer[1]) == wealth


def _keep_both(*, q: FloatND, left: PolicyAnswer, right: PolicyAnswer) -> FloatND:
    return jnp.all(q >= left[0]) & jnp.all(q >= right[0])


def test_one_raw_answer_feeds_both_readers_in_one_loop() -> None:
    """A producer read by two queries is evaluated once per chunk for both."""
    model = _native_model(enable_jit=True)
    original = model._regimes["root"]
    initial = original.solution.period_kernels[0]
    assert isinstance(initial, queries._PolicyQueryKernel)
    target = initial.definitions["target"]
    raw = replace(target, read=replace(target.read, kind=QueryKind.RAW))
    left = replace(target, inputs={"outside": "raw"}, participation=_keep)
    right = replace(left, read=replace(left.read, role="counterparty"))
    root = replace(
        initial.definitions["root"],
        inputs={"left": "left", "right": "right"},
        participation=_keep_both,
    )
    kernel = build_policy_query_kernel(
        root_context=_CONTEXTS["root"],
        period=0,
        definitions={"raw": raw, "left": left, "right": right, "root": root},
        root="root",
        chunk_size=2,
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            dict(model._regimes)
            | {
                "root": replace(
                    original,
                    solution=replace(
                        original.solution,
                        period_kernels=MappingProxyType({0: kernel}),
                    ),
                )
            }
        ),
    )
    calls, value = _solve_counting_dispatches(model=model)
    assert calls == ["_initialize", "_chunks", "_finish"]
    schedule = cast(
        "queries._QueryChunks", kernel.programs["_chunks"].function
    ).schedule
    assert sorted(step.name for step in schedule) == ["left", "raw", "right", "root"]
    np.testing.assert_array_equal(value, [[2, 0.5], [3, 1.5], [2, 5]])


def test_native_query_bank_is_owned_and_published_by_native_programs(
    *, monkeypatch
) -> None:
    """The chunk loop's bank has declared storage and compiled write/read owners."""
    model = _native_model(enable_jit=True)
    kernel = model._regimes["root"].solution.period_kernels[0]
    assert isinstance(kernel, queries._PolicyQueryKernel)
    assert kernel.programs.keys() == {"_initialize", "_chunks", "_finish"}
    for program in kernel.programs.values():
        banks = program.requirements.host_buffers["query_banks"]
        assert isinstance(banks, jax.ShapeDtypeStruct)
        assert banks.shape == (2, 4, 2)
    calls = []
    invoke = type(kernel).__call__

    def record(*, name, compiled, arguments):
        calls.append(name)
        return compiled(**arguments)

    def observe(self, **arguments):
        cores = arguments["compiled_cores"]
        arguments["compiled_cores"] = {
            name: lambda _name=name, _compiled=compiled, **kwargs: record(
                name=_name, compiled=_compiled, arguments=kwargs
            )
            for name, compiled in cores.items()
        }
        return invoke(self, **arguments)

    monkeypatch.setattr(type(kernel), "__call__", observe)
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    assert calls == ["_initialize", "_chunks", "_finish"]
    np.testing.assert_array_equal(
        result.values[0]["root"], [[2, 0.5], [3, 1.5], [2, 5]]
    )


def test_query_period_compiles_only_its_dispatched_programs(*, monkeypatch) -> None:
    """A query period lowers its bank, its chunk loop and its publication only."""
    model = _native_model(enable_jit=True)
    lowered = []
    lower = backward._lower_resolved_candidate

    def record(*, resolved, label, **kwargs):
        if label.startswith("root "):
            lowered.append(resolved.name)
        return lower(resolved=resolved, label=label, **kwargs)

    monkeypatch.setattr(backward, "_lower_resolved_candidate", record)
    model.solve(params={}, log_level="off", runtime_checks=False)
    assert sorted(lowered) == ["_chunks", "_finish", "_initialize"]


def _solve_counting_dispatches(*, model: Model) -> tuple[list[str], FloatND]:
    """Solve, returning every compiled query core dispatch and the root value."""
    kernel = model._regimes["root"].solution.period_kernels[0]
    calls = []
    invoke = type(kernel).__call__

    def record(*, name, compiled, arguments):
        calls.append(name)
        return compiled(**arguments)

    def observe(self, **arguments):
        cores = arguments["compiled_cores"]
        arguments["compiled_cores"] = {
            name: lambda _name=name, _compiled=compiled, **kwargs: record(
                name=_name, compiled=_compiled, arguments=kwargs
            )
            for name, compiled in cores.items()
        }
        return invoke(self, **arguments)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(kernel), "__call__", observe)
        result = model.solve(params={}, log_level="off", runtime_checks=False)
    return calls, result.values[0]["root"]


def test_query_dispatches_do_not_grow_with_the_number_of_chunks() -> None:
    """One period's query closure dispatches a fixed number of executables."""
    dispatches = {
        chunk_size: _solve_counting_dispatches(
            model=_native_model(enable_jit=True, source_size=8, chunk_size=chunk_size)
        )[0]
        for chunk_size in (1, 3, 8)
    }
    assert dispatches[1] == dispatches[3] == dispatches[8]


@pytest.mark.parametrize("enable_jit", [False, True])
def test_query_chunk_width_keeps_values_bitwise(*, enable_jit: bool) -> None:
    """Unit, ragged-tail and whole-domain chunks publish identical value bits."""
    values = {
        chunk_size: np.asarray(
            _solve_counting_dispatches(
                model=_native_model(
                    enable_jit=enable_jit, source_size=8, chunk_size=chunk_size
                )
            )[1]
        )
        for chunk_size in (1, 3, 8)
    }
    expected = [
        [wealth + 2, wealth + 0.5] if wealth < 2 else [wealth, wealth + 3]
        for wealth in range(8)
    ]
    np.testing.assert_array_equal(values[8], expected)
    assert values[1].tobytes() == values[3].tobytes() == values[8].tobytes()


def test_compiler_budget_selects_a_complete_smaller_common_chunk(
    *, monkeypatch
) -> None:
    """A refused graph is rebuilt before execution with one smaller shared domain."""
    records = []

    def record(**fields):
        records.append(fields)

    monkeypatch.setattr(backward, "_log_core_plan_record", record)
    for width in (32, 1):
        records.clear()
        model = _native_model(
            enable_jit=True, source_size=33, chunk_size=width, budget=2**26
        )
        model.solve(params={}, log_level="debug", runtime_checks=False)
        total = max(
            item["memory"].reservation_bytes + item["resident_bytes"]
            for item in records
        )
        if width == 32:
            wide_total = total
        else:
            narrow_total = total
    assert narrow_total < wide_total
    records.clear()
    model = _native_model(
        enable_jit=True, source_size=33, chunk_size=32, budget=narrow_total
    )
    result = model.solve(params={}, log_level="debug", runtime_checks=False)
    final = {item["triple"]: item for item in records}
    chunks = final[("root", 0, "_chunks")]
    assert chunks["regime"].solution.period_kernels[0].chunk_size < 32
    assert all(
        item["memory"].reservation_bytes + item["resident_bytes"] <= narrow_total
        for item in final.values()
    )
    expected = [
        [wealth + 2, wealth + 0.5] if wealth < 2 else [wealth, wealth + 3]
        for wealth in range(33)
    ]
    np.testing.assert_array_equal(result.values[0]["root"], expected)


@pytest.mark.parametrize("failure", ["exception", "status"])
def test_failed_query_chunks_never_publish_and_drain(
    *, monkeypatch, failure: str
) -> None:
    """A failed chunk loop or reader status publishes no value bank and drains."""
    model = _native_model(enable_jit=True)
    kernel = model._regimes["root"].solution.period_kernels[0]
    events = []
    invoke = type(kernel).__call__
    close = queries.PendingSolveWork.close

    def drain(self):
        events.append("drain")
        return close(self)

    def observe(self, **arguments):
        cores = dict(arguments["compiled_cores"])
        chunks = cores["_chunks"]

        def reject(**kwargs):
            events.append("chunks")
            if failure == "exception":
                raise RuntimeError("seeded query reader failure")
            bank, statuses, totals = chunks(**kwargs)
            return bank, statuses.at[-1].set(False), totals

        def publish(**kwargs):
            del kwargs
            pytest.fail("A failed reader must never publish.")

        arguments["compiled_cores"] = {**cores, "_chunks": reject, "_finish": publish}
        return invoke(self, **arguments)

    monkeypatch.setattr(queries.PendingSolveWork, "close", drain)
    monkeypatch.setattr(type(kernel), "__call__", observe)
    with pytest.raises(
        (RuntimeError, backward.ExecutionPlanningError), match=r"failure|'root'.*status"
    ):
        model.solve(params={}, log_level="off", runtime_checks=False)
    assert events == ["chunks", "drain"]


def test_query_closure_narrows_by_halving_the_chunk() -> None:
    """Narrowing halves the common chunk and refuses below one source cell."""
    kernel = (
        _native_model(enable_jit=True, chunk_size=2)
        ._regimes["root"]
        .solution.period_kernels[0]
    )
    assert isinstance(kernel, queries._PolicyQueryKernel)
    narrowed = kernel.narrow_workspace()
    assert isinstance(narrowed, queries._PolicyQueryKernel)
    assert narrowed.chunk_size == 1
    assert narrowed.narrow_workspace() is None


def test_narrowed_chunk_with_an_unchanged_bank_compiles_its_own_program(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget that admits only the halved chunk solves at the halved chunk.

    Four source cells pad to one bank shape at chunks 2 and 1, so the narrowed
    program has the same arguments as the refused one. It must still be compiled
    and admitted as its own executable, not stand in for the refused width.
    """
    records = []
    log_record = backward._log_core_plan_record

    def record(**kwargs):
        records.append(kwargs)
        return log_record(**kwargs)

    monkeypatch.setattr(backward, "_log_core_plan_record", record)
    _native_model(enable_jit=True, chunk_size=1, source_size=4, budget=2**30).solve(
        params={}, log_level="debug", runtime_checks=False
    )
    budget = max(
        item["memory"].reservation_bytes + item["resident_bytes"] for item in records
    )
    records.clear()
    model = _native_model(enable_jit=True, chunk_size=2, source_size=4, budget=budget)
    result = model.solve(params={}, log_level="debug", runtime_checks=False)
    (chunks,) = (item for item in records if item["triple"][2] == "_chunks")
    assert chunks["memory"].reservation_bytes + chunks["resident_bytes"] <= budget
    np.testing.assert_array_equal(
        result.values[0]["root"],
        _native_model(enable_jit=True, chunk_size=1, source_size=4)
        .solve(params={}, log_level="off", runtime_checks=False)
        .values[0]["root"],
    )


def test_exhausted_common_chunk_refuses_without_dispatch(*, monkeypatch) -> None:
    """A genuine fixed bank floor cannot be hidden by narrowing a query chunk."""
    model = _native_model(enable_jit=True, chunk_size=2, budget=1)
    kernel = model._regimes["root"].solution.period_kernels[0]

    def forbidden(self, **kwargs):
        del self, kwargs
        pytest.fail("An exhausted native closure must not execute.")

    monkeypatch.setattr(type(kernel), "__call__", forbidden)
    with pytest.raises(backward.ExecutionPlanningError, match="budget"):
        model.solve(params={}, log_level="off", runtime_checks=False)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_query_uses_current_call_and_target_fixed_params(*, enable_jit: bool) -> None:
    """Equal query declarations do not cache the previous call's parameter values."""
    model = _native_model(enable_jit=enable_jit, parameterized=True)
    for gain in (7.0, 8.0):
        result = model.solve(
            params={"root": {"utility_f": {"gain": gain}, "utility_m": {"gain": gain}}},
            log_level="off",
            runtime_checks=False,
        )
        np.testing.assert_array_equal(
            result.values[0]["root"],
            [[gain + wealth, gain + wealth + 3] for wealth in range(3)],
        )
    fixed = _native_model(enable_jit=enable_jit, parameterized=True, fixed_gain=7.0)
    result = fixed.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_array_equal(result.values[0]["root"], [[7, 10], [8, 11], [9, 12]])


def test_query_mapped_category_domain_is_mandatory() -> None:
    """A code-reading finite payoff cannot hide an invalid mapped category."""
    model = _native_model(enable_jit=True, invalid_mapped_code=True)
    with pytest.raises(backward.ExecutionPlanningError, match="status"):
        model.solve(params={}, log_level="off", runtime_checks=False)


def _vector_keep(*, q: FloatND, outside: PolicyAnswer) -> jax.Array:
    del outside
    return jnp.ones_like(q, dtype=jnp.bool_)


def test_query_kept_set_requires_one_scalar_per_action() -> None:
    """A stakeholder-shaped predicate must never become an extra action axis."""
    model = _native_model(enable_jit=True)
    original = model._regimes["root"]
    kernel = original.solution.period_kernels[0]
    assert isinstance(kernel, queries._PolicyQueryKernel)
    definitions = {
        **kernel.definitions,
        "root": replace(kernel.definitions["root"], participation=_vector_keep),
    }
    selected = build_policy_query_kernel(
        root_context=_CONTEXTS["root"],
        period=0,
        definitions=definitions,
        root="root",
        chunk_size=2,
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            {
                **model._regimes,
                "root": replace(
                    original,
                    solution=replace(
                        original.solution,
                        period_kernels=MappingProxyType({0: selected}),
                    ),
                ),
            }
        ),
    )
    with pytest.raises(ValueError, match="scalar"):
        model.solve(params={}, log_level="off", runtime_checks=False)


@categorical(ordered=False)
class _RootOnly:
    root: ScalarInt


def _unit() -> FloatND:
    return jnp.asarray(1.0)


def _two() -> FloatND:
    return jnp.asarray(2.0)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_stateless_action_free_query_has_one_native_cell(*, enable_jit: bool) -> None:
    """Empty ordinary axes still produce a real common policy, with no dummy state."""
    model = Model(
        regimes={
            "root": Regime(
                states={},
                actions={},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _unit, "m": _two})
                },
                solver=_CaptureGridSearch(),
            )
        },
        regime_id_class=_RootOnly,
        n_periods=1,
        initial_nodes=InitialNodes(by_period={0: ("root",)}),
        edges={},
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    read = replace(
        _read(kind=QueryKind.RAW, target="root"),
        projection={},
        state_inputs=frozenset(),
        operand_paths={},
    )
    context = _CONTEXTS["root"]
    kernel = build_policy_query_kernel(
        root_context=context,
        period=0,
        root="root",
        chunk_size=3,
        definitions={
            "raw": QueryNode(read=read, context=context),
            "root": QueryNode(
                read=replace(read, kind=QueryKind.FULL),
                context=context,
                inputs={"outside": "raw"},
                participation=_keep,
            ),
        },
    )
    original = model._regimes["root"]
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            {
                "root": replace(
                    original,
                    solution=replace(
                        original.solution, period_kernels=MappingProxyType({0: kernel})
                    ),
                )
            }
        ),
    )
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_array_equal(result.values[0]["root"], [1, 2])


def _finite_heavy_outside(*, q: FloatND, outside: PolicyAnswer) -> jax.Array:
    return jnp.all(jnp.isfinite(outside[0])) & jnp.all(
        q >= -jnp.sum(jnp.abs(outside[0]))
    )


def _fanout_kernel(*, model: Model, chunk_size: int = 2):
    original = model._regimes["root"]
    initial = original.solution.period_kernels[0]
    assert isinstance(initial, queries._PolicyQueryKernel)
    target = initial.definitions["target"]
    raw = replace(target, read=replace(target.read, kind=QueryKind.RAW))
    cheap = QueryNode(
        read=_read(kind=QueryKind.PARTICIPATING, target="root"),
        context=_CONTEXTS["root"],
        inputs={"outside": "raw_a"},
        participation=_finite_heavy_outside,
    )
    root = replace(
        initial.definitions["root"],
        inputs={"left": "left", "right": "right"},
        participation=_keep_both,
    )
    kernel = build_policy_query_kernel(
        root_context=_CONTEXTS["root"],
        period=0,
        root="root",
        chunk_size=chunk_size,
        definitions={
            "raw_a": raw,
            "raw_b": replace(raw, read=replace(raw.read, role="other")),
            "left": cheap,
            "right": replace(cheap, inputs={"outside": "raw_b"}),
            "root": root,
        },
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            {
                **model._regimes,
                "root": replace(
                    original,
                    solution=replace(
                        original.solution, period_kernels=MappingProxyType({0: kernel})
                    ),
                ),
            }
        ),
    )
    return kernel


def test_each_budget_narrowing_names_its_refused_cell(
    *, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A narrowing retry logs the refused triple and its budget refusal."""
    records = []
    log_record = backward._log_core_plan_record

    def record(**kwargs):
        records.append(kwargs)
        return log_record(**kwargs)

    monkeypatch.setattr(backward, "_log_core_plan_record", record)
    model = _native_model(enable_jit=True, target_roles=64, budget=2**26)
    _fanout_kernel(model=model, chunk_size=1)
    model.solve(params={}, log_level="debug", runtime_checks=False)
    budget = max(
        item["memory"].reservation_bytes + item["resident_bytes"] for item in records
    )
    model = _native_model(enable_jit=True, target_roles=64, budget=budget)
    _fanout_kernel(model=model)
    with caplog.at_level(logging.INFO, logger="lcm"):
        model.solve(params={}, log_level="progress", runtime_checks=False)
    narrowings = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Workspace narrowing ")
    ]
    assert narrowings
    for attempt, message in enumerate(narrowings, start=1):
        assert message.startswith(f"Workspace narrowing {attempt}: ")
        assert " at 'root' period 0 refused; " in message
        assert "cannot select a workspace width" in message


def test_one_planning_pass_narrows_every_refused_cell(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two core refusals in one planning pass cost one narrowing and one re-plan.

    Both refused cores belong to the root cell, whose single kernel serves them.
    """
    refusals = []
    narrowed = []
    passes = []
    plan = backward.plan_workspace
    compile_all = backward._compile_all_functions

    def refuse_first_two_selections(**kwargs):
        if len(refusals) < 2:
            refusals.append(len(passes))
            raise backward.WorkspaceBudgetError("refused")
        return plan(**kwargs)

    def count_passes(**kwargs):
        passes.append(len(passes))
        return compile_all(**kwargs)

    def keep_kernel(*, regimes, error):
        narrowed.append(error.triple[:2])
        return regimes

    monkeypatch.setattr(backward, "plan_workspace", refuse_first_two_selections)
    monkeypatch.setattr(backward, "_compile_all_functions", count_passes)
    monkeypatch.setattr(backward, "_narrow_native_closure", keep_kernel)
    model = _native_model(enable_jit=True, budget=2**26)
    model.solve(params={}, log_level="off", runtime_checks=False)
    assert (narrowed, len(passes)) == ([("root", 0)], 2)


def _vector_projection(wealth: FloatND) -> FloatND:
    return jnp.stack((wealth, wealth))


def test_query_projection_cannot_add_an_undeclared_axis() -> None:
    """A helper returns one target coordinate per source cell, never a new axis."""
    model = _native_model(enable_jit=True)
    original = model._regimes["root"]
    kernel = original.solution.period_kernels[0]
    assert isinstance(kernel, queries._PolicyQueryKernel)
    target = kernel.definitions["target"]
    selected = build_policy_query_kernel(
        root_context=_CONTEXTS["root"],
        period=0,
        root="root",
        chunk_size=2,
        definitions={
            **kernel.definitions,
            "target": replace(
                target,
                read=replace(target.read, projection={"wealth": _vector_projection}),
            ),
        },
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            {
                **model._regimes,
                "root": replace(
                    original,
                    solution=replace(
                        original.solution,
                        period_kernels=MappingProxyType({0: selected}),
                    ),
                ),
            }
        ),
    )
    with pytest.raises(ValueError, match="scalar per source cell"):
        model.solve(params={}, log_level="off", runtime_checks=False)


def _log_consumption(consumption: FloatND) -> FloatND:
    return jnp.log(consumption)


def _savings(*, wealth: FloatND, consumption: FloatND) -> FloatND:
    return wealth - consumption


@pytest.mark.parametrize("owner", ["root", "target"])
def test_query_refuses_a_declared_endogenous_grid_owner(owner: RegimeName) -> None:
    """A foreign solver declaration cannot silently receive GridSearch semantics."""
    model = _native_model(enable_jit=True)
    kernel = model._regimes["root"].solution.period_kernels[0]
    assert isinstance(kernel, queries._PolicyQueryKernel)
    grid = LinSpacedGrid(start=0.1, stop=2, n_points=3)
    declaration = ConsumptionSavingsRegime(
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        ),
        states={"wealth": grid},
        actions={"consumption": grid},
        functions={"utility": _log_consumption, "savings": _savings},
        solver=EGM(savings_grid=grid),
    )
    context = replace(
        _CONTEXTS[owner],
        user_regimes=MappingProxyType(
            {
                **_CONTEXTS[owner].user_regimes,
                owner: declaration,
            }
        ),
    )
    definitions = dict(kernel.definitions)
    definitions[owner] = replace(definitions[owner], context=context)
    with pytest.raises(queries.UnsupportedOperationError, match="GridSearch"):
        build_policy_query_kernel(
            root_context=context if owner == "root" else _CONTEXTS["root"],
            period=0,
            definitions=definitions,
            root="root",
            chunk_size=2,
        )


def test_participation_callback_reads_mapped_state_and_current_params() -> None:
    """The queried household's lagged state determines its threshold adjustment."""
    model = _native_model(enable_jit=True, parameterized=True)
    original = model._regimes["root"]
    kernel = original.solution.period_kernels[0]
    assert isinstance(kernel, queries._PolicyQueryKernel)

    def adjusted_keep(
        *, q: FloatND, outside: PolicyAnswer, wealth: FloatND, utility_f__gain: FloatND
    ) -> jax.Array:
        return jnp.all(q >= outside[0] + wealth * utility_f__gain)

    root = replace(kernel.definitions["root"], participation=adjusted_keep)
    updated = build_policy_query_kernel(
        root_context=kernel.root_context,
        period=0,
        definitions={**kernel.definitions, "root": root},
        root="root",
        chunk_size=kernel.chunk_size,
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            dict(model._regimes)
            | {
                "root": replace(
                    original,
                    solution=replace(
                        original.solution, period_kernels=MappingProxyType({0: updated})
                    ),
                )
            }
        ),
    )
    result = model.solve(
        params={"root": {"utility_f": {"gain": 1.0}, "utility_m": {"gain": 1.0}}},
        log_level="off",
        runtime_checks=False,
    )
    np.testing.assert_allclose(result.values[0]["root"], [[3, 1.5], [4, 2.5], [5, 3.5]])


def test_mapped_query_accepts_a_python_scalar_constant() -> None:
    """A constant settlement has no undeclared numerical axis."""
    model = _native_model(enable_jit=True)
    original = model._regimes["root"]
    kernel = original.solution.period_kernels[0]
    assert isinstance(kernel, queries._PolicyQueryKernel)

    def constant() -> float:
        return 0.5

    target = kernel.definitions["target"]
    updated = build_policy_query_kernel(
        root_context=kernel.root_context,
        period=0,
        definitions={
            **kernel.definitions,
            "target": replace(
                target, read=replace(target.read, projection={"wealth": constant})
            ),
        },
        root="root",
        chunk_size=kernel.chunk_size,
    )
    object.__setattr__(
        model,
        "_regimes",
        MappingProxyType(
            dict(model._regimes)
            | {
                "root": replace(
                    original,
                    solution=replace(
                        original.solution, period_kernels=MappingProxyType({0: updated})
                    ),
                )
            }
        ),
    )
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(result.values[0]["root"], [[2, 0.5], [3, 1.5], [2, 5]])


def test_unavailable_benchmark_guards_domain_poison_and_its_derivative() -> None:
    """Unavailable sqrt inputs cannot poison another source cell's tangent."""
    _native_model(enable_jit=True)
    context = _CONTEXTS["target"]

    def settlement(*, wealth: FloatND, shift: FloatND) -> FloatND:
        return jnp.sqrt(wealth + shift)

    def available(wealth: FloatND) -> jax.Array:
        return wealth >= 0

    node = QueryNode(
        read=replace(
            _read(kind=QueryKind.PARTICIPATING, target="target"),
            projection={"wealth": settlement},
        ),
        context=context,
        available=available,
    )
    core = queries._QueryCore(node=node, period=0)

    def value(shift: FloatND) -> FloatND:
        answer = core(
            **{
                queries._SOURCE: {"wealth": jnp.asarray([-1.0, 1.0])},
                queries._LIVE: jnp.ones((2,), dtype=bool),
                queries._PROJECTION: {"shift": shift},
                "period": jnp.int32(0),
                "age": jnp.int32(0),
                "action": context.state_action_space.actions["action"],
                "next_regime_to_V_arr": MappingProxyType({}),
            }
        )
        return jnp.sum(jnp.where(answer[2][:, None], answer[0], 0.0))

    np.testing.assert_allclose(jax.jit(value)(jnp.asarray(0.0)), 5.0)
    np.testing.assert_allclose(jax.jit(jax.grad(value))(jnp.asarray(0.0)), 2.5)


def _information_mesh_inputs(
    *, guarded: bool, events: dict[str, list[float]]
) -> tuple[Model, QueryNode, Mapping[str, PytreeValue]]:
    """Bind a real terminal GridSearch reader and a state-dependent Normal law."""
    _CONTEXTS.clear()

    def mean(wealth: FloatND) -> FloatND:
        return wealth

    def observed_mean(wealth: FloatND) -> FloatND:
        jax.debug.callback(lambda value: events["law"].append(float(value)), wealth)
        return mean(wealth)

    def settlement(wealth: FloatND) -> FloatND:
        jax.debug.callback(
            lambda value: events["projection"].append(float(value)), wealth
        )
        return jnp.sqrt(wealth)

    def utility(*, wealth: FloatND, eps: FloatND) -> FloatND:
        return wealth + eps

    def available(wealth: FloatND) -> BoolND:
        return wealth >= 0

    model = Model(
        regimes={
            name: Regime(
                states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": utility},
                shocks=("innovation",),
                solver=_CaptureGridSearch(),
            )
            for name in ("root", "target")
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: ("root", "target")}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=mean, sigma=1)},
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0], [1.0]]),
                    weights=np.asarray([0.5, 0.5]),
                ),
            )
        },
    )
    context = _CONTEXTS["target"]
    # Instrument only the test-owned query adapter after valid durable construction.
    plan = context.current_iid_plans[0]
    assert plan is not None
    ((group, binding, nodes),) = plan.bindings
    assert group == "innovation"
    law = binding.group.shocks["eps"]
    assert isinstance(law, Normal)
    observed_binding = replace(
        binding,
        group=replace(binding.group, shocks={"eps": replace(law, mu=observed_mean)}),
    )
    context = replace(
        context,
        current_iid_plans=MappingProxyType(
            {0: replace(plan, bindings=((group, observed_binding, nodes),))}
        ),
    )
    node = QueryNode(
        read=replace(
            _read(kind=QueryKind.PARTICIPATING, target="target"),
            projection={"wealth": settlement},
            operand_paths={},
        ),
        context=context,
        available=available if guarded else None,
    )
    arguments = queries.ScalarPolicyQueries(
        definitions={"target": node}, root="target", period=0
    ).arguments(
        context=CoreBuildContext(
            state_action_space=context.state_action_space,
            next_regime_to_V_arr={},
            next_regime_to_continuation={},
            flat_params={},
            period=0,
            ages=model._time,
        )
    )
    return model, node, cast("Mapping[str, PytreeValue]", arguments["target"])


def _information_native_keep(*, q: FloatND, outside: PolicyAnswer) -> BoolND:
    return q >= outside[0]


@pytest.mark.parametrize(
    ("kind", "gate"),
    [
        (QueryKind.RAW, "plain"),
        (QueryKind.PARTICIPATING, "plain"),
        (QueryKind.PARTICIPATING, "live"),
        (QueryKind.PARTICIPATING, "availability"),
        (QueryKind.PARTICIPATING, "admitted-poison"),
    ],
)
def test_information_native_output_omits_hidden_policy_axes(
    *, kind: QueryKind, gate: str
) -> None:
    """Native value transport keeps scalar policies and all economic guards."""
    events: dict[str, list[float]] = {"projection": [], "law": []}
    _model, original, fixed = _information_mesh_inputs(
        guarded=gate == "availability", events=events
    )
    read = replace(
        original.read,
        physical_inputs=frozenset({"eps"}),
        policy_information=frozenset({"eps"}),
    )
    parent = replace(
        original,
        read=replace(read, kind=QueryKind.FULL),
        inputs={"outside": "child"},
        participation=_information_native_keep,
    )
    child = replace(
        original,
        read=replace(read, kind=kind, price_information=frozenset({"eps"})),
        ancestors=("parent",),
    )
    definitions = {"parent": parent, "child": child}
    kernel = build_policy_query_kernel(
        root_context=parent.context,
        period=0,
        definitions=definitions,
        root="parent",
        chunk_size=2,
    )
    wealth = jnp.asarray([1.0, 4.0] if gate == "plain" else [-1.0, 1.0])
    live = jnp.asarray([gate != "live", True])
    source = {"wealth": wealth}
    information = {"parent": fixed}
    cell = InformationQueryCell(
        name="child",
        node=child,
        definitions=definitions,
        period=0,
        fixed=fixed,
        projection_operands={},
        information_arguments=information,
    )

    def scalar_child(index: IntND) -> PolicyAnswer:
        return cell(live=live[index], source={"wealth": wealth[index]}, dependencies={})

    expected = jax.block_until_ready(
        jax.jit(lambda: jax.lax.map(scalar_child, jnp.arange(2)))()
    )
    arguments = {
        **fixed,
        queries._SOURCE: source,
        queries._LIVE: live,
        queries._INFORMATION: information,
    }
    cores = cast("queries._QueryChunks", kernel.programs["_chunks"].function).cores
    lowered = jax.jit(cores["child"]).lower(**arguments)
    assert lowered.out_info[1].shape == live.shape
    actual = jax.block_until_ready(lowered.compile()(**arguments))
    np.testing.assert_array_equal(actual[1], -np.ones(live.shape, np.int32))
    assert actual[1].dtype == jnp.int32
    assert expected[1].ndim > actual[1].ndim
    for index in (0, 2):
        assert actual[index].dtype == expected[index].dtype
        np.testing.assert_array_equal(actual[index], expected[index])
    np.testing.assert_array_equal(actual[3], jnp.all(expected[3]))
    parent_cell = InformationQueryCell(
        name="parent",
        node=parent,
        definitions=definitions,
        period=0,
        fixed=fixed,
        projection_operands={},
        information_arguments={},
    )

    def scalar_parent(index: IntND) -> PolicyAnswer:
        return parent_cell(
            live=live[index],
            source={"wealth": wealth[index]},
            dependencies={
                "outside": (
                    expected[0][index],
                    expected[1][index],
                    expected[2][index],
                    jnp.all(expected[3]),
                    queries._query_landing_totals(counts=expected[4][index]),
                )
            },
        )

    expected_parent = jax.block_until_ready(
        jax.jit(lambda: jax.lax.map(scalar_parent, jnp.arange(2)))()
    )
    actual_parent = jax.block_until_ready(
        jax.jit(cores["parent"])(**{**arguments, "outside": actual})
    )
    np.testing.assert_array_equal(actual_parent[1], -np.ones(live.shape, np.int32))
    for index in (0, 2):
        np.testing.assert_array_equal(actual_parent[index], expected_parent[index])
    np.testing.assert_array_equal(actual_parent[3], jnp.all(expected_parent[3]))
    assert bool(actual[3]) == (gate != "admitted-poison")
    jax.effects_barrier()


@pytest.mark.parametrize("ancestor", [False, True])
@pytest.mark.parametrize("gate", ["live", "availability", "admitted-poison"])
def test_information_mesh_neutral_branch_keeps_axes_and_runtime_guard(
    *, ancestor: bool, gate: str
) -> None:
    """Neutral answers keep manual types while inactive unsafe inputs stay unread."""
    devices = tuple(jax.devices())[:2]
    if len(devices) != 2:
        pytest.skip("Select this test with at least two visible devices.")
    mesh = jax.make_mesh((2,), ("X",), (jax.sharding.AxisType.Auto,), devices=devices)
    events: dict[str, list[float]] = {"projection": [], "law": []}
    _model, node, fixed = _information_mesh_inputs(
        guarded=gate == "availability", events=events
    )
    definitions = {"target": node}
    own = replace(node, ancestors=("parent",), available=None) if ancestor else node
    if ancestor:
        definitions = {"parent": node, "target": own}
    cell = InformationQueryCell(
        name="target",
        node=own,
        definitions=definitions,
        period=0,
        fixed=fixed,
        projection_operands={},
        information_arguments={"parent": fixed} if ancestor else {},
    )

    def local(arguments: Mapping[str, FloatND | BoolND]) -> PolicyAnswer:
        answer = cell(
            live=arguments["live"][0],
            source={"wealth": arguments["wealth"][0]},
            dependencies={},
        )
        for name, leaf in zip(
            ("value", "action", "support", "valid", "landing_count"),
            answer,
            strict=True,
        ):
            manual = getattr(jax.typeof(leaf), "manual_axis_type", None)
            assert manual is not None
            assert manual.varying <= frozenset({"X"}), (name, manual.varying)
            if name == "value":
                assert manual.varying == frozenset({"X"})
        return (
            answer[0][None],
            answer[1][None],
            answer[2][None],
            answer[3][None],
            answer[4][None],
        )

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=({"wealth": jax.P("X"), "live": jax.P("X")},),
        out_specs=(jax.P("X"),) * 5,
        check_vma=True,
    )
    inputs = {
        "wealth": jax.device_put(
            jnp.asarray([-1.0, 1.0]), jax.NamedSharding(mesh, jax.P("X"))
        ),
        "live": jax.device_put(
            jnp.asarray([gate != "live", True]), jax.NamedSharding(mesh, jax.P("X"))
        ),
    }
    jax.effects_barrier()
    for values in events.values():
        values.clear()
    compiled = jax.jit(mapped).lower(inputs).compile()
    assert not any(events.values()), "Abstract type alignment must not run callbacks."
    answer = jax.block_until_ready(compiled(inputs))
    jax.effects_barrier()
    dimensions = (2,) if ancestor else ()
    assert tuple(leaf.shape for leaf in answer) == (
        (2, *dimensions),
        (2, *dimensions, 2),
        (2, *dimensions),
        (2, *dimensions),
        (2, *dimensions, 0),
    )
    assert tuple(leaf.dtype for leaf in answer) == (
        jnp.asarray(0.0).dtype,
        jnp.int32,
        jnp.bool_,
        jnp.bool_,
        jnp.int32,
    )
    assert all(leaf.sharding.spec == jax.P("X") for leaf in answer)
    np.testing.assert_array_equal(answer[0][1], np.full(dimensions, 2.0))
    np.testing.assert_array_equal(answer[1][1], np.zeros((*dimensions, 2), np.int32))
    np.testing.assert_array_equal(answer[2][1], np.ones(dimensions, bool))
    np.testing.assert_array_equal(answer[3][1], np.ones(dimensions, bool))
    if gate == "admitted-poison":
        assert -1.0 in events["projection"]
        assert any(not np.isfinite(value) for value in events["law"])
        np.testing.assert_array_equal(answer[3][0], np.zeros(dimensions, bool))
    else:
        np.testing.assert_array_equal(answer[0][0], np.full(dimensions, -np.inf))
        np.testing.assert_array_equal(
            answer[1][0], -np.ones((*dimensions, 2), np.int32)
        )
        np.testing.assert_array_equal(answer[2][0], np.zeros(dimensions, bool))
        np.testing.assert_array_equal(answer[3][0], np.ones(dimensions, bool))
        assert events["projection"]
        assert events["law"]
        assert all(value == 1.0 for value in events["projection"])
        assert all(value == 1.0 for value in events["law"])


@pytest.mark.parametrize("guarded", [False, True])
def test_information_mesh_unavailable_observation_keeps_producer_unexecuted(
    *, guarded: bool
) -> None:
    """The neutral frame preserves manual types without executing its physical law."""
    devices = tuple(jax.devices())[:2]
    if len(devices) != 2:
        pytest.skip("Select this test with at least two visible devices.")
    mesh = jax.make_mesh((2,), ("X",), (jax.sharding.AxisType.Auto,), devices=devices)
    events: dict[str, list[float]] = {"projection": [], "law": []}
    _model, producer, fixed = _information_mesh_inputs(guarded=guarded, events=events)
    observer = replace(
        producer,
        observation_sources={"innovation": "producer"},
        observation_aliases={"innovation": ("producer",)},
        observation_counts=(2,),
    )
    cell = InformationQueryCell(
        name="observer",
        node=observer,
        definitions={"observer": observer, "producer": producer},
        period=0,
        fixed=fixed,
        projection_operands={},
        information_arguments={"producer": fixed},
    )

    def local(wealth: FloatND) -> tuple[FloatND, BoolND, FloatND]:
        frame, weights = cell._observer_row(
            observer=observer,
            target=producer,
            group="innovation",
            fixed=fixed,
            row=jnp.int32(0),
            observed=frozenset({"eps"}),
            source={"wealth": wealth[0]},
            frames={},
        )
        return frame.physical["eps"][None], frame.valid[None], weights[None]

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=jax.P("X"),
        out_specs=(jax.P("X"),) * 3,
        check_vma=True,
    )
    wealth = jax.device_put(
        jnp.asarray([-1.0, 1.0]), jax.NamedSharding(mesh, jax.P("X"))
    )
    jax.effects_barrier()
    for values in events.values():
        values.clear()
    compiled = jax.jit(mapped).lower(wealth).compile()
    assert not any(events.values()), "Abstract frame alignment must not run callbacks."
    physical, valid, weights = jax.block_until_ready(compiled(wealth))
    jax.effects_barrier()
    assert physical.shape == valid.shape == (2,)
    assert weights.shape == (2, 1)
    np.testing.assert_array_equal(physical[1], 0.0)
    np.testing.assert_array_equal(valid[1], desired=True)
    np.testing.assert_array_equal(weights[1], [0.5])
    if guarded:
        np.testing.assert_array_equal(physical[0], 0.0)
        np.testing.assert_array_equal(valid[0], desired=True)
        np.testing.assert_array_equal(weights[0], [1.0])
        assert events["projection"]
        assert events["law"]
        assert all(value == 1.0 for value in events["projection"])
        assert all(value == 1.0 for value in events["law"])
    else:
        assert -1.0 in events["projection"]
        assert any(not np.isfinite(value) for value in events["law"])
        np.testing.assert_array_equal(valid[0], desired=False)


def test_budget_narrowing_retries_reuse_executables_already_compiled(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A narrowing retry compiles no lowering key an earlier attempt compiled."""
    records = []
    log_record = backward._log_core_plan_record

    def record(**kwargs):
        records.append(kwargs)
        return log_record(**kwargs)

    monkeypatch.setattr(backward, "_log_core_plan_record", record)
    model = _native_model(enable_jit=True, target_roles=64, budget=2**26)
    _fanout_kernel(model=model, chunk_size=1)
    model.solve(params={}, log_level="debug", runtime_checks=False)
    budget = max(
        item["memory"].reservation_bytes + item["resident_bytes"] for item in records
    )
    compiled_keys = []
    compile_and_log = backward._compile_and_log

    def count(**kwargs):
        compiled_keys.append(kwargs["lowering_key"])
        return compile_and_log(**kwargs)

    monkeypatch.setattr(backward, "_compile_and_log", count)
    narrowings = []
    narrow = backward._narrow_native_closure

    def observe(**kwargs):
        narrowings.append(kwargs["error"].triple)
        return narrow(**kwargs)

    monkeypatch.setattr(backward, "_narrow_native_closure", observe)
    model = _native_model(enable_jit=True, target_roles=64, budget=budget)
    _fanout_kernel(model=model)
    model.solve(params={}, log_level="off", runtime_checks=False)
    assert narrowings
    assert len(compiled_keys) == len(set(compiled_keys))


def test_budget_refusal_logs_the_resident_split_of_its_busiest_device(
    *, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A budget refusal warns with resident components summing to its total."""
    records = []
    log_record = backward._log_core_plan_record

    def record(**kwargs):
        records.append(kwargs)
        return log_record(**kwargs)

    monkeypatch.setattr(backward, "_log_core_plan_record", record)
    model = _native_model(enable_jit=True, target_roles=64, budget=2**26)
    _fanout_kernel(model=model, chunk_size=1)
    model.solve(params={}, log_level="debug", runtime_checks=False)
    budget = max(
        item["memory"].reservation_bytes + item["resident_bytes"] for item in records
    )
    model = _native_model(enable_jit=True, target_roles=64, budget=budget)
    kernel = _fanout_kernel(model=model)
    monkeypatch.setattr(type(kernel), "narrow_workspace", lambda _self: None)
    with (
        caplog.at_level(logging.WARNING, logger="lcm"),
        pytest.raises(backward.ExecutionPlanningError, match="budget") as error,
    ):
        model.solve(params={}, log_level="progress", runtime_checks=False)
    (resident,) = (
        int(value)
        for value in re.findall(r"keeps (\d+) bytes resident", str(error.value))
    )
    splits = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.getMessage().startswith("Resident split for regime 'root'")
    ]
    total, *components = (int(value) for value in re.findall(r"=(\d+)", splits[-1]))
    assert (total, sum(components)) == (resident, resident)


def test_resident_split_aggregates_owners_and_bounds_the_listing(
    *, caplog: pytest.LogCaptureFixture
) -> None:
    """Live groups sum per kind and owning regime; the smallest share a remainder."""
    values = {
        frozenset(
            {
                ValueArtifactAddress(
                    kind=ValueArtifactKind.REGIME_VALUE, period=period, regime=name
                )
            }
        ): size
        for name, size in zip("abcdefghi", range(10, 100, 10), strict=True)
        for period in (1, 2)
    }
    breakdown = ResidentBreakdown(
        device_id=0,
        live=MappingProxyType({**values, frozenset({"scratch"}): 1}),
        peer_bytes=2,
        internal_bytes=3,
        fixed_bytes=4,
        transfer_scratch_bytes=5,
    )
    with caplog.at_level(logging.WARNING, logger="lcm"):
        backward._log_resident_split(
            logger=logging.getLogger("lcm"),
            triple=("root", 0, "core"),
            breakdown=breakdown,
        )
    assert caplog.records[-1].getMessage() == (
        "Resident split for regime 'root' period 0 core 'core' on device 0: "
        "total=915 = live[regime_value:i]=180 + live[regime_value:h]=160 + "
        "live[regime_value:g]=140 + live[regime_value:f]=120 + "
        "live[regime_value:e]=100 + live[regime_value:d]=80 + "
        "live[regime_value:c]=60 + live[regime_value:b]=40 + "
        "live[remainder of 2 owners]=21 + internal=3 + peer_outputs=2 + "
        "fixed_owners=4 + transfer_scratch=5"
    )


def test_query_core_repr_is_bounded_by_its_name() -> None:
    """A jitted query core renders in a few hundred characters, whatever its graph."""
    _native_model(enable_jit=False)
    node = QueryNode(
        read=_read(kind=QueryKind.PARTICIPATING, target="target"),
        context=_CONTEXTS["target"],
    )
    definitions = {f"target_{i}": node for i in range(100)}
    core = queries._QueryCore(
        node=node, period=0, name="target_0", definitions=definitions
    )
    assert len(repr(core)) < 300


def test_value_reads_walk_each_query_node_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every core key reuses one walk of the graph's continuation operands."""
    _native_model(enable_jit=False)
    node = QueryNode(
        read=_read(kind=QueryKind.PARTICIPATING, target="target"),
        context=_CONTEXTS["target"],
    )
    scalar = queries.ScalarPolicyQueries(
        definitions={"first": node, "second": node}, root="first", period=0
    )
    walked = []
    node_inputs = queries._node_inputs

    def counted(
        *, node: QueryNode, period: int
    ) -> tuple[CoreArgumentBuilder, tuple[ValueRead, ...]]:
        walked.append(node)
        return node_inputs(node=node, period=period)

    monkeypatch.setattr(queries, "_node_inputs", counted)
    reads = {key: scalar.value_reads(core_key=key) for key in ("a", "b", "a")}
    assert len(walked) == 2
    assert {read.source.core_key for read in reads["a"]} <= {"a"}
    assert [
        replace(read, source=replace(read.source, core_key="a")) for read in reads["b"]
    ] == list(reads["a"])
