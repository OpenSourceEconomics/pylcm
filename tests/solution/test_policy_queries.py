"""Typed current-period reads use mapped canonical action problems.

The private adapter installed below is a stage fixture. Public Exit declarations
and integrated collective publication are qualified by their own later stages.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from _lcm import policy_queries as queries
from _lcm.execution.output_layout import StateAxesLeading, resolve_output_layout
from _lcm.policy_queries import (
    PolicyAnswer,
    QueryKind,
    QueryNode,
    QueryRead,
    build_policy_query_kernel,
    demanded_query_order,
)
from _lcm.solution import backward_induction as backward
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
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
from lcm.solvers import EGM, GridSearch
from lcm.typing import FloatND, IntND, RegimeName, ScalarInt
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
    recompute: bool = False,
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
        recompute=recompute,
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


def _keep_both(*, q: FloatND, left: PolicyAnswer, right: PolicyAnswer) -> FloatND:
    return jnp.all(q >= left[0]) & jnp.all(q >= right[0])


def test_recomputation_reruns_the_identical_native_producer(*, monkeypatch) -> None:
    """A shared Raw query can be evicted and replayed byte for byte per chunk."""
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
    observations = []
    runs = []
    invoke = type(kernel).__call__

    def record(*, name, compiled, arguments):
        output = compiled(**arguments)
        observations.append(
            (
                name,
                id(getattr(compiled, "compiled", compiled)),
                tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(output)),
            )
        )
        return output

    def observe(self, **arguments):
        cores = arguments["compiled_cores"]
        arguments["compiled_cores"] = {
            name: lambda _name=name, _compiled=compiled, **kwargs: record(
                name=_name, compiled=_compiled, arguments=kwargs
            )
            for name, compiled in cores.items()
        }
        outputs = []
        for recompute in (False, True):
            observations.clear()
            output = invoke(
                self.with_recomputation() if recompute else self, **arguments
            )
            outputs.append(np.asarray(output.value).tobytes())
            runs.append(tuple(observations))
        assert outputs[0] == outputs[1]
        return output

    monkeypatch.setattr(type(kernel), "__call__", observe)

    def install(selected):
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
                            period_kernels=MappingProxyType({0: selected}),
                        ),
                    )
                }
            ),
        )

    install(kernel)
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    retained_raw = [item for item in runs[0] if item[0] == "raw"]
    recomputed_raw = [item for item in runs[1] if item[0] == "raw"]
    assert len(retained_raw) == 2
    assert len(recomputed_raw) == 4
    assert [item[2] for item in recomputed_raw] == [
        item[2] for item in retained_raw for _ in range(2)
    ]
    assert {item[1] for item in retained_raw} == {item[1] for item in recomputed_raw}
    np.testing.assert_array_equal(
        result.values[0]["root"], [[2, 0.5], [3, 1.5], [2, 5]]
    )


def test_native_query_bank_is_owned_and_published_by_native_programs(
    *, monkeypatch
) -> None:
    """Host-loop output banks have declared storage and compiled write/read owners."""
    model = _native_model(enable_jit=True)
    kernel = model._regimes["root"].solution.period_kernels[0]
    assert isinstance(kernel, queries._PolicyQueryKernel)
    assert {"_initialize", "_source", "_publish", "_finish"} <= kernel.programs.keys()
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
    assert calls.count("_initialize") == 1
    assert calls.count("_source") == 2
    assert calls.count("_publish") == 2
    assert calls.count("_finish") == 1
    np.testing.assert_array_equal(
        result.values[0]["root"], [[2, 0.5], [3, 1.5], [2, 5]]
    )


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
    target = final[("root", 0, "target")]
    assert target["layout"].expected_value_shape[0] < 32
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
def test_failed_query_reader_keeps_inputs_uncommitted_and_drains(
    *, monkeypatch, failure: str
) -> None:
    """A failed reader neither releases its producer nor publishes a value bank."""
    model = _native_model(enable_jit=True)
    kernel = model._regimes["root"].solution.period_kernels[0]
    ledgers = []
    events = []
    held = []
    ledger_type = queries.PlannedInputLiveness
    invoke = type(kernel).__call__
    close = queries.PendingSolveWork.close

    def ledger(**kwargs):
        instance = ledger_type(**kwargs)
        ledgers.append(instance)
        return instance

    def drain(self):
        events.append("drain")
        return close(self)

    def observe(self, **arguments):
        cores = dict(arguments["compiled_cores"])
        target = cores["target"]
        root = cores["root"]

        def produce(**kwargs):
            answer = target(**kwargs)
            held.append(answer)
            return answer

        def reject(**kwargs):
            events.append("reader")
            if failure == "exception":
                raise RuntimeError("seeded query reader failure")
            answer = root(**kwargs)
            return (*answer[:3], jnp.zeros_like(answer[3]))

        def publish(**kwargs):
            del kwargs
            pytest.fail("A failed reader must never publish.")

        arguments["compiled_cores"] = {
            **cores,
            "target": produce,
            "root": reject,
            "_publish": publish,
        }
        return invoke(self, **arguments)

    monkeypatch.setattr(queries, "PlannedInputLiveness", ledger)
    monkeypatch.setattr(queries.PendingSolveWork, "close", drain)
    monkeypatch.setattr(type(kernel), "__call__", observe)
    with pytest.raises(
        (RuntimeError, backward.ExecutionPlanningError), match=r"failure|status"
    ):
        model.solve(params={}, log_level="off", runtime_checks=False)
    assert events == ["reader", "drain"]
    assert not ledgers[0].is_release_eligible(artifact=(0, "target", 0, "answer"))
    assert all(not array.is_deleted() for array in jax.tree.leaves(held))
    with pytest.raises(RuntimeError, match="uncommitted"):
        ledgers[0].assert_solve_complete()


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


def _fanout_kernel(*, model: Model, recompute: bool):
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
        chunk_size=2,
        recompute=recompute,
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


def test_recomputation_materially_admits_a_smaller_native_budget(
    *, monkeypatch
) -> None:
    """Cheap readers do not require independent heavy answers to remain co-resident."""
    records = []
    selected = []
    totals = {}
    log_record = backward._log_core_plan_record

    def record(**kwargs):
        records.append(kwargs)
        return log_record(**kwargs)

    monkeypatch.setattr(backward, "_log_core_plan_record", record)
    for recompute in (False, True):
        records.clear()
        model = _native_model(enable_jit=True, target_roles=64, budget=2**26)
        _fanout_kernel(model=model, recompute=recompute)
        result = model.solve(params={}, log_level="debug", runtime_checks=False)
        np.testing.assert_array_equal(
            result.values[0]["root"], [[0, 3], [1, 4], [2, 5]]
        )
        totals[recompute] = max(
            item["memory"].reservation_bytes + item["resident_bytes"]
            for item in records
        )
    assert totals[True] < totals[False]
    records.clear()
    model = _native_model(enable_jit=True, target_roles=64, budget=totals[True])
    kernel = _fanout_kernel(model=model, recompute=False)
    invoke = type(kernel).__call__

    def observe(self, **kwargs):
        selected.append((self.recompute, self.chunk_size))
        return invoke(self, **kwargs)

    monkeypatch.setattr(type(kernel), "__call__", observe)
    with monkeypatch.context() as blocked:
        blocked.setattr(type(kernel), "narrow_workspace", lambda _self: None)
        with pytest.raises(backward.ExecutionPlanningError, match="budget"):
            model.solve(params={}, log_level="debug", runtime_checks=False)
    assert not selected
    result = model.solve(params={}, log_level="debug", runtime_checks=False)
    assert selected == [(True, 2)]
    np.testing.assert_array_equal(result.values[0]["root"], [[0, 3], [1, 4], [2, 5]])


@pytest.mark.parametrize("mutation", ["missing", "generation", "omitted", "duplicate"])
def test_closed_schedule_refuses_incomplete_or_wrong_generation_bindings(
    mutation: str,
) -> None:
    """A claimed smaller frontier must match all actual native input references."""
    model = _native_model(enable_jit=True, budget=2**26)
    original = model._regimes["root"]
    kernel = original.solution.period_kernels[0]
    assert isinstance(kernel, queries._PolicyQueryKernel)
    schedule = list(kernel.programs["root"].requirements.internal_schedule)
    root = schedule[-2]
    if mutation == "missing":
        schedule[-2] = replace(
            root, inputs={k: v for k, v in root.inputs.items() if k != "outside"}
        )
    elif mutation == "generation":
        schedule[-2] = replace(root, inputs={**root.inputs, "outside": ("target", 99)})
    elif mutation == "omitted":
        schedule.pop(1)
    else:
        schedule.insert(1, schedule[1])
    selected = replace(
        kernel,
        programs=MappingProxyType(
            {
                name: replace(
                    program,
                    requirements=replace(
                        program.requirements, internal_schedule=tuple(schedule)
                    ),
                )
                for name, program in kernel.programs.items()
            }
        ),
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
    with pytest.raises(backward.ExecutionPlanningError, match="schedule"):
        model.solve(params={}, log_level="off", runtime_checks=False)


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
