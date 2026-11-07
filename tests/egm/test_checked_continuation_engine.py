"""Runtime checks select paired continuation counts before engine lowering."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    TypedDict,
    Unpack,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.dcegm_query import DCEGMQuery, _BlendedRow
from _lcm.egm.nested_floor import PreparedNestedFloor
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.egm.step import _EGMStep
from _lcm.execution.core_program import CoreProgram
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.solution import backward_induction as induction
from _lcm.solution import egm as egm_module
from _lcm.solution.dcegm import _DCEGMStepOutput
from _lcm.solution.diagnostics import _DiagnosticRow
from _lcm.solution.egm import _EGMArgumentBuilder
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    ContinuationReadCounts,
    FeasibilityKernelOutput,
    QueryInputStatus,
)
from _lcm.solution.kernel_output import ConsumedKernelOutput
from _lcm.typing import EconFunctionArg, EconFunctionKwargs, HostArray, PytreeValue
from lcm import Model
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import (
    EGM_CONTINUATION,
    ArtifactAuthority,
    ArtifactKey,
    KernelOutput,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    Float1D,
    FloatND,
    RegimeName,
    ScalarFloat,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_below_first_live_counts import (
    RegimeId,
    _model,
    _params,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type _Run = Literal["unchecked", "checked", "bad-unchecked", "bad-checked"]
type _FloorState = tuple[ScalarFloat, Float1D, BoundInputStatus]
type _CountMetadata = tuple[tuple[tuple[int, ...], str], ...]


class _SelectInputs(TypedDict):
    graph: Mapping[str, CoreProgram]
    retain_replay: bool
    selected_artifact_keys: NotRequired[frozenset[ArtifactKey]]
    collect_continuation_read_counts: NotRequired[bool]


class _ConsumeInputs(TypedDict):
    output: KernelOutput
    continuation_key: ArtifactKey | None
    regime_name: RegimeName
    period: int
    artifact_authorities: NotRequired[Mapping[ArtifactKey, ArtifactAuthority]]


class _DcReadInputs(TypedDict):
    query: DCEGMQuery
    liquid: Float1D
    grid: Float1D
    value: Float1D
    marginal: Float1D
    data: EGMQueryData | None
    pool: EconFunctionKwargs | None
    floor_state: NotRequired[_FloorState | None]
    active: NotRequired[BoolND | None]


class _CountedRowInputs(TypedDict):
    grid: FloatND
    value: FloatND
    marginal: FloatND
    data: EGMQueryData
    row_queries: FloatND
    row_gradients: FloatND
    weights: tuple[Float1D, ...]
    active: BoolND
    negative: BoolND
    paired_marginal_read: bool


@runtime_checkable
class _ConsumeCall(Protocol):
    """The existing consumer keyword contract retained by its observer."""

    def __call__(
        self,
        *,
        output: KernelOutput,
        continuation_key: ArtifactKey | None,
        regime_name: RegimeName,
        period: int,
        artifact_authorities: Mapping[ArtifactKey, ArtifactAuthority] = ...,
    ) -> ConsumedKernelOutput: ...


@runtime_checkable
class _DcReadCall(Protocol):
    """The existing physical-row keyword contract retained by its observer."""

    def __call__(
        self,
        *,
        query: DCEGMQuery,
        liquid: Float1D,
        grid: Float1D,
        value: Float1D,
        marginal: Float1D,
        data: EGMQueryData | None,
        pool: EconFunctionKwargs | None,
        floor_state: _FloorState | None = None,
        active: BoolND | None = None,
    ) -> tuple[Float1D, Float1D, QueryInputStatus, _FloorState]: ...


@runtime_checkable
class _StepCall(Protocol):
    """The existing unbound DC step call, including execution-plan keywords."""

    # keyword-only-exempt: library-callback=_EGMStep.__call__
    def __call__(
        self,
        step: _EGMStep,
        /,
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
        _lcm_stochastic_node_width: int | None = None,
        _lcm_cell_width: int | None = None,
        _lcm_savings_point_width: int | None = None,
        _lcm_euler_point_width: int | None = None,
        _lcm_envelope_cell_width: int = 1,
        _lcm_prepared_nested_floor: PreparedNestedFloor | None = None,
        **kwargs: EconFunctionArg,
    ) -> _DCEGMStepOutput: ...


class _StepTuple(NamedTuple):
    run: _Run
    grid: HostArray
    value: HostArray
    marginal: HostArray
    consumption: HostArray
    savings: HostArray | None
    kind: HostArray | None
    corner: HostArray | None


class _Selection(NamedTuple):
    run: _Run
    enabled: bool
    declared: CoreProgram
    selected: CoreProgram


class _Read(NamedTuple):
    run: _Run
    initial: bool
    queries: HostArray
    active: HostArray
    clock: HostArray
    floor: HostArray
    nodes: HostArray
    status: HostArray


class _Output(NamedTuple):
    run: _Run
    regime: RegimeName
    period: int
    numeric: tuple[HostArray, ...]
    bound: tuple[HostArray, ...]
    query: tuple[HostArray, ...]
    feasibility: tuple[int, int] | None
    counts_present: bool
    counts_metadata: _CountMetadata
    counts_forwarded: bool


class _HostRecord(NamedTuple):
    run: _Run
    counts: tuple[int, int] | None


@dataclass(kw_only=True)
class _Observations:
    """Test-local facts copied before rolling continuations can be released."""

    run: _Run = "unchecked"
    selections: list[_Selection] = field(default_factory=list)
    reads: list[_Read] = field(default_factory=list)
    outputs: list[_Output] = field(default_factory=list)
    first: list[tuple[_Run, float]] = field(default_factory=list)
    traces: list[_Run] = field(default_factory=list)
    host: list[_HostRecord] = field(default_factory=list)
    materializations: list[tuple[_Run, int]] = field(default_factory=list)
    step_outputs: list[_StepTuple] = field(default_factory=list)


def _parent_utility(*, consumption: ContinuousAction) -> FloatND:
    """Current consumption carries twice the continuation's log weight."""
    return 2.0 * jnp.log(consumption)


def _parent_inverse(*, marginal_continuation: FloatND) -> ContinuousAction:
    """Invert the parent's marginal felicity 2/C."""
    return 2.0 / marginal_continuation


def _engine_model() -> tuple[Model, UserParams, HostArray]:
    """Use the qualified child publication with four independent source candidates."""
    dtype = np.dtype(jnp.asarray(0.0).dtype)
    first = np.asarray(4.0 / 3.0, dtype=dtype) * np.asarray(1e-4, dtype=dtype)
    nodes = np.asarray(
        [
            np.nextafter(first, np.asarray(-np.inf, dtype=dtype)),
            first,
            np.nextafter(first, np.asarray(np.inf, dtype=dtype)),
            np.asarray(1.0, dtype=dtype),
        ],
        dtype=dtype,
    )
    seed = _model(literal=False, parent_nodes=tuple(float(node) for node in nodes))
    parent = seed.user_regimes["parent"]
    regimes = {
        **seed.user_regimes,
        "parent": parent.replace(
            functions={
                **parent.functions,
                "utility": _parent_utility,
                "inverse_marginal_utility": _parent_inverse,
            }
        ),
    }
    model = Model(
        regimes=regimes,
        ages=seed.ages,
        edges=seed.edges,
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
        enable_jit=True,
    )
    return model, _params(literal=False, parent=True), nodes


def _host_pair(value: PytreeValue) -> tuple[int, int] | None:
    """Require released diagnostic metadata to contain only Python integers."""
    if value is None:
        return None
    assert type(value) is tuple, (
        "A completed diagnostic row must release its native continuation counts."
    )
    assert len(value) == 2, (
        "A completed diagnostic row must release its native continuation counts."
    )
    below, eligible = value
    assert type(below) is int
    assert type(eligible) is int
    return below, eligible


def _numeric_snapshot(output: KernelOutput) -> tuple[HostArray, ...]:
    """Copy ordinary fields while their period still owns the device buffers."""
    leaves = [output.value]
    for channel in (output.continuations, output.replay):
        for key in sorted(
            channel, key=lambda item: (item.type_id, item.schema_version)
        ):
            leaves.extend(jax.tree.leaves(channel[key]))
    return tuple(np.asarray(leaf).copy() for leaf in leaves)


def _assert_first_tuple_from_step(*, carry: EGMCarry, snapshot: _StepTuple) -> float:
    """The actual step policy and owned continuation publish one joint first tuple."""
    leaves = carry.leaves()
    grid = np.asarray(leaves[("endog_grid",)])
    value = np.asarray(leaves[("value",)])
    marginal = np.asarray(leaves[("marginal_utility",)])
    consumption = np.asarray(snapshot.consumption)
    assert grid.ndim == value.ndim == marginal.ndim == consumption.ndim == 1
    euler_first = np.asarray(4.0 / 3.0, dtype=grid.dtype)
    expected_first = euler_first * np.asarray(1e-4, dtype=grid.dtype)
    np.testing.assert_array_equal(grid[0], expected_first)
    assert 0.0 < grid[0] < euler_first
    np.testing.assert_array_equal(consumption[0], grid[0])
    np.testing.assert_allclose(
        [value[0], marginal[0]],
        [np.log(consumption[0]) + np.log(4.0), 1.0 / consumption[0]],
        rtol=10.0**-DECIMAL_PRECISION,
        atol=0.0,
    )
    data = carry.query_data_for_transport()
    if data is not None:
        np.testing.assert_array_equal(data.savings[0], 0.0)
        np.testing.assert_array_equal(data.node_kind[0], QueryNodeKind.FINITE)
        assert data.corner_continuation is not None
        np.testing.assert_array_almost_equal(
            data.corner_continuation[0], np.log(4.0), decimal=DECIMAL_PRECISION
        )
    np.testing.assert_array_equal(snapshot.grid, grid)
    np.testing.assert_array_equal(snapshot.value, value)
    np.testing.assert_array_equal(snapshot.marginal, marginal)
    assert data is not None
    assert snapshot.savings is not None
    assert snapshot.kind is not None
    assert snapshot.corner is not None
    np.testing.assert_array_equal(snapshot.savings, np.asarray(data.savings))
    np.testing.assert_array_equal(snapshot.kind, np.asarray(data.node_kind))
    np.testing.assert_array_equal(snapshot.corner, np.asarray(data.corner_continuation))
    return float(grid[0])


def _make_step_observer(
    *, observed: _Observations, original_step: _StepCall
) -> _StepCall:
    """Capture the returned native policy before the values core drops that output."""

    # keyword-only-exempt: library-callback=_EGMStep.__call__
    def step(
        self: _EGMStep,
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
        _lcm_stochastic_node_width: int | None = None,
        _lcm_cell_width: int | None = None,
        _lcm_savings_point_width: int | None = None,
        _lcm_euler_point_width: int | None = None,
        _lcm_envelope_cell_width: int = 1,
        _lcm_prepared_nested_floor: PreparedNestedFloor | None = None,
        **kwargs: EconFunctionArg,
    ) -> _DCEGMStepOutput:
        result = original_step(
            self,
            next_regime_to_continuation=next_regime_to_continuation,
            _lcm_stochastic_node_width=_lcm_stochastic_node_width,
            _lcm_cell_width=_lcm_cell_width,
            _lcm_savings_point_width=_lcm_savings_point_width,
            _lcm_euler_point_width=_lcm_euler_point_width,
            _lcm_envelope_cell_width=_lcm_envelope_cell_width,
            _lcm_prepared_nested_floor=_lcm_prepared_nested_floor,
            **kwargs,
        )
        output = result.output if isinstance(result, BoundedResult) else result
        carry, policy = output[1], output[2]
        data = carry.query_data_for_transport()
        leaves = carry.leaves()

        # keyword-only-exempt: library-callback=jax.debug.callback
        def record(
            grid: HostArray,
            value: HostArray,
            marginal: HostArray,
            consumption: HostArray,
            savings: HostArray | None,
            kind: HostArray | None,
            corner: HostArray | None,
        ) -> None:
            observed.step_outputs.append(
                _StepTuple(
                    observed.run,
                    grid.copy(),
                    value.copy(),
                    marginal.copy(),
                    consumption.copy(),
                    None if savings is None else savings.copy(),
                    None if kind is None else kind.copy(),
                    None if corner is None else corner.copy(),
                )
            )

        jax.debug.callback(
            record,
            leaves[("endog_grid",)],
            leaves[("value",)],
            leaves[("marginal_utility",)],
            policy.policy,
            None if data is None else data.savings,
            None if data is None else data.node_kind,
            None if data is None else data.corner_continuation,
        )
        return result

    return step


def _make_consume_observer(
    *, observed: _Observations, original_consume: _ConsumeCall
) -> _ConsumeCall:
    """Keep the exact real consumer call and its bounded output observation together."""

    def consume(**kwargs: Unpack[_ConsumeInputs]) -> ConsumedKernelOutput:
        result = original_consume(**kwargs)
        output = kwargs["output"]
        regime, period = kwargs["regime_name"], kwargs["period"]
        if observed.run in ("unchecked", "checked") and (regime, period) == (
            "child",
            1,
        ):
            carry = output.continuations[EGM_CONTINUATION]
            assert isinstance(carry, EGMCarry)
            jax.effects_barrier()
            snapshots = [
                snapshot
                for snapshot in observed.step_outputs
                if snapshot.run == observed.run
            ]
            assert len(snapshots) == 1
            observed.first.append(
                (
                    observed.run,
                    _assert_first_tuple_from_step(carry=carry, snapshot=snapshots[0]),
                )
            )
        raw_counts = getattr(output, "continuation_read_counts", None)
        consumed_counts = getattr(result, "continuation_read_counts", None)
        assert raw_counts is None or isinstance(raw_counts, ContinuationReadCounts)
        metadata = (
            tuple((leaf.shape, str(leaf.dtype)) for leaf in raw_counts)
            if raw_counts is not None
            else ()
        )
        feasibility = None
        if isinstance(output, FeasibilityKernelOutput):
            empty, other = output.feasibility_counts
            feasibility = int(np.asarray(empty)), int(np.asarray(other))
        observed.outputs.append(
            _Output(
                observed.run,
                regime,
                period,
                _numeric_snapshot(output),
                tuple(
                    np.asarray(tuple(status)).copy() for status in result.bound_status
                ),
                tuple(
                    np.asarray(tuple(status)).copy()
                    for status in result.query_status or ()
                ),
                feasibility,
                raw_counts is not None,
                metadata,
                consumed_counts is raw_counts,
            )
        )
        return result

    return consume


def _make_read_observer(
    *, observed: _Observations, original_read: _DcReadCall
) -> _DcReadCall:
    """Keep the exact real row call and its deferred numeric observation together."""

    def read(
        **kwargs: Unpack[_DcReadInputs],
    ) -> tuple[Float1D, Float1D, QueryInputStatus, _FloorState]:
        result = original_read(**kwargs)
        initial = kwargs.get("active") is None
        active = kwargs.get("active")
        if active is None:
            active = jnp.ones(kwargs["liquid"].shape, dtype=jnp.bool_)
        pool = kwargs["pool"]
        assert pool is not None
        period, age = pool["period"], pool["age"]
        assert isinstance(period, jax.Array | float)
        assert isinstance(age, jax.Array | float)

        # keyword-only-exempt: library-callback=jax.debug.callback
        def record(
            queries: HostArray,
            active: HostArray,
            clock: HostArray,
            floor: HostArray,
            nodes: HostArray,
            status: HostArray,
        ) -> None:
            observed.reads.append(
                _Read(
                    observed.run,
                    initial,
                    queries.copy(),
                    active.copy(),
                    clock.copy(),
                    floor.copy(),
                    nodes.copy(),
                    status.copy(),
                )
            )

        jax.debug.callback(
            record,
            kwargs["liquid"],
            active,
            jnp.stack((period, age)),
            result[3][0],
            result[3][1],
            jnp.stack(result[3][2]),
        )
        return result

    return read


@contextmanager
def _observe_engine(*, monkeypatch: pytest.MonkeyPatch) -> Generator[_Observations]:
    """Observe each real call once and restore all hooks at the context boundary."""
    observed = _Observations()
    original_select = induction.select_programs
    original_consume = induction.consume_kernel_output
    original_read = egm_module._read_one_row_dc_child
    original_counted = DCEGMQuery._read_rows_with_counts
    original_barrier = induction._materialize_period_feasibility_counts
    original_get = jax.device_get
    original_step = _EGMStep.__call__
    step = _make_step_observer(observed=observed, original_step=original_step)

    def select(**kwargs: Unpack[_SelectInputs]) -> Mapping[str, CoreProgram]:
        selected = original_select(**kwargs)
        for name, declared in kwargs["graph"].items():
            builder = declared.argument_builder
            if (
                isinstance(builder, _EGMArgumentBuilder)
                and builder.regime_name == "parent"
            ):
                observed.selections.append(
                    _Selection(
                        observed.run,
                        kwargs.get("collect_continuation_read_counts", False),
                        declared,
                        selected[name],
                    )
                )
        return selected

    consume = _make_consume_observer(
        observed=observed, original_consume=original_consume
    )
    read = _make_read_observer(observed=observed, original_read=original_read)

    def counted(
        self: DCEGMQuery, **kwargs: Unpack[_CountedRowInputs]
    ) -> tuple[_BlendedRow, ContinuationReadCounts]:
        observed.traces.append(observed.run)
        return original_counted(self, **kwargs)

    def barrier(
        *, diagnostic_rows: list[_DiagnosticRow], period: int, enabled: bool
    ) -> None:
        def device_get(value: PytreeValue) -> PytreeValue:
            # The barrier moves a period's rows in one transfer; each set of
            # counts inside it is one materialization.
            observed.materializations.extend(
                (observed.run, period)
                for node in jax.tree.leaves(
                    value, is_leaf=lambda node: isinstance(node, ContinuationReadCounts)
                )
                if isinstance(node, ContinuationReadCounts)
            )
            return original_get(value)

        with monkeypatch.context() as transfer_patch:
            transfer_patch.setattr(jax, "device_get", device_get)
            original_barrier(
                diagnostic_rows=diagnostic_rows, period=period, enabled=enabled
            )
        for row in diagnostic_rows:
            if (row.regime_name, row.period) == ("parent", period) and period == 0:
                observed.host.append(
                    _HostRecord(
                        observed.run,
                        _host_pair(getattr(row, "continuation_read_counts", None)),
                    )
                )

    with monkeypatch.context() as patch:
        patch.setattr(induction, "select_programs", select)
        patch.setattr(induction, "consume_kernel_output", consume)
        patch.setattr(egm_module, "_read_one_row_dc_child", read)
        patch.setattr(DCEGMQuery, "_read_rows_with_counts", counted)
        patch.setattr(induction, "_materialize_period_feasibility_counts", barrier)
        patch.setattr(_EGMStep, "__call__", step)
        try:
            yield observed
        finally:
            jax.effects_barrier()
    assert induction.select_programs is original_select
    assert induction.consume_kernel_output is original_consume
    assert egm_module._read_one_row_dc_child is original_read
    assert DCEGMQuery._read_rows_with_counts is original_counted
    assert induction._materialize_period_feasibility_counts is original_barrier
    assert jax.device_get is original_get
    assert _EGMStep.__call__ is original_step


def _qualify_ordinary_events(*, observed: _Observations, nodes: HostArray) -> None:
    """Four Euler candidates and two genuine F2 candidates imply one below of six."""
    first = [value for run, value in observed.first if run == "unchecked"]
    assert len(first) == 1
    np.testing.assert_array_equal(np.asarray(first[0], dtype=nodes.dtype), nodes[1])
    initial = [
        read for read in observed.reads if read.run == "unchecked" and read.initial
    ]
    rescore = [
        read for read in observed.reads if read.run == "unchecked" and not read.initial
    ]
    assert len(initial) == len(rescore) == 1
    bits = np.uint32 if nodes.dtype.itemsize == 4 else np.uint64
    np.testing.assert_array_equal(initial[0].queries.view(bits), nodes.view(bits))
    np.testing.assert_array_equal(initial[0].active, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(rescore[0].active, np.ones(2, dtype=bool))
    assert rescore[0].queries.shape == (2,)
    assert bool(np.all(np.greater(rescore[0].queries, 0.125)))
    assert bool(np.all(np.less(rescore[0].queries, 0.75)))
    for read in (*initial, *rescore):
        np.testing.assert_array_equal(read.clock, [1, 1])
        np.testing.assert_array_equal(read.floor, 0.0)
        np.testing.assert_array_equal(read.nodes, [0.0, 0.5, 2.0])
        np.testing.assert_array_equal(read.status, [False, False])
    assert bool(nodes[0] < nodes[1])
    assert bool(np.all(np.greater_equal(nodes[1:], nodes[1])))
    assert bool(nodes[1] < 0.125)


def _assert_ordinary_parity(*, observed: _Observations) -> None:
    """The validation policy leaves ordinary values, artifacts and statuses equal."""
    unchecked = [item for item in observed.outputs if item.run == "unchecked"]
    checked = [item for item in observed.outputs if item.run == "checked"]
    assert [(item.regime, item.period) for item in unchecked] == [
        (item.regime, item.period) for item in checked
    ]
    assert unchecked
    for left, right in zip(unchecked, checked, strict=True):
        assert len(left.numeric) == len(right.numeric)
        for actual, expected in zip(right.numeric, left.numeric, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert len(left.bound) == len(right.bound)
        assert len(left.query) == len(right.query)
        for actual, expected in zip(
            right.bound + right.query, left.bound + left.query, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual, np.zeros(actual.shape, dtype=bool))
        assert left.feasibility == right.feasibility


def _healthy_solves(
    *, model: Model, params: UserParams, observed: _Observations
) -> None:
    observed.run = "unchecked"
    model.solve(params=params, log_level="warning", runtime_checks=False)
    jax.effects_barrier()
    observed.run = "checked"
    model.solve(params=params, log_level="warning", runtime_checks=True)
    jax.effects_barrier()


def test_runtime_checks_selects_actual_counted_parent_before_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual solve selects checked work without changing numerical contracts."""
    model, params, nodes = _engine_model()
    with _observe_engine(monkeypatch=monkeypatch) as observed:
        _healthy_solves(model=model, params=params, observed=observed)
        _qualify_ordinary_events(observed=observed, nodes=nodes)
        _assert_ordinary_parity(observed=observed)
        bad = {
            **params,
            "child": {
                "koopmans_aggregator": {"discount_factor": 1.0},
                "borrowing_lower": {"floor": np.nan},
            },
        }
        errors: list[str] = []
        negative_runs: tuple[tuple[_Run, bool], ...] = (
            ("bad-unchecked", False),
            ("bad-checked", True),
        )
        for run, enabled in negative_runs:
            observed.run = run
            with pytest.raises(
                InvalidValueFunctionError,
                match=r"borrowing.*nonfinite borrowing floor.*child.*period 1",
            ) as failure:
                model.solve(params=bad, log_level="warning", runtime_checks=enabled)
            errors.append(str(failure.value))
            jax.effects_barrier()
        assert errors[0] == errors[1]
        ordinary = [item for item in observed.selections if item.run == "unchecked"]
        checked = [item for item in observed.selections if item.run == "checked"]
        assert len(ordinary) == len(checked) == 1
        assert ordinary[0].enabled is False
        assert ordinary[0].selected is ordinary[0].declared
        assert "unchecked" not in observed.traces
        assert checked[0].enabled is True, (
            "runtime_checks=True must select the counted parent before materialization."
        )
        variant = checked[0].declared.checked_variant
        assert variant is not None, "The real parent must declare its counted variant."
        selected = checked[0].selected
        assert selected.function is variant.function
        assert selected.output_roles is variant.output_roles
        assert selected.internal_outputs == variant.internal_outputs
        assert selected.argument_builder is checked[0].declared.argument_builder
        assert selected.requirements is checked[0].declared.requirements
        assert selected.checked_variant is None
        role_counts = getattr(selected.output_roles, "continuation_read_counts", None)
        assert isinstance(role_counts, ContinuationReadCounts)
        roles = jax.tree.leaves(role_counts)
        assert len(roles) == 2
        for role in roles:
            assert isinstance(role, StateAxesLeading)
            assert role.shape == ()
            assert role.dtype == jnp.dtype(jnp.int32)
        assert "checked" in observed.traces
        assert all(
            not item.counts_present
            for item in observed.outputs
            if item.run == "unchecked"
        )


def test_actual_parent_counts_reach_one_period_host_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One actual parent period materializes its independent paired-read inventory."""
    model, params, nodes = _engine_model()
    with _observe_engine(monkeypatch=monkeypatch) as observed:
        _healthy_solves(model=model, params=params, observed=observed)
        _qualify_ordinary_events(observed=observed, nodes=nodes)
        _assert_ordinary_parity(observed=observed)
        parent = [
            item
            for item in observed.outputs
            if item.run == "checked" and (item.regime, item.period) == ("parent", 0)
        ]
        assert len(parent) == 1
        assert parent[0].counts_present, (
            "The actual parent kernel must publish its paired continuation counts."
        )
        assert parent[0].counts_metadata == (((), "int32"), ((), "int32"))
        assert parent[0].counts_forwarded
        assert parent[0].feasibility == (0, 0)
        host = [item for item in observed.host if item.run == "checked"]
        assert len(host) == 1
        assert host[0].counts == (1, 6)
        assert observed.materializations.count(("checked", 0)) == 1
        assert not any(run == "unchecked" for run, _period in observed.materializations)
        assert all(
            item.counts is None for item in observed.host if item.run == "unchecked"
        )
