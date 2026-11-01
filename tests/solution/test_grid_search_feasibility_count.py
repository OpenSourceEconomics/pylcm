"""Keep feasibility populations exact within the configured integer precision."""

from collections.abc import Callable
from contextlib import nullcontext
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.execution.output_layout import StateAxesLeading
from _lcm.policy_queries import _get_padded_query_bank_size, _PolicyQueryKernel
from _lcm.solution.backward_induction import _period_feasibility_counter_reservations
from _lcm.solution.diagnostics import (
    _DiagnosticRow,
    _materialize_period_feasibility_counts,
)
from _lcm.solution.feasibility import (
    CountedGridSearch,
    FeasibilityKernelOutput,
    _fail_if_feasibility_count_exceeds_capacity,
    add_feasibility_counts,
)
from _lcm.solution.grid_search import _GridSearchPeriodKernel
from _lcm.solution.kernel_output import consume_kernel_output
from _lcm.solution.landing_counts import LandingCountedGridSearch
from lcm import (
    AgeGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    LinSpacedGrid,
    Model,
    Regime,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import BoolND, FloatND
from tests.simulation.test_no_feasible_action import (
    DecisionId,
    _two_state_utility,
    _zero,
)
from tests.test_model import _output_model, _output_utility

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _evaluated(*, value: FloatND, empty: BoolND) -> tuple[FloatND, BoolND]:
    """Supply already reduced state values and their empty-action mask."""
    return value, empty


def test_feasibility_blocks_preserve_large_exact_totals() -> None:
    """Representable partial counts add to the full population without saturation."""
    part = 1 << 30 if jax.config.x64_enabled else 3
    expected = 1 << 31 if jax.config.x64_enabled else 6
    counts = (jnp.asarray(part, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))

    result = add_feasibility_counts(total=counts, block=counts)

    assert tuple(int(count) for count in result) == (expected, 0)


def test_large_count_reduction_is_exact_or_refused_before_allocation() -> None:
    """A metadata-only large population requires int64 or a count-naming refusal."""
    counted = CountedGridSearch(function=_evaluated, collective=False)
    value = jax.ShapeDtypeStruct((1 << 31,), jnp.result_type(0.0))
    empty = jax.ShapeDtypeStruct((1 << 31,), jnp.bool_)

    if jax.config.x64_enabled:
        result = jax.eval_shape(counted, value=value, empty=empty)
        assert tuple(count.dtype for count in result[1:]) == (jnp.int64, jnp.int64)
    else:
        with pytest.raises(ExecutionPlanningError, match="feasibility count"):
            jax.eval_shape(counted, value=value, empty=empty)


@pytest.mark.parametrize("parts", [(0, 0), (7, 11)])
def test_small_feasibility_blocks_keep_exact_counts(parts: tuple[int, int]) -> None:
    """Small block summaries retain their exact empty and other-infinite totals."""
    left, right = parts
    result = add_feasibility_counts(
        total=(jnp.asarray(left, dtype=jnp.int32), jnp.asarray(right, dtype=jnp.int32)),
        block=(jnp.asarray(2, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32)),
    )
    assert tuple(int(count) for count in result) == (left + 2, right + 3)


@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_feasibility_reduction_counts_states_once(
    *, collective: bool, compiled: bool
) -> None:
    """Empty and economic-infinity totals count states, including whole households."""
    value = jnp.asarray([-jnp.inf, jnp.inf, 2.0, jnp.nan])
    if collective:
        value = jnp.stack((value, jnp.ones_like(value)), axis=-1)
    empty = jnp.asarray([True, False, False, False])
    function = CountedGridSearch(function=_evaluated, collective=collective)
    run = jax.jit(function) if compiled else function

    result = run(value=value, empty=empty)
    assert isinstance(result, tuple)
    assert len(result) == 3
    published, empty_count, other_count = result

    np.testing.assert_array_equal(published, value)
    assert (int(empty_count), int(other_count)) == (1, 1)
    expected_dtype = np.dtype(np.int64 if jax.config.x64_enabled else np.int32)
    assert empty_count.shape == other_count.shape == ()
    assert empty_count.dtype == other_count.dtype == expected_dtype


@pytest.mark.parametrize("n_states", [0, 3])
@pytest.mark.parametrize("collective", [False, True])
def test_feasible_finite_and_empty_domains_have_zero_counts(
    *, n_states: int, collective: bool
) -> None:
    """Finite feasible states and empty domains publish exact zero summaries."""
    shape = (n_states, 2) if collective else (n_states,)
    value = jnp.ones(shape)
    empty = jnp.zeros((n_states,), dtype=jnp.bool_)
    result = jax.jit(CountedGridSearch(function=_evaluated, collective=collective))(
        value=value, empty=empty
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    np.testing.assert_array_equal(result[0], value)
    assert tuple(int(count) for count in result[1:]) == (0, 0)


def test_count_capacity_bound_names_the_unsupported_population() -> None:
    """The largest supported population fits; the next count is refused statically."""
    capacity = (1 << (63 if jax.config.x64_enabled else 31)) - 1
    _fail_if_feasibility_count_exceeds_capacity(n_states=capacity)
    with pytest.raises(ExecutionPlanningError, match="feasibility count"):
        _fail_if_feasibility_count_exceeds_capacity(n_states=capacity + 1)


def test_legacy_solver_counts_keep_their_nonzero_population() -> None:
    """Large legacy partials stay positive at fp32 and add exactly at fp64."""
    part = jnp.asarray(1 << 30, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    result = add_feasibility_counts(total=(part, zero), block=(part, zero))
    expected = 1 << 31 if jax.config.x64_enabled else (1 << 31) - 1
    assert (int(result[0]), int(result[1])) == (expected, 0)
    assert bool(result[0] > 0)


@pytest.mark.parametrize("dtype", ["int32", "configured"])
def test_native_carrier_forwards_exact_counts_to_host_diagnostics(dtype: str) -> None:
    """Declared signed counters retain their values through the period barrier."""
    count_dtype = np.dtype(
        np.int64 if dtype == "configured" and jax.config.x64_enabled else np.int32
    )
    expected = 1 << 31 if count_dtype == np.dtype(np.int64) else 7
    counts = (
        jnp.asarray(expected, dtype=count_dtype),
        jnp.asarray(0, dtype=count_dtype),
    )
    output = FeasibilityKernelOutput(
        value=jnp.asarray([1.0]), feasibility_counts=counts
    )
    consumed = consume_kernel_output(
        output=output, continuation_key=None, regime_name="alive", period=0
    )
    assert consumed.feasibility_counts is counts
    rows = [
        _DiagnosticRow(
            regime_name="alive", period=0, age=0.0, feasibility_counts=counts
        )
    ]

    _materialize_period_feasibility_counts(diagnostic_rows=rows, period=0, enabled=True)

    host_counts = rows[0].feasibility_counts
    assert isinstance(host_counts, tuple)
    assert host_counts == (expected, 0)
    assert all(type(count) is int for count in host_counts)
    assert not output.continuations
    assert not output.replay
    assert not output.auxiliary


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [("int16", ()), ("uint32", ()), ("float32", ()), ("int32", (1,))],
)
def test_native_feasibility_carrier_rejects_wrong_count_type_or_shape(
    *, dtype: str, shape: tuple[int, ...]
) -> None:
    """Counters require signed int32/int64 scalars, independent of value channels."""
    invalid = jnp.zeros(shape, dtype=dtype)
    with pytest.raises((TypeError, BeartypeCallHintParamViolation), match="counts"):
        FeasibilityKernelOutput(
            value=jnp.asarray([1.0]), feasibility_counts=(invalid, invalid)
        )


def test_gridsearch_roles_and_reservations_match_counter_precision() -> None:
    """Actual program roles and per-device owner charges cover the native pair."""
    model = _output_model(functions={"utility": _output_utility})
    kernel = model._regimes["alive"].solution.period_kernels[0]
    assert isinstance(kernel, _GridSearchPeriodKernel)
    program = kernel.core_programs()["main"]
    roles = program.output_roles
    assert isinstance(roles, tuple)
    count_dtype = np.dtype(np.int64 if jax.config.x64_enabled else np.int32)
    for role in roles[1:3]:
        assert isinstance(role, StateAxesLeading)
        assert role.dtype == count_dtype
        assert role.shape == ()
        assert role.state_names == ()
    reservations = _period_feasibility_counter_reservations(
        active=frozenset({"alive", "other"}),
        device_ids_by_regime={"alive": frozenset({0}), "other": frozenset({0, 1})},
    )
    assert dict(reservations) == dict.fromkeys((0, 1), 2 * count_dtype.itemsize * 3)


@pytest.mark.parametrize("extent", [46_340, 46_341])
def test_public_model_counter_guard_uses_configured_precision(extent: int) -> None:
    """Real node vectors admit int64 populations or refuse fp32 before a product."""
    unsupported = extent == 46_341 and not jax.config.x64_enabled
    outcome = (
        pytest.raises(ExecutionPlanningError, match="feasibility count")
        if unsupported
        else nullcontext()
    )
    with outcome:
        model = _counter_domain_model(extent=extent)
        for grid in model.user_regimes["working"].states.values():
            assert isinstance(grid, LinSpacedGrid)
            assert grid.n_points == extent


def _counter_domain_model(*, extent: int) -> Model:
    """Declare real node vectors without allocating their Cartesian product."""
    return Model(
        regimes={
            "working": Regime(
                states={
                    "wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=extent),
                    "ability": LinSpacedGrid(start=0.0, stop=1.0, n_points=extent),
                },
                state_transitions={
                    "wealth": fixed_transition("wealth"),
                    "ability": fixed_transition("ability"),
                },
                functions={"utility": _two_state_utility},
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "working"},
        edges={"working": {"dead": (0,)}},
        regime_id_class=DecisionId,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_landing_wrapper_keeps_exact_feasibility_and_int32_saturation(
    *, compiled: bool
) -> None:
    """Feasibility totals use their precision while landing totals retain saturation."""

    def outputs() -> tuple[jax.Array, ...]:
        return (
            jnp.ones(4),
            jnp.zeros(4, dtype=bool),
            jnp.full((4, 1), 1 << 30, dtype=jnp.int32),
        )

    counted = LandingCountedGridSearch(
        counted=CountedGridSearch(
            function=cast("Callable[..., tuple[FloatND, BoolND]]", outputs),
            collective=False,
        )
    )
    result = (jax.jit(counted) if compiled else counted)()
    assert isinstance(result, tuple)
    assert len(result) == 4
    value, empty, other, landing = result
    np.testing.assert_array_equal(value, np.ones(4))
    assert (int(empty), int(other)) == (0, 0)
    expected_dtype = np.dtype(np.int64 if jax.config.x64_enabled else np.int32)
    assert empty.dtype == other.dtype == expected_dtype
    np.testing.assert_array_equal(landing, np.asarray([(1 << 31) - 1], dtype=np.int32))
    assert landing.dtype == jnp.int32


@pytest.mark.parametrize(
    ("wealth_extent", "ability_extent", "chunk_size", "unsupported"),
    [
        (2, 2, 64, False),
        (46_340, 46_340, 64, False),
        (46_341, 46_341, 64, True),
        (46_340, 46_341, 100_000, True),
    ],
)
def test_public_query_sources_require_safe_int32_indices(
    *, wealth_extent: int, ability_extent: int, chunk_size: int, unsupported: bool
) -> None:
    """Query declarations refuse unaddressable logical or padded state products."""
    logical_overflow = wealth_extent * ability_extent > (1 << 31) - 1
    message = (
        "feasibility count"
        if logical_overflow and not jax.config.x64_enabled
        else "query.*int32"
    )
    outcome = (
        pytest.raises(ExecutionPlanningError, match=message)
        if unsupported
        else nullcontext()
    )
    with outcome:
        model = _query_counter_domain_model(
            wealth_extent=wealth_extent,
            ability_extent=ability_extent,
            chunk_size=chunk_size,
        )
        kernel = model._regimes["working"].solution.period_kernels[0]
        assert isinstance(kernel, _PolicyQueryKernel)
        assert kernel.chunk_size == min(chunk_size, wealth_extent * ability_extent)
        assert len(kernel.root_context.state_action_space.states) == 2


def _query_counter_domain_model(
    *, wealth_extent: int, ability_extent: int, chunk_size: int
) -> Model:
    """Declare a genuine unilateral query using only its two node vectors."""
    return Model(
        regimes={
            "working": Regime(
                states={
                    "wealth": LinSpacedGrid(
                        start=0.0, stop=1.0, n_points=wealth_extent
                    ),
                    "ability": LinSpacedGrid(
                        start=0.0, stop=1.0, n_points=ability_extent
                    ),
                },
                state_transitions={
                    "wealth": fixed_transition("wealth"),
                    "ability": fixed_transition("ability"),
                },
                functions={"utility": _two_state_utility},
                exits={
                    "leave": Exit(
                        targets={"self": ExitTarget(regime="dead", projection={})},
                        rule="unilateral",
                    )
                },
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "working"},
        edges={"working": {"dead": (0,)}},
        regime_id_class=DecisionId,
        execution_config=ExecutionConfig(
            axis_widths={"query_cell": chunk_size}, device_memory_bytes=None
        ),
    )


@pytest.mark.parametrize("chunk_size", [64, 100])
def test_public_query_padding_preserves_the_int32_last_position(
    chunk_size: int,
) -> None:
    """The final int32 position is admitted; later padded positions are refused."""
    outcome = (
        pytest.raises(ExecutionPlanningError, match=r"query source count.*int32")
        if chunk_size == 100
        else nullcontext()
    )
    with outcome:
        model = _query_counter_domain_model(
            wealth_extent=65_534, ability_extent=32_769, chunk_size=chunk_size
        )
        kernel = model._regimes["working"].solution.period_kernels[0]
        assert isinstance(kernel, _PolicyQueryKernel)
        assert kernel.chunk_size == 64
        states = kernel.root_context.state_action_space.states
        assert {name: value.size for name, value in states.items()} == {
            "wealth": 65_534,
            "ability": 32_769,
        }
        bank = kernel.core_programs()["_initialize"].requirements.host_buffers[
            "query_banks"
        ]
        assert isinstance(bank, jax.ShapeDtypeStruct)
        assert bank.shape == (2, 1 << 31)


@pytest.mark.parametrize(
    ("size", "chunk_size", "bank_size"),
    [
        ((1 << 31) - 1, 64, 1 << 31),
        ((1 << 31) - 64, 64, (1 << 31) - 64),
        (1 << 31, 1, None),
        ((1 << 31) - 1, 100, None),
    ],
)
def test_query_index_capacity_at_exact_integer_boundaries(
    *, size: int, chunk_size: int, bank_size: int | None
) -> None:
    """Logical count and padded positions independently obey signed-int32 bounds."""
    if bank_size is None:
        with pytest.raises(
            ExecutionPlanningError, match=rf"query source count {size}.*int32"
        ):
            _get_padded_query_bank_size(size=size, chunk_size=chunk_size)
    else:
        assert (
            _get_padded_query_bank_size(size=size, chunk_size=chunk_size) == bank_size
        )
