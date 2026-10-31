"""Landing counts never wrap: a positive population stays positive at any size.

Every reduction of raw off-grid landing counts saturates at the int32 capacity.
A total is exact below the capacity; at the capacity it means "at least that
many", which the diagnostic says explicitly and which still refuses a
`raise` grid. The expected values come from Python integers.
"""

import logging
import os
from functools import partial
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import transition_checks
from _lcm.off_grid import report_off_grid
from _lcm.regime_building.Q_and_F import _live_landing_counts
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    CountedGridSearch,
)
from _lcm.solution.landing_counts import (
    LandingCountedGridSearch,
    _feasible_landing_counts,
    fail_on_reached_landings,
)
from _lcm.typing import EconFunctionKwargs
from _lcm.utils.counts import COUNT_CAPACITY, sum_counts
from lcm import AgeGrid, ExecutionConfig, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import ExecutionPlanningError, OffGridStateError
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    IntND,
    ScalarInt,
    StateOrActionName,
)

_INT32_MAX = (1 << 31) - 1

_TOTALS = (
    0,
    1,
    _INT32_MAX - 1,
    _INT32_MAX,
    1 << 31,
    (1 << 31) + 1,
    (1 << 32) - 1,
    1 << 32,
    (1 << 32) + 1,
    1 << 33,
)


def _expected(total: int) -> int:
    return min(total, _INT32_MAX)


def _parts(total: int) -> tuple[int, ...]:
    """Split `total` into int32-representable partial counts."""
    quotient, remainder = divmod(total, _INT32_MAX)
    return (_INT32_MAX,) * quotient + ((remainder,) if remainder else ()) or (0,)


def _count_point(*, row: jax.Array, partial_counts: jax.Array) -> jax.Array:
    count = partial_counts[row.astype(jnp.int32)]
    return jnp.stack((jnp.stack((count, jnp.int32(0))),))


def _preflight_total(*, parts: tuple[int, ...], compiled: bool) -> int:
    def run(
        *,
        grid_args: dict[StateOrActionName, jax.Array],
        scalar_kwargs: EconFunctionKwargs,
    ) -> jax.Array:
        return cast(
            "jax.Array",
            transition_checks._off_grid_counts_law(
                grid_args=grid_args,
                scalar_kwargs=scalar_kwargs,
                grid_names=("row",),
                func=_count_point,
            ),
        )

    counts = (jax.jit(run) if compiled else run)(
        grid_args={"row": jnp.arange(len(parts), dtype=jnp.int32)},
        scalar_kwargs=MappingProxyType(
            {"partial_counts": jnp.asarray(parts, dtype=jnp.int32)}
        ),
    )
    return int(counts[0, 0])


def test_count_capacity_is_int32_max() -> None:
    assert int(COUNT_CAPACITY) == _INT32_MAX


@pytest.mark.parametrize("block_rows", [1, 2])
@pytest.mark.parametrize("disabled", [False, True], ids=["normal", "disable-jit"])
def test_bounded_outer_counts_accept_eager_integer_indices(
    *, block_rows: int, disabled: bool
) -> None:
    """An outer count loop preserves counts and synthetic transport status."""

    def point(*, row: ScalarInt) -> BoundedResult[IntND]:
        return BoundedResult(
            output=jnp.stack(
                (jnp.stack((row + jnp.int32(1), (row == 1).astype(jnp.int32))),)
            ),
            bound_status=(
                BoundInputStatus(
                    nonfinite_floor=row == 2,
                    invalid_shifted_grid=jnp.asarray(a=False),
                ),
            ),
        )

    with jax.disable_jit(disabled):
        result = transition_checks._off_grid_counts_law(
            grid_args={"row": jnp.arange(3, dtype=jnp.int32)},
            scalar_kwargs=MappingProxyType({}),
            grid_names=("row",),
            func=point,
            block_rows=block_rows,
        )

    assert isinstance(result, BoundedResult)
    np.testing.assert_array_equal(result.output, [[6, 1]])
    assert len(result.bound_status) == 1
    status = result.bound_status[0]
    assert status.nonfinite_floor.shape == ()
    assert status.invalid_shifted_grid.shape == ()
    np.testing.assert_array_equal(status.nonfinite_floor, desired=True)
    np.testing.assert_array_equal(status.invalid_shifted_grid, desired=False)


@pytest.mark.parametrize("total", _TOTALS)
@pytest.mark.parametrize("block_rows", [1, 2, 1 << 14])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_preflight_total_is_exact_or_saturated_across_partition_and_order(
    *,
    monkeypatch: pytest.MonkeyPatch,
    total: int,
    block_rows: int,
    reverse: bool,
    compiled: bool,
) -> None:
    monkeypatch.setattr(transition_checks, "_OFF_GRID_BLOCK_ROWS", block_rows)
    parts = _parts(total)[::-1] if reverse else _parts(total)
    assert sum(parts) == total
    assert _preflight_total(parts=parts, compiled=compiled) == _expected(total)


@pytest.mark.parametrize("total", _TOTALS)
@pytest.mark.parametrize("compiled", [False, True])
def test_sum_counts_is_exact_or_saturated(*, total: int, compiled: bool) -> None:
    parts = jnp.asarray(_parts(total), dtype=jnp.int32)
    func = jax.jit(sum_counts) if compiled else sum_counts
    assert int(func(counts=parts)) == _expected(total)


def test_sum_counts_counts_booleans() -> None:
    assert int(sum_counts(counts=jnp.array([True, False, True]))) == 2


def test_sum_counts_reduces_only_the_named_axes() -> None:
    counts = jnp.full((4, 3), 1 << 30, dtype=jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(sum_counts(counts=counts, axis=(0,))), np.full(3, _INT32_MAX)
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_elements", [3, 1 << 12, 1 << 17])
@pytest.mark.parametrize("compiled", [False, True])
def test_sum_counts_agrees_with_python_integers(
    *,
    seed: int,
    n_elements: int,
    compiled: bool,
) -> None:
    rng = np.random.default_rng(seed=seed)
    largest = rng.choice([1, 1 << 10, 1 << 20, _INT32_MAX])
    values = rng.integers(0, largest, size=n_elements, endpoint=True, dtype=np.int64)
    func = jax.jit(sum_counts) if compiled else sum_counts
    result = int(func(counts=jnp.asarray(values.astype(np.int32))))
    assert result == _expected(sum(int(value) for value in values))


def test_sum_counts_with_a_unit_bound_counts_zero_one_integers_exactly() -> None:
    counts = jnp.asarray(
        np.random.default_rng(seed=0).integers(0, 2, size=1 << 17), dtype=jnp.int32
    )
    assert int(sum_counts(counts=counts, largest=1)) == int(np.sum(counts))


def test_sum_counts_keeps_exact_totals_on_the_digit_path() -> None:
    counts = jnp.full((1 << 16,), 1 << 14, dtype=jnp.int32)
    assert int(sum_counts(counts=counts)) == 1 << 30


def test_sum_counts_saturates_on_the_digit_path() -> None:
    counts = jnp.full((1 << 17,), 1 << 20, dtype=jnp.int32)
    assert int(sum_counts(counts=counts)) == _INT32_MAX


def test_sum_counts_refuses_a_domain_int32_cannot_address() -> None:
    with pytest.raises(ExecutionPlanningError, match="int32 count"):
        jax.eval_shape(
            lambda counts: sum_counts(counts=counts),
            jax.ShapeDtypeStruct((1 << 31,), jnp.bool_),
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_action_reduction_saturates_instead_of_wrapping(compiled: bool) -> None:  # noqa: FBT001
    def run() -> jax.Array:
        return _feasible_landing_counts(
            F_arr=jnp.ones(4, dtype=bool),
            landing_counts=jnp.full((4, 1), 1 << 30, dtype=jnp.int32),
        )

    assert int((jax.jit(run) if compiled else run)()[0]) == _INT32_MAX


@pytest.mark.parametrize("compiled", [False, True])
def test_state_reduction_saturates_instead_of_wrapping(compiled: bool) -> None:  # noqa: FBT001
    def outputs(**kwargs: jax.Array) -> tuple[jax.Array, ...]:  # noqa: ARG001
        return (
            jnp.ones(4),
            jnp.zeros(4, dtype=bool),
            jnp.full((4, 1), 1 << 30, dtype=jnp.int32),
        )

    counted = LandingCountedGridSearch(
        counted=CountedGridSearch(function=outputs, collective=False)  # ty: ignore[invalid-argument-type]
    )

    def run() -> jax.Array:
        return cast("tuple[jax.Array, ...]", counted())[-1]

    assert int((jax.jit(run) if compiled else run)()[0]) == _INT32_MAX


def test_live_node_reduction_saturates_instead_of_wrapping() -> None:
    totals = _live_landing_counts(
        node_counts=jnp.full((4, 1), 1 << 30, dtype=jnp.int32),
        target_probability=jnp.asarray(1.0),
        node_weights=(jnp.full(4, 0.25),),
    )
    assert int(totals[0]) == _INT32_MAX


def test_saturated_total_still_refuses_a_raise_grid() -> None:
    totals = _feasible_landing_counts(
        F_arr=jnp.ones(4, dtype=bool),
        landing_counts=jnp.full((4, 1), 1 << 30, dtype=jnp.int32),
    )
    with pytest.raises(OffGridStateError):
        fail_on_reached_landings(
            totals=totals,
            checks=(("target", "wealth"),),
            regime_name="source",
            period=0,
        )


def test_saturated_count_is_reported_as_a_lower_bound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("lcm")
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        report_off_grid(
            count=_INT32_MAX,
            rule="clip_state",
            regime="source",
            target="target",
            period=0,
            state="wealth",
            phase="solve",
            logger=logger,
        )
    assert f"count>={_INT32_MAX}" in caplog.text


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt


def _utility(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return 0.0 * wealth + 0.0 * consumption


def _landing(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return 3.0 + 0.0 * wealth + 0.0 * consumption


def _terminal(wealth: ContinuousState) -> FloatND:
    return wealth


def _exterior_model(*, n_states: int, n_actions: int) -> Model:
    """Every feasible action lands at 3, outside the target's `raise` grid [0, 2]."""
    return Model(
        regimes={
            "source": Regime(
                states={
                    "wealth": LinSpacedGrid(
                        start=0, stop=n_states - 1, n_points=n_states
                    )
                },
                actions={
                    "consumption": LinSpacedGrid(
                        start=0, stop=n_actions - 1, n_points=n_actions
                    )
                },
                functions={"utility": _utility},
                state_transitions={"wealth": _landing},
            ),
            "target": Regime(
                states={
                    "wealth": LinSpacedGrid(
                        start=0, stop=2, n_points=2, off_grid="raise"
                    )
                },
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"source": {"target": 0}},
        initial_nodes={0: "source"},
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def test_validate_refuses_an_all_exterior_population() -> None:
    model = _exterior_model(n_states=4, n_actions=4)
    with pytest.raises(OffGridStateError):
        model.validate_transitions(params={"discount_factor": 1.0})


@pytest.mark.skipif(
    os.environ.get("LCM_RUN_LARGE_COUNT") != "1",
    reason="2**32 landings take a dedicated resource job",
)
def test_validate_refuses_two_to_the_32_exterior_landings() -> None:
    model = _exterior_model(n_states=1 << 18, n_actions=1 << 14)
    with pytest.raises(OffGridStateError):
        model.validate_transitions(params={"discount_factor": 1.0})


def test_omitted_axis_multiplicity_preserves_saturated_counts_and_status() -> None:
    """Compiled repeats preserve dynamic counts and independent true status flags."""
    counts = jnp.array([[0, 1], [COUNT_CAPACITY // 2, COUNT_CAPACITY]], dtype=jnp.int32)

    def point(*, counts: IntND) -> transition_checks._CurrentIIDOffGridCounts:
        return transition_checks._CurrentIIDOffGridCounts(
            counts=BoundedResult(
                output=counts,
                bound_status=(
                    BoundInputStatus(
                        nonfinite_floor=jnp.asarray(a=True),
                        invalid_shifted_grid=jnp.asarray(a=True),
                    ),
                ),
            ),
            invalid_law=jnp.asarray(a=True),
        )

    record = point(counts=counts)
    assert (
        transition_checks._repeat_off_grid_counts(counts=record, multiplicity=1)
        is record
    )
    with pytest.raises(ValueError, match="multiplicity must be positive"):
        transition_checks._repeat_off_grid_counts(counts=record, multiplicity=0)
    for supplied, multiplicity in (
        (counts, 1),
        (counts, 2),
        (counts, COUNT_CAPACITY + 1),
        (jnp.zeros_like(counts), 2),
    ):
        run = jax.jit(
            partial(
                transition_checks._off_grid_counts_law,
                grid_names=(),
                func=point,
                multiplicity=multiplicity,
            )
        )
        repeated = run(
            grid_args={}, scalar_kwargs=MappingProxyType({"counts": supplied})
        )
        assert isinstance(repeated, transition_checks._CurrentIIDOffGridCounts)
        assert isinstance(repeated.counts, BoundedResult)
        expected = [
            [min(int(value) * multiplicity, COUNT_CAPACITY) for value in row]
            for row in np.asarray(supplied)
        ]
        np.testing.assert_array_equal(repeated.counts.output, expected)
        assert bool(repeated.invalid_law)
        assert len(repeated.counts.bound_status) == 1
        status = repeated.counts.bound_status[0]
        assert status.nonfinite_floor.shape == ()
        assert status.invalid_shifted_grid.shape == ()
        assert bool(status.nonfinite_floor)
        assert bool(status.invalid_shifted_grid)
