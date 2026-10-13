"""Landing counts never wrap: a positive population stays positive at any size.

Every reduction of raw off-grid landing counts saturates at the int32 capacity.
A total is exact below the capacity; at the capacity it means "at least that
many", which the diagnostic says explicitly and which still refuses a
`raise` grid. The expected values come from Python integers.
"""

import logging
import os
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import transition_checks
from _lcm.off_grid import report_off_grid
from _lcm.regime_building.Q_and_F import _live_landing_counts
from _lcm.solution.feasibility import CountedGridSearch
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
        return transition_checks._off_grid_counts_law(
            grid_args=grid_args,
            scalar_kwargs=scalar_kwargs,
            grid_names=("row",),
            func=_count_point,
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
        return counted()[-1]

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
