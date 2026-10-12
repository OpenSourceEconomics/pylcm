"""Grid policies distinguish value lookup from the economic state handoff."""

from collections.abc import Callable
from functools import partial
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm import transition_checks
from _lcm.grids.continuous import _clip_grid_value
from _lcm.off_grid import count_off_grid
from _lcm.params.edges import regime_kernel_params
from _lcm.regime_building.shock_average import average_over_shock
from _lcm.regime_building.V import (
    VInterpolationInfo,
    _get_coordinate_finder,
    get_V_interpolator,
)
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.transitions import _clip_next_states
from benchmarks.asv._compile_counters import count_compile_requests
from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    DeterministicTransition,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import (
    GridInitializationError,
    InvalidValueFunctionError,
    OffGridStateError,
)
from lcm.grids import (
    ContinuousGrid,
    GridBreakpoint,
    IrregSpacedGrid,
    LogSpacedGrid,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)
from lcm.regime import Regime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    OffGrid,
    Period,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    terminal: ScalarInt


def _utility(*, consumption: ContinuousAction, reward: float) -> FloatND:
    return reward * consumption


def _terminal(wealth: ContinuousState) -> FloatND:
    return wealth**2


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth + 1 - consumption


def _capped_next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> FloatND:
    return jnp.clip(wealth + 1 - consumption, 0, 2)


def _exclude_tail(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return (wealth < 2) | (consumption > 0)


def _model(
    *,
    off_grid: Literal["extrapolate", "clip_lookup", "clip_state", "raise"],
    hand_capped: bool = False,
    mask_tail: bool = False,
    runtime_grid: bool = False,
    device_memory_bytes: int | None = None,
) -> Model:
    grid = (
        IrregSpacedGrid(n_points=3, off_grid=off_grid)
        if runtime_grid
        else LinSpacedGrid(start=0, stop=2, n_points=3, off_grid=off_grid)
    )
    return Model(
        regimes={
            "alive": Regime(
                functions={"utility": _utility},
                states={"wealth": grid},
                actions={"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)},
                state_transitions={
                    "wealth": _capped_next_wealth if hand_capped else _next_wealth
                },
                constraints={"budget": _exclude_tail} if mask_tail else {},
            ),
            "terminal": Regime(
                functions={"utility": _terminal}, states={"wealth": grid}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"alive": {"terminal": 0}},
        initial_nodes={0: "alive"},
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=device_memory_bytes),
    )


@pytest.mark.parametrize("off_grid", ["clip_lookup", "clip_state"])
@pytest.mark.parametrize("runtime_grid", [False, True])
@pytest.mark.parametrize("device_memory_bytes", [None, 2**24])
def test_clipped_continuation_matches_hand_capped_value_and_choice(
    *,
    off_grid: Literal["clip_lookup", "clip_state"],
    runtime_grid: bool,
    device_memory_bytes: int | None,
) -> None:
    """At wealth2, capped continuation selects action1 and gives value6."""
    model = _model(
        off_grid=off_grid,
        runtime_grid=runtime_grid,
        device_memory_bytes=device_memory_bytes,
    )
    params = {
        "discount_factor": 1.0,
        "reward": 2.0,
        **(
            {
                name: {"wealth": {"points": jnp.array([0.0, 0.5, 2.0])}}
                for name in ("alive", "terminal")
            }
            if runtime_grid
            else {}
        ),
    }
    solution = model.solve(params=params, log_level="off")
    oracle = _model(
        off_grid="extrapolate",
        hand_capped=True,
        runtime_grid=runtime_grid,
        device_memory_bytes=device_memory_bytes,
    ).solve(params=params, log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["alive"],
        oracle.values[0]["alive"],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["alive"],
        [2, 2.75, 6] if runtime_grid else [2, 4, 6],
        decimal=DECIMAL_PRECISION,
    )
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.array([0.0]),
            "regime_id": jnp.array([RegimeId.alive]),
            "wealth": jnp.array([2.0]),
        },
        solution=solution,
        log_level="off",
    )
    frame = result.to_dataframe(use_labels=False)
    np.testing.assert_array_equal(frame.loc[frame["period"] == 0, "consumption"], [1])


@pytest.mark.parametrize("off_grid", ["clip_lookup", "clip_state"])
def test_interpolator_clips_both_sides_with_compiled_queries(
    off_grid: Literal["clip_lookup", "clip_state"],
) -> None:
    """Lookup maps exterior queries to the closed endpoints without changing NaN."""
    grid = LinSpacedGrid(start=0, stop=2, n_points=3, off_grid=off_grid)
    info = VInterpolationInfo(
        state_names=("wealth",),
        discrete_states=MappingProxyType({}),
        continuous_states=MappingProxyType({"wealth": grid}),
    )
    read = partial(
        get_V_interpolator(
            v_interpolation_info=info, state_prefix="next_", V_arr_name="values"
        ),
        values=jnp.array([0.0, 1.0, 4.0]),
    )
    queries = jnp.array([-1.0, 0.0, 1.5, 2.0, 3.0, jnp.nan])
    got = jax.jit(jax.vmap(lambda x: read(next_wealth=x)))(queries)
    np.testing.assert_array_equal(got, [0, 0, 2.5, 4, 4, np.nan])


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period == 0, RegimeId.alive, RegimeId.terminal)


@pytest.mark.parametrize(
    ("off_grid", "expected"),
    [("extrapolate", [2, 3, 4]), ("clip_lookup", [2, 3, 4]), ("clip_state", [2, 2, 2])],
)
@pytest.mark.parametrize("varying_bounds", [False, True])
@pytest.mark.parametrize("subject_batch_size", [0, 1])
def test_grid_policy_preserves_or_clips_the_actual_later_state(
    *,
    off_grid: OffGrid,
    expected: list[int],
    varying_bounds: bool,
    subject_batch_size: int,
) -> None:
    """A later transition consumes the raw state for lookup-only clipping."""

    def bounds(age: float) -> float:
        return 2.0 / 2**age

    def build(age: float) -> LinSpacedGrid:
        return LinSpacedGrid(start=0, stop=bounds(age), n_points=3, off_grid=off_grid)

    grid = (
        AgeSpecializedGrid(build=build, signature=bounds)
        if varying_bounds
        else build(0.0)
    )
    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _utility},
                states={"wealth": grid},
                actions={"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)},
                state_transitions={"wealth": _next_wealth},
            ),
            "terminal": Regime(
                functions={"utility": _terminal}, states={"wealth": grid}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={
            "alive": Transition(
                targets={"alive": 0, "terminal": (0, 1)},
                law=DeterministicTransition(func=_next_regime),
            )
        },
        initial_nodes={0: "alive"},
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(
            axis_widths={}
            if subject_batch_size == 0
            else {"subject": subject_batch_size},
            device_memory_bytes=None,
        ),
    )
    params = {"discount_factor": 1.0, "reward": 0.0}
    solution = model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(3),
            "regime_id": jnp.full(3, RegimeId.alive),
            "wealth": jnp.full(3, 2.0),
        },
        solution=solution,
        log_level="off",
    )
    frame = result.to_dataframe(use_labels=False).sort_values("period")
    np.testing.assert_array_equal(
        frame["wealth"],
        np.repeat(
            [2, 1, 0.5] if varying_bounds and off_grid == "clip_state" else expected,
            3,
        ),
    )


_GRID_FACTORIES: tuple[Callable[..., ContinuousGrid], ...] = (
    partial(LinSpacedGrid, start=1, stop=3, n_points=3),
    partial(LogSpacedGrid, start=1, stop=3, n_points=3),
    partial(IrregSpacedGrid, points=[1, 1.5, 3]),
    partial(
        PiecewiseLinSpacedGrid,
        start=1,
        stop=3,
        breakpoints=(GridBreakpoint(value=2),),
        points_per_segment=(2, 2),
    ),
    partial(
        PiecewiseLogSpacedGrid,
        start=1,
        stop=3,
        breakpoints=(GridBreakpoint(value=2),),
        points_per_segment=(2, 2),
    ),
)


@pytest.mark.parametrize("factory", _GRID_FACTORIES)
def test_every_continuous_grid_refuses_ambiguous_clip(
    factory: Callable[..., ContinuousGrid],
) -> None:
    """A caller must specify which clipping contract it needs."""
    with pytest.raises(GridInitializationError, match="off_grid"):
        factory(off_grid="clip")


@pytest.mark.parametrize("factory", _GRID_FACTORIES)
def test_all_grid_geometries_clip_physical_values_before_mapping(
    factory: Callable[..., ContinuousGrid],
) -> None:
    """Negative physical values reach a log grid's endpoint rather than its log."""
    grid = factory(off_grid="clip_lookup")
    info = VInterpolationInfo(
        state_names=("wealth",),
        discrete_states=MappingProxyType({}),
        continuous_states=MappingProxyType({"wealth": grid}),
    )
    read = partial(
        get_V_interpolator(
            v_interpolation_info=info, state_prefix="next_", V_arr_name="values"
        ),
        values=grid.to_jax() ** 2,
    )
    got = jax.jit(jax.vmap(lambda x: read(next_wealth=x)))(
        jnp.array([-2.0, 1.0, 3.0, 4.0])
    )
    np.testing.assert_array_almost_equal(got, [1, 1, 9, 9], decimal=DECIMAL_PRECISION)


def test_runtime_points_define_the_clipping_endpoints() -> None:
    """The supplied physical nodes, rather than declared size, set the cap."""
    info = VInterpolationInfo(
        state_names=("wealth",),
        discrete_states=MappingProxyType({}),
        continuous_states=MappingProxyType(
            {"wealth": IrregSpacedGrid(n_points=3, off_grid="clip_lookup")}
        ),
    )
    read = partial(
        get_V_interpolator(
            v_interpolation_info=info, state_prefix="next_", V_arr_name="values"
        ),
        values=jnp.array([1.0, 2.25, 9.0]),
        wealth__points=jnp.array([1.0, 1.5, 3.0]),
    )
    got = jax.jit(jax.vmap(lambda x: read(next_wealth=x)))(
        jnp.array([-2.0, 1.0, 3.0, 4.0])
    )
    np.testing.assert_array_equal(got, [1, 1, 9, 9])


def test_original_raw_off_grid_count_excludes_masked_cells_and_endpoints() -> None:
    """One reached raw tail counts; masked poison and exact endpoints do not."""
    values = jnp.array([0.0, 1.0, 2.0, 3.0, 100.0, jnp.nan])
    feasible = jnp.array([True, True, True, True, False, False])
    got = count_off_grid(
        values=values, points=jnp.array([0.0, 1.0, 2.0]), feasible=feasible
    )
    assert int(got) == 1


def test_state_clipping_preserves_other_target_and_unselected_carriers() -> None:
    """Only the subject's actual handoff is capped, not its unused regime slots."""
    result = jax.jit(_clip_next_states)(
        states=MappingProxyType(
            {
                "alive": MappingProxyType({"wealth": jnp.array([3.0, 4.0, 5.0])}),
                "terminal": MappingProxyType({"wealth": jnp.array([5.0, 6.0, 7.0])}),
            }
        ),
        points={
            "alive": {"wealth": jnp.array([0.0, 2.0])},
            "terminal": {"wealth": jnp.array([0.0, 4.0])},
        },
        subjects=jnp.array([True, True, False]),
        new_regime_ids=jnp.array([RegimeId.alive, RegimeId.terminal, RegimeId.alive]),
        regime_ids=MappingProxyType(
            {"alive": RegimeId.alive, "terminal": RegimeId.terminal}
        ),
    )
    np.testing.assert_array_equal(result["alive"]["wealth"], [2.0, 4.0, 5.0])
    np.testing.assert_array_equal(result["terminal"]["wealth"], [5.0, 4.0, 7.0])


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_explicit_raise_is_mandatory_at_every_runtime_and_output_setting(
    *, runtime_checks: bool, log_level: Literal["off", "warning", "progress", "debug"]
) -> None:
    """An explicit grid policy is enforced even when optional diagnostics are off."""
    model = _model(off_grid="raise")
    with pytest.raises(
        OffGridStateError, match=r"regime='alive'.*period=0.*state='wealth'"
    ):
        model.solve(
            params={"discount_factor": 1.0, "reward": 2.0},
            log_level=log_level,
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("off_grid", ["extrapolate", "clip_lookup", "clip_state"])
def test_checked_solve_reports_the_raw_count_before_clipping(
    *, off_grid: OffGrid, caplog: pytest.LogCaptureFixture
) -> None:
    """Both clipped modes report the same original exterior cell as extrapolation."""
    model = _model(off_grid=off_grid)
    model.solve(params={"discount_factor": 1.0, "reward": 2.0}, log_level="warning")
    rows = [
        record.getMessage()
        for record in caplog.records
        if "Off-grid next states:" in record.getMessage()
    ]
    assert any(
        "regime='alive'" in row
        and "period=0" in row
        and "state='wealth'" in row
        and "count=1" in row
        for row in rows
    )


@pytest.mark.parametrize("runtime_checks", [False, True])
def test_explicit_raise_accepts_economically_capped_transitions(
    *,
    runtime_checks: bool,
) -> None:
    """A grid policy does not replace a model's economic transition cap."""
    params = {"discount_factor": 1.0, "reward": 2.0}
    solution = _model(off_grid="raise", hand_capped=True).solve(
        params=params, runtime_checks=runtime_checks, log_level="off"
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["alive"], [2, 4, 6], decimal=DECIMAL_PRECISION
    )


def test_clipping_preserves_invalid_values_instead_of_repairing_the_law() -> None:
    """Finite tails clip; NaN and both infinities remain invalid."""
    values = jnp.array([-jnp.inf, -1.0, 0.0, 2.0, 3.0, jnp.inf, jnp.nan])
    got = jax.jit(_clip_grid_value)(value=values, points=jnp.array([0.0, 2.0]))
    np.testing.assert_array_equal(got, [-np.inf, 0, 0, 2, 2, np.inf, np.nan])


@pytest.mark.parametrize("runtime_checks", [False, True])
def test_raise_excludes_infeasible_exterior_transition_cells(
    *, runtime_checks: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the excluded wealth2/action0 cell lands outside the target grid."""
    solution = _model(off_grid="raise", mask_tail=True).solve(
        params={"discount_factor": 1.0, "reward": 2.0},
        runtime_checks=runtime_checks,
        log_level="debug",
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["alive"], [2, 4, 6], decimal=DECIMAL_PRECISION
    )
    rows = [
        record.getMessage()
        for record in caplog.records
        if "Off-grid next states:" in record.getMessage()
    ]
    if runtime_checks:
        assert rows
        assert all("count=0" in row for row in rows)


@categorical(ordered=False)
class Health:
    good: ScalarInt
    tail: ScalarInt
    poor: ScalarInt


def _node_terminal(*, wealth: ContinuousState, health: DiscreteState) -> FloatND:
    return wealth**2 + 0.0 * health


def _node_model(
    *,
    tail_probability: float,
    tail_value: float,
    off_grid: OffGrid = "raise",
) -> Model:
    def probabilities() -> FloatND:
        return jnp.array([0.5, tail_probability, 0.5 - tail_probability])

    def entry(next_health: DiscreteState) -> FloatND:
        return jnp.array([1.0, tail_value, 2.0])[next_health]

    def zero() -> FloatND:
        return jnp.asarray(0.0)

    return Model(
        regimes={
            "alive": Regime(
                functions={"utility": zero},
                state_transitions={
                    "wealth": {"terminal": entry},
                    "health": {"terminal": StochasticTransition(func=probabilities)},
                },
            ),
            "terminal": Regime(
                functions={"utility": _node_terminal},
                states={
                    "wealth": LinSpacedGrid(
                        start=0, stop=2, n_points=3, off_grid=off_grid
                    ),
                    "health": DiscreteGrid(category_class=Health),
                },
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"alive": {"terminal": 0}},
        initial_nodes={0: "alive"},
        regime_id_class=RegimeId,
    )


@pytest.mark.parametrize("tail_value", [10.0, float("nan")])
def test_raise_omits_impossible_exterior_and_poison_nodes(tail_value: float) -> None:
    """An impossible discrete draw cannot create an off-grid violation."""
    solution = _node_model(tail_probability=0.0, tail_value=tail_value).solve(
        params={"discount_factor": 1.0}, log_level="off"
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["alive"], 2.5, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("tail_value", [10.0, float("nan")])
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_raise_refuses_reachable_tail_even_with_tiny_mass(
    *, tail_value: float, runtime_checks: bool
) -> None:
    """Positive support, rather than the magnitude of its mass, controls raising."""
    with pytest.raises(
        OffGridStateError, match=r"regime='alive'.*period=0.*state='wealth'"
    ):
        _node_model(tail_probability=1e-10, tail_value=tail_value).solve(
            params={"discount_factor": 1.0},
            log_level="off",
            runtime_checks=runtime_checks,
        )


@pytest.mark.parametrize("off_grid", ["extrapolate", "clip_lookup", "clip_state"])
@pytest.mark.parametrize("reachable", [False, True])
def test_clipping_cannot_hide_a_feasible_nan_transition(
    *,
    off_grid: OffGrid,
    reachable: bool,
) -> None:
    """Impossible poison is omitted; feasible poison retains the checked failure."""
    model = _node_model(
        tail_probability=0.1 if reachable else 0.0,
        tail_value=float("nan"),
        off_grid=off_grid,
    )
    if reachable:
        with pytest.raises(InvalidValueFunctionError, match="alive"):
            model.solve(params={"discount_factor": 1.0}, log_level="off")
    else:
        solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
        np.testing.assert_array_almost_equal(
            solution.values[0]["alive"],
            2.5,
            decimal=DECIMAL_PRECISION,
        )


@pytest.mark.parametrize("off_grid", ["clip_lookup", "clip_state"])
def test_averaged_shock_reads_clip_each_physical_landing(off_grid: OffGrid) -> None:
    """Average endpoint-clipped reads, rather than clipping the mean shock."""
    grid = LinSpacedGrid(start=0, stop=2, n_points=3, off_grid=off_grid)
    finder = _get_coordinate_finder(in_name="next_wealth", grid=grid)
    points = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 1.0, 4.0])
    shocks = np.array([-1.0, 1.0])
    weights = np.array([0.5, 0.5])
    knots, averaged = average_over_shock(
        values=jnp.asarray(values),
        points=jnp.asarray(points),
        coordinate=lambda value: finder(next_wealth=value),
        shocks=jnp.asarray(shocks),
        weights=jnp.asarray(weights),
    )
    queries = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    expected = sum(
        weight * np.interp(queries + shock, points, values)
        for shock, weight in zip(shocks, weights, strict=True)
    )
    np.testing.assert_array_almost_equal(
        np.interp(queries, np.asarray(knots), np.asarray(averaged)),
        expected,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("reachable", [False, True])
def test_joint_tail_support_survives_an_underflowed_probability_product(
    *,
    runtime_checks: bool,
    reachable: bool,
) -> None:
    """Normal positive factors reach a tail even when their product rounds away."""
    exponent = -512 if jnp.asarray(0.0).dtype == np.float64 else -64
    factor = jnp.ldexp(jnp.asarray(1.0), exponent)
    assert float(factor) > 0
    assert float(factor * factor) == 0

    def probabilities() -> FloatND:
        mass = factor if reachable else jnp.asarray(0.0)
        return jnp.array([1.0 - mass, mass, 0.0])

    def entry(*, next_health: DiscreteState, next_mood: DiscreteState) -> FloatND:
        return jnp.where(
            (next_health == Health.tail) & (next_mood == Health.tail), 3.0, 1.0
        )

    def zero() -> FloatND:
        return jnp.asarray(0.0)

    def terminal(
        *,
        wealth: ContinuousState,
        health: DiscreteState,
        mood: DiscreteState,
    ) -> FloatND:
        return wealth**2 + 0.0 * health + 0.0 * mood

    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": zero},
                state_transitions={
                    "wealth": {"terminal": entry},
                    "health": {"terminal": StochasticTransition(func=probabilities)},
                    "mood": {"terminal": StochasticTransition(func=probabilities)},
                },
            ),
            "terminal": Regime(
                functions={"utility": terminal},
                states={
                    "wealth": LinSpacedGrid(
                        start=0, stop=2, n_points=3, off_grid="raise"
                    ),
                    "health": DiscreteGrid(category_class=Health),
                    "mood": DiscreteGrid(category_class=Health),
                },
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"alive": {"terminal": 0}},
        initial_nodes={0: "alive"},
        regime_id_class=RegimeId,
    )
    if reachable:
        with pytest.raises(OffGridStateError, match=r"regime='alive'.*state='wealth'"):
            model.solve(
                params={"discount_factor": 1.0},
                log_level="off",
                runtime_checks=runtime_checks,
            )
    else:
        solution = model.solve(
            params={"discount_factor": 1.0},
            log_level="off",
            runtime_checks=runtime_checks,
        )
        np.testing.assert_array_equal(solution.values[0]["alive"], 1.0)


def test_budgeted_mandatory_counts_reuse_the_warm_executable() -> None:
    """Silent unchecked simulation still enforces raises without warm lowering."""
    model = _model(off_grid="raise", hand_capped=True, device_memory_bytes=2**28)
    params = {"discount_factor": 1.0, "reward": 2.0}
    solution = model.solve(params=params, log_level="off")
    initial = {
        "age": jnp.zeros(2),
        "regime_id": jnp.zeros(2, dtype=jnp.int32),
        "wealth": jnp.array([0.0, 2.0]),
    }
    with count_compile_requests() as cold_requests:
        cold = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            runtime_checks=False,
            log_level="off",
        ).to_dataframe(use_labels=False)
    with count_compile_requests() as warm_requests:
        warm = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            runtime_checks=False,
            log_level="off",
        ).to_dataframe(use_labels=False)
    pd.testing.assert_frame_equal(warm, cold, check_exact=True)
    assert (cold_requests.compile_requests > 0, warm_requests.compile_requests) == (
        True,
        0,
    )


def _shock_count_utility(*, wealth: ContinuousState, z: ContinuousState) -> FloatND:
    return wealth**2 + z**2


def _wealth_from_next_shock(next_z: ContinuousState) -> FloatND:
    return 1 + next_z


def test_budgeted_counts_read_changed_node_operands_on_the_same_model() -> None:
    """The admitted producer reads fresh supplied nodes, not captured arrays."""
    grid = LinSpacedGrid(start=0, stop=2, n_points=3, off_grid="raise")
    shock = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)
    model = Model(
        regimes={
            "alive": Regime(
                states={"wealth": grid, "z": shock},
                functions={"utility": _shock_count_utility},
                state_transitions={"wealth": _wealth_from_next_shock},
            ),
            "terminal": Regime(
                states={"wealth": grid, "z": shock},
                functions={"utility": _shock_count_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"alive": {"terminal": 0}},
        initial_nodes={0: "alive"},
        regime_id_class=RegimeId,
    )
    devices = tuple(jax.devices()[:1])
    memory = SimulationMemory(
        budget_bytes=2**28,
        devices=devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree={}),
    )
    source = model._regimes["alive"]
    target = model._regimes["terminal"]
    law = transition_checks._regime_laws_at_period(
        regime=source, regime_name="alive", period=0
    )[0][0]
    params = model._process_params({"discount_factor": 1.0})
    axes = target.solution.state_action_space(
        regime_params=regime_kernel_params(params, regime_name="terminal")
    ).states
    for node_scale, expected in ((0.1, 0), (1.0, 18)):
        summary = transition_checks._ValidationSummary(memory=memory)
        try:
            counts = transition_checks._off_grid_target_counts(
                regime=source,
                phase=source.solution,
                target="terminal",
                law=law,
                points={"wealth": grid.to_jax()},
                target_axes=MappingProxyType(
                    dict(axes) | {"z": node_scale * axes["z"]}
                ),
                period=0,
                flat_params=params,
                ages=model._time,
                summary=summary,
                process_grid_resolver=None,
            )
            np.testing.assert_array_equal(counts, [[expected, 0]])
        finally:
            summary.close()


def _advance_one(x: ContinuousState) -> FloatND:
    return x + 1.0


def _zero_utility(x: ContinuousState) -> FloatND:
    return 0.0 * x


_SUBJECT_LAYOUTS = pytest.mark.parametrize(
    ("n_subjects", "subject_width"),
    [
        pytest.param(1, 2, id="one-subject-width-clamps"),
        pytest.param(2, 1, id="two-subjects-unit-width"),
        pytest.param(2, 3, id="two-subjects-width-clamps"),
        pytest.param(3, 2, id="three-subjects-one-pad-row"),
        pytest.param(4, 2, id="four-subjects-exact-multiple"),
        pytest.param(5, 4, id="five-subjects-three-pad-rows"),
        pytest.param(6, 4, id="six-subjects-two-pad-rows"),
        pytest.param(7, 3, id="seven-subjects-two-pad-rows"),
        pytest.param(8, 3, id="eight-subjects-one-pad-row"),
        pytest.param(8, 4, id="eight-subjects-exact-multiple"),
        pytest.param(3, 0, id="three-subjects-unbatched"),
        pytest.param(8, 0, id="eight-subjects-unbatched"),
    ],
)


def _simulate_every_subject_off_grid(
    *, n_subjects: int, subject_width: int, caplog: pytest.LogCaptureFixture
) -> tuple[pd.DataFrame, list[int]]:
    """Simulate distinct dyadic subjects that all leave `[0, 1]` once.

    Returns the public panel and the realized `alive -> terminal` counts.
    """
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    model = Model(
        regimes={
            "alive": Regime(
                states={"x": grid},
                functions={"utility": _zero_utility},
                state_transitions={"x": _advance_one},
            ),
            "terminal": Regime(states={"x": grid}, functions={"utility": _terminal_x}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "alive"},
        edges={"alive": {"terminal": 0}},
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(
            device_memory_bytes=None,
            axis_widths={} if subject_width == 0 else {"subject": subject_width},
        ),
    )
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off")
    caplog.clear()
    with caplog.at_level("INFO"):
        frame = model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "age": jnp.zeros(n_subjects),
                "regime_id": jnp.full(n_subjects, RegimeId.alive),
                "x": jnp.arange(1, n_subjects + 1) / 8.0,
            },
            runtime_checks=True,
            log_level="warning",
            seed=0,
        ).to_dataframe(use_labels=False)
    counts = [
        int(message.rsplit("count=", 1)[1])
        for record in caplog.records
        if "Off-grid next states:" in (message := record.getMessage())
        and "phase='realized'" in message
        and "target='terminal'" in message
    ]
    return frame, counts


def _terminal_x(x: ContinuousState) -> FloatND:
    return x**2


@_SUBJECT_LAYOUTS
def test_realized_off_grid_count_includes_each_genuine_subject_once(
    *, n_subjects: int, subject_width: int, caplog: pytest.LogCaptureFixture
) -> None:
    """Synthetic alignment rows never add to the realized exterior count."""
    _, counts = _simulate_every_subject_off_grid(
        n_subjects=n_subjects, subject_width=subject_width, caplog=caplog
    )
    assert sum(counts) == n_subjects


@_SUBJECT_LAYOUTS
def test_realized_off_grid_panel_keeps_raw_states_in_subject_order(
    *, n_subjects: int, subject_width: int, caplog: pytest.LogCaptureFixture
) -> None:
    """The public panel holds every genuine subject's raw exterior handoff once."""
    frame, _ = _simulate_every_subject_off_grid(
        n_subjects=n_subjects, subject_width=subject_width, caplog=caplog
    )
    initial = np.arange(1, n_subjects + 1) / 8.0
    expected = pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(n_subjects), 2),
            "period": np.tile([0, 1], n_subjects),
            "x": np.column_stack((initial, initial + 1.0)).reshape(-1),
        }
    )
    pd.testing.assert_frame_equal(
        frame[["subject_id", "period", "x"]].reset_index(drop=True),
        expected,
        check_dtype=False,
        check_exact=True,
    )
