"""A state only utility reads is broadcast past the continuation, not batched into it.

The model's `alive` regime carries three states:

- `health`, a two-point Markov state its own law reads;
- `wealth`, a continuous state its own law reads;
- `habit`, a discrete state utility reads but no law does: `next_habit` reads the
  work action only, as a lagged choice would.

So the expected continuation `E[V' | x, a]` is constant along `habit`. GridSearch maps
`habit` outside the flat state cell, so the continuation is computed once per
remaining cell and broadcast along `habit`, while utility and the argmax still see
every `habit` point.

Three properties are tested:

- **Structure.** With the cell width fixed at the whole state product, the additive
  reductions the lowered `alive` programs perform (the expectation over next-period
  nodes and the regime mixture) keep the same operand shapes when `habit` grows from
  two to three points. The maximum over actions still grows with it, which shows the
  reduction census can see a `habit` axis. A planned width counts every product
  point, `habit` included, so it moves with the habit count and is not held fixed.
- **Parity.** Solving with and without the broadcast publishes the same arrays: byte
  for byte on the CPU backend; elsewhere identical shapes, dtypes, non-finite
  entries and discrete arrays, with every finite float within 4 ULP in float64 and
  1 ULP in float32. Simulating from either solve with the same initial conditions
  and seed yields the same discrete choices, states and regimes.
- **Width settings.** The broadcast is used only at cell widths that are whole
  multiples of `habit`'s extent. A pinned width or a ceiling that is not such a
  multiple is honoured unchanged and solved in the plain layout, as is a budget
  under which no window of such a multiple fits; every other solve keeps the
  broadcast. Values agree with the solve that never broadcasts.
"""

import dataclasses
import functools
import logging
import re
import tempfile
from collections.abc import Hashable, Mapping
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.execution.core_program import TiledOutputAxis
from _lcm.execution.workspace_planning import (
    CompilerMemoryReservation,
    workspace_width_candidates,
)
from _lcm.solution import backward_induction, grid_search
from _lcm.solution.period_capture import _PAYLOAD_NAME
from _lcm.typing import ArrayTree, QAndFArg, QAndFFunction
from _lcm.utils import dispatchers
from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.execution import WidthSearch, WidthSearchPolicy
from lcm.regime import Regime
from lcm.solver_api import SolutionResult
from lcm.solvers import CELL_AXIS
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    StateName,
    UserParams,
)
from tests.solution.test_covered_axis_parity import (
    _NON_FINITE_CLASSES,
    _bytes,
    _class_masks,
    _discrete_bytes,
    _shapes_and_dtypes,
    _ulp_excess,
)

_N_PERIODS = 3
_FINAL_AGE_ALIVE = _N_PERIODS - 2
_N_SUBJECTS = 200
_SIMULATION_SEED = 1
_N_HEALTH = 2
_N_WEALTH = 12
# The alive cell's width is fixed at its whole extent, so every habit count
# evaluates the whole state product in one window, or left to the planner.
_WHOLE = "whole"
_PLANNED = "planned"
_CASES = tuple((n_habits, arm) for n_habits in (2, 3) for arm in (_WHOLE, _PLANNED))
# `stablehlo.reduce(%x init: %c) applies stablehlo.<op> across dimensions = [..] :
# (tensor<AxBx..xf32>, ...` — the reduction's body operation and its operand shape.
_REDUCE = re.compile(
    r"stablehlo\.reduce\(%[^ ]+ init: %[^)]+\) applies stablehlo\.(\w+) "
    r"across dimensions = \[[\d, ]*\] : \(tensor<([\dx]*)x?[a-z]+\d+>"
)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Work:
    rest: ScalarInt
    work: ScalarInt


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=True)
class TwoHabits:
    none: ScalarInt
    some: ScalarInt


@categorical(ordered=True)
class ThreeHabits:
    none: ScalarInt
    some: ScalarInt
    strong: ScalarInt


_HABITS = {2: TwoHabits, 3: ThreeHabits}


def _utility(
    *,
    consumption: ContinuousAction,
    work: DiscreteAction,
    habit: DiscreteState,
    health: DiscreteState,
    disutility_of_work: float,
) -> FloatND:
    """Work costs more with a stronger habit and in bad health."""
    return jnp.log(consumption) - disutility_of_work * work * (1.0 + habit) * (
        2.0 - health
    )


def _next_wealth(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    work: DiscreteAction,
    interest_rate: float,
) -> ContinuousState:
    return (1.0 + interest_rate) * (wealth - consumption) + 10.0 * work


def _next_habit(*, work: DiscreteAction) -> DiscreteState:
    """Next period's habit is this period's work choice."""
    return work


def _next_health(*, health: DiscreteState) -> FloatND:
    return jnp.where(
        health == Health.good, jnp.array([0.2, 0.8]), jnp.array([0.6, 0.4])
    )


def _next_regime(*, age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


def _borrowing_constraint(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def _build(*, n_habits: int, arm: str) -> tuple[Model, UserParams]:
    """Build the model with `n_habits` habit points and its parameters."""
    model = _model(
        n_habits=n_habits,
        execution_config=ExecutionConfig(
            axis_widths=(
                {CELL_AXIS: {"alive": _N_HEALTH * n_habits * _N_WEALTH}}
                if arm == _WHOLE
                else {}
            )
        ),
    )
    return model, _params()


def _model(*, n_habits: int, execution_config: ExecutionConfig) -> Model:
    """Build the model with `n_habits` habit points under `execution_config`."""
    alive = Regime(
        actions={
            "work": DiscreteGrid(category_class=Work),
            "consumption": LinSpacedGrid(start=1, stop=60, n_points=10),
        },
        states={
            "health": DiscreteGrid(category_class=Health),
            "habit": DiscreteGrid(category_class=_HABITS[n_habits]),
            "wealth": LinSpacedGrid(start=1, stop=60, n_points=_N_WEALTH),
        },
        state_transitions={
            "health": StochasticTransition(func=_next_health),
            "habit": _next_habit,
            "wealth": _next_wealth,
        },
        constraints={"borrowing_constraint": _borrowing_constraint},
        functions={"utility": _utility},
    )
    dead = Regime(functions={"utility": lambda: 0.0})
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=_FINAL_AGE_ALIVE + 1, step="Y"),
        regime_id_class=RegimeId,
        execution_config=execution_config,
        initial_nodes={0: "alive"},
        edges={
            "alive": Transition(
                targets={"alive": 0, "dead": (0, 1)},
                law=DeterministicTransition(func=_next_regime),
            )
        },
    )


def _params() -> UserParams:
    """Return the model's parameters."""
    params: UserParams = {
        "discount_factor": 0.95,
        "alive": {
            "utility": {"disutility_of_work": 0.3},
            "next_wealth": {"interest_rate": 0.05},
        },
        "final_age_alive": _FINAL_AGE_ALIVE,
    }
    return params


@functools.cache
def _run(
    *, n_habits: int, arm: str, broadcast: bool
) -> tuple[SolutionResult, pd.DataFrame, tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Solve and simulate one case, observing the broadcast choice and the lowerings.

    Return the solution, the labelled simulated panel, every state tuple the
    broadcast selection returned, and the StableHLO text of every lowered
    `alive` program.
    """
    selections: list[tuple[str, ...]] = []
    lowered: list[str] = []
    # Absent from a revision without the broadcast, where the solve never asks.
    select = getattr(grid_search, "_continuation_unread_state_names", None)
    compile_and_log = backward_induction._compile_and_log

    def observed_select(
        *, Q_and_F: QAndFFunction, inner_state_names: tuple[StateName, ...]
    ) -> tuple[StateName, ...]:
        selected = (
            select(Q_and_F=Q_and_F, inner_state_names=inner_state_names)
            if broadcast and select is not None
            else ()
        )
        selections.append(selected)
        return selected

    def observed_compile(
        *,
        lowering_key: Hashable,
        low: jax.stages.Lowered,
        label: str,
        log_kernel_memory: bool,
        logger: logging.Logger,
        phase: str | None,
    ) -> tuple[Hashable, jax.stages.Compiled]:
        if label.startswith("alive "):
            lowered.append(low.as_text())
        return compile_and_log(
            lowering_key=lowering_key,
            low=low,
            label=label,
            log_kernel_memory=log_kernel_memory,
            logger=logger,
            phase=phase,
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            grid_search,
            "_continuation_unread_state_names",
            observed_select,
            raising=False,
        )
        monkeypatch.setattr(backward_induction, "_compile_and_log", observed_compile)
        model, params = _build(n_habits=n_habits, arm=arm)
        result = model.solve(params=params, log_level="off")
        simulation = model.simulate(
            params=params,
            initial_conditions=_initial_conditions(),
            solution=result,
            log_level="off",
            seed=_SIMULATION_SEED,
        )
    return (
        result,
        simulation.to_dataframe(terminal_rows="all"),
        tuple(selections),
        tuple(lowered),
    )


def _initial_conditions() -> dict[str, jax.Array]:
    """Subjects spread over wealth, both health states and both first habits."""
    codes = jnp.arange(_N_SUBJECTS, dtype=jnp.int32)
    return {
        "wealth": jnp.linspace(1.0, 60.0, _N_SUBJECTS),
        "health": codes % 2,
        "habit": (codes // 2) % 2,
        "age": jnp.zeros(_N_SUBJECTS),
        "regime_id": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
    }


def _reduce_shapes(
    *, lowered: tuple[str, ...], additive: bool
) -> list[tuple[str, str]]:
    """Body operation and operand shape of every additive or other reduction."""
    return sorted(
        (operation, shape)
        for text in lowered
        for operation, shape in _REDUCE.findall(text)
        if (operation == "add") is additive
    )


def _case_id(case: tuple[int, str]) -> str:
    """Name a case as `<n_habits>habits-<arm>`."""
    return f"{case[0]}habits-{case[1]}"


def test_additive_reductions_do_not_scale_with_a_state_no_law_reads() -> None:
    """With the whole state product in one window, the continuation's sums keep
    their operand shapes from two to three habits."""
    two = _run(n_habits=2, arm=_WHOLE, broadcast=True)[3]
    three = _run(n_habits=3, arm=_WHOLE, broadcast=True)[3]

    assert _reduce_shapes(lowered=two, additive=True) == _reduce_shapes(
        lowered=three, additive=True
    )


def test_the_reduction_census_sees_the_habit_axis() -> None:
    """The non-additive reductions of the same programs do grow with `habit`."""
    two = _run(n_habits=2, arm=_WHOLE, broadcast=True)[3]
    three = _run(n_habits=3, arm=_WHOLE, broadcast=True)[3]

    assert _reduce_shapes(lowered=two, additive=False) != _reduce_shapes(
        lowered=three, additive=False
    )


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_grid_search_broadcasts_exactly_the_habit_state(
    *, case: tuple[int, str]
) -> None:
    """The alive regime's continuation reads every state but `habit`."""
    n_habits, arm = case
    _, _, selections, _ = _run(n_habits=n_habits, arm=arm, broadcast=True)

    assert set(selections) - {()} == {("habit",)}


@pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="Byte parity is the CPU-backend tier."
)
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_broadcast_leaves_every_published_array_bitwise_unchanged_on_cpu(
    *, case: tuple[int, str]
) -> None:
    """On the CPU backend, values and artifacts agree byte for byte."""
    n_habits, arm = case
    on = _run(n_habits=n_habits, arm=arm, broadcast=True)[0]
    off = _run(n_habits=n_habits, arm=arm, broadcast=False)[0]

    assert _bytes(on) == _bytes(off)


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_broadcast_keeps_every_shape_and_dtype(*, case: tuple[int, str]) -> None:
    """Every published array keeps its shape and dtype."""
    n_habits, arm = case
    on = _run(n_habits=n_habits, arm=arm, broadcast=True)[0]
    off = _run(n_habits=n_habits, arm=arm, broadcast=False)[0]

    assert _shapes_and_dtypes(on) == _shapes_and_dtypes(off)


@pytest.mark.parametrize("non_finite", tuple(_NON_FINITE_CLASSES))
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_broadcast_keeps_every_non_finite_entry_in_place(
    *, case: tuple[int, str], non_finite: str
) -> None:
    """NaN, +Inf and -Inf each occupy exactly the same entries of every float
    array."""
    n_habits, arm = case
    on = _run(n_habits=n_habits, arm=arm, broadcast=True)[0]
    off = _run(n_habits=n_habits, arm=arm, broadcast=False)[0]

    assert _class_masks(result=on, non_finite=non_finite) == _class_masks(
        result=off, non_finite=non_finite
    )


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_broadcast_keeps_every_discrete_array_exact(*, case: tuple[int, str]) -> None:
    """Every published integer or boolean array is unchanged."""
    n_habits, arm = case
    on = _run(n_habits=n_habits, arm=arm, broadcast=True)[0]
    off = _run(n_habits=n_habits, arm=arm, broadcast=False)[0]

    assert _discrete_bytes(on) == _discrete_bytes(off)


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_broadcast_moves_no_finite_value_beyond_the_dtype_ulp_bound(
    *, case: tuple[int, str]
) -> None:
    """Every finite float agrees to 4 ULP in float64 and 1 ULP in float32."""
    n_habits, arm = case
    on = _run(n_habits=n_habits, arm=arm, broadcast=True)[0]
    off = _run(n_habits=n_habits, arm=arm, broadcast=False)[0]

    assert _ulp_excess(covered=on, uncovered=off) == {}


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_broadcast_keeps_simulated_discrete_choices_identical(
    *, case: tuple[int, str]
) -> None:
    """Simulating from either solve, with the same initial conditions and seed,
    yields the same work choice, habit, health and regime for every subject in
    every period, and the work choice varies across subjects."""
    n_habits, arm = case
    on = _run(n_habits=n_habits, arm=arm, broadcast=True)[1]
    off = _run(n_habits=n_habits, arm=arm, broadcast=False)[1]
    columns = list(on.select_dtypes(include="category").columns)
    finite = on[np.isfinite(on["value"].to_numpy())]
    assert finite["work"].nunique() >= 2, "work does not vary among finite rows"

    pd.testing.assert_frame_equal(on[columns], off[columns])


# The three-habit model broadcasts `habit`, so a window holds whole multiples of
# its three points.
_BROADCAST_EXTENT = 3
_BUDGET = 10**9
_REFUSED = 2**62
# Width settings, each named for its test id. A pin or ceiling that is not a
# multiple of the broadcast extent is incompatible with it; so is a budget under
# which no window of a multiple of the broadcast extent fits.
_INCOMPATIBLE = (
    "ceiling-2",
    "pin-1",
    "pin-5",
    "budget-refuses-broadcast-windows-exhaustive",
    "budget-refuses-broadcast-windows-bounded",
)
_COMPATIBLE = ("pin-6", "unbudgeted", "budget-exhaustive", "budget-bounded")
_REFUSING_BUDGETS = (
    "budget-refuses-broadcast-windows-exhaustive",
    "budget-refuses-broadcast-windows-bounded",
)


def _execution_config(*, setting: str) -> ExecutionConfig:
    """Return the execution configuration one width setting names."""
    bounded = WidthSearchPolicy(kind=WidthSearch.BOUNDED)
    return {
        "ceiling-2": ExecutionConfig(axis_width_ceilings={CELL_AXIS: 2}),
        "pin-1": ExecutionConfig(axis_widths={CELL_AXIS: {"alive": 1}}),
        "pin-5": ExecutionConfig(axis_widths={CELL_AXIS: {"alive": 5}}),
        "pin-6": ExecutionConfig(axis_widths={CELL_AXIS: {"alive": 6}}),
        "unbudgeted": ExecutionConfig(),
        "budget-exhaustive": ExecutionConfig(device_memory_bytes=_BUDGET),
        "budget-bounded": ExecutionConfig(
            device_memory_bytes=_BUDGET, width_search=bounded
        ),
        "budget-refuses-broadcast-windows-exhaustive": ExecutionConfig(
            device_memory_bytes=_BUDGET
        ),
        "budget-refuses-broadcast-windows-bounded": ExecutionConfig(
            device_memory_bytes=_BUDGET, width_search=bounded
        ),
    }[setting]


@functools.cache
def _solve_under(
    *, setting: str, broadcast: bool
) -> tuple[SolutionResult, int, frozenset[tuple[tuple[str, ...], int]]]:
    """Solve the three-habit model under one width setting.

    Under a `budget-refuses-broadcast-windows-*` setting, every compiled program
    whose cell width is at least the broadcast extent reports a reservation no
    budget admits, so only narrower windows fit.

    Return the solution, the cell width the first `alive` period dispatched at,
    and every traced cell window as its mapped state tuple and its width in
    cells. Refused budget candidates and later periods are traced too, so a
    window is attributed to the dispatched program by its width.
    """
    windows: list[tuple[tuple[str, ...], int]] = []
    select = grid_search._continuation_unread_state_names
    map_window = dispatchers._TiledProductMap.__call__
    reserve = backward_induction.compiler_memory_reservation

    def observed_map_window(
        self: dispatchers._TiledProductMap, **kwargs: QAndFArg
    ) -> ArrayTree:
        width = kwargs.get(self.width_keyword, 1)
        assert isinstance(width, int)
        windows.append((self.variables, width))
        return map_window(self, **kwargs)

    def refusing_reserve(
        *, compiled: jax.stages.Compiled, widths: Mapping[str, int]
    ) -> CompilerMemoryReservation:
        memory = reserve(compiled=compiled, widths=widths)
        if widths.get(CELL_AXIS, 0) < _BROADCAST_EXTENT:
            return memory
        return CompilerMemoryReservation(
            records=(dataclasses.replace(memory.records[0], peak_bytes=_REFUSED),)
        )

    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        tempfile.TemporaryDirectory() as capture_dir,
    ):
        monkeypatch.setattr(
            grid_search,
            "_continuation_unread_state_names",
            select if broadcast else lambda **_: (),
        )
        monkeypatch.setattr(
            dispatchers._TiledProductMap, "__call__", observed_map_window
        )
        if setting in _REFUSING_BUDGETS:
            monkeypatch.setattr(
                backward_induction, "compiler_memory_reservation", refusing_reserve
            )
        monkeypatch.setenv("LCM_CAPTURE_PERIOD", "alive@0")
        monkeypatch.setenv("LCM_CAPTURE_DIR", capture_dir)
        model = _model(
            n_habits=_BROADCAST_EXTENT,
            execution_config=_execution_config(setting=setting),
        )
        result = model.solve(params=_params(), log_level="off")
        with (Path(capture_dir) / "alive@0" / _PAYLOAD_NAME).open("rb") as stream:
            widths = cloudpickle.load(stream)["core_tile_widths"]["main"]
    cells = frozenset(window for window in windows if "wealth" in window[0])
    return result, widths[CELL_AXIS], cells


@pytest.mark.parametrize(
    ("setting", "expected"),
    [("ceiling-2", 2), ("pin-1", 1), ("pin-5", 5), ("pin-6", 6)],
)
def test_user_cell_width_setting_is_dispatched_unchanged(
    *, setting: str, expected: int
) -> None:
    """A pinned cell width is dispatched as pinned, and an unbudgeted solve under a
    ceiling dispatches at the ceiling, whether or not the width is a multiple of
    the broadcast extent."""
    assert _solve_under(setting=setting, broadcast=True)[1] == expected


@pytest.mark.parametrize("setting", _REFUSING_BUDGETS)
def test_budget_refusing_every_broadcast_window_selects_a_narrower_width(
    *, setting: str
) -> None:
    """When no window of whole broadcast-extent multiples fits the budget, the solve
    dispatches a plain width below the broadcast extent."""
    assert _solve_under(setting=setting, broadcast=True)[1] < _BROADCAST_EXTENT


@pytest.mark.parametrize("setting", _INCOMPATIBLE)
def test_incompatible_width_setting_maps_every_cell_state_in_the_window(
    *, setting: str
) -> None:
    """A width the broadcast cannot serve runs the plain layout: a cell window of
    the dispatched width maps `habit` together with the other cell states."""
    _, width, cells = _solve_under(setting=setting, broadcast=True)

    assert (("health", "habit", "wealth"), width) in cells


@pytest.mark.parametrize("setting", _COMPATIBLE)
def test_compatible_width_setting_broadcasts_habit(*, setting: str) -> None:
    """A width of whole broadcast-extent multiples keeps `habit` outside the cell
    window: the window maps the dispatched width's share of the other states."""
    _, width, cells = _solve_under(setting=setting, broadcast=True)

    assert (("health", "wealth"), width // _BROADCAST_EXTENT) in cells


@pytest.mark.parametrize("setting", [*_INCOMPATIBLE, *_COMPATIBLE])
def test_width_setting_solves_to_the_values_of_the_plain_layout(
    *, setting: str
) -> None:
    """Every finite float the solve publishes agrees with the solve that never
    broadcasts, to 4 ULP in float64 and 1 ULP in float32."""
    on = _solve_under(setting=setting, broadcast=True)[0]
    off = _solve_under(setting=setting, broadcast=False)[0]

    assert _ulp_excess(covered=on, uncovered=off) == {}


def test_budgeted_cell_frontier_rounds_widths_at_or_above_the_broadcast_extent() -> (
    None
):
    """The frontier is the plain ladder with every width at or above the broadcast
    extent rounded down onto its multiples; narrower widths stay as they are."""
    axis = TiledOutputAxis(
        name=CELL_AXIS,
        state_names=("health", "habit", "wealth"),
        extent=72,
        width_keyword="_lcm_cell_width",
        preferred_alignment=_BROADCAST_EXTENT,
    )
    candidates = workspace_width_candidates(axes=(axis,), budget_bytes=_BUDGET)

    assert [widths[CELL_AXIS] for widths in candidates] == [72, 63, 30, 15, 6, 3, 2, 1]
