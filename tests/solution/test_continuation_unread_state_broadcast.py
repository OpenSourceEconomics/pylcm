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

Two properties are tested:

- **Structure.** The additive reductions the lowered `alive` programs perform (the
  expectation over next-period nodes and the regime mixture) keep the same operand
  shapes when `habit` grows from two to three points. The maximum over actions still
  grows with it, which shows the reduction census can see a `habit` axis.
- **Parity.** Solving with and without the broadcast publishes the same arrays: byte
  for byte on the CPU backend; elsewhere identical shapes, dtypes, non-finite
  entries and discrete arrays, with every finite float within 4 ULP in float64 and
  1 ULP in float32. Simulating from either solve with the same initial conditions
  and seed yields the same discrete choices, states and regimes.
"""

import functools
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.solution import backward_induction, grid_search
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import Regime
from lcm.solvers import CELL_AXIS
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
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
_COVERED = "covered"
_PLANNED = "planned"
_CASES = tuple((n_habits, arm) for n_habits in (2, 3) for arm in (_COVERED, _PLANNED))
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
    alive = Regime(
        actions={
            "work": DiscreteGrid(category_class=Work),
            "consumption": LinSpacedGrid(start=1, stop=60, n_points=10),
        },
        states={
            "health": DiscreteGrid(category_class=Health),
            "habit": DiscreteGrid(category_class=_HABITS[n_habits]),
            "wealth": LinSpacedGrid(start=1, stop=60, n_points=12),
        },
        state_transitions={
            "health": MarkovTransition(_next_health),
            "habit": _next_habit,
            "wealth": _next_wealth,
        },
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        functions={"utility": _utility},
        active=lambda age: age <= _FINAL_AGE_ALIVE,
    )
    dead = Regime(transition=None, functions={"utility": lambda: 0.0})
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=_FINAL_AGE_ALIVE + 1, step="Y"),
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(
            covered_axes=(CELL_AXIS,) if arm == _COVERED else ()
        ),
    )
    params: UserParams = {
        "discount_factor": 0.95,
        "alive": {
            "utility": {"disutility_of_work": 0.3},
            "next_wealth": {"interest_rate": 0.05},
            "next_regime": {"final_age_alive": _FINAL_AGE_ALIVE},
        },
    }
    return model, params


@functools.cache
def _run(
    *, n_habits: int, arm: str, broadcast: bool
) -> tuple[Any, pd.DataFrame, tuple[tuple[str, ...], ...], tuple[str, ...]]:
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

    def observed_select(**kwargs: Any) -> tuple[str, ...]:
        selected = select(**kwargs) if broadcast and select is not None else ()
        selections.append(selected)
        return selected

    def observed_compile(**kwargs: Any) -> Any:
        if kwargs["label"].startswith("alive "):
            lowered.append(kwargs["low"].as_text())
        return compile_and_log(**kwargs)

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


@pytest.mark.parametrize("arm", [_COVERED, _PLANNED])
def test_additive_reductions_do_not_scale_with_a_state_no_law_reads(
    *, arm: str
) -> None:
    """The continuation's sums keep their operand shapes from two to three habits."""
    two = _run(n_habits=2, arm=arm, broadcast=True)[3]
    three = _run(n_habits=3, arm=arm, broadcast=True)[3]

    assert _reduce_shapes(lowered=two, additive=True) == _reduce_shapes(
        lowered=three, additive=True
    )


@pytest.mark.parametrize("arm", [_COVERED, _PLANNED])
def test_the_reduction_census_sees_the_habit_axis(*, arm: str) -> None:
    """The non-additive reductions of the same programs do grow with `habit`."""
    two = _run(n_habits=2, arm=arm, broadcast=True)[3]
    three = _run(n_habits=3, arm=arm, broadcast=True)[3]

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
