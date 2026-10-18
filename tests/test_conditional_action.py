"""`ConditionalAction` behaves exactly like the pin constraint it declares.

A worker chooses whether to work, the hours worked and consumption. Hours matter only
when working. The conditional declaration and the hand-written pin constraint
`(labor == yes) | (hours == inactive node)` describe the same feasible set, so the two
models solve and simulate identically.
"""

import functools
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ConditionalAction,
    DiscreteGrid,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    Periods,
    PeriodSpecializedFunction,
    Phased,
    categorical,
)
from lcm.exceptions import (
    GridInitializationError,
    InvalidSimulationInputError,
    ModelInitializationError,
)
from lcm.grids import ContinuousGrid
from lcm.persistence import load_solution
from lcm.regime import Regime
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    FunctionName,
    IntND,
    ScalarInt,
)
from tests.conftest import assert_agrees_to_ulp

HOURS_GRID = LinSpacedGrid(start=0.0, stop=40.0, n_points=5)
CONSUMPTION_GRID = LinSpacedGrid(start=0.5, stop=10.0, n_points=11)
WEALTH_GRID = LinSpacedGrid(start=1.0, stop=12.0, n_points=7)
N_SUBJECTS = 6


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Labor:
    no: ScalarInt
    yes: ScalarInt


def _utility(
    *,
    consumption: ContinuousAction,
    hours: ContinuousAction,
    labor: DiscreteAction,
    disutility: FloatND,
) -> FloatND:
    return jnp.log(consumption) - disutility * (hours / 40.0) - 0.1 * labor


def _next_wealth(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    hours: ContinuousAction,
) -> ContinuousState:
    return 1.03 * (wealth - consumption) + 0.1 * hours


def _borrowing(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _bequest(*, wealth: ContinuousState) -> FloatND:
    return jnp.sqrt(wealth)


def _hours_pin(inactive_value: float) -> Callable[..., BoolND]:
    def hours_only_when_working(
        *, labor: DiscreteAction, hours: ContinuousAction
    ) -> BoolND:
        return (labor == Labor.yes) | (hours == inactive_value)

    return hours_only_when_working


def _get_model(*, conditional: bool, inactive_value: float = 0.0) -> Model:
    if conditional:
        return _model_with_hours(
            hours=ConditionalAction(
                grid=HOURS_GRID,
                active_when={"labor": Labor.yes},
                inactive_value=inactive_value,
            )
        )
    return _model_with_hours(
        hours=HOURS_GRID, extra_constraints={"hours_only": _hours_pin(inactive_value)}
    )


def _model_with_hours(
    *,
    hours: ContinuousGrid,
    extra_constraints: dict[FunctionName, Callable[..., BoolND]] | None = None,
) -> Model:
    constraints = {"borrowing": _borrowing, **(extra_constraints or {})}
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    working = Regime(
        state_transitions={"wealth": _next_wealth},
        actions={
            "labor": DiscreteGrid(category_class=Labor),
            "hours": hours,
            "consumption": CONSUMPTION_GRID,
        },
        functions={"utility": _utility},
        constraints=constraints,
    )
    dead = Regime(functions={"utility": _bequest})
    return Model(
        regimes={"working": working, "dead": dead},
        states={"wealth": WEALTH_GRID},
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "working"},
        edges={"working": {"working": (0, 1), "dead": 2}},
    )


PARAMS = {
    "discount_factor": 0.95,
    "working": {"utility": {"disutility": 0.8}},
    "dead": {},
}


def _simulate(*, conditional: bool, inactive_value: float = 0.0) -> pd.DataFrame:
    model = _get_model(conditional=conditional, inactive_value=inactive_value)
    result = model.simulate(
        params=PARAMS,
        initial_conditions={
            "wealth": jnp.linspace(1.5, 11.5, N_SUBJECTS),
            "age": jnp.zeros(N_SUBJECTS),
            "regime_id": jnp.full(N_SUBJECTS, RegimeId.working, dtype=jnp.int32),
        },
        log_level="off",
    )
    return result.to_dataframe()


@pytest.mark.parametrize("inactive_value", [0.0, 10.0])
def test_conditional_action_solves_like_its_pin_constraint(
    inactive_value: float,
) -> None:
    """Value functions agree with the pin-constraint formulation in every period."""
    conditional = _get_model(conditional=True, inactive_value=inactive_value).solve(
        params=PARAMS, log_level="off"
    )
    pinned = _get_model(conditional=False, inactive_value=inactive_value).solve(
        params=PARAMS, log_level="off"
    )
    for period, regimes in pinned.values.items():
        for regime, expected in regimes.items():
            assert_agrees_to_ulp(
                got=conditional.values[period][regime],
                expected=expected,
                n_ulp=4,
                err_msg=f"period {period}, regime {regime}",
            )


@pytest.mark.parametrize("inactive_value", [0.0, 10.0])
def test_conditional_action_simulates_like_its_pin_constraint(
    inactive_value: float,
) -> None:
    """Simulated actions equal those of the pin-constraint formulation."""
    columns = ["subject_id", "period", "labor", "hours", "consumption"]
    conditional = _simulate(conditional=True, inactive_value=inactive_value)
    pinned = _simulate(conditional=False, inactive_value=inactive_value)
    pd.testing.assert_frame_equal(
        conditional[columns].reset_index(drop=True),
        pinned[columns].reset_index(drop=True),
    )


@pytest.mark.parametrize("inactive_value", [0.0, 10.0])
def test_inactive_conditional_action_takes_its_inactive_value(
    inactive_value: float,
) -> None:
    """Every simulated row that does not work reports the inactive hours exactly."""
    simulated = _simulate(conditional=True, inactive_value=inactive_value)
    living = simulated[simulated["regime_name"] == "working"]
    not_working = living.loc[living["labor"] == "no", "hours"].to_numpy()
    # Empty `not_working` gives an empty list, so the check cannot pass vacuously.
    assert np.unique(not_working).tolist() == [inactive_value]


def test_simulation_covers_both_labor_choices() -> None:
    """The simulated population chooses both labor options, so the comparisons
    above see active and inactive hours."""
    simulated = _simulate(conditional=True)
    living = simulated[simulated["regime_name"] == "working"]
    assert set(living["labor"]) == {"no", "yes"}


def test_conditional_action_reports_its_grid_nodes() -> None:
    """The declaration takes the nodes of the grid it wraps."""
    hours = ConditionalAction(
        grid=HOURS_GRID, active_when={"labor": Labor.yes}, inactive_value=0.0
    )
    np.testing.assert_array_equal(hours.to_jax(), HOURS_GRID.to_jax())


def test_conditional_action_normalizes_codes_to_sets() -> None:
    """A single code and a one-element tuple declare the same condition."""
    single = ConditionalAction(
        grid=HOURS_GRID, active_when={"labor": Labor.yes}, inactive_value=0.0
    )
    as_tuple = ConditionalAction(
        grid=HOURS_GRID, active_when={"labor": (Labor.yes,)}, inactive_value=0.0
    )
    assert single == as_tuple


@pytest.mark.parametrize(
    ("grid", "active_when", "inactive_value", "match"),
    [
        (HOURS_GRID, {"labor": 1}, 5.0, "not a node"),
        (HOURS_GRID, {}, 0.0, "names no discrete action"),
        (HOURS_GRID, {"labor": ()}, 0.0, "lists no value"),
        (IrregSpacedGrid(n_points=3), {"labor": 1}, 0.0, "points at runtime"),
        (
            ConditionalAction(
                grid=HOURS_GRID, active_when={"labor": 1}, inactive_value=0.0
            ),
            {"labor": 1},
            0.0,
            "plain continuous grid",
        ),
    ],
)
def test_conditional_action_rejects_invalid_declarations(
    *,
    grid: ContinuousGrid,
    active_when: dict[ActionName, int | tuple[int, ...]],
    inactive_value: float,
    match: str,
) -> None:
    """The grid must be static and plain, and the inactive value one of its nodes."""
    with pytest.raises(GridInitializationError, match=match):
        ConditionalAction(
            grid=grid, active_when=active_when, inactive_value=inactive_value
        )


@pytest.mark.parametrize(
    ("active_when", "match"),
    [
        ({"shift": 1}, "not a discrete action"),
        ({"consumption": 1}, "not a discrete action"),
        ({"wealth": 1}, "not a discrete action"),
        ({"labor": 2}, "does not take"),
    ],
)
def test_model_rejects_conditions_the_regime_cannot_evaluate(
    *, active_when: dict[ActionName, int], match: str
) -> None:
    """A condition must name a discrete action of the regime and values it takes."""
    hours = ConditionalAction(
        grid=HOURS_GRID, active_when=active_when, inactive_value=0.0
    )
    with pytest.raises(ModelInitializationError, match=match):
        _model_with_hours(hours=hours)


@pytest.mark.parametrize("inactive_value", [0.0, 10.0])
def test_lookup_policy_matches_the_pin_constraint(inactive_value: float) -> None:
    """`lookup_policy` returns the same actions as the pin-constraint formulation."""
    states = {"wealth": WEALTH_GRID.to_jax()}
    looked_up = {}
    for conditional in (True, False):
        model = _get_model(conditional=conditional, inactive_value=inactive_value)
        solution = model.solve(params=PARAMS, log_level="off")
        lookup = model.lookup_policy(
            params=PARAMS,
            solution=solution,
            period=0,
            regime_name="working",
            states=states,
        )
        assert lookup.actions is not None
        looked_up[conditional] = {
            name: np.asarray(values) for name, values in lookup.actions.items()
        }
    for name, expected in looked_up[False].items():
        np.testing.assert_array_equal(looked_up[True][name], expected, err_msg=name)


SHIFT_GRID = DiscreteGrid(values=[0, 8, 12])


def _utility_with_shift(
    *,
    consumption: ContinuousAction,
    hours: ContinuousAction,
    labor: DiscreteAction,
    shift: DiscreteAction,
    disutility: FloatND,
) -> FloatND:
    return (
        jnp.log(consumption) - disutility * (hours / 40.0) - 0.1 * labor - 0.01 * shift
    )


def _shift_pin(
    *, labor: DiscreteAction, shift: DiscreteAction, hours: ContinuousAction
) -> BoolND:
    return ((labor == Labor.yes) & ((shift == 8) | (shift == 12))) | (hours == 0.0)


def _get_shift_model(*, conditional: bool) -> Model:
    """Hours are active only when working a shift of 8 or 12 hours."""
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    if conditional:
        hours = ConditionalAction(
            grid=HOURS_GRID,
            active_when={"labor": Labor.yes, "shift": (8, 12)},
            inactive_value=0.0,
        )
        constraints = {"borrowing": _borrowing}
    else:
        hours = HOURS_GRID
        constraints = {"borrowing": _borrowing, "hours_only": _shift_pin}
    working = Regime(
        state_transitions={"wealth": _next_wealth},
        actions={
            "labor": DiscreteGrid(category_class=Labor),
            "shift": SHIFT_GRID,
            "hours": hours,
            "consumption": CONSUMPTION_GRID,
        },
        functions={"utility": _utility_with_shift},
        constraints=constraints,
    )
    dead = Regime(functions={"utility": _bequest})
    return Model(
        regimes={"working": working, "dead": dead},
        states={"wealth": WEALTH_GRID},
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "working"},
        edges={"working": {"working": (0, 1), "dead": 2}},
    )


def _assert_same_values(*, got: Model, expected: Model) -> None:
    got_solution = got.solve(params=PARAMS, log_level="off")
    expected_solution = expected.solve(params=PARAMS, log_level="off")
    for period, regimes in expected_solution.values.items():
        for regime, values in regimes.items():
            assert_agrees_to_ulp(
                got=got_solution.values[period][regime],
                expected=values,
                n_ulp=4,
                err_msg=f"period {period}, regime {regime}",
            )


def test_condition_on_two_actions_and_a_value_set_solves_like_its_pin() -> None:
    """A conjunction over two discrete actions, one of them numeric with a set of
    activating values, equals the hand-written pin."""
    _assert_same_values(
        got=_get_shift_model(conditional=True),
        expected=_get_shift_model(conditional=False),
    )


def _array_annotated_utility(
    *, consumption: FloatND, hours: FloatND, labor: IntND, disutility: FloatND
) -> FloatND:
    return jnp.log(consumption) - disutility * (hours / 40.0) - 0.1 * labor


def _array_annotated_pin(*, labor: IntND, hours: FloatND) -> BoolND:
    return (labor == Labor.yes) | (hours == 0.0)


def _array_annotated_next_wealth(
    *, wealth: FloatND, consumption: FloatND, hours: FloatND
) -> FloatND:
    return 1.03 * (wealth - consumption) + 0.1 * hours


def _array_annotated_borrowing(*, consumption: FloatND, wealth: FloatND) -> BoolND:
    return consumption <= wealth


def _phased_model(*, conditional: bool) -> Model:
    """The utility is declared per phase, and every function annotates its arguments
    as plain arrays."""
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    constraints: dict[FunctionName, Callable[..., BoolND]] = {
        "borrowing": _array_annotated_borrowing
    }
    if conditional:
        hours = ConditionalAction(
            grid=HOURS_GRID, active_when={"labor": Labor.yes}, inactive_value=0.0
        )
    else:
        hours = HOURS_GRID
        constraints["hours_only"] = _array_annotated_pin
    working = Regime(
        state_transitions={"wealth": _array_annotated_next_wealth},
        actions={
            "labor": DiscreteGrid(category_class=Labor),
            "hours": hours,
            "consumption": CONSUMPTION_GRID,
        },
        functions={
            "utility": Phased(
                solve=_array_annotated_utility, simulate=_array_annotated_utility
            )
        },
        constraints=constraints,
    )
    dead = Regime(functions={"utility": _bequest})
    return Model(
        regimes={"working": working, "dead": dead},
        states={"wealth": WEALTH_GRID},
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "working"},
        edges={"working": {"working": (0, 1), "dead": 2}},
    )


def test_conditional_action_with_a_phased_utility_solves_like_its_pin() -> None:
    """A regime whose utility is declared per phase accepts the conditional action."""
    _assert_same_values(
        got=_phased_model(conditional=True), expected=_phased_model(conditional=False)
    )


def _model_level_model(*, conditional: bool) -> Model:
    """Labor and hours are declared once on the model; the terminal regime masks
    them."""
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    if conditional:
        hours = ConditionalAction(
            grid=HOURS_GRID, active_when={"labor": Labor.yes}, inactive_value=0.0
        )
        constraints = {}
    else:
        hours = HOURS_GRID
        constraints = {"hours_only": _hours_pin(0.0)}
    working = Regime(
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": CONSUMPTION_GRID},
        functions={"utility": _utility},
        constraints={"borrowing": _borrowing},
    )
    dead = Regime(
        functions={"utility": _bequest},
        actions={"labor": None, "hours": None},
        constraints=dict.fromkeys(constraints),
    )
    return Model(
        regimes={"working": working, "dead": dead},
        states={"wealth": WEALTH_GRID},
        actions={"labor": DiscreteGrid(category_class=Labor), "hours": hours},
        constraints=constraints,
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "working"},
        edges={"working": {"working": (0, 1), "dead": 2}},
    )


def test_model_level_conditional_action_solves_like_its_pin() -> None:
    """A conditional action declared on the model equals a model-level pin."""
    _assert_same_values(
        got=_model_level_model(conditional=True),
        expected=_model_level_model(conditional=False),
    )


def test_inactive_value_matches_a_node_rounded_at_the_working_precision() -> None:
    """At float32 the node of `LinSpacedGrid(0, 1, 11)` nearest 0.9 is not the float
    nearest 0.9; the declared value still names that node."""
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=11)
    hours = ConditionalAction(
        grid=grid, active_when={"labor": Labor.yes}, inactive_value=0.9
    )
    assert hours.inactive_index == 9


@pytest.mark.parametrize(
    ("active_when", "accepted"),
    [({"labor": Labor.yes}, True), ({"labor": Labor.no}, False)],
)
def test_saved_solution_is_accepted_only_by_the_same_declaration(
    *, tmp_path: Path, active_when: dict[ActionName, ScalarInt], accepted: bool
) -> None:
    """A saved solution binds the declaration: changing `active_when` refuses it."""
    solution = _get_model(conditional=True).solve(params=PARAMS, log_level="off")
    loaded = load_solution(path=solution.save(path=tmp_path / "solution.lcm"))
    other = _model_with_hours(
        hours=ConditionalAction(
            grid=HOURS_GRID, active_when=active_when, inactive_value=0.0
        )
    )
    if accepted:
        other.simulate(
            params=PARAMS,
            solution=loaded,
            initial_conditions={
                "wealth": jnp.linspace(1.5, 11.5, N_SUBJECTS),
                "age": jnp.zeros(N_SUBJECTS),
                "regime_id": jnp.full(N_SUBJECTS, RegimeId.working, dtype=jnp.int32),
            },
            log_level="off",
        )
    else:
        with pytest.raises(InvalidSimulationInputError, match="model_fingerprint"):
            other.simulate(
                params=PARAMS,
                solution=loaded,
                initial_conditions={
                    "wealth": jnp.linspace(1.5, 11.5, N_SUBJECTS),
                    "age": jnp.zeros(N_SUBJECTS),
                    "regime_id": jnp.full(
                        N_SUBJECTS, RegimeId.working, dtype=jnp.int32
                    ),
                },
                log_level="off",
            )


def _annotated_utility(*, labor: IntND, hours: FloatND) -> FloatND:
    return 100.0 * (1 - labor) + hours


def _scaled_utility(*, labor: IntND, hours: FloatND, scale: float) -> FloatND:
    return scale * (1 - labor) + hours


def _unannotated_utility(*, labor, hours):
    return 100.0 * (1 - labor) + hours


type _Reader = Literal["annotated", "partial", "unannotated"]
type _Coordinate = Literal["age", "period"]


@dataclass(frozen=True)
class _UtilityAt:
    """Build the utility at an age or period; every age gets the same function."""

    reader: _Reader

    def __call__(self, coordinate: float) -> Callable[..., FloatND]:  # noqa: ARG002
        if self.reader == "partial":
            return functools.partial(_scaled_utility, scale=100.0)
        if self.reader == "unannotated":
            return _unannotated_utility
        return _annotated_utility


def _array_annotated_hours_pin(*, labor: IntND, hours: FloatND) -> BoolND:
    return (labor == Labor.yes) | (hours == 0.0)


def _unannotated_hours_pin(*, labor, hours):
    return (labor == Labor.yes) | (hours == 0.0)


def _age_specialized_model(
    *, conditional: bool, model_level: bool, reader: _Reader, coordinate: _Coordinate
) -> Model:
    """The utility is the only reader of labor and hours and is built per age or
    period. With `model_level`, labor and hours are declared on the model, so
    broadcast pruning sees them only through their readers."""
    grid = IrregSpacedGrid(points=(0.0, 1.0, 2.0))
    hours: ContinuousGrid = (
        ConditionalAction(
            grid=grid, active_when={"labor": Labor.yes}, inactive_value=0.0
        )
        if conditional
        else grid
    )
    actions = {"labor": DiscreteGrid(category_class=Labor), "hours": hours}
    pin = (
        _unannotated_hours_pin
        if reader == "unannotated"
        else _array_annotated_hours_pin
    )
    constraints = {} if conditional else {"hours_only": pin}
    marker_class = (
        PeriodSpecializedFunction if coordinate == "period" else AgeSpecializedFunction
    )
    working = Regime(
        actions={} if model_level else actions,
        functions={
            "utility": marker_class(
                build=_UtilityAt(reader=reader), signature=_same_key
            )
        },
        constraints={} if model_level else constraints,
    )
    dead = Regime(
        functions={"utility": _zero},
        actions={"labor": None, "hours": None} if model_level else {},
        constraints=dict.fromkeys(constraints) if model_level else {},
    )
    regimes = {"working": working, "dead": dead}
    model_actions = actions if model_level else {}
    model_constraints = constraints if model_level else {}
    if coordinate == "period":
        return Model(
            regimes=regimes,
            actions=model_actions,
            constraints=model_constraints,
            regime_id_class=RegimeId,
            n_periods=2,
            initial_nodes=InitialNodes(by_period={0: "working"}),
            edges={"working": {"dead": Periods(values=(0,))}},
        )
    ages = AgeGrid(start=0, inclusive_stop=1, step="Y")
    return Model(
        regimes=regimes,
        actions=model_actions,
        constraints=model_constraints,
        regime_id_class=RegimeId,
        ages=ages,
        initial_nodes={ages.exact_values[0]: "working"},
        edges={"working": {"dead": 0}},
    )


def _same_key(coordinate: float) -> int:  # noqa: ARG001
    return 0


def _zero() -> FloatND:
    return jnp.asarray(0.0)


_AGE_PARAMS = {"discount_factor": 0.95, "working": {}, "dead": {}}


def _age_specialized_outputs(
    *, conditional: bool, model_level: bool, reader: _Reader, coordinate: _Coordinate
) -> tuple[FloatND, dict[ActionName, np.ndarray], pd.DataFrame]:
    model = _age_specialized_model(
        conditional=conditional,
        model_level=model_level,
        reader=reader,
        coordinate=coordinate,
    )
    solution = model.solve(params=_AGE_PARAMS, log_level="off")
    lookup = model.lookup_policy(
        params=_AGE_PARAMS,
        solution=solution,
        period=0,
        regime_name="working",
        states={},
    )
    assert lookup.actions is not None
    time_column = "period" if coordinate == "period" else "age"
    frame = model.simulate(
        params=_AGE_PARAMS,
        solution=solution,
        initial_conditions={
            time_column: jnp.zeros(2, dtype=jnp.int32)
            if coordinate == "period"
            else jnp.zeros(2),
            "regime_id": jnp.full(2, RegimeId.working, dtype=jnp.int32),
        },
        log_level="off",
    ).to_dataframe()
    panel = frame.loc[
        frame["regime_name"] == "working", ["subject_id", "period", "labor", "hours"]
    ].reset_index(drop=True)
    actions = {name: np.asarray(value) for name, value in lookup.actions.items()}
    return solution.values[0]["working"], actions, panel


@pytest.mark.parametrize("coordinate", ["age", "period"])
@pytest.mark.parametrize("reader", ["annotated", "partial", "unannotated"])
@pytest.mark.parametrize("model_level", [False, True])
def test_age_specialized_reader_solves_looks_up_and_simulates_like_the_pin(
    *, model_level: bool, reader: _Reader, coordinate: _Coordinate
) -> None:
    """When the only reader of the actions is built per age or period, whether it
    annotates them as plain arrays, is a partial or is unannotated, the declaration
    matches the hand-written constraint in the value function, `lookup_policy` and
    the simulated panel."""
    values, actions, panel = _age_specialized_outputs(
        conditional=True, model_level=model_level, reader=reader, coordinate=coordinate
    )
    expected_values, expected_actions, expected_panel = _age_specialized_outputs(
        conditional=False, model_level=model_level, reader=reader, coordinate=coordinate
    )
    assert_agrees_to_ulp(got=values, expected=expected_values, n_ulp=4)
    for name, expected in expected_actions.items():
        np.testing.assert_array_equal(actions[name], expected, err_msg=name)
    pd.testing.assert_frame_equal(panel, expected_panel)


def test_lowering_module_imports_without_beartype_warnings() -> None:
    """The beartype claw decorates every function of the lowering module."""
    code = (
        "import warnings; warnings.simplefilter('error');"
        "import lcm, _lcm.regime_building.conditional_actions"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@categorical(ordered=False)
class _VisitRegimeId:
    origin: ScalarInt
    working: ScalarInt
    dead: ScalarInt


def _period_utility(period: int) -> Callable[..., FloatND]:
    def utility(*, labor: IntND, hours: FloatND) -> FloatND:
        return (100.0 + period) * (1 - labor) + hours

    return utility


def _visited_only_utility(period: int) -> Callable[..., FloatND]:
    """Build the simulated utility, which exists only where subjects are."""
    if period != 2:
        msg = f"No simulated utility at period {period}; subjects arrive at 2."
        raise ValueError(msg)
    return _period_utility(period)


def _visit_model() -> Model:
    """Subjects start in origin at period 0 and in working at period 2. The solve
    moves origin to working, the simulation moves it to dead, so working is solved
    at periods 1 and 2 but visited only at period 2."""
    working = Regime(
        actions={
            "labor": DiscreteGrid(category_class=Labor),
            "hours": ConditionalAction(
                grid=IrregSpacedGrid(points=(0.0, 1.0, 2.0)),
                active_when={"labor": Labor.yes},
                inactive_value=0.0,
            ),
        },
        functions={
            "utility": Phased(
                solve=PeriodSpecializedFunction(build=_period_utility, signature=int),
                simulate=PeriodSpecializedFunction(
                    build=_visited_only_utility, signature=int
                ),
            )
        },
    )
    return Model(
        regimes={
            "origin": Regime(functions={"utility": _zero}),
            "working": working,
            "dead": Regime(functions={"utility": _zero}),
        },
        regime_id_class=_VisitRegimeId,
        n_periods=4,
        initial_nodes=InitialNodes(by_period={0: "origin", 2: "working"}),
        edges=Phased(
            solve={
                "origin": {"working": Periods(values=(0,))},
                "working": {"dead": Periods(values=(1, 2))},
            },
            simulate={
                "origin": {"dead": Periods(values=(0,))},
                "working": {"dead": Periods(values=(1, 2))},
            },
        ),
    )


@pytest.mark.parametrize(("period", "expected"), [(1, 101.0), (2, 102.0)])
def test_simulated_reader_is_bound_where_subjects_visit(
    *, period: int, expected: float
) -> None:
    """A simulated utility built only at visited periods does not block the model,
    and working is worth `100 + period + inactive hours` with labor no."""
    solution = _visit_model().solve(
        params={"discount_factor": 0.95, "origin": {}, "working": {}, "dead": {}},
        log_level="off",
    )
    np.testing.assert_allclose(
        np.asarray(solution.values[period]["working"]), expected, rtol=1e-6
    )
