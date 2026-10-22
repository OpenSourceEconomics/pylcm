"""User functions read the numbers of a `DiscreteGrid(values=...)`.

The reference is the fully discrete deterministic model of
`tests.test_models.deterministic.discrete`, which works on category codes and
converts them by hand. The value-grid twin declares the numbers instead:

- `consumption` takes the float values `1.0, 2.0`; the reference reads level
  `1 + code` in utility and spends the code in the budget.
- `wealth` takes the values `10, 11, 12`; the twin shifts by 10 where the
  reference uses the code.

Every shift is exact, so both models evaluate the same floats at either
precision.
"""

import math
from fractions import Fraction
from functools import partial, wraps
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from _lcm.grids.discrete import MISSING_CAT_CODE, read_at_codes
from _lcm.regime_building.finalize import _authored
from _lcm.regime_building.V import _LookupWrapper
from _lcm.regime_building.value_adapter import (
    adapt_entry,
    adapt_user_function,
    collect_value_grids,
)
from _lcm.simulation.transitions import _count_drawn_off_grid_codes
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    CollectiveUtility,
    Decoupled,
    DeterministicTransition,
    DiscreteGrid,
    Gate,
    InitialNodes,
    JointTransition,
    LinSpacedGrid,
    Model,
    ParetoObjective,
    Periods,
    Phased,
    ProjectedRegimeValue,
    StakeholderRoute,
    Transition,
    case_boundary,
    categorical,
    fixed_transition,
    piece,
    ref,
)
from lcm.collective import ValueDependentConstraint
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    post_decision_lower_bound,
)
from lcm.exceptions import InvalidStateTransitionError, RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.regime import StochasticTransition
from lcm.solver_api import SolutionResult
from lcm.solvers import DCEGM
from lcm.transition import AgeSpecializedFunction
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Int1D,
    ScalarInt,
    UserFunction,
    UserParams,
)
from lcm_examples.mortality import (
    is_working,
    labor_income,
    next_wealth,
    utility_working,
)
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.deterministic import discrete
from tests.test_models.deterministic.regression import LaborSupply

_N_PERIODS = 4
_WEALTH_OFFSET = 10


def _utility(
    *, consumption: DiscreteAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    return utility_working(
        consumption=consumption,
        is_working=is_working,
        disutility_of_work=disutility_of_work,
    )


def _next_wealth(
    *,
    wealth: DiscreteState,
    consumption: DiscreteAction,
    labor_income: FloatND,
    interest_rate: float,
) -> DiscreteState:
    continuous = next_wealth(
        wealth=wealth - _WEALTH_OFFSET,
        consumption=consumption - 1.0,
        labor_income=labor_income,
        interest_rate=interest_rate,
    )
    return jnp.clip(jnp.rint(continuous), 0, 2).astype(jnp.int32) + _WEALTH_OFFSET


def _borrowing_constraint(
    *, consumption: DiscreteAction, wealth: DiscreteState
) -> BoolND:
    return consumption - 1.0 <= wealth - _WEALTH_OFFSET


def _value_grid_model(*, enable_jit: bool) -> Model:
    working_life = UserRegime(
        actions={
            "labor_supply": DiscreteGrid(category_class=LaborSupply),
            "consumption": DiscreteGrid(values=np.array([1.0, 2.0])),
        },
        states={
            "wealth": DiscreteGrid(
                values=np.array(
                    [_WEALTH_OFFSET, _WEALTH_OFFSET + 1, _WEALTH_OFFSET + 2]
                )
            )
        },
        state_transitions={"wealth": _next_wealth},
        constraints={"borrowing_constraint": _borrowing_constraint},
        functions={
            "utility": _utility,
            "labor_income": labor_income,
            "is_working": is_working,
        },
    )
    ages = AgeGrid(start=50, inclusive_stop=50 + (_N_PERIODS - 1) * 10, step="10Y")
    return Model(
        regimes={"working_life": working_life, "dead": discrete.dead},
        ages=ages,
        regime_id_class=discrete.RegimeId,
        initial_nodes={ages.exact_values[0]: "working_life"},
        edges=discrete.working_life_edges(ages),
        enable_jit=enable_jit,
    )


def _reference_model(*, enable_jit: bool) -> Model:
    if enable_jit:
        return discrete.get_model(n_periods=_N_PERIODS)
    ages = AgeGrid(start=50, inclusive_stop=50 + (_N_PERIODS - 1) * 10, step="10Y")
    return Model(
        regimes={"working_life": discrete.working_life, "dead": discrete.dead},
        ages=ages,
        regime_id_class=discrete.RegimeId,
        initial_nodes={ages.exact_values[0]: "working_life"},
        edges=discrete.working_life_edges(ages),
        enable_jit=False,
    )


_PARAMS = discrete.get_params(n_periods=_N_PERIODS, wage=1.5, interest_rate=0.0)

_INITIAL_CONDITIONS = {
    "wealth": jnp.array([0, 1, 2, 0], dtype=jnp.int32),
    "age": jnp.array([50.0, 50.0, 50.0, 50.0]),
    "regime_id": jnp.array([discrete.RegimeId.working_life] * 4),
}


@pytest.mark.parametrize("enable_jit", [True, False])
def test_value_grid_model_solves_to_the_reference_values_bitwise(enable_jit):
    """The value-grid twin and the hand-decoded reference give identical V."""
    got = (
        _value_grid_model(enable_jit=enable_jit)
        .solve(params=_PARAMS, log_level="off")
        .values
    )
    expected = (
        _reference_model(enable_jit=enable_jit)
        .solve(params=_PARAMS, log_level="off")
        .values
    )
    pairs = [
        (np.asarray(got[period][regime]), np.asarray(expected[period][regime]))
        for period in expected
        for regime in expected[period]
    ]
    assert len(pairs) == sum(len(regimes) for regimes in got.values())
    for got_values, expected_values in pairs:
        np.testing.assert_array_equal(got_values, expected_values)


def _simulated_codes(model: Model) -> pd.DataFrame:
    frame = model.simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        seed=0,
        log_level="off",
        use_categorical_codes=True,
    ).to_dataframe(use_categorical_codes=True)
    return frame.sort_values(["subject_id", "period"]).reset_index(drop=True)


def test_value_grid_model_simulates_the_reference_choices():
    """Both models simulate the same codes for every subject and period."""
    assert_frame_equal(
        _simulated_codes(_value_grid_model(enable_jit=True)),
        _simulated_codes(_reference_model(enable_jit=True)),
    )


def test_value_grid_model_fingerprint_is_reproducible():
    """Two builds of one value-grid model share their structure fingerprint."""
    assert (
        _value_grid_model(enable_jit=True)._model_structure_fingerprint
        == _value_grid_model(enable_jit=True)._model_structure_fingerprint
    )


def test_value_grid_tolerance_enters_the_model_fingerprint():
    """A model whose value grid declares another tolerance has another identity."""
    model = _value_grid_model(enable_jit=True)
    regime = model.user_regimes["working_life"]
    tolerant = regime.replace(
        actions={
            **dict(regime.actions),
            "consumption": DiscreteGrid(values=np.array([1.0, 2.0]), tolerance=1e-6),
        }
    )
    ages = AgeGrid(start=50, inclusive_stop=50 + (_N_PERIODS - 1) * 10, step="10Y")
    other = Model(
        regimes={"working_life": tolerant, "dead": discrete.dead},
        ages=ages,
        regime_id_class=discrete.RegimeId,
        initial_nodes={ages.exact_values[0]: "working_life"},
        edges=discrete.working_life_edges(ages),
    )
    assert other._model_structure_fingerprint != model._model_structure_fingerprint


_WEALTH = DiscreteGrid(values=np.array([10, 11, 12]))
_VALUE_GRIDS = MappingProxyType({"wealth": _WEALTH})


def _identity_wealth(*, wealth: DiscreteState) -> DiscreteState:
    return wealth


def _next_wealth_plus(*, wealth: DiscreteState, step: int) -> DiscreteState:
    return wealth + step


def _reads_next_wealth(*, next_wealth: DiscreteState) -> DiscreteState:
    return next_wealth


def _wealth_probabilities(*, wealth: DiscreteState) -> FloatND:
    share = (wealth - 10) / 2
    return jnp.stack([1 - share, share])


def test_adapted_function_reads_the_value_of_a_code():
    """A user function receives the value its code indexes."""
    adapted = adapt_user_function(
        func=_identity_wealth, value_grids=_VALUE_GRIDS, encode=None
    )
    assert jnp.asarray(adapted(wealth=jnp.int32(1))).tolist() == 11


def test_adapted_function_reads_the_value_of_a_next_state():
    """An argument `next_<state>` reads the next state's value."""
    adapted = adapt_user_function(
        func=_reads_next_wealth, value_grids=_VALUE_GRIDS, encode=None
    )
    assert jnp.asarray(adapted(next_wealth=jnp.int32(2))).tolist() == 12


@pytest.mark.parametrize(("step", "expected"), [(1, 2), (3, MISSING_CAT_CODE)])
def test_adapted_law_returns_the_code_of_its_value(*, step, expected):
    """A law's value becomes the code of its node, or the sentinel off the grid."""
    adapted = adapt_user_function(
        func=_next_wealth_plus, value_grids=_VALUE_GRIDS, encode=_WEALTH
    )
    code = adapted(wealth=jnp.int32(1), step=jnp.int32(step))
    assert jnp.asarray(code).tolist() == expected


def test_adapted_function_publishes_code_annotations():
    """Decoded arguments and encoded returns are annotated as codes."""
    adapted = adapt_user_function(
        func=_next_wealth_plus, value_grids=_VALUE_GRIDS, encode=_WEALTH
    )
    published = adapted.__annotations__
    assert (published["wealth"], published["return"]) == (
        "DiscreteState",
        "DiscreteState",
    )


def test_function_reading_no_value_grid_is_left_alone():
    """A function that reads no value grid is returned unchanged."""
    adapted = adapt_user_function(
        func=_reads_next_wealth, value_grids=MappingProxyType({}), encode=None
    )
    assert adapted is _reads_next_wealth


def test_stochastic_law_reads_values_and_keeps_its_probabilities():
    """A stochastic law reads values and returns probabilities over codes as is."""
    adapted = adapt_entry(
        entry=StochasticTransition(func=_wealth_probabilities),
        value_grids=_VALUE_GRIDS,
        encode=_WEALTH,
    )
    assert isinstance(adapted, StochasticTransition)
    probabilities = adapted.func(wealth=jnp.int32(1))
    aaae(probabilities, [0.5, 0.5], decimal=DECIMAL_PRECISION)


def test_model_refuses_one_name_with_different_values():
    """A variable declared with values must carry the same values everywhere."""
    with pytest.raises(RegimeInitializationError, match="same values"):
        collect_value_grids(
            regimes={
                "a": UserRegime(
                    states={"wealth": _WEALTH},
                    state_transitions={"wealth": _identity_wealth},
                    functions={"utility": lambda: 0.0},
                ),
                "b": UserRegime(
                    states={"wealth": DiscreteGrid(values=np.array([0, 1, 2]))},
                    state_transitions={"wealth": _identity_wealth},
                    functions={"utility": lambda: 0.0},
                ),
            },
            derived_categoricals={},
        )


@pytest.mark.parametrize(
    "law",
    [
        pytest.param(fixed_transition("bracket"), id="bare"),
        pytest.param(
            Phased(
                solve=fixed_transition("bracket"), simulate=fixed_transition("bracket")
            ),
            id="phased",
        ),
    ],
)
def test_fixed_value_grid_state_repeats_the_reference_per_value(law):
    """A fixed value-grid state the economics ignores repeats the reference V."""
    model = _value_grid_model(enable_jit=True)
    regime = model.user_regimes["working_life"]
    with_bracket = regime.replace(
        states={
            **dict(regime.states),
            "bracket": DiscreteGrid(values=np.array([5, 7])),
        },
        state_transitions={**dict(regime.state_transitions), "bracket": law},
        functions={**dict(regime.functions), "utility": _utility_with_bracket},
    )
    got = _model_with(working_life=with_bracket).solve(params=_PARAMS, log_level="off")
    expected = _reference_model(enable_jit=True).solve(params=_PARAMS, log_level="off")
    for period in (0, 1, 2):
        reference = np.asarray(expected.values[period]["working_life"])
        for bracket_code in (0, 1):
            np.testing.assert_array_equal(
                np.asarray(got.values[period]["working_life"])[..., bracket_code],
                reference,
            )


def _utility_with_bracket(
    *,
    consumption: DiscreteAction,
    is_working: BoolND,
    disutility_of_work: float,
    bracket: DiscreteState,
) -> FloatND:
    return (
        _utility(
            consumption=consumption,
            is_working=is_working,
            disutility_of_work=disutility_of_work,
        )
        + 0.0 * bracket
    )


def test_an_age_specialized_function_beside_a_value_grid_solves_as_unmarked():
    """An age-specialized function reads the values of a value grid."""
    model = _value_grid_model(enable_jit=True)
    regime = model.user_regimes["working_life"]
    specialized = regime.replace(
        functions={
            **dict(regime.functions),
            "is_working": AgeSpecializedFunction(
                build=lambda _age: is_working, signature=lambda _age: 0
            ),
        }
    )
    got, expected = (
        candidate.solve(params=_PARAMS, log_level="off").values
        for candidate in (_model_with(working_life=specialized), model)
    )
    pairs = [
        (np.asarray(got[period][regime]), np.asarray(expected[period][regime]))
        for period in expected
        for regime in expected[period]
    ]
    assert len(pairs) == sum(len(regimes) for regimes in got.values())
    for got_values, expected_values in pairs:
        np.testing.assert_array_equal(got_values, expected_values)


def _model_with(*, working_life: UserRegime) -> Model:
    ages = AgeGrid(start=50, inclusive_stop=50 + (_N_PERIODS - 1) * 10, step="10Y")
    return Model(
        regimes={"working_life": working_life, "dead": discrete.dead},
        ages=ages,
        regime_id_class=discrete.RegimeId,
        initial_nodes={ages.exact_values[0]: "working_life"},
        edges=discrete.working_life_edges(ages),
    )


_V = jnp.asarray([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]])


@pytest.mark.parametrize("code", [-1, 3, MISSING_CAT_CODE])
def test_value_function_read_at_an_off_grid_code_is_nan(code):
    """A code off the axis reads NaN, never a clamped or wrapped node."""
    lookup = _LookupWrapper(
        array_name="V",
        axis_names=("wealth", "health"),
        retained_axis_names=frozenset({"health"}),
    )
    assert bool(jnp.isnan(lookup(wealth=jnp.int32(code), V=_V)).all())


def test_value_function_read_at_a_code_returns_its_row():
    """A code on the axis reads its row."""
    lookup = _LookupWrapper(
        array_name="V",
        axis_names=("wealth", "health"),
        retained_axis_names=frozenset({"health"}),
    )
    assert lookup(wealth=jnp.int32(1), V=_V).tolist() == [10.0, 11.0]


@pytest.mark.parametrize("code", [-1, 3, MISSING_CAT_CODE])
def test_read_at_codes_off_the_axis_is_nan(code):
    """The shared code read gives NaN for a code off the axis."""
    read = read_at_codes(array=_V, positions=(slice(None), jnp.int32(code)))
    assert bool(jnp.isnan(read).all())


def _next_wealth_up(*, wealth: DiscreteState) -> DiscreteState:
    return wealth + 1


def _next_wealth_spent(
    *, wealth: DiscreteState, consumption: DiscreteAction
) -> DiscreteState:
    return wealth - (consumption - 1.0).astype(jnp.int32)


def _next_wealth_spent_clipped(
    *, wealth: DiscreteState, consumption: DiscreteAction
) -> DiscreteState:
    return jnp.maximum(_next_wealth_spent(wealth=wealth, consumption=consumption), 10)


# `_PARAMS` without the interest rate, which the laws below do not read.
_PARAMS_WITHOUT_RATE: UserParams = {
    "discount_factor": 0.95,
    "working_life": {
        "utility": {"disutility_of_work": 0.5},
        "labor_income": {"wage": 1.5},
    },
    "final_age_alive": 50 + (_N_PERIODS - 2) * 10,
}


def _model_with_law(law: UserFunction) -> Model:
    regime = _value_grid_model(enable_jit=True).user_regimes["working_life"]
    return _model_with(
        working_life=regime.replace(
            state_transitions={**dict(regime.state_transitions), "wealth": law}
        )
    )


def test_off_grid_law_at_a_feasible_cell_fails_the_solve():
    """With checks on, a law leaving the grid where a choice is feasible raises."""
    with pytest.raises(InvalidStateTransitionError, match="next_wealth") as error:
        _model_with_law(_next_wealth_up).solve(
            params=_PARAMS_WITHOUT_RATE, log_level="off", runtime_checks=True
        )
    assert "'wealth': 12" in str(error.value)


def test_off_grid_law_at_a_feasible_cell_fails_validate_transitions():
    """`validate_transitions` reports an off-grid deterministic law."""
    with pytest.raises(InvalidStateTransitionError, match="next_wealth"):
        _model_with_law(_next_wealth_up).validate_transitions(
            params=_PARAMS_WITHOUT_RATE
        )


def test_off_grid_law_without_checks_reads_nan_where_it_leaves_the_grid():
    """With checks off, exactly the cells whose path leaves the grid are NaN.

    `wealth + 1` leaves `[10, 11, 12]` from 12. At period 1 only wealth 12 reads
    off the grid; at period 0, wealth 11 lands on 12 and so reads period 1's NaN.
    Period 2 moves only to the terminal regime.
    """
    values = (
        _model_with_law(_next_wealth_up)
        .solve(params=_PARAMS_WITHOUT_RATE, log_level="off", runtime_checks=False)
        .values
    )
    nan_pattern = [
        np.isnan(np.asarray(values[period]["working_life"])).tolist()
        for period in (0, 1, 2)
    ]
    assert nan_pattern == [
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ]


def test_off_grid_law_only_at_infeasible_cells_solves_like_the_clipped_law():
    """Leaving the grid only where no choice is feasible changes nothing."""
    got = _model_with_law(_next_wealth_spent).solve(
        params=_PARAMS_WITHOUT_RATE, log_level="off", runtime_checks=True
    )
    expected = _model_with_law(_next_wealth_spent_clipped).solve(
        params=_PARAMS_WITHOUT_RATE, log_level="off", runtime_checks=True
    )
    for period in expected.values:
        for regime in expected.values[period]:
            np.testing.assert_array_equal(
                np.asarray(got.values[period][regime]),
                np.asarray(expected.values[period][regime]),
            )


def test_adapting_an_adapted_function_wraps_the_user_function_once():
    """A finalized regime reused in another model is not decoded twice."""
    once = adapt_user_function(
        func=_identity_wealth, value_grids=_VALUE_GRIDS, encode=None
    )
    twice = adapt_user_function(func=once, value_grids=_VALUE_GRIDS, encode=None)
    assert jnp.asarray(twice(wealth=jnp.int32(1))).tolist() == 11


def _next_room(*, wealth: DiscreteState) -> DiscreteState:
    return (wealth < 12).astype(jnp.int32)


def _next_wealth_by_room(
    *, wealth: DiscreteState, next_room: DiscreteState
) -> DiscreteState:
    return wealth + next_room


def _utility_with_room(
    *,
    consumption: DiscreteAction,
    is_working: BoolND,
    disutility_of_work: float,
    room: DiscreteState,
) -> FloatND:
    return (
        _utility(
            consumption=consumption,
            is_working=is_working,
            disutility_of_work=disutility_of_work,
        )
        + 0.0 * room
    )


def test_law_reading_a_sibling_next_state_is_checked_as_the_solve_composes_it():
    """A law reading another law's next state sees only the values it produces.

    `next_wealth = wealth + next_room` stays on `[10, 11, 12]` because
    `next_room` is 0 at wealth 12; sweeping `next_room` over its whole grid
    would wrongly reach 13.
    """
    regime = _value_grid_model(enable_jit=True).user_regimes["working_life"]
    with_room = regime.replace(
        states={**dict(regime.states), "room": DiscreteGrid(values=np.array([0, 1]))},
        state_transitions={"wealth": _next_wealth_by_room, "room": _next_room},
        functions={**dict(regime.functions), "utility": _utility_with_room},
    )
    _model_with(working_life=with_room).validate_transitions(
        params=_PARAMS_WITHOUT_RATE
    )


@categorical(ordered=False)
class _WorkingDead:
    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _HoursCodes:
    low: ScalarInt
    high: ScalarInt


_NODE_PAIRS = [
    pytest.param((10, 20), id="integers-10-20"),
    pytest.param((5, 25), id="integers-5-25"),
    pytest.param((2.5, 7.5), id="floats-2_5-7_5"),
]


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0, dtype=float)


def _one_period_model(
    *, working: UserRegime, enable_jit: bool, model_aggregator=None, n_periods: int = 1
) -> Model:
    """A working regime that moves to a terminal regime after `n_periods`."""
    targets = {"dead": n_periods - 1}
    if n_periods > 1:
        targets["working"] = tuple(range(n_periods - 1))
    options = (
        {} if model_aggregator is None else {"koopmans_aggregator": model_aggregator}
    )
    return Model(
        regimes={
            "working": working,
            "dead": UserRegime(functions={"utility": _zero_utility}),
        },
        regime_id_class=_WorkingDead,
        ages=AgeGrid(start=0, inclusive_stop=n_periods, step="Y"),
        initial_nodes={0: "working"},
        edges={"working": targets},
        enable_jit=enable_jit,
        **options,
    )


def _first_values(*, model: Model, params) -> np.ndarray:
    return np.asarray(model.solve(params=params, log_level="off").values[0]["working"])


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("location", ["regime", "model", "phased"])
@pytest.mark.parametrize("nodes", _NODE_PAIRS)
def test_koopmans_aggregator_reads_the_value_of_a_derived_categorical(
    *, enable_jit, location, nodes
):
    """W reading a value-grid derived output receives its value, not its code."""

    def bonus() -> DiscreteState:
        return jnp.asarray(nodes[1])

    def aggregate(*, utility: FloatND, CE: FloatND, bonus: DiscreteState) -> FloatND:
        return utility + bonus + 0.0 * CE

    declaration = (
        Phased(solve=aggregate, simulate=aggregate)
        if location == "phased"
        else aggregate
    )
    working = UserRegime(
        functions={"utility": _zero_utility, "bonus": bonus},
        derived_categoricals={"bonus": DiscreteGrid(values=np.array(nodes))},
        koopmans_aggregator=None if location == "model" else declaration,
    )
    model = _one_period_model(
        working=working,
        enable_jit=enable_jit,
        model_aggregator=declaration if location == "model" else None,
    )
    assert _first_values(model=model, params={}).item() == nodes[1]


def _utility_of_hours(*, hours: DiscreteAction) -> FloatND:
    return jnp.asarray(hours, dtype=float)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("nodes", _NODE_PAIRS)
@pytest.mark.parametrize("phased", [False, True])
def test_decoupled_objective_selects_by_the_values_of_an_action(
    *, enable_jit, nodes, phased
):
    """An objective on hours values picks the lower node; the value reports it."""

    def select(*, utility: FloatND, CE: FloatND, hours: DiscreteAction) -> FloatND:
        return -((hours - nodes[0]) ** 2) + 0.0 * utility + 0.0 * CE

    def report_hours(
        *, utility: FloatND, CE: FloatND, hours: DiscreteAction
    ) -> FloatND:
        return jnp.asarray(hours, dtype=float) + 0.0 * utility + 0.0 * CE

    working = UserRegime(
        actions={"hours": DiscreteGrid(values=np.array(nodes))},
        functions={"utility": _utility_of_hours},
        koopmans_aggregator=Decoupled(
            objective=Phased(solve=select, simulate=select) if phased else select,
            value=report_hours,
        ),
    )
    model = _one_period_model(working=working, enable_jit=enable_jit)
    assert _first_values(model=model, params={}).item() == nodes[0]


@pytest.mark.parametrize("nodes", _NODE_PAIRS)
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("explicit_phases", [False, True])
def test_one_function_in_both_phases_keeps_the_feasible_set(
    *, nodes, enable_jit, explicit_phases
):
    """The same authored helper in both phases is one function, not a varying one."""

    def hours_as_float(*, hours: DiscreteState) -> FloatND:
        return jnp.asarray(hours, dtype=float)

    def utility(*, h: FloatND) -> FloatND:
        return h

    def positive(*, h: FloatND) -> BoolND:
        return h > 0.0

    working = UserRegime(
        states={"hours": DiscreteGrid(values=np.array(nodes))},
        state_transitions={"hours": fixed_transition("hours")},
        functions={
            "h": (
                Phased(solve=hours_as_float, simulate=hours_as_float)
                if explicit_phases
                else hours_as_float
            ),
            "utility": utility,
        },
        constraints={"positive": positive},
    )
    got = _first_values(
        model=_one_period_model(working=working, enable_jit=enable_jit),
        params={"discount_factor": 1.0},
    )
    np.testing.assert_array_equal(got, np.asarray(nodes, dtype=float))


def _case_regime(*, nodes, value_side: str) -> UserRegime:
    boundary = case_boundary(condition=ref("wealth") < 1.5, kind="jump")

    def from_hours(*, hours: DiscreteState) -> FloatND:
        return jnp.asarray(hours, dtype=float)

    def constant() -> FloatND:
        return jnp.asarray(7.0, dtype=float)

    def utility(*, income: FloatND, wealth: ContinuousState) -> FloatND:
        return income + 0.0 * wealth

    when_func, otherwise_func = (
        (from_hours, constant) if value_side == "when" else (constant, from_hours)
    )
    return UserRegime(
        states={
            "wealth": LinSpacedGrid(start=1, stop=2, n_points=2),
            "hours": DiscreteGrid(values=np.array(nodes)),
        },
        state_transitions={
            "wealth": fixed_transition("wealth"),
            "hours": fixed_transition("hours"),
        },
        functions={
            "eligible": boundary,
            "income_below": piece(output="income", when=boundary)(when_func),
            "income_above": piece(output="income", otherwise=boundary)(otherwise_func),
            "utility": utility,
        },
    )


@pytest.mark.parametrize("nodes", _NODE_PAIRS)
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("value_side", ["when", "otherwise"])
def test_case_pieces_survive_three_reuses_of_a_finalized_regime(
    *, nodes, enable_jit, value_side
):
    """A value-reading case piece keeps its role and its values across reuses."""
    model = _one_period_model(
        working=_case_regime(nodes=nodes, value_side=value_side),
        enable_jit=enable_jit,
    )
    first = _first_values(model=model, params={"discount_factor": 1.0})
    np.testing.assert_array_equal(
        np.sort(first.ravel()), np.sort(np.asarray([*nodes, 7.0, 7.0], dtype=float))
    )
    for _ in range(3):
        model = _one_period_model(
            working=model.user_regimes["working"], enable_jit=enable_jit
        )
        np.testing.assert_array_equal(
            _first_values(model=model, params={"discount_factor": 1.0}), first
        )


def _bonus_regime(kind: str) -> UserRegime:
    def integer_bonus() -> FloatND:
        return jnp.int32(20)

    def float_bonus() -> FloatND:
        return jnp.asarray(20.0)

    def utility(*, bonus: FloatND) -> FloatND:
        return bonus + 0.0

    return UserRegime(
        functions={
            "bonus": integer_bonus if kind == "integer" else float_bonus,
            "utility": utility,
        },
        derived_categoricals={
            "bonus": DiscreteGrid(
                values=np.array([10, 20] if kind == "integer" else [10.0, 20.0])
            )
        },
    )


def _add_continuation(*, utility: FloatND, CE: FloatND) -> FloatND:
    return utility + CE


@categorical(ordered=False)
class _EarlyLate:
    early: ScalarInt
    late: ScalarInt


def _bonus_model(*, order, early_kind, late_kind) -> Model:
    declared = {"early": _bonus_regime(early_kind), "late": _bonus_regime(late_kind)}
    ages = AgeGrid(start=0, inclusive_stop=1, step="Y")
    return Model(
        regimes={name: declared[name] for name in order},
        ages=ages,
        regime_id_class=_EarlyLate,
        initial_nodes={ages.exact_values[0]: "early"},
        edges={"early": {"late": AgeRange(exclusive_stop=1)}},
        koopmans_aggregator=_add_continuation,
        enable_jit=False,
    )


@pytest.mark.parametrize("order", [("early", "late"), ("late", "early")])
def test_one_derived_name_with_integer_and_float_values_is_refused(order):
    """One name with integer values in one regime and floats in another is refused."""
    with pytest.raises(RegimeInitializationError, match="bonus"):
        _bonus_model(order=order, early_kind="integer", late_kind="float")


@pytest.mark.parametrize("kind", ["integer", "float"])
@pytest.mark.parametrize("order", [("early", "late"), ("late", "early")])
def test_one_derived_name_with_one_kind_of_values_solves_in_either_order(
    *, kind, order
):
    """The same derived grid in two regimes gives the declared values 40 and 20."""
    model = _bonus_model(order=order, early_kind=kind, late_kind=kind)
    values = model.solve(params={}, log_level="off").values
    assert (
        np.asarray(values[0]["early"]).item(),
        np.asarray(values[1]["late"]).item(),
    ) == (40.0, 20.0)


def _earnings_utility(*, earnings: FloatND) -> FloatND:
    return earnings


def _hours_as_float(*, hours: DiscreteState) -> FloatND:
    return jnp.asarray(hours, dtype=float)


def _impute_twenty() -> DiscreteState:
    return jnp.int32(20)


def _impute_high_code() -> DiscreteState:
    return jnp.int32(1)


def _hours_of_code_as_float(*, hours: DiscreteState) -> FloatND:
    return jnp.asarray(jnp.asarray([10.0, 20.0])[hours], dtype=float)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_an_imputed_value_grid_state_matches_its_twin_on_codes(*, enable_jit):
    """An imputation returning a value lands on that value's code."""
    got, expected = (
        _one_period_model(
            working=UserRegime(
                states={
                    "wealth": LinSpacedGrid(start=0, stop=30, n_points=4),
                    "hours": Phased(
                        solve=_impute_twenty if values else _impute_high_code,
                        simulate=DiscreteGrid(values=np.array([10, 20]))
                        if values
                        else DiscreteGrid(category_class=_HoursCodes),
                    ),
                },
                state_transitions={
                    "wealth": _hours_as_float if values else _hours_of_code_as_float,
                    "hours": fixed_transition("hours"),
                },
                functions={"utility": _wealth_utility},
            ),
            enable_jit=enable_jit,
            n_periods=2,
        )
        .solve(params={"discount_factor": 1.0}, log_level="off")
        .values
        for values in (True, False)
    )
    pairs = [
        (np.asarray(got[period][regime]), np.asarray(expected[period][regime]))
        for period in expected
        for regime in expected[period]
    ]
    assert len(pairs) == sum(len(regimes) for regimes in got.values())
    for got_values, expected_values in pairs:
        np.testing.assert_array_equal(got_values, expected_values)


def _wealth_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _specialize(function):
    """The same concrete function at every age, behind an age-specialization marker."""
    return AgeSpecializedFunction(build=lambda _age: function, signature=lambda _age: 0)


def _imputation_regime(*, nodes, marked, category_control=False) -> UserRegime:
    if category_control:

        def impute(*, hours: DiscreteState) -> FloatND:
            return jnp.asarray(jnp.where(hours == 0, nodes[0], nodes[1]), dtype=float)

        grid = DiscreteGrid(category_class=_HoursCodes)
    else:
        impute = _hours_as_float
        grid = DiscreteGrid(values=np.array(nodes))
    return UserRegime(
        states={
            "hours": grid,
            "earnings": Phased(
                solve=_specialize(impute) if marked else impute,
                simulate=LinSpacedGrid(start=0, stop=30, n_points=2),
            ),
        },
        state_transitions={
            "hours": fixed_transition("hours"),
            "earnings": fixed_transition("earnings"),
        },
        functions={"utility": _earnings_utility},
    )


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("nodes", _NODE_PAIRS)
@pytest.mark.parametrize("enable_jit", [False, True])
def test_an_imputation_reading_a_value_grid_matches_its_twin_on_codes(
    *, nodes, enable_jit, marked
):
    """A carried state's imputation reads the values of a value grid."""
    got, expected = (
        _first_values(
            model=_one_period_model(
                working=_imputation_regime(
                    nodes=nodes, marked=marked, category_control=category_control
                ),
                enable_jit=enable_jit,
            ),
            params={"discount_factor": 1.0},
        )
        for category_control in (False, True)
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("nodes", _NODE_PAIRS)
@pytest.mark.parametrize("enable_jit", [False, True])
def test_an_age_specialized_imputation_of_category_codes_still_solves(
    *, nodes, enable_jit
):
    """A model without value grids keeps its age-specialized imputations."""
    model = _one_period_model(
        working=_imputation_regime(nodes=nodes, marked=True, category_control=True),
        enable_jit=enable_jit,
    )
    np.testing.assert_array_equal(
        _first_values(model=model, params={"discount_factor": 1.0}),
        np.asarray(nodes, dtype=float),
    )


@pytest.mark.parametrize("nodes", _NODE_PAIRS)
def test_a_phased_age_specialized_utility_beside_a_value_grid_reads_values(nodes):
    """An age-specialized utility inside `Phased` receives each node's value."""
    marker = _specialize(_hours_as_float)
    working = UserRegime(
        states={"hours": DiscreteGrid(values=np.array(nodes))},
        state_transitions={"hours": fixed_transition("hours")},
        functions={"utility": Phased(solve=marker, simulate=marker)},
    )
    np.testing.assert_array_equal(
        _first_values(
            model=_one_period_model(working=working, enable_jit=False),
            params={"discount_factor": 1.0},
        ),
        np.asarray(nodes, dtype=float),
    )


def _choice_utility(*, choice: DiscreteAction) -> FloatND:
    return jnp.asarray(choice, dtype=float)


def _constraint_regime(*, nodes, category_control=False) -> UserRegime:
    threshold = (nodes[0] + nodes[1]) / 2
    if category_control:

        def available(*, hours: DiscreteState, choice: DiscreteAction) -> BoolND:
            physical_hours = jnp.where(hours == 0, nodes[0], nodes[1])
            return (choice == 0) | (physical_hours >= threshold)

        grid = DiscreteGrid(category_class=_HoursCodes)
    else:

        def available(*, hours: DiscreteState, choice: DiscreteAction) -> BoolND:
            return (choice == 0) | (hours >= threshold)

        grid = DiscreteGrid(values=np.array(nodes))
    return UserRegime(
        states={"hours": grid},
        state_transitions={"hours": fixed_transition("hours")},
        actions={"choice": DiscreteGrid(category_class=_HoursCodes)},
        functions={"utility": _choice_utility},
        constraints={"available": _specialize(available)},
    )


@pytest.mark.parametrize("nodes", _NODE_PAIRS)
def test_an_age_specialized_constraint_beside_a_value_grid_reads_values(nodes):
    """An age-specialized constraint compares each node's value to the threshold."""
    model = _one_period_model(working=_constraint_regime(nodes=nodes), enable_jit=False)
    np.testing.assert_array_equal(
        _first_values(model=model, params={"discount_factor": 1.0}), [0.0, 1.0]
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_an_age_specialized_constraint_on_category_codes_still_solves(enable_jit):
    """A model without value grids keeps its age-specialized constraints."""
    model = _one_period_model(
        working=_constraint_regime(nodes=(10, 20), category_control=True),
        enable_jit=enable_jit,
    )
    np.testing.assert_array_equal(
        _first_values(model=model, params={"discount_factor": 1.0}), [0.0, 1.0]
    )


def test_a_model_level_derived_value_grid_counts_for_the_refusals():
    """A model-level derived value grid makes its regimes refuse conditions."""

    def bonus() -> DiscreteState:
        return jnp.int32(20)

    def utility(*, bonus: DiscreteState) -> FloatND:
        return jnp.asarray(bonus, dtype=float)

    working = UserRegime(
        functions={"bonus": bonus, "utility": utility},
        constraints={"enough": ref("bonus") >= 15},
    )
    with pytest.raises(RegimeInitializationError, match=r"condition.*enough"):
        collect_value_grids(
            regimes={"working": working},
            derived_categoricals={"bonus": DiscreteGrid(values=np.array([10, 20]))},
        )


# The discrete-law check resolves lotteries as the solve does.
# A Markov lottery is swept over its target's codes; a joint draw takes each of
# its declared support events. The oracle enumerates the finite model in exact
# fractions, independently of the solver.


@categorical(ordered=False)
class _AliveDead:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _ThreeCodes:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


@categorical(ordered=True)
class _TwoCodes:
    low: ScalarInt
    high: ScalarInt


_LOTTERY_PARAMS: UserParams = {"discount_factor": 0.5}


def _lottery_values(*, probabilities: tuple[Fraction, ...]) -> list[float]:
    """Period-0 values of the two live periods, utility `x + a`, beta 1/2."""
    beta = Fraction(1, 2)
    codes = range(len(probabilities))
    last = [max(Fraction(x + a) for a in (0, 1)) for x in codes]
    expected_last = sum(
        (probability * last[node] for node, probability in enumerate(probabilities)),
        start=Fraction(0),
    )
    return [
        float(max(Fraction(x + a) for a in (0, 1)) + beta * expected_last)
        for x in codes
    ]


def _lottery_utility(*, x: DiscreteState, a: DiscreteAction) -> FloatND:
    return 1.0 * x + a


def _lottery_sibling_utility(
    *, x: DiscreteState, y: DiscreteState, a: DiscreteAction
) -> FloatND:
    return 1.0 * x + y + a


def _dead_utility() -> FloatND:
    return jnp.asarray(0.0)


def _every_row(*, x: DiscreteState, a: DiscreteAction) -> BoolND:
    return (x >= 0) & (a >= 0)


def _quarter_quarter_half() -> FloatND:
    return jnp.asarray([0.25, 0.25, 0.5])


def _half_half_nothing() -> FloatND:
    return jnp.asarray([0.5, 0.5, 0.0])


def _half_half() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _x_from_event(*, draw: Int1D) -> DiscreteState:
    """Events `[10]` and `[11]` give codes 0 and 1; node indices would give -10."""
    return draw[0] - 10


def _x_off_grid_from_event(*, draw: Int1D) -> DiscreteState:
    """Event `[10]` gives code 0 and event `[11]` code 2, off the grid `0, 1`."""
    return 2 * (draw[0] - 10)


def _y_off_grid_at_high_x(*, next_x: DiscreteState) -> DiscreteState:
    """Leave `y`'s grid `0, 1` exactly at `x`'s highest node."""
    return jnp.where(next_x == 2, 2, 0).astype(jnp.int32)


def _lottery_model(*, alive: UserRegime, enable_jit: bool) -> Model:
    return Model(
        n_periods=3,
        regimes={
            "alive": alive,
            "dead": UserRegime(functions={"utility": _dead_utility}),
        },
        regime_id_class=_AliveDead,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"alive": Periods(values=(0,)), "dead": Periods(values=(1,))}},
        enable_jit=enable_jit,
    )


def _markov_model(
    *, form: str, enable_jit: bool, off_grid_sibling: bool, zero_mass_last: bool
) -> Model:
    lottery = StochasticTransition(
        func=_half_half_nothing if zero_mass_last else _quarter_quarter_half
    )
    law = {
        "bare": lottery,
        "phased": Phased(solve=lottery, simulate=lottery),
        "phased_per_target": Phased(
            solve={"alive": lottery}, simulate={"alive": lottery}
        ),
    }[form]
    states = {"x": DiscreteGrid(category_class=_ThreeCodes)}
    transitions = {"x": law}
    if off_grid_sibling:
        states["y"] = DiscreteGrid(category_class=_TwoCodes)
        transitions["y"] = _y_off_grid_at_high_x
    alive = UserRegime(
        states=states,
        actions={"a": DiscreteGrid(category_class=_TwoCodes)},
        state_transitions=transitions,
        constraints={"every_row": _every_row},
        functions={
            "utility": _lottery_sibling_utility
            if off_grid_sibling
            else _lottery_utility
        },
    )
    return _lottery_model(alive=alive, enable_jit=enable_jit)


def _joint_model(*, enable_jit: bool, off_grid_output: bool) -> Model:
    kernel = JointTransition(
        support_size=2,
        support=jnp.asarray([[10], [11]], dtype=jnp.int32),
        probabilities=_half_half,
        outputs={"x": _x_off_grid_from_event if off_grid_output else _x_from_event},
    )
    alive = UserRegime(
        states={"x": DiscreteGrid(category_class=_TwoCodes)},
        actions={"a": DiscreteGrid(category_class=_TwoCodes)},
        joint_transitions={"alive": {"draw": kernel}},
        functions={"utility": _lottery_utility},
    )
    return _lottery_model(alive=alive, enable_jit=enable_jit)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("form", ["bare", "phased", "phased_per_target"])
def test_markov_lottery_in_each_phase_form_passes_the_discrete_check(
    *, form, enable_jit
):
    """A `Phased` Markov lottery keeps its row and node axes apart."""
    model = _markov_model(
        form=form, enable_jit=enable_jit, off_grid_sibling=False, zero_mass_last=False
    )
    model.validate_transitions(params=_LOTTERY_PARAMS)
    got = model.solve(params=_LOTTERY_PARAMS, log_level="off", runtime_checks=True)
    np.testing.assert_array_equal(
        np.asarray(got.values[0]["alive"]),
        _lottery_values(probabilities=(Fraction(1, 4), Fraction(1, 4), Fraction(1, 2))),
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_joint_output_is_checked_at_its_support_events(*, enable_jit):
    """A joint output reads the declared events, never the internal node index."""
    model = _joint_model(enable_jit=enable_jit, off_grid_output=False)
    model.validate_transitions(params=_LOTTERY_PARAMS)
    got = model.solve(params=_LOTTERY_PARAMS, log_level="off", runtime_checks=True)
    np.testing.assert_array_equal(
        np.asarray(got.values[0]["alive"]),
        _lottery_values(probabilities=(Fraction(1, 2), Fraction(1, 2))),
    )


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("form", ["bare", "phased", "phased_per_target"])
@pytest.mark.parametrize("zero_mass_last", [False, True])
def test_off_grid_sibling_of_a_markov_lottery_is_refused(
    *, form, zero_mass_last, enable_jit
):
    """Every node counts, including one the lottery gives no mass."""
    model = _markov_model(
        form=form,
        enable_jit=enable_jit,
        off_grid_sibling=True,
        zero_mass_last=zero_mass_last,
    )
    with pytest.raises(InvalidStateTransitionError, match="next_y"):
        model.validate_transitions(params=_LOTTERY_PARAMS)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_off_grid_joint_output_is_refused(*, enable_jit):
    """The error names the support node whose event leaves the grid."""
    model = _joint_model(enable_jit=enable_jit, off_grid_output=True)
    with pytest.raises(InvalidStateTransitionError, match="next_x") as error:
        model.validate_transitions(params=_LOTTERY_PARAMS)
    assert "'draw': 1" in str(error.value)


# A law may return its code as a Python `int`.


@categorical(ordered=False)
class _SourceTarget:
    source: ScalarInt
    target: ScalarInt


def _constant_code_values(*, next_code: int) -> tuple[list[float], list[float]]:
    """Both periods' values: utility `kind + choice`, then `100 + kind`, beta 1/2."""
    terminal = [100.0 + kind for kind in (0, 1)]
    source = [
        max(kind + choice + 0.5 * terminal[next_code] for choice in (0, 1))
        for kind in (0, 1)
    ]
    return source, terminal


def _python_low() -> int:
    return 0


def _python_high() -> int:
    return 1


def _array_low() -> DiscreteState:
    return jnp.int32(0)


def _array_high() -> DiscreteState:
    return jnp.int32(1)


def _kind_utility(*, kind: DiscreteState, choice: DiscreteAction) -> FloatND:
    return jnp.asarray(kind, dtype=float) + jnp.asarray(choice, dtype=float)


def _kind_terminal_utility(*, kind: DiscreteState) -> FloatND:
    return 100.0 + jnp.asarray(kind, dtype=float)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("runtime_checks", [False, True], ids=["unchecked", "checked"])
@pytest.mark.parametrize(
    ("law", "next_code"),
    [
        pytest.param(_python_low, 0, id="python-low"),
        pytest.param(_array_low, 0, id="array-low"),
        pytest.param(_python_high, 1, id="python-high"),
        pytest.param(_array_high, 1, id="array-high"),
    ],
)
def test_constant_category_law_may_return_a_python_int(
    *, law, next_code, runtime_checks, enable_jit
):
    """A Python `int` code solves like the same code as an array."""
    source = UserRegime(
        states={"kind": DiscreteGrid(category_class=_TwoCodes)},
        state_transitions={"kind": law},
        actions={"choice": DiscreteGrid(category_class=_TwoCodes)},
        functions={"utility": _kind_utility},
    )
    target = UserRegime(
        states={"kind": DiscreteGrid(category_class=_TwoCodes)},
        functions={"utility": _kind_terminal_utility},
    )
    model = Model(
        regimes={"source": source, "target": target},
        regime_id_class=_SourceTarget,
        ages=AgeGrid(start=20, inclusive_stop=21, step="Y"),
        initial_nodes={20: "source"},
        edges={"source": {"target": 20}},
        enable_jit=enable_jit,
    )
    got = model.solve(
        params={"discount_factor": 0.5}, log_level="off", runtime_checks=runtime_checks
    )
    source_values, terminal_values = _constant_code_values(next_code=next_code)
    np.testing.assert_array_equal(np.asarray(got.values[0]["source"]), source_values)
    np.testing.assert_array_equal(np.asarray(got.values[1]["target"]), terminal_values)


def _read_matrix_axis(
    *, values: list[list[float]], axis: int, code: int
) -> list[float]:
    """Read one axis of a matrix with literal bounds: NaN off the axis."""
    size = len(values) if axis == 0 else len(values[0])
    cells = len(values[0]) if axis == 0 else len(values)
    if not 0 <= code < size:
        return [math.nan] * cells
    return list(values[code]) if axis == 0 else [row[code] for row in values]


def _as_code(*, code: int, representation: str) -> int | jax.Array:
    if representation == "python":
        return code
    if representation == "int32":
        return jnp.int32(code)
    if not jax.config.jax_enable_x64:
        pytest.skip("An int64 code needs x64.")
    return jnp.int64(code)


@pytest.mark.parametrize("representation", ["python", "int32", "int64"])
@pytest.mark.parametrize("route", ["eager", "jit", "vmap"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("which", ["lowest", "highest", "negative", "n", "missing"])
def test_read_at_codes_is_the_same_for_every_code_representation(
    *, representation, route, axis, which
):
    """On the axis a code reads its node; off it, NaN, however it is typed."""
    literal = [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]
    values = jnp.asarray(literal)
    size = values.shape[axis]
    raw = {
        "lowest": 0,
        "highest": size - 1,
        "negative": -1,
        "n": size,
        "missing": MISSING_CAT_CODE,
    }[which]
    code = _as_code(code=raw, representation=representation)
    positions = (code,) if axis == 0 else (slice(None), code)

    # The code is captured inside the trace, as a constant law's literal is.
    def read(array: FloatND) -> FloatND:
        return read_at_codes(array=array, positions=positions)

    expected = _read_matrix_axis(values=literal, axis=axis, code=raw)
    if route == "eager":
        got = read(values)
    elif route == "jit":
        got = jax.jit(read)(values)
    else:
        got = jax.vmap(read)(jnp.stack([values, values]))
        expected = [expected, expected]
    np.testing.assert_array_equal(np.asarray(got), expected)
    assert got.dtype == values.dtype


# Finalization keeps a declared `Phased` object when it has nothing to strip.


@categorical(ordered=False)
class _CoupleTerminal:
    couple: ScalarInt
    couple_terminal: ScalarInt


@categorical(ordered=False)
class _Work:
    no: ScalarInt
    yes: ScalarInt


def _u_f(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.log(wealth) - 0.1 * work


def _u_m(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.5 * jnp.log(wealth) - 0.2 * work


def _u_m_simulate(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.5 * jnp.log(wealth) - 0.3 * work


def _u_terminal(*, wealth: ContinuousState) -> FloatND:
    return 0.5 * jnp.log(wealth)


def _household(*, m_utility: UserFunction | Phased) -> UserRegime:
    return UserRegime(
        states={"wealth": LinSpacedGrid(start=1.0, stop=3.0, n_points=3)},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={"work": DiscreteGrid(category_class=_Work)},
        functions={"utility": CollectiveUtility(utilities={"f": _u_f, "m": m_utility})},
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_phased_stakeholder_utility_without_value_grids_builds_and_solves(
    *, enable_jit
):
    """The declared `Phased` body reaches the household check as the same object."""
    phased = Phased(solve=_u_m, simulate=_u_m_simulate)
    terminal = UserRegime(
        states={"wealth": LinSpacedGrid(start=1.0, stop=3.0, n_points=3)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _u_terminal, "m": _u_terminal})
        },
    )
    model = Model(
        regimes={"couple": _household(m_utility=phased), "couple_terminal": terminal},
        edges={"couple": {"couple_terminal": 0}},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_CoupleTerminal,
        initial_nodes={0: "couple"},
        enable_jit=enable_jit,
    )
    values = model.solve(
        params={
            "couple": {"koopmans_aggregator": {"discount_factor": 0.9}},
            "couple_terminal": {},
        },
        log_level="off",
    ).values
    # m's period-0 value: her solve utility, not working, plus the discounted
    # terminal one at wealth held fixed.
    expected = 0.5 * np.log([1.0, 2.0, 3.0]) * 1.9
    aaae(np.asarray(values[0]["couple"])[..., 1], expected, decimal=DECIMAL_PRECISION)


@categorical(ordered=False)
class _SingleOrCouple:
    single: ScalarInt
    single_terminal: ScalarInt
    couple: ScalarInt
    couple_terminal: ScalarInt


def _u_single(
    *, wealth: ContinuousState, work: DiscreteAction, kind: DiscreteState
) -> FloatND:
    return jnp.log(wealth) - 0.1 * work + 0.001 * kind


def _u_single_terminal(*, wealth: ContinuousState, kind: DiscreteState) -> FloatND:
    return jnp.log(wealth) + 0.001 * kind


@pytest.mark.parametrize("enable_jit", [False, True])
def test_phased_stakeholder_utility_beside_a_value_grid_elsewhere_builds(*, enable_jit):
    """A value grid in another regime leaves the household's `Phased` body alone."""
    wealth = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)
    kind = DiscreteGrid(values=np.array([10, 20]))
    phased = Phased(solve=_u_m, simulate=_u_m_simulate)
    model = Model(
        regimes={
            "single": UserRegime(
                states={"wealth": wealth, "kind": kind},
                state_transitions={
                    "wealth": fixed_transition("wealth"),
                    "kind": fixed_transition("kind"),
                },
                actions={"work": DiscreteGrid(category_class=_Work)},
                functions={"utility": _u_single},
            ),
            "single_terminal": UserRegime(
                states={"wealth": wealth, "kind": kind},
                functions={"utility": _u_single_terminal},
            ),
            "couple": _household(m_utility=phased),
            "couple_terminal": UserRegime(
                states={"wealth": wealth},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _u_terminal, "m": _u_terminal}
                    )
                },
            ),
        },
        edges={"single": {"single_terminal": 0}, "couple": {"couple_terminal": 0}},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_SingleOrCouple,
        initial_nodes={0: ["single", "couple"]},
        enable_jit=enable_jit,
    )
    assert model.user_regimes["couple"].decomposed_functions["utility_m"] is phased


def test_a_really_replaced_phased_stakeholder_body_still_fails_the_identity_check():
    """An equal but new `Phased` object is a different body."""
    regime = _household(m_utility=Phased(solve=_u_m, simulate=_u_m_simulate))
    engine = {
        **dict(regime.decomposed_functions),
        "utility_m": Phased(solve=_u_m, simulate=_u_m_simulate),
    }
    with pytest.raises(
        RegimeInitializationError, match="different body for 'utility_m'"
    ):
        regime.with_engine_functions(engine_functions=engine)


def test_authored_keeps_a_phased_object_and_strips_a_wrapped_side():
    """Nothing to strip keeps the object; a value adapter on a side is removed."""
    plain = Phased(solve=_u_m, simulate=_u_m_simulate)
    adapted = Phased(
        solve=adapt_user_function(
            func=_identity_wealth, value_grids=_VALUE_GRIDS, encode=None
        ),
        simulate=_identity_wealth,
    )

    stripped = _authored(adapted)

    assert _authored(plain) is plain
    assert isinstance(stripped, Phased)
    assert stripped is not adapted
    assert stripped.solve is _identity_wealth
    assert stripped.simulate is _identity_wealth


# A Markov draw on an edge whose target does not carry the drawn state is resolved
# at the support nodes of the canonical plan, not on a target grid it lacks.


def _source_only_values(*, zero_mass_last: bool) -> dict[str, list[float]]:
    """Exact values: utility `x + a`, then `100 + y` with `y = 1{x' >= 1}`, beta 1/2."""
    weights = (
        (Fraction(1, 2), Fraction(1, 2), Fraction(0))
        if zero_mass_last
        else (Fraction(1, 4), Fraction(1, 4), Fraction(1, 2))
    )
    terminal = (Fraction(100), Fraction(101))
    continuation = sum(
        (mass * terminal[int(node >= 1)] for node, mass in enumerate(weights)),
        start=Fraction(0),
    )
    source = [
        float(max(Fraction(x + a) + Fraction(1, 2) * continuation for a in (0, 1)))
        for x in (0, 1, 2)
    ]
    return {"source": source, "terminal": [float(value) for value in terminal]}


def _source_utility(*, x: DiscreteState, a: DiscreteAction) -> FloatND:
    return 1.0 * x + a


def _y_terminal_utility(*, y: DiscreteState) -> FloatND:
    return 100.0 + y


def _y_from_next_x(*, next_x: DiscreteState) -> DiscreteState:
    return (next_x >= 1).astype(jnp.int32)


def _source_only_model(
    *,
    form: str,
    enable_jit: bool,
    zero_mass_last: bool,
    invalid_phase: str | None = None,
) -> Model:
    lottery = StochasticTransition(
        func=_half_half_nothing if zero_mass_last else _quarter_quarter_half
    )
    y_law = {
        None: _y_from_next_x,
        "both": _y_off_grid_at_high_x,
        "solve": Phased(solve=_y_off_grid_at_high_x, simulate=_y_from_next_x),
        "simulate": Phased(solve=_y_from_next_x, simulate=_y_off_grid_at_high_x),
    }[invalid_phase]
    return Model(
        n_periods=2,
        regimes={
            "source": UserRegime(
                states={"x": DiscreteGrid(category_class=_ThreeCodes)},
                actions={"a": DiscreteGrid(category_class=_TwoCodes)},
                state_transitions={
                    "x": (
                        lottery
                        if form == "bare"
                        else Phased(solve=lottery, simulate=lottery)
                    ),
                    "y": y_law,
                },
                functions={"utility": _source_utility},
            ),
            "target": UserRegime(
                states={"y": DiscreteGrid(category_class=_TwoCodes)},
                functions={"utility": _y_terminal_utility},
            ),
        },
        regime_id_class=_SourceTarget,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={"source": {"target": Periods(values=(0,))}},
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("form", ["bare", "phased"])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("zero_mass_last", [False, True])
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_source_only_markov_draw_validates_and_solves_exactly(
    *, form, enable_jit, zero_mass_last, runtime_checks
):
    """The target carries `y` but not `x`, whose draw lives on this edge only."""
    model = _source_only_model(
        form=form, enable_jit=enable_jit, zero_mass_last=zero_mass_last
    )
    model.validate_transitions(params=_LOTTERY_PARAMS)
    got = model.solve(
        params=_LOTTERY_PARAMS, log_level="off", runtime_checks=runtime_checks
    )
    expected = _source_only_values(zero_mass_last=zero_mass_last)
    np.testing.assert_array_equal(
        np.asarray(got.values[0]["source"]), expected["source"]
    )
    np.testing.assert_array_equal(
        np.asarray(got.values[1]["target"]), expected["terminal"]
    )


@pytest.mark.parametrize("form", ["bare", "phased"])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("supply_solution", [False, True])
def test_checked_simulation_accepts_a_source_only_markov_draw(
    *, form, enable_jit, supply_solution
):
    model = _source_only_model(form=form, enable_jit=enable_jit, zero_mass_last=False)
    solution = (
        model.solve(params=_LOTTERY_PARAMS, log_level="off", runtime_checks=False)
        if supply_solution
        else None
    )
    panel = model.simulate(
        params=_LOTTERY_PARAMS,
        initial_conditions={
            "x": jnp.asarray([0, 1, 2], dtype=jnp.int32),
            "period": jnp.asarray([0, 0, 0], dtype=jnp.int32),
            "regime_id": jnp.asarray([_SourceTarget.source] * 3, dtype=jnp.int32),
        },
        solution=solution,
        seed=19,
        log_level="off",
        runtime_checks=True,
    ).to_dataframe(use_categorical_codes=True)
    source = panel.loc[panel["period"] == 0]
    target = panel.loc[panel["period"] == 1]
    assert len(source) == len(target) == 3
    np.testing.assert_array_equal(source["a"].to_numpy(), [1, 1, 1])
    assert target["y"].notna().all()
    assert set(target["y"].to_numpy()) <= {0, 1}


@pytest.mark.parametrize("form", ["bare", "phased"])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("zero_mass_last", [False, True])
@pytest.mark.parametrize("invalid_phase", ["solve", "simulate", "both"])
@pytest.mark.parametrize("entry", ["validate", "checked_solve"])
def test_off_grid_output_of_a_source_only_markov_draw_is_refused(
    *, form, enable_jit, zero_mass_last, invalid_phase, entry
):
    """Node 2 leaves `y`'s grid in each phase that reads it, at zero mass too."""
    model = _source_only_model(
        form=form,
        enable_jit=enable_jit,
        zero_mass_last=zero_mass_last,
        invalid_phase=invalid_phase,
    )
    check = (
        partial(model.validate_transitions, params=_LOTTERY_PARAMS)
        if entry == "validate"
        else partial(
            model.solve, params=_LOTTERY_PARAMS, log_level="off", runtime_checks=True
        )
    )
    with pytest.raises(InvalidStateTransitionError, match="next_y") as error:
        check()
    assert "'next_x': 2" in str(error.value)
    assert "'source'" in str(error.value)
    assert "'target'" in str(error.value)


# The discrete-law check reads the feasibility the solve uses and the grids of the
# phase it checks.


@categorical(ordered=False)
class _SourceTerminal:
    source: ScalarInt
    terminal: ScalarInt


def _pair_flow(*, kind: DiscreteState, a: DiscreteAction) -> FloatND:
    return 1.0 - 2.0 * a + 0.0 * kind


def _pair_terminal(*, kind: DiscreteState) -> FloatND:
    return 0.0 * kind


def _kind_leaves_when_working(
    *, kind: DiscreteState, a: DiscreteAction
) -> DiscreteState:
    return jnp.where(a == 1, 2, kind).astype(jnp.int32)


def _rest_only(*, a: DiscreteAction) -> BoolND:
    return a == 0


def _participation_model(*, rest_as_ordinary: bool) -> Model:
    """`Q_alice = 1 - 2a`; only `a = 1` sends `kind` to code 2, off `{0, 1}`.

    The restriction `a == 0` sits inside the value constraint, as in the review's
    `(a == 0) & (Q_f >= 0)`, or beside it as an ordinary constraint.
    """

    def participation(*, Q_alice: FloatND, a: DiscreteAction) -> BoolND:
        return ((a == 0) | rest_as_ordinary) & (Q_alice >= 0.0)

    constraints = {"participation": ValueDependentConstraint(predicate=participation)}
    if rest_as_ordinary:
        constraints["rest_only"] = _rest_only
    return Model(
        regimes={
            "source": UserRegime(
                states={"kind": DiscreteGrid(category_class=_TwoCodes)},
                actions={"a": DiscreteGrid(category_class=_TwoCodes)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"alice": _pair_flow, "bob": _pair_flow}
                    )
                },
                state_transitions={"kind": _kind_leaves_when_working},
                constraints=constraints,
            ),
            "terminal": UserRegime(
                states={"kind": DiscreteGrid(category_class=_TwoCodes)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"alice": _pair_terminal, "bob": _pair_terminal}
                    )
                },
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"source": {"terminal": 0}},
        initial_nodes={0: "source"},
        regime_id_class=_SourceTerminal,
    )


def test_value_constraint_cannot_excuse_an_off_grid_law():
    """`a = 1` is feasible by the ordinary constraints, and its code is off the grid.

    An off-grid code has no value, so the value constraint cannot decide that
    cell; the error names the ordinary constraint as the remedy.
    """
    with pytest.raises(InvalidStateTransitionError, match="ordinary constraint"):
        _participation_model(rest_as_ordinary=False).validate_transitions(
            params={"discount_factor": 1.0}
        )


@pytest.mark.parametrize("runtime_checks", [False, True])
def test_ordinary_restriction_beside_a_value_constraint_solves(*, runtime_checks):
    """Declared as an ordinary constraint, `a == 0` excludes the off-grid action."""
    model = _participation_model(rest_as_ordinary=True)
    model.validate_transitions(params={"discount_factor": 1.0})
    got = model.solve(
        params={"discount_factor": 1.0},
        log_level="off",
        runtime_checks=runtime_checks,
    )
    np.testing.assert_array_equal(
        np.asarray(got.values[0]["source"]), [[1.0, 1.0], [1.0, 1.0]]
    )


def _impute_low() -> DiscreteState:
    return jnp.int32(0)


def _flag_utility(*, flag: DiscreteState, a: DiscreteAction) -> FloatND:
    return 1.0 * flag + a


def _flag_other_utility(
    *, flag: DiscreteState, other: DiscreteState, a: DiscreteAction
) -> FloatND:
    return 1.0 * flag + other + a


def _flag_stays(*, flag: DiscreteState) -> DiscreteState:
    return flag


def _flag_leaves(*, flag: DiscreteState) -> DiscreteState:
    return flag + 2


def _carried_flag_model(*, law: UserFunction, with_category_state: bool) -> Model:
    """`flag` is carried: imputed in the solve, a category grid in simulation."""
    states = {
        "flag": Phased(
            solve=_impute_low, simulate=DiscreteGrid(category_class=_TwoCodes)
        )
    }
    transitions = {"flag": law}
    if with_category_state:
        states["other"] = DiscreteGrid(category_class=_TwoCodes)
        transitions["other"] = fixed_transition("other")
    return Model(
        n_periods=3,
        regimes={
            "alive": UserRegime(
                states=states,
                state_transitions=transitions,
                actions={"a": DiscreteGrid(category_class=_TwoCodes)},
                functions={
                    "utility": (
                        _flag_other_utility if with_category_state else _flag_utility
                    )
                },
            ),
            "dead": UserRegime(functions={"utility": _dead_utility}),
        },
        regime_id_class=_AliveDead,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"alive": Periods(values=(0,)), "dead": Periods(values=(1,))}},
    )


@pytest.mark.parametrize("with_category_state", [False, True])
def test_off_grid_law_of_a_carried_category_state_is_refused(*, with_category_state):
    """The simulate phase checks the simulation grid of a carried state."""
    model = _carried_flag_model(
        law=_flag_leaves, with_category_state=with_category_state
    )
    with pytest.raises(InvalidStateTransitionError, match="next_flag"):
        model.validate_transitions(params=_LOTTERY_PARAMS)


@pytest.mark.parametrize("with_category_state", [False, True])
def test_on_grid_law_of_a_carried_category_state_passes(*, with_category_state):
    _carried_flag_model(
        law=_flag_stays, with_category_state=with_category_state
    ).validate_transitions(params=_LOTTERY_PARAMS)


# An EGM continuation reads its child's carry at the next discrete code; a code off
# the child's grid reads NaN, never a clamped row.


def _egm_utility(*, consumption: ContinuousAction, kind: DiscreteState) -> FloatND:
    return jnp.log(consumption) + 0.0 * kind


def _egm_inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _egm_savings(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def _egm_next_wealth(savings: ContinuousState) -> ContinuousState:
    return 1.0 + savings


def _kind_up(*, kind: DiscreteState) -> DiscreteState:
    return kind + 1


def _egm_dead_utility(*, wealth: ContinuousState, kind: DiscreteState) -> FloatND:
    return jnp.log(wealth) + 0.0 * kind


def test_dcegm_continuation_reads_nan_at_an_off_grid_child_code():
    """Exactly the cells whose path leaves `{0, 1}` are NaN, as in a grid search.

    At period 1, `kind = 1` moves to code 2 on the way to `dead`. At period 0,
    `kind = 1` reads its child's carry at code 2, and `kind = 0` lands on
    `kind = 1`, whose period-1 value is NaN. The carry pads unused nodes with
    NaN, so without the off-grid mask the code-2 rows read as an empty row and a
    finite corner value replaces NaN.
    """
    margin = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="wealth",
        post_decision_state="savings",
    )
    wealth = LinSpacedGrid(start=1.0, stop=3.0, n_points=4)
    alive = ConsumptionSavingsRegime(
        states={"wealth": wealth, "kind": DiscreteGrid(category_class=_TwoCodes)},
        actions={"consumption": LinSpacedGrid(start=0.5, stop=1.0, n_points=3)},
        state_transitions={"wealth": _egm_next_wealth, "kind": _kind_up},
        functions={
            "utility": _egm_utility,
            "inverse_marginal_utility": _egm_inverse_marginal_utility,
            "savings": _egm_savings,
        },
        constraints={
            "borrowing_limit": post_decision_lower_bound(margin=margin, lower=0.0)
        },
        liquid=margin,
        solver=DCEGM(savings_grid=LinSpacedGrid(start=0.0, stop=3.0, n_points=7)),
    )
    dead = UserRegime(
        states={"wealth": wealth, "kind": DiscreteGrid(category_class=_TwoCodes)},
        functions={"utility": _egm_dead_utility},
    )
    model = Model(
        n_periods=3,
        regimes={"alive": alive, "dead": dead},
        regime_id_class=_AliveDead,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"alive": Periods(values=(0,)), "dead": Periods(values=(1,))}},
    )
    values = model.solve(
        params={"discount_factor": 0.9}, log_level="off", runtime_checks=False
    ).values
    nan_pattern = [
        np.isnan(np.asarray(values[period]["alive"])).all(axis=1).tolist()
        for period in (0, 1)
    ]
    finite_pattern = [
        np.isfinite(np.asarray(values[period]["alive"])).all(axis=1).tolist()
        for period in (0, 1)
    ]
    assert nan_pattern == [[True, True], [False, True]]
    assert finite_pattern == [[False, False], [True, False]]


# A gated edge decides its branch only at a cell the ordinary constraints allow, so
# a law leaving the grid there is refused whatever the gate reads.


@categorical(ordered=True)
class _SourceTargetFallback:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _gated_flow(*, kind: DiscreteState, a: DiscreteAction) -> FloatND:
    return 20.0 * a + 0.0 * kind


def _kind_utility_zero(*, kind: DiscreteState) -> FloatND:
    return 0.0 * kind


def _kind_utility_low(*, kind: DiscreteState) -> FloatND:
    return -10.0 + 0.0 * kind


def _kind_stays(*, kind: DiscreteState, a: DiscreteAction) -> DiscreteState:
    return kind + 0 * a


def _invalid_when_working(code: int) -> UserFunction:
    def law(*, kind: DiscreteState, a: DiscreteAction) -> DiscreteState:
        return jnp.where(a == 1, code, kind).astype(jnp.int32)

    return law


def _window_gate(*, center: float, width: float) -> UserFunction:
    def gate(V_target: FloatND) -> BoolND:
        return jnp.abs(V_target - center) < width

    return gate


def _threshold_gate(V_target: FloatND) -> BoolND:
    return V_target > -5.0


def _project_kind(kind: DiscreteState) -> DiscreteState:
    return jnp.clip(kind, 0, 1).astype(jnp.int32)


def _gated_model(
    *,
    gate: UserFunction,
    law: UserFunction | Phased,
    enable_jit: bool,
    forbid_working: bool = False,
) -> Model:
    kind = DiscreteGrid(category_class=_TwoCodes)
    return Model(
        regimes={
            "source": UserRegime(
                states={"kind": kind},
                actions={"a": DiscreteGrid(category_class=_TwoCodes)},
                functions={"utility": _gated_flow},
                state_transitions={"kind": law},
                constraints={"rest_only": _rest_only} if forbid_working else {},
            ),
            "target": UserRegime(
                states={"kind": kind}, functions={"utility": _kind_utility_zero}
            ),
            "fallback": UserRegime(
                states={"kind": kind}, functions={"utility": _kind_utility_low}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={
            "source": Transition(
                law="target",
                gates={
                    "target": Gate(
                        predicate=gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"kind": _project_kind},
                                )
                            )
                        },
                    )
                },
            )
        },
        initial_nodes={0: "source"},
        regime_id_class=_SourceTargetFallback,
        enable_jit=enable_jit,
    )


_PARAMS_ONE: UserParams = {"discount_factor": 1.0}
_WINDOWS = [(0.0, 5.0), (0.0, 0.5), (1.0, 2.0), (-2.0, 3.0)]
_BAD_CODES = [-1, 2, 3]


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize(("center", "width"), _WINDOWS)
@pytest.mark.parametrize("bad_code", _BAD_CODES)
def test_off_grid_law_behind_a_value_gate_is_refused(
    *, bad_code, center, width, enable_jit
):
    """A finite-window gate on `V_target` cannot hide the off-grid code of `a = 1`."""
    model = _gated_model(
        gate=_window_gate(center=center, width=width),
        law=_invalid_when_working(bad_code),
        enable_jit=enable_jit,
    )
    with pytest.raises(InvalidStateTransitionError, match="ordinary constraint"):
        model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize(("center", "width"), _WINDOWS)
def test_on_grid_law_behind_a_value_gate_solves(*, center, width, enable_jit):
    """Two actions: `Q(0) = 0`, `Q(1) = 20` at either code."""
    got = _gated_model(
        gate=_window_gate(center=center, width=width),
        law=_kind_stays,
        enable_jit=enable_jit,
    ).solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)
    np.testing.assert_array_equal(np.asarray(got.values[0]["source"]), [20.0, 20.0])


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("bad_code", _BAD_CODES)
def test_ordinarily_excluded_off_grid_action_behind_a_gate_solves(
    *, bad_code, enable_jit
):
    """An ordinary constraint excluding `a = 1` keeps its off-grid code unread."""
    got = _gated_model(
        gate=_window_gate(center=0.0, width=5.0),
        law=_invalid_when_working(bad_code),
        enable_jit=enable_jit,
        forbid_working=True,
    ).solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)
    np.testing.assert_array_equal(np.asarray(got.values[0]["source"]), [0.0, 0.0])


@pytest.mark.parametrize("enable_jit", [False, True])
def test_simulation_only_off_grid_law_behind_a_value_gate_is_refused(*, enable_jit):
    """The simulate phase's law is checked before any subject is rerouted."""
    model = _gated_model(
        gate=_threshold_gate,
        law=Phased(solve=_kind_stays, simulate=_invalid_when_working(2)),
        enable_jit=enable_jit,
    )
    with pytest.raises(InvalidStateTransitionError, match="ordinary constraint"):
        model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_on_grid_phased_law_behind_a_value_gate_simulates_to_the_target(*, enable_jit):
    model = _gated_model(
        gate=_threshold_gate,
        law=Phased(solve=_kind_stays, simulate=_kind_stays),
        enable_jit=enable_jit,
    )
    solution = model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)
    panel = model.simulate(
        params=_PARAMS_ONE,
        initial_conditions={
            "kind": jnp.asarray([0, 1], dtype=jnp.int32),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, _SourceTargetFallback.source, dtype=jnp.int32),
        },
        solution=solution,
        seed=3,
        log_level="off",
        runtime_checks=True,
    ).to_dataframe(use_categorical_codes=True)
    later = panel.loc[panel["period"] == 1].sort_values("subject_id")
    assert later["regime_name"].tolist() == ["target", "target"]
    np.testing.assert_array_equal(later["kind"].to_numpy(), [0, 1])


# A continuous source state between grid nodes can make a simulated law emit a code
# the preflight never sees. Simulation checks the raw code of every subject drawn
# to a target before a gate reads it or reroutes the subject.


def _flow_with_wealth(
    *, kind: DiscreteState, wealth: ContinuousState, a: DiscreteAction
) -> FloatND:
    return 20.0 * a + 0.0 * kind + 0.0 * wealth


def _between_nodes_model(
    *,
    enable_jit: bool,
    gated: bool = True,
    bad_code: int = 2,
    lower: float = 0.0,
    upper: float = 1.0,
    n_wealth_points: int = 2,
    forbid_working: bool = False,
    valid_simulation: bool = False,
) -> Model:
    """Working sends `kind` to `bad_code` only strictly inside `(lower, upper)`."""

    def interior_law(
        *, kind: DiscreteState, wealth: ContinuousState, a: DiscreteAction
    ) -> DiscreteState:
        bad = (a == 1) & (wealth > lower) & (wealth < upper)
        return jnp.where(bad, bad_code, kind).astype(jnp.int32)

    kind = DiscreteGrid(category_class=_TwoCodes)
    gates = {
        "target": Gate(
            predicate=_threshold_gate,
            routes={
                "only": StakeholderRoute(
                    fallback=ProjectedRegimeValue(
                        regime="fallback", projection={"kind": _project_kind}
                    )
                )
            },
        )
    }
    return Model(
        regimes={
            "source": UserRegime(
                states={
                    "kind": kind,
                    "wealth": LinSpacedGrid(
                        start=0.0, stop=1.0, n_points=n_wealth_points
                    ),
                },
                actions={"a": DiscreteGrid(category_class=_TwoCodes)},
                functions={"utility": _flow_with_wealth},
                state_transitions={
                    "kind": Phased(
                        solve=_kind_stays,
                        simulate=_kind_stays if valid_simulation else interior_law,
                    ),
                    "wealth": fixed_transition("wealth"),
                },
                constraints={"rest_only": _rest_only} if forbid_working else {},
            ),
            "target": UserRegime(
                states={"kind": kind}, functions={"utility": _kind_utility_zero}
            ),
            "fallback": UserRegime(
                states={"kind": kind}, functions={"utility": _kind_utility_low}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={
            "source": Transition(law="target", gates=gates) if gated else {"target": 0}
        },
        initial_nodes={0: "source"},
        regime_id_class=_SourceTargetFallback,
        enable_jit=enable_jit,
    )


def _simulate_between_nodes(
    *, model: Model, wealth: tuple[float, float], solution: SolutionResult | None
) -> pd.DataFrame:
    return model.simulate(
        params=_PARAMS_ONE,
        initial_conditions={
            "kind": jnp.asarray([0, 1], dtype=jnp.int32),
            "wealth": jnp.asarray(wealth),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, _SourceTargetFallback.source, dtype=jnp.int32),
        },
        solution=solution,
        seed=3,
        log_level="off",
        runtime_checks=True,
    ).to_dataframe(use_categorical_codes=True)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("supply_solution", [False, True])
@pytest.mark.parametrize("gated", [True, False])
def test_off_grid_code_from_a_between_node_value_fails_simulation(
    *, gated, supply_solution, enable_jit
):
    """Wealth 0.5 lies between nodes 0 and 1; working there emits code 2."""
    model = _between_nodes_model(enable_jit=enable_jit, gated=gated)
    solution = model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)
    with pytest.raises(InvalidStateTransitionError, match="kind"):
        _simulate_between_nodes(
            model=model,
            wealth=(0.5, 0.5),
            solution=solution if supply_solution else None,
        )


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize(("lower", "upper"), [(0.125, 0.875), (0.25, 0.75)])
@pytest.mark.parametrize("bad_code", [-1, 3, MISSING_CAT_CODE])
def test_between_node_window_and_code_mutations_fail_simulation(
    *, bad_code, lower, upper, enable_jit
):
    model = _between_nodes_model(
        enable_jit=enable_jit, bad_code=bad_code, lower=lower, upper=upper
    )
    solution = model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)
    midpoint = (lower + upper) / 2
    with pytest.raises(InvalidStateTransitionError, match="kind"):
        _simulate_between_nodes(
            model=model, wealth=(midpoint, midpoint), solution=solution
        )


def _assert_reaches_target(*, panel: pd.DataFrame, action: int) -> None:
    first = panel.loc[panel["period"] == 0].sort_values("subject_id")
    later = panel.loc[panel["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(first["a"].to_numpy(), [action, action])
    assert later["regime_name"].tolist() == ["target", "target"]
    np.testing.assert_array_equal(later["kind"].to_numpy(), [0, 1])


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("supply_solution", [False, True])
def test_on_node_values_of_the_same_law_simulate_to_the_target(
    *, supply_solution, enable_jit
):
    model = _between_nodes_model(enable_jit=enable_jit)
    solution = model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)
    panel = _simulate_between_nodes(
        model=model,
        wealth=(0.0, 1.0),
        solution=solution if supply_solution else None,
    )
    _assert_reaches_target(panel=panel, action=1)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_valid_simulated_law_between_nodes_simulates_to_the_target(*, enable_jit):
    model = _between_nodes_model(enable_jit=enable_jit, valid_simulation=True)
    solution = model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)
    panel = _simulate_between_nodes(model=model, wealth=(0.5, 0.5), solution=solution)
    _assert_reaches_target(panel=panel, action=1)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_ordinarily_excluded_action_between_nodes_simulates_to_the_target(
    *, enable_jit
):
    model = _between_nodes_model(enable_jit=enable_jit, forbid_working=True)
    solution = model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)
    panel = _simulate_between_nodes(model=model, wealth=(0.5, 0.5), solution=solution)
    _assert_reaches_target(panel=panel, action=0)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_a_grid_node_inside_the_window_is_refused_before_the_solve(*, enable_jit):
    """With a node at 0.5 the preflight sees the bad code itself."""
    model = _between_nodes_model(enable_jit=enable_jit, n_wealth_points=3)
    with pytest.raises(InvalidStateTransitionError, match="ordinary constraint"):
        model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)


def _working_goes_to_fallback(*, a: DiscreteAction) -> ScalarInt:
    return jnp.where(
        a == 1, _SourceTargetFallback.fallback, _SourceTargetFallback.target
    )


def _kind_off_grid_between_nodes(
    *, kind: DiscreteState, wealth: ContinuousState, a: DiscreteAction
) -> DiscreteState:
    return jnp.where((a == 1) & (wealth > 0.0) & (wealth < 1.0), 2, kind).astype(
        jnp.int32
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_off_grid_code_to_a_target_no_subject_is_drawn_to_passes(*, enable_jit):
    """Working moves the subject to `fallback`; its code toward `target` is unread."""
    kind = DiscreteGrid(category_class=_TwoCodes)
    model = Model(
        regimes={
            "source": UserRegime(
                states={
                    "kind": kind,
                    "wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
                },
                actions={"a": DiscreteGrid(category_class=_TwoCodes)},
                functions={"utility": _flow_with_wealth},
                state_transitions={
                    "kind": Phased(
                        solve={"target": _kind_stays, "fallback": _kind_stays},
                        simulate={
                            "target": _kind_off_grid_between_nodes,
                            "fallback": _kind_stays,
                        },
                    ),
                    "wealth": fixed_transition("wealth"),
                },
            ),
            "target": UserRegime(
                states={"kind": kind}, functions={"utility": _kind_utility_zero}
            ),
            "fallback": UserRegime(
                states={"kind": kind}, functions={"utility": _kind_utility_zero}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={
            "source": Transition(
                targets={"target": (0,), "fallback": (0,)},
                law=_working_goes_to_fallback,
            )
        },
        initial_nodes={0: "source"},
        regime_id_class=_SourceTargetFallback,
        enable_jit=enable_jit,
    )
    solution = model.solve(params=_PARAMS_ONE, log_level="off", runtime_checks=True)
    panel = _simulate_between_nodes(model=model, wealth=(0.5, 0.5), solution=solution)
    later = panel.loc[panel["period"] == 1].sort_values("subject_id")
    assert later["regime_name"].tolist() == ["fallback", "fallback"]


@pytest.mark.parametrize("width", ["int32", "int64"])
def test_drawn_code_count_is_exact_at_any_signed_width(*, width):
    """A wide code is compared at its own width and never aliases a valid one."""
    if width == "int64" and not jax.config.jax_enable_x64:
        pytest.skip("An int64 code needs x64.")
    dtype = jnp.int32 if width == "int32" else jnp.int64
    codes = (
        jnp.asarray([0, 2, 3, -1, 2**31 - 1], dtype=jnp.int32)
        if width == "int32"
        else jnp.asarray([0, 2, 3, -(2**33), 2**31 + 2], dtype=dtype),
    )
    counts = _count_drawn_off_grid_codes(
        codes=codes,
        n_codes=(jnp.int32(3),),
        target_ids=(jnp.int32(1),),
        subjects=jnp.ones(5, dtype=bool),
        new_regime_ids=jnp.ones(5, dtype=jnp.int32),
    )
    assert counts.tolist() == [3]


@pytest.mark.parametrize(
    "codes",
    [
        jnp.asarray([0, 2, 3], dtype=jnp.uint32),
        jnp.asarray([0.0, 0.5, 3.0]),
    ],
    ids=["unsigned", "float"],
)
def test_drawn_code_count_refuses_codes_that_are_not_signed_integers(*, codes):
    with pytest.raises(BeartypeCallHintParamViolation):
        _count_drawn_off_grid_codes(
            codes=(codes,),
            n_codes=(jnp.int32(3),),
            target_ids=(jnp.int32(1),),
            subjects=jnp.ones(3, dtype=bool),
            new_regime_ids=jnp.ones(3, dtype=jnp.int32),
        )


# A reader of a value grid and the same reader on category codes. The value
# model declares `income` by its values; the code twin declares a category
# class and converts each code by hand before it calls the same function. Both
# models must give identical V and identical simulated codes.

_INCOME_VALUES = (1.0, 2.0, 4.0)


def _income_of_code(code: DiscreteState) -> FloatND:
    return jnp.asarray(_INCOME_VALUES)[code]


def _on_codes[F: UserFunction](func: F) -> F:
    """The same function, reading `income` as a code and converting it by hand."""

    @wraps(func)
    def on_codes(**kwargs):
        return func(**{**kwargs, "income": _income_of_code(kwargs["income"])})

    return cast("F", on_codes)


def _as_declared[F: UserFunction](func: F) -> F:
    return func


def _income_grid(*, values: bool) -> DiscreteGrid:
    if values:
        return DiscreteGrid(values=np.array(_INCOME_VALUES))
    return DiscreteGrid(category_class=_ThreeCodes)


def _income_utility(*, income: DiscreteState, a: DiscreteAction) -> FloatND:
    return income * a + 0.5 * (1 - a)


def _survives_deterministically(*, income: DiscreteState, age: float) -> ScalarInt:
    return jnp.where(
        (age == 0) & (income >= 2.0), _WorkingDead.working, _WorkingDead.dead
    )


def _survives_at_high_income(*, income: DiscreteState) -> ScalarInt:
    return jnp.where(income >= 2.0, _WorkingDead.working, _WorkingDead.dead)


def _survival_probability(*, income: DiscreteState) -> FloatND:
    return income / 5.0


def _death_probability(*, income: DiscreteState) -> FloatND:
    return 1.0 - income / 5.0


def _survival_vector(*, income: DiscreteState) -> FloatND:
    return jnp.stack([income / 5.0, 1.0 - income / 5.0])


def _regime_law(*, form: str, values: bool):
    convert = _as_declared if values else _on_codes
    return {
        "deterministic": DeterministicTransition(
            func=convert(_survives_deterministically)
        ),
        "by_age_deterministic": ByAge(
            cases={
                0: DeterministicTransition(func=convert(_survives_at_high_income)),
                1: "dead",
            }
        ),
        # The last working age has one exit, so the stochastic forms are
        # scheduled by age.
        "per_target": ByAge(
            cases={
                0: {
                    "working": StochasticTransition(
                        func=convert(_survival_probability)
                    ),
                    "dead": StochasticTransition(func=convert(_death_probability)),
                },
                1: "dead",
            }
        ),
        "phased_vector": ByAge(
            cases={
                0: Phased(
                    solve=StochasticTransition(func=convert(_survival_vector)),
                    simulate=StochasticTransition(func=convert(_survival_vector)),
                ),
                1: "dead",
            }
        ),
    }[form]


def _income_model(*, working: UserRegime, law, enable_jit: bool = True) -> Model:
    return Model(
        regimes={
            "working": working,
            "dead": UserRegime(functions={"utility": _zero_utility}),
        },
        regime_id_class=_WorkingDead,
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "working"},
        edges={
            "working": Transition(targets={"working": (0,), "dead": (0, 1)}, law=law)
        },
        enable_jit=enable_jit,
    )


def _income_regime(*, values: bool, **slots) -> UserRegime:
    convert = _as_declared if values else _on_codes
    return UserRegime(
        states={"income": _income_grid(values=values)},
        state_transitions={"income": fixed_transition("income")},
        actions={"a": DiscreteGrid(category_class=_TwoCodes)},
        functions={"utility": convert(_income_utility)},
        **slots,
    )


_INCOME_PARAMS: UserParams = {"discount_factor": 0.9}


def _assert_identical_solutions_and_simulations(
    *,
    got: Model,
    expected: Model,
    params: UserParams = _INCOME_PARAMS,
    regime_id: ScalarInt = _WorkingDead.working,
) -> None:
    got_solution = got.solve(params=params, log_level="off")
    expected_solution = expected.solve(params=params, log_level="off")
    pairs = [
        (
            np.asarray(got_solution.values[period][regime]),
            np.asarray(expected_solution.values[period][regime]),
        )
        for period in expected_solution.values
        for regime in expected_solution.values[period]
    ]
    assert len(pairs) == sum(len(r) for r in got_solution.values.values())
    for got_values, expected_values in pairs:
        np.testing.assert_array_equal(got_values, expected_values)
    initial_conditions = {
        "income": jnp.asarray([0, 1, 2, 2, 1, 0], dtype=jnp.int32),
        "age": jnp.zeros(6),
        "regime_id": jnp.full(6, regime_id),
    }
    frames = [
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            seed=3,
            log_level="off",
            use_categorical_codes=True,
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values(["subject_id", "period"])
        .reset_index(drop=True)
        for model, solution in ((got, got_solution), (expected, expected_solution))
    ]
    assert_frame_equal(frames[0], frames[1])


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize(
    "form", ["deterministic", "by_age_deterministic", "per_target", "phased_vector"]
)
def test_regime_transition_law_reads_the_values_of_a_value_grid(*, form, enable_jit):
    """A regime law reading a value grid matches its twin on converted codes."""
    got, expected = (
        _income_model(
            working=_income_regime(values=values),
            law=_regime_law(form=form, values=values),
            enable_jit=enable_jit,
        )
        for values in (True, False)
    )
    _assert_identical_solutions_and_simulations(got=got, expected=expected)


def _scaled_income_utility(
    *, income: DiscreteState, a: DiscreteAction, scale: float = 1.0
) -> FloatND:
    return scale * income * a + 0.5 * (1 - a)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_partial_user_function_reading_a_value_grid_keeps_its_bound_arguments(
    *, enable_jit
):
    """A `partial` reading values asks for no parameter its keywords fix."""
    got, expected = (
        _income_model(
            working=_income_regime(values=values).replace(
                functions={
                    "utility": partial(
                        _scaled_income_utility
                        if values
                        else _on_codes(_scaled_income_utility),
                        scale=2.0,
                    )
                }
            ),
            law=_regime_law(form="per_target", values=values),
            enable_jit=enable_jit,
        )
        for values in (True, False)
    )
    _assert_identical_solutions_and_simulations(got=got, expected=expected)


def _float_grid_built_without_x64() -> DiscreteGrid:
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", val=False)
    try:
        return DiscreteGrid(values=np.array([0.1, 0.7]))
    finally:
        jax.config.update("jax_enable_x64", val=previous)


@pytest.mark.usefixtures("x64_enabled")
@pytest.mark.parametrize("enable_jit", [False, True])
def test_float_value_grid_built_before_x64_holds_float64_values_in_the_model(
    *, enable_jit
):
    """A float64 model reads float64 values of a grid built while x64 was off."""
    working = UserRegime(
        states={"hours": _float_grid_built_without_x64()},
        state_transitions={"hours": fixed_transition("hours")},
        functions={"utility": _hours_as_float},
    )
    got = _first_values(
        model=_one_period_model(working=working, enable_jit=enable_jit),
        params={"discount_factor": 1.0},
    )
    np.testing.assert_array_equal(got, np.asarray([0.1, 0.7], dtype=np.float64))


def _income_utility_f(*, income: DiscreteState, a: DiscreteAction) -> FloatND:
    return income * a + 0.5 * (1 - a)


def _income_utility_m(*, income: DiscreteState, a: DiscreteAction) -> FloatND:
    return 0.5 * income * (1 - a) + a


def _income_utility_terminal(*, income: DiscreteState) -> FloatND:
    return 0.25 * income


def _weight_f(*, income: DiscreteState) -> FloatND:
    """At income 4 the household works: `3.5 * 0.16 > 0.5`; at its code 2 not."""
    return income**2 / 100.0


def _participation_at_income(
    *, Q_f: FloatND, a: DiscreteAction, income: DiscreteState
) -> BoolND:
    return (a == 0) | (Q_f >= 2.0 * income - 3.0)


def _couple_regime(*, values: bool, terminal: bool) -> UserRegime:
    convert = _as_declared if values else _on_codes
    if terminal:
        return UserRegime(
            states={"income": _income_grid(values=values)},
            functions={
                "utility": CollectiveUtility(
                    utilities={
                        "f": convert(_income_utility_terminal),
                        "m": convert(_income_utility_terminal),
                    }
                )
            },
        )
    return UserRegime(
        states={"income": _income_grid(values=values)},
        state_transitions={"income": fixed_transition("income")},
        actions={"a": DiscreteGrid(category_class=_TwoCodes)},
        functions={
            "utility": CollectiveUtility(
                utilities={
                    "f": convert(_income_utility_f),
                    "m": convert(_income_utility_m),
                },
                objective=ParetoObjective(weights={"f": convert(_weight_f), "m": 0.5}),
            )
        },
        constraints={
            "participation": ValueDependentConstraint(
                predicate=convert(_participation_at_income)
            )
        },
    )


def _couple_model(*, values: bool, enable_jit: bool) -> Model:
    convert = _as_declared if values else _on_codes
    law = ByAge(
        cases={
            0: {
                "couple": StochasticTransition(func=convert(_survival_probability)),
                "couple_terminal": StochasticTransition(
                    func=convert(_death_probability)
                ),
            },
            1: "couple_terminal",
        }
    )
    return Model(
        regimes={
            "couple": _couple_regime(values=values, terminal=False),
            "couple_terminal": _couple_regime(values=values, terminal=True),
        },
        regime_id_class=_CoupleTerminal,
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "couple"},
        edges={
            "couple": Transition(
                targets={"couple": (0,), "couple_terminal": (0, 1)}, law=law
            )
        },
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_collective_regime_reads_the_values_of_a_value_grid(*, enable_jit):
    """A household's utilities, weights, value constraint and law read values."""
    _assert_identical_solutions_and_simulations(
        got=_couple_model(values=True, enable_jit=enable_jit),
        expected=_couple_model(values=False, enable_jit=enable_jit),
        regime_id=_CoupleTerminal.couple,
    )


def _gate_at_high_income(*, income: DiscreteState, V_target: FloatND) -> BoolND:
    return (income >= 2.0) & (V_target > -1.0)


def _project_income_down(*, income: DiscreteState) -> DiscreteState:
    return jnp.maximum(income / 2.0, 1.0)


def _project_income_down_on_codes(*, income: DiscreteState) -> DiscreteState:
    projected = _project_income_down(income=_income_of_code(income))
    return jnp.searchsorted(jnp.asarray(_INCOME_VALUES), projected).astype(jnp.int32)


def _twice_income(*, income: DiscreteState) -> FloatND:
    return 2.0 * income


def _gated_income_model(*, values: bool, enable_jit: bool) -> Model:
    convert = _as_declared if values else _on_codes
    terminal = {
        name: UserRegime(
            states={"income": _income_grid(values=values)},
            functions={"utility": convert(utility)},
        )
        for name, utility in (
            ("target", _income_utility_terminal),
            ("fallback", _twice_income),
        )
    }
    gate = Gate(
        predicate=convert(_gate_at_high_income),
        routes={
            "only": StakeholderRoute(
                fallback=ProjectedRegimeValue(
                    regime="fallback",
                    projection={
                        "income": _project_income_down
                        if values
                        else _project_income_down_on_codes
                    },
                )
            )
        },
    )
    return Model(
        regimes={"source": _income_regime(values=values), **terminal},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"source": Transition(law="target", gates={"target": gate})},
        initial_nodes={0: "source"},
        regime_id_class=_SourceTargetFallback,
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_gated_edge_reads_and_projects_the_values_of_a_value_grid(*, enable_jit):
    """A gate and its fallback projection read values; the projection returns one."""
    _assert_identical_solutions_and_simulations(
        got=_gated_income_model(values=True, enable_jit=enable_jit),
        expected=_gated_income_model(values=False, enable_jit=enable_jit),
        regime_id=_SourceTargetFallback.source,
    )


def _income_after_event(*, draw: Int1D, income: DiscreteState) -> DiscreteState:
    """Event `[1]` doubles income up to its top value; event `[0]` keeps it."""
    return jnp.where(draw[0] == 1, jnp.minimum(2.0 * income, 4.0), income)


def _income_after_event_on_codes(
    *, draw: Int1D, income: DiscreteState
) -> DiscreteState:
    landed = _income_after_event(draw=draw, income=_income_of_code(income))
    return jnp.searchsorted(jnp.asarray(_INCOME_VALUES), landed).astype(jnp.int32)


def _raise_probabilities(*, income: DiscreteState) -> FloatND:
    return jnp.stack([income / 5.0, 1.0 - income / 5.0])


def _joint_income_model(*, values: bool, enable_jit: bool) -> Model:
    convert = _as_declared if values else _on_codes
    kernel = JointTransition(
        support_size=2,
        support=jnp.asarray([[0], [1]], dtype=jnp.int32),
        probabilities=convert(_raise_probabilities),
        outputs={
            "income": _income_after_event if values else _income_after_event_on_codes
        },
    )
    working = _income_regime(values=values).replace(
        state_transitions={}, joint_transitions={"working": {"draw": kernel}}
    )
    return _income_model(
        working=working,
        law=_regime_law(form="per_target", values=values),
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_joint_transition_reads_and_returns_the_values_of_a_value_grid(*, enable_jit):
    """A joint lottery reads values in its probabilities and outputs values."""
    _assert_identical_solutions_and_simulations(
        got=_joint_income_model(values=True, enable_jit=enable_jit),
        expected=_joint_income_model(values=False, enable_jit=enable_jit),
    )
