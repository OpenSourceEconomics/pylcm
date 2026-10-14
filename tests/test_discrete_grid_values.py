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

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from _lcm.grids.discrete import MISSING_CAT_CODE, read_at_codes
from _lcm.regime_building.V import _LookupWrapper
from _lcm.regime_building.value_adapter import (
    adapt_entry,
    adapt_user_function,
    collect_value_grids,
)
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    DeterministicTransition,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Phased,
    Transition,
    case_boundary,
    categorical,
    fixed_transition,
    piece,
    ref,
)
from lcm.exceptions import InvalidStateTransitionError, RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.regime import StochasticTransition
from lcm.transition import AgeSpecializedFunction
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
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
            "consumption": DiscreteGrid(values=[1.0, 2.0]),
        },
        states={
            "wealth": DiscreteGrid(
                values=[_WEALTH_OFFSET, _WEALTH_OFFSET + 1, _WEALTH_OFFSET + 2]
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
    ).to_dataframe(use_labels=False)
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
            "consumption": DiscreteGrid(values=[1.0, 2.0], tolerance=1e-6),
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


_WEALTH = DiscreteGrid(values=[10, 11, 12])
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
                    states={"wealth": DiscreteGrid(values=[0, 1, 2])},
                    state_transitions={"wealth": _identity_wealth},
                    functions={"utility": lambda: 0.0},
                ),
            },
            laws={},
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
        states={**dict(regime.states), "bracket": DiscreteGrid(values=[5, 7])},
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


def test_model_refuses_an_age_specialized_function_beside_a_value_grid():
    """An age-specialized function in a regime with value grids is refused."""
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
    with pytest.raises(RegimeInitializationError, match="age-specialized"):
        _model_with(working_life=specialized)


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
        states={**dict(regime.states), "room": DiscreteGrid(values=[0, 1])},
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
        derived_categoricals={"bonus": DiscreteGrid(values=nodes)},
        koopmans_aggregator=None if location == "model" else declaration,
    )
    model = _one_period_model(
        working=working,
        enable_jit=enable_jit,
        model_aggregator=declaration if location == "model" else None,
    )
    assert _first_values(model=model, params={}).item() == nodes[1]


def _report_utility(*, utility: FloatND, CE: FloatND) -> FloatND:
    return utility + 0.0 * CE


def _utility_of_hours(*, hours: DiscreteAction) -> FloatND:
    return jnp.asarray(hours, dtype=float)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("nodes", _NODE_PAIRS)
@pytest.mark.parametrize("phased", [False, True])
def test_selection_aggregator_selects_by_the_values_of_an_action(
    *, enable_jit, nodes, phased
):
    """A selection rule on hours values keeps the unique optimum at the lower node."""

    def select(*, utility: FloatND, CE: FloatND, hours: DiscreteAction) -> FloatND:
        return -((hours - nodes[0]) ** 2) + 0.0 * utility + 0.0 * CE

    working = UserRegime(
        actions={"hours": DiscreteGrid(values=nodes)},
        functions={"utility": _utility_of_hours},
        koopmans_aggregator=_report_utility,
        selection_aggregator=(
            Phased(solve=select, simulate=select) if phased else select
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
        states={"hours": DiscreteGrid(values=nodes)},
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
            "hours": DiscreteGrid(values=nodes),
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


def _working_hours_utility(*, hours: DiscreteState) -> FloatND:
    return jnp.asarray(hours, dtype=float)


def _law_at_zero(*, hours: DiscreteState) -> ScalarInt:
    return jnp.where(hours <= 15, _WorkingDead.working, _WorkingDead.dead)


def _law_at_one(*, hours: DiscreteState) -> ScalarInt:
    return jnp.full_like(hours, _WorkingDead.dead, dtype=jnp.int32)


def _flat_law(*, hours: DiscreteState, age: float) -> ScalarInt:
    return jnp.where(
        (age == 0) & (hours <= 15), _WorkingDead.working, _WorkingDead.dead
    )


@pytest.mark.parametrize("law_form", ["flat", "by_age"])
def test_regime_transition_law_reading_a_value_grid_is_refused_in_any_form(law_form):
    """A regime law that reads a value grid is refused, flat or scheduled by age."""
    law = (
        DeterministicTransition(func=_flat_law)
        if law_form == "flat"
        else ByAge(
            cases={
                0: DeterministicTransition(func=_law_at_zero),
                1: DeterministicTransition(func=_law_at_one),
            }
        )
    )
    working = UserRegime(
        states={"hours": DiscreteGrid(values=[10, 20])},
        state_transitions={"hours": fixed_transition("hours")},
        functions={"utility": _working_hours_utility},
    )
    with pytest.raises(RegimeInitializationError, match="regime transition law"):
        Model(
            regimes={
                "working": working,
                "dead": UserRegime(functions={"utility": _zero_utility}),
            },
            regime_id_class=_WorkingDead,
            ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
            initial_nodes={0: "working"},
            edges={
                "working": Transition(
                    targets={"working": (0,), "dead": (0, 1)}, law=law
                )
            },
            enable_jit=False,
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
                values=[10, 20] if kind == "integer" else [10.0, 20.0]
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


def test_an_imputation_reading_a_value_grid_is_refused():
    """A carried state's solve imputation that reads a value grid is refused."""
    working = UserRegime(
        states={
            "hours": DiscreteGrid(values=[10, 20]),
            "earnings": Phased(
                solve=_hours_as_float,
                simulate=LinSpacedGrid(start=0, stop=30, n_points=2),
            ),
        },
        state_transitions={
            "hours": fixed_transition("hours"),
            "earnings": fixed_transition("earnings"),
        },
        functions={"utility": _earnings_utility},
    )
    with pytest.raises(RegimeInitializationError, match="imputation"):
        _one_period_model(working=working, enable_jit=False)


def test_an_imputed_value_grid_state_is_refused():
    """A carried state declared with values in its simulate phase is refused."""
    working = UserRegime(
        states={
            "wealth": LinSpacedGrid(start=0, stop=30, n_points=4),
            "hours": Phased(
                solve=_impute_twenty, simulate=DiscreteGrid(values=[10, 20])
            ),
        },
        state_transitions={
            "wealth": _hours_as_float,
            "hours": fixed_transition("hours"),
        },
        functions={"utility": _wealth_utility},
    )
    with pytest.raises(RegimeInitializationError, match="imputation"):
        _one_period_model(working=working, enable_jit=False, n_periods=2)


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
        grid = DiscreteGrid(values=nodes)
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


@pytest.mark.parametrize("nodes", _NODE_PAIRS)
@pytest.mark.parametrize("marked", [False, True], ids=["plain", "age-specialized"])
def test_an_imputation_reading_a_value_grid_is_refused_in_any_spelling(
    *, nodes, marked
):
    """The refusal holds whether or not the imputation is age-specialized."""
    with pytest.raises(RegimeInitializationError, match=r"imputation.*earnings"):
        _one_period_model(
            working=_imputation_regime(nodes=nodes, marked=marked), enable_jit=False
        )


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
def test_a_phased_age_specialized_utility_beside_a_value_grid_is_refused(nodes):
    """An age-specialized utility inside `Phased` is refused beside a value grid."""
    marker = _specialize(_hours_as_float)
    working = UserRegime(
        states={"hours": DiscreteGrid(values=nodes)},
        state_transitions={"hours": fixed_transition("hours")},
        functions={"utility": Phased(solve=marker, simulate=marker)},
    )
    with pytest.raises(RegimeInitializationError, match=r"age-specialized.*utility"):
        _one_period_model(working=working, enable_jit=False)


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

        grid = DiscreteGrid(values=nodes)
    return UserRegime(
        states={"hours": grid},
        state_transitions={"hours": fixed_transition("hours")},
        actions={"choice": DiscreteGrid(category_class=_HoursCodes)},
        functions={"utility": _choice_utility},
        constraints={"available": _specialize(available)},
    )


@pytest.mark.parametrize("nodes", _NODE_PAIRS)
def test_an_age_specialized_constraint_beside_a_value_grid_is_refused(nodes):
    """An age-specialized constraint is refused in a regime with value grids."""
    with pytest.raises(RegimeInitializationError, match=r"age-specialized.*available"):
        _one_period_model(working=_constraint_regime(nodes=nodes), enable_jit=False)


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


@pytest.mark.parametrize("phased", [False, True])
def test_a_model_level_derived_value_grid_counts_for_the_refusals(phased):
    """A derived value grid declared on the model makes its regimes refuse markers."""

    def bonus() -> DiscreteState:
        return jnp.int32(20)

    def utility(*, bonus: DiscreteState) -> FloatND:
        return jnp.asarray(bonus, dtype=float)

    marker = _specialize(utility)
    working = UserRegime(
        functions={
            "bonus": bonus,
            "utility": Phased(solve=marker, simulate=marker) if phased else marker,
        }
    )
    with pytest.raises(RegimeInitializationError, match=r"age-specialized.*utility"):
        collect_value_grids(
            regimes={"working": working},
            laws={},
            derived_categoricals={"bonus": DiscreteGrid(values=[10, 20])},
        )
