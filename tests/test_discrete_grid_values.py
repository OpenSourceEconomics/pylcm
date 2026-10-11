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

from _lcm.dtypes import canonical_float_dtype
from _lcm.grids.discrete import MISSING_CAT_CODE, read_at_codes
from _lcm.regime_building.ndimage import _compute_pinned_index_and_weight
from _lcm.regime_building.V import _LookupWrapper
from _lcm.regime_building.value_adapter import (
    adapt_entry,
    adapt_user_function,
    collect_value_grids,
)
from lcm import AgeGrid, DiscreteGrid, Model, Phased, fixed_transition
from lcm.exceptions import InvalidStateTransitionError, RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.regime import StochasticTransition
from lcm.transition import AgeSpecializedFunction
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
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


def test_pinned_read_off_the_axis_carries_nan_weight():
    """A pinned coordinate outside `0..n-1` reads with weight NaN."""
    [(_, weight)] = _compute_pinned_index_and_weight(
        coordinate=jnp.asarray([-1.0, 3.0, 1.0], dtype=canonical_float_dtype()),
        input_size=3,
    )
    assert np.isnan(np.asarray(weight)).tolist() == [True, True, False]


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
