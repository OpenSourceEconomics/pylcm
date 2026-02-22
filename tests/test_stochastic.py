from collections.abc import Mapping

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from lcm import (
    AgeGrid,
    DiscreteGrid,
    DiscreteMarkovGrid,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidParamsError
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserParams,
)
from tests.test_models.stochastic import (
    HealthStatus,
    RegimeId,
    dead,
    get_model,
    get_params,
    retired,
    working,
)

# ======================================================================================
# Simulate
# ======================================================================================


def test_model_solve_and_simulate_with_stochastic_model():
    model = get_model(n_periods=4)
    params = get_params(n_periods=4)

    result = model.solve_and_simulate(
        params=params,
        initial_states={
            "health": jnp.array([1, 1, 0, 0]),
            "partner": jnp.array([0, 0, 1, 0]),
            "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
            "age": jnp.array([0.0, 0.0, 0.0, 0.0]),
        },
        initial_regimes=["working"] * 4,
    )
    df = result.to_dataframe().query('regime == "working"')

    # Verify expected columns
    required_cols = {"period", "subject_id", "partner", "labor_supply"}
    assert required_cols <= set(df.columns)
    assert len(df) > 0

    # Check partner transition follows expected pattern:
    # Partner becomes single if working and partnered, otherwise stays partnered
    period_0 = df.query("period == 0").set_index("subject_id")
    period_1 = df.query("period == 1").set_index("subject_id")
    common = period_0.index.intersection(period_1.index)

    if len(common) > 0:
        p0, p1 = period_0.loc[common], period_1.loc[common]
        should_be_single = (p0["labor_supply"] == "work") & (
            p0["partner"] == "partnered"
        )
        expected = should_be_single.map({True: "single", False: "partnered"})

        pd.testing.assert_series_equal(
            p1["partner"],
            expected,
            check_names=False,
            check_dtype=False,
            check_categorical=False,
        )


# ======================================================================================
# Solve
# ======================================================================================


def test_model_solve_with_stochastic_model():
    model = get_model(n_periods=4)
    model.solve(params=get_params(n_periods=4))


# ======================================================================================
# Comparison with deterministic results
# ======================================================================================


@pytest.fixture
def models_and_params() -> tuple[Model, Model, UserParams]:
    """Return a deterministic and stochastic model with parameters.

    TODO(@timmens): Add this to tests/test_models/stochastic.py.

    """

    def next_health_stochastic(health: DiscreteState) -> FloatND:
        return jnp.identity(2)[health]

    def next_health_deterministic(health: DiscreteState) -> DiscreteState:
        return health

    n_periods = 4
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")

    # Create deterministic model by replacing health grid transition
    working_deterministic = working.replace(
        states={
            **working.states,
            "health": DiscreteGrid(
                category_class=HealthStatus, transition=next_health_deterministic
            ),
        },
        active=lambda age: age < n_periods - 1,
    )
    retired_deterministic = retired.replace(
        states={
            **retired.states,
            "health": DiscreteGrid(
                category_class=HealthStatus, transition=next_health_deterministic
            ),
        },
        active=lambda age: age < n_periods - 1,
    )

    # Create stochastic model with identity transition function
    working_stochastic = working.replace(
        states={
            **working.states,
            "health": DiscreteMarkovGrid(
                category_class=HealthStatus, transition=next_health_stochastic
            ),
        },
        active=lambda age: age < n_periods - 1,
    )
    retired_stochastic = retired.replace(
        states={
            **retired.states,
            "health": DiscreteMarkovGrid(
                category_class=HealthStatus, transition=next_health_stochastic
            ),
        },
        active=lambda age: age < n_periods - 1,
    )

    dead_updated = dead.replace(active=lambda age: age >= n_periods - 1)

    model_deterministic = Model(
        regimes={
            "working": working_deterministic,
            "retired": retired_deterministic,
            "dead": dead_updated,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    model_stochastic = Model(
        regimes={
            "working": working_stochastic,
            "retired": retired_stochastic,
            "dead": dead_updated,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    return model_deterministic, model_stochastic, get_params(n_periods=n_periods)


def test_compare_deterministic_and_stochastic_results_value_function(
    models_and_params: tuple[Model, Model, UserParams],
) -> None:
    """Test that the deterministic and stochastic models produce the same results."""
    model_deterministic, model_stochastic, params = models_and_params

    # ==================================================================================
    # Compare value function arrays
    # ==================================================================================
    solution_deterministic: Mapping[int, Mapping[str, FloatND]] = (
        model_deterministic.solve(params)
    )
    solution_stochastic: Mapping[int, Mapping[str, FloatND]] = model_stochastic.solve(
        params
    )

    for period in range(model_deterministic.n_periods - 1):
        assert_array_almost_equal(
            solution_deterministic[period]["working"],
            solution_stochastic[period]["working"],
            decimal=14,
        )

    # ==================================================================================
    # Compare simulation results
    # ==================================================================================
    initial_states = {
        "health": jnp.array([1, 1, 0, 0]),
        "partner": jnp.array([0, 0, 0, 0]),
        "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        "age": jnp.array([0.0, 0.0, 0.0, 0.0]),
    }
    initial_regimes = ["working"] * 4

    simulation_deterministic = model_deterministic.simulate(
        params,
        V_arr_dict=solution_deterministic,
        initial_states=initial_states,
        initial_regimes=initial_regimes,
    )
    simulation_stochastic = model_stochastic.simulate(
        params,
        V_arr_dict=solution_stochastic,
        initial_states=initial_states,
        initial_regimes=initial_regimes,
    )
    df_deterministic = simulation_deterministic.to_dataframe().query(
        'regime == "working"'
    )
    df_stochastic = simulation_stochastic.to_dataframe().query('regime == "working"')
    pd.testing.assert_frame_equal(
        df_deterministic.reset_index(drop=True),
        df_stochastic.reset_index(drop=True),
    )


# ======================================================================================
# Minimal stochastic model for issue reproducers
# ======================================================================================


@categorical
class _ShockStatus:
    bad: int
    good: int


@categorical
class _ShockRegimeId:
    working: int
    dead: int


_STOCH_FINAL_AGE = 1  # 3 periods total


def _next_shock(shock: DiscreteState, shock_transition: FloatND) -> FloatND:
    """Default stochastic transition using a pre-computed transition matrix."""
    return shock_transition[shock]


def _stoch_utility(consumption: ContinuousAction, shock: DiscreteState) -> FloatND:
    bonus = jnp.where(shock == _ShockStatus.good, 1.0, 0.0)
    return jnp.log(consumption) + bonus


def _stoch_next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 2.0


def _stoch_borrowing(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _stoch_next_regime(age: float, final_age_alive: float) -> ScalarInt:
    dead = _ShockRegimeId.dead
    working = _ShockRegimeId.working
    return jnp.where(age >= final_age_alive, dead, working)


def _make_minimal_stochastic_model(shock_transition_func=None) -> Model:
    """Create a minimal stochastic model with a discrete shock state."""
    if shock_transition_func is None:
        shock_transition_func = _next_shock

    working_regime = Regime(
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=20)},
        states={
            "shock": DiscreteMarkovGrid(_ShockStatus, transition=shock_transition_func),
            "wealth": LinSpacedGrid(
                start=1, stop=10, n_points=15, transition=_stoch_next_wealth
            ),
        },
        constraints={"borrowing_constraint": _stoch_borrowing},
        transition=_stoch_next_regime,
        functions={"utility": _stoch_utility},
        active=lambda age: age <= _STOCH_FINAL_AGE,
    )
    dead_regime = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age > _STOCH_FINAL_AGE,
    )
    return Model(
        regimes={"working": working_regime, "dead": dead_regime},
        ages=AgeGrid(start=0, stop=_STOCH_FINAL_AGE + 1, step="Y"),
        regime_id_class=_ShockRegimeId,
    )


@pytest.fixture
def minimal_stochastic_model():
    return _make_minimal_stochastic_model()


@pytest.fixture
def stoch_base_params():
    """Params for the minimal stochastic model (no transition params)."""
    return {
        "discount_factor": 0.95,
        "working": {"next_regime": {"final_age_alive": _STOCH_FINAL_AGE}},
    }


# ======================================================================================
# Issue reproducers / regression guards
# ======================================================================================


def test_stochastic_next_function_with_no_arguments(stoch_base_params):
    """Issue #39 (resolved): zero-arg stochastic next functions now work.

    Regression guard -- previously the weight function machinery assumed at least
    one argument, causing a failure.
    """

    def next_shock_no_args() -> FloatND:
        return jnp.array([0.5, 0.5])

    model = _make_minimal_stochastic_model(next_shock_no_args)
    V = model.solve(stoch_base_params)
    assert all(jnp.all(jnp.isfinite(V[p]["working"])) for p in V if "working" in V[p])


def test_stochastic_next_depending_on_continuous_state(stoch_base_params):
    """Issue #35 (resolved): stochastic next functions can depend on continuous states.

    Regression guard -- previously an explicit check rejected any stochastic
    dependency that was not a discrete state.
    """

    def next_shock_continuous(wealth: ContinuousState) -> FloatND:
        p_good = jnp.clip(wealth / 10.0, 0.1, 0.9)
        return jnp.array([1.0 - p_good, p_good])

    model = _make_minimal_stochastic_model(next_shock_continuous)
    V = model.solve(stoch_base_params)
    assert all(jnp.all(jnp.isfinite(V[p]["working"])) for p in V if "working" in V[p])


@pytest.mark.parametrize("bad_shape", [(1, 2), (4, 2)])
def test_wrong_transition_matrix_shape_rejected(bad_shape, minimal_stochastic_model):
    """Issue #63: solve should reject wrong-shaped transition matrices.

    ShockStatus has 2 values, so the transition matrix must be (2, 2).
    Passing (1, 2) or (4, 2) should raise a validation error but currently
    succeeds silently (JAX clips out-of-bounds indices).
    """
    params = {
        "discount_factor": 0.95,
        "working": {
            "next_shock": {
                "shock_transition": jnp.ones(bad_shape) / bad_shape[1],
            },
            "next_regime": {"final_age_alive": _STOCH_FINAL_AGE},
        },
    }
    with pytest.raises(InvalidParamsError):
        minimal_stochastic_model.solve(params)


def test_params_template_includes_stochastic_transition_shape(
    minimal_stochastic_model,
):
    """Issue #185: params_template should include required array shapes.

    For this model, shock_transition[shock] indexes by current shock state (0 or 1)
    and returns a probability vector of length 2, so the correct shape is (2, 2).
    Currently the template only shows the type alias (FloatND).
    """
    template = minimal_stochastic_model.params_template
    shock_info = template["working"]["next_shock"]["shock_transition"]
    assert isinstance(shock_info, tuple)
