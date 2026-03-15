from collections.abc import Callable, Mapping

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
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
    RegimeId,
    dead,
    get_model,
    get_params,
    retirement,
    working_life,
)

# ======================================================================================
# Simulate
# ======================================================================================


def test_model_solve_and_simulate_with_stochastic_model():
    model = get_model(n_periods=4)
    params = get_params(n_periods=4)

    result = model.solve_and_simulate(
        params=params,
        initial_conditions={
            "health": jnp.array([1, 1, 0, 0]),
            "partner": jnp.array([0, 0, 1, 0]),
            "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
            "age": jnp.array([40.0, 40.0, 40.0, 40.0]),
            "regime_id": jnp.array([RegimeId.working_life] * 4),
        },
    )
    df = result.to_dataframe().query('regime == "working_life"')

    # Verify expected columns
    required_cols = {"period", "subject_id", "partner", "work"}
    assert required_cols <= set(df.columns)
    assert len(df) > 0

    # Check partner transition follows expected pattern:
    # Partner becomes single if working and partnered, otherwise stays partnered
    period_0 = df.query("period == 0").set_index("subject_id")
    period_1 = df.query("period == 1").set_index("subject_id")
    common = period_0.index.intersection(period_1.index)

    if len(common) > 0:
        p0, p1 = period_0.loc[common], period_1.loc[common]
        should_be_single = (p0["work"] == "work") & (p0["partner"] == "partnered")
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
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]

    # Create deterministic model by replacing health grid transition
    working_deterministic = working_life.replace(
        state_transitions={
            **working_life.state_transitions,
            "health": next_health_deterministic,
        },
        active=lambda age: age < last_age,
    )
    retirement_deterministic = retirement.replace(
        state_transitions={
            **retirement.state_transitions,
            "health": next_health_deterministic,
        },
        active=lambda age: age < last_age,
    )

    # Create stochastic model with identity transition function
    working_stochastic = working_life.replace(
        state_transitions={
            **working_life.state_transitions,
            "health": MarkovTransition(next_health_stochastic),
        },
        active=lambda age: age < last_age,
    )
    retirement_stochastic = retirement.replace(
        state_transitions={
            **retirement.state_transitions,
            "health": MarkovTransition(next_health_stochastic),
        },
        active=lambda age: age < last_age,
    )

    model_deterministic = Model(
        regimes={
            "working_life": working_deterministic,
            "retirement": retirement_deterministic,
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    model_stochastic = Model(
        regimes={
            "working_life": working_stochastic,
            "retirement": retirement_stochastic,
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    # Use survival_probs=1.0 for all but the last period so no subject dies early.
    # This test focuses on health transition equivalence, not mortality.
    params = get_params(n_periods=n_periods)
    params["survival_probs"] = jnp.concatenate(
        [jnp.full(n_periods - 2, 1.0), jnp.array([0.0])]
    )

    return model_deterministic, model_stochastic, params


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
            solution_deterministic[period]["working_life"],
            solution_stochastic[period]["working_life"],
            decimal=14,
        )

    # ==================================================================================
    # Compare simulation results
    # ==================================================================================
    initial_conditions = {
        "health": jnp.array([1, 1, 0, 0]),
        "partner": jnp.array([0, 0, 0, 0]),
        "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        "age": jnp.array([40.0, 40.0, 40.0, 40.0]),
        "regime_id": jnp.array([RegimeId.working_life] * 4),
    }

    simulation_deterministic = model_deterministic.simulate(
        params,
        V_arr_dict=solution_deterministic,
        initial_conditions=initial_conditions,
    )
    simulation_stochastic = model_stochastic.simulate(
        params,
        V_arr_dict=solution_stochastic,
        initial_conditions=initial_conditions,
    )
    df_deterministic = simulation_deterministic.to_dataframe().query(
        'regime == "working_life"'
    )
    df_stochastic = simulation_stochastic.to_dataframe().query(
        'regime == "working_life"'
    )
    pd.testing.assert_frame_equal(
        df_deterministic.reset_index(drop=True),
        df_stochastic.reset_index(drop=True),
    )


# ======================================================================================
# Minimal stochastic model for issue reproducers
# ======================================================================================


def _make_minimal_stochastic_model(next_draw: Callable[..., FloatND]) -> Model:
    """Create a minimal stochastic model with a discrete state `draw`."""

    final_age = 1

    @categorical(ordered=False)
    class ShockStatus:
        bad: int
        good: int

    @categorical(ordered=False)
    class ShockRegimeId:
        working_life: int
        dead: int

    def utility(consumption: ContinuousAction, draw: DiscreteState) -> FloatND:
        bonus = jnp.where(draw == ShockStatus.good, 1.0, 0.0)
        return jnp.log(consumption) + bonus

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption + 2.0

    def borrowing_constraint(
        consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    def next_regime(age: float, final_age_alive: float) -> ScalarInt:
        return jnp.where(
            age >= final_age_alive, ShockRegimeId.dead, ShockRegimeId.working_life
        )

    working_regime = Regime(
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=20)},
        states={
            "draw": DiscreteGrid(ShockStatus),
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=15),
        },
        state_transitions={
            "draw": MarkovTransition(next_draw),
            "wealth": next_wealth,
        },
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime,
        functions={"utility": utility},
        active=lambda age: age <= final_age,
    )
    dead_regime = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"working_life": working_regime, "dead": dead_regime},
        ages=AgeGrid(start=0, stop=final_age + 1, step="Y"),
        regime_id_class=ShockRegimeId,
    )


# ======================================================================================
# Issue reproducers / regression guards
# ======================================================================================


def test_stochastic_next_function_with_no_arguments():
    """Issue #39 (resolved): zero-arg stochastic next functions now work.

    Regression guard -- previously the weight function machinery assumed at least
    one argument, causing a failure.
    """

    def next_draw_no_args() -> FloatND:
        return jnp.array([0.5, 0.5])

    model = _make_minimal_stochastic_model(next_draw_no_args)
    params = {
        "discount_factor": 0.95,
        "working_life": {"next_regime": {"final_age_alive": 1}},
    }
    V = model.solve(params)
    assert all(
        jnp.all(jnp.isfinite(V[p]["working_life"])) for p in V if "working_life" in V[p]
    )


def test_stochastic_next_depending_on_continuous_state():
    """Issue #35 (resolved): stochastic next functions can depend on continuous states.

    Regression guard -- previously an explicit check rejected any stochastic
    dependency that was not a discrete state.
    """

    def next_draw_continuous(wealth: ContinuousState) -> FloatND:
        p_good = jnp.clip(wealth / 10.0, 0.1, 0.9)
        return jnp.array([1.0 - p_good, p_good])

    model = _make_minimal_stochastic_model(next_draw_continuous)
    params = {
        "discount_factor": 0.95,
        "working_life": {"next_regime": {"final_age_alive": 1}},
    }
    V = model.solve(params)
    assert all(
        jnp.all(jnp.isfinite(V[p]["working_life"])) for p in V if "working_life" in V[p]
    )


def test_stochastic_regime_transition_active_at_last_period_raises():
    """Non-terminal regimes active at the last period must raise an error.

    See https://github.com/OpenSourceEconomics/pylcm/issues/276.
    """
    from lcm_examples import mortality  # noqa: PLC0415

    # Deliberately set active=always to trigger the validation error.
    model = Model(
        regimes={
            "working_life": mortality.working_life.replace(active=lambda _age: True),
            "retirement": mortality.retirement.replace(active=lambda _age: True),
            "dead": mortality.dead,
        },
        ages=AgeGrid(start=40, stop=70, step="10Y"),
        regime_id_class=mortality.RegimeId,
    )

    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"Non-terminal regime.*active at the last period",
    ):
        model.solve(mortality.get_params(n_periods=4))
