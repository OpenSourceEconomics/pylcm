from collections.abc import Callable, Mapping

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.regime import Regime as UserRegime
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
    WEALTH_GRID,
    Health,
    PartnerStatus,
    RegimeId,
    dead,
    get_model,
    get_params,
    next_health,
    next_partner,
    next_wealth,
    retirement,
    working_life,
)


def test_model_simulate_with_stochastic_model():
    model = get_model(n_periods=4)
    params = get_params(n_periods=4)

    result = model.simulate(
        log_level="debug",
        params=params,
        initial_conditions={
            "health": jnp.array([1, 1, 0, 0], dtype=jnp.int32),
            "partner": jnp.array([0, 0, 1, 0], dtype=jnp.int32),
            "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
            "age": jnp.array([40.0, 40.0, 40.0, 40.0]),
            "regime_id": jnp.array([RegimeId.working_life] * 4),
        },
        period_to_regime_to_V_arr=None,
    )
    df = result.to_dataframe().query('regime_name == "working_life"')

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
            p1["partner"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            check_dtype=False,
            check_categorical=False,
        )


def test_model_solve_with_stochastic_model():
    model = get_model(n_periods=4)
    model.solve(log_level="debug", params=get_params(n_periods=4))


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
        model_deterministic.solve(log_level="debug", params=params)
    )
    solution_stochastic: Mapping[int, Mapping[str, FloatND]] = model_stochastic.solve(
        log_level="debug", params=params
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
        "health": jnp.array([1, 1, 0, 0], dtype=jnp.int32),
        "partner": jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        "age": jnp.array([40.0, 40.0, 40.0, 40.0]),
        "regime_id": jnp.array([RegimeId.working_life] * 4),
    }

    simulation_deterministic = model_deterministic.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=solution_deterministic,
        initial_conditions=initial_conditions,
    )
    simulation_stochastic = model_stochastic.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=solution_stochastic,
        initial_conditions=initial_conditions,
    )
    df_deterministic = simulation_deterministic.to_dataframe().query(
        'regime_name == "working_life"'
    )
    df_stochastic = simulation_stochastic.to_dataframe().query(
        'regime_name == "working_life"'
    )
    pd.testing.assert_frame_equal(
        df_deterministic.reset_index(drop=True),
        df_stochastic.reset_index(drop=True),
    )


def _make_minimal_stochastic_model(next_draw: Callable[..., FloatND]) -> Model:
    """Create a minimal stochastic model with a discrete state `draw`."""

    final_age = 1

    @categorical(ordered=False)
    class ShockStatus:
        bad: ScalarInt
        good: ScalarInt

    @categorical(ordered=False)
    class ShockRegimeId:
        working_life: ScalarInt
        dead: ScalarInt

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

    working_regime = UserRegime(
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
    dead_regime = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"working_life": working_regime, "dead": dead_regime},
        ages=AgeGrid(start=0, stop=final_age + 1, step="Y"),
        regime_id_class=ShockRegimeId,
    )


def _always_bad_draw() -> FloatND:
    """Degenerate `draw` transition: all mass on `bad` (zero expected bonus)."""
    return jnp.array([1.0, 0.0])


def test_stochastic_zero_arg_weight_adds_discounted_expected_bonus() -> None:
    """A zero-argument `MarkovTransition` weight feeds its probabilities into `V`.

    In `_make_minimal_stochastic_model` the `good` outcome of `draw` adds a
    utility bonus of 1.0 and `draw` is otherwise additively separable from the
    budget. A 50/50 `draw` therefore raises period-0 `V` over an otherwise
    identical model with an always-`bad` `draw` by exactly
    `discount_factor * 0.5` — the discounted expected bonus implied by the
    `[0.5, 0.5]` weights — at every state grid point.
    """
    discount_factor = 0.95
    params = {
        "discount_factor": discount_factor,
        "working_life": {"next_regime": {"final_age_alive": 1}},
    }

    def next_draw_5050() -> FloatND:
        return jnp.array([0.5, 0.5])

    V_stochastic = _make_minimal_stochastic_model(next_draw_5050).solve(
        log_level="debug", params=params
    )
    V_degenerate = _make_minimal_stochastic_model(_always_bad_draw).solve(
        log_level="debug", params=params
    )

    extra_continuation_value = (
        V_stochastic[0]["working_life"] - V_degenerate[0]["working_life"]
    )
    assert_allclose(extra_continuation_value, discount_factor * 0.5, atol=1e-6)


def test_stochastic_weight_on_continuous_state_varies_continuation_by_wealth() -> None:
    """A `MarkovTransition` weight may depend on a continuous state.

    `next_draw` sets `P(good)` to `clip(wealth / 10, 0.1, 0.9)`, so the
    discounted expected bonus in period-0 `V` varies across the wealth grid:
    `V` exceeds the always-`bad` baseline by exactly
    `discount_factor * P(good)(wealth)` at each wealth grid point.
    """
    discount_factor = 0.95
    params = {
        "discount_factor": discount_factor,
        "working_life": {"next_regime": {"final_age_alive": 1}},
    }

    def next_draw_wealth_dependent(wealth: ContinuousState) -> FloatND:
        p_good = jnp.clip(wealth / 10.0, 0.1, 0.9)
        return jnp.array([1.0 - p_good, p_good])

    V_stochastic = _make_minimal_stochastic_model(next_draw_wealth_dependent).solve(
        log_level="debug", params=params
    )
    V_degenerate = _make_minimal_stochastic_model(_always_bad_draw).solve(
        log_level="debug", params=params
    )

    extra_continuation_value = (
        V_stochastic[0]["working_life"] - V_degenerate[0]["working_life"]
    )
    # The state grid is (draw, wealth); the bonus is the same for both `draw`
    # rows, so the per-wealth expected bonus appears on every row.
    wealth_grid = jnp.linspace(1.0, 10.0, 15)
    expected_per_wealth = discount_factor * jnp.clip(wealth_grid / 10.0, 0.1, 0.9)
    expected = jnp.broadcast_to(expected_per_wealth, extra_continuation_value.shape)
    assert_allclose(extra_continuation_value, expected, atol=1e-6)


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
        model.solve(log_level="debug", params=mortality.get_params(n_periods=4))


def _build_stochastic_model(n_periods: int, subjects_batch_size: int) -> Model:
    """Build the stochastic test model with a configurable `subjects_batch_size`."""
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age, la=last_age: age < la
            ),
            "retirement": retirement.replace(active=lambda age, la=last_age: age < la),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
        subjects_batch_size=subjects_batch_size,
    )


def test_simulate_with_subjects_batch_size_matches_unbatched_V_arr():
    """`Model(subjects_batch_size>0)` chunks the per-device simulate dispatch and
    produces V_arrs that match the unchunked baseline within float tolerance.

    Exact equality is not expected: chunked dispatch via `jax.lax.map` reorders
    XLA reductions vs. a single big `vmap`, and the resulting 1-ULP drift in V
    can flip individual `argmax` decisions when two actions are nearly tied,
    which then changes downstream stochastic transitions. This is the same
    cross-configuration determinism caveat that running on different hardware
    has — the user-visible invariant is that V values agree within `atol`.
    """
    n_periods = 4
    params = get_params(n_periods=n_periods)
    initial_conditions = {
        "health": jnp.array([1, 1, 0, 0], dtype=jnp.int32),
        "partner": jnp.array([0, 0, 1, 0], dtype=jnp.int32),
        "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        "age": jnp.array([40.0, 40.0, 40.0, 40.0]),
        "regime_id": jnp.array([RegimeId.working_life] * 4),
    }
    baseline = _build_stochastic_model(
        n_periods=n_periods, subjects_batch_size=0
    ).simulate(
        log_level="debug",
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
    )
    batched = _build_stochastic_model(
        n_periods=n_periods, subjects_batch_size=2
    ).simulate(
        log_level="debug",
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
    )
    # Only the first period is float-comparable: per-subject initial state is
    # identical, so chunked vs vmapped Q evaluations agree within XLA reduction
    # tolerance. From period 1 onwards, the 1-ULP drift propagates through
    # `argmax`-on-close-Qs to different action choices and from there to
    # different stochastic-transition outcomes — by-design divergence.
    for regime_name, baseline_periods in baseline.raw_results.items():
        baseline_data = baseline_periods[0]
        batched_data = batched.raw_results[regime_name][0]
        assert baseline_data.V_arr.shape == batched_data.V_arr.shape
        assert_allclose(
            jnp.where(jnp.isnan(baseline_data.V_arr), 0.0, baseline_data.V_arr),
            jnp.where(jnp.isnan(batched_data.V_arr), 0.0, batched_data.V_arr),
            atol=1e-5,
        )


def test_solve_with_stochastic_batch_size_matches_unbatched():
    """`batch_size>0` on a stochastic DiscreteGrid chunks the shock-integration
    productmap but yields the same V_arrs as the unchunked baseline.
    """

    def build_model(health_batch_size: int) -> Model:
        wl = working_life.replace(
            states={
                "health": DiscreteGrid(Health, batch_size=health_batch_size),
                "partner": DiscreteGrid(PartnerStatus),
                "wealth": WEALTH_GRID,
            },
            state_transitions={
                "health": MarkovTransition(next_health),
                "partner": MarkovTransition(next_partner),
                "wealth": next_wealth,
            },
        )
        n_periods = 4
        ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
        last_age = ages.exact_values[-1]
        return Model(
            regimes={
                "working_life": wl.replace(active=lambda age, la=last_age: age < la),
                "retirement": retirement.replace(
                    active=lambda age, la=last_age: age < la
                ),
                "dead": dead,
            },
            ages=ages,
            regime_id_class=RegimeId,
        )

    params = get_params(n_periods=4)
    baseline = build_model(health_batch_size=0).solve(log_level="debug", params=params)
    batched = build_model(health_batch_size=1).solve(log_level="debug", params=params)

    for period in baseline:
        for regime_name in baseline[period]:
            assert_allclose(
                baseline[period][regime_name],
                batched[period][regime_name],
                atol=1e-10,
            )
