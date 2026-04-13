"""Tests for static params (fixed_params partialled at model initialization)."""

import jax.numpy as jnp
import pandas as pd
from numpy.testing import assert_array_almost_equal as aaae

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt, UserParams
from tests.test_models.regime_markov import Health
from tests.test_models.regime_markov import RegimeId as MarkovRegimeId
from tests.test_models.regime_markov import alive as markov_alive
from tests.test_models.regime_markov import dead as markov_dead


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


def _utility(consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return jnp.log(consumption + 1) + 0.01 * wealth


def _next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption)


def _borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def _next_regime(period: int) -> FloatND:
    return jnp.where(period >= 1, RegimeId.dead, RegimeId.alive)


def _make_model(n_periods=3, *, extra_fixed_params=None):
    """Create a simple 2-regime model for testing."""
    alive = Regime(
        functions={"utility": _utility},
        states={
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=5),
        },
        state_transitions={
            "wealth": _next_wealth,
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        active=lambda age, n=n_periods: age < n - 1,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age, n=n_periods: age >= n - 1,
    )

    fixed_params = extra_fixed_params or {}

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=n_periods - 1, step="Y"),
        regime_id_class=RegimeId,
        fixed_params=fixed_params,
    )


def test_fixed_param_removed_from_template():
    """Fixed params should disappear from params_template."""
    model = _make_model(
        extra_fixed_params={"interest_rate": 0.05},
    )
    # interest_rate should NOT be in the template
    alive_template = model._params_template.get("alive", {})
    all_param_names = set()
    for func_params in alive_template.values():
        all_param_names.update(func_params.keys())
    assert "interest_rate" not in all_param_names
    # discount_factor should still be there
    assert "discount_factor" in all_param_names


def test_solve_with_fewer_params():
    """Solve should work with only the non-fixed params."""
    model = _make_model(
        extra_fixed_params={"interest_rate": 0.05},
    )
    # Should NOT need interest_rate in params, only discount_factor
    params = {"discount_factor": 0.95}
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    assert len(period_to_regime_to_V_arr) > 0


def test_simulate_with_fixed_params():
    """Full solve and simulate with fixed params should produce valid results."""
    # Model without fixed_params
    model_full = _make_model()
    params_full = {"discount_factor": 0.95, "interest_rate": 0.05}
    result_full = model_full.simulate(
        params=params_full,
        initial_conditions={
            "wealth": jnp.array([5.0, 7.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime": jnp.array([RegimeId.alive] * 2),
        },
        period_to_regime_to_V_arr=None,
        log_level="off",
    )

    # Model with interest_rate as fixed param
    model_fixed = _make_model(
        extra_fixed_params={"interest_rate": 0.05},
    )
    params_fixed = {"discount_factor": 0.95}
    result_fixed = model_fixed.simulate(
        params=params_fixed,
        initial_conditions={
            "wealth": jnp.array([5.0, 7.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime": jnp.array([RegimeId.alive] * 2),
        },
        period_to_regime_to_V_arr=None,
        log_level="off",
    )

    # Results should be identical
    df_full = result_full.to_dataframe()
    df_fixed = result_fixed.to_dataframe()
    aaae(df_full["wealth"].values, df_fixed["wealth"].values)
    aaae(df_full["consumption"].values, df_fixed["consumption"].values)


def test_regime_level_fixed_param():
    """Fixed params at regime level should work."""
    model = _make_model(
        extra_fixed_params={"alive": {"interest_rate": 0.05}},
    )
    # interest_rate should be removed from alive's template
    alive_template = model._params_template.get("alive", {})
    all_param_names = set()
    for func_params in alive_template.values():
        all_param_names.update(func_params.keys())
    assert "interest_rate" not in all_param_names

    params = {"discount_factor": 0.95}
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    assert len(period_to_regime_to_V_arr) > 0


def test_all_params_fixed():
    """All params can be fixed, leaving an empty template."""
    model = _make_model(
        extra_fixed_params={"interest_rate": 0.05, "discount_factor": 0.95},
    )
    # All regime templates should be empty
    for regime_template in model._params_template.values():
        assert len(regime_template) == 0

    # Solve with empty params
    period_to_regime_to_V_arr = model.solve(params={}, log_level="off")
    assert len(period_to_regime_to_V_arr) > 0


_AGES = (60.0, 61.0, 62.0)

_PROBS_SERIES = pd.Series(
    [0.95, 0.05, 0.98, 0.02, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    index=pd.MultiIndex.from_product(
        [_AGES, ["bad", "good"], ["alive", "dead"]],
        names=["age", "health", "next_regime"],
    ),
)

_MARKOV_INITIAL_CONDITIONS = {
    "wealth": jnp.array([50.0, 80.0]),
    "health": jnp.array([Health.bad, Health.good]),
    "age": jnp.array([60.0, 60.0]),
    "regime": jnp.array([MarkovRegimeId.alive] * 2),
}


def _make_markov_model(*, fixed_params: UserParams | None = None) -> Model:
    """Create regime_markov model with optional fixed_params."""
    return Model(
        regimes={"alive": markov_alive, "dead": markov_dead},
        ages=AgeGrid(start=60, stop=62, step="Y"),
        regime_id_class=MarkovRegimeId,
        fixed_params=fixed_params or {},
    )


def test_series_as_runtime_param_works():
    """Baseline: pd.Series works as a runtime param (not fixed)."""
    model = _make_markov_model()
    result = model.simulate(
        params={"discount_factor": 0.95, "probs_array": _PROBS_SERIES},
        initial_conditions=_MARKOV_INITIAL_CONDITIONS,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe()
    assert len(df) > 0


def test_series_as_fixed_param():
    """pd.Series in fixed_params should be auto-converted like runtime params."""
    model = _make_markov_model(
        fixed_params={"probs_array": _PROBS_SERIES},
    )
    result = model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions=_MARKOV_INITIAL_CONDITIONS,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe()
    assert len(df) > 0


def test_series_fixed_param_parity_with_runtime_param():
    """Same Series value as fixed_param vs runtime param produces identical results."""
    model_runtime = _make_markov_model()
    result_runtime = model_runtime.simulate(
        params={"discount_factor": 0.95, "probs_array": _PROBS_SERIES},
        initial_conditions=_MARKOV_INITIAL_CONDITIONS,
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=0,
    )

    model_fixed = _make_markov_model(
        fixed_params={"probs_array": _PROBS_SERIES},
    )
    result_fixed = model_fixed.simulate(
        params={"discount_factor": 0.95},
        initial_conditions=_MARKOV_INITIAL_CONDITIONS,
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=0,
    )

    df_runtime = result_runtime.to_dataframe()
    df_fixed = result_fixed.to_dataframe()
    aaae(df_runtime["wealth"].to_numpy(), df_fixed["wealth"].to_numpy())


def test_mixed_series_and_scalar_fixed_params():
    """Mixed: some fixed_params are Series, others are scalars."""
    model = _make_markov_model(
        fixed_params={
            "probs_array": _PROBS_SERIES,
            "discount_factor": 0.95,
        },
    )
    result = model.simulate(
        params={},
        initial_conditions=_MARKOV_INITIAL_CONDITIONS,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe()
    assert len(df) > 0


@categorical(ordered=False)
class _WealthGroup:
    low: int
    high: int


def _wealth_group(wealth: ContinuousState) -> ScalarInt:
    return jnp.int32(wealth > 5.0)


def _utility_with_group(
    consumption: ContinuousAction,
    wealth_group: ScalarInt,
    group_bonus: FloatND,
) -> FloatND:
    return jnp.log(consumption + 1) + group_bonus[wealth_group]


def test_series_fixed_param_with_derived_categoricals():
    """Fixed pd.Series indexed by derived categorical needs derived_categoricals."""
    group_bonus = pd.Series(
        [0.0, 1.0],
        index=pd.Index(["low", "high"], name="wealth_group"),
    )
    alive = Regime(
        functions={"utility": _utility_with_group, "wealth_group": _wealth_group},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": lambda wealth: wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        active=lambda age: age < 2,
        derived_categoricals={"wealth_group": _WealthGroup},
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 2,
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
        fixed_params={"group_bonus": group_bonus},
    )
    result = model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions={
            "wealth": jnp.array([3.0, 8.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime": jnp.array([RegimeId.alive] * 2),
        },
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe()
    assert len(df) > 0


def test_model_broadcast_merges_into_regimes():
    """Model-level derived_categoricals broadcast to all regimes (raw class)."""
    alive = Regime(
        functions={"utility": _utility_with_group, "wealth_group": _wealth_group},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": lambda wealth: wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        active=lambda age: age < 2,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 2,
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
        derived_categoricals={"wealth_group": _WealthGroup},
    )
    assert isinstance(
        model.regimes["alive"].derived_categoricals["wealth_group"], DiscreteGrid
    )
    assert isinstance(
        model.regimes["dead"].derived_categoricals["wealth_group"], DiscreteGrid
    )


def test_model_broadcast_matching_regime_entry():
    """Model-level entry matching a regime entry does not conflict."""
    wg_grid = DiscreteGrid(_WealthGroup)
    alive = Regime(
        functions={"utility": _utility_with_group, "wealth_group": _wealth_group},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": lambda wealth: wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        active=lambda age: age < 2,
        derived_categoricals={"wealth_group": wg_grid},
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 2,
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
        derived_categoricals={"wealth_group": wg_grid},
    )
    assert model.regimes["alive"].derived_categoricals["wealth_group"] is wg_grid


def test_model_broadcast_conflict_raises():
    """Model-level entry conflicting with regime entry raises."""
    import pytest  # noqa: PLC0415

    @categorical(ordered=False)
    class _OtherGroup:
        a: int
        b: int
        c: int

    alive = Regime(
        functions={"utility": _utility_with_group, "wealth_group": _wealth_group},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": lambda wealth: wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        active=lambda age: age < 2,
        derived_categoricals={"wealth_group": DiscreteGrid(_OtherGroup)},
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 2,
    )
    with pytest.raises(Exception, match="conflicts"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=RegimeId,
            derived_categoricals={"wealth_group": DiscreteGrid(_WealthGroup)},
        )


def test_different_regime_derived_categoricals_with_model_broadcast():
    """Different per-regime grids coexist with a model-level broadcast."""

    @categorical(ordered=False)
    class _GroupA:
        x: int
        y: int

    @categorical(ordered=False)
    class _GroupB:
        p: int
        q: int

    @categorical(ordered=False)
    class _Shared:
        lo: int
        hi: int

    alive = Regime(
        functions={"utility": lambda: 0.0},
        transition=_next_regime,
        active=lambda age: age < 2,
        derived_categoricals={"group_a": DiscreteGrid(_GroupA)},
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 2,
        derived_categoricals={"group_b": DiscreteGrid(_GroupB)},
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
        derived_categoricals={"shared": DiscreteGrid(_Shared)},
    )
    assert "group_a" in model.regimes["alive"].derived_categoricals
    assert "shared" in model.regimes["alive"].derived_categoricals
    assert "group_a" not in model.regimes["dead"].derived_categoricals
    assert "group_b" in model.regimes["dead"].derived_categoricals
    assert "shared" in model.regimes["dead"].derived_categoricals
